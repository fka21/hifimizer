#!/usr/bin/env python3

import json
import logging
import os
import random
import shutil
import signal
import sys
from pathlib import Path

# Redirect caches that would otherwise be written under $HOME *before* any
# library that uses them is imported. In a read-only container $HOME may not be
# writable at all, in which case matplotlib and fontconfig rebuild their caches
# from scratch on every single invocation -- which can take minutes on a shared
# network filesystem and looks exactly like a hang.
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "mplconfig"))
os.environ.setdefault("PYTHONNOUSERSITE", "1")
try:
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
except OSError:
    pass

import numpy as np
import optuna
import optuna.visualization as vis
import plotly.graph_objects as go
import psutil

# NOTE: `optunahub` is deliberately NOT imported here. It is a hub client that
# contacts GitHub, so importing it at startup makes every invocation -- even
# `--help` and `--version` -- block until the network call resolves. On an
# offline compute node behind a firewall that drops (rather than rejects)
# packets, that is a multi-minute stall with no output. It is imported lazily
# in the multi-objective branch, which is the only place it is used.

from utils.assembly_eval import AssemblyEvaluator
from utils.argparser import get_args
from utils.hifiasm_command import (
    HIFIASM_PARAM_KEYS,
    build_hifiasm_command,
    collect_hifiasm_outputs,
    run_default_hifiasm_assembly,
)
from utils.objective import ObjectiveBuilder, haplotype_suffixes
from utils.optuna_callback import MultiCriteriaConvergenceDetector
from utils.optuna_plots import write_param_importances
from utils.paths import RunPaths
from utils.subprocess_logger import SubprocessLogger, TIMEOUT_EXIT_CODE

# ---------------------------------------------------------------- termination
TERMINATE_REQUESTED = False


def terminate_all_processes(sig, frame):
    """
    Kill everything we started, then exit.

    SubprocessLogger launches each external tool with ``start_new_session=True``
    so that walltime enforcement can signal the whole process group. That also
    means a plain ``children(recursive=True)`` sweep from this process can miss
    them, which is why the tool's own registry is drained first: without it,
    Ctrl-C left hifiasm running and holding every core on the node.
    """
    global TERMINATE_REQUESTED
    TERMINATE_REQUESTED = True

    SubprocessLogger.kill_all_active()

    parent = psutil.Process()
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except Exception:
            pass

    logging.info("Termination signal received. All child processes killed.")
    sys.exit(128 + sig)


signal.signal(signal.SIGINT, terminate_all_processes)
signal.signal(signal.SIGTERM, terminate_all_processes)

# --------------------------------------------------------------------- inputs
args = get_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

input_reads = Path(args.input_reads).resolve()
hic1 = Path(args.hic1).resolve() if args.hic1 else None
hic2 = Path(args.hic2).resolve() if args.hic2 else None
ul = Path(args.ul).resolve() if args.ul else None
threads = args.threads
download_path = args.busco_download_path

if bool(hic1) != bool(hic2):
    logging.error("--hic1 and --hic2 must always be provided together.")
    sys.exit(1)

_input_files = {"--input-reads": input_reads}
if hic1:
    _input_files["--hic1"] = hic1
if hic2:
    _input_files["--hic2"] = hic2
if ul:
    _input_files["--ul"] = ul

_missing = [
    f"{flag}: {path}" for flag, path in _input_files.items() if not path.exists()
]
if _missing:
    logging.error("Input file(s) not found:\n  " + "\n  ".join(_missing))
    sys.exit(1)

# ---------------------------------------------------------------------- paths
# Final results live directly under output_dir; every intermediate lives under
# output_dir/work and can be deleted without losing a result.
paths = RunPaths(args.output_dir).create()
output_dir = paths.output_dir
logs_dir = paths.logs_dir

# Kept for backwards compatibility with relative paths a user may pass on the
# command line after this point; nothing hifimizer writes depends on the CWD.
os.chdir(output_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(logs_dir / "main.log"), logging.StreamHandler()],
)

# Checked here, before anything is created or deleted: --force-rerun wipes the
# work tree, and doing that only to then reject the flag combination would
# destroy a study the user was trying to re-use.
if args.rerun_trial is not None and args.rerun_best:
    logging.error("--rerun-trial and --rerun-best are mutually exclusive.")
    sys.exit(1)
if args.rerun_trial is not None and args.force_rerun:
    logging.error("--rerun-trial and --force-rerun are mutually exclusive.")
    sys.exit(1)
if args.rerun_best and args.force_rerun:
    logging.error("--rerun-best and --force-rerun are mutually exclusive.")
    sys.exit(1)

KNOWN_GENOME_SIZE = args.genome_size  # megabases
ploidy = args.ploidy
GENOME_SIZE_BP = KNOWN_GENOME_SIZE * 1_000_000

# Per-stage wall-clock budgets. A stage that blows its budget is killed,
# loses only its own metrics, and is retired for the rest of the study.
STAGE_WALLTIMES = {
    "gfastats": args.gfastats_walltime,
    "alignment": args.align_walltime,
    "samtools_stats": args.samtools_stats_walltime,
    "sniffles": args.sniffles_walltime,
    "yak": args.yak_walltime,
    "busco": args.busco_walltime,
}

evaluator = AssemblyEvaluator(
    known_genome_size=GENOME_SIZE_BP,
    input_reads=input_reads,
    paths=paths,
    threads=threads,
    download_path=download_path,
    ont=args.ont,
    kmer_eval=args.kmer_eval,
    include_busco=args.include_busco,
    yak_k=args.kmer_k,
    yak_bloom_bits=args.yak_bloom_bits,
    stage_walltimes=STAGE_WALLTIMES,
)

# ------------------------------------------------------------ default-only run
if args.default_hifiasm:
    run_default_hifiasm_assembly(
        prefix=str(paths.default_assembly_dir / "default_run"),
        haploid_genome_size=KNOWN_GENOME_SIZE,
        threads=threads,
        primary=args.primary,
        hic1=hic1,
        hic2=hic2,
        ul=ul,
        input_reads=input_reads,
        ont=args.ont,
        logs_dir=logs_dir,
        hom_cov=args.hom_cov,
        walltime_hours=args.trial_walltime,
    )
    sys.exit(0)

# ------------------------------------------------------------------- dry run
if args.dry_run:
    _required_tools = ["hifiasm", "gfastats", "minimap2", "samtools", "sniffles"]
    if args.include_busco:
        _required_tools.append("busco")
    if args.kmer_eval:
        _required_tools.append("yak")

    _missing_tools = [t for t in _required_tools if shutil.which(t) is None]
    if _missing_tools:
        logging.error(
            "Dry-run: required tool(s) not found on PATH: " + ", ".join(_missing_tools)
        )
        sys.exit(1)

    _dry_cmd = (
        build_hifiasm_command(
            prefix=str(paths.hifiasm_prefix),
            haploid_genome_size=KNOWN_GENOME_SIZE,
            threads=threads,
            primary=args.primary,
            hic1=hic1,
            hic2=hic2,
            ul=ul,
            ont=args.ont,
            hom_cov=args.hom_cov,
        )
        + f" {input_reads}"
    )

    logging.info(
        "Dry-run checks passed.\n"
        f"  Input reads : {input_reads}\n"
        f"  Genome size : {KNOWN_GENOME_SIZE} Mb\n"
        f"  Threads     : {threads}\n"
        f"  Output dir  : {output_dir}\n"
        f"  Work dir    : {paths.work_dir}\n"
        f"  Tools found : {', '.join(_required_tools)}\n"
        f"  Trial-0 command (default params):\n    {_dry_cmd}"
    )
    sys.exit(0)

# ------------------------------------------------------------------ run setup
logging.info(f"Output directory : {output_dir}")
logging.info(f"Working directory: {paths.work_dir} (intermediates, safe to delete)")



def prepare_shared_inputs():
    """
    Build the artefacts every trial reuses: the BUSCO lineage, the yak read
    k-mer hash and the read subset.

    Called once at startup, and again after ``--force-rerun`` wipes ``work/``.
    """
    # A setup step that fails retires its stage instead of killing the run:
    # a missing BUSCO lineage is a reason to score without BUSCO, not a reason
    # to abandon a multi-day assembly optimisation.
    if evaluator.stage_enabled("busco"):
        logging.info("Preparing BUSCO lineage dataset.")
        try:
            evaluator.download_busco(lineage=args.busco_lineage)
        except Exception as e:
            logging.error(f"BUSCO lineage preparation failed: {e}")
            evaluator.retire_stage("busco", e)

    if evaluator.stage_enabled("yak"):
        # The read k-mer hash depends only on the reads, so it is built once
        # here and reused by every trial.
        logging.info("Building yak k-mer hash from the full read set (once).")
        evaluator.build_read_kmer_db()

    logging.info(
        f"Subsetting {args.num_reads} reads for the alignment-based metrics."
    )
    evaluator.read_subsetting(num_reads=args.num_reads)


def reset_previous_run():
    """
    Wipe everything ``--force-rerun`` is supposed to discard.

    This has to happen *before* prepare_shared_inputs(): it deletes
    ``work/``, which is where the read subset, the yak read hash and the BUSCO
    lineage live. Doing it afterwards -- as the previous ordering did -- threw
    away a `yak count` over the full read set and rebuilt it, which on a real
    HiFi dataset is hours of wasted CPU on every forced re-run.
    """
    logging.info("--force-rerun: deleting the existing study and work tree.")
    try:
        optuna.delete_study(study_name="no-name", storage=paths.db_uri)
    except Exception as e:
        logging.warning(
            f"Could not delete existing study (may not exist or storage issue): {e}"
        )

    removed = []
    for target in (
        paths.final_assembly_dir,
        paths.default_assembly_dir,
        paths.optuna_dir,
        paths.work_dir,
    ):
        if target.exists():
            try:
                shutil.rmtree(target)
                removed.append(target.name)
            except Exception as e:
                logging.warning(f"Failed to remove {target}: {e}")

    paths.create()
    if removed:
        logging.info(f"Cleaned up previous run artifacts: {', '.join(removed)}")

    # metric_stage_state.json went with work/, so every retired stage gets
    # another chance; the in-memory copy has to follow it.
    evaluator.stage_state = {}
    evaluator.backend_cache = {}
    evaluator.kmer_eval = args.kmer_eval
    evaluator.include_busco = args.include_busco


if args.force_rerun:
    reset_previous_run()

prepare_shared_inputs()

_retired = evaluator.disabled_stages()
if _retired:
    logging.warning(
        "Metric stage(s) retired by an earlier run and still disabled: "
        + ", ".join(evaluator.STAGES_BY_NAME[n].label for n in _retired)
        + f". State file: {paths.metric_stage_state}. "
        "Use --force-rerun to clear it and retry them."
    )
logging.info(f"Metric weights from: {evaluator.weights_source}")
logging.info(f"Scoring metrics for this run: {evaluator.metric_regime()}")

objective_builder = ObjectiveBuilder(
    evaluator=evaluator,
    input_reads=input_reads,
    haploid_genome_size=KNOWN_GENOME_SIZE,
    threads=threads,
    paths=paths,
    hic1=hic1,
    hic2=hic2,
    ul=ul,
    sensitive=args.sensitive,
    primary=args.primary,
    include_busco=args.include_busco,
    busco_lineage=args.busco_lineage,
    download_path=download_path,
    is_multi_objective=args.multi_objective,
    ont=args.ont,
    trial_walltime_hours=args.trial_walltime,
    stage_walltimes=STAGE_WALLTIMES,
    kmer_eval=evaluator.kmer_eval,
    yak_k=args.kmer_k,
    yak_bloom_bits=args.yak_bloom_bits,
    hom_cov=args.hom_cov,
)
objective = objective_builder.build_objective()

PRIMARY_SUFFIX, EXTRA_SUFFIXES = haplotype_suffixes(hic1, hic2, ul)

# ------------------------------------------------------------------ reporting
def format_metrics(metrics, title="Assembly metrics"):
    """
    Render a metrics dict for the log.

    Metrics are stored as log(value + 1) except those in
    ``AssemblyEvaluator.RAW_METRICS`` (qv, kmer_completeness and the rate-like
    samtools-stats values), which are stored raw. ``AssemblyEvaluator.raw_value``
    knows which is which, so this function never has to.
    """
    rev = AssemblyEvaluator.raw_value

    def val(key):
        return rev(key, metrics.get(key, 0))

    num_contigs = round(val("num_contigs"))
    n50 = round(val("n50"))
    num_sv = round(val("num_sv"))
    length_diff_mb = val("length_diff")

    single_copy = round(val("single_copy"))
    multi_copy = round(val("multi_copy"))
    fragmented = round(val("fragmented"))
    missing = round(val("missing"))
    total_busco = single_copy + multi_copy + fragmented + missing

    def pct(x):
        return (x / total_busco * 100) if total_busco > 0 else 0

    out = f"{title}:\n"
    # Any block can be absent: its stage may have failed or been switched off.
    if "n50" in metrics:
        out += f"  - Number of contigs          : {num_contigs:,}\n"
        out += f"  - Length difference          : {length_diff_mb:.2f} Mb\n"
        out += f"  - N50                        : {n50:,} bp\n"
    if "num_sv" in metrics:
        out += f"  - Large-scale misassemblies  : {num_sv:,}\n"

    if "reads_mapped" in metrics:
        reads_mapped = round(val("reads_mapped"))
        reads_total = round(val("reads_total"))
        out += (
            f"  - Reads mapped               : {reads_mapped:,}"
            f" / {reads_total:,} ({metrics.get('mapped_rate', 0):.2f}%)\n"
        )
        out += (
            f"  - Supplementary alignments   : "
            f"{round(val('supplementary_alignments')):,}\n"
        )
        out += (
            f"  - Error rate                 : "
            f"{metrics.get('error_rate', 0):.3f} mismatches/kb\n"
        )
        out += (
            f"  - Bases mapped / mismatches  : "
            f"{val('bases_mapped') / 1e6:.1f} Mb / {round(val('mismatches')):,}\n"
        )
    if "qv" in metrics:
        out += f"  - Consensus QV (yak)         : {metrics['qv']:.2f}\n"
    if "kmer_completeness" in metrics:
        out += (
            f"  - k-mer completeness (yak)   : "
            f"{metrics['kmer_completeness']:.2f}%\n"
        )

    if total_busco > 0:
        out += f"  - Single-copy BUSCOs         : {pct(single_copy):.2f}%\n"
        out += f"  - Multi-copy BUSCOs          : {pct(multi_copy):.2f}%\n"
        out += f"  - Fragmented BUSCOs          : {pct(fragmented):.2f}%\n"
        out += f"  - Missing BUSCOs             : {pct(missing):.2f}%\n"

    absent = [
        stage.label
        for stage in AssemblyEvaluator.STAGES
        if stage.metrics and not any(m in metrics for m in stage.metrics)
    ]
    if absent:
        out += f"  - Not measured               : {'; '.join(absent)}"
    return out.rstrip("\n")


def collect_assembly_outputs(dest_dir, label):
    """Copy the current hifiasm outputs out of work/ into a results directory."""
    collect_hifiasm_outputs(paths.hifiasm_prefix, dest_dir, label)
    return dest_dir


def run_hifiasm_with_params(params, trial_label, walltime_hours):
    """Rebuild an assembly from a recorded parameter set. Returns the GFA path."""
    kwargs = dict(
        prefix=str(paths.hifiasm_prefix),
        haploid_genome_size=KNOWN_GENOME_SIZE,
        threads=threads,
        sensitive=args.sensitive,
        primary=args.primary,
        hic1=hic1,
        hic2=hic2,
        ul=ul,
        ont=args.ont,
        hom_cov=args.hom_cov,
    )
    for key in HIFIASM_PARAM_KEYS:
        if key in params:
            kwargs[key] = params[key]

    command = build_hifiasm_command(**kwargs) + f" {input_reads}"
    logging.info(f"Running hifiasm ({trial_label}):\n{command}")

    runner = SubprocessLogger(logs_dir=logs_dir)
    rc, log_path = runner.run_command_with_logging(
        command=command,
        log_filename="hifiasm.log",
        command_name="hifiasm",
        trial_id=trial_label,
        timeout_seconds=walltime_hours * 3600,
        cwd=paths.hifiasm_dir,
    )

    if rc == TIMEOUT_EXIT_CODE:
        raise TimeoutError(
            f"hifiasm exceeded the walltime limit ({walltime_hours:.1f} h) and was "
            "killed. Re-run with a larger --trial-walltime value."
        )
    if rc != 0:
        raise RuntimeError(f"hifiasm exited with code {rc}. See log at {log_path}")

    gfa = paths.hifiasm_prefix.parent / f"{paths.hifiasm_prefix.name}.{PRIMARY_SUFFIX}.gfa"
    if not gfa.exists():
        raise FileNotFoundError(f"No GFA found at {gfa} after hifiasm completed.")
    return gfa


def evaluate_and_report(gfa, trial_label, title):
    """Convert, evaluate and log a finished assembly."""
    evaluator.trial_id = trial_label
    fasta = evaluator.trial_dir / f"{Path(gfa).stem}.fasta"
    AssemblyEvaluator.convert_gfa_to_fasta(gfa, fasta)

    extra_fastas = evaluator.convert_extra_haplotypes(
        paths.hifiasm_prefix, EXTRA_SUFFIXES
    )

    metrics = evaluator.evaluate_assembly(
        gfa_file=gfa,
        fasta_file=fasta,
        include_busco=args.include_busco,
        busco_lineage=args.busco_lineage,
        extra_fasta_files=extra_fastas,
    )
    logging.info(format_metrics(metrics, title))
    return metrics


def trial_score(trial, score_key):
    """The comparable score of a trial, or -inf when it has none."""
    try:
        return float(trial.user_attrs.get(score_key, float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def trial_params(trial):
    """Recorded hifiasm parameters, falling back to Optuna's own record."""
    return trial.user_attrs.get("params", dict(trial.params))


def find_best_trial(study, score_key, regime=None):
    """
    Best trial by ``score_key``.

    When a metric stage is retired mid-study the weighted sum stops including
    its metrics, so scores from before and after are on different scales.
    ``regime`` restricts the comparison to trials scored on the same metric
    set; if none match (e.g. an old study without the attribute) the search
    falls back to every trial and says so.
    """
    candidates = list(study.trials)

    if regime is not None:
        same = [t for t in candidates if t.user_attrs.get("metric_regime") == regime]
        if same and len(same) != len(candidates):
            logging.warning(
                f"{len(candidates) - len(same)} trial(s) were scored on a "
                "different metric set (a stage failed part-way through) and are "
                "excluded from the best-trial comparison."
            )
        if same:
            candidates = same

    bt_num = study.user_attrs.get("best_trial", None)
    if bt_num is not None:
        for t in candidates:
            if t.number == bt_num:
                return t, trial_score(t, score_key)

    best_trial, best_score = None, float("-inf")
    for t in candidates:
        score = trial_score(t, score_key)
        if score > best_score:
            best_score, best_trial = score, t
    return best_trial, best_score


# ------------------------------------------------------------------ callbacks
def convergence_callback(study, trial):
    if trial.user_attrs.get("metric_skip"):
        # Discarded, not a data point: feeding it to the detectors would look
        # like a stalled search and could trigger a false convergence.
        return

    current = trial_score(trial, "weighted_score")
    if current == float("-inf"):
        current = trial_score(trial, "aggregate_score")
    if current == float("-inf"):
        logging.warning(
            f"Trial {trial.number} has no usable score for convergence -> skipping"
        )
        return

    has_converged, converged_methods = convergence_detector.update(
        current, trial.number
    )

    if has_converged:
        study.set_user_attr("converged", True)
        study.set_user_attr("converged_methods", converged_methods)
        methods = ", ".join(converged_methods) if converged_methods else "unknown"
        logging.info(
            f"\n{'#' * 66}\n"
            f"Optimization converged at trial: {trial.number}\n"
            f"Convergence detected by: {methods}\n"
            f"{'#' * 66}"
        )
        study.stop()
    else:
        # Info-level on every single trial was pure noise; the interesting
        # event is convergence, not its absence.
        methods = ", ".join(converged_methods) if converged_methods else "none"
        logging.debug(
            f"Majority convergence not yet met. Partial convergence detected by: {methods}"
        )


def metric_skip_callback(study, trial):
    """
    Stop the study once too many trials have been thrown away.

    A trial is discarded when a metric that worked on the baseline assembly
    fails later on (see ``AssemblyEvaluator.failure_policy``). One or two of
    those is bad luck; a steady stream means the inputs or the environment
    changed under the run, and burning another eighty hifiasm assemblies to
    find that out helps nobody.

    Counting from ``study.trials`` rather than an in-memory tally keeps the
    budget correct across a resumed study.
    """
    if args.max_metric_skips <= 0:
        return

    skipped = [
        t for t in study.get_trials(deepcopy=False) if t.user_attrs.get("metric_skip")
    ]
    if not skipped:
        return

    by_stage = {}
    for t in skipped:
        stage = t.user_attrs.get("metric_skip_stage", "unknown")
        by_stage.setdefault(stage, []).append(t.number)

    detail = "; ".join(
        f"{stage} (trials {', '.join(str(n) for n in numbers)})"
        for stage, numbers in sorted(by_stage.items())
    )

    if len(skipped) < args.max_metric_skips:
        logging.warning(
            f"{len(skipped)}/{args.max_metric_skips} trials discarded so far "
            f"because a metric failed after the baseline: {detail}."
        )
        return

    if study.user_attrs.get("halted_reason"):
        return

    reason = (
        f"{len(skipped)} trials discarded because metrics failed after the "
        f"baseline assembly (limit --max-metric-skips={args.max_metric_skips})"
    )
    study.set_user_attr("halted_reason", reason)
    logging.error(
        f"\n{'#' * 72}\n"
        f"OPTIMIZATION STOPPED - too many trials discarded\n"
        f"{'#' * 72}\n"
        f"{len(skipped)} trials had to be thrown away because a metric stage "
        f"failed after it had already succeeded on the baseline assembly:\n"
        f"  {detail}\n"
        "A metric that fails on the baseline is retired quietly and costs no "
        "trials, so these are failures of a different kind: either a tool that "
        "had been working stopped, or an assembly the tools cannot handle. "
        "A data-quality check and a look at the logs are warranted before "
        f"trusting anything below.\n"
        f"  - per-command logs : {logs_dir}\n"
        f"  - stage state      : {paths.metric_stage_state}\n"
        "The best trial so far will still be rebuilt and reported. Raise "
        "--max-metric-skips to push on regardless.\n"
        f"{'#' * 72}"
    )
    study.stop()


def best_tracker_callback(study, trial):
    try:
        score_val = trial_score(trial, "weighted_score")
        if score_val == float("-inf"):
            score_val = trial_score(trial, "aggregate_score")
        if score_val == float("-inf"):
            return

        # A retired metric stage changes what the weighted sum measures, so the
        # incumbent best is no longer comparable and has to be re-established.
        regime = trial.user_attrs.get("metric_regime")
        previous_regime = study.user_attrs.get("best_regime")
        if regime is not None and previous_regime is not None and regime != previous_regime:
            logging.warning(
                "The set of scored metrics changed at trial "
                f"{trial.number} (a metric stage was retired). Resetting the "
                "running best: earlier scores are on a different scale."
            )
            study.set_user_attr("best_score", float("-inf"))
            study.set_user_attr("best_trial", None)

        best_score = study.user_attrs.get("best_score", float("-inf"))
        if best_score is None:
            best_score = float("-inf")

        if score_val > best_score:
            study.set_user_attr("best_score", float(score_val))
            study.set_user_attr("best_trial", trial.number)
            if regime is not None:
                study.set_user_attr("best_regime", regime)
            params = trial_params(trial)
            try:
                with open(paths.best_params_checkpoint, "w") as _f:
                    json.dump(
                        {
                            "trial": trial.number,
                            "score": float(score_val),
                            "params": params,
                            "metric_regime": regime,
                        },
                        _f,
                        indent=2,
                    )
            except Exception as cp_err:
                logging.debug(f"best_tracker_callback: checkpoint write failed: {cp_err}")
            logging.info(
                f"New best so far: trial {trial.number} score={score_val:.4f} "
                f"params={params}"
            )
    except Exception as e:
        logging.debug(f"best_tracker_callback: unexpected error: {e}")


def rerun_from_params(params, label, banner, title):
    """
    Rebuild, archive and evaluate an assembly from a recorded parameter set.

    Shared by ``--rerun-best`` and ``--rerun-trial``, which differed only in
    how they found the parameters. Never returns: it exits the process.
    """
    logging.info(f"\n{'#' * 60}\n{banner}\n{'#' * 60}")
    try:
        gfa = run_hifiasm_with_params(params, label, args.trial_walltime)
    except Exception as e:
        logging.error(f"{label}: {e}")
        sys.exit(1)

    collect_assembly_outputs(paths.final_assembly_dir, "final_assembly")
    try:
        evaluate_and_report(gfa, label, title)
    except Exception as e:
        logging.error(f"{label}: assembly evaluation failed: {e}")
    sys.exit(0)


# --------------------------------------------------------------- optimization
best_trial = None
best_score = float("-inf")

try:
    if args.multi_objective:
        objective_keys = objective_builder.objectives
        directions_file = Path(__file__).resolve().parent / "optim_directions.json"
        try:
            if not directions_file.exists():
                raise FileNotFoundError(
                    f"optim_directions.json not found at {directions_file}"
                )
            with open(directions_file, "r") as fh:
                mapping = json.load(fh) or {}

            directions = []
            for k in objective_keys:
                v = mapping.get(k, None)
                if v in ("maximize", "minimize"):
                    directions.append(v)
                elif v is None:
                    raise ValueError(
                        f"Missing direction for metric '{k}' in optim_directions.json"
                    )
                else:
                    raise ValueError(
                        f"Invalid direction '{v}' for metric '{k}'; "
                        "must be 'maximize' or 'minimize'"
                    )
        except Exception as e:
            logging.error(f"Failed to load optim_directions.json: {e}")
            raise

        convergence_detector = MultiCriteriaConvergenceDetector(
            directions=directions,
            min_trials=args.convergence_warmup,
            stagnation_patience=10,
            min_improvement=0,
            threshold=0.01,
            patience=10,
            plateau_threshold=1e-3,
            min_plateau_length=10,
            window_size=10,
            significance_level=0.05,
        )

        try:
            import optunahub  # lazy: contacts the network, offline-hostile

            moead_module = optunahub.load_module(
                package="samplers/moead", force_reload=False
            )
            sampler = moead_module.MOEADSampler(seed=args.seed)
            logging.info("Using MOEAD sampler for multi-objective optimization")
        except Exception as e:
            logging.warning(f"MOEAD not available ({e}), falling back to NSGAIIISampler")
            sampler = optuna.samplers.NSGAIIISampler(seed=args.seed)
    else:
        directions = ["maximize"]
        convergence_detector = MultiCriteriaConvergenceDetector(
            directions=directions,
            min_trials=args.convergence_warmup,
            stagnation_patience=15,
            min_improvement=0,
            threshold=0.01,
            patience=15,
            plateau_threshold=1e-3,
            min_plateau_length=15,
            window_size=15,
            significance_level=0.05,
        )
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        logging.info("Using TPE sampler for single-objective optimization (weighted score)")

    load_if_exists = not args.force_rerun

    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna.logging.disable_default_handler()
        logging.getLogger("optuna").setLevel(logging.WARNING)
        logging.getLogger("optuna").propagate = False
    except Exception:
        pass

    study = optuna.create_study(
        study_name="no-name",
        directions=directions,
        storage=paths.db_uri,
        load_if_exists=load_if_exists,
        sampler=sampler,
    )

    if not args.force_rerun and len(study.trials) > 0:
        logging.info(
            f"Resuming existing study with {len(study.trials)} trials. "
            "Use --force-rerun to start fresh."
        )

    score_key = "aggregate_score" if args.multi_objective else "weighted_score"

    # ----------------------------------------------------------- rerun-best
    if args.rerun_best:
        if len(study.trials) == 0:
            logging.error("No trials found in the existing study. Cannot rerun best assembly.")
            sys.exit(1)

        _converged = study.user_attrs.get("converged", False)
        _has_best = study.user_attrs.get("best_trial", None) is not None
        if not _converged and not _has_best:
            logging.error(
                "--rerun-best requested but neither a convergence flag nor a stored "
                "best_trial was found in this study. Run the full optimisation first."
            )
            sys.exit(1)
        if not _converged and _has_best:
            logging.warning(
                "--rerun-best: no explicit convergence flag found in this study. "
                "Proceeding with the best recorded trial anyway."
            )

        evaluator.reload_stage_state()
        _trial, _score = find_best_trial(study, score_key, evaluator.metric_regime())
        if _trial is None:
            logging.error(f"Could not identify a best trial (score key: '{score_key}').")
            sys.exit(1)

        _params = trial_params(_trial)
        rerun_from_params(
            _params,
            "rerun_best",
            f"--rerun-best mode\nBest trial : {_trial.number}\n"
            f"Score      : {_score:.4f}\nParams     : {_params}",
            "Rerun assembly metrics",
        )

    # ---------------------------------------------------------- rerun-trial
    if getattr(args, "rerun_trial", None) is not None:
        _num = args.rerun_trial
        if len(study.trials) == 0:
            logging.error("No trials found in the existing study. Cannot rerun trial.")
            sys.exit(1)

        _target = next((t for t in study.trials if t.number == _num), None)
        if _target is None:
            logging.error(
                f"Trial {_num} not found. Available: {[t.number for t in study.trials]}"
            )
            sys.exit(1)

        _params = trial_params(_target)
        _score = trial_score(_target, score_key)
        rerun_from_params(
            _params,
            f"rerun_trial_{_num}",
            f"--rerun-trial mode\nTrial  : {_num}\n"
            f"Score  : {_score:.4f}\nParams : {_params}",
            f"Trial {_num} rerun assembly metrics",
        )

    # ----------------------------------------------------------- optimize
    mode_str = "multi-objective" if args.multi_objective else "single-objective (weighted score)"
    logging.info(
        "Starting Optuna %s optimization with up to %d trials.", mode_str, args.num_trials
    )
    study.optimize(
        objective,
        n_trials=args.num_trials,
        callbacks=[metric_skip_callback, best_tracker_callback, convergence_callback],
    )

    # Count COMPLETE trials, not all of them: a study where every trial was
    # pruned or discarded still has a non-empty `trials` list, and asking it for
    # `best_value` in that state raises. Our own weighted_score bookkeeping is
    # the authority anyway, so find_best_trial does the work.
    _completed = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    _pruned = len(study.get_trials(deepcopy=False)) - len(_completed)

    if not _completed:
        logging.error(
            f"No trial completed successfully ({_pruned} pruned or discarded). "
            f"There is nothing to select a best assembly from; check the logs "
            f"in {logs_dir}."
        )
    else:
        _tally = f"{len(_completed)} completed" + (
            f", {_pruned} pruned or discarded" if _pruned else ""
        )
        if args.multi_objective:
            logging.info(
                f"Multi-objective optimization finished ({_tally}). "
                "Use the Pareto front to select a preferred solution."
            )
        else:
            logging.info(f"Single-objective optimization finished ({_tally}).")

        # Trials may have retired a stage since this evaluator was built;
        # without re-reading, the regime below is the one we started with and
        # the comparison picks from the wrong group of trials.
        evaluator.reload_stage_state()
        best_trial, best_score = find_best_trial(
            study, score_key, evaluator.metric_regime()
        )

        _halted = study.user_attrs.get("halted_reason")
        if _halted:
            logging.error(f"Study stopped early: {_halted}.")
        if best_trial is not None:
            logging.info(
                f"\n{'#' * 44}\n\nBest trial: {best_trial.number}\n"
                f"Score: {best_score:.2f}\n"
                f"Params: {trial_params(best_trial)}\n\n{'#' * 44}"
            )

except Exception as e:
    logging.error(f"Optimization failed: {e}", exc_info=True)
    raise

# ------------------------------------------------------- OPTUNA VISUALIZATIONS
optuna_dir = paths.optuna_dir
optuna_dir.mkdir(exist_ok=True, parents=True)

# Trial 0 (the default-parameter baseline) has no sampled parameters, which
# collapses the intersection search space used by plot_param_importances to the
# empty set. The helper below plots from a filtered copy of the study so the
# baseline stays visible in the optimization history while the importances are
# still computed.
write_param_importances(
    study,
    optuna_dir,
    objectives=objective_builder.objectives,
    multi_objective=args.multi_objective,
)

if args.multi_objective:
    for idx, metric in enumerate(objective_builder.objectives):
        try:
            metric_dir = optuna_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)

            # `i=idx` binds the loop variable at definition time; a bare `idx`
            # would let every lambda see the final value.
            for plot_fn, fname, label in (
                (vis.plot_optimization_history, "optuna_optimization_history.html", "optimization history"),
                (vis.plot_parallel_coordinate, "optuna_parallel_coordinates.html", "parallel coordinate"),
                (vis.plot_contour, "optuna_contour_plot.html", "contour"),
            ):
                try:
                    plot_fn(
                        study,
                        target=lambda t, i=idx: t.values[i],
                        target_name=metric,
                    ).write_html(metric_dir / fname)
                except Exception as e:
                    logging.warning(f"[{metric}] Failed to create {label} plot: {e}")
        except Exception as e:
            logging.warning(f"Failed to create {metric} visualizations: {e}")

    try:
        if len(objective_builder.objectives) <= 3:
            vis.plot_pareto_front(study).write_html(optuna_dir / "optuna_pareto_front.html")
        else:
            logging.info(
                f"Skipping Pareto front plot: {len(objective_builder.objectives)} "
                "objectives exceed the 3-objective limit"
            )
    except Exception as e:
        logging.warning(f"Failed to create pareto front plot: {e}")
else:
    try:
        vis.plot_optimization_history(study).write_html(
            optuna_dir / "optimization_history.html"
        )
        vis.plot_parallel_coordinate(study).write_html(
            optuna_dir / "parallel_coordinate.html"
        )
        vis.plot_slice(study).write_html(optuna_dir / "slice.html")
        if len(study.best_params) >= 2:
            vis.plot_contour(study).write_html(optuna_dir / "contour.html")
    except Exception as e:
        logging.warning(f"Failed to create single-objective plots: {e}")

# Per-metric histories, from user_attrs, for both modes. Trial 0 appears here.
try:
    metric_dir = optuna_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    for metric in objective_builder.objectives:
        xs, ys = [], []
        for t in study.trials:
            v = t.user_attrs.get(metric, None)
            if v is None:
                continue
            xs.append(t.number)
            ys.append(v)
        if len(ys) < 2:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=metric))
        fig.update_layout(
            title=f"Metric History: {metric} (trial 0 = default-parameter baseline)",
            xaxis_title="Trial",
            yaxis_title=metric,
        )
        fig.write_html(metric_dir / f"{metric}.history.html")
except Exception as e:
    logging.warning(f"Failed to create per-metric plots: {e}")

# ----------------------------------- FINAL ASSEMBLY with the best parameters
try:
    if best_trial is None:
        logging.info("No best trial identified; skipping final assembly run.")
    else:
        best_params = trial_params(best_trial)
        try:
            final_gfa = run_hifiasm_with_params(best_params, "best", args.trial_walltime)
        except Exception as e:
            logging.error(f"Final assembly failed: {e}")
            final_gfa = None

        if final_gfa is not None:
            collect_assembly_outputs(paths.final_assembly_dir, "final_assembly")
            try:
                evaluate_and_report(final_gfa, "best", "Final assembly metrics")
            except Exception as e:
                logging.error(f"Final assembly evaluation failed: {e}")
except Exception as e:
    logging.error(f"Final assembly/evaluation failed: {e}")

# --------------------------------------------------------- METRIC HEALTH
# Repeated here because a stage retired at trial 0 may be thousands of log
# lines back by the time the run ends.
try:
    evaluator.reload_stage_state()
    _health = evaluator.metric_health_report()
    if _health:
        logging.error(_health)
except Exception as e:
    logging.debug(f"Could not render the metric health report: {e}")

# ------------------------------------------------------------- FINAL CLEANUP
try:
    logging.info("Performing final cleanup of intermediate files...")
    if paths.trials_dir.exists():
        shutil.rmtree(paths.trials_dir)
    paths.trials_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        f"Results are in {output_dir}. Intermediates (including hifiasm .bin files, "
        f"the yak read hash and the BUSCO datasets) remain in {paths.work_dir}; "
        "delete it to reclaim space, at the cost of a full hifiasm recompute on the "
        "next run."
    )
except Exception as e:
    logging.warning(f"Final cleanup encountered issues: {e}")