#!/usr/bin/env python3

import json
import logging
import os
import random
import shutil
import signal
import sys
from pathlib import Path

import numpy as np
import optuna
import optuna.visualization as vis
import optunahub
import plotly.graph_objects as go
import psutil

from utils.assembly_eval import AssemblyEvaluator
from utils.argparser import get_args
from utils.hifiasm_command import build_hifiasm_command, run_default_hifiasm_assembly
from utils.objective import ObjectiveBuilder, haplotype_suffixes
from utils.optuna_callback import MultiCriteriaConvergenceDetector
from utils.optuna_plots import write_param_importances
from utils.paths import RunPaths
from utils.subprocess_logger import SubprocessLogger, TIMEOUT_EXIT_CODE

# ---------------------------------------------------------------- termination
TERMINATE_REQUESTED = False


def terminate_all_processes(sig, frame):
    global TERMINATE_REQUESTED
    TERMINATE_REQUESTED = True
    parent = psutil.Process()
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except Exception:
            pass
    logging.info("Termination signal received. All child processes killed.")
    sys.exit(0)


def get_terminate_status():
    return TERMINATE_REQUESTED


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

KNOWN_GENOME_SIZE = args.genome_size  # megabases
ploidy = args.ploidy
GENOME_SIZE_BP = KNOWN_GENOME_SIZE * 1_000_000

evaluator = AssemblyEvaluator(
    known_genome_size=GENOME_SIZE_BP,
    input_reads=input_reads,
    paths=paths,
    threads=threads,
    download_path=download_path,
    ont=args.ont,
    busco_walltime_hours=args.busco_walltime,
    craq_walltime_hours=args.craq_walltime,
    craq_mapq=args.craq_mapq,
    kmer_eval=args.kmer_eval,
    yak_k=args.kmer_k,
    yak_bloom_bits=args.yak_bloom_bits,
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
    _required_tools = ["hifiasm", "gfastats", "craq", "minimap2", "samtools", "sniffles"]
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

if args.include_busco:
    logging.info("Preparing BUSCO lineage dataset.")
    evaluator.download_busco(lineage=args.busco_lineage)

if args.kmer_eval:
    # The read k-mer hash depends only on the reads, so it is built once here
    # and reused by every trial.
    logging.info("Building yak k-mer hash from the full read set (once).")
    evaluator.build_read_kmer_db()

logging.info(f"Subsetting {args.num_reads} reads for alignment-based metrics.")
evaluator.read_subsetting(num_reads=args.num_reads)

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
    busco_walltime_hours=args.busco_walltime,
    craq_walltime_hours=args.craq_walltime,
    craq_mapq=args.craq_mapq,
    kmer_eval=evaluator.kmer_eval,
    hom_cov=args.hom_cov,
)
objective = objective_builder.build_objective()

PRIMARY_SUFFIX, EXTRA_SUFFIXES = haplotype_suffixes(hic1, hic2, ul)

HIFIASM_PARAM_KEYS = [
    "x", "y", "s", "n", "m", "p", "u",
    "D", "N", "max_kocc",
    "s_base", "f_perturb", "l_msjoin",
    "path_max", "path_min",
]


# ------------------------------------------------------------------ reporting
def format_metrics(metrics, title="Assembly metrics"):
    """
    Render a metrics dict for the log.

    Metrics are stored as log(value + 1) except those in
    ``AssemblyEvaluator.RAW_METRICS`` (qv, kmer_completeness), which are stored
    raw and must not be inverse-transformed.
    """

    def rev(key):
        v = metrics.get(key, 0)
        return max(0, np.exp(v) - 1) if v else 0

    num_contigs = int(rev("num_contigs"))
    n50 = int(rev("n50"))
    num_sv = int(rev("num_sv"))
    length_diff_mb = rev("length_diff")

    single_copy = int(rev("single_copy"))
    multi_copy = int(rev("multi_copy"))
    fragmented = int(rev("fragmented"))
    missing = int(rev("missing"))
    total_busco = single_copy + multi_copy + fragmented + missing

    def pct(x):
        return (x / total_busco * 100) if total_busco > 0 else 0

    out = f"{title}:\n"
    out += f"  - Number of contigs          : {num_contigs}\n"
    out += f"  - Length difference          : {length_diff_mb:.2f} Mb\n"
    out += f"  - N50                        : {n50} bp\n"
    out += f"  - Large-scale misassemblies  : {num_sv}\n"

    if "aqi" in metrics:
        out += f"  - AQI (CRAQ)                 : {metrics['aqi']:.2f}\n"
        out += (
            f"  - R-AQI / S-AQI              : "
            f"{metrics.get('r_aqi', 0):.2f} / {metrics.get('s_aqi', 0):.2f}\n"
        )
        out += (
            f"  - CRE / CSE per Mb           : "
            f"{metrics.get('cre_per_mb', 0):.3f} / {metrics.get('cse_per_mb', 0):.3f}\n"
        )
        out += (
            f"  - Covered / low-confidence   : "
            f"{metrics.get('craq_covered_rate', 0) * 100:.1f}% / "
            f"{metrics.get('craq_low_conf_rate', 0) * 100:.1f}%\n"
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
        out += f"  - Missing BUSCOs             : {pct(missing):.2f}%"
    return out.rstrip("\n")


def collect_assembly_outputs(dest_dir, label):
    """Copy the current hifiasm outputs out of work/ into a results directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = paths.hifiasm_prefix
    copied = 0
    for f in sorted(prefix.parent.glob(f"{prefix.name}*")):
        if not f.is_file() or f.suffix == ".bin":
            continue
        target = dest_dir / f"{label}{f.name[len(prefix.name):]}"
        try:
            shutil.copy2(f, target)
            copied += 1
        except Exception as e:
            logging.warning(f"Could not copy {f.name} -> {target}: {e}")
    logging.info(f"Copied {copied} file(s) to {dest_dir}")
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
    tdir = evaluator.trial_dir
    fasta = tdir / f"{Path(gfa).stem}.fasta"
    AssemblyEvaluator.convert_gfa_to_fasta(gfa, fasta)

    extra_fastas = []
    for extra in EXTRA_SUFFIXES:
        extra_gfa = (
            paths.hifiasm_prefix.parent
            / f"{paths.hifiasm_prefix.name}.{extra}.gfa"
        )
        if extra_gfa.exists():
            extra_fasta = tdir / f"{extra}.fasta"
            AssemblyEvaluator.convert_gfa_to_fasta(extra_gfa, extra_fasta)
            extra_fastas.append(extra_fasta)

    metrics = evaluator.evaluate_assembly(
        gfa_file=gfa,
        fasta_file=fasta,
        include_busco=args.include_busco,
        busco_lineage=args.busco_lineage,
        extra_fasta_files=extra_fastas,
    )
    logging.info(format_metrics(metrics, title))
    return metrics


def find_best_trial(study, score_key):
    best_trial, best_score = None, float("-inf")

    bt_num = study.user_attrs.get("best_trial", None)
    if bt_num is not None:
        for t in study.trials:
            if t.number == bt_num:
                return t, float(t.user_attrs.get(score_key, float("-inf")))

    for t in study.trials:
        try:
            s = float(t.user_attrs.get(score_key, float("-inf")))
        except (TypeError, ValueError):
            continue
        if s > best_score:
            best_score, best_trial = s, t
    return best_trial, best_score


# ------------------------------------------------------------------ callbacks
def convergence_callback(study, trial):
    current = trial.user_attrs.get("weighted_score", None)
    if current is None:
        current = trial.user_attrs.get("aggregate_score", None)
    if current is None:
        logging.warning(
            f"Trial {trial.number} has no usable score for convergence -> skipping"
        )
        return

    has_converged, converged_methods = convergence_detector.update(
        float(current), trial.number
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
        methods = ", ".join(converged_methods) if converged_methods else "none"
        logging.info(
            f"Majority convergence not yet met. Partial convergence detected by: {methods}"
        )


def best_tracker_callback(study, trial):
    try:
        score_val = trial.user_attrs.get("weighted_score", None)
        if score_val is None:
            score_val = trial.user_attrs.get("aggregate_score", None)
        if score_val is None:
            return

        best_score = study.user_attrs.get("best_score", float("-inf"))
        if score_val > best_score:
            study.set_user_attr("best_score", float(score_val))
            study.set_user_attr("best_trial", trial.number)
            params = trial.user_attrs.get("params", dict(trial.params))
            try:
                with open(paths.best_params_checkpoint, "w") as _f:
                    json.dump(
                        {
                            "trial": trial.number,
                            "score": float(score_val),
                            "params": params,
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

    load_if_exists = True
    if args.force_rerun:
        load_if_exists = False
        try:
            logging.info("--force-rerun specified: deleting existing Optuna study if present.")
            optuna.delete_study(study_name="no-name", storage=paths.db_uri)
        except Exception as e:
            logging.warning(
                f"Could not delete existing study (may not exist or storage issue): {e}"
            )

        removed_items = []
        for target in (
            paths.final_assembly_dir,
            paths.default_assembly_dir,
            paths.optuna_dir,
            paths.work_dir,
        ):
            if target.exists():
                try:
                    shutil.rmtree(target)
                    removed_items.append(target.name)
                except Exception as e:
                    logging.warning(f"Failed to remove {target}: {e}")

        paths.create()
        if removed_items:
            logging.info(
                f"Cleaned up previous run artifacts: {', '.join(removed_items)}"
            )
        # The work tree was just wiped; rebuild the shared setup artefacts.
        if args.include_busco:
            evaluator.download_busco(lineage=args.busco_lineage)
        if evaluator.kmer_eval:
            evaluator.build_read_kmer_db()
        evaluator.read_subsetting(num_reads=args.num_reads)

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

    # ------------------------------------------------------ rerun-* guards
    if getattr(args, "rerun_trial", None) is not None and args.rerun_best:
        logging.error("--rerun-trial and --rerun-best are mutually exclusive.")
        sys.exit(1)
    if getattr(args, "rerun_trial", None) is not None and args.force_rerun:
        logging.error("--rerun-trial and --force-rerun are mutually exclusive.")
        sys.exit(1)
    if args.rerun_best and args.force_rerun:
        logging.error("--rerun-best and --force-rerun are mutually exclusive.")
        sys.exit(1)

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

        _trial, _score = find_best_trial(study, score_key)
        if _trial is None:
            logging.error(f"Could not identify a best trial (score key: '{score_key}').")
            sys.exit(1)

        _params = _trial.user_attrs.get("params", dict(_trial.params))
        logging.info(
            f"\n{'#' * 60}\n--rerun-best mode\n"
            f"Best trial : {_trial.number}\n"
            f"Score      : {_score:.4f}\nParams     : {_params}\n{'#' * 60}"
        )

        try:
            _gfa = run_hifiasm_with_params(_params, "rerun_best", args.trial_walltime)
        except Exception as e:
            logging.error(f"--rerun-best: {e}")
            sys.exit(1)

        collect_assembly_outputs(paths.final_assembly_dir, "final_assembly")
        try:
            evaluate_and_report(_gfa, "rerun_best", "Rerun assembly metrics")
        except Exception as e:
            logging.error(f"--rerun-best: assembly evaluation failed: {e}")
        sys.exit(0)

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

        _params = _target.user_attrs.get("params", dict(_target.params))
        _score = _target.user_attrs.get(score_key, float("-inf"))
        logging.info(
            f"\n{'#' * 60}\n--rerun-trial mode\n"
            f"Trial  : {_num}\nScore  : {_score:.4f}\nParams : {_params}\n{'#' * 60}"
        )

        try:
            _gfa = run_hifiasm_with_params(
                _params, f"rerun_trial_{_num}", args.trial_walltime
            )
        except Exception as e:
            logging.error(f"--rerun-trial: {e}")
            sys.exit(1)

        collect_assembly_outputs(paths.final_assembly_dir, "final_assembly")
        try:
            evaluate_and_report(
                _gfa, f"rerun_trial_{_num}", f"Trial {_num} rerun assembly metrics"
            )
        except Exception as e:
            logging.error(f"--rerun-trial: assembly evaluation failed: {e}")
        sys.exit(0)

    # ----------------------------------------------------------- optimize
    mode_str = "multi-objective" if args.multi_objective else "single-objective (weighted score)"
    logging.info(
        "Starting Optuna %s optimization with up to %d trials.", mode_str, args.num_trials
    )
    study.optimize(
        objective,
        n_trials=args.num_trials,
        callbacks=[best_tracker_callback, convergence_callback],
    )

    if len(study.trials) == 0:
        logging.info("No successful trials were completed.")
    else:
        if args.multi_objective:
            logging.info(
                f"Multi-objective optimization completed with {len(study.trials)} trials. "
                "Use the Pareto front to select a preferred solution."
            )
        else:
            logging.info(
                f"Single-objective optimization completed with {len(study.trials)} "
                f"trials. Best weighted score: {study.best_value:.2f}"
            )

        best_trial, best_score = find_best_trial(study, score_key)
        if best_trial is not None:
            logging.info(
                f"\n{'#' * 44}\n\nBest trial: {best_trial.number}\n"
                f"Score: {best_score:.2f}\n"
                f"Params: {best_trial.user_attrs.get('params', {})}\n\n{'#' * 44}"
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
        best_params = best_trial.user_attrs.get("params", dict(best_trial.params))
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