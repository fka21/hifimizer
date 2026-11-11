#!/usr/bin/env python3

import fcntl
import json
import logging
import os
import random
import shutil
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# NOTE: `optunahub` is deliberately not used. It is a hub client that contacts
# GitHub, and on an offline compute node behind a firewall that drops (rather
# than rejects) packets that is a multi-minute stall with no output. It was
# only needed for the MOEA/D sampler, which --sampler replaced.

from utils.assembly_eval import AssemblyEvaluator
from utils.argparser import get_args
from utils.hifiasm_command import (
    HIFIASM_PARAM_KEYS,
    parse_hifiasm_extra,
    build_hifiasm_command,
    collect_hifiasm_outputs,
    delete_hifiasm_bins,
    effective_hifiasm_params,
    run_default_hifiasm_assembly,
)
from utils.objective import (
    ObjectiveBuilder,
    haplotype_suffixes,
    log_metric_report_legend,
    log_weight_groups,
)
from utils.optuna_callback import MultiCriteriaConvergenceDetector
from utils.samplers import SAMPLER_DESCRIPTIONS, build_sampler
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
def acquire_output_lock(directory):
    """
    Hold an exclusive lock on ``directory`` for the life of this process.

    Two runs sharing an output directory destroy each other: a second run's
    --force-rerun deletes the first one's study database and work tree
    mid-trial (SQLite then reports "attempt to write a readonly database").
    The lock is released by the kernel when the process exits, however it
    exits, so a killed run never leaves a stale lock behind. Returns the open
    lock file, which must stay referenced; exits if another run holds it.
    """
    lock_path = Path(directory) / ".hifimizer.lock"
    handle = open(lock_path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        holder = handle.read().strip() or "unknown process"
        logging.error(
            f"Another hifimizer run is already using {directory} ({holder}). "
            "Running two studies in one output directory corrupts both; wait "
            "for it to finish, stop it, or use a different --output-dir."
        )
        sys.exit(1)
    except OSError as e:
        # Some network filesystems do not support flock at all.
        logging.warning(
            f"Could not lock {lock_path} ({e}); make sure no other hifimizer "
            "run uses this output directory."
        )
        return handle
    import socket
    from datetime import datetime

    handle.seek(0)
    handle.truncate()
    handle.write(
        f"pid {os.getpid()} on {socket.gethostname()}, started "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}: {' '.join(sys.argv)}\n"
    )
    handle.flush()
    return handle



def start_run_logs(root, update_latest=True):
    """
    A fresh log folder for this run, ``logs/run_<time>_<pid>/``.

    Logs used to be appended to one file per command for the life of the
    output directory -- across resumes and --force-rerun alike -- so the first
    block in ``trial_0_hifiasm.log`` could be from a run days older than the
    one in progress. ``logs/latest`` points at the newest real run.
    """
    from datetime import datetime

    run_dir = Path(root) / f"run_{datetime.now():%Y%m%d-%H%M%S}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if update_latest:
        latest = Path(root) / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.name)
        except OSError:
            pass
    return run_dir


# Final results live directly under output_dir; every intermediate lives under
# output_dir/work and can be deleted without losing a result.
paths = RunPaths(args.output_dir).create()
output_dir = paths.output_dir

# Taken before anything is created, deleted or written -- this run's log
# folder and logs/latest included. --dry-run only reads, so it may validate
# inputs while a real run is in progress.
OUTPUT_LOCK = None if args.dry_run else acquire_output_lock(output_dir)

paths.logs_dir = start_run_logs(paths.logs_dir, update_latest=not args.dry_run)
logs_dir = paths.logs_dir

# Where the command was started, for resolving --weights and finding a
# weights.json next to the user's command -- before the chdir below hides it.
LAUNCH_DIR = Path.cwd()

# Kept for backwards compatibility with relative paths a user may pass on the
# command line after this point; nothing hifimizer writes depends on the CWD.
os.chdir(output_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(logs_dir / "main.log"), logging.StreamHandler()],
)

# The exact command this run was launched with, as the very first line of
# main.log: reproducing a run or diffing two runs' flags otherwise means
# reconstructing the invocation from args.* after the fact.
logging.info(f"Command: {' '.join(sys.argv)}")

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
GENOME_SIZE_BP = round(KNOWN_GENOME_SIZE * 1_000_000)

# Per-stage wall-clock budgets. A stage that blows its budget is killed,
# loses only its own metrics, and is retired for the rest of the study.
STAGE_WALLTIMES = {
    "gfastats": args.gfastats_walltime,
    "yak": args.yak_walltime,
    "busco": args.busco_walltime,
    "hic_phasing": args.hic_phasing_walltime,
}

def resolve_weights_file():
    """
    The weights.json this run scores with, or None for the bundled defaults.

    --weights PATH if given (relative to where the command was started), else a
    weights.json in the launch directory, else one in the output directory.
    """
    if args.weights:
        path = Path(args.weights)
        path = path if path.is_absolute() else LAUNCH_DIR / path
        if not path.is_file():
            logging.error(f"--weights: {path} does not exist.")
            sys.exit(1)
        return path.resolve()
    for path in (LAUNCH_DIR / "weights.json", output_dir / "weights.json"):
        if path.is_file():
            return path.resolve()
    return None


evaluator = AssemblyEvaluator(
    weights_file=resolve_weights_file(),
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
    compleasm_bin=args.compleasm_bin,
    compleasm_download_path=args.compleasm_download_path,
    use_compleasm=args.use_compleasm,
    subset_seed=args.seed,
    hic1=hic1,
    hic2=hic2,
    num_hic_reads=args.num_hic_reads,
    hic_min_mapq=args.hic_min_mapq,
    hic_phasing=args.hic_phasing,
    # --primary collapses the output to a single contig set, so the
    # per-haplotype metric names would be asked for and never appear.
    n_haplotypes=2 if ((hic1 and hic2) or (ul and not args.primary)) else 1,
    bits_cap=args.bits_cap,
)

if args.burn_in_trials is not None:
    logging.warning(
        "--burn-in-trials is ignored: metrics are now scored in bits, which "
        "are scale-free, so there is no per-metric scale to measure before "
        "optimising. TPE's own random startup trials take its place."
    )

if args.sensitive:
    logging.warning(
        "--sensitive: -D, -N and --max-kocc only act on hifiasm's overlap and "
        "error-correction stages, which it skips when cached .bin files exist. "
        "The cache is therefore deleted before every hifiasm run, and every "
        "trial recomputes error correction from the raw reads -- expect each "
        "one to take several times longer than without --sensitive."
    )

# User-supplied hifiasm arguments. Parsed once, up front, so a typo fails
# before any assembly runs rather than on the first hifiasm invocation.
try:
    PINNED_HIFIASM_PARAMS, HIFIASM_PASSTHROUGH = parse_hifiasm_extra(args.hifiasm_extra)
except ValueError as e:
    logging.error(f"--hifiasm-extra: {e}")
    sys.exit(1)

if PINNED_HIFIASM_PARAMS:
    logging.info(
        "Pinned by --hifiasm-extra, fixed for every run and not optimized: "
        + ", ".join(f"{k}={v}" for k, v in sorted(PINNED_HIFIASM_PARAMS.items()))
    )
if HIFIASM_PASSTHROUGH:
    logging.info(
        "Passed to hifiasm unchanged on every run: "
        + " ".join(HIFIASM_PASSTHROUGH)
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
        pinned=PINNED_HIFIASM_PARAMS,
        extra_args=HIFIASM_PASSTHROUGH,
    )
    sys.exit(0)

# ------------------------------------------------------------------- dry run
if args.dry_run:
    # minimap2/samtools/seqtk are still needed, but only for the Hi-C phasing
    # metric now that the read-to-assembly alignment is gone.
    _required_tools = ["hifiasm", "gfastats"]
    if args.hic1 and args.hic2 and args.hic_phasing:
        _required_tools += ["minimap2", "samtools", "seqtk"]
    if args.include_busco:
        # BUSCO is required either way: it is the fallback even when compleasm
        # is the backend that will normally run.
        _required_tools.append("busco")
    if args.kmer_eval:
        _required_tools.append("yak")

    _missing_tools = [t for t in _required_tools if shutil.which(t) is None]
    if _missing_tools:
        logging.error(
            "Dry-run: required tool(s) not found on PATH: " + ", ".join(_missing_tools)
        )
        sys.exit(1)

    if args.include_busco:
        # compleasm lives in its own conda environment and is deliberately not
        # on PATH, so shutil.which() is the wrong check for it.
        _compleasm = evaluator.compleasm_bin if args.use_compleasm else None
        if _compleasm:
            logging.info(f"  compleasm   : {_compleasm} (primary backend)")
        else:
            logging.warning(
                "  compleasm   : not found; completeness will be scored with "
                "BUSCO (metaeuk, then augustus), which is slower and needs "
                "considerably more memory. See --compleasm-bin."
                if args.use_compleasm
                else "  compleasm   : disabled by --no-compleasm"
            )

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
            # The command trial 0 will really run: hifiasm's defaults plus
            # whatever --hifiasm-extra pins or passes through.
            extra_args=HIFIASM_PASSTHROUGH,
            **effective_hifiasm_params(pinned=PINNED_HIFIASM_PARAMS),
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
        logging.info("Preparing the gene-space completeness lineage dataset.")
        try:
            evaluator.prepare_completeness_dataset(lineage=args.busco_lineage)
        except Exception as e:
            logging.error(f"Completeness lineage preparation failed: {e}")
            evaluator.retire_stage("busco", e)

    if evaluator.stage_enabled("yak"):
        # The read k-mer hash depends only on the reads, so it is built once
        # here and reused by every trial.
        logging.info("Building yak k-mer hash from the full read set (once).")
        evaluator.build_read_kmer_db()

    # Only the Hi-C subset is still drawn here. The HiFi read subset existed
    # solely for the read-to-assembly alignment metrics, which this version no
    # longer computes, so the pass over the full read set goes with them.
    if evaluator.stage_enabled("hic_phasing"):
        logging.info(
            f"Subsetting {args.num_hic_reads} Hi-C read pairs for the phasing "
            "metric."
        )
        evaluator.subset_hic_reads()


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
    # The database file, not just the study record inside it: deleting by
    # study name silently misses any other study in the same file, which is how
    # a previous run's trials survived a --force-rerun once already.
    if paths.db_path.exists():
        try:
            paths.db_path.unlink()
            logging.info(f"Deleted the previous study database: {paths.db_path}")
        except Exception as e:
            logging.warning(f"Could not delete {paths.db_path}: {e}")
            try:
                optuna.delete_study(study_name="no-name", storage=paths.db_uri)
            except Exception as inner:
                logging.warning(f"Could not delete the study either: {inner}")

    removed = []
    # The running-best checkpoint belongs to the study being discarded; left in
    # place it reported the old study's "best" until the new one found its own.
    try:
        paths.best_params_checkpoint.unlink()
        removed.append(paths.best_params_checkpoint.name)
    except FileNotFoundError:
        pass

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

    # rmtree() not raising does not guarantee the directory is actually gone:
    # on a network-mounted filesystem a delete can appear to succeed locally
    # while remnants are still visible for a few seconds (this is what
    # happened in practice -- a yak hash from the deleted work/ turned up as
    # "already present" two minutes into a --force-rerun run). work/hifiasm/
    # is the dangerous case: reusing its .bin files from an incompatible prior
    # configuration is what silently broke the GFA-naming prediction, so it
    # gets an explicit re-check with a short retry rather than a warning that
    # is easy to miss in a multi-day log.
    import time

    for attempt in range(5):
        remnants = (
            list(paths.work_dir.glob("**/*")) if paths.work_dir.exists() else []
        )
        if not remnants:
            break
        time.sleep(2)
    else:
        listing = "\n  ".join(str(p) for p in remnants[:20])
        logging.error(
            f"--force-rerun: {paths.work_dir} still contains files after "
            f"deletion and a {5 * 2}s wait -- the wipe did not actually "
            f"complete (this can happen on a network filesystem with delayed "
            f"delete propagation). Continuing would risk hifiasm reusing "
            f".bin files from the previous configuration while this run's "
            f"flags predict a different output layout, which fails in "
            f"confusing ways rather than obviously. Remaining:\n  {listing}\n"
            f"Delete {paths.work_dir} manually and re-run, or wait for the "
            f"filesystem to catch up."
        )
        sys.exit(1)

    paths.create()
    if removed:
        logging.info(f"Cleaned up previous run artifacts: {', '.join(removed)}")

    # metric_stage_state.json went with work/, so every retired stage gets
    # another chance; the in-memory copy has to follow it.
    evaluator.stage_state = {}
    evaluator.backend_cache = {}
    evaluator.kmer_eval = args.kmer_eval
    evaluator.include_busco = args.include_busco


def hifiasm_run_fingerprint():
    """
    Everything about this invocation that hifiasm's own .bin files depend on.

    Covers what changes hifiasm's output *naming* (primary/hic/ul, which pick
    the infix -- see ``objective.haplotype_suffixes``) and what changes .bin
    *content* without hifiasm re-validating it (--hifiasm-extra, --sensitive,
    --ont: verified against the source this session that hifiasm's own
    checkpoint loader checks the error-correction round count and little
    else). Two runs against the same output directory only share bins safely
    if this matches.
    """
    return {
        "primary": bool(args.primary),
        "hic": bool(hic1 and hic2),
        "ul": bool(ul),
        "ont": bool(args.ont),
        "sensitive": bool(args.sensitive),
        "hifiasm_extra": args.hifiasm_extra or "",
    }


def verify_hifiasm_bin_compatibility():
    """
    Refuse to silently reuse .bin files written under different flags.

    hifiasm resumes from whatever it finds at its ``-o`` prefix
    (``load_index_from_disk``, on by default) and does not check whether the
    run that wrote those files used the same --primary/--hic/--ul/
    --hifiasm-extra as this one. A mismatch there is what let hifiasm exit 0
    while writing output under a different naming convention than the current
    flags predict, which then surfaces many stages later as a confusing
    "GFA file not found" rather than as what it actually is. This is checked
    on every startup, not only after --force-rerun: the same hazard exists
    when resuming a study after changing --hifiasm-extra by hand.
    """
    fp_path = paths.hifiasm_fingerprint
    current = hifiasm_run_fingerprint()

    existing_bins = (
        any(paths.hifiasm_dir.glob(f"{paths.hifiasm_prefix.name}*"))
        if paths.hifiasm_dir.exists()
        else False
    )

    if not existing_bins:
        # Nothing to be incompatible with yet; record what this run is about
        # to write so the *next* invocation can check itself.
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fp_path, "w") as f:
            json.dump(current, f, indent=2)
        return

    if not fp_path.exists():
        # .bin files from before this check existed. Can't compare, so warn
        # rather than block -- but write the fingerprint now so this stops
        # being silent from here on.
        logging.warning(
            f"{paths.hifiasm_dir} has existing hifiasm files with no recorded "
            "fingerprint (from a hifimizer version before this check). "
            "Proceeding, but if the GFA naming looks wrong, re-run with "
            "--force-rerun to get a clean bin cache."
        )
        with open(fp_path, "w") as f:
            json.dump(current, f, indent=2)
        return

    with open(fp_path) as f:
        recorded = json.load(f)

    if recorded != current:
        diffs = "\n  ".join(
            f"{k}: recorded={recorded.get(k)!r} vs current={v!r}"
            for k, v in current.items()
            if recorded.get(k) != v
        )
        logging.error(
            f"{paths.hifiasm_dir} contains .bin files written under different "
            f"hifiasm flags than this run is using:\n  {diffs}\n"
            "hifiasm resumes from these files without checking that "
            "compatibility itself, so continuing risks it exiting "
            "successfully while writing output under a naming convention "
            "this run does not predict -- which is what happened here. "
            "Re-run with --force-rerun to rebuild the bins under the current "
            "flags, or restore the flags this directory was started with."
        )
        sys.exit(1)


if args.force_rerun:
    reset_previous_run()

verify_hifiasm_bin_compatibility()

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
    num_hic_reads=args.num_hic_reads,
    hic_min_mapq=args.hic_min_mapq,
    hic_phasing=args.hic_phasing,
    prune_bits=args.prune_bits,
    prune_min_metrics=args.prune_min_metrics,
    hifiasm_extra=args.hifiasm_extra,
    bits_cap=args.bits_cap,
)
# A single objective for the whole study. Trial 0 is hifiasm with default
# parameters and every later trial is scored in bits against it, which needs no
# scale to be measured first.
objective = objective_builder.build_objective()

PRIMARY_SUFFIX, EXTRA_SUFFIXES = haplotype_suffixes(hic1, hic2, ul, args.primary)

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

    num_sv = round(val("num_sv"))

    single_copy = round(val("single_copy"))
    multi_copy = round(val("multi_copy"))
    fragmented = round(val("fragmented"))
    missing = round(val("missing"))
    total_busco = single_copy + multi_copy + fragmented + missing

    def pct(x):
        return (x / total_busco * 100) if total_busco > 0 else 0

    out = f"{title}:\n"
    # Any block can be absent: its stage may have failed or been switched off.
    #
    # Contiguity is reported per haplotype when there is more than one and
    # under the plain names when there is not, matching how it is scored. Every
    # number is back-transformed out of log space first: contigs as a count,
    # N50 in bp, length difference in Mb.
    for suffix, label in (("_hap1", "hap1"), ("_hap2", "hap2"), ("", "")):
        if f"n50{suffix}" not in metrics:
            continue
        tag = f" ({label})" if label else ""
        length_key = f"{label}_length_mb" if label else "hap1_length_mb"
        out += (
            f"  - Number of contigs{tag:<10}: "
            f"{round(val('num_contigs' + suffix)):,}\n"
        )
        if length_key in metrics:
            out += (
                f"  - Assembly length{tag:<12}: "
                f"{metrics[length_key]:.2f} Mb "
                f"[target {GENOME_SIZE_BP / 1e6:.2f} Mb]\n"
            )
        out += (
            f"  - Length difference{tag:<10}: "
            f"{val('length_diff' + suffix):.2f} Mb\n"
        )
        out += f"  - N50{tag:<24}: {round(val('n50' + suffix)):,} bp\n"
    if "num_sv" in metrics:
        out += f"  - Large-scale misassemblies  : {num_sv:,}\n"
    if "trans_hap_rate" in metrics:
        informative = round(val("hic_pairs_informative"))
        out += (
            f"  - Hi-C trans-haplotype links : "
            f"{metrics['trans_hap_rate']:.2f}% of {informative:,} "
            f"informative pairs\n"
        )

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
        if "qv_hap2" in metrics:
            out += (
                f"  - Consensus QV (yak)         : {metrics['qv']:.2f} mean "
                f"({metrics['qv_hap1']:.2f} hap1 / {metrics['qv_hap2']:.2f} hap2)\n"
            )
        else:
            out += f"  - Consensus QV (yak)         : {metrics['qv']:.2f}\n"
    if "kmer_completeness" in metrics:
        out += (
            f"  - k-mer completeness (yak)   : "
            f"{metrics['kmer_completeness']:.2f}% (hap1 + hap2 combined)\n"
            if "qv_hap2" in metrics
            else f"  - k-mer completeness (yak)   : "
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


# ---------------------------------------------------- final assembly record
# final_assembly/ outlives the run that built it, and resuming a finished
# study used to rebuild it from scratch -- a full hifiasm run plus evaluation,
# to reproduce files already on disk. The record says what the directory
# holds, so a rebuild happens only when the best trial or the hifiasm flags
# have changed since.
def _json_safe(value):
    return json.loads(json.dumps(value, default=float))


def clear_final_assembly_record():
    """Called before any rebuild, so a failed one cannot vouch for old files."""
    try:
        paths.final_assembly_record.unlink()
    except FileNotFoundError:
        pass


def write_final_assembly_record(trial_number, params, metrics=None):
    record = {
        "trial": trial_number,
        "params": _json_safe(params),
        "hifiasm_flags": hifiasm_run_fingerprint(),
        "metrics": _json_safe(metrics) if metrics else None,
    }
    try:
        with open(paths.final_assembly_record, "w") as fh:
            json.dump(record, fh, indent=2)
    except OSError as e:
        logging.warning(f"Could not write {paths.final_assembly_record}: {e}")


def matching_final_assembly(trial_number, params):
    """The record of ``final_assembly/`` if it already holds this trial's build."""
    try:
        with open(paths.final_assembly_record) as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return None
    gfa = paths.final_assembly_dir / f"final_assembly.{PRIMARY_SUFFIX}.gfa"
    if (
        record.get("trial") == trial_number
        and record.get("params") == _json_safe(params)
        and record.get("hifiasm_flags") == hifiasm_run_fingerprint()
        and gfa.exists()
    ):
        return record
    return None


def run_hifiasm_with_params(params, trial_label, walltime_hours):
    """Rebuild an assembly from a recorded parameter set. Returns the GFA path."""
    kwargs = dict(
        prefix=str(paths.hifiasm_prefix),
        haploid_genome_size=KNOWN_GENOME_SIZE,
        threads=threads,
        primary=args.primary,
        hic1=hic1,
        hic2=hic2,
        ul=ul,
        ont=args.ont,
        hom_cov=args.hom_cov,
        extra_args=HIFIASM_PASSTHROUGH,
    )
    for key in HIFIASM_PARAM_KEYS:
        if key in params:
            kwargs[key] = params[key]
    # A pinned parameter is absent from the recorded trial params (it was never
    # sampled), so it has to be reapplied here or a rerun would silently drop it.
    for key, value in PINNED_HIFIASM_PARAMS.items():
        kwargs.setdefault(key, value)

    command = build_hifiasm_command(**kwargs) + f" {input_reads}"
    if args.sensitive:
        # The cache holds whichever trial ran last, with its own -D/-N/
        # --max-kocc; rebuilding this parameter set on top of it would load
        # that trial's corrected reads instead.
        removed = delete_hifiasm_bins(paths.hifiasm_prefix)
        if removed:
            logging.info(
                f"--sensitive: deleted {len(removed)} cached hifiasm .bin "
                f"file(s) before rebuilding {trial_label}."
            )
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
        found = sorted(
            f.name
            for f in paths.hifiasm_prefix.parent.glob(f"{paths.hifiasm_prefix.name}*")
        )
        listing = ", ".join(found) if found else "(nothing)"
        raise FileNotFoundError(
            f"No GFA found at {gfa} after hifiasm completed.\n"
            f"    predicted suffix : {PRIMARY_SUFFIX}\n"
            f"    hic={bool(hic1 and hic2)} ul={bool(ul)} primary={args.primary}\n"
            f"    actually at {paths.hifiasm_prefix.parent}: {listing}\n"
            "    hifiasm reported success, so this is a naming mismatch, not "
            "an assembly failure -- check work/hifiasm/.fingerprint.json "
            "against the flags above."
        )
    return gfa


def evaluate_and_report(gfa, trial_label, title):
    """Convert, evaluate and log a finished assembly."""
    evaluator.trial_id = trial_label
    fasta = evaluator.trial_dir / f"{Path(gfa).stem}.fasta"
    AssemblyEvaluator.convert_gfa_to_fasta(gfa, fasta)

    extra_fastas = evaluator.convert_extra_haplotypes(
        paths.hifiasm_prefix, EXTRA_SUFFIXES
    )
    extra_gfas = [
        paths.hifiasm_prefix.parent
        / f"{paths.hifiasm_prefix.name}.{suffix}.gfa"
        for suffix in EXTRA_SUFFIXES
    ]

    metrics = evaluator.evaluate_assembly(
        gfa_file=gfa,
        fasta_file=fasta,
        include_busco=args.include_busco,
        busco_lineage=args.busco_lineage,
        extra_fasta_files=extra_fastas,
        extra_gfa_files=extra_gfas,
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

    # Always the highest score among the candidates. The running best stored
    # by best_tracker_callback is a progress report, not the authority: it
    # used to be returned as-is whenever it was among the candidates, so a
    # running best that had been reset wrongly decided the final assembly.
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
        # A pruned trial has no score of its own, and skipping it here was why
        # convergence never fired: a study that pruned 28 of its 30 trials fed
        # the detectors two values, so every window stayed empty and no
        # detector could ever reach its patience. A pruned trial is still
        # evidence -- the search looked somewhere and found nothing usable --
        # so it is recorded as "no improvement" by replaying the running best.
        # A long enough run of them therefore converges, which is the correct
        # reading of a search that has stopped finding anything.
        best = study.user_attrs.get("best_score")
        if best is None or best == float("-inf"):
            return
        current = float(best)

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
        # Reported on every trial: without it there is no way to tell a study
        # that is closing in on convergence from one that has not moved, until
        # it either stops or hits --num-trials.
        total = len(convergence_detector.detectors)
        needed = max(1, total // 2)
        methods = ", ".join(converged_methods) if converged_methods else "none"
        logging.info(
            f"Convergence: {len(converged_methods)}/{total} detectors agree "
            f"({needed} needed to stop) - {methods}"
        )


def previous_stop_reason(study):
    """
    Why this study was stopped by an earlier run, or None if it never was.

    Every way a study can stop itself leaves a mark in its user attributes
    before calling ``study.stop()``: ``converged`` from convergence_callback,
    ``halted_reason`` from pruned_trials_callback and metric_skip_callback.
    Running out of --num-trials is deliberately not a stop -- the budget is
    per invocation, so resuming such a study is how it is extended.
    """
    halted = study.user_attrs.get("halted_reason")
    if halted:
        return f"halted: {halted}"
    if study.user_attrs.get("converged"):
        methods = study.user_attrs.get("converged_methods") or []
        detail = f" (detected by {', '.join(methods)})" if methods else ""
        return f"converged{detail}"
    return None


def consecutive_pruned_trials(study):
    """The trials pruned in a row since the last completed one, newest first."""
    run = []
    for t in sorted(
        study.get_trials(deepcopy=False), key=lambda t: t.number, reverse=True
    ):
        if t.state == optuna.trial.TrialState.COMPLETE:
            break
        if t.state == optuna.trial.TrialState.PRUNED:
            run.append(t)
    return run


def pruned_trials_callback(study, trial):
    """
    Stop once the gfastats gate has rejected everything for long enough.

    Distinct from --max-metric-skips, which counts trials thrown away because
    a *metric* failed. This counts trials the *assembly* failed: hifiasm keeps
    producing collapsed graphs and there is no reason to expect the next
    twenty parameter sets to be different.

    Counted consecutively rather than in total. Prunes scattered across a long
    study are ordinary -- the search is meant to try bad parameter sets -- and
    an unbroken run is the only version of this that means the search is stuck.
    """
    if args.max_consecutive_pruned <= 0:
        return
    if trial.user_attrs.get("metric_skip"):
        return  # metric_skip_callback owns that budget

    pruned = consecutive_pruned_trials(study)
    run = len(pruned)
    if run < args.max_consecutive_pruned:
        if run and run % 5 == 0:
            logging.warning(
                f"{run} trials pruned in a row (limit "
                f"--max-consecutive-pruned={args.max_consecutive_pruned})."
            )
        return

    if study.user_attrs.get("halted_reason"):
        return

    # Trials end up "pruned" for two very different reasons: the gfastats gate
    # rejected a collapsed assembly, or hifiasm / the evaluation failed before
    # anything was scored. The stop is the same, the advice is not -- widening
    # the gate does nothing for a hifiasm that refuses its arguments.
    gated = [t for t in pruned if t.user_attrs.get("gfastats_gate")]
    failed = [t for t in pruned if t.user_attrs.get("failure")]
    limit = f"limit --max-consecutive-pruned={args.max_consecutive_pruned}"
    if len(gated) == run:
        reason = f"{run} consecutive trials pruned at the gfastats gate ({limit})"
        detail = (
            "Every one of them collapsed on at least "
            f"{args.prune_min_metrics} of the gated contiguity metrics, by "
            f"more than {args.prune_bits:.1f} bits against the "
            "default-parameter assembly. The sampler is not finding usable "
            "parameter sets, and each rejected trial still costs a full "
            "hifiasm run.\n"
            "Raise --max-consecutive-pruned to push on, or --prune-bits to "
            "widen the gate."
        )
    else:
        causes = [f"{len(failed)} failed before scoring"]
        if gated:
            causes.append(f"{len(gated)} pruned at the gfastats gate")
        reason = (
            f"{run} consecutive trials produced no score: "
            f"{', '.join(causes)} ({limit})"
        )
        latest = failed[0].user_attrs["failure"] if failed else "see the logs"
        detail = (
            f"{len(failed)} of them failed before they could be scored -- "
            "usually hifiasm exiting with an error, which repeats for every "
            "parameter set until its cause is fixed.\n"
            f"Most recent failure (trial {failed[0].number if failed else '?'}): "
            f"{latest}\n"
            f"Per-command logs: {logs_dir}"
        )
    study.set_user_attr("halted_reason", reason)
    logging.warning(
        f"\n{'#' * 72}\n"
        f"OPTIMIZATION STOPPED - {run} trials in a row without a score\n"
        f"{'#' * 72}\n"
        f"{detail}\n"
        "The best trial so far will still be rebuilt and reported.\n"
        f"{'#' * 72}"
    )
    study.stop()


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

        # Only trials scored on the metric set this run currently expects
        # compete. A trial that lost a metric for its own assembly alone is
        # skipped rather than allowed to reset the running best: resetting on
        # every change made a one-off dropout throw away the real best, and
        # the next ordinary trial became "best" whatever its score. Only a
        # stage retired for the whole study changes the expected set, and
        # that is when the incumbent stops being comparable.
        regime = trial.user_attrs.get("metric_regime")
        if regime is not None:
            evaluator.reload_stage_state()
            expected = evaluator.metric_regime()
            if regime != expected:
                lost = sorted(set(expected.split(",")) - set(regime.split(",")))
                logging.warning(
                    f"Trial {trial.number} was scored without "
                    f"{', '.join(lost) or 'some metrics'}, so it is not "
                    "compared for the best trial."
                )
                return
            previous_regime = study.user_attrs.get("best_regime")
            if previous_regime is not None and previous_regime != expected:
                logging.warning(
                    "The set of scored metrics changed at trial "
                    f"{trial.number} (a metric stage was retired). Resetting "
                    "the running best: earlier scores are on a different scale."
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


def rerun_from_params(params, label, banner, title, trial_number):
    """
    Rebuild, archive and evaluate an assembly from a recorded parameter set.

    Shared by ``--rerun-best`` and ``--rerun-trial``, which differed only in
    how they found the parameters. Always rebuilds -- that is what was asked
    for -- and records what ``final_assembly/`` now holds, so a later resume
    knows whether it still matches the best trial. Never returns: it exits
    the process.
    """
    logging.info(f"\n{'#' * 60}\n{banner}\n{'#' * 60}")
    clear_final_assembly_record()
    try:
        gfa = run_hifiasm_with_params(params, label, args.trial_walltime)
    except Exception as e:
        logging.error(f"{label}: {e}")
        sys.exit(1)

    collect_assembly_outputs(paths.final_assembly_dir, "final_assembly")
    metrics = None
    try:
        metrics = evaluate_and_report(gfa, label, title)
    except Exception as e:
        logging.error(f"{label}: assembly evaluation failed: {e}")
    write_final_assembly_record(trial_number, params, metrics)
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

        # Same window shortening as the single-objective branch. The
        # thresholds stay small here because the multi-objective aggregate is a
        # mean of raw metric values, not of fold changes, so a log2 unit is not
        # the scale -- the Pareto objectives are left untransformed.
        convergence_detector = MultiCriteriaConvergenceDetector(
            directions=directions,
            min_trials=args.convergence_warmup,
            stagnation_patience=10,
            min_improvement=1e-3,
            threshold=0.01,
            patience=8,
            plateau_threshold=1e-2,
            min_plateau_length=8,
            window_size=8,
            significance_level=0.05,
        )
    else:
        directions = ["maximize"]
        # Tuned for log2 fold-change scores, and for actually firing.
        #
        # The windows were the structural problem: PlateauDetector needs
        # min_plateau_length values before it starts counting and then that
        # many again to trip, and StatisticalConvergenceDetector needs two full
        # windows. At 15 apiece, two of the four detectors could not reach a
        # verdict before trial 30 no matter what the scores did -- and two of
        # four is exactly the quorum. At 8 they are reachable by trial 16.
        #
        # min_improvement and plateau_threshold are in bits of the weighted
        # MEAN score, which is what makes them interpretable and the same in
        # every mode: 0.01 is one metric in a typical run of eight improving
        # by about a twelfth of a doubling (~6%), and negligible; a run whose
        # scores sit inside a 0.03 band is not moving. (They were 0.05 and
        # 0.15 when the score was a sum over ~7 units of weight.)
        convergence_detector = MultiCriteriaConvergenceDetector(
            directions=directions,
            min_trials=args.convergence_warmup,
            stagnation_patience=12,
            min_improvement=0.01,
            threshold=0.01,
            patience=10,
            plateau_threshold=0.03,
            min_plateau_length=8,
            window_size=8,
            significance_level=0.05,
        )

    # One sampler choice for both modes. The samplers' own random startup
    # trials are kept: with no burn-in there are no observations to seed them,
    # and those first trials are ordinary scored, gated trials of this study.
    #
    # Multi-objective runs used to use MOEA/D (fetched from optunahub, falling
    # back to NSGA-III). Both are generational and need many generations of a
    # population, which a study of ~100 multi-hour trials never reaches; TPE
    # handles several objectives itself and is built for small budgets.
    sampler = build_sampler(args.sampler, args.multi_objective, args.seed)
    logging.info(
        f"Using {SAMPLER_DESCRIPTIONS[args.sampler]} (--sampler {args.sampler}) "
        f"for {'multi' if args.multi_objective else 'single'}-objective "
        "optimization."
    )

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
            _trial.number,
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
            _num,
        )

    # ------------------------------------------------------- sampler lock
    # A study is tied to the sampler it started with. Optuna itself would let a
    # resumed study switch, but the trials so far were proposed under the
    # other sampler's model, so the result would be neither one's search --
    # and comparing samplers is exactly what a user switching is trying to do.
    # Studies written before the sampler was recorded also predate the
    # current scoring (bits as a weighted mean, the pseudocounts), so their
    # scores are on a different scale and cannot be continued either.
    recorded_sampler = study.user_attrs.get("sampler")
    if study.get_trials(deepcopy=False):
        if recorded_sampler is None:
            logging.error(
                f"\n{'#' * 72}\n"
                "CANNOT RESUME - STUDY FROM AN EARLIER HIFIMIZER VERSION\n"
                f"{'#' * 72}\n"
                f"The study in {paths.db_path} records no sampler, so it was "
                "written by an earlier hifimizer version, which also scored "
                "trials differently. Its scores are not comparable with new "
                "ones. Re-run with --force-rerun to start a fresh study.\n"
                f"{'#' * 72}"
            )
            sys.exit(1)
        if recorded_sampler != args.sampler:
            logging.error(
                f"\n{'#' * 72}\n"
                "CANNOT RESUME - SAMPLER CHANGED\n"
                f"{'#' * 72}\n"
                f"This study was optimised with --sampler {recorded_sampler}, "
                f"and this run asks for --sampler {args.sampler}. A study "
                "cannot switch samplers part-way: its trials so far were "
                "proposed by the other sampler, so the result would belong to "
                "neither. Nothing has been run.\n"
                f"  - to continue this study    : --sampler {recorded_sampler}\n"
                f"  - to start over with {args.sampler:<6} : --sampler "
                f"{args.sampler} --force-rerun\n"
                f"{'#' * 72}"
            )
            sys.exit(1)
    else:
        study.set_user_attr("sampler", args.sampler)

    # ----------------------------------------------------- baseline trial
    # Trial 0 is run on its own first, because nothing after it can be scored
    # until its metrics are on disk. One study holds everything: bits are the
    # same function of a trial's metrics at trial 1 as at trial 100, so no
    # value ever needs rescoring and there is no second study to import from.
    if not study.get_trials(deepcopy=False):
        logging.info(
            f"\n{'#' * 72}\n"
            "BASELINE: trial 0, hifiasm with default parameters\n"
            f"{'#' * 72}\n"
            "It becomes the reference every later trial is compared against, "
            "in bits: +1 is twice as good on that metric, -1 twice as bad. "
            "Bits mean the same thing on every metric, so the optimisation "
            "starts straight after it, with no burn-in.\n"
            "The .bin files it writes are reused by every later trial.\n"
            f"{'#' * 72}"
        )
        study.optimize(
            objective,
            n_trials=1,
            callbacks=[metric_skip_callback, best_tracker_callback],
        )

    if not paths.baseline_metrics.exists():
        logging.error(
            "There is no default-parameter baseline to score against: trial 0 "
            "did not complete, or this study was started before "
            f"{paths.baseline_metrics} was written. Check the logs in "
            f"{logs_dir}, then re-run with --force-rerun."
        )
        sys.exit(1)

    # ------------------------------------------------------- stopped study
    # A study that already stopped itself made a decision: it converged, or
    # it concluded that further trials would only be wasted. Resuming it used
    # to quietly overrule that -- a fresh convergence detector knows nothing
    # of the old history, and a halted_reason that is already set makes both
    # budget callbacks skip their own study.stop() -- so the next run spent up
    # to --num-trials hifiasm assemblies relitigating it. One check covers
    # every mode, single- and multi-objective, since both go through here.
    stop_reason = previous_stop_reason(study)
    if stop_reason and args.continue_stopped:
        logging.warning(
            f"--continue-stopped: this study had stopped ({stop_reason}). "
            f"Clearing that and running up to {args.num_trials} more trial(s)."
        )
        study.set_user_attr("converged", False)
        study.set_user_attr("converged_methods", [])
        study.set_user_attr("halted_reason", None)
        stop_reason = None
    elif stop_reason:
        _completed_before = sum(
            t.state == optuna.trial.TrialState.COMPLETE
            for t in study.get_trials(deepcopy=False)
        )
        logging.info(
            f"\n{'#' * 72}\n"
            "STUDY ALREADY STOPPED - NO NEW TRIALS\n"
            f"{'#' * 72}\n"
            f"An earlier run stopped this study: {stop_reason}.\n"
            f"It holds {len(study.trials)} trial(s), {_completed_before} of "
            "them completed. No new trials are run; the best of them is "
            "reported and the final assembly built as usual.\n"
            "  - to keep optimising this study : --continue-stopped\n"
            "  - to start again from scratch   : --force-rerun\n"
            f"{'#' * 72}"
        )

    # ----------------------------------------------------------- optimize
    if stop_reason is None:
        mode_str = "multi-objective" if args.multi_objective else "single-objective (weighted score)"
        logging.info(
            "Starting Optuna %s optimization with up to %d trials.", mode_str, args.num_trials
        )
        if args.multi_objective:
            logging.info(
                "Pareto objectives (%d): %s",
                len(objective_keys),
                ", ".join(f"{k} ({d})" for k, d in zip(objective_keys, directions)),
            )
            logging.info(
                "  No weights are applied in this mode. Every metric outside this "
                "set is still measured and logged per trial, but does not steer "
                "the search."
            )
        if not args.multi_objective:
            evaluator.reload_stage_state()
            log_weight_groups(evaluator)
            log_metric_report_legend(args.bits_cap)
        if args.prune_bits > 0 and args.prune_min_metrics > 0:
            logging.info(
                f"Early pruning: a trial is abandoned after gfastats once "
                f"{args.prune_min_metrics} of contig count, N50 and length "
                f"difference are more than {args.prune_bits:.1f} bits worse "
                f"than the default-parameter assembly (2^{args.prune_bits:.1f} = "
                f"{2 ** args.prune_bits:.1f}x). The remaining metrics are not run "
                "for that trial. The study stops after "
                f"{args.max_consecutive_pruned} such trials in a row.\n"
            )

        study.optimize(
            objective,
            n_trials=args.num_trials,
            callbacks=[
                metric_skip_callback,
                pruned_trials_callback,
                best_tracker_callback,
                convergence_callback,
            ],
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
            # user_attrs hold the stored values, which for counts and for
            # length_diff are log(value + 1). Plotting those puts a 30 Mb
            # length difference and a 3 Mb one within a hair of each other on
            # the axis. Back-transform so the y axis is Mb, bp or the metric's
            # own unit -- what the optimiser consumes is unaffected.
            ys.append(AssemblyEvaluator.raw_value(metric, v))
        if len(ys) < 2:
            continue
        unit = AssemblyEvaluator.METRIC_UNITS.get(metric, "")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=metric))
        fig.update_layout(
            title=f"Metric History: {metric} (trial 0 = default-parameter baseline)",
            xaxis_title="Trial",
            yaxis_title=f"{metric} ({unit})" if unit else metric,
        )
        fig.write_html(metric_dir / f"{metric}.history.html")
except Exception as e:
    logging.warning(f"Failed to create per-metric plots: {e}")

# ----------------------------------- FINAL ASSEMBLY with the best parameters
try:
    if best_trial is None:
        logging.info("No best trial identified; skipping final assembly run.")
    elif best_trial.user_attrs.get("baseline"):
        # Rebuilding would re-issue the identical default-parameter command and
        # land the identical assembly next to the one already sitting in
        # default_assembly/, at the cost of another full hifiasm run.
        logging.info(
            f"\n{'#' * 72}\n"
            "DEFAULT PARAMETERS WON - NO FINAL ASSEMBLY BUILT\n"
            f"{'#' * 72}\n"
            f"The best trial is {best_trial.number}, the default-parameter "
            f"baseline, scoring {best_score:+.4f}. Nothing the search sampled "
            "beat plain hifiasm on this data.\n"
            "\n"
            f"  Use this assembly: {paths.default_assembly_dir}\n"
            "\n"
            "It is already complete on disk, so the final assembly step is "
            "skipped rather than rebuilding a copy of it.\n"
            "\n"
            "If you expected tuning to help, the usual reasons are a search "
            "that stopped before it explored much (raise --num-trials, or "
            "check whether it converged early), or weights.json rewarding "
            "something other than what you care about.\n"
            f"{'#' * 72}"
        )
    elif matching_final_assembly(best_trial.number, trial_params(best_trial)):
        record = matching_final_assembly(best_trial.number, trial_params(best_trial))
        logging.info(
            f"final_assembly/ already holds trial {best_trial.number}'s "
            "assembly, built from the same parameters and hifiasm flags; "
            "reusing it instead of rebuilding. Use --rerun-best to force a "
            "rebuild."
        )
        if record.get("metrics"):
            logging.info(
                format_metrics(
                    record["metrics"], "Final assembly metrics (from when it was built)"
                )
            )
    else:
        best_params = trial_params(best_trial)
        clear_final_assembly_record()
        try:
            final_gfa = run_hifiasm_with_params(best_params, "best", args.trial_walltime)
        except Exception as e:
            logging.error(f"Final assembly failed: {e}")
            final_gfa = None

        if final_gfa is not None:
            collect_assembly_outputs(paths.final_assembly_dir, "final_assembly")
            final_metrics = None
            try:
                final_metrics = evaluate_and_report(
                    final_gfa, "best", "Final assembly metrics"
                )
            except Exception as e:
                logging.error(f"Final assembly evaluation failed: {e}")
            write_final_assembly_record(best_trial.number, best_params, final_metrics)
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