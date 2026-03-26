#!/usr/bin/env python3

import subprocess, logging, sys, optuna, psutil, signal, os, numpy as np
import random
from pathlib import Path
import optuna.visualization as vis
import optunahub
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
from utils.assembly_eval import AssemblyEvaluator
from utils.optuna_callback import MultiCriteriaConvergenceDetector
from utils.hifiasm_command import build_hifiasm_command, run_default_hifiasm_assembly
from utils.objective import ObjectiveBuilder
from utils.argparser import get_args

# Set the base path to the current working directory
base_path = Path.cwd()
os.chdir(base_path)

# Define a function to terminate all child/grandchild processes and exit the program gracefully upon receiving a termination signal
TERMINATE_REQUESTED = False


def terminate_all_processes(sig, frame):
    global TERMINATE_REQUESTED
    TERMINATE_REQUESTED = True
    parent = psutil.Process()
    for child in parent.children(recursive=True):
        child.kill()
    logging.info("Termination signal received. All child processes killed.")
    sys.exit(0)


def get_terminate_status():
    return TERMINATE_REQUESTED


signal.signal(signal.SIGINT, terminate_all_processes)
signal.signal(signal.SIGTERM, terminate_all_processes)

# Parse command line arguments
args = get_args()

# Set random seeds for reproducibility if specified
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

# Parse command line arguments
input_reads = Path(args.input_reads).resolve()
hic1 = Path(args.hic1).resolve() if args.hic1 else None
hic2 = Path(args.hic2).resolve() if args.hic2 else None
ul = Path(args.ul).resolve() if args.ul else None
threads = args.threads
download_path = args.busco_download_path

# --- Upfront input validation ---
# Fail fast before spending any time on setup or assembly.

# Hi-C reads must be provided as a pair
if bool(hic1) != bool(hic2):
    logging.error("--hic1 and --hic2 must always be provided together.")
    sys.exit(1)

# Verify all input files actually exist
_input_files = {"--input-reads": input_reads}
if hic1:
    _input_files["--hic1"] = hic1
if hic2:
    _input_files["--hic2"] = hic2
if ul:
    _input_files["--ul"] = ul

_missing = [f"{flag}: {path}" for flag, path in _input_files.items() if not path.exists()]
if _missing:
    logging.error("Input file(s) not found:\n  " + "\n  ".join(_missing))
    sys.exit(1)

# Convert output directory to absolute path to avoid confusion
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
os.chdir(output_dir)

# Create logs directory inside output directory
logs_dir = output_dir / "logs"
logs_dir.mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(logs_dir / "main.log"), logging.StreamHandler()],
)

# Set up run parameters
KNOWN_GENOME_SIZE = args.genome_size
ploidy = args.ploidy

GENOME_SIZE_BP = KNOWN_GENOME_SIZE * 1_000_000

# Initialize the evaluator
evaluator = AssemblyEvaluator(
    known_genome_size=GENOME_SIZE_BP,
    input_reads=input_reads,
    threads=threads,
    download_path=download_path,
    logs_dir=logs_dir,
    ont=args.ont
)

if args.default_hifiasm:
    run_default_hifiasm_assembly(
        prefix="default_run",
        haploid_genome_size=KNOWN_GENOME_SIZE,
        threads=threads,
        primary=args.primary,
        hic1=args.hic1,
        hic2=args.hic2,
        ul=args.ul,
        input_reads=input_reads,
        ont=args.ont
    )
    exit(0)

# --- Dry-run: validate tools and print trial-0 command, then exit ---
if args.dry_run:
    import shutil as _shutil
    _required_tools = ["hifiasm", "busco", "gfastats"]
    _missing_tools = [t for t in _required_tools if _shutil.which(t) is None]
    if _missing_tools:
        logging.error(
            "Dry-run: required tool(s) not found on PATH: "
            + ", ".join(_missing_tools)
        )
        sys.exit(1)

    _dry_cmd = build_hifiasm_command(
        prefix="trial_assembly",
        haploid_genome_size=KNOWN_GENOME_SIZE,
        threads=threads,
        primary=args.primary,
        hic1=hic1,
        hic2=hic2,
        ul=ul,
        ont=args.ont,
    ) + f" {input_reads}"

    logging.info(
        "Dry-run checks passed.\n"
        f"  Input reads : {input_reads}\n"
        f"  Genome size : {KNOWN_GENOME_SIZE} Mb\n"
        f"  Threads     : {threads}\n"
        f"  Output dir  : {output_dir}\n"
        f"  Tools found : {', '.join(_required_tools)}\n"
        f"  Trial-0 command (default params):\n    {_dry_cmd}"
    )
    sys.exit(0)

# Prepare data for evaluation
logging.info("Preparing BUSCO download and read subsetting.")


def busco_lineage_exists(download_path, lineage):
    return download_path and os.path.exists(
        os.path.join(download_path, "lineages", lineage)
    )

if download_path and busco_lineage_exists(download_path, args.busco_lineage):
    logging.info(
        f"BUSCO lineage '{args.busco_lineage}' found in '{download_path}'. Skipping download."
    )
else:
    evaluator.download_busco(lineage=args.busco_lineage)

# Always perform read subsetting, regardless of BUSCO download status
evaluator.read_subsetting(num_reads=args.num_reads)
# Initialize the objective function for Optuna
objective_builder = ObjectiveBuilder(
    evaluator=evaluator,
    input_reads=input_reads,
    haploid_genome_size=KNOWN_GENOME_SIZE,
    threads=threads,
    hic1=hic1,
    hic2=hic2,
    ul=ul,
    sensitive=args.sensitive,
    primary=args.primary,
    include_busco=args.include_busco,
    busco_lineage=args.busco_lineage,
    download_path=download_path,
    output_dir=output_dir,
    logs_dir=logs_dir,
    is_multi_objective=args.multi_objective,
    ont=args.ont,
)
# Build the objective function
objective = objective_builder.build_objective()


# Define the convergence callback function
def convergence_callback(study, trial):
    # Use weighted_score for single-objective, aggregate_score for multi-objective
    current = trial.user_attrs.get("weighted_score", None)
    if current is None:
        current = trial.user_attrs.get("aggregate_score", None)
    if current is None:
        logging.warning(f"Trial {trial.number} has no usable score for convergence → skipping")
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


# Callback to track best trial so far using weighted_score (single-obj) or aggregate_score (multi-obj)
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
            # Checkpoint best params to disk so they survive interruption
            params = trial.user_attrs.get("params", dict(trial.params))
            try:
                import json as _json
                checkpoint = {
                    "trial": trial.number,
                    "score": float(score_val),
                    "params": params,
                }
                with open(output_dir / "best_params_checkpoint.json", "w") as _f:
                    _json.dump(checkpoint, _f, indent=2)
            except Exception as cp_err:
                logging.debug(f"best_tracker_callback: checkpoint write failed: {cp_err}")
            logging.info(
                f"New best so far: trial {trial.number} score={score_val:.4f} params={params}"
            )
    except Exception as e:
        logging.debug(f"best_tracker_callback: unexpected error: {e}")


# Run optimization
try:
    # Configure optimization based on mode (single vs multi-objective)
    if args.multi_objective:
        # Multi-objective setup
        objective_keys = objective_builder.objectives

        # Load optimization directions from src/optim_directions.json (no weights.json fallback)
        # Format: { "metric_name": "maximize" | "minimize", ... }
        directions_file = Path(__file__).resolve().parent / "optim_directions.json"
        try:
            import json

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
                        f"Invalid direction '{v}' for metric '{k}'; must be 'maximize' or 'minimize'"
                    )
        except Exception as e:
            logging.error(f"Failed to load optim_directions.json: {e}")
            raise
        # Create convergence detector configured for multi-objective directions
        # Tuned for faster detection: stagnation after 10 trials, plateau range 1e-3, relative improvement < 1%
        convergence_detector = MultiCriteriaConvergenceDetector(
            directions=directions,
            stagnation_patience=10,
            min_improvement=0,
            threshold=0.01,  # 1% relative improvement threshold
            patience=10,
            plateau_threshold=1e-3,  # tighter plateau detection
            min_plateau_length=10,
            window_size=10,
            significance_level=0.05,
        )

        # Use MOEAD sampler for multi-objective optimization
        try:
            moead_module = optunahub.load_module(
                package="samplers/moead", force_reload=False
            )
            sampler = moead_module.MOEADSampler(seed=args.seed)
            logging.info("Using MOEAD sampler for multi-objective optimization")
        except Exception as e:
            logging.warning(
                f"MOEAD not available ({e}), falling back to NSGAIIISampler"
            )
            sampler = optuna.samplers.NSGAIIISampler(seed=args.seed)
    else:
        # Single-objective setup
        directions = ["maximize"]  # Single direction for weighted score
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
        logging.info(
            "Using TPE sampler for single-objective optimization (weighted score)"
        )

    # If force rerun requested, attempt to delete existing study from storage
    load_if_exists = True
    if args.force_rerun:
        load_if_exists = False
        try:
            logging.info(
                "--force-rerun specified: deleting existing Optuna study if present."
            )
            optuna.delete_study(study_name="no-name", storage=evaluator.db_uri)
            logging.info(
                "Deleted existing Optuna study 'no-name'. A fresh study will be created."
            )
        except Exception as e:
            logging.warning(
                f"Could not delete existing study (may not exist or storage issue): {e}"
            )

        # Clean up previous run results: final assembly, logs, plots, and default_assembly
        import shutil

        cwd = Path.cwd()
        removed_items = []

        # Remove final assembly files
        for pattern in ["final_assembly*"]:
            for p in cwd.glob(pattern):
                try:
                    if p.is_file():
                        p.unlink()
                        removed_items.append(str(p))
                    elif p.is_dir():
                        shutil.rmtree(p)
                        removed_items.append(str(p))
                except Exception as e:
                    logging.warning(f"Failed to remove {p}: {e}")

        # Remove old logs directory
        old_logs = cwd / "logs"
        if old_logs.exists():
            try:
                shutil.rmtree(old_logs)
                removed_items.append(str(old_logs))
            except Exception as e:
                logging.warning(f"Failed to remove logs directory: {e}")

        # Remove optuna output and plots
        optuna_output = cwd / "optuna_output"
        if optuna_output.exists():
            try:
                shutil.rmtree(optuna_output)
                removed_items.append(str(optuna_output))
            except Exception as e:
                logging.warning(f"Failed to remove optuna_output directory: {e}")

        # Remove default_assembly from previous run
        default_assembly = cwd / "default_assembly"
        if default_assembly.exists():
            try:
                shutil.rmtree(default_assembly)
                removed_items.append(str(default_assembly))
            except Exception as e:
                logging.warning(f"Failed to remove default_assembly directory: {e}")

        if removed_items:
            logging.info(
                f"Cleaned up {len(removed_items)} previous run artifacts: {', '.join([Path(p).name for p in removed_items])}"
            )

    # Reduce Optuna library verbosity (suppress per-trial value printing)
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
        storage=evaluator.db_uri,
        load_if_exists=load_if_exists,
        sampler=sampler,
    )

    # Existing study handling: if not forcing rerun, continue from existing trials.
    if not args.force_rerun and len(study.trials) > 0:
        logging.info(
            f"Resuming existing study with {len(study.trials)} trials. Use --force-rerun to start fresh."
        )

    # --rerun-best: skip optimisation entirely and immediately rerun the best
    # assembly found in a previously converged study.
    if args.rerun_best:
        if args.force_rerun:
            logging.error("--rerun-best and --force-rerun are mutually exclusive.")
            sys.exit(1)

        # Gate on recorded convergence so users cannot accidentally rerun a
        # study that never finished properly.
        if not study.user_attrs.get("converged", False):
            logging.error(
                "--rerun-best requested but no convergence was recorded for this "
                "study. Run the full optimisation first and ensure it converges "
                "before using --rerun-best."
            )
            sys.exit(1)

        if len(study.trials) == 0:
            logging.error("No trials found in the existing study. Cannot rerun best assembly.")
            sys.exit(1)

        # ---- Locate best trial (mirrors post-optimisation logic below) ----
        score_key = "aggregate_score" if args.multi_objective else "weighted_score"
        _rerun_best_trial = None
        _rerun_best_score = float("-inf")

        # Prefer the trial number stored by best_tracker_callback
        _bt_num = study.user_attrs.get("best_trial", None)
        if _bt_num is not None:
            for _t in study.trials:
                if _t.number == _bt_num:
                    _rerun_best_trial = _t
                    _rerun_best_score = _t.user_attrs.get(score_key, float("-inf"))
                    break

        # Fallback: scan all trials for highest score
        if _rerun_best_trial is None:
            for _t in study.trials:
                try:
                    _s = float(_t.user_attrs.get(score_key, float("-inf")))
                    if _s > _rerun_best_score:
                        _rerun_best_score = _s
                        _rerun_best_trial = _t
                except Exception:
                    continue

        if _rerun_best_trial is None:
            logging.error(
                "Could not identify a best trial from the study "
                f"(score key: '{score_key}'). Aborting rerun."
            )
            sys.exit(1)

        _converged_methods = study.user_attrs.get("converged_methods", [])
        _methods_str = ", ".join(_converged_methods) if _converged_methods else "unknown"
        logging.info(
            f"\n{'#' * 60}\n"
            f"--rerun-best mode\n"
            f"Convergence methods : {_methods_str}\n"
            f"Best trial          : {_rerun_best_trial.number}\n"
            f"Score ({score_key:>16}): {_rerun_best_score:.4f}\n"
            f"Params              : {_rerun_best_trial.user_attrs.get('params', dict(_rerun_best_trial.params))}\n"
            f"{'#' * 60}"
        )

        # ---- Build and run final assembly with best params ----
        try:
            _best_params = _rerun_best_trial.user_attrs.get(
                "params", dict(_rerun_best_trial.params)
            )
            _hifiasm_kwargs = dict(
                prefix="final_assembly",
                haploid_genome_size=KNOWN_GENOME_SIZE,
                threads=threads,
                sensitive=args.sensitive,
                primary=args.primary,
                hic1=hic1,
                hic2=hic2,
                ul=ul,
                ont=args.ont,
            )
            for _key in [
                "x", "y", "s", "n", "m", "p", "u",
                "D", "N", "max_kocc",
                "s_base", "f_perturb", "l_msjoin",
                "path_max", "path_min",
            ]:
                if _key in _best_params:
                    _hifiasm_kwargs[_key] = _best_params[_key]

            _final_cmd = build_hifiasm_command(**_hifiasm_kwargs) + f" {input_reads}"
        except Exception as _e:
            logging.error(f"--rerun-best: failed to build hifiasm command: {_e}")
            sys.exit(1)

        logging.info(f"Running final assembly:\n{_final_cmd}")
        _runner = SubprocessLogger(logs_dir=logs_dir)
        try:
            _rc, _log_path = _runner.run_command_with_logging(
                command=_final_cmd,
                log_filename="hifiasm.log",
                command_name="hifiasm",
                trial_id="rerun_best",
                timeout_seconds=24 * 3600,
            )
        except Exception as _e:
            logging.error(f"--rerun-best: hifiasm run failed: {_e}")
            sys.exit(1)

        if _rc != 0:
            logging.error(
                f"--rerun-best: hifiasm exited with code {_rc}. "
                f"See log at {_log_path}"
            )
            sys.exit(1)

        # ---- Locate the GFA produced by the rerun ----
        # Suffix mirrors objective.py's suffix selection logic
        if hic1 and hic2:
            _suffix = "hic.hap1.p_ctg"
        elif ul:
            _suffix = "bp.hap1.p_ctg"
        else:
            _suffix = "bp.p_ctg"

        _gfa_candidates = list(Path.cwd().glob(f"final_assembly*.{_suffix}.gfa"))
        if not _gfa_candidates:
            logging.error(
                f"--rerun-best: no GFA file matching 'final_assembly*.{_suffix}.gfa' "
                "found after hifiasm completed."
            )
            sys.exit(1)

        _final_gfa = str(_gfa_candidates[0])
        _fasta_out = str(Path(_final_gfa).with_suffix(".fasta"))

        try:
            AssemblyEvaluator.convert_gfa_to_fasta(_final_gfa, _fasta_out)
            logging.info(f"Generated FASTA from GFA: {_fasta_out}")
        except Exception as _e:
            logging.warning(f"--rerun-best: failed to convert GFA to FASTA: {_e}")

        # ---- Evaluate ----
        logging.info(f"Evaluating rerun assembly: {_final_gfa}")
        try:
            evaluator.trial_id = "rerun_best"
            _metrics = evaluator.evaluate_assembly(
                gfa_file=_final_gfa,
                fasta_file=_fasta_out,
                include_busco=args.include_busco,
                busco_lineage=args.busco_lineage,
                download_path=download_path,
            )

            def _rev_log(v):
                return max(0, np.exp(v) - 1) if v else 0

            _num_contigs   = int(_rev_log(_metrics.get("num_contigs", 0)))
            _n50           = int(_rev_log(_metrics.get("n50", 0)))
            _num_sv        = int(_rev_log(_metrics.get("num_sv", 0)))
            _error_rate    =     _rev_log(_metrics.get("error_rate", 0))
            _length_diff   =     _rev_log(_metrics.get("length_diff", 0))
            _single_copy   = int(_rev_log(_metrics.get("single_copy", 0)))
            _multi_copy    = int(_rev_log(_metrics.get("multi_copy", 0)))
            _fragmented    = int(_rev_log(_metrics.get("fragmented", 0)))
            _missing       = int(_rev_log(_metrics.get("missing", 0)))

            _total_busco = _single_copy + _multi_copy + _fragmented + _missing
            def _pct(x): return (x / _total_busco * 100) if _total_busco > 0 else 0

            logging.info(
                "Rerun assembly metrics:\n"
                f"  - Number of contigs          : {_num_contigs}\n"
                f"  - Length difference          : {_length_diff:.2f} Mb\n"
                f"  - N50                        : {_n50} bp\n"
                f"  - Mapping error rate         : {_error_rate:.6f}\n"
                f"  - Large-scale misassemblies  : {_num_sv}\n"
                f"  - Single-copy BUSCOs         : {_pct(_single_copy):.2f}%\n"
                f"  - Multi-copy BUSCOs          : {_pct(_multi_copy):.2f}%\n"
                f"  - Fragmented BUSCOs          : {_pct(_fragmented):.2f}%\n"
                f"  - Missing BUSCOs             : {_pct(_missing):.2f}%"
            )
        except Exception as _e:
            logging.error(f"--rerun-best: assembly evaluation failed: {_e}")

        sys.exit(0)
    # ---- end --rerun-best ----

    mode_str = (
        "multi-objective"
        if args.multi_objective
        else "single-objective (weighted score)"
    )
    logging.info(
        "Starting Optuna %s optimization with up to %d trials.",
        mode_str,
        args.num_trials,
    )
    # Use convergence callback for both single- and multi-objective
    callbacks = [best_tracker_callback, convergence_callback]
    study.optimize(
        objective,
        n_trials=args.num_trials,
        callbacks=callbacks,
    )
    # Optimization completed
    if len(study.trials) == 0:
        logging.info("No successful trials were completed.")
    else:
        if args.multi_objective:
            logging.info(
                f"Multi-objective optimization completed with {len(study.trials)} trials. Use the Pareto front to select a preferred solution."
            )
        else:
            logging.info(
                f"Single-objective optimization completed with {len(study.trials)} trials. Best weighted score: {study.best_value:.2f}"
            )

        # Track best trial: prefer study-stored best_trial from callback,
        # fallback to scanning trials by score user_attr (weighted_score for single-objective, aggregate_score for multi)
        best_trial = None
        best_score = float("-inf")
        score_key = "aggregate_score" if args.multi_objective else "weighted_score"

        try:
            bt_num = study.user_attrs.get("best_trial", None)
            if bt_num is not None:
                for t in study.trials:
                    if t.number == bt_num:
                        best_trial = t
                        best_score = t.user_attrs.get(score_key, float("-inf"))
                        break
        except Exception:
            best_trial = None
            best_score = float("-inf")

        if best_trial is None:
            best_score = float("-inf")
            for t in study.trials:
                try:
                    s = t.user_attrs.get(score_key, None)
                    if s is None:
                        continue
                    s_val = float(s)
                    if s_val > best_score:
                        best_score = s_val
                        best_trial = t
                except Exception:
                    continue

        if best_trial is not None:
            logging.info(
                f"\n############################################\n"
                f"\nBest trial: {best_trial.number}\nScore: {best_score:.2f}\n"
                f"Params: {best_trial.user_attrs.get('params', {})}\n"
                f"\n############################################"
            )

except Exception as e:
    logging.error(f"Optimization failed: {e}", exc_info=True)
    raise

# --- OPTUNA VISUALIZATIONS ---

# Create output directory
optuna_dir = Path("optuna_output")
optuna_dir.mkdir(exist_ok=True, parents=True)

try:
    vis.plot_param_importances(study, target=None).write_html(
        optuna_dir / "optuna_param_importance.html"
    )
except Exception as e:
    logging.warning(f"Failed to create param importance plot: {e}")

# Save visualizations as interactive HTML files
if args.multi_objective:
    # Multi-objective: create plots per metric
    for idx, metric in enumerate(objective_builder.objectives):
        try:
            metric_dir = optuna_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)

            try:
                vis.plot_optimization_history(
                    study, target=lambda t: t.values[idx], target_name=metric
                ).write_html(metric_dir / "optuna_optimization_history.html")
            except Exception as e:
                logging.warning(
                    f"[{metric}] Failed to create optimization history plot: {e}"
                )
            try:
                vis.plot_parallel_coordinate(
                    study, target=lambda t: t.values[idx], target_name=metric
                ).write_html(metric_dir / "optuna_parallel_coordinates.html")
            except Exception as e:
                logging.warning(
                    f"[{metric}] Failed to create parallel coordinate plot: {e}"
                )

            try:
                vis.plot_contour(
                    study, target=lambda t: t.values[idx], target_name=metric
                ).write_html(metric_dir / "optuna_contour_plot.html")
            except Exception as e:
                logging.warning(f"[{metric}] Failed to create contour plot: {e}")

        except Exception as e:
            logging.warning(f"Failed to create {metric} visualizations: {e}")

    # Additional per-metric plots and Pareto front (for multi-objective)
    try:
        # Pareto front: only works with 2-3 objectives
        if len(objective_builder.objectives) <= 3:
            try:
                vis.plot_pareto_front(study).write_html(
                    optuna_dir / "optuna_pareto_front.html"
                )
            except Exception as e:
                logging.warning(f"Failed to create pareto front plot: {e}")
        else:
            logging.info(
                f"Skipping Pareto front plot: {len(objective_builder.objectives)} objectives exceed 3-objective limit"
            )

        # Per-objective optimization history from trial.values
        for idx, metric in enumerate(objective_builder.objectives):
            try:
                fig = go.Figure()
                trials = [t for t in study.trials if t.values is not None]
                trial_numbers = [t.number for t in trials]
                values = [t.values[idx] for t in trials]

                fig.add_trace(
                    go.Scatter(
                        x=trial_numbers, y=values, mode="lines+markers", name=metric
                    )
                )
                fig.update_layout(
                    title=f"Optimization History: {metric}",
                    xaxis_title="Trial",
                    yaxis_title=metric,
                )
                fig.write_html(optuna_dir / metric / "optimization_history.html")
            except Exception as e:
                logging.warning(
                    f"Failed to create optimization history for {metric}: {e}"
                )
    except Exception as e:
        logging.warning(f"Failed to create additional metric visualizations: {e}")
else:
    # Single-objective: create standard plots
    try:
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(optuna_dir / "optimization_history.html")

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(optuna_dir / "parallel_coordinate.html")

        # Slice plot
        fig = vis.plot_slice(study)
        fig.write_html(optuna_dir / "slice.html")

        # Contour plot (for 2D parameter space)
        if len(study.best_params) >= 2:
            fig = vis.plot_contour(study)
            fig.write_html(optuna_dir / "contour.html")
            
    # Per-metric histories (same spirit as multi-objective)
        metric_dir = optuna_dir / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)

        metrics_to_plot = objective_builder.objectives  # same metric list as elsewhere

        trials = [t for t in study.trials if t.user_attrs is not None]
        xs = [t.number for t in trials]

        for metric in metrics_to_plot:
            ys = []
            x2 = []
            for t in trials:
                v = t.user_attrs.get(metric, None)
                if v is None:
                    continue
                x2.append(t.number)
                ys.append(v)

            if len(ys) < 2:
                continue

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x2, y=ys, mode="lines+markers", name=metric))
            fig.update_layout(
                title=f"Metric History: {metric}",
                xaxis_title="Trial",
                yaxis_title=metric,
            )
            fig.write_html(metric_dir / f"{metric}.history.html")

    except Exception as e:
        logging.warning(f"Failed to create per-metric plots for single-objective: {e}")


# --- FINAL ASSEMBLY: run with best parameters and evaluate ---
try:
    if best_trial is not None:
        # Extract params (prefer stored user_attrs, fallback to trial.params)
        try:
            best_params = best_trial.user_attrs.get("params", dict(best_trial.params))
        except Exception:
            best_params = dict(best_trial.params)

        # Build hifiasm command for final assembly
        hifiasm_kwargs = dict(
            prefix="final_assembly",
            haploid_genome_size=KNOWN_GENOME_SIZE,
            threads=threads,
            sensitive=args.sensitive,
            primary=args.primary,
            hic1=hic1,
            hic2=hic2,
            ul=ul,
            ont=args.ont
        )

        # copy relevant optimized parameters if present
        for key in [
            "x",
            "y",
            "s",
            "n",
            "m",
            "p",
            "u",
            # sensitive tuning params
            "D",
            "N",
            "max_kocc",
            # Hi-C specific params
            "s_base",
            "f_perturb",
            "l_msjoin",
            # UL-specific params
            "path_max",
            "path_min",
        ]:
            if key in best_params:
                hifiasm_kwargs[key] = best_params[key]

        try:
            final_cmd = build_hifiasm_command(**hifiasm_kwargs) + f" {input_reads}"
        except Exception as e:
            logging.error(f"Failed to build final hifiasm command: {e}")
            final_cmd = None

        if final_cmd:
            from utils.subprocess_logger import SubprocessLogger

            runner = SubprocessLogger(logs_dir=logs_dir)
            logging.info(f"Running final assembly with best params: {final_cmd}")
            try:
                rc, log_path = runner.run_command_with_logging(
                    command=final_cmd,
                    log_filename="hifiasm.log",
                    command_name="hifiasm",
                    trial_id="best",
                    timeout_seconds=24 * 3600,
                )
            except Exception as e:
                logging.error(f"Final hifiasm run failed: {e}")
                rc = -1

            if rc == 0:
                # Find generated GFA file(s) for final assembly
                gfa_candidates = list(Path.cwd().glob("final_assembly*.bp.p_ctg.gfa"))
                if not gfa_candidates:
                    logging.warning(
                        "No final assembly GFA files found after hifiasm run"
                    )
                else:
                    final_gfa = str(gfa_candidates[0])
                    # Choose fasta output name based on GFA
                    fasta_out = str(Path(final_gfa).with_suffix(".fasta"))

                    # Explicitly generate FASTA from GFA since hifiasm may not produce one
                    try:
                        AssemblyEvaluator.convert_gfa_to_fasta(final_gfa, fasta_out)
                        logging.info(f"Generated FASTA from GFA: {fasta_out}")
                    except Exception as e:
                        logging.warning(f"Failed to auto-generate FASTA from GFA: {e}")

                    # Ensure evaluator uses 'best' trial id so logs are prefixed accordingly
                    try:
                        evaluator.trial_id = "best"
                    except Exception:
                        pass

                    logging.info(f"Evaluating final assembly: {final_gfa}")
                    try:
                        metrics = evaluator.evaluate_assembly(
                            gfa_file=final_gfa,
                            fasta_file=fasta_out,
                            include_busco=args.include_busco,
                            busco_lineage=args.busco_lineage,
                            download_path=download_path,
                        )

                        # Format and log the metrics
                        # Metrics from evaluate_assembly are log-transformed: np.log(value + 1)
                        # Reverse transformation: exp(log_value) - 1
                        def reverse_log(log_val):
                            return max(0, np.exp(log_val) - 1) if log_val else 0

                        # Reverse log transformation for each metric
                        num_contigs = int(reverse_log(metrics.get("num_contigs", 0)))
                        n50 = int(reverse_log(metrics.get("n50", 0)))
                        num_sv = int(reverse_log(metrics.get("num_sv", 0)))
                        error_rate = reverse_log(metrics.get("error_rate", 0))

                        # Length difference is stored as log of (abs_diff_mb + 1)
                        length_diff_mb = reverse_log(metrics.get("length_diff", 0))

                        # BUSCO counts need reverse log transformation, then convert to percentages
                        single_copy = int(reverse_log(metrics.get("single_copy", 0)))
                        multi_copy = int(reverse_log(metrics.get("multi_copy", 0)))
                        fragmented = int(reverse_log(metrics.get("fragmented", 0)))
                        missing = int(reverse_log(metrics.get("missing", 0)))

                        total_busco = single_copy + multi_copy + fragmented + missing
                        single_copy_pct = (
                            (single_copy / total_busco * 100) if total_busco > 0 else 0
                        )
                        multi_copy_pct = (
                            (multi_copy / total_busco * 100) if total_busco > 0 else 0
                        )
                        fragmented_pct = (
                            (fragmented / total_busco * 100) if total_busco > 0 else 0
                        )
                        missing_pct = (
                            (missing / total_busco * 100) if total_busco > 0 else 0
                        )

                        metrics_str = "Final assembly metrics:\n"
                        metrics_str += f"  - Number of contigs: {num_contigs}\n"
                        metrics_str += (
                            f"  - Length difference: {length_diff_mb:.2f} Mb\n"
                        )
                        metrics_str += f"  - N50: {n50} bp\n"
                        metrics_str += f"  - Mapping error rate: {error_rate:.6f}\n"
                        metrics_str += (
                            f"  - Number of large-scale misassemblies: {num_sv}\n"
                        )
                        metrics_str += (
                            f"  - Single copy BUSCOs: {single_copy_pct:.2f}%\n"
                        )
                        metrics_str += f"  - Multi-copy BUSCOs: {multi_copy_pct:.2f}%\n"
                        metrics_str += f"  - Fragmented BUSCOs: {fragmented_pct:.2f}%\n"
                        metrics_str += f"  - Missing BUSCOs: {missing_pct:.2f}%"

                        logging.info(metrics_str)
                    except Exception as e:
                        logging.error(f"Final assembly evaluation failed: {e}")

            else:
                logging.error(f"Final hifiasm run exited with code {rc}")

        # Remove only top-level trial artifacts and sniffles outputs (non-recursive)
        try:
            import shutil

            removed = 0
            cwd = Path.cwd()
            default_assembly_dir = cwd / "default_assembly"

            # Remove top-level files and directories starting with 'trial'
            for p in cwd.glob("trial*"):
                try:
                    # Skip anything inside the logs directory
                    try:
                        if logs_dir is not None and logs_dir.exists():
                            resolved_logs = logs_dir.resolve()
                            try:
                                if (
                                    resolved_logs in p.resolve().parents
                                    or p.resolve() == resolved_logs
                                ):
                                    continue
                            except Exception:
                                # If resolution fails, be conservative and skip deletion
                                continue
                    except Exception:
                        pass

                    # Skip default_assembly directory (preserve trial 0 results)
                    try:
                        if p.resolve() == default_assembly_dir.resolve():
                            continue
                    except Exception:
                        pass

                    if p.is_file():
                        p.unlink()
                        removed += 1
                    elif p.is_dir():
                        shutil.rmtree(p)
                        removed += 1
                except Exception:
                    logging.warning(f"Failed to remove trial artifact: {p}")

            # Remove sniffles output files produced per trial in top-level output directory
            for s in cwd.glob("sniffles_output_trial_*.vcf"):
                try:
                    if s.is_file():
                        s.unlink()
                        removed += 1
                except Exception:
                    logging.warning(f"Failed to remove sniffles output: {s}")

            logging.info(
                f"Cleaned up {removed} trial and sniffles artifacts (non-recursive)."
            )
        except Exception as e:
            logging.warning(f"Failed to clean trial assemblies: {e}")
    else:
        logging.info("No best trial identified; skipping final assembly run.")
except Exception as e:
    logging.error(f"Final assembly/evaluation failed: {e}")

# --- FINAL CLEANUP ---
try:
    logging.info("Performing final cleanup of intermediate files...")
    evaluator.cleanup_intermediate_files()
except Exception as e:
    logging.warning(f"Final cleanup encountered issues: {e}")