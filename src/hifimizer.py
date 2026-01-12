#!/usr/bin/env python3

import subprocess, logging, sys, optuna, psutil, signal, os, numpy as np
from pathlib import Path
import optuna.visualization as vis
import optunahub
import matplotlib.pyplot as plt
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

# Setup sampler visualizer
module = optunahub.load_module(
    package="visualization/tpe_acquisition_visualizer", force_reload=False
)

tpe_acquisition_visualizer = module.TPEAcquisitionVisualizer()

# Parse command line arguments
args = get_args()

# Parse command line arguments
input_reads = Path(args.input_reads).resolve()
hic1 = Path(args.hic1).resolve() if args.hic1 else None
hic2 = Path(args.hic2).resolve() if args.hic2 else None
ul = Path(args.ul).resolve() if args.ul else None
threads = args.threads
download_path = args.busco_download_path

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
os.chdir(output_dir)

# Create logs directory - using base_path to consolidate in single logs directory
logs_dir = base_path / "logs"
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
    )
    exit(0)

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
)
# Build the objective function
objective = objective_builder.build_objective()


# Define the convergence callback function
def convergence_callback(study, trial):
    # For multi-objective studies, use the trial values aggregated according to
    # the detector's internal directions. The detector accepts either a scalar
    # or a sequence of values.
    current = None
    if trial.values is not None:
        current = tuple(trial.values)
    else:
        # Fallback to None (detectors should handle this safely)
        current = None

    has_converged, converged_methods = convergence_detector.update(
        current, trial.number
    )

    if has_converged:
        study.set_user_attr("converged", True)
        study.set_user_attr("converged_methods", converged_methods)

        methods = ", ".join(converged_methods) if converged_methods else "unknown"
        msg = (
            f"\n##################################################################\n"
            f"\nOptimization converged at trial: {trial.number}\n"
            f"Convergence detected by: {methods}\n"
            f"\n##################################################################"
        )
        logging.info(msg)
        study.stop()
    else:
        methods = ", ".join(converged_methods) if converged_methods else "none"
        logging.info(
            f"Majority convergence not yet met. Partial convergence detected by: {methods}"
        )


# Callback to track best trial so far using aggregate_score user attribute
def best_tracker_callback(study, trial):
    try:
        s = trial.user_attrs.get("aggregate_score", None)
        if s is None:
            return

        best_score = study.user_attrs.get("best_score", float("-inf"))
        if s > best_score:
            # update stored best
            study.set_user_attr("best_score", float(s))
            study.set_user_attr("best_trial", trial.number)
            params = trial.user_attrs.get("params", dict(trial.params))
            logging.info(
                f"New best so far: trial {trial.number} score={s} params={params}"
            )
    except Exception:
        return


# Run optimization
try:
    # Create a multi-objective Optuna study. Directions derived from evaluator weights.
    objective_keys = objective_builder.objectives
    directions = [
        "maximize" if evaluator.weights.get(k, 0) > 0 else "minimize"
        for k in objective_keys
    ]
    # Create convergence detector configured for multi-objective directions
    convergence_detector = MultiCriteriaConvergenceDetector(directions=directions)

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
        sampler=optuna.samplers.TPESampler(),
    )

    # Existing study handling: if not forcing rerun, continue from existing trials.
    if not args.force_rerun and len(study.trials) > 0:
        logging.info(
            f"Resuming existing study with {len(study.trials)} trials. Use --force-rerun to start fresh."
        )

    logging.info(
        "Starting Optuna multi-objective optimization with up to %d trials.",
        args.num_trials,
    )
    # No single-objective convergence callbacks are used for multi-objective runs
    study.optimize(
        objective,
        n_trials=args.num_trials,
        callbacks=[best_tracker_callback, convergence_callback],
    )
    # Multi-objective optimization completed. No single "best" trial exists.
    if len(study.trials) == 0:
        logging.info("No successful trials were completed.")
    else:
        logging.info(
            f"Study completed with {len(study.trials)} trials. Use the Pareto front and per-trial metrics to select a preferred solution."
        )

        # Track best trial by aggregate_score user attribute (higher is better)
        best_trial = None
        best_score = float("-inf")
        for t in study.trials:
            try:
                s = t.user_attrs.get("aggregate_score", None)
                if s is not None and s > best_score:
                    best_score = s
                    best_trial = t
            except Exception:
                continue

        if best_trial is not None:
            logging.info(
                f"\n############################################\n"
                f"\nBest trial by aggregate_score: {best_trial.number} score={best_score} params={best_trial.user_attrs.get('params', {})}\n"
                f"\n############################################\n"
            )

except Exception as e:
    logging.error(f"Optimization failed: {e}", exc_info=True)
    raise

# --- OPTUNA VISUALIZATIONS ---

# Create output directory
optuna_dir = Path("optuna_output")
optuna_dir.mkdir(exist_ok=True, parents=True)

# Save visualizations as interactive HTML files per metric (multi-objective requires a target)
for metric in objective_builder.objectives:
    try:
        metric_dir = optuna_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        try:
            vis.plot_optimization_history(study, target=metric).write_html(
                metric_dir / "optuna_optimization_history.html"
            )
        except Exception as e:
            logging.warning(
                f"[{metric}] Failed to create optimization history plot: {e}"
            )

        try:
            vis.plot_param_importances(study, target=metric).write_html(
                metric_dir / "optuna_param_importance.html"
            )
        except Exception as e:
            logging.warning(f"[{metric}] Failed to create param importance plot: {e}")

        try:
            vis.plot_parallel_coordinate(study, target=metric).write_html(
                metric_dir / "optuna_parallel_coordinates.html"
            )
        except Exception as e:
            logging.warning(
                f"[{metric}] Failed to create parallel coordinate plot: {e}"
            )

        try:
            vis.plot_contour(study, target=metric).write_html(
                metric_dir / "optuna_contour_plot.html"
            )
        except Exception as e:
            logging.warning(f"[{metric}] Failed to create contour plot: {e}")

    except Exception as e:
        logging.warning(f"Failed to create visualizations for metric {metric}: {e}")

# Additional per-metric plots and Pareto front (for multi-objective)
try:
    # Determine metric keys: for multi-objective, use objective_builder.objectives
    # Pareto front
    try:
        vis.plot_pareto_front(study).write_html(optuna_dir / "optuna_pareto_front.html")
    except Exception as e:
        logging.warning(f"Failed to create pareto front plot: {e}")

    # Per-objective optimization history from trial.values
    for idx, metric in enumerate(objective_builder.objectives):
        xs = []
        ys = []
        for t in study.trials:
            if t.values is None:
                continue
            try:
                val = t.values[idx]
                if val is None or (
                    isinstance(val, float) and (math.isnan(val) or math.isinf(val))
                ):
                    continue
                xs.append(t.number)
                ys.append(val)
            except Exception:
                continue
        if xs and ys:
            plt.figure()
            plt.scatter(xs, ys)
            plt.xlabel("trial")
            plt.ylabel(metric)
            plt.title(f"Optimization history: {metric}")
            # save PNG per-metric into metric directory if created, else optuna_dir
            metric_dir = optuna_dir / metric
            if not metric_dir.exists():
                metric_dir = optuna_dir
            plt.savefig(metric_dir / f"optuna_history_{metric}.png")
            plt.close()
except Exception as e:
    logging.warning(f"Failed to create additional metric visualizations: {e}")

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
        )

        # copy relevant optimized parameters if present
        for key in [
            "x",
            "y",
            "s",
            "n",
            "m",
            "p",
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
                        logging.info(f"Final assembly metrics: {metrics}")
                    except Exception as e:
                        logging.error(f"Final assembly evaluation failed: {e}")

            else:
                logging.error(f"Final hifiasm run exited with code {rc}")

        # Remove trial assemblies and associated files
        try:
            import shutil

            removed = 0
            for p in Path.cwd().rglob("trial_assembly*"):
                try:
                    if p.is_file():
                        p.unlink()
                        removed += 1
                    elif p.is_dir():
                        shutil.rmtree(p)
                        removed += 1
                except Exception:
                    logging.warning(f"Failed to remove trial artifact: {p}")
            logging.info(f"Cleaned up {removed} trial assembly artifacts.")
        except Exception as e:
            logging.warning(f"Failed to clean trial assemblies: {e}")
    else:
        logging.info("No best trial identified; skipping final assembly run.")
except Exception as e:
    logging.error(f"Final assembly/evaluation failed: {e}")
