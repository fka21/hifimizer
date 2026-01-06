#!/usr/bin/env python3

import subprocess, logging, sys, optuna, psutil, signal, os, numpy as np
from pathlib import Path
import optuna.visualization as vis
import optunahub
import matplotlib.pyplot as plt
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

# Create logs directory - FIXED VERSION
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

# Instantiate the early stopping callback with default parameters
convergence_detector = MultiCriteriaConvergenceDetector(max_trials=args.num_trials)


# Define the convergence callback function
def convergence_callback(study, trial):
    has_converged, converged_methods = convergence_detector.update(
        study.best_value, trial.number
    )

    if has_converged:
        # Setting user attributes for convergence status
        study.set_user_attr("converged", True)
        study.set_user_attr("converged_methods", converged_methods)

        methods = ", ".join(converged_methods) if converged_methods else "unknown"
        msg = (
            f"##################################################################\n"
            f"\nOptimization converged at trial: {trial.number}\n"
            f"Best objective value: {study.best_value:.6f}\n"
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


# Run optimization
try:
    study = optuna.create_study(
        study_name="no-name",
        direction="maximize",
        storage=evaluator.db_uri,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
    )

    # --- Check if previous convergence has been reached ---
    if not args.force_rerun:
        logging.info("Checking for previous convergence status in the study...")
        # Detect if there was already convergence in the stored study
        if len(study.trials) > 0:
            logging.info(
                f"Study has {len(study.trials)} trials. Checking convergence status..."
            )
            best_trial = study.best_trial
            best_value = best_trial.value if best_trial else None

            # Check if previous convergence marker exists
            if study.user_attrs.get("converged", False):
                logging.info(
                    f"Convergence was already met in a previous run. "
                    f"Best trial: {best_trial.number if best_trial else 'unknown'}. "
                    f"Use --force_rerun to override."
                )
                sys.exit(0)
    else:
        logging.info("Force rerun requested. Ignoring previous convergence status.")

    logging.info("Starting Optuna optimization with up to %d trials.", args.num_trials)
    study.optimize(
        objective,
        n_trials=args.num_trials,
        callbacks=[convergence_callback, tpe_acquisition_visualizer],
    )

    if study.best_trial is not None:
        trial = study.best_trial
        # Cleanup trial assemblies if optimization finished successfully
        patterns = ["trial_*", "alignment.sam", "subset_reads.fa"]

        # Check if best trial is the default (trial 0)
        if trial.number == 0:
            logging.info(
                "Best trial is the default assembly (trial 0). "
                "No improvement over baseline was found. "
                "You may try increasing the number of trials, "
                "though this does not guarantee a better solution."
                "You can find the assembly in the default_assembly/ directory."
            )

            logging.info(f"Cleaning up intermediate files in {output_dir}...")
            for pattern in patterns:
                for item in output_dir.glob(pattern):
                    try:
                        if item.is_dir():
                            import shutil

                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        logging.info(f"Removed {item}")
                    except Exception as cleanup_err:
                        logging.warning(f"Failed to remove {item}: {cleanup_err}")
            # Nothing more to do here — skip tuned assembly
        else:
            logging.info("Best trial found:")
            logging.info(f"  Value: {trial.value}")
            logging.info("  Params:")
            for key, value in trial.params.items():
                logging.info(f"    {key}: {value}")

            # Setting up the final assembly command
            base_name = "final_assembly"

            # Choose the suffix based on inputs
            if args.hic1 and args.hic2:
                suffix = "hic.hap1.p_ctg"
            elif args.ul:
                suffix = "bp.hap1.p_ctg"
            else:
                suffix = "bp.p_ctg"

            # Run final assembly with best parameters
            best_x = trial.params["x"]
            best_y = trial.params["y"]
            best_s = trial.params["s"]
            best_n = trial.params["n"]
            best_m = trial.params["m"]
            best_p = trial.params["p"]

            final_hic_params = {}
            final_ont_params = {}

            if args.hic1 and args.hic2:
                final_hic_params.update(
                    {
                        "s_base": trial.params["s_base"],
                        "f_perturb": trial.params["f_perturb"],
                        "l_msjoin": trial.params["l_msjoin"],
                    }
                )

            if args.ul:
                final_ont_params.update(
                    {
                        "path_max": trial.params["path_max"],
                        "path_min": trial.params["path_min"],
                    }
                )

            if args.sensitive:
                final_command = build_hifiasm_command(
                    prefix=base_name,
                    x=best_x,
                    y=best_y,
                    s=best_s,
                    n=best_n,
                    m=best_m,
                    p=best_p,
                    haploid_genome_size=KNOWN_GENOME_SIZE,
                    threads=threads,
                    sensitive=True,
                    D=trial.params["D"],
                    N=trial.params["N"],
                    max_kocc=trial.params["max_kocc"],
                    hic1=hic1,
                    hic2=hic2,
                    ul=ul,
                    **final_hic_params,
                    **final_ont_params,
                    primary=args.primary,
                )
            else:
                final_command = build_hifiasm_command(
                    prefix=base_name,
                    x=best_x,
                    y=best_y,
                    s=best_s,
                    n=best_n,
                    m=best_m,
                    p=best_p,
                    haploid_genome_size=KNOWN_GENOME_SIZE,
                    threads=threads,
                    sensitive=False,
                    hic1=hic1,
                    hic2=hic2,
                    ul=ul,
                    **final_hic_params,
                    **final_ont_params,
                    primary=args.primary,
                )

            final_command += f" {input_reads}"

            # Define file names
            gfa_file = f"{base_name}.{suffix}.gfa"
            fasta_file = f"{base_name}.{suffix}.fasta"

            if Path(fasta_file).exists():
                logging.info(
                    f"Final assembly FASTA {fasta_file} already exists. Skipping final assembly step."
                )
            else:
                logging.info(
                    "Running final assembly with best parameters:\n%s", final_command
                )
                final_hifiasm_log = logs_dir / "final_hifiasm.log"
                with open(final_hifiasm_log, "w") as logf:
                    subprocess.run(
                        final_command, shell=True, stdout=logf, stderr=subprocess.STDOUT
                    )

                evaluator.convert_gfa_to_fasta(
                    gfa_file=gfa_file, output_fasta=fasta_file
                )

                logging.info("Final assembly completed.")

                # Evaluate the final assembly using AssemblyEvaluator
                try:
                    logging.info("Evaluating final assembly with AssemblyEvaluator")
                    final_score = evaluator.evaluate_assembly(
                        gfa_file=gfa_file,
                        fasta_file=fasta_file,
                        include_busco=args.include_busco,
                        busco_lineage=args.busco_lineage,
                        download_path=download_path,
                    )
                    logging.info(f"Final assembly evaluation score: {final_score}")
                except Exception as eval_err:
                    logging.error(
                        f"Final assembly evaluation failed: {eval_err}", exc_info=True
                    )

            # Cleanup trial assemblies if optimization finished successfully
            logging.info(f"Cleaning up intermediate files in {output_dir}...")
            for pattern in patterns:
                for item in output_dir.glob(pattern):
                    try:
                        if item.is_dir():
                            import shutil

                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        logging.info(f"Removed {item}")
                    except Exception as cleanup_err:
                        logging.warning(f"Failed to remove {item}: {cleanup_err}")

    else:
        logging.info("No successful trials were completed.")

except Exception as e:
    logging.error(f"Optimization failed: {e}", exc_info=True)
    raise

# --- OPTUNA VISUALIZATIONS ---

# Create output directory
optuna_dir = Path("optuna_output")
optuna_dir.mkdir(exist_ok=True, parents=True)

# Save visualizations as interactive HTML files
vis.plot_optimization_history(study).write_html(
    optuna_dir / "optuna_optimization_history.html"
)
vis.plot_param_importances(study).write_html(
    optuna_dir / "optuna_param_importance.html"
)
vis.plot_parallel_coordinate(study).write_html(
    optuna_dir / "optuna_parallel_coordinates.html"
)
vis.plot_contour(study).write_html(optuna_dir / "optuna_contour_plot.html")

# Extract parameter names for acquisition function visualization
param_names = list(study.best_trial.params.keys())
# Remove params that are not always present
for t in study.trials:
    t.params = {k: v for k, v in t.params.items() if v is not None}
# Extract TPE samples trials
available_trials = list(tpe_acquisition_visualizer.log_objects.keys())

output_dir = "optuna_output/acquisition_plots"
os.makedirs(output_dir, exist_ok=True)

for trial_num in available_trials:
    if (
        trial_num == 0
    ):  # skip the default assembly trial as this does not parameters set
        continue
    for param in param_names:
        try:
            fig = tpe_acquisition_visualizer.plot(study, trial_num, param)
            fig.savefig(
                os.path.join(output_dir, f"acquisition_trial_{trial_num}_{param}.png")
            )
            plt.close(fig)
        except Exception as e:
            print(f"Failed to plot trial {trial_num}, param {param}: {e}")
