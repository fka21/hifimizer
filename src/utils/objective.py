import subprocess
import shutil
import logging
from pathlib import Path
from utils.hifiasm_command import build_hifiasm_command
from utils.subprocess_logger import SubprocessLogger
from utils.assembly_eval import AssemblyEvaluator

def _load_directions_map():
    import json
    from pathlib import Path

    # Same convention you already use elsewhere
    directions_file = Path(__file__).resolve().parent.parent / "optim_directions.json"
    if directions_file.exists():
        with open(directions_file, "r") as fh:
            return json.load(fh) or {}
    return {}


class ObjectiveBuilder:
    def __init__(
        self,
        evaluator,
        input_reads,
        haploid_genome_size,
        threads,
        hic1=None,
        hic2=None,
        ul=None,
        sensitive=False,
        primary=False,
        include_busco=True,
        busco_lineage="metazoa_odb12",
        download_path=None,
        output_dir=None,
        logs_dir=None,
        # multi-objective is the only supported mode now
        objectives=None,
        is_multi_objective=False,
    ):
        """
        Initialize the objective builder with necessary configuration.

        Args:
            evaluator: Assembly evaluator instance
            input_reads: Path to input reads
            haploid_genome_size: Genome size
            threads: Number of threads to use
            hic1: Path to Hi-C R1 reads file (optional)
            hic2: Path to Hi-C R2 reads file (optional)
            ul: Path to ultra-long ONT reads file (optional)
            sensitive: Whether to optimize for sensitivity
            primary: Whether to perform primary assembly only
            include_busco: Whether to include BUSCO metrics
        """
        self.evaluator = evaluator
        self.input_reads = input_reads
        self.haploid_genome_size = haploid_genome_size
        self.threads = threads
        self.hic1 = hic1
        self.hic2 = hic2
        self.ul = ul
        self.logs_dir = logs_dir if logs_dir else Path.cwd() / "logs"
        self.sensitive = sensitive
        self.primary = primary
        self.include_busco = include_busco
        self.busco_lineage = busco_lineage
        self.download_path = download_path
        self.subprocess_logger = SubprocessLogger(logs_dir=Path.cwd() / "logs")

        # Set up output directory for default assembly results
        if output_dir is None:
            self.output_dir = Path.cwd()
        else:
            self.output_dir = Path(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Support both single and multi-objective optimization
        self.is_multi_objective = is_multi_objective
        if objectives:
            self.objectives = objectives
        else:
            # preserve order from evaluator.weights
            try:
                self.objectives = list(self.evaluator.weights.keys())
            except Exception:
                self.objectives = ["n50", "single_copy", "missing"]

    def build_objective(self):
        """
        Build and return the objective function for Optuna optimization.
        """

        def objective(trial):
            # Get trial number for logging
            trial_id = trial.number

            # Initialize evaluator with trial ID
            evaluator = AssemblyEvaluator(
                known_genome_size=self.evaluator.known_genome_size,
                input_reads=self.evaluator.input_reads,
                threads=self.threads,
                trial_id=trial_id,
                download_path=self.download_path,
                logs_dir=self.logs_dir,
            )

            # Define base assembly name
            base_name = "trial_assembly"

            # Choose the suffix based on inputs
            if self.hic1 and self.hic2:
                suffix = "hic.hap1.p_ctg"
            elif self.ul:
                suffix = "bp.hap1.p_ctg"
            else:
                suffix = "bp.p_ctg"

            # Define file names
            gfa_file = f"{base_name}.{suffix}.gfa"
            fasta_file = f"{base_name}.{suffix}.fasta"

            # Assenbly to compare to should be the assembly generated with default values
            if trial_id == 0:
                params = {
                    "base_name": "default_assembly",
                    "haploid_genome_size": self.haploid_genome_size,
                    "threads": self.threads,
                    "sensitive": self.sensitive,
                    "hic1": self.hic1,
                    "hic2": self.hic2,
                    "ul": self.ul,
                    "primary": self.primary,
                }
                # Only keep non-None params
                params = {k: v for k, v in params.items() if v is not None}

                command = build_hifiasm_command(**params)
                command += f" {self.input_reads}"
            else:
                # Parameters to optimize by default
                x = trial.suggest_float("x", 0.59, 0.99, step=0.01)
                y = trial.suggest_float("y", 0.01, 0.41, step=0.01)
                s = trial.suggest_float("s", 0.55, 1, step=0.01)
                n = trial.suggest_int("n", 0, 10)
                m = trial.suggest_int("m", 500_000, 20_000_000, log=True)
                p = trial.suggest_int("p", 1, 10_000, log=True)
                u = trial.suggest_categorical("u", [0, 1])

                hic_params = {}
                ont_params = {}

                # Hi-C parameter space
                if self.hic1 and self.hic2:
                    hic_params.update(
                        {
                            "s_base": trial.suggest_float("s_base", 0, 1, step=0.05),
                            "f_perturb": trial.suggest_float(
                                "f_perturb", 0, 1, step=0.05
                            ),
                            "l_msjoin": trial.suggest_int(
                                "l_msjoin", 0, 10_000_000, log=True
                            ),
                        }
                    )

                # UL ONT parameter space
                if self.ul:
                    ont_params.update(
                        {
                            "path_max": trial.suggest_float(
                                "path_max", 0.0, 1.0, step=0.05
                            ),
                            "path_min": trial.suggest_float(
                                "path_min", 0.0, 1.0, step=0.05
                            ),
                        }
                    )

                if self.sensitive:
                    D = trial.suggest_int("D", 3, 20, step=1)
                    N = trial.suggest_int("N", 50, 400, step=10)
                    max_kocc = trial.suggest_int("max_kocc", 1000, 5000, step=100)

                    command = build_hifiasm_command(
                        x=x,
                        y=y,
                        s=s,
                        n=n,
                        m=m,
                        p=p,
                        u=u,
                        haploid_genome_size=self.haploid_genome_size,
                        threads=self.threads,
                        sensitive=True,
                        D=D,
                        N=N,
                        max_kocc=max_kocc,
                        hic1=self.hic1,
                        hic2=self.hic2,
                        ul=self.ul,
                        **hic_params,
                        **ont_params,
                        primary=self.primary,
                    )
                else:
                    command = build_hifiasm_command(
                        x=x,
                        y=y,
                        s=s,
                        n=n,
                        m=m,
                        p=p,
                        u=u,
                        haploid_genome_size=self.haploid_genome_size,
                        threads=self.threads,
                        sensitive=False,
                        hic1=self.hic1,
                        hic2=self.hic2,
                        ul=self.ul,
                        **hic_params,
                        **ont_params,
                        primary=self.primary,
                    )

                command += f" {self.input_reads}"

            try:
                # Run hifiasm with dedicated logging
                return_code, log_path = self.subprocess_logger.run_command_with_logging(
                    command=command,
                    log_filename="hifiasm.log",
                    command_name="hifiasm",
                    trial_id=trial_id,
                    timeout_seconds=24 * 3600,
                )

                if return_code != 0:
                    raise RuntimeError(f"Hifiasm failed - see {log_path}")

                # Check if output exists
                if not Path(gfa_file).exists():
                    raise FileNotFoundError(f"GFA file not found: {gfa_file}")

                # Evaluate assembly
                logging.info(f"Trial {trial_id}: Evaluating assembly")
                # Request raw metrics (multi-criteria optimization)
                metrics = evaluator.evaluate_assembly(
                    gfa_file=gfa_file,
                    fasta_file=fasta_file,
                    include_busco=self.include_busco,
                    busco_lineage=self.busco_lineage,
                    download_path=self.download_path,
                )

                if metrics is None:
                    raise RuntimeError("Evaluation returned no metrics")

                # Attach metrics to the trial for later analysis/visualization
                try:
                    for k, v in metrics.items():
                        trial.set_user_attr(k, float(v))
                except Exception:
                    pass

                # Save results of default assembly into separate folder for trial 0
                if trial_id == 0:
                    default_dir = self.output_dir / "default_assembly"
                    default_dir.mkdir(parents=True, exist_ok=True)
                    for file in Path(".").glob("trial_assembly*"):
                        shutil.copy(str(file), default_dir / file.name)
                    logging.info(
                        f"Default setting based assembly results moved to {default_dir.resolve()}"
                    )

                # Return based on optimization mode
                if self.is_multi_objective:
                    # Multi-objective: return tuple of objective values
                    objective_values = tuple(
                        float(metrics.get(k, 0)) for k in self.objectives
                    )

                    # Compute a simple aggregate score for bookkeeping: signed average
                    # Use optim_directions.json to determine optimization direction
                    try:
                        import json

                        directions_file = (
                            Path(__file__).resolve().parent.parent
                            / "optim_directions.json"
                        )
                        if directions_file.exists():
                            with open(directions_file, "r") as fh:
                                directions_map = json.load(fh)
                        else:
                            directions_map = {}

                        # Score normalization: maximize means positive contribution, minimize means negative
                        signs = []
                        for obj in self.objectives:
                            dir_str = directions_map.get(obj, "maximize")
                            sign = 1 if dir_str == "maximize" else -1
                            signs.append(sign)

                        # Normalize metric values to [0, 1] scale when possible for fair aggregation
                        normalized_metrics = []
                        for k in self.objectives:
                            raw_val = float(metrics.get(k, 0))
                            # For aggregate scoring, we just use the raw values with appropriate sign
                            # The actual Pareto optimization happens via the multi-objective framework
                            normalized_metrics.append(raw_val)

                        agg = sum(
                            s * v for s, v in zip(signs, normalized_metrics)
                        ) / max(1, len(self.objectives))
                    except Exception as e:
                        logging.warning(f"Failed to compute aggregate score: {e}")
                        agg = 0.0

                    # Store aggregate score and trial parameters as user attributes for later inspection
                    try:
                        trial.set_user_attr("aggregate_score", float(agg))
                        trial.set_user_attr("params", dict(trial.params))
                    except Exception:
                        pass

                    # Minimal logging: success and parameters
                    logging.info(
                        f"Trial {trial_id}: Completed successfully. Params: {dict(trial.params)}"
                    )
                    return objective_values
                else:
                    # Single-objective: return weighted sum
                    weighted_score = self.evaluator.calculate_weighted_sum(metrics)

                    # Analyze metric contributions
                    contribution_analysis = self.evaluator.analyze_metric_contributions(
                        metrics
                    )

                    # Store score as user attribute for tracking
                    try:
                        trial.set_user_attr("weighted_score", float(weighted_score))
                        trial.set_user_attr("params", dict(trial.params))
                    except Exception:
                        pass

                    # Log the weighted score and metric contributions
                    logging.info(
                        f"Trial {trial_id}: Completed successfully. Weighted score: {weighted_score:.2f}"
                    )

                    # Log metric contributions
                    directions_map = _load_directions_map()

                    contribs = contribution_analysis["contributions"]
                    pos_sum = float(contribution_analysis.get("positive_sum", 0.0))
                    neg_sum = float(contribution_analysis.get("negative_sum", 0.0))

                    # Partition by desired optimization direction, not by sign
                    maximize_metrics = []
                    minimize_metrics = []
                    unknown_metrics = []

                    for m, d in contribs.items():
                        direction = directions_map.get(m, "unknown")
                        if direction == "maximize":
                            maximize_metrics.append((m, d))
                        elif direction == "minimize":
                            minimize_metrics.append((m, d))
                        else:
                            unknown_metrics.append((m, d))

                    def _log_block(title, items, denom_pos, denom_neg):
                        logging.info(title)
                        for metric_name, data in items:
                            contrib = float(data["contribution"])
                            # “reward share” if positive, “penalty share” if negative
                            if contrib >= 0 and denom_pos > 0:
                                share = 100.0 * (contrib / denom_pos)
                                share_label = "reward_share"
                            elif contrib < 0 and denom_neg > 0:
                                share = 100.0 * (abs(contrib) / denom_neg)
                                share_label = "penalty_share"
                            else:
                                share = 0.0
                                share_label = "share"

                            logging.info(
                                f"  {metric_name:25s} | log_value: {data['log_value']:12.2f} | "
                                f"weight: {data['weight']:6.2f} | "
                                f"contribution: {contrib:9.4f} | "
                                f"{share_label}: {share:6.2f}%"
                            )

                    logging.info(f"\nTrial {trial_id} Metric Contributions (direction-aware):")

                    _log_block("Maximize metrics (higher is better):", maximize_metrics, pos_sum, neg_sum)
                    _log_block("Minimize metrics (lower is better):", minimize_metrics, pos_sum, neg_sum)

                    if unknown_metrics:
                        _log_block("Unknown-direction metrics (check optim_directions.json):", unknown_metrics, pos_sum, neg_sum)

                    logging.info(
                        f"  {'TOTAL':25s} | Positive sum: {pos_sum:.4f} | Negative sum: {neg_sum:.4f}\n"
                    )


                    return weighted_score

            except (
                TimeoutError,
                FileNotFoundError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
            ) as e:
                stage = self._determine_failure_stage(e, gfa_file)
                logging.error(f"Trial {trial_id}: Failed at {stage} - {str(e)}")
                import optuna

                raise optuna.exceptions.TrialPruned(f"Trial pruned at {stage}: {e}")

        return objective

    def _determine_failure_stage(self, error, gfa_file):
        """Determine which stage the trial failed at."""
        if "hifiasm" in str(error).lower():
            return "hifiasm assembly"
        elif not Path(gfa_file).exists():
            return "assembly output generation"
        else:
            return "assembly evaluation"
