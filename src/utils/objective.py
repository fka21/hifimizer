import json
import shutil
import subprocess
import logging
from pathlib import Path

from utils.hifiasm_command import build_hifiasm_command
from utils.subprocess_logger import SubprocessLogger, TIMEOUT_EXIT_CODE
from utils.assembly_eval import AssemblyEvaluator
from utils.paths import RunPaths


def _load_directions_map():
    directions_file = Path(__file__).resolve().parent.parent / "optim_directions.json"
    if directions_file.exists():
        with open(directions_file, "r") as fh:
            return json.load(fh) or {}
    return {}


def haplotype_suffixes(hic1, hic2, ul):
    """
    Return (primary_suffix, extra_haplotype_suffixes) for the hifiasm outputs.

    The extra suffixes are only used to widen the k-mer completeness estimate
    across both haplotypes; QV and everything else stay on the primary.
    """
    if hic1 and hic2:
        return "hic.hap1.p_ctg", ["hic.hap2.p_ctg"]
    if ul:
        return "bp.hap1.p_ctg", ["bp.hap2.p_ctg"]
    # Plain HiFi: bp.p_ctg is already the primary (haplotype-collapsed) set.
    return "bp.p_ctg", []


class ObjectiveBuilder:
    def __init__(
        self,
        evaluator,
        input_reads,
        haploid_genome_size,
        threads,
        paths: RunPaths,
        hic1=None,
        hic2=None,
        ul=None,
        sensitive=False,
        primary=False,
        include_busco=True,
        busco_lineage="metazoa_odb12",
        download_path=None,
        ont=False,
        trial_walltime_hours=24.0,
        busco_walltime_hours=6.0,
        craq_walltime_hours=6.0,
        craq_mapq=20,
        kmer_eval=True,
        objectives=None,
        is_multi_objective=False,
        hom_cov=None,
    ):
        self.evaluator = evaluator
        self.input_reads = input_reads
        self.haploid_genome_size = haploid_genome_size
        self.threads = threads
        self.paths = paths
        self.hic1 = hic1
        self.hic2 = hic2
        self.ul = ul
        self.sensitive = sensitive
        self.primary = primary
        self.include_busco = include_busco
        self.busco_lineage = busco_lineage
        self.download_path = download_path
        self.ont = ont
        self.hom_cov = hom_cov
        self.trial_walltime_hours = trial_walltime_hours
        self.busco_walltime_hours = busco_walltime_hours
        self.craq_walltime_hours = craq_walltime_hours
        self.craq_mapq = craq_mapq
        self.kmer_eval = kmer_eval

        self.subprocess_logger = SubprocessLogger(logs_dir=paths.logs_dir)
        self.output_dir = paths.output_dir

        self.is_multi_objective = is_multi_objective
        if objectives:
            self.objectives = objectives
        else:
            try:
                self.objectives = list(self.evaluator.active_weights().keys())
            except Exception:
                self.objectives = ["n50", "single_copy", "missing"]

        self.directions_map = _load_directions_map()

    # ------------------------------------------------------------------ files
    def _archive_default_assembly(self):
        """
        Copy trial 0's outputs into ``output_dir/default_assembly/``.

        Trial 0 runs hifiasm with default parameters but under the *same*
        prefix as every other trial, so that hifiasm's error-corrected read
        and overlap .bin files are reused by trials 1..N.  Its results are
        therefore copied out before trial 1 overwrites them.
        """
        dest = self.paths.default_assembly_dir
        dest.mkdir(parents=True, exist_ok=True)
        prefix = self.paths.hifiasm_prefix
        copied = 0
        for f in sorted(prefix.parent.glob(f"{prefix.name}*")):
            if not f.is_file() or f.suffix == ".bin":
                continue
            target = dest / f"default_assembly{f.name[len(prefix.name):]}"
            try:
                shutil.copy2(f, target)
                copied += 1
            except Exception as e:
                logging.warning(f"Could not copy {f.name} to {dest}: {e}")
        logging.info(f"Copied {copied} default-assembly file(s) to {dest}")

    # -------------------------------------------------------------- objective
    def build_objective(self):
        """Build and return the objective function for Optuna."""

        def objective(trial):
            trial_id = trial.number

            evaluator = AssemblyEvaluator(
                known_genome_size=self.evaluator.known_genome_size,
                input_reads=self.evaluator.input_reads,
                paths=self.paths,
                threads=self.threads,
                trial_id=trial_id,
                download_path=self.download_path,
                ont=self.ont,
                busco_walltime_hours=self.busco_walltime_hours,
                craq_walltime_hours=self.craq_walltime_hours,
                craq_mapq=self.craq_mapq,
                kmer_eval=self.kmer_eval,
            )

            prefix = self.paths.hifiasm_prefix
            suffix, extra_suffixes = haplotype_suffixes(self.hic1, self.hic2, self.ul)

            gfa_file = prefix.parent / f"{prefix.name}.{suffix}.gfa"
            fasta_file = evaluator.trial_dir / f"{prefix.name}.{suffix}.fasta"

            # Trial 0 is the baseline: hifiasm with default parameters. It stays
            # in the study so that it shows up in the optimisation history as a
            # reference point; the parameter-importance plots filter it out
            # separately (see hifimizer.plot_param_importances).
            if trial_id == 0:
                params = {
                    "prefix": str(prefix),
                    "haploid_genome_size": self.haploid_genome_size,
                    "threads": self.threads,
                    "sensitive": self.sensitive,
                    "hic1": self.hic1,
                    "hic2": self.hic2,
                    "ul": self.ul,
                    "primary": self.primary,
                    "ont": self.ont,
                    "hom_cov": self.hom_cov,
                }
                params = {k: v for k, v in params.items() if v is not None}
                command = build_hifiasm_command(**params) + f" {self.input_reads}"
            else:
                x = trial.suggest_float("x", 0.59, 0.99, step=0.01)
                y = trial.suggest_float("y", 0.01, 0.41, step=0.01)
                s = trial.suggest_float("s", 0.55, 1, step=0.01)
                n = trial.suggest_int("n", 0, 10)
                m = trial.suggest_int("m", 500_000, 20_000_000, log=True)
                p = trial.suggest_int("p", 1, 10_000, log=True)
                u = trial.suggest_categorical("u", [0, 1])

                hic_params = {}
                ont_params = {}

                if self.hic1 and self.hic2:
                    hic_params.update(
                        {
                            "s_base": trial.suggest_float("s_base", 0, 1, step=0.05),
                            "f_perturb": trial.suggest_float(
                                "f_perturb", 0, 1, step=0.05
                            ),
                            "l_msjoin": trial.suggest_int(
                                "l_msjoin", 1, 10_000_000, log=True
                            ),
                        }
                    )

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

                sensitive_params = {}
                if self.sensitive:
                    sensitive_params.update(
                        {
                            "D": trial.suggest_int("D", 3, 20, step=1),
                            "N": trial.suggest_int("N", 50, 400, step=10),
                            "max_kocc": trial.suggest_int(
                                "max_kocc", 1000, 5000, step=100
                            ),
                        }
                    )

                command = build_hifiasm_command(
                    prefix=str(prefix),
                    x=x,
                    y=y,
                    s=s,
                    n=n,
                    m=m,
                    p=p,
                    u=u,
                    haploid_genome_size=self.haploid_genome_size,
                    threads=self.threads,
                    sensitive=self.sensitive,
                    hic1=self.hic1,
                    hic2=self.hic2,
                    ul=self.ul,
                    **sensitive_params,
                    **hic_params,
                    **ont_params,
                    primary=self.primary,
                    ont=self.ont,
                    hom_cov=self.hom_cov,
                )
                command += f" {self.input_reads}"

            try:
                return_code, log_path = self.subprocess_logger.run_command_with_logging(
                    command=command,
                    log_filename="hifiasm.log",
                    command_name="hifiasm",
                    trial_id=trial_id,
                    timeout_seconds=self.trial_walltime_hours * 3600,
                    cwd=self.paths.hifiasm_dir,
                )

                if return_code == TIMEOUT_EXIT_CODE:
                    logging.warning(
                        f"Trial {trial_id}: hifiasm exceeded the per-trial walltime "
                        f"({self.trial_walltime_hours:.1f} h) and was killed. "
                        "Consider increasing --trial-walltime."
                    )
                    raise RuntimeError(
                        f"Trial {trial_id} timed out after "
                        f"{self.trial_walltime_hours:.1f} h"
                    )

                if return_code != 0:
                    raise RuntimeError(f"Hifiasm failed - see {log_path}")

                if not gfa_file.exists():
                    raise FileNotFoundError(f"GFA file not found: {gfa_file}")

                # Sibling haplotype GFAs, used only for k-mer completeness.
                extra_fastas = []
                for extra in extra_suffixes:
                    extra_gfa = prefix.parent / f"{prefix.name}.{extra}.gfa"
                    if extra_gfa.exists():
                        extra_fasta = evaluator.trial_dir / f"{extra}.fasta"
                        try:
                            AssemblyEvaluator.convert_gfa_to_fasta(
                                extra_gfa, extra_fasta
                            )
                            extra_fastas.append(extra_fasta)
                        except Exception as e:
                            logging.warning(
                                f"Trial {trial_id}: could not convert {extra_gfa.name}: {e}"
                            )

                logging.info(f"Trial {trial_id}: Evaluating assembly")
                metrics = evaluator.evaluate_assembly(
                    gfa_file=gfa_file,
                    fasta_file=fasta_file,
                    include_busco=self.include_busco,
                    busco_lineage=self.busco_lineage,
                    extra_fasta_files=extra_fastas,
                )

                if not metrics:
                    raise RuntimeError("Evaluation returned no metrics")

                try:
                    for k, v in metrics.items():
                        trial.set_user_attr(k, float(v))
                except Exception as e:
                    logging.debug(
                        f"Trial {trial_id}: failed to set metric user attrs: {e}"
                    )

                if trial_id == 0:
                    self._archive_default_assembly()

                if self.is_multi_objective:
                    objective_values = tuple(
                        float(metrics.get(k, 0)) for k in self.objectives
                    )

                    try:
                        signs = [
                            1
                            if self.directions_map.get(obj, "maximize") == "maximize"
                            else -1
                            for obj in self.objectives
                        ]
                        agg = sum(
                            sign * float(metrics.get(k, 0))
                            for sign, k in zip(signs, self.objectives)
                        ) / max(1, len(self.objectives))
                    except Exception as e:
                        logging.warning(f"Failed to compute aggregate score: {e}")
                        agg = 0.0

                    try:
                        trial.set_user_attr("aggregate_score", float(agg))
                        trial.set_user_attr("params", dict(trial.params))
                    except Exception as e:
                        logging.debug(
                            f"Trial {trial_id}: failed to set aggregate user attrs: {e}"
                        )

                    logging.info(
                        f"Trial {trial_id}: Completed successfully. "
                        f"Params: {dict(trial.params)}"
                    )
                    return objective_values

                weighted_score = self.evaluator.calculate_weighted_sum(metrics)
                contribution_analysis = self.evaluator.analyze_metric_contributions(
                    metrics
                )

                try:
                    trial.set_user_attr("weighted_score", float(weighted_score))
                    trial.set_user_attr("params", dict(trial.params))
                except Exception as e:
                    logging.debug(
                        f"Trial {trial_id}: failed to set weighted_score user attrs: {e}"
                    )

                logging.info(
                    f"Trial {trial_id}: Completed successfully. "
                    f"Weighted score: {weighted_score:.2f}"
                )

                self._log_contributions(trial_id, contribution_analysis)
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

            finally:
                # Trial-local evaluation artefacts (sam/bam/vcf/busco) are large;
                # the assembly itself stays in work/hifiasm for .bin reuse.
                try:
                    evaluator.cleanup_intermediate_files(trial_id)
                except Exception:
                    pass

        return objective

    # ---------------------------------------------------------------- logging
    def _log_contributions(self, trial_id, contribution_analysis):
        contribs = contribution_analysis["contributions"]
        pos_sum = float(contribution_analysis.get("positive_sum", 0.0))
        neg_sum = float(contribution_analysis.get("negative_sum", 0.0))

        maximize_metrics, minimize_metrics, unknown_metrics = [], [], []
        for metric_name, metric_data in contribs.items():
            direction = self.directions_map.get(metric_name, "unknown")
            if direction == "maximize":
                maximize_metrics.append((metric_name, metric_data))
            elif direction == "minimize":
                minimize_metrics.append((metric_name, metric_data))
            else:
                unknown_metrics.append((metric_name, metric_data))

        def _log_block(title, items):
            logging.info(title)
            for metric_name, data in items:
                contrib = float(data["contribution"])
                if contrib >= 0 and pos_sum > 0:
                    share, share_label = 100.0 * contrib / pos_sum, "reward_share"
                elif contrib < 0 and neg_sum > 0:
                    share, share_label = 100.0 * abs(contrib) / neg_sum, "penalty_share"
                else:
                    share, share_label = 0.0, "share"

                logging.info(
                    f"  {metric_name:25s} | value: {data['log_value']:12.2f} | "
                    f"weight: {data['weight']:6.2f} | "
                    f"contribution: {contrib:9.4f} | "
                    f"{share_label}: {share:6.2f}%"
                )

        logging.info(f"\nTrial {trial_id} Metric Contributions (direction-aware):")
        _log_block("Maximize metrics (higher is better):", maximize_metrics)
        _log_block("Minimize metrics (lower is better):", minimize_metrics)
        if unknown_metrics:
            _log_block(
                "Unknown-direction metrics (check optim_directions.json):",
                unknown_metrics,
            )
        logging.info(
            f"  {'TOTAL':25s} | Positive sum: {pos_sum:.4f} | "
            f"Negative sum: {neg_sum:.4f}\n"
        )

    def _determine_failure_stage(self, error, gfa_file):
        if "hifiasm" in str(error).lower():
            return "hifiasm assembly"
        elif not Path(gfa_file).exists():
            return "assembly output generation"
        else:
            return "assembly evaluation"