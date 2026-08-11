import json
import subprocess
import logging
from pathlib import Path

import optuna

from utils.hifiasm_command import build_hifiasm_command, collect_hifiasm_outputs
from utils.subprocess_logger import SubprocessLogger, TIMEOUT_EXIT_CODE
from utils.assembly_eval import AssemblyEvaluator, MetricStageFailure
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
        stage_walltimes=None,
        kmer_eval=True,
        yak_k=31,
        yak_bloom_bits=37,
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
        self.stage_walltimes = stage_walltimes or {}
        self.kmer_eval = kmer_eval
        self.yak_k = yak_k
        self.yak_bloom_bits = yak_bloom_bits

        self.subprocess_logger = SubprocessLogger(logs_dir=paths.logs_dir)

        self.is_multi_objective = is_multi_objective
        #: objectives whose stage died mid-study; warned about once each
        self._reported_missing_objectives = set()
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
        collect_hifiasm_outputs(
            self.paths.hifiasm_prefix,
            self.paths.default_assembly_dir,
            "default_assembly",
        )

    # ------------------------------------------------------------ evaluators
    def make_evaluator(self, trial_id=None):
        """
        Build an :class:`AssemblyEvaluator` for one trial.

        A fresh instance per trial is deliberate: it re-reads
        ``metric_stage_state.json`` from disk, which is how a stage retired
        during an earlier trial stays retired for the next one.
        """
        return AssemblyEvaluator(
            known_genome_size=self.evaluator.known_genome_size,
            input_reads=self.evaluator.input_reads,
            paths=self.paths,
            threads=self.threads,
            trial_id=trial_id,
            download_path=self.download_path,
            ont=self.ont,
            kmer_eval=self.kmer_eval,
            include_busco=self.include_busco,
            yak_k=self.yak_k,
            yak_bloom_bits=self.yak_bloom_bits,
            stage_walltimes=self.stage_walltimes,
        )

    def _objective_values(self, metrics, evaluator, trial_id):
        """
        Build the fixed-length objective vector for multi-objective mode.

        Optuna fixes the number of objectives when the study is created, so a
        metric whose stage has since been retired cannot simply be dropped the
        way it is in single-objective mode. It is reported as a constant 0.0
        instead, which makes that axis non-discriminating between later
        trials -- but does leave earlier trials on a different footing, so it
        is warned about loudly.
        """
        values = []
        newly_missing = []
        for key in self.objectives:
            if key in metrics:
                values.append(float(metrics[key]))
                continue
            values.append(0.0)
            if key not in self._reported_missing_objectives:
                self._reported_missing_objectives.add(key)
                newly_missing.append(key)

        if newly_missing:
            stages = sorted(
                {
                    stage.label
                    for stage in evaluator.STAGES
                    for key in newly_missing
                    if key in stage.metrics
                }
            )
            logging.error(
                f"Objective(s) {', '.join(newly_missing)} are no longer being "
                f"produced ({'; '.join(stages) or 'stage unknown'} failed). "
                "Optuna fixes the number of objectives when the study is "
                f"created, so from trial {trial_id} on they are reported as a "
                "constant 0.0 rather than dropped. The Pareto front now mixes "
                "two metric regimes: fix the underlying tool and re-run with "
                "--force-rerun for a clean front, or switch to single-objective "
                "mode, which drops retired metrics cleanly."
            )
        return tuple(values)

    # -------------------------------------------------------------- objective
    def build_objective(self):
        """Build and return the objective function for Optuna."""

        def objective(trial):
            trial_id = trial.number

            evaluator = self.make_evaluator(trial_id)

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
                # default_only stops build_hifiasm_command before the tunable
                # block. The result is identical to relying on every tunable
                # being None, but it says so rather than implying it.
                command = (
                    build_hifiasm_command(default_only=True, **params)
                    + f" {self.input_reads}"
                )
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
                extra_fastas = evaluator.convert_extra_haplotypes(
                    prefix, extra_suffixes
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
                    # Which metrics this trial was actually scored on. Trials
                    # from different regimes are not score-comparable, and
                    # find_best_trial uses this to avoid comparing them.
                    trial.set_user_attr(
                        "metric_regime", evaluator.metric_regime(metrics)
                    )
                    failed = [
                        n for n, ok in evaluator.stage_outcomes.items() if not ok
                    ]
                    if failed:
                        trial.set_user_attr("failed_stages", failed)
                except Exception as e:
                    logging.debug(
                        f"Trial {trial_id}: failed to set metric user attrs: {e}"
                    )

                if trial_id == 0:
                    self._archive_default_assembly()

                if self.is_multi_objective:
                    objective_values = self._objective_values(
                        metrics, evaluator, trial_id
                    )

                    try:
                        signs = [
                            1
                            if self.directions_map.get(obj, "maximize") == "maximize"
                            else -1
                            for obj in self.objectives
                        ]
                        agg = sum(
                            sign * value
                            for sign, value in zip(signs, objective_values)
                        ) / max(1, len(objective_values))
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

                # The trial-local evaluator, not self.evaluator: only it has
                # re-read metric_stage_state.json and therefore knows which
                # metrics are still in the weighted sum.
                weighted_score = evaluator.calculate_weighted_sum(metrics)
                contribution_analysis = evaluator.analyze_metric_contributions(
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

                self._log_contributions(trial_id, contribution_analysis, evaluator)
                return weighted_score

            except MetricStageFailure as e:
                # A metric that worked on the baseline failed here. The trial is
                # thrown away rather than scored on a smaller metric set: mixing
                # scoring regimes inside one study makes the values meaningless.
                # hifimizer's metric_skip_callback counts these and stops the
                # run once --max-metric-skips is reached.
                try:
                    trial.set_user_attr("metric_skip", True)
                    trial.set_user_attr("metric_skip_stage", e.stage_name)
                    trial.set_user_attr("params", dict(trial.params))
                except Exception:
                    pass
                logging.error(
                    f"Trial {trial_id}: DISCARDED - {e.reason}. The metric "
                    "stays enabled for later trials; this trial's result does "
                    "not enter the study."
                )
                raise optuna.exceptions.TrialPruned(
                    f"Trial {trial_id} discarded: {e}"
                )

            except (
                TimeoutError,
                FileNotFoundError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
            ) as e:
                stage = self._determine_failure_stage(e, gfa_file)
                logging.error(f"Trial {trial_id}: Failed at {stage} - {str(e)}")
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
    def _log_contributions(self, trial_id, contribution_analysis, evaluator=None):
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

        def _fmt_raw(data):
            """Raw (back-transformed) value plus unit, for human eyes."""
            raw = float(data.get("raw_value", data["log_value"]))
            unit = data.get("unit", "")
            if abs(raw) >= 1000:
                text = f"{raw:,.0f}"
            elif abs(raw) >= 10:
                text = f"{raw:.1f}"
            else:
                text = f"{raw:.3f}"
            return f"{text} {unit}".strip()

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
                    f"  {metric_name:25s} | value: {_fmt_raw(data):>16s} | "
                    f"log: {data['log_value']:8.3f} | "
                    f"weight: {data['weight']:6.2f} | "
                    f"contribution: {contrib:9.4f} | "
                    f"{share_label}: {share:6.2f}%"
                )

        # `value` is the real-world number; `log` is what the weighted sum
        # actually consumes (log(v+1), except for the metrics listed in
        # AssemblyEvaluator.RAW_METRICS, where the two columns coincide).
        logging.info(f"\nTrial {trial_id} Metric Contributions (direction-aware):")
        if evaluator is not None:
            skipped = [
                name for name, ok in evaluator.stage_outcomes.items() if not ok
            ]
            if skipped:
                logging.info(
                    "  (scored without: "
                    + ", ".join(
                        evaluator.STAGES_BY_NAME[n].label for n in skipped
                    )
                    + ")"
                )
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

    @staticmethod
    def _determine_failure_stage(error, gfa_file):
        """
        Label a *trial-fatal* failure.

        Individual metric tools no longer reach this path: they are absorbed
        by ``AssemblyEvaluator._run_stage``. What is left is hifiasm itself,
        a missing GFA, and the case where every metric stage failed at once.
        """
        if "hifiasm" in str(error).lower():
            return "hifiasm assembly"
        if not Path(gfa_file).exists():
            return "assembly output generation"
        return "assembly evaluation"