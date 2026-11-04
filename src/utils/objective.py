import json
import subprocess
import logging
from pathlib import Path

import optuna

from utils.hifiasm_command import (
    build_hifiasm_command,
    parse_hifiasm_extra,
    collect_hifiasm_outputs,
    delete_hifiasm_bins,
    dual_scaf_supported,
)
from utils.subprocess_logger import SubprocessLogger, TIMEOUT_EXIT_CODE
from utils.assembly_eval import AssemblyEvaluator, MetricStageFailure, BITS_CAP
from utils.paths import RunPaths


#: Objectives used when ``--multi-objective`` is set. Gene space is represented
#: by missing BUSCOs: single_copy is no longer scored (see RETIRED_WEIGHT_KEYS).
PARETO_OBJECTIVES = ("n50", "missing", "qv", "length_diff")

#: Discrete values sampled in --sensitive mode, each including hifiasm 0.25.0's
#: default (k 51, w 51, max-od-final 0.03). Smaller k / w and a larger maximum
#: overlap difference are more lenient. hifiasm rejects k >= 64.
SENSITIVE_K_CHOICES = (21, 31, 51, 61)
SENSITIVE_W_CHOICES = (21, 31, 51, 61, 71)
SENSITIVE_MAX_OD_FINAL_CHOICES = (0.03, 0.05, 0.07)
#: --max-od-ec defaults to 0.04, but --ont raises it to 0.07, so the lenient
#: values are counted from whichever default the run starts from.
SENSITIVE_MAX_OD_EC_CHOICES = {False: (0.04, 0.06, 0.08), True: (0.07, 0.09, 0.11)}

#: Added to the Pareto front for Hi-C runs. Without it the front trades
#: contiguity against completeness while the phasing parameters
#: (--s-base, --f-perturb, --l-msjoin) move nothing that is being optimised.
HIC_PARETO_OBJECTIVE = "trans_hap_rate"


def _load_directions_map():
    directions_file = Path(__file__).resolve().parent.parent / "optim_directions.json"
    if directions_file.exists():
        with open(directions_file, "r") as fh:
            return json.load(fh) or {}
    return {}


def haplotype_suffixes(hic1, hic2, ul, primary=False):
    """
    Return (primary_suffix, extra_haplotype_suffixes) for the hifiasm outputs.

    The extra suffixes are only used to widen the k-mer completeness estimate
    across both haplotypes; QV and everything else stay on the primary.

    A hifiasm output name is an infix plus a contig-set suffix, and one flag
    picks both. ``get_outfile_name`` (Overlaps.cpp) emits the ``.bp`` infix
    only while ``HA_F_PARTITION`` is set, and ``--primary`` clears that flag
    (CommandLines.cpp: ``flag -= HA_F_PARTITION``). The output dispatch then
    falls through to ``output_contig_graph_primary`` /
    ``output_contig_graph_alternative``, which write one collapsed contig set
    rather than a hap1/hap2 pair. So a primary run produces
    ``<prefix>.p_ctg.gfa`` and ``<prefix>.a_ctg.gfa``.

    Hi-C is checked first here because hifiasm checks it first too: its
    ``.hic`` branch precedes the ``HA_F_PARTITION`` branch, so ``--primary``
    does not change Hi-C output names.
    """
    if hic1 and hic2:
        return "hic.hap1.p_ctg", ["hic.hap2.p_ctg"]
    if primary:
        # a_ctg is the alternate set, not a second haplotype assembly, so it
        # is deliberately not scored as one.
        return "p_ctg", []
    if ul:
        return "bp.hap1.p_ctg", ["bp.hap2.p_ctg"]
    # Plain HiFi: bp.p_ctg is already the primary (haplotype-collapsed) set.
    return "bp.p_ctg", []


def log_weight_groups(evaluator):
    """
    Show what share of the score each metric group carries in this run.

    Weights are set per metric, but metrics inside a group move together, so
    the group total is the vote that matters. Shown once per run because it
    depends on the mode: Hi-C adds the phasing group, a retired stage removes
    one, and per-haplotype runs split contiguity across hap1 and hap2.
    """
    rows = evaluator.weight_groups()
    if not rows:
        return
    logging.info(
        f"\nWhat the score is made of (weights from {evaluator.weights_source})"
    )
    logging.info(f"    {'group':<12}{'weight':>8}{'share':>8}  metrics")
    for group, weight, share, members in rows:
        detail = ", ".join(
            f"{m} {evaluator.weights[m]:g}" for m in members
        )
        logging.info(f"    {group:<12}{weight:>8.2f}{share:>7.0%}  {detail}")
    logging.info(
        "  The share is the most of the score a group can move. Metrics inside "
        "a group move together (contig count and N50 nearly always do), so to "
        "shift the balance between properties, scale a whole group's weights "
        "in weights.json rather than one metric.\n"
    )


def log_metric_report_legend(bits_cap=BITS_CAP):
    """Explain the per-trial metric table. Called once, before optimisation."""
    logging.info("\nHow to read the per-trial metric reports below")
    logging.info(f"  {'value':<8}the real measurement, in its natural units")
    logging.info(
        f"  {'bits':<8}how that compares with the default-parameter assembly "
        "(trial 0), as a"
    )
    logging.info(
        f"  {'':<8}log2 fold change signed so POSITIVE IS ALWAYS BETTER. +1.00 "
        "is twice as"
    )
    logging.info(
        f"  {'':<8}good, -1.00 twice as bad, 0.00 no change -- the same for "
        "every metric,"
    )
    logging.info(f"  {'':<8}so no metric needs rescaling before it is scored.")
    if bits_cap and bits_cap > 0:
        logging.info(
            f"  {'capped':<8}those bits soft-capped at +/-{bits_cap:.1f} "
            f"(cap x tanh(bits / cap)), so"
        )
        logging.info(
            f"  {'':<8}one runaway metric cannot outvote all the others. This "
            "is what the"
        )
        logging.info(f"  {'':<8}weight multiplies.")
    else:
        logging.info(f"  {'capped':<8}identical to bits: --bits-cap is 0.")
    logging.info(
        f"  {'':<8}Score = weighted MEAN of the capped bits, weighted by the "
        "importances in"
    )
    logging.info(
        f"  {'':<8}weights.json, so +0.10 is a tenth of a doubling on average "
        "in every mode."
    )
    logging.info(
        f"  {'':<8}Trial 0 scores exactly 0, so a positive score beat plain "
        "hifiasm.\n"
    )


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
        num_hic_reads=1_000_000,
        hic_min_mapq=20,
        hic_phasing=True,
        prune_bits=1.0,
        prune_min_metrics=2,
        hifiasm_extra=None,
        bits_cap=BITS_CAP,
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
        self.num_hic_reads = num_hic_reads
        self.hic_min_mapq = hic_min_mapq
        self.hic_phasing = hic_phasing
        self.prune_bits = prune_bits
        self.prune_min_metrics = prune_min_metrics
        self.bits_cap = bits_cap
        # Anything the user pinned is fixed for every run and never sampled;
        # anything hifimizer does not recognise is passed straight through.
        self.pinned_params, self.hifiasm_passthrough = parse_hifiasm_extra(
            hifiasm_extra
        )

        self.subprocess_logger = SubprocessLogger(logs_dir=paths.logs_dir)

        self.is_multi_objective = is_multi_objective
        #: objectives whose stage died mid-study; warned about once each
        self._reported_missing_objectives = set()
        if objectives:
            self.objectives = objectives
        elif is_multi_objective:
            self.objectives = list(PARETO_OBJECTIVES)
            if self.hic1 and self.hic2 and self.hic_phasing:
                self.objectives.append(HIC_PARETO_OBJECTIVE)
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
            # The completeness settings come from the setup evaluator, not from
            # this builder's own arguments: that evaluator is the one built
            # from the command line, and the one that has already resolved
            # (and if needed fetched) the lineage. Leaving any of them out made
            # every trial run compleasm against the empty default library --
            # downloading the lineage again despite --compleasm-download-path --
            # and silently ignored --no-compleasm and --compleasm-bin.
            download_path=self.evaluator.download_path,
            compleasm_bin=self.evaluator.compleasm_bin,
            compleasm_download_path=self.evaluator.compleasm_download_path,
            use_compleasm=self.evaluator.use_compleasm,
            subset_seed=self.evaluator.subset_seed,
            ont=self.ont,
            kmer_eval=self.kmer_eval,
            include_busco=self.include_busco,
            yak_k=self.yak_k,
            yak_bloom_bits=self.yak_bloom_bits,
            stage_walltimes=self.stage_walltimes,
            hic1=self.hic1,
            hic2=self.hic2,
            num_hic_reads=self.num_hic_reads,
            hic_min_mapq=self.hic_min_mapq,
            hic_phasing=self.hic_phasing,
            # --primary collapses the output to a single contig set, so the
            # per-haplotype metric names would be asked for and never appear.
            n_haplotypes=2
            if ((self.hic1 and self.hic2) or (self.ul and not self.primary))
            else 1,
            bits_cap=self.bits_cap,
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

    # ---------------------------------------------------------------- pruning
    #: Which way each gated metric is allowed to move. "up" means a larger raw
    #: value is worse (more contigs, a bigger length deviation); "down" means a
    #: smaller one is worse (N50).
    GATED_METRICS = {
        "num_contigs": "up",
        "length_diff": "up",
        "n50": "down",
    }

    def _make_gate(self, trial, evaluator):
        """
        Build the post-gfastats check that prunes a collapsed assembly early.

        Only contiguity is gated. It comes from the cheapest stage, it is the
        first thing to fall apart when hifiasm's graph-cleaning parameters go
        wrong, and it is the only metric available before the expensive stages
        run. A trial that fails it is pruned rather than scored, so the study
        moves on without paying for yak and completeness on an assembly already
        known to be bad.

        The threshold is in bits against the *default-parameter baseline* --
        the same per-metric bits the score is built from, so ``--prune-bits
        1`` prunes exactly what would score -1 or worse on two gated metrics --
        and the bar is identical for the whole study. Measured
        against the running best it ratcheted: every improvement tightened the
        gate on every later trial, and a study that found one good assembly
        early then pruned nearly everything after it.

        A single metric is never enough. Contig count, N50 and length
        difference are strongly coupled, so one drifting past the threshold is
        ordinary variation; ``prune_min_metrics`` going at once is a collapse.

        Compared on uncapped bits: the cap bounds a metric's pull on the score,
        it is not meant to hide the size of a collapse from the gate.

        Returns None (no gating) for the baseline, which has nothing to be
        compared against yet.
        """
        if self.prune_bits <= 0 or self.prune_min_metrics <= 0:
            return None
        if not evaluator.has_baseline:
            return None

        def gate(metrics):
            breaches = []
            for name, value in sorted(metrics.items()):
                if name.split("_hap")[0] not in self.GATED_METRICS:
                    continue
                bits = evaluator.fold_change(name, float(value))
                if bits is None:
                    continue
                # Already signed so that negative means worse than the
                # baseline, whichever way the metric runs.
                if bits < -self.prune_bits:
                    breaches.append((name, bits))

            if len(breaches) < self.prune_min_metrics:
                return

            detail = ", ".join(f"{n} {v:+.2f} bits" for n, v in breaches)
            trial.set_user_attr("gfastats_gate", [n for n, _ in breaches])
            trial.set_user_attr(
                "gfastats_gate_bits", {n: float(v) for n, v in breaches}
            )
            logging.info(
                f"Trial {trial.number}: PRUNED - {len(breaches)} of the gated "
                f"metrics are more than {self.prune_bits:.1f} bits worse "
                f"than the default-parameter baseline ({detail}). Skipping the "
                "remaining metrics and moving to the next trial."
            )
            raise optuna.exceptions.TrialPruned(
                f"{len(breaches)} gated metrics beyond -{self.prune_bits:.1f} bits"
            )

        return gate

    # -------------------------------------------------------------- objective
    def build_objective(self):
        """
        Build and return the objective function for Optuna.

        Trial 0 runs hifiasm with default parameters and becomes the reference
        every later trial is scored against, so it scores exactly 0 and the
        sign of any other trial's score says whether it beat plain hifiasm.

        Every trial after it is scored in bits against that reference from the
        start. There is no burn-in phase: bits are scale-free, so there is no
        per-metric spread to measure before scoring can begin, and every trial
        -- including TPE's random startup trials -- is an ordinary, gated
        trial of the one study.
        """

        def objective(trial):
            trial_id = trial.number

            evaluator = self.make_evaluator(trial_id)

            prefix = self.paths.hifiasm_prefix
            suffix, extra_suffixes = haplotype_suffixes(
                self.hic1, self.hic2, self.ul, self.primary
            )

            gfa_file = prefix.parent / f"{prefix.name}.{suffix}.gfa"
            fasta_file = evaluator.trial_dir / f"{prefix.name}.{suffix}.fasta"

            # Trial 0 is the baseline: hifiasm with default parameters. It stays
            # in the study so that it shows up in the optimisation history as a
            # reference point; the parameter-importance plots filter it out
            # separately (see hifimizer.plot_param_importances).
            # Trial 0 rather than "no baseline yet": the evaluator derives its
            # failure policy from the trial number (0 retires a failing stage
            # for the study, later trials are discarded instead), so the two
            # must agree on which trial is the baseline.
            is_baseline = trial_id == 0
            if is_baseline:
                # Also recorded as an attribute, which is what the
                # final-assembly step checks to know whether the winner is it.
                trial.set_user_attr("baseline", True)
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
                    build_hifiasm_command(
                        default_only=True,
                        extra_args=self.hifiasm_passthrough,
                        dual_scaf=bool(self.pinned_params.get("dual_scaf", False)),
                        **params,
                    )
                    + f" {self.input_reads}"
                )
            else:
                # A pinned parameter is returned as-is and never handed to
                # trial.suggest_*, which is what keeps it out of the search
                # space: Optuna only models parameters it was asked to sample.
                def pick(name, sampler):
                    if name in self.pinned_params:
                        return self.pinned_params[name]
                    return sampler()

                x = pick("x", lambda: trial.suggest_float("x", 0.59, 0.99, step=0.01))
                y = pick("y", lambda: trial.suggest_float("y", 0.01, 0.41, step=0.01))
                s = pick("s", lambda: trial.suggest_float("s", 0.55, 1, step=0.01))
                n = pick("n", lambda: trial.suggest_int("n", 0, 10))
                m = pick(
                    "m", lambda: trial.suggest_int("m", 500_000, 20_000_000, log=True)
                )
                p = pick("p", lambda: trial.suggest_int("p", 1, 10_000, log=True))
                u = pick("u", lambda: trial.suggest_categorical("u", [0, 1]))

                hic_params = {}
                ont_params = {}

                if self.hic1 and self.hic2:
                    hic_params.update(
                        {
                            "s_base": pick(
                                "s_base",
                                lambda: trial.suggest_float("s_base", 0, 1, step=0.05),
                            ),
                            "f_perturb": pick(
                                "f_perturb",
                                lambda: trial.suggest_float(
                                    "f_perturb", 0, 1, step=0.05
                                ),
                            ),
                            "l_msjoin": pick(
                                "l_msjoin",
                                lambda: trial.suggest_int(
                                    "l_msjoin", 1, 10_000_000, log=True
                                ),
                            ),
                        }
                    )

                # Scaffolding one haplotype through the other only changes the
                # scored assembly when hap1/hap2 are what is scored: Hi-C and
                # ultra-long runs, never --primary.
                if dual_scaf_supported(self.hic1, self.hic2, self.ul, self.primary):
                    hic_params["dual_scaf"] = pick(
                        "dual_scaf",
                        lambda: trial.suggest_categorical("dual_scaf", [False, True]),
                    )

                if self.ul:
                    ont_params.update(
                        {
                            "path_max": pick(
                                "path_max",
                                lambda: trial.suggest_float(
                                    "path_max", 0.0, 1.0, step=0.05
                                ),
                            ),
                            "path_min": pick(
                                "path_min",
                                lambda: trial.suggest_float(
                                    "path_min", 0.0, 1.0, step=0.05
                                ),
                            ),
                        }
                    )

                sensitive_params = {}
                if self.sensitive:
                    sensitive_params.update(
                        {
                            "D": pick(
                                "D", lambda: trial.suggest_int("D", 3, 20, step=1)
                            ),
                            "N": pick(
                                "N", lambda: trial.suggest_int("N", 50, 400, step=10)
                            ),
                            "max_kocc": pick(
                                "max_kocc",
                                lambda: trial.suggest_int(
                                    "max_kocc", 1000, 5000, step=100
                                ),
                            ),
                            "k": pick(
                                "k",
                                lambda: trial.suggest_categorical(
                                    "k", SENSITIVE_K_CHOICES
                                ),
                            ),
                            "w": pick(
                                "w",
                                lambda: trial.suggest_categorical(
                                    "w", SENSITIVE_W_CHOICES
                                ),
                            ),
                            "max_od_ec": pick(
                                "max_od_ec",
                                lambda: trial.suggest_categorical(
                                    "max_od_ec",
                                    SENSITIVE_MAX_OD_EC_CHOICES[bool(self.ont)],
                                ),
                            ),
                            "max_od_final": pick(
                                "max_od_final",
                                lambda: trial.suggest_categorical(
                                    "max_od_final", SENSITIVE_MAX_OD_FINAL_CHOICES
                                ),
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
                    extra_args=self.hifiasm_passthrough,
                )
                command += f" {self.input_reads}"

            try:
                if self.sensitive:
                    # -D, -N and --max-kocc act on the stages hifiasm skips when
                    # it finds cached .bin files, so every --sensitive run has
                    # to start from the raw reads.
                    removed = delete_hifiasm_bins(prefix)
                    if removed:
                        logging.info(
                            f"Trial {trial_id}: --sensitive - deleted "
                            f"{len(removed)} cached hifiasm .bin file(s) so the "
                            "sampled -D/-N/--max-kocc take effect."
                        )
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
                    # hifiasm exited 0 but the name it wrote does not match
                    # the one haplotype_suffixes() predicted from this run's
                    # flags -- which is exactly the failure mode of loading
                    # .bin files written under a different configuration
                    # (verify_hifiasm_bin_compatibility should catch that
                    # before any trial runs, but this makes the failure
                    # self-diagnosing rather than a bare "not found" if it
                    # ever happens anyway). List what is actually at the
                    # prefix instead of leaving that to be reconstructed from
                    # the hifiasm log afterwards.
                    found = sorted(
                        f.name for f in prefix.parent.glob(f"{prefix.name}*")
                    )
                    listing = ", ".join(found) if found else "(nothing)"
                    raise FileNotFoundError(
                        f"GFA file not found: {gfa_file}\n"
                        f"    predicted suffix : {suffix}\n"
                        f"    hic={bool(self.hic1 and self.hic2)} "
                        f"ul={bool(self.ul)} primary={self.primary}\n"
                        f"    actually at {prefix.parent}: {listing}\n"
                        "    hifiasm reported success (exit 0), so this is a "
                        "naming mismatch, not an assembly failure -- most "
                        "often stale .bin files from a run with different "
                        "flags. Check work/hifiasm/.fingerprint.json against "
                        "the flags above, or re-run with --force-rerun."
                    )

                # Sibling haplotype outputs. The FASTAs feed the diploid
                # alignment, QV and k-mer completeness; the GFAs let gfastats
                # measure hap2 as well as hap1.
                extra_fastas = evaluator.convert_extra_haplotypes(
                    prefix, extra_suffixes
                )
                extra_gfas = [
                    prefix.parent / f"{prefix.name}.{s}.gfa"
                    for s in extra_suffixes
                ]

                logging.info(f"Trial {trial_id}: Evaluating assembly")
                metrics = evaluator.evaluate_assembly(
                    gfa_file=gfa_file,
                    fasta_file=fasta_file,
                    include_busco=self.include_busco,
                    busco_lineage=self.busco_lineage,
                    extra_fasta_files=extra_fastas,
                    extra_gfa_files=extra_gfas,
                    gate=None if is_baseline else self._make_gate(trial, evaluator),
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

                if is_baseline:
                    # Adopted before scoring, so trial 0 is a fold change
                    # against itself and lands at exactly 0.
                    evaluator.save_baseline_metrics(metrics)
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
                # Lets the --max-consecutive-pruned stop say what went wrong
                # instead of blaming the gfastats gate.
                try:
                    trial.set_user_attr("failure", f"{stage}: {e}"[:500])
                except Exception:
                    pass
                raise optuna.exceptions.TrialPruned(f"Trial pruned at {stage}: {e}")

            finally:
                # Trial-local evaluation artefacts (sam/bam/vcf/compleasm/busco)
                # are large;
                # the assembly itself stays in work/hifiasm for .bin reuse.
                try:
                    evaluator.cleanup_intermediate_files(trial_id)
                except Exception:
                    pass

        return objective

    # ---------------------------------------------------------------- logging
    def _log_contributions(self, trial_id, contribution_analysis, evaluator=None):
        """
        Per-trial metric table: the measurement, and how it compares with the
        default-parameter baseline.

        There is no separate reward/penalty split. Fold changes are signed so
        that positive is better for every metric, so there are not two pools to
        divide the score between -- a metric either improved on trial 0 or it
        did not.
        """
        contribs = contribution_analysis["contributions"]

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

        def _fmt(data, key):
            value = data.get(key)
            return "     -" if value is None else f"{value:+.2f}"

        def _log_block(title, items):
            if not items:
                return
            logging.info(f"  {title}")
            for metric_name, data in items:
                logging.info(
                    f"    {metric_name:<26}{_fmt_raw(data):>18}"
                    f"{_fmt(data, 'fc'):>10}{_fmt(data, 'bits'):>9}"
                )

        logging.info(f"\nTrial {trial_id} metric report")

        if evaluator is not None:
            skipped = [
                name for name, ok in evaluator.stage_outcomes.items() if not ok
            ]
            if skipped:
                logging.info(
                    "  Scored without: "
                    + ", ".join(
                        evaluator.STAGES_BY_NAME[n].label for n in skipped
                    )
                )

        logging.info(
            f"\n    {'metric':<26}{'value':>18}{'bits':>10}{'capped':>9}"
        )
        _log_block("Maximize (higher is better)", maximize_metrics)
        _log_block("Minimize (lower is better)", minimize_metrics)
        if unknown_metrics:
            _log_block(
                "Unknown direction (add these to optim_directions.json)",
                unknown_metrics,
            )

        groups = contribution_analysis.get("groups") or {}
        if groups:
            logging.info(
                "\n  By group: "
                + " | ".join(f"{g} {v:+.3f}" for g, v in groups.items())
            )

        score = float(contribution_analysis["total_score"])
        verdict = (
            "better than the default assembly"
            if score > 0
            else ("the default assembly itself" if score == 0 else "worse than default")
        )
        cap = evaluator.bits_cap if evaluator is not None else self.bits_cap
        unit = (
            f"bits soft-capped at +/-{cap:.1f}"
            if cap and cap > 0
            else "bits, uncapped"
        )
        logging.info(
            f"\n  Score {score:+.4f}  (importance-weighted mean of {unit} -- "
            f"{verdict})\n"
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