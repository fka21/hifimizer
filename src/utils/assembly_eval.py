import os
import re
import gc
import json
import ctypes
import shutil
import subprocess
import logging
import gzip
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
from pathlib import Path

from utils.subprocess_logger import SubprocessLogger, TIMEOUT_EXIT_CODE
from utils.paths import RunPaths


#: log2 fold changes are clipped to this many doublings. A collapsed assembly
#: can be a 500-fold regression on contig count, which at weight 1 would swamp
#: every other metric combined; the gate should reject those anyway, and past
#: ~32x a trial is bad by any reading.
FC_CLIP = 5.0

#: Added to count-like quantities before the ratio. Several are legitimately
#: zero on a good assembly (missing BUSCOs, multi-copy, SVs), and x/0 is not a
#: fold change.
FC_PSEUDOCOUNT = 1.0

#: Floor for the "deficit" quantities (k-mer completeness, trans-hap rate),
#: which reach exactly zero on a perfect assembly.
FC_DEFICIT_FLOOR = 0.01

#: Standardised scores are clipped to this many scale units. A metric whose
#: burn-in movement was tiny can otherwise turn an ordinary trial into a
#: 50-unit outlier purely by having a small denominator.
Z_CLIP = 5.0

#: Below this a metric did not move at all during the burn-in: it cannot tell
#: one parameter set from another, so it is dropped rather than divided by.
SCALE_MIN = 1e-6

#: Burn-in trials a metric needs before it gets a scale.
SCALE_MIN_OBSERVATIONS = 3


class CompletenessFailedError(RuntimeError):
    """
    Raised when every gene-space completeness backend fails or times out.

    The chain is compleasm, then BUSCO/metaeuk, then BUSCO/augustus; this is
    only raised once all of them are exhausted.
    """


#: Retained under its old name: ``metric_stage_state.json`` files and any
#: downstream ``except`` clauses written against the BUSCO-only implementation
#: keep working.
BuscoFailedError = CompletenessFailedError


class MetricStageFailure(RuntimeError):
    """
    A metric stage failed in a trial whose results cannot be trusted without it.

    Raised only under the ``strict`` failure policy -- i.e. for trials after
    the baseline, where the tool has already been shown to work. The trial is
    discarded rather than scored on a smaller metric set, because a score
    computed from a different set of metrics is not comparable with the rest
    of the study.
    """

    def __init__(self, stage_name, label, error, reason=None):
        self.stage_name = stage_name
        self.label = label
        self.error = error
        #: why the trial was invalidated, phrased for a log line
        self.reason = reason or f"{label} failed"
        super().__init__(f"{self.reason}: {error}")


@dataclass(frozen=True)
class MetricStage:
    """
    One external tool invocation that contributes metrics to a trial.

    Stages are independently fallible: if one fails, the trial is still scored
    on whatever the others produced (see
    :meth:`AssemblyEvaluator.evaluate_assembly`). Which stage produced which
    metric has to be declared here so that a failed stage's metrics can be
    removed from the weighted sum -- leaving them at their 0.0 default would
    silently *reward* the failure for every negatively-weighted metric.
    """

    name: str
    label: str
    #: metric keys this stage contributes; may be empty for a pure prerequisite
    metrics: Tuple[str, ...] = ()
    #: stages that must have succeeded before this one can run
    requires: Tuple[str, ...] = ()
    #: name of the CLI walltime option that bounds it (for error messages)
    walltime_flag: str = ""
    #: an essential stage cannot be dropped -- without it there is nothing
    #: meaningful left to optimise, so its failure always invalidates the trial
    essential: bool = False


class AssemblyEvaluator:
    """
    AssemblyEvaluator provides a unified interface to evaluate genome assemblies.

    It integrates:
    - Assembly statistics with `gfastats`
    - Gene-space completeness with `compleasm`, falling back to `BUSCO`
    - k-mer completeness and consensus QV with `yak`
    - Hi-C phasing consistency with `minimap2` + `samtools` (Hi-C runs only)

    All intermediate artefacts are written beneath ``paths.work_dir``; nothing
    is written relative to the current working directory.

    Metric conventions
    ------------------
    Every metric except those listed in :attr:`RAW_METRICS` is stored
    log-transformed as ``log(value + 1)``.  ``qv`` (already a Phred-scaled,
    i.e. logarithmic, quantity) and the bounded percentages
    (``kmer_completeness``, ``trans_hap_rate``) are stored raw: log-transforming
    them would compress their variance to the point of invisibility.

    Use :meth:`raw_value` to undo the transform for display; the optimiser
    always consumes the stored (log) values.
    """

    #: metrics that are NOT log-transformed
    RAW_METRICS = frozenset(
        {
            # QV is already logarithmic by construction (Phred), so logging it
            # again would be a double transform.
            "qv",
            "kmer_completeness",
            # Hi-C phasing: a percentage, like kmer_completeness.
            "trans_hap_rate",
            # Per-haplotype assembly lengths, reported in Mb for the log.
            "hap1_length_mb",
            "hap2_length_mb",
            "qv_hap1",
            "qv_hap2",
            "kmer_completeness_hap1",
            "kmer_completeness_hap2",
        }
    )

    #: units used when echoing raw values into the log
    METRIC_UNITS = {
        "length_diff": "Mb",
        "n50": "bp",
        "qv": "Phred",
        "kmer_completeness": "%",
        "trans_hap_rate": "%",
        "hic_pairs_informative": "pairs",
        "hap1_length_mb": "Mb",
        "hap2_length_mb": "Mb",
        "qv_hap1": "Phred",
        "qv_hap2": "Phred",
        "kmer_completeness_hap1": "%",
        "kmer_completeness_hap2": "%",
        "num_contigs_hap1": "",
        "num_contigs_hap2": "",
        "n50_hap1": "bp",
        "n50_hap2": "bp",
        "length_diff_hap1": "Mb",
        "length_diff_hap2": "Mb",
    }

    #: Metrics that are scored per haplotype when a second haplotype exists.
    #: Splitting them is what stops a good hap1 hiding a poor hap2: a summary
    #: across haplotypes lets one make up for the other, whereas two separate
    #: penalties cannot cancel.
    HAPLOTYPE_SPLIT_METRICS = ("num_contigs", "n50", "length_diff")

    # ------------------------------------------------------------ transforms
    @classmethod
    def raw_value(cls, name, value):
        """
        Undo the ``log(v + 1)`` storage transform for *display purposes only*.

        Metrics in :attr:`RAW_METRICS` are returned unchanged. Everything the
        optimiser sees stays log-scaled; only the log lines show raw numbers.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if name in cls.RAW_METRICS:
            return v
        return max(0.0, float(np.expm1(v)))

    @classmethod
    def raw_metrics(cls, metrics):
        """Whole-dict version of :meth:`raw_value`."""
        return {k: cls.raw_value(k, v) for k, v in metrics.items()}

    # --------------------------------------------------------------- stages
    #: Metric-producing stages, in the order ``evaluate_assembly`` runs them.
    #: ``alignment`` contributes no metrics of its own but gates the two
    #: stages that consume its BAM.
    STAGES = (
        MetricStage(
            name="gfastats",
            label="assembly statistics (gfastats)",
            metrics=(
                "num_contigs",
                "length_diff",
                "n50",
                "num_contigs_hap1",
                "num_contigs_hap2",
                "n50_hap1",
                "n50_hap2",
                "length_diff_hap1",
                "length_diff_hap2",
                "hap1_length_mb",
                "hap2_length_mb",
            ),
            walltime_flag="--gfastats-walltime",
            # Contig count, length and N50 are the backbone of the objective.
            # Continuing without them would be optimising nothing.
            essential=True,
        ),
        MetricStage(
            name="yak",
            label="k-mer QV and completeness (yak)",
            metrics=(
                "qv",
                "kmer_completeness",
                "qv_hap1",
                "qv_hap2",
                "kmer_completeness_hap1",
                "kmer_completeness_hap2",
            ),
            walltime_flag="--yak-walltime",
        ),
        # Named "busco" for continuity: the name is the key in
        # metric_stage_state.json and in the backend cache, and renaming it
        # would silently un-retire the stage in every existing output
        # directory. The tool behind it is compleasm first, BUSCO second.
        MetricStage(
            name="busco",
            label="gene-space completeness (compleasm, falling back to BUSCO)",
            metrics=("single_copy", "multi_copy", "fragmented", "missing"),
            walltime_flag="--busco-walltime",
        ),
        # Only runs when --hic1/--hic2 are given: without Hi-C reads there is
        # nothing to measure and no phasing parameters being tuned.
        MetricStage(
            name="hic_phasing",
            label="Hi-C phasing consistency (minimap2 + samtools)",
            metrics=("trans_hap_rate", "hic_pairs_informative"),
            walltime_flag="--hic-phasing-walltime",
        ),
    )

    STAGES_BY_NAME = {stage.name: stage for stage in STAGES}

    #: default per-stage wall-clock budgets, in hours (CLI overrides these)
    DEFAULT_STAGE_WALLTIMES = {
        "gfastats": 0.5,
        "yak": 2.0,
        "busco": 6.0,
        "hic_phasing": 4.0,
    }

    def __init__(
        self,
        known_genome_size,
        input_reads,
        paths: RunPaths,
        trial_id=None,
        threads=None,
        download_path=None,
        ont=False,
        kmer_eval=True,
        include_busco=True,
        yak_k=31,
        yak_bloom_bits=37,
        stage_walltimes=None,
        failure_policy=None,
        compleasm_bin=None,
        compleasm_download_path=None,
        use_compleasm=True,
        subset_seed=42,
        hic1=None,
        hic2=None,
        num_hic_reads=1_000_000,
        hic_min_mapq=20,
        hic_phasing=True,
        n_haplotypes=1,
    ):
        """
        Args:
            known_genome_size: Haploid genome size in base pairs.
            input_reads: Path to the full input read set.
            paths: :class:`RunPaths` instance describing the run layout.
            trial_id: Optuna trial number (or a string like "best").
            threads: CPU threads handed to the external tools.
            download_path: User-supplied BUSCO dataset directory. When None,
                ``paths.busco_downloads_dir`` is used.
            ont: Input reads are ONT (selects the minimap2 preset).
            kmer_eval: Enable the yak QV / k-mer completeness metrics.
            include_busco: Enable the gene-space completeness metrics
                (compleasm or BUSCO). Named for the ``--no-busco`` flag it
                backs.
            compleasm_bin: Path to the ``compleasm`` executable. When None it
                is discovered; see :meth:`_discover_compleasm`.
            compleasm_download_path: compleasm lineage library. When None,
                ``paths.compleasm_downloads_dir`` is used.
            use_compleasm: Try compleasm before BUSCO. Set False to go
                straight to BUSCO (``--no-compleasm``).
            subset_seed: Seed for the read subsample, so a re-run against the
                same reads produces the same subset.
            stage_walltimes: ``{stage_name: hours}`` overriding
                :attr:`DEFAULT_STAGE_WALLTIMES`. A value of ``None`` or 0
                means "no limit".
            failure_policy: Override for how a stage failure is handled; see
                :attr:`failure_policy`. Normally left as ``None`` so it is
                derived from ``trial_id``.
        """
        self.known_genome_size = known_genome_size
        self.input_reads = Path(input_reads)
        self.paths = paths
        self.trial_id = trial_id
        self.threads = threads
        self.ont = ont
        self.kmer_eval = kmer_eval
        self.include_busco = include_busco
        self.yak_k = yak_k
        self.yak_bloom_bits = yak_bloom_bits
        self._failure_policy = failure_policy
        self.use_compleasm = use_compleasm
        self.subset_seed = subset_seed
        self.hic1 = Path(hic1) if hic1 else None
        self.hic2 = Path(hic2) if hic2 else None
        self.num_hic_reads = num_hic_reads
        self.hic_min_mapq = hic_min_mapq
        self.hic_phasing = hic_phasing
        # 2 for Hi-C / ultra-long runs, which produce hap1 and hap2. Decides
        # whether the split metrics are scored per haplotype or unsuffixed.
        self.n_haplotypes = max(1, int(n_haplotypes))

        self.stage_walltimes = dict(self.DEFAULT_STAGE_WALLTIMES)
        self.stage_walltimes.update(
            {k: v for k, v in (stage_walltimes or {}).items() if k in self.STAGES_BY_NAME}
        )

        # BUSCO datasets: user override, else our own work/ subdirectory.
        self.download_path = (
            Path(download_path).resolve()
            if download_path
            else paths.busco_downloads_dir
        )
        # compleasm keeps its lineages miniprot-indexed and in its own layout,
        # so it gets a separate directory even when the user supplied one for
        # BUSCO. Pointing compleasm at a BUSCO download path makes it
        # re-download rather than fail, which is merely wasteful, but keeping
        # them apart means neither tool ever sees the other's half-written
        # files.
        self.compleasm_download_path = (
            Path(compleasm_download_path).resolve()
            if compleasm_download_path
            else paths.compleasm_downloads_dir
        )
        self._compleasm_bin = compleasm_bin
        self._compleasm_resolved = False

        self.subprocess_logger = SubprocessLogger(logs_dir=paths.logs_dir)
        # `trial_id or 'main'` mislabelled trial 0 -- which is the default-
        # parameter baseline, i.e. the one trial you most want to find in a log.
        label = "main" if trial_id is None else trial_id
        self.logger = logging.getLogger(f"AssemblyEval_{label}")

        self._compile_patterns()
        self.weights_source = "built-in defaults"
        self.weights = self._load_weights()

        # Cache of which BUSCO gene-prediction backend actually works in this
        # environment, so a failing one is only paid for once.
        self.cache_path = paths.busco_backend_cache
        self.backend_cache = self._read_json(self.cache_path, "BUSCO backend cache")

        # Cross-trial record of which metric stages have failed. Re-read from
        # disk on every construction, because Optuna builds a fresh evaluator
        # per trial and this is how a disabled stage propagates forward.
        self.stage_state = self._read_json(
            self.paths.metric_stage_state, "metric stage state"
        )
        # The default-parameter assembly every trial is scored against. Empty
        # until trial 0 has been measured.
        self.baseline_metrics = self._read_json(
            self.paths.baseline_metrics, "baseline metrics"
        )
        # How far each metric typically moves, measured over the burn-in.
        # Empty during the burn-in itself.
        self.metric_scales = self._read_json(
            self.paths.metric_scales, "metric scales"
        )
        #: per-evaluation outcome, ``{stage_name: bool}``; reset by
        #: :meth:`evaluate_assembly`
        self.stage_outcomes = {}

    # ------------------------------------------------------------------ misc
    def _release_memory(self, context=""):
        """
        Collect garbage and hand freed arenas back to the kernel.

        ``gc.collect()`` alone is not enough. CPython returns freed blocks to
        glibc's allocator, which keeps them in per-arena free lists rather than
        ``munmap``-ing them, so RSS stays at its high-water mark for the life
        of the process even when nothing is referenced any more. The optimiser
        then looks like it is leaking when it is only failing to give memory
        back -- and that unreturned memory is exactly what a completeness run
        collides with fifty trials later.

        ``malloc_trim(0)`` is glibc-specific; on anything else the lookup fails
        and only the ``gc.collect()`` takes effect, which is the correct
        degradation.
        """
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            return
        if context:
            self.logger.debug(f"Released allocator arenas after {context}")

    @property
    def trial_dir(self) -> Path:
        return self.paths.trial_dir(self.trial_id)

    # ------------------------------------------------------------ json cache
    def _read_json(self, path, description):
        """Load a small JSON side-file, tolerating absence and corruption."""
        path = Path(path)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f) or {}
            except Exception:
                self.logger.warning(
                    f"Failed to load {description} from {path}; starting fresh"
                )
        return {}

    def _write_json(self, path, payload, description):
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save {description}: {e}")

    def _save_backend_cache(self):
        self._write_json(
            self.cache_path, self.backend_cache, "BUSCO backend cache"
        )

    def reload_stage_state(self):
        """
        Re-read the stage state from disk.

        Each trial gets a fresh evaluator and therefore fresh state, but the
        long-lived evaluator hifimizer keeps for setup and the final assembly
        was constructed before any trial ran. Without this it still believes
        every stage works, and would both re-run a retired tool and compute the
        wrong metric regime when picking the best trial.
        """
        self.stage_state = self._read_json(
            self.paths.metric_stage_state, "metric stage state"
        )
        self.backend_cache = self._read_json(
            self.cache_path, "BUSCO backend cache"
        )
        # Trial 0 writes this after the long-lived evaluator was built, so
        # without picking it up here the final assembly would have no baseline
        # to compare against.
        self.baseline_metrics = self._read_json(
            self.paths.baseline_metrics, "baseline metrics"
        )
        self.metric_scales = self._read_json(
            self.paths.metric_scales, "metric scales"
        )
        return self.stage_state

    def _save_stage_state(self):
        self._write_json(
            self.paths.metric_stage_state, self.stage_state, "metric stage state"
        )

    # -------------------------------------------------------- stage handling
    def stage_walltime_seconds(self, name):
        """Wall-clock budget for a stage, in seconds (``None`` = unlimited)."""
        hours = self.stage_walltimes.get(name)
        return hours * 3600 if hours else None

    def stage_disabled(self, name) -> bool:
        """True if the stage has been switched off after repeated failures."""
        return bool(self.stage_state.get(name, {}).get("disabled", False))

    def stage_off_by_user(self, name) -> bool:
        """
        True if the *user* switched the stage off (``--no-busco`` /
        ``--no-kmer-eval``).

        Kept distinct from :meth:`stage_disabled` so that a deliberate opt-out
        is not reported as a failure on every single trial.
        """
        if name == "yak":
            return not self.kmer_eval
        if name == "busco":
            return not self.include_busco
        if name == "hic_phasing":
            # Also the "no Hi-C data" case: not having run Hi-C is a choice
            # about the experiment, not a stage failing, so it is reported
            # the same way as an explicit --no-hic-phasing.
            return not (self.hic_phasing and self.hic1 and self.hic2)
        return False

    def stage_enabled(self, name) -> bool:
        """
        True if the stage can still contribute metrics.

        Covers four reasons a stage may be off: the user disabled it
        (``--no-busco`` / ``--no-kmer-eval``), it failed often enough to be
        retired, a stage it *depends on* was retired, or it is not a real
        stage name.
        """
        stage = self.STAGES_BY_NAME.get(name)
        if stage is None:
            return False
        if self.stage_off_by_user(name):
            return False
        if self.stage_disabled(name):
            return False
        return all(self.stage_enabled(req) for req in stage.requires)

    def disabled_stages(self):
        """Names of stages retired after failures, in declaration order."""
        return [s.name for s in self.STAGES if self.stage_disabled(s.name)]

    # ------------------------------------------------------ failure policy
    #: Failure of a stage retires that metric for the whole study. Used for the
    #: baseline trial (trial 0) and for setup: a tool that cannot produce a
    #: number even once is a tool this run cannot use.
    BASELINE = "baseline"
    #: Failure invalidates the trial. Used from trial 1 on, where the stage has
    #: already been shown to work on this data: something anomalous happened,
    #: and a score built from a different metric set would not be comparable.
    STRICT = "strict"
    #: Failure is absorbed and reported. Used for the final assembly and the
    #: --rerun-* evaluations, which are reports rather than optimisation trials.
    LENIENT = "lenient"

    @property
    def failure_policy(self) -> str:
        """
        How this evaluator reacts to a metric stage failing.

        Derived from ``trial_id`` unless overridden at construction:

        ===================  ==========  ====================================
        trial_id             policy      effect of a stage failing
        ===================  ==========  ====================================
        ``0``                baseline    metric retired for the whole study
        ``1``, ``2``, ...    strict      trial discarded, metric set unchanged
        ``"best"``, ``None`` lenient     absorbed; the report omits the metric
        ===================  ==========  ====================================

        The asymmetry is deliberate. At trial 0 a failure means the tool cannot
        run here at all -- wrong lineage, missing database, unreadable input --
        so there is no point paying for it another ninety-nine times. After
        trial 0 the tool has demonstrably worked, so a failure says something
        about *this* assembly or a transient fault, and silently scoring the
        trial on fewer metrics would put an incomparable number into the study.
        """
        if self._failure_policy is not None:
            return self._failure_policy
        if isinstance(self.trial_id, bool) or not isinstance(self.trial_id, int):
            return self.LENIENT
        return self.BASELINE if self.trial_id == 0 else self.STRICT

    @property
    def is_baseline(self) -> bool:
        return self.failure_policy == self.BASELINE

    def _stage_entry(self, name, error):
        """Common bookkeeping for any stage failure."""
        entry = dict(self.stage_state.get(name, {}))
        entry["failures"] = int(entry.get("failures", 0)) + 1
        entry["last_error"] = str(error)[:500]
        entry["last_failed_trial"] = self.trial_id
        entry["last_failed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(error, TimeoutError):
            entry["timeouts"] = int(entry.get("timeouts", 0)) + 1
        return entry

    def retire_stage(self, name, error):
        """
        Disable a stage for the remainder of the study and record why.

        Called for baseline-trial failures and for setup failures (a BUSCO
        lineage that would not download, a `yak count` that would not run),
        where retrying per trial cannot possibly help.

        The record is written to ``work/cache/metric_stage_state.json`` so it
        survives both the next trial (which builds a new evaluator) and the
        next invocation of hifimizer against the same output directory.
        """
        stage = self.STAGES_BY_NAME[name]
        entry = self._stage_entry(name, error)
        entry["disabled"] = True
        entry["disabled_after_trial"] = self.trial_id

        where = "on the baseline assembly" if self.is_baseline else "during setup"
        dropped = [m for m in stage.metrics if m in self.weights]
        self.logger.error(
            f"Metric stage '{name}' ({stage.label}) failed {where} "
            f"and is DISABLED for the rest of this study. No trial "
            f"will attempt it again. Metrics dropped from the score: "
            f"{', '.join(dropped) if dropped else 'none (prerequisite stage)'}. "
            f"Reason: {entry['last_error']}"
        )
        if isinstance(error, TimeoutError) and stage.walltime_flag:
            self.logger.error(
                f"  It timed out rather than erroring. If that was the only "
                f"problem, raise {stage.walltime_flag} and re-run with "
                f"--force-rerun."
            )

        self.stage_state[name] = entry
        self._save_stage_state()

    def note_trial_failure(self, name, error):
        """
        Record a post-baseline stage failure without retiring the stage.

        The counter is kept for the end-of-run health summary; the stage stays
        enabled because it worked on the baseline and is expected to work again.
        """
        stage = self.STAGES_BY_NAME[name]
        entry = self._stage_entry(name, error)
        entry["trial_failures"] = int(entry.get("trial_failures", 0)) + 1
        self.logger.error(
            f"Metric stage '{name}' ({stage.label}) failed on trial "
            f"{self.trial_id}, having succeeded on the baseline. This trial is "
            f"discarded rather than scored on a reduced metric set. "
            f"Reason: {entry['last_error']}"
        )
        self.stage_state[name] = entry
        self._save_stage_state()

    def baseline_failures(self):
        """Stages retired *by the baseline trial*, in declaration order."""
        return [
            s.name
            for s in self.STAGES
            if self.stage_state.get(s.name, {}).get("disabled")
        ]

    def metric_health_report(self) -> str:
        """
        Human-readable summary of every stage that has ever misbehaved.

        Emitted after the baseline trial and again at the end of the run, so a
        degraded objective is impossible to miss in a multi-day log.
        """
        retired, flaky = [], []
        for stage in self.STAGES:
            entry = self.stage_state.get(stage.name)
            if not entry:
                continue
            if entry.get("disabled"):
                retired.append((stage, entry))
            elif entry.get("trial_failures"):
                flaky.append((stage, entry))

        if not retired and not flaky:
            return ""

        lines = [
            "",
            "#" * 72,
            "METRIC HEALTH WARNING - the objective is not the one you asked for",
            "#" * 72,
        ]

        if retired:
            lines.append(
                "Retired on the baseline assembly (never attempted again):"
            )
            for stage, entry in retired:
                dropped = [m for m in stage.metrics if m in self.weights]
                lines.append(f"  - {stage.label}")
                lines.append(
                    f"      metrics dropped : "
                    f"{', '.join(dropped) if dropped else 'none (prerequisite)'}"
                )
                lines.append(f"      reason          : {entry.get('last_error', '?')}")

        if flaky:
            lines.append("Failed on individual trials (those trials discarded):")
            for stage, entry in flaky:
                lines.append(
                    f"  - {stage.label}: {entry['trial_failures']} trial(s), "
                    f"last on trial {entry.get('last_failed_trial')}"
                )

        lines += [
            "",
            "ACTION REQUIRED: check your input data and read the tool logs before",
            "trusting these results. A metric that cannot be computed is usually a",
            "symptom of the inputs, not of the assembler:",
            f"  - per-command logs : {self.paths.logs_dir}",
            f"  - stage state      : {self.paths.metric_stage_state}",
            "  - verify the reads, the assembly FASTA and (for BUSCO) the lineage",
            "    dataset, then re-run with --force-rerun to retry the failed stage.",
            f"Metrics still scored: {self.metric_regime() or '(none)'}",
            "#" * 72,
        ]
        return "\n".join(lines)

    def _run_stage(self, name, func):
        """
        Run one metric stage and apply :attr:`failure_policy` to any failure.

        Raises:
            MetricStageFailure: under the ``strict`` policy, or for an
                essential stage under any policy.

        Returns:
            ``(ok, value)``. ``value`` is whatever ``func`` returned on
            success, and ``None`` on failure or when the stage was skipped.
        """
        stage = self.STAGES_BY_NAME[name]

        # A stage the user turned off is not a failure and must not be recorded
        # as one: otherwise every `--no-busco` trial reports a "failed stage".
        if self.stage_off_by_user(name):
            return False, None

        if not self.stage_enabled(name):
            if self.stage_disabled(name):
                self.logger.info(
                    f"Skipping {stage.label}: retired on the baseline assembly."
                )
            self.stage_outcomes[name] = False
            return False, None

        unmet = [r for r in stage.requires if not self.stage_outcomes.get(r)]
        if unmet:
            self.logger.warning(
                f"Skipping {stage.label}: it needs "
                f"{', '.join(self.STAGES_BY_NAME[u].label for u in unmet)}, "
                "which did not succeed for this assembly."
            )
            self.stage_outcomes[name] = False
            return False, None

        try:
            value = func()
        except Exception as e:  # noqa: BLE001 - the whole point is to absorb it
            self.stage_outcomes[name] = False
            self._handle_stage_failure(stage, e)
            return False, None

        self.stage_outcomes[name] = True
        return True, value

    def _handle_stage_failure(self, stage, error):
        """Route a stage failure according to the active policy."""
        policy = self.failure_policy

        # gfastats carries contig count, length and N50. There is no version of
        # "carry on with the metrics that worked" that survives losing those,
        # so it invalidates the evaluation no matter who is asking.
        if stage.essential:
            self.logger.error(
                f"{stage.label} is essential and failed; this assembly cannot "
                f"be scored at all. Reason: {error}"
            )
            self.stage_state[stage.name] = self._stage_entry(stage.name, error)
            self._save_stage_state()
            raise MetricStageFailure(
                stage.name,
                stage.label,
                error,
                reason=(
                    f"{stage.label} is essential and cannot be dropped, so this "
                    "assembly cannot be scored"
                ),
            )

        if policy == self.BASELINE:
            # Trial 0: the tool cannot run here. Retire it and carry on with a
            # smaller, honest metric set for the whole study.
            self.retire_stage(stage.name, error)
            return

        if policy == self.STRICT:
            # It worked on the baseline, so this failure is about this trial.
            self.note_trial_failure(stage.name, error)
            raise MetricStageFailure(
                stage.name,
                stage.label,
                error,
                reason=(
                    f"{stage.label} failed after succeeding on the baseline "
                    "assembly"
                ),
            )

        # LENIENT: a final-assembly or --rerun-* report. Nothing to invalidate
        # and nothing to retire; just say the number is missing.
        self.logger.warning(
            f"{stage.label} failed for this report; its metrics will be "
            f"absent from the summary below. Reason: {error}"
        )

    def _compile_patterns(self):
        """Pre-compile regexes for parsing the output of the evaluation tools."""
        self.gfastats_patterns = {
            "num_contigs": re.compile(r"# contigs:\s+(\d+)"),
            "length_diff": re.compile(r"Total contig length:\s+(\d+)"),
            "n50": re.compile(r"Contig N50:\s+(\d+)"),
        }

    def run_command(
        self, command, command_name="command", timeout_seconds=None, cwd=None
    ):
        """
        Run a command through the subprocess logger.

        Args:
            command: Shell command string.
            command_name: Used for the log filename and error messages.
            timeout_seconds: Wall-clock limit; on expiry the whole process
                group is killed and a TimeoutError is raised.
            cwd: Working directory for the command.

        Returns:
            The contents of the log file the command wrote to.
        """
        try:
            return_code, log_path = self.subprocess_logger.run_command_with_logging(
                command=command,
                log_filename=f"{command_name}.log",
                command_name=command_name,
                trial_id=self.trial_id,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
            )

            if return_code == TIMEOUT_EXIT_CODE:
                self.logger.error(
                    f"{command_name} exceeded its walltime and was killed. "
                    f"See log: {log_path}"
                )
                raise TimeoutError(
                    f"{command_name} timed out after {timeout_seconds} s "
                    f"- see {log_path}"
                )

            if return_code != 0:
                self.logger.error(
                    f"{command_name} failed (return code: {return_code}). "
                    f"See log: {log_path}"
                )
                raise RuntimeError(f"{command_name} failed - see {log_path}")

            with open(log_path, "r") as f:
                return f.read()

        except (RuntimeError, TimeoutError):
            raise
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise

    # ------------------------------------------------------------------ setup
    def download_busco(self, lineage="metazoa_odb12"):
        """
        Download the BUSCO lineage dataset into the work/ tree if absent.

        Called from setup when BUSCO is the primary backend, and lazily from
        the completeness chain when a compleasm failure sends it to BUSCO for
        the first time.
        """
        if not self.stage_enabled("busco"):
            self.logger.info(
                "Completeness scoring is disabled for this run; skipping the "
                "BUSCO lineage download."
            )
            return None

        lineage_dir = self.download_path / "lineages" / lineage
        if lineage_dir.exists():
            self.logger.debug(
                f"BUSCO lineage '{lineage}' already present in {self.download_path}."
            )
            return lineage_dir

        self.download_path.mkdir(parents=True, exist_ok=True)
        command = f"busco --download_path {self.download_path} --download {lineage}"
        try:
            self.run_command(command, command_name="busco_download")
        except Exception as e:
            self.logger.error(f"BUSCO download failed: {e}")
            raise
        return lineage_dir

    def prepare_completeness_dataset(self, lineage="metazoa_odb12"):
        """
        Fetch the lineage for whichever backend will actually run first.

        Downloading both formats costs a few hundred megabytes and several
        minutes for a dataset that may never be touched, so only the primary
        is fetched here. If compleasm later fails, the completeness chain
        downloads the BUSCO lineage on its way to the fallback.
        """
        if not self.stage_enabled("busco"):
            self.logger.info(
                "Completeness scoring is disabled for this run; skipping the "
                "lineage download."
            )
            return None

        if self.compleasm_available():
            self.logger.info(
                f"Preparing compleasm lineage '{lineage}' (primary backend). "
                "The BUSCO lineage is fetched only if compleasm fails."
            )
            return self.download_compleasm_lineage(lineage=lineage)

        self.logger.info(
            f"Preparing BUSCO lineage '{lineage}' (compleasm unavailable)."
        )
        return self.download_busco(lineage=lineage)

    # ------------------------------------------------------------------- yak
    def build_read_kmer_db(self, force=False):
        """
        Build the yak k-mer hash table for the *full* read set, once.

        The read hash is a property of the reads, not of any assembly, so it is
        built during setup and reused by every trial.

        Command (see https://github.com/lh3/yak):
            yak count -k31 -b37 -t<threads> -o reads.yak <reads>

        ``-b37`` uses a Bloom filter to discard singleton k-mers, which is what
        lh3 recommends for high-coverage read sets and keeps memory bounded.
        """
        if not self.kmer_eval:
            return None

        if self.paths.reads_yak.exists() and not force:
            self.logger.info(
                f"yak read hash already present at {self.paths.reads_yak}; "
                "skipping `yak count`."
            )
            return self.paths.reads_yak

        self.paths.reads_yak.parent.mkdir(parents=True, exist_ok=True)
        bloom = f"-b{self.yak_bloom_bits} " if self.yak_bloom_bits else ""
        command = (
            f"yak count -k{self.yak_k} {bloom}"
            f"-t{self.threads} -o {self.paths.reads_yak} {self.input_reads}"
        )
        self.logger.info(f"Building yak read k-mer hash: {command}")
        try:
            self.run_command(
                command,
                command_name="yak_count",
                timeout_seconds=self.stage_walltime_seconds("yak"),
            )
        except Exception as e:
            self.logger.error(
                f"yak count failed: {e}. k-mer metrics will be unavailable; "
                "re-run with --no-kmer-eval to silence this."
            )
            self.kmer_eval = False
            self.retire_stage("yak", e)
            return None

        return self.paths.reads_yak

    @staticmethod
    def parse_yak_qv(qv_file):
        """
        Parse the output of ``yak qv``.

        yak writes tab-separated records (see main.c in lh3/yak):

            CT  <occ>  <read_kmer_count>  <asm_kmer_count>  <adjusted_count>
            FR  <fpr_lower>  <fpr_upper>
            ER  <total_input_kmers>  <adjusted_error_kmers>
            CV  <cov>
            QV  <qv_raw>  <qv_adjusted>

        ``CV`` is the fraction of read k-mers at the modal occurrence that are
        found in the assembly, i.e. a k-mer completeness estimate in [0, 1].

        ``QV`` carries two values: the naive estimate and the model-calibrated
        one.  yak sets the calibrated value to -1 when the read histogram is
        too shallow to fit the model (``max_c <= 4``), in which case we fall
        back to the raw estimate.

        Returns:
            dict with ``qv`` (Phred) and ``kmer_completeness`` (percent).
        """
        qv_raw = qv_adj = cov = None

        with open(qv_file, "r") as fh:
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if not fields:
                    continue
                tag = fields[0]
                try:
                    if tag == "CV" and len(fields) >= 2:
                        cov = float(fields[1])
                    elif tag == "QV" and len(fields) >= 3:
                        qv_raw = float(fields[1])
                        qv_adj = float(fields[2])
                except ValueError:
                    continue

        # Prefer the calibrated QV; -1 means yak declined to calibrate.
        if qv_adj is not None and qv_adj > 0:
            qv = qv_adj
        elif qv_raw is not None and qv_raw > 0:
            qv = qv_raw
        else:
            qv = 0.0

        completeness = (cov * 100.0) if cov and cov > 0 else 0.0
        # CV is a ratio of estimates and can marginally exceed 1.0.
        completeness = min(completeness, 100.0)

        return {"qv": float(qv), "kmer_completeness": float(completeness)}

    def _combine_fastas(self, fasta_files, out_path):
        with open(out_path, "wb") as out:
            for f in fasta_files:
                with open(f, "rb") as fh:
                    shutil.copyfileobj(fh, out)
        return out_path

    def run_yak_qv(self, fasta_file, extra_fasta_files=None):
        """
        Compute consensus QV and k-mer completeness with ``yak qv``.

        Command:
            yak qv -t<threads> -K<chunk> reads.yak asm.fa > yak_qv.txt

        ``-p`` (per-sequence QV) is deliberately omitted: it adds one line per
        contig to stdout and we only consume the whole-assembly summary.

        QV is measured on ``fasta_file`` alone, because it is a per-haplotype
        base-accuracy statistic.  k-mer completeness is measured on the union
        of ``fasta_file`` and ``extra_fasta_files`` (i.e. hap1 + hap2), because
        completeness of a single haplotype is structurally capped by
        heterozygosity: scoring hap1 alone rewards collapsed assemblies.
        """
        if not self.kmer_eval:
            return {}

        if not self.paths.reads_yak.exists():
            self.logger.warning(
                "yak read hash not found; skipping k-mer metrics for this trial."
            )
            return {}

        tdir = self.trial_dir
        # -K batches sequence loading; sizing it near the haploid genome length
        # follows the `-K3.2g` example in the yak README.
        chunk = max(100_000_000, int(self.known_genome_size))

        # --- QV on the primary haplotype ---
        qv_out = tdir / "yak_qv.primary.txt"
        command = (
            f"yak qv -t{self.threads} -K{chunk} "
            f"{self.paths.reads_yak} {fasta_file} > {qv_out}"
        )
        self.run_command(
            command,
            command_name="yak_qv",
            timeout_seconds=self.stage_walltime_seconds("yak"),
        )
        primary = self.parse_yak_qv(qv_out)
        extra_fasta_files = [f for f in (extra_fasta_files or []) if Path(f).exists()]

        if not extra_fasta_files:
            return primary

        primary["qv_hap1"] = primary["qv"]
        primary["kmer_completeness_hap1"] = primary["kmer_completeness"]

        # --- second haplotype, measured on its own ---
        # Consensus accuracy and k-mer completeness are both per-haplotype
        # quantities, so with two haplotypes there are two of each and scoring
        # only the first leaves hap2 out of the objective entirely.
        hap2_out = tdir / "yak_qv.hap2.txt"
        self.run_command(
            f"yak qv -t{self.threads} -K{chunk} "
            f"{self.paths.reads_yak} {extra_fasta_files[0]} > {hap2_out}",
            command_name="yak_qv_hap2",
            timeout_seconds=self.stage_walltime_seconds("yak"),
        )
        hap2 = self.parse_yak_qv(hap2_out)
        primary["qv_hap2"] = hap2["qv"]
        primary["kmer_completeness_hap2"] = hap2["kmer_completeness"]
        primary["qv"] = (primary["qv_hap1"] + hap2["qv"]) / 2.0

        # --- completeness on the full (diploid) assembly ---
        # Kept alongside the per-haplotype figures rather than replaced by
        # them. Per-haplotype completeness is structurally capped by
        # heterozygosity and is *higher* for a collapsed assembly that packed
        # both alleles into one haplotype, so on its own it rewards exactly
        # the failure mode Hi-C phasing is meant to avoid. The combined figure
        # is the only metric here that penalises that collapse.
        combined = self._combine_fastas(
            [fasta_file] + list(extra_fasta_files), tdir / "combined_haps.fasta"
        )
        comb_out = tdir / "yak_qv.combined.txt"
        self.run_command(
            f"yak qv -t{self.threads} -K{chunk * 2} "
            f"{self.paths.reads_yak} {combined} > {comb_out}",
            command_name="yak_qv_combined",
            timeout_seconds=self.stage_walltime_seconds("yak"),
        )
        primary["kmer_completeness"] = self.parse_yak_qv(comb_out)[
            "kmer_completeness"
        ]
        try:
            combined.unlink()
        except Exception:
            pass

        return primary

    # ---------------------------------------------------------------- gfastats
    def run_gfastats(self, gfa_file, extra_gfa_files=None):
        """
        Assembly statistics, over every haplotype rather than just the first.

        With a second haplotype present, scoring hap1 alone leaves half the
        assembly unmeasured: hap2's contig count, N50 and length never enter
        the objective, so a parameter set that builds a good hap1 and a poor
        hap2 scores the same as one that builds two good haplotypes.

        The three scored numbers are reduced across haplotypes as:

        ``num_contigs``  summed  -- fragmentation anywhere is fragmentation
        ``n50``          mean    -- so one contiguous haplotype cannot hide
                                    a shattered one
        ``length_diff``  mean of each haplotype's own deviation from the
                         *haploid* genome size, which is the size each
                         haplotype should individually be

        The individual haplotype lengths are also returned, unweighted, so the
        log shows both rather than only their summary.
        """
        gfas = [Path(gfa_file)] + [
            Path(f) for f in (extra_gfa_files or []) if Path(f).exists()
        ]

        per_hap = []
        for index, gfa in enumerate(gfas, start=1):
            # A distinct command_name per haplotype is load-bearing, not
            # cosmetic. run_command returns the *contents of the log file*, and
            # the subprocess logger opens it in append mode, so two runs
            # sharing a name write into one file. parse_gfastats_output uses
            # re.search, which returns the first match -- so hap2 would be
            # parsed out of hap1's output and both haplotypes would report
            # identical contig counts, N50 and length.
            name = "gfastats" if len(gfas) == 1 else f"gfastats_hap{index}"
            try:
                stdout = self.run_command(
                    f"gfastats --discover-paths {gfa}",
                    name,
                    timeout_seconds=self.stage_walltime_seconds("gfastats"),
                )
            except (RuntimeError, TimeoutError):
                self.logger.error(f"gfastats analysis failed for {gfa.name}")
                raise
            per_hap.append(self.parse_gfastats_output(stdout))

        return self.combine_gfastats(per_hap)

    @classmethod
    def combine_gfastats(cls, per_hap):
        """
        Turn per-haplotype gfastats dicts into the scored metric set.

        With one haplotype the metric names are unsuffixed and nothing about
        scoring changes. With two, ``num_contigs``, ``n50`` and ``length_diff``
        are emitted *per haplotype* (``n50_hap1``, ``n50_hap2``, ...) and the
        unsuffixed names are not produced at all.

        Splitting rather than summarising is the point: a mean N50 lets one
        contiguous haplotype hide a shattered one, and a mean length deviation
        lets +10 Mb on hap1 cancel -10 Mb on hap2. Two separate penalties
        cannot cancel each other.
        """
        if not per_hap:
            raise RuntimeError("gfastats produced no output for any haplotype")

        metrics = {}
        single = len(per_hap) == 1

        for index, hap in enumerate(per_hap, start=1):
            suffix = "" if single else f"_hap{index}"
            for key in cls.HAPLOTYPE_SPLIT_METRICS:
                if key in hap:
                    metrics[f"{key}{suffix}"] = float(hap[key])
            if "total_length" in hap and index <= 2:
                metrics[f"hap{index}_length_mb"] = hap["total_length"] / 1e6
        return metrics

    def parse_gfastats_output(self, output):
        metrics = {}
        for key, pattern in self.gfastats_patterns.items():
            match = re.search(pattern, output)
            if match:
                value = int(match.group(1))
                if key == "length_diff":
                    # Kept alongside the log-scaled deviation so the log can
                    # show what each haplotype actually measured, in Mb.
                    metrics["total_length"] = float(value)
                    metrics[key] = np.log(
                        (abs(value - self.known_genome_size) / 1_000_000) + 1
                    )
                else:
                    metrics[key] = np.log(value + 1)
        return metrics

    def convert_extra_haplotypes(self, prefix, suffixes):
        """
        Convert the sibling haplotype GFAs next to ``prefix`` into FASTAs.

        Used only to widen the k-mer completeness estimate across both
        haplotypes. Missing or unconvertible files are skipped with a warning
        rather than failing the trial: completeness on the primary haplotype
        alone is still a usable number.

        Returns:
            list of FASTA paths that were successfully written.
        """
        prefix = Path(prefix)
        out = []
        for suffix in suffixes or []:
            gfa = prefix.parent / f"{prefix.name}.{suffix}.gfa"
            if not gfa.exists():
                continue
            fasta = self.trial_dir / f"{suffix}.fasta"
            try:
                self.convert_gfa_to_fasta(gfa, fasta)
                out.append(fasta)
            except Exception as e:
                self.logger.warning(f"Could not convert {gfa.name}: {e}")
        return out

    @staticmethod
    def convert_gfa_to_fasta(gfa_file, output_fasta):
        """Extract the S-lines of a GFA into a FASTA file."""
        command = ["awk", '$1 == "S" {print ">"$2"\\n"$3}', str(gfa_file)]
        Path(output_fasta).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fasta, "w") as out_file:
            subprocess.run(command, stdout=out_file, check=True)
        return True

    # ----------------------------------------------------- gene completeness
    #: Completeness backends, attempted in this order. compleasm is a
    #: miniprot-based reimplementation of BUSCO and is both faster and far
    #: lighter on memory, which is why it leads. BUSCO's own miniprot backend
    #: is deliberately absent: it would re-run the same aligner compleasm
    #: already tried, so a compleasm failure tells us nothing new about it.
    #: metaeuk and augustus are genuinely different gene finders and are worth
    #: the fallback.
    COMPLETENESS_BACKENDS = ("compleasm", "busco:metaeuk", "busco:augustus")

    #: BUSCO gene predictors reachable as a fallback, in order of cost
    BUSCO_BACKENDS = ("metaeuk", "augustus")

    # ------------------------------------------------------ compleasm lookup
    def _discover_compleasm(self):
        """
        Locate the ``compleasm`` executable.

        compleasm and BUSCO cannot share a conda environment -- their pinned
        dependencies conflict -- so compleasm is installed into a *separate*
        environment and is not on the optimiser's own PATH. Search order:

        1. an explicit ``--compleasm-bin`` / constructor argument
        2. ``$COMPLEASM_BIN`` (what the Docker/Singularity image sets)
        3. ``compleasm`` on PATH, for anyone who did get them co-installed
        4. a sibling conda environment named ``compleasm``, resolved relative
           to the running interpreter, which covers a local ``conda create``
        5. the image's canonical location

        Returns:
            ``Path`` to the executable, or None if nothing was found.
        """
        candidates = []
        if self._compleasm_bin:
            candidates.append(Path(self._compleasm_bin))

        env_bin = os.environ.get("COMPLEASM_BIN")
        if env_bin:
            candidates.append(Path(env_bin))

        on_path = shutil.which("compleasm")
        if on_path:
            candidates.append(Path(on_path))

        # sys.prefix is .../envs/optimizer; its sibling is .../envs/compleasm
        import sys

        sibling = Path(sys.prefix).parent / "compleasm" / "bin" / "compleasm"
        candidates.append(sibling)
        candidates.append(Path("/opt/conda/envs/compleasm/bin/compleasm"))

        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
        return None

    @property
    def compleasm_bin(self):
        """Cached result of :meth:`_discover_compleasm` (None if unavailable)."""
        if not self._compleasm_resolved:
            self._compleasm_bin = self._discover_compleasm()
            self._compleasm_resolved = True
            if self._compleasm_bin:
                self.logger.info(f"Using compleasm at {self._compleasm_bin}")
            else:
                self.logger.info(
                    "compleasm not found; gene-space completeness will use "
                    "BUSCO. Set --compleasm-bin or $COMPLEASM_BIN to point at "
                    "it, or install it into a conda environment named "
                    "'compleasm' alongside this one."
                )
        return self._compleasm_bin

    def _compleasm_env_prefix(self):
        """
        ``PATH=...`` prefix that puts compleasm's own environment first.

        compleasm shells out to ``miniprot`` and ``hmmsearch`` by bare name.
        Running the executable by absolute path is not enough: without this
        prefix it would find the *optimiser* environment's miniprot, which is
        a different build pinned against BUSCO's dependency set. Prepending
        rather than replacing keeps ``sh``, ``awk`` and friends resolvable.
        """
        bin_dir = self.compleasm_bin.parent
        return f"PATH={bin_dir}:$PATH "

    def compleasm_available(self) -> bool:
        """True if compleasm should and can be used."""
        return bool(self.use_compleasm and self.compleasm_bin)

    # ---------------------------------------------------------- compleasm run
    def download_compleasm_lineage(self, lineage="metazoa_odb12"):
        """
        Fetch the compleasm lineage library if it is not already present.

        compleasm downloads on demand during ``run``, but doing it here keeps
        a slow first download out of the per-attempt walltime, where it would
        look like a hung gene predictor.
        """
        if not self.compleasm_available():
            return None

        lineage_dir = self.compleasm_download_path / lineage
        if lineage_dir.exists() and any(lineage_dir.iterdir()):
            self.logger.info(
                f"compleasm lineage '{lineage}' already present in "
                f"{self.compleasm_download_path}. Skipping download."
            )
            return lineage_dir

        self.compleasm_download_path.mkdir(parents=True, exist_ok=True)
        command = (
            f"{self._compleasm_env_prefix()}{self.compleasm_bin} download "
            f"{lineage} -L {self.compleasm_download_path}"
        )
        self.logger.info(f"Downloading compleasm lineage '{lineage}'")
        self.run_command(command, command_name="compleasm_download")
        return lineage_dir

    def run_compleasm(self, fasta_file, lineage="metazoa_odb12"):
        """
        Score gene-space completeness with compleasm.

            compleasm run -a <asm.fa> -o <outdir> -l <lineage> -t N -L <lib>

        compleasm writes ``<outdir>/summary.txt``, which
        :meth:`parse_compleasm_summary` reads. Bounded by ``--busco-walltime``
        like every other completeness attempt.
        """
        if not self.compleasm_available():
            raise RuntimeError("compleasm is not available in this environment")

        out_dir = self.trial_dir / "compleasm_output"
        # compleasm appends to an existing run directory rather than replacing
        # it, so a stale directory from a killed attempt would be read back as
        # if it were this trial's result.
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        command = (
            f"{self._compleasm_env_prefix()}{self.compleasm_bin} run "
            f"-a {fasta_file} -o {out_dir} -l {lineage} "
            f"-t {self.threads} -L {self.compleasm_download_path}"
        )

        self.logger.info(
            f"Running compleasm (walltime: {self.stage_walltimes.get('busco')} h)"
        )
        self.run_command(
            command,
            command_name="compleasm",
            timeout_seconds=self.stage_walltime_seconds("busco"),
        )

        summary = out_dir / "summary.txt"
        if not summary.exists():
            matches = list(out_dir.glob("**/summary.txt"))
            if not matches:
                raise FileNotFoundError(
                    f"compleasm finished but no summary.txt was written under {out_dir}"
                )
            summary = matches[0]

        return self.parse_compleasm_summary(summary)

    #: ``S:90.03%, 605`` -- category, percentage, count
    _COMPLEASM_CATEGORY = re.compile(
        r"^\s*([SDFIM])\s*:\s*([\d.]+)\s*%\s*,\s*(\d+)\s*$", re.MULTILINE
    )
    #: ``N:672`` -- total markers in the lineage
    _COMPLEASM_TOTAL = re.compile(r"^\s*N\s*:\s*(\d+)\s*$", re.MULTILINE)

    @classmethod
    def parse_compleasm_summary(cls, summary_file):
        """
        Parse compleasm's ``summary.txt`` into the BUSCO metric schema.

        The file looks like::

            ## lineage: metazoa_odb12
            S:90.03%, 605
            D:5.51%, 37
            F:2.38%, 16
            I:0.00%, 0
            M:2.08%, 14
            N:672

        Category mapping
        ----------------
        ``S``, ``D`` and ``M`` map straight onto ``single_copy``,
        ``multi_copy`` and ``missing``. ``I`` has no BUSCO equivalent:
        compleasm splits partially recovered markers into *fragmented* (a
        truncated alignment) and *incomplete* (a gene found, but not to the
        completeness threshold), where BUSCO reports both as ``Fragmented``.
        ``I`` is therefore folded into ``fragmented``, which keeps
        ``S + D + F + M == N`` and keeps a compleasm-scored trial comparable
        with a BUSCO-scored one. It also keeps the four counts summing to the
        marker total, which is what ``format_metrics`` divides by to get the
        percentages it prints.

        Counts are stored log-transformed, exactly as
        :meth:`parse_busco_results` does, so nothing downstream can tell which
        tool produced them.

        Raises:
            ValueError: if no category line parsed, so that a truncated or
                reformatted summary triggers the BUSCO fallback instead of
                silently scoring the trial as 100% missing.
        """
        text = Path(summary_file).read_text()

        counts = {}
        for category, _pct, count in cls._COMPLEASM_CATEGORY.findall(text):
            counts[category] = int(count)

        if not counts:
            raise ValueError(
                f"No S/D/F/I/M category lines found in {summary_file}; "
                "compleasm's output format may have changed."
            )

        missing_categories = {"S", "D", "F", "M"} - counts.keys()
        if missing_categories:
            raise ValueError(
                f"compleasm summary {summary_file} is missing the "
                f"{', '.join(sorted(missing_categories))} categor"
                f"{'y' if len(missing_categories) == 1 else 'ies'}."
            )

        single = counts["S"]
        duplicated = counts["D"]
        # "Incomplete" is compleasm-only; see the docstring.
        fragmented = counts["F"] + counts.get("I", 0)
        absent = counts["M"]

        total_match = cls._COMPLEASM_TOTAL.search(text)
        if total_match:
            total = int(total_match.group(1))
            summed = single + duplicated + fragmented + absent
            if total and summed != total:
                logging.getLogger(__name__).warning(
                    f"compleasm categories sum to {summed} but the lineage has "
                    f"{total} markers ({summary_file}); percentages in the "
                    "metric summary will be computed against the sum."
                )

        return {
            "single_copy": np.log(single + 1),
            "multi_copy": np.log(duplicated + 1),
            "fragmented": np.log(fragmented + 1),
            "missing": np.log(absent + 1),
        }

    # -------------------------------------------------------------- BUSCO run
    def run_busco(
        self, fasta_file, lineage="metazoa_odb12", mode="genome", backend="metaeuk"
    ):
        """
        Run BUSCO once, with an explicit gene predictor.

        Backend selection has moved up into :meth:`run_completeness`, which
        walks compleasm and the BUSCO predictors as one ordered chain. This
        method performs a single attempt so that the caller decides what a
        failure means.

        The attempt is bounded by ``--busco-walltime``, enforced by
        ``SubprocessLogger`` killing the whole process group. GNU ``timeout``
        is deliberately *not* used: it signals only the direct child, leaving
        BUSCO's metaeuk/augustus/hmmsearch grandchildren running.
        """
        tdir = self.trial_dir
        out_name = "busco_output"
        out_dir = tdir / out_name

        # BUSCO's -o must be a bare name; the location is set with --out_path.
        command = (
            f"busco -i {fasta_file} -l {lineage} -m {mode} "
            f"-o {out_name} --out_path {tdir} "
            f"-c {self.threads} --skip_bbtools --force "
            f"--download_path {self.download_path} --{backend}"
        )

        self.logger.info(
            f"Running BUSCO with {backend} "
            f"(walltime: {self.stage_walltimes.get('busco')} h)"
        )
        self.run_command(
            command,
            f"busco_{backend}",
            timeout_seconds=self.stage_walltime_seconds("busco"),
        )

        matches = list(out_dir.glob(f"short_summary.specific.{lineage}.*.json"))
        if not matches:
            matches = list(out_dir.glob("short_summary.*.json"))
        if not matches:
            raise FileNotFoundError(f"BUSCO summary JSON not found in {out_dir}")

        return self.parse_busco_results(str(matches[0]))

    @staticmethod
    def parse_busco_results(busco_json_file):
        with open(busco_json_file, "r") as f:
            data = json.load(f)

        return {
            "single_copy": np.log(data["results"]["Single copy BUSCOs"] + 1),
            "multi_copy": np.log(data["results"]["Multi copy BUSCOs"] + 1),
            "fragmented": np.log(data["results"]["Fragmented BUSCOs"] + 1),
            "missing": np.log(data["results"]["Missing BUSCOs"] + 1),
        }

    # ------------------------------------------------------ completeness chain
    def _completeness_order(self):
        """
        Backends to try, best-known-working first.

        The cache records whichever backend last succeeded, so a run that fell
        through to BUSCO/augustus on trial 0 does not re-pay for the compleasm
        and metaeuk failures on trials 1..99. Backends that cannot run at all
        in this environment are dropped rather than ordered.
        """
        available = [
            b
            for b in self.COMPLETENESS_BACKENDS
            if b != "compleasm" or self.compleasm_available()
        ]

        cached = self.backend_cache.get("completeness")

        # Cache written before compleasm existed: the key was "busco" and the
        # value a bare predictor name. It still carries real information --
        # "augustus" means metaeuk had already failed on this data -- so it is
        # used to order the BUSCO entries. It does not promote BUSCO above
        # compleasm: compleasm is a different tool and has never been tried.
        if cached is None:
            legacy = self.backend_cache.get("busco")
            if legacy in self.BUSCO_BACKENDS:
                legacy_id = f"busco:{legacy}"
                available = [b for b in available if b != legacy_id]
                insert_at = 1 if available and available[0] == "compleasm" else 0
                available.insert(insert_at, legacy_id)
            return available

        if cached in available:
            return [cached] + [b for b in available if b != cached]
        return available

    def _run_completeness_backend(self, backend, fasta_file, lineage):
        """Dispatch one backend id from :attr:`COMPLETENESS_BACKENDS`."""
        if backend == "compleasm":
            return self.run_compleasm(fasta_file, lineage=lineage)

        predictor = backend.split(":", 1)[1]
        # A fallback to BUSCO needs BUSCO's own lineage format, which the
        # compleasm-first setup path will not have fetched. Done here rather
        # than inside the attempt so a multi-hundred-megabyte download is not
        # charged against the gene predictor's walltime.
        self.download_busco(lineage=lineage)
        return self.run_busco(fasta_file, lineage=lineage, backend=predictor)

    def run_completeness(self, fasta_file, lineage="metazoa_odb12"):
        """
        Score gene-space completeness, trying each backend in turn.

        compleasm runs first: it is miniprot-based, typically minutes rather
        than hours, and its peak memory is a fraction of a threaded metaeuk or
        augustus run -- which matters because this stage runs last in a trial,
        when the process is already at its high-water mark. BUSCO with metaeuk
        and then augustus are the fallbacks. BUSCO's *own* miniprot backend is
        not in the chain: compleasm already is miniprot, so re-running it under
        BUSCO would buy nothing but another walltime.

        Whichever backend succeeds is cached in
        ``work/cache/busco_backend_cache.json`` so later trials start with it.

        Returns:
            dict with ``single_copy``, ``multi_copy``, ``fragmented`` and
            ``missing``, log-transformed, regardless of which tool ran.

        Raises:
            CompletenessFailedError: every available backend failed.
        """
        order = self._completeness_order()
        if not order:
            raise CompletenessFailedError(
                "No gene-space completeness backend is available: compleasm "
                "was not found and BUSCO is not usable. Re-run with --no-busco "
                "to score without completeness metrics."
            )

        last_error = None
        for backend in order:
            try:
                metrics = self._run_completeness_backend(
                    backend, fasta_file, lineage
                )
            except (RuntimeError, TimeoutError, ValueError, FileNotFoundError) as e:
                last_error = e
                self.logger.warning(
                    f"Completeness backend '{backend}' failed or exceeded its "
                    f"walltime: {e}"
                )
                continue

            if self.backend_cache.get("completeness") != backend:
                self.backend_cache["completeness"] = backend
                self._save_backend_cache()
            self.logger.info(f"Gene-space completeness scored with {backend}")
            return metrics

        raise CompletenessFailedError(
            f"Gene-space completeness failed or exceeded the "
            f"{self.stage_walltimes.get('busco')} h walltime with every backend "
            f"({', '.join(order)}). This usually means a gene-prediction step "
            "is hanging or broken in this environment. Re-run with --no-busco "
            f"to proceed without completeness scoring. Last error: {last_error}"
        )

    # ------------------------------------------------------- Hi-C phasing
    def build_diploid_fasta(self, fasta_file, extra_fasta_files):
        """
        Concatenate the haplotype FASTAs, prefixing contig names by haplotype.

        Contigs become ``h1_<name>`` / ``h2_<name>``, which is what makes the
        phasing metric computable at all: a Hi-C pair's two mates only tell us
        something about phasing if we can say which haplotype each landed on,
        and hifiasm gives the two haplotypes overlapping contig names.

        Returns:
            Path to the combined FASTA, or None if there is no second
            haplotype (i.e. nothing to phase).
        """
        extra = [Path(f) for f in (extra_fasta_files or []) if Path(f).exists()]
        if not extra:
            return None

        out = self.trial_dir / "diploid_prefixed.fasta"
        with open(out, "w") as fh:
            for index, source in enumerate([Path(fasta_file)] + extra, start=1):
                subprocess.run(
                    [
                        "awk",
                        "-v", f"p=h{index}_",
                        '/^>/ {print ">" p substr($0, 2); next} {print}',
                        str(source),
                    ],
                    stdout=fh,
                    check=True,
                )
        return out

    def subset_hic_reads(self, force=False):
        """
        Draw a fixed subset of Hi-C pairs, once, and reuse it every trial.

        Both mates are sampled with the same seed so the pairing survives.

        Returns:
            ``(r1, r2)`` paths, or ``(None, None)`` when no Hi-C reads were
            given.
        """
        if not (self.hic1 and self.hic2):
            return None, None

        outputs = []
        for index, source in enumerate((self.hic1, self.hic2), start=1):
            target = self.paths.reads_dir / f"hic_subset_{index}.fq"
            outputs.append(target)
            if target.exists() and not force:
                continue
            self.paths.reads_dir.mkdir(parents=True, exist_ok=True)
            command = (
                f"set -o pipefail; seqtk sample -s{self.subset_seed} "
                f"{source} {self.num_hic_reads} > {target}"
            )
            self.run_command(
                command,
                command_name=f"seqtk_sample_hic{index}",
                timeout_seconds=self.stage_walltime_seconds("hic_phasing"),
            )
        return outputs[0], outputs[1]

    #: Count Hi-C pairs whose mates sit on different contigs, split by whether
    #: those contigs belong to the same haplotype. Only read 1 of each pair is
    #: counted (``-f 0x40``) so a pair contributes once, and 0x90c drops
    #: unmapped, mate-unmapped, secondary and supplementary records.
    _HIC_PAIR_AWK = (
        '{ if ($7 != "=" && $7 != "*") { '
        'a = substr($3, 1, 2); b = substr($7, 1, 2); '
        'if (a == b) cis++; else trans++ } } '
        'END { printf "%d %d\\n", cis + 0, trans + 0 }'
    )

    #: The counts line: two integers alone on a line. ``run_command`` returns
    #: the whole log file, which the subprocess logger prefixes with a banner
    #: and a header, so the numbers have to be found rather than assumed to be
    #: the first thing in the output.
    _HIC_COUNTS_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$", re.MULTILINE)

    @classmethod
    def parse_hic_pair_counts(cls, output):
        """
        Turn the ``cis trans`` line from :attr:`_HIC_PAIR_AWK` into metrics.

        ``trans_hap_rate`` is the percentage of informative pairs whose two
        mates landed on *different* haplotypes. Hi-C contacts happen within a
        physical chromosome, so in a correctly phased assembly this should be
        small; a high value means contigs have been assigned to the wrong
        haplotype. It is the only metric here that responds directly to
        --s-base, --f-perturb and --l-msjoin.

        ``hic_pairs_informative`` is unweighted and exists so a suspiciously
        good rate computed from a handful of pairs is visible rather than
        silently trusted.

        The last match wins: the log is opened in append mode, so a retry
        within one trial leaves the earlier attempt's counts above the current
        ones.
        """
        matches = cls._HIC_COUNTS_RE.findall(output or "")
        if not matches:
            raise RuntimeError(
                "Could not find the Hi-C pair counts in the command output. "
                "Check the hic_phasing log for a minimap2 or samtools error."
            )
        cis, trans = (int(v) for v in matches[-1])
        informative = cis + trans
        if informative == 0:
            raise RuntimeError(
                "No informative Hi-C pairs: every pair was unmapped, "
                "low-quality, or had both mates on one contig. Raise "
                "--num-hic-reads or lower --hic-min-mapq."
            )
        return {
            "trans_hap_rate": 100.0 * trans / informative,
            "hic_pairs_informative": float(np.log(informative + 1)),
        }

    def run_hic_phasing(self, diploid_fasta):
        """
        Measure how consistently Hi-C links stay inside one haplotype.

            minimap2 -ax sr diploid.fa hic_1.fq hic_2.fq
              | samtools view -q <mapq> -F 0x90c -f 0x40 -
              | awk '<count cis vs trans>'

        The MAPQ filter is doing the real work. Two haplotypes of one
        individual are nearly identical, so most Hi-C reads map equally well
        to both and get MAPQ 0. What survives the filter is the subset of
        reads that overlap a haplotype-distinguishing variant -- which is
        exactly the subset that carries phasing information, and the same
        signal hifiasm partitions on. It does mean the metric is computed
        from a minority of the sampled pairs, which is why the surviving
        count is reported alongside the rate.

        Circularity is worth naming: hifiasm phased using these same reads, so
        this is a fit statistic rather than an independent test. It still
        separates parameter sets that phase from ones that do not. Holding
        out a fraction of the pairs from the hifiasm invocation would make it
        a genuine cross-validation; that is not done here because it would
        change what the final assembly is built from.
        """
        if diploid_fasta is None:
            raise RuntimeError(
                "Hi-C phasing needs two haplotype assemblies; only one was "
                "produced. This stage is only meaningful for --hic1/--hic2 runs."
            )

        r1, r2 = self.subset_hic_reads()
        if not (r1 and r2):
            raise RuntimeError("Hi-C read subset unavailable")

        command = (
            f"set -o pipefail; "
            f"minimap2 -ax sr -t {self.threads} {diploid_fasta} {r1} {r2} "
            f"| samtools view -q {self.hic_min_mapq} -F 0x90c -f 0x40 - "
            f"| awk '{self._HIC_PAIR_AWK}'"
        )

        self.logger.info("Scoring Hi-C phasing consistency (minimap2 -x sr)")
        output = self.run_command(
            command,
            command_name="hic_phasing",
            timeout_seconds=self.stage_walltime_seconds("hic_phasing"),
        )
        metrics = self.parse_hic_pair_counts(output)
        self.logger.info(
            "Hi-C phasing: %.2f%% of %d informative pairs link across "
            "haplotypes",
            metrics["trans_hap_rate"],
            round(float(np.expm1(metrics["hic_pairs_informative"]))),
        )
        return metrics

    # ---------------------------------------------------------------- scoring
    def _load_weights(self):
        """Load metric weights from weights.json, falling back to defaults."""
        # Importances, not signed weights. Direction is already handled: every
        # metric is scored as a log2 fold change signed so that positive means
        # better, so a weight here answers one question -- what is a doubling
        # (or halving) of this quantity worth, relative to the others?
        default_weights = {
            # Contiguity. Halving the contig count and doubling the N50 are
            # the two headline improvements, priced equally.
            "num_contigs": 1.0,
            "length_diff": 0.5,
            "n50": 1.0,
            # Per-haplotype versions, used instead of the three above when a
            # second haplotype exists. Each carries half the single-haplotype
            # weight so that splitting a metric in two does not silently
            # double how much that property counts for.
            "num_contigs_hap1": 0.5,
            "num_contigs_hap2": 0.5,
            "length_diff_hap1": 0.25,
            "length_diff_hap2": 0.25,
            "n50_hap1": 0.5,
            "n50_hap2": 0.5,
            # Gene space. The three deficits carry the signal, because they can
            # halve or double; single_copy is a bounded count that barely moves
            # proportionally (605 -> 610 out of 672 is 0.8%), so it is priced
            # low rather than pretending otherwise.
            "single_copy": 0.2,
            "multi_copy": 0.7,
            "fragmented": 0.7,
            "missing": 1.0,
            # k-mer metrics, both scored on what the assembly is *short of*
            # rather than the percentage itself (see `badness`), so halving the
            # missing k-mers or halving the error rate behind the QV each move
            # one unit, like everything else here.
            "qv": 1.0,
            "kmer_completeness": 1.0,
            "kmer_completeness_hap1": 0.5,
            "kmer_completeness_hap2": 0.5,
            # Hi-C phasing. Halving the cross-haplotype link rate is worth as
            # much as halving the contig count: a well-phased assembly and a
            # contiguous one are meant to trade off, not for one to dominate.
            "trans_hap_rate": 1.0,
        }

        candidates = [
            Path.cwd() / "weights.json",
            Path(__file__).parent / "weights.json",
            Path(__file__).resolve().parents[2] / "weights.json",
        ]

        log = logging.getLogger("AssemblyEval")

        for candidate in candidates:
            try:
                if not candidate.exists():
                    continue
                with open(candidate, "r") as fh:
                    loaded = json.load(fh) or {}

                validated = {}
                negative_keys = []
                for key, default in default_weights.items():
                    if key not in loaded:
                        validated[key] = default
                        continue
                    try:
                        raw_weight = float(loaded[key])
                        # Weights used to carry direction as a sign. They no
                        # longer need to -- the fold change is already signed
                        # so positive means better -- so a negative weight here
                        # would invert the metric. The magnitude is used and
                        # the file is flagged once.
                        if raw_weight < 0:
                            negative_keys.append(key)
                        validated[key] = abs(raw_weight)
                    except (TypeError, ValueError):
                        log.warning(
                            f"Invalid weight for '{key}' in {candidate}; "
                            f"using the default ({default})"
                        )
                        validated[key] = default

                if negative_keys:
                    log.warning(
                        f"{candidate} gives a negative weight to "
                        f"{', '.join(sorted(negative_keys))}. Weights are now "
                        "importances: each metric's direction is handled by the "
                        "fold change itself, so the magnitude has been used. "
                        "Make them positive to silence this."
                    )

                # Silently ignoring these used to make a typo look like it had
                # worked; the objective simply never changed.
                unknown = sorted(set(loaded) - set(default_weights))
                if unknown:
                    log.warning(
                        f"Ignoring unrecognised weight key(s) in {candidate}: "
                        f"{', '.join(unknown)}. Known metrics are: "
                        f"{', '.join(sorted(default_weights))}."
                    )

                # The first candidate is the *current working directory*, which
                # hifimizer has already chdir'ed into the output directory, so
                # say out loud which file actually won.
                self.weights_source = str(candidate)
                log.debug(f"Loaded metric weights from {candidate}")
                return validated
            except Exception as e:
                log.warning(f"Failed to load weights from {candidate}: {e}")

        self.weights_source = "built-in defaults"
        log.debug("No weights.json found; using built-in default weights.")
        return default_weights

    def active_weights(self):
        """
        Weights restricted to the metrics this run can still actually produce.

        A metric whose stage is off -- switched off by the user, or retired
        after repeated failures -- must be dropped rather than left at its 0.0
        default. ``calculate_weighted_sum`` uses ``metrics.get(name, 0.0)``, so
        a missing negatively-weighted metric (``missing``, ``num_sv``,
        ``error_rate``, ...) would otherwise contribute a penalty of zero and
        make the broken trial look like the best one in the study.
        """
        weights = dict(self.weights)
        for stage in self.STAGES:
            if self.stage_enabled(stage.name):
                continue
            for metric in stage.metrics:
                weights.pop(metric, None)

        # A run produces either the unsuffixed split metrics or the
        # per-haplotype ones, never both. Leaving the unproduced names in here
        # would put metrics that can never appear into metric_regime and into
        # the single-objective metric list.
        split = list(self.HAPLOTYPE_SPLIT_METRICS)
        if self.n_haplotypes < 2:
            drop = [f"{m}_hap{i}" for m in split + ["kmer_completeness"]
                    for i in (1, 2)]
        else:
            drop = split
        for metric in drop:
            weights.pop(metric, None)

        # A metric the baseline never measured has nothing to be a fold change
        # against, so it cannot be scored. Dropping it here rather than at
        # scoring time keeps metric_regime honest about what the study is
        # actually optimising.
        if self.baseline_metrics:
            for metric in [m for m in weights if m not in self.baseline_metrics]:
                weights.pop(metric, None)

        # Likewise a metric that never moved during the burn-in has no scale to
        # be divided by, and scoring it would only add noise.
        if self.metric_scales:
            for metric in [m for m in weights if m not in self.metric_scales]:
                weights.pop(metric, None)
        return weights

    def weights_for(self, metrics):
        """
        The weights that actually apply to one evaluation's results.

        ``active_weights`` answers "what should this run be able to measure";
        this answers "what did this assembly actually yield". They differ when
        a stage failed only for *this* trial -- below the retirement threshold,
        or skipped because a prerequisite failed. Scoring on the intersection
        is what keeps a transient failure from handing the trial free points
        on every metric it was supposed to be penalised by.
        """
        return {
            name: weight
            for name, weight in self.active_weights().items()
            if name in metrics
        }

    def metric_regime(self, metrics=None) -> str:
        """
        Stable identifier for a scored-metric set.

        With ``metrics``, describes what a particular trial was scored on;
        without, what this run currently expects to be able to score on. Two
        trials are only score-comparable when their regimes match, which is
        why it is recorded as a trial attribute and consumed by
        ``find_best_trial``.
        """
        keys = self.active_weights() if metrics is None else self.weights_for(metrics)
        return ",".join(sorted(keys))

    # -------------------------------------------------------- fold changes
    #: Metrics where a larger raw value is better. Everything else is treated
    #: as "smaller is better", so these are inverted before the ratio.
    HIGHER_IS_BETTER = frozenset(
        {"n50", "n50_hap1", "n50_hap2", "single_copy", "hic_pairs_informative"}
    )

    @classmethod
    def badness(cls, name, raw):
        """
        Turn a raw measurement into a quantity where *smaller is better*.

        Every metric is then compared by the same ratio against the baseline,
        so nothing downstream needs to know which way a metric runs. Three
        families need more than a pseudocount:

        ``qv``
            Phred is already the logarithm of an error rate, so a ratio of QVs
            is a ratio of logarithms: 50 -> 45 reads as a 9% change when it is
            really 3.2x more errors. Converting back to the error rate makes a
            3.01 dB gain exactly one halving.
        ``kmer_completeness``
            Bounded at 100%, so the ratio of the percentages barely moves --
            99.0 -> 99.5 is a factor of 1.005. What actually halved is the
            *deficit*, and that is the quantity worth scoring.
        ``trans_hap_rate``
            Also a percentage, and legitimately zero on a perfectly phased
            assembly, so it takes the same floor as the deficit above.
        """
        if name.startswith("qv"):
            return 10.0 ** (-raw / 10.0)
        if name.startswith("kmer_completeness"):
            return max(100.0 - raw, FC_DEFICIT_FLOOR)
        if name == "trans_hap_rate":
            return max(raw, FC_DEFICIT_FLOOR)
        if name in cls.HIGHER_IS_BETTER:
            # 1/x turns "more is better" into "less is better" without needing
            # a sign anywhere downstream.
            return 1.0 / (raw + FC_PSEUDOCOUNT)
        return raw + FC_PSEUDOCOUNT

    @staticmethod
    def compute_metric_scales(observations, keys):
        """
        The typical size of each metric's fold change during the burn-in.

        ``observations`` is one ``{metric: log2FC}`` dict per burn-in trial,
        **excluding the baseline** -- its fold change against itself is zero by
        construction and would only deflate the scale.

        The statistic is the root mean square about *zero*, not the standard
        deviation about the burn-in mean, because the scores it will divide are
        themselves measured from the baseline rather than from the mean. If
        every random parameter set lands 4x worse than default but tightly
        clustered, the standard deviation is near zero and an entirely typical
        trial scores -28; the RMS is 2.0 and the same trial scores -1.0, which
        is the honest reading.

        A metric that never moved gets no scale, which drops it from the score
        for the rest of the study: it cannot separate one trial from another.
        """
        scales = {}
        for key in keys:
            values = [
                float(o[key])
                for o in observations
                if key in o and o[key] is not None and np.isfinite(o[key])
            ]
            if len(values) < SCALE_MIN_OBSERVATIONS:
                continue
            rms = float(np.sqrt(np.mean(np.square(values))))
            if not np.isfinite(rms) or rms <= SCALE_MIN:
                continue
            scales[key] = {"scale": rms, "n": len(values)}
        return scales

    def save_metric_scales(self, scales):
        """Persist the burn-in scales and adopt them for this evaluator."""
        self.metric_scales = dict(scales)
        self._write_json(self.paths.metric_scales, self.metric_scales, "metric scales")

    def standardised(self, name, stored_value):
        """
        The metric's fold change divided by how much it typically moves.

        Returns None during the burn-in, when there is no scale yet, so callers
        can fall back to the raw fold change.
        """
        fc = self.fold_change(name, stored_value)
        if fc is None:
            return None
        entry = self.metric_scales.get(name)
        if not entry:
            return None
        scale = float(entry.get("scale", 0.0))
        if not np.isfinite(scale) or scale <= SCALE_MIN:
            return None
        return float(np.clip(fc / scale, -Z_CLIP, Z_CLIP))

    @property
    def is_standardised(self) -> bool:
        """True once the burn-in has produced usable scales."""
        return bool(self.metric_scales)

    def save_baseline_metrics(self, metrics):
        """Record the default-parameter assembly and adopt it as the reference."""
        self.baseline_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if np.isfinite(float(v))
        }
        self._write_json(
            self.paths.baseline_metrics, self.baseline_metrics, "baseline metrics"
        )

    def fold_change(self, name, stored_value):
        """
        log2 fold change of one metric against the baseline, signed so that
        **positive always means better**.

        log2 rather than a plain ratio because a plain ratio is asymmetric: a
        doubling moves 1.0 away from "no change" while a halving moves only
        0.5, so summing plain ratios quietly rewards metrics that worsened more
        than it rewards ones that improved. Under log2 a doubling is +1 and a
        halving is -1, whichever direction the metric runs.

        Computed on *raw* values, never the stored log(v+1) ones: a ratio of
        logarithms is not a fold change. num_contigs going 1 -> 1000 is a ratio
        of 9.97 in stored units and a 500x collapse in reality.

        Returns None when the baseline never measured this metric, which is the
        signal that it cannot be scored (see :meth:`active_weights`).
        """
        base = self.baseline_metrics.get(name)
        if base is None:
            return None
        try:
            b_trial = self.badness(name, self.raw_value(name, stored_value))
            b_base = self.badness(name, self.raw_value(name, base))
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(b_trial) and np.isfinite(b_base)):
            return 0.0
        if b_trial <= 0 or b_base <= 0:
            return 0.0
        # badness is "smaller is better", so baseline/trial above 1 means this
        # trial improved on the baseline.
        lfc = float(np.log2(b_base / b_trial))
        if not np.isfinite(lfc):
            return 0.0
        return float(np.clip(lfc, -FC_CLIP, FC_CLIP))

    def scored_value(self, name, value):
        """
        What the weight multiplies.

        Before the burn-in has produced scales, the raw log2 fold change. After
        it, that fold change divided by how far the metric typically moves, so
        that a weight buys the same thing on a volatile metric as on a steady
        one.
        """
        fc = self.fold_change(name, value)
        if fc is None:
            return 0.0
        if not self.metric_scales:
            return fc
        z = self.standardised(name, value)
        return 0.0 if z is None else z

    @property
    def has_baseline(self) -> bool:
        """True once the default-parameter assembly has been measured."""
        return bool(self.baseline_metrics)

    # ---------------------------------------------------------------- scoring
    def calculate_weighted_sum(self, metrics):
        """
        Sum of ``importance x log2 fold change`` over every scored metric.

        The baseline assembly scores exactly 0 by construction, so the sign of
        a trial's score says directly whether it beat plain hifiasm.
        """
        return sum(
            weight * self.scored_value(name, metrics[name])
            for name, weight in self.weights_for(metrics).items()
        )

    def analyze_metric_contributions(self, metrics):
        """
        Per-metric breakdown of the score.

        ``raw_value`` is the measurement back-transformed out of log space so
        the log is readable (a contig N50 of 34 Mb is useful information,
        ``17.34`` is not); ``fc`` is its log2 fold change against the baseline,
        positive when the trial is better, and None for a metric the baseline
        never measured.
        """
        rows = {}
        total = 0.0

        for name, weight in self.weights_for(metrics).items():
            value = float(metrics[name])
            fc = self.fold_change(name, value)
            total += weight * self.scored_value(name, value)
            rows[name] = {
                "log_value": value,
                "raw_value": self.raw_value(name, value),
                "unit": self.METRIC_UNITS.get(name, ""),
                "weight": weight,
                "fc": fc,
                "z": self.standardised(name, value),
            }

        return {"total_score": total, "contributions": rows}

    # ------------------------------------------------------------- evaluation
    def evaluate_assembly(
        self,
        gfa_file,
        fasta_file,
        include_busco=None,
        busco_lineage="metazoa_odb12",
        extra_fasta_files=None,
        extra_gfa_files=None,
        gate=None,
    ):
        """
        Run the evaluation pipeline for one assembly.

        Every metric-producing tool runs inside :meth:`_run_stage`, which
        applies :attr:`failure_policy`:

        * **Baseline (trial 0).** A tool that crashes, exits non-zero or blows
          through its walltime is retired for the whole study; the baseline is
          still scored on what remains, and that reduced set becomes the metric
          set every later trial uses. The decision is written to disk, so it
          also survives a restart.
        * **Trial 1 onwards.** The same failure raises
          :class:`MetricStageFailure`. The stage stays enabled and the *trial*
          is discarded, because a score computed from a different metric set is
          not comparable with the rest of the study.
        * **Reports** (final assembly, ``--rerun-*``). Absorbed; the metric is
          simply missing from the summary.

        Unrecoverable in every mode: a missing GFA, a failure of an essential
        stage (gfastats), and every stage failing at once.

        Args:
            gfa_file: Primary GFA produced by hifiasm.
            fasta_file: Where to write the FASTA derived from ``gfa_file``.
            include_busco: Run BUSCO. Defaults to the constructor setting.
            busco_lineage: BUSCO lineage dataset name.
            extra_fasta_files: Additional haplotype FASTAs. When present the
                assembly is scored as a diploid: reads are aligned to both
                haplotypes, QV is averaged over them, and k-mer completeness
                is measured on their union.
            extra_gfa_files: The matching haplotype GFAs, so gfastats measures
                every haplotype rather than only the first.
            gate: Optional callable invoked with the gfastats metrics as soon
                as they exist, before anything expensive runs. Whatever it
                raises propagates untouched, which is how the caller aborts a
                trial whose contiguity has already collapsed without paying
                for alignment, yak and completeness first.

        Returns:
            dict of metrics (see the class docstring for the log convention).
        """
        if include_busco is not None:
            self.include_busco = include_busco

        gfa_file = Path(gfa_file)
        if not gfa_file.exists():
            raise FileNotFoundError(f"GFA file not found: {gfa_file}")

        # Not a stage: without a FASTA there is nothing any tool can look at.
        self.logger.info("Converting GFA to FASTA")
        self.convert_gfa_to_fasta(gfa_file, fasta_file)

        self.stage_outcomes = {}
        metrics = {}

        def absorb(ok, value):
            if not (ok and value):
                return
            # A non-finite measurement cannot be summed, averaged or
            # standardised, so it is dropped here rather than allowed to reach
            # the score. yak reports a QV of `inf` when it finds no k-mer
            # errors at all, which is a real result but not a usable number:
            # it made the burn-in mean `inf` and the spread `nan`, and every
            # trial afterwards scored `nan`.
            for name, v in value.items():
                try:
                    finite = np.isfinite(float(v))
                except (TypeError, ValueError):
                    finite = False
                if finite:
                    metrics[name] = v
                else:
                    self.logger.warning(
                        f"Metric '{name}' came back as {v!r}, which cannot be "
                        "scored; it is dropped for this assembly."
                    )

        absorb(
            *self._run_stage(
                "gfastats",
                lambda: self.run_gfastats(gfa_file, extra_gfa_files=extra_gfa_files),
            )
        )

        # Cheapest stage first, then the decision to keep going. gfastats costs
        # seconds; everything below it costs hours.
        if gate is not None and metrics:
            gate(metrics)

        # The combined haplotype FASTA exists for the phasing metric only: it
        # is the one measurement that needs to know which haplotype a contig
        # belongs to.
        try:
            diploid_fasta = self.build_diploid_fasta(fasta_file, extra_fasta_files)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                f"Could not build the combined haplotype FASTA ({e}); "
                "the Hi-C phasing metric will be unavailable for this trial."
            )
            diploid_fasta = None

        absorb(
            *self._run_stage(
                "yak",
                lambda: self.run_yak_qv(
                    fasta_file, extra_fasta_files=extra_fasta_files
                ),
            )
        )
        # Completeness runs last and is the memory spike that ends runs. Every
        # stage above it has finished with its buffers by now, so give those
        # pages back to the kernel before the spike rather than holding them
        # against it.
        self._release_memory("the alignment and k-mer stages")

        # Completeness stays on hap1 deliberately. Each haplotype should carry
        # the full gene set once, so the question "is the gene space complete
        # and single-copy" is already answered by one haplotype -- and running
        # the most expensive stage twice to average two nearly identical
        # numbers is not worth doubling the cost of every trial. Collapsed
        # phasing still shows up here, as multi_copy on hap1.
        absorb(
            *self._run_stage(
                "busco",
                lambda: self.run_completeness(fasta_file, lineage=busco_lineage),
            )
        )

        absorb(
            *self._run_stage(
                "hic_phasing", lambda: self.run_hic_phasing(diploid_fasta)
            )
        )

        succeeded = [n for n, ok in self.stage_outcomes.items() if ok]
        failed = [n for n, ok in self.stage_outcomes.items() if not ok]

        if not metrics:
            raise MetricStageFailure(
                "all",
                "every metric stage",
                f"none of {', '.join(failed) or 'the stages'} produced a value",
                reason="no metric stage produced a value for this assembly",
            )

        if failed:
            self.logger.warning(
                f"Evaluation completed with {len(succeeded)}/"
                f"{len(succeeded) + len(failed)} stages: "
                f"succeeded [{', '.join(succeeded)}], "
                f"failed or skipped [{', '.join(failed)}]."
            )

        # The baseline decides the metric set for the entire study, so if it
        # lost anything, say so once and loudly rather than leaving it to be
        # inferred from a per-stage warning fifty trials back in the log.
        if self.is_baseline:
            report = self.metric_health_report()
            if report:
                self.logger.error(report)
            if not self.active_weights():
                raise RuntimeError(
                    "The baseline assembly produced no scorable metrics at all "
                    "(every stage failed or is switched off). There is nothing "
                    "to optimise; check the logs in "
                    f"{self.paths.logs_dir} before re-running."
                )

        return metrics

    # ---------------------------------------------------------------- cleanup
    def cleanup_intermediate_files(self, trial_id=None):
        """
        Remove a trial's evaluation scratch directory.

        Everything a trial produces during evaluation lives in
        ``work/trials/trial_<id>/``, so cleanup is a single rmtree. The
        assembly itself (in ``work/hifiasm/``) is left alone, since hifiasm
        reuses its .bin files across trials.
        """
        tid = trial_id if trial_id is not None else self.trial_id
        target = self.paths.trials_dir / f"trial_{tid if tid is not None else 'main'}"
        try:
            if target.exists():
                shutil.rmtree(target)
                self.logger.info(f"Removed intermediate directory: {target}")
        except Exception as e:
            self.logger.warning(f"Cleanup failed for {target}: {e}")