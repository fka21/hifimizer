import os
import re
import json
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
from Bio import SeqIO

from utils.subprocess_logger import SubprocessLogger, TIMEOUT_EXIT_CODE
from utils.paths import RunPaths


class BuscoFailedError(RuntimeError):
    """Raised when every BUSCO gene-prediction backend fails or times out."""


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
    - Read-to-assembly alignment with `minimap2` + `samtools sort/index`
    - Alignment-based quality metrics with `samtools stats`
    - Structural-variant counting with `sniffles2`, reusing that alignment
    - Gene-space completeness with `BUSCO`
    - k-mer completeness and consensus QV with `yak`

    All intermediate artefacts are written beneath ``paths.work_dir``; nothing
    is written relative to the current working directory.

    Metric conventions
    ------------------
    Every metric except those listed in :attr:`RAW_METRICS` is stored
    log-transformed as ``log(value + 1)``.  ``qv`` (already a Phred-scaled,
    i.e. logarithmic, quantity), ``kmer_completeness`` (a bounded percentage)
    and the rate-like ``samtools stats`` values are stored raw:
    log-transforming them would compress their variance to the point of
    invisibility next to ``n50``.

    Use :meth:`raw_value` to undo the transform for display; the optimiser
    always consumes the stored (log) values.
    """

    #: metrics that are NOT log-transformed
    RAW_METRICS = frozenset(
        {
            "qv",
            "kmer_completeness",
            # samtools stats: rates / averages, meaningless under log(v+1)
            "error_rate",
            "mapped_rate",
            "average_read_length",
            "average_quality",
        }
    )

    #: units used when echoing raw values into the log
    METRIC_UNITS = {
        "length_diff": "Mb",
        "n50": "bp",
        "qv": "Phred",
        "kmer_completeness": "%",
        "error_rate": "per kb",
        "mapped_rate": "%",
        "average_read_length": "bp",
        "bases_mapped": "bp",
    }

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
            metrics=("num_contigs", "length_diff", "n50"),
            walltime_flag="--gfastats-walltime",
            # Contig count, length and N50 are the backbone of the objective.
            # Continuing without them would be optimising nothing.
            essential=True,
        ),
        MetricStage(
            name="alignment",
            label="read alignment (minimap2 | samtools sort)",
            walltime_flag="--align-walltime",
        ),
        MetricStage(
            name="samtools_stats",
            label="alignment statistics (samtools stats)",
            metrics=(
                "reads_mapped",
                "supplementary_alignments",
                "error_rate",
                "reads_total",
                "reads_unmapped",
                "reads_mq0",
                "bases_mapped",
                "mismatches",
                "mapped_rate",
                "average_read_length",
                "average_quality",
            ),
            requires=("alignment",),
            walltime_flag="--samtools-stats-walltime",
        ),
        MetricStage(
            name="sniffles",
            label="structural variants (sniffles2)",
            metrics=("num_sv",),
            requires=("alignment",),
            walltime_flag="--sniffles-walltime",
        ),
        MetricStage(
            name="yak",
            label="k-mer QV and completeness (yak)",
            metrics=("qv", "kmer_completeness"),
            walltime_flag="--yak-walltime",
        ),
        MetricStage(
            name="busco",
            label="gene-space completeness (BUSCO)",
            metrics=("single_copy", "multi_copy", "fragmented", "missing"),
            walltime_flag="--busco-walltime",
        ),
    )

    STAGES_BY_NAME = {stage.name: stage for stage in STAGES}

    #: default per-stage wall-clock budgets, in hours (CLI overrides these)
    DEFAULT_STAGE_WALLTIMES = {
        "gfastats": 0.5,
        "alignment": 6.0,
        "samtools_stats": 1.0,
        "sniffles": 2.0,
        "yak": 2.0,
        "busco": 6.0,
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
            include_busco: Enable the BUSCO completeness metrics.
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
        #: per-evaluation outcome, ``{stage_name: bool}``; reset by
        #: :meth:`evaluate_assembly`
        self.stage_outcomes = {}

    # ------------------------------------------------------------------ misc
    @property
    def subset_reads(self) -> Path:
        return self.paths.subset_reads

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
        return False

    def stage_enabled(self, name) -> bool:
        """
        True if the stage can still contribute metrics.

        Covers four reasons a stage may be off: the user disabled it
        (``--no-busco`` / ``--no-kmer-eval``), it failed often enough to be
        retired, a stage it *depends on* was retired, or it is not a real
        stage name. The dependency case matters: ``alignment`` produces no
        metrics of its own, so losing it would otherwise leave samtools stats
        and sniffles nominally "enabled" while producing nothing.
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

        # `samtools stats` summary-number lines look like:
        #     SN\treads mapped:\t98213\t# comment
        # The value may be an integer, a float, or scientific notation
        # ("error rate:\t2.383865e-03").
        self.samtools_sn = re.compile(
            r"^SN\t([^:]+):\t([-+0-9.eE]+)", re.MULTILINE
        )

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
        """Download the BUSCO lineage dataset into the work/ tree if absent."""
        if not self.stage_enabled("busco"):
            self.logger.info(
                "BUSCO is disabled for this run; skipping the lineage download."
            )
            return None

        lineage_dir = self.download_path / "lineages" / lineage
        if lineage_dir.exists():
            self.logger.info(
                f"BUSCO lineage '{lineage}' already present in {self.download_path}. "
                "Skipping download."
            )
            return

        self.download_path.mkdir(parents=True, exist_ok=True)
        command = f"busco --download_path {self.download_path} --download {lineage}"
        try:
            return self.run_command(command, command_name="busco_download")
        except Exception as e:
            self.logger.error(f"BUSCO download failed: {e}")
            raise

    def read_subsetting(self, num_reads):
        """
        Subsample a **fixed number** of reads (``num_reads``, i.e.
        ``--num-reads``) from the input file into ``work/reads/subset_reads.fa``.

        This is a fixed *count*, not a coverage target: the resulting depth
        depends on the read-length distribution and the genome size. The
        implied coverage is computed from the sampled bases and logged below
        so that ``--num-reads`` can be tuned against it.

        The subset is always written as FASTA regardless of input format: the
        downstream consumers (minimap2, sniffles) do not use base qualities,
        and a fixed filename keeps every trial pointing at the same file.
        """
        fname = self.input_reads.name

        if fname.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz")):
            fmt = "fastq"
        elif fname.endswith((".fasta", ".fa", ".fasta.gz", ".fa.gz")):
            fmt = "fasta"
        else:
            raise ValueError(
                f"Input file {fname} is not in a recognized FASTA or FASTQ format."
            )

        open_func = gzip.open if fname.endswith(".gz") else open

        with open_func(self.input_reads, "rt") as handle:
            records = list(SeqIO.parse(handle, fmt))
        sampled = random.sample(records, min(num_reads, len(records)))

        self.subset_reads.parent.mkdir(parents=True, exist_ok=True)
        with open(self.subset_reads, "wt") as out_handle:
            SeqIO.write(sampled, out_handle, "fasta")

        sampled_bases = sum(len(r.seq) for r in sampled)
        mean_len = sampled_bases / len(sampled) if sampled else 0
        coverage = (
            sampled_bases / self.known_genome_size if self.known_genome_size else 0
        )

        self.logger.info(
            f"Read subsetting: fixed count of {num_reads} requested "
            f"(--num-reads), {len(sampled)} of {len(records)} available reads "
            f"written to {self.subset_reads}"
        )
        self.logger.info(
            f"  Sampled bases      : {sampled_bases / 1e6:.1f} Mb "
            f"(mean read length {mean_len:,.0f} bp)"
        )
        self.logger.info(
            f"  Implied coverage   : ~{coverage:.1f}x of a "
            f"{self.known_genome_size / 1e6:.0f} Mb haploid genome"
        )

        if coverage < 10:
            self.logger.warning(
                f"The read subset gives only ~{coverage:.1f}x coverage. The "
                "alignment-based metrics (samtools stats, sniffles) become "
                "noisy below ~10x; raise --num-reads if the per-trial scores "
                "look unstable."
            )

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

        # --- completeness on the full (diploid) assembly ---
        extra_fasta_files = [f for f in (extra_fasta_files or []) if Path(f).exists()]
        if extra_fasta_files:
            combined = self._combine_fastas(
                [fasta_file] + list(extra_fasta_files), tdir / "combined_haps.fasta"
            )
            comb_out = tdir / "yak_qv.combined.txt"
            command = (
                f"yak qv -t{self.threads} -K{chunk * 2} "
                f"{self.paths.reads_yak} {combined} > {comb_out}"
            )
            self.run_command(
                command,
                command_name="yak_qv_combined",
                timeout_seconds=self.stage_walltime_seconds("yak"),
            )
            combined_stats = self.parse_yak_qv(comb_out)
            primary["kmer_completeness"] = combined_stats["kmer_completeness"]
            try:
                combined.unlink()
            except Exception:
                pass

        return primary

    # ---------------------------------------------------------------- gfastats
    def run_gfastats(self, gfa_file):
        command = f"gfastats --discover-paths {gfa_file}"
        try:
            stdout = self.run_command(
                command,
                "gfastats",
                timeout_seconds=self.stage_walltime_seconds("gfastats"),
            )
            return self.parse_gfastats_output(stdout)
        except (RuntimeError, TimeoutError):
            self.logger.error("gfastats analysis failed")
            raise

    def parse_gfastats_output(self, output):
        metrics = {}
        for key, pattern in self.gfastats_patterns.items():
            match = re.search(pattern, output)
            if match:
                value = int(match.group(1))
                if key == "length_diff":
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

    # ------------------------------------------------------------------ BUSCO
    #: attempted in order; the first that succeeds is cached for later trials
    BUSCO_BACKENDS = ("miniprot", "metaeuk", "augustus")

    def run_busco(self, fasta_file, lineage="metazoa_odb12", mode="genome"):
        """
        Run BUSCO, trying gene-prediction backends in order of increasing cost.

        Miniprot has been the default for eukaryotic genome mode since BUSCO
        v5.7.0 and is typically minutes rather than hours; metaeuk and augustus
        are only reached if it fails.  Whichever backend succeeds is recorded
        in the backend cache so subsequent trials do not re-pay for the
        failures.

        Each attempt is bounded by ``--busco-walltime``, enforced by
        ``SubprocessLogger`` killing the whole process group.  GNU ``timeout``
        is deliberately *not* used: it signals only the direct child, leaving
        BUSCO's metaeuk/augustus/hmmsearch grandchildren running.
        """
        tdir = self.trial_dir
        out_name = "busco_output"
        out_dir = tdir / out_name

        # BUSCO's -o must be a bare name; the location is set with --out_path.
        base_cmd = (
            f"busco -i {fasta_file} -l {lineage} -m {mode} "
            f"-o {out_name} --out_path {tdir} "
            f"-c {self.threads} --skip_bbtools --force "
            f"--download_path {self.download_path} --offline"
        )

        timeout_seconds = self.stage_walltime_seconds("busco")

        cached = self.backend_cache.get("busco")
        if cached in self.BUSCO_BACKENDS:
            order = [cached] + [b for b in self.BUSCO_BACKENDS if b != cached]
        else:
            order = list(self.BUSCO_BACKENDS)

        last_error = None
        backend_used = None
        for backend in order:
            cmd = f"{base_cmd} --{backend}"
            try:
                self.logger.info(
                    f"Running BUSCO with {backend} "
                    f"(walltime: {self.stage_walltimes.get('busco')} h)"
                )
                self.run_command(cmd, f"busco_{backend}", timeout_seconds=timeout_seconds)
                backend_used = backend
                break
            except (RuntimeError, TimeoutError) as e:
                last_error = e
                self.logger.warning(
                    f"BUSCO/{backend} failed or exceeded its walltime: {e}"
                )

        if backend_used is None:
            raise BuscoFailedError(
                f"BUSCO failed or exceeded the {self.stage_walltimes.get('busco')} h walltime "
                f"with every backend ({', '.join(order)}). This usually means a "
                "gene-prediction step is hanging or broken in this environment. "
                "Re-run with --no-busco to proceed without completeness scoring. "
                f"Last error: {last_error}"
            )

        if self.backend_cache.get("busco") != backend_used:
            self.backend_cache["busco"] = backend_used
            self._save_backend_cache()

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

    # ------------------------------------------------------- read alignment
    @property
    def minimap2_preset(self) -> str:
        return "map-ont" if self.ont else "map-hifi"

    def align_reads(self, fasta_file, force=False):
        """
        Map the read subset back onto the assembly and produce a sorted,
        indexed BAM.

            minimap2 -a -x <preset> --secondary=no -t N asm.fa reads.fa
              | samtools sort -@ N -o aln.bam -
            samtools index aln.bam

        ``--secondary=no`` suppresses secondary alignments but keeps
        *supplementary* ones, which are the split-read signal we score with
        (and which sniffles needs). One alignment per trial serves both
        ``samtools stats`` and sniffles.

        Returns:
            Path to the sorted BAM.
        """
        tdir = self.trial_dir
        bam = tdir / "reads_to_assembly.bam"

        if bam.exists() and not force:
            return bam

        if not Path(self.subset_reads).exists():
            raise FileNotFoundError(
                f"Read subset not found at {self.subset_reads}; "
                "read_subsetting() must run before alignment."
            )

        timeout_seconds = self.stage_walltime_seconds("alignment")

        # `set -o pipefail` so a minimap2 failure is not masked by a
        # successful samtools sort of an empty stream.
        command = (
            f"set -o pipefail; "
            f"minimap2 -a -x {self.minimap2_preset} --secondary=no "
            f"-t {self.threads} {fasta_file} {self.subset_reads} "
            f"| samtools sort -@ {self.threads} -o {bam} - "
            f"&& samtools index -@ {self.threads} {bam}"
        )

        self.logger.info(
            f"Aligning read subset to the assembly (minimap2 -x "
            f"{self.minimap2_preset})"
        )
        try:
            self.run_command(
                command, "minimap2_sort", timeout_seconds=timeout_seconds
            )
        except (RuntimeError, TimeoutError):
            self.logger.error("Read alignment (minimap2 | samtools sort) failed")
            raise

        if not bam.exists():
            raise FileNotFoundError(f"Alignment finished but {bam} was not written")

        return bam

    # -------------------------------------------------------- samtools stats
    def run_samtools_stats(self, bam_file):
        """
        Alignment-based quality metrics from ``samtools stats``.

        Three of the summary numbers feed the score (see ``weights.json``):

        ``reads_mapped``
            How many of the subset reads placed on the assembly at all. The
            subset size is constant across trials, so the raw count is
            directly comparable; missing sequence shows up here first.
        ``error_rate``
            ``mismatches / bases mapped (cigar)`` -- samtools' own per-base
            divergence between reads and assembly. A consensus-accuracy proxy
            that, unlike a raw mismatch count, is independent of how much
            sequence got mapped.
        ``supplementary_alignments``
            Split reads: one read placed in two pieces. The clearest cheap
            signal for chimeric joins and local misassembly, and the metric
            CRAQ's clip-based CRE/CSE was ultimately derived from.

        Everything else parsed here is recorded for the log and the trial
        attributes but carries no weight.

        Returns:
            dict of metrics. Counts are stored log-transformed; the rate-like
            values listed in :attr:`RAW_METRICS` are stored raw.
        """
        stats_out = self.trial_dir / "samtools_stats.txt"
        command = f"samtools stats -@ {self.threads} {bam_file} > {stats_out}"

        try:
            self.run_command(
                command,
                "samtools_stats",
                timeout_seconds=self.stage_walltime_seconds("samtools_stats"),
            )
        except (RuntimeError, TimeoutError):
            self.logger.error("samtools stats failed")
            raise

        with open(stats_out, "r") as fh:
            return self.parse_samtools_stats(fh.read())

    def parse_samtools_stats(self, output):
        """
        Parse the ``SN`` block of ``samtools stats`` output.

        Fields consumed (samtools >= 1.10 names them all):
            sequences, reads mapped, reads unmapped, supplementary alignments,
            reads MQ0, bases mapped (cigar), mismatches, error rate,
            average length, average quality
        """
        raw = {}
        for match in self.samtools_sn.finditer(output):
            key = match.group(1).strip()
            try:
                raw[key] = float(match.group(2))
            except ValueError:
                continue

        if not raw:
            self.logger.warning(
                "samtools stats produced no parsable SN block; alignment "
                "metrics unavailable for this trial."
            )
            return {}

        total = raw.get("sequences", 0.0)
        mapped = raw.get("reads mapped", 0.0)
        unmapped = raw.get("reads unmapped", 0.0)
        supplementary = raw.get("supplementary alignments", 0.0)
        # samtools reports the error rate as a fraction of aligned bases; per
        # kb keeps it on a scale where a sensible weight is O(1) rather than
        # O(1000).
        error_rate_per_kb = raw.get("error rate", 0.0) * 1000.0

        metrics = {
            # --- scored ---------------------------------------------------
            "reads_mapped": np.log(mapped + 1),
            "supplementary_alignments": np.log(supplementary + 1),
            "error_rate": error_rate_per_kb,
            # --- reported only --------------------------------------------
            "reads_total": np.log(total + 1),
            "reads_unmapped": np.log(unmapped + 1),
            "reads_mq0": np.log(raw.get("reads MQ0", 0.0) + 1),
            "bases_mapped": np.log(raw.get("bases mapped (cigar)", 0.0) + 1),
            "mismatches": np.log(raw.get("mismatches", 0.0) + 1),
            "mapped_rate": (mapped / total * 100.0) if total > 0 else 0.0,
            "average_read_length": raw.get("average length", 0.0),
            "average_quality": raw.get("average quality", 0.0),
        }

        if total > 0 and (mapped / total) < 0.8:
            self.logger.warning(
                f"Only {mapped / total * 100:.1f}% of the subset reads mapped back "
                "to this assembly. That usually means the assembly is badly "
                "fragmented or a large fraction of the genome is missing."
            )

        return metrics

    # --------------------------------------------------------------- sniffles
    def run_sniffles2(self, bam_file, vcf_file=None):
        """Call SVs from the sorted, indexed BAM built by :meth:`align_reads`."""
        if vcf_file is None:
            vcf_file = self.trial_dir / "sniffles_output.vcf"

        command = f"sniffles -i {bam_file} -v {vcf_file} --allow-overwrite"
        try:
            self.run_command(
                command,
                "sniffles2",
                timeout_seconds=self.stage_walltime_seconds("sniffles"),
            )
            return self.parse_sniffles_vcf(vcf_file)
        except (RuntimeError, TimeoutError):
            self.logger.error("sniffles2 analysis failed")
            raise

    def parse_sniffles_vcf(self, vcf_file):
        metrics = {"num_sv": 0}
        try:
            if not os.path.exists(vcf_file):
                self.logger.warning(f"Sniffles VCF file not found: {vcf_file}")
                return metrics

            with open(vcf_file, "r") as f:
                sv_count = sum(
                    1 for line in f if line.strip() and not line.startswith("#")
                )

            metrics["num_sv"] = np.log(sv_count + 1)
            self.logger.debug(f"Detected {sv_count} structural variants")
        except Exception as e:
            self.logger.warning(f"Failed to parse sniffles VCF {vcf_file}: {e}")
            metrics["num_sv"] = 0
        return metrics

    # ---------------------------------------------------------------- scoring
    def _load_weights(self):
        """Load metric weights from weights.json, falling back to defaults."""
        default_weights = {
            "num_contigs": -0.8,
            "length_diff": -1,
            "n50": 1,
            "single_copy": 1,
            "multi_copy": -0.7,
            "fragmented": -0.7,
            "missing": -1,
            "num_sv": -0.5,
            # samtools stats. reads_mapped and supplementary_alignments are
            # log-scale counts, so they sit naturally alongside n50.
            "reads_mapped": 0.8,
            "supplementary_alignments": -0.7,
            # error_rate is raw, in mismatches per kb of aligned sequence
            # (typically 1-5 for HiFi, higher for ONT).
            "error_rate": -0.6,
            # Raw (non-log) metrics. QV is Phred (~40-60) and completeness is
            # a percentage (~95-100), so the weights are deliberately small to
            # keep their contributions on the same order as log-scale n50.
            "qv": 0.1,
            "kmer_completeness": 0.1,
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
                for key, default in default_weights.items():
                    if key not in loaded:
                        validated[key] = default
                        continue
                    try:
                        validated[key] = float(loaded[key])
                    except (TypeError, ValueError):
                        log.warning(
                            f"Invalid weight for '{key}' in {candidate}; "
                            f"using the default ({default})"
                        )
                        validated[key] = default

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

    def calculate_weighted_sum(self, metrics):
        return sum(
            weight * float(metrics[name])
            for name, weight in self.weights_for(metrics).items()
        )

    def analyze_metric_contributions(self, metrics):
        """
        Break the weighted score down per metric.

        ``log_value`` is what the optimiser actually multiplies by the weight;
        ``raw_value`` is the same number back-transformed out of log space and
        exists purely so the log lines are readable (a contig N50 of 34 Mb is
        useful information, ``17.34`` is not).
        """
        contributions = {}
        weighted_sum = 0.0

        for metric_name, weight in self.weights_for(metrics).items():
            value = float(metrics[metric_name])
            contribution = weight * value
            weighted_sum += contribution
            contributions[metric_name] = {
                "log_value": value,
                "raw_value": self.raw_value(metric_name, value),
                "unit": self.METRIC_UNITS.get(metric_name, ""),
                "weight": weight,
                "contribution": contribution,
            }

        positive_contributions = sum(
            c["contribution"] for c in contributions.values() if c["contribution"] > 0
        )
        negative_contributions = abs(
            sum(
                c["contribution"]
                for c in contributions.values()
                if c["contribution"] < 0
            )
        )

        for data in contributions.values():
            if positive_contributions > 0 and data["contribution"] > 0:
                data["proportion"] = (
                    data["contribution"] / positive_contributions
                ) * 100
            elif negative_contributions > 0 and data["contribution"] < 0:
                data["proportion"] = (
                    abs(data["contribution"]) / negative_contributions
                ) * 100
            else:
                data["proportion"] = 0.0

        return {
            "total_score": weighted_sum,
            "positive_sum": positive_contributions,
            "negative_sum": negative_contributions,
            "contributions": contributions,
        }

    # ------------------------------------------------------------- evaluation
    def evaluate_assembly(
        self,
        gfa_file,
        fasta_file,
        include_busco=None,
        busco_lineage="metazoa_odb12",
        extra_fasta_files=None,
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
            extra_fasta_files: Additional haplotype FASTAs, used only to make
                the k-mer completeness estimate reflect the whole diploid
                assembly rather than one haplotype.

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
            if ok and value:
                metrics.update(value)

        absorb(*self._run_stage("gfastats", lambda: self.run_gfastats(gfa_file)))

        # One alignment, two consumers: samtools stats and sniffles2.
        aligned, bam = self._run_stage(
            "alignment", lambda: self.align_reads(fasta_file)
        )

        if aligned:
            absorb(
                *self._run_stage(
                    "samtools_stats", lambda: self.run_samtools_stats(bam)
                )
            )
            absorb(
                *self._run_stage("sniffles", lambda: self.run_sniffles2(bam))
            )
        else:
            # Mark the dependants so _run_stage reports the real reason.
            for dependent in ("samtools_stats", "sniffles"):
                self._run_stage(dependent, lambda: None)

        absorb(
            *self._run_stage(
                "yak",
                lambda: self.run_yak_qv(
                    fasta_file, extra_fasta_files=extra_fasta_files
                ),
            )
        )
        absorb(
            *self._run_stage(
                "busco",
                lambda: self.run_busco(fasta_file, lineage=busco_lineage),
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