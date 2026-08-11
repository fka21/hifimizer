# utils/paths.py
"""
Centralised definition of the hifimizer directory layout.

Everything the user is meant to keep lives directly under ``output_dir``.
Everything that is a means to an end lives under ``output_dir/work`` and can
be deleted without losing a result.

    output_dir/
    ├── default_assembly/          final: trial-0 (default hifiasm parameters)
    ├── final_assembly/            final: assembly built from the best parameters
    ├── optuna_output/             final: interactive plots
    ├── logs/                      final: main.log + per-command logs
    ├── optuna_study.db            final: the study
    ├── best_params_checkpoint.json
    └── work/                      intermediate, safe to delete
        ├── hifiasm/               shared hifiasm prefix -> .bin files are reused
        ├── reads/                 subset_reads.fa
        ├── kmers/                 reads.yak
        ├── busco_downloads/       lineage datasets
        ├── cache/                 busco_backend_cache.json,
        │                          metric_stage_state.json
        └── trials/trial_<id>/     sam, bam, vcf, busco output, yak qv txt

Note on the shared ``work/hifiasm`` directory: every trial uses the same
hifiasm ``-o`` prefix (``trial_assembly``) so that hifiasm re-uses the
``*.ec.bin`` / ``*.ovlp.*.bin`` files it wrote on the first trial.  Giving each
trial its own prefix would force a full error-correction and all-vs-all overlap
pass per trial, which is by far the most expensive part of the run.  The
per-trial *evaluation* artefacts do get their own directory, under
``work/trials/``.
"""

from pathlib import Path


class RunPaths:
    """Resolve and create every directory hifimizer writes to."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir).resolve()

        # --- final results -------------------------------------------------
        self.logs_dir = self.output_dir / "logs"
        self.optuna_dir = self.output_dir / "optuna_output"
        self.default_assembly_dir = self.output_dir / "default_assembly"
        self.final_assembly_dir = self.output_dir / "final_assembly"

        # --- intermediates -------------------------------------------------
        self.work_dir = self.output_dir / "work"
        self.hifiasm_dir = self.work_dir / "hifiasm"
        self.reads_dir = self.work_dir / "reads"
        self.kmers_dir = self.work_dir / "kmers"
        self.busco_downloads_dir = self.work_dir / "busco_downloads"
        self.cache_dir = self.work_dir / "cache"
        self.trials_dir = self.work_dir / "trials"

    # ---------------------------------------------------------------- files
    @property
    def db_path(self) -> Path:
        return self.output_dir / "optuna_study.db"

    @property
    def db_uri(self) -> str:
        return f"sqlite:///{self.db_path}"

    @property
    def best_params_checkpoint(self) -> Path:
        return self.output_dir / "best_params_checkpoint.json"

    @property
    def subset_reads(self) -> Path:
        return self.reads_dir / "subset_reads.fa"

    @property
    def reads_yak(self) -> Path:
        return self.kmers_dir / "reads.yak"

    @property
    def busco_backend_cache(self) -> Path:
        return self.cache_dir / "busco_backend_cache.json"

    @property
    def metric_stage_state(self) -> Path:
        """
        Cross-trial record of which metric stages have failed / been retired.

        Lives under ``work/cache`` so that ``--force-rerun`` (which wipes
        ``work/``) also clears it: a fresh run should retry every stage.
        """
        return self.cache_dir / "metric_stage_state.json"

    @property
    def hifiasm_prefix(self) -> Path:
        """Shared hifiasm ``-o`` prefix; keeps .bin reuse across trials."""
        return self.hifiasm_dir / "trial_assembly"

    # ----------------------------------------------------------- directories
    def trial_dir(self, trial_id) -> Path:
        """Per-trial evaluation scratch directory (created on demand)."""
        d = self.trials_dir / f"trial_{trial_id if trial_id is not None else 'main'}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create(self) -> "RunPaths":
        for d in (
            self.logs_dir,
            self.optuna_dir,
            self.work_dir,
            self.hifiasm_dir,
            self.reads_dir,
            self.kmers_dir,
            self.busco_downloads_dir,
            self.cache_dir,
            self.trials_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def __repr__(self) -> str:
        return f"RunPaths(output_dir={self.output_dir}, work_dir={self.work_dir})"