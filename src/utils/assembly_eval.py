import os
import re
import json
import shutil
import subprocess
import logging
import gzip
import random
import numpy as np
from pathlib import Path
from Bio import SeqIO

from utils.subprocess_logger import SubprocessLogger, TIMEOUT_EXIT_CODE
from utils.paths import RunPaths


class BuscoFailedError(RuntimeError):
    """Raised when every BUSCO gene-prediction backend fails or times out."""


class AssemblyEvaluator:
    """
    AssemblyEvaluator provides a unified interface to evaluate genome assemblies.

    It integrates:
    - Assembly statistics with `gfastats`
    - Reference-free error detection with `CRAQ` (clip-based CRE/CSE, AQI)
    - Structural-variant counting with `sniffles2`, reusing CRAQ's alignment
    - Gene-space completeness with `BUSCO`
    - k-mer completeness and consensus QV with `yak`

    All intermediate artefacts are written beneath ``paths.work_dir``; nothing
    is written relative to the current working directory.

    Metric conventions
    ------------------
    Every metric except those listed in :attr:`RAW_METRICS` is stored
    log-transformed as ``log(value + 1)``.  ``qv`` (already a Phred-scaled,
    i.e. logarithmic, quantity) and ``kmer_completeness`` (a bounded
    percentage) are stored raw: log-transforming them would compress their
    variance to the point of invisibility next to ``n50``.
    """

    #: metrics that are NOT log-transformed
    RAW_METRICS = frozenset(
        {
            "qv",
            "kmer_completeness",
            "aqi",
            "r_aqi",
            "s_aqi",
            "cre_per_mb",
            "cse_per_mb",
            "craq_covered_rate",
            "craq_low_conf_rate",
        }
    )

    def __init__(
        self,
        known_genome_size,
        input_reads,
        paths: RunPaths,
        trial_id=None,
        threads=None,
        download_path=None,
        ont=False,
        busco_walltime_hours=6.0,
        craq_walltime_hours=6.0,
        craq_mapq=20,
        kmer_eval=True,
        yak_k=31,
        yak_bloom_bits=37,
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
            ont: Input reads are ONT (selects the CRAQ/minimap2 preset).
            busco_walltime_hours: Wall-clock budget per BUSCO backend attempt.
            craq_walltime_hours: Wall-clock budget for the CRAQ run.
            craq_mapq: Minimum mapping quality passed to CRAQ (-q).
            kmer_eval: Enable the yak QV / k-mer completeness metrics.
        """
        self.known_genome_size = known_genome_size
        self.input_reads = Path(input_reads)
        self.paths = paths
        self.trial_id = trial_id
        self.threads = threads
        self.ont = ont
        self.busco_walltime_hours = busco_walltime_hours
        self.craq_walltime_hours = craq_walltime_hours
        self.craq_mapq = craq_mapq
        self.kmer_eval = kmer_eval
        self.yak_k = yak_k
        self.yak_bloom_bits = yak_bloom_bits

        # BUSCO datasets: user override, else our own work/ subdirectory.
        self.download_path = (
            Path(download_path).resolve()
            if download_path
            else paths.busco_downloads_dir
        )

        self.subprocess_logger = SubprocessLogger(logs_dir=paths.logs_dir)
        self.logger = logging.getLogger(f"AssemblyEval_{trial_id or 'main'}")

        self._compile_patterns()
        self.weights = self._load_weights()

        # Cache of which backend (aligner, BUSCO gene predictor) actually works
        # in this environment, so a failing one is only paid for once.
        self.cache_path = paths.busco_backend_cache
        self.backend_cache = self._load_backend_cache()

    # ------------------------------------------------------------------ misc
    @property
    def subset_reads(self) -> Path:
        return self.paths.subset_reads

    @property
    def trial_dir(self) -> Path:
        return self.paths.trial_dir(self.trial_id)

    def _load_backend_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    return json.load(f)
            except Exception:
                self.logger.warning(
                    "Failed to load BUSCO backend cache, starting fresh"
                )
        return {}

    def _save_backend_cache(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.backend_cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save backend cache: {e}")

    def _compile_patterns(self):
        """Pre-compile regexes for parsing the output of the evaluation tools."""
        self.gfastats_patterns = {
            "num_contigs": re.compile(r"# contigs:\s+(\d+)"),
            "length_diff": re.compile(r"Total contig length:\s+(\d+)"),
            "n50": re.compile(r"Contig N50:\s+(\d+)"),
        }

        # CRAQ's short report puts the per-metric AQI in parentheses, e.g.
        # "0.312(94.221)" for "Avg.CRE(R-AQI)".
        self.craq_value_paren = re.compile(r"^([-+0-9.eE]+)\(([-+0-9.eE]+)\)$")

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
            Tuple of (log_file_contents, "", return_code)
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
                content = f.read()

            return content, "", return_code

        except (RuntimeError, TimeoutError):
            raise
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise

    # ------------------------------------------------------------------ setup
    def download_busco(self, lineage="metazoa_odb12"):
        """Download the BUSCO lineage dataset into the work/ tree if absent."""
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
        Subsample ``num_reads`` reads from the input file into
        ``work/reads/subset_reads.fa``.

        The subset is always written as FASTA regardless of input format: the
        downstream consumers (CRAQ/minimap2, sniffles) do not use base
        qualities, and a fixed filename keeps every trial pointing at the same
        file.
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

        self.logger.info(
            f"Wrote {len(sampled)} subsampled reads to {self.subset_reads}"
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
            self.run_command(command, command_name="yak_count")
        except Exception as e:
            self.logger.error(
                f"yak count failed: {e}. k-mer metrics will be unavailable; "
                "re-run with --no-kmer-eval to silence this."
            )
            self.kmer_eval = False
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
        self.run_command(command, command_name="yak_qv")
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
            self.run_command(command, command_name="yak_qv_combined")
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
            stdout, _, _ = self.run_command(command, "gfastats")
            return self.parse_gfastats_output(stdout)
        except RuntimeError:
            self.logger.error("Gfastats analysis failed")
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

        Each attempt is bounded by ``busco_walltime_hours``, enforced by
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
            f"--download_path {self.download_path}"
        )

        timeout_seconds = (
            self.busco_walltime_hours * 3600 if self.busco_walltime_hours else None
        )

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
                    f"(walltime: {self.busco_walltime_hours} h)"
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
                f"BUSCO failed or exceeded the {self.busco_walltime_hours} h walltime "
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

    # ------------------------------------------------------------------ CRAQ
    def run_craq(self, fasta_file):
        """
        Reference-free error detection with CRAQ (Clipping Reveals Assembly Quality).

        CRAQ maps the reads back to the assembly with minimap2 and converts the
        clipped-alignment signal into error coordinates and quality indices:

            craq -g asm.fa -sms reads.fa -x <preset> -t <threads> -D <outdir>

        Only the SMS (long-read) arm is used -- we have no NGS short reads --
        so `-ngs` is omitted. Per CRAQ's own documentation, running without NGS
        data means fewer CREs and CRHs are detected (more so for ONT-based
        assemblies), while CSE/CSH detection, which is what the long reads are
        good for, is unaffected.

        Why AQI rather than a raw count: R-AQI and S-AQI are *normalised*
        (errors per unit length, mapped onto 0-100), so they degrade gracefully
        as coverage drops instead of collapsing toward zero the way a raw
        variant count does. CRAQ's own quality bands: >90 reference, 80-90
        high, 60-80 draft, <60 low.

        Returns:
            dict of CRAQ metrics (all raw, i.e. not log-transformed).
        """
        preset = "map-ont" if self.ont else "map-hifi"

        # CRAQ creates its -D output directory itself and aborts if it already
        # exists ("cannot create directory ... already exists, Exit !"). We must
        # therefore NOT pre-create it, and must clear any stale copy left by an
        # earlier attempt at this trial (a re-run, a retried prune, etc.). The
        # parent (self.trial_dir) is created on access; only the craq/ leaf is
        # handed to CRAQ, and it must be absent.
        craq_dir = self.trial_dir / "craq"
        if craq_dir.exists():
            shutil.rmtree(craq_dir)

        command = (
            f"craq -g {fasta_file} -sms {self.subset_reads} "
            f"-x {preset} -q {self.craq_mapq} -t {self.threads} "
            f"-pl F -D {craq_dir}"
        )

        timeout_seconds = (
            self.craq_walltime_hours * 3600 if self.craq_walltime_hours else None
        )

        try:
            self.run_command(command, "craq", timeout_seconds=timeout_seconds)
        except (RuntimeError, TimeoutError):
            self.logger.error("CRAQ analysis failed")
            raise

        reports = list(craq_dir.glob("**/runAQI_out/out_final.Report"))
        if not reports:
            reports = list(craq_dir.glob("**/out_final.Report"))
        if not reports:
            raise FileNotFoundError(f"CRAQ report not found under {craq_dir}")

        return self.parse_craq_report(reports[0])

    def craq_bam(self):
        """
        Path to the sorted, indexed long-read BAM that CRAQ already produced.

        CRAQ writes ``LRout/LR_sort.bam`` (+ .bai) as a by-product of its own
        minimap2 run, so sniffles can consume it directly instead of us paying
        for a second alignment of the same reads against the same assembly.
        """
        candidates = list((self.trial_dir / "craq").glob("**/LRout/LR_sort.bam"))
        return candidates[0] if candidates else None

    def parse_craq_report(self, report_file):
        """
        Parse ``runAQI_out/out_final.Report``.

        The file is a short report with one row per sequence plus a whole-
        assembly row labelled ``Genome``. Columns (see
        ``src/format_results_addAQI.pl`` in JiaoLaboratory/CRAQ):

            #Chr  Covered.Rate  Low-conf.Rate  Avg.CRH  Avg.CSH
                  Avg.CRE(R-AQI)  Avg.CSE(S-AQI)  AQI

        AQI is the harmonic mean of R-AQI and S-AQI.
        """
        metrics = {}

        row = None
        with open(report_file, "r") as fh:
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if fields and fields[0] == "Genome":
                    row = fields
                    break

        if row is None or len(row) < 7:
            self.logger.warning(
                f"No whole-assembly 'Genome' row found in {report_file}; "
                "CRAQ metrics unavailable for this trial."
            )
            return metrics

        def _split_paren(text):
            m = self.craq_value_paren.match(text.strip())
            if not m:
                return 0.0, 0.0
            return float(m.group(1)), float(m.group(2))

        try:
            covered = float(row[1])
            low_conf = float(row[2])
            cre, r_aqi = _split_paren(row[5])
            cse, s_aqi = _split_paren(row[6])
        except (ValueError, IndexError) as e:
            self.logger.warning(f"Failed to parse CRAQ report {report_file}: {e}")
            return metrics

        if len(row) >= 8:
            try:
                aqi = float(row[7])
            except ValueError:
                aqi = 0.0
        else:
            aqi = 0.0

        # Older CRAQ builds omit the final AQI column; recompute it.
        if aqi <= 0 and (r_aqi + s_aqi) > 0:
            aqi = 2 * r_aqi * s_aqi / (r_aqi + s_aqi)

        metrics = {
            "aqi": aqi,
            "r_aqi": r_aqi,
            "s_aqi": s_aqi,
            "cre_per_mb": cre,
            "cse_per_mb": cse,
            "craq_covered_rate": covered,
            "craq_low_conf_rate": low_conf,
        }

        if low_conf > 0.5:
            self.logger.warning(
                f"CRAQ flagged {low_conf * 100:.1f}% of the assembly as low-confidence. "
                "This almost always means the read subset is too shallow; consider "
                "raising --num-reads so CRAQ sees at least ~10x coverage."
            )

        return metrics

    # --------------------------------------------------------------- sniffles
    def run_sniffles2(self, bam_file, vcf_file=None):
        """Call SVs from CRAQ's already-sorted, already-indexed long-read BAM."""
        if vcf_file is None:
            vcf_file = self.trial_dir / "sniffles_output.vcf"

        command = f"sniffles -i {bam_file} -v {vcf_file} --allow-overwrite"
        try:
            self.run_command(command, "sniffles2")
            return self.parse_sniffles_vcf(vcf_file)
        except RuntimeError:
            self.logger.error("Sniffles2 analysis failed")
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
            # Raw (non-log) metrics. QV is Phred (~40-60), completeness is a
            # percentage (~95-100) and AQI is 0-100, so the weights are
            # deliberately small to keep their contributions on the same order
            # as log-scale n50.
            "qv": 0.1,
            "kmer_completeness": 0.1,
            # CRAQ's overall AQI (harmonic mean of R-AQI and S-AQI). r_aqi and
            # s_aqi are recorded as trial attributes but deliberately left out
            # of the weighted sum: they are collinear with aqi by construction.
            "aqi": 0.1,
        }

        candidates = [
            Path.cwd() / "weights.json",
            Path(__file__).parent / "weights.json",
            Path(__file__).resolve().parents[2] / "weights.json",
        ]

        for p in candidates:
            try:
                if p.exists():
                    with open(p, "r") as fh:
                        loaded = json.load(fh)
                    validated = {}
                    for k, v in default_weights.items():
                        if k in loaded:
                            try:
                                validated[k] = float(loaded[k])
                            except Exception:
                                logging.getLogger("AssemblyEval").warning(
                                    f"Invalid weight for {k} in {p}; using default"
                                )
                                validated[k] = v
                        else:
                            validated[k] = v
                    return validated
            except Exception as e:
                logging.getLogger("AssemblyEval").warning(
                    f"Failed to load weights from {p}: {e}"
                )

        return default_weights

    #: subset of RAW_METRICS produced by yak, dropped when --no-kmer-eval
    YAK_METRICS = frozenset({"qv", "kmer_completeness"})

    def active_weights(self):
        """Weights restricted to the metrics this run will actually produce."""
        weights = dict(self.weights)
        if not self.kmer_eval:
            for k in self.YAK_METRICS:
                weights.pop(k, None)
        return weights

    def calculate_weighted_sum(self, metrics):
        weighted_sum = 0.0
        for metric_name, weight in self.active_weights().items():
            weighted_sum += weight * metrics.get(metric_name, 0.0)
        return weighted_sum

    def analyze_metric_contributions(self, metrics):
        contributions = {}
        weighted_sum = 0.0

        for metric_name, weight in self.active_weights().items():
            value = float(metrics.get(metric_name, 0.0))
            contribution = weight * value
            weighted_sum += contribution
            contributions[metric_name] = {
                "log_value": value,
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
        include_busco=True,
        busco_lineage="metazoa_odb12",
        extra_fasta_files=None,
    ):
        """
        Run the full evaluation pipeline for one assembly.

        Args:
            gfa_file: Primary GFA produced by hifiasm.
            fasta_file: Where to write the FASTA derived from ``gfa_file``.
            include_busco: Run BUSCO.
            busco_lineage: BUSCO lineage dataset name.
            extra_fasta_files: Additional haplotype FASTAs, used only to make
                the k-mer completeness estimate reflect the whole diploid
                assembly rather than one haplotype.

        Returns:
            dict of metrics (see the class docstring for the log convention).
        """
        gfa_file = Path(gfa_file)
        if not gfa_file.exists():
            raise FileNotFoundError(f"GFA file not found: {gfa_file}")

        tdir = self.trial_dir

        try:
            self.logger.info("Converting GFA to FASTA")
            self.convert_gfa_to_fasta(gfa_file, fasta_file)

            self.logger.info("Running CRAQ for reference-free error detection")
            metrics_craq = self.run_craq(fasta_file)

            self.logger.info("Running gfastats")
            metrics_gfastats = self.run_gfastats(gfa_file)

            combined_metrics = {**metrics_gfastats, **metrics_craq}

            # Reuse the sorted+indexed BAM CRAQ already built rather than
            # aligning the same reads to the same assembly a second time.
            sorted_bam_file = self.craq_bam()
            if sorted_bam_file is None:
                self.logger.warning(
                    "CRAQ did not leave an LRout/LR_sort.bam behind; "
                    "skipping sniffles2 for this trial."
                )
            else:
                self.logger.info("Running sniffles2 for structural variant detection")
                combined_metrics.update(self.run_sniffles2(sorted_bam_file))

            if self.kmer_eval:
                self.logger.debug("Running yak for QV and k-mer completeness")
                combined_metrics.update(
                    self.run_yak_qv(fasta_file, extra_fasta_files=extra_fasta_files)
                )

            if include_busco:
                self.logger.info("Running BUSCO evaluation")
                combined_metrics.update(
                    self.run_busco(fasta_file, lineage=busco_lineage)
                )

            return combined_metrics

        except Exception as e:
            stage_info = self._get_current_stage(e)
            self.logger.error(f"Assembly evaluation failed at stage: {stage_info}")
            raise

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

    def _get_current_stage(self, error):
        error_str = str(error).lower()
        if "gfastats" in error_str:
            return "Assembly statistics"
        elif "gfa" in error_str or "convert" in error_str:
            return "GFA to FASTA conversion"
        elif "craq" in error_str or "aqi" in error_str:
            return "CRAQ error detection"
        elif "sniffles" in error_str:
            return "Structural variant detection"
        elif "yak" in error_str:
            return "k-mer evaluation"
        elif "busco" in error_str:
            return "BUSCO evaluation"
        else:
            return "Unknown stage"