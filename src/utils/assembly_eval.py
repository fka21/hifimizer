import os
import re
import json
import subprocess
import logging
import random
import gzip
import numpy as np
from pathlib import Path
from Bio import SeqIO
from utils.subprocess_logger import SubprocessLogger


class AssemblyEvaluator:
    """
    AssemblyEvaluator provides a unified interface to evaluate and optimize genome assemblies.

    It integrates:
    - Assembly evaluation with `gfastats`
    - Read-to-assembly alignment and evaluation using `gfalign`
    - Completeness scoring with `BUSCO`

    The metrics gathered from these steps are parsed and scored with predefined weights to guide optimization routines
    (e.g., Optuna) in search of parameter configurations that produce the best assemblies.

    Attributes:
        known_genome_size (int): Known genome size to compare against.
        input_reads (str): Path to the full input reads.
        subset_reads (str): Path to the subsetted reads file.
        aln_file (str): Alignment file produced by gfalign.
        db_path (str): Path to the SQLite database used for Optuna studies.
        db_uri (str): Full SQLite URI to the study database.
        threads (int): Number of threads to use in BUSCO and gfalign.
    """

    def __init__(
        self,
        known_genome_size,
        input_reads,
        trial_id=None,
        threads=None,
        download_path=None,
    ):
        self.known_genome_size = known_genome_size
        self.input_reads = input_reads
        self.number_reads = 1000
        self.subset_reads = "subset_reads.fa"
        self.aln_file = "alignment.sam"
        self.db_path = "optuna_study.db"
        self.db_uri = f"sqlite:///{self.db_path}"
        self.trial_id = trial_id
        self.threads = threads
        self.download_path = download_path
        self._compile_patterns()

        # Initialize subprocess logger
        self.subprocess_logger = SubprocessLogger()

        # Initialize main logger for this evaluator
        self.logger = logging.getLogger(f"AssemblyEval_{trial_id or 'main'}")

        # Compile regex patterns for parsing outputs
        self._compile_patterns()
        # Load metric weights from config (or use defaults)
        self.weights = self._load_weights()

    def _compile_patterns(self):
        """
        Pre-compile regular expression patterns for parsing output of evaluation tools.
        """
        self.gfastats_patterns = {
            "num_contigs": re.compile(r"# contigs:\s+(\d+)"),
            "length_diff": re.compile(r"Total contig length:\s+(\d+)"),
            "n50": re.compile(r"Contig N50:\s+(\d+)"),
            "l50": re.compile(r"Contig L50:\s+(\d+)"),
        }

        self.stats_patterns = {
            "reads_mapped": re.compile(r"reads mapped:\s+(\d+)"),
            "bases_mapped": re.compile(r"bases mapped \(cigar\):\s+(\d+)"),
            "error_rate": re.compile(
                r"error rate:\s+([0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?)"
            ),
        }

        self.sniffles_patterns = {
            "num_sv": re.compile(r"Total SVs:\s+(\d+)"),
        }

    @staticmethod
    def run_command(self, command, command_name="command"):
        """
        Run a command with logging - updated to use subprocess logger.
        """
        try:
            return_code, log_path = self.subprocess_logger.run_command_with_logging(
                command=command,
                log_filename=f"{command_name}.log",
                command_name=command_name,
                trial_id=self.trial_id,
            )

            if return_code != 0:
                self.logger.error(
                    f"{command_name} failed (return code: {return_code}). See log: {log_path}"
                )
                raise RuntimeError(f"{command_name} failed - see {log_path}")

            # For compatibility, return stdout from log file
            with open(log_path, "r") as f:
                content = f.read()

            return content, "", return_code

        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise

    def download_busco(self, lineage="metazoa_odb12"):
        """
        Download the BUSCO lineage dataset if not already present.

        Args:
            lineage (str): BUSCO lineage name.
        """
        busco_dir = Path(f"busco_downloads/{lineage}")
        if busco_dir.exists():
            self.logger.info("BUSCO dataset already downloaded. Skipping download.")
            return
        try:
            command = f"busco --download {lineage}"
            return self.run_command(
                self, command=command, command_name="busco_download"
            )
        except Exception as e:
            self.logger.error(f"BUSCO download failed: {e}")
            raise

    def run_gfastats(self, gfa_file):
        """
        Run gfastats on the given GFA file.

        Args:
            gfa_file (str): Path to GFA file.

        Returns:
            dict: Parsed gfastats metrics.
        """
        command = f"gfastats --discover-paths {gfa_file}"
        try:
            stdout, _, _ = self.run_command(self, command, "gfastats")
            return self.parse_gfastats_output(stdout)
        except RuntimeError:
            self.logger.error("Gfastats analysis failed")
            raise

    def read_subsetting(self, num_reads=None):
        """
        Subsample reads from the input FASTA/FASTQ file and write to subset_reads.

        Args:
            num_reads (int): Number of reads to sample.
        """
        # Initialize number of reads if not provided, and get input file name
        fname = self.input_reads.name
        if num_reads is None:
            num_reads = self.number_reads

        # Determine file format
        if fname.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz")):
            fmt = "fastq"
        elif fname.endswith((".fasta", ".fa", ".fasta.gz", ".fa.gz")):
            fmt = "fasta"
        else:
            raise ValueError(
                f"Input file {fname} is not in a recognized FASTA or FASTQ format."
            )

        open_func = gzip.open if fname.endswith(".gz") else open

        # Read and sample records
        with open_func(self.input_reads, "rt") as handle:
            records = list(SeqIO.parse(handle, fmt))
        sampled = random.sample(records, min(num_reads, len(records)))

        # Write sampled reads
        output_is_gz = str(self.subset_reads).endswith(".gz")
        open_func_out = gzip.open if output_is_gz else open

        with open_func_out(self.subset_reads, "wt") as out_handle:
            SeqIO.write(sampled, out_handle, fmt)

    @staticmethod
    def convert_gfa_to_fasta(gfa_file, output_fasta):
        """
        Convert GFA file to FASTA format.

        Args:
            gfa_file (_type_): _description_
            output_fasta (_type_): _description_

        Returns:
            _type_: _description_
        """
        command = ["awk", '$1 == "S" {print ">"$2"\\n"$3}', gfa_file]
        with open(output_fasta, "w") as out_file:
            subprocess.run(command, stdout=out_file, check=True)
        return True

    def run_busco(
        self, fasta_file, lineage="metazoa_odb12", mode="genome", download_path=None
    ):
        """
        Run BUSCO on the given FASTA assembly.

        Args:
            fasta_file (str): Path to the FASTA file.
            lineage (str): BUSCO lineage dataset.
            mode (str): BUSCO mode ('genome', 'proteins', etc.).
            download_path (str, optional): Custom path to BUSCO datasets.

        Returns:
            dict: Parsed BUSCO metrics.
        """
        output_dir = f"busco_output_{os.path.basename(fasta_file).split('.')[0]}"

        command = (
            f"busco -i {fasta_file} -l {lineage} -m {mode} -o {output_dir} "
            f"-c {self.threads} --metaeuk --skip_bbtools --force"
        )
        if download_path:
            command += f" --download_path {download_path}"

        try:
            self.run_command(self, command, "busco")
            # Locate the BUSCO short summary JSON file produced in the output directory.
            out_dir_path = Path(output_dir)
            # BUSCO may include different suffixes in the short_summary filename
            # depending on the provided -o value; use a glob to find the JSON.
            json_pattern = f"short_summary.specific.{lineage}.*.json"
            matches = list(out_dir_path.glob(json_pattern))
            if not matches:
                # fallback to a more generic pattern
                matches = list(out_dir_path.glob("short_summary.*.json"))
            if not matches:
                raise FileNotFoundError(
                    f"BUSCO summary JSON not found in {output_dir} (pattern {json_pattern})"
                )
            busco_json_file = str(matches[0])
            return self.parse_busco_results(busco_json_file)
        except RuntimeError:
            self.logger.error("BUSCO evaluation failed")
            raise

    def parse_gfastats_output(self, output):
        """
        Parse gfastats output and extract relevant metrics.

        Args:
            output (str): Raw gfastats stdout output.

        Returns:
            dict: Dictionary of log-transformed gfastats metrics.
        """
        metrics = {}
        for key, pattern in self.gfastats_patterns.items():
            match = re.search(pattern, output)
            if match:
                value = int(match.group(1))
                if key == "length_diff":
                    metrics[key] = np.log10(
                        abs(value - self.known_genome_size) / 1_000_000
                    )  # Convert to Mbp
                elif key == "n50":
                    metrics[key] = np.log10(value / 1_000_000)  # Convert to Mbp
                else:
                    metrics[key] = np.log10(value)
        return metrics

    def run_minimap2_align(self, fasta_file, reads_file, sam_file, threads=None):
        """
        Align reads to the assembly with minimap2.

        Args:
            fasta_file (str): Path to the FASTA assembly.
            reads_file (str): Path to the reads (FASTQ).
            sam_file (str): Desired output SAM file.
            threads (int, optional): Number of CPU threads. If not provided, use self.threads.

        Raises:
            RuntimeError: If minimap2 exits with a non-zero code.
        """
        threads = threads or self.threads
        command = f"minimap2 -t {threads} -ax map-hifi -o {sam_file} {fasta_file} {reads_file}"

        try:
            self.run_command(self, command, "minimap2")
            return sam_file
        except RuntimeError:
            self.logger.error("Minimap2 alignment failed")
            raise

    def parse_samtools_stats(self, sam_file):
        """
        Parse samtools stats output to extract average alignment length and mapping quality.

        Args:
            sam_file (str): Path to the aligned SAM file.

        Prints:
            avg_alignment_length, avg_mapping_quality
        """
        command = f"samtools stats {sam_file}"
        stdout, _, _ = self.run_command(self, command, "samtools_stats")

        stats = {}
        for key, pattern in self.stats_patterns.items():
            match = pattern.search(stdout)
            if match:
                try:
                    if key == "bases_mapped":
                        # Convert to float for average length
                        value = float(match.group(1))
                        stats[key] = np.log10(value / 1_000_000)
                    else:
                        value = float(match.group(1))
                        stats[key] = np.log10(value + 1)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Value conversion failed for {key}: {e}")
                    stats[key] = 0
            else:
                self.logger.warning(f"Pattern not found for {key} in samtools output")
                stats[key] = 0

        return stats

    def run_sniffles2(self, sam_file, vcf_file=None):
        """
        Run sniffles2 on the aligned SAM file to detect structural variants.

        Args:
            sam_file (str): Path to the aligned SAM file.
            vcf_file (str, optional): Output VCF file. Defaults to sniffles_output.vcf.

        Returns:
            dict: Parsed sniffles2 metrics.
        """
        if vcf_file is None:
            vcf_file = "sniffles_output.vcf"

        command = f"sniffles -i {sam_file} -v {vcf_file}"

        try:
            self.run_command(self, command, "sniffles2")
            return self.parse_sniffles_vcf(vcf_file)
        except RuntimeError:
            self.logger.error("Sniffles2 analysis failed")
            raise

    def parse_sniffles_vcf(self, vcf_file):
        """
        Parse sniffles2 VCF output to extract structural variant metrics.

        Args:
            vcf_file (str): Path to sniffles VCF file.

        Returns:
            dict: Dictionary of log-transformed sniffles metrics.
        """
        metrics = {
            "num_sv": 0,
        }

        try:
            with open(vcf_file, "r") as f:
                sv_count = 0
                for line in f:
                    if line.startswith("#"):
                        continue
                    sv_count += 1
                
                # Log-transform the count
                metrics["num_sv"] = np.log10(sv_count + 1)
                
        except FileNotFoundError:
            self.logger.warning(f"Sniffles VCF file not found: {vcf_file}")
            metrics["num_sv"] = 0
        except Exception as e:
            self.logger.warning(f"Failed to parse sniffles VCF: {e}")
            metrics["num_sv"] = 0

        return metrics

    @staticmethod
    def parse_busco_results(busco_json_file):
        """
        Parse BUSCO results from JSON file.

        Args:
            busco_json_file (str): Path to BUSCO JSON summary.

        Returns:
            dict: Dictionary of log-transformed BUSCO metrics.
        """
        with open(busco_json_file, "r") as f:
            data = json.load(f)

        metrics = {
            "single_copy": np.log10(data["results"]["Single copy BUSCOs"] + 1),
            "multi_copy": np.log10(data["results"]["Multi copy BUSCOs"] + 1),
            "fragmented": np.log10(data["results"]["Fragmented BUSCOs"] + 1),
            "missing": np.log10(data["results"]["Missing BUSCOs"] + 1),
        }
        return metrics

    def _load_weights(self):
        """Load metric weights from a JSON config file, falling back to defaults.

        Search order:
        - ./weights.json (current working dir)
        - repository root weights.json (two levels up from this file)
        """
        default_weights = {
            "num_contigs": -0.8,
            "length_diff": -1,
            "n50": 1,
            "l50": -0.4,
            "single_copy": 1,
            "multi_copy": -0.7,
            "fragmented": -0.7,
            "missing": -1,
            "reads_mapped": 0.8,
            "bases_mapped": 0.8,
            "error_rate": -1,
            "num_sv": -0.5,
        }

        # Candidate locations for user-editable config
        candidates = [
            Path.cwd() / "weights.json",
            Path(__file__).resolve().parents[2] / "weights.json",
        ]

        for p in candidates:
            try:
                if p.exists():
                    with open(p, "r") as fh:
                        loaded = json.load(fh)
                    # Validate and coerce numeric values
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
        return default_weights

    def calculate_weighted_sum(self, metrics):
        """
        Calculate weighted sum of metrics using loaded weights.

        Args:
            metrics (dict): Dictionary of metric scores.

        Returns:
            float: Weighted sum.
        """
        # Weighted-sum scoring removed — multicriteria optimization only.
        raise RuntimeError(
            "Weighted-sum scoring is disabled; use multicriteria optimization."
        )

    def evaluate_assembly(
        self,
        gfa_file,
        fasta_file,
        include_busco=True,
        busco_lineage="metazoa_odb12",
        download_path=None,
    ):
        """
        Perform a full evaluation pipeline for a given assembly.

        Args:
            gfa_file (str): Path to the input GFA file.
            fasta_file (str): Path to the output FASTA file.
            include_busco (bool): Whether to run BUSCO.

        Returns:
            float: Weighted score of assembly quality.
        """
        if not Path(gfa_file).exists():
            raise FileNotFoundError(f"GFA file not found: {gfa_file}")

        try:
            # Stage 1: GFA to FASTA conversion
            self.logger.info("Converting GFA to FASTA")
            self.convert_gfa_to_fasta(gfa_file, fasta_file)

            # Stage 2: Read alignment
            self.logger.info("Running minimap2 alignment")
            aln_file = self.run_minimap2_align(
                fasta_file, self.subset_reads, self.aln_file, threads=self.threads
            )

            # Stage 3: Parse alignment stats
            self.logger.info("Parsing alignment statistics")
            metrics_minimap2 = self.parse_samtools_stats(aln_file)

            # Stage 4: Assembly statistics
            self.logger.info("Running gfastats")
            metrics_gfastats = self.run_gfastats(gfa_file)

            combined_metrics = {**metrics_gfastats, **metrics_minimap2}

            # Stage 5: Structural variant detection with sniffles2
            self.logger.info("Running sniffles2 for structural variant detection")
            metrics_sniffles = self.run_sniffles2(aln_file)
            combined_metrics.update(metrics_sniffles)

            # Stage 6: BUSCO evaluation (optional)
            if include_busco:
                self.logger.info("Running BUSCO evaluation")
                metrics_busco = self.run_busco(
                    fasta_file, lineage=busco_lineage, download_path=self.download_path
                )
                combined_metrics.update(metrics_busco)

            # Return raw metrics dict for multicriteria optimization
            return combined_metrics

        except Exception as e:
            stage_info = self._get_current_stage(e)
            self.logger.error(f"Assembly evaluation failed at stage: {stage_info}")

    def _get_current_stage(self, error):
        """Determine which stage failed based on error type/message."""
        error_str = str(error).lower()
        if "gfa" in error_str or "convert" in error_str:
            return "GFA to FASTA conversion"
        elif "minimap2" in error_str or "alignment" in error_str:
            return "Read alignment"
        elif "samtools" in error_str or "stats" in error_str:
            return "Alignment statistics parsing"
        elif "gfastats" in error_str:
            return "Assembly statistics"
        elif "sniffles" in error_str or "sv" in error_str:
            return "Structural variant detection"
        elif "busco" in error_str:
            return "BUSCO evaluation"
        else:
            return "Unknown stage"
