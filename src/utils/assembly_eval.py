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
        logs_dir=None,
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

        # Initialize subprocess logger with provided logs_dir
        self.subprocess_logger = SubprocessLogger(
            logs_dir=logs_dir if logs_dir else Path.cwd() / "logs"
        )

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
        }

        self.stats_patterns = {
            "reads_mapped": re.compile(r"reads mapped:\s+(\d+)"),
            "error_rate": re.compile(
                r"error rate:\s+([0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?)"
            ),
            "supplementary_alignments": re.compile(
                r"supplementary alignments:\s+(\d+)"
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
            dict: Dictionary of raw gfastats metrics.
        """
        metrics = {}
        for key, pattern in self.gfastats_patterns.items():
            match = re.search(pattern, output)
            if match:
                value = int(match.group(1))
                if key == "length_diff":
                    # Return raw difference in megabases
                    metrics[key] = np.log(
                        (abs(value - self.known_genome_size) / 1_000_000) + 1
                    )
                elif key == "n50":
                    # Return raw N50 value
                    metrics[key] = np.log(value + 1)
                else:
                    # Return raw value
                    metrics[key] = np.log(value + 1)
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

    def convert_sam_to_bam(self, sam_file, bam_file=None, threads=None):
        """
        Convert SAM file to BAM format and sort it.

        Args:
            sam_file (str): Path to the input SAM file.
            bam_file (str, optional): Output BAM file. Defaults to input with .bam extension.
            threads (int, optional): Number of CPU threads. If not provided, use self.threads.

        Returns:
            str: Path to the output BAM file.
        """
        threads = threads or self.threads
        if bam_file is None:
            bam_file = sam_file.replace(".sam", ".bam")

        # Convert SAM to BAM
        command = f"samtools view -b -h -o {bam_file} {sam_file}"
        try:
            self.run_command(self, command, "samtools_view")
        except RuntimeError:
            self.logger.error("SAM to BAM conversion failed")
            raise

        return bam_file

    def sort_bam(self, bam_file, sorted_bam_file=None, threads=None):
        """
        Sort a BAM file.

        Args:
            bam_file (str): Path to the input BAM file.
            sorted_bam_file (str, optional): Output sorted BAM file. Defaults to input with .sorted.bam.
            threads (int, optional): Number of CPU threads. If not provided, use self.threads.

        Returns:
            str: Path to the sorted BAM file.
        """
        threads = threads or self.threads
        if sorted_bam_file is None:
            sorted_bam_file = bam_file.replace(".bam", ".sorted.bam")

        command = f"samtools sort -@ {threads} -o {sorted_bam_file} {bam_file}"
        try:
            self.run_command(self, command, "samtools_sort")
        except RuntimeError:
            self.logger.error("BAM sorting failed")
            raise

        return sorted_bam_file

    def index_bam(self, bam_file):
        """
        Index a sorted BAM file to create a .bai file.

        Args:
            bam_file (str): Path to the sorted BAM file.

        Returns:
            str: Path to the BAM index file.
        """
        bam_index_file = f"{bam_file}.bai"
        command = f"samtools index {bam_file}"
        try:
            self.run_command(self, command, "samtools_index")
        except RuntimeError:
            self.logger.error("BAM indexing failed")
            raise

        return bam_index_file

    def parse_samtools_stats(self, sam_file):
        """
        Parse samtools stats output to extract average alignment length and mapping quality.

        Args:
            sam_file (str): Path to the aligned SAM file.

        Returns:
            dict: Dictionary of raw alignment statistics.
        """
        command = f"samtools stats {sam_file}"
        stdout, _, _ = self.run_command(self, command, "samtools_stats")

        stats = {}
        for key, pattern in self.stats_patterns.items():
            match = pattern.search(stdout)
            if match:
                try:
                    value = float(match.group(1))
                    # Return raw values
                    stats[key] = np.log(value + 1)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Value conversion failed for {key}: {e}")
                    stats[key] = 0
            else:
                self.logger.warning(f"Pattern not found for {key} in samtools output")
                stats[key] = 0

        return stats

    def run_sniffles2(self, bam_file, vcf_file=None):
        """
        Run sniffles2 on the sorted BAM file to detect structural variants.

        Args:
            bam_file (str): Path to the sorted BAM file.
            vcf_file (str, optional): Output VCF file. Defaults to sniffles_output.vcf.

        Returns:
            dict: Parsed sniffles2 metrics.
        """
        if vcf_file is None:
            # Use trial ID if available for unique filenames
            trial_suffix = (
                f"trial_{self.trial_id}" if self.trial_id is not None else "default"
            )
            vcf_file = f"sniffles_output_{trial_suffix}.vcf"

        command = f"sniffles -i {bam_file} -v {vcf_file} --allow-overwrite"

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
            dict: Dictionary of raw sniffles metrics.
        """
        metrics = {
            "num_sv": 0,
        }

        try:
            if not os.path.exists(vcf_file):
                self.logger.warning(f"Sniffles VCF file not found: {vcf_file}")
                return metrics

            with open(vcf_file, "r") as f:
                sv_count = 0
                for line in f:
                    # Skip header lines (start with # or ##)
                    if line.startswith("#"):
                        continue
                    # Count non-empty lines that aren't comments
                    if line.strip():
                        sv_count += 1

                # Return raw count
                metrics["num_sv"] = np.log(sv_count + 1)
                self.logger.debug(f"Detected {sv_count} structural variants")

        except Exception as e:
            self.logger.warning(f"Failed to parse sniffles VCF {vcf_file}: {e}")
            metrics["num_sv"] = 0

        return metrics

    @staticmethod
    def parse_busco_results(busco_json_file):
        """
        Parse BUSCO results from JSON file.

        Args:
            busco_json_file (str): Path to BUSCO JSON summary.

        Returns:
            dict: Dictionary of raw BUSCO metrics.
        """
        with open(busco_json_file, "r") as f:
            data = json.load(f)

        metrics = {
            "single_copy": np.log(data["results"]["Single copy BUSCOs"] + 1),
            "multi_copy": np.log(data["results"]["Multi copy BUSCOs"] + 1),
            "fragmented": np.log(data["results"]["Fragmented BUSCOs"] + 1),
            "missing": np.log(data["results"]["Missing BUSCOs"] + 1),
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
            "single_copy": 1,
            "multi_copy": -0.7,
            "fragmented": -0.7,
            "missing": -1,
            "reads_mapped": 0.8,
            "error_rate": -1,
            "num_sv": -0.5,
            "supplementary_alignments": -0.6,
        }

        # Candidate locations for user-editable config
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
        Calculate weighted sum of metrics using log-transformed values directly.

        Args:
            metrics (dict): Dictionary of log-transformed metric scores (from parsing stage).

        Returns:
            float: Weighted sum of log-transformed metrics (sum of weight × log_value).
        """
        weighted_sum = 0.0
        for metric_name, weight in self.weights.items():
            log_value = metrics.get(metric_name, 0.0)
            weighted_sum += weight * log_value
        return weighted_sum

    def analyze_metric_contributions(self, metrics):
        """
        Analyze the contribution of each metric to the weighted score.

        Returns detailed breakdown of how each metric contributes to the final score.

        Args:
            metrics (dict): Already log-transformed metric values (from evaluation parsing).

        Returns:
            dict: Detailed contribution analysis including log-transformed values, weights,
                  contributions, and proportions.
        """
        contributions = {}
        weighted_sum = 0.0

        # Calculate contribution of each metric
        for metric_name, weight in self.weights.items():
            log_value = float(metrics.get(metric_name, 0.0))
            contribution = weight * log_value
            weighted_sum += contribution

            contributions[metric_name] = {
                "log_value": log_value,
                "weight": weight,
                "contribution": contribution,
            }

        # Calculate proportions (percentage of total positive contributions)
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

        for metric_name, data in contributions.items():
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

            # Stage 4: Convert SAM to sorted BAM for sniffles2
            self.logger.info("Converting SAM to sorted BAM and indexing")
            bam_file = self.convert_sam_to_bam(aln_file)
            sorted_bam_file = self.sort_bam(bam_file)
            self.index_bam(sorted_bam_file)

            # Stage 5: Assembly statistics
            self.logger.info("Running gfastats")
            metrics_gfastats = self.run_gfastats(gfa_file)

            combined_metrics = {**metrics_gfastats, **metrics_minimap2}

            # Stage 6: Structural variant detection with sniffles2
            self.logger.info("Running sniffles2 for structural variant detection")
            metrics_sniffles = self.run_sniffles2(sorted_bam_file)
            combined_metrics.update(metrics_sniffles)

            # Stage 7: BUSCO evaluation (optional)
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

    def cleanup_intermediate_files(self, trial_id=None):
        """
        Clean up intermediate files generated during assembly evaluation.
        Removes SAM, unsorted BAM files, and other temporary files.

        Args:
            trial_id (int, optional): Trial ID for prefix matching.
        """
        try:
            import glob

            # List of patterns for intermediate files to remove
            patterns_to_remove = [
                "*.sam",  # SAM alignment files
                "*.bam",  # Unsorted BAM files (keep only sorted)
                "subset_reads.*",  # Subsetted reads (can be regenerated)
                "sniffles_output_*.vcf",  # Trial-specific sniffles VCF files
                "busco_output_trial_assembly/*",  # Trial BUSCO outputs
            ]

            removed_count = 0
            for pattern in patterns_to_remove:
                for filepath in glob.glob(pattern):
                    try:
                        # Don't remove sorted BAM files
                        if filepath.endswith(".sorted.bam") or filepath.endswith(
                            ".sorted.bam.bai"
                        ):
                            continue

                        if os.path.isfile(filepath):
                            os.remove(filepath)
                            removed_count += 1
                            self.logger.debug(f"Removed intermediate file: {filepath}")
                        elif os.path.isdir(filepath):
                            import shutil

                            shutil.rmtree(filepath)
                            removed_count += 1
                            self.logger.debug(
                                f"Removed intermediate directory: {filepath}"
                            )
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {filepath}: {e}")

            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} intermediate files")

        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

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
