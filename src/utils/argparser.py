# args_parser.py
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Optimize hifiasm assembled de novo genomes with Optuna. It enables various parameter optimizations for hifiasm assembly, including parameters associated with Hi-C and ultra-long reads. By default it optimizes the parameters: x, y, s, n, m, p. If sensitive mode is enabled, it also optimizes D, N, and max_kocc parameters. The script can also run hifiasm with default settings, Hi-C reads, and ultra-long reads. It also supports primary assembly only mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required Inputs
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--genome-size",
        type=int,
        required=True,
        help="Haploid genome size in Mb (e.g., 300 for 300Mb)",
    )
    required.add_argument(
        "--input-reads", type=str, required=True, help="Input HiFi reads file path"
    )

    # Optional General Settings
    general = parser.add_argument_group("General settings")
    general.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to store output files.",
    )
    general.add_argument(
        "--threads", type=int, default=40, help="Number of threads to use"
    )
    general.add_argument("--ploidy", type=int, default=2, help="Ploidy of the genome")
    general.add_argument(
        "--busco-download-path",
        type=str,
        help="Custom BUSCO download path. If set, BUSCO datasets will not be (re)downloaded.",
    )

    # Optimization Parameters
    optimization = parser.add_argument_group("Optimization options")
    optimization.add_argument(
        "--sensitive",
        action="store_true",
        help="Optimize D, N, and max_kocc for possibly higher quality (longer runtime). Can be used in combination with --primary, --hic1, --hic2, and --ul to optimize Hi-C and ultra-long read parameters as well. Will also optimize x, y, s, n, m, and p parameters.",
    )
    optimization.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of trials for optimization. First 20 trials will always run, afterwards a custom multi-criteria convergence detector is used to detect convergence.",
    )
    optimization.add_argument(
        "--num-reads",
        type=int,
        default=10000,
        help="Number of reads to subset for minimap2",
    )
    optimization.add_argument(
        "--no-busco",
        dest="include_busco",
        action="store_false",
        help="Disable BUSCO metrics during evaluation. By default, BUSCO metrics are included.",
    )
    optimization.add_argument(
        "--busco-lineage",
        type=str,
        default="metazoa_odb12",
        help="BUSCO lineage database name",
    )
    optimization.add_argument(
        "--default-hifiasm",
        action="store_true",
        help="Run hifiasm assembly without optimized parameters, i.e. use all default parameter settings. Note: default behaviour of hifimizer saves the default assembly results into a default_assembly folder in the output directory.",
    )
    optimization.add_argument(
        "--primary",
        action="store_true",
        help="Perform primary assembly only. Can be used in combination with --default, --hic1, --hic2, and --ul to run hifiasm with default settings, Hi-C and ultra-long reads.",
    )
    optimization.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun of optimization and assembly even if convergence was previously reached.",
    )
    optimization.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. If not set, results may vary between runs.",
    )

    # Multi-objective options
    # Multi-criteria optimization is used by default (no CLI toggle).

    # Optional Input Data
    optional_inputs = parser.add_argument_group(
        "Optional sequencing data or hifiasm settings"
    )
    optional_inputs.add_argument("--hic1", type=str, help="Hi-C R1 reads file")
    optional_inputs.add_argument("--hic2", type=str, help="Hi-C R2 reads file")
    optional_inputs.add_argument("--ul", type=str, help="Ultra-long ONT reads file")

    return parser.parse_args()
