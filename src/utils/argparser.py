# args_parser.py
import argparse


def parse_genome_size(value: str) -> int:
    """
    Parse a genome size string into an integer number of megabases.

    Accepted suffixes (case-insensitive):
        G / Gb / GBp  →  gigabases  (multiply by 1 000)
        M / Mb / MBp  →  megabases  (no conversion, this is the internal unit)
        K / Kb / KBp  →  kilobases  (divide by 1 000, rounded to nearest Mb)

    A bare number with no suffix is assumed to be megabases, preserving
    backwards-compatibility with the previous integer-only behaviour.

    Examples:
        "3G" → 3000, "1.5G" → 1500, "300M" → 300, "300" → 300, "750k" → 1
    """
    raw = value.strip()
    suffixes = {
        "gbp": 1_000,
        "gb": 1_000,
        "g": 1_000,
        "mbp": 1,
        "mb": 1,
        "m": 1,
        "kbp": 1e-3,
        "kb": 1e-3,
        "k": 1e-3,
    }

    lower = raw.lower()
    multiplier = 1  # default: treat bare number as Mb
    numeric_str = raw

    for suffix, mult in sorted(suffixes.items(), key=lambda x: -len(x[0])):
        if lower.endswith(suffix):
            multiplier = mult
            numeric_str = raw[: -len(suffix)]
            break

    try:
        result = round(float(numeric_str) * multiplier)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid genome size '{value}'. "
            "Expected a number optionally followed by G/Gb, M/Mb, or K/Kb "
            "(e.g. 3G, 1.5Gb, 300M, 300, 750k)."
        )

    if result <= 0:
        raise argparse.ArgumentTypeError(
            f"Genome size must be greater than zero (got '{value}' → {result} Mb)."
        )

    return result


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize hifiasm de novo genome assemblies with Optuna. "
            "Supports parameter optimization for standard HiFi, Hi-C, and ultra-long "
            "ONT assemblies. By default optimizes: x, y, s, n, m, p. "
            "Sensitive mode additionally optimizes D, N, and max_kocc. "
            "Genome size can be specified with a G/Gb (gigabases), M/Mb (megabases), "
            "or K/Kb (kilobases) suffix, or as a plain integer interpreted as megabases."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 1.0.9")

    # Required Inputs
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--genome-size",
        type=parse_genome_size,
        required=True,
        help=(
            "Haploid genome size. Accepts a plain integer (treated as Mb) or a value "
            "with a suffix: G/Gb for gigabases, M/Mb for megabases, K/Kb for kilobases "
            "(e.g. 3G, 1.5Gb, 300M, 300, 750k). Internally converted to whole megabases."
        ),
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
        default=100000,
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
        "--multi-objective",
        action="store_true",
        help="Use multi-objective optimization (Pareto front). Default is single-objective optimization with weighted score.",
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
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs and environment without running any assemblies. "
            "Checks that all input files exist, required tools (hifiasm, busco, "
            "gfastats) are on PATH, and prints the trial-0 hifiasm command, then exits."
        ),
    )
    optimization.add_argument(
        "--rerun-best",
        action="store_true",
        help=(
            "Skip optimization and rerun hifiasm using the best parameters recorded in an "
            "existing study. Requires that the previous run reached convergence. "
            "Incompatible with --force-rerun."
        ),
    )
    optimization.add_argument(
        "--rerun-trial",
        type=int,
        default=None,
        metavar="TRIAL_NUM",
        help=(
            "Skip optimization and rerun hifiasm using the parameters of a specific trial "
            "number from an existing study. Incompatible with --force-rerun and --rerun-best."
        ),
    )
    optimization.add_argument(
        "--trial-walltime",
        type=float,
        default=24.0,
        metavar="HOURS",
        help=(
            "Maximum wall-clock time in hours allowed for a single hifiasm trial. "
            "Trials that exceed this limit are killed, logged as timed-out, and pruned "
            "from the Optuna study. The final assembly step uses the same limit. "
            "Default: 24 hours."
        ),
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
    optional_inputs.add_argument(
        "--ont",
        action="store_true",
        help="Use this flag if as input you provide ONT R10 simplex reads.",
    )

    return parser.parse_args()
