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

    parser.add_argument("--version", action="version", version="%(prog)s 2.0.0")

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
    general.add_argument(
        "--compleasm-download-path",
        type=str,
        help=(
            "Custom compleasm lineage library (compleasm's -L). Kept separate "
            "from --busco-download-path: compleasm stores lineages "
            "miniprot-indexed and in its own layout, so a BUSCO download "
            "directory is not interchangeable with it."
        ),
    )
    general.add_argument(
        "--compleasm-bin",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the compleasm executable. compleasm and BUSCO cannot "
            "share a conda environment, so the container installs compleasm "
            "into its own and advertises it through $COMPLEASM_BIN. Set this "
            "only if you installed it somewhere else. When neither is set, "
            "hifimizer looks on PATH and then for a conda environment named "
            "'compleasm' next to its own."
        ),
    )

    general.add_argument(
        "--hom-cov",
        type=int,
        default=None,
        metavar="COV",
        help=(
            "Homozygous read coverage passed to hifiasm --hom-cov option. "
            "If not set, hifiasm auto-detects it from the read depth histogram."
        ),
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
        help=(
            "Maximum number of optimisation trials, on top of the burn-in. "
            "The first --convergence-warmup always run; after that a "
            "multi-criteria convergence detector may stop the study early."
        ),
    )
    optimization.add_argument(
        "--burn-in-trials",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Random-parameter trials run before the optimisation, on top of "
            "trial 0 (hifiasm with default parameters, the reference every "
            "trial is compared against). They measure how far each metric "
            "actually moves when the parameters change, and every score "
            "afterwards is divided by that - so a metric that happens to swing "
            "wildly cannot outvote a steady one purely by swinging. They are "
            "never pruned, and are carried into the optimisation as its "
            "starting observations rather than discarded. Below about 5 the "
            "scale estimates are too noisy to divide by."
        ),
    )
    optimization.add_argument(
        "--no-busco",
        dest="include_busco",
        action="store_false",
        help=(
            "Disable the gene-space completeness metrics (single_copy, "
            "multi_copy, fragmented, missing) during evaluation. These are "
            "produced by compleasm where available and by BUSCO otherwise; "
            "this flag switches off the whole stage regardless of backend. "
            "By default they are included."
        ),
    )
    optimization.add_argument(
        "--no-compleasm",
        dest="use_compleasm",
        action="store_false",
        help=(
            "Skip compleasm and score gene-space completeness with BUSCO "
            "directly. compleasm is the default because it is a miniprot-based "
            "reimplementation of BUSCO that is both faster and considerably "
            "lighter on peak memory; use this to reproduce BUSCO numbers "
            "exactly, or if compleasm misbehaves on your lineage."
        ),
    )
    optimization.add_argument(
        "--busco-lineage",
        type=str,
        default="metazoa_odb12",
        help=(
            "Lineage dataset name, passed to whichever completeness backend "
            "runs. compleasm and BUSCO accept the same lineage names but store "
            "them in incompatible formats, so each downloads its own copy."
        ),
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
            "gfastats, yak) are on PATH, and prints the trial-0 hifiasm command, "
            "then exits."
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
        "--no-kmer-eval",
        dest="kmer_eval",
        action="store_false",
        help=(
            "Disable the yak-based k-mer metrics (consensus QV and k-mer "
            "completeness). By default they are included; the read k-mer hash is "
            "built once during setup and reused by every trial."
        ),
    )
    optimization.add_argument(
        "--kmer-k",
        type=int,
        default=31,
        metavar="K",
        help="k-mer length passed to `yak count` (-k).",
    )
    optimization.add_argument(
        "--yak-bloom-bits",
        type=int,
        default=37,
        metavar="BITS",
        help=(
            "Bloom-filter size passed to `yak count` (-b), used to discard "
            "singleton k-mers. 37 is lh3's recommendation for human-scale, "
            "high-coverage read sets. Set to 0 to disable the Bloom filter "
            "(needed for low-coverage read sets, at the cost of memory)."
        ),
    )
    optimization.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. If not set, results may vary between runs.",
    )

    # Per-stage walltimes and failure handling
    stages = parser.add_argument_group(
        "Metric stage walltimes and failure handling",
        (
            "Each metric-producing tool runs as an independent 'stage' with its "
            "own wall-clock budget. What happens when one fails or times out "
            "depends on when it happens. On the baseline assembly (trial 0) the "
            "tool is judged unusable here, so it is retired for the whole study "
            "and no later trial attempts it; the baseline is still scored on "
            "the remaining metrics, and that reduced set becomes the objective. "
            "From trial 1 on the tool has already been shown to work, so a "
            "failure means the trial is discarded rather than scored on a "
            "different metric set -- see --max-metric-skips. Either way the "
            "run tells you loudly that a data-quality check is warranted. Set "
            "any walltime to 0 for no limit."
        ),
    )
    stages.add_argument(
        "--gfastats-walltime",
        type=float,
        default=0.5,
        metavar="HOURS",
        help="Walltime for gfastats (num_contigs, length_diff, n50).",
    )
    stages.add_argument(
        "--yak-walltime",
        type=float,
        default=2.0,
        metavar="HOURS",
        help=(
            "Walltime for each yak invocation: the one-off `yak count` during "
            "setup and the per-trial `yak qv` (qv, kmer_completeness)."
        ),
    )
    stages.add_argument(
        "--busco-walltime",
        type=float,
        default=6.0,
        metavar="HOURS",
        help=(
            "Walltime for a single gene-space completeness attempt. The chain "
            "is compleasm, then BUSCO with metaeuk, then BUSCO with augustus; "
            "each attempt gets this budget and the whole process group is "
            "killed on expiry. Only when all of them are exhausted does the "
            "stage count as failed. BUSCO's own miniprot backend is not in the "
            "chain because compleasm already is miniprot."
        ),
    )
    stages.add_argument(
        "--hic-phasing-walltime",
        type=float,
        default=4.0,
        metavar="HOURS",
        help=(
            "Walltime for the Hi-C phasing metric (trans_hap_rate). Only runs "
            "when --hic1/--hic2 are given."
        ),
    )
    stages.add_argument(
        "--prune-fc",
        type=float,
        default=1.0,
        metavar="LOG2FC",
        help=(
            "Abandon a trial straight after gfastats if its contig count, N50 "
            "or length difference is worse than the default-parameter assembly "
            "by more than this many log2 units (1.0 = twice as bad). gfastats "
            "costs seconds and everything after it costs hours, so a collapsed "
            "assembly is recognised before any of that is paid for. Measured "
            "against trial 0 rather than the best trial so far, which stops "
            "the bar ratcheting tighter every time the study improves. Set to "
            "0 to disable pruning."
        ),
    )
    stages.add_argument(
        "--prune-min-metrics",
        type=int,
        default=2,
        metavar="N",
        help=(
            "How many of the gated contiguity metrics must be past --prune-fc "
            "before a trial is pruned. Contig count, N50 and length difference "
            "are strongly coupled, so one of them drifting out is ordinary "
            "variation and only several going at once is a collapsed assembly. "
            "Set to 1 for the old behaviour of pruning on any single metric, "
            "or 0 to disable pruning."
        ),
    )
    stages.add_argument(
        "--max-consecutive-pruned",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Stop the study after this many trials are pruned back to back "
            "without one completing. Each pruned trial still costs a full "
            "hifiasm run, and an unbroken run of them means the sampler is not "
            "finding usable parameter sets. The best trial so far is still "
            "assembled and reported. Counted consecutively, not in total: "
            "prunes scattered across a long study are expected. Set to 0 to "
            "never give up."
        ),
    )
    stages.add_argument(
        "--max-metric-skips",
        type=int,
        default=5,
        metavar="N",
        help=(
            "How many trials may be discarded because a metric failed before "
            "the optimization gives up. Only post-baseline failures count: a "
            "metric that fails on trial 0 is retired for the whole study and "
            "costs no trials at all, whereas a metric that worked on trial 0 "
            "and then fails signals that something changed, so that trial is "
            "thrown away instead of being scored on a different metric set. "
            "Reaching this limit stops the study cleanly - the best trial so "
            "far is still assembled and reported. Set to 0 to never give up."
        ),
    )
    stages.add_argument(
        "--convergence-warmup",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Number of initial trials during which convergence detection is "
            "suppressed, so a short run cannot stop on its own noise."
        ),
    )

    # Optional Input Data
    optional_inputs = parser.add_argument_group(
        "Optional sequencing data or hifiasm settings"
    )
    optional_inputs.add_argument("--hic1", type=str, help="Hi-C R1 reads file")
    optional_inputs.add_argument("--hic2", type=str, help="Hi-C R2 reads file")
    optional_inputs.add_argument("--ul", type=str, help="Ultra-long ONT reads file")
    optional_inputs.add_argument(
        "--num-hic-reads",
        type=int,
        default=1_000_000,
        metavar="N",
        help=(
            "Hi-C read pairs subsampled once and reused by every trial to "
            "score phasing. Only the pairs that survive --hic-min-mapq are "
            "informative, and that is a minority, so this needs to be "
            "generous; the surviving count is logged per trial as "
            "hic_pairs_informative."
        ),
    )
    optional_inputs.add_argument(
        "--hic-min-mapq",
        type=int,
        default=20,
        metavar="Q",
        help=(
            "Minimum MAPQ for a Hi-C mate to count towards phasing. The two "
            "haplotypes are nearly identical, so reads that do not overlap a "
            "distinguishing variant map ambiguously and carry no phasing "
            "information; this filter is what removes them."
        ),
    )
    optional_inputs.add_argument(
        "--hifiasm-extra",
        type=str,
        default=None,
        metavar="ARGS",
        help=(
            "Extra hifiasm arguments, quoted, applied to every run including "
            "the default-parameter baseline. Anything hifimizer normally "
            "optimizes is pinned to your value and dropped from the search "
            "space (e.g. '-s 0.5' fixes -s instead of sampling it); anything "
            "else is passed to hifiasm verbatim (e.g. '--telo-m CCCTAA'). "
            "Flags hifimizer sets itself (-o, -t, --hg-size, --h1, --h2, "
            "--ul, --primary, --ont, --hom-cov) are rejected -- use their "
            "dedicated options."
        ),
    )
    optional_inputs.add_argument(
        "--no-hic-phasing",
        dest="hic_phasing",
        action="store_false",
        help=(
            "Skip the Hi-C phasing metric even when --hic1/--hic2 are given. "
            "Without it the Hi-C parameters (--s-base, --f-perturb, "
            "--l-msjoin) are tuned against an objective that cannot see "
            "phasing quality."
        ),
    )
    optional_inputs.add_argument(
        "--ont",
        action="store_true",
        help="Use this flag if as input you provide ONT R10 simplex reads.",
    )

    return parser.parse_args()