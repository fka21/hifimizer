# args_parser.py
import argparse

from utils.assembly_eval import BITS_CAP
from utils.samplers import SAMPLER_CHOICES, SAMPLER_DESCRIPTIONS


def parse_genome_size(value: str) -> int:
    """
    Parse a genome size string into a number of megabases.

    Accepted suffixes (case-insensitive):
        G / Gb / GBp  →  gigabases  (multiply by 1 000)
        M / Mb / MBp  →  megabases  (no conversion, this is the internal unit)
        K / Kb / KBp  →  kilobases  (divide by 1 000)

    A bare number with no suffix is assumed to be megabases, preserving
    backwards-compatibility with the previous integer-only behaviour.

    Examples:
        "3G" → 3000, "1.5G" → 1500, "300M" → 300, "300" → 300, "750k" → 0.75
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

    # Not rounded to whole megabases: the value is both hifiasm's --hg-size and
    # the target length_diff is measured against, and rounding made 4.64m
    # count as 5 Mb, 2.5m as 2 Mb and anything under 500 kb as zero.
    try:
        result = float(numeric_str) * multiplier
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

    parser.add_argument("--version", action="version", version="%(prog)s 2.0.1")

    # Required Inputs
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--genome-size",
        type=parse_genome_size,
        required=True,
        help=(
            "Haploid genome size. Accepts a plain integer (treated as Mb) or a value "
            "with a suffix: G/Gb for gigabases, M/Mb for megabases, K/Kb for kilobases "
            "(e.g. 3G, 1.5Gb, 300M, 300, 750k). Used exactly, not rounded."
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
        help=(
            "Existing BUSCO dataset location: the download root (the folder "
            "holding lineages/), its lineages/ folder, or the lineage folder "
            "itself. When the lineage is found there, BUSCO runs with "
            "--offline and nothing is downloaded or updated."
        ),
    )
    general.add_argument(
        "--compleasm-download-path",
        type=str,
        help=(
            "Existing compleasm lineage library (compleasm's -L), or the "
            "lineage folder inside it. Used by every trial. compleasm decides "
            "what is present from its <lineage>.done markers and always wants "
            "eukaryota_<odb> as well, so a library without those is completed "
            "once at startup rather than per trial. Kept separate from "
            "--busco-download-path: compleasm stores lineages miniprot-indexed "
            "and in its own layout, so a BUSCO download directory is not "
            "interchangeable with it."
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
        help=(
            "Also optimize -D, -N and --max-kocc, plus -k (21/31/51/61), "
            "-w (21/31/51/61/71), --max-od-ec (0.04/0.06/0.08; with --ont "
            "0.07/0.09/0.11) and --max-od-final (0.03/0.05/0.07) as discrete "
            "choices that include hifiasm's defaults. These act on hifiasm's "
            "k-mer filtering, overlap and error-correction stages, which "
            "hifiasm skips whenever it finds cached .bin files -- so in this "
            "mode the cache is deleted before every hifiasm run and each trial "
            "recomputes error correction from the raw reads. Expect every "
            "trial to take several times longer. Combines with --primary, "
            "--hic1/--hic2 and --ul, and still optimizes x, y, s, n, m and p."
        ),
    )
    optimization.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help=(
            "Maximum number of optimisation trials, on top of trial 0 (the "
            "default-parameter baseline). "
            "The first --convergence-warmup always run; after that a "
            "multi-criteria convergence detector may stop the study early."
        ),
    )
    optimization.add_argument(
        "--bits-cap",
        type=float,
        default=BITS_CAP,
        metavar="BITS",
        help=(
            "Soft cap on how many bits any single metric can add to or take "
            "from the weighted score. Every metric is scored as a log2 fold "
            "change against trial 0 (+1 bit = twice as good, -1 = twice as "
            "bad), which is scale-free, so no burn-in is needed to put the "
            "metrics on one footing; but it is not bounded, and a single "
            "runaway metric (a 500x contig-count collapse is -9 bits) would "
            "otherwise outvote all the others. Applied as "
            "cap * tanh(bits / cap): nearly linear below half the cap, "
            "saturating smoothly above it, and still order-preserving. "
            "A cap of 4 bits is a 16x change. Set to 0 to disable."
        ),
    )
    # Accepted and ignored, so command lines written for the burn-in version
    # keep working. hifimizer warns once when it is given.
    optimization.add_argument(
        "--burn-in-trials",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
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
        "--sampler",
        type=str.lower,
        choices=SAMPLER_CHOICES,
        default=SAMPLER_CHOICES[0],
        help=(
            "How new parameter sets are proposed: "
            + "; ".join(f"'{k}' = {v}" for k, v in SAMPLER_DESCRIPTIONS.items())
            + ". 'gp' is single-objective only. A study is tied to the sampler "
            "it started with: resuming it with a different one stops with an "
            "error, and switching needs --force-rerun."
        ),
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
        "--continue-stopped",
        action="store_true",
        help=(
            "Resume a study that has already stopped -- because it converged, "
            "or was halted by --max-consecutive-pruned or --max-metric-skips "
            "-- and run up to --num-trials more trials in it. Without this, "
            "resuming a stopped study runs no new trials, in either single- or "
            "multi-objective mode: it only reports the best trial and builds "
            "the final assembly. The stop is cleared, so the study can stop "
            "again on its own criteria; a study halted by a trial budget will "
            "usually need that budget raised as well, since the trials that "
            "used it up are still counted."
        ),
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
        "--prune-bits",
        type=float,
        default=1.0,
        metavar="BITS",
        help=(
            "Abandon a trial straight after gfastats if its contig count, N50 "
            "or length difference is worse than the default-parameter assembly "
            "by more than this many bits -- the same unit the score uses, "
            "before the --bits-cap (1.0 = twice as bad, 2.0 = four times). "
            "gfastats costs seconds and everything after it costs hours, so a "
            "collapsed assembly is recognised before any of that is paid for. "
            "Measured against trial 0 rather than the best trial so far, which "
            "stops the bar ratcheting tighter every time the study improves. "
            "Set to 0 to disable pruning."
        ),
    )
    # The pre-bits name, kept so existing command lines keep working.
    stages.add_argument(
        "--prune-fc",
        dest="prune_bits",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    stages.add_argument(
        "--prune-min-metrics",
        type=int,
        default=2,
        metavar="N",
        help=(
            "How many of the gated contiguity metrics must be past --prune-bits "
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

    args = parser.parse_args()
    # Checked here rather than when the sampler is built, which happens after
    # the read k-mer hash and the completeness lineage are prepared -- hours
    # into a run, for an error that is known from the command line alone.
    if args.sampler == "gp" and args.multi_objective:
        parser.error(
            "--sampler gp supports a single objective only; use --sampler tpe "
            "with --multi-objective."
        )
    return args