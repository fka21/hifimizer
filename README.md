# Hifimizer

**Hifimizer** is a framework for optimizing *de novo genome assembly* parameters using Bayesian optimization.
It wraps **hifiasm** in an automated optimization loop powered by **Optuna**, enabling systematic exploration
of the assembly parameter space instead of manual trial-and-error.

The primary goal is to identify parameter configurations that maximize assembly quality for a given dataset,
scored across contiguity, gene-space completeness (BUSCO), k-mer completeness and consensus accuracy (yak),
and reference-free misassembly signal (CRAQ).

Hifimizer supports standard **PacBio HiFi**, **Hi-C** integrated, **ultra-long ONT** integrated,
and **ONT R10 simplex** assemblies.

---

## Core idea

Genome assemblers use dozens of parameters, many of which interact non-linearly.
Hifimizer treats assembly as an optimization problem:

- parameter space → hifiasm arguments
- objective function → assembly quality metrics
- optimizer → Bayesian optimization (Optuna)

Each trial runs hifiasm with a sampled parameter set, evaluates the resulting assembly, and returns either a
single weighted score or a vector of objectives (for multi-objective / Pareto-front mode). A multi-criteria
convergence detector stops the study early once the score stops improving.

> **Note**
>
> Due to the stochastic nature of Bayesian optimization and adaptive sampling,
> the *exact sequence of trials and the final best solution* may vary between runs,
> even when random seeds are set.
> While individual components are seeded where possible, full end-to-end
> determinism is not guaranteed, especially under parallel execution.

### Workflow overview

![Hifimizer workflow](flowchart.svg)

---

## What gets evaluated

Every assembly a trial produces is scored on the following metrics. Directions (whether higher or lower is
better) live in `src/optim_directions.json`; weights for the single-objective weighted score live in
`weights.json` (see [Tuning the objective](#tuning-the-objective)).

| Source | Metric | Meaning |
| --- | --- | --- |
| **gfastats** | `n50` | Contig N50 (contiguity) |
| | `num_contigs` | Number of contigs (fragmentation) |
| | `length_diff` | \|assembly length − expected haploid size\| |
| **BUSCO** | `single_copy` | Complete single-copy BUSCOs |
| | `multi_copy` | Complete duplicated BUSCOs |
| | `fragmented` | Fragmented BUSCOs |
| | `missing` | Missing BUSCOs |
| **yak** | `qv` | Consensus accuracy (Phred-scaled) from read k-mers |
| | `kmer_completeness` | Fraction of solid read k-mers present in the assembly |
| **sniffles2** | `num_sv` | Structural variants called against the assembly |

A few implementation details worth knowing:

- **k-mer metrics (yak).** The read k-mer hash is built **once** during setup from the full read set and
  reused by every trial. QV is measured per-haplotype; k-mer completeness is measured on the combined
  haplotypes (hap1 + hap2) when Hi-C or ultra-long data yield a second haplotype, so that heterozygous
  sequence isn't scored as "missing". Disable with `--no-kmer-eval`.
- **BUSCO** is attempted with **miniprot** first, then **metaeuk**, then **augustus**; the first backend that
  succeeds within `--busco-walltime` is cached and reused for later trials.

---

## Installation options

You can run Hifimizer in three supported ways.

### Option 1: Conda (native execution)

Create the environment (its name, `optimizer`, is defined in `environment.yml`):

```bash
conda env create -f environment.yml
conda activate optimizer
```

Run:

```bash
python3 src/hifimizer.py -h
```

### Option 2: Docker

Pull the prebuilt image:

```bash
docker pull fka21/hifimizer:latest
```

Run:

```bash
docker run --rm \
  -v $(pwd):/wd \
  fka21/hifimizer:latest \
  src/hifimizer.py -h
```

> **Note**
>
> If the HPC environment cannot download the BUSCO database through BUSCO, please download it *a priori*,
> copy it to your HPC environment, and pass `--busco-download-path` when running `hifimizer`.

---

## Requirements

* Long reads: **PacBio HiFi**, or **ONT R10 simplex** (with `--ont`)
* A haploid genome size estimate
* Optionally: **Hi-C** reads (`--hic1`/`--hic2`) and/or **ultra-long ONT** reads (`--ul`)
* The following tools on `$PATH` (all provided by the conda environment / container):
  `hifiasm`, `gfastats`, `busco`, `yak`, `craq`, `minimap2`, `samtools`, `sniffles`
* Sufficient computational power for repeated assemblies
* Patience

---

## Quick start

Optimize a HiFi assembly:

```bash
python3 src/hifimizer.py \
  --genome-size 1.2G \
  --input-reads reads.hifi.fastq.gz \
  --threads 48 \
  --output-dir my_run
```

ONT R10 simplex:

```bash
python3 src/hifimizer.py --genome-size 300M --input-reads reads.ont.fastq.gz --ont
```

Hi-C integrated assembly (also optimizes `--s-base`, `--f-perturb`, `--l-msjoin`):

```bash
python3 src/hifimizer.py \
  --genome-size 3G --input-reads hifi.fq.gz \
  --hic1 hic_R1.fq.gz --hic2 hic_R2.fq.gz
```

Validate inputs and environment without assembling anything:

```bash
python3 src/hifimizer.py --genome-size 300M --input-reads reads.fq.gz --dry-run
```

The genome size accepts a suffix — `3G`, `1.5Gb`, `300M`, `750k` — or a bare integer interpreted as megabases.

---

## Reruns and resuming

Hifimizer persists its study in `optuna_study.db`, so runs are resumable and specific results can be
reproduced without re-optimizing:

- `--force-rerun` — discard any existing study and artifacts and start fresh.
- `--rerun-best` — skip optimization and rebuild the assembly from the best parameters in an existing study.
- `--rerun-trial N` — skip optimization and rebuild the assembly from the parameters of trial `N`.

Without `--force-rerun`, an existing study in the output directory is resumed.

---

## Walltime controls

Long-running external steps are individually bounded. When a step exceeds its limit the entire process group
is killed (not just the direct child, which matters for tools like BUSCO that spawn their own gene-prediction
subprocesses), and the trial is pruned.

- `--trial-walltime HOURS` (default 24) — per-trial hifiasm limit; also applied to the final assembly.
- `--busco-walltime HOURS` (default 6) — per gene-prediction-backend attempt.

---

## Tuning the objective

By default the tool uses a single weighted score for the objective function. The weights for each assembly
metric are defined in `weights.json`; any metric absent from the file falls back to a built-in default. The
optimization direction of each metric (maximize vs. minimize) is defined in `src/optim_directions.json` and is
also used to aggregate the multi-objective mode into a scalar for convergence detection.

Use `--multi-objective` to optimize a Pareto front over the individual metrics instead of a single weighted
sum.

> **Note**
>
> The `--sensitive` setting **can improve assemblies occasionally**, however it will significantly
> **increase runtime** as the read overlaps will be repeatedly re-calculated. It additionally optimizes
> hifiasm's `D`, `N`, and `max_kocc` parameters.

---

## Output

By default an `output/` directory is created in the current working directory. **Final results live directly
under the output directory; all intermediates live under `work/` and can be deleted to reclaim space** (at the
cost of a full hifiasm recompute on the next run, since hifiasm's error-corrected read and overlap `.bin`
files are cached there).

```
output/
├── final_assembly/              # Optimized assembly (best parameters)
│
├── default_assembly/            # Baseline hifiasm run (default parameters, trial 0)
│
├── logs/                        # main.log + per-command, per-trial logs (incl. per-trial metrics)
│
├── optuna_output/               # Optuna plots + per-metric history (incl. the baseline as a reference point)
│
├── optuna_study.db              # Optuna study database (all trials, params and metrics)
│
├── best_params_checkpoint.json  # Best parameters seen so far
│
└── work/                        # Intermediates — safe to delete
    ├── hifiasm/                 # Shared hifiasm prefix (.bin files reused across trials)
    ├── reads/                   # Subsampled reads for the alignment-based metrics
    ├── kmers/                   # yak read k-mer hash (built once, reused by all trials)
    ├── busco_downloads/         # BUSCO lineage datasets
    ├── cache/                   # Cached working BUSCO backend
    └── trials/trial_<id>/       # Per-trial CRAQ / sniffles / BUSCO / yak scratch
```

---

## Manual

Below are the available options for running the tool.

```bash
usage: hifimizer.py [-h] [--version] --genome-size GENOME_SIZE --input-reads
                    INPUT_READS [--output-dir OUTPUT_DIR] [--threads THREADS]
                    [--ploidy PLOIDY]
                    [--busco-download-path BUSCO_DOWNLOAD_PATH]
                    [--hom-cov COV] [--sensitive] [--num-trials NUM_TRIALS]
                    [--num-reads NUM_READS] [--no-busco]
                    [--busco-lineage BUSCO_LINEAGE] [--multi-objective]
                    [--default-hifiasm] [--primary] [--force-rerun]
                    [--dry-run] [--rerun-best] [--rerun-trial TRIAL_NUM]
                    [--trial-walltime HOURS] [--busco-walltime HOURS]
                    [--craq-walltime HOURS] [--craq-mapq MAPQ]
                    [--no-kmer-eval] [--kmer-k K] [--yak-bloom-bits BITS]
                    [--seed SEED] [--hic1 HIC1] [--hic2 HIC2] [--ul UL]
                    [--ont]

Optimize hifiasm de novo genome assemblies with Optuna. Supports parameter
optimization for standard HiFi, Hi-C, and ultra-long ONT assemblies. By
default optimizes: x, y, s, n, m, p. Sensitive mode additionally optimizes D,
N, and max_kocc. Genome size can be specified with a G/Gb (gigabases), M/Mb
(megabases), or K/Kb (kilobases) suffix, or as a plain integer interpreted as
megabases.

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

Required arguments:
  --genome-size GENOME_SIZE
                        Haploid genome size. Accepts a plain integer (treated
                        as Mb) or a value with a suffix: G/Gb for gigabases,
                        M/Mb for megabases, K/Kb for kilobases (e.g. 3G,
                        1.5Gb, 300M, 300, 750k). Internally converted to whole
                        megabases. (default: None)
  --input-reads INPUT_READS
                        Input HiFi reads file path (default: None)

General settings:
  --output-dir OUTPUT_DIR
                        Directory to store output files. (default: output)
  --threads THREADS     Number of threads to use (default: 40)
  --ploidy PLOIDY       Ploidy of the genome (default: 2)
  --busco-download-path BUSCO_DOWNLOAD_PATH
                        Custom BUSCO download path. If set, BUSCO datasets
                        will not be (re)downloaded. (default: None)
  --hom-cov COV         Homozygous read coverage passed to hifiasm --hom-cov
                        option. If not set, hifiasm auto-detects it from the
                        read depth histogram. (default: None)

Optimization options:
  --sensitive           Optimize D, N, and max_kocc for possibly higher
                        quality (longer runtime). Can be used in combination
                        with --primary, --hic1, --hic2, and --ul to optimize
                        Hi-C and ultra-long read parameters as well. Will also
                        optimize x, y, s, n, m, and p parameters. (default:
                        False)
  --num-trials NUM_TRIALS
                        Number of trials for optimization. First 20 trials
                        will always run, afterwards a custom multi-criteria
                        convergence detector is used to detect convergence.
                        (default: 100)
  --num-reads NUM_READS
                        Number of reads to subset for the alignment-based
                        metrics (CRAQ, sniffles2). CRAQ's AQI is a normalised
                        metric and so degrades gracefully with coverage, but
                        aim for at least ~10x: below that, large fractions of
                        the assembly get flagged low-confidence. (default:
                        100000)
  --no-busco            Disable BUSCO metrics during evaluation. By default,
                        BUSCO metrics are included. (default: True)
  --busco-lineage BUSCO_LINEAGE
                        BUSCO lineage database name (default: metazoa_odb12)
  --multi-objective     Use multi-objective optimization (Pareto front).
                        Default is single-objective optimization with weighted
                        score. (default: False)
  --default-hifiasm     Run hifiasm assembly without optimized parameters,
                        i.e. use all default parameter settings. Note: default
                        behaviour of hifimizer saves the default assembly
                        results into a default_assembly folder in the output
                        directory. (default: False)
  --primary             Perform primary assembly only. Can be used in
                        combination with --default, --hic1, --hic2, and --ul
                        to run hifiasm with default settings, Hi-C and ultra-
                        long reads. (default: False)
  --force-rerun         Force rerun of optimization and assembly even if
                        convergence was previously reached. (default: False)
  --dry-run             Validate inputs and environment without running any
                        assemblies. Checks that all input files exist,
                        required tools (hifiasm, busco, gfastats, yak) are on
                        PATH, and prints the trial-0 hifiasm command, then
                        exits. (default: False)
  --rerun-best          Skip optimization and rerun hifiasm using the best
                        parameters recorded in an existing study. Requires
                        that the previous run reached convergence.
                        Incompatible with --force-rerun. (default: False)
  --rerun-trial TRIAL_NUM
                        Skip optimization and rerun hifiasm using the
                        parameters of a specific trial number from an existing
                        study. Incompatible with --force-rerun and --rerun-
                        best. (default: None)
  --trial-walltime HOURS
                        Maximum wall-clock time in hours allowed for a single
                        hifiasm trial. Trials that exceed this limit are
                        killed, logged as timed-out, and pruned from the
                        Optuna study. The final assembly step uses the same
                        limit. Default: 24 hours. (default: 24.0)
  --busco-walltime HOURS
                        Maximum wall-clock time in hours allowed for a single
                        BUSCO gene-prediction attempt. BUSCO is tried with
                        miniprot, then metaeuk, then augustus; each attempt
                        gets this budget and the whole process group is killed
                        on expiry. Default: 6 hours. (default: 6.0)
  --no-kmer-eval        Disable the yak-based k-mer metrics (consensus QV and
                        k-mer completeness). By default they are included; the
                        read k-mer hash is built once during setup and reused
                        by every trial. (default: True)
  --kmer-k K            k-mer length passed to `yak count` (-k). (default: 31)
  --yak-bloom-bits BITS
                        Bloom-filter size passed to `yak count` (-b), used to
                        discard singleton k-mers. 37 is lh3's recommendation
                        for human-scale, high-coverage read sets. Set to 0 to
                        disable the Bloom filter (needed for low-coverage read
                        sets, at the cost of memory). (default: 37)
  --seed SEED           Random seed for reproducibility. If not set, results
                        may vary between runs. (default: 42)

Optional sequencing data or hifiasm settings:
  --hic1 HIC1           Hi-C R1 reads file (default: None)
  --hic2 HIC2           Hi-C R2 reads file (default: None)
  --ul UL               Ultra-long ONT reads file (default: None)
  --ont                 Use this flag if as input you provide ONT R10 simplex
                        reads. (default: False)
```