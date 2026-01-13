# Hifimizer

**Hifimizer** is a framework for optimizing *HiFi genome assembly* parameters using Bayesian optimization.
It wraps **hifiasm** in an automated optimization loop powered by **Optuna**, enabling systematic exploration
of the assembly parameter space instead of manual trial-and-error.

The primary goal is to identify parameter configurations that maximize assembly quality metrics
(e.g. contiguity, BUSCO completeness) for a given dataset.

---

## Core idea

Genome assemblers use dozens of parameters, many of which interact non-linearly.
Hifimizer treats assembly as an optimization problem:

- parameter space → hifiasm arguments
- objective function → assembly quality metrics
- optimizer → Bayesian optimization (Optuna)

> **Note**
>
> Due to the stochastic nature of Bayesian optimization and adaptive sampling,
> the *exact sequence of trials and the final best solution* may vary between runs,
> even when random seeds are set.
> While individual components are seeded where possible, full end-to-end
> determinism is not guaranteed, especially under parallel execution.

---

## Installation options

You can run Hifimizer in three supported ways.

### Option 1: Conda (native execution)

Create the environment:

```bash
conda env create -f environment.yml
conda activate hifimizer
```

Run
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
  -v $(pwd):/opt/project \
  fka21/hifimizer:latest \
  src/hifimizer.py -h
```

### Option 3: Singularity (for HPC environments)

Two images are provided:

* `hifimizer-online.sif` - Downloads BUSCO databases at runtime.
* `hifimizer-offline.sif` - Requires BUSCO databases to be downloaded in advance.

Example:

```bash
apptainer exec \
  --bind $(pwd):/opt/project \
  hifimizer-online.sif \
  src/hifimizer.py -h
```

## Requirements
* HiFi (PacBio CCS) reads
* Genome size estimate
* hifiasm
* Sufficient compute for repeated assemblies
* Patience
