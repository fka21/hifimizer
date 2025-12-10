# Hifimizer

This repository contains my attempt on trying to improve genome assembler pipelines. The idea revolves around using Bayesian optimization framework from [optuna](https://optuna.readthedocs.io/en/stable/) to try to find the best assembly parameter space in [hifiasm](https://hifiasm.readthedocs.io/en/latest/).

## Prerequisites

There are 3 options for using this tool. 

1.  Use the `src/hifimizer.py` directly. For this please setup a [conda](https://bioconda.github.io) environemnt using the `environment.yml` file. 
2.  Use the `Dockerfile` provided to build your own Docker image.
3.  Use the `Singularity.def` to build a singularity image.

> **NOTE**
>  
> For the Singularity option there are two `.sif` files. One of them, `hifimizer-online.sif` is able to download [busco](https://busco.ezlab.org) database online. The other one, `hifimizer-offline.sif` is set up to work offline. Therefore [busco](https://busco.ezlab.org) database needs to be downloaded *a priori* for usage.

## Usage 

### Manual
Please use the manual of the tool to navigate the possible options.
```
# If using the first installation option please use:
src/hifimizer.py -h

# If using Docker please use:
docker run -v [path-to-this-repository]:/opt/project fka21/hifimizer:1.0.5 src/hifimizer.py -h

# If using Singularity please use:
apptainer exec --bind [path-to-this-repository]:/opt/project hifimizer-online.sif src/hifimizer.py -h
```

### Example

Toy reads are provided to test the tool. Please use the following command to test it.

```
# Run this minimal example
nohup docker run \
    -v [path-to-this-repository]:/opt/project \
    fka21/hifimizer:1.0.5 \
    src/hifimizer.py \
    --input-reads example/SRR10971019_subseq.fastq.gz \
    --genome-size 5 \
    --busco-lineage enterobacterales_odb10 \
    --threads 20 \
    --num-trials 25 > hifimizer.log 2>&1 &
```