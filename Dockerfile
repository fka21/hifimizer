FROM continuumio/miniconda3

WORKDIR /opt/project

# Name and location of the conda environment created from environment.yml.
ENV CONDA_ENV_NAME=optimizer
ENV CONDA_ENV_PATH=/opt/conda/envs/optimizer

# ----------------------------
# 1. System dependencies FIRST
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 2. Conda env
# ----------------------------
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# ----------------------------------------------------------------------------
# 3. Make the environment's interpreter *the* interpreter for this image
# ----------------------------------------------------------------------------
# This is the fix for "numpy is not installed".
#
# The previous image relied solely on the entrypoint running `conda activate`.
# That covers `docker run <image> <cmd>` and nothing else. It is bypassed by:
#
#   docker exec <container> python src/hifimizer.py ...
#   docker run --entrypoint python <image> ...
#   singularity/apptainer exec <image.sif> ...     (Docker ENTRYPOINT ignored)
#   any HPC runner that invokes the image's command directly
#
# In all of those, PATH still starts with /opt/conda/bin, so `python` is the
# *base* miniconda interpreter -- which has no numpy, no optuna, no Bio. The
# symptom is exactly "ModuleNotFoundError: No module named 'numpy'".
#
# Baking the env's bin directory into ENV PATH removes that whole class of
# failure. The entrypoint is kept as well, because activation additionally
# sources the env's activate.d hooks (R, perl, GDK_PIXBUF, ...).
ENV PATH="${CONDA_ENV_PATH}/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=optimizer

# Redirect caches that would otherwise land in a possibly read-only $HOME.
ENV MPLCONFIGDIR=/tmp/mplconfig \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1

# ----------------------------
# 4. Install yak (k-mer QV + completeness)
# ----------------------------
# environment.yml already pulls yak from bioconda so that a plain
# `conda env create` works without Docker. The env's bin directory takes
# precedence, so the source build is installed straight into it to make sure
# the version we actually run is this one.
RUN git clone --depth 1 https://github.com/lh3/yak.git && \
    cd yak && \
    make && \
    cp yak "${CONDA_ENV_PATH}/bin/" && \
    yak version && \
    cd .. && rm -rf yak

# ----------------------------------------------------------------------------
# 5. Fail the *build* if the environment is incomplete
# ----------------------------------------------------------------------------
# Without this, a solver that quietly dropped a package (or a `pip:` section
# that did not run) is only discovered hours into a run on a cluster.
RUN python -c "import sys, numpy, scipy, optuna, plotly, psutil, Bio; \
print('interpreter:', sys.executable); \
print('numpy      :', numpy.__version__); \
print('optuna     :', optuna.__version__)" && \
    for tool in hifiasm minimap2 samtools sniffles gfastats busco yak; do \
        command -v "$tool" >/dev/null || { echo "MISSING TOOL: $tool" >&2; exit 1; }; \
    done && echo "all external tools present"

# ----------------------------
# 6. Your code
# ----------------------------
COPY src/ ./src/
ENV PATH="/opt/project/src:${PATH}"

# ----------------------------
# 7. Entrypoint
# ----------------------------
COPY src/utils/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3", "src/hifimizer.py", "--help"]