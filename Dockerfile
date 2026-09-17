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
RUN conda config --system --set auto_activate_base false
RUN conda env create -f environment.yml && conda clean -afy

# ----------------------------------------------------------------------------
# 2b. compleasm, in a SEPARATE environment
# ----------------------------------------------------------------------------
COPY environment.compleasm.yml .
RUN conda env create -f environment.compleasm.yml --solver=libmamba && conda clean -afy
ENV COMPLEASM_BIN=/opt/conda/envs/compleasm/bin/compleasm

# ----------------------------------------------------------------------------
# 3. Make the environment's interpreter *the* interpreter for this image
# ----------------------------------------------------------------------------
ENV PATH="${CONDA_ENV_PATH}/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=optimizer

# Redirect caches that would otherwise land in a possibly read-only $HOME.
ENV MPLCONFIGDIR=/tmp/mplconfig \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1

# ----------------------------
# 4. Install yak (k-mer QV + completeness)
# ----------------------------
RUN git clone --depth 1 https://github.com/lh3/yak.git && \
    cd yak && \
    make && \
    cp yak "${CONDA_ENV_PATH}/bin/" && \
    yak version && \
    cd .. && rm -rf yak

# ----------------------------------------------------------------------------
# 5. Fail the *build* if the environment is incomplete
# ----------------------------------------------------------------------------
RUN python -c "import sys, numpy, scipy, optuna, plotly, psutil, Bio; \
print('interpreter:', sys.executable); \
print('numpy      :', numpy.__version__); \
print('optuna     :', optuna.__version__)" && \
    for tool in hifiasm minimap2 samtools gfastats busco yak seqtk; do \
        command -v "$tool" >/dev/null || { echo "MISSING TOOL: $tool" >&2; exit 1; }; \
    done && echo "all external tools present"

RUN PATH="/opt/conda/envs/compleasm/bin:${PATH}" "${COMPLEASM_BIN}" --version && \
    PATH="/opt/conda/envs/compleasm/bin:${PATH}" command -v miniprot >/dev/null || \
    { echo "compleasm environment is incomplete" >&2; exit 1; }

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