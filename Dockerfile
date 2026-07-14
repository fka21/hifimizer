FROM continuumio/miniconda3

WORKDIR /opt/project

# ----------------------------
# 1. System dependencies FIRST
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 2. Conda env
# ----------------------------
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate conda AFTER system deps
SHELL ["conda", "run", "-n", "optimizer", "/bin/bash", "-c"]

# ----------------------------
# 3. Install yak (k-mer QV + completeness)
# ----------------------------
# environment.yml already pulls yak from bioconda so that a plain
# `conda env create` works without Docker. The conda env's bin directory takes
# precedence over /usr/local/bin, so we install the source build straight into
# $CONDA_PREFIX/bin to make sure the version we actually run is this one.
RUN git clone https://github.com/lh3/yak.git && \
    cd yak && \
    make && \
    cp yak "${CONDA_PREFIX}/bin/" && \
    yak version

# ----------------------------
# 4. Your code
# ----------------------------
COPY src/ ./src/
ENV PATH="/opt/project/src:${PATH}"

# ----------------------------
# 5. Entrypoint
# ----------------------------
COPY src/utils/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh