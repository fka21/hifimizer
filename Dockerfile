FROM continuumio/miniconda3

WORKDIR /opt/project

# ----------------------------
# 1. System dependencies FIRST
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 2. Conda env
# ----------------------------
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate conda AFTER system deps
SHELL ["conda", "run", "-n", "optimizer", "/bin/bash", "-c"]

# ----------------------------
# 3. Install mm2plus
# ----------------------------
RUN git clone https://github.com/at-cg/mm2-plus.git && \
    cd mm2-plus && \
    make deps && \
    make && \
    cp mm2plus /usr/local/bin/

# ----------------------------
# 4. Install compleasm from GitHub release
# ----------------------------
# We install compleasm from the upstream release tarball rather than from
# bioconda. The bioconda recipe pins dendropy<4.6.0, which conflicts with
# the dendropy>=5.0.1 requirement of sepp and pasta in this environment.
# The release tarball bundles compleasm with its own miniprot and
# hmmsearch binaries and only needs `pandas` at runtime, which is already
# provided by the conda environment.
ARG COMPLEASM_VERSION=0.2.7
RUN wget -q \
    https://github.com/huangnengCSU/compleasm/releases/download/v${COMPLEASM_VERSION}/compleasm-${COMPLEASM_VERSION}_x64-linux.tar.bz2 \
    -O /tmp/compleasm.tar.bz2 && \
    tar -jxf /tmp/compleasm.tar.bz2 -C /opt && \
    rm /tmp/compleasm.tar.bz2 && \
    chmod +x /opt/compleasm_kit/compleasm.py && \
    printf '#!/bin/bash\nexec python /opt/compleasm_kit/compleasm.py "$@"\n' \
        > /usr/local/bin/compleasm && \
    chmod +x /usr/local/bin/compleasm && \
    compleasm --help > /dev/null

# ----------------------------
# 5. Copy the code itself
# ----------------------------
COPY src/ ./src/
RUN chmod +x /opt/project/src/hifimizer.py
ENV PATH="/opt/project/src:${PATH}"

# ----------------------------
# 6. Entrypoint
# ----------------------------
COPY src/utils/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]