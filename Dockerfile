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
# 4. Your code
# ----------------------------
COPY src/ ./src/
ENV PATH="/opt/project/src:${PATH}"

# ----------------------------
# 5. Entrypoint
# ----------------------------
COPY src/utils/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]