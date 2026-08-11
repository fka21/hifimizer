#!/usr/bin/env bash
# Activates the `optimizer` conda environment, then runs whatever command was
# passed to `docker run` (default: hifimizer --help).
#
# NOTE: as of the Dockerfile rewrite this is a convenience, not a requirement.
# /opt/conda/envs/optimizer/bin is baked into the image's PATH, so numpy,
# hifiasm, busco, yak, samtools, minimap2 and sniffles resolve correctly even
# when this script never runs -- which is the case for `docker exec`,
# `docker run --entrypoint ...` and `singularity exec`. Activation is still
# done here because it also sources the environment's activate.d hooks.
#
# `set -u` is deliberately NOT used: conda's own activate scripts reference
# unbound variables (PS1, _CE_M, ...) and would abort the container under it.
set -eo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-optimizer}"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    . /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || \
        echo "entrypoint: could not activate '${CONDA_ENV_NAME}'; " \
             "falling back to the PATH baked into the image" >&2
fi

exec "$@"