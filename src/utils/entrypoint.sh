#!/bin/bash --login
set -e
conda activate optimizer
exec "$@"
