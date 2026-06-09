#!/usr/bin/env bash
# Create the reproducible conda environment (task m1-t2) and install the package.
set -euo pipefail
ENV_NAME="${1:-govgridmarl}"

if command -v conda >/dev/null 2>&1; then
  conda env create -f environment.yml -n "$ENV_NAME" || conda env update -f environment.yml -n "$ENV_NAME"
  echo "Activate with: conda activate $ENV_NAME"
else
  echo "conda not found — falling back to venv + pip"
  python -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e ".[dev]"
fi
