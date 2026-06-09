#!/usr/bin/env bash
# Reproducible environment setup (task m1-t2).
# Prefers conda (environment.yml); falls back to uv or pip using requirements.lock.
set -euo pipefail
ENV_NAME="${1:-govgridmarl}"

if command -v conda >/dev/null 2>&1; then
  conda env create -f environment.yml -n "$ENV_NAME" \
    || conda env update -f environment.yml -n "$ENV_NAME"
  echo "Done. Activate with: conda activate $ENV_NAME"
elif command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.lock
  uv pip install -e .
  echo "Done. Activated .venv (uv)."
else
  python -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.lock
  pip install -e .
  echo "Done. Activated .venv (pip)."
fi
