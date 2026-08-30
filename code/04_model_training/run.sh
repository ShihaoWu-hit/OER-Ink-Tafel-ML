#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPOSITORY_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "Ink_Tafel_Study requires Python 3.12; found: $("${PYTHON_BIN}" --version 2>&1)" >&2
  exit 2
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {curate|refit|evaluate|report|optimize|feature-develop|cardinal-classify|posterior-cardinal-classify|posterior-cardinal-threeclass-optuna|posterior-cardinal-contiguous-threeclass|cardinal-develop|feature-associate|all} [options]" >&2
  exit 2
fi

export PYTHONPATH="${REPOSITORY_DIR}/code/Ink_Tafel_Study/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m ink_tafel_study.cli "$@"
