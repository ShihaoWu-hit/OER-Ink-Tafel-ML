#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Create the workspace .venv before running." >&2
  exit 2
fi

export PYTHONPATH="${WORKSPACE_DIR}/code/Tafel_refitting/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m tafel_refit.cli run \
  --config "${WORKSPACE_DIR}/code/Tafel_refitting/config/default.json" \
  --workbook "${WORKSPACE_DIR}/Data/01_raw/tafel_refitting/tafel_curve_source.xlsx" \
  --mapping-csv "${WORKSPACE_DIR}/Data/01_raw/ink_provenance.csv" \
  --canonical-csv "${WORKSPACE_DIR}/Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv" \
  --output-root "${WORKSPACE_DIR}/Data/03_outputs/tafel_refitting/outputs" \
  "$@"
