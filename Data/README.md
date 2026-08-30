# Data layout

All data and numerical artifacts are grouped by lifecycle stage:

- `01_raw/`: original formulation and Tafel workbooks, raw property tables,
  and Excel archives.
- `02_inputs/`: canonical 97-record dataset, model-ready tables, and posterior
  target inputs.
- `03_outputs/`: generated study and Tafel-refitting outputs.
- `04_results/RESULTS/`: released paper results, quality-control records,
  model workbooks, reports, and data exports.

The top-level `artifact_manifest.json` records the contents of this complete
data tree.
