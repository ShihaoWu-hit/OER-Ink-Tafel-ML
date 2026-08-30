# OER Tafel / PRESTO paper code and data

This release contains the code, input data, frozen outputs, and numerical
results associated with the manuscript and Supporting Information.

## Layout

```text
code/                         All code, packages, tests, and configurations
  01_data_curation/           Data curation and provenance
  02_tafel_refitting/         Tafel curve preprocessing and Bayesian fitting
  03_targets_features/        Target and feature construction
  04_model_training/          Model training and grouped validation
  05_policy_high_slope/       PRESTO/high-slope decision layer
  06_interpretability/        Feature reliance and ALE analyses
  07_runtime_dependencies/    Importable workflow modules
  08_tests/                   Tests
  configs/                    Study, model, property, and fitting contracts
  Ink_Tafel_Study/            Main study package
  Tafel_refitting/            Tafel fitting package

Data/                         All data, outputs, and released results
  01_raw/                     Original workbooks and source tables
  02_inputs/                  Canonical and model-ready inputs
  03_outputs/                 Frozen workflow outputs and fitting outputs
  04_results/RESULTS/         Paper result tables, QA records, and reports

```

`Data/01_raw` is the source layer. `Data/02_inputs` contains the 97-record
canonical dataset and model inputs. `Data/03_outputs` contains generated
workflow artifacts. `Data/04_results/RESULTS` contains the released paper
results, with unique numbered sections from `01` to `08`.

A complete Bayesian refit and grouped OOF retraining remain compute-intensive
optional runs; the released numerical outputs are supplied in `Data/04_results`.
