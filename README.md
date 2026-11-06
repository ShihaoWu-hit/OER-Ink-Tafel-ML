# OER Tafel / PRESTO code and data

This repository contains the code, input data, and results for the OER Tafel analysis.

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
  09_extended_analysis/       Extended statistical and model analyses
  configs/                    Study, model, property, and fitting contracts
  Ink_Tafel_Study/            Main study package
  Tafel_refitting/            Tafel fitting package

Data/                         All data, outputs, and released results
  01_raw/                     Original workbooks and source tables
  02_inputs/                  Canonical and model-ready inputs
  03_outputs/                 Frozen workflow outputs and fitting outputs
  04_results/RESULTS/         Analysis result tables

```

`Data/01_raw` is the source layer. `Data/02_inputs` contains the 97-record
canonical dataset and model inputs. `Data/03_outputs` contains generated
workflow artifacts. `Data/04_results/RESULTS` contains the released numerical
results, with unique numbered sections from `01` to `08`.

## Extended analysis

The extended analyses use the released posterior draws, model inputs, and
grouped splits. LR baselines use eight inputs. Outputs are saved to the
specified directory.

Use Python 3.12 and install:

```bash
python -m pip install numpy==2.4.6 pandas==3.0.3 scipy==1.17.1 scikit-learn==1.8.0 joblib==1.5.3 matplotlib==3.10.9 PyYAML==6.0.3 optuna==4.9.0 openpyxl==3.1.5
```

Run from the repository root:

```bash
python code/09_extended_analysis/run_analysis.py all --output analysis_runs/full --jobs 3
```

Replace `all` with a stage to run an individual analysis:

| Stage | Analysis |
|---|---|
| `variance` | Posterior variance decomposition |
| `baseline` | Classification baselines and paired comparisons |
| `posterior` | Posterior resolution and High-region membership |
| `four` | Four-region model comparison |
| `boundaries` | Boundary sensitivity |
| `probability` | Probability-model reliance stability |
| `high-focus` | High-focus reliance stability |
| `ale` | Cross-fitted ALE and bootstrap intervals |

Use a separate output directory for each run. `--jobs` sets parallel workers;
`--resume` reuses locally generated fold caches.
