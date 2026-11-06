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
  revision_analysis/         Additional manuscript and SI analyses

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

## Revision analyses

The additional analyses use the released posterior draws, model inputs, and
grouped splits. LR baselines use eight inputs. Outputs are saved to the
specified directory.

Use Python 3.12 and install:

```bash
python -m pip install numpy==2.4.6 pandas==3.0.3 scipy==1.17.1 scikit-learn==1.8.0 joblib==1.5.3 matplotlib==3.10.9 PyYAML==6.0.3 optuna==4.9.0 openpyxl==3.1.5
```

Run from the repository root:

```bash
python code/revision_analysis/run_revision.py all --output analysis_runs/full --jobs 3
```

Replace `all` with a stage to run an individual analysis:

| Stage | Analysis | Paper location |
|---|---|---|
| `variance` | Posterior variance decomposition | Fig. S1 |
| `baseline` | Classification baselines and paired comparisons | Fig. 3; Table S11 |
| `posterior` | Posterior resolution and High-region membership | Figs. S99-S100, S104 |
| `four` | Four-region model comparison | Fig. S101 |
| `boundaries` | Boundary sensitivity | Fig. S102 |
| `probability` | Probability-model reliance stability | Fig. S103 |
| `high-focus` | High-focus reliance stability | Fig. S104 |
| `ale` | Cross-fitted ALE and bootstrap intervals | Fig. 4; S4, S7.4 |

Use a separate output directory for each run. `--jobs` sets parallel workers;
`--resume` reuses locally generated fold caches.
