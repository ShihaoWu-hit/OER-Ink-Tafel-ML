# Individual feature importance for PRESTO_policy_v2_complete

The active summary reports each of the final model's nine logical inputs
separately. Grouped importance and panel-selection frequency were archived.

## Recommended primary ranking: probability reliance

The probability ranking uses stage-aware permutation on untouched outer-test
records. Outer-training data, fold-local preprocessing, fitted models,
penalties, and policy thresholds remain fixed. Importance is
`permuted soft log loss - baseline soft log loss`; positive values mean that
the fitted probability engine relied on the feature.

| Rank | Feature | Delta soft log loss | 95% connected-group CI |
|---:|---|---:|---:|
| 1 | Rotation rate | 0.024832 | [-0.011295, 0.056234] |
| 2 | Catalyst family | 0.013147 | [-0.011272, 0.035422] |
| 3 | iR compensation | 0.011899 | [-0.016727, 0.034967] |
| 4 | Solvent polarity coordinate | 0.008805 | [-0.012196, 0.033462] |
| 5 | Electrolyte scope | 0.008726 | [-0.005100, 0.026718] |
| 6 | Electrolyte molarity | 0.001623 | [-0.014677, 0.016220] |
| 7 | Catalyst loading | 0.001295 | [-0.006981, 0.010325] |
| 8 | Ionomer / solvent ratio | -0.003890 | [-0.020126, 0.011342] |
| 9 | Scan rate | -0.003993 | [-0.011658, 0.002751] |

All nine intervals cross zero. The point-estimate order is a descriptive
ranking; it does not identify a stable full-probability winner.

## Operational high-alert policy reliance

For the thresholded policy, importance is
`baseline posterior-soft F0.5 - permuted F0.5`. This is a different estimand,
so it must not be merged with the probability ranking.

| Rank | Feature | Delta soft F0.5 | 95% connected-group CI |
|---:|---|---:|---:|
| 1 | Solvent polarity coordinate | 0.400470 | [0.106120, 0.663195] |
| 2 | Electrolyte scope | 0.361150 | [0.014861, 0.739494] |
| 3 | Catalyst family | 0.249706 | [0.029356, 0.539394] |
| 4 | iR compensation | 0.239944 | [0.000567, 0.557160] |
| 5 | Rotation rate | 0.204371 | approximately [0, 0.493977] |
| 6 | Scan rate | 0.127219 | [0.002355, 0.227270] |
| 7 | Catalyst loading | 0.075315 | [0, 0.241525] |
| 8 | Electrolyte molarity | 0.034113 | [-0.187808, 0.267991] |
| 9 | Ionomer / solvent ratio | 0 | structurally zero |

This second ranking is conditional on the post-hoc policy and only six
baseline high alerts. Ionomer ratio is zero here because it enters only the
low-versus-120 conditional branch, never the high-alert gate.

## What can be said about ink formulation

Solvent polarity is the largest operational policy-reliance point estimate,
and its conditional interval is positive. That supports the narrow statement
that the selected high-alert policy depends on its polarity input. It does not
show that ink formulation as a whole has a confirmed physical or causal effect:

- full-coverage probability reliance for polarity still crosses zero;
- all registered formulation descriptors have zero within-connected-group
  variation, so the evidence is between-group;
- the Gower design gives the singleton polarity block weight 0.5 while the
  seven-feature context block jointly receives 0.5, so the large policy
  reliance partly reflects a frozen architectural choice;
- policy-v2 was globally selected after inspecting this same closed dataset.

The polarity coordinate is one actual model feature formed fold-locally from
surface tension, density, and dielectric. Splitting those correlated raw
properties into three fitted-feature importances would change the final model.

## Why permutation + ALE, not SHAP

The final pipeline combines a PRESTO-inspired continuation-ratio gate, Gower
novelty, a multiscale Gower conditional branch, repeated grouped CV, and a
thresholded policy. It has no single natural TreeSHAP or linear-SHAP explainer.
KernelSHAP would require an arbitrary background distribution and can create
off-support combinations in this 97-record correlated dataset.

Outer-test permutation therefore remains the primary global reliance method.
Cross-fitted ALE is retained only to show response shape for polarity and
ionomer ratio; ALE is not an importance score.

## Files

- `feature_importance_summary.csv`: compact two-ranking handoff.
- `presto_policy_v2_feature_reliance.csv`: full 9-feature audit.
- `figures/presto_policy_v2_stage_aware_permutation_reliance.png`: both
  reliance estimands with uncertainty.
- `figures/presto_cross_fitted_ale.png`: supported response shapes.
- `tables/`: replicate, connected-group bootstrap, record-level, and ALE
  audit tables.
- `run_feature_importance.py`, `run_cross_fitted_ale.py`: deterministic
  reproduction code.
- `qa.json`: permutation source hashes and numerical reproduction checks.
- `ale_qa.json`: ALE support, bootstrap, simplex, and refit checks.

None of these results authorize causal, mechanistic, or independently
confirmed feature claims.
