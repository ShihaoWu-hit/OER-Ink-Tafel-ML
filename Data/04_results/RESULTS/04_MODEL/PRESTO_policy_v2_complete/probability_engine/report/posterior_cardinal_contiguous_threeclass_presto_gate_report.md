# PRESTO-inspired high-off gate development sensitivity

## Outcome

This package evaluates a probability-valid PRESTO-inspired fused continuation-ratio gate and a separately tuned recall-oriented decision policy. Probability quality and high-recall decisions are not conflated.

| evaluation_model | soft_log_loss | multiclass_brier | modal_accuracy | balanced_accuracy | macro_f1 | posterior_expected_hard_accuracy | gate_soft_log_loss | predicted_high_count | modal_high_recall | modal_high_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multiscale_soft_gower | 0.821016 | 0.280094 | 0.711340 | 0.424375 | 0.417075 | 0.672611 | 0.301883 | 0 | 0.000000 | 0.000000 |
| fold_local_contiguous_threeclass_soft_prior | 0.864023 | 0.304044 | 0.659794 | 0.333333 | 0.265010 | 0.641460 | 0.315671 | 0 | 0.000000 | 0.000000 |
| presto_high_gate_direct_gower_conditional | 0.819451 | 0.272466 | 0.711340 | 0.424375 | 0.417075 | 0.672611 | 0.300266 | 0 | 0.000000 | 0.000000 |
| presto_inspired_fused_continuation_ratio | 0.847855 | 0.294320 | 0.659794 | 0.333333 | 0.265010 | 0.641460 | 0.300266 | 0 | 0.000000 | 0.000000 |

## Recall-oriented policy

Each policy threshold was selected using only grouped inner-OOF soft labels inside its outer-training split. PRESTO and the direct-Gower comparator received the identical soft-F2 utility and expected-FPR <= 0.20 constraint.

| decision_policy | modal_accuracy | balanced_accuracy | macro_f1 | posterior_expected_hard_accuracy | predicted_high_count | modal_high_recall | modal_high_precision | modal_high_false_positive_rate | posterior_expected_high_recall | posterior_expected_high_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_gower_inner_oof_soft_f2_fpr20_policy | 0.597938 | 0.399375 | 0.373741 | 0.577369 | 17 | 0.250000 | 0.117647 | 0.168539 | 0.257834 | 0.137162 |
| presto_inner_oof_soft_f2_fpr20_policy | 0.690722 | 0.543542 | 0.526065 | 0.658866 | 11 | 0.500000 | 0.363636 | 0.078652 | 0.459882 | 0.378091 |

## Paired uncertainty

PRESTO-hybrid minus direct-Gower soft-log-loss delta: -0.001565, connected-group bootstrap 95% CI [-0.071018, 0.066893].

PRESTO-policy minus direct-policy modal high-recall delta: 0.250000, connected-group bootstrap 95% CI [-0.333333, 0.750000].

## Model definition

The model uses one fold-local X-only Gower novelty score from panel `context_plus_polarity` and the mean distance to 5 nearest records outside the same connected group. It jointly models advancement from low to 120/high and from 120/high to high, shrinking adjacent transition slopes together.

This is not the paper's exact cumulative-odds PRESTO. Declared deviations are the continuation-ratio link, smooth-L1 penalty, posterior-soft-label extension, and a small fixed ridge stabilizer. The parameterization was chosen because it cannot produce negative middle-class probabilities.

Penalty and threshold selections by outer split:

| repeat | fold | selected_penalty | selected_presto_threshold | selected_direct_threshold |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0.003000 | 0.175000 | 0.200000 |
| 0 | 1 | 0.010000 | 0.150000 | 0.125000 |
| 0 | 2 | 0.003000 | 0.125000 | 0.150000 |
| 0 | 3 | 0.300000 | 1.000000 | 0.125000 |
| 0 | 4 | 0.010000 | 0.150000 | 0.150000 |
| 1 | 0 | 0.030000 | 0.125000 | 0.125000 |
| 1 | 1 | 1.000000 | 1.000000 | 0.150000 |
| 1 | 2 | 0.030000 | 0.100000 | 0.100000 |
| 1 | 3 | 0.003000 | 0.275000 | 0.125000 |
| 1 | 4 | 0.003000 | 0.150000 | 0.150000 |
| 2 | 0 | 0.003000 | 0.200000 | 0.200000 |
| 2 | 1 | 0.030000 | 0.125000 | 0.150000 |
| 2 | 2 | 0.030000 | 0.150000 | 0.125000 |
| 2 | 3 | 0.300000 | 1.000000 | 0.125000 |
| 2 | 4 | 0.030000 | 0.125000 | 0.100000 |
| 3 | 0 | 0.010000 | 0.175000 | 0.150000 |
| 3 | 1 | 0.030000 | 0.150000 | 0.150000 |
| 3 | 2 | 0.010000 | 0.175000 | 0.150000 |
| 3 | 3 | 0.030000 | 0.150000 | 0.175000 |
| 3 | 4 | 0.030000 | 0.125000 | 0.100000 |
| 4 | 0 | 0.030000 | 0.125000 | 0.150000 |
| 4 | 1 | 0.003000 | 0.175000 | 0.150000 |
| 4 | 2 | 0.003000 | 0.150000 | 0.125000 |
| 4 | 3 | 1.000000 | 1.000000 | 0.100000 |
| 4 | 4 | 0.003000 | 0.175000 | 0.150000 |
| 5 | 0 | 0.003000 | 0.150000 | 0.150000 |
| 5 | 1 | 0.010000 | 0.125000 | 0.150000 |
| 5 | 2 | 0.030000 | 0.175000 | 0.125000 |
| 5 | 3 | 0.300000 | 1.000000 | 0.125000 |
| 5 | 4 | 0.010000 | 0.225000 | 0.175000 |
| 6 | 0 | 1.000000 | 1.000000 | 0.125000 |
| 6 | 1 | 0.030000 | 0.125000 | 0.125000 |
| 6 | 2 | 0.010000 | 0.200000 | 0.200000 |
| 6 | 3 | 0.010000 | 0.250000 | 0.125000 |
| 6 | 4 | 0.003000 | 0.125000 | 0.125000 |
| 7 | 0 | 0.100000 | 0.100000 | 0.100000 |
| 7 | 1 | 1.000000 | 1.000000 | 0.125000 |
| 7 | 2 | 0.300000 | 1.000000 | 0.150000 |
| 7 | 3 | 0.003000 | 0.150000 | 0.200000 |
| 7 | 4 | 0.003000 | 0.175000 | 0.175000 |
| 8 | 0 | 0.030000 | 0.150000 | 0.125000 |
| 8 | 1 | 0.003000 | 0.250000 | 0.125000 |
| 8 | 2 | 0.300000 | 1.000000 | 0.175000 |
| 8 | 3 | 0.300000 | 1.000000 | 0.150000 |
| 8 | 4 | 0.030000 | 0.150000 | 0.100000 |
| 9 | 0 | 0.030000 | 0.100000 | 0.100000 |
| 9 | 1 | 0.003000 | 0.200000 | 0.225000 |
| 9 | 2 | 0.030000 | 0.125000 | 0.150000 |
| 9 | 3 | 0.030000 | 0.125000 | 0.150000 |
| 9 | 4 | 0.003000 | 0.250000 | 0.175000 |

## Interpretation boundary

The architecture, novelty coordinate, candidate grid, and decision utility were developed after inspecting results on the same closed 97-record dataset. The current nested folds protect outer-test rows from within-run tuning, but they do not account for this global development adaptivity. The result is post-selection development evidence, not independent confirmation.

A recall-policy gain is a cost-sensitive operating-point change. It must be read together with false positives, precision, overall accuracy, and probability scores. It is not evidence that formulation features cause high slopes.

Primary method sources:

- Faletto and Bien (2023), PRESTO: https://proceedings.mlr.press/v202/faletto23a.html
- Author implementation: https://github.com/gregfaletto/presto

Artifact QA status: `complete_posthoc_adaptive_development`; scientifically releasable: `false`.
