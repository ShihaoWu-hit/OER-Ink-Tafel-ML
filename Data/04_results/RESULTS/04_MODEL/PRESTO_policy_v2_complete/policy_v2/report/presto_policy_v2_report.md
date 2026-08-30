# PRESTO policy-only F0.5 mean-margin development sensitivity

The PRESTO-v1 probability OOF predictions are inherited without refit or recalibration. This package changes only a separate cost-sensitive action policy.

| decision_policy | modal_accuracy | balanced_accuracy | macro_f1 | posterior_expected_hard_accuracy | predicted_high_count | modal_high_recall | modal_high_precision | modal_high_false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parent_v1_presto_soft_f2_majority_vote_policy | 0.690722 | 0.543542 | 0.526065 | 0.658866 | 11 | 0.500000 | 0.363636 | 0.078652 |
| presto_policy_only_soft_f05_mean_margin_v2 | 0.742268 | 0.577708 | 0.602513 | 0.706129 | 6 | 0.500000 | 0.666667 | 0.022472 |
| presto_policy_only_soft_f05_vote5_sensitivity_v2 | 0.690722 | 0.551667 | 0.538566 | 0.658098 | 11 | 0.500000 | 0.363636 | 0.078652 |
| presto_policy_only_soft_f05_vote6_sensitivity_v2 | 0.731959 | 0.572500 | 0.587128 | 0.695893 | 7 | 0.500000 | 0.571429 | 0.033708 |

Primary-policy minus parent-v1 accuracy delta: 0.051546, conditional connected-group bootstrap 95% CI [0.009901, 0.109893].

The policy penalty and threshold are selected only from grouped inner-OOF soft targets using soft F0.5 under expected FPR <= 0.20. Repeated OOF alerts use mean(raw score - selected threshold) >= 0; five-vote and six-vote rules are fixed sensitivities.

This exact utility and aggregation rule were chosen after inspecting outer results on the same closed dataset. Nested CV and the conditional bootstrap do not correct that global post-selection. The package is not independently confirmed or scientifically releasable and cannot replace any frozen result.

Method sources: Faletto and Bien (2023), https://proceedings.mlr.press/v202/faletto23a.html; Varma and Simon (2006), https://doi.org/10.1186/1471-2105-7-91.

Artifact QA: `complete_posthoc_global_postselection_development`; probability maximum absolute mismatch: `0.0`.
