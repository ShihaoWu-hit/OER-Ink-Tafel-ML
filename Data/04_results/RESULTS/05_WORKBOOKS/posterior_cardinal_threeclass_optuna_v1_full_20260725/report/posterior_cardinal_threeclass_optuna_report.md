# Posterior-aware three-class Optuna exploration

This is a post-hoc exploratory three-class task. It does not replace or amend the frozen four-class v4 result.

## Endpoint

The response is the posterior-soft vector `low_30_40_60_like / 120_like / off_cardinal`. Each experimental record remains one ML row; posterior draws are not pseudo-samples.

| class_label       |   effective_posterior_mass |   modal_record_count |   modal_connected_group_count |   mean_probability |   width | width_id   |
|:------------------|---------------------------:|---------------------:|------------------------------:|-------------------:|--------:|:-----------|
| low_30_40_60_like |                    46.7512 |                   55 |                            48 |           0.481972 |     0.2 | width_20   |
| 120_like          |                    16.8087 |                   16 |                            13 |           0.173286 |     0.2 | width_20   |
| off_cardinal      |                    33.44   |                   26 |                            24 |           0.344742 |     0.2 | width_20   |

## Honest nested-CV performance

Optuna selected the feature panel, model family, and hyperparameters using connected-group inner folds only. Outer test records evaluated the complete adaptive procedure.

|   width | width_id   | evaluation_model                 |   n_records |   n_connected_groups |   soft_log_loss |   target_entropy |   excess_cross_entropy_KL |   multiclass_brier |   modal_accuracy |   balanced_accuracy |   macro_f1 |   posterior_expected_hard_accuracy |   oracle_posterior_expected_hard_accuracy |   mean_normalized_predictive_entropy |   uncertain_count |   uncertain_fraction |   full_coverage_three_class_soft_log_loss |   multiclass_brier_mean_sum_squared_error |   gate_soft_log_loss |   gate_brier |   conditional_soft_log_loss |   conditional_brier |   conditional_soft_log_loss_weighted_by_q_cardinal |   conditional_brier_weighted_by_q_cardinal |   mean_observed_cardinal_probability |   mean_predicted_cardinal_probability |
|--------:|:-----------|:---------------------------------|------------:|---------------------:|----------------:|-----------------:|--------------------------:|-------------------:|-----------------:|--------------------:|-----------:|-----------------------------------:|------------------------------------------:|-------------------------------------:|------------------:|---------------------:|------------------------------------------:|------------------------------------------:|---------------------:|-------------:|----------------------------:|--------------------:|---------------------------------------------------:|-------------------------------------------:|-------------------------------------:|--------------------------------------:|
|     0.2 | width_20   | fold_local_threeclass_soft_prior |          97 |                   80 |         1.02302 |          0.64002 |                  0.382999 |           0.232681 |         0.56701  |            0.333333 |   0.241228 |                           0.481972 |                                  0.716311 |                             0.930852 |                97 |             1        |                                   1.02302 |                                  0.232681 |             0.644387 |     0.111524 |                    0.577836 |            0.272264 |                                           0.577836 |                                   0.272264 |                             0.655258 |                              0.655243 |
|     0.2 | width_20   | nested_optuna_selected_pipeline  |          97 |                   80 |         1.01636 |          0.64002 |                  0.376337 |           0.234687 |         0.505155 |            0.31049  |   0.263725 |                           0.458236 |                                  0.716311 |                             0.894264 |                48 |             0.494845 |                                   1.01636 |                                  0.234687 |             0.658824 |     0.124808 |                    0.545636 |            0.24733  |                                           0.545636 |                                   0.24733  |                             0.655258 |                              0.6528   |

The nested selected procedure has descriptive modal accuracy `0.505155` and posterior-expected hard accuracy `0.458236`. Neither quantity was used by Optuna; three-class soft log loss is the primary selection and evaluation metric.

## Feature and model selection stability

```json
{
  "feature_panel": {
    "candidate": "context_core",
    "selection_count": 20,
    "selection_fraction": 0.4,
    "stable": false
  },
  "global_single_best_claim_authorized": false,
  "joint_pipeline": {
    "candidate": "context_core::two_stage_soft_extra_trees",
    "selection_count": 15,
    "selection_fraction": 0.3,
    "stable": false
  },
  "model_family": {
    "candidate": "two_stage_soft_extra_trees",
    "selection_count": 32,
    "selection_fraction": 0.64,
    "stable": true
  },
  "outer_test_metrics_used_to_choose_winner": false,
  "overall_status": "no_stable_joint_winner",
  "selection_basis": "frequency_of_inner_CV_selections_across_outer_training_splits",
  "separate_validation_available": false,
  "smoke_structural_only": false,
  "stability_threshold": 0.5
}
```

## Selected pipeline minus fold-local prior

| metric                                 |   observed_mean |   bootstrap_mean |   bootstrap_median |   ci95_low |   ci95_high |   probability_below_zero |   n_bootstrap |   n_records |   n_connected_groups |
|:---------------------------------------|----------------:|-----------------:|-------------------:|-----------:|------------:|-------------------------:|--------------:|------------:|---------------------:|
| delta_soft_log_loss                    |     -0.00666205 |      -0.0059403  |        -0.00654471 | -0.0469935 |   0.0420653 |                   0.615  |          2000 |          97 |                   80 |
| delta_multiclass_brier                 |      0.00200667 |       0.00228959 |         0.00199112 | -0.0228983 |   0.0303507 |                   0.446  |          2000 |          97 |                   80 |
| delta_posterior_expected_hard_accuracy |     -0.0237358  |      -0.0240012  |        -0.0232122  | -0.066471  |   0.0126562 |                   0.8845 |          2000 |          97 |                   80 |

## Interpretation boundary

The classes are apparent Tafel-slope neighborhoods, not reaction mechanisms. Cross-publication/laboratory prediction is associational and cannot establish a causal formulation effect.
