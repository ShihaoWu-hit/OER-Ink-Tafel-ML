# Contiguous posterior-aware three-class sensitivity

**Post-hoc exploratory changed endpoint; not an independent confirmation and not a replacement for v4 or classic v1.**

## Response definition

- `low_60_side`: slope < 84.85281374 mV dec-1
- `120_side`: 84.85281374 <= slope <= 144.0 mV dec-1
- `high_off`: slope > 144.0 mV dec-1

| class_label   |   effective_posterior_mass |   modal_record_count |   modal_connected_group_count |   mean_probability |
|:--------------|---------------------------:|---------------------:|------------------------------:|-------------------:|
| low_60_side   |                   62.2216  |                   64 |                            55 |          0.64146   |
| 120_side      |                   25.7348  |                   25 |                            22 |          0.265307  |
| high_off      |                    9.04363 |                    8 |                             6 |          0.0932332 |

## Nested connected-group OOF performance

- Stack soft log loss: 0.854774
- Prior soft log loss: 0.864023
- Stack modal accuracy: 0.680412
- Prior modal accuracy: 0.659794
- Posterior-expected oracle: 0.857933

## Stagewise mean stack weights

| stage       | candidate          |   mean_weight |   median_weight |   minimum_weight |   maximum_weight |   nonzero_fraction |   dominant_count |   n_outer_splits |
|:------------|:-------------------|--------------:|----------------:|-----------------:|-----------------:|-------------------:|-----------------:|-----------------:|
| conditional | soft_gower         |   0.452231    |     0.430077    |                0 |      1           |               0.92 |               20 |               50 |
| conditional | gaussian_prior_MAP |   0.301921    |     0.222881    |                0 |      1           |               0.66 |               18 |               50 |
| conditional | soft_extra_trees   |   0.180145    |     5.27754e-17 |                0 |      0.982119    |               0.42 |               10 |               50 |
| conditional | soft_catboost      |   0.0657035   |     4.70518e-18 |                0 |      0.930324    |               0.26 |                2 |               50 |
| conditional | prior              |   1.68357e-17 |     1.94142e-18 |                0 |      1.41082e-16 |               0    |                0 |               50 |
| gate        | soft_gower         |   0.620123    |     0.753331    |                0 |      1           |               0.84 |               31 |               50 |
| gate        | soft_catboost      |   0.230068    |     5.20417e-18 |                0 |      1           |               0.42 |               13 |               50 |
| gate        | gaussian_prior_MAP |   0.142235    |     0           |                0 |      1           |               0.28 |                6 |               50 |
| gate        | soft_extra_trees   |   0.00757521  |     0           |                0 |      0.191472    |               0.1  |                0 |               50 |
| gate        | prior              |   2.76085e-17 |     1.16496e-18 |                0 |      4.44089e-16 |               0    |                0 |               50 |

## Most frequently inner-selected panel within each family

| stage       | model_family       | panel                               |   selected_outer_splits |   mean_selected_alpha |   median_selected_alpha |   mean_inner_oof_soft_log_loss |   mean_stack_weight |   nonzero_stack_weight_fraction |   dominant_stack_count |   selection_fraction_within_family | most_frequently_selected_within_family   |
|:------------|:-------------------|:------------------------------------|------------------------:|----------------------:|------------------------:|-------------------------------:|--------------------:|--------------------------------:|-----------------------:|-----------------------------------:|:-----------------------------------------|
| conditional | gaussian_prior_MAP | context_plus_catalyst_concentration |                      19 |            0.617374   |                0.586328 |                       0.576821 |            0.284114 |                        0.684211 |                      6 |                               0.38 | True                                     |
| conditional | soft_catboost      | context_core                        |                      21 |            0.375201   |                0.430498 |                       0.589179 |            0.103142 |                        0.285714 |                      2 |                               0.42 | True                                     |
| conditional | soft_extra_trees   | context_plus_catalyst_concentration |                      12 |            0.673035   |                0.700121 |                       0.572852 |            0.284444 |                        0.5      |                      4 |                               0.24 | True                                     |
| conditional | soft_extra_trees   | context_plus_ethanol_isopropanol    |                      12 |            0.767068   |                0.802802 |                       0.573693 |            0.125617 |                        0.5      |                      1 |                               0.24 | True                                     |
| conditional | soft_gower         | context_core                        |                      20 |            0.562336   |                0.613017 |                       0.570674 |            0.468217 |                        0.85     |                      8 |                               0.4  | True                                     |
| gate        | gaussian_prior_MAP | context_core                        |                      25 |            0.170231   |                0        |                       0.310794 |            0.127463 |                        0.16     |                      3 |                               0.5  | True                                     |
| gate        | soft_catboost      | context_core                        |                      22 |            0.0926749  |                0        |                       0.304421 |            0.105256 |                        0.136364 |                      3 |                               0.44 | True                                     |
| gate        | soft_extra_trees   | context_core                        |                      28 |            0.00294644 |                0        |                       0.309806 |            0        |                        0        |                      0 |                               0.56 | True                                     |
| gate        | soft_gower         | context_plus_ionomer_ratio          |                      39 |            0.600188   |                0.620725 |                       0.294917 |            0.723639 |                        0.948718 |                     29 |                               0.78 | True                                     |

## Paired connected-group bootstrap

| metric                                 |   observed_mean |   bootstrap_mean |   bootstrap_median |   ci95_low |   ci95_high |   probability_below_zero |   n_bootstrap |   n_records |   n_connected_groups |
|:---------------------------------------|----------------:|-----------------:|-------------------:|-----------:|------------:|-------------------------:|--------------:|------------:|---------------------:|
| delta_soft_log_loss                    |     -0.00924975 |      -0.00921223 |        -0.00939745 | -0.0460907 |   0.0335456 |                   0.6765 |          2000 |          97 |                   80 |
| delta_multiclass_brier                 |     -0.00682469 |      -0.00689079 |        -0.00704045 | -0.034167  |   0.020847  |                   0.68   |          2000 |          97 |                   80 |
| delta_posterior_expected_hard_accuracy |      0.00911469 |       0.00943566 |         0.0101171  | -0.0377962 |   0.0503414 |                   0.3105 |          2000 |          97 |                   80 |

## Interpretation boundary

The 0.70 comparison is descriptive. Hard accuracy, posterior argmax labels, random record splits, and outer-test results were not used to tune labels, features, alpha, model families, or stack weights. This endpoint intentionally differs from the classic-neighbourhood task, so absolute accuracy is not a same-task improvement claim.
