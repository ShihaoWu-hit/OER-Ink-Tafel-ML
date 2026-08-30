# Posterior-aware fixed-data feature association

## Bottom line

This is a post-hoc, fixed-dataset associational analysis of all 97 records as one primary population. It does not identify causal or mechanistic effects, and it does not publish a separate 90-record alkaline result.

The audit was restricted to 28 hash-bound registered predictors. 20 passed the X-only descriptive support rule in the all-97 scope; 8 were thin, constant, or otherwise unsupported.

No block passed every registered validation gate; therefore no adjusted effect curve was released.

## Feature support and redundancy

| feature_name                                                      | kind        |   n_nonmissing |   n_distinct |   n_nonzero_groups |   n_within_group_variable_groups | support_status              |
|:------------------------------------------------------------------|:------------|---------------:|-------------:|-------------------:|---------------------------------:|:----------------------------|
| catalyst_active_material_type                                     | categorical |             97 |            3 |                nan |                                4 | thin_category_group_support |
| catalyst_support                                                  | categorical |             97 |            2 |                nan |                                1 | thin_category_group_support |
| catalyst_concentration_mg_mL                                      | numeric     |             97 |           51 |                 80 |                                0 | supported_descriptive       |
| ionomer_to_solvent_ratio                                          | numeric     |             97 |           35 |                 79 |                                0 | supported_descriptive       |
| ionomer_concentration_wt_percent                                  | numeric     |             97 |            2 |                 80 |                                0 | limited_numeric_levels      |
| conductive_agent_present                                          | categorical |             97 |            2 |                nan |                                0 | supported_descriptive       |
| conductive_agent_dose_mg_per_mgcat                                | numeric     |             97 |            8 |                 14 |                                0 | supported_descriptive       |
| solvent_fraction_acetone                                          | numeric     |             97 |            2 |                  1 |                                0 | thin_nonzero_group_support  |
| solvent_fraction_water                                            | numeric     |             97 |           25 |                 55 |                                0 | supported_descriptive       |
| solvent_fraction_ethanol                                          | numeric     |             97 |           17 |                 46 |                                0 | supported_descriptive       |
| solvent_fraction_isopropanol                                      | numeric     |             97 |           15 |                 26 |                                0 | supported_descriptive       |
| solvent_fraction_methanol                                         | numeric     |             97 |            2 |                  2 |                                0 | thin_nonzero_group_support  |
| catalyst_loading_mg_cm2_selected                                  | numeric     |             93 |           55 |                 77 |                                2 | supported_descriptive       |
| electrolyte_identity                                              | categorical |             97 |            5 |                nan |                                1 | thin_category_group_support |
| electrolyte_molarity_M                                            | numeric     |             97 |            4 |                 80 |                                1 | supported_descriptive       |
| rpm                                                               | numeric     |             94 |            8 |                 38 |                                2 | supported_descriptive       |
| scan_rate_mV_s                                                    | numeric     |             95 |            6 |                 78 |                                1 | supported_descriptive       |
| ir_correction_status                                              | categorical |             97 |            2 |                nan |                                0 | supported_descriptive       |
| dispersion_method                                                 | categorical |             96 |            2 |                nan |                                0 | thin_category_group_support |
| sonication_time_min                                               | numeric     |             73 |            8 |                 61 |                                0 | supported_descriptive       |
| drop_volume_uL                                                    | numeric     |             86 |           17 |                 70 |                                0 | supported_descriptive       |
| electrode_area_cm2                                                | numeric     |             85 |            7 |                 70 |                                0 | supported_descriptive       |
| solvent_mixture_property_proxy_viscosity_mPa_s                    | numeric     |             97 |           33 |                 80 |                                0 | supported_descriptive       |
| solvent_mixture_property_proxy_surface_tension_mN_m               | numeric     |             97 |           33 |                 80 |                                0 | supported_descriptive       |
| solvent_mixture_property_proxy_density_kg_m3                      | numeric     |             97 |           33 |                 80 |                                0 | supported_descriptive       |
| solvent_mixture_property_proxy_dielectric_constant                | numeric     |             66 |           19 |                 54 |                                0 | supported_descriptive       |
| solvent_mixture_property_proxy_dielectric_constant_best_available | numeric     |             97 |           33 |                 80 |                                0 | supported_descriptive       |
| solvent_mixture_property_proxy_dielectric_constant_estimated_flag | numeric     |             97 |            2 |                 26 |                                0 | limited_numeric_levels      |

Strong numeric redundancy pairs (all-97 scope):

| feature_a                                           | feature_b                                                         |   spearman_rho | redundancy_status     |
|:----------------------------------------------------|:------------------------------------------------------------------|---------------:|:----------------------|
| solvent_mixture_property_proxy_dielectric_constant  | solvent_mixture_property_proxy_dielectric_constant_best_available |       1        | near_exact_affine     |
| solvent_mixture_property_proxy_surface_tension_mN_m | solvent_mixture_property_proxy_dielectric_constant                |       0.999653 | high_rank_correlation |
| solvent_mixture_property_proxy_density_kg_m3        | solvent_mixture_property_proxy_dielectric_constant                |       0.998959 | high_rank_correlation |
| solvent_mixture_property_proxy_surface_tension_mN_m | solvent_mixture_property_proxy_dielectric_constant_best_available |       0.995123 | high_rank_correlation |
| solvent_mixture_property_proxy_density_kg_m3        | solvent_mixture_property_proxy_dielectric_constant_best_available |       0.99443  | high_rank_correlation |
| solvent_fraction_water                              | solvent_mixture_property_proxy_dielectric_constant                |       0.993012 | high_rank_correlation |
| solvent_fraction_ethanol                            | solvent_mixture_property_proxy_dielectric_constant                |      -0.986666 | high_rank_correlation |
| solvent_fraction_water                              | solvent_mixture_property_proxy_density_kg_m3                      |       0.986206 | high_rank_correlation |
| solvent_mixture_property_proxy_surface_tension_mN_m | solvent_mixture_property_proxy_density_kg_m3                      |       0.985461 | high_rank_correlation |
| solvent_fraction_isopropanol                        | solvent_mixture_property_proxy_dielectric_constant_estimated_flag |       0.976827 | high_rank_correlation |
| solvent_fraction_water                              | solvent_mixture_property_proxy_dielectric_constant_best_available |       0.975881 | high_rank_correlation |
| solvent_fraction_water                              | solvent_mixture_property_proxy_surface_tension_mN_m               |       0.963172 | high_rank_correlation |

## Posterior-aware marginal atlas

The atlas contains 28 scope-feature summaries. Each plot shows one point and the formal posterior 95% interval per record. Numeric Spearman rho or categorical log-scale eta-squared is recomputed across selected posterior draws. No smoother, p-value ranking, or target-driven feature selection is used.

## Fixed block-wise grouped comparison

Validation used 5-fold connected-group holdout repeated 10 times. All preprocessing and feature transformations were learned inside each training fold.

| contrast_id                          | model           |   observed_delta |   ci95_low |   ci95_high |   repeat_improvement_fraction |   posterior_draw_improvement_fraction |
|:-------------------------------------|:----------------|-----------------:|-----------:|------------:|------------------------------:|--------------------------------------:|
| catalyst_concentration_minus_context | bayesian_linear |       0.0349305  | -0.380711  |    0.386608 |                           0.5 |                                  0.44 |
| catalyst_concentration_minus_context | gam_low_df      |       0.135871   | -0.306666  |    0.54884  |                           0.2 |                                  0.22 |
| conductive_minus_context             | bayesian_linear |       0.214835   | -0.297147  |    0.824129 |                           0.3 |                                  0.1  |
| conductive_minus_context             | gam_low_df      |       0.157864   | -0.165493  |    0.581216 |                           0.2 |                                  0.06 |
| context_minus_prior                  | bayesian_linear |      -0.0143544  | -0.918313  |    0.930361 |                           0.1 |                                  0.52 |
| context_minus_prior                  | gam_low_df      |      -0.356531   | -1.3019    |    0.559079 |                           0.7 |                                  0.81 |
| ethanol_isopropanol_minus_context    | bayesian_linear |       0.00542957 | -0.347075  |    0.30777  |                           0.8 |                                  0.47 |
| ethanol_isopropanol_minus_context    | gam_low_df      |       0.453551   | -0.213474  |    1.22284  |                           0   |                                  0.03 |
| ionomer_ratio_minus_context          | bayesian_linear |      -0.0216165  | -0.383354  |    0.315384 |                           0.8 |                                  0.5  |
| ionomer_ratio_minus_context          | gam_low_df      |       0.131703   | -0.135951  |    0.380553 |                           0.2 |                                  0.12 |
| polarity_minus_context               | bayesian_linear |      -0.0993724  | -1.12621   |    0.906849 |                           0.6 |                                  0.64 |
| polarity_minus_context               | gam_low_df      |       0.269551   | -0.549212  |    1.12534  |                           0.1 |                                  0.22 |
| viscosity_minus_context              | bayesian_linear |       0.0739913  | -0.358873  |    0.420372 |                           0.5 |                                  0.31 |
| viscosity_minus_context              | gam_low_df      |       0.227479   | -0.0509642 |    0.540556 |                           0.1 |                                  0.06 |
| water_organic_minus_context          | bayesian_linear |       0.0888893  | -0.573246  |    0.751179 |                           0.6 |                                  0.39 |
| water_organic_minus_context          | gam_low_df      |       0.266499   | -0.16914   |    0.706676 |                           0.1 |                                  0.11 |

Primary-scope aggregate metrics:

| panel                               | model           |   posterior_expected_mae_mV_dec |   posterior_expected_mae_log10 |   posterior_expected_nll_log10 |   rmse_mV_dec |   spearman |         r2 |
|:------------------------------------|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------:|-----------:|-----------:|
| context_core                        | bayesian_linear |                         34.649  |                       0.182606 |                       0.144156 |       53.5733 | -0.168664  | -0.0704811 |
| context_core                        | gam_low_df      |                         34.3069 |                       0.180736 |                       0.141236 |       53.3636 | -0.13329   | -0.0621188 |
| context_plus_catalyst_concentration | bayesian_linear |                         34.684  |                       0.182888 |                       0.139819 |       53.5412 | -0.262516  | -0.0692007 |
| context_plus_catalyst_concentration | gam_low_df      |                         34.4427 |                       0.181619 |                       0.137577 |       53.2741 | -0.182064  | -0.0585603 |
| context_plus_conductive_sensitivity | bayesian_linear |                         34.8639 |                       0.183666 |                       0.157238 |       53.6354 | -0.110908  | -0.0729678 |
| context_plus_conductive_sensitivity | gam_low_df      |                         34.4647 |                       0.181587 |                       0.15266  |       53.4676 | -0.0824778 | -0.0662635 |
| context_plus_ethanol_isopropanol    | bayesian_linear |                         34.6545 |                       0.182719 |                       0.135575 |       53.5543 | -0.267277  | -0.0697218 |
| context_plus_ethanol_isopropanol    | gam_low_df      |                         34.7604 |                       0.182966 |                       0.157616 |       53.5033 | -0.0886189 | -0.0676873 |
| context_plus_ionomer_ratio          | bayesian_linear |                         34.6274 |                       0.18257  |                       0.135374 |       53.5112 | -0.274246  | -0.0680023 |
| context_plus_ionomer_ratio          | gam_low_df      |                         34.4386 |                       0.18149  |                       0.133691 |       53.3964 | -0.227905  | -0.0634232 |
| context_plus_polarity               | bayesian_linear |                         34.5497 |                       0.181399 |                       0.138355 |       53.1251 |  0.0747586 | -0.0526469 |
| context_plus_polarity               | gam_low_df      |                         34.5764 |                       0.181792 |                       0.1478   |       53.2553 |  0.0148597 | -0.0578108 |
| context_plus_viscosity              | bayesian_linear |                         34.723  |                       0.183124 |                       0.138511 |       53.6    | -0.299482  | -0.0715488 |
| context_plus_viscosity              | gam_low_df      |                         34.5343 |                       0.18204  |                       0.147358 |       53.5676 | -0.232048  | -0.070255  |
| context_plus_water_organic          | bayesian_linear |                         34.7379 |                       0.182771 |                       0.138756 |       53.2672 | -0.038188  | -0.0582847 |
| context_plus_water_organic          | gam_low_df      |                         34.5734 |                       0.182052 |                       0.145418 |       53.3659 | -0.0963643 | -0.0622106 |
| prior                               | bayesian_linear |                         34.6634 |                       0.182723 |                       0.100026 |       53.2012 | -0.593081  | -0.0556629 |
| prior                               | gam_low_df      |                         34.6634 |                       0.182723 |                       0.100026 |       53.2012 | -0.593081  | -0.0556629 |

## Release decisions

| contrast_id                          | block_id                       | block_gate_passed   | panel_effect_eligible   | effect_release_pass   | reason                                                                                                                                                                                                                                                        |
|:-------------------------------------|:-------------------------------|:--------------------|:------------------------|:----------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| context_minus_prior                  | context                        | False               | True                    | False                 | failed: primary_ci95_high_below_zero|augmented_vs_prior_ci95_high_below_zero|minimum_repeat_improvement_fraction|minimum_posterior_draw_improvement_fraction|corroborating_gam_ci95_high_below_zero                                                           |
| catalyst_concentration_minus_context | catalyst_concentration         | False               | True                    | False                 | failed: primary_estimate_below_zero|primary_ci95_high_below_zero|augmented_vs_prior_ci95_high_below_zero|minimum_repeat_improvement_fraction|minimum_posterior_draw_improvement_fraction|corroborating_gam_ci95_high_below_zero|holm_one_sided_primary_blocks |
| ionomer_ratio_minus_context          | ionomer_to_solvent_ratio       | False               | True                    | False                 | failed: primary_ci95_high_below_zero|augmented_vs_prior_ci95_high_below_zero|minimum_posterior_draw_improvement_fraction|corroborating_gam_ci95_high_below_zero|holm_one_sided_primary_blocks                                                                 |
| water_organic_minus_context          | water_to_organic_balance       | False               | True                    | False                 | failed: primary_estimate_below_zero|primary_ci95_high_below_zero|augmented_vs_prior_ci95_high_below_zero|minimum_repeat_improvement_fraction|minimum_posterior_draw_improvement_fraction|corroborating_gam_ci95_high_below_zero|holm_one_sided_primary_blocks |
| ethanol_isopropanol_minus_context    | ethanol_to_isopropanol_balance | False               | True                    | False                 | failed: primary_estimate_below_zero|primary_ci95_high_below_zero|augmented_vs_prior_ci95_high_below_zero|minimum_posterior_draw_improvement_fraction|corroborating_gam_ci95_high_below_zero|holm_one_sided_primary_blocks                                     |
| viscosity_minus_context              | solvent_viscosity              | False               | True                    | False                 | failed: primary_estimate_below_zero|primary_ci95_high_below_zero|augmented_vs_prior_ci95_high_below_zero|minimum_repeat_improvement_fraction|minimum_posterior_draw_improvement_fraction|corroborating_gam_ci95_high_below_zero|holm_one_sided_primary_blocks |
| polarity_minus_context               | solvent_polarity               | False               | True                    | False                 | failed: primary_ci95_high_below_zero|augmented_vs_prior_ci95_high_below_zero|minimum_repeat_improvement_fraction|minimum_posterior_draw_improvement_fraction|corroborating_gam_ci95_high_below_zero|holm_one_sided_primary_blocks                             |
| conductive_minus_context             | conductive_agent               | False               | False                   | False                 | failed: primary_estimate_below_zero|primary_ci95_high_below_zero|augmented_vs_prior_ci95_high_below_zero|minimum_repeat_improvement_fraction|minimum_posterior_draw_improvement_fraction|corroborating_gam_ci95_high_below_zero                               |

## Posterior policy

Training used one equally weighted row per record and the log10 formal posterior median. Posterior draws were used only for scoring/uncertainty propagation; 0.227% of draws were nonpositive, retained for original-scale MAE and excluded from log-scale metrics.

## Interpretation boundary

A passed block means reproducible out-of-connected-group incremental predictive association under this frozen workflow. A released curve is an adjusted associational description indexed by training-fold feature quantile, restricted to the preregistered joint-support common-record set. It is not a fixed physical-dose intervention. Neither result proves a reaction mechanism, rate-determining step, or causal formulation effect.
