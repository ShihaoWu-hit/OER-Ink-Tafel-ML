from pathlib import Path

from ink_tafel_study.config import load_config


def test_registered_protocol_is_locked() -> None:
    config = load_config()
    assert config.value("study.causal_claim_allowed") is False
    assert config.value("targets.fixed_current_mA_cm2") == 2.0
    assert config.value("targets.numeric_minimum_branch_forbidden") is True
    assert config.value("validation.outer_folds") == 5
    assert config.value("validation.outer_repeats") == 10
    assert config.value("validation.inner_folds") == 4
    assert config.value("feature_panels.primary_comparison.reference") == "context"
    assert config.value("feature_panels.primary_comparison.augmented") == "context_plus_ink"
    assert {
        "context_plus_ink_amounts",
        "context_plus_property_proxy_core",
        "context_plus_property_proxy_4p_source_backed",
        "context_plus_property_proxy_4p_estimated",
    }.issubset(set(config.value("feature_panels.enabled")))
    assert (
        config.value("feature_panels.property_protocol_amendment.timing")
        == "before_formal_outer_results"
    )
    assert (
        config.value(
            "feature_panels.property_protocol_amendment.property_increment_reference"
        )
        == "context_plus_ink_amounts"
    )
    assert (
        config.value(
            "feature_panels.property_protocol_amendment.property_representation_reference"
        )
        == "context_plus_ink"
    )
    assert (
        config.value("interpretation.importance_panel")
        == "context_plus_property_proxy_4p_estimated"
    )
    assert config.value("curation.source_mode") == "canonical_97"
    assert config.value("curation.expected_curves") == 97
    assert config.value("curation.expected_evidence_curves") == 97
    assert config.value("study.version") == "7.1.0-ir-semantic-revision"
    assert (
        config.value("feature_panels.solvent_composition_amendment.decision")
        == "include_methanol_as_fifth_fraction"
    )
    assert (
        config.value(
            "feature_panels.solvent_composition_amendment.context_plus_ink_feature_count"
        )
        == 18
    )
    assert config.value("targets.primary") == "b_bayesian_local_line_mV_dec"
    assert config.value("targets.expected_primary_eligible") == 97
    assert config.value("targets.expected_primary_alkaline_eligible") == 90
    assert config.value("targets.expected_low_branch_eligible") == 97
    assert config.value("targets.expected_single_regime_eligible") == 0
    assert config.value("targets.overpotential_fit_policy.eta_min_V") == 0.15
    assert (
        config.value("targets.overpotential_fit_policy.upper_selection_rule")
        == "first_endpoint_meeting_frozen_support_and_linearity_gates"
    )
    assert (
        config.value("targets.overpotential_fit_policy.common_current_span_required")
        is False
    )
    assert config.value("targets.exact_2mA_policy.included_in_ml") is False
    assert config.value("interpretation.outer_test_target_use") == "posthoc_metric_only"
    assert config.value("interpretation.mechanistic_claim_allowed") is False
    assert config.value("performance_optimization.target") == "b_bayesian_local_line_mV_dec"
    assert config.value("performance_optimization.outer_repeats") == 10
    assert config.value("performance_optimization.outer_folds") == 5
    assert config.value("performance_optimization.inner_folds") == 4
    assert config.value("performance_optimization.candidate_budget") == 96
    assert (
        config.value("performance_optimization.require_catboost_candidate_for_full")
        is True
    )
    assert (
        config.value("performance_optimization.selection_metric")
        == "pooled_inner_grouped_cv_mae_original_mV_dec"
    )
    assert config.value("performance_optimization.outer_results_may_not_select_fixed_pipeline") is True
    assert config.value("feature_development.target") == "b_bayesian_local_line_mV_dec"
    assert config.value("feature_development.primary_scope") == "all"
    assert config.value("feature_development.sensitivity_scope") == "alkaline"
    assert config.value("feature_development.outer_repeats") == 10
    assert config.value("feature_development.outer_folds") == 5
    assert config.value("feature_development.inner_folds") == 4
    assert config.value("feature_development.target_transform") == "log10"
    assert config.value("feature_development.fixed_rbf_svr") == {
        "C": 1.0,
        "epsilon": 0.05,
        "gamma": 0.1,
    }
    assert config.value("feature_development.near_tie_tolerance_mV_dec") == 0.5
    assert config.value("feature_development.stable_selection_minimum_outer_folds") == 30
    assert config.value("cardinal_classification.status") == "exploratory_posthoc"
    assert config.value("cardinal_classification.mechanistic_claim_allowed") is False
    assert (
        config.value("cardinal_classification.label_scheme.id")
        == "classic_hierarchical_band20_v2"
    )
    assert config.value("cardinal_classification.label_scheme.anchors_mV_dec") == [
        30.0,
        40.0,
        60.0,
        120.0,
    ]
    assert (
        config.value("cardinal_classification.label_scheme.relative_half_width")
        == 0.20
    )
    assert config.value(
        "cardinal_classification.label_scheme.training_sensitivity_half_widths"
    ) == [0.15, 0.20, 0.25]
    assert config.value(
        "cardinal_classification.label_scheme.supervised_class_order"
    ) == [
        "low_30_40_like",
        "60_like",
        "120_like",
        "off_cardinal",
    ]
    assert (
        config.value(
            "cardinal_classification.label_scheme.low_anchor_subdivision.status"
        )
        == "descriptive_only_support_gated"
    )
    assert (
        config.value(
            "cardinal_classification.label_scheme.low_anchor_subdivision.affects_primary_selection"
        )
        is False
    )
    assert (
        config.value(
            "cardinal_classification.label_scheme.low_anchor_subdivision.min_effective_mass_per_subclass"
        )
        == 5.0
    )
    assert "soft_ordinal_logistic" in config.value(
        "cardinal_classification.models.enabled"
    )
    assert (
        config.value("cardinal_classification.validation.primary_metric")
        == "soft_log_loss"
    )
    assert set(
        config.value("cardinal_classification.source_snapshot_sha256")
    ) == {
        "records.csv",
        "registered_panel_schema.csv",
        "property_proxy_qa.json",
        "integration_input_manifest.json",
        "target_registry.csv",
        "posterior_slope_draws.csv.gz",
        "target_qa.json",
        "bayesian_audit.csv",
        "ppc_binding_audit.csv",
        "stage_refit.json",
        "source_input_manifest.json",
    }


def test_canonical_dataset_is_the_only_active_data_source() -> None:
    config = load_config()
    dataset = config.resolve_path("canonical_dataset")
    assert dataset.name == (
        "canonical_dataset_ir_corrected_v2"
    )
    assert (dataset / "master.csv").is_file()
    assert (dataset / "ml_ready.csv").is_file()
    assert config.resolve_path("active_source_run").name == (
        "current_97_ir_compensation_corrected_v2_20260724"
    )


def test_fixed_feature_association_protocol_is_locked() -> None:
    config = load_config()

    assert (
        config.value("feature_association.status")
        == "exploratory_posthoc_fixed_dataset"
    )
    assert config.value("feature_association.primary_scope") == "all97"
    assert config.value("feature_association.analysis_scopes") == ["all97"]
    assert (
        config.value("feature_association.sensitivity_scope", None) is None
    )
    assert config.value("feature_association.causal_claim_allowed") is False
    assert (
        config.value("feature_association.mechanistic_claim_allowed") is False
    )
    assert config.value("feature_association.frozen_panels") == [
        "context_core",
        "context_plus_catalyst_concentration",
        "context_plus_ionomer_ratio",
        "context_plus_water_organic",
        "context_plus_ethanol_isopropanol",
        "context_plus_viscosity",
        "context_plus_polarity",
        "context_plus_conductive_sensitivity",
    ]
    assert config.value("feature_association.models.enabled") == [
        "bayesian_linear",
        "gam_low_df",
    ]
    assert (
        config.value("feature_association.models.accurate_method_name")
        == "empirical_Bayes_shrinkage_additive_spline"
    )
    assert (
        config.value(
            "feature_association.posterior_policy.posterior_draws_expanded_as_samples"
        )
        is False
    )
    assert (
        config.value(
            "feature_association.validation.models_and_spline_complexity_are_fixed"
        )
        is True
    )
    assert (
        config.value("feature_association.validation.inner_cv_required")
        is False
    )
    assert (
        config.value(
            "feature_association.feature_reduction."
            "outcome_driven_feature_selection_used"
        )
        is False
    )
    assert (
        config.value(
            "feature_association.feature_reduction."
            "one_feature_per_augmented_panel"
        )
        is True
    )
    assert (
        config.value(
            "feature_association.release_gate.failed_blocks_emit_effect_curves"
        )
        is False
    )
    assert (
        config.value("feature_association.effects.extrapolation_allowed")
        is False
    )
    assert (
        config.value("feature_association.effects.joint_support_distance")
        == "fold_local_block_balanced_gower"
    )
    assert (
        config.value(
            "feature_association.effects.minimum_common_record_joint_support_coverage"
        )
        == 0.80
    )
    assert (
        config.value(
            "feature_association.effects.categorical_adjusted_contrasts"
        )
        == "not_released_without_prespecified_common_reference"
    )
    assert (
        config.value(
            "feature_association.release_gate.gam_only_rescue_allowed"
        )
        is False
    )
    assert (
        config.value("feature_association.effects.cross_fitted_g_computation")
        is True
    )
    contrasts = config.value("feature_association.frozen_contrasts")
    assert [(row["reference"], row["augmented"]) for row in contrasts] == [
        ("prior", "context_core"),
        ("context_core", "context_plus_catalyst_concentration"),
        ("context_core", "context_plus_ionomer_ratio"),
        ("context_core", "context_plus_water_organic"),
        ("context_core", "context_plus_ethanol_isopropanol"),
        ("context_core", "context_plus_viscosity"),
        ("context_core", "context_plus_polarity"),
        ("context_core", "context_plus_conductive_sensitivity"),
    ]
    assert config.value("feature_association.atlas.scopes") == ["all97"]
    assert (
        config.value("feature_association.release_gate.applies_to_scope")
        == "all97"
    )


def test_every_registered_input_is_hash_frozen() -> None:
    config = load_config()
    policy = config.section("input_hashes")
    registered = set(policy["expected"])
    configured = set(config.section("paths")) - {
        "output_root",
        "active_source_run",
    }

    assert policy["enforce"] is True
    assert registered == configured
    assert all(len(str(value)) == 64 for value in policy["expected"].values())
    assert registered == {"canonical_dataset", "ppc_review_registry"}
