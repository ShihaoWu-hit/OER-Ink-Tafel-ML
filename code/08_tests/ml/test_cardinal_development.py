from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ink_tafel_study.cardinal_development import run_cardinal_development
from ink_tafel_study.cardinal_feature_engineering import (
    CARDINAL_FEATURE_RECIPES,
    PRIMARY_CARDINAL_FEATURE_RECIPES,
    SENSITIVITY_CARDINAL_FEATURE_RECIPES,
    CardinalFeatureBuilder,
    required_cardinal_feature_columns,
)
from ink_tafel_study.cardinal_models import EngineeredCardinalClassifier
from ink_tafel_study.cardinal_stacking import fit_convex_stack_weight
from ink_tafel_study.config import load_config
from ink_tafel_study.orthogonal_kinetics import audit_orthogonal_kinetics


def _records() -> pd.DataFrame:
    return pd.read_csv(
        "Data/03_outputs/study/outputs/current_97/data/records.csv"
    )


def test_cardinal_development_mainline_is_fixed_and_excludes_search_expansion() -> None:
    settings = load_config().section("cardinal_development")
    assert (
        settings["active_design"]
        == "fixed_ir_stack_plus_ru_ir_hierarchical_map"
    )
    assert settings["retired_recipe_search"]["enabled"] is False
    assert settings["ir_fixed_comparison"]["recipe_selection"] == (
        "fixed_no_joint_selector"
    )
    assert settings["ir_fixed_comparison"]["optuna_used"] is False
    assert settings["hierarchical_partial_pooling"][
        "fixed_feature_recipes"
    ] == ["context_log", "legacy_safe_compact"]
    assert settings["hierarchical_partial_pooling"]["optuna_used"] is False
    assert settings["excluded_from_mainline"] == {
        "clustering": "exploratory_formulation_archetypes_only",
        "SMOTE": True,
        "deep_learning": True,
        "large_CatBoost_XGBoost_search": True,
    }


def test_cardinal_recipe_catalog_is_finite_and_mutually_exclusive() -> None:
    assert PRIMARY_CARDINAL_FEATURE_RECIPES == (
        "context_log",
        "context_log_plus_ink_log",
        "intensive_formulation",
        "compact_composition",
    )
    assert SENSITIVITY_CARDINAL_FEATURE_RECIPES == (
        "sensitivity_composition_ilr",
        "sensitivity_lean_property",
        "sensitivity_compact_process",
        "sensitivity_compact_catalyst_interactions",
    )
    assert (
        CARDINAL_FEATURE_RECIPES["legacy_safe_compact"].fixed_comparison_only
        is True
    )
    assert "legacy_safe_compact" not in PRIMARY_CARDINAL_FEATURE_RECIPES
    assert "legacy_safe_compact" not in SENSITIVITY_CARDINAL_FEATURE_RECIPES
    for name, recipe in CARDINAL_FEATURE_RECIPES.items():
        columns = required_cardinal_feature_columns(name)
        assert len(columns) == len(set(columns))
        assert not any(
            token in column.lower()
            for column in columns
            for token in ("slope", "posterior", "target", "tafel")
        )
        assert not (
            recipe.representation == "lean_property"
            and any(column.startswith("solvent_fraction_") for column in columns)
        )


def test_legacy_safe_compact_has_three_composition_coordinates_without_intensive() -> None:
    records = _records()
    columns = required_cardinal_feature_columns("legacy_safe_compact")
    assert not set(columns).intersection(
        {
            "catalyst_mass_mg",
            "conductive_agent_mass_mg",
            "total_liquid_volume_uL",
            "drop_volume_uL",
        }
    )
    builder = CardinalFeatureBuilder("legacy_safe_compact").fit(
        records.loc[:, columns]
    )
    transformed = builder.transform(records.loc[:, columns])
    composition = tuple(builder.get_blocks_out()["composition"])
    assert composition == (
        "log10_water_to_organic_balance",
        "log10_ethanol_to_isopropanol_balance",
        "solvent_entropy",
    )
    assert "log10_water_to_alcohol_balance" not in transformed


def test_convex_stack_weight_uses_soft_log_loss_and_respects_endpoints() -> None:
    observed = np.asarray(
        [
            [0.90, 0.05, 0.03, 0.02],
            [0.05, 0.85, 0.05, 0.05],
            [0.02, 0.03, 0.90, 0.05],
            [0.04, 0.06, 0.10, 0.80],
        ]
    )
    uninformative = np.full_like(observed, 0.25)
    gower_best = fit_convex_stack_weight(
        observed, observed, uninformative
    )
    logistic_best = fit_convex_stack_weight(
        observed, uninformative, observed
    )
    assert gower_best["gower_weight"] == pytest.approx(1.0)
    assert gower_best["logistic_weight"] == pytest.approx(0.0)
    assert logistic_best["gower_weight"] == pytest.approx(0.0)
    assert logistic_best["logistic_weight"] == pytest.approx(1.0)


def test_cardinal_feature_builder_is_fold_local_finite_and_block_safe() -> None:
    records = _records()
    for name in CARDINAL_FEATURE_RECIPES:
        columns = required_cardinal_feature_columns(name)
        builder = clone(CardinalFeatureBuilder(name)).fit(records.loc[:, columns])
        transformed = builder.transform(records.loc[:, columns])
        numeric = transformed.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy(float)).all()
        assert len(transformed.columns) == len(set(transformed.columns))
        blocks = builder.get_blocks_out()
        flattened = [column for values in blocks.values() for column in values]
        assert len(flattened) == len(set(flattened))
        assert set(flattened) == set(transformed.columns)


def test_compact_composition_is_scale_invariant_and_rejects_invalid_rows() -> None:
    records = _records()
    name = "compact_composition"
    columns = required_cardinal_feature_columns(name)
    X = records.loc[:, columns].copy()
    builder = CardinalFeatureBuilder(name).fit(X)
    original = builder.transform(X)
    fractions = [
        column for column in columns if column.startswith("solvent_fraction_")
    ]
    scaled = X.copy()
    scaled.loc[:, fractions] = scaled.loc[:, fractions] * 100.0
    repeated = builder.transform(scaled)
    composition_columns = list(builder.get_blocks_out()["composition"])
    np.testing.assert_allclose(
        original.loc[:, composition_columns],
        repeated.loc[:, composition_columns],
        atol=1e-12,
    )

    invalid = X.copy()
    invalid.loc[0, fractions[0]] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        CardinalFeatureBuilder(name).fit(invalid)
    invalid = X.copy()
    invalid.loc[0, fractions] = 0.0
    with pytest.raises(ValueError, match="positive row sum"):
        CardinalFeatureBuilder(name).fit(invalid)


def test_engineered_cardinal_classifier_is_cloneable_and_probabilistic() -> None:
    records = _records().iloc[:40].copy()
    columns = required_cardinal_feature_columns("compact_composition")
    targets = np.tile(np.asarray([[0.2, 0.4, 0.2, 0.2]]), (len(records), 1))
    estimator = EngineeredCardinalClassifier(
        feature_recipe="compact_composition",
        base_model_family="soft_gower_knn",
        base_model_params={"n_neighbors": 9, "distance_power": 1.0},
        random_seed=7,
    )
    fitted = clone(estimator).fit(records.loc[:, columns], targets)
    probabilities = fitted.predict_proba(records.loc[:2, columns])
    assert probabilities.shape == (3, 4)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert set(fitted.engineered_blocks_) == {
        "context",
        "ink",
        "composition",
    }


def test_empty_orthogonal_contract_never_imputes_missing_kinetics(
    tmp_path: Path,
) -> None:
    audit = audit_orthogonal_kinetics(
        _records(),
        "Data/orthogonal_kinetics",
        (
            "Data/03_outputs/tafel_refitting/outputs/20260721_112453/csv/"
            "binned_curve_points.csv"
        ),
        tmp_path,
    )
    assert audit.qa["eligible_predictor_columns"] == []
    assert audit.qa["missing_values_inferred"] is False
    assert audit.qa["binned_points_hash_matches"] is True
    assert audit.qa["n_binned_curves"] == 97
    slope = audit.predictor_eligibility.loc[
        audit.predictor_eligibility["field_or_stream"].eq(
            "local_slope_vs_potential"
        )
    ].iloc[0]
    assert bool(slope["eligible_for_cardinal_predictor"]) is False
    assert bool(slope["eligible_for_auxiliary_analysis"]) is True


def test_cardinal_development_smoke_trains_user_requested_catalyst_family(
    tmp_path: Path,
) -> None:
    config = load_config()
    destination = tmp_path / "cardinal_development_smoke"
    result = run_cardinal_development(
        config,
        destination,
        source_run=config.resolve_path("active_source_run"),
        smoke=True,
        n_jobs=2,
    )
    qa = json.loads(
        (destination / "cardinal_development_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "complete_smoke"
    assert qa["retired_recipe_search_executed"] is False
    assert qa["independent_catalyst_models_fitted"] is False
    assert qa["n_records_IrO2_plus_Ir"] == 43
    assert qa["n_records_RuO2"] == 47
    assert qa["metal_Ir_merged_into_IrO2"] is True
    assert qa["missing_orthogonal_values_inferred"] is False
    registry = pd.read_csv(
        destination / "registry" / "catalyst_strata_eligibility.csv"
    ).set_index("catalyst_stratum")
    assert int(registry.loc["IrO2_plus_Ir", "n_records"]) == 43
    assert int(registry.loc["RuO2", "n_records"]) == 47
    ir_records = pd.read_csv(
        destination
        / "analyses"
        / "catalyst_type_strata"
        / "IrO2_plus_Ir"
        / "stratum_record_registry.csv"
    )
    assert len(ir_records) == 43
    assert set(ir_records["catalyst_type_original"]) == {"Ir", "IrO2"}
    ir_dir = (
        destination
        / "analyses"
        / "catalyst_type_strata"
        / "IrO2_plus_Ir"
    )
    fixed_qa = json.loads(
        (ir_dir / "analysis_qa.json").read_text(encoding="utf-8")
    )
    assert fixed_qa["fixed_recipes"] == [
        "context_log",
        "legacy_safe_compact",
    ]
    assert fixed_qa["joint_recipe_selector_used"] is False
    assert fixed_qa["optuna_used"] is False
    assert fixed_qa["outer_test_targets_used_for_stacking_weight"] is False
    weights = pd.read_csv(ir_dir / "stacking_weights.csv")
    assert len(weights) == 4
    assert set(weights["recipe"]) == {
        "context_log",
        "legacy_safe_compact",
    }
    assert weights["gower_weight"].between(0.0, 1.0).all()
    fixed_registry = pd.read_csv(ir_dir / "fixed_pipeline_registry.csv")
    assert not fixed_registry["recipe_or_model_search"].astype(bool).any()
    fixed_oof = pd.read_csv(ir_dir / "oof_aggregated.csv")
    assert set(fixed_oof["model_role"]) == {"fixed_stacked"}
    assert set(fixed_oof["model_family"]) == {
        "soft_gower_logistic_stack"
    }
    assert (
        ir_dir / "abstention" / "abstention_qa.json"
    ).is_file()
    hierarchical_dir = destination / "analyses" / "hierarchical_map"
    hierarchical_qa = json.loads(
        (hierarchical_dir / "hierarchical_comparison_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert hierarchical_qa["fixed_feature_recipes"] == [
        "context_log",
        "legacy_safe_compact",
    ]
    assert hierarchical_qa["recipe_selector_used"] is False
    assert hierarchical_qa["optuna_used"] is False
    assert hierarchical_qa["posterior_draws_expanded"] is False
    assert hierarchical_qa["one_fractional_soft_response_per_record"] is True
    assert hierarchical_qa["shared_outer_manifests_across_recipes"] is True
    assert hierarchical_qa["shared_inner_manifests_across_recipes"] is True
    assert (
        hierarchical_dir / "context_log" / "model_definition.json"
    ).is_file()
    assert (
        hierarchical_dir
        / "legacy_safe_compact"
        / "model_definition.json"
    ).is_file()
    assert (
        hierarchical_dir / "abstention" / "abstention_qa.json"
    ).is_file()
    assert (
        destination / "report" / "cardinal_development_report.md"
    ).is_file()
