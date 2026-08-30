from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from ink_tafel_study.feature_development_components import (
    FEATURE_DEVELOPMENT_RECIPES,
    PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES,
    RAW_SOLVENT_PREDICTORS,
    SENSITIVITY_FEATURE_DEVELOPMENT_RECIPE_NAMES,
    SOLVENT_FRACTIONS,
    FeatureDevelopmentBuilder,
    feature_definition_registry,
    feature_development_registry,
    make_feature_development_pipeline,
    required_feature_development_source_columns,
)


def _frame(n: int = 18) -> pd.DataFrame:
    index = np.arange(n)
    fractions = np.tile([0.1, 0.4, 0.2, 0.2, 0.1], (n, 1)).astype(float)
    fractions[index % 3 == 0] = [0.0, 0.75, 0.0, 0.25, 0.0]
    frame = pd.DataFrame(
        {
            "catalyst_loading_mg_cm2_selected": 0.2 + 0.01 * index,
            "electrolyte_molarity_M": np.where(index % 2, 0.5, 1.0),
            "rpm": np.where(index % 2, 900.0, 1600.0),
            "scan_rate_mV_s": np.where(index % 2, 5.0, 10.0),
            "catalyst_type": np.where(index % 2, "IrO2", "RuO2"),
            "electrolyte_identity": np.where(index % 2, "KOH", "HClO4"),
            "ir_correction_status": np.where(
                index % 2, "reported_corrected", "not_reported_or_unverifiable"
            ),
            "catalyst_concentration_mg_mL": 8.0 + 0.1 * index,
            "ionomer_to_solvent_ratio": 0.02 + 0.002 * index,
            "ionomer_concentration_wt_percent": np.where(index % 2, 5.0, 20.0),
            "conductive_agent_dose_mg_per_mgcat": 0.25 + 0.01 * index,
            "conductive_agent_present": np.where(index % 2, 1, 0),
            "conductive_agent_class": np.where(index % 2, "carbon_black", "none"),
            "catalyst_mass_mg": 2.0 + 0.1 * index,
            "conductive_agent_mass_mg": 0.5 + 0.02 * index,
            "total_liquid_volume_uL": 250.0 + 2.0 * index,
            "drop_volume_uL": 50.0 + index,
            "ionomer_volume_uL_per_mgcat": 20.0 + index,
            "solvent_mixture_property_proxy_viscosity_mPa_s": 2.0 + 0.1 * index,
            "solvent_mixture_property_proxy_surface_tension_mN_m": 30.0 + index,
            "solvent_mixture_property_proxy_density_kg_m3": 1000.0 - index,
            "solvent_mixture_property_proxy_dielectric_constant__best_available": 40.0 + index,
            "sonication_time_min": 10.0 + index,
            "drying_temperature_C": 20.0 + index,
            "drying_time_min": 30.0 + index,
            "dispersion_method": np.where(index % 2, "bath", "probe"),
            "drying_process": np.where(index % 2, "air", "vacuum"),
            "solvent_1_type": "ignored_a",
            "solvent_2_type": "ignored_b",
            "solvent_1_volume_uL": 999.0,
            "solvent_2_volume_uL": 777.0,
        }
    )
    for position, name in enumerate(SOLVENT_FRACTIONS):
        frame[name] = fractions[:, position]
    return frame


def test_frozen_primary_ladder_and_sensitivities_are_auditable() -> None:
    assert PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES == (
        "context_core",
        "context_plus_recorded_amounts",
        "si_intensive_amounts",
        "si_composition_summary",
        "composition_ilr",
        "property_lean",
        "property_physics",
        "property_interactions",
    )
    assert set(SENSITIVITY_FEATURE_DEVELOPMENT_RECIPE_NAMES) == {
        "sensitivity_fractions_plus_properties",
        "sensitivity_process_features",
        "sensitivity_batch_masses",
    }
    registry = feature_development_registry().set_index("name")
    assert registry.loc[list(PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES), "sensitivity_only"].eq(False).all()  # noqa: E712
    assert registry.loc[
        "sensitivity_fractions_plus_properties", "block_solvent_fractions_sensitivity"
    ]
    assert registry.loc["sensitivity_fractions_plus_properties", "block_property_physics"]
    for name in PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES:
        blocks = set(FEATURE_DEVELOPMENT_RECIPES[name].blocks)
        assert not ({"composition_ilr", "si_composition_summary"} & blocks and {"property_lean", "property_physics"} & blocks)


def test_allowlists_forbid_raw_solvent_and_target_fields() -> None:
    for recipe in FEATURE_DEVELOPMENT_RECIPES:
        columns = required_feature_development_source_columns(recipe)
        assert not RAW_SOLVENT_PREDICTORS.intersection(columns)
        assert not any(
            token in name.lower()
            for name in columns
            for token in ("tafel", "slope", "posterior", "selected_model")
        )
    definitions = feature_definition_registry()
    assert {"feature_name", "unit", "formula", "source_columns"}.issubset(definitions)


def test_intensive_and_physics_formulas_have_registered_units() -> None:
    frame = _frame()
    intensive = FeatureDevelopmentBuilder("si_intensive_amounts").fit_transform(frame)
    np.testing.assert_allclose(intensive.loc[0, "log10_solid_concentration_mg_mL"], 1.0)
    np.testing.assert_allclose(intensive.loc[0, "ink_dilution_uL_mgsolid"], 100.0)
    np.testing.assert_allclose(intensive.loc[0, "deposited_catalyst_dose_mg"], 0.4)
    np.testing.assert_allclose(intensive.loc[0, "deposited_solid_dose_mg"], 0.5)
    np.testing.assert_allclose(intensive.loc[0, "ionomer_solution_dose_uL_mgcat"], 20.0)
    np.testing.assert_allclose(intensive.loc[0, "log10_catalyst_loading_mg_cm2"], np.log10(0.2))
    np.testing.assert_allclose(intensive.loc[0, "log10_total_solid_loading_mg_cm2"], np.log10(0.25))

    physics = FeatureDevelopmentBuilder("property_physics").fit_transform(frame)
    np.testing.assert_allclose(physics.loc[0, "kinematic_viscosity_m2_s"], 2.0e-6)
    np.testing.assert_allclose(physics.loc[0, "capillary_flow_proxy_m_s"], 15.0)
    units = feature_definition_registry().set_index("feature_name")["unit"]
    assert units.loc["kinematic_viscosity_m2_s"] == "m^2 s^-1"
    assert units.loc["capillary_flow_proxy_m_s"] == "m s^-1"

    summary = FeatureDevelopmentBuilder("si_composition_summary").fit_transform(frame)
    expected_entropy = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    np.testing.assert_allclose(summary.loc[0, "solvent_entropy"], expected_entropy)

    interactions = FeatureDevelopmentBuilder("property_interactions").fit_transform(frame)
    expected_log_viscosity = np.log10(2.0)
    np.testing.assert_allclose(
        interactions.loc[0, "log10_solvent_viscosity_mPa_s"],
        expected_log_viscosity,
    )
    np.testing.assert_allclose(
        interactions.loc[0, "log10_viscosity_x_log10_solid_concentration"],
        expected_log_viscosity,
    )
    np.testing.assert_allclose(
        interactions.loc[0, "capillary_flow_x_log10_ionomer_solvent_ratio"],
        interactions.loc[0, "capillary_flow_proxy_m_s"]
        * np.log10(frame.loc[0, "ionomer_to_solvent_ratio"]),
    )
    catalyst_interactions = [
        name
        for name in interactions
        if name.startswith("catalyst_type_") and "_x_" in name
    ]
    assert len(catalyst_interactions) == 2

    with_zero = frame.copy()
    with_zero.loc[0, "ionomer_to_solvent_ratio"] = 0.0
    zero_builder = FeatureDevelopmentBuilder("property_interactions").fit(with_zero)
    assert zero_builder.ionomer_ratio_delta_ == 1.0e-3
    zero_features = zero_builder.transform(with_zero)
    np.testing.assert_allclose(
        zero_features.loc[0, "capillary_flow_x_log10_ionomer_solvent_ratio"],
        zero_features.loc[0, "capillary_flow_proxy_m_s"] * np.log10(1.0e-3),
    )

    definitions = feature_definition_registry().set_index("feature_name")
    assert "Sheet2 reviewed iR evidence" in definitions.loc[
        "ir_correction_status", "data_source"
    ]


def test_composition_and_property_learning_is_fold_local_and_raw_solvents_are_ignored() -> None:
    frame = _frame()
    builder = FeatureDevelopmentBuilder("composition_ilr").fit(frame.iloc[:12])
    original = builder.transform(frame.iloc[12:])
    scaled = frame.iloc[12:].copy()
    scaled.loc[:, list(SOLVENT_FRACTIONS)] *= 9.0
    changed_raw = scaled.copy()
    changed_raw["solvent_1_type"] = "poison"
    changed_raw["solvent_1_volume_uL"] = -1.0e12
    np.testing.assert_allclose(
        original.filter(like="solvent_ilr_"),
        builder.transform(scaled).filter(like="solvent_ilr_"),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(builder.transform(scaled).select_dtypes(include=[np.number]), builder.transform(changed_raw).select_dtypes(include=[np.number]))
    assert len([name for name in original if name.startswith("solvent_ilr_")]) == 4

    prop = FeatureDevelopmentBuilder("property_lean").fit(frame.iloc[:12])
    components = prop.polarity_pca_.components_.copy()
    extreme = frame.iloc[12:].copy()
    extreme.loc[:, "solvent_mixture_property_proxy_dielectric_constant__best_available"] *= 1.0e6
    prop.transform(extreme)
    np.testing.assert_allclose(prop.polarity_pca_.components_, components)


def test_all_builders_and_fixed_models_are_cloneable_and_finite() -> None:
    frame = _frame()
    y = 60.0 + 2.0 * np.arange(len(frame))
    for name in FEATURE_DEVELOPMENT_RECIPES:
        transformed = clone(FeatureDevelopmentBuilder(name)).fit_transform(frame.iloc[:12])
        held_out = FeatureDevelopmentBuilder(name).fit(frame.iloc[:12]).transform(frame.iloc[12:])
        assert list(transformed) == list(held_out)
        assert np.isfinite(transformed.select_dtypes(include=[np.number])).all().all()
        assert np.isfinite(held_out.select_dtypes(include=[np.number])).all().all()
    for model in ("rbf_svr", "bayesian_ridge", "dummy_mean", "dummy_median"):
        pipeline = clone(make_feature_development_pipeline("property_interactions", model=model))
        pipeline.fit(frame.iloc[:14], y[:14])
        prediction = pipeline.predict(frame.iloc[14:])
        assert np.isfinite(prediction).all()
        assert (prediction > 0).all()

    pipeline = make_feature_development_pipeline("context_core", model="rbf_svr")
    target_wrapper = pipeline.named_steps["target"]
    assert target_wrapper.transform == "log10"
    assert target_wrapper.standardize is False
    params = target_wrapper.estimator.get_params()
    assert {name: params[name] for name in ("kernel", "C", "epsilon", "gamma")} == {
        "kernel": "rbf",
        "C": 1.0,
        "epsilon": 0.05,
        "gamma": 0.1,
    }
    fitted = clone(pipeline).fit(frame.iloc[:14], y[:14])
    engineered = fitted.named_steps["features"].transform(frame.iloc[14:])
    encoded = fitted.named_steps["preprocess"].transform(engineered)
    encoded = fitted.named_steps["near_zero_variance"].transform(encoded)
    fitted_target = fitted.named_steps["target"]
    assert fitted_target.y_mean_ == 0.0
    assert fitted_target.y_scale_ == 1.0
    direct = np.power(10.0, fitted_target.estimator_.predict(encoded))
    np.testing.assert_allclose(fitted.predict(frame.iloc[14:]), direct)
