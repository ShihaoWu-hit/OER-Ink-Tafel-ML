from __future__ import annotations

import pandas as pd
import pytest

from ink_tafel_study.evaluate import _interpretation_blocks
from ink_tafel_study.features import (
    PANEL_FEATURES,
    resolve_feature_panel,
    validate_no_target_leakage,
)


def _feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "catalyst_type": ["RuO2"],
            "catalyst_identity": ["alias_must_not_be_selected_too"],
            "catalyst_source": ["commercial"],
            "catalyst_support": ["none"],
            "electrode_type": ["RDE"],
            "catalyst_loading_mg_cm2": [0.2],
            "electrolyte_identity": ["KOH"],
            "electrolyte_molarity_M": [1.0],
            "rotation_rate_rpm": [1600],
            "scan_rate_mV_s": [5.0],
            "ir_correction_status": ["reported_corrected"],
            "catalyst_concentration_mg_mL": [2.0],
            "nafion_to_solvent_ratio": [0.1],
            "ionomer_concentration_wt_percent": [5.0],
            "conductive_agent_present": ["no"],
            "conductive_agent_dose_mg_mgcat": [0.0],
            "solvent_fraction_acetone": [0.0],
            "solvent_fraction_water": [0.25],
            "solvent_fraction_ethanol": [0.0],
            "solvent_fraction_isopropanol": [0.75],
            "solvent_fraction_methanol": [0.0],
            "sonication_time_min": [30.0],
            "dispersion_method": ["ultrasonication"],
            "drop_volume_uL": [10.0],
            "electrode_area_cm2": [0.196],
            "drying_method": ["air"],
            "drying_temperature_C": [25.0],
            "drying_time_min": [10.0],
            "solvent_mixture_property_proxy_viscosity_mPa_s": [1.5],
            "solvent_mixture_property_proxy_surface_tension_mN_m": [30.0],
            "solvent_mixture_property_proxy_density_kg_m3": [900.0],
            "solvent_mixture_property_proxy_dielectric_constant": [30.0],
            "solvent_mixture_property_proxy_dielectric_constant_estimated": [31.0],
            "solvent_mixture_property_proxy_dielectric_constant_best_available": [30.0],
            "solvent_mixture_property_proxy_dielectric_constant_estimated_flag": [0],
            "b_refit_at_2mA_mV_dec": [60.0],
            "y_overpotential_10mA_mV": [300.0],
        }
    )


def test_all_preregistered_panels_resolve_without_duplicate_physical_expressions() -> None:
    frame = _feature_table()
    assert set(PANEL_FEATURES) == {
        "dummy",
        "catalyst_only",
        "ink_only",
        "conditions_only",
        "context",
        "context_plus_ink_amounts",
        "context_plus_ink",
        "context_plus_ink_process",
        "context_plus_property_proxy_core",
        "context_plus_property_proxy_4p_source_backed",
        "context_plus_property_proxy_4p_estimated",
    }
    for panel_name in PANEL_FEATURES:
        panel = resolve_feature_panel(frame, panel_name, require_all=True)
        assert len(panel.columns) == len(set(panel.columns))
        # Ordered aliases select one expression; the second catalyst alias is
        # present in the table but not duplicated in a model panel.
        assert "catalyst_identity" not in panel.columns


def test_property_proxy_replaces_solvent_composition() -> None:
    frame = _feature_table()
    ink_amounts = resolve_feature_panel(
        frame, "context_plus_ink_amounts", require_all=True
    )
    composition = resolve_feature_panel(frame, "context_plus_ink", require_all=True)
    core = resolve_feature_panel(frame, "context_plus_property_proxy_core", require_all=True)
    source_backed = resolve_feature_panel(
        frame, "context_plus_property_proxy_4p_source_backed", require_all=True
    )
    estimated = resolve_feature_panel(
        frame, "context_plus_property_proxy_4p_estimated", require_all=True
    )

    assert "catalyst_concentration_mg_mL" in ink_amounts.columns
    assert "solvent_fraction_water" not in ink_amounts.columns
    assert "solvent_mixture_property_proxy_viscosity_mPa_s" not in ink_amounts.columns
    assert "solvent_fraction_water" in composition.columns
    assert "solvent_fraction_methanol" in composition.columns
    assert "solvent_mixture_property_proxy_viscosity_mPa_s" not in composition.columns
    for properties in (core, source_backed, estimated):
        assert "solvent_mixture_property_proxy_viscosity_mPa_s" in properties.columns
        assert "solvent_fraction_water" not in properties.columns
    assert "solvent_mixture_property_proxy_dielectric_constant" not in core.columns
    assert "solvent_mixture_property_proxy_dielectric_constant" in source_backed.columns
    assert (
        "solvent_mixture_property_proxy_dielectric_constant_best_available"
        in estimated.columns
    )
    assert "solvent_mixture_property_proxy_dielectric_constant" not in estimated.columns
    core_property_columns = {
        "solvent_mixture_property_proxy_viscosity_mPa_s",
        "solvent_mixture_property_proxy_surface_tension_mN_m",
        "solvent_mixture_property_proxy_density_kg_m3",
    }
    assert set(core.blocks["solvent_property_proxy"]) == core_property_columns
    assert set(source_backed.blocks["solvent_property_proxy"]) == (
        core_property_columns
        | {"solvent_mixture_property_proxy_dielectric_constant"}
    )
    # Interpretation may create separate conditional-sensitivity groupings,
    # but the registered Gower distance block remains joint.
    assert set(estimated.blocks["solvent_property_proxy"]) == (
        core_property_columns
        | {"solvent_mixture_property_proxy_dielectric_constant_best_available"}
    )
    assert estimated.blocks["solvent_property_proxy_estimation_flag"] == (
        "solvent_mixture_property_proxy_dielectric_constant_estimated_flag",
    )


def test_property_interpretation_splits_proxies_without_changing_gower_blocks() -> None:
    core = resolve_feature_panel(
        _feature_table(), "context_plus_property_proxy_core", require_all=True
    )

    # The fitted Gower model retains one joint, equally weighted property block.
    assert set(core.blocks["solvent_property_proxy"]) == {
        "solvent_mixture_property_proxy_viscosity_mPa_s",
        "solvent_mixture_property_proxy_surface_tension_mN_m",
        "solvent_mixture_property_proxy_density_kg_m3",
    }

    interpretation = _interpretation_blocks(core.name, core)
    assert "solvent_property_proxy" not in interpretation
    assert interpretation["solvent_property_proxy_viscosity"] == (
        "solvent_mixture_property_proxy_viscosity_mPa_s",
    )
    assert interpretation["solvent_property_proxy_surface_tension"] == (
        "solvent_mixture_property_proxy_surface_tension_mN_m",
    )
    assert interpretation["solvent_property_proxy_density"] == (
        "solvent_mixture_property_proxy_density_kg_m3",
    )
    assert {column for members in interpretation.values() for column in members} == set(
        core.columns
    )


def test_unverified_legacy_physics_columns_cannot_enter_property_panel() -> None:
    frame = _feature_table().drop(
        columns=[column for column in _feature_table() if column.startswith("solvent_mixture_property_proxy_")]
    )
    frame["physics_viscosity_Pa_s"] = 0.001
    frame["physics_surface_tension_N_m"] = 0.03
    for panel_name in (
        "context_plus_property_proxy_core",
        "context_plus_property_proxy_4p_source_backed",
        "context_plus_property_proxy_4p_estimated",
    ):
        panel = resolve_feature_panel(frame, panel_name)
        assert "physics_viscosity_Pa_s" not in panel.columns
        assert "physics_surface_tension_N_m" not in panel.columns
        assert set(panel.missing_logical_features).issuperset(
            {"mixture_viscosity_proxy", "mixture_surface_tension_proxy"}
        )


def test_target_and_curve_derived_columns_are_rejected() -> None:
    for leaking in (
        "b_refit_at_2mA_mV_dec",
        "b_single_regime_mV_dec",
        "y_tafel_slope_mV_dec",
        "overpotential_10mA_mV",
        "posterior_slope_mean",
        "fit_derived_plateau",
    ):
        with pytest.raises(ValueError, match="forbidden predictors"):
            validate_no_target_leakage(["electrolyte_molarity_M", leaking])


def test_missing_features_are_manifested_or_can_be_strict() -> None:
    frame = _feature_table().drop(columns=["scan_rate_mV_s"])
    panel = resolve_feature_panel(frame, "conditions_only")
    assert panel.missing_logical_features == ("scan_rate",)
    with pytest.raises(ValueError, match="missing required logical features"):
        resolve_feature_panel(frame, "conditions_only", require_all=True)
