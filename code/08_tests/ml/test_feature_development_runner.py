from __future__ import annotations

import numpy as np
import pandas as pd

from ink_tafel_study.feature_development import (
    ANALYSES,
    RBF_PARAMETERS,
    _evaluate_recipes,
    choose_feature_recipe,
)


def test_fixed_model_and_analysis_scopes_are_frozen() -> None:
    assert RBF_PARAMETERS == {
        "kernel": "rbf",
        "C": 1.0,
        "epsilon": 0.05,
        "gamma": 0.1,
    }
    assert [(item.analysis_id, item.scope, item.include_sensitivity_recipes) for item in ANALYSES] == [
        ("primary_all", "all", False),
        ("sensitivity_alkaline", "alkaline", False),
        ("sensitivity_feature_recipes_all", "all", True),
    ]


def test_recipe_tie_uses_lower_feature_complexity_only_inside_half_mv() -> None:
    scores = pd.DataFrame(
        [
            {
                "recipe": "complex",
                "inner_mae_mV_dec": 10.0,
                "status": "complete",
                "feature_complexity": 20,
                "complexity_rank": 8,
            },
            {
                "recipe": "simple",
                "inner_mae_mV_dec": 10.4,
                "status": "complete",
                "feature_complexity": 7,
                "complexity_rank": 1,
            },
        ]
    )
    strict, _ = choose_feature_recipe(scores, tie_tolerance_mV_dec=0.1)
    tied, details = choose_feature_recipe(scores, tie_tolerance_mV_dec=0.5)
    assert strict == "complex"
    assert tied == "simple"
    assert details["simplicity_tie_break_applied"] is True


def test_inner_recipe_scores_do_not_read_outer_test_targets() -> None:
    rng = np.random.default_rng(7)
    n = 18
    loading = np.linspace(0.05, 1.0, n)
    frame = pd.DataFrame(
        {
            "catalyst_type": np.where(np.arange(n) % 2, "IrO2", "RuO2"),
            "catalyst_support": "unsupported",
            "catalyst_loading_mg_cm2_selected": loading,
            "electrolyte_molarity_M": 1.0,
            "rpm": rng.choice([0.0, 400.0, 1600.0], n),
            "scan_rate_mV_s": 10.0,
            "electrolyte_identity": "KOH",
            "ir_correction_status": "reported_corrected",
        }
    )
    train = np.arange(12)
    outer_test = np.arange(12, n)
    y = 60.0 + 120.0 * loading
    splits = [(np.arange(0, 6), np.arange(6, 12)), (np.arange(6, 12), np.arange(0, 6))]
    first, _ = _evaluate_recipes(("context_core",), frame.iloc[train].reset_index(drop=True), y[train], splits, n_jobs=1)
    poisoned = y.copy()
    poisoned[outer_test] = 1.0e12
    second, _ = _evaluate_recipes(("context_core",), frame.iloc[train].reset_index(drop=True), poisoned[train], splits, n_jobs=1)
    pd.testing.assert_frame_equal(first, second)


def test_fixed_rbf_selector_recovers_a_synthetic_property_signal() -> None:
    rng = np.random.default_rng(9)
    n = 80
    signal = rng.uniform(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "catalyst_loading_mg_cm2_selected": 0.2,
            "electrolyte_molarity_M": 1.0,
            "rpm": 0.0,
            "scan_rate_mV_s": 10.0,
            "catalyst_type": np.where(np.arange(n) % 2, "IrO2", "RuO2"),
            "electrolyte_identity": "KOH",
            "ir_correction_status": "reported_corrected",
            "catalyst_concentration_mg_mL": 10.0,
            "ionomer_to_solvent_ratio": 0.02,
            "ionomer_concentration_wt_percent": 5.0,
            "conductive_agent_dose_mg_per_mgcat": 0.0,
            "conductive_agent_present": 0,
            "conductive_agent_class": "none",
            "catalyst_mass_mg": 2.0,
            "conductive_agent_mass_mg": 0.0,
            "total_liquid_volume_uL": 200.0,
            "drop_volume_uL": 50.0,
            "ionomer_volume_uL_per_mgcat": 20.0,
            "solvent_mixture_property_proxy_viscosity_mPa_s": 10.0**signal,
            "solvent_mixture_property_proxy_surface_tension_mN_m": 30.0
            + 2.0 * signal,
            "solvent_mixture_property_proxy_density_kg_m3": 900.0
            + 10.0 * signal,
            "solvent_mixture_property_proxy_dielectric_constant_best_available": 40.0
            + 5.0 * signal,
        }
    )
    target = 10.0 ** (1.6 + 0.8 * signal)
    all_positions = np.arange(n)
    splits = []
    for fold in range(4):
        validation = np.arange(fold, n, 4)
        train = np.setdiff1d(all_positions, validation)
        splits.append((train, validation))

    scores, _ = _evaluate_recipes(
        ("context_core", "property_lean"),
        frame,
        target,
        splits,
        n_jobs=1,
    )
    selected, _ = choose_feature_recipe(scores, tie_tolerance_mV_dec=0.5)
    indexed = scores.set_index("recipe")

    assert selected == "property_lean"
    assert (
        indexed.loc["property_lean", "inner_mae_mV_dec"]
        < indexed.loc["context_core", "inner_mae_mV_dec"] - 0.5
    )
