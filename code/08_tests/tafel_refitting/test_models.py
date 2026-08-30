from __future__ import annotations

import numpy as np

from tafel_refit.models import (
    LOG10,
    LOW_CURRENT_FLOOR_SENSITIVITY_MODEL_NAMES,
    MODEL_SPECS,
    PRIMARY_MODEL_NAMES,
    classify_tafel_regime,
    fit_model,
    harmonic_plateau_log10,
    model_spec,
    predict_log10,
)


def test_paper_equation_matches_legacy_parameterization() -> None:
    voltage = np.linspace(0.0, 0.5, 31)
    alpha = 18.0
    legacy_b = 3.2
    legacy_ilim = 4.5
    legacy = (legacy_b + alpha * voltage - np.log(legacy_ilim + np.exp(alpha * voltage))) / LOG10

    parameters = {
        "a": (legacy_b - np.log(legacy_ilim)) / LOG10,
        "log_m": np.log(alpha / LOG10),
        "L": legacy_b / LOG10,
    }
    modern = predict_log10("paper_single_plateau", voltage, parameters, 0.01)
    np.testing.assert_allclose(modern, legacy, rtol=1e-12, atol=1e-12)


def test_harmonic_plateau_is_numerically_stable() -> None:
    kinetic = np.array([-1000.0, -2.0, 2.0, 1000.0])
    predicted = harmonic_plateau_log10(kinetic, 1.0)
    assert np.all(np.isfinite(predicted))
    np.testing.assert_allclose(predicted[-1], 1.0, rtol=1e-8, atol=1e-8)


def test_single_plateau_recovers_known_slope() -> None:
    rng = np.random.default_rng(12)
    x = np.linspace(0.0, 0.42, 61)
    truth = {"a": -1.5, "log_m": np.log(1000.0 / 60.0), "L": 1.2}
    y = predict_log10("paper_single_plateau", x, truth, 0.01)
    y = y + rng.standard_t(df=4, size=len(x)) * 0.015
    fit = fit_model(
        "paper_single_plateau",
        x,
        y,
        transition_width_V=0.01,
        n_starts=12,
        seed=42,
    )
    recovered = 1000.0 / np.exp(fit.parameters["log_m"])
    assert abs(recovered - 60.0) < 6.0


def test_two_regime_model_recovers_direction_of_change() -> None:
    rng = np.random.default_rng(24)
    x = np.linspace(0.0, 0.48, 61)
    truth = {
        "a": -2.0,
        "log_m1": np.log(1000.0 / 40.0),
        "log_m2": np.log(1000.0 / 120.0),
        "c": 0.24,
        "L": 5.0,
    }
    y = predict_log10("two_regime_plateau", x, truth, 0.012)
    y = y + rng.standard_t(df=4, size=len(x)) * 0.012
    fit = fit_model(
        "two_regime_plateau",
        x,
        y,
        transition_width_V=0.012,
        n_starts=16,
        seed=84,
    )
    low = 1000.0 / np.exp(fit.parameters["log_m1"])
    high = 1000.0 / np.exp(fit.parameters["log_m2"])
    assert low < high
    assert abs(low - 40.0) < 12.0
    assert abs(high - 120.0) < 30.0
    assert abs(fit.parameters["c"] - 0.24) < 0.06


def test_registered_low_current_floor_sensitivity_recovers_known_slope() -> None:
    rng = np.random.default_rng(19)
    x = np.linspace(0.0, 0.45, 61)
    truth = {
        "B": -1.8,
        "a": -3.2,
        "log_m": np.log(1000.0 / 70.0),
        "L": 1.3,
    }
    y = predict_log10("single_background_plateau", x, truth, 0.012)
    y = y + rng.standard_t(df=4, size=len(x)) * 0.012
    floor_sensitivity = fit_model(
        "single_background_plateau",
        x,
        y,
        transition_width_V=0.012,
        n_starts=16,
        seed=19,
    )
    two_regime = fit_model(
        "two_regime_plateau",
        x,
        y,
        transition_width_V=0.012,
        n_starts=16,
        seed=29,
    )
    recovered = 1000.0 / np.exp(floor_sensitivity.parameters["log_m"])
    assert abs(recovered - 70.0) < 12.0
    assert floor_sensitivity.bic < two_regime.bic


def test_primary_registry_has_one_family_and_three_floor_free_variants() -> None:
    assert all(name == spec.model_name for name, spec in MODEL_SPECS.items())
    assert len({spec.analysis_seed_offset for spec in MODEL_SPECS.values()}) == len(
        MODEL_SPECS
    )
    assert PRIMARY_MODEL_NAMES == (
        "linear",
        "paper_single_plateau",
        "two_regime_plateau",
    )
    primary_specs = [model_spec(name) for name in PRIMARY_MODEL_NAMES]
    assert {spec.kinetic_structure for spec in primary_specs} == {
        "constant",
        "potential_dependent",
    }
    assert not any(spec.has_low_current_floor for spec in primary_specs)
    assert all(spec.role == "primary" for spec in primary_specs)
    assert all(
        model_spec(name).role == "low_current_floor_sensitivity"
        for name in LOW_CURRENT_FLOOR_SENSITIVITY_MODEL_NAMES
    )


def test_regime_boundaries_match_knn_rules() -> None:
    assert classify_tafel_regime(90.0) == "low_40_60"
    assert classify_tafel_regime(90.0001) == "120_like"
    assert classify_tafel_regime(122.0) == "120_like"
    assert classify_tafel_regime(122.0001) == "high_gt120"
