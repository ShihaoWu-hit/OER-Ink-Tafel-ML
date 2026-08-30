from __future__ import annotations

import numpy as np
import pandas as pd

from ink_tafel_study.posterior_functional import (
    evaluate_posterior_draw_frame,
    evaluate_posterior_slope_draw_frame,
)


def test_single_regime_primary_uses_constant_posterior_slope() -> None:
    posterior = pd.DataFrame(
        {"chain": [0, 0], "draw": [0, 1], "log_m": np.log([10.0, 20.0])}
    )
    result = evaluate_posterior_slope_draw_frame(
        posterior, curve_id="sr001", sample_record=1, model_name="linear"
    )

    assert np.allclose(result["primary_slope_mV_dec"], [100.0, 50.0])
    assert result["posterior_source_variable"].eq("slope_mV_dec").all()
    assert result["target_name"].eq("b_low_potential_mV_dec").all()


def test_two_regime_primary_uses_low_potential_coordinate_not_numeric_minimum() -> None:
    posterior = pd.DataFrame(
        {
            "chain": [0],
            "draw": [0],
            # b_low = 100 mV/dec and b_high = 50 mV/dec.  A numerical-minimum
            # rule would incorrectly return the high-potential branch.
            "log_m1": [np.log(10.0)],
            "log_m2": [np.log(20.0)],
            "change_potential_relative_V": [0.25],
        }
    )
    result = evaluate_posterior_slope_draw_frame(
        posterior,
        curve_id="sr002",
        sample_record=2,
        model_name="two_regime_plateau",
    )

    assert np.isclose(result.loc[0, "primary_slope_mV_dec"], 100.0)
    assert np.isclose(result.loc[0, "slope_high_mV_dec"], 50.0)
    assert result.loc[0, "posterior_source_variable"] == "slope_low_mV_dec"


def test_complete_plateau_model_is_inverted_per_chain_draw() -> None:
    posterior = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "draw": [0, 1, 0, 1],
            "a": [-1.0, -1.0, -1.0, -1.0],
            "log_m": [np.log(10.0)] * 4,
            "L_log10_current": [1.0] * 4,
        }
    )
    result = evaluate_posterior_draw_frame(
        posterior,
        curve_id="sr001",
        sample_record=1,
        model_name="paper_single_plateau",
        transition_width_V=0.01,
        potential_min_V=0.0,
        potential_max_V=0.5,
        observed_current_min_mA_cm2=0.1,
        observed_current_max_mA_cm2=8.0,
        target_current_mA_cm2=2.0,
    )

    assert len(result) == 4
    assert not result.duplicated(["chain", "draw"]).any()
    assert result["eligible"].all()
    assert np.isfinite(result["relative_potential_at_current_V"]).all()
    # The complete model locates the potential at 2 mA, but the registered
    # Tafel target is the local kinetic slope at that coordinate.  Plateau
    # flattening belongs only to the apparent-slope diagnostic.
    assert np.allclose(
        result["local_slope_mV_dec"], result["kinetic_slope_mV_dec"]
    )
    assert (
        result["apparent_slope_mV_dec"] > result["kinetic_slope_mV_dec"]
    ).all()
    assert result["target_slope_semantics"].eq(
        "local_kinetic_slope_at_complete_model_2mA_potential"
    ).all()
    assert not result["extrapolated"].any()


def test_posterior_draw_outside_complete_model_support_is_not_extrapolated() -> None:
    posterior = pd.DataFrame(
        {
            "chain": [0],
            "draw": [0],
            "B_log10_background": [np.log10(3.0)],
            "a": [-2.0],
            "log_m": [np.log(10.0)],
            "L_log10_current": [2.0],
        }
    )
    result = evaluate_posterior_draw_frame(
        posterior,
        curve_id="sr002",
        sample_record=2,
        model_name="single_background_plateau",
        transition_width_V=0.01,
        potential_min_V=0.0,
        potential_max_V=0.5,
        observed_current_min_mA_cm2=0.1,
        observed_current_max_mA_cm2=10.0,
        target_current_mA_cm2=2.0,
    )

    assert not result.loc[0, "eligible"]
    assert not result.loc[0, "extrapolated"]
    assert "does_not_support" in result.loc[0, "exclusion_reason"]
