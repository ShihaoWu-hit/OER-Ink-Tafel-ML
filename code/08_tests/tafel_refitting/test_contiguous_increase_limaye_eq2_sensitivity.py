from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import contiguous_increase_limaye_eq2_sensitivity as sensitivity
import finalize_contiguous_increase_limaye_eq2_sensitivity as finalize


def test_representative_window_uses_unperturbed_strict_points() -> None:
    summary = pd.Series(
        {
            "sample_record": 1,
            "window_start_eta_median_V": 0.20,
            "window_end_eta_median_V": 0.25,
        }
    )
    frames = []
    for replicate, start in [(0, 0.20), (1, 0.205)]:
        eta = np.linspace(start, start + 0.05, 6)
        frames.append(
            pd.DataFrame(
                {
                    "selection_replicate": replicate,
                    "observed_eta_V": eta,
                    "observed_potential_V": eta + 1.23,
                    "observed_log10_current": np.linspace(-1.0, -0.4, 6),
                    "potential_bin_index": np.arange(6),
                }
            )
        )
    selected, replicate = sensitivity.choose_representative_window(
        summary, pd.concat(frames, ignore_index=True)
    )
    assert replicate == 0
    assert np.all(np.diff(selected["unshifted_log10_current"]) > 0)
    assert np.isclose(selected["fit_log10_current"].min(), 0.0)
    np.testing.assert_allclose(
        selected["potential_V"] - 1.23, selected["observed_eta_V"]
    )


def _attempt_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = np.arange(1, 98)
    base = pd.DataFrame(
        {
            "sample_record": records,
            "curve_id": [f"sr{record:03d}" for record in records],
            "fit_status": finalize.COMPLETE,
            "sampler_diagnostics_pass": True,
            "posterior_prior_boundary_warning": False,
            "posterior_slope_median_mV_dec": 60.0,
        }
    )
    base.loc[base["sample_record"].isin([23, 74]), "fit_status"] = (
        "unsupported_no_strict_window"
    )
    first = base.copy()
    first.loc[first["sample_record"].eq(1), "sampler_diagnostics_pass"] = False
    first.loc[first["sample_record"].eq(1), "posterior_slope_median_mV_dec"] = 80.0
    second = base.copy()
    second.loc[~second["sample_record"].eq(1), "fit_status"] = np.nan
    second.loc[second["sample_record"].eq(1), "posterior_slope_median_mV_dec"] = 65.0
    return first, second


def test_merge_uses_retry_only_after_initial_diagnostic_failure() -> None:
    first, second = _attempt_frames()
    merged = finalize.merge_attempts(first, second).set_index("sample_record")
    assert merged.loc[1, "selected_sampling_attempt"] == 2
    assert merged.loc[1, "posterior_slope_median_mV_dec"] == 65.0
    assert merged.loc[2, "selected_sampling_attempt"] == 1
    assert merged.loc[23, "fit_status"] == "unsupported_no_strict_window"
    assert bool(merged.loc[1, "third_automatic_attempt_forbidden"])
