from __future__ import annotations

import numpy as np
import pandas as pd

from tafel_refit.preprocess import preprocess_curve


def _config() -> dict[str, float | int | str]:
    return {
        "max_effective_bins": 60,
        "minimum_bin_width_V": 0.002,
        "minimum_fit_bins": 12,
        "minimum_fit_span_V": 0.08,
        "minimum_log_current_span_dec": 0.5,
        "minimum_spearman_logj_vs_potential": 0.0,
        "manual_review_spearman_below": 0.9,
        "left_censoring_limit_status": "unknown_not_measured_no_formal_censored_likelihood",
    }


def _manifest() -> pd.Series:
    return pd.Series(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "n_unique_potential": 51,
            "n_finite_pairs": 51,
            "current_unit_raw": "unknown_current_density",
        }
    )


def test_long_leading_nonpositive_prefix_is_left_censored_and_never_logged() -> None:
    potential = np.linspace(1.2, 1.5, 51)
    current = 10 ** ((potential - 1.2) * 12.0 - 2.0)
    current[:20] = 0.0
    points = pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "sheet_index": 1,
            "sheet_name": "Sheet1",
            "excel_row": np.arange(2, 53),
            "potential_raw": potential,
            "current_raw": current,
            "finite_pair": True,
            "positive_current": current > 0,
        }
    )
    processed = preprocess_curve(_manifest(), points, _config())
    assert processed.fit_eligible
    assert "leading_nonpositive_left_censored_status_only" in processed.qc_flags
    assert "left_censoring_limit_unverified" in processed.qc_flags
    assert processed.metrics["n_nonpositive_selected_branch"] == 20
    assert processed.metrics["n_leading_left_censored"] == 20
    assert processed.metrics["n_nonpositive_after_first_reliable_positive"] == 0
    assert processed.metrics["nonpositive_fraction_selected_branch"] > 0.35
    assert processed.metrics["nonpositive_confined_to_pre_reliable_region"]
    assert np.isfinite(processed.binned_points["log10_current"]).all()
    assert (processed.raw_points.loc[processed.raw_points["used_for_log_fit"], "current_raw"] > 0).all()
    assert processed.raw_points["left_censored_nonpositive"].sum() == 20


def test_small_pre_onset_positive_run_does_not_define_reliable_region() -> None:
    potential = np.linspace(1.2, 1.5, 51)
    current = 10 ** ((potential - 1.2) * 12.0 - 2.0)
    current[10] = 0.0
    points = pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "sheet_index": 1,
            "sheet_name": "Sheet1",
            "excel_row": np.arange(2, 53),
            "potential_raw": potential,
            "current_raw": current,
            "finite_pair": True,
            "positive_current": current > 0,
        }
    )
    processed = preprocess_curve(_manifest(), points, _config())
    assert processed.fit_eligible
    assert processed.metrics["reliable_positive_run_ordinal"] == 2
    assert processed.metrics["n_pre_reliable_positive_points"] == 10
    assert processed.metrics["n_nonpositive_after_first_numeric_positive"] == 1
    assert processed.metrics["n_nonpositive_after_first_reliable_positive"] == 0
    assert processed.raw_points["pre_reliable_onset_positive"].sum() == 10
    assert processed.raw_points["left_censored_nonpositive"].sum() == 1
    assert processed.raw_points["used_for_log_fit"].sum() == 40


def test_nonpositive_after_established_reliable_region_is_excluded() -> None:
    potential = np.linspace(1.2, 1.65, 91)
    current = 10 ** ((potential - 1.2) * 8.0 - 2.0)
    current[31] = 0.0
    points = pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "sheet_index": 1,
            "sheet_name": "Sheet1",
            "excel_row": np.arange(2, 93),
            "potential_raw": potential,
            "current_raw": current,
            "finite_pair": True,
            "positive_current": current > 0,
        }
    )
    manifest = _manifest().copy()
    manifest["n_unique_potential"] = len(points)
    manifest["n_finite_pairs"] = len(points)
    processed = preprocess_curve(manifest, points, _config())
    assert not processed.fit_eligible
    assert processed.metrics["reliable_positive_run_ordinal"] == 1
    assert processed.metrics["n_nonpositive_after_first_reliable_positive"] == 1
    assert (
        processed.exclusion_reason
        == "nonpositive_after_first_reliable_positive"
    )
    assert not processed.raw_points["used_for_log_fit"].any()


def test_decreasing_positive_region_fails_monotonicity_gate() -> None:
    potential = np.linspace(1.2, 1.5, 51)
    current = 10 ** (2.0 - (potential - 1.2) * 12.0)
    points = pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "sheet_index": 1,
            "sheet_name": "Sheet1",
            "excel_row": np.arange(2, 53),
            "potential_raw": potential,
            "current_raw": current,
            "finite_pair": True,
            "positive_current": True,
        }
    )
    processed = preprocess_curve(_manifest(), points, _config())
    assert not processed.fit_eligible
    assert np.isclose(processed.metrics["spearman_logj_vs_potential"], -1.0)
    assert "insufficient_positive_region_monotonicity" in processed.exclusion_reason


def test_zero_direction_positive_region_fails_monotonicity_gate() -> None:
    potential = np.linspace(1.2, 1.31, 12)
    zero_correlation_ranks = np.array([2, 10, 9, 5, 4, 3, 8, 6, 1, 0, 7, 11])
    log_current = zero_correlation_ranks / 3.0 - 2.0
    current = 10**log_current
    points = pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "sheet_index": 1,
            "sheet_name": "Sheet1",
            "excel_row": np.arange(2, 14),
            "potential_raw": potential,
            "current_raw": current,
            "finite_pair": True,
            "positive_current": True,
        }
    )
    processed = preprocess_curve(_manifest(), points, _config())
    assert not processed.fit_eligible
    assert abs(processed.metrics["spearman_logj_vs_potential"]) < 1e-12
    assert "insufficient_positive_region_monotonicity" in processed.exclusion_reason


def test_decreasing_acquisition_is_checked_from_low_to_high_potential() -> None:
    potential = np.linspace(1.5, 1.2, 51)
    current = 10 ** ((potential - 1.2) * 12.0 - 2.0)
    current[-20:] = 0.0
    points = pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "sheet_index": 1,
            "sheet_name": "Sheet1",
            "excel_row": np.arange(2, 53),
            "potential_raw": potential,
            "current_raw": current,
            "finite_pair": True,
            "positive_current": current > 0,
        }
    )
    processed = preprocess_curve(_manifest(), points, _config())
    assert processed.fit_eligible
    assert processed.metrics["potential_order"] == "decreasing_reversed"
    assert processed.metrics["n_leading_left_censored"] == 20
    assert processed.metrics["n_nonpositive_after_first_reliable_positive"] == 0


def test_scan_reset_step_is_not_joined_to_reverse_branch() -> None:
    up = np.linspace(1.39, 1.70, 160)
    reverse_tail = np.linspace(1.38, 1.30, 42)
    potential = np.r_[up, reverse_tail]
    current = 10 ** ((potential - 1.2) * 8.0 - 2.0)
    points = pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "sheet_index": 1,
            "sheet_name": "Sheet1",
            "excel_row": np.arange(2, len(potential) + 2),
            "potential_raw": potential,
            "current_raw": current,
            "finite_pair": True,
            "positive_current": True,
        }
    )
    manifest = _manifest().copy()
    manifest["n_unique_potential"] = len(points)
    manifest["n_finite_pairs"] = len(points)
    processed = preprocess_curve(manifest, points, _config())
    assert processed.fit_eligible
    assert processed.metrics["potential_order"] == "increasing"
    assert processed.metrics["n_branch_points"] == 160
    assert processed.metrics["fit_potential_span_V"] > 0.30


def test_density_invariance_after_equal_potential_binning() -> None:
    sparse_potential = np.linspace(1.2, 1.6, 81)
    dense_potential = np.linspace(1.2, 1.6, 801)

    def build(potential: np.ndarray) -> pd.DataFrame:
        current = 10 ** ((potential - 1.2) * (1000.0 / 60.0) - 2.0)
        return pd.DataFrame(
            {
                "curve_id": "sr001",
                "sample_record": 1,
                "sheet_index": 1,
                "sheet_name": "Sheet1",
                "excel_row": np.arange(2, len(potential) + 2),
                "potential_raw": potential,
                "current_raw": current,
                "finite_pair": True,
                "positive_current": True,
            }
        )

    sparse_manifest = _manifest().copy()
    sparse_manifest["n_unique_potential"] = len(sparse_potential)
    sparse_manifest["n_finite_pairs"] = len(sparse_potential)
    dense_manifest = _manifest().copy()
    dense_manifest["n_unique_potential"] = len(dense_potential)
    dense_manifest["n_finite_pairs"] = len(dense_potential)
    sparse = preprocess_curve(sparse_manifest, build(sparse_potential), _config())
    dense = preprocess_curve(dense_manifest, build(dense_potential), _config())
    sparse_slope = np.polyfit(
        sparse.binned_points["relative_potential_V"],
        sparse.binned_points["log10_current"],
        1,
    )[0]
    dense_slope = np.polyfit(
        dense.binned_points["relative_potential_V"],
        dense.binned_points["log10_current"],
        1,
    )[0]
    assert abs(sparse_slope - dense_slope) < 0.05
    assert len(sparse.binned_points) <= 61
    assert len(dense.binned_points) <= 61
