from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
for value in (str(ANALYSIS_DIR),):
    if value not in sys.path:
        sys.path.insert(0, value)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


formal = _load("formal_first_supported_for_contiguous_tests", "formal_first_supported_target.py")
contiguous = _load("contiguous_increase_for_tests", "contiguous_increase_region.py")


def _points(eta: np.ndarray, log_current: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "potential_bin_index": np.arange(len(eta), dtype=int),
            "potential_V": 1.23 + eta,
            "relative_potential_V": eta - eta.min(),
            "log10_current": log_current,
        }
    )


def test_selected_window_is_strictly_increasing_in_observed_coordinates() -> None:
    eta = np.linspace(0.15, 0.34, 30)
    prefix = np.repeat(-1.2, 8)
    signal = np.linspace(-1.0, 0.8, 22)
    points = _points(eta, np.concatenate([prefix, signal]))
    perturbed = formal.perturb_points(
        points,
        potential_resolution_V=0.003,
        log_current_resolution_dec=0.08,
        seed=7,
    )
    audit, selected = contiguous.select_contiguous_increase_region(
        perturbed, formal.RegionConfig()
    )
    assert audit["region_supported"]
    observed = selected["observed_log10_current"].to_numpy(dtype=float)
    assert np.all(np.diff(observed) > 0.0)
    assert audit["window_observed_tie_steps"] == 0
    assert audit["window_observed_decrease_steps"] == 0
    assert selected["potential_bin_index"].min() >= 7


def test_jitter_cannot_turn_an_observed_staircase_into_support() -> None:
    eta = np.linspace(0.15, 0.32, 36)
    log_current = np.repeat(np.linspace(-1.0, 0.4, 9), 4)
    points = _points(eta, log_current)
    for seed in (1, 11, 101):
        perturbed = formal.perturb_points(
            points,
            potential_resolution_V=0.004,
            log_current_resolution_dec=0.10,
            seed=seed,
        )
        audit, selected = contiguous.select_contiguous_increase_region(
            perturbed, formal.RegionConfig()
        )
        assert not audit["region_supported"]
        assert selected.empty


def test_tie_or_reversal_forces_selection_to_a_later_clean_run() -> None:
    eta = np.linspace(0.15, 0.38, 38)
    contaminated = np.array(
        [-1.3, -1.15, -1.0, -0.85, -0.85, -0.70, -0.82, -0.72]
    )
    recovery = np.linspace(-0.6, 0.8, len(eta) - len(contaminated))
    points = _points(eta, np.concatenate([contaminated, recovery]))
    perturbed = formal.perturb_points(
        points,
        potential_resolution_V=0.001,
        log_current_resolution_dec=0.01,
        seed=19,
    )
    audit, selected = contiguous.select_contiguous_increase_region(
        perturbed, formal.RegionConfig()
    )
    assert audit["region_supported"]
    # A window cannot begin on the endpoint of the reversal; it starts only
    # once the next observed positive-current step is present.
    assert selected["potential_bin_index"].min() >= 7
    assert np.all(
        np.diff(selected["observed_log10_current"].to_numpy(dtype=float)) > 0
    )


def test_valid_high_slope_is_not_rejected_by_a_slope_cap() -> None:
    log_current = np.linspace(-0.8, 0.8, 24)
    eta = 0.16 + 0.30 * (log_current - log_current.min())
    points = _points(eta, log_current)
    audit, selected = contiguous.select_contiguous_increase_region(
        points, formal.RegionConfig()
    )
    assert audit["region_supported"]
    assert not selected.empty
    assert audit["selection_used_slope_magnitude"] is False


def test_named_ppc_failures_use_later_strict_windows_on_frozen_source() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "20260721_112453"
        / "csv"
        / "binned_curve_points.csv"
    )
    points = pd.read_csv(source)
    named = [1, 2, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 89, 97]
    starts: dict[int, float] = {}
    for record in named:
        audit, selected = contiguous.select_contiguous_increase_region(
            points.loc[points["sample_record"].eq(record)].copy(),
            formal.RegionConfig(),
        )
        assert audit["region_supported"], record
        observed = selected["observed_log10_current"].to_numpy(dtype=float)
        assert np.all(np.diff(observed) > 0.0), record
        assert float(selected.iloc[0]["observed_step_from_previous"]) > 0.0
        starts[record] = float(selected["observed_eta_V"].iloc[0])
    assert starts[89] >= 0.28
    assert starts[97] >= 0.24


def test_coarsely_digitized_records_remain_unsupported_without_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "20260721_112453"
        / "csv"
        / "binned_curve_points.csv"
    )
    points = pd.read_csv(source)
    for record in (23, 74):
        audit, selected = contiguous.select_contiguous_increase_region(
            points.loc[points["sample_record"].eq(record)].copy(),
            formal.RegionConfig(),
        )
        assert not audit["region_supported"], record
        assert selected.empty


def test_registered_quantization_fallback_recovers_records_23_and_74() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "20260721_112453"
        / "csv"
        / "binned_curve_points.csv"
    )
    points = pd.read_csv(source)
    for record in (23, 74):
        audit, selected = contiguous.select_bayesian_local_line_with_fallback(
            points.loc[points["sample_record"].eq(record)].copy(),
            formal.RegionConfig(),
        )
        assert audit["region_supported"], record
        assert audit["quantization_aware_fallback_used"], record
        assert audit["fallback_raw_tie_steps"] > 0
        assert audit["window_strict_progression_pass"]
        assert float(selected.iloc[0]["observed_step_from_previous"]) > 0.0
        assert np.all(
            np.diff(selected["observed_log10_current"].to_numpy(dtype=float))
            > 0.0
        )


def test_quantization_fallback_refuses_a_pure_platform() -> None:
    eta = np.linspace(0.15, 0.40, 48)
    points = _points(eta, np.repeat(-0.8, len(eta)))
    audit, selected = contiguous.select_bayesian_local_line_with_fallback(
        points, formal.RegionConfig()
    )
    assert not audit["region_supported"]
    assert selected.empty
