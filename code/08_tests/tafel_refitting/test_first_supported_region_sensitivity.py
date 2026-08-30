from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "first_supported_region_sensitivity.py"
)


def _load_module(name: str = "first_supported_region_sensitivity_for_tests") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


first_supported = _load_module()


def _points(eta: np.ndarray, log_current: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "potential_bin_index": np.arange(len(eta), dtype=int),
            "potential_V": 1.23 + eta,
            "relative_potential_V": eta - np.min(eta),
            "log10_current": log_current,
        }
    )


def test_isotonic_projection_is_non_decreasing_and_preserves_length() -> None:
    projected = first_supported.isotonic_non_decreasing([0.0, 0.4, 0.2, 0.8, 0.7])
    assert len(projected) == 5
    assert np.all(np.diff(projected) >= -1e-12)
    np.testing.assert_allclose(projected, [0.0, 0.3, 0.3, 0.75, 0.75])


def test_selector_skips_long_digitization_platform_without_reading_slope() -> None:
    floor_eta = np.linspace(0.15, 0.20, 10)
    floor_current = np.repeat(-1.0, len(floor_eta))
    signal_current = np.linspace(-1.0, 1.0, 36)
    signal_eta = 0.205 + 0.09 * (signal_current + 1.0)
    points = _points(
        np.concatenate([floor_eta, signal_eta]),
        np.concatenate([floor_current, signal_current]),
    )
    audit, selected, ledger = first_supported.select_first_supported_region(
        points, first_supported.RegionConfig()
    )
    assert audit["region_supported"]
    assert audit["window_start_bin"] >= 7
    assert audit["window_raw_log_current_span_dec"] >= 0.50
    assert audit["window_unique_current_levels"] >= 5
    assert not audit["selection_used_slope_magnitude"]
    assert not audit["selection_used_record_identity"]
    assert ledger.loc[ledger["selected"], "candidate_start_rank"].tolist() == [
        audit["window_scan_rank"]
    ]
    assert len(selected) == audit["window_n_bins"]


def test_selector_uses_minimal_endpoint_for_each_candidate_start() -> None:
    log_current = np.linspace(-1.0, 1.0, 21)
    eta = 0.15 + 0.10 * (log_current + 1.0)
    audit, selected, ledger = first_supported.select_first_supported_region(
        _points(eta, log_current), first_supported.RegionConfig()
    )
    assert audit["region_supported"]
    assert selected["log10_current"].iloc[-1] - selected["log10_current"].iloc[0] >= 0.50
    if len(selected) > first_supported.RegionConfig().minimum_bins:
        shortened = selected.iloc[:-1]
        projected = first_supported.isotonic_non_decreasing(shortened["log10_current"])
        assert projected[-1] - projected[0] < 0.50
    selected_row = ledger.loc[ledger["selected"]].iloc[0]
    assert selected_row["candidate_has_minimum_support"]


def test_uninformative_curve_returns_unsupported_not_extreme_slope() -> None:
    eta = np.linspace(0.15, 0.45, 30)
    log_current = np.linspace(-1.0, -0.9, 30)
    audit, selected, _ = first_supported.select_first_supported_region(
        _points(eta, log_current), first_supported.RegionConfig()
    )
    assert not audit["region_supported"]
    assert audit["target_method_level"] == "L3_unsupported"
    assert selected.empty


def test_repeated_current_levels_use_global_quantization_fallback() -> None:
    points = first_supported.make_repeated_quantized_curve(
        sample_record=1,
        true_slope_mV_dec=120.0,
    ).assign(curve_id="sr001", sample_record=1)
    audit, selected, _ = first_supported.select_first_supported_region(
        points, first_supported.RegionConfig()
    )
    assert audit["region_supported"]
    assert audit["target_method_level"] == "L2_collapsed_quantized_window_dry_run"
    assert audit["fit_point_rule"] == "median_potential_per_exact_log_current_level"
    assert len(selected) == selected["log10_current"].nunique()
    assert selected["n_collapsed_bins"].max() == 4


def test_gibbs_bayesian_region_recovers_known_slope_without_hard_ceiling() -> None:
    true_slope = 120.0
    log_current = np.linspace(-1.0, 1.0, 30)
    eta = 0.18 + true_slope / 1000.0 * (log_current + 1.0)
    points = _points(eta, log_current)
    audit, selected, _ = first_supported.select_first_supported_region(
        points, first_supported.RegionConfig()
    )
    assert audit["region_supported"]
    summary, draws = first_supported.fit_bayesian_region(
        selected,
        model_kind=str(audit["regional_model_kind"]),
        sampler=first_supported.SamplerConfig(
            draws_per_chain=250,
            tune_per_chain=150,
            chains=3,
            seed=123,
        ),
    )
    assert abs(summary["posterior_target_median_mV_dec"] - true_slope) / true_slope < 0.15
    assert summary["posterior_has_hard_upper_bound"] is False
    assert summary["regional_support_pass_dry_run"]
    assert len(draws) == 750


def test_synthetic_recovery_smoke_has_supported_and_rejected_scenarios() -> None:
    frame, passed = first_supported.run_synthetic_recovery_smoke(
        first_supported.RegionConfig(),
        first_supported.SamplerConfig(
            draws_per_chain=160,
            tune_per_chain=120,
            chains=2,
            seed=55,
        ),
    )
    assert passed
    assert len(frame) == 6
    assert frame.loc[frame["scenario"].eq("uninformative_platform"), "region_supported"].eq(False).all()
    assert frame.loc[~frame["scenario"].eq("uninformative_platform"), "region_supported"].all()
    assert frame["scenario_pass"].all()


def test_top_level_import_does_not_require_pymc_or_arviz(monkeypatch) -> None:
    real_import = builtins.__import__
    forbidden = {"arviz", "pymc", "pytensor"}
    attempted: list[str] = []

    def guarded_import(name: str, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] in forbidden:
            attempted.append(name)
            raise AssertionError(f"Bayesian stack imported at module load: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    reloaded = _load_module("first_supported_region_without_pymc")
    assert reloaded.TARGET_NAME == "b_first_supported_apparent_tafel_mV_dec"
    assert attempted == []
