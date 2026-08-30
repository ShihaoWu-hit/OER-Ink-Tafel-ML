from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "low_potential_window_sensitivity.py"
)


def _load_module(name: str = "low_potential_window_sensitivity_for_tests") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


window_sensitivity = _load_module()


def _curve_points(log_current: list[float]) -> pd.DataFrame:
    relative_potential = np.arange(len(log_current), dtype=float) / 10.0
    return pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "potential_bin_index": np.arange(len(log_current), dtype=int),
            "potential_V": 1.30 + relative_potential,
            "relative_potential_V": relative_potential,
            "log10_current": log_current,
        }
    )


def _selection_row(
    curve_id: str,
    sample_record: int,
    window: float,
    *,
    supported: bool = True,
    eligible: bool = True,
) -> dict[str, object]:
    return {
        "curve_id": curve_id,
        "sample_record": sample_record,
        "window_decades_requested": window,
        "span_supported": supported,
        "bayesian_fit_eligible": eligible,
        "n_window_bins": 20 if supported else 0,
        "window_log_current_span_dec": window if supported else np.nan,
        "ineligibility_reason": (
            "" if eligible else "observed_log_current_span_below_window"
        ),
    }


def _fit_result(
    curve_id: str,
    sample_record: int,
    window: float,
    slope: float,
    *,
    status: str = "bayesian_sampling_complete",
    diagnostics_pass: bool = True,
    prior_boundary_warning: bool = False,
) -> dict[str, object]:
    return {
        "curve_id": curve_id,
        "sample_record": sample_record,
        "window_decades_requested": window,
        "fit_status": status,
        "posterior_slope_median_mV_dec": slope,
        "posterior_slope_hdi_low_mV_dec": slope - 5.0,
        "posterior_slope_hdi_high_mV_dec": slope + 5.0,
        "max_r_hat": 1.001,
        "min_ess_bulk": 800.0,
        "divergences": 0,
        "sampler_diagnostics_pass": diagnostics_pass,
        "posterior_slope_prior_boundary_warning": prior_boundary_warning,
    }


def _summary_and_baseline(
    records: list[tuple[str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        {"curve_id": [item[0] for item in records], "sample_record": [item[1] for item in records]}
    )
    baseline = pd.DataFrame(
        {
            "curve_id": [item[0] for item in records],
            "model_name": "linear",
            "posterior_slope_variable": "slope_mV_dec",
            "posterior_slope_median_mV_dec": 100.0,
            "posterior_slope_draw_hdi_low_mV_dec": 90.0,
            "posterior_slope_draw_hdi_high_mV_dec": 110.0,
            "sampler_diagnostics_pass": True,
        }
    )
    return summary, baseline


def test_selects_shortest_low_potential_prefix_at_first_span_crossing() -> None:
    ordered = _curve_points([0.20, -0.10, 0.40, 1.45, 1.20, 1.80])
    shuffled = ordered.iloc[[5, 2, 0, 4, 1, 3]].reset_index(drop=True)

    audit, selected = window_sensitivity.select_low_potential_window(
        shuffled,
        1.5,
        min_bins=3,
    )

    assert audit["span_supported"]
    assert audit["bayesian_fit_eligible"]
    assert audit["n_window_bins"] == 4
    assert np.isclose(audit["window_log_current_span_dec"], 1.55)
    assert np.isclose(audit["window_span_overshoot_dec"], 0.05)
    assert audit["window_end_relative_potential_V"] == 0.3
    np.testing.assert_allclose(
        selected["relative_potential_V"].to_numpy(dtype=float),
        [0.0, 0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        selected["log10_current"].to_numpy(dtype=float),
        [0.20, -0.10, 0.40, 1.45],
    )


def test_insufficient_span_is_unsupported_and_not_counted_as_unstable() -> None:
    points = _curve_points([0.0, 0.2, 0.1, 0.5, 0.8, 1.0])
    selection_rows = []
    for window in window_sensitivity.WINDOWS:
        audit, selected = window_sensitivity.select_low_potential_window(
            points,
            window,
            min_bins=3,
        )
        selection_rows.append(audit)
        assert not audit["span_supported"]
        assert not audit["bayesian_fit_eligible"]
        assert audit["ineligibility_reason"] == "observed_log_current_span_below_window"
        assert selected.empty

    summary, baseline = _summary_and_baseline([("sr001", 1)])
    stability = window_sensitivity.build_stability_table(
        pd.DataFrame(selection_rows),
        pd.DataFrame(),
        summary,
        baseline,
        requested_records={1},
    )
    row = stability.iloc[0]
    assert not bool(row["stability_evaluable"])
    assert row["stability_status_reason"] == "window_1p5_ineligible"
    assert pd.isna(row["window_local_posterior_median_ratio_within_threshold"])


def test_stability_requires_both_successful_diagnostic_passes() -> None:
    records = [
        ("sr001", 1),
        ("sr002", 2),
        ("sr003", 3),
        ("sr004", 4),
        ("sr005", 5),
        ("sr006", 6),
    ]
    selection_rows: list[dict[str, object]] = []
    for curve_id, sample_record in records:
        selection_rows.append(_selection_row(curve_id, sample_record, 1.5))
        selection_rows.append(
            _selection_row(
                curve_id,
                sample_record,
                2.0,
                supported=curve_id != "sr005",
                eligible=curve_id != "sr005",
            )
        )

    results = pd.DataFrame(
        [
            _fit_result("sr001", 1, 1.5, 100.0),
            _fit_result("sr001", 1, 2.0, 125.0),
            _fit_result("sr002", 2, 1.5, 100.0),
            _fit_result("sr002", 2, 2.0, 126.0),
            _fit_result("sr003", 3, 1.5, 100.0),
            _fit_result(
                "sr003",
                3,
                2.0,
                np.nan,
                status="bayesian_sampling_failed",
                diagnostics_pass=False,
            ),
            _fit_result("sr004", 4, 1.5, 100.0),
            _fit_result("sr004", 4, 2.0, 110.0, diagnostics_pass=False),
            _fit_result("sr005", 5, 1.5, 100.0),
            _fit_result("sr006", 6, 1.5, 100.0),
            _fit_result(
                "sr006", 6, 2.0, 105.0, prior_boundary_warning=True
            ),
        ]
    )
    summary, baseline = _summary_and_baseline(records)

    stability = window_sensitivity.build_stability_table(
        pd.DataFrame(selection_rows),
        results,
        summary,
        baseline,
        requested_records={1, 2, 3, 4, 5, 6},
    ).set_index("curve_id")

    exact_threshold = stability.loc["sr001"]
    assert bool(exact_threshold["stability_evaluable"])
    assert exact_threshold["slope_fold_ratio_1p5_vs_2p0"] == 1.25
    assert exact_threshold["stability_ratio_threshold"] == 1.25
    assert bool(
        exact_threshold["window_local_posterior_median_ratio_within_threshold"]
    )
    assert bool(exact_threshold["formal_target_window_stability_evaluable"])
    assert bool(
        exact_threshold["formal_linear_target_stable_within_ratio_threshold"]
    )

    above_threshold = stability.loc["sr002"]
    assert bool(above_threshold["stability_evaluable"])
    assert above_threshold["slope_fold_ratio_1p5_vs_2p0"] == 1.26
    assert not bool(
        above_threshold["window_local_posterior_median_ratio_within_threshold"]
    )

    failed = stability.loc["sr003"]
    assert not bool(failed["stability_evaluable"])
    assert failed["stability_status_reason"] == "bayesian_sampling_failed_or_missing"
    assert pd.isna(
        failed["window_local_posterior_median_ratio_within_threshold"]
    )

    diagnostics_failed = stability.loc["sr004"]
    assert not bool(diagnostics_failed["stability_evaluable"])
    assert diagnostics_failed["stability_status_reason"] == "sampler_diagnostics_failed"
    assert pd.isna(
        diagnostics_failed["window_local_posterior_median_ratio_within_threshold"]
    )

    unsupported = stability.loc["sr005"]
    assert not bool(unsupported["stability_evaluable"])
    assert unsupported["stability_status_reason"] == "window_2p0_ineligible"
    assert pd.isna(
        unsupported["window_local_posterior_median_ratio_within_threshold"]
    )

    boundary = stability.loc["sr006"]
    assert not bool(boundary["stability_evaluable"])
    assert (
        boundary["stability_status_reason"]
        == "posterior_slope_prior_boundary_warning"
    )
    assert pd.isna(
        boundary["window_local_posterior_median_ratio_within_threshold"]
    )


def test_top_level_import_does_not_require_pymc_stack(monkeypatch) -> None:
    real_import = builtins.__import__
    forbidden = {"arviz", "pymc", "pytensor"}
    attempted: list[str] = []

    def guarded_import(name: str, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] in forbidden:
            attempted.append(name)
            raise AssertionError(f"Bayesian stack imported at module load: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    reloaded = _load_module("low_potential_window_sensitivity_without_pymc")

    assert reloaded.WINDOWS == (1.5, 2.0)
    assert attempted == []


def test_sensitivity_ppc_uses_relative_potential_without_reference_claim(
    tmp_path,
) -> None:
    x = np.array([0.0, 0.1, 0.2])
    observed = np.array([-2.0, -1.5, -1.0])
    draws = np.vstack([observed - 0.05, observed, observed + 0.05])
    output = tmp_path / "ppc.png"

    metrics = window_sensitivity.render_relative_potential_ppc(
        curve_id="sr001",
        window_decades=1.5,
        relative_potential_V=x,
        observed_log10_current=observed,
        posterior_predictive_draws=draws,
        output_path=output,
    )

    assert output.is_file()
    assert metrics["relative_potential_min_V"] == 0.0
    assert metrics["relative_potential_max_V"] == 0.2
    assert not any("overpotential" in key for key in metrics)


def test_completed_fit_resume_validates_identity_and_rejects_partial_set(
    tmp_path,
) -> None:
    fit_dir = tmp_path / "complete"
    fit_dir.mkdir()
    paths = window_sensitivity._fit_artifact_paths(fit_dir)
    for key in ("netcdf", "posterior_summary", "slope_draws", "diagnostics", "ppc"):
        paths[key].write_text(key, encoding="utf-8")
    provenance = {
        "run_configuration_sha256": "config-hash",
        "source_binned_points_sha256": "points-hash",
    }
    result = {
        "curve_id": "sr001",
        "sample_record": 1,
        "window_decades_requested": 1.5,
        "sampling_seed": 123,
        "draws_per_chain": 100,
        "tune_per_chain": 50,
        "chains": 2,
        "target_accept": 0.95,
        "window_points_sha256": "window-hash",
        **provenance,
        "artifact_sha256": {
            key: window_sensitivity.sha256_file(paths[key])
            for key in (
                "netcdf",
                "posterior_summary",
                "slope_draws",
                "diagnostics",
                "ppc",
            )
        },
    }
    paths["result"].write_text(json.dumps(result), encoding="utf-8")

    loaded = window_sensitivity._load_completed_fit(
        fit_dir,
        curve_id="sr001",
        sample_record=1,
        window_decades=1.5,
        sampling_seed=123,
        draws=100,
        tune=50,
        chains=2,
        target_accept=0.95,
        window_points_sha256="window-hash",
        provenance=provenance,
    )
    assert loaded == result
    with pytest.raises(RuntimeError, match="identity mismatch"):
        window_sensitivity._load_completed_fit(
            fit_dir,
            curve_id="sr001",
            sample_record=1,
            window_decades=2.0,
            sampling_seed=123,
            draws=100,
            tune=50,
            chains=2,
            target_accept=0.95,
            window_points_sha256="window-hash",
            provenance=provenance,
        )

    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    (partial_dir / "posterior.nc").write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Partial Bayesian artifact set"):
        window_sensitivity._load_completed_fit(
            partial_dir,
            curve_id="sr001",
            sample_record=1,
            window_decades=1.5,
            sampling_seed=123,
            draws=100,
            tune=50,
            chains=2,
            target_accept=0.95,
            window_points_sha256="window-hash",
            provenance=provenance,
        )
