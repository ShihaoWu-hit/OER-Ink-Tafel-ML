from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "uniform_equation2_targets.py"
)


def _load_module(name: str = "uniform_equation2_targets_for_tests") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


targets = _load_module()


def _binned_points() -> pd.DataFrame:
    potential = np.array([1.30, 1.38, 1.40, 1.45, 1.50])
    return pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "potential_bin_index": np.arange(len(potential)),
            "potential_V": potential,
            "log10_current": np.linspace(-1.0, 1.0, len(potential)),
        }
    )


def _raw_points() -> pd.DataFrame:
    potential = np.array([1.25, 1.30, 1.35, 1.40, 1.45])
    return pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "excel_row": np.arange(2, 7),
            "potential_raw": potential,
            "current_raw": 10 ** np.linspace(-2.0, 1.0, len(potential)),
            "used_for_log_fit": [True, False, True, True, True],
        }
    )


def test_paper_equation2_slope_mapping_recovers_known_slope() -> None:
    slope = 120.0
    alpha = 1000.0 / (targets.LOG10E * slope)
    recovered = targets.paper_alpha_to_slope_mV_dec(alpha)
    np.testing.assert_allclose(recovered, slope)


def test_paper_equation2_is_linear_well_below_plateau() -> None:
    slope = 100.0
    alpha = 1000.0 / (targets.LOG10E * slope)
    x = np.linspace(0.0, 0.08, 9)
    prediction = targets.paper_equation2_log10(x, alpha, 8.0, 1e6)
    fitted_slope = np.polyfit(x, prediction, 1)[0]
    np.testing.assert_allclose(fitted_slope, 1000.0 / slope, rtol=1e-3)


def test_eta015_selector_is_inclusive_and_uses_all_remaining_bins() -> None:
    audit, selected = targets.select_eta015_binned(_binned_points())
    np.testing.assert_allclose(selected["potential_V"], [1.38, 1.40, 1.45, 1.50])
    np.testing.assert_allclose(selected["eta_V"], [0.15, 0.17, 0.22, 0.27])
    assert audit["target_id"] == targets.TARGET_EQ2_ETA015
    assert audit["n_observations"] == 4
    assert not audit["common_current_span_required"]


def test_paper_selector_uses_all_accepted_raw_points_and_shifts_log_current() -> None:
    summary = pd.Series({"log_m": np.nan, "a": np.nan, "L": np.nan})
    audit, selected = targets.select_paper_raw(_raw_points(), summary)
    assert len(selected) == 4
    assert selected["excel_row"].tolist() == [2, 4, 5, 6]
    assert np.isclose(selected["fit_log10_current"].min(), 0.0)
    assert audit["all_available_points_within_curve_rule"]
    assert audit["paper_fixed_sigma_log10"] == 0.10
    assert audit["paper_prior_multiplier"] == 10.0


def test_paper_preconditioner_produces_positive_tenfold_prior_bounds() -> None:
    x = np.linspace(0.0, 0.25, 18)
    truth = np.array([28.0, 4.0, 20.0])
    y = targets.paper_equation2_log10(x, *truth)
    fit = targets.optimize_paper_preconditioner(x, y)
    assert fit["paper_theta_star_alpha_per_V"] > 0
    assert fit["paper_theta_star_b_natural"] > 0
    assert fit["paper_theta_star_ilim_ratio"] > 0
    assert np.isclose(
        fit["paper_prior_alpha_upper_per_V"],
        10 * fit["paper_theta_star_alpha_per_V"],
    )
    assert fit["paper_preconditioner_sse"] < 1e-6


def test_target_names_are_distinct_and_noncanonical() -> None:
    assert set(targets.TARGET_IDS) == {
        "uniform_eq2_eta015",
        "uniform_eq2_paper_replication",
    }
    assert len(set(targets.TARGET_NAMES.values())) == 2
    assert targets.ANALYSIS_ROLE == "sensitivity_target_not_canonical"


def test_top_level_import_does_not_require_bayesian_stack(monkeypatch) -> None:
    real_import = builtins.__import__
    forbidden = {"arviz", "pymc", "pytensor"}
    attempted: list[str] = []

    def guarded_import(name: str, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] in forbidden:
            attempted.append(name)
            raise AssertionError(f"Bayesian stack imported at module load: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    reloaded = _load_module("uniform_equation2_targets_without_pymc")
    assert reloaded.TARGET_IDS == targets.TARGET_IDS
    assert attempted == []
