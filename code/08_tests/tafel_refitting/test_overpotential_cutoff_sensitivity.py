from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "overpotential_cutoff_sensitivity.py"
)


def _load_module(name: str = "overpotential_cutoff_sensitivity_for_tests") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cutoff_sensitivity = _load_module()


def _points() -> pd.DataFrame:
    potential = np.array([1.20, 1.23, 1.30, 1.33, 1.38, 1.45])
    return pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "potential_bin_index": np.arange(len(potential)),
            "potential_V": potential,
            "relative_potential_V": potential - potential.min(),
            "log10_current": np.linspace(-2.0, 1.0, len(potential)),
        }
    )


def test_eta_cutoff_is_inclusive_and_uses_all_remaining_points() -> None:
    audit, selected = cutoff_sensitivity.select_points_after_eta_cutoff(
        _points(), 0.10, min_bins=3
    )
    np.testing.assert_allclose(selected["potential_V"], [1.33, 1.38, 1.45])
    np.testing.assert_allclose(selected["eta_V"], [0.10, 0.15, 0.22])
    np.testing.assert_allclose(selected["cutoff_relative_potential_V"], [0.0, 0.05, 0.12])
    assert audit["bayesian_fit_eligible"]
    assert audit["retained_n_bins"] == 3
    assert not audit["common_current_span_required"]
    assert not audit["common_decade_span_required"]


def test_cutoff_requires_only_minimum_computational_point_count() -> None:
    audit, selected = cutoff_sensitivity.select_points_after_eta_cutoff(
        _points(), 0.15, min_bins=3
    )
    assert len(selected) == 2
    assert not audit["bayesian_fit_eligible"]
    assert audit["ineligibility_reason"] == "fewer_than_3_bins_after_cutoff"


@pytest.mark.parametrize(
    ("model_name", "variable"),
    [
        ("linear", "slope_mV_dec"),
        ("paper_single_plateau", "slope_mV_dec"),
        ("two_regime_plateau", "slope_low_mV_dec"),
    ],
)
def test_selected_model_slope_variable_is_preserved(model_name: str, variable: str) -> None:
    assert cutoff_sensitivity.expected_slope_variable(model_name) == variable


def test_manifest_has_one_row_per_curve_and_registered_cutoff() -> None:
    points = pd.concat(
        [_points(), _points().assign(curve_id="sr002", sample_record=2)],
        ignore_index=True,
    )
    summary = pd.DataFrame(
        {
            "curve_id": ["sr001", "sr002"],
            "selected_model": ["linear", "two_regime_plateau"],
            "transition_width_V": [0.01, 0.02],
            "qc_flags": ["", ""],
        }
    )
    manifest, selected = cutoff_sensitivity.build_cutoff_manifest(
        points,
        summary,
        cutoffs=cutoff_sensitivity.ETA_CUTOFFS_V,
        min_bins=3,
    )
    assert len(manifest) == 8
    assert set(manifest["eta_min_V"]) == set(cutoff_sensitivity.ETA_CUTOFFS_V)
    assert set(selected["eta_min_V"]) == set(cutoff_sensitivity.ETA_CUTOFFS_V)
    assert manifest.loc[
        manifest["curve_id"].eq("sr002"), "posterior_slope_variable"
    ].eq("slope_low_mV_dec").all()


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
    reloaded = _load_module("overpotential_cutoff_sensitivity_without_pymc")
    assert reloaded.ETA_CUTOFFS_V == (0.0, 0.10, 0.15, 0.20)
    assert attempted == []
