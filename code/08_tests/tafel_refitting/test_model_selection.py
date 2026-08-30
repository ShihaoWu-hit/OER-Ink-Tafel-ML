import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tafel_refit.analysis as analysis_module
from tafel_refit.analysis import analyze_curve
from tafel_refit.models import FitResult, predict_log10
from tafel_refit.preprocess import ProcessedCurve


def _config() -> dict:
    config = json.loads(Path("config/default.json").read_text())
    config["optimizer_starts_cv"] = 3
    config["optimizer_starts_bootstrap"] = 2
    return config


def _processed_curve(
    model_name: str,
    parameters: dict[str, float],
    *,
    noise_seed: int,
) -> tuple[ProcessedCurve, pd.Series]:
    x = np.linspace(0.0, 0.30, 45)
    transition_width = 2.0 * (x[1] - x[0])
    mean = predict_log10(model_name, x, parameters, transition_width)
    rng = np.random.default_rng(noise_seed)
    y = mean + rng.normal(0.0, 0.012, len(x))
    potential = 1.2 + x
    raw = pd.DataFrame(
        {
            "potential_raw": potential,
            "current_raw": np.power(10.0, y),
            "used_for_log_fit": True,
        }
    )
    binned = pd.DataFrame(
        {
            "potential_V": potential,
            "relative_potential_V": x,
            "log10_current": y,
        }
    )
    processed = ProcessedCurve(
        curve_id="synthetic_null",
        sample_record=1,
        raw_points=raw,
        binned_points=binned,
        metrics={
            "n_effective_bins": len(x),
            "fit_potential_span_V": float(np.ptp(x)),
            "fit_log_current_span_dec": float(np.ptp(y)),
            "potential_bin_width_V": float(x[1] - x[0]),
        },
        fit_eligible=True,
        exclusion_reason="",
        qc_flags=("potential_unit_inferred_V_unverified",),
        bin_width_V=float(x[1] - x[0]),
    )
    manifest = pd.Series(
        {
            "curve_variant": 1,
            "sheet_index": 1,
            "sheet_name": "synthetic",
            "tafel_slope_source": "synthetic",
            "publication_id": "synthetic",
            "doi": "",
            "workbook_doi": "",
            "y_tafel_slope_mV_dec": 60.0,
            "potential_unit_raw": "V",
            "potential_unit_assumption": "synthetic_verified",
            "current_unit_raw": "synthetic",
            "potential_reference_raw": "synthetic",
            "ir_status": "synthetic",
        }
    )
    return processed, manifest


@pytest.mark.parametrize("noise_seed", [7, 29])
def test_full_selection_does_not_flag_synthetic_single_slope_nulls(
    noise_seed: int,
):
    processed, manifest = _processed_curve(
        "paper_single_plateau",
        {"a": -2.0, "log_m": np.log(1000.0 / 60.0), "L": 1.0},
        noise_seed=noise_seed,
    )
    result = analyze_curve(
        processed,
        manifest,
        _config(),
        bootstrap_repetitions=40,
        optimizer_starts=10,
        seed=1000 + noise_seed,
    )

    assert result.summary["fit_status"] == "success"
    assert not result.summary["potential_dependent_supported"]


def test_low_current_floor_shape_is_not_a_primary_model_candidate() -> None:
    processed, manifest = _processed_curve(
        "single_background_plateau",
        {
            "B": -2.2,
            "a": -4.0,
            "log_m": np.log(1000.0 / 60.0),
            "L": 1.0,
        },
        noise_seed=7,
    )
    result = analyze_curve(
        processed,
        manifest,
        _config(),
        bootstrap_repetitions=20,
        optimizer_starts=10,
        seed=1007,
    )

    assert set(result.model_fits["model_name"]) == {
        "linear",
        "paper_single_plateau",
        "two_regime_plateau",
    }
    assert result.summary["selected_model"] in set(result.model_fits["model_name"])
    assert result.summary["primary_model_count"] == 3
    assert (
        result.summary["low_current_floor_primary_policy"]
        == "excluded_without_independent_blank"
    )
    assert result.summary["low_current_floor_sensitivity_implementation_available"]
    assert not result.summary["low_current_floor_sensitivity_run"]
    if result.summary["potential_dependent_supported"]:
        assert (
            result.summary["potential_dependence_floor_ambiguity"]
            == "requires_separate_sensitivity_or_independent_blank"
        )


def test_failed_linear_fit_falls_back_to_a_converged_primary_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed, manifest = _processed_curve(
        "paper_single_plateau",
        {"a": -2.0, "log_m": np.log(1000.0 / 60.0), "L": 1.0},
        noise_seed=11,
    )
    real_fit_model = analysis_module.fit_model

    def fail_linear(model_name: str, x: np.ndarray, y: np.ndarray, **kwargs):
        if model_name != "linear":
            return real_fit_model(model_name, x, y, **kwargs)
        return FitResult(
            model_name="linear",
            parameters={},
            nll=np.inf,
            bic=np.inf,
            mae=np.inf,
            success=False,
            optimizer_message="synthetic failure",
            boundary_hits=(),
            predictions=np.full_like(y, np.nan),
            n_observations=len(y),
            transition_width_V=float(kwargs["transition_width_V"]),
        )

    monkeypatch.setattr(analysis_module, "fit_model", fail_linear)
    result = analyze_curve(
        processed,
        manifest,
        _config(),
        bootstrap_repetitions=10,
        optimizer_starts=8,
        seed=1011,
    )

    assert result.summary["fit_status"] == "success"
    assert result.summary["selected_model"] != "linear"
    assert result.fit_results[result.summary["selected_model"]].success
