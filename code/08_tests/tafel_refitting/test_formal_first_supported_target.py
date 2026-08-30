from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "formal_first_supported_target.py"
)
ANALYSIS = str(SCRIPT.parent)
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)
spec = importlib.util.spec_from_file_location("formal_first_supported_for_tests", SCRIPT)
assert spec is not None and spec.loader is not None
formal = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = formal
spec.loader.exec_module(formal)


def _floor_curve() -> pd.DataFrame:
    floor_eta = np.linspace(0.15, 0.19, 10)
    signal_logj = np.linspace(-1.0, 1.0, 36)
    signal_eta = 0.195 + 0.09 * (signal_logj + 1.0)
    eta = np.concatenate([floor_eta, signal_eta])
    logj = np.concatenate([np.repeat(-1.0, 10), signal_logj])
    return pd.DataFrame(
        {
            "curve_id": "sr001",
            "sample_record": 1,
            "potential_bin_index": np.arange(len(eta)),
            "potential_V": 1.23 + eta,
            "relative_potential_V": eta - eta.min(),
            "log10_current": logj,
        }
    )


def test_unified_selector_skips_observed_floor_after_jitter() -> None:
    points = _floor_curve()
    perturbed = formal.perturb_points(
        points,
        potential_resolution_V=0.001,
        log_current_resolution_dec=0.05,
        seed=7,
    )
    audit, selected = formal.select_unified_linear_region(
        perturbed, formal.RegionConfig()
    )
    assert audit["region_supported"]
    assert audit["target_method_level"] == "unified_bayesian_local_linear"
    assert selected["potential_bin_index"].min() >= 7
    assert not audit["quadratic_fallback_used"]
    assert not audit["exact_level_collapse_used"]


def test_interval_perturbation_preserves_rows_instead_of_collapsing_levels() -> None:
    logj = np.repeat(np.linspace(-0.5, 0.2, 10), 4)
    eta = 0.19 + 0.12 * (logj + 0.5) + np.tile(np.linspace(-0.002, 0.002, 4), 10)
    points = pd.DataFrame(
        {
            "curve_id": "sr023",
            "sample_record": 23,
            "potential_bin_index": np.arange(len(eta)),
            "potential_V": 1.23 + eta,
            "relative_potential_V": eta - eta.min(),
            "log10_current": logj,
        }
    )
    perturbed = formal.perturb_points(
        points,
        potential_resolution_V=0.001,
        log_current_resolution_dec=0.05,
        seed=11,
    )
    audit, selected = formal.select_unified_linear_region(
        perturbed, formal.RegionConfig()
    )
    assert audit["region_supported"]
    assert selected["log10_current"].nunique() == len(selected)
    assert audit["exact_level_collapse_used"] is False


def test_window_mixture_has_one_common_estimand_and_fixed_draw_count() -> None:
    points = _floor_curve()
    summary, windows, draws, _ = formal.fit_curve_window_mixture(
        points,
        region_config=formal.RegionConfig(),
        formal_config=formal.FormalConfig(
            selection_replicates=4,
            pooled_draws_per_curve=200,
            seed=19,
        ),
        sampler_config=formal.SamplerConfig(
            draws_per_chain=100,
            tune_per_chain=80,
            chains=2,
            seed=29,
        ),
        curve_seed_offset=1,
    )
    assert summary["target_structurally_eligible"]
    assert summary["selection_uncertainty_propagated"]
    assert summary["posterior_has_hard_upper_bound"] is False
    assert len(windows) == 4
    assert len(draws) == 200
    assert draws["selection_replicate"].nunique() >= 2
    assert 60 < summary["posterior_target_median_mV_dec"] < 130
