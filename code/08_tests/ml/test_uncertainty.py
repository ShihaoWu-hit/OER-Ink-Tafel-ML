from pathlib import Path

import pandas as pd

from ink_tafel_study.uncertainty import score_functional_target_draws


def test_functional_draws_are_synchronized(tmp_path: Path) -> None:
    oof = pd.DataFrame(
        {
                "target": ["b_bayesian_local_line_mV_dec"] * 4,
            "scope": ["alkaline"] * 4,
            "panel": ["context", "context", "context_plus_ink", "context_plus_ink"],
            "model": ["nested_selected"] * 4,
            "record_id": [1, 2, 1, 2],
            "predicted_mV_dec": [50.0, 100.0, 55.0, 95.0],
        }
    )
    draws = pd.DataFrame(
        {
            "sample_record": [1, 2, 1, 2],
            "bootstrap_replicate": [10, 10, 11, 11],
            "primary_slope_mV_dec": [52.0, 98.0, 54.0, 96.0],
        }
    )
    metrics, summary = score_functional_target_draws(oof, draws, tmp_path, max_draws=2)
    assert metrics["target_draw"].nunique() == 2
    assert set(metrics.groupby(["panel", "target_draw"])["n_records"].first()) == {2}
    assert not summary.empty


def test_duplicate_record_draw_is_rejected(tmp_path: Path) -> None:
    oof = pd.DataFrame(
        {
                "target": ["b_bayesian_local_line_mV_dec"],
            "scope": ["alkaline"],
            "panel": ["context"],
            "model": ["nested_selected"],
            "record_id": [1],
            "predicted_mV_dec": [50.0],
        }
    )
    draws = pd.DataFrame(
        {
            "sample_record": [1, 1],
            "bootstrap_replicate": [0, 0],
            "primary_slope_mV_dec": [50.0, 51.0],
        }
    )
    try:
        score_functional_target_draws(oof, draws, tmp_path)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("Expected duplicate functional target draws to fail")


def test_posterior_draw_index_is_used_when_available(tmp_path: Path) -> None:
    oof = pd.DataFrame(
        {
                "target": ["b_bayesian_local_line_mV_dec"],
            "scope": ["alkaline"],
            "panel": ["context"],
            "model": ["nested_selected"],
            "record_id": [1],
            "predicted_mV_dec": [50.0],
        }
    )
    draws = pd.DataFrame(
        {
            "sample_record": [1, 1],
            "posterior_draw_index": [3, 4],
            "primary_slope_mV_dec": [51.0, 52.0],
        }
    )
    metrics, _ = score_functional_target_draws(oof, draws, tmp_path, max_draws=2)
    qa = pd.read_json(tmp_path / "target_draw_qa.json", typ="series")
    assert metrics["target_draw"].nunique() == 2
    assert qa["synchronized_by"] == "posterior_draw_index"
