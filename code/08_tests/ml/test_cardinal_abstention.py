from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.cardinal_abstention import (
    DEFAULT_COVERAGE_CURVE_THRESHOLDS,
    FIXED_THRESHOLD_SOURCE,
    abstention_coverage_curve,
    entropy_abstention_decisions,
    normalized_predictive_entropy,
    write_entropy_abstention_artifacts,
)
from ink_tafel_study.cardinal_metrics import (
    PREDICTION_COLUMNS,
    SOFT_LABEL_COLUMNS,
)
from ink_tafel_study.config import load_config


def _aggregated_oof() -> pd.DataFrame:
    observed = np.asarray(
        [
            [0.80, 0.10, 0.05, 0.05],
            [0.10, 0.70, 0.10, 0.10],
            [0.05, 0.10, 0.75, 0.10],
            [0.10, 0.10, 0.10, 0.70],
        ]
    )
    predicted = np.asarray(
        [
            [0.97, 0.01, 0.01, 0.01],
            [0.25, 0.25, 0.25, 0.25],
            [0.60, 0.20, 0.10, 0.10],
            [0.34, 0.33, 0.18, 0.15],
        ]
    )
    frame = pd.DataFrame(
        {
            "analysis_id": "test",
            "scope": "alkaline_IrO2_plus_Ir",
            "panel_role": "fixed_recipe_comparison",
            "panel": "legacy_safe_compact",
            "model_role": "fixed_stacked",
            "model_family": "soft_gower_logistic_stack",
            "record_id": [1, 2, 3, 4],
            "cv_group_id": ["g1", "g2", "g3", "g4"],
            "n_repeated_oof": 10,
        }
    )
    for index, column in enumerate(SOFT_LABEL_COLUMNS):
        frame[column] = observed[:, index]
    for index, column in enumerate(PREDICTION_COLUMNS):
        frame[column] = predicted[:, index]
    return frame


def test_abstention_policy_is_preregistered_in_config() -> None:
    settings = load_config().section("cardinal_development")["abstention"]
    assert settings["enabled"] is True
    assert settings["normalized_entropy_threshold"] == 0.90
    assert settings["threshold_source"] == FIXED_THRESHOLD_SOURCE
    assert tuple(settings["coverage_curve_thresholds"]) == (
        DEFAULT_COVERAGE_CURVE_THRESHOLDS
    )
    assert settings["full_coverage_primary_metrics_unchanged"] is True


def test_normalized_entropy_has_expected_extremes() -> None:
    entropy, normalized = normalized_predictive_entropy(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
            ]
        )
    )
    np.testing.assert_allclose(entropy, [0.0, np.log(4.0)], atol=1e-12)
    np.testing.assert_allclose(normalized, [0.0, 1.0], atol=1e-12)


def test_entropy_abstention_never_forces_high_entropy_class() -> None:
    decisions = entropy_abstention_decisions(
        _aggregated_oof(), normalized_entropy_threshold=0.90
    )
    uniform = decisions.loc[decisions["record_id"].eq(2)].iloc[0]
    assert bool(uniform["abstained"]) is True
    assert uniform["reported_decision"] == "uncertain"
    assert uniform["predicted_class_without_abstention"] in {
        "low_30_40_like",
        "60_like",
        "120_like",
        "off_cardinal",
    }
    confident = decisions.loc[decisions["record_id"].eq(1)].iloc[0]
    assert bool(confident["accepted"]) is True
    assert confident["reported_decision"] == "low_30_40_like"
    assert (
        decisions["threshold_used_outer_test_labels"].astype(bool).sum() == 0
    )
    assert (
        decisions["threshold_selection_used_outer_test_data"]
        .astype(bool)
        .sum()
        == 0
    )
    assert decisions["threshold_source"].eq(FIXED_THRESHOLD_SOURCE).all()


def test_non_preregistered_threshold_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="preregistered fixed"):
        entropy_abstention_decisions(
            _aggregated_oof(),
            threshold_source="chosen_on_outer_test_accuracy",
        )


def test_coverage_curve_is_fixed_and_full_coverage_at_one() -> None:
    curve = abstention_coverage_curve(_aggregated_oof())
    assert tuple(
        sorted(curve["normalized_entropy_threshold"].unique())
    ) == DEFAULT_COVERAGE_CURVE_THRESHOLDS
    assert curve["outer_test_metrics_used_to_select_threshold"].eq(
        False
    ).all()
    full = curve.loc[
        curve["normalized_entropy_threshold"].eq(1.0)
    ].iloc[0]
    assert float(full["coverage"]) == 1.0
    primary = curve.loc[curve["is_primary_threshold"]]
    assert len(primary) == 1
    assert float(primary.iloc[0]["normalized_entropy_threshold"]) == 0.90


def test_entropy_abstention_artifacts_are_auditable(
    tmp_path: Path,
) -> None:
    result = write_entropy_abstention_artifacts(
        _aggregated_oof(), tmp_path
    )
    assert len(result["decisions"]) == 4
    assert (
        tmp_path / "abstention_record_decisions.csv"
    ).is_file()
    assert (tmp_path / "abstention_summary.csv").is_file()
    assert (tmp_path / "abstention_coverage_curve.csv").is_file()
    definition = json.loads(
        (tmp_path / "abstention_definition.json").read_text(
            encoding="utf-8"
        )
    )
    assert definition["outer_test_labels_used_to_choose_threshold"] is False
    assert (
        definition[
            "outer_test_predictions_or_labels_used_to_choose_threshold"
        ]
        is False
    )
    assert definition["full_coverage_logloss_and_brier_remain_primary"] is True
    qa = json.loads(
        (tmp_path / "abstention_qa.json").read_text(encoding="utf-8")
    )
    assert qa["status"] == "passed"
    assert qa["coverage_curve_thresholds_match_preregistered_grid"] is True
