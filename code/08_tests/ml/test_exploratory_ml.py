from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from ink_tafel_study.exploratory_ml import (
    AUDIT_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROPERTY_PROXY_CATEGORICAL_FEATURES,
    PROPERTY_PROXY_FEATURE_COLUMNS,
    PROPERTY_PROXY_NUMERIC_FEATURES,
    TARGET_COLUMN,
    export_context_plus_ink_dataset,
    export_property_proxy_dataset,
    run_exploratory_benchmark,
    validate_ml_dataset,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_registered_context_plus_ink_export_has_exact_schema_and_grain(tmp_path: Path) -> None:
    output = tmp_path / "ml.csv"
    frame = export_context_plus_ink_dataset(
        records_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/data/records.csv",
        targets_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/targets/targets_wide.csv",
        panel_schema_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/data/registered_panel_schema.csv",
        output_path=output,
    )
    assert frame.shape == (97, 26)
    assert tuple(frame.columns) == AUDIT_COLUMNS + FEATURE_COLUMNS + (TARGET_COLUMN,)
    assert len(FEATURE_COLUMNS) == 18
    assert "solvent_fraction_methanol" in FEATURE_COLUMNS
    assert set(CATEGORICAL_FEATURES).isdisjoint(NUMERIC_FEATURES)
    assert set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) == set(FEATURE_COLUMNS)
    assert frame[TARGET_COLUMN].notna().all()
    assert frame["record_id"].is_unique
    assert frame["record_uid"].is_unique
    assert frame["electrolyte_scope"].value_counts().to_dict() == {
        "alkaline": 90,
        "acidic": 5,
        "neutral": 2,
    }
    pd.testing.assert_frame_equal(frame, pd.read_csv(output), check_dtype=False)


def test_dataset_validator_rejects_target_leakage_shape() -> None:
    columns = list(AUDIT_COLUMNS + FEATURE_COLUMNS + (TARGET_COLUMN,))
    frame = pd.DataFrame(columns=columns)
    frame.insert(len(AUDIT_COLUMNS), "posterior_interval", [])
    with pytest.raises(ValueError, match="columns or order"):
        validate_ml_dataset(frame)


def test_registered_property_proxy_export_replaces_solvent_coordinates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "property_proxy.csv"
    frame = export_property_proxy_dataset(
        records_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/data/records.csv",
        targets_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/targets/targets_wide.csv",
        panel_schema_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/data/registered_panel_schema.csv",
        output_path=output,
    )

    assert frame.shape == (97, 26)
    assert tuple(frame.columns) == (
        AUDIT_COLUMNS + PROPERTY_PROXY_FEATURE_COLUMNS + (TARGET_COLUMN,)
    )
    assert len(PROPERTY_PROXY_FEATURE_COLUMNS) == 18
    assert not any(column.startswith("solvent_fraction_") for column in frame.columns)
    assert set(PROPERTY_PROXY_CATEGORICAL_FEATURES).isdisjoint(
        PROPERTY_PROXY_NUMERIC_FEATURES
    )
    assert set(PROPERTY_PROXY_CATEGORICAL_FEATURES) | set(
        PROPERTY_PROXY_NUMERIC_FEATURES
    ) == set(PROPERTY_PROXY_FEATURE_COLUMNS)
    property_columns = [
        column
        for column in PROPERTY_PROXY_FEATURE_COLUMNS
        if column.startswith("solvent_mixture_property_proxy_")
    ]
    assert len(property_columns) == 5
    assert frame[property_columns].notna().all().all()
    assert (
        frame[
            "solvent_mixture_property_proxy_dielectric_constant_estimated_flag"
        ].sum()
        == 31
    )
    assert frame["electrolyte_scope"].value_counts().to_dict() == {
        "alkaline": 90,
        "acidic": 5,
        "neutral": 2,
    }
    pd.testing.assert_frame_equal(frame, pd.read_csv(output), check_dtype=False)


@pytest.mark.skipif(importlib.util.find_spec("catboost") is None, reason="CatBoost optional")
def test_actual_dataset_smoke_benchmark_writes_complete_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml.csv"
    frame = export_context_plus_ink_dataset(
        records_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/data/records.csv",
        targets_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/targets/targets_wide.csv",
        panel_schema_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/data/registered_panel_schema.csv",
        output_path=dataset_path,
    )
    output_dir = tmp_path / "smoke"
    result = run_exploratory_benchmark(
        frame,
        output_dir=output_dir,
        smoke=True,
        optuna_trials=2,
    )
    assert len(result.metrics) == 6
    assert len(result.predictions) == 6 * 20
    assert result.audit["n_train"] == 77
    assert result.audit["n_test"] == 20
    assert result.audit["connected_groups_crossing_train_test"] == 7
    assert result.audit["n_test_records_in_overlapping_groups"] == 8
    assert set(result.predictions["model"]) == set(result.metrics["model"])
    assert result.predictions["predicted_mV_dec"].gt(0).all()
    for name in (
        "benchmark_metrics.csv",
        "test_predictions.csv",
        "split_manifest.csv",
        "best_params.json",
        "optuna_trials.csv",
        "benchmark_audit.json",
        "README.md",
    ):
        assert (output_dir / name).is_file()
    assert not (output_dir / "catboost_info").exists()


@pytest.mark.skipif(importlib.util.find_spec("catboost") is None, reason="CatBoost optional")
def test_property_proxy_dataset_smoke_runs_all_models(tmp_path: Path) -> None:
    dataset_path = tmp_path / "property_proxy.csv"
    frame = export_property_proxy_dataset(
        records_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/data/records.csv",
        targets_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/targets/targets_wide.csv",
        panel_schema_path=REPOSITORY_ROOT
        / "Data/03_outputs/study/outputs/current_97/data/registered_panel_schema.csv",
        output_path=dataset_path,
    )
    output_dir = tmp_path / "smoke"
    result = run_exploratory_benchmark(
        frame,
        output_dir=output_dir,
        smoke=True,
        optuna_trials=2,
    )

    assert len(result.metrics) == 6
    assert len(result.predictions) == 6 * 20
    assert result.audit["feature_panel"] == "context_plus_property_proxy_4p_estimated"
    assert result.audit["n_predictors"] == 18
    assert result.audit["n_train"] == 77
    assert result.audit["n_test"] == 20
    assert result.audit["connected_groups_crossing_train_test"] == 7
    assert result.audit["n_test_records_in_overlapping_groups"] == 8
    assert set(result.predictions["model"]) == set(result.metrics["model"])
    assert result.predictions["predicted_mV_dec"].gt(0).all()
    assert not (output_dir / "catboost_info").exists()
