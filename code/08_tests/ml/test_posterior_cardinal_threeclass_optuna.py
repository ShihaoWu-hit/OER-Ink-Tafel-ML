from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.config import load_config
from ink_tafel_study.config import StudyConfig
from ink_tafel_study.posterior_cardinal_threeclass_optuna import (
    MODEL_FAMILIES,
    P_COLUMNS,
    Q_COLUMNS,
    SEARCH_PANELS,
    THREE_CLASS_LABELS,
    _load_contract,
    collapse_to_three_classes,
    run_posterior_cardinal_threeclass_optuna,
)


def test_threeclass_merge_preserves_fractional_probability_mass() -> None:
    source = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "cv_group_id": ["g1", "g2"],
            "q_low_30_40_like": [0.20, 0.10],
            "q_60_like": [0.30, 0.20],
            "q_120_like": [0.10, 0.40],
            "q_off_cardinal": [0.40, 0.30],
        }
    )

    merged = collapse_to_three_classes(source)

    assert THREE_CLASS_LABELS == (
        "low_30_40_60_like",
        "120_like",
        "off_cardinal",
    )
    assert np.allclose(
        merged.loc[:, Q_COLUMNS].to_numpy(float),
        [[0.50, 0.10, 0.40], [0.30, 0.40, 0.30]],
    )
    assert np.allclose(merged.loc[:, Q_COLUMNS].sum(axis=1), 1.0)
    assert all(column in merged for column in source)


def test_threeclass_contract_is_config_hash_bound() -> None:
    config = load_config()
    contract, path, digest = _load_contract(config)
    assert contract["contract_id"] == (
        "posterior_cardinal_threeclass_optuna_v1"
    )
    assert path.name == "posterior_cardinal_threeclass_optuna_v1.yaml"
    assert digest == config.value(
        "posterior_cardinal_threeclass_optuna.contract_sha256"
    )

    raw = deepcopy(config.raw)
    raw["posterior_cardinal_threeclass_optuna"]["contract_sha256"] = "0" * 64
    drifted = StudyConfig(
        path=config.path,
        project_root=config.project_root,
        repository_root=config.repository_root,
        raw=raw,
    )
    with pytest.raises(ValueError, match="contract hash mismatch"):
        _load_contract(drifted)


def test_frozen_real_input_end_to_end_smoke(tmp_path: Path) -> None:
    config = load_config()
    destination = tmp_path / "threeclass_optuna_smoke"

    result = run_posterior_cardinal_threeclass_optuna(
        config,
        destination,
        source_run=config.resolve_path("active_source_run"),
        smoke=True,
        n_jobs=1,
        optuna_trials=3,
    )

    qa = json.loads(
        (destination / "analyses" / "analysis_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "complete_smoke"
    assert qa["status"] == "complete_smoke"
    assert qa["scientifically_releasable"] is False
    assert qa["formal_v4_result_modified"] is False
    assert qa["n_records"] == 97
    assert qa["n_connected_groups"] == 80
    assert qa["n_posterior_draws"] == 776_000
    assert qa["draws_per_record"] == 8_000
    assert qa["class_order"] == list(THREE_CLASS_LABELS)
    assert qa["outer_repeats"] == 1
    assert qa["outer_folds"] == 2
    assert qa["inner_folds"] == 2
    assert qa["posterior_argmax_used_for_training_or_selection"] is False
    assert (
        qa["outer_test_used_for_feature_model_or_hyperparameter_selection"]
        is False
    )
    assert np.isclose(
        qa["posterior_expected_hard_accuracy_oracle_ceiling"],
        0.7163105670103093,
    )

    labels = pd.read_csv(
        destination / "registry" / "three_class_soft_labels.csv"
    )
    assert len(labels) == 97
    assert labels["record_id"].nunique() == 97
    assert np.allclose(labels.loc[:, Q_COLUMNS].sum(axis=1), 1.0)
    assert np.allclose(
        labels.loc[:, Q_COLUMNS].sum(axis=0),
        [46.75125, 16.80875, 33.44],
    )

    outer = pd.read_csv(destination / "folds" / "outer_split_manifest.csv")
    inner = pd.read_csv(destination / "folds" / "inner_split_manifest.csv")
    assert outer["record_id"].nunique() == 97
    assert outer["cv_group_id"].nunique() == 80
    assert set(outer["fold"]) == {0, 1}
    assert inner["inner_fold"].nunique() == 2
    assert (
        outer.groupby(["repeat", "cv_group_id"])["fold"].nunique().eq(1).all()
    )
    assert (
        inner.groupby(
            ["outer_repeat", "outer_fold", "cv_group_id"]
        )["inner_fold"]
        .nunique()
        .eq(1)
        .all()
    )

    trials = pd.read_csv(destination / "analyses" / "optuna_trials.csv")
    assert set(trials["model_family"]) == set(MODEL_FAMILIES)
    assert set(trials["panel"]).issubset(set(SEARCH_PANELS))
    assert trials["state"].eq("complete").all()
    assert len(trials) == 2 * 3

    selections = pd.read_csv(
        destination / "analyses" / "outer_selections.csv"
    )
    assert not selections["outer_test_used_for_selection"].astype(bool).any()
    assert selections["selected_panel"].isin(SEARCH_PANELS).all()
    assert selections["selected_model_family"].isin(MODEL_FAMILIES).all()

    oof = pd.read_csv(destination / "analyses" / "oof_probabilities.csv")
    assert len(oof) == 2 * 97
    assert set(oof["evaluation_model"]) == {
        "fold_local_threeclass_soft_prior",
        "nested_optuna_selected_pipeline",
    }
    assert np.isfinite(oof.loc[:, P_COLUMNS].to_numpy(float)).all()
    assert (oof.loc[:, P_COLUMNS].to_numpy(float) >= 0).all()
    assert np.allclose(oof.loc[:, P_COLUMNS].sum(axis=1), 1.0)

    metrics = pd.read_csv(
        destination / "analyses" / "aggregate_metrics.csv"
    )
    assert {
        "full_coverage_three_class_soft_log_loss",
        "multiclass_brier_mean_sum_squared_error",
        "gate_soft_log_loss",
        "gate_brier",
        "conditional_soft_log_loss_weighted_by_q_cardinal",
        "conditional_brier_weighted_by_q_cardinal",
    }.issubset(metrics.columns)

    summary = json.loads(
        (
            destination / "analyses" / "best_pipeline_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["smoke_structural_only"] is True
    assert summary["global_single_best_claim_authorized"] is False
    assert summary["outer_test_metrics_used_to_choose_winner"] is False

    required = (
        "registry/contract_snapshot.yaml",
        "registry/three_class_support.csv",
        "analyses/aggregate_metrics.csv",
        "analyses/connected_group_bootstrap.csv",
        "analyses/connected_group_bootstrap_summary.csv",
        "provenance/contract_binding.json",
        "provenance/input_binding.json",
        "provenance/v4_manifest_binding.json",
        "report/posterior_cardinal_threeclass_optuna_report.md",
        "report/validation_report.json",
    )
    assert all((destination / relative).is_file() for relative in required)


def test_runner_rejects_output_inside_immutable_v4_before_writing() -> None:
    config = load_config()
    v4_root = (
        config.repository_root
        / "RESULTS"
        / "05_WORKBOOKS"
        / "posterior_cardinal_all97_v4_ir_compensation_corrected_20260724"
    )
    forbidden = v4_root / "__threeclass_forbidden_test_child__"
    assert not forbidden.exists()

    with pytest.raises(ValueError, match="must be disjoint"):
        run_posterior_cardinal_threeclass_optuna(
            config,
            forbidden,
            source_run=config.resolve_path("active_source_run"),
            smoke=True,
            optuna_trials=1,
        )

    assert not forbidden.exists()


def test_full_run_rejects_trial_budget_contract_deviation(
    tmp_path: Path,
) -> None:
    config = load_config()
    with pytest.raises(ValueError, match="hash-bound 45 Optuna trials"):
        run_posterior_cardinal_threeclass_optuna(
            config,
            tmp_path / "full_budget_deviation",
            source_run=config.resolve_path("active_source_run"),
            smoke=False,
            optuna_trials=21,
        )
