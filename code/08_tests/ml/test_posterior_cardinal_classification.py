from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.config import load_config
from ink_tafel_study.posterior_cardinal_classification import (
    ALL_PANELS,
    FORMAL_PANELS,
    PRIMARY_MODEL,
    SECONDARY_MODEL,
    SENSITIVITY_PANELS,
    build_release_decisions,
    run_posterior_cardinal_classification,
)


def _relative_tree(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
        )
    )


def test_runner_rejects_stale_run_directory_before_modeling(
    tmp_path: Path,
) -> None:
    source = tmp_path / "formal_source"
    source.mkdir()
    destination = tmp_path / "stale_run"
    destination.mkdir()
    (destination / "stale.txt").write_text("old run", encoding="utf-8")
    tree_before = _relative_tree(destination)

    with pytest.raises(FileExistsError, match="new or contain only"):
        run_posterior_cardinal_classification(
            object(),  # Path validation must precede all config/input access.
            destination,
            source_run=source,
            smoke=True,
        )

    assert _relative_tree(destination) == tree_before
    for name in ("inputs", "registry", "folds", "analyses", "effects", "report"):
        assert not (destination / name).exists()


@pytest.mark.parametrize(
    "overlap",
    ("destination_below_source", "source_below_destination"),
)
def test_runner_rejects_source_destination_overlap_before_modeling(
    tmp_path: Path,
    overlap: str,
) -> None:
    if overlap == "destination_below_source":
        source = tmp_path / "formal_source"
        source.mkdir()
        destination = source / "classification"
        existing_root = source
    else:
        destination = tmp_path / "classification"
        source = destination / "formal_source"
        source.mkdir(parents=True)
        existing_root = destination
    tree_before = _relative_tree(existing_root)

    with pytest.raises(ValueError, match="paths must be disjoint"):
        run_posterior_cardinal_classification(
            object(),
            destination,
            source_run=source,
            smoke=True,
        )

    assert _relative_tree(existing_root) == tree_before
    assert not (destination / "analyses").exists()
    assert not (destination / "effects").exists()


def _inference_with_prior_only_feature_signal() -> dict[str, pd.DataFrame]:
    bootstrap_rows: list[dict[str, Any]] = [
        {
            "width": 0.20,
            "contrast_id": "context_minus_prior",
            "metric": "delta_soft_log_loss",
            "observed_mean": -0.01,
            "ci95_high": -0.001,
        }
    ]
    repeat_rows: list[dict[str, Any]] = []
    p_value_rows: list[dict[str, Any]] = []
    selected_panel = FORMAL_PANELS[0]

    for panel in (*FORMAL_PANELS, *SENSITIVITY_PANELS):
        token = panel.removeprefix("context_plus_")
        prior_ci_high = -0.01 if panel == selected_panel else 0.02
        bootstrap_rows.append(
            {
                "width": 0.20,
                "contrast_id": f"{token}_minus_prior",
                "metric": "delta_soft_log_loss",
                "observed_mean": -0.02 if panel == selected_panel else 0.01,
                "ci95_high": prior_ci_high,
            }
        )
        for width in (0.15, 0.20, 0.25):
            bootstrap_rows.append(
                {
                    "width": width,
                    "contrast_id": f"{token}_minus_context",
                    "metric": "delta_soft_log_loss",
                    "observed_mean": -0.001,
                    # The incremental confidence gate fails for every feature.
                    "ci95_high": 0.01,
                }
            )
        bootstrap_rows.append(
            {
                "width": 0.20,
                "contrast_id": f"{token}_minus_context",
                "metric": "delta_multiclass_brier",
                "observed_mean": -0.001,
                "ci95_high": 0.01,
            }
        )
        repeat_rows.extend(
            {
                "width": 0.20,
                "contrast_id": f"{token}_minus_context",
                "repeat": repeat,
                "delta_soft_log_loss": -0.001,
            }
            for repeat in range(10)
        )
        if panel in FORMAL_PANELS:
            p_value_rows.append(
                {
                    "panel": panel,
                    "p_value_lower_tail": 0.50,
                    "holm_adjusted_p": 1.0,
                }
            )

    return {
        "bootstrap_summary": pd.DataFrame(bootstrap_rows),
        "repeat_deltas": pd.DataFrame(repeat_rows),
        "signflip_summary": pd.DataFrame(p_value_rows),
    }


def test_feature_interpretation_stops_when_prior_gate_alone_passes() -> None:
    decisions, stopping = build_release_decisions(
        _inference_with_prior_only_feature_signal(),
        smoke=False,
    )
    formal = decisions.loc[
        decisions["role"].eq("formal_single_feature_increment")
    ]

    assert formal["augmented_vs_prior_pass"].astype(bool).any()
    assert not formal["primary_release_gate_pass"].astype(bool).any()
    assert formal["width_label"].eq(
        "not_applicable_failed_primary_gate"
    ).all()
    assert stopping["any_formal_augmented_panel_beats_prior"] is True
    assert stopping["baseline_information_stop"] is False
    assert stopping["feature_claim_release_stopped"] is True
    assert stopping["feature_interpretation_stopped"] is True
    assert (
        stopping["decision"]
        == "no_formal_feature_passed_all_incremental_gates"
    )


def test_frozen_real_input_end_to_end_smoke(tmp_path: Path) -> None:
    config = load_config()
    source = config.resolve_path("active_source_run")
    destination = tmp_path / "posterior_cardinal_smoke"

    result = run_posterior_cardinal_classification(
        config,
        destination,
        source_run=source,
        smoke=True,
        n_jobs=1,
    )

    qa = json.loads(
        (destination / "analyses" / "analysis_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "complete_smoke"
    assert qa["status"] == "complete_smoke"
    assert qa["n_records"] == 97
    assert qa["n_connected_groups"] == 80
    assert qa["n_posterior_draws"] == 776_000
    assert qa["draws_per_record"] == 8_000
    assert qa["n_widths_fully_refit"] == 3
    assert qa["outer_repeats"] == 1
    assert qa["outer_folds"] == 2
    assert qa["inner_folds"] == 2
    assert qa["outer_manifest_shared_across_widths_panels_models"] is True
    assert qa["inner_manifest_shared_across_widths_panels_models"] is True
    assert qa["input_snapshot_verified_after_modeling"] is True
    assert qa["reduced_registry_verified_after_modeling"] is True
    assert qa["formal_source_run_unchanged"] is True

    draw_rows = sum(
        len(chunk)
        for chunk in pd.read_csv(
            destination / "inputs" / "posterior_slope_draws.csv.gz",
            usecols=["sample_record"],
            chunksize=100_000,
        )
    )
    assert draw_rows == 776_000

    outer = pd.read_csv(destination / "folds" / "outer_split_manifest.csv")
    assert len(outer) == 97
    assert outer["record_id"].nunique() == 97
    assert outer["cv_group_id"].nunique() == 80
    assert outer["manifest_id"].unique().tolist() == [
        "primary_width_20_shared_v2"
    ]
    assert set(outer["fold"]) == {0, 1}
    assert outer.groupby("record_id").size().eq(1).all()
    assert outer.groupby("cv_group_id")["fold"].nunique().eq(1).all()

    inner = pd.read_csv(destination / "folds" / "inner_split_manifest.csv")
    assert inner["manifest_id"].unique().tolist() == [
        "primary_width_20_shared_v2"
    ]
    assert set(inner["inner_fold"]) == {0, 1}
    assert (
        inner.groupby(
            ["outer_repeat", "outer_fold", "cv_group_id"]
        )["inner_fold"]
        .nunique()
        .eq(1)
        .all()
    )

    for width_id in ("width_15", "width_20", "width_25"):
        labels = pd.read_csv(
            destination
            / "registry"
            / width_id
            / "hierarchical_soft_labels.csv"
        )
        assert len(labels) == 97
        assert labels["record_id"].nunique() == 97

    oof = pd.read_csv(destination / "analyses" / "oof_probabilities.csv")
    assert set(np.round(oof["width"], 2)) == {0.15, 0.20, 0.25}
    assert len(oof) == 3 * 97 * (1 + 2 * len(ALL_PANELS))
    assert set(oof["model_family"]) == {
        "fold_local_soft_prior",
        PRIMARY_MODEL,
        SECONDARY_MODEL,
    }
    coverage = oof.groupby(
        ["width", "panel", "model_family", "record_id"]
    ).size()
    assert coverage.eq(1).all()
    assert (
        oof.groupby(["width", "panel", "model_family"])["record_id"]
        .nunique()
        .eq(97)
        .all()
    )
    probability_columns = [
        "p_low_30_40_like",
        "p_60_like",
        "p_120_like",
        "p_off_cardinal",
    ]
    assert np.allclose(oof.loc[:, probability_columns].sum(axis=1), 1.0)

    aggregated = pd.read_csv(
        destination / "analyses" / "oof_aggregated.csv"
    )
    assert len(aggregated) == len(oof)
    assert aggregated["n_repeated_oof"].eq(1).all()

    decisions = pd.read_csv(
        destination / "effects" / "feature_release_decisions.csv"
    )
    assert len(decisions) == len(FORMAL_PANELS) + len(SENSITIVITY_PANELS)
    assert not decisions["primary_release_gate_pass"].astype(bool).any()
    curves = pd.read_csv(
        destination / "effects" / "adjusted_probability_curves.csv"
    )
    assert curves.empty

    required_artifacts = (
        "registry/posterior_cardinal_all97_v4.yaml",
        "analyses/aggregate_metrics.csv",
        "analyses/inference/paired_connected_group_bootstrap_summary.csv",
        "analyses/inference/one_sided_holm_summary.csv",
        "effects/stopping_decision.json",
        "report/posterior_cardinal_report.md",
        "report/validation_report.json",
        "provenance/posterior_cardinal_contract_binding.json",
        "provenance/input_snapshot_verification_after_modeling.json",
        "provenance/source_run_verification.json",
    )
    assert all((destination / relative).is_file() for relative in required_artifacts)
