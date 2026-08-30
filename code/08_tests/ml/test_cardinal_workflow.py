import json
from pathlib import Path
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.cardinal_workflow import (
    _low_anchor_subdivision_gate,
    _validate_frozen_cardinal_settings,
    run_cardinal_classification,
)
from ink_tafel_study.config import load_config


def test_cardinal_workflow_smoke_is_self_contained_and_nonmechanistic(
    tmp_path: Path,
) -> None:
    config = load_config()
    source_run = config.resolve_path("active_source_run")
    destination = tmp_path / "cardinal_smoke"

    result = run_cardinal_classification(
        config,
        destination,
        source_run=source_run,
        smoke=True,
        n_jobs=1,
    )

    qa = json.loads(
        (destination / "cardinal_classification_qa.json").read_text(
            encoding="utf-8"
        )
    )
    decision = json.loads(
        (destination / "cardinal_classification_decision.json").read_text(
            encoding="utf-8"
        )
    )
    labels = pd.read_csv(
        destination / "registry" / "posterior_soft_labels.csv"
    )
    hierarchical_labels = pd.read_csv(
        destination / "registry" / "hierarchical_soft_labels.csv"
    )
    final_spec = json.loads(
        (destination / "final_pipeline" / "exact_final_pipeline.json").read_text(
            encoding="utf-8"
        )
    )
    low_gate = json.loads(
        (
            destination / "registry" / "low_anchor_subdivision_gate.json"
        ).read_text(encoding="utf-8")
    )

    assert result["status"] == "complete_smoke"
    assert qa["n_records_all"] == 97
    assert qa["n_records_primary_alkaline"] == 90
    assert qa["n_posterior_draws"] == 776_000
    assert qa["n_nonpositive_draws"] == 1_760
    assert qa["source_run_unchanged"] is True
    assert qa["input_snapshot_unchanged_during_modeling"] is True
    assert qa["continuous_regression_outputs_modified"] is False
    assert qa["mechanistic_claim_allowed"] is False
    assert qa["n_panel_sensitivity_contrasts"] == 5
    assert qa["primary_and_panel_sensitivity_outer_assignments_identical"] is True
    assert decision["continuous_regression_replaced"] is False
    assert decision["mechanistic_claim_allowed"] is False
    assert labels["n_draws"].eq(8_000).all()
    assert {
        "q_30_like",
        "q_40_like",
        "q_60_like",
        "q_120_like",
        "q_off_cardinal",
    }.issubset(labels.columns)
    assert {
        "q_low_30_40_like",
        "q_60_like",
        "q_120_like",
        "q_off_cardinal",
    }.issubset(hierarchical_labels.columns)
    np.testing.assert_allclose(
        hierarchical_labels["q_low_30_40_like"],
        hierarchical_labels["q_30_like"] + hierarchical_labels["q_40_like"],
    )
    assert (
        destination / "inputs" / "posterior_slope_draws.csv.gz"
    ).is_file()
    assert (
        destination / "final_pipeline" / "hierarchical_cardinal_pipeline.joblib"
    ).is_file()
    assert not (
        destination / "final_pipeline" / "two_stage_cardinal_pipeline.joblib"
    ).exists()
    assert (
        destination / "report" / "cardinal_classification_report.md"
    ).is_file()
    assert low_gate["status"] == "not_fitted_insufficient_30_support"
    assert low_gate["training_attempted"] is False
    assert low_gate["affects_primary_selection"] is False
    assert decision["final_pipeline"] == final_spec
    assert final_spec["relative_half_width"] == 0.20
    assert final_spec["supervised_class_order"] == [
        "low_30_40_like",
        "60_like",
        "120_like",
        "off_cardinal",
    ]
    assert final_spec["training_scope"] == "alkaline"
    assert final_spec["n_training_records"] == 90
    assert (
        final_spec["low_anchor_subdivision"]["status"]
        == "not_fitted_insufficient_30_support"
    )

    fitted_pipeline = joblib.load(
        destination / "final_pipeline" / final_spec["model_path"]
    )
    records = pd.read_csv(destination / "inputs" / "records.csv")
    probabilities = fitted_pipeline.predict_proba(
        records.loc[:2, final_spec["feature_columns"]]
    )
    assert probabilities.shape == (3, 4)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert list(fitted_pipeline.classes_) == final_spec["supervised_class_order"]

    registry_filenames = {
        "cardinal_label_definition.json",
        "posterior_soft_labels.csv",
        "hierarchical_soft_labels.csv",
        "class_support.csv",
        "hierarchical_class_support.csv",
        "low_anchor_subdivision.csv",
        "low_anchor_subdivision_gate.json",
    }
    for width_tag in ("width_15", "width_20", "width_25"):
        width_registry = destination / "registry" / "label_widths" / width_tag
        assert registry_filenames.issubset(
            {path.name for path in width_registry.iterdir() if path.is_file()}
        )

    for width_tag in ("width_15", "width_25"):
        for analysis_name in ("primary_alkaline", "population_sensitivity_all"):
            analysis_dir = (
                destination
                / "analyses"
                / "label_width_sensitivity"
                / width_tag
                / analysis_name
            )
            for filename in (
                "analysis_qa.json",
                "split_manifest.csv",
                "inner_split_manifest.csv",
                "oof_aggregated.csv",
                "aggregate_metrics.csv",
                "paired_group_bootstrap_summary.csv",
            ):
                assert (analysis_dir / filename).is_file()

    width_summary = pd.read_csv(
        destination / "label_width_sensitivity_summary.csv"
    )
    assert set(width_summary["relative_half_width"]) == {0.15, 0.20, 0.25}
    assert set(width_summary["scheme_role"]) == {"primary", "sensitivity"}
    assert width_summary["absolute_log_loss_comparable_across_widths"].eq(
        False
    ).all()
    assert width_summary.loc[
        width_summary["relative_half_width"].ne(0.20),
        "can_override_primary_width_decision",
    ].eq(False).all()

    candidate_catalog = pd.read_csv(
        destination / "analyses" / "primary_alkaline" / "candidate_catalog.csv"
    )
    assert "soft_ordinal_logistic" in set(candidate_catalog["model_family"])
    assert qa["fully_trained_relative_half_widths"] == [0.15, 0.20, 0.25]
    assert qa["label_width_outer_and_inner_assignments_identical"] is True
    assert qa["absolute_log_loss_comparable_across_widths"] is False

    primary_manifest = pd.read_csv(
        destination / "analyses" / "primary_alkaline" / "split_manifest.csv"
    )
    comparison_columns = [
        "repeat",
        "fold",
        "record_id",
        "cv_group_id",
        "split_seed",
    ]
    for width_tag in ("width_15", "width_25"):
        sensitivity_manifest = pd.read_csv(
            destination
            / "analyses"
            / "label_width_sensitivity"
            / width_tag
            / "primary_alkaline"
            / "split_manifest.csv"
        )
        pd.testing.assert_frame_equal(
            primary_manifest.loc[:, comparison_columns].sort_values(
                comparison_columns, kind="stable"
            ).reset_index(drop=True),
            sensitivity_manifest.loc[:, comparison_columns].sort_values(
                comparison_columns, kind="stable"
            ).reset_index(drop=True),
        )
    contrasts = pd.read_csv(
        destination
        / "analyses"
        / "panel_sensitivity_alkaline"
        / "paired_contrast_registry.csv"
    )
    assert set(contrasts["contrast_role"]) == {
        "process_increment",
        "property_increment",
        "property_representation",
    }


def test_frozen_label_policy_drift_fails_before_execution() -> None:
    settings = deepcopy(load_config().section("cardinal_classification"))
    settings["label_scheme"]["overlap_policy"] = "first_band"
    with pytest.raises(ValueError, match="overlap_policy"):
        _validate_frozen_cardinal_settings(settings)

    settings = deepcopy(load_config().section("cardinal_classification"))
    settings["panel_sensitivity_contrasts"][0]["augmented"] = "context"
    with pytest.raises(ValueError, match="panel_sensitivity_contrasts"):
        _validate_frozen_cardinal_settings(settings)

    settings = deepcopy(load_config().section("cardinal_classification"))
    settings["label_scheme"]["training_sensitivity_half_widths"] = [
        0.15,
        0.25,
    ]
    with pytest.raises(
        ValueError, match="training_sensitivity_half_widths"
    ):
        _validate_frozen_cardinal_settings(settings)

    settings = deepcopy(load_config().section("cardinal_classification"))
    settings["label_scheme"]["supervised_class_order"] = [
        "60_like",
        "low_30_40_like",
        "120_like",
        "off_cardinal",
    ]
    with pytest.raises(ValueError, match="supervised_class_order"):
        _validate_frozen_cardinal_settings(settings)

    settings = deepcopy(load_config().section("cardinal_classification"))
    settings["models"]["enabled"].remove("soft_ordinal_logistic")
    with pytest.raises(ValueError, match="soft_ordinal_logistic_enabled"):
        _validate_frozen_cardinal_settings(settings)


def test_low_anchor_subdivision_gate_is_descriptive_and_fails_closed() -> None:
    labels = pd.DataFrame(
        {
            "record_id": [f"r{index}" for index in range(6)],
            "cv_group_id": [f"g{index}" for index in range(6)],
            "electrolyte_scope": ["alkaline"] * 6,
            "q_30_like": [0.10] * 6,
            "q_40_like": [0.60] * 6,
        }
    )
    gate = _low_anchor_subdivision_gate(
        labels,
        load_config().section("cardinal_classification"),
        relative_half_width=0.20,
    )

    assert gate["status"] == "not_fitted_insufficient_30_support"
    assert gate["support_gate_passed"] is False
    assert gate["training_attempted"] is False
    assert gate["affects_primary_selection"] is False
    assert gate["assessment_population"] == "alkaline"
