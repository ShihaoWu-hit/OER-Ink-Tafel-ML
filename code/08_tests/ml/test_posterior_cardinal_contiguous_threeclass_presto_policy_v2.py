from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ink_tafel_study.posterior_cardinal_contiguous_threeclass_presto_policy_v2 as stage

from ink_tafel_study.posterior_cardinal_contiguous_threeclass_presto_policy_v2 import (
    CONTRACT_SHA256,
    POLICY_PENALTIES,
    PRIMARY_POLICY,
    VOTE5_POLICY,
    VOTE6_POLICY,
    _aggregate_candidate_policy,
    _select_policy_penalty_threshold,
    load_presto_policy_v2_contract,
    run_presto_policy_v2_sensitivity,
)
from ink_tafel_study.provenance import verify_artifact_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    REPOSITORY_ROOT
    / "RESULTS"
    / "05_WORKBOOKS"
    / "posterior_cardinal_contiguous_threeclass_presto_gate_v1_20260726"
)


def test_policy_v2_contract_is_hash_bound_and_globally_postselected() -> None:
    contract, path, digest = load_presto_policy_v2_contract()

    assert digest == CONTRACT_SHA256
    assert path.is_file()
    assert contract["policy_score_model"]["exact_presto_replication"] is False
    assert contract["development_boundary"][
        "f0_5_utility_and_mean_margin_rule_selected_after_inspecting_closed_outer_results"
    ] is True
    assert contract["inherited_probability_model"][
        "probability_refit_or_recalibration"
    ] == "forbidden"


def test_parent_presto_refit_code_and_environment_guards_match() -> None:
    code = stage._verify_parent_refit_code(PARENT)
    environment = stage._verify_parent_refit_environment(PARENT)

    assert code["status"] == "pass"
    assert code["recorded_tree_sha256"] == code["current_tree_sha256"]
    assert code["all_parent_manifest_files_match_current_source"] is True
    assert set(stage.PARENT_REFIT_REQUIRED_PATHS).issubset(
        {row["path"] for row in code["files"]}
    )
    assert environment["status"] == "pass"
    assert environment["python_matches"] is True
    assert environment["all_required_package_versions_match"] is True


def test_parent_presto_refit_code_guard_fails_closed(
    monkeypatch,
) -> None:
    original = stage.sha256_file

    def mismatched(path):
        if Path(path).name == "presto_ordinal.py":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(stage, "sha256_file", mismatched)
    with pytest.raises(RuntimeError, match="does not match the immutable"):
        stage._verify_parent_refit_code(PARENT)


def test_parent_presto_refit_environment_guard_fails_closed(
    monkeypatch,
) -> None:
    original = stage.environment_manifest

    def mismatched(repository_root):
        current = original(repository_root)
        current["packages"]["scipy"] = "0.0.invalid"
        return current

    monkeypatch.setattr(stage, "environment_manifest", mismatched)
    with pytest.raises(RuntimeError, match="environment does not match"):
        stage._verify_parent_refit_environment(PARENT)


def test_joint_soft_f05_selection_uses_only_inner_soft_rows() -> None:
    rows: list[dict[str, object]] = []
    q_high = [0.0, 0.0, 1.0, 1.0]
    predictions = {
        0.003: [0.01, 0.02, 0.80, 0.90],
        0.01: [0.10, 0.10, 0.20, 0.20],
        0.03: [0.90, 0.80, 0.02, 0.01],
    }
    for penalty in POLICY_PENALTIES:
        for index, target in enumerate(q_high):
            rows.append(
                {
                    "penalty": penalty,
                    "inner_fold": index // 2,
                    "record_id": str(index + 1),
                    "q_high_off": target,
                    "p_high_off": predictions[penalty][index],
                    "fit_failed": False,
                }
            )
    selected, grid = _select_policy_penalty_threshold(pd.DataFrame(rows))

    assert selected["penalty"] == 0.003
    assert selected["threshold"] == 0.75
    assert selected["soft_f0_5"] == 1.0
    assert selected["outer_hard_labels_used"] is False
    assert grid["selected"].sum() == 1
    assert grid.loc[grid["selected"], "passes_expected_fpr_cap"].all()


def test_mean_margin_and_vote_sensitivities_do_not_read_target_labels() -> None:
    rows: list[dict[str, object]] = []
    margins = {
        "a": [0.5] * 4 + [-0.1] * 6,
        "b": [0.01] * 6 + [-0.5] * 4,
    }
    for record_id, values in margins.items():
        for repeat, margin in enumerate(values):
            rows.append(
                {
                    "repeat": repeat,
                    "fold": 0,
                    "record_id": record_id,
                    "cv_group_id": f"g_{record_id}",
                    "q_low_60_side": 0.7,
                    "q_120_side": 0.2,
                    "q_high_off": 0.1,
                    "p_low_60_side": 0.7,
                    "p_120_side": 0.2,
                    "p_high_off": 0.1,
                    "selected_policy_penalty": 0.003,
                    "selected_high_threshold": 0.2,
                    "high_score": 0.2 + margin,
                    "raw_threshold_margin": margin,
                    "predicted_high": margin >= 0,
                }
            )
    frame = pd.DataFrame(rows)
    original = _aggregate_candidate_policy(frame)
    permuted = frame.copy()
    permuted[["q_low_60_side", "q_120_side", "q_high_off"]] = permuted[
        ["q_high_off", "q_low_60_side", "q_120_side"]
    ].to_numpy()
    changed_targets = _aggregate_candidate_policy(permuted)

    keys = ["decision_policy", "record_id"]
    original = original.sort_values(keys).reset_index(drop=True)
    changed_targets = changed_targets.sort_values(keys).reset_index(drop=True)
    assert original["predicted_class_index"].equals(
        changed_targets["predicted_class_index"]
    )
    lookup = original.set_index(["decision_policy", "record_id"])[
        "predicted_high"
    ]
    assert bool(lookup.loc[(PRIMARY_POLICY, "a")]) is True
    assert bool(lookup.loc[(VOTE5_POLICY, "a")]) is False
    assert bool(lookup.loc[(VOTE6_POLICY, "a")]) is False
    assert bool(lookup.loc[(PRIMARY_POLICY, "b")]) is False
    assert bool(lookup.loc[(VOTE5_POLICY, "b")]) is True
    assert bool(lookup.loc[(VOTE6_POLICY, "b")]) is True


def test_policy_v2_smoke_preserves_parent_probability_rows(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "presto-policy-v2-smoke"
    result = run_presto_policy_v2_sensitivity(
        destination,
        n_jobs=2,
        smoke=True,
    )

    assert result["status"] == "complete_smoke"
    assert result["maximum_parent_probability_numeric_error"] == 0.0
    assert verify_artifact_manifest(destination)["status"] == "pass"
    qa = json.loads(
        (destination / "analyses" / "analysis_qa.json").read_text()
    )
    assert qa["probability_refit_or_recalibration_performed"] is False
    assert qa["maximum_parent_probability_numeric_error"] == 0.0
    assert qa["outer_test_used_for_penalty_threshold_or_aggregation_selection"] is False
    assert qa["global_policy_rule_selected_after_closed_outer_results_seen"] is True
    assert qa["scientifically_releasable"] is False
    assert qa["parent_refit_code_verified_before_policy_refits"] is True
    assert qa["parent_refit_code_verified_after_modeling"] is True
    assert qa["parent_refit_code_tree_unchanged_during_run"] is True
    assert qa["parent_refit_environment_verified_before_policy_refits"] is True
    assert qa["parent_refit_environment_verified_after_modeling"] is True
    assert qa["parent_refit_environment_versions_unchanged_during_run"] is True
    assert (destination / "inputs" / "parent_code_manifest.json").is_file()
    assert (destination / "inputs" / "parent_environment_manifest.json").is_file()
    code_binding = json.loads(
        (
            destination
            / "provenance"
            / "parent_refit_code_binding.json"
        ).read_text()
    )
    assert code_binding["code_tree_unchanged_during_run"] is True

    expected = pd.read_csv(PARENT / "analyses" / "oof_probabilities.csv")
    expected = expected.loc[expected["repeat"].eq(0)].copy()
    actual = pd.read_csv(destination / "analyses" / "oof_probabilities.csv")
    key = ["evaluation_model", "repeat", "fold", "record_id"]
    merged = expected.merge(
        actual,
        on=key,
        suffixes=("_expected", "_actual"),
        validate="one_to_one",
    )
    assert len(merged) == len(expected) == len(actual)
    for column in ("p_low_60_side", "p_120_side", "p_high_off"):
        assert np.array_equal(
            merged[f"{column}_expected"].to_numpy(float),
            merged[f"{column}_actual"].to_numpy(float),
        )
    selections = pd.read_csv(destination / "analyses" / "outer_selections.csv")
    assert set(selections["selected_policy_penalty"]) <= set(POLICY_PENALTIES)
    assert not selections["outer_hard_labels_used"].astype(bool).any()
    metrics = pd.read_csv(
        destination / "analyses" / "policy_aggregate_metrics.csv"
    )
    assert {PRIMARY_POLICY, VOTE5_POLICY, VOTE6_POLICY}.issubset(
        set(metrics["decision_policy"])
    )
