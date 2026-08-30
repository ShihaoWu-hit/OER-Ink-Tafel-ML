from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ink_tafel_study.posterior_cardinal_contiguous_threeclass_shrunk_calibrated_gower as stage
from ink_tafel_study.posterior_cardinal_contiguous_threeclass import (
    P_COLUMNS,
    Q_COLUMNS,
)
from ink_tafel_study.posterior_cardinal_contiguous_threeclass_direct_gower import (
    NEIGHBOR_COUNTS,
)
from ink_tafel_study.provenance import (
    sha256_file,
    verify_artifact_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    REPOSITORY_ROOT
    / "RESULTS"
    / "05_WORKBOOKS"
    / "posterior_cardinal_contiguous_threeclass_direct_gower_v1_20260725"
)


def _synthetic_component_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = 12
    base = np.asarray(
        [
            [0.72, 0.20, 0.08],
            [0.58, 0.29, 0.13],
            [0.48, 0.34, 0.18],
            [0.63, 0.24, 0.13],
        ],
        dtype=float,
    )
    base = np.tile(base, (3, 1))
    components = []
    shifts = np.linspace(-0.035, 0.035, len(NEIGHBOR_COUNTS))
    for shift in shifts:
        candidate = base.copy()
        candidate[:, 0] += shift
        candidate[:, 1] -= 0.6 * shift
        candidate[:, 2] -= 0.4 * shift
        components.append(candidate)
    cube = np.stack(components, axis=1)
    true_weights = np.asarray([0.12, 0.14, 0.18, 0.22, 0.20, 0.14])
    mixture = stage.mix_component_probabilities(cube, true_weights)
    targets = stage.apply_identity_shrunk_calibration(
        mixture,
        np.asarray([0.92, -0.015, 0.01]),
    )
    folds = np.repeat(np.arange(4), 3)
    return cube, targets, folds


def test_contract_is_hash_bound_and_freezes_mean_loss_shrinkage() -> None:
    contract, path, digest = stage.load_shrunk_calibrated_gower_contract()

    assert contract["contract_id"] == stage.CONTRACT_ID
    assert digest == stage.CONTRACT_SHA256
    assert sha256_file(path) == stage.CONTRACT_SHA256
    assert contract["inner_validation"]["loss_aggregation"] == (
        "fold_equal_weighted_mean"
    )
    assert contract["model"]["multiscale_weights"]["shrinkage_lambda"] == (
        stage.WEIGHT_SHRINKAGE
    )
    assert contract["model"]["calibration"]["shrinkage_lambda"] == (
        stage.CALIBRATION_SHRINKAGE
    )
    assert contract["development_boundary"]["calibration_source_boundary"][
        "exact_guo_et_al_neural_network_method_replication"
    ] is False


def test_parent_refit_code_guard_matches_immutable_manifest() -> None:
    audit = stage._verify_parent_refit_code(PARENT)

    assert audit["status"] == "pass"
    assert audit["all_parent_manifest_files_match_current_source"] is True
    assert audit["recorded_tree_sha256"] == audit["current_tree_sha256"]
    assert set(stage.PARENT_REFIT_REQUIRED_PATHS).issubset(
        {row["path"] for row in audit["files"]}
    )
    assert audit["n_manifest_files_checked"] == len(audit["files"])


def test_parent_refit_code_guard_fails_closed_on_current_hash_mismatch(
    monkeypatch,
) -> None:
    original = stage.sha256_file
    direct_path = (
        "posterior_cardinal_contiguous_threeclass_direct_gower.py"
    )

    def mismatched(path):
        if Path(path).name == direct_path:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(stage, "sha256_file", mismatched)
    with pytest.raises(RuntimeError, match="does not match the immutable"):
        stage._verify_parent_refit_code(PARENT)


def test_parent_environment_guard_matches_critical_runtime() -> None:
    audit = stage._verify_parent_environment(PARENT)

    assert audit["status"] == "pass"
    assert audit["python_matches"] is True
    assert audit["all_critical_packages_match"] is True
    assert {row["package"] for row in audit["critical_packages"]} == set(
        stage.CRITICAL_ENVIRONMENT_PACKAGES
    )
    assert all(row["matches"] for row in audit["critical_packages"])


def test_parent_environment_guard_fails_closed_on_package_drift(
    monkeypatch,
) -> None:
    original = stage.environment_manifest

    def drifted(repository_root):
        payload = original(repository_root)
        payload["packages"] = dict(payload["packages"])
        payload["packages"]["numpy"] = "0.0.0-test-drift"
        return payload

    monkeypatch.setattr(stage, "environment_manifest", drifted)
    with pytest.raises(RuntimeError, match="critical package"):
        stage._verify_parent_environment(PARENT)


def test_fold_equal_weights_give_each_fold_identical_total_mass() -> None:
    folds = np.asarray([0, 0, 1, 1, 1, 2, 2, 2, 2])
    weights = stage.fold_equal_weights(folds)

    assert np.isclose(weights.sum(), 1.0)
    totals = [weights[folds == fold].sum() for fold in np.unique(folds)]
    assert np.allclose(totals, np.repeat(1.0 / 3.0, 3))


def test_shrunk_weights_and_calibration_are_simplex_and_identity_centered() -> None:
    cube, targets, folds = _synthetic_component_cube()
    sample_weight = stage.fold_equal_weights(folds)
    equal = np.repeat(1.0 / len(NEIGHBOR_COUNTS), len(NEIGHBOR_COUNTS))
    equal_prediction = stage.mix_component_probabilities(cube, equal)

    weights, weight_audit = stage.fit_shrunk_multiscale_weights(
        cube,
        targets,
        sample_weight,
    )
    weighted = stage.mix_component_probabilities(cube, weights)
    calibration, calibration_audit = stage.fit_identity_shrunk_calibrator(
        weighted,
        targets,
        sample_weight,
    )
    calibrated = stage.apply_identity_shrunk_calibration(
        weighted,
        calibration,
    )

    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.allclose(
        stage.apply_identity_shrunk_calibration(
            equal_prediction,
            np.asarray([1.0, 0.0, 0.0]),
        ),
        equal_prediction,
    )
    assert weight_audit["shrinkage_lambda"] == stage.WEIGHT_SHRINKAGE
    assert calibration_audit["shrinkage_lambda"] == (
        stage.CALIBRATION_SHRINKAGE
    )
    assert weight_audit["selected_raw_soft_log_loss"] <= (
        weight_audit["equal_weight_raw_soft_log_loss"] + 1.0e-10
    )
    assert calibration_audit["selected_raw_soft_log_loss"] <= (
        calibration_audit["identity_raw_soft_log_loss"] + 1.0e-10
    )
    assert abs(sum(calibration_audit["class_intercepts"].values())) < 1.0e-12


def test_parent_outer_component_helper_ignores_stored_target_columns() -> None:
    components = pd.read_csv(
        PARENT / "analyses" / "gower_component_oof_probabilities.csv"
    )
    components["record_id"] = components["record_id"].astype(str)
    selected = components.loc[
        components["repeat"].eq(0) & components["fold"].eq(0)
    ].copy()
    test_ids = selected["record_id"].drop_duplicates().astype(str).tolist()
    expected = stage._outer_component_cube(
        selected,
        repeat=0,
        fold=0,
        test_ids=test_ids,
    )
    selected.loc[:, list(Q_COLUMNS)] = np.nan
    actual = stage._outer_component_cube(
        selected,
        repeat=0,
        fold=0,
        test_ids=test_ids,
    )

    assert np.array_equal(actual, expected)


def test_optimizer_failures_fall_back_to_equal_weights_and_identity(
    monkeypatch,
) -> None:
    cube, targets, folds = _synthetic_component_cube()
    sample_weight = stage.fold_equal_weights(folds)
    failed = SimpleNamespace(
        success=False,
        x=np.repeat(1.0 / len(NEIGHBOR_COUNTS), len(NEIGHBOR_COUNTS)),
        message="synthetic failure",
        nit=0,
    )
    monkeypatch.setattr(stage, "minimize", lambda *args, **kwargs: failed)

    weights, weight_audit = stage.fit_shrunk_multiscale_weights(
        cube,
        targets,
        sample_weight,
    )
    assert np.array_equal(
        weights,
        np.repeat(1.0 / len(NEIGHBOR_COUNTS), len(NEIGHBOR_COUNTS)),
    )
    assert weight_audit["fallback_used"] is True

    failed.x = np.asarray([1.0, 0.0, 0.0])
    mixture = stage.mix_component_probabilities(cube, weights)
    calibration, calibration_audit = stage.fit_identity_shrunk_calibrator(
        mixture,
        targets,
        sample_weight,
    )
    assert np.array_equal(calibration, np.asarray([1.0, 0.0, 0.0]))
    assert calibration_audit["fallback_used"] is True


def test_smoke_is_disjoint_audited_and_exactly_reconstructable(
    tmp_path: Path,
) -> None:
    parent_before = verify_artifact_manifest(PARENT)
    destination = tmp_path / "shrunk_calibrated_gower_smoke"

    result = stage.run_shrunk_calibrated_gower_sensitivity(
        destination,
        parent_run=PARENT,
        n_jobs=2,
        smoke=True,
    )

    qa = json.loads(
        (destination / "analyses" / "analysis_qa.json").read_text(
            encoding="utf-8"
        )
    )
    oof = pd.read_csv(destination / "analyses" / "oof_probabilities.csv")
    inner = pd.read_csv(
        destination / "analyses" / "inner_component_oof_probabilities.csv"
    )
    parameters = pd.read_csv(
        destination / "analyses" / "outer_split_meta_parameters.csv"
    )
    parent_components = pd.read_csv(
        destination
        / "inputs"
        / "parent_gower_component_oof_probabilities.csv"
    )
    refit_binding = json.loads(
        (
            destination
            / "provenance"
            / "parent_refit_code_binding.json"
        ).read_text(encoding="utf-8")
    )
    environment_binding = json.loads(
        (
            destination
            / "provenance"
            / "parent_environment_binding.json"
        ).read_text(encoding="utf-8")
    )
    for frame in (oof, inner, parent_components):
        frame["record_id"] = frame["record_id"].astype(str)

    assert result["status"] == "complete_smoke"
    assert qa["status"] == "complete_smoke"
    assert qa["scientifically_releasable"] is False
    assert qa["outer_test_used_for_weight_calibration_or_selection"] is False
    assert qa["outer_parent_component_target_columns_passed_to_fit"] is False
    assert qa["parent_outer_components_reused_without_refit"] is True
    assert qa["parent_tree_unchanged"] is True
    assert qa["parent_refit_code_verified_before_inner_refits"] is True
    assert qa["parent_refit_code_verified_after_modeling"] is True
    assert qa["parent_refit_code_tree_unchanged_during_run"] is True
    assert refit_binding["code_tree_unchanged_during_run"] is True
    assert (
        refit_binding["verified_before_inner_refits"]["current_tree_sha256"]
        == refit_binding["verified_after_modeling"]["current_tree_sha256"]
    )
    assert (
        destination / "inputs" / "parent_code_manifest.json"
    ).is_file()
    assert qa["parent_environment_verified_before_inner_refits"] is True
    assert qa["parent_environment_verified_after_modeling"] is True
    assert qa["parent_python_matches_current"] is True
    assert qa["parent_critical_packages_match_current"] is True
    assert qa["parent_critical_environment_unchanged_during_run"] is True
    assert environment_binding[
        "critical_environment_unchanged_during_run"
    ] is True
    assert (
        destination / "inputs" / "parent_environment_manifest.json"
    ).is_file()
    assert qa["n_outer_splits_evaluated"] == 2
    assert len(parameters) == 2
    assert set(oof["evaluation_model"]) == {
        stage.MODEL,
        "direct_multiscale_soft_gower",
        "fold_local_contiguous_threeclass_soft_prior",
    }
    assert inner.groupby(["repeat", "fold", "record_id"]).size().eq(
        len(NEIGHBOR_COUNTS)
    ).all()
    assert not parameters["outer_test_used_for_fit"].astype(bool).any()

    first = parameters.sort_values(["repeat", "fold"]).iloc[0]
    repeat = int(first["repeat"])
    fold = int(first["fold"])
    candidate = oof.loc[
        oof["evaluation_model"].eq(stage.MODEL)
        & oof["repeat"].eq(repeat)
        & oof["fold"].eq(fold)
    ].copy()
    test_ids = candidate["record_id"].astype(str).tolist()
    cube = stage._outer_component_cube(
        parent_components,
        repeat=repeat,
        fold=fold,
        test_ids=test_ids,
    )
    weights = np.asarray(
        [first[f"weight_k_{count}"] for count in NEIGHBOR_COUNTS],
        dtype=float,
    )
    calibration = np.asarray(
        [
            first["calibration_slope"],
            first["calibration_intercept_low_60_side"],
            first["calibration_intercept_120_side"],
        ],
        dtype=float,
    )
    reconstructed = stage.apply_identity_shrunk_calibration(
        stage.mix_component_probabilities(cube, weights),
        calibration,
    )

    assert np.allclose(
        reconstructed,
        candidate.loc[:, P_COLUMNS].to_numpy(float),
        atol=1.0e-12,
        rtol=0,
    )
    assert verify_artifact_manifest(destination)["status"] == "pass"
    parent_after = verify_artifact_manifest(PARENT)
    assert parent_before["actual_tree_sha256"] == (
        parent_after["actual_tree_sha256"]
    )
