"""Post-selected PRESTO policy-only refinement for contiguous classes.

This stage never refits or recalibrates the immutable PRESTO-v1 probability
model.  It reuses the parent's grouped inner-OOF PRESTO candidates to choose a
separate policy-only penalty and high-alert threshold, refits only that score
inside each outer-training split, and aggregates repeated decisions by the
mean raw score-minus-threshold margin.

The F0.5 utility and mean-margin rule were chosen after inspecting results on
the same closed 97-record dataset.  Nested grouped folds prevent row-level
outer-test leakage within this packaged run, but do not remove that global
method-selection bias.  Outputs are development evidence only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import yaml

from .posterior_cardinal_contiguous_threeclass import (
    CLASS_LABELS,
    P_COLUMNS,
    Q_COLUMNS,
    _metric_tables,
    aggregate_contiguous_oof,
)
from .posterior_cardinal_contiguous_threeclass_presto_gate import (
    PRESTO_HYBRID_MODEL,
    PRESTO_POLICY as PARENT_PRESTO_POLICY,
    _decision_metrics,
    _fit_presto,
    _fold_equal_weights,
    _fold_local_novelty_features,
    _indexed,
    _jsonable,
    _probability_argmax_metrics,
    _resolve_path,
)
from .provenance import (
    artifact_manifest,
    environment_manifest,
    inventory_files,
    sha256_file,
    tree_sha256,
    verify_artifact_manifest,
    write_json,
)
from .threeclass_sensitivity_models import probability_matrix_audit


CONTRACT_ID = "posterior_cardinal_contiguous_threeclass_presto_policy_v2"
CONTRACT_FILENAME = (
    "posterior_cardinal_contiguous_threeclass_presto_policy_v2.yaml"
)
CONTRACT_SHA256 = (
    "85694b3ff89b10458e56cf159fa8d2da2fa771948845469c8951a14a7ee93dcd"
)
STAGE_NAME = "posterior-cardinal-contiguous-threeclass-presto-policy-v2"

PRIMARY_POLICY = "presto_policy_only_soft_f05_mean_margin_v2"
VOTE5_POLICY = "presto_policy_only_soft_f05_vote5_sensitivity_v2"
VOTE6_POLICY = "presto_policy_only_soft_f05_vote6_sensitivity_v2"
PARENT_POLICY = "parent_v1_presto_soft_f2_majority_vote_policy"
POLICY_SCORE_MODEL = "presto_inspired_policy_only_high_score_v2"

POLICY_PENALTIES: tuple[float, ...] = (0.003, 0.01, 0.03)
THRESHOLD_GRID: tuple[float, ...] = (
    0.03,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
    0.225,
    0.25,
    0.275,
    0.30,
    0.35,
    0.40,
    0.50,
    0.75,
    1.00,
)
F_BETA = 0.5
EXPECTED_FPR_CAP = 0.20
ENTROPY_THRESHOLD = 0.90
BOOTSTRAP_REPETITIONS = 2000
SMOKE_BOOTSTRAP_REPETITIONS = 100
BOOTSTRAP_SEED = 20260726
TIE_TOLERANCE = 1.0e-12

PARENT_REFIT_REQUIRED_PATHS: tuple[str, ...] = (
    "code/Ink_Tafel_Study/config/contracts/"
    "posterior_cardinal_contiguous_threeclass_presto_gate_v1.yaml",
    "code/Ink_Tafel_Study/src/ink_tafel_study/cardinal_metrics.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/cardinal_models.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/cardinal_reduced_models.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/feature_association_features.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/features.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/models.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/"
    "posterior_cardinal_contiguous_threeclass.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/"
    "posterior_cardinal_contiguous_threeclass_direct_gower.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/"
    "posterior_cardinal_contiguous_threeclass_presto_gate.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/posterior_cardinal_inference.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/presto_ordinal.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/provenance.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/threeclass_sensitivity_models.py",
)
PARENT_REFIT_REQUIRED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "PyYAML",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "contracts" / CONTRACT_FILENAME
)


def load_presto_policy_v2_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> tuple[dict[str, Any], Path, str]:
    """Load and fail-closed validate the frozen v2 development contract."""

    contract_path = _resolve_path(path, base=REPOSITORY_ROOT)
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    digest = sha256_file(contract_path)
    if CONTRACT_SHA256.startswith("__"):
        raise RuntimeError("PRESTO policy-v2 contract hash is not frozen")
    if digest != CONTRACT_SHA256:
        raise ValueError(
            "PRESTO policy-v2 contract hash mismatch: "
            f"expected={CONTRACT_SHA256}, actual={digest}"
        )
    loaded = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("PRESTO policy-v2 contract must be a mapping")
    contract = dict(loaded)
    if (
        contract.get("contract_id") != CONTRACT_ID
        or contract.get("stage") != STAGE_NAME
        or contract.get("status")
        != "posthoc_global_postselection_development_frozen_before_packaged_v2_run"
    ):
        raise ValueError("PRESTO policy-v2 contract identity changed")

    scope = dict(contract.get("scientific_scope", {}))
    if (
        scope.get("population") != "all97"
        or int(scope.get("expected_records", -1)) != 97
        or int(scope.get("expected_connected_groups", -1)) != 80
        or scope.get("independent_confirmation") is not False
        or scope.get("scientifically_releasable") is not False
        or scope.get("may_replace_presto_gate_v1_result") is not False
    ):
        raise ValueError("PRESTO policy-v2 scientific scope changed")

    inherited = dict(contract.get("inherited_probability_model", {}))
    if (
        inherited.get("evaluation_model") != PRESTO_HYBRID_MODEL
        or inherited.get("probability_refit_or_recalibration") != "forbidden"
        or inherited.get("probability_values_must_match_parent_exactly")
        is not True
    ):
        raise ValueError("Inherited PRESTO-v1 probability policy changed")

    model = dict(contract.get("policy_score_model", {}))
    selection = dict(contract.get("inner_policy_selection", {}))
    aggregation = dict(contract.get("repeat_aggregation", {}))
    primary = dict(aggregation.get("primary", {}))
    if (
        model.get("exact_presto_replication") is not False
        or tuple(float(value) for value in model.get("policy_penalty_grid", ()))
        != POLICY_PENALTIES
        or tuple(
            float(value) for value in selection.get("threshold_grid", ())
        )
        != THRESHOLD_GRID
        or float(selection.get("expected_soft_fpr_cap", -1))
        != EXPECTED_FPR_CAP
        or float(selection.get("beta", -1)) != F_BETA
        or selection.get("joint_penalty_threshold_selection") is not True
        or primary.get("rule")
        != "high_if_mean_raw_score_minus_selected_threshold_is_nonnegative"
    ):
        raise ValueError("PRESTO policy-v2 selection policy changed")
    boundary = dict(contract.get("development_boundary", {}))
    if (
        boundary.get(
            "f0_5_utility_and_mean_margin_rule_selected_after_inspecting_closed_outer_results"
        )
        is not True
        or boundary.get("paired_bootstrap_is_conditional_on_selected_rule")
        is not True
    ):
        raise ValueError("Global post-selection boundary was removed")
    return contract, contract_path, digest


def _resolve_parent(
    contract: Mapping[str, Any], parent_run: str | Path | None
) -> Path:
    settings = dict(contract["immutable_parent"])
    value = parent_run if parent_run is not None else settings["default_run"]
    return _resolve_path(value, base=REPOSITORY_ROOT)


def _assert_fresh_disjoint_destination(destination: Path, parent: Path) -> None:
    output = destination.resolve()
    immutable = parent.resolve()
    if output == immutable or immutable in output.parents:
        raise ValueError("Policy-v2 output must be disjoint from its parent")
    if output in immutable.parents:
        raise ValueError("Policy-v2 output cannot contain its parent")
    if output.exists():
        raise FileExistsError(f"Policy-v2 output must be absent: {output}")


def _verify_parent(parent: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if not parent.is_dir():
        raise FileNotFoundError(f"Immutable PRESTO-v1 parent is missing: {parent}")
    settings = dict(contract["immutable_parent"])
    verification = verify_artifact_manifest(parent)
    if verification["status"] != "pass":
        raise RuntimeError(f"PRESTO-v1 parent manifest failed: {verification}")
    expected_tree = str(settings["expected_artifact_tree_sha256"])
    if verification["actual_tree_sha256"] != expected_tree:
        raise RuntimeError(
            "PRESTO-v1 parent tree changed: "
            f"expected={expected_tree}, actual={verification['actual_tree_sha256']}"
        )
    files: list[dict[str, Any]] = []
    for relative, expected in dict(settings["required_files"]).items():
        path = parent / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing parent file: {relative}")
        actual = sha256_file(path)
        if actual != str(expected):
            raise RuntimeError(
                f"Parent file changed: {relative}; expected={expected}, actual={actual}"
            )
        files.append({"path": str(relative), "sha256": actual})
    return {
        "status": "pass",
        "parent_run": str(parent),
        "artifact_manifest_verification": verification,
        "required_files": files,
    }


def _verify_parent_refit_code(parent: Path) -> dict[str, Any]:
    """Bind the current PRESTO refit call chain to immutable parent code.

    The parent artifact tree binds its saved ``code_manifest.json``.  This
    guard additionally verifies that every file named by that manifest still
    exists in the current repository with the recorded size and SHA256 before
    or after any policy-score refit can be accepted.
    """

    manifest_path = parent / "provenance" / "code_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Parent PRESTO refit code manifest is missing: {manifest_path}"
        )
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("files"), list):
        raise TypeError("Parent PRESTO refit code manifest is malformed")
    recorded_rows = loaded["files"]
    if not recorded_rows or not all(
        isinstance(row, dict)
        and {"path", "size_bytes", "sha256"}.issubset(row)
        for row in recorded_rows
    ):
        raise ValueError("Parent PRESTO refit code manifest rows are malformed")
    recorded_paths = [str(row["path"]) for row in recorded_rows]
    if len(recorded_paths) != len(set(recorded_paths)):
        raise ValueError("Parent PRESTO refit code manifest has duplicate paths")
    missing_required = sorted(
        set(PARENT_REFIT_REQUIRED_PATHS) - set(recorded_paths)
    )
    if missing_required:
        raise RuntimeError(
            "Parent PRESTO refit code manifest omits required dependencies: "
            f"{missing_required}"
        )
    recorded_tree = str(loaded.get("tree_sha256", ""))
    recomputed_recorded_tree = tree_sha256(recorded_rows)
    if recorded_tree != recomputed_recorded_tree:
        raise RuntimeError(
            "Parent PRESTO refit code manifest tree is inconsistent: "
            f"recorded={recorded_tree}, recomputed={recomputed_recorded_tree}"
        )

    current_rows: list[dict[str, Any]] = []
    for row in recorded_rows:
        relative = str(row["path"])
        current_path = (REPOSITORY_ROOT / relative).resolve()
        try:
            current_path.relative_to(REPOSITORY_ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"Parent PRESTO refit path escapes repository root: {relative}"
            ) from error
        if not current_path.is_file():
            raise FileNotFoundError(
                f"Current PRESTO refit dependency is missing: {relative}"
            )
        current_hash = sha256_file(current_path)
        current_size = int(current_path.stat().st_size)
        if (
            current_hash != str(row["sha256"])
            or current_size != int(row["size_bytes"])
        ):
            raise RuntimeError(
                "Current PRESTO refit code does not match the immutable "
                f"parent manifest: {relative}, "
                f"expected_sha256={row['sha256']}, actual_sha256={current_hash}, "
                f"expected_size={row['size_bytes']}, actual_size={current_size}"
            )
        current_rows.append(
            {
                "path": relative,
                "size_bytes": current_size,
                "sha256": current_hash,
                "required_for_presto_policy_refit": (
                    relative in PARENT_REFIT_REQUIRED_PATHS
                ),
            }
        )
    current_tree_rows = [
        {
            "path": row["path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in current_rows
    ]
    current_tree = tree_sha256(current_tree_rows)
    if current_tree != recorded_tree:
        raise RuntimeError(
            "Current PRESTO refit code tree does not match the immutable "
            f"parent: expected={recorded_tree}, actual={current_tree}"
        )
    return {
        "status": "pass",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "recorded_tree_sha256": recorded_tree,
        "current_tree_sha256": current_tree,
        "all_parent_manifest_files_match_current_source": True,
        "required_refit_paths": list(PARENT_REFIT_REQUIRED_PATHS),
        "n_manifest_files_checked": int(len(current_rows)),
        "files": current_rows,
    }


def _verify_parent_refit_environment(parent: Path) -> dict[str, Any]:
    """Fail closed on Python or core numerical-package version drift."""

    manifest_path = parent / "provenance" / "environment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Parent PRESTO environment manifest is missing: {manifest_path}"
        )
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("packages"), dict):
        raise TypeError("Parent PRESTO environment manifest is malformed")
    parent_python = str(loaded.get("python", ""))
    if not parent_python:
        raise ValueError("Parent PRESTO environment lacks Python version")
    parent_packages = dict(loaded["packages"])
    missing = sorted(set(PARENT_REFIT_REQUIRED_PACKAGES) - set(parent_packages))
    if missing:
        raise RuntimeError(
            f"Parent PRESTO environment omits required packages: {missing}"
        )
    current = environment_manifest(REPOSITORY_ROOT)
    current_python = str(current.get("python", ""))
    current_packages = dict(current.get("packages", {}))
    comparisons: list[dict[str, Any]] = []
    for package in PARENT_REFIT_REQUIRED_PACKAGES:
        parent_version = parent_packages.get(package)
        current_version = current_packages.get(package)
        matches = parent_version == current_version and parent_version is not None
        comparisons.append(
            {
                "package": package,
                "parent_version": parent_version,
                "current_version": current_version,
                "matches": matches,
            }
        )
        if not matches:
            raise RuntimeError(
                "Current PRESTO refit environment does not match the immutable "
                f"parent: package={package}, parent={parent_version}, "
                f"current={current_version}"
            )
    if current_python != parent_python:
        raise RuntimeError(
            "Current PRESTO refit Python does not match the immutable parent: "
            f"parent={parent_python!r}, current={current_python!r}"
        )
    return {
        "status": "pass",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "python_matches": True,
        "parent_python": parent_python,
        "current_python": current_python,
        "required_packages": list(PARENT_REFIT_REQUIRED_PACKAGES),
        "all_required_package_versions_match": True,
        "package_comparisons": comparisons,
        "parent_platform": loaded.get("platform"),
        "current_platform": current.get("platform"),
    }


def _load_parent_tables(
    parent: Path,
) -> dict[str, pd.DataFrame]:
    paths = {
        "records": parent / "inputs" / "records.csv",
        "labels": parent / "registry" / "contiguous_three_class_soft_labels.csv",
        "outer": parent / "folds" / "outer_split_manifest.csv",
        "inner": parent / "folds" / "inner_split_manifest.csv",
        "probability": parent / "analyses" / "oof_probabilities.csv",
        "candidates": parent / "analyses" / "presto_inner_oof_candidates.csv",
        "parent_policy": parent / "analyses" / "policy_oof_decisions.csv",
        "parent_selections": parent / "analyses" / "outer_selections.csv",
    }
    tables = {name: pd.read_csv(path) for name, path in paths.items()}
    for name, frame in tables.items():
        if name != "parent_selections":
            if "record_id" not in frame.columns:
                raise KeyError(f"Parent {name} table lacks record_id")
            frame["record_id"] = frame["record_id"].astype(str)
    records = tables["records"]
    labels = tables["labels"]
    outer = tables["outer"]
    inner = tables["inner"]
    probability = tables["probability"]
    candidates = tables["candidates"]
    parent_policy = tables["parent_policy"]
    if (
        len(records) != 97
        or records["record_id"].nunique() != 97
        or records["cv_group_id"].astype(str).nunique() != 80
        or len(labels) != 97
        or len(outer) != 970
        or len(inner) != 3880
    ):
        raise ValueError("PRESTO-v1 parent population/folds changed")
    q = labels.loc[:, Q_COLUMNS].to_numpy(float)
    if (
        not np.isfinite(q).all()
        or np.any(q < 0)
        or not np.allclose(q.sum(axis=1), 1.0, atol=1.0e-12, rtol=0)
    ):
        raise ValueError("Parent soft labels are invalid")
    if probability.duplicated(
        ["evaluation_model", "repeat", "fold", "record_id"]
    ).any():
        raise ValueError("Parent probability OOF keys are not unique")
    probability_matrix_audit(probability.loc[:, P_COLUMNS].to_numpy(float))
    hybrid = probability.loc[
        probability["evaluation_model"].astype(str).eq(PRESTO_HYBRID_MODEL)
    ]
    if len(hybrid) != 970:
        raise ValueError("Parent hybrid probability coverage changed")
    low_candidates = candidates.loc[
        candidates["penalty"].astype(float).isin(POLICY_PENALTIES)
    ]
    if (
        low_candidates["fit_failed"].astype(bool).any()
        or low_candidates.duplicated(
            ["repeat", "fold", "penalty", "record_id"]
        ).any()
    ):
        raise ValueError("Required parent low-penalty candidate OOF is invalid")
    parent_policy = parent_policy.loc[
        parent_policy["decision_policy"].astype(str).eq(PARENT_PRESTO_POLICY)
    ].copy()
    if len(parent_policy) != 970:
        raise ValueError("Parent PRESTO-v1 policy coverage changed")
    tables["parent_policy"] = parent_policy
    return tables


def _soft_f_beta(precision: float, recall: float, beta: float) -> float:
    beta_squared = float(beta) ** 2
    denominator = beta_squared * float(precision) + float(recall)
    if denominator <= 0:
        return 0.0
    return float(
        (1.0 + beta_squared)
        * float(precision)
        * float(recall)
        / denominator
    )


def _select_policy_penalty_threshold(
    candidate_oof: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Jointly select only from grouped inner-OOF soft targets."""

    required = {
        "penalty",
        "inner_fold",
        "record_id",
        "q_high_off",
        "p_high_off",
        "fit_failed",
    }
    missing = sorted(required - set(candidate_oof.columns))
    if missing:
        raise KeyError(f"Policy candidate OOF lacks columns: {missing}")
    rows: list[dict[str, Any]] = []
    expected_records: set[str] | None = None
    for penalty in POLICY_PENALTIES:
        frame = candidate_oof.loc[
            np.isclose(
                candidate_oof["penalty"].to_numpy(float),
                float(penalty),
                atol=0,
                rtol=0,
            )
        ].copy()
        ids = set(frame["record_id"].astype(str))
        if expected_records is None:
            expected_records = ids
        if (
            not ids
            or ids != expected_records
            or len(frame) != len(ids)
            or frame["fit_failed"].astype(bool).any()
        ):
            raise ValueError("Low-penalty inner-OOF coverage is incomplete")
        target = frame["q_high_off"].to_numpy(float)
        prediction = frame["p_high_off"].to_numpy(float)
        weights = _fold_equal_weights(frame["inner_fold"].to_numpy(int))
        for threshold in THRESHOLD_GRID:
            high = prediction >= float(threshold)
            tp = float(np.sum(weights * target * high))
            fn = float(np.sum(weights * target * (~high)))
            fp = float(np.sum(weights * (1.0 - target) * high))
            tn = float(np.sum(weights * (1.0 - target) * (~high)))
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
            rows.append(
                {
                    "penalty": float(penalty),
                    "threshold": float(threshold),
                    "soft_high_recall": recall,
                    "soft_high_precision": precision,
                    "soft_high_false_positive_rate": fpr,
                    "soft_f0_5": _soft_f_beta(precision, recall, F_BETA),
                    "weighted_predicted_high_fraction": float(
                        np.sum(weights * high)
                    ),
                    "passes_expected_fpr_cap": bool(
                        fpr <= EXPECTED_FPR_CAP + TIE_TOLERANCE
                    ),
                    "outer_hard_labels_used": False,
                }
            )
    table = pd.DataFrame(rows)
    eligible = table.loc[table["passes_expected_fpr_cap"]].copy()
    if eligible.empty:
        raise RuntimeError("No policy candidate satisfies the expected-FPR cap")
    selected = eligible.sort_values(
        [
            "soft_f0_5",
            "soft_high_recall",
            "soft_high_false_positive_rate",
            "threshold",
            "penalty",
        ],
        ascending=[False, False, True, False, False],
        kind="stable",
    ).iloc[0]
    table["selected"] = np.isclose(
        table["penalty"].to_numpy(float),
        float(selected["penalty"]),
        atol=0,
        rtol=0,
    ) & np.isclose(
        table["threshold"].to_numpy(float),
        float(selected["threshold"]),
        atol=0,
        rtol=0,
    )
    return (
        {key: _jsonable(value) for key, value in selected.to_dict().items()},
        table,
    )


def _run_outer_split(
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    candidates: pd.DataFrame,
    inherited_hybrid: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    test_ids: Sequence[str],
) -> dict[str, Any]:
    ordered = records_by_id.index.astype(str).tolist()
    test = [str(value) for value in test_ids]
    test_set = set(test)
    train = [record_id for record_id in ordered if record_id not in test_set]
    train_groups = set(records_by_id.loc[train, "cv_group_id"].astype(str))
    test_groups = set(records_by_id.loc[test, "cv_group_id"].astype(str))
    if train_groups & test_groups:
        raise AssertionError("A connected group crosses a policy-v2 outer fold")

    candidate_split = candidates.loc[
        candidates["repeat"].eq(int(repeat))
        & candidates["fold"].eq(int(fold))
    ].copy()
    selected, grid = _select_policy_penalty_threshold(candidate_split)
    penalty = float(selected["penalty"])
    threshold = float(selected["threshold"])

    X_train = records_by_id.loc[train].reset_index(drop=True)
    X_test = records_by_id.loc[test].reset_index(drop=True)
    y_train = labels_by_id.loc[train, Q_COLUMNS].to_numpy(float)
    novelty_train, novelty_test, novelty_audit = _fold_local_novelty_features(
        X_train,
        records_by_id.loc[train, "cv_group_id"].astype(str).tolist(),
        X_test,
    )
    policy_probability, optimizer = _fit_presto(
        novelty_train,
        y_train,
        novelty_test,
        penalty=penalty,
    )
    high_score = policy_probability[:, 2]

    inherited = inherited_hybrid.loc[
        inherited_hybrid["repeat"].eq(int(repeat))
        & inherited_hybrid["fold"].eq(int(fold))
    ].copy()
    inherited = inherited.set_index("record_id").loc[test].reset_index()
    inherited_probability = inherited.loc[:, P_COLUMNS].to_numpy(float)
    probability_matrix_audit(inherited_probability)
    high = high_score >= threshold
    decision = np.argmax(inherited_probability[:, :2], axis=1)
    decision[high] = 2

    rows: list[dict[str, Any]] = []
    for index, record_id in enumerate(test):
        identity = labels_by_id.loc[record_id]
        rows.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "record_id": str(record_id),
                "cv_group_id": str(identity["cv_group_id"]),
                **{column: float(identity[column]) for column in Q_COLUMNS},
                "decision_policy": PRIMARY_POLICY,
                "probability_model": PRESTO_HYBRID_MODEL,
                "policy_score_model": POLICY_SCORE_MODEL,
                **{
                    column: float(inherited_probability[index, class_index])
                    for class_index, column in enumerate(P_COLUMNS)
                },
                "selected_policy_penalty": penalty,
                "selected_high_threshold": threshold,
                "high_score": float(high_score[index]),
                "raw_threshold_margin": float(high_score[index] - threshold),
                "predicted_class_index": int(decision[index]),
                "predicted_class": str(CLASS_LABELS[int(decision[index])]),
                "predicted_high": bool(high[index]),
                "outer_test_used_for_selection": False,
            }
        )
    selection = {
        "repeat": int(repeat),
        "fold": int(fold),
        "selected_policy_penalty": penalty,
        "selected_high_threshold": threshold,
        "inner_soft_f0_5": float(selected["soft_f0_5"]),
        "inner_soft_high_recall": float(selected["soft_high_recall"]),
        "inner_soft_high_precision": float(selected["soft_high_precision"]),
        "inner_soft_high_false_positive_rate": float(
            selected["soft_high_false_positive_rate"]
        ),
        "outer_hard_labels_used": False,
        "selection_audit_json": json.dumps(selected, sort_keys=True),
    }
    audit = {
        "repeat": int(repeat),
        "fold": int(fold),
        "train_record_ids": train,
        "test_record_ids": test,
        "selection": selected,
        "novelty": novelty_audit,
        "optimizer": optimizer,
    }
    return {
        "policy_rows": pd.DataFrame(rows),
        "selection_grid": grid.assign(repeat=int(repeat), fold=int(fold)),
        "selection": pd.DataFrame([selection]),
        "audit": audit,
    }


def _aggregate_candidate_policy(
    policy_oof: pd.DataFrame,
) -> pd.DataFrame:
    """Apply three fixed repeat aggregation rules without consulting labels."""

    aggregation = (
        policy_oof.groupby(
            ["record_id", "cv_group_id"], as_index=False, sort=True
        )
        .agg(
            {
                **{column: "mean" for column in Q_COLUMNS},
                **{column: "mean" for column in P_COLUMNS},
                "selected_policy_penalty": "mean",
                "selected_high_threshold": "mean",
                "high_score": "mean",
                "raw_threshold_margin": "mean",
                "predicted_high": "mean",
            }
        )
        .rename(columns={"predicted_high": "predicted_high_vote_fraction"})
    )
    definitions = (
        (
            PRIMARY_POLICY,
            "high_if_mean_raw_score_minus_selected_threshold_is_nonnegative",
            aggregation["raw_threshold_margin"].to_numpy(float) >= 0.0,
        ),
        (
            VOTE5_POLICY,
            "high_if_at_least_5_of_10_repeat_votes",
            aggregation["predicted_high_vote_fraction"].to_numpy(float) >= 0.5,
        ),
        (
            VOTE6_POLICY,
            "high_if_at_least_6_of_10_repeat_votes",
            aggregation["predicted_high_vote_fraction"].to_numpy(float) >= 0.6,
        ),
    )
    outputs: list[pd.DataFrame] = []
    for policy, rule, high in definitions:
        output = aggregation.copy()
        decision = np.argmax(output.loc[:, P_COLUMNS[:2]].to_numpy(float), axis=1)
        decision[high] = 2
        output["decision_policy"] = policy
        output["record_aggregation_rule"] = rule
        output["predicted_class_index"] = decision
        output["predicted_class"] = [CLASS_LABELS[int(value)] for value in decision]
        output["predicted_high"] = high
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def _aggregate_parent_policy(parent_oof: pd.DataFrame) -> pd.DataFrame:
    aggregation = (
        parent_oof.groupby(
            ["record_id", "cv_group_id"], as_index=False, sort=True
        )
        .agg(
            {
                **{column: "mean" for column in Q_COLUMNS},
                **{column: "mean" for column in P_COLUMNS},
                "selected_high_threshold": "mean",
                "high_score": "mean",
                "predicted_high": "mean",
            }
        )
        .rename(columns={"predicted_high": "predicted_high_vote_fraction"})
    )
    high = aggregation["predicted_high_vote_fraction"].to_numpy(float) >= 0.5
    decision = np.argmax(aggregation.loc[:, P_COLUMNS[:2]].to_numpy(float), axis=1)
    decision[high] = 2
    aggregation["decision_policy"] = PARENT_POLICY
    aggregation["record_aggregation_rule"] = (
        "parent_v1_high_if_at_least_half_of_repeat_votes"
    )
    aggregation["predicted_class_index"] = decision
    aggregation["predicted_class"] = [CLASS_LABELS[int(value)] for value in decision]
    aggregation["predicted_high"] = high
    return aggregation


def _policy_metric_tables(
    new_oof: pd.DataFrame,
    parent_oof: pd.DataFrame,
    aggregated: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    aggregate_rows: list[dict[str, Any]] = []
    for policy, frame in aggregated.groupby("decision_policy", sort=True):
        aggregate_rows.append(
            {
                "decision_policy": str(policy),
                "record_aggregation_rule": str(
                    frame["record_aggregation_rule"].iloc[0]
                ),
                **_decision_metrics(
                    frame.loc[:, Q_COLUMNS].to_numpy(float),
                    frame["predicted_class_index"].to_numpy(int),
                ),
            }
        )
    repeat_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for policy, source in ((PRIMARY_POLICY, new_oof), (PARENT_POLICY, parent_oof)):
        for (repeat, fold), frame in source.groupby(["repeat", "fold"], sort=True):
            split_rows.append(
                {
                    "decision_policy": policy,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    **_decision_metrics(
                        frame.loc[:, Q_COLUMNS].to_numpy(float),
                        frame["predicted_class_index"].to_numpy(int),
                    ),
                }
            )
        for repeat, frame in source.groupby("repeat", sort=True):
            repeat_rows.append(
                {
                    "decision_policy": policy,
                    "repeat": int(repeat),
                    **_decision_metrics(
                        frame.loc[:, Q_COLUMNS].to_numpy(float),
                        frame["predicted_class_index"].to_numpy(int),
                    ),
                }
            )
    return {
        "policy_aggregate_metrics": pd.DataFrame(aggregate_rows),
        "policy_repeat_metrics": pd.DataFrame(repeat_rows),
        "policy_split_metrics": pd.DataFrame(split_rows),
    }


def _paired_policy_bootstrap(
    aggregated: pd.DataFrame,
    *,
    focal_policy: str,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focal = aggregated.loc[
        aggregated["decision_policy"].eq(str(focal_policy))
    ].copy()
    parent = aggregated.loc[
        aggregated["decision_policy"].eq(PARENT_POLICY)
    ].copy()
    merged = focal.merge(
        parent,
        on=["record_id", "cv_group_id"],
        suffixes=("_focal", "_parent"),
        validate="one_to_one",
    )
    q = merged.loc[:, [f"{column}_focal" for column in Q_COLUMNS]].to_numpy(float)
    focal_decision = merged["predicted_class_index_focal"].to_numpy(int)
    parent_decision = merged["predicted_class_index_parent"].to_numpy(int)
    group_values = merged["cv_group_id"].astype(str).to_numpy()
    groups = np.asarray(sorted(set(group_values)), dtype=object)
    if len(groups) != 80:
        raise ValueError("Policy-v2 bootstrap requires 80 connected groups")
    metric_names = (
        "modal_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "posterior_expected_hard_accuracy",
        "modal_high_recall",
        "modal_high_precision",
        "modal_high_false_positive_rate",
        "posterior_expected_high_recall",
        "posterior_expected_high_precision",
        "posterior_expected_high_false_positive_rate",
        "predicted_high_fraction",
    )
    focal_observed = _decision_metrics(q, focal_decision)
    parent_observed = _decision_metrics(q, parent_decision)
    lookup = {
        group: np.flatnonzero(group_values == group) for group in groups
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrast = f"{focal_policy}_minus_{PARENT_POLICY}"
    rows: list[dict[str, Any]] = []
    for replicate in range(int(repetitions)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([lookup[str(group)] for group in sampled])
        focal_metrics = _decision_metrics(q[indices], focal_decision[indices])
        parent_metrics = _decision_metrics(q[indices], parent_decision[indices])
        rows.append(
            {
                "contrast": contrast,
                "replicate": int(replicate),
                **{
                    f"delta_{name}": (
                        float(focal_metrics[name]) - float(parent_metrics[name])
                    )
                    for name in metric_names
                },
            }
        )
    draws = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for name in metric_names:
        values = draws[f"delta_{name}"].to_numpy(float)
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "contrast": contrast,
                "metric": f"delta_{name}",
                "observed_delta": float(focal_observed[name])
                - float(parent_observed[name]),
                "bootstrap_mean": float(np.mean(finite)),
                "bootstrap_median": float(np.median(finite)),
                "ci95_low": float(np.quantile(finite, 0.025)),
                "ci95_high": float(np.quantile(finite, 0.975)),
                "probability_below_zero": float(np.mean(finite < 0)),
                "probability_above_zero": float(np.mean(finite > 0)),
                "n_bootstrap": int(repetitions),
                "n_valid_bootstrap": int(len(finite)),
                "n_records": int(len(merged)),
                "n_connected_groups": int(len(groups)),
                "conditional_on_globally_postselected_rule": True,
            }
        )
    return draws, pd.DataFrame(summary_rows)


def _copy_bound_inputs(
    destination: Path,
    parent: Path,
    contract_path: Path,
) -> None:
    copies = {
        parent / "inputs" / "records.csv": destination / "inputs" / "records.csv",
        parent / "registry" / "contiguous_three_class_soft_labels.csv": (
            destination / "registry" / "contiguous_three_class_soft_labels.csv"
        ),
        parent / "folds" / "outer_split_manifest.csv": (
            destination / "folds" / "outer_split_manifest.csv"
        ),
        parent / "folds" / "inner_split_manifest.csv": (
            destination / "folds" / "inner_split_manifest.csv"
        ),
        parent / "analyses" / "oof_probabilities.csv": (
            destination / "inputs" / "parent_oof_probabilities.csv"
        ),
        parent / "analyses" / "presto_inner_oof_candidates.csv": (
            destination / "inputs" / "parent_presto_inner_oof_candidates.csv"
        ),
        parent / "analyses" / "policy_oof_decisions.csv": (
            destination / "inputs" / "parent_policy_oof_decisions.csv"
        ),
        parent / "registry" / "contract_snapshot.yaml": (
            destination / "inputs" / "parent_contract_snapshot.yaml"
        ),
        parent / "provenance" / "code_manifest.json": (
            destination / "inputs" / "parent_code_manifest.json"
        ),
        parent / "provenance" / "environment_manifest.json": (
            destination / "inputs" / "parent_environment_manifest.json"
        ),
        contract_path: destination / "registry" / "contract_snapshot.yaml",
    }
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_code_manifest(destination: Path, contract_path: Path) -> None:
    sources = [
        Path(__file__).resolve(),
        contract_path.resolve(),
        PROJECT_ROOT
        / "src"
        / "ink_tafel_study"
        / "posterior_cardinal_contiguous_threeclass_presto_gate.py",
        PROJECT_ROOT / "src" / "ink_tafel_study" / "presto_ordinal.py",
        PROJECT_ROOT
        / "src"
        / "ink_tafel_study"
        / "posterior_cardinal_contiguous_threeclass.py",
        PROJECT_ROOT
        / "src"
        / "ink_tafel_study"
        / "threeclass_sensitivity_models.py",
        PROJECT_ROOT / "src" / "ink_tafel_study" / "provenance.py",
    ]
    files = inventory_files(sources, relative_to=REPOSITORY_ROOT)
    write_json(
        destination / "provenance" / "code_manifest.json",
        {"tree_sha256": tree_sha256(files), "files": files},
    )


def _write_report(
    destination: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    qa: Mapping[str, Any],
) -> Path:
    columns = (
        "decision_policy",
        "modal_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "posterior_expected_hard_accuracy",
        "predicted_high_count",
        "modal_high_recall",
        "modal_high_precision",
        "modal_high_false_positive_rate",
    )
    display = metrics.loc[:, columns].copy()
    lines = [
        "# PRESTO policy-only F0.5 mean-margin development sensitivity",
        "",
        "The PRESTO-v1 probability OOF predictions are inherited without refit or recalibration. This package changes only a separate cost-sensitive action policy.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                f"{value:.6f}" if isinstance(value, float) else str(value)
                for value in row
            )
            + " |"
        )
    focal = bootstrap.loc[
        bootstrap["contrast"].str.startswith(PRIMARY_POLICY)
        & bootstrap["metric"].eq("delta_modal_accuracy")
    ]
    if not focal.empty:
        row = focal.iloc[0]
        lines.extend(
            [
                "",
                (
                    "Primary-policy minus parent-v1 accuracy delta: "
                    f"{float(row['observed_delta']):.6f}, conditional connected-group "
                    f"bootstrap 95% CI [{float(row['ci95_low']):.6f}, "
                    f"{float(row['ci95_high']):.6f}]."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "The policy penalty and threshold are selected only from grouped inner-OOF soft targets using soft F0.5 under expected FPR <= 0.20. Repeated OOF alerts use mean(raw score - selected threshold) >= 0; five-vote and six-vote rules are fixed sensitivities.",
            "",
            "This exact utility and aggregation rule were chosen after inspecting outer results on the same closed dataset. Nested CV and the conditional bootstrap do not correct that global post-selection. The package is not independently confirmed or scientifically releasable and cannot replace any frozen result.",
            "",
            "Method sources: Faletto and Bien (2023), https://proceedings.mlr.press/v202/faletto23a.html; Varma and Simon (2006), https://doi.org/10.1186/1471-2105-7-91.",
            "",
            f"Artifact QA: `{qa['status']}`; probability maximum absolute mismatch: `{qa['maximum_parent_probability_numeric_error']}`.",
            "",
        ]
    )
    report = destination / "report" / "presto_policy_v2_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_presto_policy_v2_sensitivity(
    destination: str | Path,
    *,
    parent_run: str | Path | None = None,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    n_jobs: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the isolated policy-only refinement without changing probabilities."""

    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    contract, resolved_contract, contract_hash = load_presto_policy_v2_contract(
        contract_path
    )
    parent = _resolve_parent(contract, parent_run)
    output = _resolve_path(destination, base=Path.cwd())
    _assert_fresh_disjoint_destination(output, parent)
    parent_before = _verify_parent(parent, contract)
    # These guards must pass before loading refit inputs and before any call to
    # the inherited PRESTO-v1 novelty builder or optimizer.
    refit_code_before = _verify_parent_refit_code(parent)
    refit_environment_before = _verify_parent_refit_environment(parent)
    tables = _load_parent_tables(parent)

    records = tables["records"]
    labels = tables["labels"]
    records_by_id = _indexed(records, name="records")
    labels_with_group = labels.drop(columns=["cv_group_id"], errors="ignore").merge(
        records.loc[:, ["record_id", "cv_group_id"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    labels_by_id = _indexed(labels_with_group, name="labels")
    inherited_hybrid = tables["probability"].loc[
        tables["probability"]["evaluation_model"].astype(str).eq(
            PRESTO_HYBRID_MODEL
        )
    ].copy()

    selected_manifest = tables["outer"].copy()
    if smoke:
        selected_manifest = selected_manifest.loc[
            selected_manifest["repeat"].eq(
                int(selected_manifest["repeat"].min())
            )
        ].copy()
    split_groups = [
        (int(repeat), int(fold), frame["record_id"].astype(str).tolist())
        for (repeat, fold), frame in selected_manifest.groupby(
            ["repeat", "fold"], sort=True
        )
    ]
    results = Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(_run_outer_split)(
            records_by_id,
            labels_by_id,
            tables["candidates"],
            inherited_hybrid,
            repeat=repeat,
            fold=fold,
            test_ids=test_ids,
        )
        for repeat, fold, test_ids in split_groups
    )
    new_policy_oof = pd.concat(
        [result["policy_rows"] for result in results], ignore_index=True
    )
    selection_grid = pd.concat(
        [result["selection_grid"] for result in results], ignore_index=True
    )
    selections = pd.concat(
        [result["selection"] for result in results], ignore_index=True
    ).sort_values(["repeat", "fold"], kind="stable")
    outer_audits = [result["audit"] for result in results]

    selected_keys = selected_manifest.loc[:, ["repeat", "fold", "record_id"]]
    probability_oof = tables["probability"].merge(
        selected_keys,
        on=["repeat", "fold", "record_id"],
        how="inner",
        validate="many_to_one",
        sort=False,
    )
    parent_policy_oof = tables["parent_policy"].merge(
        selected_keys,
        on=["repeat", "fold", "record_id"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(new_policy_oof) != len(selected_manifest):
        raise AssertionError("Policy-v2 OOF coverage changed")
    if new_policy_oof["outer_test_used_for_selection"].astype(bool).any():
        raise AssertionError("Outer-test selection flag became true")

    aggregated_probability = aggregate_contiguous_oof(probability_oof)
    probability_tables = _metric_tables(
        probability_oof,
        aggregated_probability,
        entropy_threshold=ENTROPY_THRESHOLD,
    )
    probability_argmax = _probability_argmax_metrics(aggregated_probability)
    candidate_aggregated = _aggregate_candidate_policy(new_policy_oof)
    parent_aggregated = _aggregate_parent_policy(parent_policy_oof)
    aggregated_policy = pd.concat(
        [candidate_aggregated, parent_aggregated], ignore_index=True, sort=False
    )
    policy_tables = _policy_metric_tables(
        new_policy_oof, parent_policy_oof, aggregated_policy
    )

    repetitions = (
        SMOKE_BOOTSTRAP_REPETITIONS if smoke else BOOTSTRAP_REPETITIONS
    )
    bootstrap_draws: list[pd.DataFrame] = []
    bootstrap_summaries: list[pd.DataFrame] = []
    for focal in (PRIMARY_POLICY, VOTE5_POLICY, VOTE6_POLICY):
        draws, summary = _paired_policy_bootstrap(
            aggregated_policy,
            focal_policy=focal,
            repetitions=repetitions,
        )
        bootstrap_draws.append(draws)
        bootstrap_summaries.append(summary)
    bootstrap_draws_frame = pd.concat(bootstrap_draws, ignore_index=True)
    bootstrap_summary = pd.concat(bootstrap_summaries, ignore_index=True)

    parent_after = _verify_parent(parent, contract)
    refit_code_after = _verify_parent_refit_code(parent)
    refit_environment_after = _verify_parent_refit_environment(parent)
    parent_tree_unchanged = (
        parent_before["artifact_manifest_verification"]["actual_tree_sha256"]
        == parent_after["artifact_manifest_verification"]["actual_tree_sha256"]
    )
    if not parent_tree_unchanged:
        raise AssertionError("Immutable PRESTO-v1 parent changed during policy fit")
    refit_code_tree_unchanged = (
        refit_code_before["current_tree_sha256"]
        == refit_code_after["current_tree_sha256"]
    )
    if not refit_code_tree_unchanged:
        raise AssertionError("PRESTO-v1 refit code changed during policy fit")
    refit_environment_unchanged = (
        refit_environment_before["current_python"]
        == refit_environment_after["current_python"]
        and refit_environment_before["package_comparisons"]
        == refit_environment_after["package_comparisons"]
    )
    if not refit_environment_unchanged:
        raise AssertionError("PRESTO-v1 refit environment changed during policy fit")

    output.mkdir(parents=True)
    for name in ("inputs", "registry", "folds", "analyses", "provenance", "report"):
        (output / name).mkdir()
    _copy_bound_inputs(output, parent, resolved_contract)
    if smoke:
        probability_oof.to_csv(
            output / "analyses" / "oof_probabilities.csv", index=False
        )
        aggregated_probability.to_csv(
            output / "analyses" / "oof_aggregated.csv", index=False
        )
    else:
        shutil.copy2(
            parent / "analyses" / "oof_probabilities.csv",
            output / "analyses" / "oof_probabilities.csv",
        )
        shutil.copy2(
            parent / "analyses" / "oof_aggregated.csv",
            output / "analyses" / "oof_aggregated.csv",
        )
    frames = {
        "policy_oof_decisions": pd.concat(
            [new_policy_oof, parent_policy_oof], ignore_index=True, sort=False
        ),
        "policy_record_aggregated": aggregated_policy,
        "policy_inner_selection_grid": selection_grid,
        "outer_selections": selections,
        "probability_argmax_metrics": probability_argmax,
        "policy_connected_group_bootstrap": bootstrap_draws_frame,
        "policy_connected_group_bootstrap_summary": bootstrap_summary,
        **probability_tables,
        **policy_tables,
    }
    for name, frame in frames.items():
        frame.to_csv(output / "analyses" / f"{name}.csv", index=False)
    write_json(
        output / "analyses" / "outer_fit_audits.json", _jsonable(outer_audits)
    )

    written_probability = pd.read_csv(
        output / "analyses" / "oof_probabilities.csv"
    )
    written_probability["record_id"] = written_probability["record_id"].astype(str)
    expected_probability = probability_oof.copy()
    expected_probability["record_id"] = expected_probability["record_id"].astype(str)
    key = ["evaluation_model", "repeat", "fold", "record_id"]
    check = expected_probability.merge(
        written_probability,
        on=key,
        suffixes=("_expected", "_written"),
        validate="one_to_one",
    )
    if len(check) != len(expected_probability):
        raise AssertionError("Written inherited probability coverage changed")
    probability_error = float(
        np.max(
            np.abs(
                check.loc[
                    :, [f"{column}_expected" for column in P_COLUMNS]
                ].to_numpy(float)
                - check.loc[
                    :, [f"{column}_written" for column in P_COLUMNS]
                ].to_numpy(float)
            ),
            initial=0.0,
        )
    )
    if probability_error != 0.0:
        raise AssertionError("Inherited PRESTO-v1 probabilities changed")
    byte_identical = bool(
        not smoke
        and sha256_file(output / "analyses" / "oof_probabilities.csv")
        == sha256_file(parent / "analyses" / "oof_probabilities.csv")
    )
    if not smoke and not byte_identical:
        raise AssertionError("Full probability OOF CSV is not byte-identical")

    aggregate_metrics = policy_tables["policy_aggregate_metrics"]
    primary_metric = aggregate_metrics.loc[
        aggregate_metrics["decision_policy"].eq(PRIMARY_POLICY)
    ].iloc[0]
    parent_metric = aggregate_metrics.loc[
        aggregate_metrics["decision_policy"].eq(PARENT_POLICY)
    ].iloc[0]
    qa = {
        "status": (
            "complete_smoke"
            if smoke
            else "complete_posthoc_global_postselection_development"
        ),
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_hash,
        "analysis_role": "posthoc_global_postselection_policy_development_sensitivity",
        "scientifically_releasable": False,
        "independent_confirmation": False,
        "may_replace_presto_gate_v1": False,
        "may_replace_or_rescue_v4": False,
        "global_policy_rule_selected_after_closed_outer_results_seen": True,
        "nested_cv_does_not_cover_global_method_selection": True,
        "bootstrap_conditional_on_globally_postselected_rule": True,
        "n_records_in_registry": int(len(records)),
        "n_connected_groups_in_registry": int(
            records["cv_group_id"].astype(str).nunique()
        ),
        "n_outer_splits_evaluated": int(len(split_groups)),
        "n_policy_oof_rows": int(len(new_policy_oof)),
        "outer_test_used_for_penalty_threshold_or_aggregation_selection": False,
        "posterior_argmax_used_for_policy_selection": False,
        "posterior_draws_expanded": False,
        "class_rows_expanded": False,
        "probability_refit_or_recalibration_performed": False,
        "maximum_parent_probability_numeric_error": probability_error,
        "full_probability_oof_byte_identical_to_parent": byte_identical,
        "parent_probability_sha256": sha256_file(
            parent / "analyses" / "oof_probabilities.csv"
        ),
        "output_probability_sha256": sha256_file(
            output / "analyses" / "oof_probabilities.csv"
        ),
        "selected_policy_penalties": sorted(
            set(selections["selected_policy_penalty"].astype(float))
        ),
        "selected_threshold_one_count": int(
            np.isclose(
                selections["selected_high_threshold"].to_numpy(float),
                1.0,
                atol=0,
                rtol=0,
            ).sum()
        ),
        "parent_manifest_verified_before": parent_before["status"] == "pass",
        "parent_manifest_verified_after": parent_after["status"] == "pass",
        "parent_refit_code_verified_before_policy_refits": (
            refit_code_before["status"] == "pass"
        ),
        "parent_refit_code_verified_after_modeling": (
            refit_code_after["status"] == "pass"
        ),
        "parent_refit_code_manifest_sha256": refit_code_before[
            "manifest_sha256"
        ],
        "parent_refit_code_tree_sha256": refit_code_before[
            "current_tree_sha256"
        ],
        "parent_refit_code_file_count": refit_code_before[
            "n_manifest_files_checked"
        ],
        "parent_refit_code_tree_unchanged_during_run": refit_code_tree_unchanged,
        "parent_refit_environment_verified_before_policy_refits": (
            refit_environment_before["status"] == "pass"
        ),
        "parent_refit_environment_verified_after_modeling": (
            refit_environment_after["status"] == "pass"
        ),
        "parent_refit_environment_manifest_sha256": refit_environment_before[
            "manifest_sha256"
        ],
        "parent_refit_environment_versions_unchanged_during_run": (
            refit_environment_unchanged
        ),
        "parent_tree_unchanged": parent_tree_unchanged,
        "primary_policy_modal_accuracy": float(primary_metric["modal_accuracy"]),
        "primary_policy_balanced_accuracy": float(
            primary_metric["balanced_accuracy"]
        ),
        "primary_policy_macro_f1": float(primary_metric["macro_f1"]),
        "primary_policy_modal_high_recall": float(
            primary_metric["modal_high_recall"]
        ),
        "primary_policy_modal_high_precision": float(
            primary_metric["modal_high_precision"]
        ),
        "primary_policy_modal_high_false_positive_rate": float(
            primary_metric["modal_high_false_positive_rate"]
        ),
        "parent_policy_modal_accuracy": float(parent_metric["modal_accuracy"]),
    }
    write_json(output / "analyses" / "analysis_qa.json", _jsonable(qa))
    report = _write_report(output, aggregate_metrics, bootstrap_summary, qa)
    write_json(
        output / "report" / "validation_report.json",
        {
            "overall_assessment": (
                "structural_smoke_only"
                if smoke
                else "posthoc_global_postselection_development_not_confirmatory"
            ),
            "probability_values_inherited_exactly": probability_error == 0.0,
            "policy_claim_ready": False,
            "independent_confirmation": False,
            "scientifically_releasable": False,
            "qa": _jsonable(qa),
            "report": str(report),
        },
    )
    _write_code_manifest(output, resolved_contract)
    write_json(
        output / "provenance" / "environment_manifest.json",
        environment_manifest(REPOSITORY_ROOT),
    )
    write_json(
        output / "provenance" / "contract_binding.json",
        {
            "contract_id": CONTRACT_ID,
            "contract_path": str(resolved_contract),
            "contract_sha256": contract_hash,
        },
    )
    write_json(output / "provenance" / "parent_binding.json", parent_before)
    write_json(
        output / "provenance" / "parent_refit_code_binding.json",
        {
            "verified_before_policy_refits": refit_code_before,
            "verified_after_modeling": refit_code_after,
            "code_tree_unchanged_during_run": refit_code_tree_unchanged,
        },
    )
    write_json(
        output / "provenance" / "parent_refit_environment_binding.json",
        {
            "verified_before_policy_refits": refit_environment_before,
            "verified_after_modeling": refit_environment_after,
            "versions_unchanged_during_run": refit_environment_unchanged,
        },
    )
    write_json(
        output / "provenance" / "run_mode.json",
        {
            "smoke": bool(smoke),
            "n_jobs": int(n_jobs),
            "parallel_backend": "joblib_threads",
            "n_outer_splits": int(len(split_groups)),
            "bootstrap_repetitions": int(repetitions),
        },
    )
    manifest_path = output / "artifact_manifest.json"
    write_json(manifest_path, artifact_manifest(output))
    verification = verify_artifact_manifest(output, manifest_path)
    if verification["status"] != "pass":
        raise RuntimeError(f"Policy-v2 artifact manifest failed: {verification}")
    return {
        "status": qa["status"],
        "run_dir": str(output),
        "report": str(report),
        "contract_sha256": contract_hash,
        "maximum_parent_probability_numeric_error": probability_error,
        "primary_policy_modal_accuracy": qa["primary_policy_modal_accuracy"],
        "primary_policy_modal_high_recall": qa[
            "primary_policy_modal_high_recall"
        ],
        "artifact_tree_sha256": verification["actual_tree_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PRESTO F0.5 mean-margin policy-only v2 sensitivity"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--parent-run")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_presto_policy_v2_sensitivity(
        args.run_dir,
        parent_run=args.parent_run,
        contract_path=args.contract,
        n_jobs=args.n_jobs,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_SHA256",
    "PARENT_POLICY",
    "PARENT_REFIT_REQUIRED_PACKAGES",
    "PARENT_REFIT_REQUIRED_PATHS",
    "POLICY_PENALTIES",
    "PRIMARY_POLICY",
    "VOTE5_POLICY",
    "VOTE6_POLICY",
    "_aggregate_candidate_policy",
    "_select_policy_penalty_threshold",
    "_verify_parent_refit_code",
    "_verify_parent_refit_environment",
    "load_presto_policy_v2_contract",
    "run_presto_policy_v2_sensitivity",
]
