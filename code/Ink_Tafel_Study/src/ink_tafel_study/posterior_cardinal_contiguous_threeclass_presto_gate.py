"""PRESTO-inspired rare-ordered-event gate development sensitivity.

This workflow is isolated from the immutable contiguous-v2, direct-Gower-v1,
and formal v4 results.  It reuses the direct-Gower result's connected-group
folds and evaluates a probability-valid fused continuation-ratio ordinal gate.

The probability model and the decision policy are deliberately separate:

* the probability gate selects only its smooth fused-L1 penalty using grouped
  inner-OOF posterior-soft cross-entropy;
* a recall-oriented high alert threshold is selected from the same
  outer-training inner-OOF predictions using soft F2 under a fixed expected
  false-positive-rate cap; and
* the identical threshold protocol is applied to a refitted direct-Gower
  comparator so that a recall gain is not incorrectly attributed to PRESTO
  when it is only a changed decision rule.

This is post-selection development evidence on the same closed 97-record
dataset.  It is not independent confirmation and cannot replace any frozen
scientific result.
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
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
import yaml

from .cardinal_metrics import normalize_probabilities
from .models import GowerKNeighborsRegressor
from .posterior_cardinal_contiguous_threeclass import (
    CLASS_LABELS,
    P_COLUMNS,
    PRIOR_MODEL,
    Q_COLUMNS,
    _manifest_inner_splits,
    _metric_tables,
    aggregate_contiguous_oof,
)
from .posterior_cardinal_contiguous_threeclass_direct_gower import (
    DIRECT_MODEL,
    fit_predict_direct_multiscale_gower,
)
from .posterior_cardinal_inference import paired_connected_group_bootstrap
from .presto_ordinal import SoftFusedContinuationRatioClassifier
from .provenance import (
    artifact_manifest,
    environment_manifest,
    inventory_files,
    sha256_file,
    tree_sha256,
    verify_artifact_manifest,
    write_json,
)
from .threeclass_sensitivity_models import (
    _fold_local_engineered_features,
    probability_matrix_audit,
)


CONTRACT_ID = "posterior_cardinal_contiguous_threeclass_presto_gate_v1"
CONTRACT_FILENAME = (
    "posterior_cardinal_contiguous_threeclass_presto_gate_v1.yaml"
)
CONTRACT_SHA256 = (
    "964428f61150981f15c882ba12e2df65f389f12f653e238678dcadf539ffde3c"
)
STAGE_NAME = "posterior-cardinal-contiguous-threeclass-presto-gate"

PRESTO_FULL_MODEL = "presto_inspired_fused_continuation_ratio"
PRESTO_HYBRID_MODEL = "presto_high_gate_direct_gower_conditional"
PRESTO_POLICY = "presto_inner_oof_soft_f2_fpr20_policy"
DIRECT_POLICY = "direct_gower_inner_oof_soft_f2_fpr20_policy"

PANEL = "context_plus_polarity"
NOVELTY_NEIGHBORS = 5
PENALTY_GRID: tuple[float, ...] = (
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
)
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
EXPECTED_FPR_CAP = 0.20
RARE_MIN_FREQUENCY = 3
RIDGE = 1.0e-6
L1_SMOOTHING = 1.0e-6
ENTROPY_THRESHOLD = 0.90
BOOTSTRAP_REPETITIONS = 2000
SMOKE_BOOTSTRAP_REPETITIONS = 100
BOOTSTRAP_SEED = 20260726
TIE_TOLERANCE = 1.0e-12

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "contracts" / CONTRACT_FILENAME
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _assert_fresh_disjoint_destination(
    destination: Path,
    parent: Path,
) -> None:
    output = destination.resolve()
    immutable = parent.resolve()
    if output == immutable or immutable in output.parents:
        raise ValueError("PRESTO output must be disjoint from its parent")
    if output in immutable.parents:
        raise ValueError("PRESTO output cannot contain its parent")
    if output.exists():
        raise FileExistsError(
            f"PRESTO output must be new and absent: {output}"
        )


def load_presto_gate_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> tuple[dict[str, Any], Path, str]:
    contract_path = _resolve_path(path, base=REPOSITORY_ROOT)
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"PRESTO gate contract is missing: {contract_path}"
        )
    digest = sha256_file(contract_path)
    if CONTRACT_SHA256.startswith("__"):
        raise RuntimeError(
            "PRESTO gate contract hash has not been frozen in source"
        )
    if digest != CONTRACT_SHA256:
        raise ValueError(
            "PRESTO gate contract hash mismatch: "
            f"expected={CONTRACT_SHA256}, actual={digest}"
        )
    loaded = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("PRESTO gate contract must be a mapping")
    contract = dict(loaded)
    if (
        contract.get("contract_id") != CONTRACT_ID
        or contract.get("stage") != STAGE_NAME
        or contract.get("status")
        != "posthoc_adaptive_development_frozen_before_packaged_run"
    ):
        raise ValueError("PRESTO gate contract identity changed")

    scope = dict(contract.get("scientific_scope", {}))
    if (
        scope.get("population") != "all97"
        or int(scope.get("expected_records", -1)) != 97
        or int(scope.get("expected_connected_groups", -1)) != 80
        or scope.get("independent_confirmation") is not False
        or scope.get("scientifically_releasable") is not False
        or scope.get("posterior_draws_expanded_as_samples") is not False
    ):
        raise ValueError("PRESTO gate scientific scope changed")

    model = dict(contract.get("model", {}))
    novelty = dict(model.get("fold_local_gower_novelty", {}))
    if (
        model.get("estimator_id") != PRESTO_HYBRID_MODEL
        or model.get("ordinal_parameterization")
        != "forward_continuation_ratio"
        or model.get("exact_presto_replication") is not False
        or model.get("feature_input") != "single_x_only_gower_novelty_score"
        or novelty.get("panel") != PANEL
        or int(novelty.get("neighbors", -1)) != NOVELTY_NEIGHBORS
        or tuple(float(value) for value in model.get("penalty_grid", ()))
        != PENALTY_GRID
    ):
        raise ValueError("PRESTO gate model contract changed")

    policy = dict(contract.get("high_recall_policy", {}))
    if (
        tuple(float(value) for value in policy.get("threshold_grid", ()))
        != THRESHOLD_GRID
        or float(policy.get("expected_fpr_cap", -1))
        != EXPECTED_FPR_CAP
        or policy.get("selection_utility") != "soft_f2"
        or policy.get("baseline_receives_identical_protocol") is not True
    ):
        raise ValueError("PRESTO high-recall policy contract changed")
    return contract, contract_path, digest


def _resolve_parent(
    contract: Mapping[str, Any],
    parent_run: str | Path | None,
) -> Path:
    settings = dict(contract["immutable_parent"])
    value = parent_run if parent_run is not None else settings["default_run"]
    return _resolve_path(value, base=REPOSITORY_ROOT)


def _verify_parent(
    parent: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not parent.is_dir():
        raise FileNotFoundError(f"PRESTO parent is missing: {parent}")
    settings = dict(contract["immutable_parent"])
    verification = verify_artifact_manifest(parent)
    if verification["status"] != "pass":
        raise RuntimeError(
            "PRESTO parent manifest verification failed: "
            f"{verification}"
        )
    expected_tree = str(settings["expected_artifact_tree_sha256"])
    if verification["actual_tree_sha256"] != expected_tree:
        raise RuntimeError(
            "PRESTO parent tree changed: "
            f"expected={expected_tree}, "
            f"actual={verification['actual_tree_sha256']}"
        )
    rows: list[dict[str, Any]] = []
    for relative, expected in dict(settings["required_files"]).items():
        path = parent / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing parent file: {relative}")
        actual = sha256_file(path)
        if actual != str(expected):
            raise RuntimeError(
                "PRESTO parent file changed: "
                f"{relative}, expected={expected}, actual={actual}"
            )
        rows.append(
            {
                "path": str(relative),
                "sha256": actual,
                "size_bytes": int(path.stat().st_size),
            }
        )
    return {
        "status": "pass",
        "parent_run": str(parent),
        "artifact_manifest_verification": verification,
        "required_files": rows,
    }


def _load_parent_tables(
    parent: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    records = pd.read_csv(parent / "inputs" / "records.csv")
    labels = pd.read_csv(
        parent / "registry" / "contiguous_three_class_soft_labels.csv"
    )
    outer = pd.read_csv(parent / "folds" / "outer_split_manifest.csv")
    inner = pd.read_csv(parent / "folds" / "inner_split_manifest.csv")
    parent_oof = pd.read_csv(parent / "analyses" / "oof_probabilities.csv")
    for frame in (records, labels, outer, inner, parent_oof):
        if "record_id" not in frame.columns:
            raise KeyError("A PRESTO parent table is missing record_id")
        frame["record_id"] = frame["record_id"].astype(str)

    if (
        len(records) != 97
        or records["record_id"].nunique() != 97
        or len(labels) != 97
        or labels["record_id"].nunique() != 97
        or records["cv_group_id"].astype(str).nunique() != 80
    ):
        raise ValueError("PRESTO parent population is not 97/80")
    q = labels.loc[:, Q_COLUMNS].to_numpy(float)
    if (
        not np.isfinite(q).all()
        or np.any(q < 0)
        or not np.allclose(q.sum(axis=1), 1.0, atol=1.0e-12, rtol=0)
    ):
        raise ValueError("PRESTO parent soft labels are invalid")
    if len(outer) != 970 or len(inner) != 3880:
        raise ValueError("PRESTO parent fold manifests changed")
    if (
        outer.groupby(["repeat", "record_id"]).size().ne(1).any()
        or outer.groupby(["repeat", "cv_group_id"])["fold"].nunique().gt(1).any()
        or inner.groupby(
            ["outer_repeat", "outer_fold", "cv_group_id"]
        )["inner_fold"].nunique().gt(1).any()
    ):
        raise ValueError("A connected group crosses a PRESTO fold")

    direct = parent_oof.loc[
        parent_oof["evaluation_model"].astype(str).eq(DIRECT_MODEL)
    ].copy()
    if (
        len(direct) != 970
        or direct.duplicated(["repeat", "fold", "record_id"]).any()
    ):
        raise ValueError("Direct-Gower parent OOF coverage changed")
    direct.loc[:, P_COLUMNS] = normalize_probabilities(
        direct.loc[:, P_COLUMNS].to_numpy(float)
    )
    return records, labels, outer, inner, direct


def _indexed(
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["record_id"] = output["record_id"].astype(str)
    if output["record_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate record IDs")
    return output.set_index("record_id", drop=False)


def _k_smallest_mean(distances: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(distances, dtype=float)
    if values.ndim != 2 or values.shape[1] < int(k):
        raise ValueError("Not enough external records for Gower novelty")
    partitioned = np.partition(values, int(k) - 1, axis=1)
    selected = partitioned[:, : int(k)]
    if not np.isfinite(selected).all():
        raise ValueError("Gower novelty contains no finite external neighbor")
    return np.mean(selected, axis=1)


def _fold_local_novelty_features(
    X_train: pd.DataFrame,
    train_groups: Sequence[str],
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = np.asarray(list(map(str, train_groups)), dtype=object)
    if groups.shape != (len(X_train),):
        raise ValueError("Training group IDs do not align with novelty rows")
    builder, engineered_train, engineered_test, source_columns = (
        _fold_local_engineered_features(
            X_train,
            X_test,
            panel=PANEL,
            rare_min_frequency=RARE_MIN_FREQUENCY,
        )
    )
    distance_model = GowerKNeighborsRegressor(
        n_neighbors=1,
        distance_power=1.0,
        numeric_features=builder.numeric_columns_out_,
        categorical_features=builder.categorical_columns_out_,
        blocks=builder.get_blocks_out(),
        epsilon=1.0e-12,
    ).fit(
        engineered_train,
        np.zeros(len(engineered_train), dtype=float),
    )
    train_distances = distance_model._pairwise_distances(engineered_train)
    train_distances[groups[:, None] == groups[None, :]] = np.inf
    test_distances = distance_model._pairwise_distances(engineered_test)
    train_score = _k_smallest_mean(
        train_distances,
        NOVELTY_NEIGHBORS,
    )
    test_score = _k_smallest_mean(
        test_distances,
        NOVELTY_NEIGHBORS,
    )
    train_frame = pd.DataFrame({"gower_novelty": train_score})
    test_frame = pd.DataFrame({"gower_novelty": test_score})
    return train_frame, test_frame, {
        "panel": PANEL,
        "neighbors": NOVELTY_NEIGHBORS,
        "same_connected_group_excluded_from_training_novelty": True,
        "source_columns": list(source_columns),
        "engineered_feature_names": [
            str(value) for value in builder.get_feature_names_out()
        ],
        "gower_blocks": {
            str(block): [str(value) for value in members]
            for block, members in builder.get_blocks_out().items()
        },
        "training_novelty_minimum": float(np.min(train_score)),
        "training_novelty_maximum": float(np.max(train_score)),
        "test_novelty_minimum": float(np.min(test_score)),
        "test_novelty_maximum": float(np.max(test_score)),
    }


def _fit_presto(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    *,
    penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    estimator = SoftFusedContinuationRatioClassifier(
        numeric_columns=("gower_novelty",),
        categorical_columns=(),
        class_names=CLASS_LABELS,
        penalty=float(penalty),
        ridge=RIDGE,
        l1_smoothing=L1_SMOOTHING,
        max_iter=2000,
        tolerance=1.0e-9,
    ).fit(X_train, y_train)
    if estimator.fit_status_ != "complete":
        raise RuntimeError(
            "PRESTO candidate did not converge and is invalid: "
            f"{estimator.optimization_audit_}"
        )
    prediction = estimator.predict_proba(X_test)
    probability_matrix_audit(prediction)
    return prediction, dict(estimator.optimization_audit_)


def _fold_equal_weights(fold_ids: np.ndarray) -> np.ndarray:
    folds = np.asarray(fold_ids, dtype=int)
    output = np.zeros(len(folds), dtype=float)
    for fold in np.unique(folds):
        selected = folds == int(fold)
        output[selected] = 1.0 / float(selected.sum())
    total = float(output.sum())
    if total <= 0:
        raise ValueError("Inner fold-equal weights are empty")
    return output / total


def _binary_soft_log_loss(
    q_high: np.ndarray,
    p_high: np.ndarray,
    weights: np.ndarray,
) -> float:
    target = np.asarray(q_high, dtype=float)
    prediction = np.clip(
        np.asarray(p_high, dtype=float),
        1.0e-15,
        1.0 - 1.0e-15,
    )
    return -float(
        np.sum(
            weights
            * (
                target * np.log(prediction)
                + (1.0 - target) * np.log(1.0 - prediction)
            )
        )
        / np.sum(weights)
    )


def _full_soft_log_loss(
    q: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
) -> float:
    target = np.asarray(q, dtype=float)
    prediction = np.clip(np.asarray(p, dtype=float), 1.0e-15, 1.0)
    return -float(
        np.sum(weights[:, None] * target * np.log(prediction))
        / np.sum(weights)
    )


def _threshold_table(
    q_high: np.ndarray,
    p_high: np.ndarray,
    fold_ids: np.ndarray,
    *,
    model: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = np.asarray(q_high, dtype=float)
    prediction = np.asarray(p_high, dtype=float)
    weights = _fold_equal_weights(np.asarray(fold_ids, dtype=int))
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        flag = prediction >= float(threshold)
        tp = float(np.sum(weights * target * flag))
        fn = float(np.sum(weights * target * (~flag)))
        fp = float(np.sum(weights * (1.0 - target) * flag))
        tn = float(np.sum(weights * (1.0 - target) * (~flag)))
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
        specificity = 1.0 - fpr
        denominator = 4.0 * precision + recall
        soft_f2 = (
            5.0 * precision * recall / denominator
            if denominator > 0
            else 0.0
        )
        rows.append(
            {
                "evaluation_model": str(model),
                "threshold": float(threshold),
                "soft_high_recall": recall,
                "soft_high_precision": precision,
                "soft_high_false_positive_rate": fpr,
                "soft_high_specificity": specificity,
                "soft_high_balanced_accuracy": 0.5
                * (recall + specificity),
                "soft_f2": soft_f2,
                "weighted_predicted_high_fraction": float(
                    np.sum(weights * flag)
                ),
                "passes_expected_fpr_cap": bool(
                    fpr <= EXPECTED_FPR_CAP + TIE_TOLERANCE
                ),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.loc[table["passes_expected_fpr_cap"]].copy()
    if eligible.empty:
        raise RuntimeError("No threshold satisfies the expected FPR cap")
    selected = eligible.sort_values(
        [
            "soft_f2",
            "soft_high_recall",
            "soft_high_false_positive_rate",
            "threshold",
        ],
        ascending=[False, False, True, False],
        kind="stable",
    ).iloc[0]
    table["selected"] = np.isclose(
        table["threshold"].to_numpy(float),
        float(selected["threshold"]),
        atol=0,
        rtol=0,
    )
    return table, {
        key: _jsonable(value)
        for key, value in selected.to_dict().items()
    }


def _select_penalty(
    candidate_oof: pd.DataFrame,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for penalty, frame in candidate_oof.groupby("penalty", sort=True):
        weights = _fold_equal_weights(
            frame["inner_fold"].to_numpy(int)
        )
        q = frame.loc[:, Q_COLUMNS].to_numpy(float)
        p = frame.loc[:, P_COLUMNS].to_numpy(float)
        rows.append(
            {
                "penalty": float(penalty),
                "inner_high_gate_soft_log_loss": _binary_soft_log_loss(
                    q[:, 2],
                    p[:, 2],
                    weights,
                ),
                "inner_full_threeclass_soft_log_loss": _full_soft_log_loss(
                    q,
                    p,
                    weights,
                ),
                "inner_high_gate_brier": float(
                    np.sum(weights * np.square(p[:, 2] - q[:, 2]))
                    / np.sum(weights)
                ),
                "fit_failures": int(frame["fit_failed"].astype(bool).sum()),
            }
        )
    metrics = pd.DataFrame(rows)
    valid = metrics.loc[metrics["fit_failures"].eq(0)].copy()
    if valid.empty:
        raise RuntimeError("Every PRESTO penalty candidate failed")
    selected = valid.sort_values(
        [
            "inner_high_gate_soft_log_loss",
            "inner_full_threeclass_soft_log_loss",
            "inner_high_gate_brier",
            "penalty",
        ],
        ascending=[True, True, True, False],
        kind="stable",
    ).iloc[0]
    selected_penalty = float(selected["penalty"])
    metrics["selected"] = np.isclose(
        metrics["penalty"].to_numpy(float),
        selected_penalty,
        atol=0,
        rtol=0,
    )
    selected_oof = candidate_oof.loc[
        np.isclose(
            candidate_oof["penalty"].to_numpy(float),
            selected_penalty,
            atol=0,
            rtol=0,
        )
    ].copy()
    return selected_penalty, metrics, selected_oof


def _inner_oof_development(
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    outer_train_ids: Sequence[str],
) -> dict[str, Any]:
    ordered = [str(value) for value in outer_train_ids]
    position = {record_id: index for index, record_id in enumerate(ordered)}
    splits = _manifest_inner_splits(
        ordered,
        inner_manifest,
        repeat=int(repeat),
        fold=int(fold),
    )
    candidate_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    novelty_audits: list[dict[str, Any]] = []
    optimizer_audits: list[dict[str, Any]] = []

    for train_ids, validation_ids, inner_fold in splits:
        train_groups = (
            records_by_id.loc[train_ids, "cv_group_id"]
            .astype(str)
            .tolist()
        )
        validation_groups = set(
            records_by_id.loc[validation_ids, "cv_group_id"].astype(str)
        )
        if set(train_groups).intersection(validation_groups):
            raise AssertionError("A connected group crosses an inner fold")
        X_train_raw = records_by_id.loc[train_ids].reset_index(drop=True)
        X_validation_raw = records_by_id.loc[
            validation_ids
        ].reset_index(drop=True)
        y_train = labels_by_id.loc[train_ids, Q_COLUMNS].to_numpy(float)
        X_train, X_validation, novelty_audit = (
            _fold_local_novelty_features(
                X_train_raw,
                train_groups,
                X_validation_raw,
            )
        )
        novelty_audit.update(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "inner_fold": int(inner_fold),
                "n_train": int(len(train_ids)),
                "n_validation": int(len(validation_ids)),
            }
        )
        novelty_audits.append(novelty_audit)

        for penalty in PENALTY_GRID:
            failed = False
            failure_message = ""
            try:
                prediction, optimizer = _fit_presto(
                    X_train,
                    y_train,
                    X_validation,
                    penalty=float(penalty),
                )
            except (RuntimeError, ValueError) as error:
                failed = True
                failure_message = str(error)
                prediction = np.repeat(
                    y_train.mean(axis=0)[None, :],
                    len(validation_ids),
                    axis=0,
                )
                prediction = normalize_probabilities(prediction)
                optimizer = {
                    "success": False,
                    "fallback": "inner_training_soft_prior_for_invalid_candidate",
                    "message": failure_message,
                }
            optimizer_audits.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "inner_fold": int(inner_fold),
                    "penalty": float(penalty),
                    **_jsonable(optimizer),
                }
            )
            for index, record_id in enumerate(validation_ids):
                identity = labels_by_id.loc[record_id]
                candidate_rows.append(
                    {
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "inner_fold": int(inner_fold),
                        "record_id": str(record_id),
                        "cv_group_id": str(identity["cv_group_id"]),
                        "penalty": float(penalty),
                        "fit_failed": bool(failed),
                        "failure_message": failure_message,
                        **{
                            column: float(identity[column])
                            for column in Q_COLUMNS
                        },
                        **{
                            column: float(prediction[index, class_index])
                            for class_index, column in enumerate(P_COLUMNS)
                        },
                    }
                )

        direct_prediction, _, direct_audit = (
            fit_predict_direct_multiscale_gower(
                X_train_raw,
                y_train,
                X_validation_raw,
            )
        )
        probability_matrix_audit(direct_prediction)
        for index, record_id in enumerate(validation_ids):
            identity = labels_by_id.loc[record_id]
            direct_rows.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "inner_fold": int(inner_fold),
                    "record_id": str(record_id),
                    "cv_group_id": str(identity["cv_group_id"]),
                    **{
                        column: float(identity[column])
                        for column in Q_COLUMNS
                    },
                    **{
                        column: float(
                            direct_prediction[index, class_index]
                        )
                        for class_index, column in enumerate(P_COLUMNS)
                    },
                }
            )
        optimizer_audits.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "inner_fold": int(inner_fold),
                "model": DIRECT_MODEL,
                "audit": _jsonable(direct_audit),
            }
        )

    candidates = pd.DataFrame(candidate_rows)
    direct_oof = pd.DataFrame(direct_rows)
    for frame, name in (
        (candidates, "PRESTO candidate"),
        (direct_oof, "direct-Gower inner OOF"),
    ):
        coverage = (
            frame.loc[
                np.isclose(
                    frame["penalty"].to_numpy(float),
                    PENALTY_GRID[0],
                    atol=0,
                    rtol=0,
                )
            ]
            if "penalty" in frame.columns
            else frame
        )
        if (
            len(coverage) != len(ordered)
            or set(coverage["record_id"].astype(str)) != set(ordered)
        ):
            raise AssertionError(f"{name} does not cover outer training")
    selected_penalty, penalty_metrics, selected_oof = _select_penalty(
        candidates
    )
    presto_thresholds, presto_threshold = _threshold_table(
        selected_oof["q_high_off"].to_numpy(float),
        selected_oof["p_high_off"].to_numpy(float),
        selected_oof["inner_fold"].to_numpy(int),
        model=PRESTO_POLICY,
    )
    direct_thresholds, direct_threshold = _threshold_table(
        direct_oof["q_high_off"].to_numpy(float),
        direct_oof["p_high_off"].to_numpy(float),
        direct_oof["inner_fold"].to_numpy(int),
        model=DIRECT_POLICY,
    )
    threshold_metrics = pd.concat(
        [presto_thresholds, direct_thresholds],
        ignore_index=True,
    )
    return {
        "selected_penalty": selected_penalty,
        "presto_threshold": float(presto_threshold["threshold"]),
        "direct_threshold": float(direct_threshold["threshold"]),
        "penalty_metrics": penalty_metrics,
        "threshold_metrics": threshold_metrics,
        "candidate_oof": candidates,
        "direct_oof": direct_oof,
        "selected_presto_oof": selected_oof,
        "presto_threshold_audit": presto_threshold,
        "direct_threshold_audit": direct_threshold,
        "novelty_audits": novelty_audits,
        "optimizer_audits": optimizer_audits,
        "outer_training_position_checksum": int(
            sum((index + 1) * (position[value] + 1) for index, value in enumerate(ordered))
        ),
    }


def _matched_direct_outer(
    direct_parent_oof: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    test_ids: Sequence[str],
) -> np.ndarray:
    selected = direct_parent_oof.loc[
        direct_parent_oof["repeat"].eq(int(repeat))
        & direct_parent_oof["fold"].eq(int(fold))
    ].copy()
    selected = selected.set_index("record_id")
    ordered = [str(value) for value in test_ids]
    if set(selected.index.astype(str)) != set(ordered):
        raise ValueError("Direct-Gower outer comparator does not match test IDs")
    prediction = selected.loc[ordered, P_COLUMNS].to_numpy(float)
    probability_matrix_audit(prediction)
    return prediction


def _hybrid_probabilities(
    presto: np.ndarray,
    direct: np.ndarray,
) -> np.ndarray:
    presto_probability = np.asarray(presto, dtype=float)
    direct_probability = np.asarray(direct, dtype=float)
    if presto_probability.shape != direct_probability.shape:
        raise ValueError("PRESTO and direct probabilities do not align")
    non_high = direct_probability[:, :2].sum(axis=1)
    conditional = np.divide(
        direct_probability[:, :2],
        non_high[:, None],
        out=np.full((len(direct_probability), 2), 0.5, dtype=float),
        where=non_high[:, None] > 0,
    )
    high = presto_probability[:, 2]
    hybrid = np.column_stack(
        [
            (1.0 - high) * conditional[:, 0],
            (1.0 - high) * conditional[:, 1],
            high,
        ]
    )
    probability_matrix_audit(hybrid)
    if not np.allclose(
        hybrid[:, 2],
        presto_probability[:, 2],
        atol=1.0e-15,
        rtol=0,
    ):
        raise AssertionError("Hybrid did not preserve PRESTO high probability")
    hybrid_conditional = hybrid[:, :2] / hybrid[:, :2].sum(
        axis=1,
        keepdims=True,
    )
    if not np.allclose(
        hybrid_conditional,
        conditional,
        atol=1.0e-12,
        rtol=0,
    ):
        raise AssertionError("Hybrid did not preserve Gower conditional")
    return hybrid


def _decision_indices(
    probabilities: np.ndarray,
    high_score: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float)
    score = np.asarray(high_score, dtype=float)
    high_flag = score >= float(threshold)
    decision = np.argmax(p[:, :2], axis=1)
    decision[high_flag] = 2
    return decision.astype(int), high_flag.astype(bool)


def _run_outer_split(
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    direct_parent_oof: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    test_ids: Sequence[str],
) -> dict[str, Any]:
    ordered_ids = records_by_id.index.astype(str).tolist()
    test = [str(value) for value in test_ids]
    test_set = set(test)
    train = [value for value in ordered_ids if value not in test_set]
    if len(train) + len(test) != len(ordered_ids) or set(train) & test_set:
        raise AssertionError("PRESTO outer partition is invalid")
    train_groups = set(
        records_by_id.loc[train, "cv_group_id"].astype(str)
    )
    test_groups = set(
        records_by_id.loc[test, "cv_group_id"].astype(str)
    )
    if train_groups & test_groups:
        raise AssertionError("A connected group crosses an outer fold")

    development = _inner_oof_development(
        records_by_id,
        labels_by_id,
        inner_manifest,
        repeat=int(repeat),
        fold=int(fold),
        outer_train_ids=train,
    )
    X_train_raw = records_by_id.loc[train].reset_index(drop=True)
    X_test_raw = records_by_id.loc[test].reset_index(drop=True)
    y_train = labels_by_id.loc[train, Q_COLUMNS].to_numpy(float)
    X_train, X_test, novelty_audit = _fold_local_novelty_features(
        X_train_raw,
        records_by_id.loc[train, "cv_group_id"].astype(str).tolist(),
        X_test_raw,
    )
    presto, final_optimizer = _fit_presto(
        X_train,
        y_train,
        X_test,
        penalty=float(development["selected_penalty"]),
    )
    direct = _matched_direct_outer(
        direct_parent_oof,
        repeat=int(repeat),
        fold=int(fold),
        test_ids=test,
    )
    hybrid = _hybrid_probabilities(presto, direct)
    prior = normalize_probabilities(
        np.repeat(y_train.mean(axis=0)[None, :], len(test), axis=0)
    )

    presto_decision, presto_flag = _decision_indices(
        hybrid,
        hybrid[:, 2],
        float(development["presto_threshold"]),
    )
    direct_decision, direct_flag = _decision_indices(
        direct,
        direct[:, 2],
        float(development["direct_threshold"]),
    )

    probability_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for index, record_id in enumerate(test):
        identity = labels_by_id.loc[record_id]
        common = {
            "repeat": int(repeat),
            "fold": int(fold),
            "record_id": str(record_id),
            "cv_group_id": str(identity["cv_group_id"]),
            **{
                column: float(identity[column])
                for column in Q_COLUMNS
            },
        }
        for model, prediction in (
            (PRESTO_FULL_MODEL, presto),
            (PRESTO_HYBRID_MODEL, hybrid),
            (DIRECT_MODEL, direct),
            (PRIOR_MODEL, prior),
        ):
            probability_rows.append(
                {
                    **common,
                    "evaluation_model": str(model),
                    **{
                        column: float(prediction[index, class_index])
                        for class_index, column in enumerate(P_COLUMNS)
                    },
                    "outer_test_used_for_selection": False,
                }
            )
        for policy, base_model, decision, flag, threshold, score in (
            (
                PRESTO_POLICY,
                PRESTO_HYBRID_MODEL,
                presto_decision,
                presto_flag,
                development["presto_threshold"],
                hybrid[:, 2],
            ),
            (
                DIRECT_POLICY,
                DIRECT_MODEL,
                direct_decision,
                direct_flag,
                development["direct_threshold"],
                direct[:, 2],
            ),
        ):
            policy_rows.append(
                {
                    **common,
                    "decision_policy": str(policy),
                    "probability_model": str(base_model),
                    **{
                        column: float(
                            (hybrid if policy == PRESTO_POLICY else direct)[
                                index, class_index
                            ]
                        )
                        for class_index, column in enumerate(P_COLUMNS)
                    },
                    "selected_high_threshold": float(threshold),
                    "high_score": float(score[index]),
                    "predicted_class_index": int(decision[index]),
                    "predicted_class": str(
                        CLASS_LABELS[int(decision[index])]
                    ),
                    "predicted_high": bool(flag[index]),
                    "outer_test_used_for_selection": False,
                }
            )

    selection_row = {
        "repeat": int(repeat),
        "fold": int(fold),
        "selected_penalty": float(development["selected_penalty"]),
        "selected_presto_threshold": float(
            development["presto_threshold"]
        ),
        "selected_direct_threshold": float(
            development["direct_threshold"]
        ),
        "presto_inner_threshold_audit_json": json.dumps(
            _jsonable(development["presto_threshold_audit"]),
            sort_keys=True,
        ),
        "direct_inner_threshold_audit_json": json.dumps(
            _jsonable(development["direct_threshold_audit"]),
            sort_keys=True,
        ),
    }
    audit = {
        "repeat": int(repeat),
        "fold": int(fold),
        "train_record_ids": train,
        "test_record_ids": test,
        "selected_penalty": float(development["selected_penalty"]),
        "selected_presto_threshold": float(
            development["presto_threshold"]
        ),
        "selected_direct_threshold": float(
            development["direct_threshold"]
        ),
        "outer_novelty": novelty_audit,
        "final_optimizer": final_optimizer,
        "inner_novelty": development["novelty_audits"],
        "inner_optimizers": development["optimizer_audits"],
        "presto_threshold_audit": development[
            "presto_threshold_audit"
        ],
        "direct_threshold_audit": development[
            "direct_threshold_audit"
        ],
    }
    return {
        "probability_rows": pd.DataFrame(probability_rows),
        "policy_rows": pd.DataFrame(policy_rows),
        "penalty_metrics": development["penalty_metrics"].assign(
            repeat=int(repeat),
            fold=int(fold),
        ),
        "threshold_metrics": development["threshold_metrics"].assign(
            repeat=int(repeat),
            fold=int(fold),
        ),
        "candidate_oof": development["candidate_oof"],
        "direct_inner_oof": development["direct_oof"],
        "selection_row": pd.DataFrame([selection_row]),
        "audit": audit,
    }


def _decision_metrics(
    q: np.ndarray,
    decision: np.ndarray,
) -> dict[str, float | int]:
    target = np.asarray(q, dtype=float)
    predicted = np.asarray(decision, dtype=int)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("Decision metrics require three soft target columns")
    if predicted.shape != (len(target),) or np.any(
        (predicted < 0) | (predicted > 2)
    ):
        raise ValueError("Decision indices are invalid")
    target_modal = np.argmax(target, axis=1)
    high_target = target_modal == 2
    high_prediction = predicted == 2
    hard_tp = int(np.sum(high_target & high_prediction))
    hard_fn = int(np.sum(high_target & (~high_prediction)))
    hard_fp = int(np.sum((~high_target) & high_prediction))
    hard_tn = int(np.sum((~high_target) & (~high_prediction)))
    hard_high_recall = (
        hard_tp / (hard_tp + hard_fn)
        if hard_tp + hard_fn > 0
        else np.nan
    )
    hard_high_precision = (
        hard_tp / (hard_tp + hard_fp)
        if hard_tp + hard_fp > 0
        else 0.0
    )
    hard_high_fpr = (
        hard_fp / (hard_fp + hard_tn)
        if hard_fp + hard_tn > 0
        else np.nan
    )

    q_high = target[:, 2]
    soft_tp = float(np.sum(q_high * high_prediction))
    soft_fn = float(np.sum(q_high * (~high_prediction)))
    soft_fp = float(np.sum((1.0 - q_high) * high_prediction))
    soft_tn = float(np.sum((1.0 - q_high) * (~high_prediction)))
    soft_recall = soft_tp / (soft_tp + soft_fn)
    soft_precision = (
        soft_tp / (soft_tp + soft_fp)
        if soft_tp + soft_fp > 0
        else 0.0
    )
    soft_fpr = soft_fp / (soft_fp + soft_tn)
    denominator = 4.0 * soft_precision + soft_recall
    soft_f2 = (
        5.0 * soft_precision * soft_recall / denominator
        if denominator > 0
        else 0.0
    )
    positions = np.arange(len(target))
    matrix = confusion_matrix(
        target_modal,
        predicted,
        labels=[0, 1, 2],
    )
    return {
        "n_records": int(len(target)),
        "modal_accuracy": float(np.mean(predicted == target_modal)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target_modal, predicted)
        ),
        "macro_f1": float(
            f1_score(
                target_modal,
                predicted,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "posterior_expected_hard_accuracy": float(
            np.mean(target[positions, predicted])
        ),
        "predicted_high_count": int(high_prediction.sum()),
        "predicted_high_fraction": float(np.mean(high_prediction)),
        "modal_high_true_positives": hard_tp,
        "modal_high_false_negatives": hard_fn,
        "modal_high_false_positives": hard_fp,
        "modal_high_true_negatives": hard_tn,
        "modal_high_recall": float(hard_high_recall),
        "modal_high_precision": float(hard_high_precision),
        "modal_high_false_positive_rate": float(hard_high_fpr),
        "posterior_expected_high_recall": float(soft_recall),
        "posterior_expected_high_precision": float(soft_precision),
        "posterior_expected_high_false_positive_rate": float(soft_fpr),
        "posterior_expected_high_f2": float(soft_f2),
        "confusion_low_low": int(matrix[0, 0]),
        "confusion_low_120": int(matrix[0, 1]),
        "confusion_low_high": int(matrix[0, 2]),
        "confusion_120_low": int(matrix[1, 0]),
        "confusion_120_120": int(matrix[1, 1]),
        "confusion_120_high": int(matrix[1, 2]),
        "confusion_high_low": int(matrix[2, 0]),
        "confusion_high_120": int(matrix[2, 1]),
        "confusion_high_high": int(matrix[2, 2]),
    }


def _policy_tables(
    policy_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    split_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for (policy, repeat, fold), frame in policy_rows.groupby(
        ["decision_policy", "repeat", "fold"],
        sort=True,
    ):
        split_rows.append(
            {
                "decision_policy": str(policy),
                "repeat": int(repeat),
                "fold": int(fold),
                **_decision_metrics(
                    frame.loc[:, Q_COLUMNS].to_numpy(float),
                    frame["predicted_class_index"].to_numpy(int),
                ),
            }
        )
    for (policy, repeat), frame in policy_rows.groupby(
        ["decision_policy", "repeat"],
        sort=True,
    ):
        if (
            len(frame) != 97
            or frame["record_id"].astype(str).nunique() != 97
        ):
            raise AssertionError("A policy repeat does not cover 97 records")
        repeat_rows.append(
            {
                "decision_policy": str(policy),
                "repeat": int(repeat),
                **_decision_metrics(
                    frame.loc[:, Q_COLUMNS].to_numpy(float),
                    frame["predicted_class_index"].to_numpy(int),
                ),
            }
        )

    aggregation = (
        policy_rows.groupby(
            [
                "decision_policy",
                "probability_model",
                "record_id",
                "cv_group_id",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            {
                **{column: "mean" for column in Q_COLUMNS},
                **{column: "mean" for column in P_COLUMNS},
                "predicted_high": "mean",
                "selected_high_threshold": "mean",
                "high_score": "mean",
            }
        )
        .rename(columns={"predicted_high": "predicted_high_vote_fraction"})
    )
    aggregate_rows: list[dict[str, Any]] = []
    aggregated_frames: list[pd.DataFrame] = []
    for policy, frame in aggregation.groupby("decision_policy", sort=True):
        output = frame.copy()
        high = (
            output["predicted_high_vote_fraction"].to_numpy(float) >= 0.5
        )
        decision = np.argmax(
            output.loc[:, P_COLUMNS[:2]].to_numpy(float),
            axis=1,
        )
        decision[high] = 2
        output["predicted_class_index"] = decision
        output["predicted_class"] = [
            CLASS_LABELS[int(value)] for value in decision
        ]
        output["predicted_high"] = high
        aggregated_frames.append(output)
        aggregate_rows.append(
            {
                "decision_policy": str(policy),
                "record_aggregation_rule": (
                    "high_if_at_least_half_of_repeat_votes_else_"
                    "argmax_mean_non_high_probabilities"
                ),
                **_decision_metrics(
                    output.loc[:, Q_COLUMNS].to_numpy(float),
                    decision,
                ),
            }
        )
    aggregated = pd.concat(aggregated_frames, ignore_index=True)
    return aggregated, {
        "policy_aggregate_metrics": pd.DataFrame(aggregate_rows),
        "policy_repeat_metrics": pd.DataFrame(repeat_rows),
        "policy_split_metrics": pd.DataFrame(split_rows),
    }


def _probability_argmax_metrics(
    aggregated: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, frame in aggregated.groupby("evaluation_model", sort=True):
        probability = frame.loc[:, P_COLUMNS].to_numpy(float)
        rows.append(
            {
                "evaluation_model": str(model),
                "decision_rule": "argmax_mean_repeated_oof_probability",
                **_decision_metrics(
                    frame.loc[:, Q_COLUMNS].to_numpy(float),
                    np.argmax(probability, axis=1),
                ),
            }
        )
    return pd.DataFrame(rows)


def _probability_contrast(
    aggregated: pd.DataFrame,
    *,
    model: str,
    comparator: str,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    focal = aggregated.loc[
        aggregated["evaluation_model"].eq(str(model))
    ].copy()
    baseline = aggregated.loc[
        aggregated["evaluation_model"].eq(str(comparator))
    ].copy()
    merged = focal.merge(
        baseline,
        on=["record_id", "cv_group_id"],
        suffixes=("_model", "_comparator"),
        validate="one_to_one",
    )
    q = merged.loc[
        :, [f"{column}_model" for column in Q_COLUMNS]
    ].to_numpy(float)
    p_model = merged.loc[
        :, [f"{column}_model" for column in P_COLUMNS]
    ].to_numpy(float)
    p_comparator = merged.loc[
        :, [f"{column}_comparator" for column in P_COLUMNS]
    ].to_numpy(float)
    model_loss = -np.sum(
        q * np.log(np.clip(p_model, 1.0e-15, 1.0)),
        axis=1,
    )
    comparator_loss = -np.sum(
        q * np.log(np.clip(p_comparator, 1.0e-15, 1.0)),
        axis=1,
    )
    model_brier = np.sum(np.square(p_model - q), axis=1)
    comparator_brier = np.sum(np.square(p_comparator - q), axis=1)
    q_high = q[:, 2]
    model_high_loss = -(
        q_high * np.log(np.clip(p_model[:, 2], 1.0e-15, 1.0))
        + (1.0 - q_high)
        * np.log(np.clip(1.0 - p_model[:, 2], 1.0e-15, 1.0))
    )
    comparator_high_loss = -(
        q_high
        * np.log(np.clip(p_comparator[:, 2], 1.0e-15, 1.0))
        + (1.0 - q_high)
        * np.log(
            np.clip(1.0 - p_comparator[:, 2], 1.0e-15, 1.0)
        )
    )
    contrast = f"{model}_minus_{comparator}"
    scores = pd.DataFrame(
        {
            "contrast": contrast,
            "record_id": merged["record_id"].astype(str),
            "cv_group_id": merged["cv_group_id"].astype(str),
            "delta_soft_log_loss": model_loss - comparator_loss,
            "delta_multiclass_brier": model_brier - comparator_brier,
            "delta_high_gate_soft_log_loss": (
                model_high_loss - comparator_high_loss
            ),
            "delta_high_probability_squared_error": (
                np.square(p_model[:, 2] - q_high)
                - np.square(p_comparator[:, 2] - q_high)
            ),
        }
    )
    draws, summary = paired_connected_group_bootstrap(
        scores,
        [
            "delta_soft_log_loss",
            "delta_multiclass_brier",
            "delta_high_gate_soft_log_loss",
            "delta_high_probability_squared_error",
        ],
        replicates=int(repetitions),
        seed=BOOTSTRAP_SEED,
    )
    draws.insert(0, "contrast", contrast)
    summary.insert(0, "contrast", contrast)
    return scores, draws, summary


def _policy_group_bootstrap(
    aggregated_policy: pd.DataFrame,
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    presto = aggregated_policy.loc[
        aggregated_policy["decision_policy"].eq(PRESTO_POLICY)
    ].copy()
    direct = aggregated_policy.loc[
        aggregated_policy["decision_policy"].eq(DIRECT_POLICY)
    ].copy()
    merged = presto.merge(
        direct,
        on=["record_id", "cv_group_id"],
        suffixes=("_presto", "_direct"),
        validate="one_to_one",
    )
    q = merged.loc[
        :, [f"{column}_presto" for column in Q_COLUMNS]
    ].to_numpy(float)
    presto_decision = merged["predicted_class_index_presto"].to_numpy(int)
    direct_decision = merged["predicted_class_index_direct"].to_numpy(int)
    group_values = merged["cv_group_id"].astype(str).to_numpy()
    groups = np.asarray(sorted(set(group_values)), dtype=object)
    if len(groups) != 80:
        raise ValueError("Policy bootstrap requires 80 connected groups")

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
    observed_presto = _decision_metrics(q, presto_decision)
    observed_direct = _decision_metrics(q, direct_decision)
    observed = {
        name: float(observed_presto[name]) - float(observed_direct[name])
        for name in metric_names
    }

    lookup = {
        group: np.flatnonzero(group_values == group)
        for group in groups
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw_rows: list[dict[str, Any]] = []
    for replicate in range(int(repetitions)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([lookup[str(group)] for group in sampled])
        presto_metrics = _decision_metrics(
            q[indices],
            presto_decision[indices],
        )
        direct_metrics = _decision_metrics(
            q[indices],
            direct_decision[indices],
        )
        draw_rows.append(
            {
                "replicate": int(replicate),
                **{
                    f"delta_{name}": (
                        float(presto_metrics[name])
                        - float(direct_metrics[name])
                    )
                    for name in metric_names
                },
            }
        )
    draws = pd.DataFrame(draw_rows)
    summary_rows: list[dict[str, Any]] = []
    for name in metric_names:
        column = f"delta_{name}"
        values = draws[column].to_numpy(float)
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "contrast": f"{PRESTO_POLICY}_minus_{DIRECT_POLICY}",
                "metric": column,
                "observed_delta": observed[name],
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
            }
        )
    return draws, pd.DataFrame(summary_rows)


def _copy_bound_inputs(
    destination: Path,
    parent: Path,
    contract_path: Path,
) -> None:
    copies = {
        parent / "inputs" / "records.csv": (
            destination / "inputs" / "records.csv"
        ),
        parent / "registry" / "contiguous_three_class_soft_labels.csv": (
            destination
            / "registry"
            / "contiguous_three_class_soft_labels.csv"
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
        parent / "registry" / "contract_snapshot.yaml": (
            destination / "inputs" / "parent_contract_snapshot.yaml"
        ),
        contract_path: destination / "registry" / "contract_snapshot.yaml",
    }
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_code_manifest(
    destination: Path,
    contract_path: Path,
) -> None:
    paths = [
        Path(__file__).resolve(),
        contract_path.resolve(),
        PROJECT_ROOT / "src" / "ink_tafel_study" / "presto_ordinal.py",
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "posterior_cardinal_contiguous_threeclass_direct_gower.py"
        ),
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "posterior_cardinal_contiguous_threeclass.py"
        ),
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "threeclass_sensitivity_models.py"
        ),
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "feature_association_features.py"
        ),
        PROJECT_ROOT / "src" / "ink_tafel_study" / "features.py",
        PROJECT_ROOT / "src" / "ink_tafel_study" / "cardinal_models.py",
        PROJECT_ROOT / "src" / "ink_tafel_study" / "models.py",
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "cardinal_reduced_models.py"
        ),
        PROJECT_ROOT / "src" / "ink_tafel_study" / "cardinal_metrics.py",
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "posterior_cardinal_inference.py"
        ),
        PROJECT_ROOT / "src" / "ink_tafel_study" / "provenance.py",
    ]
    files = inventory_files(paths, relative_to=REPOSITORY_ROOT)
    write_json(
        destination / "provenance" / "code_manifest.json",
        {
            "tree_sha256": tree_sha256(files),
            "files": files,
        },
    )


def _markdown_metric_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> list[str]:
    output = frame.loc[:, list(columns)].copy()
    for column in output.columns:
        if pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].map(
                lambda value: (
                    "NA" if not np.isfinite(value) else f"{value:.6f}"
                )
            )
    header = "| " + " | ".join(map(str, output.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(output.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in output.itertuples(index=False, name=None)
    ]
    return [header, separator, *rows]


def _write_report(
    destination: Path,
    *,
    aggregate_metrics: pd.DataFrame,
    argmax_metrics: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    probability_bootstrap: pd.DataFrame,
    policy_bootstrap: pd.DataFrame,
    selections: pd.DataFrame,
    qa: Mapping[str, Any],
) -> Path:
    probability_columns = (
        "evaluation_model",
        "soft_log_loss",
        "multiclass_brier",
        "modal_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "posterior_expected_hard_accuracy",
        "gate_soft_log_loss",
    )
    policy_columns = (
        "decision_policy",
        "modal_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "posterior_expected_hard_accuracy",
        "predicted_high_count",
        "modal_high_recall",
        "modal_high_precision",
        "modal_high_false_positive_rate",
        "posterior_expected_high_recall",
        "posterior_expected_high_precision",
    )
    probability_display = aggregate_metrics.merge(
        argmax_metrics.loc[
            :,
            [
                "evaluation_model",
                "modal_high_recall",
                "modal_high_precision",
                "predicted_high_count",
            ],
        ],
        on="evaluation_model",
        validate="one_to_one",
    )
    probability_columns = (
        *probability_columns,
        "predicted_high_count",
        "modal_high_recall",
        "modal_high_precision",
    )

    hybrid_bootstrap = probability_bootstrap.loc[
        probability_bootstrap["contrast"].eq(
            f"{PRESTO_HYBRID_MODEL}_minus_{DIRECT_MODEL}"
        )
    ]
    high_recall_bootstrap = policy_bootstrap.loc[
        policy_bootstrap["metric"].eq("delta_modal_high_recall")
    ]
    lines = [
        "# PRESTO-inspired high-off gate development sensitivity",
        "",
        "## Outcome",
        "",
        (
            "This package evaluates a probability-valid PRESTO-inspired "
            "fused continuation-ratio gate and a separately tuned "
            "recall-oriented decision policy. Probability quality and "
            "high-recall decisions are not conflated."
        ),
        "",
        *(_markdown_metric_table(
            probability_display,
            probability_columns,
        )),
        "",
        "## Recall-oriented policy",
        "",
        (
            "Each policy threshold was selected using only grouped inner-OOF "
            "soft labels inside its outer-training split. PRESTO and the "
            "direct-Gower comparator received the identical soft-F2 utility "
            f"and expected-FPR <= {EXPECTED_FPR_CAP:.2f} constraint."
        ),
        "",
        *(_markdown_metric_table(policy_metrics, policy_columns)),
        "",
        "## Paired uncertainty",
        "",
    ]
    if not hybrid_bootstrap.empty:
        row = hybrid_bootstrap.loc[
            hybrid_bootstrap["metric"].eq("delta_soft_log_loss")
        ].iloc[0]
        lines.append(
            "PRESTO-hybrid minus direct-Gower soft-log-loss delta: "
            f"{float(row['observed_mean']):.6f}, connected-group bootstrap "
            f"95% CI [{float(row['ci95_low']):.6f}, "
            f"{float(row['ci95_high']):.6f}]."
        )
    if not high_recall_bootstrap.empty:
        row = high_recall_bootstrap.iloc[0]
        lines.extend(
            [
                "",
                (
                    "PRESTO-policy minus direct-policy modal high-recall "
                    f"delta: {float(row['observed_delta']):.6f}, "
                    "connected-group bootstrap 95% CI "
                    f"[{float(row['ci95_low']):.6f}, "
                    f"{float(row['ci95_high']):.6f}]."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Model definition",
            "",
            (
                "The model uses one fold-local X-only Gower novelty score "
                f"from panel `{PANEL}` and the mean distance to "
                f"{NOVELTY_NEIGHBORS} nearest records outside the same "
                "connected group. It jointly models advancement from low to "
                "120/high and from 120/high to high, shrinking adjacent "
                "transition slopes together."
            ),
            "",
            (
                "This is not the paper's exact cumulative-odds PRESTO. "
                "Declared deviations are the continuation-ratio link, "
                "smooth-L1 penalty, posterior-soft-label extension, and a "
                "small fixed ridge stabilizer. The parameterization was "
                "chosen because it cannot produce negative middle-class "
                "probabilities."
            ),
            "",
            "Penalty and threshold selections by outer split:",
            "",
            *(_markdown_metric_table(
                selections,
                (
                    "repeat",
                    "fold",
                    "selected_penalty",
                    "selected_presto_threshold",
                    "selected_direct_threshold",
                ),
            )),
            "",
            "## Interpretation boundary",
            "",
            (
                "The architecture, novelty coordinate, candidate grid, and "
                "decision utility were developed after inspecting results on "
                "the same closed 97-record dataset. The current nested folds "
                "protect outer-test rows from within-run tuning, but they do "
                "not account for this global development adaptivity. The "
                "result is post-selection development evidence, not "
                "independent confirmation."
            ),
            "",
            (
                "A recall-policy gain is a cost-sensitive operating-point "
                "change. It must be read together with false positives, "
                "precision, overall accuracy, and probability scores. It is "
                "not evidence that formulation features cause high slopes."
            ),
            "",
            "Primary method sources:",
            "",
            (
                "- Faletto and Bien (2023), PRESTO: "
                "https://proceedings.mlr.press/v202/faletto23a.html"
            ),
            (
                "- Author implementation: "
                "https://github.com/gregfaletto/presto"
            ),
            "",
            (
                f"Artifact QA status: `{qa['status']}`; scientifically "
                f"releasable: `{str(qa['scientifically_releasable']).lower()}`."
            ),
            "",
        ]
    )
    report = (
        destination
        / "report"
        / "posterior_cardinal_contiguous_threeclass_presto_gate_report.md"
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_presto_gate_sensitivity(
    destination: str | Path,
    *,
    parent_run: str | Path | None = None,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    n_jobs: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the isolated PRESTO-inspired gate development sensitivity."""

    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    contract, resolved_contract, contract_hash = load_presto_gate_contract(
        contract_path
    )
    parent = _resolve_parent(contract, parent_run)
    output = _resolve_path(destination, base=Path.cwd())
    _assert_fresh_disjoint_destination(output, parent)
    parent_before = _verify_parent(parent, contract)
    records, labels, outer, inner, direct_parent = _load_parent_tables(parent)
    records_by_id = _indexed(records, name="records")
    labels_with_group = labels.drop(
        columns=["cv_group_id"],
        errors="ignore",
    ).merge(
        records.loc[:, ["record_id", "cv_group_id"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    labels_by_id = _indexed(labels_with_group, name="labels")

    selected_manifest = outer.copy()
    if smoke:
        first_repeat = int(selected_manifest["repeat"].min())
        selected_manifest = selected_manifest.loc[
            selected_manifest["repeat"].eq(first_repeat)
        ].copy()
    split_groups = [
        (
            int(repeat),
            int(fold),
            frame["record_id"].astype(str).tolist(),
        )
        for (repeat, fold), frame in selected_manifest.groupby(
            ["repeat", "fold"],
            sort=True,
        )
    ]
    results = Parallel(
        n_jobs=int(n_jobs),
        prefer="threads",
    )(
        delayed(_run_outer_split)(
            records_by_id,
            labels_by_id,
            inner,
            direct_parent,
            repeat=repeat,
            fold=fold,
            test_ids=test_ids,
        )
        for repeat, fold, test_ids in split_groups
    )
    probability_oof = pd.concat(
        [result["probability_rows"] for result in results],
        ignore_index=True,
    )
    policy_oof = pd.concat(
        [result["policy_rows"] for result in results],
        ignore_index=True,
    )
    penalty_metrics = pd.concat(
        [result["penalty_metrics"] for result in results],
        ignore_index=True,
    )
    threshold_metrics = pd.concat(
        [result["threshold_metrics"] for result in results],
        ignore_index=True,
    )
    inner_candidates = pd.concat(
        [result["candidate_oof"] for result in results],
        ignore_index=True,
    )
    direct_inner_oof = pd.concat(
        [result["direct_inner_oof"] for result in results],
        ignore_index=True,
    )
    selections = pd.concat(
        [result["selection_row"] for result in results],
        ignore_index=True,
    ).sort_values(["repeat", "fold"], kind="stable")
    outer_audits = [result["audit"] for result in results]

    expected_test_records = sum(
        len(test_ids) for _, _, test_ids in split_groups
    )
    if len(probability_oof) != expected_test_records * 4:
        raise AssertionError("PRESTO probability OOF row count changed")
    if len(policy_oof) != expected_test_records * 2:
        raise AssertionError("PRESTO policy OOF row count changed")
    if probability_oof.duplicated(
        ["evaluation_model", "repeat", "fold", "record_id"]
    ).any():
        raise AssertionError("PRESTO probability OOF keys are not unique")
    if policy_oof.duplicated(
        ["decision_policy", "repeat", "fold", "record_id"]
    ).any():
        raise AssertionError("PRESTO policy OOF keys are not unique")
    if (
        probability_oof["outer_test_used_for_selection"].astype(bool).any()
        or policy_oof["outer_test_used_for_selection"].astype(bool).any()
    ):
        raise AssertionError("Outer-test selection flag became true")
    probability_matrix_audit(
        probability_oof.loc[:, P_COLUMNS].to_numpy(float)
    )

    key = ["repeat", "fold", "record_id"]
    full = probability_oof.loc[
        probability_oof["evaluation_model"].eq(PRESTO_FULL_MODEL),
        [*key, *P_COLUMNS],
    ]
    hybrid = probability_oof.loc[
        probability_oof["evaluation_model"].eq(PRESTO_HYBRID_MODEL),
        [*key, *P_COLUMNS],
    ]
    direct = probability_oof.loc[
        probability_oof["evaluation_model"].eq(DIRECT_MODEL),
        [*key, *P_COLUMNS],
    ]
    high_check = full.merge(
        hybrid,
        on=key,
        suffixes=("_full", "_hybrid"),
        validate="one_to_one",
    )
    maximum_high_identity_error = float(
        np.max(
            np.abs(
                high_check["p_high_off_full"].to_numpy(float)
                - high_check["p_high_off_hybrid"].to_numpy(float)
            )
        )
    )
    conditional_check = hybrid.merge(
        direct,
        on=key,
        suffixes=("_hybrid", "_direct"),
        validate="one_to_one",
    )
    hybrid_non_high = conditional_check[
        ["p_low_60_side_hybrid", "p_120_side_hybrid"]
    ].to_numpy(dtype=float, copy=True)
    hybrid_non_high /= hybrid_non_high.sum(axis=1, keepdims=True)
    direct_non_high = conditional_check[
        ["p_low_60_side_direct", "p_120_side_direct"]
    ].to_numpy(dtype=float, copy=True)
    direct_non_high /= direct_non_high.sum(axis=1, keepdims=True)
    maximum_conditional_identity_error = float(
        np.max(np.abs(hybrid_non_high - direct_non_high))
    )
    if (
        maximum_high_identity_error > 1.0e-12
        or maximum_conditional_identity_error > 1.0e-12
    ):
        raise AssertionError("PRESTO hybrid identity audit failed")

    aggregated_probability = aggregate_contiguous_oof(probability_oof)
    probability_tables = _metric_tables(
        probability_oof,
        aggregated_probability,
        entropy_threshold=ENTROPY_THRESHOLD,
    )
    probability_argmax = _probability_argmax_metrics(
        aggregated_probability
    )
    aggregated_policy, policy_tables = _policy_tables(policy_oof)

    repetitions = (
        SMOKE_BOOTSTRAP_REPETITIONS
        if smoke
        else BOOTSTRAP_REPETITIONS
    )
    score_frames: list[pd.DataFrame] = []
    probability_draw_frames: list[pd.DataFrame] = []
    probability_summary_frames: list[pd.DataFrame] = []
    for model in (PRESTO_FULL_MODEL, PRESTO_HYBRID_MODEL):
        scores, draws, summary = _probability_contrast(
            aggregated_probability,
            model=model,
            comparator=DIRECT_MODEL,
            repetitions=repetitions,
        )
        score_frames.append(scores)
        probability_draw_frames.append(draws)
        probability_summary_frames.append(summary)
    probability_scores = pd.concat(score_frames, ignore_index=True)
    probability_bootstrap_draws = pd.concat(
        probability_draw_frames,
        ignore_index=True,
    )
    probability_bootstrap_summary = pd.concat(
        probability_summary_frames,
        ignore_index=True,
    )
    policy_bootstrap_draws, policy_bootstrap_summary = (
        _policy_group_bootstrap(
            aggregated_policy,
            repetitions=repetitions,
        )
    )

    parent_after = _verify_parent(parent, contract)
    parent_tree_unchanged = (
        parent_before["artifact_manifest_verification"][
            "actual_tree_sha256"
        ]
        == parent_after["artifact_manifest_verification"][
            "actual_tree_sha256"
        ]
    )
    if not parent_tree_unchanged:
        raise AssertionError("PRESTO parent changed during modeling")

    output.mkdir(parents=True)
    for directory in (
        "inputs",
        "registry",
        "folds",
        "analyses",
        "provenance",
        "report",
    ):
        (output / directory).mkdir()
    _copy_bound_inputs(output, parent, resolved_contract)
    tables_to_write = {
        "oof_probabilities": probability_oof,
        "oof_aggregated": aggregated_probability,
        "policy_oof_decisions": policy_oof,
        "policy_record_aggregated": aggregated_policy,
        "penalty_inner_oof_metrics": penalty_metrics,
        "threshold_inner_oof_metrics": threshold_metrics,
        "presto_inner_oof_candidates": inner_candidates,
        "direct_gower_inner_oof": direct_inner_oof,
        "outer_selections": selections,
        "probability_argmax_metrics": probability_argmax,
        "probability_record_level_contrasts": probability_scores,
        "probability_connected_group_bootstrap": (
            probability_bootstrap_draws
        ),
        "probability_connected_group_bootstrap_summary": (
            probability_bootstrap_summary
        ),
        "policy_connected_group_bootstrap": policy_bootstrap_draws,
        "policy_connected_group_bootstrap_summary": (
            policy_bootstrap_summary
        ),
        **probability_tables,
        **policy_tables,
    }
    for name, frame in tables_to_write.items():
        frame.to_csv(output / "analyses" / f"{name}.csv", index=False)
    write_json(
        output / "analyses" / "outer_fit_audits.json",
        _jsonable(outer_audits),
    )

    probability_metrics = probability_tables["aggregate_metrics"]
    policy_metrics = policy_tables["policy_aggregate_metrics"]
    hybrid_metric = probability_metrics.loc[
        probability_metrics["evaluation_model"].eq(PRESTO_HYBRID_MODEL)
    ].iloc[0]
    direct_metric = probability_metrics.loc[
        probability_metrics["evaluation_model"].eq(DIRECT_MODEL)
    ].iloc[0]
    presto_policy_metric = policy_metrics.loc[
        policy_metrics["decision_policy"].eq(PRESTO_POLICY)
    ].iloc[0]
    direct_policy_metric = policy_metrics.loc[
        policy_metrics["decision_policy"].eq(DIRECT_POLICY)
    ].iloc[0]
    raw_presto_argmax = probability_argmax.loc[
        probability_argmax["evaluation_model"].eq(PRESTO_HYBRID_MODEL)
    ].iloc[0]
    qa = {
        "status": (
            "complete_smoke"
            if smoke
            else "complete_posthoc_adaptive_development"
        ),
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_hash,
        "analysis_role": "posthoc_adaptive_prediction_development_sensitivity",
        "scientifically_releasable": False,
        "independent_confirmation": False,
        "may_replace_or_rescue_v4": False,
        "may_replace_contiguous_v2": False,
        "may_replace_direct_gower_v1": False,
        "n_records_in_registry": int(len(records)),
        "n_connected_groups_in_registry": int(
            records["cv_group_id"].astype(str).nunique()
        ),
        "n_outer_splits_evaluated": int(len(split_groups)),
        "n_probability_oof_rows": int(len(probability_oof)),
        "n_policy_oof_rows": int(len(policy_oof)),
        "outer_test_used_for_model_penalty_or_threshold_selection": False,
        "parent_same_dataset_results_visible_during_development": True,
        "exact_presto_replication": False,
        "presto_inspired_deviations": [
            "forward_continuation_ratio_not_cumulative_odds",
            "smooth_l1_not_exact_l1",
            "posterior_soft_label_extension",
            "fixed_small_ridge_stabilizer",
            "x_only_fold_local_gower_novelty_input",
        ],
        "posterior_argmax_used_for_training_penalty_or_threshold_selection": False,
        "posterior_draws_expanded": False,
        "class_rows_expanded": False,
        "probability_and_decision_policy_reported_separately": True,
        "baseline_receives_identical_threshold_protocol": True,
        "maximum_presto_high_identity_error": maximum_high_identity_error,
        "maximum_gower_conditional_identity_error": (
            maximum_conditional_identity_error
        ),
        "probability_audit": probability_matrix_audit(
            probability_oof.loc[:, P_COLUMNS].to_numpy(float)
        ),
        "parent_manifest_verified_before": (
            parent_before["status"] == "pass"
        ),
        "parent_manifest_verified_after": (
            parent_after["status"] == "pass"
        ),
        "parent_tree_unchanged": parent_tree_unchanged,
        "presto_hybrid_soft_log_loss": float(
            hybrid_metric["soft_log_loss"]
        ),
        "direct_gower_soft_log_loss": float(
            direct_metric["soft_log_loss"]
        ),
        "presto_hybrid_modal_accuracy": float(
            hybrid_metric["modal_accuracy"]
        ),
        "direct_gower_modal_accuracy": float(
            direct_metric["modal_accuracy"]
        ),
        "presto_hybrid_argmax_modal_high_recall": float(
            raw_presto_argmax["modal_high_recall"]
        ),
        "presto_policy_modal_high_recall": float(
            presto_policy_metric["modal_high_recall"]
        ),
        "direct_policy_modal_high_recall": float(
            direct_policy_metric["modal_high_recall"]
        ),
        "presto_policy_predicted_high_count": int(
            presto_policy_metric["predicted_high_count"]
        ),
        "direct_policy_predicted_high_count": int(
            direct_policy_metric["predicted_high_count"]
        ),
        "presto_policy_modal_high_precision": float(
            presto_policy_metric["modal_high_precision"]
        ),
        "direct_policy_modal_high_precision": float(
            direct_policy_metric["modal_high_precision"]
        ),
        "presto_policy_modal_accuracy": float(
            presto_policy_metric["modal_accuracy"]
        ),
        "direct_policy_modal_accuracy": float(
            direct_policy_metric["modal_accuracy"]
        ),
    }
    write_json(output / "analyses" / "analysis_qa.json", _jsonable(qa))
    report = _write_report(
        output,
        aggregate_metrics=probability_metrics,
        argmax_metrics=probability_argmax,
        policy_metrics=policy_metrics,
        probability_bootstrap=probability_bootstrap_summary,
        policy_bootstrap=policy_bootstrap_summary,
        selections=selections,
        qa=qa,
    )
    validation = {
        "overall_assessment": (
            "structural_smoke_only"
            if smoke
            else "share_with_caveats_posthoc_development_not_formal_release"
        ),
        "methodology": (
            "presto_inspired_soft_fused_continuation_ratio_"
            "nested_grouped_oof_with_separate_recall_policy"
        ),
        "probability_claim_ready": bool(
            not smoke
            and parent_tree_unchanged
            and maximum_high_identity_error <= 1.0e-12
            and maximum_conditional_identity_error <= 1.0e-12
        ),
        "recall_policy_claim_ready_with_cost_sensitive_caveat": bool(
            not smoke and parent_tree_unchanged
        ),
        "independent_confirmation": False,
        "scientifically_releasable": False,
        "qa": qa,
        "report": str(report),
    }
    write_json(
        output / "report" / "validation_report.json",
        _jsonable(validation),
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
    write_json(
        output / "provenance" / "parent_binding.json",
        parent_before,
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
    manifest_verification = verify_artifact_manifest(output, manifest_path)
    if manifest_verification["status"] != "pass":
        raise RuntimeError(
            "PRESTO artifact manifest verification failed: "
            f"{manifest_verification}"
        )
    return {
        "status": qa["status"],
        "run_dir": str(output),
        "report": str(report),
        "contract_sha256": contract_hash,
        "presto_hybrid_soft_log_loss": qa[
            "presto_hybrid_soft_log_loss"
        ],
        "presto_hybrid_modal_accuracy": qa[
            "presto_hybrid_modal_accuracy"
        ],
        "presto_argmax_high_recall": qa[
            "presto_hybrid_argmax_modal_high_recall"
        ],
        "presto_policy_high_recall": qa[
            "presto_policy_modal_high_recall"
        ],
        "direct_policy_high_recall": qa[
            "direct_policy_modal_high_recall"
        ],
        "artifact_tree_sha256": manifest_verification[
            "actual_tree_sha256"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PRESTO-inspired rare ordered high-off gate sensitivity"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--parent-run")
    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT_PATH),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_presto_gate_sensitivity(
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
    "DIRECT_POLICY",
    "PRESTO_FULL_MODEL",
    "PRESTO_HYBRID_MODEL",
    "PRESTO_POLICY",
    "_decision_metrics",
    "_fold_local_novelty_features",
    "_hybrid_probabilities",
    "_threshold_table",
    "load_presto_gate_contract",
    "run_presto_gate_sensitivity",
]
