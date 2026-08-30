"""Direct multiscale Gower development sensitivity for contiguous classes.

This module intentionally does not modify or re-run the frozen contiguous-v2
stagewise estimator.  It reads that immutable result as a parent, reuses its
connected-group outer folds and posterior-soft labels, and evaluates one fixed
development candidate:

* fit the fold-local ``context_plus_ionomer_ratio`` feature panel;
* fit a direct three-class soft Gower model at six predeclared neighbourhood
  sizes; and
* average the six probability vectors with fixed equal weights.

The estimator never hardens posterior labels and never expands posterior draws
or class rows.  Its result is explicitly post-selection development evidence:
the parent result was already visible when this architecture was proposed, so
the output cannot be an independent confirmation or replace the frozen v4
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
import yaml

from .cardinal_metrics import normalize_probabilities
from .cardinal_models import SoftGowerKNeighborsClassifier
from .posterior_cardinal_contiguous_threeclass import (
    CLASS_LABELS,
    P_COLUMNS,
    PRIOR_MODEL,
    Q_COLUMNS,
    STACK_MODEL,
    _metric_tables,
    aggregate_contiguous_oof,
)
from .posterior_cardinal_inference import paired_connected_group_bootstrap
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


CONTRACT_ID = "posterior_cardinal_contiguous_threeclass_direct_gower_v1"
CONTRACT_FILENAME = (
    "posterior_cardinal_contiguous_threeclass_direct_gower_v1.yaml"
)
CONTRACT_SHA256 = (
    "9467f660eff0efa2e664b290037a002b826234dd869a3440fd27141ca83d973c"
)
STAGE_NAME = "posterior-cardinal-contiguous-threeclass-direct-gower"
DIRECT_MODEL = "direct_multiscale_soft_gower"
PARENT_MODEL = STACK_MODEL
PANEL = "context_plus_ionomer_ratio"
NEIGHBOR_COUNTS: tuple[int, ...] = (9, 15, 21, 25, 31, 40)
DISTANCE_POWER = 1.0
EPSILON = 1.0e-12
RARE_MIN_FREQUENCY = 3
ENTROPY_THRESHOLD = 0.90
FULL_BOOTSTRAP_REPETITIONS = 2000
SMOKE_BOOTSTRAP_REPETITIONS = 100
BOOTSTRAP_SEED = 20260725

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
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_direct_gower_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> tuple[dict[str, Any], Path, str]:
    """Load and strictly validate the frozen development contract."""

    contract_path = _resolve_path(path, base=REPOSITORY_ROOT)
    if contract_path.name != CONTRACT_FILENAME:
        raise ValueError(
            f"Direct-Gower contract must be named {CONTRACT_FILENAME}"
        )
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    digest = sha256_file(contract_path)
    if digest != CONTRACT_SHA256:
        raise ValueError(
            "Direct-Gower contract hash mismatch: "
            f"expected={CONTRACT_SHA256}, actual={digest}"
        )
    loaded = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("Direct-Gower contract must be a mapping")
    contract = dict(loaded)
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError("Direct-Gower contract ID changed")
    if contract.get("stage") != STAGE_NAME:
        raise ValueError("Direct-Gower stage name changed")
    if contract.get("status") != (
        "posthoc_development_frozen_before_packaged_run"
    ):
        raise ValueError("Direct-Gower development status changed")

    scope = dict(contract.get("scientific_scope", {}))
    if (
        scope.get("population") != "all97"
        or int(scope.get("expected_records", -1)) != 97
        or int(scope.get("expected_connected_groups", -1)) != 80
        or scope.get("one_ml_row_per_record") is not True
        or scope.get("posterior_draws_expanded_as_samples") is not False
        or scope.get("independent_confirmation") is not False
    ):
        raise ValueError("Direct-Gower scientific scope changed")

    labels = dict(contract.get("label_scheme", {}))
    if (
        tuple(labels.get("supervised_class_order", ())) != CLASS_LABELS
        or tuple(labels.get("soft_label_columns", ())) != Q_COLUMNS
        or tuple(labels.get("prediction_columns", ())) != P_COLUMNS
        or labels.get("posterior_argmax_training_or_selection_use")
        != "forbidden"
    ):
        raise ValueError("Direct-Gower soft-label contract changed")

    model = dict(contract.get("model", {}))
    gower = dict(model.get("gower", {}))
    aggregation = dict(model.get("multiscale_aggregation", {}))
    if (
        model.get("estimator_id") != DIRECT_MODEL
        or model.get("feature_panel") != PANEL
        or tuple(int(value) for value in gower.get("n_neighbors", ()))
        != NEIGHBOR_COUNTS
        or float(gower.get("distance_power", -1)) != DISTANCE_POWER
        or float(gower.get("epsilon", -1)) != EPSILON
        or gower.get("neighbor_mode") != "knn"
        or aggregation.get("rule") != "arithmetic_mean_probability_vectors"
        or dict(model.get("additional_prior_shrinkage", {})).get("enabled")
        is not False
        or dict(model.get("optuna", {})).get("enabled") is not False
    ):
        raise ValueError("Direct-Gower model contract changed")
    weights = np.asarray(aggregation.get("weights", ()), dtype=float)
    if (
        weights.shape != (len(NEIGHBOR_COUNTS),)
        or not np.allclose(
            weights,
            np.repeat(1.0 / len(NEIGHBOR_COUNTS), len(NEIGHBOR_COUNTS)),
            atol=1.0e-15,
            rtol=0,
        )
    ):
        raise ValueError("Direct-Gower multiscale weights changed")
    return contract, contract_path, digest


def _resolve_parent(
    contract: Mapping[str, Any],
    parent_run: str | Path | None,
) -> Path:
    parent = dict(contract["immutable_parent"])
    value = parent_run if parent_run is not None else parent["default_run"]
    return _resolve_path(value, base=REPOSITORY_ROOT)


def _assert_fresh_disjoint_destination(
    destination: Path,
    parent: Path,
) -> None:
    destination = destination.resolve()
    parent = parent.resolve()
    if destination == parent or parent in destination.parents:
        raise ValueError(
            "Direct-Gower output must be disjoint from the immutable parent"
        )
    if destination in parent.parents:
        raise ValueError(
            "Direct-Gower output cannot contain the immutable parent"
        )
    if destination.exists():
        raise FileExistsError(
            f"Direct-Gower output must be new and absent: {destination}"
        )


def _verify_parent(
    parent: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not parent.is_dir():
        raise FileNotFoundError(f"Immutable parent run is missing: {parent}")
    settings = dict(contract["immutable_parent"])
    verification = verify_artifact_manifest(parent)
    if verification["status"] != "pass":
        raise RuntimeError(
            "Immutable parent artifact manifest failed verification: "
            f"{verification}"
        )
    expected_tree = str(settings["expected_artifact_tree_sha256"])
    if verification["actual_tree_sha256"] != expected_tree:
        raise RuntimeError(
            "Immutable parent tree hash changed: "
            f"expected={expected_tree}, "
            f"actual={verification['actual_tree_sha256']}"
        )
    file_rows: list[dict[str, Any]] = []
    for relative, expected_hash in dict(
        settings["required_files"]
    ).items():
        path = parent / str(relative)
        if not path.is_file():
            raise FileNotFoundError(
                f"Immutable parent file is missing: {relative}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != str(expected_hash):
            raise RuntimeError(
                "Immutable parent file hash changed: "
                f"{relative}, expected={expected_hash}, actual={actual_hash}"
            )
        file_rows.append(
            {
                "path": str(relative),
                "sha256": actual_hash,
                "size_bytes": int(path.stat().st_size),
            }
        )
    return {
        "status": "pass",
        "parent_run": str(parent),
        "artifact_manifest_verification": verification,
        "required_files": file_rows,
    }


def _load_parent_tables(
    parent: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(parent / "inputs" / "records.csv")
    labels = pd.read_csv(
        parent / "registry" / "contiguous_three_class_soft_labels.csv"
    )
    outer = pd.read_csv(parent / "folds" / "outer_split_manifest.csv")
    inner = pd.read_csv(parent / "folds" / "inner_split_manifest.csv")
    parent_oof = pd.read_csv(parent / "analyses" / "oof_probabilities.csv")

    for frame in (records, labels, outer, inner, parent_oof):
        if "record_id" not in frame.columns:
            raise KeyError("A parent table is missing record_id")
        frame["record_id"] = frame["record_id"].astype(str)
    if (
        len(records) != 97
        or records["record_id"].nunique() != 97
        or len(labels) != 97
        or labels["record_id"].nunique() != 97
        or records["cv_group_id"].astype(str).nunique() != 80
    ):
        raise ValueError("Immutable parent population is not 97 records/80 groups")
    q = labels.loc[:, Q_COLUMNS].to_numpy(float)
    if (
        not np.isfinite(q).all()
        or np.any(q < 0)
        or not np.allclose(q.sum(axis=1), 1.0, atol=1.0e-12, rtol=0)
    ):
        raise ValueError("Immutable parent soft labels are invalid")
    if len(outer) != 970:
        raise ValueError("Immutable parent outer manifest must contain 970 rows")
    coverage = outer.groupby(["repeat", "record_id"]).size()
    if (
        outer["repeat"].nunique() != 10
        or outer["fold"].nunique() != 5
        or not coverage.eq(1).all()
    ):
        raise ValueError("Immutable parent outer manifest coverage changed")
    if len(inner) != 3880:
        raise ValueError("Immutable parent inner manifest must contain 3880 rows")

    parent_stack = parent_oof.loc[
        parent_oof["evaluation_model"].astype(str).eq(PARENT_MODEL)
    ].copy()
    if len(parent_stack) != 970:
        raise ValueError("Immutable parent stack must contain 970 OOF rows")
    required = {
        "repeat",
        "fold",
        "record_id",
        "cv_group_id",
        *Q_COLUMNS,
        *P_COLUMNS,
    }
    missing = sorted(required - set(parent_stack.columns))
    if missing:
        raise KeyError(f"Immutable parent OOF is missing columns: {missing}")
    parent_stack.loc[:, P_COLUMNS] = normalize_probabilities(
        parent_stack.loc[:, P_COLUMNS].to_numpy(float)
    )
    return records, labels, outer, inner, parent_stack


def fit_predict_direct_multiscale_gower(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    *,
    panel: str = PANEL,
    neighbor_counts: Sequence[int] = NEIGHBOR_COUNTS,
    rare_min_frequency: int = RARE_MIN_FREQUENCY,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    """Fit the fixed direct soft-label Gower ensemble on one training fold."""

    counts = tuple(int(value) for value in neighbor_counts)
    if not counts or len(set(counts)) != len(counts) or min(counts) < 1:
        raise ValueError("neighbor_counts must be unique positive integers")
    targets = np.asarray(y_train, dtype=float)
    if (
        targets.shape != (len(X_train), len(CLASS_LABELS))
        or not np.isfinite(targets).all()
        or np.any(targets < 0)
        or not np.allclose(targets.sum(axis=1), 1.0, atol=1.0e-12, rtol=0)
    ):
        raise ValueError("Direct-Gower training targets are invalid")

    builder, engineered_train, engineered_test, source_columns = (
        _fold_local_engineered_features(
            X_train,
            X_test,
            panel=str(panel),
            rare_min_frequency=int(rare_min_frequency),
        )
    )
    numeric = tuple(builder.numeric_columns_out_)
    categorical = tuple(builder.categorical_columns_out_)
    blocks = builder.get_blocks_out()
    component_predictions: dict[int, np.ndarray] = {}
    component_audits: dict[str, Any] = {}
    for count in counts:
        estimator = SoftGowerKNeighborsClassifier(
            numeric_columns=numeric,
            categorical_columns=categorical,
            class_names=CLASS_LABELS,
            blocks=blocks,
            block_weights=None,
            feature_weights=None,
            n_neighbors=int(count),
            distance_power=DISTANCE_POWER,
            epsilon=EPSILON,
            neighbor_mode="knn",
            bandwidth=1.0,
        ).fit(engineered_train, targets)
        prediction = estimator.predict_proba(engineered_test)
        component_predictions[int(count)] = prediction
        component_audits[str(count)] = {
            "requested_neighbors": int(count),
            "maximum_effective_neighbors_before_ties": int(
                min(count, len(engineered_train))
            ),
            "probability_audit": probability_matrix_audit(prediction),
        }
    stacked = np.stack(
        [component_predictions[count] for count in counts], axis=0
    )
    ensemble = np.mean(stacked, axis=0)
    ensemble = normalize_probabilities(ensemble)
    audit = {
        "status": "complete",
        "estimator_id": DIRECT_MODEL,
        "panel": str(panel),
        "neighbor_counts": list(counts),
        "component_weights": {
            str(count): float(1.0 / len(counts)) for count in counts
        },
        "distance_power": DISTANCE_POWER,
        "epsilon": EPSILON,
        "n_train_records": int(len(engineered_train)),
        "n_test_records": int(len(engineered_test)),
        "source_columns": list(source_columns),
        "engineered_feature_names": [
            str(value) for value in builder.get_feature_names_out()
        ],
        "numeric_features": list(numeric),
        "categorical_features": list(categorical),
        "gower_blocks": {
            str(block): [str(value) for value in features]
            for block, features in blocks.items()
        },
        "component_audits": component_audits,
        "ensemble_probability_audit": probability_matrix_audit(ensemble),
        "posterior_draws_expanded": False,
        "class_rows_expanded": False,
        "posterior_argmax_used": False,
        "outer_test_used_for_selection": False,
        "feature_engineering_fit_scope": "outer_training_records_only",
    }
    return ensemble, component_predictions, audit


def _run_outer_split(
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    test_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered_ids = records_by_id.index.astype(str).tolist()
    test = [str(value) for value in test_ids]
    test_set = set(test)
    train = [record_id for record_id in ordered_ids if record_id not in test_set]
    if set(train) & test_set or len(train) + len(test) != len(ordered_ids):
        raise AssertionError("Outer training/test partition is invalid")

    X_train = records_by_id.loc[train].reset_index(drop=True)
    X_test = records_by_id.loc[test].reset_index(drop=True)
    y_train = labels_by_id.loc[train, Q_COLUMNS].to_numpy(float)
    ensemble, components, audit = fit_predict_direct_multiscale_gower(
        X_train,
        y_train,
        X_test,
    )
    prior = np.repeat(y_train.mean(axis=0)[None, :], len(test), axis=0)
    prior = normalize_probabilities(prior)

    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for position, record_id in enumerate(test):
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
        rows.append(
            {
                **common,
                "evaluation_model": DIRECT_MODEL,
                **{
                    column: float(ensemble[position, class_index])
                    for class_index, column in enumerate(P_COLUMNS)
                },
                "outer_test_used_for_selection": False,
            }
        )
        rows.append(
            {
                **common,
                "evaluation_model": PRIOR_MODEL,
                **{
                    column: float(prior[position, class_index])
                    for class_index, column in enumerate(P_COLUMNS)
                },
                "outer_test_used_for_selection": False,
            }
        )
        for count in NEIGHBOR_COUNTS:
            prediction = components[count][position]
            component_rows.append(
                {
                    **common,
                    "evaluation_model": DIRECT_MODEL,
                    "n_neighbors": int(count),
                    "component_weight": float(
                        1.0 / len(NEIGHBOR_COUNTS)
                    ),
                    **{
                        column: float(prediction[class_index])
                        for class_index, column in enumerate(P_COLUMNS)
                    },
                }
            )
    audit.update(
        {
            "repeat": int(repeat),
            "fold": int(fold),
            "train_record_ids": train,
            "test_record_ids": test,
            "outer_training_prior": {
                label: float(prior[0, index])
                for index, label in enumerate(CLASS_LABELS)
            },
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(component_rows), audit


def _parent_comparator_rows(
    parent_stack: pd.DataFrame,
    selected_splits: set[tuple[int, int]],
) -> pd.DataFrame:
    mask = [
        (int(repeat), int(fold)) in selected_splits
        for repeat, fold in zip(
            parent_stack["repeat"],
            parent_stack["fold"],
            strict=True,
        )
    ]
    output = parent_stack.loc[mask].copy()
    output["evaluation_model"] = PARENT_MODEL
    output["outer_test_used_for_selection"] = False
    keep = [
        "repeat",
        "fold",
        "record_id",
        "cv_group_id",
        *Q_COLUMNS,
        "evaluation_model",
        *P_COLUMNS,
        "outer_test_used_for_selection",
    ]
    return output.loc[:, keep]


def _paired_contrast(
    aggregated: pd.DataFrame,
    *,
    comparator: str,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direct = aggregated.loc[
        aggregated["evaluation_model"].eq(DIRECT_MODEL)
    ].copy()
    baseline = aggregated.loc[
        aggregated["evaluation_model"].eq(str(comparator))
    ].copy()
    merged = direct.merge(
        baseline,
        on=["record_id", "cv_group_id"],
        suffixes=("_direct", "_comparator"),
        validate="one_to_one",
    )
    q = merged.loc[
        :, [f"{column}_direct" for column in Q_COLUMNS]
    ].to_numpy(float)
    p_direct = merged.loc[
        :, [f"{column}_direct" for column in P_COLUMNS]
    ].to_numpy(float)
    p_comparator = merged.loc[
        :, [f"{column}_comparator" for column in P_COLUMNS]
    ].to_numpy(float)
    direct_loss = -np.sum(
        q * np.log(np.clip(p_direct, 1.0e-15, 1.0)), axis=1
    )
    comparator_loss = -np.sum(
        q * np.log(np.clip(p_comparator, 1.0e-15, 1.0)), axis=1
    )
    direct_brier = np.sum(np.square(p_direct - q), axis=1)
    comparator_brier = np.sum(np.square(p_comparator - q), axis=1)
    target_modal = np.argmax(q, axis=1)
    direct_modal = np.argmax(p_direct, axis=1)
    comparator_modal = np.argmax(p_comparator, axis=1)
    positions = np.arange(len(q))
    contrast = f"{DIRECT_MODEL}_minus_{comparator}"
    scores = pd.DataFrame(
        {
            "contrast": contrast,
            "record_id": merged["record_id"].astype(str),
            "cv_group_id": merged["cv_group_id"].astype(str),
            "direct_soft_log_loss": direct_loss,
            "comparator_soft_log_loss": comparator_loss,
            "delta_soft_log_loss": direct_loss - comparator_loss,
            "direct_multiclass_brier": direct_brier,
            "comparator_multiclass_brier": comparator_brier,
            "delta_multiclass_brier": direct_brier - comparator_brier,
            "direct_posterior_expected_hard_accuracy": q[
                positions, direct_modal
            ],
            "comparator_posterior_expected_hard_accuracy": q[
                positions, comparator_modal
            ],
            "delta_posterior_expected_hard_accuracy": (
                q[positions, direct_modal]
                - q[positions, comparator_modal]
            ),
            "direct_modal_correct": (
                direct_modal == target_modal
            ).astype(float),
            "comparator_modal_correct": (
                comparator_modal == target_modal
            ).astype(float),
            "delta_modal_accuracy": (
                (direct_modal == target_modal).astype(float)
                - (comparator_modal == target_modal).astype(float)
            ),
        }
    )
    draws, summary = paired_connected_group_bootstrap(
        scores,
        [
            "delta_soft_log_loss",
            "delta_multiclass_brier",
            "delta_posterior_expected_hard_accuracy",
            "delta_modal_accuracy",
        ],
        replicates=int(repetitions),
        seed=BOOTSTRAP_SEED,
    )
    draws.insert(0, "contrast", contrast)
    summary.insert(0, "contrast", contrast)
    return scores, draws, summary


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
        contract_path: destination / "registry" / "contract_snapshot.yaml",
    }
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_code_manifest(destination: Path, contract_path: Path) -> None:
    paths = [
        Path(__file__).resolve(),
        contract_path.resolve(),
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "cardinal_models.py"
        ),
        PROJECT_ROOT / "src" / "ink_tafel_study" / "models.py",
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "feature_association_features.py"
        ),
        PROJECT_ROOT / "src" / "ink_tafel_study" / "features.py",
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
            / "posterior_cardinal_contiguous_threeclass.py"
        ),
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "posterior_cardinal_inference.py"
        ),
        (
            PROJECT_ROOT
            / "src"
            / "ink_tafel_study"
            / "cardinal_metrics.py"
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


def _write_report(
    destination: Path,
    *,
    aggregate_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    qa: Mapping[str, Any],
) -> Path:
    metrics = aggregate_metrics.set_index("evaluation_model")
    direct = metrics.loc[DIRECT_MODEL]
    prior = metrics.loc[PRIOR_MODEL]
    parent = metrics.loc[PARENT_MODEL]
    lines = [
        "# Direct multiscale soft-Gower development sensitivity",
        "",
        (
            "**Post-selection development evidence only. This result is not "
            "an independent confirmation and cannot replace contiguous v2 or "
            "formal v4.**"
        ),
        "",
        "## Fixed estimator",
        "",
        (
            f"- Directly predicts `{CLASS_LABELS[0]} / {CLASS_LABELS[1]} / "
            f"{CLASS_LABELS[2]}` soft probabilities."
        ),
        f"- Fold-local panel: `{PANEL}`.",
        (
            "- Equal probability average over Gower kNN neighbourhoods: "
            + ", ".join(str(value) for value in NEIGHBOR_COUNTS)
            + "."
        ),
        "- No posterior argmax, draw expansion, Optuna, or outer-test selection.",
        "",
        "## Repeated connected-group OOF results",
        "",
        "| model | soft log loss | Brier | modal accuracy | balanced accuracy | macro F1 |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for name, row in (
        (DIRECT_MODEL, direct),
        (PARENT_MODEL, parent),
        (PRIOR_MODEL, prior),
    ):
        lines.append(
            f"| `{name}` | "
            f"{row['full_coverage_three_class_soft_log_loss']:.6f} | "
            f"{row['multiclass_brier_mean_sum_squared_error']:.6f} | "
            f"{row['modal_accuracy']:.6f} | "
            f"{row['balanced_accuracy']:.6f} | "
            f"{row['macro_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired connected-group bootstrap",
            "",
            bootstrap_summary.to_markdown(index=False),
            "",
            "## Interpretation boundary",
            "",
            (
                "The candidate architecture and panel were developed after "
                "reviewing the same closed 97-record dataset. The repeated "
                "connected-group OOF predictions are leakage-checked, but the "
                "model-development decision is adaptive to prior results. "
                "Independent confirmation would require new data or a future "
                "precommitted evaluation, neither of which exists here."
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
        / "posterior_cardinal_contiguous_threeclass_direct_gower_report.md"
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_direct_multiscale_gower_sensitivity(
    destination: str | Path,
    *,
    parent_run: str | Path | None = None,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    n_jobs: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the fixed direct-Gower candidate on immutable parent folds."""

    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    contract, resolved_contract, contract_hash = (
        load_direct_gower_contract(contract_path)
    )
    parent = _resolve_parent(contract, parent_run)
    output = _resolve_path(destination, base=Path.cwd())
    _assert_fresh_disjoint_destination(output, parent)
    parent_before = _verify_parent(parent, contract)
    records, labels, outer, inner, parent_stack = _load_parent_tables(parent)

    selected_manifest = outer.copy()
    if smoke:
        selected_keys = (
            selected_manifest.loc[:, ["repeat", "fold"]]
            .drop_duplicates()
            .sort_values(["repeat", "fold"], kind="stable")
            .head(2)
        )
        selected_pairs = set(
            zip(
                selected_keys["repeat"].astype(int),
                selected_keys["fold"].astype(int),
                strict=True,
            )
        )
        selected_manifest = selected_manifest.loc[
            [
                (int(repeat), int(fold)) in selected_pairs
                for repeat, fold in zip(
                    selected_manifest["repeat"],
                    selected_manifest["fold"],
                    strict=True,
                )
            ]
        ].copy()
    else:
        selected_pairs = set(
            zip(
                selected_manifest["repeat"].astype(int),
                selected_manifest["fold"].astype(int),
                strict=True,
            )
        )
    split_groups = [
        (int(repeat), int(fold), group["record_id"].astype(str).tolist())
        for (repeat, fold), group in selected_manifest.groupby(
            ["repeat", "fold"], sort=True
        )
    ]

    records_by_id = records.set_index("record_id", drop=False)
    labels_by_id = labels.set_index("record_id", drop=False)
    results = Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(_run_outer_split)(
            records_by_id,
            labels_by_id,
            repeat=repeat,
            fold=fold,
            test_ids=test_ids,
        )
        for repeat, fold, test_ids in split_groups
    )
    oof_frames = [value[0] for value in results]
    component_frames = [value[1] for value in results]
    audits = [value[2] for value in results]
    new_oof = pd.concat(oof_frames, ignore_index=True)
    components = pd.concat(component_frames, ignore_index=True)
    comparator = _parent_comparator_rows(parent_stack, selected_pairs)
    oof = (
        pd.concat([new_oof, comparator], ignore_index=True)
        .sort_values(
            ["evaluation_model", "repeat", "fold", "record_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if oof.duplicated(
        ["evaluation_model", "repeat", "fold", "record_id"]
    ).any():
        raise AssertionError("Direct-Gower OOF keys are not unique")
    if oof["outer_test_used_for_selection"].astype(bool).any():
        raise AssertionError("Outer-test selection flag became true")
    probability_matrix_audit(oof.loc[:, P_COLUMNS].to_numpy(float))

    component_mean = (
        components.groupby(["repeat", "fold", "record_id"], sort=False)[
            list(P_COLUMNS)
        ]
        .mean()
        .reset_index()
    )
    direct_rows = new_oof.loc[
        new_oof["evaluation_model"].eq(DIRECT_MODEL),
        ["repeat", "fold", "record_id", *P_COLUMNS],
    ]
    ensemble_check = direct_rows.merge(
        component_mean,
        on=["repeat", "fold", "record_id"],
        suffixes=("_saved", "_components"),
        validate="one_to_one",
    )
    maximum_ensemble_error = max(
        float(
            np.max(
                np.abs(
                    ensemble_check[f"{column}_saved"].to_numpy(float)
                    - ensemble_check[f"{column}_components"].to_numpy(float)
                )
            )
        )
        for column in P_COLUMNS
    )
    if maximum_ensemble_error > 1.0e-12:
        raise AssertionError("Saved ensemble is not the component mean")

    aggregated = aggregate_contiguous_oof(oof)
    tables = _metric_tables(
        oof,
        aggregated,
        entropy_threshold=ENTROPY_THRESHOLD,
    )
    repetitions = (
        SMOKE_BOOTSTRAP_REPETITIONS
        if smoke
        else FULL_BOOTSTRAP_REPETITIONS
    )
    score_frames: list[pd.DataFrame] = []
    draw_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for baseline in (PRIOR_MODEL, PARENT_MODEL):
        scores, draws, summary = _paired_contrast(
            aggregated,
            comparator=baseline,
            repetitions=repetitions,
        )
        score_frames.append(scores)
        draw_frames.append(draws)
        summary_frames.append(summary)
    scores = pd.concat(score_frames, ignore_index=True)
    bootstrap_draws = pd.concat(draw_frames, ignore_index=True)
    bootstrap_summary = pd.concat(summary_frames, ignore_index=True)

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
    oof.to_csv(output / "analyses" / "oof_probabilities.csv", index=False)
    components.to_csv(
        output / "analyses" / "gower_component_oof_probabilities.csv",
        index=False,
    )
    aggregated.to_csv(
        output / "analyses" / "oof_aggregated.csv", index=False
    )
    for name, table in tables.items():
        table.to_csv(output / "analyses" / f"{name}.csv", index=False)
    scores.to_csv(
        output / "analyses" / "record_level_contrasts.csv", index=False
    )
    bootstrap_draws.to_csv(
        output / "analyses" / "connected_group_bootstrap.csv", index=False
    )
    bootstrap_summary.to_csv(
        output / "analyses" / "connected_group_bootstrap_summary.csv",
        index=False,
    )
    write_json(
        output / "analyses" / "outer_fit_audits.json",
        _jsonable(audits),
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

    parent_after = _verify_parent(parent, contract)
    direct_metrics = tables["aggregate_metrics"].loc[
        tables["aggregate_metrics"]["evaluation_model"].eq(DIRECT_MODEL)
    ].iloc[0]
    qa = {
        "status": (
            "complete_smoke" if smoke else "complete_posthoc_development"
        ),
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_hash,
        "analysis_role": (
            "posthoc_prediction_development_sensitivity"
        ),
        "scientifically_releasable": False,
        "independent_confirmation": False,
        "may_replace_or_rescue_v4": False,
        "may_replace_contiguous_v2": False,
        "n_records_in_registry": int(len(records)),
        "n_connected_groups_in_registry": int(
            records["cv_group_id"].astype(str).nunique()
        ),
        "n_outer_splits_evaluated": int(len(split_groups)),
        "n_oof_rows": int(len(oof)),
        "n_component_oof_rows": int(len(components)),
        "n_neighbors": list(NEIGHBOR_COUNTS),
        "feature_panel": PANEL,
        "fixed_equal_multiscale_weights": True,
        "optuna_used": False,
        "additional_prior_shrinkage_used": False,
        "posterior_argmax_used_for_training_or_selection": False,
        "posterior_draws_expanded": False,
        "class_rows_expanded": False,
        "outer_test_used_for_model_or_feature_selection": False,
        "outer_test_selection_flag_scope": (
            "current_packaged_run_only"
        ),
        "parent_same_dataset_results_visible_during_method_development": True,
        "maximum_component_mean_reconstruction_error": (
            maximum_ensemble_error
        ),
        "probability_audit": probability_matrix_audit(
            oof.loc[:, P_COLUMNS].to_numpy(float)
        ),
        "parent_manifest_verified_before": (
            parent_before["status"] == "pass"
        ),
        "parent_manifest_verified_after": (
            parent_after["status"] == "pass"
        ),
        "parent_tree_unchanged": (
            parent_before["artifact_manifest_verification"][
                "actual_tree_sha256"
            ]
            == parent_after["artifact_manifest_verification"][
                "actual_tree_sha256"
            ]
        ),
        "inner_manifest_bound_but_not_used_for_fixed_model_selection": True,
        "direct_soft_log_loss": float(
            direct_metrics[
                "full_coverage_three_class_soft_log_loss"
            ]
        ),
        "direct_modal_accuracy_descriptive_only": float(
            direct_metrics["modal_accuracy"]
        ),
        "accuracy_reaches_0_70_descriptive_only": bool(
            float(direct_metrics["modal_accuracy"]) >= 0.70
        ),
    }
    write_json(output / "analyses" / "analysis_qa.json", qa)
    report = _write_report(
        output,
        aggregate_metrics=tables["aggregate_metrics"],
        bootstrap_summary=bootstrap_summary,
        qa=qa,
    )
    write_json(
        output / "report" / "validation_report.json",
        {
            "overall_assessment": (
                "structural_smoke_only"
                if smoke
                else "ready_for_posthoc_development_review_not_formal_release"
            ),
            "methodology": (
                "fixed_direct_multiscale_soft_gower_repeated_connected_group_oof"
            ),
            "qa": qa,
            "report": str(report),
        },
    )
    manifest_path = output / "artifact_manifest.json"
    write_json(manifest_path, artifact_manifest(output))
    manifest_verification = verify_artifact_manifest(output, manifest_path)
    if manifest_verification["status"] != "pass":
        raise RuntimeError(
            "Direct-Gower artifact manifest verification failed: "
            f"{manifest_verification}"
        )
    return {
        "status": qa["status"],
        "run_dir": str(output),
        "report": str(report),
        "contract_sha256": contract_hash,
        "direct_soft_log_loss": qa["direct_soft_log_loss"],
        "direct_modal_accuracy": (
            qa["direct_modal_accuracy_descriptive_only"]
        ),
        "artifact_tree_sha256": manifest_verification[
            "actual_tree_sha256"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed direct multiscale soft-Gower development sensitivity"
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
    result = run_direct_multiscale_gower_sensitivity(
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
    "DIRECT_MODEL",
    "NEIGHBOR_COUNTS",
    "PANEL",
    "fit_predict_direct_multiscale_gower",
    "load_direct_gower_contract",
    "run_direct_multiscale_gower_sensitivity",
]
