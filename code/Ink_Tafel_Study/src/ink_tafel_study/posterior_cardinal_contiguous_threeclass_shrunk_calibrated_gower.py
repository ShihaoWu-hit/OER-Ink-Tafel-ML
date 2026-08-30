"""Strongly shrunk multiscale weighting and calibration for direct Gower.

This isolated post-hoc development stage leaves the immutable direct-Gower-v1
result unchanged.  For every connected-group outer split it builds six Gower
component predictions on the outer-training records using the parent's four
grouped inner folds.  A simplex weight vector is strongly shrunk toward the
fixed equal-weight ensemble, followed by a three-parameter calibration map
shrunk toward the identity.  Only those grouped inner-OOF predictions and
their posterior-soft targets may fit the meta-parameters.

The outer component predictions are reused byte-for-byte from the immutable
direct-Gower-v1 artifact.  Their stored target columns are never passed to the
meta-fit functions.  This remains adaptive development evidence on the same
closed 97-record dataset and cannot replace any frozen scientific result.
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
from scipy.optimize import minimize
import yaml

from .cardinal_metrics import normalize_probabilities
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
    NEIGHBOR_COUNTS,
    PANEL,
    fit_predict_direct_multiscale_gower,
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
from .threeclass_sensitivity_models import probability_matrix_audit


CONTRACT_ID = (
    "posterior_cardinal_contiguous_threeclass_shrunk_calibrated_gower_v1"
)
CONTRACT_FILENAME = f"{CONTRACT_ID}.yaml"
CONTRACT_SHA256 = (
    "f5e197bf69b1fb9d040fabae576653a67ce82a5f3e7d996634cedc24e306682f"
)
STAGE_NAME = (
    "posterior-cardinal-contiguous-threeclass-shrunk-calibrated-gower"
)
MODEL = "shrunk_calibrated_direct_multiscale_soft_gower"

WEIGHT_SHRINKAGE = 16.0
CALIBRATION_SHRINKAGE = 0.25
CALIBRATION_IDENTITY = np.asarray([1.0, 0.0, 0.0], dtype=float)
CALIBRATION_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.5, 1.5),
    (-0.5, 0.5),
    (-0.5, 0.5),
)
SCORE_EPSILON = 1.0e-15
OPTIMIZER_MAX_ITER = 500
OPTIMIZER_TOLERANCE = 1.0e-12
ACCEPTANCE_TOLERANCE = 1.0e-10
ENTROPY_THRESHOLD = 0.90
FULL_BOOTSTRAP_REPETITIONS = 2000
SMOKE_BOOTSTRAP_REPETITIONS = 100
BOOTSTRAP_SEED = 20260726

PARENT_REFIT_REQUIRED_PATHS: tuple[str, ...] = (
    "code/Ink_Tafel_Study/config/contracts/"
    "posterior_cardinal_contiguous_threeclass_direct_gower_v1.yaml",
    "code/Ink_Tafel_Study/src/ink_tafel_study/cardinal_metrics.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/cardinal_models.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/feature_association_features.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/features.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/models.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/"
    "posterior_cardinal_contiguous_threeclass.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/"
    "posterior_cardinal_contiguous_threeclass_direct_gower.py",
    "code/Ink_Tafel_Study/src/ink_tafel_study/threeclass_sensitivity_models.py",
)
CRITICAL_ENVIRONMENT_PACKAGES: tuple[str, ...] = (
    "PyYAML",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
)

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
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_shrunk_calibrated_gower_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> tuple[dict[str, Any], Path, str]:
    """Load and strictly validate the frozen development contract."""

    contract_path = _resolve_path(path, base=REPOSITORY_ROOT)
    if contract_path.name != CONTRACT_FILENAME:
        raise ValueError(f"Contract must be named {CONTRACT_FILENAME}")
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    digest = sha256_file(contract_path)
    if digest != CONTRACT_SHA256:
        raise ValueError(
            "Shrunk-calibrated-Gower contract hash mismatch: "
            f"expected={CONTRACT_SHA256}, actual={digest}"
        )
    loaded = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("Contract must be a mapping")
    contract = dict(loaded)
    if (
        contract.get("contract_id") != CONTRACT_ID
        or contract.get("stage") != STAGE_NAME
        or contract.get("status")
        != "posthoc_adaptive_development_frozen_before_packaged_run"
    ):
        raise ValueError("Contract identity changed")

    scope = dict(contract.get("scientific_scope", {}))
    if (
        scope.get("population") != "all97"
        or int(scope.get("expected_records", -1)) != 97
        or int(scope.get("expected_connected_groups", -1)) != 80
        or scope.get("independent_confirmation") is not False
        or scope.get("scientifically_releasable") is not False
        or scope.get("posterior_draws_expanded_as_samples") is not False
    ):
        raise ValueError("Scientific scope changed")

    labels = dict(contract.get("label_scheme", {}))
    if (
        tuple(labels.get("supervised_class_order", ())) != CLASS_LABELS
        or tuple(labels.get("soft_label_columns", ())) != Q_COLUMNS
        or tuple(labels.get("prediction_columns", ())) != P_COLUMNS
    ):
        raise ValueError("Soft-label scheme changed")

    model = dict(contract.get("model", {}))
    gower = dict(model.get("gower", {}))
    weights = dict(model.get("multiscale_weights", {}))
    calibration = dict(model.get("calibration", {}))
    if (
        model.get("estimator_id") != MODEL
        or model.get("feature_panel") != PANEL
        or tuple(int(value) for value in gower.get("n_neighbors", ()))
        != NEIGHBOR_COUNTS
        or float(weights.get("shrinkage_lambda", -1)) != WEIGHT_SHRINKAGE
        or float(calibration.get("shrinkage_lambda", -1))
        != CALIBRATION_SHRINKAGE
        or tuple(float(value) for value in calibration.get("slope_bounds", ()))
        != CALIBRATION_BOUNDS[0]
        or tuple(
            float(value)
            for value in calibration.get("free_intercept_bounds", ())
        )
        != CALIBRATION_BOUNDS[1]
    ):
        raise ValueError("Model or shrinkage contract changed")
    return contract, contract_path, digest


def _assert_fresh_disjoint_destination(
    destination: Path,
    parent: Path,
) -> None:
    output = destination.resolve()
    immutable = parent.resolve()
    if output == immutable or immutable in output.parents:
        raise ValueError("Output must be disjoint from its immutable parent")
    if output in immutable.parents:
        raise ValueError("Output cannot contain its immutable parent")
    if output.exists():
        raise FileExistsError(f"Output must be new and absent: {output}")


def _resolve_parent(
    contract: Mapping[str, Any],
    parent_run: str | Path | None,
) -> Path:
    parent = dict(contract["immutable_parent"])
    value = parent_run if parent_run is not None else parent["default_run"]
    return _resolve_path(value, base=REPOSITORY_ROOT)


def _verify_parent(
    parent: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not parent.is_dir():
        raise FileNotFoundError(f"Immutable parent is missing: {parent}")
    settings = dict(contract["immutable_parent"])
    verification = verify_artifact_manifest(parent)
    if verification["status"] != "pass":
        raise RuntimeError(f"Parent artifact verification failed: {verification}")
    expected_tree = str(settings["expected_artifact_tree_sha256"])
    if verification["actual_tree_sha256"] != expected_tree:
        raise RuntimeError(
            "Immutable parent tree changed: "
            f"expected={expected_tree}, "
            f"actual={verification['actual_tree_sha256']}"
        )
    file_rows: list[dict[str, Any]] = []
    for relative, expected_hash in dict(settings["required_files"]).items():
        path = parent / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Immutable parent file missing: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != str(expected_hash):
            raise RuntimeError(
                "Immutable parent file changed: "
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


def _verify_parent_refit_code(parent: Path) -> dict[str, Any]:
    """Fail closed unless current refit code matches the parent's code manifest.

    The parent artifact manifest already binds the saved code manifest.  This
    second check connects those recorded hashes to the *current* source files
    that will execute the inner Gower refits.  Every entry in the parent code
    manifest is checked, and the core direct-Gower refit dependency set must be
    present explicitly.
    """

    manifest_path = parent / "provenance" / "code_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Parent refit code manifest is missing: {manifest_path}"
        )
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("files"), list):
        raise TypeError("Parent refit code manifest is malformed")
    recorded_rows = loaded["files"]
    if not recorded_rows or not all(
        isinstance(row, dict)
        and {"path", "size_bytes", "sha256"}.issubset(row)
        for row in recorded_rows
    ):
        raise ValueError("Parent refit code manifest file rows are malformed")
    recorded_paths = [str(row["path"]) for row in recorded_rows]
    if len(recorded_paths) != len(set(recorded_paths)):
        raise ValueError("Parent refit code manifest contains duplicate paths")
    missing_required = sorted(
        set(PARENT_REFIT_REQUIRED_PATHS) - set(recorded_paths)
    )
    if missing_required:
        raise RuntimeError(
            "Parent refit code manifest omits required dependencies: "
            f"{missing_required}"
        )
    recorded_tree = str(loaded.get("tree_sha256", ""))
    recomputed_recorded_tree = tree_sha256(recorded_rows)
    if recorded_tree != recomputed_recorded_tree:
        raise RuntimeError(
            "Parent refit code manifest tree is internally inconsistent: "
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
                f"Parent refit code path escapes repository root: {relative}"
            ) from error
        if not current_path.is_file():
            raise FileNotFoundError(
                f"Current parent-refit dependency is missing: {relative}"
            )
        current_hash = sha256_file(current_path)
        current_size = int(current_path.stat().st_size)
        if (
            current_hash != str(row["sha256"])
            or current_size != int(row["size_bytes"])
        ):
            raise RuntimeError(
                "Current parent-refit code does not match the immutable "
                f"parent manifest: {relative}, "
                f"expected_sha256={row['sha256']}, actual_sha256={current_hash}, "
                f"expected_size={row['size_bytes']}, actual_size={current_size}"
            )
        current_rows.append(
            {
                "path": relative,
                "size_bytes": current_size,
                "sha256": current_hash,
                "required_for_direct_gower_inner_refit": (
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
            "Current parent-refit code tree does not match the immutable "
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


def _verify_parent_environment(parent: Path) -> dict[str, Any]:
    """Fail closed on Python or critical package version drift."""

    manifest_path = parent / "provenance" / "environment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Parent environment manifest is missing: {manifest_path}"
        )
    parent_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(parent_payload, dict) or not isinstance(
        parent_payload.get("packages"), dict
    ):
        raise TypeError("Parent environment manifest is malformed")
    parent_python = str(parent_payload.get("python", ""))
    if not parent_python:
        raise ValueError("Parent environment manifest has no Python version")
    parent_packages = dict(parent_payload["packages"])
    missing_parent_packages = sorted(
        set(CRITICAL_ENVIRONMENT_PACKAGES) - set(parent_packages)
    )
    if missing_parent_packages:
        raise RuntimeError(
            "Parent environment manifest omits critical packages: "
            f"{missing_parent_packages}"
        )

    current_payload = environment_manifest(REPOSITORY_ROOT)
    current_python = str(current_payload.get("python", ""))
    current_packages = dict(current_payload.get("packages", {}))
    if current_python != parent_python:
        raise RuntimeError(
            "Current Python does not match the immutable parent environment: "
            f"expected={parent_python!r}, actual={current_python!r}"
        )
    package_rows: list[dict[str, Any]] = []
    for package in CRITICAL_ENVIRONMENT_PACKAGES:
        expected = parent_packages.get(package)
        actual = current_packages.get(package)
        if expected is None or actual is None or str(actual) != str(expected):
            raise RuntimeError(
                "Current critical package does not match the immutable parent "
                f"environment: package={package}, expected={expected!r}, "
                f"actual={actual!r}"
            )
        package_rows.append(
            {
                "package": package,
                "parent_version": str(expected),
                "current_version": str(actual),
                "matches": True,
            }
        )
    return {
        "status": "pass",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "python_parent": parent_python,
        "python_current": current_python,
        "python_matches": True,
        "critical_packages": package_rows,
        "all_critical_packages_match": True,
        "parent_python_executable_recorded": parent_payload.get(
            "python_executable"
        ),
        "current_python_executable": current_payload.get("python_executable"),
        "executable_path_compared": False,
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
    components = pd.read_csv(
        parent / "analyses" / "gower_component_oof_probabilities.csv"
    )
    for frame in (records, labels, outer, inner, components):
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
    if len(outer) != 970 or len(inner) != 3880:
        raise ValueError("Immutable parent fold manifests changed")
    if (
        outer.groupby(["repeat", "record_id"]).size().ne(1).any()
        or outer.groupby(["repeat", "cv_group_id"])["fold"]
        .nunique()
        .gt(1)
        .any()
        or inner.groupby(
            ["outer_repeat", "outer_fold", "cv_group_id"]
        )["inner_fold"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("A connected group crosses a parent fold")

    expected_component_rows = 97 * 10 * len(NEIGHBOR_COUNTS)
    if (
        len(components) != expected_component_rows
        or components.duplicated(
            ["repeat", "fold", "record_id", "n_neighbors"]
        ).any()
        or set(components["n_neighbors"].astype(int)) != set(NEIGHBOR_COUNTS)
        or components.groupby(["repeat", "record_id"]).size()
        .ne(len(NEIGHBOR_COUNTS))
        .any()
    ):
        raise ValueError("Immutable parent component coverage changed")
    probability_matrix_audit(components.loc[:, P_COLUMNS].to_numpy(float))
    return records, labels, outer, inner, components


def fold_equal_weights(fold_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return row weights that give every represented inner fold equal mass."""

    folds = np.asarray(fold_ids, dtype=int)
    if folds.ndim != 1 or len(folds) == 0:
        raise ValueError("fold_ids must be a non-empty one-dimensional array")
    output = np.zeros(len(folds), dtype=float)
    for fold in np.unique(folds):
        selected = folds == int(fold)
        output[selected] = 1.0 / float(selected.sum())
    total = float(output.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Fold-equal weights are invalid")
    output /= total
    return output


def _validate_soft_targets(
    targets: Any,
    *,
    n_rows: int | None = None,
) -> np.ndarray:
    q = np.asarray(targets, dtype=float)
    if (
        q.ndim != 2
        or q.shape[1] != len(CLASS_LABELS)
        or (n_rows is not None and q.shape[0] != int(n_rows))
        or not np.isfinite(q).all()
        or np.any(q < 0)
        or not np.allclose(q.sum(axis=1), 1.0, atol=1.0e-12, rtol=0)
    ):
        raise ValueError("Soft targets are invalid")
    return q


def _validate_component_cube(components: Any) -> np.ndarray:
    cube = np.asarray(components, dtype=float)
    if (
        cube.ndim != 3
        or cube.shape[1:] != (len(NEIGHBOR_COUNTS), len(CLASS_LABELS))
        or not np.isfinite(cube).all()
        or np.any(cube < 0)
        or not np.allclose(cube.sum(axis=2), 1.0, atol=1.0e-12, rtol=0)
    ):
        raise ValueError("Gower component probability cube is invalid")
    return cube


def mean_soft_cross_entropy(
    targets: Any,
    probabilities: Any,
    sample_weight: Any,
) -> float:
    """Fold-weighted mean soft cross-entropy; weights are normalized here."""

    p = normalize_probabilities(np.asarray(probabilities, dtype=float))
    q = _validate_soft_targets(targets, n_rows=len(p))
    weights = np.asarray(sample_weight, dtype=float)
    if (
        weights.shape != (len(q),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or float(weights.sum()) <= 0
    ):
        raise ValueError("sample_weight is invalid")
    weights = weights / weights.sum()
    return -float(
        np.sum(
            weights[:, None]
            * q
            * np.log(np.clip(p, SCORE_EPSILON, 1.0))
        )
    )


def mix_component_probabilities(
    components: Any,
    weights: Any,
) -> np.ndarray:
    cube = _validate_component_cube(components)
    values = np.asarray(weights, dtype=float)
    if (
        values.shape != (len(NEIGHBOR_COUNTS),)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0, atol=1.0e-10, rtol=0)
    ):
        raise ValueError("Multiscale weights must lie on the simplex")
    output = np.einsum("nmc,m->nc", cube, values, optimize=True)
    return normalize_probabilities(output)


def fit_shrunk_multiscale_weights(
    components: Any,
    targets: Any,
    sample_weight: Any,
    *,
    shrinkage: float = WEIGHT_SHRINKAGE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit strongly equal-weight-shrunk simplex weights on inner OOF."""

    cube = _validate_component_cube(components)
    q = _validate_soft_targets(targets, n_rows=len(cube))
    row_weights = np.asarray(sample_weight, dtype=float)
    if not np.isfinite(float(shrinkage)) or float(shrinkage) < 0:
        raise ValueError("Weight shrinkage must be finite and nonnegative")
    uniform = np.repeat(1.0 / len(NEIGHBOR_COUNTS), len(NEIGHBOR_COUNTS))
    baseline_loss = mean_soft_cross_entropy(
        q,
        mix_component_probabilities(cube, uniform),
        row_weights,
    )

    def objective(candidate: np.ndarray) -> float:
        # SLSQP's finite-difference probes need not satisfy the equality
        # constraint to machine precision.  Normalize only inside the
        # objective; the accepted final solution is still checked against the
        # explicit nonnegative simplex constraint below.
        normalized = np.maximum(np.asarray(candidate, dtype=float), 0.0)
        total = float(normalized.sum())
        if not np.isfinite(total) or total <= 0:
            return float("inf")
        normalized /= total
        return mean_soft_cross_entropy(
            q,
            mix_component_probabilities(cube, normalized),
            row_weights,
        ) + float(shrinkage) * float(
            np.sum(np.square(normalized - uniform))
        )

    result = minimize(
        objective,
        uniform,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(uniform),
        constraints={
            "type": "eq",
            "fun": lambda candidate: float(candidate.sum() - 1.0),
        },
        options={
            "maxiter": OPTIMIZER_MAX_ITER,
            "ftol": OPTIMIZER_TOLERANCE,
            "disp": False,
        },
    )
    candidate = np.asarray(result.x, dtype=float)
    feasible = (
        bool(result.success)
        and candidate.shape == uniform.shape
        and np.isfinite(candidate).all()
        and np.min(candidate) >= -1.0e-8
        and abs(float(candidate.sum()) - 1.0) <= 1.0e-7
    )
    fallback_reason = ""
    if feasible:
        candidate = np.maximum(candidate, 0.0)
        candidate /= candidate.sum()
        fitted_loss = mean_soft_cross_entropy(
            q,
            mix_component_probabilities(cube, candidate),
            row_weights,
        )
        feasible = bool(
            np.isfinite(fitted_loss)
            and fitted_loss <= baseline_loss + ACCEPTANCE_TOLERANCE
        )
        if not feasible:
            fallback_reason = "optimized_raw_loss_worse_than_equal_weights"
    else:
        fitted_loss = baseline_loss
        fallback_reason = "optimizer_failure_or_infeasible_solution"
    fallback_used = not feasible
    selected = uniform if fallback_used else candidate
    selected_loss = baseline_loss if fallback_used else fitted_loss
    return selected, {
        "status": "fallback" if fallback_used else "complete",
        "optimizer": "scipy_SLSQP",
        "success": bool(result.success),
        "message": str(result.message),
        "n_iterations": int(getattr(result, "nit", -1)),
        "shrinkage_lambda": float(shrinkage),
        "objective_loss_aggregation": "fold_equal_weighted_mean",
        "equal_weight_raw_soft_log_loss": baseline_loss,
        "selected_raw_soft_log_loss": selected_loss,
        "selected_penalized_objective": float(objective(selected)),
        "weights": {
            str(count): float(selected[index])
            for index, count in enumerate(NEIGHBOR_COUNTS)
        },
        "maximum_absolute_deviation_from_equal": float(
            np.max(np.abs(selected - uniform))
        ),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "posterior_argmax_used": False,
        "outer_test_used": False,
    }


def apply_identity_shrunk_calibration(
    probabilities: Any,
    parameters: Any,
) -> np.ndarray:
    p = normalize_probabilities(np.asarray(probabilities, dtype=float))
    theta = np.asarray(parameters, dtype=float)
    if theta.shape != (3,) or not np.isfinite(theta).all():
        raise ValueError("Calibration parameters must be three finite values")
    slope, intercept_0, intercept_1 = theta
    intercepts = np.asarray(
        [intercept_0, intercept_1, -intercept_0 - intercept_1],
        dtype=float,
    )
    logits = slope * np.log(np.clip(p, SCORE_EPSILON, 1.0)) + intercepts
    logits -= np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    output = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    probability_matrix_audit(output)
    return output


def fit_identity_shrunk_calibrator(
    probabilities: Any,
    targets: Any,
    sample_weight: Any,
    *,
    shrinkage: float = CALIBRATION_SHRINKAGE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a shared log-probability slope and two sum-to-zero intercepts."""

    p = normalize_probabilities(np.asarray(probabilities, dtype=float))
    q = _validate_soft_targets(targets, n_rows=len(p))
    row_weights = np.asarray(sample_weight, dtype=float)
    if not np.isfinite(float(shrinkage)) or float(shrinkage) < 0:
        raise ValueError("Calibration shrinkage must be finite and nonnegative")
    identity_loss = mean_soft_cross_entropy(q, p, row_weights)

    def penalty(theta: np.ndarray) -> float:
        slope, intercept_0, intercept_1 = theta
        intercept_2 = -intercept_0 - intercept_1
        return float(
            np.square(slope - 1.0)
            + np.square(intercept_0)
            + np.square(intercept_1)
            + np.square(intercept_2)
        )

    def objective(theta: np.ndarray) -> float:
        return mean_soft_cross_entropy(
            q,
            apply_identity_shrunk_calibration(p, theta),
            row_weights,
        ) + float(shrinkage) * penalty(theta)

    result = minimize(
        objective,
        CALIBRATION_IDENTITY.copy(),
        method="L-BFGS-B",
        bounds=list(CALIBRATION_BOUNDS),
        options={
            "maxiter": OPTIMIZER_MAX_ITER,
            "ftol": OPTIMIZER_TOLERANCE,
        },
    )
    candidate = np.asarray(result.x, dtype=float)
    feasible = (
        bool(result.success)
        and candidate.shape == (3,)
        and np.isfinite(candidate).all()
    )
    fallback_reason = ""
    if feasible:
        fitted_loss = mean_soft_cross_entropy(
            q,
            apply_identity_shrunk_calibration(p, candidate),
            row_weights,
        )
        feasible = bool(
            np.isfinite(fitted_loss)
            and fitted_loss <= identity_loss + ACCEPTANCE_TOLERANCE
        )
        if not feasible:
            fallback_reason = "optimized_raw_loss_worse_than_identity"
    else:
        fitted_loss = identity_loss
        fallback_reason = "optimizer_failure_or_infeasible_solution"
    fallback_used = not feasible
    selected = CALIBRATION_IDENTITY.copy() if fallback_used else candidate
    selected_loss = identity_loss if fallback_used else fitted_loss
    intercepts = [
        float(selected[1]),
        float(selected[2]),
        float(-selected[1] - selected[2]),
    ]
    return selected, {
        "status": "fallback" if fallback_used else "complete",
        "optimizer": "scipy_L-BFGS-B",
        "success": bool(result.success),
        "message": str(result.message),
        "n_iterations": int(getattr(result, "nit", -1)),
        "shrinkage_lambda": float(shrinkage),
        "objective_loss_aggregation": "fold_equal_weighted_mean",
        "identity_raw_soft_log_loss": identity_loss,
        "selected_raw_soft_log_loss": selected_loss,
        "selected_penalized_objective": float(objective(selected)),
        "slope": float(selected[0]),
        "class_intercepts": {
            label: intercepts[index]
            for index, label in enumerate(CLASS_LABELS)
        },
        "intercept_sum": float(sum(intercepts)),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "posterior_argmax_used": False,
        "outer_test_used": False,
    }


def _component_cube_from_long_table(
    component_rows: pd.DataFrame,
    record_order: Sequence[str],
) -> np.ndarray:
    ordered = [str(value) for value in record_order]
    indexed = component_rows.copy()
    indexed["record_id"] = indexed["record_id"].astype(str)
    if indexed.duplicated(["record_id", "n_neighbors"]).any():
        raise ValueError("Component table contains duplicate record/k rows")
    lookup = {
        (str(row["record_id"]), int(row["n_neighbors"])): row
        for _, row in indexed.iterrows()
    }
    expected = {
        (record_id, count)
        for record_id in ordered
        for count in NEIGHBOR_COUNTS
    }
    if set(lookup) != expected:
        raise ValueError("Component table does not match requested records/k")
    cube = np.stack(
        [
            np.stack(
                [
                    np.asarray(
                        [lookup[(record_id, count)][column] for column in P_COLUMNS],
                        dtype=float,
                    )
                    for count in NEIGHBOR_COUNTS
                ],
                axis=0,
            )
            for record_id in ordered
        ],
        axis=0,
    )
    return _validate_component_cube(cube)


def build_grouped_inner_component_oof(
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    outer_train_ids: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build six-component grouped inner-OOF predictions for one outer fold."""

    ordered = [str(value) for value in outer_train_ids]
    splits = _manifest_inner_splits(
        ordered,
        inner_manifest,
        repeat=int(repeat),
        fold=int(fold),
    )
    rows: list[dict[str, Any]] = []
    validation_order: list[str] = []
    validation_folds: list[int] = []
    for train_ids, validation_ids, inner_fold in splits:
        train_groups = set(
            records_by_id.loc[train_ids, "cv_group_id"].astype(str)
        )
        validation_groups = set(
            records_by_id.loc[validation_ids, "cv_group_id"].astype(str)
        )
        if train_groups & validation_groups:
            raise AssertionError("A connected group crosses an inner fold")
        _, components, audit = fit_predict_direct_multiscale_gower(
            records_by_id.loc[train_ids].reset_index(drop=True),
            labels_by_id.loc[train_ids, Q_COLUMNS].to_numpy(float),
            records_by_id.loc[validation_ids].reset_index(drop=True),
        )
        if audit["outer_test_used_for_selection"] is not False:
            raise AssertionError("Base Gower audit reports test selection")
        for position, record_id in enumerate(validation_ids):
            identity = labels_by_id.loc[record_id]
            validation_order.append(str(record_id))
            validation_folds.append(int(inner_fold))
            for count in NEIGHBOR_COUNTS:
                prediction = components[int(count)][position]
                rows.append(
                    {
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "inner_fold": int(inner_fold),
                        "record_id": str(record_id),
                        "cv_group_id": str(identity["cv_group_id"]),
                        "n_neighbors": int(count),
                        **{
                            column: float(identity[column])
                            for column in Q_COLUMNS
                        },
                        **{
                            column: float(prediction[class_index])
                            for class_index, column in enumerate(P_COLUMNS)
                        },
                        "outer_test_used_for_fit": False,
                    }
                )
    if (
        len(validation_order) != len(ordered)
        or set(validation_order) != set(ordered)
        or len(set(validation_order)) != len(ordered)
    ):
        raise AssertionError("Inner OOF does not cover outer training exactly once")
    table = pd.DataFrame(rows)
    cube = _component_cube_from_long_table(table, validation_order)
    targets = labels_by_id.loc[validation_order, Q_COLUMNS].to_numpy(float)
    weights = fold_equal_weights(validation_folds)
    return table, cube, targets, weights


def fit_shrunk_calibrated_meta_model(
    inner_components: Any,
    targets: Any,
    sample_weight: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    component_weights, weight_audit = fit_shrunk_multiscale_weights(
        inner_components,
        targets,
        sample_weight,
    )
    inner_mixture = mix_component_probabilities(
        inner_components,
        component_weights,
    )
    calibration, calibration_audit = fit_identity_shrunk_calibrator(
        inner_mixture,
        targets,
        sample_weight,
    )
    calibrated = apply_identity_shrunk_calibration(
        inner_mixture,
        calibration,
    )
    return component_weights, calibration, {
        "status": "complete",
        "weight_fit": weight_audit,
        "calibration_fit": calibration_audit,
        "inner_calibrated_raw_soft_log_loss": mean_soft_cross_entropy(
            targets,
            calibrated,
            sample_weight,
        ),
        "loss_aggregation": "fold_equal_weighted_mean",
        "posterior_argmax_used": False,
        "outer_test_used": False,
    }


def _outer_component_cube(
    parent_components: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    test_ids: Sequence[str],
) -> np.ndarray:
    """Return only immutable outer P components; stored q columns are ignored."""

    selected = parent_components.loc[
        parent_components["repeat"].eq(int(repeat))
        & parent_components["fold"].eq(int(fold)),
        ["record_id", "n_neighbors", *P_COLUMNS],
    ].copy()
    return _component_cube_from_long_table(selected, test_ids)


def _run_outer_split(
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    parent_components: pd.DataFrame,
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
        raise AssertionError("Outer partition is invalid")
    train_groups = set(records_by_id.loc[train, "cv_group_id"].astype(str))
    test_groups = set(records_by_id.loc[test, "cv_group_id"].astype(str))
    if train_groups & test_groups:
        raise AssertionError("A connected group crosses an outer fold")

    inner_rows, inner_cube, inner_targets, inner_weights = (
        build_grouped_inner_component_oof(
            records_by_id,
            labels_by_id,
            inner_manifest,
            repeat=int(repeat),
            fold=int(fold),
            outer_train_ids=train,
        )
    )
    component_weights, calibration, meta_audit = (
        fit_shrunk_calibrated_meta_model(
            inner_cube,
            inner_targets,
            inner_weights,
        )
    )

    # This helper deliberately receives no outer-test targets.
    outer_cube = _outer_component_cube(
        parent_components,
        repeat=int(repeat),
        fold=int(fold),
        test_ids=test,
    )
    direct = mix_component_probabilities(
        outer_cube,
        np.repeat(
            1.0 / len(NEIGHBOR_COUNTS),
            len(NEIGHBOR_COUNTS),
        ),
    )
    weighted = mix_component_probabilities(outer_cube, component_weights)
    calibrated = apply_identity_shrunk_calibration(weighted, calibration)
    y_train = labels_by_id.loc[train, Q_COLUMNS].to_numpy(float)
    prior = normalize_probabilities(
        np.repeat(y_train.mean(axis=0)[None, :], len(test), axis=0)
    )

    probability_rows: list[dict[str, Any]] = []
    for position, record_id in enumerate(test):
        identity = labels_by_id.loc[record_id]
        common = {
            "repeat": int(repeat),
            "fold": int(fold),
            "record_id": str(record_id),
            "cv_group_id": str(identity["cv_group_id"]),
            **{column: float(identity[column]) for column in Q_COLUMNS},
        }
        for model, prediction in (
            (MODEL, calibrated),
            (DIRECT_MODEL, direct),
            (PRIOR_MODEL, prior),
        ):
            probability_rows.append(
                {
                    **common,
                    "evaluation_model": str(model),
                    **{
                        column: float(prediction[position, class_index])
                        for class_index, column in enumerate(P_COLUMNS)
                    },
                    "outer_test_used_for_selection": False,
                }
            )

    intercepts = [
        float(calibration[1]),
        float(calibration[2]),
        float(-calibration[1] - calibration[2]),
    ]
    parameter_row = {
        "repeat": int(repeat),
        "fold": int(fold),
        **{
            f"weight_k_{count}": float(component_weights[index])
            for index, count in enumerate(NEIGHBOR_COUNTS)
        },
        "calibration_slope": float(calibration[0]),
        **{
            f"calibration_intercept_{label}": intercepts[index]
            for index, label in enumerate(CLASS_LABELS)
        },
        "weight_fallback_used": bool(
            meta_audit["weight_fit"]["fallback_used"]
        ),
        "calibration_fallback_used": bool(
            meta_audit["calibration_fit"]["fallback_used"]
        ),
        "inner_equal_weight_soft_log_loss": float(
            meta_audit["weight_fit"]["equal_weight_raw_soft_log_loss"]
        ),
        "inner_weighted_soft_log_loss": float(
            meta_audit["weight_fit"]["selected_raw_soft_log_loss"]
        ),
        "inner_calibrated_soft_log_loss": float(
            meta_audit["inner_calibrated_raw_soft_log_loss"]
        ),
        "outer_test_used_for_fit": False,
    }
    audit = {
        "repeat": int(repeat),
        "fold": int(fold),
        "train_record_ids": train,
        "test_record_ids": test,
        "n_inner_oof_records": int(len(inner_targets)),
        "n_inner_component_rows": int(len(inner_rows)),
        "inner_fold_total_weights": {
            str(inner_fold): float(
                inner_weights[
                    inner_rows.drop_duplicates("record_id")["inner_fold"]
                    .to_numpy(int)
                    == int(inner_fold)
                ].sum()
            )
            for inner_fold in sorted(
                inner_rows["inner_fold"].astype(int).unique()
            )
        },
        "meta_fit": meta_audit,
        "outer_component_source": (
            "immutable_parent_gower_component_oof_probabilities"
        ),
        "outer_component_target_columns_passed_to_fit": False,
        "outer_test_used_for_fit": False,
        "posterior_argmax_used_for_fit": False,
        "outer_probability_audit": probability_matrix_audit(calibrated),
    }
    return {
        "probability_rows": pd.DataFrame(probability_rows),
        "inner_component_rows": inner_rows,
        "parameter_row": pd.DataFrame([parameter_row]),
        "audit": audit,
    }


def _paired_contrast(
    aggregated: pd.DataFrame,
    *,
    comparator: str,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate = aggregated.loc[
        aggregated["evaluation_model"].eq(MODEL)
    ].copy()
    baseline = aggregated.loc[
        aggregated["evaluation_model"].eq(str(comparator))
    ].copy()
    merged = candidate.merge(
        baseline,
        on=["record_id", "cv_group_id"],
        suffixes=("_candidate", "_comparator"),
        validate="one_to_one",
    )
    q = merged.loc[
        :, [f"{column}_candidate" for column in Q_COLUMNS]
    ].to_numpy(float)
    p_candidate = merged.loc[
        :, [f"{column}_candidate" for column in P_COLUMNS]
    ].to_numpy(float)
    p_comparator = merged.loc[
        :, [f"{column}_comparator" for column in P_COLUMNS]
    ].to_numpy(float)
    candidate_loss = -np.sum(
        q * np.log(np.clip(p_candidate, SCORE_EPSILON, 1.0)), axis=1
    )
    comparator_loss = -np.sum(
        q * np.log(np.clip(p_comparator, SCORE_EPSILON, 1.0)), axis=1
    )
    candidate_brier = np.sum(np.square(p_candidate - q), axis=1)
    comparator_brier = np.sum(np.square(p_comparator - q), axis=1)
    target_modal = np.argmax(q, axis=1)
    candidate_modal = np.argmax(p_candidate, axis=1)
    comparator_modal = np.argmax(p_comparator, axis=1)
    positions = np.arange(len(q))
    contrast = f"{MODEL}_minus_{comparator}"
    scores = pd.DataFrame(
        {
            "contrast": contrast,
            "record_id": merged["record_id"].astype(str),
            "cv_group_id": merged["cv_group_id"].astype(str),
            "candidate_soft_log_loss": candidate_loss,
            "comparator_soft_log_loss": comparator_loss,
            "delta_soft_log_loss": candidate_loss - comparator_loss,
            "candidate_multiclass_brier": candidate_brier,
            "comparator_multiclass_brier": comparator_brier,
            "delta_multiclass_brier": candidate_brier - comparator_brier,
            "candidate_posterior_expected_hard_accuracy": q[
                positions, candidate_modal
            ],
            "comparator_posterior_expected_hard_accuracy": q[
                positions, comparator_modal
            ],
            "delta_posterior_expected_hard_accuracy": (
                q[positions, candidate_modal]
                - q[positions, comparator_modal]
            ),
            "candidate_modal_correct": (
                candidate_modal == target_modal
            ).astype(float),
            "comparator_modal_correct": (
                comparator_modal == target_modal
            ).astype(float),
            "delta_modal_accuracy": (
                (candidate_modal == target_modal).astype(float)
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
        parent / "analyses" / "gower_component_oof_probabilities.csv": (
            destination / "inputs" / "parent_gower_component_oof_probabilities.csv"
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
    paths = [
        Path(__file__).resolve(),
        contract_path.resolve(),
        PROJECT_ROOT
        / "src"
        / "ink_tafel_study"
        / "posterior_cardinal_contiguous_threeclass_direct_gower.py",
        PROJECT_ROOT
        / "src"
        / "ink_tafel_study"
        / "posterior_cardinal_contiguous_threeclass.py",
        PROJECT_ROOT / "src" / "ink_tafel_study" / "cardinal_models.py",
        PROJECT_ROOT / "src" / "ink_tafel_study" / "models.py",
        PROJECT_ROOT
        / "src"
        / "ink_tafel_study"
        / "threeclass_sensitivity_models.py",
        PROJECT_ROOT
        / "src"
        / "ink_tafel_study"
        / "posterior_cardinal_inference.py",
        PROJECT_ROOT / "src" / "ink_tafel_study" / "cardinal_metrics.py",
        PROJECT_ROOT / "src" / "ink_tafel_study" / "provenance.py",
    ]
    files = inventory_files(paths, relative_to=REPOSITORY_ROOT)
    write_json(
        destination / "provenance" / "code_manifest.json",
        {"tree_sha256": tree_sha256(files), "files": files},
    )


def _write_report(
    destination: Path,
    *,
    aggregate_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    qa: Mapping[str, Any],
) -> Path:
    metrics = aggregate_metrics.set_index("evaluation_model")
    lines = [
        "# Strongly shrunk and calibrated direct-Gower development sensitivity",
        "",
        (
            "**Post-selection development evidence only. This result is not "
            "independent confirmation and cannot replace direct-Gower v1, "
            "contiguous v2, or formal v4.**"
        ),
        "",
        "## Frozen estimator",
        "",
        f"- Fold-local panel: `{PANEL}`.",
        (
            "- Gower neighbourhoods: "
            + ", ".join(str(value) for value in NEIGHBOR_COUNTS)
            + "."
        ),
        (
            f"- Simplex weights use grouped inner-OOF mean soft cross-entropy "
            f"with equal-weight shrinkage lambda {WEIGHT_SHRINKAGE:g}."
        ),
        (
            "- Calibration uses one shared log-probability slope and two "
            f"sum-to-zero class intercepts with identity shrinkage lambda "
            f"{CALIBRATION_SHRINKAGE:g}."
        ),
        "- Outer-test targets never fit weights or calibration.",
        "",
        "## Repeated connected-group OOF results",
        "",
        "| model | soft log loss | Brier | modal accuracy | balanced accuracy | macro F1 |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for name in (MODEL, DIRECT_MODEL, PRIOR_MODEL):
        row = metrics.loc[name]
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
                "The architecture and shrinkage constants were chosen after "
                "reviewing the same closed dataset. Nested outer evaluation "
                "protects each outer test fold from fitting, but it does not "
                "erase this global development adaptivity."
            ),
            "",
            (
                f"Artifact QA status: `{qa['status']}`; scientifically "
                f"releasable: `{str(qa['scientifically_releasable']).lower()}`."
            ),
            "",
        ]
    )
    path = (
        destination
        / "report"
        / "posterior_cardinal_contiguous_threeclass_shrunk_calibrated_gower_report.md"
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_shrunk_calibrated_gower_sensitivity(
    destination: str | Path,
    *,
    parent_run: str | Path | None = None,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    n_jobs: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the isolated grouped-inner-OOF shrunk/calibrated Gower stage."""

    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    contract, resolved_contract, contract_hash = (
        load_shrunk_calibrated_gower_contract(contract_path)
    )
    parent = _resolve_parent(contract, parent_run)
    output = _resolve_path(destination, base=Path.cwd())
    _assert_fresh_disjoint_destination(output, parent)
    parent_before = _verify_parent(parent, contract)
    # This must pass before _load_parent_tables and, critically, before any
    # grouped inner refit calls fit_predict_direct_multiscale_gower.
    refit_code_before = _verify_parent_refit_code(parent)
    environment_before = _verify_parent_environment(parent)
    records, labels, outer, inner, parent_components = _load_parent_tables(parent)

    selected_manifest = outer.copy()
    split_keys = (
        selected_manifest.loc[:, ["repeat", "fold"]]
        .drop_duplicates()
        .sort_values(["repeat", "fold"], kind="stable")
    )
    if smoke:
        split_keys = split_keys.head(2)
    selected_pairs = set(
        zip(
            split_keys["repeat"].astype(int),
            split_keys["fold"].astype(int),
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
            inner,
            parent_components,
            repeat=repeat,
            fold=fold,
            test_ids=test_ids,
        )
        for repeat, fold, test_ids in split_groups
    )
    oof = (
        pd.concat(
            [result["probability_rows"] for result in results],
            ignore_index=True,
        )
        .sort_values(
            ["evaluation_model", "repeat", "fold", "record_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    inner_components = pd.concat(
        [result["inner_component_rows"] for result in results],
        ignore_index=True,
    )
    parameters = pd.concat(
        [result["parameter_row"] for result in results],
        ignore_index=True,
    ).sort_values(["repeat", "fold"], kind="stable")
    audits = [result["audit"] for result in results]
    if oof.duplicated(
        ["evaluation_model", "repeat", "fold", "record_id"]
    ).any():
        raise AssertionError("OOF keys are not unique")
    if oof["outer_test_used_for_selection"].astype(bool).any():
        raise AssertionError("Outer-test selection flag became true")
    probability_matrix_audit(oof.loc[:, P_COLUMNS].to_numpy(float))

    # Check that the saved direct comparator is exactly the equal component mean.
    selected_parent = parent_components.loc[
        [
            (int(repeat), int(fold)) in selected_pairs
            for repeat, fold in zip(
                parent_components["repeat"],
                parent_components["fold"],
                strict=True,
            )
        ]
    ]
    component_mean = (
        selected_parent.groupby(["repeat", "fold", "record_id"], sort=False)[
            list(P_COLUMNS)
        ]
        .mean()
        .reset_index()
    )
    direct_rows = oof.loc[
        oof["evaluation_model"].eq(DIRECT_MODEL),
        ["repeat", "fold", "record_id", *P_COLUMNS],
    ]
    reconstruction = direct_rows.merge(
        component_mean,
        on=["repeat", "fold", "record_id"],
        suffixes=("_saved", "_components"),
        validate="one_to_one",
    )
    maximum_direct_reconstruction_error = max(
        float(
            np.max(
                np.abs(
                    reconstruction[f"{column}_saved"].to_numpy(float)
                    - reconstruction[f"{column}_components"].to_numpy(float)
                )
            )
        )
        for column in P_COLUMNS
    )
    if maximum_direct_reconstruction_error > 1.0e-12:
        raise AssertionError("Direct comparator is not the parent component mean")

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
    for comparator in (DIRECT_MODEL, PRIOR_MODEL):
        scores, draws, summary = _paired_contrast(
            aggregated,
            comparator=comparator,
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
    inner_components.to_csv(
        output / "analyses" / "inner_component_oof_probabilities.csv",
        index=False,
    )
    parameters.to_csv(
        output / "analyses" / "outer_split_meta_parameters.csv",
        index=False,
    )
    aggregated.to_csv(output / "analyses" / "oof_aggregated.csv", index=False)
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
    refit_code_after = _verify_parent_refit_code(parent)
    environment_after = _verify_parent_environment(parent)
    write_json(
        output / "provenance" / "parent_refit_code_binding.json",
        {
            "verified_before_inner_refits": refit_code_before,
            "verified_after_modeling": refit_code_after,
            "code_tree_unchanged_during_run": (
                refit_code_before["current_tree_sha256"]
                == refit_code_after["current_tree_sha256"]
            ),
        },
    )
    write_json(
        output / "provenance" / "parent_environment_binding.json",
        {
            "verified_before_inner_refits": environment_before,
            "verified_after_modeling": environment_after,
            "critical_environment_unchanged_during_run": (
                environment_before["python_current"]
                == environment_after["python_current"]
                and environment_before["critical_packages"]
                == environment_after["critical_packages"]
            ),
        },
    )
    metrics = tables["aggregate_metrics"].set_index("evaluation_model")
    candidate_metrics = metrics.loc[MODEL]
    direct_metrics = metrics.loc[DIRECT_MODEL]
    qa = {
        "status": "complete_smoke" if smoke else "complete_posthoc_development",
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
        "n_oof_rows": int(len(oof)),
        "n_inner_component_oof_rows": int(len(inner_components)),
        "n_meta_parameter_rows": int(len(parameters)),
        "feature_panel": PANEL,
        "n_neighbors": list(NEIGHBOR_COUNTS),
        "weight_shrinkage_lambda": WEIGHT_SHRINKAGE,
        "calibration_shrinkage_lambda": CALIBRATION_SHRINKAGE,
        "inner_loss_aggregation": "fold_equal_weighted_mean",
        "posterior_argmax_used_for_training_or_selection": False,
        "posterior_draws_expanded": False,
        "class_rows_expanded": False,
        "outer_test_used_for_weight_calibration_or_selection": False,
        "outer_parent_component_target_columns_passed_to_fit": False,
        "parent_outer_components_reused_without_refit": True,
        "weight_fallback_count": int(
            parameters["weight_fallback_used"].astype(bool).sum()
        ),
        "calibration_fallback_count": int(
            parameters["calibration_fallback_used"].astype(bool).sum()
        ),
        "maximum_direct_component_mean_reconstruction_error": (
            maximum_direct_reconstruction_error
        ),
        "probability_audit": probability_matrix_audit(
            oof.loc[:, P_COLUMNS].to_numpy(float)
        ),
        "parent_manifest_verified_before": parent_before["status"] == "pass",
        "parent_manifest_verified_after": parent_after["status"] == "pass",
        "parent_refit_code_verified_before_inner_refits": (
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
        "parent_refit_code_tree_unchanged_during_run": (
            refit_code_before["current_tree_sha256"]
            == refit_code_after["current_tree_sha256"]
        ),
        "parent_environment_verified_before_inner_refits": (
            environment_before["status"] == "pass"
        ),
        "parent_environment_verified_after_modeling": (
            environment_after["status"] == "pass"
        ),
        "parent_environment_manifest_sha256": environment_before[
            "manifest_sha256"
        ],
        "parent_python_matches_current": environment_before[
            "python_matches"
        ],
        "parent_critical_packages_match_current": environment_before[
            "all_critical_packages_match"
        ],
        "parent_critical_package_versions": {
            row["package"]: row["current_version"]
            for row in environment_before["critical_packages"]
        },
        "parent_critical_environment_unchanged_during_run": (
            environment_before["python_current"]
            == environment_after["python_current"]
            and environment_before["critical_packages"]
            == environment_after["critical_packages"]
        ),
        "parent_tree_unchanged": (
            parent_before["artifact_manifest_verification"][
                "actual_tree_sha256"
            ]
            == parent_after["artifact_manifest_verification"][
                "actual_tree_sha256"
            ]
        ),
        "candidate_soft_log_loss": float(
            candidate_metrics["full_coverage_three_class_soft_log_loss"]
        ),
        "direct_soft_log_loss": float(
            direct_metrics["full_coverage_three_class_soft_log_loss"]
        ),
        "candidate_minus_direct_soft_log_loss": float(
            candidate_metrics["full_coverage_three_class_soft_log_loss"]
            - direct_metrics["full_coverage_three_class_soft_log_loss"]
        ),
        "candidate_modal_accuracy_descriptive_only": float(
            candidate_metrics["modal_accuracy"]
        ),
        "same_closed_dataset_results_visible_during_development": True,
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
                "grouped_inner_oof_strongly_shrunk_multiscale_weights_and_identity_calibration"
            ),
            "qa": qa,
            "report": str(report),
        },
    )
    manifest_path = output / "artifact_manifest.json"
    write_json(manifest_path, artifact_manifest(output))
    verification = verify_artifact_manifest(output, manifest_path)
    if verification["status"] != "pass":
        raise RuntimeError(f"Artifact manifest verification failed: {verification}")
    return {
        "status": qa["status"],
        "run_dir": str(output),
        "report": str(report),
        "contract_sha256": contract_hash,
        "candidate_soft_log_loss": qa["candidate_soft_log_loss"],
        "direct_soft_log_loss": qa["direct_soft_log_loss"],
        "candidate_minus_direct_soft_log_loss": (
            qa["candidate_minus_direct_soft_log_loss"]
        ),
        "artifact_tree_sha256": verification["actual_tree_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run grouped-inner-OOF shrunk/calibrated direct Gower"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--parent-run")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_shrunk_calibrated_gower_sensitivity(
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
    "CALIBRATION_SHRINKAGE",
    "CONTRACT_ID",
    "CONTRACT_SHA256",
    "MODEL",
    "WEIGHT_SHRINKAGE",
    "apply_identity_shrunk_calibration",
    "build_grouped_inner_component_oof",
    "fit_identity_shrunk_calibrator",
    "fit_shrunk_calibrated_meta_model",
    "fit_shrunk_multiscale_weights",
    "fold_equal_weights",
    "load_shrunk_calibrated_gower_contract",
    "mean_soft_cross_entropy",
    "mix_component_probabilities",
    "run_shrunk_calibrated_gower_sensitivity",
]
