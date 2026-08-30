"""Prediction-oriented contiguous three-class posterior sensitivity.

This stage is independent of both the frozen four-class v4 analysis and the
completed classic-neighbourhood three-class v1 analysis.  The changed endpoint
assigns every posterior slope draw to one of three contiguous intervals:

* ``low_60_side``: slope < sqrt(60 * 120)
* ``120_side``: sqrt(60 * 120) <= slope <= 144
* ``high_off``: slope > 144

Within every outer-training split, the below-high gate and the low-versus-120
conditional stage are tuned independently.  Each stage selects one
configuration per model family using grouped inner OOF predictions, selects a
fold-local prior shrinkage coefficient, and fits a nonnegative convex OOF
stack that includes the fold-local prior.  Only then are the stage components
refitted on the complete outer-training split and evaluated on untouched
outer-test records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize_scalar
import yaml

from .cardinal_cv import (
    make_inner_soft_group_splits,
    make_soft_stratified_group_manifest,
)
from .cardinal_inputs import (
    assert_disjoint_cardinal_paths,
    snapshot_cardinal_inputs,
    verify_cardinal_input_snapshot,
    verify_source_run_unchanged,
)
from .cardinal_metrics import (
    calibration_table,
    expected_confusion_table,
    multiclass_brier,
    normalize_probabilities,
    soft_cross_entropy,
)
from .config import StudyConfig
from .contiguous_threeclass_labels import (
    CONTIGUOUS_THREE_CLASS_LABELS,
    CONTIGUOUS_THREE_CLASS_P_COLUMNS,
    CONTIGUOUS_THREE_CLASS_Q_COLUMNS,
    HIGH_OFF_BOUNDARY_MV_DEC,
    LOW_120_LOG_MIDPOINT_MV_DEC,
    contiguous_threeclass_support,
    derive_contiguous_threeclass_soft_labels,
)
from .feature_association_features import FROZEN_ASSOCIATION_PANELS
from .posterior_cardinal_classification import (
    IDENTITY_COLUMNS,
    _assert_fresh_disjoint_run_directory,
    _validate_run_frame,
)
from .posterior_cardinal_contract import (
    EXPECTED_FORMAL_PANELS,
    load_frozen_posterior_cardinal_contract,
    snapshot_reduced_feature_registries,
    verify_reduced_feature_registry_snapshot,
)
from .posterior_cardinal_inference import (
    descriptive_probability_metrics,
    paired_connected_group_bootstrap,
)
from .provenance import (
    directory_tree_summary,
    sha256_file,
    verify_artifact_manifest,
    write_json,
)
from .threeclass_sensitivity_models import (
    ConvexOOFProbabilityStacker,
    FoldLocalBinarySoftPriorClassifier,
    SENSITIVITY_CONDITIONAL_LABELS,
    SENSITIVITY_GATE_LABELS,
    combine_two_stage_probabilities,
    fit_predict_soft_binary_stage,
    prior_shrink_probabilities,
)


CONTRACT_FILENAME = "posterior_cardinal_contiguous_threeclass_v2.yaml"
CONTRACT_ID = "posterior_cardinal_contiguous_threeclass_v2"
STAGE_NAME = "posterior-cardinal-contiguous-threeclass"
CONFIG_SECTION = "posterior_cardinal_contiguous_threeclass"

CLASS_LABELS = CONTIGUOUS_THREE_CLASS_LABELS
Q_COLUMNS = CONTIGUOUS_THREE_CLASS_Q_COLUMNS
P_COLUMNS = CONTIGUOUS_THREE_CLASS_P_COLUMNS
SEARCH_PANELS = ("context_core", *EXPECTED_FORMAL_PANELS)
MODEL_FAMILIES = (
    "gaussian_prior_MAP",
    "soft_gower",
    "soft_extra_trees",
    "soft_catboost",
)
MODEL_COMPLEXITY_RANK = {
    "gaussian_prior_MAP": 1,
    "soft_gower": 2,
    "soft_extra_trees": 3,
    "soft_catboost": 4,
}
STAGES = ("gate", "conditional")
STAGE_SEED_OFFSETS = {"gate": 61_000, "conditional": 71_000}
PRIOR_MODEL = "fold_local_contiguous_threeclass_soft_prior"
STACK_MODEL = "stagewise_inner_oof_convex_stack"
MANIFEST_ID = "contiguous_threeclass_v2"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_contract(config: StudyConfig) -> tuple[dict[str, Any], Path, str]:
    settings = config.section(CONFIG_SECTION)
    if settings.get("contract_id") != CONTRACT_ID:
        raise ValueError("Configured contiguous three-class contract ID changed")
    if (
        settings.get("status")
        != "exploratory_posthoc_frozen_before_first_full_fit"
    ):
        raise ValueError("Configured contiguous three-class status changed")
    configured = Path(str(settings.get("contract_path", ""))).expanduser()
    path = (
        configured
        if configured.is_absolute()
        else (config.project_root / configured).resolve()
    )
    if path.name != CONTRACT_FILENAME or not path.is_file():
        raise FileNotFoundError(
            f"Contiguous three-class contract is missing or renamed: {path}"
        )
    expected_hash = str(settings.get("contract_sha256", "")).strip().lower()
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Contiguous three-class contract hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("Contiguous three-class contract must be a mapping")
    contract = dict(loaded)
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError("Contiguous three-class contract ID changed")
    if contract.get("stage") != STAGE_NAME:
        raise ValueError("Contiguous three-class contract stage changed")
    if (
        contract.get("status")
        != "exploratory_posthoc_frozen_before_first_full_fit"
    ):
        raise ValueError("Contiguous three-class contract status changed")

    scope = dict(contract.get("scientific_scope", {}))
    if (
        scope.get("population") != "all97"
        or int(scope.get("expected_records", -1)) != 97
        or int(scope.get("expected_connected_groups", -1)) != 80
        or scope.get("one_ml_row_per_record") is not True
        or scope.get("posterior_draws_expanded_as_samples") is not False
    ):
        raise ValueError("Contiguous three-class scientific scope changed")
    parent_boundary = dict(
        contract.get("immutable_parent_boundaries", {})
    )
    if (
        parent_boundary.get(
            "parent_artifact_manifests_verified_before_input_snapshot_or_"
            "modeling"
        )
        is not True
        or parent_boundary.get(
            "classic_fold_manifest_hashes_verified_before_modeling"
        )
        is not True
        or parent_boundary.get(
            "cli_preflight_disjoint_from_both_parent_result_roots"
        )
        is not True
    ):
        raise ValueError("Contiguous immutable-parent policy changed")

    label = dict(contract.get("label_scheme", {}))
    boundaries = dict(label.get("boundaries", {}))
    if tuple(label.get("supervised_class_order", ())) != CLASS_LABELS:
        raise ValueError("Contiguous three-class class order changed")
    if tuple(label.get("soft_label_columns", ())) != Q_COLUMNS:
        raise ValueError("Contiguous three-class soft-label columns changed")
    if tuple(label.get("prediction_columns", ())) != P_COLUMNS:
        raise ValueError("Contiguous three-class prediction columns changed")
    if not np.isclose(
        float(dict(boundaries.get("low_60_side", {}))["upper_exclusive"]),
        LOW_120_LOG_MIDPOINT_MV_DEC,
        atol=0,
        rtol=1e-15,
    ):
        raise ValueError("Contiguous lower boundary changed")
    if float(dict(boundaries.get("120_side", {}))["upper_inclusive"]) != (
        HIGH_OFF_BOUNDARY_MV_DEC
    ):
        raise ValueError("Contiguous upper boundary changed")
    if label.get("posterior_argmax_training_or_tuning_use") != "forbidden":
        raise ValueError("Posterior argmax must remain forbidden")

    features = dict(contract.get("feature_engineering", {}))
    if tuple(features.get("selectable_panels", ())) != SEARCH_PANELS:
        raise ValueError("Contiguous feature-panel search space changed")
    models = dict(contract.get("models", {}))
    if tuple(models.get("selectable_families", ())) != MODEL_FAMILIES:
        raise ValueError("Contiguous model-family search space changed")
    model_seeding = dict(models.get("stage_model_seeding", {}))
    if (
        model_seeding.get("formula")
        != "outer_split_seed + stage_seed_offset + trial_number"
        or int(model_seeding.get("gate_stage_seed_offset", -1))
        != STAGE_SEED_OFFSETS["gate"]
        or int(model_seeding.get("conditional_stage_seed_offset", -1))
        != STAGE_SEED_OFFSETS["conditional"]
        or model_seeding.get(
            "same_seed_reused_across_inner_folds_for_one_trial"
        )
        is not True
    ):
        raise ValueError("Contiguous stage model-seeding contract changed")
    if (
        dict(models.get("soft_catboost", {})).get(
            "zero_weight_records_excluded_before_fit"
        )
        is not True
    ):
        raise ValueError("Contiguous CatBoost zero-weight policy changed")
    inner = dict(contract.get("inner_validation", {}))
    if (
        tuple(inner.get("separate_studies", ()))
        != ("non_high_vs_high_off_gate", "low_60_vs_120_conditional")
        or inner.get("all_prior_shrinkage_selected_on_inner_oof_only") is not True
        or inner.get("all_convex_stack_weights_selected_on_inner_oof_only")
        is not True
        or inner.get(
            "zero_conditional_weight_records_excluded_before_"
            "feature_engineering_and_fit"
        )
        is not True
    ):
        raise ValueError("Stagewise inner-validation contract changed")
    optuna_settings = dict(contract.get("optuna", {}))
    if (
        optuna_settings.get("sampler") != "TPESampler"
        or optuna_settings.get("pruner") != "NopPruner"
        or optuna_settings.get("study_n_jobs") != 1
        or optuna_settings.get("outer_test_predictions_visible_to_study")
        is not False
    ):
        raise ValueError("Contiguous Optuna contract changed")
    expected_gower_coverage = [
        {
            "panel": "context_core",
            "neighbor_mode": "exponential_kernel",
            "bandwidth": 0.50,
        },
        {
            "panel": "context_core",
            "neighbor_mode": "rbf_kernel",
            "bandwidth": 0.50,
        },
        {
            "panel": "context_plus_water_organic",
            "neighbor_mode": "exponential_kernel",
            "bandwidth": 0.50,
            "block_weight_profile": "context_2x",
        },
        {
            "panel": "context_plus_water_organic",
            "neighbor_mode": "rbf_kernel",
            "bandwidth": 0.50,
            "block_weight_profile": "formulation_2x",
        },
    ]
    if (
        optuna_settings.get(
            "enqueue_gower_structural_coverage_before_TPE"
        )
        != expected_gower_coverage
        or int(
            optuna_settings.get(
                "full_unenqueued_TPE_trials_per_stage_per_outer_split",
                -1,
            )
        )
        != 4
    ):
        raise ValueError("Contiguous Gower structural coverage changed")
    stack = dict(contract.get("inner_oof_convex_stack", {}))
    if (
        stack.get("fitted_separately_for_gate_and_conditional") is not True
        or dict(stack.get("constraints", {})).get("nonnegative_weights")
        is not True
        or stack.get("fallback") != "fold_local_stage_prior"
        or float(stack.get("score_clip_epsilon", -1)) != 1e-15
        or stack.get(
            "candidates_equal_to_prior_collapsed_before_weight_fit"
        )
        is not True
        or float(stack.get("collapsed_candidate_reported_weight", -1))
        != 0.0
    ):
        raise ValueError("Contiguous inner-OOF stack contract changed")
    return contract, path, actual_hash


def resolve_contiguous_threeclass_v4_result_root(config: StudyConfig) -> Path:
    contract, _, _ = _load_contract(config)
    boundary = dict(contract["immutable_parent_boundaries"])
    root = Path(str(boundary["source_run"])).expanduser()
    if not root.is_absolute():
        root = (config.repository_root / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Bound v4 result root is missing: {root}")
    return root


def resolve_contiguous_threeclass_classic_result_root(
    config: StudyConfig,
) -> Path:
    contract, _, _ = _load_contract(config)
    boundary = dict(contract["immutable_parent_boundaries"])
    root = Path(
        str(boundary["classic_threeclass_source_run"])
    ).expanduser()
    if not root.is_absolute():
        root = (config.repository_root / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Bound classic three-class result root is missing: {root}"
        )
    return root


def _verify_parent_artifact_manifest(
    root: Path,
    *,
    label: str,
) -> dict[str, Any]:
    manifest = root / "artifact_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Bound {label} artifact manifest is missing: {manifest}"
        )
    verification = verify_artifact_manifest(root, manifest)
    if verification.get("status") != "pass":
        raise ValueError(
            f"Bound {label} artifact manifest verification failed: "
            f"{verification}"
        )
    return dict(verification)


def _verify_bound_classic_fold_manifests(
    root: Path,
    boundary: Mapping[str, Any],
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for key in ("classic_outer_manifest", "classic_inner_manifest"):
        definition = dict(boundary[key])
        path = (root / str(definition["path"])).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(
                f"Bound classic fold manifest is missing: {path}"
            )
        expected = str(definition["sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Bound {key} hash changed: expected={expected}, "
                f"actual={actual}"
            )
        files[key] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": int(path.stat().st_size),
        }
    return {"status": "pass", "files": files}


def _records_by_id(records: pd.DataFrame) -> pd.DataFrame:
    output = records.copy()
    output["record_id"] = output["record_id"].astype(str)
    if output["record_id"].duplicated().any():
        raise ValueError("Records must be unique by record_id")
    return output.set_index("record_id", drop=False)


def _labels_by_id(labels: pd.DataFrame) -> pd.DataFrame:
    selected = labels.loc[:, ["record_id", *Q_COLUMNS]].copy()
    selected["record_id"] = selected["record_id"].astype(str)
    if selected["record_id"].duplicated().any():
        raise ValueError("Labels must be unique by record_id")
    return selected.set_index("record_id")


def _primary_frame(records: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    frame = records.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    selected = labels.loc[:, ["record_id", *Q_COLUMNS]].copy()
    selected["record_id"] = selected["record_id"].astype(str)
    merged = frame.merge(selected, on="record_id", how="left", validate="one_to_one")
    if merged.loc[:, Q_COLUMNS].isna().any().any():
        raise ValueError("Contiguous labels did not cover all records")
    return merged


def build_contiguous_outer_manifest(
    primary_frame: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    smoke: bool,
) -> pd.DataFrame:
    outer = dict(contract["outer_validation"])
    smoke_settings = dict(contract["smoke_mode"])
    folds = int(smoke_settings["outer_folds"]) if smoke else int(outer["folds"])
    repeats = (
        int(smoke_settings["outer_repeats"])
        if smoke
        else int(outer["repeats"])
    )
    seeds = tuple(int(value) for value in outer["repeat_seeds"])[:repeats]
    manifest = make_soft_stratified_group_manifest(
        primary_frame,
        Q_COLUMNS,
        group_col="cv_group_id",
        record_id_col="record_id",
        folds=folds,
        repeats=repeats,
        repeat_seeds=seeds,
    )
    manifest.insert(0, "manifest_id", MANIFEST_ID)
    return manifest


def build_contiguous_inner_manifest(
    primary_frame: pd.DataFrame,
    outer_manifest: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    smoke: bool,
) -> pd.DataFrame:
    inner_folds = (
        int(contract["smoke_mode"]["inner_folds"])
        if smoke
        else int(contract["inner_validation"]["folds"])
    )
    rows: list[dict[str, Any]] = []
    for (repeat, fold), outer_test in outer_manifest.groupby(
        ["repeat", "fold"], sort=True
    ):
        test_ids = set(outer_test["record_id"].astype(str))
        train = primary_frame.loc[
            ~primary_frame["record_id"].astype(str).isin(test_ids)
        ].reset_index(drop=True)
        split_seed = int(outer_test["split_seed"].iloc[0]) + 10_000 + int(fold)
        _, assignments = make_inner_soft_group_splits(
            train,
            Q_COLUMNS,
            group_col="cv_group_id",
            folds=inner_folds,
            random_seed=split_seed,
        )
        for position, inner_fold in enumerate(assignments):
            rows.append(
                {
                    "manifest_id": MANIFEST_ID,
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "inner_fold": int(inner_fold),
                    "record_id": str(train.iloc[position]["record_id"]),
                    "cv_group_id": str(train.iloc[position]["cv_group_id"]),
                    "inner_split_seed": int(split_seed),
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["outer_repeat", "outer_fold", "inner_fold", "record_id"],
        kind="stable",
    ).reset_index(drop=True)
    if (
        manifest.groupby(
            ["outer_repeat", "outer_fold", "cv_group_id"]
        )["inner_fold"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError("A connected group crosses inner folds")
    return manifest


def _manifest_inner_splits(
    outer_train_ids: Sequence[str],
    inner_manifest: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
) -> list[tuple[list[str], list[str], int]]:
    selected = inner_manifest.loc[
        inner_manifest["outer_repeat"].eq(int(repeat))
        & inner_manifest["outer_fold"].eq(int(fold))
    ]
    lookup = dict(
        zip(
            selected["record_id"].astype(str),
            selected["inner_fold"].astype(int),
            strict=True,
        )
    )
    ordered = [str(value) for value in outer_train_ids]
    if set(lookup) != set(ordered):
        raise ValueError("Inner manifest does not match outer-training records")
    assignments = np.asarray([lookup[value] for value in ordered], dtype=int)
    output: list[tuple[list[str], list[str], int]] = []
    for inner_fold in sorted(np.unique(assignments)):
        validation = [
            value
            for value, assignment in zip(ordered, assignments, strict=True)
            if int(assignment) == int(inner_fold)
        ]
        training = [
            value
            for value, assignment in zip(ordered, assignments, strict=True)
            if int(assignment) != int(inner_fold)
        ]
        if not training or not validation:
            raise ValueError("Grouped inner split contains an empty partition")
        output.append((training, validation, int(inner_fold)))
    return output


def _stage_targets(
    labels_indexed: pd.DataFrame,
    record_ids: Sequence[str],
    *,
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    q = labels_indexed.loc[list(record_ids), list(Q_COLUMNS)].to_numpy(float)
    non_high = q[:, :2].sum(axis=1)
    if stage == "gate":
        return non_high, np.ones(len(q), dtype=float)
    if stage != "conditional":
        raise KeyError(f"Unknown stage {stage!r}")
    conditional = np.divide(
        q[:, 0],
        non_high,
        out=np.full(len(q), 0.5, dtype=float),
        where=non_high > 0,
    )
    return conditional, non_high


def _stage_class_names(stage: str) -> tuple[str, str]:
    if stage == "gate":
        return SENSITIVITY_GATE_LABELS
    if stage == "conditional":
        return SENSITIVITY_CONDITIONAL_LABELS
    raise KeyError(f"Unknown stage {stage!r}")


def _binary_matrix(first_probability: np.ndarray) -> np.ndarray:
    values = np.asarray(first_probability, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Binary first-class probabilities must lie in [0, 1]")
    return np.column_stack([values, 1.0 - values])


def _binary_soft_loss(
    target_first: np.ndarray,
    predicted_first: np.ndarray,
    sample_weight: np.ndarray,
    *,
    epsilon: float = 1e-15,
) -> float:
    target = _binary_matrix(target_first)
    predicted = _binary_matrix(predicted_first)
    return soft_cross_entropy(
        target,
        predicted,
        sample_weight=np.asarray(sample_weight, dtype=float),
        epsilon=float(epsilon),
    )


def _fold_equal_weights(
    base_weight: np.ndarray,
    fold_ids: np.ndarray,
) -> np.ndarray:
    base = np.asarray(base_weight, dtype=float).reshape(-1)
    folds = np.asarray(fold_ids, dtype=int).reshape(-1)
    if base.shape != folds.shape or np.any(base < 0) or not np.isfinite(base).all():
        raise ValueError("Fold-equal weights received invalid arrays")
    output = np.zeros_like(base)
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        total = float(base[selected].sum())
        if total <= 0:
            raise ValueError(f"Inner fold {fold} has zero validation weight")
        output[selected] = base[selected] / total
    return output


def _optimize_prior_alpha(
    model_first: np.ndarray,
    prior_first: np.ndarray,
    target_first: np.ndarray,
    fold_equal_weight: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    model = _binary_matrix(model_first)
    prior = _binary_matrix(prior_first)

    def objective(alpha: float) -> float:
        mixed = prior_shrink_probabilities(model, prior, float(alpha))
        return soft_cross_entropy(
            _binary_matrix(target_first),
            mixed,
            sample_weight=fold_equal_weight,
            epsilon=1e-15,
        )

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    candidates: list[tuple[float, float, str]] = [
        (objective(0.0), 0.0, "explicit_prior_endpoint"),
        (objective(1.0), 1.0, "explicit_model_endpoint"),
    ]
    if bool(result.success) and np.isfinite(float(result.fun)):
        candidates.append(
            (objective(float(result.x)), float(result.x), "bounded_interior")
        )
    loss, alpha, source = min(candidates, key=lambda row: (row[0], row[1]))
    mixed, mixture_audit = prior_shrink_probabilities(
        model, prior, alpha, return_audit=True
    )
    return (
        float(alpha),
        mixed[:, 0],
        {
            "alpha": float(alpha),
            "objective_soft_log_loss": float(loss),
            "selection_source": source,
            "tie_break": "lower_alpha_toward_prior",
            "bounded_optimizer_success": bool(result.success),
            "bounded_optimizer_message": str(result.message),
            "bounded_optimizer_x": (
                float(result.x) if np.isfinite(float(result.x)) else None
            ),
            "endpoint_losses": {
                "alpha_0_prior": float(candidates[0][0]),
                "alpha_1_model": float(candidates[1][0]),
            },
            "mixture": mixture_audit,
        },
    )


def _suggest_trial_configuration(
    trial: optuna.Trial,
) -> tuple[str, str, dict[str, Any]]:
    panel = str(trial.suggest_categorical("panel", list(SEARCH_PANELS)))
    family = str(
        trial.suggest_categorical("model_family", list(MODEL_FAMILIES))
    )
    if family == "gaussian_prior_MAP":
        params = {
            "feature_prior_scale": trial.suggest_float(
                "feature_prior_scale", 0.10, 4.0, log=True
            ),
            "intercept_prior_scale": 5.0,
            "max_iter": 1000,
            "tolerance": 1e-8,
        }
    elif family == "soft_gower":
        mode = str(
            trial.suggest_categorical(
                "neighbor_mode",
                ["knn", "exponential_kernel", "rbf_kernel"],
            )
        )
        params = {
            "neighbor_mode": mode,
            "epsilon": 1e-12,
            "block_weight_profile": (
                "equal"
                if panel == "context_core"
                else str(
                    trial.suggest_categorical(
                        "block_weight_profile",
                        ["equal", "context_2x", "formulation_2x"],
                    )
                )
            ),
        }
        if mode == "knn":
            params.update(
                {
                    "n_neighbors": int(
                        trial.suggest_categorical(
                            "n_neighbors", [3, 5, 7, 9, 11, 15, 21]
                        )
                    ),
                    "distance_power": float(
                        trial.suggest_categorical(
                            "distance_power", [1.0, 2.0]
                        )
                    ),
                    "bandwidth": 1.0,
                }
            )
        else:
            params.update(
                {
                    "n_neighbors": 9,
                    "distance_power": 1.0,
                    "bandwidth": float(
                        trial.suggest_categorical(
                            "bandwidth", [0.10, 0.25, 0.50, 1.00, 2.00]
                        )
                    ),
                }
            )
    elif family == "soft_extra_trees":
        params = {
            "n_estimators": int(
                trial.suggest_int("n_estimators", 300, 800, step=100)
            ),
            "max_depth": int(trial.suggest_int("max_depth", 2, 5)),
            "min_samples_leaf": int(
                trial.suggest_int("min_samples_leaf", 3, 12)
            ),
            "max_features": float(
                trial.suggest_float("max_features", 0.40, 1.00)
            ),
        }
    else:
        params = {
            "iterations": int(
                trial.suggest_int("iterations", 100, 500, step=100)
            ),
            "depth": int(trial.suggest_int("depth", 2, 4)),
            "learning_rate": float(
                trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
            ),
            "l2_leaf_reg": float(
                trial.suggest_float("l2_leaf_reg", 3.0, 50.0, log=True)
            ),
            "border_count": 32,
            "random_strength": 0.0,
        }
    return panel, family, params


def _default_trial(
    panel: str,
    family: str,
) -> dict[str, Any]:
    base: dict[str, Any] = {"panel": panel, "model_family": family}
    if family == "gaussian_prior_MAP":
        return {**base, "feature_prior_scale": 1.0}
    if family == "soft_gower":
        output = {
            **base,
            "neighbor_mode": "knn",
            "n_neighbors": 9,
            "distance_power": 1.0,
        }
        if panel != "context_core":
            output["block_weight_profile"] = "equal"
        return output
    if family == "soft_extra_trees":
        return {
            **base,
            "n_estimators": 400,
            "max_depth": 3,
            "min_samples_leaf": 6,
            "max_features": 0.7,
        }
    return {
        **base,
        "iterations": 200,
        "depth": 3,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
    }


def _default_enqueued_candidates(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        smoke_panels = {
            "gaussian_prior_MAP": "context_core",
            "soft_gower": "context_plus_water_organic",
            "soft_extra_trees": "context_plus_polarity",
            "soft_catboost": "context_plus_ionomer_ratio",
        }
        return [
            _default_trial(smoke_panels[family], family)
            for family in MODEL_FAMILIES
        ]
    panel_family_defaults = [
        _default_trial(panel, family)
        for panel in SEARCH_PANELS
        for family in MODEL_FAMILIES
    ]
    gower_structural_coverage = [
        {
            "panel": "context_core",
            "model_family": "soft_gower",
            "neighbor_mode": "exponential_kernel",
            "bandwidth": 0.50,
        },
        {
            "panel": "context_core",
            "model_family": "soft_gower",
            "neighbor_mode": "rbf_kernel",
            "bandwidth": 0.50,
        },
        {
            "panel": "context_plus_water_organic",
            "model_family": "soft_gower",
            "neighbor_mode": "exponential_kernel",
            "bandwidth": 0.50,
            "block_weight_profile": "context_2x",
        },
        {
            "panel": "context_plus_water_organic",
            "model_family": "soft_gower",
            "neighbor_mode": "rbf_kernel",
            "bandwidth": 0.50,
            "block_weight_profile": "formulation_2x",
        },
    ]
    return [*panel_family_defaults, *gower_structural_coverage]


def _crossfit_stage_configuration(
    records_indexed: pd.DataFrame,
    labels_indexed: pd.DataFrame,
    outer_train_ids: Sequence[str],
    inner_splits: Sequence[tuple[list[str], list[str], int]],
    *,
    stage: str,
    panel: str,
    model_family: str,
    params: Mapping[str, Any],
    model_seed: int,
    rare_min_frequency: int,
) -> dict[str, Any]:
    ordered = [str(value) for value in outer_train_ids]
    position = {record_id: index for index, record_id in enumerate(ordered)}
    target, base_weight = _stage_targets(
        labels_indexed, ordered, stage=stage
    )
    model_first = np.full(len(ordered), np.nan, dtype=float)
    prior_first = np.full(len(ordered), np.nan, dtype=float)
    fold_ids = np.full(len(ordered), -1, dtype=int)
    fold_audits: list[dict[str, Any]] = []
    for train_ids, validation_ids, inner_fold in inner_splits:
        train_target, train_weight = _stage_targets(
            labels_indexed, train_ids, stage=stage
        )
        validation_target, validation_weight = _stage_targets(
            labels_indexed, validation_ids, stage=stage
        )
        prediction, fit_audit = fit_predict_soft_binary_stage(
            records_indexed.loc[list(train_ids)].reset_index(drop=True),
            train_target,
            records_indexed.loc[list(validation_ids)].reset_index(drop=True),
            panel=panel,
            model_family=model_family,
            params=params,
            seed=int(model_seed),
            sample_weight=train_weight,
            rare_min_frequency=int(rare_min_frequency),
        )
        prior_estimator = FoldLocalBinarySoftPriorClassifier(
            class_names=_stage_class_names(stage)
        ).fit(
            records_indexed.loc[list(train_ids)].reset_index(drop=True),
            train_target,
            sample_weight=train_weight,
        )
        prior_prediction = prior_estimator.predict_proba(
            records_indexed.loc[list(validation_ids)].reset_index(drop=True)
        )[:, 0]
        validation_positions = np.asarray(
            [position[value] for value in validation_ids], dtype=int
        )
        model_first[validation_positions] = prediction
        prior_first[validation_positions] = prior_prediction
        fold_ids[validation_positions] = int(inner_fold)
        fold_audits.append(
            {
                "inner_fold": int(inner_fold),
                "n_training_records": int(len(train_ids)),
                "n_validation_records": int(len(validation_ids)),
                "validation_weight_sum": float(validation_weight.sum()),
                "raw_model_soft_log_loss": _binary_soft_loss(
                    validation_target,
                    prediction,
                    validation_weight,
                ),
                "prior_soft_log_loss": _binary_soft_loss(
                    validation_target,
                    prior_prediction,
                    validation_weight,
                ),
                "fit_audit": fit_audit,
                "prior_audit": prior_estimator.fit_audit_,
            }
        )
    if (
        not np.isfinite(model_first).all()
        or not np.isfinite(prior_first).all()
        or np.any(fold_ids < 0)
    ):
        raise AssertionError("Inner OOF stage predictions are incomplete")
    fold_equal_weight = _fold_equal_weights(base_weight, fold_ids)
    alpha, shrunk_first, alpha_audit = _optimize_prior_alpha(
        model_first,
        prior_first,
        target,
        fold_equal_weight,
    )
    raw_loss = _binary_soft_loss(
        target, model_first, fold_equal_weight
    )
    prior_loss = _binary_soft_loss(
        target, prior_first, fold_equal_weight
    )
    shrunk_loss = _binary_soft_loss(
        target, shrunk_first, fold_equal_weight
    )
    for row in fold_audits:
        selected = fold_ids == int(row["inner_fold"])
        row["shrunk_soft_log_loss"] = _binary_soft_loss(
            target[selected],
            shrunk_first[selected],
            base_weight[selected],
        )
        row["prior_shrinkage_alpha"] = float(alpha)
    return {
        "target_first": target,
        "base_weight": base_weight,
        "fold_equal_weight": fold_equal_weight,
        "fold_ids": fold_ids,
        "model_first": model_first,
        "prior_first": prior_first,
        "shrunk_first": shrunk_first,
        "alpha": float(alpha),
        "raw_model_soft_log_loss": float(raw_loss),
        "prior_soft_log_loss": float(prior_loss),
        "shrunk_soft_log_loss": float(shrunk_loss),
        "alpha_audit": alpha_audit,
        "fold_audits": fold_audits,
    }


def _trial_simplicity_key(
    trial: optuna.trial.FrozenTrial,
) -> tuple[Any, ...]:
    panel = str(trial.params["panel"])
    family = str(trial.params["model_family"])
    return (
        MODEL_COMPLEXITY_RANK[family],
        int(FROZEN_ASSOCIATION_PANELS[panel].complexity_rank),
        int(trial.number),
    )


def _best_trial_per_family(
    study: optuna.Study,
) -> dict[str, optuna.trial.FrozenTrial]:
    output: dict[str, optuna.trial.FrozenTrial] = {}
    for family in MODEL_FAMILIES:
        completed = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(float(trial.value))
            and trial.params.get("model_family") == family
        ]
        if not completed:
            raise RuntimeError(
                f"No complete inner-OOF trial exists for family {family}"
            )
        output[family] = min(
            completed,
            key=lambda trial: (
                float(trial.value),
                _trial_simplicity_key(trial),
            ),
        )
    return output


def _run_stage_study(
    records_indexed: pd.DataFrame,
    labels_indexed: pd.DataFrame,
    outer_train_ids: Sequence[str],
    inner_splits: Sequence[tuple[list[str], list[str], int]],
    *,
    stage: str,
    repeat: int,
    fold: int,
    n_trials: int,
    sampler_seed: int,
    outer_split_seed: int,
    smoke: bool,
    rare_min_frequency: int,
) -> tuple[
    optuna.Study,
    dict[str, optuna.trial.FrozenTrial],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    sampler = optuna.samplers.TPESampler(
        seed=int(sampler_seed),
        multivariate=False,
        n_startup_trials=min(10, max(1, int(n_trials))),
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        study_name=f"{stage}_repeat_{repeat:02d}_fold_{fold:02d}",
    )
    defaults = _default_enqueued_candidates(smoke=smoke)
    for candidate in defaults[: int(n_trials)]:
        study.enqueue_trial(candidate)

    def objective(trial: optuna.Trial) -> float:
        panel, family, params = _suggest_trial_configuration(trial)
        model_seed = (
            int(outer_split_seed)
            + int(STAGE_SEED_OFFSETS[stage])
            + int(trial.number)
        )
        try:
            crossfit = _crossfit_stage_configuration(
                records_indexed,
                labels_indexed,
                outer_train_ids,
                inner_splits,
                stage=stage,
                panel=panel,
                model_family=family,
                params=params,
                model_seed=model_seed,
                rare_min_frequency=rare_min_frequency,
            )
        except Exception as error:
            trial.set_user_attr("failure", repr(error))
            raise RuntimeError(repr(error)) from error
        trial.set_user_attr(
            "resolved_model_params_json",
            json.dumps(_jsonable(params), sort_keys=True),
        )
        trial.set_user_attr("prior_shrinkage_alpha", crossfit["alpha"])
        trial.set_user_attr(
            "raw_model_soft_log_loss",
            crossfit["raw_model_soft_log_loss"],
        )
        trial.set_user_attr(
            "prior_soft_log_loss",
            crossfit["prior_soft_log_loss"],
        )
        trial.set_user_attr(
            "alpha_audit_json",
            json.dumps(
                _jsonable(crossfit["alpha_audit"]), sort_keys=True
            ),
        )
        trial.set_user_attr(
            "inner_fold_scores",
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"fit_audit", "prior_audit"}
                }
                for row in crossfit["fold_audits"]
            ],
        )
        return float(crossfit["shrunk_soft_log_loss"])

    study.optimize(
        objective,
        n_trials=int(n_trials),
        n_jobs=1,
        show_progress_bar=False,
        catch=(RuntimeError,),
    )
    selected = _best_trial_per_family(study)
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        trial_rows.append(
            {
                "outer_repeat": int(repeat),
                "outer_fold": int(fold),
                "stage": stage,
                "trial_number": int(trial.number),
                "state": trial.state.name.lower(),
                "objective_shrunk_soft_log_loss": (
                    float(trial.value)
                    if trial.value is not None
                    else np.nan
                ),
                "panel": trial.params.get("panel"),
                "model_family": trial.params.get("model_family"),
                "parameters_json": json.dumps(
                    _jsonable(trial.params), sort_keys=True
                ),
                "resolved_model_params_json": trial.user_attrs.get(
                    "resolved_model_params_json"
                ),
                "prior_shrinkage_alpha": trial.user_attrs.get(
                    "prior_shrinkage_alpha", np.nan
                ),
                "raw_model_soft_log_loss": trial.user_attrs.get(
                    "raw_model_soft_log_loss", np.nan
                ),
                "prior_soft_log_loss": trial.user_attrs.get(
                    "prior_soft_log_loss", np.nan
                ),
                "failure": trial.user_attrs.get("failure"),
            }
        )
        for item in trial.user_attrs.get("inner_fold_scores", ()):
            fold_rows.append(
                {
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "stage": stage,
                    "trial_number": int(trial.number),
                    **dict(item),
                }
            )
    return study, selected, trial_rows, fold_rows


def _selected_trial_configuration(
    trial: optuna.trial.FrozenTrial,
) -> tuple[str, str, dict[str, Any]]:
    payload = trial.user_attrs.get("resolved_model_params_json")
    if not isinstance(payload, str):
        raise ValueError("Selected trial is missing resolved model parameters")
    params = json.loads(payload)
    if not isinstance(params, dict):
        raise TypeError("Selected model parameter payload must be a mapping")
    return (
        str(trial.params["panel"]),
        str(trial.params["model_family"]),
        dict(params),
    )


def _fit_inner_stage_stack(
    records_indexed: pd.DataFrame,
    labels_indexed: pd.DataFrame,
    outer_train_ids: Sequence[str],
    inner_splits: Sequence[tuple[list[str], list[str], int]],
    selected_trials: Mapping[str, optuna.trial.FrozenTrial],
    *,
    stage: str,
    repeat: int,
    fold: int,
    outer_split_seed: int,
    rare_min_frequency: int,
) -> dict[str, Any]:
    ordered = [str(value) for value in outer_train_ids]
    candidate_probabilities: dict[str, np.ndarray] = {}
    component_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_configs: dict[str, dict[str, Any]] = {}
    collapsed_candidates: dict[str, str] = {}
    reference: dict[str, Any] | None = None
    for family in MODEL_FAMILIES:
        trial = selected_trials[family]
        panel, selected_family, params = _selected_trial_configuration(trial)
        if selected_family != family:
            raise AssertionError("Selected family registry is inconsistent")
        model_seed = (
            int(outer_split_seed)
            + int(STAGE_SEED_OFFSETS[stage])
            + int(trial.number)
        )
        crossfit = _crossfit_stage_configuration(
            records_indexed,
            labels_indexed,
            ordered,
            inner_splits,
            stage=stage,
            panel=panel,
            model_family=family,
            params=params,
            model_seed=model_seed,
            rare_min_frequency=rare_min_frequency,
        )
        expected_alpha = float(
            trial.user_attrs.get("prior_shrinkage_alpha", np.nan)
        )
        if not np.isclose(
            crossfit["alpha"], expected_alpha, atol=1e-8, rtol=0
        ):
            raise AssertionError("Selected trial alpha did not reproduce")
        if not np.isclose(
            crossfit["shrunk_soft_log_loss"],
            float(trial.value),
            atol=1e-8,
            rtol=0,
        ):
            raise AssertionError("Selected trial OOF loss did not reproduce")
        if reference is None:
            reference = crossfit
            candidate_probabilities["prior"] = _binary_matrix(
                crossfit["prior_first"]
            )
        else:
            for field in (
                "target_first",
                "base_weight",
                "fold_equal_weight",
                "fold_ids",
                "prior_first",
            ):
                if not np.allclose(
                    np.asarray(crossfit[field]),
                    np.asarray(reference[field]),
                    atol=1e-12,
                    rtol=0,
                ):
                    raise AssertionError(
                        f"Selected stage candidates disagree on {field}"
                    )
        family_probabilities = _binary_matrix(
            crossfit["shrunk_first"]
        )
        collapsed_to_prior = bool(
            np.allclose(
                family_probabilities,
                candidate_probabilities["prior"],
                atol=1e-15,
                rtol=0,
            )
        )
        if collapsed_to_prior:
            collapsed_candidates[family] = (
                "inner_oof_shrunk_predictions_equal_fold_local_prior"
            )
        else:
            candidate_probabilities[family] = family_probabilities
        selected_configs[family] = {
            "panel": panel,
            "model_family": family,
            "params": params,
            "model_seed": int(model_seed),
            "trial_number": int(trial.number),
            "alpha": float(crossfit["alpha"]),
        }
        alpha_rows.append(
            {
                "outer_repeat": int(repeat),
                "outer_fold": int(fold),
                "stage": stage,
                "model_family": family,
                "panel": panel,
                "trial_number": int(trial.number),
                "alpha": float(crossfit["alpha"]),
                "raw_model_soft_log_loss": float(
                    crossfit["raw_model_soft_log_loss"]
                ),
                "prior_soft_log_loss": float(
                    crossfit["prior_soft_log_loss"]
                ),
                "shrunk_soft_log_loss": float(
                    crossfit["shrunk_soft_log_loss"]
                ),
                "alpha_audit_json": json.dumps(
                    _jsonable(crossfit["alpha_audit"]), sort_keys=True
                ),
                "collapsed_to_prior_before_stack": collapsed_to_prior,
            }
        )
        candidate_rows.append(
            {
                "outer_repeat": int(repeat),
                "outer_fold": int(fold),
                "stage": stage,
                "model_family": family,
                "panel": panel,
                "trial_number": int(trial.number),
                "parameters_json": json.dumps(
                    _jsonable(params), sort_keys=True
                ),
                "alpha": float(crossfit["alpha"]),
                "inner_oof_soft_log_loss": float(
                    crossfit["shrunk_soft_log_loss"]
                ),
                "model_seed": int(model_seed),
                "active_stack_candidate": not collapsed_to_prior,
                "collapsed_to_prior_before_stack": collapsed_to_prior,
                "collapse_reason": collapsed_candidates.get(family, ""),
            }
        )
        for record_id, inner_fold, raw, prior, shrunk in zip(
            ordered,
            crossfit["fold_ids"],
            crossfit["model_first"],
            crossfit["prior_first"],
            crossfit["shrunk_first"],
            strict=True,
        ):
            component_rows.append(
                {
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "stage": stage,
                    "record_id": record_id,
                    "inner_fold": int(inner_fold),
                    "model_family": family,
                    "panel": panel,
                    "trial_number": int(trial.number),
                    "raw_model_p_class0": float(raw),
                    "fold_local_prior_p_class0": float(prior),
                    "shrunk_p_class0": float(shrunk),
                    "alpha": float(crossfit["alpha"]),
                    "collapsed_to_prior_before_stack": (
                        collapsed_to_prior
                    ),
                }
            )
    if reference is None:
        raise AssertionError("No selected stage candidates were available")
    stacker = ConvexOOFProbabilityStacker(
        epsilon=1e-15,
        max_iter=1000,
        tolerance=1e-10,
        acceptance_tolerance=1e-10,
        fallback_candidate="prior",
    ).fit(
        candidate_probabilities,
        _binary_matrix(reference["target_first"]),
        sample_weight=reference["fold_equal_weight"],
    )
    stacked_inner = stacker.predict_proba(candidate_probabilities)
    stack_loss = soft_cross_entropy(
        _binary_matrix(reference["target_first"]),
        stacked_inner,
        sample_weight=reference["fold_equal_weight"],
        epsilon=1e-15,
    )
    stacker.fit_audit_["collapsed_candidates"] = dict(
        collapsed_candidates
    )
    fitted_weights = dict(stacker.fit_audit_["candidate_weights"])
    weights = {
        candidate: float(fitted_weights.get(candidate, 0.0))
        for candidate in ("prior", *MODEL_FAMILIES)
    }
    stacker.fit_audit_[
        "reported_candidate_weights_including_collapsed"
    ] = dict(weights)
    stack_weight_rows = [
        {
            "outer_repeat": int(repeat),
            "outer_fold": int(fold),
            "stage": stage,
            "candidate": candidate,
            "weight": float(weight),
            "dominant": False,
            "stack_soft_log_loss": float(stack_loss),
            "fallback_used": bool(stacker.fit_audit_["fallback_used"]),
            "active_stack_candidate": candidate in fitted_weights,
            "collapsed_to_prior_before_stack": (
                candidate in collapsed_candidates
            ),
            "collapse_reason": collapsed_candidates.get(candidate, ""),
        }
        for candidate, weight in weights.items()
    ]
    dominant = min(
        weights,
        key=lambda candidate: (-float(weights[candidate]), candidate),
    )
    for row in stack_weight_rows:
        row["dominant"] = row["candidate"] == dominant
    return {
        "stacker": stacker,
        "selected_configs": selected_configs,
        "component_rows": component_rows,
        "alpha_rows": alpha_rows,
        "candidate_rows": candidate_rows,
        "stack_weight_rows": stack_weight_rows,
        "stack_audit": stacker.fit_audit_,
        "dominant_candidate": dominant,
        "inner_oof_stack_soft_log_loss": float(stack_loss),
    }


def _predict_outer_stage(
    records_indexed: pd.DataFrame,
    labels_indexed: pd.DataFrame,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    *,
    stage: str,
    selected_configs: Mapping[str, Mapping[str, Any]],
    stacker: ConvexOOFProbabilityStacker,
    rare_min_frequency: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    train_target, train_weight = _stage_targets(
        labels_indexed, train_ids, stage=stage
    )
    prior_estimator = FoldLocalBinarySoftPriorClassifier(
        class_names=_stage_class_names(stage)
    ).fit(
        records_indexed.loc[list(train_ids)].reset_index(drop=True),
        train_target,
        sample_weight=train_weight,
    )
    prior = prior_estimator.predict_proba(
        records_indexed.loc[list(test_ids)].reset_index(drop=True)
    )
    candidates: dict[str, np.ndarray] = {"prior": prior}
    fit_rows: list[dict[str, Any]] = []
    active_candidates = set(stacker.candidate_names_)
    collapsed_candidates = dict(
        stacker.fit_audit_.get("collapsed_candidates", {})
    )
    for family in MODEL_FAMILIES:
        config = dict(selected_configs[family])
        if family not in active_candidates:
            fit_rows.append(
                {
                    "stage": stage,
                    "model_family": family,
                    "panel": str(config["panel"]),
                    "trial_number": int(config["trial_number"]),
                    "model_seed": int(config["model_seed"]),
                    "alpha": float(config["alpha"]),
                    "parameters_json": json.dumps(
                        _jsonable(config["params"]), sort_keys=True
                    ),
                    "fit_audit_json": json.dumps(
                        {
                            "fit_status": "not_refit",
                            "reason": collapsed_candidates.get(
                                family,
                                "inactive_inner_oof_stack_candidate",
                            ),
                            "outer_test_predictions_generated": False,
                        },
                        sort_keys=True,
                    ),
                    "shrinkage_audit_json": "{}",
                }
            )
            continue
        model_first, fit_audit = fit_predict_soft_binary_stage(
            records_indexed.loc[list(train_ids)].reset_index(drop=True),
            train_target,
            records_indexed.loc[list(test_ids)].reset_index(drop=True),
            panel=str(config["panel"]),
            model_family=family,
            params=dict(config["params"]),
            seed=int(config["model_seed"]),
            sample_weight=train_weight,
            rare_min_frequency=rare_min_frequency,
        )
        shrunk, shrink_audit = prior_shrink_probabilities(
            _binary_matrix(model_first),
            prior,
            float(config["alpha"]),
            return_audit=True,
        )
        candidates[family] = shrunk
        fit_rows.append(
            {
                "stage": stage,
                "model_family": family,
                "panel": str(config["panel"]),
                "trial_number": int(config["trial_number"]),
                "model_seed": int(config["model_seed"]),
                "alpha": float(config["alpha"]),
                "parameters_json": json.dumps(
                    _jsonable(config["params"]), sort_keys=True
                ),
                "fit_audit_json": json.dumps(
                    _jsonable(fit_audit), sort_keys=True
                ),
                "shrinkage_audit_json": json.dumps(
                    _jsonable(shrink_audit), sort_keys=True
                ),
            }
        )
    prediction = stacker.predict_proba(candidates)
    return (
        prediction,
        fit_rows,
        {
            "stage": stage,
            "prior_audit": prior_estimator.fit_audit_,
            "stack_audit": stacker.fit_audit_,
        },
    )


def _run_outer_split(
    records_indexed: pd.DataFrame,
    labels_indexed: pd.DataFrame,
    outer_manifest: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    n_trials: int,
    smoke: bool,
    rare_min_frequency: int,
) -> dict[str, Any]:
    selected_outer = outer_manifest.loc[
        outer_manifest["repeat"].eq(int(repeat))
        & outer_manifest["fold"].eq(int(fold))
    ]
    test_ids = selected_outer["record_id"].astype(str).tolist()
    all_ids = records_indexed.index.astype(str).tolist()
    test_set = set(test_ids)
    train_ids = [record_id for record_id in all_ids if record_id not in test_set]
    train_groups = set(
        records_indexed.loc[train_ids, "cv_group_id"].astype(str)
    )
    test_groups = set(
        records_indexed.loc[test_ids, "cv_group_id"].astype(str)
    )
    if train_groups & test_groups:
        raise AssertionError("A connected group crosses the outer split")
    inner_splits = _manifest_inner_splits(
        train_ids,
        inner_manifest,
        repeat=repeat,
        fold=fold,
    )
    outer_split_seed = int(selected_outer["split_seed"].iloc[0])
    stage_results: dict[str, dict[str, Any]] = {}
    all_trial_rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        sampler_offset = 41_000 if stage == "gate" else 51_000
        (
            _study,
            selected_trials,
            trial_rows,
            fold_rows,
        ) = _run_stage_study(
            records_indexed,
            labels_indexed,
            train_ids,
            inner_splits,
            stage=stage,
            repeat=repeat,
            fold=fold,
            n_trials=n_trials,
            sampler_seed=outer_split_seed + sampler_offset + int(fold),
            outer_split_seed=outer_split_seed,
            smoke=smoke,
            rare_min_frequency=rare_min_frequency,
        )
        stage_results[stage] = _fit_inner_stage_stack(
            records_indexed,
            labels_indexed,
            train_ids,
            inner_splits,
            selected_trials,
            stage=stage,
            repeat=repeat,
            fold=fold,
            outer_split_seed=outer_split_seed,
            rare_min_frequency=rare_min_frequency,
        )
        all_trial_rows.extend(trial_rows)
        all_fold_rows.extend(fold_rows)

    gate_prediction, gate_fit_rows, gate_audit = _predict_outer_stage(
        records_indexed,
        labels_indexed,
        train_ids,
        test_ids,
        stage="gate",
        selected_configs=stage_results["gate"]["selected_configs"],
        stacker=stage_results["gate"]["stacker"],
        rare_min_frequency=rare_min_frequency,
    )
    conditional_prediction, conditional_fit_rows, conditional_audit = (
        _predict_outer_stage(
            records_indexed,
            labels_indexed,
            train_ids,
            test_ids,
            stage="conditional",
            selected_configs=stage_results["conditional"]["selected_configs"],
            stacker=stage_results["conditional"]["stacker"],
            rare_min_frequency=rare_min_frequency,
        )
    )
    stacked_prediction, composition_audit = combine_two_stage_probabilities(
        gate_prediction,
        conditional_prediction,
        return_audit=True,
    )
    train_q = labels_indexed.loc[train_ids, list(Q_COLUMNS)].to_numpy(float)
    test_q = labels_indexed.loc[test_ids, list(Q_COLUMNS)].to_numpy(float)
    prior = normalize_probabilities(
        np.mean(train_q, axis=0, keepdims=True)
    )[0]
    prior_prediction = np.repeat(prior[None, :], len(test_ids), axis=0)
    prediction_rows: list[dict[str, Any]] = []
    for evaluation_model, probabilities in (
        (PRIOR_MODEL, prior_prediction),
        (STACK_MODEL, stacked_prediction),
    ):
        for position, record_id in enumerate(test_ids):
            row: dict[str, Any] = {
                "evaluation_model": evaluation_model,
                "repeat": int(repeat),
                "fold": int(fold),
                "record_id": str(record_id),
                "cv_group_id": str(
                    records_indexed.loc[record_id, "cv_group_id"]
                ),
            }
            row.update(
                {
                    column: float(test_q[position, index])
                    for index, column in enumerate(Q_COLUMNS)
                }
            )
            row.update(
                {
                    column: float(probabilities[position, index])
                    for index, column in enumerate(P_COLUMNS)
                }
            )
            prediction_rows.append(row)
    selection = {
        "outer_repeat": int(repeat),
        "outer_fold": int(fold),
        "outer_split_seed": int(outer_split_seed),
        "n_trials_per_stage": int(n_trials),
        "gate_dominant_candidate": stage_results["gate"][
            "dominant_candidate"
        ],
        "conditional_dominant_candidate": stage_results["conditional"][
            "dominant_candidate"
        ],
        "gate_stack_weights_json": json.dumps(
            _jsonable(
                stage_results["gate"]["stack_audit"][
                    "reported_candidate_weights_including_collapsed"
                ]
            ),
            sort_keys=True,
        ),
        "conditional_stack_weights_json": json.dumps(
            _jsonable(
                stage_results["conditional"]["stack_audit"][
                    "reported_candidate_weights_including_collapsed"
                ]
            ),
            sort_keys=True,
        ),
        "gate_selected_configs_json": json.dumps(
            _jsonable(stage_results["gate"]["selected_configs"]),
            sort_keys=True,
        ),
        "conditional_selected_configs_json": json.dumps(
            _jsonable(stage_results["conditional"]["selected_configs"]),
            sort_keys=True,
        ),
        "gate_inner_oof_stack_soft_log_loss": float(
            stage_results["gate"]["inner_oof_stack_soft_log_loss"]
        ),
        "conditional_inner_oof_stack_soft_log_loss": float(
            stage_results["conditional"]["inner_oof_stack_soft_log_loss"]
        ),
        "outer_test_used_for_selection": False,
    }
    fit_rows = []
    for row in [*gate_fit_rows, *conditional_fit_rows]:
        fit_rows.append(
            {
                "outer_repeat": int(repeat),
                "outer_fold": int(fold),
                **row,
            }
        )
    fit_rows.append(
        {
            "outer_repeat": int(repeat),
            "outer_fold": int(fold),
            "stage": "composition",
            "model_family": STACK_MODEL,
            "panel": "stage_specific",
            "trial_number": -1,
            "model_seed": int(outer_split_seed),
            "alpha": np.nan,
            "parameters_json": "{}",
            "fit_audit_json": json.dumps(
                _jsonable(
                    {
                        "gate": gate_audit,
                        "conditional": conditional_audit,
                        "composition": composition_audit,
                    }
                ),
                sort_keys=True,
            ),
            "shrinkage_audit_json": "{}",
        }
    )
    return {
        "predictions": prediction_rows,
        "trials": all_trial_rows,
        "inner_fold_scores": all_fold_rows,
        "selection": selection,
        "component_oof": [
            *stage_results["gate"]["component_rows"],
            *stage_results["conditional"]["component_rows"],
        ],
        "alpha_selections": [
            *stage_results["gate"]["alpha_rows"],
            *stage_results["conditional"]["alpha_rows"],
        ],
        "candidate_registry": [
            *stage_results["gate"]["candidate_rows"],
            *stage_results["conditional"]["candidate_rows"],
        ],
        "stack_weights": [
            *stage_results["gate"]["stack_weight_rows"],
            *stage_results["conditional"]["stack_weight_rows"],
        ],
        "fits": fit_rows,
    }


def run_nested_stagewise_oof(
    records: pd.DataFrame,
    labels: pd.DataFrame,
    outer_manifest: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    *,
    n_trials: int,
    smoke: bool,
    rare_min_frequency: int,
    n_jobs: int,
) -> dict[str, pd.DataFrame]:
    """Return honest repeated OOF predictions for the complete adaptive stack."""

    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    if int(n_trials) < len(MODEL_FAMILIES):
        raise ValueError(
            "Each stage requires at least one enqueued trial per model family"
        )
    if not smoke and int(n_trials) < len(SEARCH_PANELS) * len(MODEL_FAMILIES):
        raise ValueError(
            "A full run requires one default for every panel/family pair"
        )
    records_indexed = _records_by_id(records)
    labels_indexed = _labels_by_id(labels)
    tasks = [
        (int(row.repeat), int(row.fold))
        for row in outer_manifest.loc[:, ["repeat", "fold"]]
        .drop_duplicates()
        .sort_values(["repeat", "fold"], kind="stable")
        .itertuples(index=False)
    ]
    results = Parallel(
        n_jobs=int(n_jobs),
        prefer="threads",
        verbose=0,
    )(
        delayed(_run_outer_split)(
            records_indexed,
            labels_indexed,
            outer_manifest,
            inner_manifest,
            repeat=repeat,
            fold=fold,
            n_trials=int(n_trials),
            smoke=smoke,
            rare_min_frequency=int(rare_min_frequency),
        )
        for repeat, fold in tasks
    )
    frames = {
        "oof_probabilities": pd.DataFrame(
            row for result in results for row in result["predictions"]
        ),
        "optuna_trials": pd.DataFrame(
            row for result in results for row in result["trials"]
        ),
        "optuna_inner_fold_scores": pd.DataFrame(
            row for result in results for row in result["inner_fold_scores"]
        ),
        "outer_selections": pd.DataFrame(
            result["selection"] for result in results
        ),
        "stage_component_oof_predictions": pd.DataFrame(
            row for result in results for row in result["component_oof"]
        ),
        "stage_alpha_selections": pd.DataFrame(
            row for result in results for row in result["alpha_selections"]
        ),
        "stage_candidate_registry": pd.DataFrame(
            row for result in results for row in result["candidate_registry"]
        ),
        "stage_stack_weights": pd.DataFrame(
            row for result in results for row in result["stack_weights"]
        ),
        "model_fit_registry": pd.DataFrame(
            row for result in results for row in result["fits"]
        ),
    }
    oof = frames["oof_probabilities"]
    expected_repeats = int(outer_manifest["repeat"].nunique())
    coverage = oof.groupby(["evaluation_model", "record_id"]).size()
    if not coverage.eq(expected_repeats).all():
        raise AssertionError("Repeated contiguous three-class OOF is incomplete")
    probabilities = oof.loc[:, P_COLUMNS].to_numpy(float)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0
        )
    ):
        raise AssertionError("Contiguous OOF predictions are invalid")
    return frames


def aggregate_contiguous_oof(oof: pd.DataFrame) -> pd.DataFrame:
    keys = ["evaluation_model", "record_id", "cv_group_id"]
    aggregated = (
        oof.groupby(keys, as_index=False, dropna=False)
        .agg(
            **{column: (column, "first") for column in Q_COLUMNS},
            **{column: (column, "mean") for column in P_COLUMNS},
            n_repeated_oof=("repeat", "nunique"),
        )
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    aggregated.loc[:, P_COLUMNS] = normalize_probabilities(
        aggregated.loc[:, P_COLUMNS].to_numpy(float)
    )
    return aggregated


def _contiguous_probability_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    entropy_threshold: float,
) -> dict[str, float | int]:
    q = normalize_probabilities(observed)
    p = normalize_probabilities(predicted)
    q_non_high = q[:, :2].sum(axis=1)
    p_non_high = p[:, :2].sum(axis=1)
    q_gate = np.column_stack([q_non_high, q[:, 2]])
    p_gate = np.column_stack([p_non_high, p[:, 2]])
    q_conditional = np.divide(
        q[:, :2],
        q_non_high[:, None],
        out=np.full((len(q), 2), 0.5, dtype=float),
        where=q_non_high[:, None] > 0,
    )
    p_conditional = np.divide(
        p[:, :2],
        p_non_high[:, None],
        out=np.full((len(p), 2), 0.5, dtype=float),
        where=p_non_high[:, None] > 0,
    )
    overall = descriptive_probability_metrics(
        q,
        p,
        entropy_threshold=float(entropy_threshold),
    )
    conditional_log_loss = soft_cross_entropy(
        q_conditional,
        p_conditional,
        sample_weight=q_non_high,
    )
    conditional_brier = multiclass_brier(
        q_conditional,
        p_conditional,
        sample_weight=q_non_high,
    )
    return {
        **overall,
        "full_coverage_three_class_soft_log_loss": float(
            overall["soft_log_loss"]
        ),
        "multiclass_brier_mean_sum_squared_error": float(
            overall["multiclass_brier"]
        ),
        "gate_soft_log_loss": soft_cross_entropy(q_gate, p_gate),
        "gate_brier": multiclass_brier(q_gate, p_gate),
        "conditional_soft_log_loss_weighted_by_q_non_high": float(
            conditional_log_loss
        ),
        "conditional_brier_weighted_by_q_non_high": float(
            conditional_brier
        ),
        "mean_observed_non_high_probability": float(np.mean(q_non_high)),
        "mean_predicted_non_high_probability": float(np.mean(p_non_high)),
    }


def _metric_tables(
    oof: pd.DataFrame,
    aggregated: pd.DataFrame,
    *,
    entropy_threshold: float,
) -> dict[str, pd.DataFrame]:
    aggregate_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    for model, frame in aggregated.groupby("evaluation_model", sort=True):
        q = frame.loc[:, Q_COLUMNS].to_numpy(float)
        p = frame.loc[:, P_COLUMNS].to_numpy(float)
        aggregate_rows.append(
            {
                "evaluation_model": str(model),
                "n_records": int(len(frame)),
                "n_connected_groups": int(frame["cv_group_id"].nunique()),
                **_contiguous_probability_metrics(
                    q, p, entropy_threshold=entropy_threshold
                ),
            }
        )
        calibration_frames.append(
            calibration_table(
                q, p, class_labels=CLASS_LABELS, n_bins=8
            ).assign(evaluation_model=str(model))
        )
        confusion_frames.append(
            expected_confusion_table(
                q, p, class_labels=CLASS_LABELS
            ).assign(evaluation_model=str(model))
        )
    repeat_rows: list[dict[str, Any]] = []
    for (model, repeat), frame in oof.groupby(
        ["evaluation_model", "repeat"], sort=True
    ):
        repeat_rows.append(
            {
                "evaluation_model": str(model),
                "repeat": int(repeat),
                **_contiguous_probability_metrics(
                    frame.loc[:, Q_COLUMNS].to_numpy(float),
                    frame.loc[:, P_COLUMNS].to_numpy(float),
                    entropy_threshold=entropy_threshold,
                ),
            }
        )
    split_rows: list[dict[str, Any]] = []
    for (model, repeat, fold), frame in oof.groupby(
        ["evaluation_model", "repeat", "fold"], sort=True
    ):
        split_rows.append(
            {
                "evaluation_model": str(model),
                "repeat": int(repeat),
                "fold": int(fold),
                "n_test_records": int(len(frame)),
                **_contiguous_probability_metrics(
                    frame.loc[:, Q_COLUMNS].to_numpy(float),
                    frame.loc[:, P_COLUMNS].to_numpy(float),
                    entropy_threshold=entropy_threshold,
                ),
            }
        )
    return {
        "aggregate_metrics": pd.DataFrame(aggregate_rows),
        "repeat_metrics": pd.DataFrame(repeat_rows),
        "split_metrics": pd.DataFrame(split_rows),
        "calibration": pd.concat(calibration_frames, ignore_index=True),
        "expected_confusion": pd.concat(confusion_frames, ignore_index=True),
    }


def _record_delta_inference(
    aggregated: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stacked = aggregated.loc[
        aggregated["evaluation_model"].eq(STACK_MODEL)
    ].copy()
    prior = aggregated.loc[
        aggregated["evaluation_model"].eq(PRIOR_MODEL)
    ].copy()
    merged = stacked.merge(
        prior,
        on=["record_id", "cv_group_id"],
        suffixes=("_stack", "_prior"),
        validate="one_to_one",
    )
    if len(merged) != 97:
        raise AssertionError("Paired inference must retain all 97 records")
    q = merged.loc[
        :, [f"{column}_stack" for column in Q_COLUMNS]
    ].to_numpy(float)
    p_stack = merged.loc[
        :, [f"{column}_stack" for column in P_COLUMNS]
    ].to_numpy(float)
    p_prior = merged.loc[
        :, [f"{column}_prior" for column in P_COLUMNS]
    ].to_numpy(float)
    epsilon = 1e-15
    stack_loss = -np.sum(
        q * np.log(np.clip(p_stack, epsilon, 1.0)), axis=1
    )
    prior_loss = -np.sum(
        q * np.log(np.clip(p_prior, epsilon, 1.0)), axis=1
    )
    stack_brier = np.sum(np.square(p_stack - q), axis=1)
    prior_brier = np.sum(np.square(p_prior - q), axis=1)
    stack_modal = np.argmax(p_stack, axis=1)
    prior_modal = np.argmax(p_prior, axis=1)
    positions = np.arange(len(q))
    scores = pd.DataFrame(
        {
            "record_id": merged["record_id"].astype(str),
            "cv_group_id": merged["cv_group_id"].astype(str),
            "stack_soft_log_loss": stack_loss,
            "prior_soft_log_loss": prior_loss,
            "delta_soft_log_loss": stack_loss - prior_loss,
            "stack_multiclass_brier": stack_brier,
            "prior_multiclass_brier": prior_brier,
            "delta_multiclass_brier": stack_brier - prior_brier,
            "stack_posterior_expected_hard_accuracy": q[
                positions, stack_modal
            ],
            "prior_posterior_expected_hard_accuracy": q[
                positions, prior_modal
            ],
            "delta_posterior_expected_hard_accuracy": (
                q[positions, stack_modal] - q[positions, prior_modal]
            ),
        }
    )
    draws, summary = paired_connected_group_bootstrap(
        scores,
        [
            "delta_soft_log_loss",
            "delta_multiclass_brier",
            "delta_posterior_expected_hard_accuracy",
        ],
        replicates=int(bootstrap_repetitions),
        seed=int(bootstrap_seed),
    )
    return scores, draws, summary


def _stack_weight_summary(weights: pd.DataFrame) -> pd.DataFrame:
    return (
        weights.groupby(["stage", "candidate"], as_index=False)
        .agg(
            mean_weight=("weight", "mean"),
            median_weight=("weight", "median"),
            minimum_weight=("weight", "min"),
            maximum_weight=("weight", "max"),
            nonzero_fraction=("weight", lambda values: float(np.mean(values > 1e-8))),
            dominant_count=("dominant", "sum"),
            n_outer_splits=("outer_fold", "size"),
        )
        .sort_values(
            ["stage", "mean_weight", "candidate"],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _stage_selection_summary(
    candidates: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    model_weights = weights.loc[
        weights["candidate"].isin(MODEL_FAMILIES),
        [
            "outer_repeat",
            "outer_fold",
            "stage",
            "candidate",
            "weight",
            "dominant",
        ],
    ].rename(columns={"candidate": "model_family"})
    merged = candidates.merge(
        model_weights,
        on=["outer_repeat", "outer_fold", "stage", "model_family"],
        how="left",
        validate="one_to_one",
    )
    if merged[["weight", "dominant"]].isna().any().any():
        raise AssertionError(
            "Every selected family configuration needs a stack weight"
        )
    summary = (
        merged.groupby(
            ["stage", "model_family", "panel"],
            as_index=False,
            sort=True,
        )
        .agg(
            selected_outer_splits=("outer_fold", "size"),
            mean_selected_alpha=("alpha", "mean"),
            median_selected_alpha=("alpha", "median"),
            mean_inner_oof_soft_log_loss=(
                "inner_oof_soft_log_loss",
                "mean",
            ),
            mean_stack_weight=("weight", "mean"),
            nonzero_stack_weight_fraction=(
                "weight",
                lambda values: float(np.mean(values > 1e-8)),
            ),
            dominant_stack_count=("dominant", "sum"),
        )
        .reset_index(drop=True)
    )
    totals = summary.groupby(
        ["stage", "model_family"]
    )["selected_outer_splits"].transform("sum")
    maxima = summary.groupby(
        ["stage", "model_family"]
    )["selected_outer_splits"].transform("max")
    summary["selection_fraction_within_family"] = (
        summary["selected_outer_splits"] / totals
    )
    summary["most_frequently_selected_within_family"] = (
        summary["selected_outer_splits"].eq(maxima)
    )
    return summary.sort_values(
        [
            "stage",
            "model_family",
            "selected_outer_splits",
            "mean_inner_oof_soft_log_loss",
            "panel",
        ],
        ascending=[True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _write_report(
    destination: Path,
    *,
    metrics: pd.DataFrame,
    support: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    weight_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    smoke: bool,
) -> Path:
    stack = metrics.loc[metrics["evaluation_model"].eq(STACK_MODEL)].iloc[0]
    prior = metrics.loc[metrics["evaluation_model"].eq(PRIOR_MODEL)].iloc[0]
    lines = [
        "# Contiguous posterior-aware three-class sensitivity",
        "",
        (
            "**Structural smoke only; no scientific performance claim.**"
            if smoke
            else "**Post-hoc exploratory changed endpoint; not an independent "
            "confirmation and not a replacement for v4 or classic v1.**"
        ),
        "",
        "## Response definition",
        "",
        f"- `low_60_side`: slope < {LOW_120_LOG_MIDPOINT_MV_DEC:.8f} mV dec-1",
        (
            f"- `120_side`: {LOW_120_LOG_MIDPOINT_MV_DEC:.8f} <= slope "
            f"<= {HIGH_OFF_BOUNDARY_MV_DEC:.1f} mV dec-1"
        ),
        f"- `high_off`: slope > {HIGH_OFF_BOUNDARY_MV_DEC:.1f} mV dec-1",
        "",
        support.to_markdown(index=False),
        "",
        "## Nested connected-group OOF performance",
        "",
        (
            f"- Stack soft log loss: "
            f"{float(stack['full_coverage_three_class_soft_log_loss']):.6f}"
        ),
        (
            f"- Prior soft log loss: "
            f"{float(prior['full_coverage_three_class_soft_log_loss']):.6f}"
        ),
        f"- Stack modal accuracy: {float(stack['modal_accuracy']):.6f}",
        f"- Prior modal accuracy: {float(prior['modal_accuracy']):.6f}",
        (
            "- Posterior-expected oracle: "
            f"{float(stack['oracle_posterior_expected_hard_accuracy']):.6f}"
        ),
        "",
        "## Stagewise mean stack weights",
        "",
        weight_summary.to_markdown(index=False),
        "",
        "## Most frequently inner-selected panel within each family",
        "",
        selection_summary.loc[
            selection_summary["most_frequently_selected_within_family"]
        ].to_markdown(index=False),
        "",
        "## Paired connected-group bootstrap",
        "",
        bootstrap_summary.to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        (
            "The 0.70 comparison is descriptive. Hard accuracy, posterior "
            "argmax labels, random record splits, and outer-test results were "
            "not used to tune labels, features, alpha, model families, or "
            "stack weights. This endpoint intentionally differs from the "
            "classic-neighbourhood task, so absolute accuracy is not a "
            "same-task improvement claim."
        ),
        "",
    ]
    path = (
        destination
        / "report"
        / "posterior_cardinal_contiguous_threeclass_report.md"
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _validate_label_contract(
    labels: pd.DataFrame,
    support: pd.DataFrame,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    label_contract = dict(contract["label_scheme"])
    expected_mass = {
        str(key): float(value)
        for key, value in dict(
            label_contract["expected_effective_posterior_mass"]
        ).items()
    }
    observed_mass = {
        str(row.class_label): float(row.effective_posterior_mass)
        for row in support.itertuples(index=False)
    }
    if set(observed_mass) != set(expected_mass):
        raise ValueError("Contiguous three-class support labels changed")
    for label in CLASS_LABELS:
        if not np.isclose(
            observed_mass[label],
            expected_mass[label],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(
                "Contiguous three-class posterior mass changed for "
                f"{label}: expected={expected_mass[label]}, "
                f"observed={observed_mass[label]}"
            )

    expected_modal = {
        str(key): int(value)
        for key, value in dict(
            label_contract["expected_modal_record_counts"]
        ).items()
    }
    observed_modal = (
        labels["threeclass_modal_label"].astype(str).value_counts().to_dict()
    )
    observed_modal = {
        label: int(observed_modal.get(label, 0)) for label in CLASS_LABELS
    }
    if observed_modal != expected_modal:
        raise ValueError(
            "Contiguous three-class modal counts changed: "
            f"expected={expected_modal}, observed={observed_modal}"
        )

    observed_nonpositive = int(labels["n_nonpositive_draws"].sum())
    expected_nonpositive = int(
        label_contract["expected_total_nonpositive_draws"]
    )
    if observed_nonpositive != expected_nonpositive:
        raise ValueError(
            "The audited number of nonpositive finite posterior draws "
            f"changed: expected={expected_nonpositive}, "
            f"observed={observed_nonpositive}"
        )

    probabilities = labels.loc[:, Q_COLUMNS].to_numpy(float)
    oracle = float(np.mean(np.max(probabilities, axis=1)))
    expected_oracle = float(
        label_contract["expected_posterior_expected_oracle"]
    )
    if not np.isclose(oracle, expected_oracle, atol=1e-15, rtol=0):
        raise ValueError(
            "Contiguous three-class posterior-expected oracle changed: "
            f"expected={expected_oracle}, observed={oracle}"
        )
    return {
        "effective_posterior_mass": observed_mass,
        "modal_record_counts": observed_modal,
        "total_nonpositive_finite_draws": observed_nonpositive,
        "posterior_expected_oracle": oracle,
    }


def _manifest_mass_range(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> tuple[float, float]:
    lookup = labels.loc[:, ["record_id", "q_high_off"]].copy()
    lookup["record_id"] = lookup["record_id"].astype(str)
    selected = manifest.copy()
    selected["record_id"] = selected["record_id"].astype(str)
    merged = selected.merge(
        lookup,
        on="record_id",
        how="left",
        validate="many_to_one",
    )
    if merged["q_high_off"].isna().any():
        raise ValueError("Fold manifest contains an unknown record")
    values = merged.groupby(
        list(group_columns), sort=True
    )["q_high_off"].sum()
    return float(values.min()), float(values.max())


def run_posterior_cardinal_contiguous_threeclass(
    config: StudyConfig,
    run_dir: str | Path,
    *,
    source_run: str | Path,
    smoke: bool = False,
    n_jobs: int = 1,
    optuna_trials: int | None = None,
) -> dict[str, Any]:
    """Run the independent contiguous three-class nested sensitivity."""

    destination = Path(run_dir).expanduser().resolve()
    source = Path(source_run).expanduser().resolve()
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    if optuna_trials is not None and int(optuna_trials) < len(MODEL_FAMILIES):
        raise ValueError(
            "optuna_trials must cover all four fixed model families"
        )
    _assert_fresh_disjoint_run_directory(destination, source)
    contract, contract_path, contract_hash = _load_contract(config)
    full_trials = int(
        contract["optuna"]["full_trials_per_stage_per_outer_split"]
    )
    if (
        not smoke
        and optuna_trials is not None
        and int(optuna_trials) != full_trials
    ):
        raise ValueError(
            "Full contiguous three-class runs use the hash-bound "
            f"{full_trials} Optuna trials per stage and outer split; "
            "--optuna-trials is a smoke-only override"
        )

    parent = dict(contract["immutable_parent_boundaries"])
    v4_result_root = resolve_contiguous_threeclass_v4_result_root(config)
    assert_disjoint_cardinal_paths(v4_result_root, destination)
    classic_result_root = (
        resolve_contiguous_threeclass_classic_result_root(config)
    )
    assert_disjoint_cardinal_paths(classic_result_root, destination)

    v4_contract, v4_contract_path, v4_contract_hash = (
        load_frozen_posterior_cardinal_contract(config)
    )
    if (
        parent.get("source_contract_id") != v4_contract["contract_id"]
        or parent.get("source_contract_sha256") != v4_contract_hash
    ):
        raise ValueError(
            "Contiguous three-class contract is not bound to active v4"
        )
    classic_contract_path = Path(
        str(parent["classic_threeclass_contract_path"])
    ).expanduser()
    if not classic_contract_path.is_absolute():
        classic_contract_path = (
            config.project_root / classic_contract_path
        ).resolve()
    classic_contract_hash = sha256_file(classic_contract_path)
    if (
        classic_contract_hash
        != parent["classic_threeclass_contract_sha256"]
    ):
        raise ValueError("Bound classic three-class contract hash changed")
    classic_contract_payload = yaml.safe_load(
        classic_contract_path.read_text(encoding="utf-8")
    )
    if (
        not isinstance(classic_contract_payload, dict)
        or classic_contract_payload.get("contract_id")
        != parent["classic_threeclass_contract_id"]
    ):
        raise ValueError("Bound classic three-class contract ID changed")

    parent_artifact_verifications = {
        "v4_result": _verify_parent_artifact_manifest(
            v4_result_root,
            label="v4 result",
        ),
        "classic_threeclass_result": _verify_parent_artifact_manifest(
            classic_result_root,
            label="classic three-class result",
        ),
    }
    classic_fold_verification = _verify_bound_classic_fold_manifests(
        classic_result_root,
        parent,
    )

    parent_summaries_before = {
        "v4_result": directory_tree_summary(v4_result_root),
        "classic_threeclass_result": directory_tree_summary(
            classic_result_root
        ),
    }
    for name in (
        "inputs",
        "inputs/reduced_feature_registry",
        "registry",
        "folds",
        "analyses",
        "report",
        "provenance",
    ):
        (destination / name).mkdir(parents=True, exist_ok=True)
    contract_snapshot = destination / "registry" / "contract_snapshot.yaml"
    shutil.copy2(contract_path, contract_snapshot)
    if sha256_file(contract_snapshot) != contract_hash:
        raise RuntimeError(
            "Contiguous three-class contract changed while copying"
        )

    reduced_binding = snapshot_reduced_feature_registries(
        config,
        v4_contract,
        destination / "inputs" / "reduced_feature_registry",
    )
    source_hashes = v4_contract["formal_input_snapshot"][
        "source_file_sha256"
    ]
    records, targets, draws, input_binding = snapshot_cardinal_inputs(
        source,
        destination / "inputs",
        canonical_dataset=config.resolve_path("canonical_dataset"),
        expected_source_hashes=source_hashes,
        require_formal_approval=not smoke,
    )
    del targets
    _validate_run_frame(records)
    parsed_snapshot_verification = verify_cardinal_input_snapshot(
        destination / "inputs", source_hashes
    )
    parsed_registry_verification = verify_reduced_feature_registry_snapshot(
        destination / "inputs" / "reduced_feature_registry",
        v4_contract,
    )

    labels = derive_contiguous_threeclass_soft_labels(
        draws,
        records.loc[:, IDENTITY_COLUMNS].copy(),
        expected_draws_per_record=int(
            contract["label_scheme"]["expected_draws_per_record"]
        ),
    )
    support = contiguous_threeclass_support(labels)
    label_audit = _validate_label_contract(labels, support, contract)
    labels.to_csv(
        destination / "registry" / "contiguous_three_class_soft_labels.csv",
        index=False,
    )
    support.to_csv(
        destination / "registry" / "contiguous_three_class_support.csv",
        index=False,
    )
    write_json(
        destination / "registry" / "contiguous_three_class_label_audit.json",
        _jsonable(label_audit),
    )

    primary_frame = _primary_frame(records, labels)
    outer_manifest = build_contiguous_outer_manifest(
        primary_frame, contract, smoke=smoke
    )
    inner_manifest = build_contiguous_inner_manifest(
        primary_frame,
        outer_manifest,
        contract,
        smoke=smoke,
    )
    if (
        outer_manifest.groupby(["repeat", "cv_group_id"])["fold"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError("A connected group crosses outer folds")
    if (
        outer_manifest.groupby(["repeat", "record_id"]).size().ne(1).any()
    ):
        raise AssertionError(
            "Every record must occur once in each outer repeat"
        )
    outer_path = destination / "folds" / "outer_split_manifest.csv"
    inner_path = destination / "folds" / "inner_split_manifest.csv"
    outer_manifest.to_csv(outer_path, index=False)
    inner_manifest.to_csv(inner_path, index=False)
    outer_hash = sha256_file(outer_path)
    inner_hash = sha256_file(inner_path)
    outer_high_mass_range = _manifest_mass_range(
        outer_manifest,
        labels,
        group_columns=("repeat", "fold"),
    )
    inner_high_mass_range = _manifest_mass_range(
        inner_manifest,
        labels,
        group_columns=("outer_repeat", "outer_fold", "inner_fold"),
    )
    if not smoke:
        outer_contract = dict(contract["outer_validation"])
        inner_contract = dict(contract["inner_validation"])
        if (
            len(outer_manifest) != int(outer_contract["expected_full_rows"])
            or outer_hash
            != str(outer_contract["expected_full_manifest_sha256"])
        ):
            raise ValueError(
                "Full outer manifest did not reproduce the frozen "
                "contiguous-endpoint manifest"
            )
        if (
            len(inner_manifest) != int(inner_contract["expected_full_rows"])
            or inner_hash
            != str(inner_contract["expected_full_manifest_sha256"])
        ):
            raise ValueError(
                "Full inner manifest did not reproduce the frozen "
                "contiguous-endpoint manifest"
            )
        if not np.allclose(
            outer_high_mass_range,
            outer_contract["observed_high_off_validation_mass_range"],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Full outer high-off mass range changed")
        if not np.allclose(
            inner_high_mass_range,
            inner_contract["observed_high_off_validation_mass_range"],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Full inner high-off mass range changed")

    requested_trials = int(
        optuna_trials
        if optuna_trials is not None
        else (
            contract["smoke_mode"][
                "optuna_trials_per_stage_per_outer_split"
            ]
            if smoke
            else full_trials
        )
    )
    if requested_trials < len(MODEL_FAMILIES):
        raise ValueError(
            "Each stage needs at least one Optuna trial per fixed family"
        )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    nested = run_nested_stagewise_oof(
        records,
        labels,
        outer_manifest,
        inner_manifest,
        n_trials=requested_trials,
        smoke=smoke,
        rare_min_frequency=int(
            contract["feature_engineering"]["rare_min_frequency"]
        ),
        n_jobs=int(n_jobs),
    )
    oof = nested["oof_probabilities"]
    aggregated = aggregate_contiguous_oof(oof)
    metric_tables = _metric_tables(
        oof,
        aggregated,
        entropy_threshold=float(
            contract["metrics"]["predictive_entropy_threshold"]
        ),
    )
    bootstrap_repetitions = int(
        contract["smoke_mode"]["bootstrap_repetitions"]
        if smoke
        else contract["connected_group_inference"]["bootstrap"][
            "full_repetitions"
        ]
    )
    record_scores, bootstrap_draws, bootstrap_summary = (
        _record_delta_inference(
            aggregated,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=int(
                contract["connected_group_inference"]["bootstrap"]["seed"]
            ),
        )
    )
    weight_summary = _stack_weight_summary(
        nested["stage_stack_weights"]
    )
    selection_summary = _stage_selection_summary(
        nested["stage_candidate_registry"],
        nested["stage_stack_weights"],
    )

    for name, frame in nested.items():
        frame.to_csv(
            destination / "analyses" / f"{name}.csv", index=False
        )
    aggregated.to_csv(
        destination / "analyses" / "oof_aggregated.csv", index=False
    )
    for name, frame in metric_tables.items():
        frame.to_csv(destination / "analyses" / f"{name}.csv", index=False)
    record_scores.to_csv(
        destination / "analyses" / "record_level_scores.csv", index=False
    )
    bootstrap_draws.to_csv(
        destination / "analyses" / "connected_group_bootstrap.csv",
        index=False,
    )
    bootstrap_summary.to_csv(
        destination
        / "analyses"
        / "connected_group_bootstrap_summary.csv",
        index=False,
    )
    weight_summary.to_csv(
        destination / "analyses" / "stack_weight_summary.csv", index=False
    )
    selection_summary.to_csv(
        destination / "analyses" / "stage_selection_summary.csv",
        index=False,
    )

    modeled_snapshot_verification = verify_cardinal_input_snapshot(
        destination / "inputs", source_hashes
    )
    modeled_registry_verification = verify_reduced_feature_registry_snapshot(
        destination / "inputs" / "reduced_feature_registry",
        v4_contract,
    )
    source_verification = verify_source_run_unchanged(
        source,
        expected_tree_sha256=input_binding[
            "source_run_tree_sha256_before"
        ],
    )
    parent_summaries_after = {
        "v4_result": directory_tree_summary(v4_result_root),
        "classic_threeclass_result": directory_tree_summary(
            classic_result_root
        ),
    }
    parent_unchanged = {
        key: (
            parent_summaries_after[key]["tree_sha256"]
            == parent_summaries_before[key]["tree_sha256"]
        )
        for key in parent_summaries_before
    }
    if not all(parent_unchanged.values()):
        raise RuntimeError(
            "An immutable parent result changed during the sensitivity run"
        )
    write_json(
        destination / "provenance" / "contract_binding.json",
        {
            "contract_id": CONTRACT_ID,
            "status": contract["status"],
            "source_path": str(contract_path),
            "snapshot_path": str(contract_snapshot),
            "sha256": contract_hash,
            "snapshot_sha256": sha256_file(contract_snapshot),
            "posthoc_exploratory": True,
            "frozen_before_first_full_fit": True,
            "structural_smoke_completed_before_full_fit": True,
        },
    )
    write_json(
        destination / "provenance" / "parent_contract_bindings.json",
        {
            "v4": {
                "contract_id": v4_contract["contract_id"],
                "source_path": str(v4_contract_path),
                "sha256": v4_contract_hash,
                "result_root": str(v4_result_root),
                "result_tree_sha256": parent_summaries_after["v4_result"][
                    "tree_sha256"
                ],
                "artifact_manifest_verification": (
                    parent_artifact_verifications["v4_result"]
                ),
            },
            "classic_threeclass_v1": {
                "contract_id": parent["classic_threeclass_contract_id"],
                "source_path": str(classic_contract_path),
                "sha256": classic_contract_hash,
                "result_root": str(classic_result_root),
                "result_tree_sha256": parent_summaries_after[
                    "classic_threeclass_result"
                ]["tree_sha256"],
                "artifact_manifest_verification": (
                    parent_artifact_verifications[
                        "classic_threeclass_result"
                    ]
                ),
                "frozen_fold_manifest_verification": (
                    classic_fold_verification
                ),
            },
            "parents_unchanged": parent_unchanged,
        },
    )
    write_json(
        destination / "provenance" / "input_binding.json", input_binding
    )
    write_json(
        destination / "provenance" / "reduced_feature_registry_binding.json",
        reduced_binding,
    )
    write_json(
        destination
        / "provenance"
        / "input_snapshot_verification_after_parse.json",
        parsed_snapshot_verification,
    )
    write_json(
        destination
        / "provenance"
        / "input_snapshot_verification_after_modeling.json",
        modeled_snapshot_verification,
    )
    write_json(
        destination
        / "provenance"
        / "reduced_registry_verification_after_parse.json",
        parsed_registry_verification,
    )
    write_json(
        destination
        / "provenance"
        / "reduced_registry_verification_after_modeling.json",
        modeled_registry_verification,
    )
    write_json(
        destination / "provenance" / "source_run_verification.json",
        source_verification,
    )
    write_json(
        destination / "provenance" / "fold_manifest_binding.json",
        {
            "endpoint_specific_manifests": True,
            "classic_v1_manifests_reused": False,
            "outer_manifest_sha256": outer_hash,
            "inner_manifest_sha256": inner_hash,
            "outer_high_off_validation_mass_range": list(
                outer_high_mass_range
            ),
            "inner_high_off_validation_mass_range": list(
                inner_high_mass_range
            ),
            "matches_expected_full_hashes": bool(
                not smoke
                and outer_hash
                == contract["outer_validation"][
                    "expected_full_manifest_sha256"
                ]
                and inner_hash
                == contract["inner_validation"][
                    "expected_full_manifest_sha256"
                ]
            ),
        },
    )

    metrics = metric_tables["aggregate_metrics"]
    stack_metrics = metrics.loc[
        metrics["evaluation_model"].eq(STACK_MODEL)
    ].iloc[0]
    prior_metrics = metrics.loc[
        metrics["evaluation_model"].eq(PRIOR_MODEL)
    ].iloc[0]
    n_outer_splits = int(
        outer_manifest.loc[:, ["repeat", "fold"]].drop_duplicates().shape[0]
    )
    qa = {
        "status": "complete_smoke" if smoke else "complete_exploratory",
        "scientifically_releasable": False,
        "analysis_role": "posthoc_prediction_oriented_sensitivity",
        "changed_endpoint": True,
        "formal_v4_result_modified": False,
        "classic_threeclass_v1_result_modified": False,
        "may_replace_or_rescue_v4_result": False,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_hash,
        "n_records": int(len(records)),
        "n_connected_groups": int(records["cv_group_id"].nunique()),
        "n_posterior_draws": int(len(draws)),
        "draws_per_record": int(
            draws.groupby("sample_record").size().iloc[0]
        ),
        "class_order": list(CLASS_LABELS),
        **label_audit,
        "outer_repeats": int(outer_manifest["repeat"].nunique()),
        "outer_folds": int(outer_manifest["fold"].nunique()),
        "inner_folds": int(inner_manifest["inner_fold"].nunique()),
        "outer_manifest_sha256": outer_hash,
        "inner_manifest_sha256": inner_hash,
        "outer_high_off_validation_mass_range": list(
            outer_high_mass_range
        ),
        "inner_high_off_validation_mass_range": list(
            inner_high_mass_range
        ),
        "optuna_trials_per_stage_per_outer_split": requested_trials,
        "n_outer_splits": n_outer_splits,
        "n_optuna_trials_total": int(len(nested["optuna_trials"])),
        "n_optuna_trials_complete": int(
            nested["optuna_trials"]["state"].eq("complete").sum()
        ),
        "gate_and_conditional_studies_independent": True,
        "gate_and_conditional_hyperparameters_independent": True,
        "feature_engineering_fit_inside_inner_and_outer_training_folds": True,
        "prior_shrinkage_selected_on_grouped_inner_oof_only": True,
        "stage_stack_weights_selected_on_grouped_inner_oof_only": True,
        "fold_local_prior_in_every_stage_stack": True,
        "outer_test_used_for_selection_shrinkage_or_stacking": False,
        "one_ml_row_per_record": True,
        "posterior_draws_expanded_as_ml_samples": False,
        "posterior_argmax_used_for_training_or_selection": False,
        "hard_accuracy_used_for_training_or_selection": False,
        "primary_metric": "full_coverage_three_class_soft_log_loss",
        "stack_soft_log_loss": float(
            stack_metrics["full_coverage_three_class_soft_log_loss"]
        ),
        "prior_soft_log_loss": float(
            prior_metrics["full_coverage_three_class_soft_log_loss"]
        ),
        "stack_modal_accuracy_descriptive_only": float(
            stack_metrics["modal_accuracy"]
        ),
        "prior_modal_accuracy_descriptive_only": float(
            prior_metrics["modal_accuracy"]
        ),
        "stack_posterior_expected_hard_accuracy": float(
            stack_metrics["posterior_expected_hard_accuracy"]
        ),
        "posterior_expected_hard_accuracy_oracle_ceiling": float(
            stack_metrics["oracle_posterior_expected_hard_accuracy"]
        ),
        "requested_accuracy_0_70_oracle_reachable": bool(
            stack_metrics["oracle_posterior_expected_hard_accuracy"] >= 0.70
        ),
        "nested_oof_modal_accuracy_reaches_0_70": bool(
            stack_metrics["modal_accuracy"] >= 0.70
        ),
        "input_snapshot_verified_after_modeling": (
            modeled_snapshot_verification["status"] == "pass"
        ),
        "reduced_registry_verified_after_modeling": (
            modeled_registry_verification["status"] == "pass"
        ),
        "formal_source_run_unchanged": bool(
            source_verification["unchanged"]
        ),
        "immutable_parent_runs_unchanged": parent_unchanged,
        "immutable_parent_artifact_manifests_verified": bool(
            all(
                value.get("status") == "pass"
                for value in parent_artifact_verifications.values()
            )
        ),
        "classic_v1_frozen_fold_manifests_verified": bool(
            classic_fold_verification["status"] == "pass"
        ),
        "causal_claim_allowed": False,
        "mechanistic_claim_allowed": False,
    }
    write_json(
        destination / "analyses" / "analysis_qa.json", _jsonable(qa)
    )
    report = _write_report(
        destination,
        metrics=metrics,
        support=support,
        bootstrap_summary=bootstrap_summary,
        weight_summary=weight_summary,
        selection_summary=selection_summary,
        smoke=smoke,
    )
    write_json(
        destination / "report" / "validation_report.json",
        {
            "overall_assessment": (
                "structural_smoke_only"
                if smoke
                else "ready_for_posthoc_exploratory_review_not_formal_release"
            ),
            "methodology": (
                "all97_contiguous_soft_labels_connected_group_nested_"
                "stagewise_optuna_prior_shrinkage_convex_stack"
            ),
            "qa": _jsonable(qa),
            "report": str(report),
        },
    )
    return _jsonable(
        {
            "status": qa["status"],
            "run_dir": str(destination),
            "source_run": str(source),
            "contract_id": CONTRACT_ID,
            "contract_sha256": contract_hash,
            "qa": qa,
            "report": str(report),
        }
    )


__all__ = [
    "CLASS_LABELS",
    "CONTRACT_ID",
    "MODEL_FAMILIES",
    "P_COLUMNS",
    "PRIOR_MODEL",
    "Q_COLUMNS",
    "SEARCH_PANELS",
    "STACK_MODEL",
    "aggregate_contiguous_oof",
    "build_contiguous_inner_manifest",
    "build_contiguous_outer_manifest",
    "resolve_contiguous_threeclass_classic_result_root",
    "resolve_contiguous_threeclass_v4_result_root",
    "run_nested_stagewise_oof",
    "run_posterior_cardinal_contiguous_threeclass",
]
