"""Exploratory posterior-aware three-class classification with nested Optuna.

This workflow is deliberately independent of the frozen four-class v4
analysis.  It merges the posterior-soft ``low`` and ``60`` responses into
``low_30_40_60_like`` and evaluates the resulting
``low_30_40_60_like / 120_like / off_cardinal`` task.

Feature-panel, model-family, and hyperparameter selection occur exclusively
inside connected-group inner folds of each outer-training split.  The outer
test records are used once to evaluate the complete adaptive procedure.
Posterior argmax labels and hard accuracy are never used for fitting or
selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import optuna
import pandas as pd
import yaml

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
from .feature_association_features import (
    FROZEN_ASSOCIATION_PANELS,
)
from .posterior_cardinal_classification import (
    IDENTITY_COLUMNS,
    PRIMARY_WIDTH,
    _assert_fresh_disjoint_run_directory,
    _merge_primary_labels,
    _validate_run_frame,
    _width_id,
    build_shared_inner_manifest,
    build_shared_outer_manifest,
    derive_all_width_labels,
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
from .provenance import sha256_file, write_json
from .threeclass_optuna_models import (
    MODEL_SEARCH_SPACE,
    THREE_CLASS_LABELS,
    THREE_CLASS_MODEL_FAMILIES,
    fit_predict_pipeline,
    suggest_params,
)


CONTRACT_FILENAME = "posterior_cardinal_threeclass_optuna_v1.yaml"
CONTRACT_ID = "posterior_cardinal_threeclass_optuna_v1"
STAGE_NAME = "posterior-cardinal-threeclass-optuna"

Q_COLUMNS = tuple(f"q_{label}" for label in THREE_CLASS_LABELS)
P_COLUMNS = tuple(f"p_{label}" for label in THREE_CLASS_LABELS)
PRIMARY_Q_SOURCE_COLUMNS = (
    "q_low_30_40_like",
    "q_60_like",
    "q_120_like",
    "q_off_cardinal",
)
SEARCH_PANELS = ("context_core", *EXPECTED_FORMAL_PANELS)
MODEL_FAMILIES = THREE_CLASS_MODEL_FAMILIES
EVALUATED_WIDTHS = (PRIMARY_WIDTH,)
PRIOR_MODEL = "fold_local_threeclass_soft_prior"
SELECTED_MODEL = "nested_optuna_selected_pipeline"

# The full run intentionally reuses the exact v4 connected-group manifests.
EXPECTED_FULL_OUTER_MANIFEST_SHA256 = (
    "5fa6629fd7a5b8d0937434d5b7cf79b18b06f9da5d4d68e274f80f92f799489a"
)
EXPECTED_FULL_INNER_MANIFEST_SHA256 = (
    "7c1776bf98151e443285c965c57151e36310f378f5a412cfe4e34f6322f8bcf4"
)


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
    settings = config.section("posterior_cardinal_threeclass_optuna")
    if settings.get("contract_id") != CONTRACT_ID:
        raise ValueError("Configured three-class contract ID changed")
    if settings.get("status") != (
        "exploratory_posthoc_frozen_before_first_fit"
    ):
        raise ValueError("Configured three-class contract status changed")
    configured_path = Path(str(settings.get("contract_path", ""))).expanduser()
    path = (
        configured_path
        if configured_path.is_absolute()
        else (config.project_root / configured_path).resolve()
    )
    if path.name != CONTRACT_FILENAME:
        raise ValueError("Configured three-class contract path changed")
    if not path.is_file():
        raise FileNotFoundError(f"Three-class Optuna contract is missing: {path}")
    expected_hash = str(settings.get("contract_sha256", "")).strip().lower()
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Three-class Optuna contract hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("Three-class Optuna contract must be a mapping")
    contract = dict(loaded)
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError("Three-class Optuna contract ID changed")
    if contract.get("stage") != STAGE_NAME:
        raise ValueError("Three-class Optuna contract stage changed")
    if contract.get("status") != (
        "exploratory_posthoc_frozen_before_first_fit"
    ):
        raise ValueError(
            "Three-class contract must remain explicitly post-hoc exploratory"
        )

    scope = dict(contract.get("scientific_scope", {}))
    if (
        scope.get("population") != "all97"
        or int(scope.get("expected_records", -1)) != 97
        or int(scope.get("expected_connected_groups", -1)) != 80
    ):
        raise ValueError("Three-class scientific scope changed")
    if scope.get("one_ml_row_per_record") is not True:
        raise ValueError("Every record must remain one ML sample")
    if scope.get("posterior_draws_expanded_as_samples") is not False:
        raise ValueError("Posterior draws may not be expanded as ML samples")

    label = dict(contract.get("label_scheme", {}))
    if tuple(label.get("supervised_class_order", ())) != THREE_CLASS_LABELS:
        raise ValueError("Three-class supervised response order changed")
    merge_rule = dict(label.get("merge_rule", {}))
    if merge_rule.get("q_low_30_40_60_like") != (
        "q_low_30_40_like + q_60_like"
    ):
        raise ValueError("Three-class label merge changed")
    if float(label.get("relative_half_width", -1)) != PRIMARY_WIDTH:
        raise ValueError("Three-class primary label width changed")
    if label.get("posterior_argmax_training_or_tuning_use") != "forbidden":
        raise ValueError("Posterior argmax must remain forbidden for selection")

    features = dict(contract.get("feature_engineering", {}))
    if tuple(features.get("selectable_panels", ())) != SEARCH_PANELS:
        raise ValueError("Three-class feature-panel search space changed")
    if (
        features.get("fitted_inside_every_inner_and_outer_training_fold")
        is not True
    ):
        raise ValueError("Feature engineering must remain fold-local")
    if features.get("conductive_agent_panel_in_search") is not False:
        raise ValueError("Conductive-agent sensitivity entered formal selection")

    models = dict(contract.get("models", {}))
    if tuple(models.get("selectable_families", ())) != MODEL_FAMILIES:
        raise ValueError("Three-class model-family search space changed")
    gaussian = dict(models.get("two_stage_gaussian_prior_MAP", {}))
    gaussian_scale = dict(gaussian.get("feature_prior_scale", {}))
    expected_gaussian = MODEL_SEARCH_SPACE["two_stage_gaussian_prior_MAP"]
    if (
        gaussian_scale.get("distribution") != "log_float"
        or float(gaussian_scale.get("low", -1))
        != float(expected_gaussian["feature_prior_scale_low"])
        or float(gaussian_scale.get("high", -1))
        != float(expected_gaussian["feature_prior_scale_high"])
        or float(gaussian.get("intercept_prior_scale", -1))
        != float(expected_gaussian["intercept_prior_scale"])
        or int(gaussian.get("max_iter", -1))
        != int(expected_gaussian["max_iter"])
        or float(gaussian.get("tolerance", -1))
        != float(expected_gaussian["tolerance"])
    ):
        raise ValueError("Gaussian MAP code and contract search spaces differ")
    gower = dict(models.get("two_stage_soft_gower_knn", {}))
    expected_gower = MODEL_SEARCH_SPACE["two_stage_soft_gower_knn"]
    if (
        tuple(dict(gower.get("n_neighbors", {})).get("choices", ()))
        != tuple(expected_gower["n_neighbors"])
        or tuple(dict(gower.get("distance_power", {})).get("choices", ()))
        != tuple(expected_gower["distance_power"])
        or float(gower.get("epsilon", -1))
        != float(expected_gower["epsilon"])
    ):
        raise ValueError("Gower kNN code and contract search spaces differ")
    trees = dict(models.get("two_stage_soft_extra_trees", {}))
    expected_trees = MODEL_SEARCH_SPACE["two_stage_soft_extra_trees"]
    tree_fields = {
        "n_estimators": (
            "n_estimators_low",
            "n_estimators_high",
            "n_estimators_step",
        ),
        "max_depth": ("max_depth_low", "max_depth_high", None),
        "min_samples_leaf": (
            "min_samples_leaf_low",
            "min_samples_leaf_high",
            None,
        ),
        "max_features": ("max_features_low", "max_features_high", None),
    }
    for field, (low_key, high_key, step_key) in tree_fields.items():
        definition = dict(trees.get(field, {}))
        if (
            float(definition.get("low", -1))
            != float(expected_trees[low_key])
            or float(definition.get("high", -1))
            != float(expected_trees[high_key])
            or (
                step_key is not None
                and int(definition.get("step", -1))
                != int(expected_trees[step_key])
            )
        ):
            raise ValueError(
                f"Extra Trees code and contract differ for {field}"
            )
    if trees.get("estimator_n_jobs") != 1:
        raise ValueError("Extra Trees must remain single-threaded per fit")
    optuna_settings = dict(contract.get("optuna", {}))
    inner = dict(contract.get("inner_validation", {}))
    if inner.get("objective") != (
        "fold_equal_mean_full_coverage_three_class_soft_log_loss"
    ) or optuna_settings.get("trial_objective") != "inner_validation.objective":
        raise ValueError("Optuna objective changed")
    if optuna_settings.get("outer_test_predictions_visible_to_study") is not False:
        raise ValueError("Outer-test access must remain forbidden during tuning")
    if optuna_settings.get("study_n_jobs") != 1:
        raise ValueError("Each Optuna study must remain single-threaded")
    if (
        optuna_settings.get("sampler") != "TPESampler"
        or optuna_settings.get("pruner") != "NopPruner"
        or optuna_settings.get("direction") != "minimize"
        or int(optuna_settings.get("full_trials_per_outer_split", -1)) != 45
    ):
        raise ValueError("Optuna implementation and contract differ")
    if optuna_settings.get("sampler_seed_formula") != (
        "outer_split_seed + 30000 + outer_fold"
    ):
        raise ValueError("Optuna sampler seed formula changed")
    return contract, path, actual_hash


def resolve_threeclass_v4_result_root(config: StudyConfig) -> Path:
    """Return the immutable v4 result root bound by the three-class contract."""

    contract, _, _ = _load_contract(config)
    boundary = dict(contract["immutable_v4_boundary"])
    root = Path(str(boundary["source_run"])).expanduser()
    if not root.is_absolute():
        root = (config.repository_root / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Bound v4 result root is missing: {root}")
    return root


def collapse_to_three_classes(labels: pd.DataFrame) -> pd.DataFrame:
    """Merge low and 60 posterior mass without hardening the response."""

    missing = sorted(
        {"record_id", *PRIMARY_Q_SOURCE_COLUMNS} - set(labels.columns)
    )
    if missing:
        raise KeyError(f"Four-class labels are missing columns: {missing}")
    output = labels.copy()
    output[Q_COLUMNS[0]] = (
        pd.to_numeric(output["q_low_30_40_like"], errors="raise")
        + pd.to_numeric(output["q_60_like"], errors="raise")
    )
    output[Q_COLUMNS[1]] = pd.to_numeric(
        output["q_120_like"], errors="raise"
    )
    output[Q_COLUMNS[2]] = pd.to_numeric(
        output["q_off_cardinal"], errors="raise"
    )
    values = output.loc[:, Q_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or (values < -1e-12).any():
        raise ValueError("Three-class labels contain invalid probability mass")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Three-class posterior probabilities must sum to one")
    output.loc[:, Q_COLUMNS] = normalize_probabilities(values)
    modal_index = np.argmax(output.loc[:, Q_COLUMNS].to_numpy(float), axis=1)
    output["threeclass_modal_label"] = np.asarray(
        THREE_CLASS_LABELS, dtype=object
    )[modal_index]
    output["threeclass_max_probability"] = np.max(values, axis=1)
    output["threeclass_entropy"] = -np.sum(
        values * np.log(np.clip(values, 1e-300, 1.0)), axis=1
    )
    return output


def _label_support(labels: pd.DataFrame) -> pd.DataFrame:
    values = labels.loc[:, Q_COLUMNS].to_numpy(float)
    modal = np.argmax(values, axis=1)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(THREE_CLASS_LABELS):
        selected = modal == index
        rows.append(
            {
                "class_label": label,
                "effective_posterior_mass": float(values[:, index].sum()),
                "modal_record_count": int(selected.sum()),
                "modal_connected_group_count": int(
                    labels.loc[selected, "cv_group_id"].astype(str).nunique()
                ),
                "mean_probability": float(values[:, index].mean()),
            }
        )
    return pd.DataFrame(rows)


def _labels_by_id(labels: pd.DataFrame) -> pd.DataFrame:
    selected = labels.loc[:, ["record_id", *Q_COLUMNS]].copy()
    selected["record_id"] = selected["record_id"].astype(str)
    if selected["record_id"].duplicated().any():
        raise ValueError("Three-class labels must be unique by record_id")
    return selected.set_index("record_id")


def _records_by_id(records: pd.DataFrame) -> pd.DataFrame:
    output = records.copy()
    output["record_id"] = output["record_id"].astype(str)
    if output["record_id"].duplicated().any():
        raise ValueError("Records must be unique by record_id")
    return output.set_index("record_id", drop=False)


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
    ].copy()
    lookup = dict(
        zip(
            selected["record_id"].astype(str),
            selected["inner_fold"].astype(int),
            strict=True,
        )
    )
    ordered_ids = [str(record_id) for record_id in outer_train_ids]
    missing = sorted(set(ordered_ids) - set(lookup))
    extra = sorted(set(lookup) - set(ordered_ids))
    if missing or extra:
        raise ValueError(
            "Inner manifest does not exactly match outer training records: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    assignments = np.asarray([lookup[record_id] for record_id in ordered_ids])
    output: list[tuple[list[str], list[str], int]] = []
    for inner_fold in sorted(np.unique(assignments)):
        validation = [
            record_id
            for record_id, assignment in zip(
                ordered_ids, assignments, strict=True
            )
            if int(assignment) == int(inner_fold)
        ]
        training = [
            record_id
            for record_id, assignment in zip(
                ordered_ids, assignments, strict=True
            )
            if int(assignment) != int(inner_fold)
        ]
        if not training or not validation:
            raise ValueError("Grouped inner split contains an empty partition")
        output.append((training, validation, int(inner_fold)))
    return output


def _suggest_trial_parameters(
    trial: optuna.Trial,
    *,
    smoke: bool,
) -> tuple[str, str, dict[str, Any]]:
    del smoke
    panel = trial.suggest_categorical("panel", list(SEARCH_PANELS))
    model_family = trial.suggest_categorical(
        "model_family", list(MODEL_FAMILIES)
    )
    params = suggest_params(trial, str(model_family))
    return str(panel), str(model_family), params


def _default_enqueued_candidates(*, smoke: bool) -> list[dict[str, Any]]:
    defaults = {
        "two_stage_gaussian_prior_MAP": {
            "feature_prior_scale": 1.0,
        },
        "two_stage_soft_gower_knn": {
            "n_neighbors": 9,
            "distance_power": 1.0,
        },
        "two_stage_soft_extra_trees": {
            "n_estimators": 400,
            "max_depth": 3,
            "min_samples_leaf": 6,
            "max_features": 0.7,
        },
    }
    if smoke:
        # Exercise the context, composition, and property feature paths plus
        # all three models. Smoke selection is never a scientific winner.
        return [
            {
                "panel": "context_core",
                "model_family": "two_stage_gaussian_prior_MAP",
                **defaults["two_stage_gaussian_prior_MAP"],
            },
            {
                "panel": "context_plus_water_organic",
                "model_family": "two_stage_soft_gower_knn",
                **defaults["two_stage_soft_gower_knn"],
            },
            {
                "panel": "context_plus_polarity",
                "model_family": "two_stage_soft_extra_trees",
                **defaults["two_stage_soft_extra_trees"],
            },
        ]
    return [
        {
            "panel": panel,
            "model_family": model,
            **defaults[model],
        }
        for panel in SEARCH_PANELS
        for model in MODEL_FAMILIES
    ]


def _trial_simplicity_key(trial: optuna.trial.FrozenTrial) -> tuple[Any, ...]:
    params = trial.params
    panel_rank = int(
        FROZEN_ASSOCIATION_PANELS[str(params["panel"])].complexity_rank
    )
    model = str(params["model_family"])
    model_rank = {
        "two_stage_gaussian_prior_MAP": 1,
        "two_stage_soft_gower_knn": 2,
        "two_stage_soft_extra_trees": 3,
    }[model]
    return (
        model_rank,
        panel_rank,
        int(trial.number),
    )


def _select_best_trial(
    study: optuna.Study,
) -> optuna.trial.FrozenTrial:
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(float(trial.value))
    ]
    if not completed:
        raise RuntimeError("All Optuna trials failed")
    return min(
        completed,
        key=lambda trial: (
            float(trial.value),
            _trial_simplicity_key(trial),
        ),
    )


def _trial_rows(
    study: optuna.Study,
    *,
    repeat: int,
    fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        state = trial.state.name.lower()
        row = {
            "outer_repeat": int(repeat),
            "outer_fold": int(fold),
            "trial_number": int(trial.number),
            "state": state,
            "objective_soft_log_loss": (
                float(trial.value) if trial.value is not None else np.nan
            ),
            "panel": trial.params.get("panel"),
            "model_family": trial.params.get("model_family"),
            "parameters_json": json.dumps(
                _jsonable(trial.params), ensure_ascii=False, sort_keys=True
            ),
            "fold_standard_error": trial.user_attrs.get(
                "fold_standard_error", np.nan
            ),
            "failure": trial.user_attrs.get("failure"),
        }
        trials.append(row)
        for item in trial.user_attrs.get("inner_fold_scores", ()):
            fold_rows.append(
                {
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "trial_number": int(trial.number),
                    **dict(item),
                }
            )
    return trials, fold_rows


def _tune_outer_training_split(
    records_indexed: pd.DataFrame,
    primary_labels_indexed: pd.DataFrame,
    outer_train_ids: Sequence[str],
    inner_splits: Sequence[tuple[list[str], list[str], int]],
    *,
    repeat: int,
    fold: int,
    n_trials: int,
    sampler_seed: int,
    model_seed: int,
    smoke: bool,
    rare_min_frequency: int,
) -> tuple[
    optuna.trial.FrozenTrial,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    del outer_train_ids  # Coverage is enforced when constructing inner_splits.
    sampler = optuna.samplers.TPESampler(
        seed=int(sampler_seed),
        multivariate=False,
        n_startup_trials=min(10, max(1, int(n_trials))),
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        study_name=f"outer_repeat_{repeat:02d}_fold_{fold:02d}",
    )
    for candidate in _default_enqueued_candidates(smoke=smoke)[: int(n_trials)]:
        study.enqueue_trial(candidate)

    def objective(trial: optuna.Trial) -> float:
        panel, model_family, params = _suggest_trial_parameters(
            trial, smoke=smoke
        )
        scores: list[float] = []
        fold_audits: list[dict[str, Any]] = []
        try:
            for train_ids, validation_ids, inner_fold in inner_splits:
                prediction, audit = fit_predict_pipeline(
                    records_indexed.loc[list(train_ids)].reset_index(drop=True),
                    primary_labels_indexed.loc[
                        list(train_ids), list(Q_COLUMNS)
                    ].to_numpy(float),
                    records_indexed.loc[list(validation_ids)].reset_index(
                        drop=True
                    ),
                    panel=panel,
                    model_family=model_family,
                    params=params,
                    seed=int(model_seed),
                    rare_min_frequency=int(rare_min_frequency),
                )
                observed = primary_labels_indexed.loc[
                    list(validation_ids), list(Q_COLUMNS)
                ].to_numpy(float)
                loss = soft_cross_entropy(observed, prediction)
                scores.append(float(loss))
                fold_audits.append(
                    {
                        "inner_fold": int(inner_fold),
                        "n_training_records": int(len(train_ids)),
                        "n_validation_records": int(len(validation_ids)),
                        "soft_log_loss": float(loss),
                        "fit_audit_json": json.dumps(
                            _jsonable(audit),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
        except Exception as error:
            trial.set_user_attr("failure", repr(error))
            raise
        standard_error = (
            float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
            if len(scores) > 1
            else 0.0
        )
        trial.set_user_attr("inner_fold_scores", fold_audits)
        trial.set_user_attr("fold_standard_error", standard_error)
        trial.set_user_attr("selection_metric", "three_class_soft_log_loss")
        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=int(n_trials),
        n_jobs=1,
        show_progress_bar=False,
        catch=(ValueError, RuntimeError, FloatingPointError),
    )
    selected = _select_best_trial(study)
    trial_rows, fold_rows = _trial_rows(study, repeat=repeat, fold=fold)
    return selected, trial_rows, fold_rows


def _run_outer_split(
    records_indexed: pd.DataFrame,
    labels_by_width_indexed: Mapping[float, pd.DataFrame],
    outer_manifest: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    n_trials: int,
    base_sampler_seed: int,
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
    if set(train_ids) & test_set or len(train_ids) + len(test_ids) != len(all_ids):
        raise AssertionError("Outer split does not partition the records")
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
    sampler_seed = (
        outer_split_seed + int(base_sampler_seed) + int(fold)
    )
    (
        selected_trial,
        trial_rows,
        inner_fold_rows,
    ) = _tune_outer_training_split(
        records_indexed,
        labels_by_width_indexed[PRIMARY_WIDTH],
        train_ids,
        inner_splits,
        repeat=repeat,
        fold=fold,
        n_trials=n_trials,
        sampler_seed=sampler_seed,
        model_seed=outer_split_seed,
        smoke=smoke,
        rare_min_frequency=rare_min_frequency,
    )
    panel = str(selected_trial.params["panel"])
    model_family = str(selected_trial.params["model_family"])
    model_params = {
        key: value
        for key, value in selected_trial.params.items()
        if key not in {"panel", "model_family"}
    }
    prediction_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for width in EVALUATED_WIDTHS:
        indexed_labels = labels_by_width_indexed[float(width)]
        train_target = indexed_labels.loc[
            train_ids, list(Q_COLUMNS)
        ].to_numpy(float)
        test_target = indexed_labels.loc[
            test_ids, list(Q_COLUMNS)
        ].to_numpy(float)
        prediction, fit_audit = fit_predict_pipeline(
            records_indexed.loc[train_ids].reset_index(drop=True),
            train_target,
            records_indexed.loc[test_ids].reset_index(drop=True),
            panel=panel,
            model_family=model_family,
            params=model_params,
            seed=outer_split_seed,
            rare_min_frequency=rare_min_frequency,
        )
        prior = normalize_probabilities(
            np.mean(train_target, axis=0, keepdims=True)
        )[0]
        prior_prediction = np.repeat(prior[None, :], len(test_ids), axis=0)
        for evaluation_model, probabilities in (
            (PRIOR_MODEL, prior_prediction),
            (SELECTED_MODEL, prediction),
        ):
            for position, record_id in enumerate(test_ids):
                row: dict[str, Any] = {
                    "width": float(width),
                    "width_id": _width_id(float(width)),
                    "evaluation_model": evaluation_model,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "record_id": str(record_id),
                    "cv_group_id": str(
                        records_indexed.loc[record_id, "cv_group_id"]
                    ),
                    "selected_panel": (
                        panel if evaluation_model == SELECTED_MODEL else "prior"
                    ),
                    "selected_model_family": (
                        model_family
                        if evaluation_model == SELECTED_MODEL
                        else PRIOR_MODEL
                    ),
                    "selected_trial_number": (
                        int(selected_trial.number)
                        if evaluation_model == SELECTED_MODEL
                        else -1
                    ),
                }
                row.update(
                    {
                        column: float(test_target[position, index])
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
        fit_rows.append(
            {
                "outer_repeat": int(repeat),
                "outer_fold": int(fold),
                "width": float(width),
                "width_id": _width_id(float(width)),
                "selected_panel": panel,
                "selected_model_family": model_family,
                "selected_trial_number": int(selected_trial.number),
                "selected_parameters_json": json.dumps(
                    _jsonable(model_params),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "fit_audit_json": json.dumps(
                    _jsonable(fit_audit),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    selection = {
        "outer_repeat": int(repeat),
        "outer_fold": int(fold),
        "sampler_seed": int(sampler_seed),
        "model_random_seed": int(outer_split_seed),
        "n_trials_requested": int(n_trials),
        "selected_trial_number": int(selected_trial.number),
        "selected_inner_soft_log_loss": float(selected_trial.value),
        "selected_inner_fold_standard_error": float(
            selected_trial.user_attrs.get("fold_standard_error", 0.0)
        ),
        "selected_panel": panel,
        "selected_feature": (
            "none_context_only"
            if panel == "context_core"
            else "|".join(FROZEN_ASSOCIATION_PANELS[panel].added_features)
        ),
        "selected_model_family": model_family,
        "selected_parameters_json": json.dumps(
            _jsonable(model_params), ensure_ascii=False, sort_keys=True
        ),
        "selection_rule": (
            "minimum_inner_soft_log_loss_then_model_panel_trial_tie_break"
        ),
        "outer_test_used_for_selection": False,
    }
    return {
        "predictions": prediction_rows,
        "trials": trial_rows,
        "inner_fold_scores": inner_fold_rows,
        "selection": selection,
        "fits": fit_rows,
    }


def run_nested_optuna_oof(
    records: pd.DataFrame,
    labels_by_width: Mapping[float, pd.DataFrame],
    outer_manifest: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    *,
    n_trials: int,
    base_sampler_seed: int,
    smoke: bool,
    rare_min_frequency: int,
    n_jobs: int,
) -> dict[str, pd.DataFrame]:
    """Tune inside every outer training split and return honest OOF results."""

    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    if int(n_trials) < 1:
        raise ValueError("n_trials must be positive")
    if not smoke and int(n_trials) < len(SEARCH_PANELS) * len(MODEL_FAMILIES):
        raise ValueError(
            "A full run needs at least one enqueued trial for every "
            "feature-panel/model-family pair"
        )
    records_indexed = _records_by_id(records)
    labels_indexed = {
        float(width): _labels_by_id(labels)
        for width, labels in labels_by_width.items()
    }
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
            base_sampler_seed=int(base_sampler_seed),
            smoke=smoke,
            rare_min_frequency=int(rare_min_frequency),
        )
        for repeat, fold in tasks
    )
    predictions = pd.DataFrame(
        row for result in results for row in result["predictions"]
    )
    trials = pd.DataFrame(row for result in results for row in result["trials"])
    inner_scores = pd.DataFrame(
        row for result in results for row in result["inner_fold_scores"]
    )
    selections = pd.DataFrame(result["selection"] for result in results)
    fits = pd.DataFrame(row for result in results for row in result["fits"])

    expected_repeats = int(outer_manifest["repeat"].nunique())
    coverage = predictions.groupby(
        ["width", "evaluation_model", "record_id"]
    ).size()
    if not coverage.eq(expected_repeats).all():
        raise AssertionError("Repeated three-class OOF coverage is incomplete")
    probabilities = predictions.loc[:, P_COLUMNS].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0
    ):
        raise AssertionError("OOF predictions are not valid probabilities")
    return {
        "oof_probabilities": predictions,
        "optuna_trials": trials,
        "inner_fold_scores": inner_scores,
        "outer_selections": selections,
        "model_fits": fits,
    }


def aggregate_threeclass_oof(oof: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "width",
        "width_id",
        "evaluation_model",
        "record_id",
        "cv_group_id",
    ]
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


def _threeclass_probability_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    entropy_threshold: float,
) -> dict[str, float | int]:
    observed_array = normalize_probabilities(observed)
    predicted_array = normalize_probabilities(predicted)
    observed_cardinal = observed_array[:, :2].sum(axis=1)
    predicted_cardinal = predicted_array[:, :2].sum(axis=1)
    observed_gate = np.column_stack(
        [observed_cardinal, observed_array[:, 2]]
    )
    predicted_gate = np.column_stack(
        [predicted_cardinal, predicted_array[:, 2]]
    )
    conditional_observed = np.divide(
        observed_array[:, :2],
        observed_cardinal[:, None],
        out=np.full((len(observed_array), 2), 0.5, dtype=float),
        where=observed_cardinal[:, None] > 0,
    )
    conditional_predicted = np.divide(
        predicted_array[:, :2],
        predicted_cardinal[:, None],
        out=np.full((len(predicted_array), 2), 0.5, dtype=float),
        where=predicted_cardinal[:, None] > 0,
    )
    overall = descriptive_probability_metrics(
        observed_array,
        predicted_array,
        entropy_threshold=entropy_threshold,
    )
    conditional_log_loss = soft_cross_entropy(
        conditional_observed,
        conditional_predicted,
        sample_weight=observed_cardinal,
    )
    conditional_brier = multiclass_brier(
        conditional_observed,
        conditional_predicted,
        sample_weight=observed_cardinal,
    )
    return {
        **overall,
        "full_coverage_three_class_soft_log_loss": float(
            overall["soft_log_loss"]
        ),
        "multiclass_brier_mean_sum_squared_error": float(
            overall["multiclass_brier"]
        ),
        "gate_soft_log_loss": soft_cross_entropy(
            observed_gate, predicted_gate
        ),
        "gate_brier": multiclass_brier(observed_gate, predicted_gate),
        "conditional_soft_log_loss": conditional_log_loss,
        "conditional_brier": conditional_brier,
        "conditional_soft_log_loss_weighted_by_q_cardinal": (
            conditional_log_loss
        ),
        "conditional_brier_weighted_by_q_cardinal": conditional_brier,
        "mean_observed_cardinal_probability": float(
            np.mean(observed_cardinal)
        ),
        "mean_predicted_cardinal_probability": float(
            np.mean(predicted_cardinal)
        ),
    }


def _metric_rows(
    oof: pd.DataFrame,
    aggregated: pd.DataFrame,
    *,
    entropy_threshold: float,
) -> dict[str, pd.DataFrame]:
    aggregate_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    for (width, width_id, model), frame in aggregated.groupby(
        ["width", "width_id", "evaluation_model"], sort=True
    ):
        observed = frame.loc[:, Q_COLUMNS].to_numpy(float)
        predicted = frame.loc[:, P_COLUMNS].to_numpy(float)
        aggregate_rows.append(
            {
                "width": float(width),
                "width_id": str(width_id),
                "evaluation_model": str(model),
                "n_records": int(len(frame)),
                "n_connected_groups": int(frame["cv_group_id"].nunique()),
                **_threeclass_probability_metrics(
                    observed,
                    predicted,
                    entropy_threshold=entropy_threshold,
                ),
            }
        )
        calibration_frames.append(
            calibration_table(
                observed,
                predicted,
                class_labels=THREE_CLASS_LABELS,
                n_bins=8,
            ).assign(
                width=float(width),
                width_id=str(width_id),
                evaluation_model=str(model),
            )
        )
        confusion_frames.append(
            expected_confusion_table(
                observed,
                predicted,
                class_labels=THREE_CLASS_LABELS,
            ).assign(
                width=float(width),
                width_id=str(width_id),
                evaluation_model=str(model),
            )
        )

    repeat_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for keys, frame in oof.groupby(
        ["width", "width_id", "evaluation_model", "repeat"], sort=True
    ):
        width, width_id, model, repeat = keys
        repeat_rows.append(
            {
                "width": float(width),
                "width_id": str(width_id),
                "evaluation_model": str(model),
                "repeat": int(repeat),
                **_threeclass_probability_metrics(
                    frame.loc[:, Q_COLUMNS].to_numpy(float),
                    frame.loc[:, P_COLUMNS].to_numpy(float),
                    entropy_threshold=entropy_threshold,
                ),
            }
        )
    for keys, frame in oof.groupby(
        ["width", "width_id", "evaluation_model", "repeat", "fold"],
        sort=True,
    ):
        width, width_id, model, repeat, fold = keys
        split_rows.append(
            {
                "width": float(width),
                "width_id": str(width_id),
                "evaluation_model": str(model),
                "repeat": int(repeat),
                "fold": int(fold),
                "n_test_records": int(len(frame)),
                **_threeclass_probability_metrics(
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
    width: float,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = aggregated.loc[
        aggregated["width"].eq(float(width))
        & aggregated["evaluation_model"].eq(SELECTED_MODEL)
    ].copy()
    prior = aggregated.loc[
        aggregated["width"].eq(float(width))
        & aggregated["evaluation_model"].eq(PRIOR_MODEL)
    ].copy()
    merged = selected.merge(
        prior,
        on=["record_id", "cv_group_id", "width", "width_id"],
        suffixes=("_selected", "_prior"),
        validate="one_to_one",
    )
    if len(merged) != 97:
        raise AssertionError("Paired inference must retain all 97 records")
    q = merged.loc[
        :, [f"{column}_selected" for column in Q_COLUMNS]
    ].to_numpy(float)
    p_selected = merged.loc[
        :, [f"{column}_selected" for column in P_COLUMNS]
    ].to_numpy(float)
    p_prior = merged.loc[
        :, [f"{column}_prior" for column in P_COLUMNS]
    ].to_numpy(float)
    epsilon = 1e-15
    selected_loss = -np.sum(
        q * np.log(np.clip(p_selected, epsilon, 1.0)), axis=1
    )
    prior_loss = -np.sum(
        q * np.log(np.clip(p_prior, epsilon, 1.0)), axis=1
    )
    selected_brier = np.sum(np.square(p_selected - q), axis=1)
    prior_brier = np.sum(np.square(p_prior - q), axis=1)
    selected_modal = np.argmax(p_selected, axis=1)
    prior_modal = np.argmax(p_prior, axis=1)
    positions = np.arange(len(q))
    record_scores = pd.DataFrame(
        {
            "record_id": merged["record_id"].astype(str),
            "cv_group_id": merged["cv_group_id"].astype(str),
            "selected_soft_log_loss": selected_loss,
            "prior_soft_log_loss": prior_loss,
            "delta_soft_log_loss": selected_loss - prior_loss,
            "selected_multiclass_brier": selected_brier,
            "prior_multiclass_brier": prior_brier,
            "delta_multiclass_brier": selected_brier - prior_brier,
            "selected_posterior_expected_hard_accuracy": q[
                positions, selected_modal
            ],
            "prior_posterior_expected_hard_accuracy": q[
                positions, prior_modal
            ],
            "delta_posterior_expected_hard_accuracy": (
                q[positions, selected_modal] - q[positions, prior_modal]
            ),
        }
    )
    draws, summary = paired_connected_group_bootstrap(
        record_scores,
        [
            "delta_soft_log_loss",
            "delta_multiclass_brier",
            "delta_posterior_expected_hard_accuracy",
        ],
        replicates=int(bootstrap_repetitions),
        seed=int(bootstrap_seed),
    )
    return record_scores, draws, summary


def _selection_tables(
    selections: pd.DataFrame,
    *,
    stability_threshold: float,
    smoke: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_splits = len(selections)
    tables: list[pd.DataFrame] = []
    for dimension, column in (
        ("feature_panel", "selected_panel"),
        ("model_family", "selected_model_family"),
    ):
        frequency = (
            selections.groupby(column, as_index=False)
            .size()
            .rename(columns={column: "candidate", "size": "selection_count"})
        )
        frequency["selection_fraction"] = (
            frequency["selection_count"] / max(n_splits, 1)
        )
        frequency["dimension"] = dimension
        tables.append(frequency)
    joint = (
        selections.groupby(
            ["selected_panel", "selected_model_family"], as_index=False
        )
        .size()
        .rename(
            columns={
                "selected_panel": "feature_panel",
                "selected_model_family": "model_family",
                "size": "selection_count",
            }
        )
    )
    joint["selection_fraction"] = joint["selection_count"] / max(n_splits, 1)
    joint["dimension"] = "joint_pipeline"
    joint["candidate"] = (
        joint["feature_panel"].astype(str)
        + "::"
        + joint["model_family"].astype(str)
    )
    frequency = pd.concat(
        [
            *tables,
            joint.loc[
                :,
                [
                    "candidate",
                    "selection_count",
                    "selection_fraction",
                    "dimension",
                ],
            ],
        ],
        ignore_index=True,
    ).sort_values(
        ["dimension", "selection_count", "candidate"],
        ascending=[True, False, True],
        kind="stable",
    )

    def winner(dimension: str) -> dict[str, Any]:
        selected = frequency.loc[frequency["dimension"].eq(dimension)].sort_values(
            ["selection_count", "candidate"],
            ascending=[False, True],
            kind="stable",
        )
        first = selected.iloc[0]
        fraction = float(first["selection_fraction"])
        return {
            "candidate": str(first["candidate"]),
            "selection_count": int(first["selection_count"]),
            "selection_fraction": fraction,
            "stable": bool(
                not smoke and fraction >= float(stability_threshold)
            ),
        }

    summary = {
        "selection_basis": (
            "frequency_of_inner_CV_selections_across_outer_training_splits"
        ),
        "outer_test_metrics_used_to_choose_winner": False,
        "global_single_best_claim_authorized": False,
        "separate_validation_available": False,
        "stability_threshold": float(stability_threshold),
        "smoke_structural_only": bool(smoke),
        "feature_panel": winner("feature_panel"),
        "model_family": winner("model_family"),
        "joint_pipeline": winner("joint_pipeline"),
    }
    summary["overall_status"] = (
        "structural_smoke_no_winner"
        if smoke
        else "descriptive_selection_concentration_without_winner_claim"
        if summary["joint_pipeline"]["stable"]
        else "no_stable_joint_winner"
    )
    return frequency.reset_index(drop=True), summary


def _write_report(
    destination: Path,
    *,
    metrics: pd.DataFrame,
    support: pd.DataFrame,
    selection_summary: Mapping[str, Any],
    bootstrap_summary: pd.DataFrame,
    smoke: bool,
) -> Path:
    primary = metrics.loc[metrics["width"].eq(PRIMARY_WIDTH)].copy()
    selected = primary.loc[
        primary["evaluation_model"].eq(SELECTED_MODEL)
    ].iloc[0]
    lines = [
        "# Posterior-aware three-class Optuna exploration",
        "",
        (
            "**Structural smoke only; no performance or winner claim is "
            "permitted.**"
            if smoke
            else (
                "This is a post-hoc exploratory three-class task. It does not "
                "replace or amend the frozen four-class v4 result."
            )
        ),
        "",
        "## Endpoint",
        "",
        (
            "The response is the posterior-soft vector "
            "`low_30_40_60_like / 120_like / off_cardinal`. Each experimental "
            "record remains one ML row; posterior draws are not pseudo-samples."
        ),
        "",
        support.to_markdown(index=False),
        "",
        "## Honest nested-CV performance",
        "",
        (
            "Optuna selected the feature panel, model family, and "
            "hyperparameters using connected-group inner folds only. Outer "
            "test records evaluated the complete adaptive procedure."
        ),
        "",
        primary.to_markdown(index=False),
        "",
        (
            f"The nested selected procedure has descriptive modal accuracy "
            f"`{float(selected['modal_accuracy']):.6f}` and "
            f"posterior-expected hard accuracy "
            f"`{float(selected['posterior_expected_hard_accuracy']):.6f}`. "
            "Neither quantity was used by Optuna; three-class soft log loss "
            "is the primary selection and evaluation metric."
        ),
        "",
        "## Feature and model selection stability",
        "",
        "```json",
        json.dumps(
            _jsonable(selection_summary),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Selected pipeline minus fold-local prior",
        "",
        bootstrap_summary.to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        (
            "The classes are apparent Tafel-slope neighborhoods, not reaction "
            "mechanisms. Cross-publication/laboratory prediction is "
            "associational and cannot establish a causal formulation effect."
        ),
        "",
    ]
    report = (
        destination
        / "report"
        / "posterior_cardinal_threeclass_optuna_report.md"
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_posterior_cardinal_threeclass_optuna(
    config: StudyConfig,
    run_dir: str | Path,
    *,
    source_run: str | Path,
    smoke: bool = False,
    n_jobs: int = 1,
    optuna_trials: int | None = None,
) -> dict[str, Any]:
    """Run the independent all-97 three-class nested Optuna exploration."""

    destination = Path(run_dir).expanduser().resolve()
    source = Path(source_run).expanduser().resolve()
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    if optuna_trials is not None and int(optuna_trials) < 1:
        raise ValueError("optuna_trials must be positive")
    _assert_fresh_disjoint_run_directory(destination, source)
    contract, contract_path, contract_hash = _load_contract(config)
    contract_trials = int(contract["optuna"]["full_trials_per_outer_split"])
    if (
        not smoke
        and optuna_trials is not None
        and int(optuna_trials) != contract_trials
    ):
        raise ValueError(
            "Full three-class runs use the hash-bound "
            f"{contract_trials} Optuna trials per outer split; "
            "--optuna-trials is a smoke-only override"
        )
    v4_result_root = resolve_threeclass_v4_result_root(config)
    assert_disjoint_cardinal_paths(v4_result_root, destination)
    v4_contract, v4_contract_path, v4_contract_hash = (
        load_frozen_posterior_cardinal_contract(config)
    )
    parent = dict(contract.get("immutable_v4_boundary", {}))
    if (
        parent.get("source_contract_id") != v4_contract["contract_id"]
        or parent.get("source_contract_sha256") != v4_contract_hash
    ):
        raise ValueError("Three-class contract is not bound to active v4")

    for name in (
        "inputs",
        "inputs/reduced_feature_registry",
        "registry",
        "folds",
        "analyses",
        "analyses/inference",
        "report",
        "provenance",
    ):
        (destination / name).mkdir(parents=True, exist_ok=True)
    contract_snapshot = destination / "registry" / "contract_snapshot.yaml"
    shutil.copy2(contract_path, contract_snapshot)
    if sha256_file(contract_snapshot) != contract_hash:
        raise RuntimeError("Three-class contract changed while copying")

    reduced_binding = snapshot_reduced_feature_registries(
        config,
        v4_contract,
        destination / "inputs" / "reduced_feature_registry",
    )
    source_hashes = v4_contract["formal_input_snapshot"]["source_file_sha256"]
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

    raw_by_width, four_class_by_width = derive_all_width_labels(records, draws)
    three_class_by_width = {
        PRIMARY_WIDTH: collapse_to_three_classes(
            four_class_by_width[PRIMARY_WIDTH]
        )
    }
    support_frames: list[pd.DataFrame] = []
    for width in EVALUATED_WIDTHS:
        width_dir = destination / "registry" / _width_id(float(width))
        width_dir.mkdir(parents=True, exist_ok=True)
        raw_by_width[float(width)].to_csv(
            width_dir / "raw_anchor_soft_labels.csv", index=False
        )
        four_class_by_width[float(width)].to_csv(
            width_dir / "four_class_soft_labels.csv", index=False
        )
        three_class_by_width[float(width)].to_csv(
            width_dir / "three_class_soft_labels.csv", index=False
        )
        support = _label_support(three_class_by_width[float(width)]).assign(
            width=float(width), width_id=_width_id(float(width))
        )
        support.to_csv(width_dir / "three_class_support.csv", index=False)
        support_frames.append(support)
    label_support = pd.concat(support_frames, ignore_index=True)
    label_support.to_csv(
        destination / "registry" / "three_class_support_all_widths.csv",
        index=False,
    )
    three_class_by_width[PRIMARY_WIDTH].to_csv(
        destination / "registry" / "three_class_soft_labels.csv",
        index=False,
    )
    label_support.loc[label_support["width"].eq(PRIMARY_WIDTH)].to_csv(
        destination / "registry" / "three_class_support.csv",
        index=False,
    )

    # Smoke generates smaller structural folds from the parent v4 response.
    # Full mode copies the authoritative v4 manifests byte-for-byte.
    parent_primary_frame = _merge_primary_labels(
        records, four_class_by_width[PRIMARY_WIDTH]
    )
    outer_path = destination / "folds" / "outer_split_manifest.csv"
    inner_path = destination / "folds" / "inner_split_manifest.csv"
    if smoke:
        outer_manifest = build_shared_outer_manifest(
            parent_primary_frame, v4_contract, smoke=True
        )
        inner_manifest = build_shared_inner_manifest(
            parent_primary_frame,
            outer_manifest,
            v4_contract,
            smoke=True,
        )
        outer_manifest.to_csv(outer_path, index=False)
        inner_manifest.to_csv(inner_path, index=False)
    else:
        outer_definition = dict(parent["source_outer_manifest"])
        inner_definition = dict(parent["source_inner_manifest"])
        source_outer = (
            v4_result_root / str(outer_definition["path"])
        ).resolve()
        source_inner = (
            v4_result_root / str(inner_definition["path"])
        ).resolve()
        if (
            sha256_file(source_outer) != outer_definition["sha256"]
            or sha256_file(source_inner) != inner_definition["sha256"]
        ):
            raise ValueError("Authoritative v4 fold manifest hash mismatch")
        shutil.copy2(source_outer, outer_path)
        shutil.copy2(source_inner, inner_path)
        outer_manifest = pd.read_csv(outer_path)
        inner_manifest = pd.read_csv(inner_path)
    outer_hash = sha256_file(outer_path)
    inner_hash = sha256_file(inner_path)
    if not smoke and (
        outer_hash != EXPECTED_FULL_OUTER_MANIFEST_SHA256
        or inner_hash != EXPECTED_FULL_INNER_MANIFEST_SHA256
    ):
        raise ValueError(
            "Full three-class run did not reproduce the exact v4 fold manifests"
        )

    optuna_settings = dict(contract["optuna"])
    smoke_settings = dict(contract["smoke_mode"])
    requested_trials = int(
        optuna_trials
        if optuna_trials is not None
        else (
            smoke_settings["optuna_trials_per_outer_split"]
            if smoke
            else optuna_settings["full_trials_per_outer_split"]
        )
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    nested = run_nested_optuna_oof(
        records,
        three_class_by_width,
        outer_manifest,
        inner_manifest,
        n_trials=requested_trials,
        base_sampler_seed=30_000,
        smoke=smoke,
        rare_min_frequency=int(
            contract["feature_engineering"]["rare_min_frequency"]
        ),
        n_jobs=int(n_jobs),
    )
    oof = nested["oof_probabilities"]
    aggregated = aggregate_threeclass_oof(oof)
    metric_tables = _metric_rows(
        oof,
        aggregated,
        entropy_threshold=float(
            contract["metrics"]["predictive_entropy_threshold"]
        ),
    )
    bootstrap_repetitions = int(
        smoke_settings["bootstrap_repetitions"]
        if smoke
        else contract["connected_group_inference"]["bootstrap"][
            "full_repetitions"
        ]
    )
    record_scores, bootstrap_draws, bootstrap_summary = (
        _record_delta_inference(
            aggregated,
            width=PRIMARY_WIDTH,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=int(
                contract["connected_group_inference"]["bootstrap"]["seed"]
            ),
        )
    )
    selection_frequency, selection_summary = _selection_tables(
        nested["outer_selections"],
        stability_threshold=float(
            contract["winner_reporting"]["stability_threshold"]
        ),
        smoke=smoke,
    )

    nested["oof_probabilities"].to_csv(
        destination / "analyses" / "oof_probabilities.csv", index=False
    )
    aggregated.to_csv(
        destination / "analyses" / "oof_aggregated.csv", index=False
    )
    nested["optuna_trials"].to_csv(
        destination / "analyses" / "optuna_trials.csv", index=False
    )
    nested["inner_fold_scores"].to_csv(
        destination / "analyses" / "optuna_inner_fold_scores.csv", index=False
    )
    nested["outer_selections"].to_csv(
        destination / "analyses" / "outer_selections.csv", index=False
    )
    nested["model_fits"].to_csv(
        destination / "analyses" / "model_fit_registry.csv", index=False
    )
    selection_frequency.to_csv(
        destination / "analyses" / "selection_frequency.csv", index=False
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
    write_json(
        destination / "analyses" / "best_pipeline_summary.json",
        _jsonable(selection_summary),
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
            "frozen_before_first_fit": True,
        },
    )
    write_json(
        destination / "provenance" / "parent_v4_contract_binding.json",
        {
            "contract_id": v4_contract["contract_id"],
            "source_path": str(v4_contract_path),
            "sha256": v4_contract_hash,
        },
    )
    write_json(
        destination / "provenance" / "input_binding.json",
        input_binding,
    )
    write_json(
        destination / "provenance" / "reduced_feature_registry_binding.json",
        reduced_binding,
    )
    write_json(
        destination / "provenance" / "input_snapshot_verification_after_parse.json",
        parsed_snapshot_verification,
    )
    write_json(
        destination / "provenance" / "input_snapshot_verification_after_modeling.json",
        modeled_snapshot_verification,
    )
    write_json(
        destination / "provenance" / "reduced_registry_verification_after_parse.json",
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
        destination / "provenance" / "v4_manifest_binding.json",
        {
            "full_run_reuses_exact_v4_manifests": not smoke,
            "outer_manifest_sha256": outer_hash,
            "inner_manifest_sha256": inner_hash,
            "expected_full_outer_manifest_sha256": (
                EXPECTED_FULL_OUTER_MANIFEST_SHA256
            ),
            "expected_full_inner_manifest_sha256": (
                EXPECTED_FULL_INNER_MANIFEST_SHA256
            ),
            "matches_expected_full_hashes": bool(
                not smoke
                and outer_hash == EXPECTED_FULL_OUTER_MANIFEST_SHA256
                and inner_hash == EXPECTED_FULL_INNER_MANIFEST_SHA256
            ),
        },
    )

    metrics = metric_tables["aggregate_metrics"]
    primary_selected = metrics.loc[
        metrics["width"].eq(PRIMARY_WIDTH)
        & metrics["evaluation_model"].eq(SELECTED_MODEL)
    ].iloc[0]
    support_primary = label_support.loc[
        label_support["width"].eq(PRIMARY_WIDTH)
    ]
    oracle = float(
        primary_selected["oracle_posterior_expected_hard_accuracy"]
    )
    qa = {
        "status": "complete_smoke" if smoke else "complete_exploratory",
        "scientifically_releasable": False,
        "formal_v4_result_modified": False,
        "posthoc_exploratory_task": True,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_hash,
        "parent_v4_contract_id": v4_contract["contract_id"],
        "parent_v4_contract_sha256": v4_contract_hash,
        "n_records": int(len(records)),
        "n_connected_groups": int(records["cv_group_id"].nunique()),
        "n_posterior_draws": int(len(draws)),
        "draws_per_record": int(
            draws.groupby("sample_record").size().iloc[0]
        ),
        "class_order": list(THREE_CLASS_LABELS),
        "primary_effective_class_mass": {
            str(row.class_label): float(row.effective_posterior_mass)
            for row in support_primary.itertuples(index=False)
        },
        "outer_repeats": int(outer_manifest["repeat"].nunique()),
        "outer_folds": int(outer_manifest["fold"].nunique()),
        "inner_folds": int(inner_manifest["inner_fold"].nunique()),
        "optuna_trials_per_outer_split": int(requested_trials),
        "n_outer_selections": int(len(nested["outer_selections"])),
        "n_optuna_trials_total": int(len(nested["optuna_trials"])),
        "n_optuna_trials_complete": int(
            nested["optuna_trials"]["state"].eq("complete").sum()
        ),
        "feature_engineering_fit_inside_training_folds": True,
        "optuna_selection_inside_outer_training_only": True,
        "outer_test_used_for_feature_model_or_hyperparameter_selection": False,
        "one_ml_row_per_record": True,
        "posterior_draws_expanded_as_ml_samples": False,
        "posterior_argmax_used_for_training_or_selection": False,
        "hard_accuracy_used_for_training_or_selection": False,
        "primary_metric": "full_coverage_three_class_soft_log_loss",
        "modal_accuracy_descriptive_only": True,
        "nested_oof_modal_accuracy": float(
            primary_selected["modal_accuracy"]
        ),
        "nested_oof_posterior_expected_hard_accuracy": float(
            primary_selected["posterior_expected_hard_accuracy"]
        ),
        "posterior_expected_hard_accuracy_oracle_ceiling": oracle,
        "requested_accuracy_0_70_oracle_reachable": bool(oracle >= 0.70),
        "nested_oof_modal_accuracy_reaches_0_70": bool(
            float(primary_selected["modal_accuracy"]) >= 0.70
        ),
        "selection_summary": selection_summary,
        "full_manifest_hashes_match_v4": bool(
            not smoke
            and outer_hash == EXPECTED_FULL_OUTER_MANIFEST_SHA256
            and inner_hash == EXPECTED_FULL_INNER_MANIFEST_SHA256
        ),
        "input_snapshot_verified_after_modeling": (
            modeled_snapshot_verification["status"] == "pass"
        ),
        "reduced_registry_verified_after_modeling": (
            modeled_registry_verification["status"] == "pass"
        ),
        "formal_source_run_unchanged": bool(source_verification["unchanged"]),
        "mechanistic_claim_allowed": False,
        "causal_claim_allowed": False,
    }
    write_json(destination / "analyses" / "analysis_qa.json", _jsonable(qa))
    report = _write_report(
        destination,
        metrics=metrics,
        support=support_primary,
        selection_summary=selection_summary,
        bootstrap_summary=bootstrap_summary,
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
                "all97_posterior_soft_connected_group_nested_optuna"
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
    "CONTRACT_ID",
    "MODEL_FAMILIES",
    "P_COLUMNS",
    "PRIOR_MODEL",
    "Q_COLUMNS",
    "SEARCH_PANELS",
    "SELECTED_MODEL",
    "STAGE_NAME",
    "THREE_CLASS_LABELS",
    "aggregate_threeclass_oof",
    "collapse_to_three_classes",
    "resolve_threeclass_v4_result_root",
    "run_nested_optuna_oof",
    "run_posterior_cardinal_threeclass_optuna",
]
