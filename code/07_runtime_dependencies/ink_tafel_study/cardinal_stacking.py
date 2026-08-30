"""Fixed-recipe, fold-local probability stacking for cardinal development.

This module deliberately implements a narrow comparison rather than another
recipe/model search.  For each outer-training set and each predeclared feature
recipe, soft Gower kNN and soft multinomial logistic probabilities are
generated for every record by connected-group inner OOF prediction.  A single
convex weight is fitted from those inner OOF probabilities and then applied to
the outer-test probabilities from models refitted on the complete
outer-training set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .cardinal_cv import (
    make_inner_soft_group_splits,
    make_soft_stratified_group_manifest,
)
from .cardinal_evaluate import (
    CardinalEvaluationConfig,
    evaluate_paired_panel_contrasts,
)
from .cardinal_feature_engineering import (
    CardinalFeatureBuilder,
    required_cardinal_feature_columns,
)
from .cardinal_metrics import (
    CLASS_LABELS,
    PREDICTION_COLUMNS,
    SOFT_LABEL_COLUMNS,
    aggregate_repeated_oof,
    calibration_table,
    collapsed_hard_metrics,
    expected_confusion_table,
    normalize_probabilities,
    soft_cross_entropy,
    two_stage_probability_metrics,
)
from .cardinal_models import EngineeredCardinalClassifier
from .evaluate import (
    DEFAULT_RELATION_COLUMNS,
    assert_no_relational_leakage,
)
from .provenance import write_json


DEFAULT_FIXED_RECIPES = ("context_log", "legacy_safe_compact")
DEFAULT_GOWER_PARAMS: Mapping[str, Any] = {
    "n_neighbors": 15,
    "distance_power": 1.0,
}
DEFAULT_LOGISTIC_PARAMS: Mapping[str, Any] = {"C": 1.0}


@dataclass
class FixedRecipeStackingResult:
    split_manifest: pd.DataFrame
    inner_split_manifest: pd.DataFrame
    stacking_weights: pd.DataFrame
    inner_oof_component_probabilities: pd.DataFrame
    oof_probabilities: pd.DataFrame
    oof_aggregated: pd.DataFrame
    split_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    paired_panel_deltas: pd.DataFrame
    paired_bootstrap: pd.DataFrame
    paired_bootstrap_summary: pd.DataFrame
    paired_contrast_registry: pd.DataFrame
    qa: dict[str, Any]


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return int((int(base_seed) + digest) % (2**31 - 1))


def _stack_probabilities(
    gower: np.ndarray,
    logistic: np.ndarray,
    gower_weight: float,
) -> np.ndarray:
    weight = float(gower_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("gower_weight must be in [0, 1]")
    gower_array = normalize_probabilities(gower)
    logistic_array = normalize_probabilities(logistic)
    if gower_array.shape != logistic_array.shape:
        raise ValueError("Stacking component probability shapes differ")
    return normalize_probabilities(
        weight * gower_array + (1.0 - weight) * logistic_array
    )


def fit_convex_stack_weight(
    observed: np.ndarray,
    gower_probabilities: np.ndarray,
    logistic_probabilities: np.ndarray,
) -> dict[str, float]:
    """Fit the one-dimensional convex weight using soft-label log loss."""

    y = normalize_probabilities(observed)
    gower = normalize_probabilities(gower_probabilities)
    logistic = normalize_probabilities(logistic_probabilities)
    if y.shape != gower.shape or y.shape != logistic.shape:
        raise ValueError("Observed and component probability shapes differ")

    def objective(weight: float) -> float:
        return soft_cross_entropy(
            y, _stack_probabilities(gower, logistic, float(weight))
        )

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    candidates = [
        (0.0, objective(0.0)),
        (1.0, objective(1.0)),
    ]
    if result.success and np.isfinite(result.fun):
        bounded_weight = float(np.clip(result.x, 0.0, 1.0))
        if bounded_weight < 1.0e-7:
            bounded_weight = 0.0
        elif bounded_weight > 1.0 - 1.0e-7:
            bounded_weight = 1.0
        candidates.append((bounded_weight, objective(bounded_weight)))
    # Stable tie handling: prefer the less extreme weight, then the smaller
    # Gower weight.  This choice never consults outer-test targets.
    weight, loss = min(
        candidates,
        key=lambda item: (
            round(float(item[1]), 14),
            abs(float(item[0]) - 0.5),
            float(item[0]),
        ),
    )
    return {
        "gower_weight": float(weight),
        "logistic_weight": float(1.0 - weight),
        "inner_stacked_soft_log_loss": float(loss),
        "inner_gower_soft_log_loss": float(objective(1.0)),
        "inner_logistic_soft_log_loss": float(objective(0.0)),
    }


def _make_component(
    recipe: str,
    family: str,
    params: Mapping[str, Any],
    *,
    seed: int,
) -> EngineeredCardinalClassifier:
    return EngineeredCardinalClassifier(
        feature_recipe=recipe,
        base_model_family=family,
        base_model_params=dict(params),
        random_seed=int(seed),
    )


def _inner_oof_components(
    X: pd.DataFrame,
    y: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    recipe: str,
    base_seed: int,
    gower_params: Mapping[str, Any],
    logistic_params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_classes = y.shape
    gower_oof = np.full((n_rows, n_classes), np.nan, dtype=float)
    logistic_oof = np.full((n_rows, n_classes), np.nan, dtype=float)
    coverage = np.zeros(n_rows, dtype=int)
    for inner_fold, (train_positions, validation_positions) in enumerate(
        inner_splits
    ):
        gower = _make_component(
            recipe,
            "soft_gower_knn",
            gower_params,
            seed=_stable_seed(base_seed, recipe, "gower", inner_fold),
        )
        logistic = _make_component(
            recipe,
            "soft_multinomial_logistic",
            logistic_params,
            seed=_stable_seed(base_seed, recipe, "logistic", inner_fold),
        )
        gower.fit(X.iloc[train_positions], y[train_positions])
        logistic.fit(X.iloc[train_positions], y[train_positions])
        gower_oof[validation_positions] = gower.predict_proba(
            X.iloc[validation_positions]
        )
        logistic_oof[validation_positions] = logistic.predict_proba(
            X.iloc[validation_positions]
        )
        coverage[validation_positions] += 1
    if not np.array_equal(coverage, np.ones(n_rows, dtype=int)):
        raise AssertionError(
            "Inner OOF stacking requires exactly one prediction per "
            "outer-training record"
        )
    return (
        normalize_probabilities(gower_oof),
        normalize_probabilities(logistic_oof),
    )


def _metric_row(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    return {
        **two_stage_probability_metrics(observed, predicted),
        **collapsed_hard_metrics(observed, predicted),
    }


def _aggregate_metric_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "analysis_id",
        "scope",
        "panel_role",
        "panel",
        "model_role",
        "model_family",
    ]
    for values, group in oof.groupby(keys, dropna=False):
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_records": int(len(group)),
                "n_connected_groups": int(group["cv_group_id"].nunique()),
                "min_repeated_oof": int(group["n_repeated_oof"].min()),
                "max_repeated_oof": int(group["n_repeated_oof"].max()),
                **_metric_row(
                    group.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float),
                    group.loc[:, PREDICTION_COLUMNS].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def _calibration_tables(
    oof: pd.DataFrame, *, n_bins: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_rows: list[pd.DataFrame] = []
    confusion_rows: list[pd.DataFrame] = []
    keys = [
        "analysis_id",
        "scope",
        "panel_role",
        "panel",
        "model_role",
        "model_family",
    ]
    for values, group in oof.groupby(keys, dropna=False):
        metadata = dict(zip(keys, values, strict=True))
        observed = group.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
        predicted = group.loc[:, PREDICTION_COLUMNS].to_numpy(float)
        calibration = calibration_table(
            observed,
            predicted,
            class_labels=CLASS_LABELS,
            n_bins=int(n_bins),
        )
        confusion = expected_confusion_table(
            observed,
            predicted,
            class_labels=CLASS_LABELS,
        )
        for key, value in reversed(tuple(metadata.items())):
            calibration.insert(0, key, value)
            confusion.insert(0, key, value)
        calibration_rows.append(calibration)
        confusion_rows.append(confusion)
    return (
        pd.concat(calibration_rows, ignore_index=True),
        pd.concat(confusion_rows, ignore_index=True),
    )


def run_fixed_recipe_stacking_evaluation(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    analysis_id: str,
    scope: str,
    panel_role: str,
    config: CardinalEvaluationConfig,
    recipes: Sequence[str] = DEFAULT_FIXED_RECIPES,
    reference_recipe: str = "context_log",
    augmented_recipe: str = "legacy_safe_compact",
    gower_params: Mapping[str, Any] = DEFAULT_GOWER_PARAMS,
    logistic_params: Mapping[str, Any] = DEFAULT_LOGISTIC_PARAMS,
    smoke: bool = False,
    record_id_col: str = "record_id",
    group_col: str = "cv_group_id",
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> FixedRecipeStackingResult:
    """Evaluate two fixed recipes with train-fold-only convex stacking."""

    recipe_names = tuple(map(str, recipes))
    if recipe_names != (str(reference_recipe), str(augmented_recipe)):
        raise ValueError(
            "Fixed stacking requires recipes ordered as "
            "(reference_recipe, augmented_recipe)"
        )
    if len(set(recipe_names)) != 2:
        raise ValueError("Fixed stacking requires two distinct recipes")
    if set(config.model_names) != set(recipe_names):
        raise ValueError(
            "Fixed stacking config.model_names must contain only the two "
            "predeclared recipe names"
        )
    if tuple(config.panel_names) != tuple(recipe_names):
        raise ValueError(
            "Fixed stacking config.panel_names must equal the recipe order"
        )

    settings = config
    if smoke:
        settings = CardinalEvaluationConfig(
            **{
                **asdict(config),
                "outer_folds": 2,
                "outer_repeats": 1,
                "inner_folds": 2,
                "outer_seeds": (config.random_seed,),
                "bootstrap_replicates": min(config.bootstrap_replicates, 50),
                "n_jobs": 1,
            }
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = frame.reset_index(drop=True).copy()
    required = {
        record_id_col,
        group_col,
        *SOFT_LABEL_COLUMNS,
        *required_cardinal_feature_columns(recipe_names),
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Fixed stacking frame is missing columns: {missing}")
    if data[record_id_col].astype(str).duplicated().any():
        raise ValueError("Fixed stacking requires one row per record")
    labels = data.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
    if not np.allclose(labels.sum(axis=1), 1.0):
        raise ValueError("Soft-label rows must sum to one")

    manifest = make_soft_stratified_group_manifest(
        data,
        SOFT_LABEL_COLUMNS,
        group_col=group_col,
        record_id_col=record_id_col,
        folds=settings.outer_folds,
        repeats=settings.outer_repeats,
        random_seed=settings.random_seed,
        repeat_seeds=settings.outer_seeds,
        relation_columns=relation_columns,
    )
    id_to_position = {
        str(record_id): position
        for position, record_id in enumerate(data[record_id_col])
    }
    inner_manifest_rows: list[dict[str, object]] = []
    inner_prediction_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    outer_selection_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for repeat in sorted(manifest["repeat"].unique()):
        repeat_manifest = manifest.loc[manifest["repeat"].eq(repeat)]
        for fold in sorted(repeat_manifest["fold"].unique()):
            test_ids = set(
                repeat_manifest.loc[
                    repeat_manifest["fold"].eq(fold), "record_id"
                ].astype(str)
            )
            test_positions = np.asarray(
                sorted(id_to_position[value] for value in test_ids), dtype=int
            )
            test_set = set(test_positions.tolist())
            train_positions = np.asarray(
                [
                    position
                    for position in range(len(data))
                    if position not in test_set
                ],
                dtype=int,
            )
            train = data.iloc[train_positions].reset_index(drop=True)
            test = data.iloc[test_positions].reset_index(drop=True)
            assert_no_relational_leakage(
                train,
                test,
                group_col=group_col,
                relation_columns=relation_columns,
            )
            inner_seed = (
                settings.random_seed + int(repeat) * 10_000 + int(fold) * 100
            )
            inner_splits, inner_assignments = make_inner_soft_group_splits(
                train,
                SOFT_LABEL_COLUMNS,
                group_col=group_col,
                folds=settings.inner_folds,
                random_seed=inner_seed,
                relation_columns=relation_columns,
            )
            for position, inner_fold in enumerate(inner_assignments):
                inner_manifest_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "scope": scope,
                        "panel_role": panel_role,
                        "outer_repeat": int(repeat),
                        "outer_fold": int(fold),
                        "inner_fold": int(inner_fold),
                        "record_id": str(train.iloc[position][record_id_col]),
                        "cv_group_id": str(train.iloc[position][group_col]),
                        "inner_seed": int(inner_seed),
                    }
                )

            y_train = train.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
            y_test = test.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
            for recipe in recipe_names:
                columns = list(required_cardinal_feature_columns(recipe))
                X_train = train.loc[:, columns]
                X_test = test.loc[:, columns]
                gower_inner, logistic_inner = _inner_oof_components(
                    X_train,
                    y_train,
                    inner_splits,
                    recipe=recipe,
                    base_seed=inner_seed,
                    gower_params=gower_params,
                    logistic_params=logistic_params,
                )
                weight = fit_convex_stack_weight(
                    y_train, gower_inner, logistic_inner
                )
                stacked_inner = _stack_probabilities(
                    gower_inner,
                    logistic_inner,
                    weight["gower_weight"],
                )
                for position, row in train.iterrows():
                    inner_row: dict[str, object] = {
                        "analysis_id": analysis_id,
                        "scope": scope,
                        "panel_role": panel_role,
                        "recipe": recipe,
                        "outer_repeat": int(repeat),
                        "outer_fold": int(fold),
                        "inner_fold": int(inner_assignments[position]),
                        "record_id": str(row[record_id_col]),
                        "cv_group_id": str(row[group_col]),
                    }
                    for column, value in zip(
                        SOFT_LABEL_COLUMNS, y_train[position], strict=True
                    ):
                        inner_row[column] = float(value)
                    for prefix, probabilities in (
                        ("gower", gower_inner[position]),
                        ("logistic", logistic_inner[position]),
                        ("stacked", stacked_inner[position]),
                    ):
                        for label, value in zip(
                            CLASS_LABELS, probabilities, strict=True
                        ):
                            inner_row[f"p_{prefix}_{label}"] = float(value)
                    inner_prediction_rows.append(inner_row)

                gower = _make_component(
                    recipe,
                    "soft_gower_knn",
                    gower_params,
                    seed=_stable_seed(inner_seed, recipe, "gower", "outer"),
                )
                logistic = _make_component(
                    recipe,
                    "soft_multinomial_logistic",
                    logistic_params,
                    seed=_stable_seed(
                        inner_seed, recipe, "logistic", "outer"
                    ),
                )
                gower.fit(X_train, y_train)
                logistic.fit(X_train, y_train)
                predicted = _stack_probabilities(
                    gower.predict_proba(X_test),
                    logistic.predict_proba(X_test),
                    weight["gower_weight"],
                )
                weight_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "scope": scope,
                        "panel_role": panel_role,
                        "recipe": recipe,
                        "outer_repeat": int(repeat),
                        "outer_fold": int(fold),
                        "n_outer_train": int(len(train)),
                        "n_inner_folds": int(settings.inner_folds),
                        **weight,
                        "weight_fit_source": (
                            "outer_training_inner_connected_group_oof_only"
                        ),
                        "gower_params": json.dumps(
                            dict(gower_params), sort_keys=True
                        ),
                        "logistic_params": json.dumps(
                            dict(logistic_params), sort_keys=True
                        ),
                    }
                )
                outer_selection_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "scope": scope,
                        "panel_role": panel_role,
                        "panel": recipe,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "selection_scope": "fixed_recipe_fixed_components",
                        "model_family": "convex_probability_stack",
                        "selected_model_family": (
                            "soft_gower_knn_plus_soft_multinomial_logistic"
                        ),
                        "selected_candidate_id": (
                            f"{recipe}__fixed_gower_logistic_stack"
                        ),
                        "selected_params": json.dumps(
                            {
                                "gower": dict(gower_params),
                                "logistic": dict(logistic_params),
                                "gower_weight": weight["gower_weight"],
                            },
                            sort_keys=True,
                        ),
                        "n_engineered_features": int(
                            len(gower.engineered_feature_names_)
                        ),
                        "engineered_feature_names": json.dumps(
                            list(gower.engineered_feature_names_)
                        ),
                        "engineered_blocks": json.dumps(
                            {
                                key: list(values)
                                for key, values in gower.engineered_blocks_.items()
                            },
                            sort_keys=True,
                        ),
                        "inner_soft_log_loss_mean": float(
                            weight["inner_stacked_soft_log_loss"]
                        ),
                        "inner_soft_log_loss_se": np.nan,
                        "selection_rule": (
                            "fixed_recipe_inner_oof_convex_stacking"
                        ),
                    }
                )
                for local_index, row in test.iterrows():
                    oof_row: dict[str, object] = {
                        "analysis_id": analysis_id,
                        "scope": scope,
                        "panel_role": panel_role,
                        "panel": recipe,
                        "model_role": "fixed_stacked",
                        "model_family": "soft_gower_logistic_stack",
                        "selected_model_family": (
                            "soft_gower_knn_plus_soft_multinomial_logistic"
                        ),
                        "selected_candidate_id": (
                            f"{recipe}__fixed_gower_logistic_stack"
                        ),
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "record_id": str(row[record_id_col]),
                        "cv_group_id": str(row[group_col]),
                    }
                    for column, value in zip(
                        SOFT_LABEL_COLUMNS, y_test[local_index], strict=True
                    ):
                        oof_row[column] = float(value)
                    for column, value in zip(
                        PREDICTION_COLUMNS,
                        predicted[local_index],
                        strict=True,
                    ):
                        oof_row[column] = float(value)
                    oof_rows.append(oof_row)
                split_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "scope": scope,
                        "panel_role": panel_role,
                        "panel": recipe,
                        "model_role": "fixed_stacked",
                        "model_family": "soft_gower_logistic_stack",
                        "selected_model_family": (
                            "soft_gower_knn_plus_soft_multinomial_logistic"
                        ),
                        "selected_candidate_id": (
                            f"{recipe}__fixed_gower_logistic_stack"
                        ),
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "n_train_groups": int(train[group_col].nunique()),
                        "n_test_groups": int(test[group_col].nunique()),
                        **_metric_row(y_test, predicted),
                    }
                )

    inner_manifest = pd.DataFrame(inner_manifest_rows)
    inner_predictions = pd.DataFrame(inner_prediction_rows)
    weights = pd.DataFrame(weight_rows)
    selections = pd.DataFrame(outer_selection_rows)
    oof = pd.DataFrame(oof_rows)
    split_metrics = pd.DataFrame(split_rows)
    aggregated = aggregate_repeated_oof(oof)
    aggregate_metrics = _aggregate_metric_table(aggregated)
    calibration, confusion = _calibration_tables(
        aggregated, n_bins=settings.calibration_bins
    )
    # The generic paired-contrast helper predates fixed pipelines and filters
    # on its historical selected-model labels.  Remap only disposable copies;
    # the persisted OOF and metric tables retain explicit fixed-stack labels.
    contrast_oof = aggregated.copy()
    contrast_splits = split_metrics.copy()
    for contrast_frame in (contrast_oof, contrast_splits):
        contrast_frame["model_role"] = "joint_selected"
        contrast_frame["model_family"] = "nested_selected"
    deltas, bootstrap, summary, registry = evaluate_paired_panel_contrasts(
        contrast_oof,
        contrast_splits,
        [
            {
                "id": f"{augmented_recipe}_minus_{reference_recipe}",
                "role": "fixed_recipe_convex_stacking_comparison",
                "reference": reference_recipe,
                "augmented": augmented_recipe,
            }
        ],
        analysis_id=analysis_id,
        scope=scope,
        panel_role=panel_role,
        replicates=settings.bootstrap_replicates,
        random_seed=settings.random_seed,
    )
    expected_oof = settings.outer_repeats
    coverage_ok = bool(
        not aggregated.empty
        and aggregated["n_repeated_oof"].eq(expected_oof).all()
    )
    inner_weight_count_ok = bool(
        len(weights)
        == settings.outer_repeats
        * settings.outer_folds
        * len(recipe_names)
    )
    qa = {
        "analysis_id": analysis_id,
        "scope": scope,
        "panel_role": panel_role,
        "status": "complete_smoke" if smoke else "complete",
        "n_records": int(len(data)),
        "n_connected_groups": int(data[group_col].nunique()),
        "outer_folds": int(settings.outer_folds),
        "outer_repeats": int(settings.outer_repeats),
        "inner_folds": int(settings.inner_folds),
        "fixed_recipes": list(recipe_names),
        "joint_recipe_selector_used": False,
        "optuna_used": False,
        "component_models": [
            "soft_gower_knn",
            "soft_multinomial_logistic",
        ],
        "stacking_weight_fit_source": (
            "outer_training_inner_connected_group_oof_only"
        ),
        "outer_test_targets_used_for_stacking_weight": False,
        "all_gower_weights_in_unit_interval": bool(
            weights["gower_weight"].between(0.0, 1.0).all()
        ),
        "one_weight_per_recipe_outer_fold": inner_weight_count_ok,
        "repeated_oof_count_matches": coverage_ok,
        "mechanistic_claim_allowed": False,
    }
    if not coverage_ok or not inner_weight_count_ok:
        raise AssertionError("Fixed stacking audit checks failed")

    pipeline_registry = pd.DataFrame(
        [
            {
                "recipe": recipe,
                "role": (
                    "reference"
                    if recipe == reference_recipe
                    else "augmented"
                ),
                "required_source_columns": "|".join(
                    required_cardinal_feature_columns(recipe)
                ),
                "base_components": (
                    "soft_gower_knn|soft_multinomial_logistic"
                ),
                "combination": (
                    "convex_weight_from_outer_training_inner_group_oof"
                ),
                "recipe_or_model_search": False,
            }
            for recipe in recipe_names
        ]
    )
    artifacts = {
        "split_manifest.csv": manifest,
        "inner_split_manifest.csv": inner_manifest,
        "fixed_pipeline_registry.csv": pipeline_registry,
        "stacking_weights.csv": weights,
        "inner_oof_component_probabilities.csv": inner_predictions,
        "outer_selections.csv": selections,
        "oof_probabilities.csv": oof,
        "oof_aggregated.csv": aggregated,
        "split_metrics.csv": split_metrics,
        "aggregate_metrics.csv": aggregate_metrics,
        "calibration_by_class.csv": calibration,
        "expected_confusion.csv": confusion,
        "paired_panel_deltas.csv": deltas,
        "paired_group_bootstrap.csv": bootstrap,
        "paired_group_bootstrap_summary.csv": summary,
        "paired_contrast_registry.csv": registry,
    }
    for filename, artifact in artifacts.items():
        artifact.to_csv(output / filename, index=False)
    write_json(output / "analysis_qa.json", qa)
    return FixedRecipeStackingResult(
        split_manifest=manifest,
        inner_split_manifest=inner_manifest,
        stacking_weights=weights,
        inner_oof_component_probabilities=inner_predictions,
        oof_probabilities=oof,
        oof_aggregated=aggregated,
        split_metrics=split_metrics,
        aggregate_metrics=aggregate_metrics,
        paired_panel_deltas=deltas,
        paired_bootstrap=bootstrap,
        paired_bootstrap_summary=summary,
        paired_contrast_registry=registry,
        qa=qa,
    )


__all__ = [
    "DEFAULT_FIXED_RECIPES",
    "FixedRecipeStackingResult",
    "fit_convex_stack_weight",
    "run_fixed_recipe_stacking_evaluation",
]
