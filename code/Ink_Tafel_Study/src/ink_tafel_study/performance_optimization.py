"""Independent joint feature/model optimization for Bayesian Tafel slopes.

The selector evaluates complete candidate pipelines inside grouped inner CV,
then predicts an untouched connected-group outer fold.  Existing registered
``current_97`` results are neither read nor modified; inputs are bound directly
to the frozen ``Data/02_inputs/canonical_dataset_ir_corrected_v2`` package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, ParameterGrid

from .evaluate import (
    DEFAULT_RELATION_COLUMNS,
    assert_no_relational_leakage,
    build_connected_group_ids,
    make_outer_split_manifest,
    regression_metrics,
)
from .features import is_target_like_column, validate_no_target_leakage
from .optimization_components import (
    FEATURE_RECIPES,
    PROPERTY_DIELECTRIC_BEST,
    PROPERTY_DIELECTRIC_STRICT,
    FeatureRecipe,
    OptimizationModelSpec,
    available_model_registry,
    feature_recipe_registry,
    get_feature_recipe,
    make_candidate_regressor_specs,
    make_optimization_pipeline,
    normalize_optimization_columns,
    property_columns,
    required_source_columns,
)
from .provenance import directory_tree_summary, sha256_file, write_json
from .uncertainty import score_functional_target_draws


TARGET_NAME = "b_bayesian_local_line_mV_dec"
TARGET_COLUMN = "target_value_mV_dec"
TARGET_INTERVAL_LOW = "target_interval_low_mV_dec"
TARGET_INTERVAL_HIGH = "target_interval_high_mV_dec"


@dataclass(frozen=True)
class OptimizationConfig:
    outer_folds: int = 5
    outer_repeats: int = 10
    inner_folds: int = 4
    outer_seeds: tuple[int, ...] = (
        42,
        142,
        242,
        342,
        442,
        542,
        642,
        742,
        842,
        942,
    )
    random_seed: int = 42
    candidate_budget: int = 96
    tie_tolerance_mV_dec: float = 0.1
    bootstrap_replicates: int = 2000
    target_draw_repetitions: int = 100
    include_catboost: bool = True
    n_jobs: int = 1
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("Nested grouped CV requires at least two inner and outer folds")
        if self.outer_repeats < 1:
            raise ValueError("outer_repeats must be positive")
        if len(self.outer_seeds) != self.outer_repeats:
            raise ValueError("outer_seeds must contain one seed per repeat")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if self.tie_tolerance_mV_dec < 0:
            raise ValueError("tie_tolerance_mV_dec cannot be negative")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates cannot be negative")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be positive")


@dataclass(frozen=True)
class AnalysisSpec:
    analysis_id: str
    scope: str
    recipe_policy: str
    strict_complete_cases: bool = False


@dataclass(frozen=True)
class CandidateDefinition:
    candidate_id: str
    recipe: str | None
    numeric_transform: str
    target_transform: str
    model: str
    model_params: Mapping[str, Any]
    complexity_rank: int
    feature_complexity: int

    def row(self) -> dict[str, object]:
        recipe = get_feature_recipe(self.recipe) if self.recipe else None
        return {
            "candidate_id": self.candidate_id,
            "recipe": self.recipe or "dummy",
            "base": recipe.base if recipe else "none",
            "include_ink_amounts": recipe.include_ink_amounts if recipe else False,
            "solvent_representation": (
                recipe.solvent_representation if recipe else "none"
            ),
            "include_property_amount_interactions": (
                recipe.include_property_amount_interactions if recipe else False
            ),
            "include_dielectric_estimated_flag": (
                recipe.include_dielectric_estimated_flag if recipe else False
            ),
            "include_process_features": (
                recipe.include_process_features if recipe else False
            ),
            "strict_source_backed_dielectric": (
                recipe.strict_source_backed_dielectric if recipe else False
            ),
            "sensitivity_only": recipe.sensitivity_only if recipe else False,
            "complete_case_required": (
                recipe.complete_case_required if recipe else False
            ),
            "numeric_transform": self.numeric_transform,
            "target_transform": self.target_transform,
            "model": self.model,
            "model_params_json": json.dumps(
                dict(self.model_params), sort_keys=True, separators=(",", ":")
            ),
            "complexity_rank": self.complexity_rank,
            "feature_complexity": self.feature_complexity,
        }


@dataclass
class AnalysisResult:
    analysis_id: str
    split_manifest: pd.DataFrame
    inner_split_manifest: pd.DataFrame
    candidate_scores: pd.DataFrame
    inner_fold_scores: pd.DataFrame
    outer_selections: pd.DataFrame
    oof_predictions: pd.DataFrame
    oof_aggregated: pd.DataFrame
    metrics: pd.DataFrame
    bootstrap_metrics: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    weighted_sensitivity: pd.DataFrame
    selection_frequency: pd.DataFrame


ANALYSES = (
    AnalysisSpec("primary_alkaline", "alkaline", "primary"),
    # Same selector library as primary: isolates the electrolyte-scope change.
    AnalysisSpec("sensitivity_all", "all", "primary"),
    # Separately named because it changes the selector library as well as using
    # the all-electrolyte population.
    AnalysisSpec(
        "sensitivity_feature_recipe_search_all", "all", "all_non_strict"
    ),
    AnalysisSpec(
        "sensitivity_strict_source_backed_complete_cases",
        "all",
        "strict_complete_case",
        strict_complete_cases=True,
    ),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def optimization_config_from_settings(
    settings: Mapping[str, Any] | None,
    *,
    n_jobs: int,
) -> OptimizationConfig:
    """Resolve the separate optimization registry without altering formal CV."""

    if not settings:
        return OptimizationConfig(n_jobs=int(n_jobs))
    defaults = OptimizationConfig()
    seeds = tuple(int(value) for value in settings.get("outer_seeds", defaults.outer_seeds))
    repeats = int(settings.get("outer_repeats", len(seeds)))
    return OptimizationConfig(
        outer_folds=int(settings.get("outer_folds", defaults.outer_folds)),
        outer_repeats=repeats,
        inner_folds=int(settings.get("inner_folds", defaults.inner_folds)),
        outer_seeds=seeds,
        random_seed=int(settings.get("random_seed", defaults.random_seed)),
        candidate_budget=int(
            settings.get("candidate_budget", defaults.candidate_budget)
        ),
        tie_tolerance_mV_dec=float(
            settings.get(
                "near_tie_tolerance_mV_dec", defaults.tie_tolerance_mV_dec
            )
        ),
        bootstrap_replicates=int(
            settings.get("group_bootstrap_repetitions", defaults.bootstrap_replicates)
        ),
        target_draw_repetitions=int(
            settings.get("target_draw_repetitions", defaults.target_draw_repetitions)
        ),
        include_catboost=bool(
            settings.get(
                "require_catboost_candidate_for_full",
                settings.get("include_optional_catboost", defaults.include_catboost),
            )
        ),
        n_jobs=int(n_jobs),
    )


def _candidate_key(
    recipe: str | None,
    numeric_transform: str,
    target_transform: str,
    model: str,
    model_params: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        {
            "recipe": recipe,
            "numeric_transform": numeric_transform,
            "target_transform": target_transform,
            "model": model,
            "model_params": _jsonable(dict(model_params)),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "pipe_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parameter_options(spec: OptimizationModelSpec) -> list[dict[str, Any]]:
    return [dict(values) for values in ParameterGrid(dict(spec.param_grid))] or [{}]


def _feature_complexity(recipe_name: str | None) -> int:
    if recipe_name is None:
        return 0
    recipe = get_feature_recipe(recipe_name)
    complexity = len(required_source_columns(recipe))
    if recipe.solvent_representation == "ilr":
        complexity -= 1  # five fractions become four ILR coordinates
    elif recipe.solvent_representation in {
        "properties_pca2",
        "fractions_plus_properties_pca2",
    }:
        complexity -= 2  # four properties become two PCs
    elif recipe.solvent_representation == "properties_lean":
        complexity -= 2  # four properties become two lean coordinates
    if recipe.include_property_amount_interactions:
        complexity += 4
    return complexity


def recipes_for_policy(policy: str) -> tuple[str, ...]:
    if policy == "primary":
        selected = [
            name
            for name, recipe in FEATURE_RECIPES.items()
            if not recipe.sensitivity_only and not recipe.complete_case_required
        ]
    elif policy == "all_non_strict":
        selected = [
            name
            for name, recipe in FEATURE_RECIPES.items()
            if not recipe.complete_case_required
        ]
    elif policy == "strict_complete_case":
        # This sensitivity changes the eligible cohort, not the selector
        # library.  On retained records the best-available dielectric value
        # is exactly the source-backed value, so adding parallel "strict"
        # recipes would duplicate candidates and give that representation
        # extra opportunities to win.
        selected = [
            name
            for name, recipe in FEATURE_RECIPES.items()
            if not recipe.sensitivity_only and not recipe.complete_case_required
        ]
    else:
        raise KeyError(f"Unknown optimization recipe policy: {policy!r}")
    return tuple(sorted(selected))


def build_candidate_catalog(
    recipe_names: Sequence[str],
    specs: Mapping[str, OptimizationModelSpec],
    *,
    budget: int,
    random_seed: int,
) -> tuple[CandidateDefinition, ...]:
    """Build one data-independent finite catalog reused in every outer fold."""

    if not recipe_names:
        raise ValueError("At least one feature recipe is required")
    rng = np.random.default_rng(random_seed)
    definitions: dict[str, CandidateDefinition] = {}

    def add(
        recipe: str | None,
        numeric_transform: str,
        target_transform: str,
        model: str,
        params: Mapping[str, Any],
    ) -> None:
        key = _candidate_key(
            recipe, numeric_transform, target_transform, model, params
        )
        definitions.setdefault(
            key,
            CandidateDefinition(
                candidate_id=key,
                recipe=recipe,
                numeric_transform=numeric_transform,
                target_transform=target_transform,
                model=model,
                model_params=_jsonable(dict(params)),
                complexity_rank=int(specs[model].complexity_rank),
                feature_complexity=_feature_complexity(recipe),
            ),
        )

    # All four target-coordinate/summary dummy baselines are explicit.
    for model in ("dummy_mean", "dummy_median"):
        for target_transform in ("identity", "log10"):
            add(None, "raw", target_transform, model, {})

    model_names = [name for name in specs if not name.startswith("dummy_")]
    # Every feature recipe is represented in both registered numeric
    # coordinates.  The model/target coordinate rotates independently, so
    # raw-vs-log is never silently omitted for a recipe.
    for index, recipe in enumerate(recipe_names):
        for numeric_index, numeric_transform in enumerate(("raw", "log")):
            model = model_names[(2 * index + numeric_index) % len(model_names)]
            options = _parameter_options(specs[model])
            add(
                recipe,
                numeric_transform,
                "identity" if (index + numeric_index) % 2 == 0 else "log10",
                model,
                options[(2 * index + numeric_index) % len(options)],
            )

    # Every model family is explicitly compared in both target coordinates.
    # These coverage anchors rotate across recipes; no privileged feature
    # representation receives the full model menu.
    for index, model in enumerate(model_names):
        options = _parameter_options(specs[model])
        midpoint = options[len(options) // 2]
        for target_index, target_transform in enumerate(("identity", "log10")):
            distributed_recipe = recipe_names[
                (2 * index + target_index) % len(recipe_names)
            ]
            add(
                distributed_recipe,
                "raw" if (index + target_index) % 2 == 0 else "log",
                target_transform,
                model,
                midpoint,
            )

    if len(definitions) > budget:
        raise ValueError(
            "candidate_budget is smaller than the balanced mandatory coverage "
            f"catalog ({budget} requested, {len(definitions)} required)"
        )

    # Fill the remaining frozen budget with reproducible conditional draws.
    attempts = 0
    maximum_attempts = max(10_000, budget * 200)
    while len(definitions) < budget and attempts < maximum_attempts:
        attempts += 1
        current = list(definitions.values())
        recipe_counts = {
            recipe: sum(item.recipe == recipe for item in current)
            for recipe in recipe_names
        }
        least_recipe_count = min(recipe_counts.values())
        recipe = str(
            rng.choice(
                [
                    name
                    for name, count in recipe_counts.items()
                    if count == least_recipe_count
                ]
            )
        )
        model_counts = {
            model: sum(item.model == model for item in current)
            for model in model_names
        }
        least_model_count = min(model_counts.values())
        model = str(
            rng.choice(
                [
                    name
                    for name, count in model_counts.items()
                    if count == least_model_count
                ]
            )
        )
        options = _parameter_options(specs[model])
        params = options[int(rng.integers(0, len(options)))]
        recipe_items = [item for item in current if item.recipe == recipe]
        numeric_counts = {
            transform: sum(
                item.numeric_transform == transform for item in recipe_items
            )
            for transform in ("raw", "log")
        }
        target_counts = {
            transform: sum(
                item.target_transform == transform for item in recipe_items
            )
            for transform in ("identity", "log10")
        }
        numeric_minimum = min(numeric_counts.values())
        target_minimum = min(target_counts.values())
        add(
            recipe,
            str(
                rng.choice(
                    [
                        name
                        for name, count in numeric_counts.items()
                        if count == numeric_minimum
                    ]
                )
            ),
            str(
                rng.choice(
                    [
                        name
                        for name, count in target_counts.items()
                        if count == target_minimum
                    ]
                )
            ),
            model,
            params,
        )
    if len(definitions) < budget:
        raise RuntimeError(
            f"Could not create {budget} unique pipeline candidates after {attempts} attempts"
        )
    return tuple(sorted(definitions.values(), key=lambda item: item.candidate_id))


def _candidate_pipeline(
    candidate: CandidateDefinition,
    specs: Mapping[str, OptimizationModelSpec],
) -> BaseEstimator:
    return make_optimization_pipeline(
        candidate.recipe,
        specs[candidate.model],
        target_transform=candidate.target_transform,
        numeric_transform=candidate.numeric_transform,
        model_params=candidate.model_params,
    )


def _make_inner_splits(
    frame: pd.DataFrame,
    *,
    group_col: str,
    inner_folds: int,
    seed: int,
    relation_columns: Sequence[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    groups = frame[group_col].astype(str).to_numpy()
    folds = min(int(inner_folds), len(np.unique(groups)))
    if folds < 2:
        raise ValueError("Outer training data contain fewer than two connected groups")
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    assignment = np.full(len(frame), -1, dtype=int)
    for fold, (train_positions, validation_positions) in enumerate(
        splitter.split(frame, groups=groups)
    ):
        assert_no_relational_leakage(
            frame.iloc[train_positions],
            frame.iloc[validation_positions],
            group_col=group_col,
            relation_columns=relation_columns,
        )
        assignment[validation_positions] = fold
        splits.append((train_positions, validation_positions))
    if np.any(assignment < 0):
        raise AssertionError("Inner grouped CV did not assign every outer-training record")
    return splits, assignment


def _evaluate_candidate_inner(
    candidate: CandidateDefinition,
    specs: Mapping[str, OptimizationModelSpec],
    X: pd.DataFrame,
    y: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    predictions = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    active_inner_fold = -1
    try:
        for inner_fold, (train_positions, validation_positions) in enumerate(inner_splits):
            active_inner_fold = int(inner_fold)
            estimator = _candidate_pipeline(candidate, specs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                estimator.fit(X.iloc[train_positions], y[train_positions])
                predicted = np.asarray(
                    estimator.predict(X.iloc[validation_positions]), dtype=float
                ).reshape(-1)
            if len(predicted) != len(validation_positions) or not np.isfinite(predicted).all():
                raise ValueError("Candidate returned non-finite or misaligned inner predictions")
            predictions[validation_positions] = predicted
            fold_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "inner_fold": int(inner_fold),
                    "n_train": int(len(train_positions)),
                    "n_validation": int(len(validation_positions)),
                    "mae_mV_dec": float(
                        mean_absolute_error(y[validation_positions], predicted)
                    ),
                    "status": "complete",
                }
            )
        if not np.isfinite(predictions).all():
            raise AssertionError("Candidate does not cover every inner validation row")
        metrics = regression_metrics(y, predictions)
        row = {
            "candidate_id": candidate.candidate_id,
            "inner_mae_mV_dec": metrics["mae"],
            "inner_rmse_mV_dec": metrics["rmse"],
            "inner_median_ae_mV_dec": metrics["median_ae"],
            "inner_spearman": metrics["spearman"],
            "inner_r2": metrics["r2"],
            "status": "complete",
            "error": "",
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    except Exception as error:
        row = {
            "candidate_id": candidate.candidate_id,
            "inner_mae_mV_dec": np.nan,
            "inner_rmse_mV_dec": np.nan,
            "inner_median_ae_mV_dec": np.nan,
            "inner_spearman": np.nan,
            "inner_r2": np.nan,
            "status": "failed",
            "error": repr(error)[:1000],
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        fold_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "inner_fold": active_inner_fold,
                "n_train": (
                    int(len(inner_splits[active_inner_fold][0]))
                    if active_inner_fold >= 0
                    else 0
                ),
                "n_validation": (
                    int(len(inner_splits[active_inner_fold][1]))
                    if active_inner_fold >= 0
                    else 0
                ),
                "mae_mV_dec": np.nan,
                "status": "failed",
                "error": repr(error)[:1000],
            }
        )
    return row, fold_rows


def evaluate_candidate_catalog(
    candidates: Sequence[CandidateDefinition],
    specs: Mapping[str, OptimizationModelSpec],
    X: pd.DataFrame,
    y: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every complete candidate on one shared inner-split object."""

    if n_jobs == 1:
        results = [
            _evaluate_candidate_inner(candidate, specs, X, y, inner_splits)
            for candidate in candidates
        ]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_evaluate_candidate_inner)(candidate, specs, X, y, inner_splits)
            for candidate in candidates
        )
    scores = pd.DataFrame([result[0] for result in results])
    folds = pd.DataFrame([row for result in results for row in result[1]])
    catalog = pd.DataFrame([candidate.row() for candidate in candidates])
    scores = scores.merge(catalog, on="candidate_id", validate="one_to_one")
    return scores, folds


def choose_inner_winner(
    scores: pd.DataFrame,
    *,
    tie_tolerance_mV_dec: float,
    candidate_lookup: Mapping[str, CandidateDefinition],
    candidates: Iterable[str] | None = None,
) -> tuple[CandidateDefinition, dict[str, object]]:
    """Choose minimum inner MAE; use simplicity only inside a near-tie."""

    eligible = scores.loc[scores["status"].eq("complete")].copy()
    if candidates is not None:
        allowed = set(map(str, candidates))
        eligible = eligible.loc[eligible["candidate_id"].astype(str).isin(allowed)]
    eligible = eligible.loc[pd.to_numeric(eligible["inner_mae_mV_dec"], errors="coerce").notna()]
    if eligible.empty:
        raise RuntimeError("No candidate pipeline completed inner grouped CV")
    best_value = float(eligible["inner_mae_mV_dec"].min())
    tied = eligible.loc[
        eligible["inner_mae_mV_dec"] <= best_value + float(tie_tolerance_mV_dec)
    ].copy()
    tied = tied.sort_values(
        ["complexity_rank", "feature_complexity", "inner_mae_mV_dec", "candidate_id"]
    )
    selected_id = str(tied.iloc[0]["candidate_id"])
    raw_best_id = str(
        eligible.sort_values(["inner_mae_mV_dec", "candidate_id"]).iloc[0][
            "candidate_id"
        ]
    )
    return candidate_lookup[selected_id], {
        "selected_candidate_id": selected_id,
        "raw_minimum_candidate_id": raw_best_id,
        "raw_minimum_inner_mae_mV_dec": best_value,
        "selected_inner_mae_mV_dec": float(tied.iloc[0]["inner_mae_mV_dec"]),
        "tie_tolerance_mV_dec": float(tie_tolerance_mV_dec),
        "n_near_tied_candidates": int(len(tied)),
        "simplicity_tie_break_applied": bool(selected_id != raw_best_id),
    }


def select_outer_training_winner(
    outer_train: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: Sequence[CandidateDefinition],
    specs: Mapping[str, OptimizationModelSpec],
    *,
    inner_folds: int,
    inner_seed: int,
    tie_tolerance_mV_dec: float,
    n_jobs: int,
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> tuple[
    CandidateDefinition,
    CandidateDefinition,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    dict[str, object],
    dict[str, object],
]:
    """Select using only outer-training rows; outer-test targets are absent."""

    inner_splits, assignments = _make_inner_splits(
        outer_train,
        group_col="cv_group_id",
        inner_folds=inner_folds,
        seed=inner_seed,
        relation_columns=relation_columns,
    )
    scores, folds = evaluate_candidate_catalog(
        candidates, specs, X_train, y_train, inner_splits, n_jobs=n_jobs
    )
    lookup = {candidate.candidate_id: candidate for candidate in candidates}
    winner, winner_details = choose_inner_winner(
        scores,
        tie_tolerance_mV_dec=tie_tolerance_mV_dec,
        candidate_lookup=lookup,
    )
    dummy_ids = [
        candidate.candidate_id
        for candidate in candidates
        if candidate.model in {"dummy_mean", "dummy_median"}
    ]
    dummy, dummy_details = choose_inner_winner(
        scores,
        tie_tolerance_mV_dec=tie_tolerance_mV_dec,
        candidate_lookup=lookup,
        candidates=dummy_ids,
    )
    return (
        winner,
        dummy,
        scores,
        folds,
        assignments,
        winner_details,
        dummy_details,
    )


def _safe_predict_fitted(
    candidate: CandidateDefinition,
    specs: Mapping[str, OptimizationModelSpec],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
) -> tuple[BaseEstimator, np.ndarray]:
    estimator = _candidate_pipeline(candidate, specs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        estimator.fit(X_train, y_train)
        prediction = np.asarray(estimator.predict(X_test), dtype=float).reshape(-1)
    if len(prediction) != len(X_test) or not np.isfinite(prediction).all():
        raise ValueError("Selected pipeline returned invalid outer-test predictions")
    return estimator, prediction


def _aggregate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    keys = ["analysis_id", "prediction_role", "record_id", "cv_group_id"]
    aggregated = (
        oof.groupby(keys, as_index=False, dropna=False)
        .agg(
            observed_mV_dec=("observed_mV_dec", "first"),
            predicted_mV_dec=("predicted_mV_dec", "mean"),
            target_interval_low_mV_dec=("target_interval_low_mV_dec", "first"),
            target_interval_high_mV_dec=("target_interval_high_mV_dec", "first"),
            n_repeated_oof=("repeat", "count"),
        )
    )
    aggregated["absolute_error_mV_dec"] = np.abs(
        aggregated["predicted_mV_dec"] - aggregated["observed_mV_dec"]
    )
    return aggregated


def extended_regression_metrics(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    base = regression_metrics(observed, predicted)
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed
    tail_threshold = float(np.quantile(observed, 0.8))
    tail = observed >= tail_threshold
    if np.ptp(predicted) > 0 and len(predicted) > 1:
        calibration_slope, calibration_intercept = np.polyfit(predicted, observed, 1)
        residual_calibration = spearmanr(np.abs(residual), predicted).statistic
    else:
        calibration_slope = np.nan
        calibration_intercept = np.nan
        residual_calibration = np.nan
    return {
        "mae_mV_dec": base["mae"],
        "rmse_mV_dec": base["rmse"],
        "median_ae_mV_dec": base["median_ae"],
        "r2": base["r2"],
        "spearman": base["spearman"],
        "tail_threshold_q80_mV_dec": tail_threshold,
        "tail_mae_mV_dec": float(mean_absolute_error(observed[tail], predicted[tail])),
        "prediction_min_mV_dec": float(np.min(predicted)),
        "prediction_max_mV_dec": float(np.max(predicted)),
        "prediction_range_mV_dec": float(np.ptp(predicted)),
        "mean_residual_pred_minus_obs_mV_dec": float(np.mean(residual)),
        "calibration_intercept_mV_dec": float(calibration_intercept),
        "calibration_slope": float(calibration_slope),
        "abs_residual_vs_prediction_spearman": (
            float(residual_calibration) if np.isfinite(residual_calibration) else np.nan
        ),
        "n_negative_predictions": int(np.sum(predicted < 0)),
    }


def _metric_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (analysis_id, role), group in aggregated.groupby(
        ["analysis_id", "prediction_role"], sort=True
    ):
        rows.append(
            {
                "analysis_id": analysis_id,
                "prediction_role": role,
                "n_records": int(len(group)),
                **extended_regression_metrics(
                    group["observed_mV_dec"].to_numpy(float),
                    group["predicted_mV_dec"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_oof(
    aggregated: pd.DataFrame,
    *,
    replicates: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replicates <= 0:
        return pd.DataFrame(), pd.DataFrame()
    wide = aggregated.pivot(
        index=["record_id", "cv_group_id", "observed_mV_dec"],
        columns="prediction_role",
        values="predicted_mV_dec",
    ).reset_index()
    required = {"joint_selector", "dummy_selector"}
    if not required.issubset(wide.columns):
        raise KeyError(f"OOF bootstrap is missing prediction roles: {sorted(required - set(wide))}")
    groups = wide["cv_group_id"].astype(str).unique()
    positions = {
        group: np.flatnonzero(wide["cv_group_id"].astype(str).to_numpy() == group)
        for group in groups
    }
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        selected = np.concatenate([positions[group] for group in sampled_groups])
        observed = wide.iloc[selected]["observed_mV_dec"].to_numpy(float)
        joint = wide.iloc[selected]["joint_selector"].to_numpy(float)
        dummy = wide.iloc[selected]["dummy_selector"].to_numpy(float)
        joint_metrics = extended_regression_metrics(observed, joint)
        dummy_mae = float(mean_absolute_error(observed, dummy))
        rows.append(
            {
                "bootstrap_replicate": replicate,
                "n_sampled_groups": int(len(sampled_groups)),
                "n_sampled_records_with_multiplicity": int(len(selected)),
                "joint_mae_mV_dec": joint_metrics["mae_mV_dec"],
                "joint_rmse_mV_dec": joint_metrics["rmse_mV_dec"],
                "joint_median_ae_mV_dec": joint_metrics["median_ae_mV_dec"],
                "joint_r2": joint_metrics["r2"],
                "joint_spearman": joint_metrics["spearman"],
                "dummy_mae_mV_dec": dummy_mae,
                "delta_mae_joint_minus_dummy_mV_dec": (
                    joint_metrics["mae_mV_dec"] - dummy_mae
                ),
            }
        )
    draws = pd.DataFrame(rows)
    summary_rows = []
    for metric in (
        "joint_mae_mV_dec",
        "joint_rmse_mV_dec",
        "joint_median_ae_mV_dec",
        "joint_r2",
        "joint_spearman",
        "dummy_mae_mV_dec",
        "delta_mae_joint_minus_dummy_mV_dec",
    ):
        values = pd.to_numeric(draws[metric], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric": metric,
                "n_bootstrap": int(len(values)),
                "median": float(values.median()),
                "q025": float(values.quantile(0.025)),
                "q975": float(values.quantile(0.975)),
                "probability_below_zero": float(np.mean(values < 0)),
            }
        )
    return draws, pd.DataFrame(summary_rows)


def _inverse_uncertainty_weighted_metrics(
    aggregated: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (analysis_id, role), group in aggregated.groupby(
        ["analysis_id", "prediction_role"], sort=True
    ):
        observed = group["observed_mV_dec"].to_numpy(float)
        predicted = group["predicted_mV_dec"].to_numpy(float)
        width = (
            group["target_interval_high_mV_dec"].to_numpy(float)
            - group["target_interval_low_mV_dec"].to_numpy(float)
        )
        sigma = width / (2.0 * 1.96)
        finite_positive = sigma[np.isfinite(sigma) & (sigma > 0)]
        floor = float(np.min(finite_positive)) if finite_positive.size else 1.0
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, floor)
        weights = 1.0 / np.square(sigma)
        weights = weights / np.sum(weights)
        residual = predicted - observed
        observed_mean = float(np.sum(weights * observed))
        denominator = float(np.sum(weights * np.square(observed - observed_mean)))
        rows.append(
            {
                "analysis_id": analysis_id,
                "prediction_role": role,
                "n_records": int(len(group)),
                "weighted_mae_mV_dec": float(np.sum(weights * np.abs(residual))),
                "weighted_rmse_mV_dec": float(
                    np.sqrt(np.sum(weights * np.square(residual)))
                ),
                "weighted_bias_pred_minus_obs_mV_dec": float(
                    np.sum(weights * residual)
                ),
                "weighted_r2": (
                    float(1.0 - np.sum(weights * np.square(residual)) / denominator)
                    if denominator > 0
                    else np.nan
                ),
                "effective_sample_size": float(1.0 / np.sum(np.square(weights))),
                "estimand": "inverse_posterior_variance_weighted_score_only_no_retraining",
            }
        )
    return pd.DataFrame(rows)


def _selection_frequency(selections: pd.DataFrame) -> pd.DataFrame:
    dimensions = (
        "recipe",
        "base",
        "include_ink_amounts",
        "solvent_representation",
        "include_property_amount_interactions",
        "include_dielectric_estimated_flag",
        "include_process_features",
        "numeric_transform",
        "target_transform",
        "model",
    )
    rows: list[dict[str, object]] = []
    total = len(selections)
    for dimension in dimensions:
        for value, count in selections[dimension].value_counts(dropna=False).items():
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "count": int(count),
                    "frequency": float(count / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _candidate_search_health(
    scores: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict[str, object]:
    outer_cells = scores.loc[:, ["repeat", "fold"]].drop_duplicates()
    completed = scores.loc[scores["status"].eq("complete")].copy()
    missing_models: list[str] = []
    for cell in outer_cells.itertuples(index=False):
        cell_complete = completed.loc[
            completed["repeat"].eq(cell.repeat) & completed["fold"].eq(cell.fold)
        ]
        for model in sorted(catalog["model"].astype(str).unique()):
            if not cell_complete["model"].astype(str).eq(model).any():
                missing_models.append(f"repeat={cell.repeat},fold={cell.fold},model={model}")
    missing_recipes: list[str] = []
    for cell in outer_cells.itertuples(index=False):
        cell_complete = completed.loc[
            completed["repeat"].eq(cell.repeat) & completed["fold"].eq(cell.fold)
        ]
        for recipe in sorted(catalog["recipe"].astype(str).unique()):
            if not cell_complete["recipe"].astype(str).eq(recipe).any():
                missing_recipes.append(
                    f"repeat={cell.repeat},fold={cell.fold},recipe={recipe}"
                )
    failed = scores.loc[~scores["status"].eq("complete")]
    return {
        "status": (
            "pass" if not missing_models and not missing_recipes else "fail"
        ),
        "n_candidate_outer_evaluations": int(len(scores)),
        "n_completed_candidate_outer_evaluations": int(len(completed)),
        "n_failed_candidate_outer_evaluations": int(len(failed)),
        "completion_fraction": float(len(completed) / len(scores)) if len(scores) else 0.0,
        "all_model_families_have_a_completed_candidate_in_every_outer_fold": not missing_models,
        "all_feature_recipes_have_a_completed_candidate_in_every_outer_fold": not missing_recipes,
        "missing_model_fold_coverage": missing_models,
        "missing_recipe_fold_coverage": missing_recipes,
        "failed_candidate_ids": sorted(failed["candidate_id"].astype(str).unique()),
    }


def _candidate_source_union(candidates: Sequence[CandidateDefinition]) -> tuple[str, ...]:
    columns: list[str] = []
    for candidate in candidates:
        if candidate.recipe is not None:
            columns.extend(required_source_columns(candidate.recipe))
    result = tuple(dict.fromkeys(columns))
    validate_no_target_leakage(result)
    forbidden = sorted(column for column in result if is_target_like_column(column))
    if forbidden:
        raise AssertionError(f"Candidate source union contains forbidden columns: {forbidden}")
    return result


def run_analysis(
    frame: pd.DataFrame,
    analysis: AnalysisSpec,
    output_dir: Path,
    *,
    config: OptimizationConfig,
    specs: Mapping[str, OptimizationModelSpec],
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> tuple[AnalysisResult, tuple[CandidateDefinition, ...]]:
    scoped = frame.copy()
    if analysis.scope == "alkaline":
        scoped = scoped.loc[scoped["electrolyte_scope"].astype(str).eq("alkaline")]
    elif analysis.scope != "all":
        raise ValueError(f"Unsupported optimization scope: {analysis.scope!r}")
    if analysis.strict_complete_cases:
        scoped = scoped.loc[
            pd.to_numeric(scoped[PROPERTY_DIELECTRIC_STRICT], errors="coerce").notna()
        ]
        strict_values = pd.to_numeric(
            scoped[PROPERTY_DIELECTRIC_STRICT], errors="raise"
        ).to_numpy(float)
        best_values = pd.to_numeric(
            scoped[PROPERTY_DIELECTRIC_BEST], errors="raise"
        ).to_numpy(float)
        if not np.array_equal(strict_values, best_values):
            raise ValueError(
                "Strict complete-case sensitivity requires best-available and "
                "source-backed dielectric values to be identical on retained records"
            )
    scoped = scoped.reset_index(drop=True)
    recipe_names = recipes_for_policy(analysis.recipe_policy)
    if config.smoke:
        smoke_names = {
            "conditions_noink_none",
            "context_ink_fractions",
            "context_ink_properties_pca2",
            "context_ink_properties_pca2_interactions",
            "context_ink_fractions_plus_properties_pca2",
            "context_ink_fractions_with_process",
        }
        recipe_names = tuple(name for name in recipe_names if name in smoke_names)
    candidates = build_candidate_catalog(
        recipe_names,
        specs,
        budget=config.candidate_budget,
        random_seed=config.random_seed,
    )
    candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
    candidate_catalog = pd.DataFrame([candidate.row() for candidate in candidates])
    source_columns = _candidate_source_union(candidates)
    missing_sources = sorted(set(source_columns) - set(scoped.columns))
    if missing_sources:
        raise KeyError(f"Optimization cohort is missing predictor sources: {missing_sources}")
    X = scoped.loc[:, list(source_columns)].copy()
    y = pd.to_numeric(scoped[TARGET_COLUMN], errors="raise").to_numpy(float)
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("Primary Bayesian kinetic slope targets must be finite and positive")

    split_manifest = make_outer_split_manifest(
        scoped,
        group_col="cv_group_id",
        record_id_col="record_id",
        folds=config.outer_folds,
        repeats=config.outer_repeats,
        repeat_seeds=config.outer_seeds,
        relation_columns=relation_columns,
    )
    inner_manifest_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    fold_score_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []

    outer_cells = list(split_manifest.groupby(["repeat", "fold"], sort=True))
    for cell_number, (
        (repeat_index, fold_index),
        test_manifest,
    ) in enumerate(outer_cells, start=1):
        print(
            f"[optimize:{analysis.analysis_id}] outer fold "
            f"{cell_number}/{len(outer_cells)} "
            f"(repeat={int(repeat_index)}, fold={int(fold_index)}, "
            f"candidates={len(candidates)})",
            flush=True,
        )
        test_positions = np.sort(test_manifest["row_position"].to_numpy(int))
        test_mask = np.zeros(len(scoped), dtype=bool)
        test_mask[test_positions] = True
        train_positions = np.flatnonzero(~test_mask)
        outer_train = scoped.iloc[train_positions].reset_index(drop=True)
        outer_test = scoped.iloc[test_positions].reset_index(drop=True)
        assert_no_relational_leakage(
            outer_train,
            outer_test,
            group_col="cv_group_id",
            relation_columns=relation_columns,
        )
        X_train = X.iloc[train_positions].reset_index(drop=True)
        X_test = X.iloc[test_positions].reset_index(drop=True)
        y_train = y[train_positions]
        y_test = y[test_positions]
        inner_seed = int(config.outer_seeds[int(repeat_index)] + int(fold_index) * 10_003)
        (
            winner,
            dummy,
            scores,
            fold_scores,
            assignments,
            winner_details,
            dummy_details,
        ) = select_outer_training_winner(
            outer_train,
            X_train,
            y_train,
            candidates,
            specs,
            inner_folds=config.inner_folds,
            inner_seed=inner_seed,
            tie_tolerance_mV_dec=config.tie_tolerance_mV_dec,
            n_jobs=config.n_jobs,
            relation_columns=relation_columns,
        )
        scores.insert(0, "analysis_id", analysis.analysis_id)
        scores.insert(1, "repeat", int(repeat_index))
        scores.insert(2, "fold", int(fold_index))
        scores["shared_inner_split_seed"] = inner_seed
        scores["n_shared_inner_splits"] = int(len(np.unique(assignments)))
        score_frames.append(scores)
        fold_scores.insert(0, "analysis_id", analysis.analysis_id)
        fold_scores.insert(1, "repeat", int(repeat_index))
        fold_scores.insert(2, "fold", int(fold_index))
        fold_score_frames.append(fold_scores)

        for position, inner_fold in enumerate(assignments):
            inner_manifest_rows.append(
                {
                    "analysis_id": analysis.analysis_id,
                    "outer_repeat": int(repeat_index),
                    "outer_fold": int(fold_index),
                    "inner_fold": int(inner_fold),
                    "record_id": str(outer_train.iloc[position]["record_id"]),
                    "cv_group_id": str(outer_train.iloc[position]["cv_group_id"]),
                    "inner_role": "validation",
                    "inner_seed": inner_seed,
                    "candidate_catalog_size": len(candidates),
                }
            )

        _, joint_prediction = _safe_predict_fitted(
            winner, specs, X_train, y_train, X_test
        )
        _, dummy_prediction = _safe_predict_fitted(
            dummy, specs, X_train, y_train, X_test
        )
        fixed_predictions: dict[str, tuple[CandidateDefinition, np.ndarray]] = {
            "joint_selector": (winner, joint_prediction),
            "dummy_selector": (dummy, dummy_prediction),
        }
        for fixed_model, role in (
            ("dummy_mean", "dummy_mean_raw"),
            ("dummy_median", "dummy_median_raw"),
        ):
            fixed_candidate = next(
                candidate
                for candidate in candidates
                if candidate.model == fixed_model
                and candidate.target_transform == "identity"
            )
            _, fixed_prediction = _safe_predict_fitted(
                fixed_candidate, specs, X_train, y_train, X_test
            )
            fixed_predictions[role] = (fixed_candidate, fixed_prediction)

        winner_row = winner.row()
        selection_rows.append(
            {
                "analysis_id": analysis.analysis_id,
                "repeat": int(repeat_index),
                "fold": int(fold_index),
                "n_outer_train": int(len(train_positions)),
                "n_outer_test": int(len(test_positions)),
                "n_outer_train_groups": int(outer_train["cv_group_id"].nunique()),
                "n_outer_test_groups": int(outer_test["cv_group_id"].nunique()),
                **winner_row,
                **{f"selection_{key}": value for key, value in winner_details.items()},
                "dummy_candidate_id": dummy.candidate_id,
                "dummy_model": dummy.model,
                "dummy_target_transform": dummy.target_transform,
                "dummy_inner_mae_mV_dec": dummy_details[
                    "selected_inner_mae_mV_dec"
                ],
            }
        )
        for role, (candidate, predicted) in fixed_predictions.items():
            for local_position, record in outer_test.iterrows():
                oof_rows.append(
                    {
                        "analysis_id": analysis.analysis_id,
                        "scope": analysis.scope,
                        "prediction_role": role,
                        "repeat": int(repeat_index),
                        "fold": int(fold_index),
                        "record_id": str(record["record_id"]),
                        "sample_record": int(record["record_id"]),
                        "curve_id": str(record["curve_id"]),
                        "cv_group_id": str(record["cv_group_id"]),
                        "observed_mV_dec": float(y_test[local_position]),
                        "predicted_mV_dec": float(predicted[local_position]),
                        "absolute_error_mV_dec": float(
                            abs(predicted[local_position] - y_test[local_position])
                        ),
                        "target_interval_low_mV_dec": float(
                            record[TARGET_INTERVAL_LOW]
                        ),
                        "target_interval_high_mV_dec": float(
                            record[TARGET_INTERVAL_HIGH]
                        ),
                        "candidate_id": candidate.candidate_id,
                        "recipe": candidate.recipe or "dummy",
                        "model": candidate.model,
                        "numeric_transform": candidate.numeric_transform,
                        "target_transform": candidate.target_transform,
                    }
                )

    candidate_scores = pd.concat(score_frames, ignore_index=True)
    inner_fold_scores = pd.concat(fold_score_frames, ignore_index=True)
    outer_selections = pd.DataFrame(selection_rows)
    oof_predictions = pd.DataFrame(oof_rows)
    oof_aggregated = _aggregate_oof(oof_predictions)
    metrics = _metric_table(oof_aggregated)
    bootstrap_metrics, bootstrap_summary = _bootstrap_oof(
        oof_aggregated,
        replicates=config.bootstrap_replicates,
        random_seed=config.random_seed,
    )
    weighted = _inverse_uncertainty_weighted_metrics(oof_aggregated)
    frequency = _selection_frequency(outer_selections)
    inner_split_manifest = pd.DataFrame(inner_manifest_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_catalog.to_csv(output_dir / "candidate_catalog.csv", index=False)
    split_manifest.to_csv(output_dir / "outer_split_manifest.csv", index=False)
    inner_split_manifest.to_csv(output_dir / "inner_split_manifest.csv", index=False)
    candidate_scores.to_csv(output_dir / "inner_candidate_scores.csv", index=False)
    inner_fold_scores.to_csv(output_dir / "inner_fold_scores.csv", index=False)
    outer_selections.to_csv(output_dir / "outer_selected_pipelines.csv", index=False)
    oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    oof_aggregated.to_csv(output_dir / "oof_aggregated.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    bootstrap_metrics.to_csv(output_dir / "connected_group_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(
        output_dir / "connected_group_bootstrap_summary.csv", index=False
    )
    weighted.to_csv(
        output_dir / "inverse_uncertainty_weighted_score_sensitivity.csv",
        index=False,
    )
    frequency.to_csv(output_dir / "selection_frequency.csv", index=False)
    search_health = _candidate_search_health(candidate_scores, candidate_catalog)
    write_json(output_dir / "candidate_search_health.json", search_health)
    write_json(
        output_dir / "analysis_definition.json",
        {
            **asdict(analysis),
            "target": TARGET_NAME,
            "target_training_label": "posterior_median_equal_record_weight",
            "selection_metric": "pooled_inner_grouped_cv_mae_original_mV_dec",
            "candidate_catalog_size": len(candidates),
            "candidate_catalog_sha256": hashlib.sha256(
                candidate_catalog.to_csv(index=False).encode("utf-8")
            ).hexdigest(),
            "n_records": len(scoped),
            "n_connected_groups": int(scoped["cv_group_id"].nunique()),
            "n_outer_predictions": len(oof_predictions),
            "candidate_search_health": search_health,
        },
    )
    result = AnalysisResult(
        analysis_id=analysis.analysis_id,
        split_manifest=split_manifest,
        inner_split_manifest=inner_split_manifest,
        candidate_scores=candidate_scores,
        inner_fold_scores=inner_fold_scores,
        outer_selections=outer_selections,
        oof_predictions=oof_predictions,
        oof_aggregated=oof_aggregated,
        metrics=metrics,
        bootstrap_metrics=bootstrap_metrics,
        bootstrap_summary=bootstrap_summary,
        weighted_sensitivity=weighted,
        selection_frequency=frequency,
    )
    return result, candidates


def _decision_from_primary(result: AnalysisResult) -> dict[str, object]:
    metrics = result.metrics.set_index("prediction_role")
    joint_mae = float(metrics.loc["joint_selector", "mae_mV_dec"])
    dummy_mae = float(metrics.loc["dummy_selector", "mae_mV_dec"])
    joint_r2 = float(metrics.loc["joint_selector", "r2"])
    bootstrap = result.bootstrap_summary.set_index("metric")
    delta = bootstrap.loc["delta_mae_joint_minus_dummy_mV_dec"]
    delta_point = joint_mae - dummy_mae
    interval_supports = bool(float(delta["q975"]) < 0)
    point_improves = bool(delta_point < 0)
    if point_improves and interval_supports and joint_r2 > 0:
        classification = "Established improvement"
    elif point_improves:
        classification = "Promising only"
    else:
        classification = "No useful predictive model"
    return {
        "classification": classification,
        "decision_rule": {
            "established": "joint MAE below dummy, paired connected-group bootstrap 95% interval below zero, and aggregated OOF R2 > 0",
            "promising": "point MAE below dummy but interval crosses zero or aggregated OOF R2 <= 0",
            "no_useful": "joint selector MAE is not below dummy",
        },
        "joint_selector_mae_mV_dec": joint_mae,
        "dummy_selector_mae_mV_dec": dummy_mae,
        "delta_mae_joint_minus_dummy_mV_dec": delta_point,
        "delta_bootstrap_q025_mV_dec": float(delta["q025"]),
        "delta_bootstrap_q975_mV_dec": float(delta["q975"]),
        "bootstrap_probability_delta_below_zero": float(
            delta["probability_below_zero"]
        ),
        "joint_selector_r2": joint_r2,
        "point_improves": point_improves,
        "interval_supports_improvement": interval_supports,
        "development_status": "internal_performance_optimization_after_prior_outer_results_were_viewed",
        "classification_scope": "internal_nested_cv_decision_rule_only",
        "independent_confirmation_required": "future prospective connected-group holdout from newly added publications",
    }


def _fit_final_primary_pipeline(
    primary_frame: pd.DataFrame,
    candidates: Sequence[CandidateDefinition],
    specs: Mapping[str, OptimizationModelSpec],
    output_dir: Path,
    *,
    config: OptimizationConfig,
    nested_metrics: pd.DataFrame,
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> dict[str, object]:
    source_columns = _candidate_source_union(candidates)
    X = primary_frame.loc[:, list(source_columns)].reset_index(drop=True)
    y = primary_frame[TARGET_COLUMN].to_numpy(float)
    splits, assignments = _make_inner_splits(
        primary_frame.reset_index(drop=True),
        group_col="cv_group_id",
        inner_folds=config.inner_folds,
        seed=config.random_seed,
        relation_columns=relation_columns,
    )
    scores, fold_scores = evaluate_candidate_catalog(
        candidates, specs, X, y, splits, n_jobs=config.n_jobs
    )
    lookup = {candidate.candidate_id: candidate for candidate in candidates}
    winner, details = choose_inner_winner(
        scores,
        tie_tolerance_mV_dec=config.tie_tolerance_mV_dec,
        candidate_lookup=lookup,
    )
    estimator, _ = _safe_predict_fitted(winner, specs, X, y, X)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "full_data_inner_candidate_scores.csv", index=False)
    fold_scores.to_csv(output_dir / "full_data_inner_fold_scores.csv", index=False)
    pd.DataFrame(
        {
            "record_id": primary_frame["record_id"].astype(str),
            "cv_group_id": primary_frame["cv_group_id"].astype(str),
            "inner_fold": assignments,
        }
    ).to_csv(output_dir / "full_data_inner_split_manifest.csv", index=False)
    model_path = output_dir / "fitted_final_pipeline.joblib"
    joblib.dump(estimator, model_path)
    engineered_features: list[str] = []
    if hasattr(estimator, "named_steps") and "features" in estimator.named_steps:
        engineered_features = [
            str(value)
            for value in estimator.named_steps["features"].get_feature_names_out()
        ]
    metric_rows = nested_metrics.loc[
        nested_metrics["prediction_role"].eq("joint_selector")
    ].to_dict(orient="records")
    specification = {
        "target": TARGET_NAME,
        "target_unit": "mV dec^-1",
        "training_label": "posterior median; equal record weight",
        "population": "90 alkaline records with a Bayesian local-line target",
        "selection": details,
        "pipeline": winner.row(),
        "required_source_columns": list(
            required_source_columns(winner.recipe) if winner.recipe else ()
        ),
        "fitted_engineered_features": engineered_features,
        "n_training_records": int(len(primary_frame)),
        "n_connected_groups": int(primary_frame["cv_group_id"].nunique()),
        "performance_source": (
            "internally nested repeated connected-group outer OOF selector; outer tests are "
            "untouched within this run, but this is not prospective independent confirmation"
        ),
        "nested_outer_oof_metrics": metric_rows[0] if metric_rows else {},
        "serialized_pipeline": str(model_path),
        "serialized_pipeline_sha256": sha256_file(model_path),
        "artifact_role": (
            "structural_smoke_test_only_not_locked"
            if config.smoke
            else "final_full_data_pipeline_locked_by_grouped_inner_cv"
        ),
        "dielectric_policy": (
            "primary property candidates use the 97/97 best-available numeric proxy; "
            "31 values are estimates and the estimate flag is sensitivity-only"
        ),
    }
    write_json(output_dir / "exact_final_pipeline.json", _jsonable(specification))
    return specification


def _verify_canonical_package(
    canonical_dir: Path,
    *,
    expected_tree_sha256: str,
) -> dict[str, object]:
    manifest_path = canonical_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Canonical artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Canonical artifact manifest has no registered files")
    registered = {str(row["path"]): row for row in rows}
    actual = {
        str(path.relative_to(canonical_dir))
        for path in canonical_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(registered) != actual:
        raise ValueError(
            "Canonical artifact manifest membership mismatch: "
            f"missing={sorted(set(registered) - actual)}, "
            f"unexpected={sorted(actual - set(registered))}"
        )
    mismatches: list[str] = []
    for relative, row in registered.items():
        path = canonical_dir / relative
        if int(row.get("size_bytes", -1)) != path.stat().st_size:
            mismatches.append(relative)
            continue
        if str(row.get("sha256", "")) != sha256_file(path):
            mismatches.append(relative)
    if mismatches:
        raise ValueError(
            f"Canonical artifact manifest hash/size mismatch: {sorted(mismatches)}"
        )
    tree = directory_tree_summary(canonical_dir)
    actual_tree = str(tree["tree_sha256"])
    if not expected_tree_sha256 or actual_tree != str(expected_tree_sha256):
        raise ValueError(
            "Canonical package tree hash does not match the frozen configuration: "
            f"expected={expected_tree_sha256!r}, actual={actual_tree!r}"
        )
    return {
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "registered_files": len(registered),
        "tree_sha256": actual_tree,
        "expected_tree_sha256": str(expected_tree_sha256),
        "tree_hash_matches_expected": True,
    }


def _prepare_canonical_inputs(
    canonical_dir: Path,
    *,
    expected_tree_sha256: str,
    approved_target_registry: Path,
    target_qa_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    master_path = canonical_dir / "master.csv"
    # The frozen canonical package contains the pre-review target evidence.
    # Named PPC decisions are rebound in the active formal run, so optimization
    # must combine canonical predictors with that approved target registry.
    target_path = Path(approved_target_registry).expanduser().resolve()
    target_qa_path = Path(target_qa_path).expanduser().resolve()
    draws_path = canonical_dir / "posterior_slope_draws.csv.gz"
    for path in (master_path, target_path, target_qa_path, draws_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required canonical optimization input is missing: {path}")
    package_verification = _verify_canonical_package(
        canonical_dir, expected_tree_sha256=expected_tree_sha256
    )
    target_qa = json.loads(target_qa_path.read_text(encoding="utf-8"))
    if not bool(target_qa.get("formal_evaluation_ready", False)):
        raise ValueError(
            "Active target QA is not formally evaluation-ready: "
            f"{target_qa_path}"
        )
    if int(target_qa.get("n_formal_ppc_approved", -1)) != 97:
        raise ValueError(
            "Optimization requires 97 named formal PPC approvals; "
            f"found {target_qa.get('n_formal_ppc_approved')!r}"
        )
    records = normalize_optimization_columns(pd.read_csv(master_path))
    targets = pd.read_csv(target_path)
    if records["record_id"].astype(str).duplicated().any():
        raise ValueError("Canonical master must contain exactly one row per record_id")
    selected = targets.loc[
        targets["target_name"].astype(str).eq(TARGET_NAME)
        & targets["target_role"].astype(str).eq("primary")
        & targets["target_eligible"].astype(bool)
        & targets["formal_target_approved"].astype(bool),
        [
            "record_id",
            TARGET_COLUMN,
            TARGET_INTERVAL_LOW,
            TARGET_INTERVAL_HIGH,
        ],
    ].copy()
    if selected["record_id"].astype(str).duplicated().any():
        raise ValueError("Primary target registry has duplicate record_id values")
    records["record_id"] = records["record_id"].astype(str)
    selected["record_id"] = selected["record_id"].astype(str)
    frame = records.merge(selected, on="record_id", how="inner", validate="one_to_one")
    if len(frame) != 97:
        raise ValueError(f"Optimization requires 97 approved Bayesian local-line targets; found {len(frame)}")
    frame["cv_group_id"] = build_connected_group_ids(
        frame,
        relation_columns=DEFAULT_RELATION_COLUMNS,
        record_id_col="record_id",
    )
    if int(frame["electrolyte_scope"].astype(str).eq("alkaline").sum()) != 90:
        raise ValueError("Optimization primary cohort must contain exactly 90 alkaline records")
    strict_count = int(
        pd.to_numeric(frame[PROPERTY_DIELECTRIC_STRICT], errors="coerce").notna().sum()
    )
    estimated_count = int(
        pd.to_numeric(
            frame[
                "solvent_mixture_property_proxy_dielectric_constant_estimated_flag"
            ],
            errors="coerce",
        ).fillna(0).sum()
    )
    binding = {
        "canonical_dir": str(canonical_dir),
        "master_csv": str(master_path),
        "master_sha256": sha256_file(master_path),
        "target_registry_csv": str(target_path),
        "target_registry_sha256": sha256_file(target_path),
        "target_qa_json": str(target_qa_path),
        "target_qa_sha256": sha256_file(target_qa_path),
        "target_qa_formal_evaluation_ready": True,
        "target_qa_formal_ppc_approved": 97,
        "posterior_slope_draws": str(draws_path),
        "posterior_slope_draws_sha256": sha256_file(draws_path),
        "target": TARGET_NAME,
        "n_records_all": int(len(frame)),
        "n_records_alkaline": int(
            frame["electrolyte_scope"].astype(str).eq("alkaline").sum()
        ),
        "n_connected_groups_all": int(frame["cv_group_id"].nunique()),
        "connected_group_relation_columns_present": [
            column for column in DEFAULT_RELATION_COLUMNS if column in frame.columns
        ],
        "connected_group_relation_columns_absent": [
            column for column in DEFAULT_RELATION_COLUMNS if column not in frame.columns
        ],
        "verified_lab_group_id_available": bool("lab_group_id" in frame.columns),
        "lab_validation_boundary": (
            "canonical master has no verified lab_group_id; connected groups use "
            "available publication/recipe/sample relations, unknown labs remain "
            "publication-independent, and this is not definitive lab-held-out validation"
        ),
        "n_strict_source_backed_dielectric_complete": strict_count,
        "n_best_available_dielectric_estimates": estimated_count,
        "forbidden_predictor_policy": "explicit allowlist only; identifiers/groups/targets/posterior/fit/provenance excluded",
        "canonical_package_verification": package_verification,
    }
    return frame, binding


def _score_posterior_draws(
    result: AnalysisResult,
    draws_path: Path,
    destination: Path,
    *,
    max_draws: int,
    random_seed: int,
) -> dict[str, object]:
    oof = result.oof_aggregated.copy()
    oof["target"] = TARGET_NAME
    oof["scope"] = result.analysis_id
    oof["panel"] = "joint_feature_model_search"
    oof["model"] = oof["prediction_role"]
    oof["record_id"] = pd.to_numeric(oof["record_id"], errors="raise").astype(int)
    score_functional_target_draws(
        oof,
        draws_path,
        destination,
        target_name=TARGET_NAME,
        max_draws=max_draws,
        random_seed=random_seed,
    )
    qa_path = destination / "target_draw_qa.json"
    if not qa_path.is_file():
        raise FileNotFoundError(f"Posterior-draw QA was not written: {qa_path}")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(qa, dict):
        raise ValueError(f"Posterior-draw QA is not a JSON object: {qa_path}")
    return qa


def _write_summary_report(
    destination: Path,
    decision: Mapping[str, object],
    primary: AnalysisResult,
    final_spec: Mapping[str, object],
    *,
    smoke: bool,
) -> None:
    metrics = primary.metrics.set_index("prediction_role")
    joint = metrics.loc["joint_selector"]
    dummy = metrics.loc["dummy_selector"]
    status = (
        "Structural smoke pass" if smoke else str(decision["classification"])
    )
    primary_heading = (
        "## Structural coverage metrics (not scientific estimates)"
        if smoke
        else "## Primary 90-record alkaline result"
    )
    pipeline_heading = (
        "## Serialized smoke-test pipeline artifact (not locked)"
        if smoke
        else "## Final locked full-data pipeline"
    )
    lines = [
        "# Bayesian local-line Tafel slope performance optimization",
        "",
        f"Status: **{status}**",
        "",
        *(
            [
                "**Smoke mode is structural validation only. The displayed metrics do not support a performance classification, and the serialized pipeline is a test artifact rather than a locked model.**",
                "",
            ]
            if smoke
            else []
        ),
        "This is an independent development/internal optimization stage. It does not replace or modify the registered `current_97` context-vs-ink analysis.",
        "",
        primary_heading,
        "",
        f"- Joint-selector record-aggregated OOF MAE: {float(joint['mae_mV_dec']):.3f} mV dec⁻¹",
        f"- Dummy-selector record-aggregated OOF MAE: {float(dummy['mae_mV_dec']):.3f} mV dec⁻¹",
        f"- ΔMAE (joint − dummy): {float(decision['delta_mae_joint_minus_dummy_mV_dec']):+.3f} mV dec⁻¹",
        f"- Connected-group bootstrap 95% interval: [{float(decision['delta_bootstrap_q025_mV_dec']):+.3f}, {float(decision['delta_bootstrap_q975_mV_dec']):+.3f}]",
        f"- Aggregated OOF R²: {float(joint['r2']):.3f}",
        f"- Aggregated OOF Spearman: {float(joint['spearman']):.3f}",
        f"- High-tail (top 20%) MAE: {float(joint['tail_mae_mV_dec']):.3f} mV dec⁻¹",
        "",
        pipeline_heading,
        "",
        f"- Candidate: `{final_spec['pipeline']['candidate_id']}`",
        f"- Feature recipe: `{final_spec['pipeline']['recipe']}`",
        f"- Numeric transform: `{final_spec['pipeline']['numeric_transform']}`",
        f"- Target transform: `{final_spec['pipeline']['target_transform']}`",
        f"- Model: `{final_spec['pipeline']['model']}`",
        f"- Parameters: `{final_spec['pipeline']['model_params_json']}`",
        "",
        (
            "This pipeline was selected and serialized only to exercise the full code path; it must not be reported as a locked model."
            if smoke
            else "Its quoted performance is the internally nested selector outer-OOF result above, not training-set fit and not the outer performance of a retrospectively chosen fixed combination. It is not prospective independent confirmation."
        ),
        "",
        "## Boundaries",
        "",
        "- All imputation, rare-level pooling, ILR, PCA, scaling, near-zero-variance removal, model selection, and hyperparameter selection are fitted inside inner-training folds.",
        "- Posterior interval width is never a predictor. Posterior draws are used only to rescore fixed OOF predictions.",
        "- The inverse-uncertainty result is a score-only sensitivity with a different estimand.",
        "- Property values are solvent-mixture proxies, not measured complete-ink properties.",
        "- Primary property candidates use the 97/97 best-available dielectric number; 31 values are estimates, while the estimate flag competes only in sensitivity recipes.",
        "- The canonical snapshot has no verified `lab_group_id`; connected groups use available publication/recipe/sample relations. Unknown labs remain publication-independent, so this is not definitive lab-held-out validation.",
        "- Because previous outer results were already viewed, independent confirmation requires future publications as a prospective connected-group holdout.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_performance_optimization(
    run_dir: Path,
    *,
    project_root: Path | None = None,
    smoke: bool = False,
    n_jobs: int = 1,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Run the independent optimization workflow and return a JSON-safe summary."""

    destination = Path(run_dir).expanduser().resolve()
    repository_root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    formal_run = (
        repository_root / "Data" / "03_outputs" / "study" / "outputs" / "current_97"
    ).resolve()
    if destination == formal_run or formal_run in destination.parents:
        raise ValueError(
            "Performance optimization cannot write into the registered formal run: "
            f"{formal_run}"
        )
    owned_outputs = (
        "registry",
        "analyses",
        "final_pipeline",
        "performance_decision.json",
        "optimization_qa.json",
        "performance_optimization_report.md",
        "all_analysis_metrics.csv",
        "all_selection_frequencies.csv",
        "artifact_manifest.json",
    )
    stale = [name for name in owned_outputs if (destination / name).exists()]
    if stale:
        raise FileExistsError(
            "Optimization destination already contains stage-owned artifacts; use a new "
            f"run directory to avoid stale output mixing: {stale}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "provenance").mkdir(exist_ok=True)
    formal_tree_before = (
        directory_tree_summary(formal_run)["tree_sha256"]
        if formal_run.is_dir()
        else None
    )
    canonical_dir = repository_root / "Data" / "02_inputs" / "canonical_dataset_ir_corrected_v2"
    config = optimization_config_from_settings(settings, n_jobs=int(n_jobs))
    if smoke:
        config = replace(
            config,
            outer_folds=3,
            outer_repeats=1,
            inner_folds=2,
            outer_seeds=(42,),
            candidate_budget=24,
            bootstrap_replicates=100,
            target_draw_repetitions=5,
            smoke=True,
        )
    expected_tree_sha256 = str(
        (settings or {}).get("expected_canonical_tree_sha256", "")
    )
    if not expected_tree_sha256:
        config_path = repository_root / "code" / "Ink_Tafel_Study" / "config" / "study.yaml"
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected_tree_sha256 = str(
            loaded["input_hashes"]["expected"]["canonical_dataset"]
        )
    frame, binding = _prepare_canonical_inputs(
        canonical_dir,
        expected_tree_sha256=expected_tree_sha256,
        approved_target_registry=formal_run / "targets" / "target_registry.csv",
        target_qa_path=formal_run / "targets" / "target_qa.json",
    )
    specs, skipped_models = make_candidate_regressor_specs(
        random_state=config.random_seed,
        include_catboost=config.include_catboost,
    )
    if smoke:
        smoke_models = {
            "dummy_mean",
            "dummy_median",
            "ridge",
            "gower_knn",
            "extra_trees",
        }
        for name in sorted(set(specs) - smoke_models):
            skipped_models[name] = "not evaluated in structural smoke subset"
        specs = {name: spec for name, spec in specs.items() if name in smoke_models}
    elif config.include_catboost and "catboost" in skipped_models:
        raise RuntimeError(
            "Full performance optimization requires the frozen CatBoost candidate, but "
            f"it is unavailable: {skipped_models['catboost']}"
        )

    (destination / "registry").mkdir(exist_ok=True)
    (destination / "analyses").mkdir(exist_ok=True)
    (destination / "final_pipeline").mkdir(exist_ok=True)
    feature_recipe_registry().to_csv(
        destination / "registry" / "feature_recipes.csv", index=False
    )
    model_registry, _ = available_model_registry(
        random_state=config.random_seed,
        include_catboost=config.include_catboost,
    )
    model_registry = model_registry.loc[model_registry["model"].isin(specs)].copy()
    model_registry.to_csv(destination / "registry" / "model_search_space.csv", index=False)
    write_json(destination / "registry" / "skipped_optional_models.json", skipped_models)
    write_json(destination / "provenance" / "optimization_input_binding.json", binding)
    write_json(
        destination / "provenance" / "optimization_config.json", asdict(config)
    )

    analyses_to_run = ANALYSES
    results: dict[str, AnalysisResult] = {}
    catalogs: dict[str, tuple[CandidateDefinition, ...]] = {}
    target_draw_qa: dict[str, dict[str, object]] = {}
    for analysis in analyses_to_run:
        print(f"[optimize] starting analysis {analysis.analysis_id}", flush=True)
        result, candidates = run_analysis(
            frame,
            analysis,
            destination / "analyses" / analysis.analysis_id,
            config=config,
            specs=specs,
        )
        results[analysis.analysis_id] = result
        catalogs[analysis.analysis_id] = candidates
        target_draw_qa[analysis.analysis_id] = _score_posterior_draws(
            result,
            canonical_dir / "posterior_slope_draws.csv.gz",
            destination / "analyses" / analysis.analysis_id / "target_uncertainty",
            max_draws=config.target_draw_repetitions,
            random_seed=config.random_seed,
        )

    primary = results["primary_alkaline"]
    decision = _decision_from_primary(primary)
    if smoke:
        decision.update(
            {
                "classification": "Structural smoke pass",
                "classification_scope": "not_applicable_structural_smoke_only",
                "decision_rule_applied": False,
                "scientific_classification_suppressed": True,
                "smoke_metrics_role": "structural diagnostics only",
            }
        )
    primary_frame = frame.loc[
        frame["electrolyte_scope"].astype(str).eq("alkaline")
    ].reset_index(drop=True)
    print("[optimize] locking full-data primary pipeline", flush=True)
    final_spec = _fit_final_primary_pipeline(
        primary_frame,
        catalogs["primary_alkaline"],
        specs,
        destination / "final_pipeline",
        config=config,
        nested_metrics=primary.metrics,
    )
    write_json(destination / "performance_decision.json", decision)

    combined_metrics = pd.concat(
        [result.metrics for result in results.values()], ignore_index=True
    )
    combined_metrics.to_csv(destination / "all_analysis_metrics.csv", index=False)
    combined_frequencies = pd.concat(
        [
            result.selection_frequency.assign(analysis_id=analysis_id)
            for analysis_id, result in results.items()
        ],
        ignore_index=True,
    )
    combined_frequencies.to_csv(
        destination / "all_selection_frequencies.csv", index=False
    )

    formal_tree_after = (
        directory_tree_summary(formal_run)["tree_sha256"]
        if formal_run.is_dir()
        else None
    )
    formal_run_unchanged = bool(formal_tree_before == formal_tree_after)

    coverage_checks: dict[str, object] = {}
    for analysis_id, result in results.items():
        counts = result.oof_aggregated["n_repeated_oof"]
        catalog_frame = pd.DataFrame(
            [candidate.row() for candidate in catalogs[analysis_id]]
        )
        search_health = _candidate_search_health(
            result.candidate_scores, catalog_frame
        )
        coverage_checks[analysis_id] = {
            "n_records_joint_selector": int(
                result.oof_aggregated.loc[
                    result.oof_aggregated["prediction_role"].eq("joint_selector")
                ]["record_id"].nunique()
            ),
            "expected_repeated_oof": config.outer_repeats,
            "all_roles_have_expected_repeated_oof": bool(
                counts.eq(config.outer_repeats).all()
            ),
            "candidate_catalog_reused_in_every_outer_fold": bool(
                result.candidate_scores.groupby(["repeat", "fold"])["candidate_id"]
                .nunique()
                .eq(len(catalogs[analysis_id]))
                .all()
            ),
            "candidate_search_health": search_health,
        }
    posterior_draw_qa_pass = bool(target_draw_qa) and all(
        item.get("status") == "pass" for item in target_draw_qa.values()
    )
    qa_pass = all(
        check["all_roles_have_expected_repeated_oof"]
        and check["candidate_catalog_reused_in_every_outer_fold"]
        and check["candidate_search_health"]["status"] == "pass"
        for check in coverage_checks.values()
    ) and formal_run_unchanged and posterior_draw_qa_pass
    qa = {
        "status": "pass" if qa_pass else "fail",
        "smoke": smoke,
        "formal_scientific_estimate": False,
        "internal_nested_oof_estimate_available": not smoke,
        "prospective_independent_confirmation": False,
        "posterior_draw_score_qa": target_draw_qa,
        "all_posterior_draw_score_qa_pass": posterior_draw_qa_pass,
        "input_binding": binding,
        "coverage": coverage_checks,
        "forbidden_predictors_checked": True,
        "target_or_interval_as_predictor": False,
        "best_available_dielectric_estimates_in_primary_property_recipes": binding[
            "n_best_available_dielectric_estimates"
        ],
        "dielectric_estimate_flag_role": "sensitivity_only",
        "connected_group_relation_columns_present": binding[
            "connected_group_relation_columns_present"
        ],
        "connected_group_relation_columns_absent": binding[
            "connected_group_relation_columns_absent"
        ],
        "verified_lab_group_id_available": binding[
            "verified_lab_group_id_available"
        ],
        "definitive_lab_held_out_validation": False,
        "fold_local_steps": [
            "numeric imputation",
            "categorical missing handling and rare pooling",
            "ILR zero replacement",
            "property PCA",
            "scaling and one-hot encoding",
            "near-zero-variance removal",
            "target transform standardization",
            "feature/model/hyperparameter selection",
        ],
        "current_97_modified": not formal_run_unchanged,
        "current_97_tree_sha256_before": formal_tree_before,
        "current_97_tree_sha256_after": formal_tree_after,
        "development_internal_only": True,
    }
    write_json(destination / "optimization_qa.json", qa)
    if not qa_pass:
        raise RuntimeError(f"Performance optimization QA failed: {qa}")
    _write_summary_report(
        destination / "performance_optimization_report.md",
        decision,
        primary,
        final_spec,
        smoke=smoke,
    )
    return _jsonable(
        {
            "status": "complete_smoke" if smoke else "complete",
            "run_dir": destination,
            "target": TARGET_NAME,
            "primary_records": 90,
            "all_records": 97,
            "analyses": list(results),
            "decision": decision,
            "final_pipeline": final_spec["pipeline"],
            "report": destination / "performance_optimization_report.md",
            "qa": destination / "optimization_qa.json",
        }
    )
