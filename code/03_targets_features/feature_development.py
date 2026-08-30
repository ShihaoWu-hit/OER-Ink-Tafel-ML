"""Fixed-RBF feature engineering for the 97-record Bayesian local-line target.

Only the feature recipe is selected inside grouped inner CV.  The target
coordinate and RBF-SVR hyperparameters are immutable.  The registered
``current_97`` analysis and frozen canonical inputs are read-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error

from .evaluate import (
    DEFAULT_RELATION_COLUMNS,
    assert_no_relational_leakage,
    make_outer_split_manifest,
)
from .feature_development_components import (
    FEATURE_DEVELOPMENT_RECIPES,
    PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES,
    RAW_SOLVENT_PREDICTORS,
    SENSITIVITY_FEATURE_DEVELOPMENT_RECIPE_NAMES,
    feature_definition_registry,
    feature_development_recipe_registry,
    make_fixed_feature_development_pipeline,
    normalize_feature_development_columns,
    required_feature_development_source_columns,
)
from .features import is_target_like_column, validate_no_target_leakage
from .performance_optimization import (
    TARGET_COLUMN,
    TARGET_INTERVAL_HIGH,
    TARGET_INTERVAL_LOW,
    TARGET_NAME,
    _make_inner_splits,
    _prepare_canonical_inputs,
    extended_regression_metrics,
)
from .provenance import (
    artifact_manifest,
    directory_tree_summary,
    sha256_file,
    verify_artifact_manifest,
    write_json,
)
from .uncertainty import score_functional_target_draws


RBF_PARAMETERS = {"kernel": "rbf", "C": 1.0, "epsilon": 0.05, "gamma": 0.1}
PREDICTION_ROLE = "fixed_rbf_recipe_selector"
DUMMY_ROLE = "inner_selected_dummy"
BAYESIAN_RIDGE_ROLE = "bayesian_ridge_sanity"


@dataclass(frozen=True)
class FeatureDevelopmentConfig:
    outer_folds: int = 5
    outer_repeats: int = 10
    inner_folds: int = 4
    outer_seeds: tuple[int, ...] = (42, 142, 242, 342, 442, 542, 642, 742, 842, 942)
    random_seed: int = 42
    tie_tolerance_mV_dec: float = 0.5
    bootstrap_replicates: int = 2000
    target_draw_repetitions: int = 100
    stable_selection_count: int = 30
    n_jobs: int = 1
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.outer_folds < 2 or self.inner_folds < 2 or self.outer_repeats < 1:
            raise ValueError("Grouped nested CV requires positive repeats and at least two folds")
        if len(self.outer_seeds) != self.outer_repeats:
            raise ValueError("outer_seeds must contain one seed per repeat")
        if self.tie_tolerance_mV_dec < 0 or self.n_jobs < 1:
            raise ValueError("Invalid feature-development configuration")


@dataclass(frozen=True)
class AnalysisSpec:
    analysis_id: str
    scope: str
    include_sensitivity_recipes: bool = False


@dataclass
class AnalysisResult:
    spec: AnalysisSpec
    metrics: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    outer_selections: pd.DataFrame
    oof_predictions: pd.DataFrame
    oof_aggregated: pd.DataFrame
    recipe_frequency: pd.DataFrame
    block_frequency: pd.DataFrame


ANALYSES = (
    AnalysisSpec("primary_all", "all", False),
    AnalysisSpec("sensitivity_alkaline", "alkaline", False),
    AnalysisSpec("sensitivity_feature_recipes_all", "all", True),
)


def _settings(config: Any) -> Mapping[str, Any]:
    if hasattr(config, "section"):
        return config.section("feature_development")
    if isinstance(config, Mapping):
        value = config.get("feature_development", config)
        return value if isinstance(value, Mapping) else {}
    return {}


def feature_development_config(config: Any, *, smoke: bool, n_jobs: int) -> FeatureDevelopmentConfig:
    values = _settings(config)
    defaults = FeatureDevelopmentConfig()
    seeds = tuple(int(x) for x in values.get("outer_seeds", defaults.outer_seeds))
    resolved = FeatureDevelopmentConfig(
        outer_folds=int(values.get("outer_folds", defaults.outer_folds)),
        outer_repeats=int(values.get("outer_repeats", len(seeds))),
        inner_folds=int(values.get("inner_folds", defaults.inner_folds)),
        outer_seeds=seeds,
        random_seed=int(values.get("random_seed", defaults.random_seed)),
        tie_tolerance_mV_dec=float(values.get("near_tie_tolerance_mV_dec", values.get("tie_tolerance_mV_dec", 0.5))),
        bootstrap_replicates=int(values.get("group_bootstrap_repetitions", defaults.bootstrap_replicates)),
        target_draw_repetitions=int(values.get("target_draw_repetitions", defaults.target_draw_repetitions)),
        stable_selection_count=int(
            values.get(
                "stable_selection_minimum_outer_folds",
                values.get("stable_selection_count", defaults.stable_selection_count),
            )
        ),
        n_jobs=int(n_jobs),
    )
    if smoke:
        return replace(
            resolved, outer_folds=3, outer_repeats=1, inner_folds=2,
            outer_seeds=(42,), bootstrap_replicates=100,
            target_draw_repetitions=5, smoke=True,
        )
    return resolved


def choose_feature_recipe(scores: pd.DataFrame, *, tie_tolerance_mV_dec: float) -> tuple[str, dict[str, object]]:
    eligible = scores.loc[scores["status"].eq("complete")].copy()
    eligible = eligible.loc[pd.to_numeric(eligible["inner_mae_mV_dec"], errors="coerce").notna()]
    if eligible.empty:
        raise RuntimeError("No feature recipe completed grouped inner CV")
    minimum = float(eligible["inner_mae_mV_dec"].min())
    tied = eligible.loc[eligible["inner_mae_mV_dec"] <= minimum + tie_tolerance_mV_dec].copy()
    tied = tied.sort_values(["feature_complexity", "complexity_rank", "inner_mae_mV_dec", "recipe"])
    selected = str(tied.iloc[0]["recipe"])
    raw = str(eligible.sort_values(["inner_mae_mV_dec", "recipe"]).iloc[0]["recipe"])
    return selected, {
        "selected_recipe": selected,
        "raw_minimum_recipe": raw,
        "raw_minimum_inner_mae_mV_dec": minimum,
        "selected_inner_mae_mV_dec": float(tied.iloc[0]["inner_mae_mV_dec"]),
        "tie_tolerance_mV_dec": float(tie_tolerance_mV_dec),
        "n_near_tied_recipes": int(len(tied)),
        "simplicity_tie_break_applied": selected != raw,
    }


def _fit_predict(estimator: BaseEstimator, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> tuple[BaseEstimator, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        estimator.fit(X_train, y_train)
        prediction = np.asarray(estimator.predict(X_test), dtype=float).reshape(-1)
    if len(prediction) != len(X_test) or not np.isfinite(prediction).all():
        raise ValueError("Pipeline returned invalid predictions")
    return estimator, prediction


def _evaluate_recipe(recipe: str, X: pd.DataFrame, y: np.ndarray, splits: Sequence[tuple[np.ndarray, np.ndarray]], metadata: Mapping[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    predicted = np.full(len(y), np.nan)
    folds: list[dict[str, object]] = []
    try:
        for fold, (train, validation) in enumerate(splits):
            _, values = _fit_predict(make_fixed_feature_development_pipeline(recipe, model="rbf_svr"), X.iloc[train], y[train], X.iloc[validation])
            predicted[validation] = values
            folds.append({"recipe": recipe, "inner_fold": fold, "n_train": len(train), "n_validation": len(validation), "mae_mV_dec": mean_absolute_error(y[validation], values), "status": "complete"})
        metrics = extended_regression_metrics(y, predicted)
        return {"recipe": recipe, **metadata, "inner_mae_mV_dec": metrics["mae_mV_dec"], "inner_rmse_mV_dec": metrics["rmse_mV_dec"], "inner_r2": metrics["r2"], "status": "complete", "error": ""}, folds
    except Exception as error:
        return {"recipe": recipe, **metadata, "inner_mae_mV_dec": np.nan, "inner_rmse_mV_dec": np.nan, "inner_r2": np.nan, "status": "failed", "error": repr(error)[:1000]}, folds


def _evaluate_recipes(recipe_names: Sequence[str], X: pd.DataFrame, y: np.ndarray, splits: Sequence[tuple[np.ndarray, np.ndarray]], *, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = feature_development_recipe_registry().set_index("name")
    jobs = [(name, {"complexity_rank": int(registry.loc[name, "complexity_rank"]), "feature_complexity": int(registry.loc[name, "feature_complexity"])}) for name in recipe_names]
    if n_jobs == 1:
        results = [_evaluate_recipe(name, X, y, splits, meta) for name, meta in jobs]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_evaluate_recipe)(name, X, y, splits, meta) for name, meta in jobs)
    return pd.DataFrame([x[0] for x in results]), pd.DataFrame([row for x in results for row in x[1]])


def _dummy_inner(X: pd.DataFrame, y: np.ndarray, splits: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[str, pd.DataFrame, dict[str, object]]:
    rows = []
    for model in ("dummy_mean", "dummy_median"):
        prediction = np.full(len(y), np.nan)
        for train, validation in splits:
            _, values = _fit_predict(make_fixed_feature_development_pipeline("context_core", model=model), X.iloc[train], y[train], X.iloc[validation])
            prediction[validation] = values
        rows.append({"dummy_model": model, "inner_mae_mV_dec": mean_absolute_error(y, prediction)})
    frame = pd.DataFrame(rows).sort_values(["inner_mae_mV_dec", "dummy_model"])
    winner = str(frame.iloc[0]["dummy_model"])
    return winner, frame, frame.iloc[0].to_dict()


def _aggregate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    keys = ["analysis_id", "prediction_role", "record_id", "cv_group_id"]
    out = oof.groupby(keys, as_index=False).agg(
        observed_mV_dec=("observed_mV_dec", "first"),
        predicted_mV_dec=("predicted_mV_dec", "mean"),
        target_interval_low_mV_dec=("target_interval_low_mV_dec", "first"),
        target_interval_high_mV_dec=("target_interval_high_mV_dec", "first"),
        n_repeated_oof=("repeat", "count"),
    )
    out["absolute_error_mV_dec"] = np.abs(out["predicted_mV_dec"] - out["observed_mV_dec"])
    return out


def _metric_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (analysis_id, role), group in aggregated.groupby(["analysis_id", "prediction_role"]):
        rows.append({"analysis_id": analysis_id, "prediction_role": role, "n_records": len(group), **extended_regression_metrics(group["observed_mV_dec"].to_numpy(float), group["predicted_mV_dec"].to_numpy(float))})
    return pd.DataFrame(rows)


def _bootstrap(aggregated: pd.DataFrame, *, replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replicates <= 0:
        return pd.DataFrame(), pd.DataFrame()
    wide = aggregated.pivot(index=["record_id", "cv_group_id", "observed_mV_dec"], columns="prediction_role", values="predicted_mV_dec").reset_index()
    groups = wide["cv_group_id"].astype(str).unique()
    positions = {group: np.flatnonzero(wide["cv_group_id"].astype(str).to_numpy() == group) for group in groups}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        selected = np.concatenate([positions[group] for group in sampled])
        observed = wide.iloc[selected]["observed_mV_dec"].to_numpy(float)
        rbf = wide.iloc[selected][PREDICTION_ROLE].to_numpy(float)
        dummy = wide.iloc[selected][DUMMY_ROLE].to_numpy(float)
        ridge = wide.iloc[selected][BAYESIAN_RIDGE_ROLE].to_numpy(float)
        metrics = extended_regression_metrics(observed, rbf)
        dummy_mae = mean_absolute_error(observed, dummy)
        ridge_mae = mean_absolute_error(observed, ridge)
        rows.append({"bootstrap_replicate": replicate, "rbf_mae_mV_dec": metrics["mae_mV_dec"], "rbf_rmse_mV_dec": metrics["rmse_mV_dec"], "rbf_median_ae_mV_dec": metrics["median_ae_mV_dec"], "rbf_tail_mae_mV_dec": metrics["tail_mae_mV_dec"], "rbf_r2": metrics["r2"], "rbf_spearman": metrics["spearman"], "dummy_mae_mV_dec": dummy_mae, "bayesian_ridge_mae_mV_dec": ridge_mae, "delta_mae_rbf_minus_dummy_mV_dec": metrics["mae_mV_dec"] - dummy_mae, "delta_mae_ridge_minus_dummy_mV_dec": ridge_mae - dummy_mae})
    draws = pd.DataFrame(rows)
    summary = []
    for metric in draws.columns.drop("bootstrap_replicate"):
        values = pd.to_numeric(draws[metric], errors="coerce").dropna()
        summary.append({"metric": metric, "n_bootstrap": len(values), "median": values.median(), "q025": values.quantile(.025), "q975": values.quantile(.975), "probability_below_zero": np.mean(values < 0)})
    return draws, pd.DataFrame(summary)


def _frequencies(selections: pd.DataFrame, *, stable_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = len(selections)
    recipe = selections["recipe"].value_counts().rename_axis("recipe").reset_index(name="count")
    recipe["frequency"] = recipe["count"] / total
    registry = feature_development_recipe_registry().set_index("name")
    rows = []
    for column in [value for value in registry.columns if value.startswith("block_")]:
        count = int(sum(bool(registry.loc[name, column]) for name in selections["recipe"]))
        rows.append({"feature_block": column.removeprefix("block_"), "count": count, "total_outer_folds": total, "frequency": count / total, "stable_selection_threshold_count": stable_count, "stable_selection": bool(total == 50 and count >= stable_count)})
    return recipe, pd.DataFrame(rows)


def _run_analysis(frame: pd.DataFrame, spec: AnalysisSpec, output: Path, *, settings: FeatureDevelopmentConfig) -> AnalysisResult:
    scoped = frame if spec.scope == "all" else frame.loc[frame["electrolyte_scope"].astype(str).eq("alkaline")]
    scoped = scoped.reset_index(drop=True)
    recipes = tuple(PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES)
    if spec.include_sensitivity_recipes:
        recipes += tuple(SENSITIVITY_FEATURE_DEVELOPMENT_RECIPE_NAMES)
    sources = tuple(dict.fromkeys(column for recipe in recipes for column in required_feature_development_source_columns(recipe)))
    validate_no_target_leakage(sources)
    forbidden = [column for column in sources if is_target_like_column(column)]
    if forbidden:
        raise AssertionError(f"Forbidden predictors: {forbidden}")
    missing = sorted(set(sources) - set(scoped))
    if missing:
        raise KeyError(f"Feature-development sources are missing: {missing}")
    X = scoped.loc[:, list(sources)].copy()
    y = pd.to_numeric(scoped[TARGET_COLUMN], errors="raise").to_numpy(float)
    manifest = make_outer_split_manifest(scoped, group_col="cv_group_id", record_id_col="record_id", folds=settings.outer_folds, repeats=settings.outer_repeats, repeat_seeds=settings.outer_seeds, relation_columns=DEFAULT_RELATION_COLUMNS)
    score_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    dummy_score_frames: list[pd.DataFrame] = []
    inner_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    cells = list(manifest.groupby(["repeat", "fold"], sort=True))
    for cell_number, ((repeat, fold), test_manifest) in enumerate(cells, 1):
        print(f"[feature-develop:{spec.analysis_id}] outer fold {cell_number}/{len(cells)}", flush=True)
        test_pos = np.sort(test_manifest["row_position"].to_numpy(int))
        mask = np.zeros(len(scoped), dtype=bool); mask[test_pos] = True
        train_pos = np.flatnonzero(~mask)
        outer_train = scoped.iloc[train_pos].reset_index(drop=True)
        outer_test = scoped.iloc[test_pos].reset_index(drop=True)
        assert_no_relational_leakage(outer_train, outer_test, group_col="cv_group_id", relation_columns=DEFAULT_RELATION_COLUMNS)
        X_train, X_test = X.iloc[train_pos].reset_index(drop=True), X.iloc[test_pos].reset_index(drop=True)
        y_train, y_test = y[train_pos], y[test_pos]
        inner_seed = int(settings.outer_seeds[int(repeat)] + int(fold) * 10003)
        splits, assignments = _make_inner_splits(outer_train, group_col="cv_group_id", inner_folds=settings.inner_folds, seed=inner_seed, relation_columns=DEFAULT_RELATION_COLUMNS)
        scores, fold_scores = _evaluate_recipes(recipes, X_train, y_train, splits, n_jobs=settings.n_jobs)
        selected, details = choose_feature_recipe(scores, tie_tolerance_mV_dec=settings.tie_tolerance_mV_dec)
        dummy_model, dummy_scores, dummy_details = _dummy_inner(X_train, y_train, splits)
        scores.insert(0, "analysis_id", spec.analysis_id); scores.insert(1, "repeat", repeat); scores.insert(2, "fold", fold)
        score_frames.append(scores)
        fold_scores.insert(0, "analysis_id", spec.analysis_id); fold_scores.insert(1, "repeat", repeat); fold_scores.insert(2, "fold", fold)
        fold_frames.append(fold_scores)
        dummy_scores.insert(0, "analysis_id", spec.analysis_id)
        dummy_scores.insert(1, "repeat", repeat)
        dummy_scores.insert(2, "fold", fold)
        dummy_score_frames.append(dummy_scores)
        for position, inner_fold in enumerate(assignments):
            inner_rows.append({"analysis_id": spec.analysis_id, "outer_repeat": repeat, "outer_fold": fold, "inner_fold": inner_fold, "record_id": str(outer_train.iloc[position]["record_id"]), "cv_group_id": str(outer_train.iloc[position]["cv_group_id"]), "inner_seed": inner_seed})
        _, rbf_prediction = _fit_predict(make_fixed_feature_development_pipeline(selected, model="rbf_svr"), X_train, y_train, X_test)
        _, dummy_prediction = _fit_predict(make_fixed_feature_development_pipeline("context_core", model=dummy_model), X_train, y_train, X_test)
        _, ridge_prediction = _fit_predict(make_fixed_feature_development_pipeline(selected, model="bayesian_ridge"), X_train, y_train, X_test)
        selected_registry = feature_development_recipe_registry().set_index("name").loc[selected]
        selection_rows.append({"analysis_id": spec.analysis_id, "repeat": repeat, "fold": fold, "recipe": selected, **details, "selected_feature_complexity": int(selected_registry["feature_complexity"]), "selected_complexity_rank": int(selected_registry["complexity_rank"]), "dummy_model": dummy_model, "dummy_inner_mae_mV_dec": dummy_details["inner_mae_mV_dec"], "n_outer_train": len(train_pos), "n_outer_test": len(test_pos), "sensitivity_recipes_eligible": spec.include_sensitivity_recipes})
        for role, model, recipe_name, prediction in ((PREDICTION_ROLE, "rbf_svr", selected, rbf_prediction), (DUMMY_ROLE, dummy_model, "dummy", dummy_prediction), (BAYESIAN_RIDGE_ROLE, "bayesian_ridge", selected, ridge_prediction)):
            for local, record in outer_test.iterrows():
                oof_rows.append({"analysis_id": spec.analysis_id, "scope": spec.scope, "prediction_role": role, "repeat": repeat, "fold": fold, "record_id": str(record["record_id"]), "sample_record": int(record["record_id"]), "curve_id": str(record["curve_id"]), "cv_group_id": str(record["cv_group_id"]), "observed_mV_dec": y_test[local], "predicted_mV_dec": prediction[local], "absolute_error_mV_dec": abs(prediction[local] - y_test[local]), "target_interval_low_mV_dec": record[TARGET_INTERVAL_LOW], "target_interval_high_mV_dec": record[TARGET_INTERVAL_HIGH], "recipe": recipe_name, "model": model, "target_transform": "log10", "target_standardized": False})
    selections = pd.DataFrame(selection_rows)
    oof = pd.DataFrame(oof_rows)
    aggregated = _aggregate_oof(oof)
    metrics = _metric_table(aggregated)
    draws, summary = _bootstrap(aggregated, replicates=settings.bootstrap_replicates, seed=settings.random_seed)
    recipe_frequency, block_frequency = _frequencies(selections, stable_count=settings.stable_selection_count)
    output.mkdir(parents=True, exist_ok=True)
    feature_development_recipe_registry().loc[lambda value: value["name"].isin(recipes)].to_csv(output / "recipe_catalog.csv", index=False)
    manifest.to_csv(output / "outer_split_manifest.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output / "inner_split_manifest.csv", index=False)
    pd.concat(score_frames, ignore_index=True).to_csv(output / "inner_recipe_scores.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(output / "inner_fold_scores.csv", index=False)
    pd.concat(dummy_score_frames, ignore_index=True).to_csv(
        output / "inner_dummy_scores.csv", index=False
    )
    selections.to_csv(output / "outer_selected_recipes.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    aggregated.to_csv(output / "oof_aggregated.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    draws.to_csv(output / "connected_group_bootstrap.csv", index=False)
    summary.to_csv(output / "connected_group_bootstrap_summary.csv", index=False)
    recipe_frequency.to_csv(output / "recipe_selection_frequency.csv", index=False)
    block_frequency.to_csv(output / "feature_block_selection_frequency.csv", index=False)
    write_json(output / "analysis_definition.json", {**asdict(spec), "target": TARGET_NAME, "target_transform": "log10", "target_standardization": False, "fixed_rbf_parameters": RBF_PARAMETERS, "n_records": len(scoped), "n_recipes": len(recipes)})
    return AnalysisResult(spec, metrics, summary, selections, oof, aggregated, recipe_frequency, block_frequency)


def _decision(primary: AnalysisResult, *, smoke: bool) -> dict[str, object]:
    metrics = primary.metrics.set_index("prediction_role")
    rbf = metrics.loc[PREDICTION_ROLE]
    dummy = metrics.loc[DUMMY_ROLE]
    delta = float(rbf["mae_mV_dec"] - dummy["mae_mV_dec"])
    row = primary.bootstrap_summary.set_index("metric").loc["delta_mae_rbf_minus_dummy_mV_dec"]
    improves = delta < 0
    supports = float(row["q975"]) < 0
    label = "Internally supported improvement" if improves and supports and float(rbf["r2"]) > 0 else ("Promising only" if improves else "No useful predictive model")
    if smoke:
        label = "Structural smoke pass"
    return {"classification": label, "decision_rule_applied": not smoke, "rbf_mae_mV_dec": float(rbf["mae_mV_dec"]), "dummy_mae_mV_dec": float(dummy["mae_mV_dec"]), "delta_mae_rbf_minus_dummy_mV_dec": delta, "delta_bootstrap_q025_mV_dec": float(row["q025"]), "delta_bootstrap_q975_mV_dec": float(row["q975"]), "bootstrap_probability_delta_below_zero": float(row["probability_below_zero"]), "rbf_r2": float(rbf["r2"]), "primary_population": "all_97_electrolytes"}


def _fit_final(frame: pd.DataFrame, output: Path, *, settings: FeatureDevelopmentConfig, nested_metrics: pd.DataFrame) -> dict[str, object]:
    recipes = tuple(PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES)
    sources = tuple(dict.fromkeys(column for recipe in recipes for column in required_feature_development_source_columns(recipe)))
    X = frame.loc[:, list(sources)].reset_index(drop=True)
    y = frame[TARGET_COLUMN].to_numpy(float)
    splits, assignments = _make_inner_splits(frame.reset_index(drop=True), group_col="cv_group_id", inner_folds=settings.inner_folds, seed=settings.random_seed, relation_columns=DEFAULT_RELATION_COLUMNS)
    scores, folds = _evaluate_recipes(recipes, X, y, splits, n_jobs=settings.n_jobs)
    selected, details = choose_feature_recipe(scores, tie_tolerance_mV_dec=settings.tie_tolerance_mV_dec)
    estimator, _ = _fit_predict(make_fixed_feature_development_pipeline(selected, model="rbf_svr"), X, y, X)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "full_data_inner_recipe_scores.csv", index=False)
    folds.to_csv(output / "full_data_inner_fold_scores.csv", index=False)
    pd.DataFrame({"record_id": frame["record_id"].astype(str), "cv_group_id": frame["cv_group_id"].astype(str), "inner_fold": assignments}).to_csv(output / "full_data_inner_split_manifest.csv", index=False)
    model_path = output / "fitted_final_pipeline.joblib"
    joblib.dump(estimator, model_path)
    feature_builder = estimator.named_steps["features"]
    target_wrapper = estimator.named_steps["target"]
    fitted_rbf_params = {
        name: target_wrapper.estimator.get_params()[name]
        for name in ("kernel", "C", "epsilon", "gamma")
    }
    engineered = list(map(str, feature_builder.get_feature_names_out()))
    interaction_encoding = None
    if hasattr(feature_builder, "catalyst_interaction_levels_"):
        interaction_encoding = {
            "levels": list(feature_builder.catalyst_interaction_levels_),
            "reference": feature_builder.catalyst_interaction_reference_,
            "ionomer_ratio_zero_replacement": feature_builder.ionomer_ratio_delta_,
            "ionomer_ratio_missing_imputation": feature_builder.ionomer_ratio_median_,
        }
    spec = {
        "target": TARGET_NAME,
        "target_unit": "mV dec^-1",
        "target_transform": "log10",
        "target_standardization": False,
        "population": "all 97 canonical records",
        "selection": details,
        "recipe": selected,
        "model": "RBF-SVR",
        "model_parameters": fitted_rbf_params,
        "required_source_columns": list(
            required_feature_development_source_columns(selected)
        ),
        "fitted_engineered_features": engineered,
        "fitted_interaction_encoding": interaction_encoding,
        "n_training_records": int(len(frame)),
        "nested_outer_oof_metrics": nested_metrics.to_dict(orient="records"),
        "performance_source": (
            "primary_all record-aggregated repeated nested outer OOF selector; "
            "not training fit and not a post-hoc fixed-recipe score"
        ),
        "serialized_pipeline": str(model_path),
        "serialized_pipeline_sha256": sha256_file(model_path),
        "artifact_role": (
            "structural_smoke_test_only_not_locked"
            if settings.smoke
            else "final_full_data_pipeline_locked_by_grouped_inner_cv"
        ),
    }
    write_json(output / "exact_final_pipeline.json", spec)
    return spec


def _posterior_scores(result: AnalysisResult, draws_path: Path, output: Path, *, settings: FeatureDevelopmentConfig) -> dict[str, object]:
    table = result.oof_aggregated.copy()
    table["target"] = TARGET_NAME; table["scope"] = result.spec.analysis_id; table["panel"] = "fixed_rbf_feature_development"; table["model"] = table["prediction_role"]
    table["record_id"] = pd.to_numeric(table["record_id"], errors="raise").astype(int)
    score_functional_target_draws(table, draws_path, output, target_name=TARGET_NAME, max_draws=settings.target_draw_repetitions, random_seed=settings.random_seed)
    return json.loads((output / "target_draw_qa.json").read_text(encoding="utf-8"))


def _write_report(
    path: Path,
    decision: Mapping[str, object],
    results: Mapping[str, AnalysisResult],
    final: Mapping[str, object],
    *,
    smoke: bool,
) -> None:
    primary = results["primary_all"]
    metrics = primary.metrics.set_index("prediction_role")
    rbf = metrics.loc[PREDICTION_ROLE]
    dummy = metrics.loc[DUMMY_ROLE]
    ridge = metrics.loc[BAYESIAN_RIDGE_ROLE]
    stable_blocks = primary.block_frequency.loc[
        primary.block_frequency["stable_selection"], "feature_block"
    ].astype(str).tolist()
    total_outer = int(len(primary.outer_selections))
    top_recipes = ", ".join(
        f"`{row.recipe}` ({int(row.count)}/{total_outer})"
        for row in primary.recipe_frequency.head(3).itertuples(index=False)
    )
    alkaline = results["sensitivity_alkaline"].metrics.set_index(
        "prediction_role"
    ).loc[PREDICTION_ROLE]
    expanded = results["sensitivity_feature_recipes_all"].metrics.set_index(
        "prediction_role"
    ).loc[PREDICTION_ROLE]
    pipeline_heading = (
        "## Serialized smoke pipeline (not locked)"
        if smoke
        else "## Final full-97 pipeline"
    )
    lines = [
        "# Fixed RBF-SVR feature development for Bayesian kinetic Tafel slope",
        "",
        f"Status: **{decision['classification']}**",
        "",
        *(
            [
                "**Smoke mode is structural validation only; its numerical scores are not a performance result.**",
                "",
            ]
            if smoke
            else []
        ),
        "All 97 records are the primary population. Ink-property descriptors are included because of their scientifically plausible influence on ink film formation and measured Tafel slope.",
        "",
        "## Primary 97-record nested OOF result",
        "",
        f"- Fixed-RBF recipe selector: MAE {rbf['mae_mV_dec']:.3f}, RMSE {rbf['rmse_mV_dec']:.3f}, median AE {rbf['median_ae_mV_dec']:.3f} mV dec⁻¹.",
        f"- R² {rbf['r2']:.3f}; Spearman {rbf['spearman']:.3f}; high-tail MAE {rbf['tail_mae_mV_dec']:.3f} mV dec⁻¹.",
        f"- Inner-selected dummy MAE: {dummy['mae_mV_dec']:.3f} mV dec⁻¹.",
        f"- ΔMAE (fixed RBF − dummy): {decision['delta_mae_rbf_minus_dummy_mV_dec']:+.3f} mV dec⁻¹; connected-group bootstrap 95% interval [{decision['delta_bootstrap_q025_mV_dec']:+.3f}, {decision['delta_bootstrap_q975_mV_dec']:+.3f}].",
        f"- Same-recipe Bayesian Ridge sanity comparator MAE: {ridge['mae_mV_dec']:.3f} mV dec⁻¹.",
        "",
        "## Feature-selection result",
        "",
        f"- Most frequent outer winners: {top_recipes or 'none'}.",
        (
            "- Stable blocks (at least 30/50 outer winners): "
            + ", ".join(f"`{name}`" for name in stable_blocks)
            + "."
            if stable_blocks
            else (
                "- Stable-block status is not evaluated in smoke mode."
                if smoke
                else "- No block reached the 30/50 stable-selection threshold."
            )
        ),
        "",
        pipeline_heading,
        "",
        f"- Recipe selected by grouped inner CV on all 97 records: `{final['recipe']}`.",
        "- Model: RBF-SVR with C=1, epsilon=0.05 and gamma=0.1; y is log10(slope) with no additional target standardization.",
        "- Quoted performance comes only from the nested outer OOF selector above, not the full-data training fit.",
        "",
        "## Sensitivities",
        "",
        f"- 90-record alkaline fixed-RBF selector MAE: {alkaline['mae_mV_dec']:.3f} mV dec⁻¹.",
        f"- 97-record expanded-recipe sensitivity selector MAE: {expanded['mae_mV_dec']:.3f} mV dec⁻¹.",
        "- Neither sensitivity can choose the final pipeline.",
        "",
        "## Validation boundary",
        "",
        (
            "All feature learning and recipe selection are fitted only on training folds. Smoke metrics are structural diagnostics only."
            if smoke
            else "All feature learning and recipe selection are fitted only on training folds. This is the repeated connected-group outer-OOF internal-development estimate."
        ),
        *(
            [
                "",
                "The fixed-RBF performance gate was not passed, so this workflow does not expand the black-box model search.",
            ]
            if not smoke
            and decision["classification"] == "No useful predictive model"
            else []
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _forbidden_predictor_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule": "raw_solvent_1_2_identity_and_volume",
                "match_type": "exact_allowlist_denial",
                "values": "|".join(sorted(RAW_SOLVENT_PREDICTORS)),
                "reason": "raw Solvent 1/2 identity and volumes are excluded",
            },
            {
                "rule": "target_posterior_curve_fit_and_selected_model",
                "match_type": "semantic_and_prefix_denial",
                "values": "b_*|target_*|posterior_*|slope_*|fit_*|selected_model*|curve-coordinate fields|overpotential fields",
                "reason": "response-derived and curve-fit information cannot be predictors",
            },
            {
                "rule": "identifiers_groups_and_provenance",
                "match_type": "metadata_denial",
                "values": "record_id|curve_id|publication_id|cv_group_id|recipe_hash|evidence hashes",
                "reason": "identifiers and grouping/provenance fields are split or audit metadata only",
            },
        ]
    )


def run_feature_development(config: Any, run_dir: Path, smoke: bool = False, n_jobs: int = 1) -> dict[str, object]:
    """Run fixed-RBF recipe development; public entry point used by the CLI."""
    destination = Path(run_dir).expanduser().resolve()
    repository_root = Path(getattr(config, "repository_root", Path(__file__).resolve().parents[3])).resolve()
    formal = (repository_root / "Data" / "03_outputs" / "study" / "outputs" / "current_97").resolve()
    if destination == formal or formal in destination.parents:
        raise ValueError("feature-develop cannot write into current_97")
    owned = ("registry", "analyses", "final_pipeline", "performance_decision.json", "feature_development_qa.json", "feature_development_report.md", "all_analysis_metrics.csv", "all_recipe_selection_frequencies.csv", "all_feature_block_selection_frequencies.csv", "artifact_manifest.json")
    stale = [name for name in owned if (destination / name).exists()]
    if stale:
        raise FileExistsError(f"Feature-development destination contains stage-owned artifacts: {stale}")
    destination.mkdir(parents=True, exist_ok=True)
    settings = feature_development_config(config, smoke=smoke, n_jobs=n_jobs)
    expected = str(getattr(config, "value", lambda *_: "")("input_hashes.expected.canonical_dataset", ""))
    canonical = repository_root / "Data" / "02_inputs" / "canonical_dataset_ir_corrected_v2"
    protected = {"current_97": formal, "canonical_97": canonical, "performance_optimization_v1": repository_root / "Data" / "03_outputs" / "study" / "outputs" / "performance_optimization_v1"}
    before = {name: directory_tree_summary(path)["tree_sha256"] if path.is_dir() else None for name, path in protected.items()}
    frame, binding = _prepare_canonical_inputs(
        canonical,
        expected_tree_sha256=expected,
        approved_target_registry=formal / "targets" / "target_registry.csv",
        target_qa_path=formal / "targets" / "target_qa.json",
    )
    frame = normalize_feature_development_columns(frame)
    (destination / "registry").mkdir(exist_ok=True)
    (destination / "analyses").mkdir(exist_ok=True)
    (destination / "provenance").mkdir(exist_ok=True)
    recipe_registry = feature_development_recipe_registry()
    recipe_registry.to_csv(
        destination / "registry" / "feature_recipes.csv", index=False
    )
    feature_definition_registry().to_csv(
        destination / "registry" / "feature_definitions.csv", index=False
    )
    _forbidden_predictor_registry().to_csv(
        destination / "registry" / "forbidden_predictors.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "model": "rbf_svr",
                **RBF_PARAMETERS,
                "target_transform": "log10",
                "target_standardization": False,
                "role": "selection_anchor",
            },
            {
                "model": "bayesian_ridge",
                "target_transform": "log10",
                "target_standardization": False,
                "role": "sanity_comparator",
            },
            {
                "model": "dummy_mean_or_median",
                "target_transform": "log10",
                "target_standardization": False,
                "role": "inner_selected_baseline",
            },
        ]
    ).to_csv(destination / "registry" / "fixed_models.csv", index=False)
    write_json(destination / "provenance" / "feature_development_input_binding.json", binding)
    write_json(destination / "provenance" / "feature_development_config.json", asdict(settings))
    results: dict[str, AnalysisResult] = {}; draw_qa: dict[str, object] = {}
    for analysis in ANALYSES:
        result = _run_analysis(frame, analysis, destination / "analyses" / analysis.analysis_id, settings=settings)
        results[analysis.analysis_id] = result
        draw_qa[analysis.analysis_id] = _posterior_scores(result, canonical / "posterior_slope_draws.csv.gz", destination / "analyses" / analysis.analysis_id / "target_uncertainty", settings=settings)
    primary = results["primary_all"]
    decision = _decision(primary, smoke=smoke)
    final = _fit_final(frame.reset_index(drop=True), destination / "final_pipeline", settings=settings, nested_metrics=primary.metrics)
    write_json(destination / "performance_decision.json", decision)
    pd.concat([value.metrics for value in results.values()], ignore_index=True).to_csv(destination / "all_analysis_metrics.csv", index=False)
    pd.concat([value.recipe_frequency.assign(analysis_id=name) for name, value in results.items()], ignore_index=True).to_csv(destination / "all_recipe_selection_frequencies.csv", index=False)
    pd.concat([value.block_frequency.assign(analysis_id=name) for name, value in results.items()], ignore_index=True).to_csv(destination / "all_feature_block_selection_frequencies.csv", index=False)
    after = {
        name: directory_tree_summary(path)["tree_sha256"] if path.is_dir() else None
        for name, path in protected.items()
    }
    expected_roles = {PREDICTION_ROLE, DUMMY_ROLE, BAYESIAN_RIDGE_ROLE}
    coverage: dict[str, dict[str, object]] = {}
    candidate_search_checks: dict[str, dict[str, object]] = {}
    for name, value in results.items():
        expected_records = 90 if value.spec.scope == "alkaline" else 97
        observed_roles = set(value.oof_predictions["prediction_role"].astype(str))
        role_record_counts = {
            role: int(
                value.oof_aggregated.loc[
                    value.oof_aggregated["prediction_role"].eq(role), "record_id"
                ].nunique()
            )
            for role in expected_roles
        }
        coverage[name] = {
            "expected_records": expected_records,
            "observed_records_by_role": role_record_counts,
            "observed_roles": sorted(observed_roles),
            "roles_match_expected": observed_roles == expected_roles,
            "all_roles_expected_oof_count": bool(
                value.oof_aggregated["n_repeated_oof"]
                .eq(settings.outer_repeats)
                .all()
            ),
            "expected_repeated_prediction_rows": int(
                expected_records * settings.outer_repeats * len(expected_roles)
            ),
            "observed_repeated_prediction_rows": int(len(value.oof_predictions)),
        }
        expected_recipe_count = 11 if value.spec.include_sensitivity_recipes else 8
        score_table = pd.read_csv(
            destination / "analyses" / name / "inner_recipe_scores.csv"
        )
        dummy_table = pd.read_csv(
            destination / "analyses" / name / "inner_dummy_scores.csv"
        )
        outer_cells = settings.outer_folds * settings.outer_repeats
        candidate_search_checks[name] = {
            "expected_recipe_count": expected_recipe_count,
            "observed_recipe_count": int(score_table["recipe"].nunique()),
            "all_recipe_scores_complete": bool(score_table["status"].eq("complete").all()),
            "expected_recipe_score_rows": int(outer_cells * expected_recipe_count),
            "observed_recipe_score_rows": int(len(score_table)),
            "dummy_models_per_outer_fold": bool(
                dummy_table.groupby(["repeat", "fold"])["dummy_model"]
                .nunique()
                .eq(2)
                .all()
            ),
            "outer_selection_rows": int(len(value.outer_selections)),
            "expected_outer_selection_rows": int(outer_cells),
        }

    primary_outer = pd.read_csv(
        destination / "analyses" / "primary_all" / "outer_split_manifest.csv"
    )
    expanded_outer = pd.read_csv(
        destination
        / "analyses"
        / "sensitivity_feature_recipes_all"
        / "outer_split_manifest.csv"
    )
    primary_inner = pd.read_csv(
        destination / "analyses" / "primary_all" / "inner_split_manifest.csv"
    ).drop(columns=["analysis_id"])
    expanded_inner = pd.read_csv(
        destination
        / "analyses"
        / "sensitivity_feature_recipes_all"
        / "inner_split_manifest.csv"
    ).drop(columns=["analysis_id"])
    shared_all_scope_assignments = {
        "outer": bool(primary_outer.equals(expanded_outer)),
        "inner": bool(primary_inner.equals(expanded_inner)),
    }
    all_sources = {
        column
        for recipe in FEATURE_DEVELOPMENT_RECIPES
        for column in required_feature_development_source_columns(recipe)
    }
    raw_solvent_fields_excluded = not bool(
        set(RAW_SOLVENT_PREDICTORS).intersection(all_sources)
    )
    target_like_fields_excluded = not any(
        is_target_like_column(column) for column in all_sources
    )
    primary_selection_is_primary_only = bool(
        set(results["primary_all"].outer_selections["recipe"])
        <= set(PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES)
    )
    electrolyte_scope_counts = {
        str(key): int(value)
        for key, value in frame["electrolyte_scope"].astype(str).value_counts().items()
    }
    fixed_pipeline_verified = bool(
        final["model_parameters"] == RBF_PARAMETERS
        and final["target_transform"] == "log10"
        and final["target_standardization"] is False
    )
    coverage_pass = all(
        item["roles_match_expected"]
        and item["all_roles_expected_oof_count"]
        and item["observed_repeated_prediction_rows"]
        == item["expected_repeated_prediction_rows"]
        and all(
            count == item["expected_records"]
            for count in item["observed_records_by_role"].values()
        )
        for item in coverage.values()
    )
    search_pass = all(
        item["expected_recipe_count"] == item["observed_recipe_count"]
        and item["all_recipe_scores_complete"]
        and item["expected_recipe_score_rows"] == item["observed_recipe_score_rows"]
        and item["dummy_models_per_outer_fold"]
        and item["outer_selection_rows"] == item["expected_outer_selection_rows"]
        for item in candidate_search_checks.values()
    )
    qa_pass = bool(
        len(frame) == 97
        and int(frame["electrolyte_scope"].astype(str).eq("alkaline").sum()) == 90
        and len(PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES) == 8
        and len(SENSITIVITY_FEATURE_DEVELOPMENT_RECIPE_NAMES) == 3
        and coverage_pass
        and search_pass
        and all(shared_all_scope_assignments.values())
        and before == after
        and all(value.get("status") == "pass" for value in draw_qa.values())
        and final["recipe"] in PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES
        and primary_selection_is_primary_only
        and raw_solvent_fields_excluded
        and target_like_fields_excluded
        and fixed_pipeline_verified
    )
    qa = {
        "status": "pass" if qa_pass else "fail",
        "smoke": smoke,
        "primary_population_records": int(len(frame)),
        "alkaline_sensitivity_records": int(
            frame["electrolyte_scope"].astype(str).eq("alkaline").sum()
        ),
        "electrolyte_scope_counts": electrolyte_scope_counts,
        "primary_recipe_count": len(PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES),
        "sensitivity_only_recipe_count": len(
            SENSITIVITY_FEATURE_DEVELOPMENT_RECIPE_NAMES
        ),
        "coverage": coverage,
        "candidate_search_checks": candidate_search_checks,
        "shared_all_scope_group_assignments": shared_all_scope_assignments,
        "posterior_draw_score_qa": draw_qa,
        "fixed_rbf_parameters": RBF_PARAMETERS,
        "fixed_target_transform": "log10",
        "fixed_target_standardization": False,
        "fixed_pipeline_verified": fixed_pipeline_verified,
        "recipe_only_inner_selection": True,
        "bayesian_ridge_participates_in_selection": False,
        "sensitivity_affects_final_recipe": False,
        "primary_outer_selections_are_primary_eligible": primary_selection_is_primary_only,
        "final_recipe_is_primary_eligible": final["recipe"]
        in PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES,
        "fold_local_transformations": True,
        "forbidden_predictors_checked": True,
        "raw_solvent_fields_excluded": raw_solvent_fields_excluded,
        "target_like_fields_excluded": target_like_fields_excluded,
        "feature_complexity_definition": "logical feature families before fold-local categorical expansion",
        "protected_tree_sha256_before": before,
        "protected_tree_sha256_after": after,
        "protected_trees_unchanged": before == after,
    }
    write_json(destination / "feature_development_qa.json", qa)
    if not qa_pass:
        raise RuntimeError(f"Feature-development QA failed: {qa}")
    _write_report(
        destination / "feature_development_report.md",
        decision,
        results,
        final,
        smoke=smoke,
    )
    write_json(destination / "artifact_manifest.json", artifact_manifest(destination))
    verification = verify_artifact_manifest(destination)
    if verification["status"] != "pass":
        raise RuntimeError(f"Feature-development artifact manifest failed: {verification}")
    return {"status": "complete_smoke" if smoke else "complete", "run_dir": str(destination), "target": TARGET_NAME, "primary_records": 97, "sensitivity_alkaline_records": 90, "analyses": list(results), "decision": decision, "final_recipe": final["recipe"], "report": str(destination / "feature_development_report.md"), "qa": str(destination / "feature_development_qa.json")}
