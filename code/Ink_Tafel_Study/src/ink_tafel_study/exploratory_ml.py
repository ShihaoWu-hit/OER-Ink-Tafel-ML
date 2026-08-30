"""Isolated exploratory benchmark for the 97-record low-potential target.

This module intentionally does not participate in the registered nested-CV
workflow.  It provides the user-requested naive random-split benchmark while
preserving enough metadata to quantify connected-group leakage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import optuna
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import resolve_feature_panel, validate_no_target_leakage
from .models import GowerKNeighborsRegressor


AUDIT_COLUMNS: tuple[str, ...] = (
    "record_uid",
    "record_id",
    "curve_id",
    "publication_id",
    "recipe_hash",
    "cv_group_id",
    "electrolyte_scope",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "catalyst_active_material_type",
    "catalyst_support",
    "electrolyte_identity",
    "ir_correction_status",
    "conductive_agent_present",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "catalyst_loading_mg_cm2_selected",
    "electrolyte_molarity_M",
    "rpm",
    "scan_rate_mV_s",
    "catalyst_concentration_mg_mL",
    "ionomer_to_solvent_ratio",
    "ionomer_concentration_wt_percent",
    "conductive_agent_dose_mg_per_mgcat",
    "solvent_fraction_acetone",
    "solvent_fraction_water",
    "solvent_fraction_ethanol",
    "solvent_fraction_isopropanol",
    "solvent_fraction_methanol",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "catalyst_active_material_type",
    "catalyst_support",
    "catalyst_loading_mg_cm2_selected",
    "electrolyte_identity",
    "electrolyte_molarity_M",
    "rpm",
    "scan_rate_mV_s",
    "ir_correction_status",
    "catalyst_concentration_mg_mL",
    "ionomer_to_solvent_ratio",
    "ionomer_concentration_wt_percent",
    "conductive_agent_present",
    "conductive_agent_dose_mg_per_mgcat",
    "solvent_fraction_acetone",
    "solvent_fraction_water",
    "solvent_fraction_ethanol",
    "solvent_fraction_isopropanol",
    "solvent_fraction_methanol",
)

TARGET_COLUMN = "b_bayesian_local_line_mV_dec"

GOWER_BLOCKS: Mapping[str, tuple[str, ...]] = {
    "catalyst": (
        "catalyst_active_material_type",
        "catalyst_support",
    ),
    "electrode": ("catalyst_loading_mg_cm2_selected",),
    "electrolyte": (
        "electrolyte_identity",
        "electrolyte_molarity_M",
    ),
    "measurement": ("rpm", "scan_rate_mV_s"),
    "ir_correction": ("ir_correction_status",),
    "ink_amounts": (
        "catalyst_concentration_mg_mL",
        "ionomer_to_solvent_ratio",
        "ionomer_concentration_wt_percent",
    ),
    "conductive_agent": (
        "conductive_agent_present",
        "conductive_agent_dose_mg_per_mgcat",
    ),
    "solvent_composition": (
        "solvent_fraction_acetone",
        "solvent_fraction_water",
        "solvent_fraction_ethanol",
        "solvent_fraction_isopropanol",
        "solvent_fraction_methanol",
    ),
}

PROPERTY_PROXY_CATEGORICAL_FEATURES: tuple[str, ...] = CATEGORICAL_FEATURES

PROPERTY_PROXY_NUMERIC_FEATURES: tuple[str, ...] = (
    "catalyst_loading_mg_cm2_selected",
    "electrolyte_molarity_M",
    "rpm",
    "scan_rate_mV_s",
    "catalyst_concentration_mg_mL",
    "ionomer_to_solvent_ratio",
    "ionomer_concentration_wt_percent",
    "conductive_agent_dose_mg_per_mgcat",
    "solvent_mixture_property_proxy_viscosity_mPa_s",
    "solvent_mixture_property_proxy_surface_tension_mN_m",
    "solvent_mixture_property_proxy_density_kg_m3",
    "solvent_mixture_property_proxy_dielectric_constant_best_available",
    "solvent_mixture_property_proxy_dielectric_constant_estimated_flag",
)

PROPERTY_PROXY_FEATURE_COLUMNS: tuple[str, ...] = (
    "catalyst_active_material_type",
    "catalyst_support",
    "catalyst_loading_mg_cm2_selected",
    "electrolyte_identity",
    "electrolyte_molarity_M",
    "rpm",
    "scan_rate_mV_s",
    "ir_correction_status",
    "catalyst_concentration_mg_mL",
    "ionomer_to_solvent_ratio",
    "ionomer_concentration_wt_percent",
    "conductive_agent_present",
    "conductive_agent_dose_mg_per_mgcat",
    "solvent_mixture_property_proxy_viscosity_mPa_s",
    "solvent_mixture_property_proxy_surface_tension_mN_m",
    "solvent_mixture_property_proxy_density_kg_m3",
    "solvent_mixture_property_proxy_dielectric_constant_best_available",
    "solvent_mixture_property_proxy_dielectric_constant_estimated_flag",
)

PROPERTY_PROXY_GOWER_BLOCKS: Mapping[str, tuple[str, ...]] = {
    "catalyst": (
        "catalyst_active_material_type",
        "catalyst_support",
    ),
    "electrode": ("catalyst_loading_mg_cm2_selected",),
    "electrolyte": (
        "electrolyte_identity",
        "electrolyte_molarity_M",
    ),
    "measurement": ("rpm", "scan_rate_mV_s"),
    "ir_correction": ("ir_correction_status",),
    "ink_amounts": (
        "catalyst_concentration_mg_mL",
        "ionomer_to_solvent_ratio",
        "ionomer_concentration_wt_percent",
    ),
    "conductive_agent": (
        "conductive_agent_present",
        "conductive_agent_dose_mg_per_mgcat",
    ),
    "solvent_property_proxy": (
        "solvent_mixture_property_proxy_viscosity_mPa_s",
        "solvent_mixture_property_proxy_surface_tension_mN_m",
        "solvent_mixture_property_proxy_density_kg_m3",
        "solvent_mixture_property_proxy_dielectric_constant_best_available",
    ),
    "solvent_property_proxy_estimation_flag": (
        "solvent_mixture_property_proxy_dielectric_constant_estimated_flag",
    ),
}

MODEL_NAMES: tuple[str, ...] = (
    "dummy_median",
    "bayesian_ridge",
    "random_forest",
    "extra_trees",
    "catboost",
    "gower_knn",
)


@dataclass(frozen=True)
class ExploratoryPanelSpec:
    name: str
    feature_columns: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    gower_blocks: Mapping[str, tuple[str, ...]]


PANEL_SPECS: Mapping[str, ExploratoryPanelSpec] = {
    "context_plus_ink": ExploratoryPanelSpec(
        name="context_plus_ink",
        feature_columns=FEATURE_COLUMNS,
        categorical_features=CATEGORICAL_FEATURES,
        numeric_features=NUMERIC_FEATURES,
        gower_blocks=GOWER_BLOCKS,
    ),
    "context_plus_property_proxy_4p_estimated": ExploratoryPanelSpec(
        name="context_plus_property_proxy_4p_estimated",
        feature_columns=PROPERTY_PROXY_FEATURE_COLUMNS,
        categorical_features=PROPERTY_PROXY_CATEGORICAL_FEATURES,
        numeric_features=PROPERTY_PROXY_NUMERIC_FEATURES,
        gower_blocks=PROPERTY_PROXY_GOWER_BLOCKS,
    ),
}


@dataclass(frozen=True)
class BenchmarkResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    split_manifest: pd.DataFrame
    best_params: dict[str, dict[str, Any]]
    trials: pd.DataFrame
    audit: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def panel_spec(panel_name: str) -> ExploratoryPanelSpec:
    try:
        return PANEL_SPECS[panel_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown exploratory panel {panel_name!r}; choose from {sorted(PANEL_SPECS)}"
        ) from exc


def infer_panel_spec(frame: pd.DataFrame) -> ExploratoryPanelSpec:
    actual = tuple(frame.columns)
    matching = [
        spec
        for spec in PANEL_SPECS.values()
        if actual == AUDIT_COLUMNS + spec.feature_columns + (TARGET_COLUMN,)
    ]
    if len(matching) != 1:
        expected = {
            name: AUDIT_COLUMNS + spec.feature_columns + (TARGET_COLUMN,)
            for name, spec in PANEL_SPECS.items()
        }
        raise ValueError(
            "ML dataset columns or order do not match exactly one approved exploratory panel: "
            f"actual={actual}, expected={expected}"
        )
    return matching[0]


def export_registered_panel_dataset(
    *,
    panel_name: str,
    records_path: Path,
    targets_path: Path,
    panel_schema_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Export one exact registered panel plus audit columns and the response."""

    spec = panel_spec(panel_name)
    records = pd.read_csv(records_path)
    targets = pd.read_csv(targets_path)
    schema = pd.read_csv(panel_schema_path)
    _require_columns(
        records,
        AUDIT_COLUMNS + spec.feature_columns,
        label="records table",
    )
    target_binding_columns = (
        "record_id",
        "curve_id",
        "cv_group_id",
        "electrolyte_scope",
        TARGET_COLUMN,
    )
    _require_columns(targets, target_binding_columns, label="targets table")
    _require_columns(
        schema,
        ("panel", "selected_column", "status"),
        label="registered panel schema",
    )

    registered = tuple(
        schema.loc[
            schema["panel"].eq(panel_name) & schema["status"].eq("present"),
            "selected_column",
        ].astype(str)
    )
    if registered != spec.feature_columns:
        raise ValueError(
            f"Registered {panel_name} schema drifted from the approved exploratory export: "
            f"registered={registered}, expected={spec.feature_columns}"
        )
    resolved = resolve_feature_panel(records, panel_name, require_all=True)
    if tuple(resolved.columns) != spec.feature_columns:
        raise ValueError(
            f"Resolved {panel_name} panel does not match the approved export order: "
            f"resolved={resolved.columns}, expected={spec.feature_columns}"
        )
    validate_no_target_leakage(spec.feature_columns)

    if records["record_id"].duplicated().any():
        duplicated = records.loc[
            records["record_id"].duplicated(False), "record_id"
        ].tolist()
        raise ValueError(f"records table has duplicate record_id values: {duplicated}")
    if targets["record_id"].duplicated().any():
        duplicated = targets.loc[
            targets["record_id"].duplicated(False), "record_id"
        ].tolist()
        raise ValueError(f"targets table has duplicate record_id values: {duplicated}")

    target_values = targets.loc[:, list(target_binding_columns)].rename(
        columns={
            "curve_id": "target_curve_id",
            "cv_group_id": "target_cv_group_id",
            "electrolyte_scope": "target_electrolyte_scope",
        }
    )
    exported = records.loc[:, list(AUDIT_COLUMNS + spec.feature_columns)].merge(
        target_values,
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    binding_pairs = (
        ("curve_id", "target_curve_id"),
        ("cv_group_id", "target_cv_group_id"),
        ("electrolyte_scope", "target_electrolyte_scope"),
    )
    binding_mismatches = {
        record_column: exported.loc[
            exported[record_column].astype(str).ne(exported[target_column].astype(str)),
            "record_id",
        ].tolist()
        for record_column, target_column in binding_pairs
        if exported[record_column].astype(str).ne(exported[target_column].astype(str)).any()
    }
    if binding_mismatches:
        raise ValueError(
            "Target bindings disagree with the registered records table: "
            f"{binding_mismatches}"
        )
    exported = exported.loc[
        :, list(AUDIT_COLUMNS + spec.feature_columns + (TARGET_COLUMN,))
    ]
    validate_ml_dataset(exported, panel_name=panel_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False)
    return exported


def export_context_plus_ink_dataset(
    *,
    records_path: Path,
    targets_path: Path,
    panel_schema_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Export the exact registered 18-feature table plus audit columns/target."""
    return export_registered_panel_dataset(
        panel_name="context_plus_ink",
        records_path=records_path,
        targets_path=targets_path,
        panel_schema_path=panel_schema_path,
        output_path=output_path,
    )


def export_property_proxy_dataset(
    *,
    records_path: Path,
    targets_path: Path,
    panel_schema_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Export the registered 18-feature best-available property-proxy panel."""

    return export_registered_panel_dataset(
        panel_name="context_plus_property_proxy_4p_estimated",
        records_path=records_path,
        targets_path=targets_path,
        panel_schema_path=panel_schema_path,
        output_path=output_path,
    )


def validate_ml_dataset(
    frame: pd.DataFrame,
    *,
    expected_records: int = 97,
    panel_name: str | None = None,
) -> ExploratoryPanelSpec:
    """Fail closed on schema, grain, target support, and predictor leakage."""

    spec = infer_panel_spec(frame) if panel_name is None else panel_spec(panel_name)
    expected_columns = AUDIT_COLUMNS + spec.feature_columns + (TARGET_COLUMN,)
    if tuple(frame.columns) != expected_columns:
        raise ValueError(
            "ML dataset columns or order differ from the approved schema: "
            f"actual={tuple(frame.columns)}, expected={expected_columns}"
        )
    if len(frame) != expected_records:
        raise ValueError(f"ML dataset must have {expected_records} rows; found {len(frame)}")
    if frame["record_id"].duplicated().any() or frame["record_uid"].duplicated().any():
        raise ValueError("ML dataset record_id and record_uid must both be unique")
    if frame.loc[:, list(AUDIT_COLUMNS)].isna().any().any():
        raise ValueError("ML dataset audit identifiers cannot be missing")
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("ML target must be finite, positive, and complete")
    scope_counts = frame["electrolyte_scope"].value_counts().to_dict()
    if expected_records == 97 and scope_counts != {"alkaline": 90, "acidic": 5, "neutral": 2}:
        raise ValueError(f"Unexpected electrolyte_scope counts: {scope_counts}")
    validate_no_target_leakage(spec.feature_columns)
    return spec


def load_ml_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_ml_dataset(frame)
    return frame


def _preprocessor(
    *, scale_numeric: bool, spec: ExploratoryPanelSpec
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(spec.numeric_features)),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(spec.categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _catboost_frame(
    frame: pd.DataFrame, *, spec: ExploratoryPanelSpec
) -> pd.DataFrame:
    prepared = frame.loc[:, list(spec.feature_columns)].copy()
    for column in spec.numeric_features:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in spec.categorical_features:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
    return prepared


def _catboost_class() -> type[BaseEstimator]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover - exercised in base-only environments
        raise RuntimeError(
            "CatBoost is required for this exploratory benchmark. Install it with "
            "`.venv/bin/pip install -r code/Ink_Tafel_Study/requirements-exploratory-ml.txt`."
        ) from exc
    return CatBoostRegressor


def _build_estimator(
    model_name: str,
    params: Mapping[str, Any],
    *,
    seed: int,
    spec: ExploratoryPanelSpec,
) -> BaseEstimator:
    if model_name == "dummy_median":
        return DummyRegressor(strategy="median")
    if model_name == "bayesian_ridge":
        return Pipeline(
            [
                ("preprocess", _preprocessor(scale_numeric=True, spec=spec)),
                ("regressor", BayesianRidge(compute_score=True)),
            ]
        )
    if model_name in {"random_forest", "extra_trees"}:
        regressor_class = RandomForestRegressor if model_name == "random_forest" else ExtraTreesRegressor
        return Pipeline(
            [
                ("preprocess", _preprocessor(scale_numeric=False, spec=spec)),
                (
                    "regressor",
                    regressor_class(
                        **dict(params),
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if model_name == "gower_knn":
        return GowerKNeighborsRegressor(
            n_neighbors=int(params["n_neighbors"]),
            distance_power=float(params["distance_power"]),
            numeric_features=spec.numeric_features,
            categorical_features=spec.categorical_features,
            blocks=spec.gower_blocks,
        )
    if model_name == "catboost":
        regressor_class = _catboost_class()
        return regressor_class(
            **dict(params),
            cat_features=list(spec.categorical_features),
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
            thread_count=1,
            bootstrap_type="Bayesian",
        )
    raise KeyError(f"Unknown exploratory model: {model_name}")


def _model_frame(
    model_name: str, frame: pd.DataFrame, *, spec: ExploratoryPanelSpec
) -> pd.DataFrame:
    if model_name == "catboost":
        return _catboost_frame(frame, spec=spec)
    return frame.loc[:, list(spec.feature_columns)]


def _cross_validated_mae(
    *,
    model_name: str,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    y_log10: np.ndarray,
    y_original: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
    spec: ExploratoryPanelSpec,
) -> float:
    fold_mae: list[float] = []
    for fold, (train_index, validation_index) in enumerate(splits):
        estimator = _build_estimator(
            model_name, params, seed=seed + fold, spec=spec
        )
        train_X = _model_frame(model_name, X.iloc[train_index], spec=spec)
        validation_X = _model_frame(
            model_name, X.iloc[validation_index], spec=spec
        )
        estimator.fit(train_X, y_log10[train_index])
        predicted = np.power(10.0, np.asarray(estimator.predict(validation_X), dtype=float))
        fold_mae.append(float(mean_absolute_error(y_original[validation_index], predicted)))
    return float(np.mean(fold_mae))


def _tree_trial_params(
    trial: optuna.Trial, *, smoke: bool
) -> dict[str, int | float]:
    if smoke:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 40, 80, step=40),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def _catboost_trial_params(
    trial: optuna.Trial, *, smoke: bool
) -> dict[str, int | float]:
    if smoke:
        return {
            "iterations": trial.suggest_int("iterations", 40, 80, step=40),
            "depth": trial.suggest_int("depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.01, 2.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }
    return {
        "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
        "depth": trial.suggest_int("depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1.0e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
    }


def _optuna_search(
    *,
    model_name: str,
    X: pd.DataFrame,
    y_log10: np.ndarray,
    y_original: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
    trials: int,
    smoke: bool,
    spec: ExploratoryPanelSpec,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    def objective(trial: optuna.Trial) -> float:
        if model_name in {"random_forest", "extra_trees"}:
            params = _tree_trial_params(trial, smoke=smoke)
        elif model_name == "catboost":
            params = _catboost_trial_params(trial, smoke=smoke)
        else:  # pragma: no cover - guarded by callers
            raise KeyError(model_name)
        return _cross_validated_mae(
            model_name=model_name,
            params=params,
            X=X,
            y_log10=y_log10,
            y_original=y_original,
            splits=splits,
            seed=seed,
            spec=spec,
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"exploratory_{model_name}",
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    rows = [
        {
            "model": model_name,
            "search_method": "optuna_tpe",
            "trial_number": int(trial.number),
            "cv_mae_mV_dec": float(trial.value) if trial.value is not None else np.nan,
            "state": trial.state.name,
            "params_json": json.dumps(trial.params, sort_keys=True),
        }
        for trial in study.trials
    ]
    return dict(study.best_params), float(study.best_value), rows


def _gower_search(
    *,
    X: pd.DataFrame,
    y_log10: np.ndarray,
    y_original: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
    smoke: bool,
    spec: ExploratoryPanelSpec,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    neighbors = (3, 5) if smoke else (3, 5, 7, 9, 11, 15, 21)
    powers = (1.0,) if smoke else (1.0, 2.0)
    rows: list[dict[str, Any]] = []
    for n_neighbors in neighbors:
        for distance_power in powers:
            params = {
                "n_neighbors": n_neighbors,
                "distance_power": distance_power,
            }
            value = _cross_validated_mae(
                model_name="gower_knn",
                params=params,
                X=X,
                y_log10=y_log10,
                y_original=y_original,
                splits=splits,
                seed=seed,
                spec=spec,
            )
            rows.append(
                {
                    "model": "gower_knn",
                    "search_method": "exhaustive_grid",
                    "trial_number": len(rows),
                    "cv_mae_mV_dec": value,
                    "state": "COMPLETE",
                    "params_json": json.dumps(params, sort_keys=True),
                }
            )
    best = min(rows, key=lambda row: row["cv_mae_mV_dec"])
    return json.loads(best["params_json"]), float(best["cv_mae_mV_dec"]), rows


def _score_predictions(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        spearman = float(spearmanr(observed, predicted).statistic)
    return {
        "mae_mV_dec": float(mean_absolute_error(observed, predicted)),
        "rmse_mV_dec": float(np.sqrt(mean_squared_error(observed, predicted))),
        "median_ae_mV_dec": float(median_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "spearman": spearman,
    }


def run_exploratory_benchmark(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    test_size: float = 0.20,
    seed: int = 42,
    optuna_trials: int = 100,
    smoke: bool = False,
) -> BenchmarkResult:
    """Run the isolated naive split benchmark and materialize audit artifacts."""

    spec = validate_ml_dataset(frame)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between zero and one")
    if optuna_trials < 1:
        raise ValueError("optuna_trials must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    all_indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train = frame.iloc[train_indices].reset_index(drop=True)
    test = frame.iloc[test_indices].reset_index(drop=True)
    X_train = train.loc[:, list(spec.feature_columns)]
    X_test = test.loc[:, list(spec.feature_columns)]
    y_train_original = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_test_original = test[TARGET_COLUMN].to_numpy(dtype=float)
    y_train_log10 = np.log10(y_train_original)

    cv_folds = 2 if smoke else 5
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_splits = list(splitter.split(X_train))
    effective_trials = min(optuna_trials, 2) if smoke else optuna_trials

    overlap_groups = sorted(set(train["cv_group_id"]) & set(test["cv_group_id"]))
    overlapping_test_records = sorted(
        test.loc[test["cv_group_id"].isin(overlap_groups), "record_id"].astype(int).tolist()
    )
    split_manifest = frame.loc[:, list(AUDIT_COLUMNS)].copy()
    split_manifest["split"] = "train"
    split_manifest.loc[test_indices, "split"] = "test"
    split_manifest["connected_group_crosses_split"] = split_manifest["cv_group_id"].isin(
        overlap_groups
    )

    best_params: dict[str, dict[str, Any]] = {}
    cv_values: dict[str, float] = {}
    trial_rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        if model_name in {"random_forest", "extra_trees", "catboost"}:
            params, cv_mae, rows = _optuna_search(
                model_name=model_name,
                X=X_train,
                y_log10=y_train_log10,
                y_original=y_train_original,
                splits=cv_splits,
                seed=seed,
                trials=effective_trials,
                smoke=smoke,
                spec=spec,
            )
        elif model_name == "gower_knn":
            params, cv_mae, rows = _gower_search(
                X=X_train,
                y_log10=y_train_log10,
                y_original=y_train_original,
                splits=cv_splits,
                seed=seed,
                smoke=smoke,
                spec=spec,
            )
        else:
            params = {}
            cv_mae = _cross_validated_mae(
                model_name=model_name,
                params=params,
                X=X_train,
                y_log10=y_train_log10,
                y_original=y_train_original,
                splits=cv_splits,
                seed=seed,
                spec=spec,
            )
            rows = []
        best_params[model_name] = params
        cv_values[model_name] = cv_mae
        trial_rows.extend(rows)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        estimator = _build_estimator(
            model_name, best_params[model_name], seed=seed, spec=spec
        )
        estimator.fit(_model_frame(model_name, X_train, spec=spec), y_train_log10)
        predicted_log10 = np.asarray(
            estimator.predict(_model_frame(model_name, X_test, spec=spec)), dtype=float
        )
        predicted = np.power(10.0, predicted_log10)
        if not np.isfinite(predicted).all() or (predicted <= 0).any():
            raise RuntimeError(f"{model_name} produced invalid test predictions")
        metric_rows.append(
            {
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
                "seed": seed,
                "test_size": test_size,
                "inner_cv_folds": cv_folds,
                "search_trials": effective_trials if model_name in {"random_forest", "extra_trees", "catboost"} else 0,
                "cv_mae_mV_dec": cv_values[model_name],
                **_score_predictions(y_test_original, predicted),
            }
        )
        rows = test.loc[:, list(AUDIT_COLUMNS)].copy()
        rows.insert(0, "model", model_name)
        rows["observed_mV_dec"] = y_test_original
        rows["predicted_mV_dec"] = predicted
        rows["residual_mV_dec"] = y_test_original - predicted
        prediction_rows.append(rows)

    metrics = pd.DataFrame(metric_rows).sort_values("mae_mV_dec").reset_index(drop=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    trials_frame = pd.DataFrame(trial_rows)
    audit = {
        "analysis_status": "naive_exploratory_random_split_not_formal_evaluation",
        "feature_panel": spec.name,
        "n_predictors": len(spec.feature_columns),
        "target": TARGET_COLUMN,
        "target_transform": "log10_fit_metrics_on_original_mV_dec_scale",
        "n_records": len(frame),
        "n_train": len(train),
        "n_test": len(test),
        "seed": seed,
        "test_size": test_size,
        "smoke": smoke,
        "inner_cv_folds": cv_folds,
        "optuna_trials_per_tuned_model": effective_trials,
        "connected_groups_crossing_train_test": len(overlap_groups),
        "overlap_group_ids": overlap_groups,
        "test_records_in_overlapping_groups": overlapping_test_records,
        "n_test_records_in_overlapping_groups": len(overlapping_test_records),
    }

    metrics.to_csv(output_dir / "benchmark_metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    trials_frame.to_csv(output_dir / "optuna_trials.csv", index=False)
    (output_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "benchmark_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    best = metrics.iloc[0]
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Exploratory ML benchmark",
                "",
                f"Status: {'smoke test' if smoke else 'full exploratory benchmark'}.",
                f"Feature panel: `{spec.name}` ({len(spec.feature_columns)} predictors).",
                "",
                "This is a naive ordinary random 80/20 split and is not the registered grouped nested-CV analysis.",
                f"The split places {len(overlap_groups)} connected groups across train and test, affecting {len(overlapping_test_records)} test records; metrics may be optimistic.",
                "",
                f"Lowest test MAE in this run: `{best['model']}` at {best['mae_mV_dec']:.4f} mV dec^-1.",
                "",
                "No feature importance or mechanistic interpretation is authorized from this benchmark.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return BenchmarkResult(
        metrics=metrics,
        predictions=predictions,
        split_manifest=split_manifest,
        best_params=best_params,
        trials=trials_frame,
        audit=audit,
    )
