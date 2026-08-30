"""Fold-local model helpers for the exploratory three-class soft target.

The three response classes are ``low_30_40_60_like``, ``120_like``, and
``off_cardinal``.  This module deliberately does not create cross-validation
splits or run an Optuna study.  Its helpers are intended to be called from a
grouped *inner* validation objective after the outer-test records have already
been removed.

Every model consumes one three-component probability vector per experimental
record.  It first fits a soft cardinal/off-cardinal gate and then a soft
low-30/40/60-versus-120 conditional model.  Both stages operate directly on
fractional targets and preserve one experimental row per record; posterior
draws and class labels are never expanded into training rows.  The exact source
allowlist and :class:`AssociationFeatureBuilder` keep feature engineering,
imputation, rare-level pooling, composition replacement, and property PCA
local to the supplied training fold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import optuna
import pandas as pd

from .cardinal_models import (
    SoftExtraTreesClassifier,
    SoftGowerKNeighborsClassifier,
)
from .cardinal_reduced_models import GaussianMAPSoftmaxClassifier
from .feature_association_features import (
    AssociationFeatureBuilder,
    required_association_source_columns,
)


THREE_CLASS_LABELS: tuple[str, ...] = (
    "low_30_40_60_like",
    "120_like",
    "off_cardinal",
)
THREE_CLASS_SOFT_COLUMNS: tuple[str, ...] = tuple(
    f"q_{label}" for label in THREE_CLASS_LABELS
)
THREE_CLASS_PREDICTION_COLUMNS: tuple[str, ...] = tuple(
    f"p_{label}" for label in THREE_CLASS_LABELS
)
THREE_CLASS_MODEL_FAMILIES: tuple[str, ...] = (
    "two_stage_gaussian_prior_MAP",
    "two_stage_soft_gower_knn",
    "two_stage_soft_extra_trees",
)
# Backward-compatible short IDs for callers that construct their own search
# registry. They resolve to the same contract-defined two-stage estimators.
MODEL_FAMILIES: tuple[str, ...] = (
    "gaussian_map",
    "gower_knn",
    "extra_trees",
)
THREE_CLASS_GATE_LABELS: tuple[str, ...] = ("cardinal", "off_cardinal")
THREE_CLASS_CONDITIONAL_LABELS: tuple[str, ...] = THREE_CLASS_LABELS[:2]
MODEL_SEARCH_SPACE: Mapping[str, Mapping[str, Any]] = {
    "two_stage_gaussian_prior_MAP": {
        "feature_prior_scale_low": 0.10,
        "feature_prior_scale_high": 4.0,
        "intercept_prior_scale": 5.0,
        "max_iter": 1000,
        "tolerance": 1.0e-8,
    },
    "two_stage_soft_gower_knn": {
        "n_neighbors": (3, 5, 7, 9, 11, 15, 21),
        "distance_power": (1.0, 2.0),
        "epsilon": 1.0e-12,
    },
    "two_stage_soft_extra_trees": {
        "n_estimators_low": 300,
        "n_estimators_high": 1000,
        "n_estimators_step": 100,
        "max_depth_low": 2,
        "max_depth_high": 6,
        "min_samples_leaf_low": 3,
        "min_samples_leaf_high": 12,
        "max_features_low": 0.40,
        "max_features_high": 1.0,
    },
}

_MODEL_ALIASES: Mapping[str, str] = {
    "gaussian_map": "two_stage_gaussian_prior_MAP",
    "direct_gaussian_map": "two_stage_gaussian_prior_MAP",
    "gaussian_prior_map": "two_stage_gaussian_prior_MAP",
    "direct_gaussian_prior_map": "two_stage_gaussian_prior_MAP",
    "two_stage_gaussian_prior_map": "two_stage_gaussian_prior_MAP",
    "gower": "two_stage_soft_gower_knn",
    "gower_knn": "two_stage_soft_gower_knn",
    "soft_gower_knn": "two_stage_soft_gower_knn",
    "two_stage_soft_gower_knn": "two_stage_soft_gower_knn",
    "extra_trees": "two_stage_soft_extra_trees",
    "soft_extra_trees": "two_stage_soft_extra_trees",
    "two_stage_soft_extra_trees": "two_stage_soft_extra_trees",
}

_ALLOWED_PARAMS: Mapping[str, frozenset[str]] = {
    "two_stage_gaussian_prior_MAP": frozenset(
        {
            "feature_prior_scale",
            "intercept_prior_scale",
            "max_iter",
            "tolerance",
        }
    ),
    "two_stage_soft_gower_knn": frozenset(
        {
            "n_neighbors",
            "distance_power",
            "epsilon",
        }
    ),
    "two_stage_soft_extra_trees": frozenset(
        {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
        }
    ),
}


def _canonical_model_family(model_family: str) -> str:
    token = str(model_family).strip().lower()
    try:
        return _MODEL_ALIASES[token]
    except KeyError as error:
        raise KeyError(
            f"Unknown three-class model family {model_family!r}; choose from "
            f"{THREE_CLASS_MODEL_FAMILIES}"
        ) from error


def _validated_params(
    model_family: str,
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values = dict(params or {})
    unknown = sorted(set(values) - set(_ALLOWED_PARAMS[model_family]))
    if unknown:
        raise TypeError(
            f"Unknown {model_family} parameters: {unknown}; allowed="
            f"{sorted(_ALLOWED_PARAMS[model_family])}"
        )
    return values


def _as_soft_targets(y: Any, *, n_rows: int) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        plain = list(THREE_CLASS_LABELS)
        prefixed = list(THREE_CLASS_SOFT_COLUMNS)
        if all(column in y.columns for column in plain):
            values = y.loc[:, plain].to_numpy(dtype=float)
        elif all(column in y.columns for column in prefixed):
            values = y.loc[:, prefixed].to_numpy(dtype=float)
        else:
            raise ValueError(
                "Three-class soft-target tables must contain either "
                f"{plain} or {prefixed}"
            )
    else:
        values = np.asarray(y, dtype=float)
    expected = (int(n_rows), len(THREE_CLASS_LABELS))
    if values.ndim != 2 or values.shape != expected:
        raise ValueError(
            f"Three-class soft targets must have shape {expected}; "
            f"received {tuple(values.shape)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Three-class soft targets contain NaN or infinite values")
    if np.any(values < -1e-12):
        raise ValueError("Three-class soft-target probabilities cannot be negative")
    clipped = np.maximum(values, 0.0)
    totals = clipped.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every three-class soft-target row must have positive mass")
    if not np.allclose(totals, 1.0, rtol=1e-8, atol=1e-8):
        raise ValueError("Every three-class soft-target row must sum to one")
    return clipped / totals[:, None]


def _feature_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    panel: str,
    rare_min_frequency: int,
) -> tuple[
    AssociationFeatureBuilder,
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
]:
    if not isinstance(X_train, pd.DataFrame) or not isinstance(
        X_test, pd.DataFrame
    ):
        raise TypeError("X_train and X_test must both be pandas DataFrames")
    if len(X_train) < 2:
        raise ValueError("At least two training records are required")
    if int(rare_min_frequency) < 1:
        raise ValueError("rare_min_frequency must be positive")
    source_columns = tuple(required_association_source_columns(panel))
    missing_train = sorted(set(source_columns) - set(X_train.columns))
    missing_test = sorted(set(source_columns) - set(X_test.columns))
    if missing_train or missing_test:
        raise KeyError(
            "Three-class panel source columns are missing: "
            f"train={missing_train}, test={missing_test}"
        )

    # Select the explicit allowlist before the fold-local builder sees data.
    # Audit identifiers, curve coordinates, targets, and posterior diagnostics
    # therefore cannot enter a model even if callers retain them in X.
    train_source = X_train.loc[:, list(source_columns)].copy()
    test_source = X_test.loc[:, list(source_columns)].copy()
    builder = AssociationFeatureBuilder(
        panel,
        rare_min_frequency=int(rare_min_frequency),
    ).fit(train_source)
    engineered_train = builder.transform(train_source)
    engineered_test = builder.transform(test_source)
    return builder, engineered_train, engineered_test, source_columns


def suggest_params(
    trial: optuna.Trial,
    model_family: str,
) -> dict[str, Any]:
    """Suggest a conservative conditional search space for one model family.

    The caller should invoke this only inside grouped inner CV.  A joint Optuna
    study may first suggest ``model_family`` and ``panel`` categorically, then
    call this helper for the selected family.  Study-level parallelism should
    remain one for deterministic TPE trial ordering.
    """

    family = _canonical_model_family(model_family)
    if family == "two_stage_gaussian_prior_MAP":
        space = MODEL_SEARCH_SPACE[family]
        return {
            "feature_prior_scale": trial.suggest_float(
                "feature_prior_scale",
                float(space["feature_prior_scale_low"]),
                float(space["feature_prior_scale_high"]),
                log=True,
            ),
            # Keep the weakly informative intercept prior and optimizer
            # controls fixed; tuning them would add search multiplicity without
            # materially expanding the useful small-data model class.
            "intercept_prior_scale": float(
                space["intercept_prior_scale"]
            ),
            "max_iter": int(space["max_iter"]),
            "tolerance": float(space["tolerance"]),
        }
    if family == "two_stage_soft_gower_knn":
        space = MODEL_SEARCH_SPACE[family]
        return {
            "n_neighbors": trial.suggest_categorical(
                "n_neighbors",
                list(space["n_neighbors"]),
            ),
            "distance_power": trial.suggest_categorical(
                "distance_power",
                list(space["distance_power"]),
            ),
            "epsilon": float(space["epsilon"]),
        }
    space = MODEL_SEARCH_SPACE[family]
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            int(space["n_estimators_low"]),
            int(space["n_estimators_high"]),
            step=int(space["n_estimators_step"]),
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            int(space["max_depth_low"]),
            int(space["max_depth_high"]),
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            int(space["min_samples_leaf_low"]),
            int(space["min_samples_leaf_high"]),
        ),
        "max_features": trial.suggest_float(
            "max_features",
            float(space["max_features_low"]),
            float(space["max_features_high"]),
        ),
    }


def _probabilities(
    values: Any,
    *,
    n_rows: int,
    n_classes: int = len(THREE_CLASS_LABELS),
) -> tuple[np.ndarray, float]:
    probabilities = np.asarray(values, dtype=float)
    expected = (int(n_rows), int(n_classes))
    if probabilities.ndim != 2 or probabilities.shape != expected:
        raise ValueError(
            f"Three-class predictions must have shape {expected}; "
            f"received {tuple(probabilities.shape)}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Three-class predictions contain NaN or infinite values")
    if np.any(probabilities < -1e-12):
        raise ValueError("Three-class predictions contain negative probabilities")
    clipped = np.maximum(probabilities, 0.0)
    totals = clipped.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every predicted probability row must have positive mass")
    maximum_simplex_error = (
        float(np.max(np.abs(totals - 1.0))) if len(totals) else 0.0
    )
    return clipped / totals[:, None], maximum_simplex_error


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_stage_estimator(
    *,
    family: str,
    class_names: Sequence[str],
    numeric: Sequence[str],
    categorical: Sequence[str],
    blocks: Mapping[str, Sequence[str]],
    params: Mapping[str, Any],
    seed: int,
) -> Any:
    if family == "two_stage_gaussian_prior_MAP":
        return GaussianMAPSoftmaxClassifier(
            numeric_columns=tuple(numeric),
            categorical_columns=tuple(categorical),
            class_names=tuple(class_names),
            feature_prior_scale=float(params.get("feature_prior_scale", 1.0)),
            intercept_prior_scale=float(
                params.get("intercept_prior_scale", 5.0)
            ),
            max_iter=int(params.get("max_iter", 1000)),
            tolerance=float(params.get("tolerance", 1.0e-8)),
        )
    if family == "two_stage_soft_gower_knn":
        return SoftGowerKNeighborsClassifier(
            numeric_columns=tuple(numeric),
            categorical_columns=tuple(categorical),
            class_names=tuple(class_names),
            blocks=blocks,
            n_neighbors=int(params.get("n_neighbors", 9)),
            distance_power=float(params.get("distance_power", 1.0)),
            epsilon=float(params.get("epsilon", 1.0e-12)),
        )
    return SoftExtraTreesClassifier(
        numeric_columns=tuple(numeric),
        categorical_columns=tuple(categorical),
        class_names=tuple(class_names),
        n_estimators=int(params.get("n_estimators", 400)),
        max_depth=int(params.get("max_depth", 3)),
        min_samples_leaf=int(params.get("min_samples_leaf", 6)),
        max_features=params.get("max_features", 0.7),
        random_state=int(seed),
        n_jobs=1,
    )


def _stage_audit(estimator: Any, *, family: str) -> dict[str, Any]:
    if family == "two_stage_gaussian_prior_MAP":
        return {
            "fit_status": str(estimator.fit_status_),
            "optimization": _jsonable(estimator.optimization_audit_),
            "feature_prior_scale": float(estimator.feature_prior_scale_),
            "intercept_prior_scale": float(estimator.intercept_prior_scale_),
            "n_training_records": int(estimator.n_training_records_),
            "total_fractional_weight": float(
                estimator.total_fractional_weight_
            ),
        }
    if family == "two_stage_soft_gower_knn":
        return {
            "fit_status": "complete",
            "n_neighbors_requested": int(estimator.n_neighbors),
            "n_neighbors_effective": int(
                min(estimator.n_neighbors, len(estimator.soft_targets_))
            ),
            "distance_power": float(estimator.distance_power),
            "n_training_records": int(len(estimator.soft_targets_)),
            "equal_block_weighting": True,
        }
    return {
        "fit_status": "complete",
        "n_estimators": int(estimator.n_estimators),
        "max_depth": int(estimator.max_depth),
        "min_samples_leaf": int(estimator.min_samples_leaf),
        "max_features": _jsonable(estimator.max_features),
        "random_state": _jsonable(estimator.random_state),
        "n_jobs": 1,
    }


def fit_predict_pipeline(
    X_train: pd.DataFrame,
    y_train: Any,
    X_test: pd.DataFrame,
    panel: str,
    model_family: str,
    params: Mapping[str, Any] | None,
    seed: int,
    rare_min_frequency: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fold-local soft-label pipeline and predict three probabilities.

    This helper makes no model or feature choice itself.  It is safe to use as
    one trial inside an Optuna objective only when ``X_train`` is an
    outer-training/inner-training subset and ``X_test`` is that trial's inner
    validation subset.  The same helper can refit the chosen trial on the full
    outer-training subset before predicting its untouched outer-test fold.
    """

    family = _canonical_model_family(model_family)
    resolved_params = _validated_params(family, params)
    targets = _as_soft_targets(y_train, n_rows=len(X_train))
    (
        builder,
        engineered_train,
        engineered_test,
        source_columns,
    ) = _feature_frames(
        X_train,
        X_test,
        panel=str(panel),
        rare_min_frequency=int(rare_min_frequency),
    )
    numeric = tuple(builder.numeric_columns_out_)
    categorical = tuple(builder.categorical_columns_out_)
    blocks = builder.get_blocks_out()

    cardinal_mass = targets[:, :2].sum(axis=1)
    gate_targets = np.column_stack([cardinal_mass, targets[:, 2]])
    conditional_targets = np.divide(
        targets[:, :2],
        cardinal_mass[:, None],
        out=np.full((len(targets), 2), 0.5, dtype=float),
        where=cardinal_mass[:, None] > 0,
    )

    gate_estimator = _make_stage_estimator(
        family=family,
        class_names=THREE_CLASS_GATE_LABELS,
        numeric=numeric,
        categorical=categorical,
        blocks=blocks,
        params=resolved_params,
        seed=int(seed),
    ).fit(engineered_train, gate_targets)
    conditional_estimator = _make_stage_estimator(
        family=family,
        class_names=THREE_CLASS_CONDITIONAL_LABELS,
        numeric=numeric,
        categorical=categorical,
        blocks=blocks,
        params=resolved_params,
        seed=int(seed),
    ).fit(
        engineered_train,
        conditional_targets,
        sample_weight=cardinal_mass,
    )
    if family == "two_stage_gaussian_prior_MAP":
        for stage_name, estimator in (
            ("gate", gate_estimator),
            ("conditional", conditional_estimator),
        ):
            if estimator.fit_status_ != "complete":
                raise RuntimeError(
                    f"Gaussian MAP {stage_name} optimizer did not converge; "
                    f"audit={estimator.optimization_audit_}"
                )

    gate, _ = _probabilities(
        gate_estimator.predict_proba(engineered_test),
        n_rows=len(engineered_test),
        n_classes=2,
    )
    conditional, _ = _probabilities(
        conditional_estimator.predict_proba(engineered_test),
        n_rows=len(engineered_test),
        n_classes=2,
    )
    combined = np.column_stack(
        [
            gate[:, 0] * conditional[:, 0],
            gate[:, 0] * conditional[:, 1],
            gate[:, 1],
        ]
    )
    model_audit = {
        "fit_status": "complete",
        "architecture": "cardinal_gate_then_low60_vs_120_conditional",
        "gate": _stage_audit(gate_estimator, family=family),
        "conditional": _stage_audit(
            conditional_estimator,
            family=family,
        ),
        "conditional_total_weight": float(cardinal_mass.sum()),
    }

    probabilities, simplex_error = _probabilities(
        combined,
        n_rows=len(engineered_test),
    )
    audit = {
        "panel": str(panel),
        "model_family": family,
        "model_params": _jsonable(resolved_params),
        "seed": int(seed),
        "rare_min_frequency": int(rare_min_frequency),
        "class_labels": list(THREE_CLASS_LABELS),
        "n_train_records": int(len(X_train)),
        "n_test_records": int(len(X_test)),
        "source_columns": list(source_columns),
        "engineered_feature_names": [
            str(value) for value in builder.get_feature_names_out()
        ],
        "numeric_features": list(numeric),
        "categorical_features": list(categorical),
        "gower_blocks": {
            str(block): list(features)
            for block, features in blocks.items()
        },
        "soft_target_shape": list(targets.shape),
        "posterior_draws_expanded": False,
        "class_rows_expanded": False,
        "two_stage_structure": True,
        "gate_class_labels": list(THREE_CLASS_GATE_LABELS),
        "conditional_class_labels": list(
            THREE_CLASS_CONDITIONAL_LABELS
        ),
        "feature_engineering_fit_scope": "supplied_training_rows_only",
        "prediction_shape": list(probabilities.shape),
        "prediction_simplex_max_abs_error_before_renormalization": (
            simplex_error
        ),
        "model_audit": model_audit,
    }
    return probabilities, audit


__all__: Sequence[str] = (
    "THREE_CLASS_LABELS",
    "THREE_CLASS_CONDITIONAL_LABELS",
    "THREE_CLASS_GATE_LABELS",
    "MODEL_SEARCH_SPACE",
    "THREE_CLASS_MODEL_FAMILIES",
    "THREE_CLASS_PREDICTION_COLUMNS",
    "THREE_CLASS_SOFT_COLUMNS",
    "fit_predict_pipeline",
    "suggest_params",
)
