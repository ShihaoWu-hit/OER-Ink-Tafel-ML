"""Partially pooled soft cardinal classification for Ru and Ir families.

This module implements a *MAP approximation* to a hierarchical Bayesian
multinomial-logit model.  It is deliberately not described as NUTS, MCMC, or
full posterior inference: Gaussian priors are represented by quadratic
penalties and the posterior mode is found with SciPy L-BFGS-B.

For every outcome stage and non-reference class the linear predictor is

``global intercept + X @ global coefficients
  + family intercept deviation + X @ family coefficient deviations``.

Family deviations are constrained to have a training-mass-weighted mean of
zero and receive separate Gaussian shrinkage priors for intercepts and feature
coefficients.  Consequently RuO2 and ``IrO2_plus_Ir`` share a global model,
while retaining partially pooled family-specific departures.

Every experimental record contributes exactly one fractional multinomial
response vector.  Posterior slope draws are never expanded into pseudo-rows.
Feature engineering and all learned preprocessing are fitted inside each call
to :meth:`fit`, so the estimator can be used safely inside connected-group
inner and outer cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cardinal_cv import (
    make_inner_soft_group_splits,
    make_soft_stratified_group_manifest,
)
from .cardinal_feature_engineering import CardinalFeatureBuilder
from .cardinal_metrics import (
    PREDICTION_COLUMNS,
    SOFT_LABEL_COLUMNS,
    aggregate_repeated_oof,
    collapsed_hard_metrics,
    soft_cross_entropy,
    two_stage_probability_metrics,
)
from .cardinal_models import (
    CARDINAL_CLASSES,
    CARDINAL_GATE_CLASSES,
    CLASSIC_CARDINAL_CLASSES,
)
from .evaluate import (
    DEFAULT_RELATION_COLUMNS,
    assert_no_relational_leakage,
    validate_split_manifest,
)


HIERARCHICAL_FAMILY_COLUMN = "__catalyst_family__"
HIERARCHICAL_FAMILIES: tuple[str, ...] = ("IrO2_plus_Ir", "RuO2")
CATALYST_TO_HIERARCHICAL_FAMILY: Mapping[str, str] = {
    "Ir": "IrO2_plus_Ir",
    "IrO2": "IrO2_plus_Ir",
    "IrO2_plus_Ir": "IrO2_plus_Ir",
    "RuO2": "RuO2",
}


def canonical_catalyst_family(values: Sequence[Any]) -> np.ndarray:
    """Map the persisted material identity to the two registered families."""

    output: list[str] = []
    unknown: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        mapped = CATALYST_TO_HIERARCHICAL_FAMILY.get(value)
        if mapped is None:
            unknown.add(value)
        else:
            output.append(mapped)
    if unknown:
        raise ValueError(
            "Hierarchical cardinal model accepts only RuO2, IrO2, metallic Ir, "
            f"or a pre-mapped family label; unknown={sorted(unknown)}"
        )
    return np.asarray(output, dtype=object)


def _as_frame(
    X: Any, expected_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Hierarchical cardinal estimators require a pandas DataFrame"
        )
    frame = X.copy(deep=False)
    if expected_columns is not None:
        missing = sorted(set(expected_columns) - set(frame.columns))
        if missing:
            raise ValueError(
                f"Prediction table is missing fitted features: {missing}"
            )
        frame = frame.loc[:, list(expected_columns)]
    return frame


def _class_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(map(str, values))
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_names must contain at least two unique names")
    return names


def _soft_targets(
    y: Any, *, n_rows: int, class_names: Sequence[str]
) -> np.ndarray:
    names = _class_names(class_names)
    if isinstance(y, pd.DataFrame):
        plain = list(names)
        prefixed = [f"q_{name}" for name in names]
        if all(column in y.columns for column in plain):
            values = y.loc[:, plain].to_numpy(float)
        elif all(column in y.columns for column in prefixed):
            values = y.loc[:, prefixed].to_numpy(float)
        else:
            raise ValueError(
                "Soft-target table must contain either plain or q_-prefixed "
                f"class columns for {list(names)}"
            )
    else:
        values = np.asarray(y, dtype=float)
    if values.shape != (n_rows, len(names)):
        raise ValueError(
            "Soft targets must have shape "
            f"({n_rows}, {len(names)}); received {tuple(values.shape)}"
        )
    if not np.isfinite(values).all() or np.any(values < -1e-12):
        raise ValueError("Soft targets must be finite and nonnegative")
    values = np.maximum(values, 0.0)
    totals = values.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every soft-target row must have positive mass")
    return values / totals[:, None]


def _sample_weights(values: Any, n_rows: int) -> np.ndarray:
    if values is None:
        return np.ones(n_rows, dtype=float)
    weights = np.asarray(values, dtype=float).reshape(-1)
    if weights.shape != (n_rows,):
        raise ValueError(
            f"sample_weight must have shape ({n_rows},), got {weights.shape}"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and nonnegative")
    if not np.any(weights > 0):
        raise ValueError("At least one sample weight must be positive")
    return weights


def _positive_scale(value: Any, name: str) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return scale


def _preprocessor(
    numeric_columns: Sequence[str], categorical_columns: Sequence[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median", keep_empty_features=True
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_columns),
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                list(categorical_columns),
            )
        )
    if not transformers:
        raise ValueError("At least one non-family predictor is required")
    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )


class HierarchicalGaussianMAPSoftmaxClassifier(
    ClassifierMixin, BaseEstimator
):
    """Gaussian-prior hierarchical softmax fitted by posterior-mode MAP.

    This is an auditable deterministic approximation to a hierarchical
    Bayesian classifier, not posterior sampling.  The reference class has
    zero logit.  A null-space contrast parameterization forces family
    deviations to have zero weighted mean in the training fold.
    """

    def __init__(
        self,
        *,
        numeric_columns: Sequence[str] = (),
        categorical_columns: Sequence[str] = (),
        family_column: str = HIERARCHICAL_FAMILY_COLUMN,
        expected_families: Sequence[str] = HIERARCHICAL_FAMILIES,
        class_names: Sequence[str] = CARDINAL_CLASSES,
        global_feature_scale: float = 2.0,
        global_intercept_scale: float = 5.0,
        family_feature_scale: float = 0.5,
        family_intercept_scale: float = 1.0,
        max_iter: int = 1000,
        tolerance: float = 1e-8,
        prior_smoothing: float = 1e-6,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.family_column = family_column
        self.expected_families = expected_families
        self.class_names = class_names
        self.global_feature_scale = global_feature_scale
        self.global_intercept_scale = global_intercept_scale
        self.family_feature_scale = family_feature_scale
        self.family_intercept_scale = family_intercept_scale
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.prior_smoothing = prior_smoothing

    def _validate_columns(
        self, frame: pd.DataFrame
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        family_column = str(self.family_column)
        if family_column not in frame:
            raise KeyError(
                f"Hierarchical family column {family_column!r} is missing"
            )
        numeric = tuple(map(str, self.numeric_columns))
        categorical = tuple(map(str, self.categorical_columns))
        overlap = sorted(set(numeric).intersection(categorical))
        typed = set(numeric + categorical)
        unknown = sorted(typed - set(frame.columns))
        expected = set(frame.columns) - {family_column}
        untyped = sorted(expected - typed)
        extra = sorted(typed - expected)
        if overlap or unknown or untyped or extra:
            raise ValueError(
                "numeric_columns and categorical_columns must be disjoint and "
                "exhaustive after excluding the family column; "
                f"overlap={overlap}, unknown={unknown}, untyped={untyped}, "
                f"extra={extra}"
            )
        return numeric, categorical

    def _family_indices(
        self, frame: pd.DataFrame, *, fitting: bool
    ) -> np.ndarray:
        values = (
            frame[str(self.family_column)]
            .where(frame[str(self.family_column)].notna(), "__MISSING__")
            .astype(str)
            .to_numpy(object)
        )
        lookup = {
            str(name): index
            for index, name in enumerate(self.family_levels_)
        }
        indices = np.asarray(
            [lookup.get(str(value), -1) for value in values], dtype=int
        )
        if fitting and np.any(indices < 0):
            unknown = sorted(
                {str(value) for value, index in zip(values, indices) if index < 0}
            )
            raise ValueError(
                f"Training data contain unexpected catalyst families: {unknown}"
            )
        return indices

    def _unpack(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        global_size = self.design_width_ * self.nonreference_classes_
        global_coef = theta[:global_size].reshape(
            self.design_width_, self.nonreference_classes_
        )
        contrast_coef = theta[global_size:].reshape(
            self.n_family_contrasts_,
            self.design_width_,
            self.nonreference_classes_,
        )
        family_deviation = np.einsum(
            "gh,hdk->gdk",
            self.family_contrast_basis_,
            contrast_coef,
            optimize=True,
        )
        return global_coef, contrast_coef, family_deviation

    def _objective_and_gradient(
        self,
        theta: np.ndarray,
        design: np.ndarray,
        family_indices: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        global_coef, contrast_coef, family_deviation = self._unpack(theta)
        selected_deviation = family_deviation[family_indices]
        logits_nonreference = (
            design @ global_coef
            + np.einsum(
                "nd,ndk->nk",
                design,
                selected_deviation,
                optimize=True,
            )
        )
        logits = np.column_stack(
            [logits_nonreference, np.zeros(len(design), dtype=float)]
        )
        log_probabilities = logits - logsumexp(
            logits, axis=1, keepdims=True
        )
        probabilities = np.exp(log_probabilities)
        objective = -float(
            np.sum(weights[:, None] * targets * log_probabilities)
        )
        residual = (
            weights[:, None]
            * (probabilities - targets)[:, : self.nonreference_classes_]
        )
        gradient_global = design.T @ residual
        gradient_family = np.zeros_like(family_deviation)
        for family_index in range(len(self.family_levels_)):
            selected = family_indices == family_index
            if selected.any():
                gradient_family[family_index] = (
                    design[selected].T @ residual[selected]
                )
        gradient_contrast = np.einsum(
            "gh,gdk->hdk",
            self.family_contrast_basis_,
            gradient_family,
            optimize=True,
        )

        global_intercept_scale = float(self.global_intercept_scale_)
        global_feature_scale = float(self.global_feature_scale_)
        family_intercept_scale = float(self.family_intercept_scale_)
        family_feature_scale = float(self.family_feature_scale_)
        objective += 0.5 * float(
            np.sum(np.square(global_coef[0] / global_intercept_scale))
        )
        gradient_global[0] += (
            global_coef[0] / global_intercept_scale**2
        )
        if self.design_width_ > 1:
            objective += 0.5 * float(
                np.sum(np.square(global_coef[1:] / global_feature_scale))
            )
            gradient_global[1:] += (
                global_coef[1:] / global_feature_scale**2
            )
        objective += 0.5 * float(
            np.sum(
                np.square(
                    contrast_coef[:, 0, :] / family_intercept_scale
                )
            )
        )
        gradient_contrast[:, 0, :] += (
            contrast_coef[:, 0, :] / family_intercept_scale**2
        )
        if self.design_width_ > 1:
            objective += 0.5 * float(
                np.sum(
                    np.square(
                        contrast_coef[:, 1:, :] / family_feature_scale
                    )
                )
            )
            gradient_contrast[:, 1:, :] += (
                contrast_coef[:, 1:, :] / family_feature_scale**2
            )
        gradient = np.concatenate(
            [gradient_global.ravel(), gradient_contrast.ravel()]
        )
        return objective, gradient

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "HierarchicalGaussianMAPSoftmaxClassifier":
        frame = _as_frame(X)
        names = _class_names(self.class_names)
        numeric, categorical = self._validate_columns(frame)
        targets = _soft_targets(y, n_rows=len(frame), class_names=names)
        weights = _sample_weights(sample_weight, len(frame))
        positive = weights > 0
        frame = frame.loc[positive].reset_index(drop=True)
        targets = targets[positive]
        weights = weights[positive]

        expected = tuple(map(str, self.expected_families))
        if len(expected) < 2 or len(set(expected)) != len(expected):
            raise ValueError(
                "expected_families must contain at least two unique families"
            )
        observed = set(frame[str(self.family_column)].astype(str))
        missing_families = sorted(set(expected) - observed)
        unexpected = sorted(observed - set(expected))
        if missing_families or unexpected:
            raise ValueError(
                "Every training fold must contain the registered catalyst "
                f"families; missing={missing_families}, unexpected={unexpected}"
            )
        self.family_levels_ = np.asarray(expected, dtype=object)
        family_indices = self._family_indices(frame, fitting=True)

        self.global_feature_scale_ = _positive_scale(
            self.global_feature_scale, "global_feature_scale"
        )
        self.global_intercept_scale_ = _positive_scale(
            self.global_intercept_scale, "global_intercept_scale"
        )
        self.family_feature_scale_ = _positive_scale(
            self.family_feature_scale, "family_feature_scale"
        )
        self.family_intercept_scale_ = _positive_scale(
            self.family_intercept_scale, "family_intercept_scale"
        )
        tolerance = _positive_scale(self.tolerance, "tolerance")
        smoothing = float(self.prior_smoothing)
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("prior_smoothing must be finite and positive")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be positive")

        self.preprocessor_ = _preprocessor(numeric, categorical)
        transformed = np.asarray(
            self.preprocessor_.fit_transform(frame), dtype=float
        )
        if transformed.ndim != 2 or not np.isfinite(transformed).all():
            raise ValueError("Fold-local transformed predictors are not finite")
        design = np.column_stack(
            [np.ones(len(transformed), dtype=float), transformed]
        )
        family_mass = np.bincount(
            family_indices,
            weights=weights,
            minlength=len(self.family_levels_),
        ).astype(float)
        family_proportion = family_mass / family_mass.sum()
        contrast_basis = null_space(family_proportion[None, :])
        if contrast_basis.shape != (
            len(self.family_levels_),
            len(self.family_levels_) - 1,
        ):
            raise RuntimeError(
                "Failed to construct family-centering contrast basis"
            )

        self.classes_ = np.asarray(names, dtype=object)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = int(frame.shape[1])
        self.predictor_names_in_ = np.asarray(
            numeric + categorical, dtype=object
        )
        self.transformed_feature_names_ = np.asarray(
            self.preprocessor_.get_feature_names_out(), dtype=object
        )
        self.design_width_ = int(design.shape[1])
        self.nonreference_classes_ = len(names) - 1
        self.n_family_contrasts_ = len(self.family_levels_) - 1
        self.family_weights_ = family_proportion
        self.family_contrast_basis_ = contrast_basis

        prior_mass = np.sum(targets * weights[:, None], axis=0) + smoothing
        prior = prior_mass / prior_mass.sum()
        initial_global = np.zeros(
            (self.design_width_, self.nonreference_classes_), dtype=float
        )
        initial_global[0] = np.log(
            prior[: self.nonreference_classes_] / prior[-1]
        )
        initial_contrast = np.zeros(
            (
                self.n_family_contrasts_,
                self.design_width_,
                self.nonreference_classes_,
            ),
            dtype=float,
        )
        initial = np.concatenate(
            [initial_global.ravel(), initial_contrast.ravel()]
        )
        result = minimize(
            self._objective_and_gradient,
            initial,
            args=(design, family_indices, targets, weights),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(self.max_iter),
                "ftol": tolerance,
                "gtol": tolerance,
                "maxls": 50,
            },
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(
                "Hierarchical Gaussian-prior MAP optimization returned "
                "non-finite parameters"
            )
        (
            global_coef,
            contrast_coef,
            family_deviation,
        ) = self._unpack(np.asarray(result.x, dtype=float))
        self.global_intercept_ = global_coef[0].copy()
        self.global_coef_ = global_coef[1:].copy()
        self.family_intercept_deviation_ = family_deviation[:, 0, :].copy()
        self.family_coef_deviation_ = family_deviation[:, 1:, :].copy()
        self.family_contrast_coef_ = contrast_coef.copy()
        self.optimizer_success_ = bool(result.success)
        self.optimizer_status_ = int(result.status)
        self.optimizer_message_ = str(result.message)
        self.optimizer_iterations_ = int(result.nit)
        self.map_objective_ = float(result.fun)
        self.n_training_records_ = int(len(frame))
        self.total_fractional_weight_ = float(weights.sum())
        self.posterior_draws_expanded_ = False
        self.inference_method_ = (
            "hierarchical_gaussian_prior_MAP_scipy_L-BFGS-B"
        )
        return self

    def _linear_predictor(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = np.asarray(
            self.preprocessor_.transform(frame), dtype=float
        )
        design = np.column_stack(
            [np.ones(len(transformed), dtype=float), transformed]
        )
        global_coef = np.vstack(
            [self.global_intercept_[None, :], self.global_coef_]
        )
        family_deviation = np.concatenate(
            [
                self.family_intercept_deviation_[:, None, :],
                self.family_coef_deviation_,
            ],
            axis=1,
        )
        indices = self._family_indices(frame, fitting=False)
        logits = design @ global_coef
        known = indices >= 0
        if known.any():
            logits[known] += np.einsum(
                "nd,ndk->nk",
                design[known],
                family_deviation[indices[known]],
                optimize=True,
            )
        return logits

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "preprocessor_"):
            raise RuntimeError(
                "HierarchicalGaussianMAPSoftmaxClassifier must be fitted first"
            )
        frame = _as_frame(X, expected_columns=self.feature_names_in_)
        logits_nonreference = self._linear_predictor(frame)
        logits = np.column_stack(
            [logits_nonreference, np.zeros(len(frame), dtype=float)]
        )
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def predict(self, X: Any) -> np.ndarray:
        probability = self.predict_proba(X)
        return self.classes_[np.argmax(probability, axis=1)]

    def family_parameter_frame(self) -> pd.DataFrame:
        """Return MAP family deviations for auditing partial pooling."""

        if not hasattr(self, "family_coef_deviation_"):
            raise RuntimeError("Estimator must be fitted first")
        feature_names = ("__intercept__", *map(str, self.transformed_feature_names_))
        deviations = np.concatenate(
            [
                self.family_intercept_deviation_[:, None, :],
                self.family_coef_deviation_,
            ],
            axis=1,
        )
        rows: list[dict[str, object]] = []
        for family_index, family in enumerate(self.family_levels_):
            for feature_index, feature in enumerate(feature_names):
                for class_index, class_name in enumerate(self.classes_[:-1]):
                    rows.append(
                        {
                            "catalyst_family": str(family),
                            "feature": str(feature),
                            "class_vs_reference": str(class_name),
                            "reference_class": str(self.classes_[-1]),
                            "map_deviation": float(
                                deviations[
                                    family_index, feature_index, class_index
                                ]
                            ),
                            "inference_method": self.inference_method_,
                        }
                    )
        return pd.DataFrame(rows)


class HierarchicalMAPCardinalClassifier(ClassifierMixin, BaseEstimator):
    """Fold-local two-stage partially pooled cardinal classifier.

    A binary cardinal/off-cardinal gate and a conditional low/60/120 model are
    fitted separately.  Both stages use the same Gaussian-prior MAP hierarchy.
    """

    def __init__(
        self,
        *,
        feature_recipe: str = "context_log",
        global_feature_scale: float = 2.0,
        global_intercept_scale: float = 5.0,
        family_feature_scale: float = 0.5,
        family_intercept_scale: float = 1.0,
        max_iter: int = 1000,
        tolerance: float = 1e-8,
    ) -> None:
        self.feature_recipe = feature_recipe
        self.global_feature_scale = global_feature_scale
        self.global_intercept_scale = global_intercept_scale
        self.family_feature_scale = family_feature_scale
        self.family_intercept_scale = family_intercept_scale
        self.max_iter = max_iter
        self.tolerance = tolerance

    def _engineered(
        self, X: pd.DataFrame, *, fitting: bool
    ) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
        if "catalyst_type" not in X:
            raise KeyError(
                "Hierarchical cardinal model requires persisted catalyst_type"
            )
        families = canonical_catalyst_family(X["catalyst_type"])
        if fitting:
            self.feature_builder_ = CardinalFeatureBuilder(
                self.feature_recipe
            ).fit(X)
        engineered = self.feature_builder_.transform(X).copy()
        # Family membership is represented only through the hierarchy.  This
        # avoids duplicating the same Ru/Ir contrast as an ordinary one-hot
        # predictor.  The original catalyst_type remains in the input registry.
        if "catalyst_type" in engineered:
            engineered = engineered.drop(columns=["catalyst_type"])
        engineered[HIERARCHICAL_FAMILY_COLUMN] = families
        numeric = tuple(
            name
            for name in self.feature_builder_.numeric_columns_out_
            if name in engineered
        )
        categorical = tuple(
            name
            for name in self.feature_builder_.categorical_columns_out_
            if name in engineered and name != "catalyst_type"
        )
        expected = set(engineered.columns) - {HIERARCHICAL_FAMILY_COLUMN}
        if set(numeric + categorical) != expected:
            raise AssertionError(
                "Hierarchical engineered predictor typing is not exhaustive"
            )
        return engineered, numeric, categorical

    def _stage(
        self,
        class_names: Sequence[str],
        numeric: Sequence[str],
        categorical: Sequence[str],
    ) -> HierarchicalGaussianMAPSoftmaxClassifier:
        return HierarchicalGaussianMAPSoftmaxClassifier(
            numeric_columns=tuple(numeric),
            categorical_columns=tuple(categorical),
            family_column=HIERARCHICAL_FAMILY_COLUMN,
            expected_families=HIERARCHICAL_FAMILIES,
            class_names=tuple(class_names),
            global_feature_scale=float(self.global_feature_scale),
            global_intercept_scale=float(self.global_intercept_scale),
            family_feature_scale=float(self.family_feature_scale),
            family_intercept_scale=float(self.family_intercept_scale),
            max_iter=int(self.max_iter),
            tolerance=float(self.tolerance),
        )

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "HierarchicalMAPCardinalClassifier":
        frame = _as_frame(X)
        targets = _soft_targets(
            y, n_rows=len(frame), class_names=CARDINAL_CLASSES
        )
        weights = _sample_weights(sample_weight, len(frame))
        engineered, numeric, categorical = self._engineered(
            frame, fitting=True
        )
        cardinal_mass = targets[:, :-1].sum(axis=1)
        gate_targets = np.column_stack(
            [cardinal_mass, targets[:, -1]]
        )
        self.gate_estimator_ = self._stage(
            CARDINAL_GATE_CLASSES, numeric, categorical
        ).fit(engineered, gate_targets, sample_weight=weights)
        eligible = cardinal_mass > 1e-12
        if not eligible.any():
            raise ValueError(
                "Conditional cardinal hierarchy has no posterior mass"
            )
        conditional_targets = (
            targets[eligible, :-1] / cardinal_mass[eligible, None]
        )
        conditional_weights = weights[eligible] * cardinal_mass[eligible]
        self.conditional_estimator_ = self._stage(
            CLASSIC_CARDINAL_CLASSES, numeric, categorical
        ).fit(
            engineered.loc[eligible].reset_index(drop=True),
            conditional_targets,
            sample_weight=conditional_weights,
        )
        self.classes_ = np.asarray(CARDINAL_CLASSES, dtype=object)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = int(frame.shape[1])
        self.engineered_feature_names_ = tuple(
            name
            for name in engineered.columns
            if name != HIERARCHICAL_FAMILY_COLUMN
        )
        self.posterior_draws_expanded_ = False
        self.inference_method_ = (
            "two_stage_hierarchical_gaussian_prior_MAP"
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "gate_estimator_"):
            raise RuntimeError(
                "HierarchicalMAPCardinalClassifier must be fitted first"
            )
        frame = _as_frame(X, expected_columns=self.feature_names_in_)
        engineered, _, _ = self._engineered(frame, fitting=False)
        gate = self.gate_estimator_.predict_proba(engineered)
        conditional = self.conditional_estimator_.predict_proba(engineered)
        combined = np.column_stack(
            [conditional * gate[:, [0]], gate[:, 1]]
        )
        return combined / combined.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        probability = self.predict_proba(X)
        return self.classes_[np.argmax(probability, axis=1)]

    def convergence_summary(self) -> dict[str, object]:
        if not hasattr(self, "gate_estimator_"):
            raise RuntimeError("Estimator must be fitted first")
        return {
            "inference_method": self.inference_method_,
            "posterior_sampling": False,
            "posterior_draws_expanded": False,
            "gate_optimizer_success": bool(
                self.gate_estimator_.optimizer_success_
            ),
            "conditional_optimizer_success": bool(
                self.conditional_estimator_.optimizer_success_
            ),
            "gate_optimizer_message": str(
                self.gate_estimator_.optimizer_message_
            ),
            "conditional_optimizer_message": str(
                self.conditional_estimator_.optimizer_message_
            ),
            "gate_map_objective": float(
                self.gate_estimator_.map_objective_
            ),
            "conditional_map_objective": float(
                self.conditional_estimator_.map_objective_
            ),
            "gate_training_record_weight": float(
                self.gate_estimator_.total_fractional_weight_
            ),
            "conditional_training_record_weight": float(
                self.conditional_estimator_.total_fractional_weight_
            ),
        }


@dataclass(frozen=True)
class HierarchicalMAPCandidate:
    """One preregistered family-shrinkage setting."""

    candidate_id: str
    family_feature_scale: float
    family_intercept_scale: float
    complexity_rank: int


@dataclass
class HierarchicalNestedResult:
    """Nested connected-group OOF outputs for the MAP hierarchy."""

    split_manifest: pd.DataFrame
    inner_split_manifest: pd.DataFrame
    inner_candidate_scores: pd.DataFrame
    outer_selections: pd.DataFrame
    oof_probabilities: pd.DataFrame
    oof_aggregated: pd.DataFrame
    aggregate_metrics: pd.DataFrame


def default_hierarchical_candidates() -> tuple[HierarchicalMAPCandidate, ...]:
    """Finite inner-CV grid; smaller scales encode stronger pooling."""

    return (
        HierarchicalMAPCandidate(
            "strong_pooling", 0.25, 0.5, complexity_rank=0
        ),
        HierarchicalMAPCandidate(
            "moderate_pooling", 0.5, 1.0, complexity_rank=1
        ),
        HierarchicalMAPCandidate(
            "weak_pooling", 1.0, 2.0, complexity_rank=2
        ),
    )


def _frozen_inner_splits(
    train: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    outer_repeat: int,
    outer_fold: int,
    inner_folds: int,
    record_id_col: str,
    group_col: str,
    relation_columns: Sequence[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    required = {
        "outer_repeat",
        "outer_fold",
        "inner_fold",
        "record_id",
        "cv_group_id",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"Frozen inner manifest is missing required columns: {missing}"
        )
    selected = manifest.loc[
        manifest["outer_repeat"].eq(outer_repeat)
        & manifest["outer_fold"].eq(outer_fold)
    ].copy()
    record_ids = train[record_id_col].astype(str)
    if set(selected["record_id"].astype(str)) != set(record_ids):
        raise ValueError(
            "Frozen inner manifest record set differs from outer training data"
        )
    fold_lookup = (
        selected.assign(_record_key=selected["record_id"].astype(str))
        .set_index("_record_key")["inner_fold"]
        .astype(int)
    )
    assignments = np.asarray(
        [int(fold_lookup.loc[value]) for value in record_ids], dtype=int
    )
    if sorted(np.unique(assignments).tolist()) != list(range(inner_folds)):
        raise ValueError("Frozen inner manifest has unexpected fold labels")
    expected_groups = train[group_col].astype(str).to_numpy()
    recorded_groups = (
        selected.assign(_record_key=selected["record_id"].astype(str))
        .set_index("_record_key")["cv_group_id"]
        .astype(str)
    )
    if any(
        recorded_groups.loc[record_id] != group
        for record_id, group in zip(record_ids, expected_groups)
    ):
        raise ValueError(
            "Frozen inner manifest connected groups disagree with training data"
        )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for inner_fold in range(inner_folds):
        validation = np.flatnonzero(assignments == inner_fold)
        training = np.flatnonzero(assignments != inner_fold)
        assert_no_relational_leakage(
            train.iloc[training],
            train.iloc[validation],
            group_col=group_col,
            relation_columns=relation_columns,
        )
        splits.append((training, validation))
    return splits, assignments


def _one_se_pooling_selection(scores: pd.DataFrame) -> pd.Series:
    valid = scores.loc[
        scores["status"].eq("ok")
        & np.isfinite(scores["inner_soft_log_loss_mean"])
    ].copy()
    if valid.empty:
        raise RuntimeError(
            "Every hierarchical MAP shrinkage candidate failed: "
            + json.dumps(
                scores.loc[:, ["candidate_id", "error"]].to_dict("records")
            )
        )
    best = valid.sort_values(
        ["inner_soft_log_loss_mean", "candidate_id"], kind="stable"
    ).iloc[0]
    threshold = float(
        best["inner_soft_log_loss_mean"]
        + best["inner_soft_log_loss_se"]
    )
    return (
        valid.loc[
            valid["inner_soft_log_loss_mean"] <= threshold + 1e-15
        ]
        .sort_values(
            [
                "complexity_rank",
                "inner_soft_log_loss_mean",
                "candidate_id",
            ],
            kind="stable",
        )
        .iloc[0]
    )


def run_hierarchical_map_nested_oof(
    frame: pd.DataFrame,
    *,
    feature_recipe: str,
    output_dir: str | Path | None = None,
    outer_folds: int = 5,
    outer_repeats: int = 10,
    inner_folds: int = 4,
    random_seed: int = 42,
    outer_seeds: Sequence[int] | None = None,
    candidates: Sequence[HierarchicalMAPCandidate] | None = None,
    global_feature_scale: float = 2.0,
    global_intercept_scale: float = 5.0,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
    record_id_col: str = "record_id",
    group_col: str = "cv_group_id",
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
    frozen_outer_manifest: pd.DataFrame | None = None,
    frozen_inner_manifest: pd.DataFrame | None = None,
    analysis_id: str = "hierarchical_map_ru_ir",
    scope: str = "alkaline_RuO2_and_IrO2_plus_Ir",
) -> HierarchicalNestedResult:
    """Run nested connected-group OOF evaluation of one fixed feature recipe.

    Only shrinkage strength is chosen in grouped inner CV.  The feature recipe
    is fixed by the caller and is never searched.  Supplying the frozen outer
    and inner manifests from another analysis makes comparisons exactly
    fold-paired.
    """

    required = {
        record_id_col,
        group_col,
        "catalyst_type",
        *SOFT_LABEL_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"Hierarchical MAP evaluation is missing columns: {missing}"
        )
    if (frozen_outer_manifest is None) != (frozen_inner_manifest is None):
        raise ValueError(
            "Frozen outer and inner manifests must be supplied together"
        )
    data = frame.reset_index(drop=True).copy()
    canonical = canonical_catalyst_family(data["catalyst_type"])
    observed_families = set(map(str, canonical))
    if observed_families != set(HIERARCHICAL_FAMILIES):
        raise ValueError(
            "Joint hierarchy requires both RuO2 and IrO2_plus_Ir; "
            f"observed={sorted(observed_families)}"
        )
    if data[record_id_col].astype(str).duplicated().any():
        raise ValueError("Hierarchical MAP evaluation requires unique records")
    y_all = data.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
    _soft_targets(y_all, n_rows=len(data), class_names=CARDINAL_CLASSES)

    if frozen_outer_manifest is None:
        outer_manifest = make_soft_stratified_group_manifest(
            data,
            SOFT_LABEL_COLUMNS,
            group_col=group_col,
            record_id_col=record_id_col,
            folds=int(outer_folds),
            repeats=int(outer_repeats),
            random_seed=int(random_seed),
            repeat_seeds=outer_seeds,
            relation_columns=relation_columns,
        )
    else:
        outer_manifest = frozen_outer_manifest.copy()
        validate_split_manifest(
            data,
            outer_manifest,
            group_col=group_col,
            record_id_col=record_id_col,
            expected_repeats=int(outer_repeats),
            expected_folds=int(outer_folds),
            relation_columns=relation_columns,
        )
    candidate_rows = tuple(candidates or default_hierarchical_candidates())
    if not candidate_rows:
        raise ValueError("At least one hierarchical MAP candidate is required")
    if len({row.candidate_id for row in candidate_rows}) != len(candidate_rows):
        raise ValueError("Hierarchical MAP candidate IDs must be unique")
    id_to_position = {
        str(record_id): position
        for position, record_id in enumerate(data[record_id_col])
    }
    inner_manifest_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []

    for repeat in sorted(outer_manifest["repeat"].unique()):
        repeat_rows = outer_manifest.loc[
            outer_manifest["repeat"].eq(repeat)
        ]
        for fold in sorted(repeat_rows["fold"].unique()):
            test_ids = set(
                repeat_rows.loc[
                    repeat_rows["fold"].eq(fold), "record_id"
                ].astype(str)
            )
            test_positions = np.asarray(
                sorted(id_to_position[value] for value in test_ids),
                dtype=int,
            )
            train_positions = np.asarray(
                sorted(
                    set(range(len(data))) - set(test_positions.tolist())
                ),
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
                int(random_seed) + int(repeat) * 10_000 + int(fold) * 100
            )
            if frozen_inner_manifest is None:
                inner_splits, assignments = make_inner_soft_group_splits(
                    train,
                    SOFT_LABEL_COLUMNS,
                    group_col=group_col,
                    folds=int(inner_folds),
                    random_seed=inner_seed,
                    relation_columns=relation_columns,
                )
            else:
                inner_splits, assignments = _frozen_inner_splits(
                    train,
                    frozen_inner_manifest,
                    outer_repeat=int(repeat),
                    outer_fold=int(fold),
                    inner_folds=int(inner_folds),
                    record_id_col=record_id_col,
                    group_col=group_col,
                    relation_columns=relation_columns,
                )
            for position, assignment in enumerate(assignments):
                inner_manifest_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "outer_repeat": int(repeat),
                        "outer_fold": int(fold),
                        "inner_fold": int(assignment),
                        "record_id": str(
                            train.iloc[position][record_id_col]
                        ),
                        "cv_group_id": str(
                            train.iloc[position][group_col]
                        ),
                        "inner_seed": inner_seed,
                    }
                )
            X_train = train.loc[
                :,
                [
                    column
                    for column in train.columns
                    if column not in SOFT_LABEL_COLUMNS
                ],
            ]
            y_train = train.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
            fold_scores: list[dict[str, object]] = []
            for candidate in candidate_rows:
                losses: list[float] = []
                error_text = ""
                try:
                    for inner_train, inner_validation in inner_splits:
                        estimator = HierarchicalMAPCardinalClassifier(
                            feature_recipe=feature_recipe,
                            global_feature_scale=global_feature_scale,
                            global_intercept_scale=global_intercept_scale,
                            family_feature_scale=(
                                candidate.family_feature_scale
                            ),
                            family_intercept_scale=(
                                candidate.family_intercept_scale
                            ),
                            max_iter=max_iter,
                            tolerance=tolerance,
                        ).fit(
                            X_train.iloc[inner_train],
                            y_train[inner_train],
                        )
                        predicted = estimator.predict_proba(
                            X_train.iloc[inner_validation]
                        )
                        losses.append(
                            soft_cross_entropy(
                                y_train[inner_validation], predicted
                            )
                        )
                    status = "ok"
                except Exception as error:
                    status = "failed"
                    error_text = repr(error)
                values = np.asarray(losses, dtype=float)
                row = {
                    "analysis_id": analysis_id,
                    "scope": scope,
                    "feature_recipe": feature_recipe,
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "candidate_id": candidate.candidate_id,
                    "family_feature_scale": float(
                        candidate.family_feature_scale
                    ),
                    "family_intercept_scale": float(
                        candidate.family_intercept_scale
                    ),
                    "complexity_rank": int(candidate.complexity_rank),
                    "n_inner_folds": int(len(inner_splits)),
                    "inner_soft_log_loss_mean": (
                        float(values.mean()) if len(values) else np.nan
                    ),
                    "inner_soft_log_loss_sd": (
                        float(values.std(ddof=1))
                        if len(values) > 1
                        else 0.0
                        if len(values)
                        else np.nan
                    ),
                    "inner_soft_log_loss_se": (
                        float(values.std(ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else 0.0
                        if len(values)
                        else np.nan
                    ),
                    "status": status,
                    "error": error_text,
                }
                fold_scores.append(row)
                score_rows.append(row)
            selected = _one_se_pooling_selection(pd.DataFrame(fold_scores))
            candidate = next(
                item
                for item in candidate_rows
                if item.candidate_id == selected["candidate_id"]
            )
            estimator = HierarchicalMAPCardinalClassifier(
                feature_recipe=feature_recipe,
                global_feature_scale=global_feature_scale,
                global_intercept_scale=global_intercept_scale,
                family_feature_scale=candidate.family_feature_scale,
                family_intercept_scale=candidate.family_intercept_scale,
                max_iter=max_iter,
                tolerance=tolerance,
            ).fit(X_train, y_train)
            X_test = test.loc[
                :,
                [
                    column
                    for column in test.columns
                    if column not in SOFT_LABEL_COLUMNS
                ],
            ]
            predicted = estimator.predict_proba(X_test)
            convergence = estimator.convergence_summary()
            selection_rows.append(
                {
                    "analysis_id": analysis_id,
                    "scope": scope,
                    "feature_recipe": feature_recipe,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "selected_candidate_id": candidate.candidate_id,
                    "family_feature_scale": float(
                        candidate.family_feature_scale
                    ),
                    "family_intercept_scale": float(
                        candidate.family_intercept_scale
                    ),
                    "selection_rule": (
                        "one_standard_error_strongest_pooling"
                    ),
                    "inner_soft_log_loss_mean": float(
                        selected["inner_soft_log_loss_mean"]
                    ),
                    "inner_soft_log_loss_se": float(
                        selected["inner_soft_log_loss_se"]
                    ),
                    **convergence,
                }
            )
            y_test = test.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
            for local_index, (_, record) in enumerate(test.iterrows()):
                row: dict[str, object] = {
                    "analysis_id": analysis_id,
                    "scope": scope,
                    "panel": feature_recipe,
                    "model_role": "fixed_hierarchical_MAP",
                    "model_family": "hierarchical_gaussian_prior_MAP",
                    "selected_candidate_id": candidate.candidate_id,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "record_id": str(record[record_id_col]),
                    "cv_group_id": str(record[group_col]),
                    "catalyst_family": str(
                        canonical_catalyst_family(
                            [record["catalyst_type"]]
                        )[0]
                    ),
                }
                for column, value in zip(
                    SOFT_LABEL_COLUMNS, y_test[local_index], strict=True
                ):
                    row[column] = float(value)
                for column, value in zip(
                    PREDICTION_COLUMNS, predicted[local_index], strict=True
                ):
                    row[column] = float(value)
                oof_rows.append(row)

    inner_manifest_frame = pd.DataFrame(inner_manifest_rows)
    inner_scores = pd.DataFrame(score_rows)
    selections = pd.DataFrame(selection_rows)
    oof = pd.DataFrame(oof_rows)
    aggregated = aggregate_repeated_oof(
        oof,
        key_columns=(
            "analysis_id",
            "scope",
            "panel",
            "model_role",
            "model_family",
            "record_id",
            "cv_group_id",
            "catalyst_family",
        ),
    )
    observed = aggregated.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
    predicted = aggregated.loc[:, PREDICTION_COLUMNS].to_numpy(float)
    metrics = pd.DataFrame(
        [
            {
                "analysis_id": analysis_id,
                "scope": scope,
                "feature_recipe": feature_recipe,
                "model_family": "hierarchical_gaussian_prior_MAP",
                "n_records": int(len(aggregated)),
                "n_connected_groups": int(
                    aggregated["cv_group_id"].nunique()
                ),
                **two_stage_probability_metrics(observed, predicted),
                **collapsed_hard_metrics(observed, predicted),
            }
        ]
    )
    if output_dir is not None:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        outer_manifest.to_csv(output / "outer_split_manifest.csv", index=False)
        inner_manifest_frame.to_csv(
            output / "inner_split_manifest.csv", index=False
        )
        inner_scores.to_csv(
            output / "inner_candidate_scores.csv", index=False
        )
        selections.to_csv(output / "outer_selections.csv", index=False)
        oof.to_csv(output / "oof_probabilities.csv", index=False)
        aggregated.to_csv(output / "oof_aggregated.csv", index=False)
        metrics.to_csv(output / "aggregate_metrics.csv", index=False)
        (output / "model_definition.json").write_text(
            json.dumps(
                {
                    "model": "two_stage_hierarchical_gaussian_prior_MAP",
                    "approximation": (
                        "posterior mode via SciPy L-BFGS-B; no NUTS/MCMC"
                    ),
                    "feature_recipe": feature_recipe,
                    "families": list(HIERARCHICAL_FAMILIES),
                    "metallic_Ir_mapping": "IrO2_plus_Ir",
                    "posterior_draws_expanded": False,
                    "one_fractional_response_per_record": True,
                    "global_feature_scale": float(global_feature_scale),
                    "global_intercept_scale": float(
                        global_intercept_scale
                    ),
                    "shrinkage_selection": (
                        "connected-group inner CV, one-SE strongest pooling"
                    ),
                    "outer_performance_source": (
                        "connected-group outer OOF aggregated once per record"
                    ),
                    "mechanistic_claim_allowed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return HierarchicalNestedResult(
        split_manifest=outer_manifest,
        inner_split_manifest=inner_manifest_frame,
        inner_candidate_scores=inner_scores,
        outer_selections=selections,
        oof_probabilities=oof,
        oof_aggregated=aggregated,
        aggregate_metrics=metrics,
    )


__all__ = [
    "CATALYST_TO_HIERARCHICAL_FAMILY",
    "HIERARCHICAL_FAMILIES",
    "HIERARCHICAL_FAMILY_COLUMN",
    "HierarchicalGaussianMAPSoftmaxClassifier",
    "HierarchicalMAPCardinalClassifier",
    "HierarchicalMAPCandidate",
    "HierarchicalNestedResult",
    "canonical_catalyst_family",
    "default_hierarchical_candidates",
    "run_hierarchical_map_nested_oof",
]
