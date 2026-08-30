"""Reduced-panel classifiers for the all-97 posterior-soft cardinal target.

The estimators in this module preserve the registered experimental-record
grain.  A row contributes one fractional class-probability vector to each
objective; neither classes nor posterior slope draws are expanded into
pseudo-observations.

``GaussianMAPSoftmaxClassifier`` is a deterministic Gaussian-prior MAP fit,
not full Bayesian posterior inference.  Its posterior mode is obtained with
SciPy L-BFGS-B.  The two reduced-panel wrappers fit
``AssociationFeatureBuilder`` inside every call to :meth:`fit`, keeping all
feature engineering and preprocessing local to the supplied training fold.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cardinal_models import (
    CARDINAL_CLASSES,
    CARDINAL_GATE_CLASSES,
    CLASSIC_CARDINAL_CLASSES,
    SoftGowerKNeighborsClassifier,
    TwoStageCardinalClassifier,
)
from .feature_association_features import (
    AssociationFeatureBuilder,
    AssociationFeaturePanel,
)


def _as_frame(
    X: Any, expected_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy(deep=False)
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        columns = (
            list(expected_columns)
            if expected_columns is not None
            else [f"x{index}" for index in range(array.shape[1])]
        )
        if len(columns) != array.shape[1]:
            raise ValueError(
                f"X has {array.shape[1]} columns but {len(columns)} fitted "
                "feature names were supplied"
            )
        frame = pd.DataFrame(array, columns=columns)
    if expected_columns is not None:
        expected = list(map(str, expected_columns))
        missing = sorted(set(expected) - set(map(str, frame.columns)))
        if missing:
            raise ValueError(
                f"Prediction table is missing fitted features: {missing}"
            )
        frame = frame.loc[:, expected]
    return frame


def _class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(map(str, class_names))
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
            selected = plain
        elif all(column in y.columns for column in prefixed):
            selected = prefixed
        else:
            raise ValueError(
                "Soft-target table must contain either plain or q_-prefixed "
                f"class columns for {plain}"
            )
        values = y.loc[:, selected].to_numpy(dtype=float)
    else:
        values = np.asarray(y, dtype=float)
    expected_shape = (n_rows, len(names))
    if values.ndim != 2 or values.shape != expected_shape:
        raise ValueError(
            f"Soft targets must have shape {expected_shape}; "
            f"received {tuple(values.shape)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Soft targets must be finite")
    if np.any(values < -1e-12):
        raise ValueError("Soft-target probabilities cannot be negative")
    values = np.maximum(values, 0.0)
    totals = values.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every soft-target row must have positive mass")
    return values / totals[:, None]


def _sample_weights(sample_weight: Any, n_rows: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_rows, dtype=float)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.shape != (n_rows,):
        raise ValueError(
            f"sample_weight must have shape ({n_rows},); "
            f"received {tuple(weights.shape)}"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and nonnegative")
    if not np.any(weights > 0):
        raise ValueError("At least one sample weight must be positive")
    return weights


def _positive_float(value: Any, name: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _resolve_columns(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = tuple(map(str, numeric_columns))
    categorical = tuple(map(str, categorical_columns))
    overlap = sorted(set(numeric).intersection(categorical))
    typed = set(numeric + categorical)
    available = set(map(str, frame.columns))
    unknown = sorted(typed - available)
    untyped = sorted(available - typed)
    if overlap or unknown or untyped:
        raise ValueError(
            "numeric_columns and categorical_columns must be disjoint and "
            "exhaustive; "
            f"overlap={overlap}, unknown={unknown}, untyped={untyped}"
        )
    if not numeric and not categorical:
        raise ValueError("At least one predictor is required")
    return numeric, categorical


def _preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
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
                                strategy="median",
                                keep_empty_features=True,
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
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _normalized_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Logits must be a two-dimensional multiclass matrix")
    log_probabilities = values - logsumexp(values, axis=1, keepdims=True)
    probabilities = np.exp(log_probabilities)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Softmax produced non-finite probabilities")
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class GaussianMAPSoftmaxClassifier(ClassifierMixin, BaseEstimator):
    """Multinomial softmax MAP fit to one soft vector per record.

    The final class is the reference class with an identically zero logit.
    Feature and intercept coefficients receive independent zero-mean Gaussian
    priors with the supplied scales.  A finite L-BFGS-B solution is retained
    even when the optimizer reports non-convergence; the complete status is
    exposed through ``optimization_audit_`` and ``fit_status_``.
    """

    def __init__(
        self,
        *,
        numeric_columns: Sequence[str] = (),
        categorical_columns: Sequence[str] = (),
        class_names: Sequence[str] = CARDINAL_CLASSES,
        feature_prior_scale: float = 2.0,
        intercept_prior_scale: float = 5.0,
        max_iter: int = 1000,
        tolerance: float = 1e-8,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.class_names = class_names
        self.feature_prior_scale = feature_prior_scale
        self.intercept_prior_scale = intercept_prior_scale
        self.max_iter = max_iter
        self.tolerance = tolerance

    def _unpack(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float).reshape(
            self.design_width_, self.nonreference_classes_
        )

    def _objective_and_gradient(
        self,
        theta: np.ndarray,
        design: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        coefficients = self._unpack(theta)
        logits_nonreference = design @ coefficients
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
        gradient = design.T @ residual

        objective += 0.5 * float(
            np.sum(
                np.square(
                    coefficients[0] / self.intercept_prior_scale_
                )
            )
        )
        gradient[0] += (
            coefficients[0] / self.intercept_prior_scale_**2
        )
        if self.design_width_ > 1:
            objective += 0.5 * float(
                np.sum(
                    np.square(
                        coefficients[1:] / self.feature_prior_scale_
                    )
                )
            )
            gradient[1:] += (
                coefficients[1:] / self.feature_prior_scale_**2
            )
        return objective, gradient.ravel()

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "GaussianMAPSoftmaxClassifier":
        frame = _as_frame(X)
        names = _class_names(self.class_names)
        numeric, categorical = _resolve_columns(
            frame, self.numeric_columns, self.categorical_columns
        )
        targets = _soft_targets(
            y, n_rows=len(frame), class_names=names
        )
        weights = _sample_weights(sample_weight, len(frame))

        self.feature_prior_scale_ = _positive_float(
            self.feature_prior_scale, "feature_prior_scale"
        )
        self.intercept_prior_scale_ = _positive_float(
            self.intercept_prior_scale, "intercept_prior_scale"
        )
        tolerance = _positive_float(self.tolerance, "tolerance")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be positive")

        self.n_input_records_ = int(len(frame))
        self.soft_target_shape_ = tuple(targets.shape)
        positive = weights > 0
        frame = frame.loc[positive].reset_index(drop=True)
        targets = targets[positive]
        weights = weights[positive]

        self.preprocessor_ = _preprocessor(numeric, categorical)
        transformed = np.asarray(
            self.preprocessor_.fit_transform(frame), dtype=float
        )
        if transformed.ndim != 2 or not np.isfinite(transformed).all():
            raise ValueError("Fold-local transformed predictors are not finite")
        design = np.column_stack(
            [np.ones(len(transformed), dtype=float), transformed]
        )

        self.classes_ = np.asarray(names, dtype=object)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = int(frame.shape[1])
        self.numeric_columns_ = numeric
        self.categorical_columns_ = categorical
        self.transformed_feature_names_ = np.asarray(
            self.preprocessor_.get_feature_names_out(), dtype=object
        )
        self.design_width_ = int(design.shape[1])
        self.nonreference_classes_ = len(names) - 1

        prior_mass = np.sum(targets * weights[:, None], axis=0)
        safe_prior = np.maximum(
            prior_mass / prior_mass.sum(), np.finfo(float).eps
        )
        safe_prior /= safe_prior.sum()
        initial = np.zeros(
            (self.design_width_, self.nonreference_classes_), dtype=float
        )
        initial[0] = np.log(
            safe_prior[: self.nonreference_classes_] / safe_prior[-1]
        )

        result = minimize(
            self._objective_and_gradient,
            initial.ravel(),
            args=(design, targets, weights),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(self.max_iter),
                "ftol": tolerance,
                "gtol": tolerance,
                "maxls": 50,
            },
        )
        result_x = np.asarray(result.x, dtype=float)
        result_fun = float(result.fun)
        self.optimizer_success_ = bool(result.success)
        self.optimizer_status_ = int(result.status)
        self.optimizer_message_ = str(result.message)
        self.optimizer_iterations_ = int(getattr(result, "nit", -1))
        self.optimizer_function_evaluations_ = int(
            getattr(result, "nfev", -1)
        )
        self.optimizer_gradient_evaluations_ = int(
            getattr(result, "njev", -1)
        )
        self.map_objective_ = result_fun
        self.optimization_audit_ = {
            "success": self.optimizer_success_,
            "status": self.optimizer_status_,
            "message": self.optimizer_message_,
            "iterations": self.optimizer_iterations_,
            "function_evaluations": self.optimizer_function_evaluations_,
            "gradient_evaluations": self.optimizer_gradient_evaluations_,
            "objective": self.map_objective_,
            "method": "L-BFGS-B",
        }
        if not np.isfinite(result_fun) or not np.isfinite(result_x).all():
            self.fit_status_ = "failed_nonfinite_optimizer_output"
            raise RuntimeError(
                "Gaussian-prior MAP optimization returned non-finite output; "
                f"status={self.optimizer_status_}, "
                f"message={self.optimizer_message_!r}"
            )

        coefficients = self._unpack(result_x)
        self.intercept_ = coefficients[0].copy()
        # Follow the conventional sklearn public shape:
        # (non-reference classes, transformed features).
        self.coef_ = coefficients[1:].T.copy()
        self.n_training_records_ = int(len(frame))
        self.objective_row_count_ = int(len(design))
        self.total_fractional_weight_ = float(weights.sum())
        self.class_rows_expanded_ = False
        self.posterior_draws_expanded_ = False
        self.inference_method_ = (
            "gaussian_prior_MAP_scipy_L-BFGS-B"
        )
        self.fit_status_ = (
            "complete"
            if self.optimizer_success_
            else "optimizer_nonconverged_finite_solution"
        )
        return self

    def _linear_predictor(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = np.asarray(
            self.preprocessor_.transform(frame), dtype=float
        )
        return self.intercept_[None, :] + transformed @ self.coef_.T

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "preprocessor_"):
            raise RuntimeError(
                "GaussianMAPSoftmaxClassifier must be fitted first"
            )
        frame = _as_frame(X, expected_columns=self.feature_names_in_)
        logits_nonreference = self._linear_predictor(frame)
        logits = np.column_stack(
            [logits_nonreference, np.zeros(len(frame), dtype=float)]
        )
        return _normalized_probabilities(logits)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def _estimator_audit(estimator: BaseEstimator) -> Mapping[str, Any]:
    audit = getattr(estimator, "optimization_audit_", None)
    if audit is not None:
        return dict(audit)
    return {
        "success": True,
        "status": "conditional_prior_fallback",
        "message": "No optimizable conditional cardinal mass was available",
        "method": "training_fold_soft_prior",
    }


class ReducedPanelTwoStageMAPClassifier(ClassifierMixin, BaseEstimator):
    """Fold-local reduced panel followed by the registered two-stage MAP fit."""

    def __init__(
        self,
        *,
        panel: str | AssociationFeaturePanel = "context_core",
        rare_min_frequency: int = 3,
        feature_prior_scale: float = 2.0,
        intercept_prior_scale: float = 5.0,
        max_iter: int = 1000,
        tolerance: float = 1e-8,
        cardinal_epsilon: float = 1e-12,
    ) -> None:
        self.panel = panel
        self.rare_min_frequency = rare_min_frequency
        self.feature_prior_scale = feature_prior_scale
        self.intercept_prior_scale = intercept_prior_scale
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.cardinal_epsilon = cardinal_epsilon

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "ReducedPanelTwoStageMAPClassifier":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ReducedPanelTwoStageMAPClassifier requires a pandas DataFrame"
            )
        self.feature_builder_ = AssociationFeatureBuilder(
            self.panel,
            rare_min_frequency=self.rare_min_frequency,
        ).fit(X)
        engineered = self.feature_builder_.transform(X)
        numeric = self.feature_builder_.numeric_columns_out_
        categorical = self.feature_builder_.categorical_columns_out_

        def stage(class_names: Sequence[str]) -> GaussianMAPSoftmaxClassifier:
            return GaussianMAPSoftmaxClassifier(
                numeric_columns=numeric,
                categorical_columns=categorical,
                class_names=class_names,
                feature_prior_scale=self.feature_prior_scale,
                intercept_prior_scale=self.intercept_prior_scale,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
            )

        self.model_ = TwoStageCardinalClassifier(
            gate_estimator=stage(CARDINAL_GATE_CLASSES),
            conditional_estimator=stage(CLASSIC_CARDINAL_CLASSES),
            cardinal_epsilon=self.cardinal_epsilon,
        ).fit(engineered, y, sample_weight=sample_weight)
        self.gate_estimator_ = self.model_.gate_estimator_
        self.conditional_estimator_ = self.model_.conditional_estimator_
        self.optimization_audit_ = {
            "gate": _estimator_audit(self.gate_estimator_),
            "conditional": _estimator_audit(self.conditional_estimator_),
        }
        self.fit_status_ = (
            "complete"
            if all(
                bool(audit.get("success", False))
                for audit in self.optimization_audit_.values()
            )
            else "optimizer_nonconverged_finite_solution"
        )
        self.classes_ = np.asarray(CARDINAL_CLASSES, dtype=object)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = int(X.shape[1])
        self.engineered_feature_names_ = (
            self.feature_builder_.get_feature_names_out()
        )
        self.numeric_columns_ = tuple(numeric)
        self.categorical_columns_ = tuple(categorical)
        self.blocks_ = self.feature_builder_.get_blocks_out()
        self.n_training_records_ = int(len(X))
        self.posterior_draws_expanded_ = False
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "feature_builder_"):
            raise RuntimeError(
                "ReducedPanelTwoStageMAPClassifier must be fitted first"
            )
        engineered = self.feature_builder_.transform(X)
        probabilities = self.model_.predict_proba(engineered)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class ReducedPanelGowerClassifier(ClassifierMixin, BaseEstimator):
    """Fold-local reduced panel followed by two-stage soft Gower kNN."""

    def __init__(
        self,
        *,
        panel: str | AssociationFeaturePanel = "context_core",
        rare_min_frequency: int = 3,
        k: int = 7,
        p: float = 1.0,
        cardinal_epsilon: float = 1e-12,
        epsilon: float = 1e-12,
    ) -> None:
        self.panel = panel
        self.rare_min_frequency = rare_min_frequency
        self.k = k
        self.p = p
        self.cardinal_epsilon = cardinal_epsilon
        self.epsilon = epsilon

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "ReducedPanelGowerClassifier":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ReducedPanelGowerClassifier requires a pandas DataFrame"
            )
        self.feature_builder_ = AssociationFeatureBuilder(
            self.panel,
            rare_min_frequency=self.rare_min_frequency,
        ).fit(X)
        engineered = self.feature_builder_.transform(X)
        numeric = self.feature_builder_.numeric_columns_out_
        categorical = self.feature_builder_.categorical_columns_out_
        blocks = self.feature_builder_.get_blocks_out()

        def stage(class_names: Sequence[str]) -> SoftGowerKNeighborsClassifier:
            return SoftGowerKNeighborsClassifier(
                numeric_columns=numeric,
                categorical_columns=categorical,
                class_names=class_names,
                blocks=blocks,
                n_neighbors=self.k,
                distance_power=self.p,
                epsilon=self.epsilon,
            )

        self.model_ = TwoStageCardinalClassifier(
            gate_estimator=stage(CARDINAL_GATE_CLASSES),
            conditional_estimator=stage(CLASSIC_CARDINAL_CLASSES),
            cardinal_epsilon=self.cardinal_epsilon,
        ).fit(engineered, y, sample_weight=sample_weight)
        self.gate_estimator_ = self.model_.gate_estimator_
        self.conditional_estimator_ = self.model_.conditional_estimator_
        self.classes_ = np.asarray(CARDINAL_CLASSES, dtype=object)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = int(X.shape[1])
        self.engineered_feature_names_ = (
            self.feature_builder_.get_feature_names_out()
        )
        self.numeric_columns_ = tuple(numeric)
        self.categorical_columns_ = tuple(categorical)
        self.blocks_ = blocks
        self.n_training_records_ = int(len(X))
        self.posterior_draws_expanded_ = False
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "feature_builder_"):
            raise RuntimeError(
                "ReducedPanelGowerClassifier must be fitted first"
            )
        engineered = self.feature_builder_.transform(X)
        probabilities = self.model_.predict_proba(engineered)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


__all__ = [
    "GaussianMAPSoftmaxClassifier",
    "ReducedPanelGowerClassifier",
    "ReducedPanelTwoStageMAPClassifier",
]
