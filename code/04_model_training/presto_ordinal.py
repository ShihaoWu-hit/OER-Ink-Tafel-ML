"""Probability-valid PRESTO-inspired ordinal models for soft labels.

The original PRESTO estimator of Faletto and Bien (2023) fits non-parallel
cumulative logits and applies an L1 penalty to differences between adjacent
decision-boundary coefficient vectors.  Non-parallel cumulative logits can
cross and therefore can yield negative class probabilities.  That is not
acceptable for the posterior-soft response used in this project.

This module keeps PRESTO's central borrowing-of-strength mechanism but uses a
forward continuation-ratio parameterization:

``advance_1 = P(Y >= 120_side | X)``
``advance_2 = P(Y = high_off | Y >= 120_side, X)``

The resulting class probabilities are

``P(low) = 1 - advance_1``
``P(120) = advance_1 * (1 - advance_2)``
``P(high) = advance_1 * advance_2``.

Separate slope vectors govern the two transitions, while a smooth
approximation to PRESTO's L1 penalty shrinks their difference toward zero.
Every experimental record contributes one three-component posterior target;
neither posterior draws nor class rows are expanded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

from .cardinal_reduced_models import (
    _as_frame,
    _class_names,
    _positive_float,
    _preprocessor,
    _resolve_columns,
    _sample_weights,
    _soft_targets,
)


PRESTO_CLASS_NAMES: tuple[str, ...] = (
    "low_60_side",
    "120_side",
    "high_off",
)


def _smooth_absolute(
    values: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a zero-at-origin smooth absolute value and its derivative."""

    array = np.asarray(values, dtype=float)
    scale = float(smoothing)
    root = np.sqrt(np.square(array) + scale**2)
    return root - scale, array / root


def _probability_audit(probabilities: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("PRESTO probabilities must have three columns")
    if not np.isfinite(values).all():
        raise ValueError("PRESTO probabilities must be finite")
    if np.any(values < -1.0e-12) or np.any(values > 1.0 + 1.0e-12):
        raise ValueError("PRESTO probabilities must lie in [0, 1]")
    row_error = np.abs(values.sum(axis=1) - 1.0)
    if np.max(row_error, initial=0.0) > 1.0e-10:
        raise ValueError("PRESTO probabilities must sum to one")
    return {
        "n_rows": int(len(values)),
        "minimum_probability": float(np.min(values, initial=1.0)),
        "maximum_probability": float(np.max(values, initial=0.0)),
        "maximum_simplex_error": float(np.max(row_error, initial=0.0)),
    }


class SoftFusedContinuationRatioClassifier(
    ClassifierMixin,
    BaseEstimator,
):
    """Fit a probability-valid PRESTO-inspired model to three soft classes.

    Parameters
    ----------
    numeric_columns, categorical_columns:
        Disjoint exhaustive columns of the fold-local engineered feature
        table.  Median imputation, one-hot encoding, and scaling are fitted
        only on the supplied training records.
    penalty:
        Common coefficient for the smooth L1 penalties on the first
        transition slopes and on adjacent-transition slope differences.  This
        mirrors PRESTO's ``|beta_1| + |beta_2 - beta_1|`` structure.
    ridge:
        Small numerical stabilizer applied to both slope vectors.  It is not
        part of the PRESTO interpretation and is exposed in the fit audit.
    l1_smoothing:
        Positive scale in ``sqrt(x^2 + scale^2) - scale``.
    """

    def __init__(
        self,
        *,
        numeric_columns: Sequence[str] = (),
        categorical_columns: Sequence[str] = (),
        class_names: Sequence[str] = PRESTO_CLASS_NAMES,
        penalty: float = 0.1,
        ridge: float = 1.0e-6,
        l1_smoothing: float = 1.0e-6,
        max_iter: int = 2000,
        tolerance: float = 1.0e-9,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.class_names = class_names
        self.penalty = penalty
        self.ridge = ridge
        self.l1_smoothing = l1_smoothing
        self.max_iter = max_iter
        self.tolerance = tolerance

    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(theta, dtype=float).reshape(
            2,
            self.design_width_,
        )
        return matrix[:, 0], matrix[:, 1:]

    @staticmethod
    def _transition_to_class_probabilities(
        transition_probabilities: np.ndarray,
    ) -> np.ndarray:
        transition = np.asarray(transition_probabilities, dtype=float)
        if transition.ndim != 2 or transition.shape[1] != 2:
            raise ValueError("Transition probabilities must have two columns")
        first = transition[:, 0]
        second = transition[:, 1]
        probabilities = np.column_stack(
            [
                1.0 - first,
                first * (1.0 - second),
                first * second,
            ]
        )
        _probability_audit(probabilities)
        return probabilities

    def _objective_and_gradient(
        self,
        theta: np.ndarray,
        design: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        intercept, slopes = self._unpack(theta)
        logits = intercept[None, :] + design @ slopes.T
        transition = expit(logits)

        advance_first = targets[:, 1] + targets[:, 2]
        advance_second_mass = targets[:, 2]
        residual = np.column_stack(
            [
                transition[:, 0] - advance_first,
                advance_first * transition[:, 1] - advance_second_mass,
            ]
        )
        weighted_residual = weights[:, None] * residual
        total_weight = float(weights.sum())

        probabilities = self._transition_to_class_probabilities(transition)
        log_probabilities = np.log(
            np.clip(probabilities, np.finfo(float).tiny, 1.0)
        )
        objective = -float(
            np.sum(weights[:, None] * targets * log_probabilities)
            / total_weight
        )
        gradient_intercept = weighted_residual.sum(axis=0) / total_weight
        gradient_slopes = (
            weighted_residual.T @ design
        ) / total_weight

        first_absolute, first_derivative = _smooth_absolute(
            slopes[0],
            self.l1_smoothing_,
        )
        difference = slopes[1] - slopes[0]
        difference_absolute, difference_derivative = _smooth_absolute(
            difference,
            self.l1_smoothing_,
        )
        objective += self.penalty_ * float(
            first_absolute.sum() + difference_absolute.sum()
        )
        gradient_slopes[0] += self.penalty_ * (
            first_derivative - difference_derivative
        )
        gradient_slopes[1] += (
            self.penalty_ * difference_derivative
        )

        if self.ridge_ > 0:
            objective += 0.5 * self.ridge_ * float(
                np.square(slopes).sum()
            )
            gradient_slopes += self.ridge_ * slopes

        gradient = np.column_stack(
            [gradient_intercept, gradient_slopes]
        )
        return float(objective), gradient.ravel()

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "SoftFusedContinuationRatioClassifier":
        frame = _as_frame(X)
        names = _class_names(self.class_names)
        if names != PRESTO_CLASS_NAMES:
            raise ValueError(
                "SoftFusedContinuationRatioClassifier requires the fixed "
                f"ordered classes {PRESTO_CLASS_NAMES}"
            )
        numeric, categorical = _resolve_columns(
            frame,
            self.numeric_columns,
            self.categorical_columns,
        )
        targets = _soft_targets(
            y,
            n_rows=len(frame),
            class_names=names,
        )
        weights = _sample_weights(sample_weight, len(frame))

        penalty = float(self.penalty)
        ridge = float(self.ridge)
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty must be finite and nonnegative")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        self.penalty_ = penalty
        self.ridge_ = ridge
        self.l1_smoothing_ = _positive_float(
            self.l1_smoothing,
            "l1_smoothing",
        )
        tolerance = _positive_float(self.tolerance, "tolerance")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be positive")

        positive = weights > 0
        frame = frame.loc[positive].reset_index(drop=True)
        targets = targets[positive]
        weights = weights[positive]
        self.preprocessor_ = _preprocessor(numeric, categorical)
        transformed = np.asarray(
            self.preprocessor_.fit_transform(frame),
            dtype=float,
        )
        if transformed.ndim != 2 or not np.isfinite(transformed).all():
            raise ValueError("Fold-local transformed predictors are not finite")

        self.classes_ = np.asarray(names, dtype=object)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = int(frame.shape[1])
        self.numeric_columns_ = numeric
        self.categorical_columns_ = categorical
        self.transformed_feature_names_ = np.asarray(
            self.preprocessor_.get_feature_names_out(),
            dtype=object,
        )
        self.design_width_ = int(transformed.shape[1] + 1)

        total_weight = float(weights.sum())
        first_prior = float(
            np.sum(weights * (targets[:, 1] + targets[:, 2]))
            / total_weight
        )
        second_denominator = float(
            np.sum(weights * (targets[:, 1] + targets[:, 2]))
        )
        if (
            not np.isfinite(second_denominator)
            or second_denominator <= 0
        ):
            raise ValueError(
                "The second ordinal transition has zero training mass"
            )
        second_prior = float(
            np.sum(weights * targets[:, 2]) / second_denominator
        )
        epsilon = 1.0e-8
        priors = np.clip(
            np.asarray([first_prior, second_prior], dtype=float),
            epsilon,
            1.0 - epsilon,
        )
        initial_intercept = np.log(priors / (1.0 - priors))
        initial = np.zeros((2, self.design_width_), dtype=float)
        initial[:, 0] = initial_intercept

        result = minimize(
            self._objective_and_gradient,
            initial.ravel(),
            args=(transformed, targets, weights),
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
        if not np.isfinite(result_fun) or not np.isfinite(result_x).all():
            raise RuntimeError(
                "PRESTO-inspired optimizer returned non-finite output: "
                f"status={int(result.status)}, message={str(result.message)!r}"
            )

        intercept, slopes = self._unpack(result_x)
        self.intercept_ = intercept.copy()
        self.coef_ = slopes.copy()
        self.n_input_records_ = int(len(positive))
        self.n_training_records_ = int(len(frame))
        self.total_fractional_weight_ = total_weight
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
        self.objective_ = result_fun
        self.slope_difference_l1_ = float(
            np.abs(slopes[1] - slopes[0]).sum()
        )
        self.slope_first_l1_ = float(np.abs(slopes[0]).sum())
        self.fit_status_ = (
            "complete"
            if self.optimizer_success_
            else "optimizer_nonconverged_finite_solution"
        )
        self.inference_method_ = (
            "soft_fused_continuation_ratio_scipy_L-BFGS-B"
        )
        self.posterior_draws_expanded_ = False
        self.class_rows_expanded_ = False
        self.optimization_audit_ = {
            "success": self.optimizer_success_,
            "status": self.optimizer_status_,
            "message": self.optimizer_message_,
            "iterations": self.optimizer_iterations_,
            "function_evaluations": self.optimizer_function_evaluations_,
            "gradient_evaluations": self.optimizer_gradient_evaluations_,
            "objective": self.objective_,
            "method": "L-BFGS-B",
            "model": "PRESTO-inspired_probability_valid_continuation_ratio",
            "penalty": self.penalty_,
            "ridge": self.ridge_,
            "l1_smoothing": self.l1_smoothing_,
            "slope_first_l1": self.slope_first_l1_,
            "slope_difference_l1": self.slope_difference_l1_,
            "posterior_argmax_used": False,
            "posterior_draws_expanded": False,
            "class_rows_expanded": False,
        }
        training_probabilities = self.predict_proba(frame)
        self.training_probability_audit_ = _probability_audit(
            training_probabilities
        )
        return self

    def predict_transition_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "preprocessor_"):
            raise RuntimeError(
                "SoftFusedContinuationRatioClassifier must be fitted first"
            )
        frame = _as_frame(X, expected_columns=self.feature_names_in_)
        transformed = np.asarray(
            self.preprocessor_.transform(frame),
            dtype=float,
        )
        if not np.isfinite(transformed).all():
            raise ValueError("Transformed prediction features are not finite")
        logits = self.intercept_[None, :] + transformed @ self.coef_.T
        transition = expit(logits)
        if (
            not np.isfinite(transition).all()
            or np.any(transition < 0)
            or np.any(transition > 1)
        ):
            raise RuntimeError("Invalid PRESTO transition probabilities")
        return transition

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._transition_to_class_probabilities(
            self.predict_transition_proba(X)
        )

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


__all__ = [
    "PRESTO_CLASS_NAMES",
    "SoftFusedContinuationRatioClassifier",
]
