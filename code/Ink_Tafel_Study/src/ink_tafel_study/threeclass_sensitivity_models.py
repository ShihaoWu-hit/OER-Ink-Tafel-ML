"""Soft-label model components for the post-hoc three-class sensitivity.

This module contains only fold-local estimators and probability combiners.  It
does not construct cross-validation splits, inspect outer-test performance, or
select a feature panel.  In particular, :class:`ConvexOOFProbabilityStacker`
must be fitted only on grouped inner out-of-fold predictions produced inside
one outer-training set.

The three-class order used by the helpers is
``low_60_side / 120_side / high_off``.  A two-stage estimator first
predicts the probability of being below the high-off boundary and then
predicts low-versus-120 conditional on that mass.  Every estimator consumes
one fractional target per experimental record; posterior draws are never
expanded into pseudo-records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin

from .cardinal_models import (
    SoftExtraTreesClassifier,
    SoftGowerKNeighborsClassifier,
)
from .cardinal_reduced_models import GaussianMAPSoftmaxClassifier
from .feature_association_features import (
    AssociationFeatureBuilder,
    required_association_source_columns,
)


SENSITIVITY_CLASS_LABELS: tuple[str, ...] = (
    "low_60_side",
    "120_side",
    "high_off",
)
SENSITIVITY_GATE_LABELS: tuple[str, ...] = (
    "below_high_off",
    "high_off",
)
SENSITIVITY_CONDITIONAL_LABELS: tuple[str, ...] = (
    "low_60_side",
    "120_side",
)
SOFT_BINARY_STAGE_MODEL_FAMILIES: tuple[str, ...] = (
    "gaussian_map",
    "gower",
    "extra_trees",
    "catboost",
)

_SOFT_BINARY_STAGE_ALIASES: Mapping[str, str] = {
    "gaussian_map": "gaussian_map",
    "gaussian_prior_map": "gaussian_map",
    "map": "gaussian_map",
    "two_stage_gaussian_prior_map": "gaussian_map",
    "gower": "gower",
    "gower_knn": "gower",
    "soft_gower": "gower",
    "soft_gower_knn": "gower",
    "two_stage_soft_gower_knn": "gower",
    "extra_trees": "extra_trees",
    "soft_extra_trees": "extra_trees",
    "two_stage_soft_extra_trees": "extra_trees",
    "catboost": "catboost",
    "soft_catboost": "catboost",
    "two_stage_soft_catboost": "catboost",
}

_SOFT_BINARY_STAGE_ALLOWED_PARAMS: Mapping[str, frozenset[str]] = {
    "gaussian_map": frozenset(
        {
            "feature_prior_scale",
            "intercept_prior_scale",
            "max_iter",
            "tolerance",
        }
    ),
    "gower": frozenset(
        {
            "n_neighbors",
            "distance_power",
            "epsilon",
            "neighbor_mode",
            "bandwidth",
            "block_weight_profile",
            "block_weights",
            "feature_weights",
        }
    ),
    "extra_trees": frozenset(
        {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
        }
    ),
    "catboost": frozenset(
        {
            "iterations",
            "depth",
            "learning_rate",
            "l2_leaf_reg",
            "border_count",
            "random_strength",
        }
    ),
}


def _n_rows(X: Any) -> int:
    if not hasattr(X, "shape") or len(X.shape) != 2:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        return int(values.shape[0])
    return int(X.shape[0])


def _validate_sample_weight(
    sample_weight: Any,
    *,
    n_rows: int,
) -> np.ndarray:
    if sample_weight is None:
        return np.ones(int(n_rows), dtype=float)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.shape != (int(n_rows),):
        raise ValueError(
            f"sample_weight must have shape ({int(n_rows)},); "
            f"received {tuple(weights.shape)}"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and nonnegative")
    if not np.any(weights > 0):
        raise ValueError("At least one sample weight must be positive")
    return weights


def validate_probability_matrix(
    values: Any,
    *,
    n_rows: int | None = None,
    n_classes: int | None = None,
    name: str = "probabilities",
    simplex_tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Validate a probability matrix without silently repairing it.

    No clipping or renormalization is applied.  This fail-closed behavior is
    important for auditing the exact ``alpha=0`` and ``alpha=1`` shrinkage
    endpoints.
    """

    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional probability matrix; "
            f"received shape {tuple(probabilities.shape)}"
        )
    if n_rows is not None and probabilities.shape[0] != int(n_rows):
        raise ValueError(
            f"{name} must contain {int(n_rows)} rows; "
            f"received {probabilities.shape[0]}"
        )
    if n_classes is not None and probabilities.shape[1] != int(n_classes):
        raise ValueError(
            f"{name} must contain {int(n_classes)} classes; "
            f"received {probabilities.shape[1]}"
        )
    if probabilities.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two classes")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(f"{name} contains values outside [0, 1]")
    if not np.isfinite(float(simplex_tolerance)) or simplex_tolerance < 0:
        raise ValueError("simplex_tolerance must be finite and nonnegative")
    totals = probabilities.sum(axis=1)
    if not np.allclose(
        totals,
        1.0,
        rtol=0.0,
        atol=float(simplex_tolerance),
    ):
        maximum_error = float(np.max(np.abs(totals - 1.0)))
        raise ValueError(
            f"{name} rows must sum to one; maximum absolute error="
            f"{maximum_error:.12g}"
        )
    return probabilities.copy()


def probability_matrix_audit(values: Any) -> dict[str, Any]:
    """Return compact probability-range and simplex diagnostics."""

    probabilities = validate_probability_matrix(values)
    totals = probabilities.sum(axis=1)
    return {
        "n_rows": int(probabilities.shape[0]),
        "n_classes": int(probabilities.shape[1]),
        "minimum_probability": float(np.min(probabilities)),
        "maximum_probability": float(np.max(probabilities)),
        "maximum_simplex_abs_error": float(
            np.max(np.abs(totals - 1.0))
        ),
        "all_finite": True,
        "normalization_applied": False,
    }


def _validate_soft_binary_target(
    y: Any,
    *,
    n_rows: int,
) -> np.ndarray:
    """Return the probability of the first binary class."""

    values = np.asarray(y, dtype=float)
    if values.ndim == 1:
        if values.shape != (int(n_rows),):
            raise ValueError(
                f"Binary soft target must have shape ({int(n_rows)},); "
                f"received {tuple(values.shape)}"
            )
        first_probability = values
    elif values.ndim == 2 and values.shape == (int(n_rows), 2):
        matrix = validate_probability_matrix(
            values,
            n_rows=int(n_rows),
            n_classes=2,
            name="binary soft target",
        )
        first_probability = matrix[:, 0]
    else:
        raise ValueError(
            "Binary soft target must be a one-dimensional first-class "
            f"probability or a ({int(n_rows)}, 2) probability matrix; "
            f"received {tuple(values.shape)}"
        )
    if not np.isfinite(first_probability).all():
        raise ValueError("Binary soft target contains NaN or infinite values")
    if np.any(first_probability < 0) or np.any(first_probability > 1):
        raise ValueError("Binary soft target values must lie in [0, 1]")
    return np.asarray(first_probability, dtype=float)


def _validate_soft_targets(
    y: Any,
    *,
    n_rows: int,
    n_classes: int,
) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        if all(column in y.columns for column in SENSITIVITY_CLASS_LABELS):
            values = y.loc[:, list(SENSITIVITY_CLASS_LABELS)].to_numpy(float)
        elif all(
            f"q_{column}" in y.columns
            for column in SENSITIVITY_CLASS_LABELS
        ):
            values = y.loc[
                :,
                [f"q_{column}" for column in SENSITIVITY_CLASS_LABELS],
            ].to_numpy(float)
        else:
            values = y.to_numpy(float)
    else:
        values = np.asarray(y, dtype=float)
    return validate_probability_matrix(
        values,
        n_rows=int(n_rows),
        n_classes=int(n_classes),
        name="soft target",
    )


class FoldLocalBinarySoftPriorClassifier(ClassifierMixin, BaseEstimator):
    """Weighted fold-local prior for a fractional binary target.

    A one-dimensional target is interpreted as the probability of
    ``class_names[0]``.  A two-column target must use the same class order.
    """

    def __init__(
        self,
        *,
        class_names: Sequence[str] = SENSITIVITY_GATE_LABELS,
    ) -> None:
        self.class_names = class_names

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "FoldLocalBinarySoftPriorClassifier":
        n_rows = _n_rows(X)
        names = tuple(map(str, self.class_names))
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError("class_names must contain exactly two unique names")
        first_probability = _validate_soft_binary_target(y, n_rows=n_rows)
        weights = _validate_sample_weight(
            sample_weight,
            n_rows=n_rows,
        )
        total_weight = float(weights.sum())
        weighted_first = float(
            np.dot(weights, first_probability) / total_weight
        )
        self.classes_ = np.asarray(names, dtype=object)
        self.prior_probabilities_ = np.asarray(
            [weighted_first, 1.0 - weighted_first],
            dtype=float,
        )
        self.n_training_records_ = int(n_rows)
        self.total_fractional_weight_ = total_weight
        self.fit_audit_ = {
            "estimator": "fold_local_binary_soft_prior",
            "class_labels": list(names),
            "first_class_target_semantics": (
                "one-dimensional y is probability of class_names[0]"
            ),
            "n_training_records": int(n_rows),
            "total_sample_weight": total_weight,
            "weighted_first_class_mass": float(
                np.dot(weights, first_probability)
            ),
            "prior_probabilities": self.prior_probabilities_.tolist(),
            "fit_scope": "supplied_training_rows_only",
            "posterior_draws_expanded": False,
            "class_rows_expanded": False,
        }
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "prior_probabilities_"):
            raise RuntimeError(
                "FoldLocalBinarySoftPriorClassifier must be fitted before "
                "predict_proba"
            )
        return np.repeat(
            self.prior_probabilities_[None, :],
            _n_rows(X),
            axis=0,
        )

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def prior_shrink_probabilities(
    model_probabilities: Any,
    prior_probabilities: Any,
    alpha: float,
    *,
    return_audit: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Mix model and fold-local prior probabilities on the simplex.

    ``alpha=0`` returns an exact copy of the supplied prior matrix and
    ``alpha=1`` returns an exact copy of the supplied model matrix.  A
    one-dimensional prior vector is broadcast across prediction rows.
    """

    model = validate_probability_matrix(
        model_probabilities,
        name="model probabilities",
    )
    prior_values = np.asarray(prior_probabilities, dtype=float)
    if prior_values.ndim == 1:
        if prior_values.shape != (model.shape[1],):
            raise ValueError(
                "One-dimensional prior must have one value per class; "
                f"expected {(model.shape[1],)}, received "
                f"{tuple(prior_values.shape)}"
            )
        prior = np.repeat(prior_values[None, :], len(model), axis=0)
    else:
        prior = prior_values
    prior = validate_probability_matrix(
        prior,
        n_rows=len(model),
        n_classes=model.shape[1],
        name="prior probabilities",
    )
    try:
        resolved_alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be a finite scalar in [0, 1]") from error
    if (
        not np.isfinite(resolved_alpha)
        or resolved_alpha < 0
        or resolved_alpha > 1
    ):
        raise ValueError("alpha must be a finite scalar in [0, 1]")

    if resolved_alpha == 0.0:
        mixed = prior.copy()
        endpoint = "exact_prior"
    elif resolved_alpha == 1.0:
        mixed = model.copy()
        endpoint = "exact_model"
    else:
        mixed = (
            resolved_alpha * model
            + (1.0 - resolved_alpha) * prior
        )
        endpoint = "interior_convex_mixture"
    validated = validate_probability_matrix(
        mixed,
        n_rows=len(model),
        n_classes=model.shape[1],
        name="prior-shrunk probabilities",
    )
    audit = {
        "operation": "fold_local_prior_shrinkage",
        "alpha_model": resolved_alpha,
        "alpha_prior": 1.0 - resolved_alpha,
        "endpoint_behavior": endpoint,
        "n_rows": int(model.shape[0]),
        "n_classes": int(model.shape[1]),
        "model_probability_audit": probability_matrix_audit(model),
        "prior_probability_audit": probability_matrix_audit(prior),
        "output_probability_audit": probability_matrix_audit(validated),
        "normalization_applied": False,
    }
    if return_audit:
        return validated, audit
    return validated


def _soft_log_loss(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    sample_weight: np.ndarray,
    epsilon: float,
) -> float:
    clipped = np.clip(probabilities, float(epsilon), 1.0)
    record_losses = -np.sum(targets * np.log(clipped), axis=1)
    return float(np.average(record_losses, weights=sample_weight))


def _candidate_probability_cube(
    predictions: Mapping[str, Any],
    *,
    n_rows: int | None = None,
    n_classes: int | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    if not isinstance(predictions, Mapping) or not predictions:
        raise TypeError(
            "predictions must be a non-empty mapping from candidate name "
            "to probability matrix"
        )
    names = tuple(sorted(map(str, predictions)))
    if len(set(names)) != len(names):
        raise ValueError("Candidate names must be unique after string conversion")
    converted = {str(name): value for name, value in predictions.items()}
    matrices: list[np.ndarray] = []
    resolved_rows = n_rows
    resolved_classes = n_classes
    for name in names:
        matrix = validate_probability_matrix(
            converted[name],
            n_rows=resolved_rows,
            n_classes=resolved_classes,
            name=f"candidate {name!r} probabilities",
        )
        if resolved_rows is None:
            resolved_rows = int(matrix.shape[0])
        if resolved_classes is None:
            resolved_classes = int(matrix.shape[1])
        matrices.append(matrix)
    # Rows first makes the fitted weighted sum transparent:
    # output[n, k] = sum_m weights[m] * cube[n, m, k].
    return names, np.stack(matrices, axis=1)


class ConvexOOFProbabilityStacker(BaseEstimator):
    """Fit nonnegative, sum-to-one weights on inner-OOF probabilities.

    Candidate names are sorted before optimization, and deterministic
    multi-start SLSQP is checked against the best single candidate.  If no
    feasible solution is acceptable, ``fallback_candidate`` is used when it
    is present (the contract default is ``prior``); otherwise the best single
    candidate is used.  Exact best-single ties prefer ``prior`` and then
    lexical candidate order.
    """

    def __init__(
        self,
        *,
        epsilon: float = 1.0e-12,
        max_iter: int = 1000,
        tolerance: float = 1.0e-10,
        acceptance_tolerance: float = 1.0e-10,
        fallback_candidate: str = "prior",
    ) -> None:
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.acceptance_tolerance = acceptance_tolerance
        self.fallback_candidate = fallback_candidate

    def fit(
        self,
        predictions: Mapping[str, Any],
        y: Any,
        sample_weight: Any = None,
    ) -> "ConvexOOFProbabilityStacker":
        if not np.isfinite(float(self.epsilon)) or not 0 < float(
            self.epsilon
        ) < 0.5:
            raise ValueError("epsilon must be finite and lie in (0, 0.5)")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be positive")
        if not np.isfinite(float(self.tolerance)) or float(
            self.tolerance
        ) <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(float(self.acceptance_tolerance)) or float(
            self.acceptance_tolerance
        ) < 0:
            raise ValueError(
                "acceptance_tolerance must be finite and nonnegative"
            )
        requested_fallback = str(self.fallback_candidate).strip()
        if not requested_fallback:
            raise ValueError("fallback_candidate must be a non-empty name")

        candidate_names, cube = _candidate_probability_cube(predictions)
        n_rows, n_candidates, n_classes = cube.shape
        targets = _validate_soft_targets(
            y,
            n_rows=n_rows,
            n_classes=n_classes,
        )
        weights = _validate_sample_weight(
            sample_weight,
            n_rows=n_rows,
        )
        individual_losses = np.asarray(
            [
                _soft_log_loss(
                    targets,
                    cube[:, candidate_index, :],
                    sample_weight=weights,
                    epsilon=float(self.epsilon),
                )
                for candidate_index in range(n_candidates)
            ],
            dtype=float,
        )
        minimum_single_loss = float(np.min(individual_losses))
        tied = np.flatnonzero(
            individual_losses
            <= minimum_single_loss + float(self.acceptance_tolerance)
        )
        best_single_index = min(
            tied.tolist(),
            key=lambda index: (
                candidate_names[index] != "prior",
                candidate_names[index],
            ),
        )
        fallback_index = (
            candidate_names.index(requested_fallback)
            if requested_fallback in candidate_names
            else best_single_index
        )
        fallback_loss = float(individual_losses[fallback_index])
        fallback_weights = np.zeros(n_candidates, dtype=float)
        fallback_weights[fallback_index] = 1.0

        def objective(candidate_weights: np.ndarray) -> float:
            combined = np.einsum(
                "m,nmk->nk",
                candidate_weights,
                cube,
                optimize=True,
            )
            return _soft_log_loss(
                targets,
                combined,
                sample_weight=weights,
                epsilon=float(self.epsilon),
            )

        best_single_weights = np.zeros(n_candidates, dtype=float)
        best_single_weights[best_single_index] = 1.0
        named_starts = [
            (
                "equal_weights",
                np.full(
                    n_candidates,
                    1.0 / n_candidates,
                    dtype=float,
                ),
            ),
            ("best_single_candidate", best_single_weights),
        ]
        if "prior" in candidate_names:
            prior_start = np.zeros(n_candidates, dtype=float)
            prior_start[candidate_names.index("prior")] = 1.0
            named_starts.append(("prior_candidate", prior_start))
        named_starts.append(("fallback_candidate", fallback_weights.copy()))
        starts: list[tuple[str, np.ndarray]] = []
        for start_name, start in named_starts:
            if not any(
                np.array_equal(start, existing)
                for _, existing in starts
            ):
                starts.append((start_name, start))

        feasible_solutions: list[
            tuple[float, int, np.ndarray, Any]
        ] = []
        solver_records: list[dict[str, Any]] = []
        for start_index, (start_name, start) in enumerate(starts):
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n_candidates,
                constraints={
                    "type": "eq",
                    "fun": lambda candidate_weights: float(
                        np.sum(candidate_weights) - 1.0
                    ),
                },
                options={
                    "maxiter": int(self.max_iter),
                    "ftol": float(self.tolerance),
                    "disp": False,
                },
            )
            raw_weights = np.asarray(result.x, dtype=float)
            feasible = (
                bool(result.success)
                and raw_weights.shape == (n_candidates,)
                and np.isfinite(raw_weights).all()
                and np.min(raw_weights) >= -1.0e-8
                and abs(float(raw_weights.sum()) - 1.0) <= 1.0e-7
            )
            loss = float("inf")
            normalized = raw_weights
            if feasible:
                normalized = np.maximum(raw_weights, 0.0)
                normalized /= normalized.sum()
                loss = float(objective(normalized))
                feasible = np.isfinite(loss)
            solver_records.append(
                {
                    "start_index": int(start_index),
                    "start_name": start_name,
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": str(result.message),
                    "n_iterations": int(getattr(result, "nit", -1)),
                    "feasible_after_check": bool(feasible),
                    "objective": loss if np.isfinite(loss) else None,
                }
            )
            if feasible:
                feasible_solutions.append(
                    (loss, start_index, normalized, result)
                )

        fallback_used = True
        fallback_reason = "no_feasible_optimizer_solution"
        selected_weights = fallback_weights
        selected_loss = fallback_loss
        selected_start_index: int | None = None
        if feasible_solutions:
            best_loss, best_start, best_weights, _ = min(
                feasible_solutions,
                key=lambda item: (item[0], item[1]),
            )
            if (
                best_loss
                <= minimum_single_loss + float(self.acceptance_tolerance)
            ):
                selected_weights = best_weights
                selected_loss = float(best_loss)
                selected_start_index = int(best_start)
                fallback_used = bool(
                    np.array_equal(selected_weights, fallback_weights)
                )
                fallback_reason = (
                    "optimizer_selected_fallback_vertex"
                    if fallback_used
                    else ""
                )
            else:
                fallback_reason = (
                    "optimizer_solution_worse_than_best_single_candidate"
                )

        self.candidate_names_ = candidate_names
        self.weights_ = np.asarray(selected_weights, dtype=float)
        self.n_candidates_in_ = int(n_candidates)
        self.n_classes_ = int(n_classes)
        self.n_training_records_ = int(n_rows)
        self.classes_ = np.arange(n_classes)
        self.fit_audit_ = {
            "estimator": "convex_inner_oof_probability_stacker",
            "required_fit_prediction_scope": (
                "grouped_inner_oof_within_outer_training_rows"
            ),
            "candidate_names": list(candidate_names),
            "candidate_weights": {
                name: float(self.weights_[index])
                for index, name in enumerate(candidate_names)
            },
            "nonnegative_weights": bool(np.all(self.weights_ >= 0)),
            "weight_sum": float(self.weights_.sum()),
            "prior_candidate_present": "prior" in candidate_names,
            "individual_soft_log_losses": {
                name: float(individual_losses[index])
                for index, name in enumerate(candidate_names)
            },
            "best_single_candidate": candidate_names[best_single_index],
            "best_single_soft_log_loss": minimum_single_loss,
            "fallback_candidate_requested": requested_fallback,
            "fallback_candidate_resolved": candidate_names[fallback_index],
            "fallback_candidate_present": (
                requested_fallback in candidate_names
            ),
            "fallback_candidate_soft_log_loss": fallback_loss,
            "stack_soft_log_loss": float(selected_loss),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "selected_solver_start_index": selected_start_index,
            "solver": "scipy.optimize.SLSQP",
            "solver_runs": solver_records,
            "n_training_records": int(n_rows),
            "n_candidates": int(n_candidates),
            "n_classes": int(n_classes),
            "total_sample_weight": float(weights.sum()),
            "posterior_argmax_used": False,
            "outer_test_predictions_permitted_for_fit": False,
        }
        return self

    def predict_proba(
        self,
        predictions: Mapping[str, Any],
    ) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError(
                "ConvexOOFProbabilityStacker must be fitted before "
                "predict_proba"
            )
        candidate_names, cube = _candidate_probability_cube(
            predictions,
            n_classes=self.n_classes_,
        )
        if candidate_names != self.candidate_names_:
            raise ValueError(
                "Prediction candidates do not match fitted candidates; "
                f"expected {self.candidate_names_}, received {candidate_names}"
            )
        probabilities = np.einsum(
            "m,nmk->nk",
            self.weights_,
            cube,
            optimize=True,
        )
        return validate_probability_matrix(
            probabilities,
            n_rows=cube.shape[0],
            n_classes=self.n_classes_,
            name="stacked probabilities",
        )


def combine_two_stage_probabilities(
    gate_probabilities: Any,
    conditional_probabilities: Any,
    *,
    return_audit: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Combine below-high/off gate and low/120 conditional probabilities."""

    gate = validate_probability_matrix(
        gate_probabilities,
        n_classes=2,
        name="gate probabilities",
    )
    conditional = validate_probability_matrix(
        conditional_probabilities,
        n_rows=len(gate),
        n_classes=2,
        name="conditional probabilities",
    )
    combined = np.column_stack(
        [
            gate[:, 0] * conditional[:, 0],
            gate[:, 0] * conditional[:, 1],
            gate[:, 1],
        ]
    )
    validated = validate_probability_matrix(
        combined,
        n_rows=len(gate),
        n_classes=3,
        name="combined two-stage probabilities",
    )
    audit = {
        "architecture": (
            "below_high_off_gate_then_low60_vs_120_conditional"
        ),
        "gate_class_labels": list(SENSITIVITY_GATE_LABELS),
        "conditional_class_labels": list(
            SENSITIVITY_CONDITIONAL_LABELS
        ),
        "output_class_labels": list(SENSITIVITY_CLASS_LABELS),
        "gate_probability_audit": probability_matrix_audit(gate),
        "conditional_probability_audit": probability_matrix_audit(
            conditional
        ),
        "output_probability_audit": probability_matrix_audit(validated),
    }
    if return_audit:
        return validated, audit
    return validated


def _require_catboost() -> tuple[Any, str]:
    try:
        module = importlib.import_module("catboost")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "TwoStageSoftCatBoostClassifier requires the optional "
            "'exploratory-ml' dependency catboost; installation is missing, "
            "so the estimator fails closed"
        ) from error
    classifier = getattr(module, "CatBoostClassifier", None)
    if classifier is None:
        raise RuntimeError(
            "Imported catboost does not expose CatBoostClassifier; "
            "the estimator fails closed"
        )
    return classifier, str(getattr(module, "__version__", "unknown"))


def _feature_frame(
    X: Any,
    *,
    expected_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] = (),
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if expected_columns is None:
            columns = [f"x{index}" for index in range(values.shape[1])]
        else:
            columns = list(map(str, expected_columns))
            if len(columns) != values.shape[1]:
                raise ValueError(
                    "X column count does not match fitted feature names"
                )
        frame = pd.DataFrame(values, columns=columns)
    frame.columns = frame.columns.map(str)
    if expected_columns is not None:
        expected = list(map(str, expected_columns))
        missing = sorted(set(expected) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(expected))
        if missing or extra:
            raise ValueError(
                "Prediction feature columns differ from fitted columns; "
                f"missing={missing}, extra={extra}"
            )
        frame = frame.loc[:, expected]
    categorical = tuple(map(str, categorical_columns))
    missing_categorical = sorted(set(categorical) - set(frame.columns))
    if missing_categorical:
        raise ValueError(
            f"Categorical columns are missing from X: {missing_categorical}"
        )
    for column in frame.columns:
        if column in categorical:
            frame[column] = (
                frame[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _fold_local_engineered_features(
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
    source_columns = tuple(required_association_source_columns(str(panel)))
    missing_train = sorted(set(source_columns) - set(X_train.columns))
    missing_test = sorted(set(source_columns) - set(X_test.columns))
    if missing_train or missing_test:
        raise KeyError(
            "Binary stage panel source columns are missing: "
            f"train={missing_train}, test={missing_test}"
        )
    train_source = X_train.loc[:, list(source_columns)].copy()
    test_source = X_test.loc[:, list(source_columns)].copy()
    builder = AssociationFeatureBuilder(
        str(panel),
        rare_min_frequency=int(rare_min_frequency),
    ).fit(train_source)
    engineered_train = builder.transform(train_source)
    engineered_test = builder.transform(test_source)
    return (
        builder,
        engineered_train,
        engineered_test,
        source_columns,
    )


def _canonical_soft_binary_stage_family(model_family: str) -> str:
    token = str(model_family).strip().lower()
    try:
        return _SOFT_BINARY_STAGE_ALIASES[token]
    except KeyError as error:
        raise KeyError(
            f"Unknown binary soft stage family {model_family!r}; choose from "
            f"{SOFT_BINARY_STAGE_MODEL_FAMILIES}"
        ) from error


def _validated_soft_binary_stage_params(
    family: str,
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resolved = dict(params or {})
    unknown = sorted(
        set(resolved) - set(_SOFT_BINARY_STAGE_ALLOWED_PARAMS[family])
    )
    if unknown:
        raise TypeError(
            f"Unknown {family} binary-stage parameters: {unknown}; allowed="
            f"{sorted(_SOFT_BINARY_STAGE_ALLOWED_PARAMS[family])}"
        )
    return resolved


def _audit_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _audit_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_audit_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_audit_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_gower_block_weight_profile(
    blocks: Mapping[str, Sequence[str]],
    *,
    panel: str,
    profile: str,
    explicit_block_weights: Mapping[str, float] | None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    normalized = str(profile).strip().lower()
    allowed = {"equal", "context_2x", "formulation_2x"}
    if normalized not in allowed:
        raise ValueError(
            "block_weight_profile must be one of equal, context_2x, or "
            "formulation_2x"
        )
    if explicit_block_weights is not None:
        if normalized != "equal":
            raise ValueError(
                "Specify either a non-equal block_weight_profile or explicit "
                "block_weights, not both"
            )
        resolved_explicit = {
            str(name): float(weight)
            for name, weight in explicit_block_weights.items()
        }
        return resolved_explicit, {
            "profile": "explicit_block_weights",
            "requested_profile": normalized,
            "context_block": "context" if "context" in blocks else None,
            "increment_block": None,
            "resolved_block_weights": resolved_explicit,
            "mapping_status": "explicit",
        }

    block_names = tuple(map(str, blocks))
    if "context" not in block_names:
        raise ValueError(
            "Cannot map Gower block_weight_profile because the engineered "
            f"panel {panel!r} has no 'context' block; blocks={block_names}"
        )
    increment_blocks = tuple(
        name for name in block_names if name != "context"
    )
    if normalized == "equal":
        return None, {
            "profile": normalized,
            "requested_profile": normalized,
            "context_block": "context",
            "increment_block": (
                increment_blocks[0] if len(increment_blocks) == 1 else None
            ),
            "resolved_block_weights": {
                name: 1.0 for name in block_names
            },
            "mapping_status": "complete",
        }
    if len(increment_blocks) != 1:
        raise ValueError(
            "Cannot reliably map non-equal Gower block_weight_profile: "
            f"panel={panel!r}, context_block='context', "
            f"increment_blocks={increment_blocks}. A non-equal profile "
            "requires exactly one surviving increment block in this "
            "training fold."
        )
    increment = increment_blocks[0]
    if normalized == "context_2x":
        resolved = {"context": 2.0, increment: 1.0}
    else:
        resolved = {"context": 1.0, increment: 2.0}
    return resolved, {
        "profile": normalized,
        "requested_profile": normalized,
        "context_block": "context",
        "increment_block": increment,
        "resolved_block_weights": resolved,
        "mapping_status": "complete",
    }


def fit_predict_soft_binary_stage(
    X_train: pd.DataFrame,
    target_probability: Any,
    X_test: pd.DataFrame,
    panel: str,
    model_family: str,
    params: Mapping[str, Any] | None,
    seed: int,
    sample_weight: Any = None,
    rare_min_frequency: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fold-local binary soft-label stage and return P(class 0).

    The input frames contain raw allowlisted source columns.  Feature
    engineering, imputation, scaling, rare-level pooling, composition
    replacement, and property PCA are therefore fitted only on ``X_train``.
    This function makes no split, panel, family, or hyperparameter choice.
    Callers must make those choices using grouped inner OOF predictions from
    the enclosing outer-training set.
    """

    family = _canonical_soft_binary_stage_family(model_family)
    resolved_params = _validated_soft_binary_stage_params(family, params)
    n_source_train_records = len(X_train)
    source_first_target = _validate_soft_binary_target(
        target_probability,
        n_rows=n_source_train_records,
    )
    source_weights = _validate_sample_weight(
        sample_weight,
        n_rows=n_source_train_records,
    )
    positive_weight = source_weights > 0
    excluded_zero_weight_records = int(np.sum(~positive_weight))
    filtered_train = X_train.iloc[
        np.flatnonzero(positive_weight)
    ].reset_index(drop=True)
    first_target = source_first_target[positive_weight]
    weights = source_weights[positive_weight]
    (
        builder,
        engineered_train,
        engineered_test,
        source_columns,
    ) = _fold_local_engineered_features(
        filtered_train,
        X_test,
        panel=str(panel),
        rare_min_frequency=int(rare_min_frequency),
    )
    binary_targets = np.column_stack(
        [first_target, 1.0 - first_target]
    )
    numeric = tuple(builder.numeric_columns_out_)
    categorical = tuple(builder.categorical_columns_out_)
    blocks = builder.get_blocks_out()
    class_names = ("class0", "class1")

    backend = family
    stage_audit: dict[str, Any]
    if family == "gaussian_map":
        estimator = GaussianMAPSoftmaxClassifier(
            numeric_columns=numeric,
            categorical_columns=categorical,
            class_names=class_names,
            feature_prior_scale=float(
                resolved_params.get("feature_prior_scale", 1.0)
            ),
            intercept_prior_scale=float(
                resolved_params.get("intercept_prior_scale", 5.0)
            ),
            max_iter=int(resolved_params.get("max_iter", 1000)),
            tolerance=float(resolved_params.get("tolerance", 1.0e-8)),
        ).fit(
            engineered_train,
            binary_targets,
            sample_weight=weights,
        )
        if estimator.fit_status_ != "complete":
            raise RuntimeError(
                "Gaussian MAP binary-stage optimizer did not converge; "
                f"audit={estimator.optimization_audit_}"
            )
        stage_probabilities = estimator.predict_proba(engineered_test)
        stage_audit = {
            "backend": "gaussian_prior_MAP",
            "fit_status": str(estimator.fit_status_),
            "optimization": _audit_value(
                estimator.optimization_audit_
            ),
            "feature_prior_scale": float(
                estimator.feature_prior_scale_
            ),
            "intercept_prior_scale": float(
                estimator.intercept_prior_scale_
            ),
        }
    elif family == "gower":
        block_weights, block_profile_audit = (
            _resolve_gower_block_weight_profile(
                blocks,
                panel=str(panel),
                profile=str(
                    resolved_params.get(
                        "block_weight_profile",
                        "equal",
                    )
                ),
                explicit_block_weights=resolved_params.get(
                    "block_weights"
                ),
            )
        )
        estimator = SoftGowerKNeighborsClassifier(
            numeric_columns=numeric,
            categorical_columns=categorical,
            class_names=class_names,
            blocks=blocks,
            block_weights=block_weights,
            feature_weights=resolved_params.get("feature_weights"),
            n_neighbors=int(resolved_params.get("n_neighbors", 9)),
            distance_power=float(
                resolved_params.get("distance_power", 1.0)
            ),
            epsilon=float(resolved_params.get("epsilon", 1.0e-12)),
            neighbor_mode=str(
                resolved_params.get("neighbor_mode", "knn")
            ),
            bandwidth=float(resolved_params.get("bandwidth", 1.0)),
        ).fit(
            engineered_train,
            binary_targets,
            sample_weight=weights,
        )
        stage_probabilities = estimator.predict_proba(engineered_test)
        kernel_mode = estimator.neighbor_mode_ != "knn"
        stage_audit = {
            "backend": "soft_gower",
            "neighbor_mode": str(estimator.neighbor_mode_),
            "n_neighbors_requested": int(estimator.n_neighbors),
            "n_neighbors_effective": int(
                len(estimator.soft_targets_)
                if kernel_mode
                else min(
                    estimator.n_neighbors,
                    len(estimator.soft_targets_),
                )
            ),
            "neighbor_scope": (
                "all_training_records"
                if kernel_mode
                else "k_with_exact_and_boundary_ties"
            ),
            "distance_power": (
                None
                if kernel_mode
                else float(estimator.distance_power)
            ),
            "bandwidth": (
                float(estimator.bandwidth) if kernel_mode else None
            ),
            "block_weights": _audit_value(estimator.block_weights),
            "feature_weights": _audit_value(estimator.feature_weights),
            "block_weight_profile": block_profile_audit,
            "boundary_ties_included": True,
        }
    elif family == "extra_trees":
        estimator = SoftExtraTreesClassifier(
            numeric_columns=numeric,
            categorical_columns=categorical,
            class_names=class_names,
            n_estimators=int(
                resolved_params.get("n_estimators", 400)
            ),
            max_depth=int(resolved_params.get("max_depth", 3)),
            min_samples_leaf=int(
                resolved_params.get("min_samples_leaf", 6)
            ),
            max_features=resolved_params.get("max_features", 0.7),
            random_state=int(seed),
            n_jobs=1,
        ).fit(
            engineered_train,
            binary_targets,
            sample_weight=weights,
        )
        stage_probabilities = estimator.predict_proba(engineered_test)
        stage_audit = {
            "backend": "soft_extra_trees",
            "n_estimators": int(estimator.n_estimators),
            "max_depth": int(estimator.max_depth),
            "min_samples_leaf": int(estimator.min_samples_leaf),
            "max_features": _audit_value(estimator.max_features),
            "random_state": int(seed),
            "n_jobs": 1,
        }
    else:
        classifier_type, catboost_version = _require_catboost()
        catboost_wrapper = TwoStageSoftCatBoostClassifier(
            categorical_columns=categorical,
            iterations=int(resolved_params.get("iterations", 200)),
            depth=int(resolved_params.get("depth", 3)),
            learning_rate=float(
                resolved_params.get("learning_rate", 0.03)
            ),
            l2_leaf_reg=float(
                resolved_params.get("l2_leaf_reg", 5.0)
            ),
            random_seed=int(seed),
            thread_count=1,
            border_count=int(
                resolved_params.get("border_count", 32)
            ),
            random_strength=float(
                resolved_params.get("random_strength", 0.0)
            ),
        )
        catboost_parameters = (
            catboost_wrapper._validated_parameters()
        )
        catboost_train = _feature_frame(
            engineered_train,
            categorical_columns=categorical,
        )
        catboost_test = _feature_frame(
            engineered_test,
            expected_columns=list(catboost_train.columns),
            categorical_columns=categorical,
        )
        (
            estimator,
            backend,
            stage_audit,
        ) = catboost_wrapper._fit_stage(
            classifier_type=classifier_type,
            frame=catboost_train,
            first_probability=first_target,
            sample_weight=weights,
            class_names=class_names,
            parameters=catboost_parameters,
        )
        first_prediction = catboost_wrapper._predict_first_class(
            estimator,
            backend,
            catboost_test,
        )
        stage_probabilities = np.column_stack(
            [first_prediction, 1.0 - first_prediction]
        )
        stage_audit.update(
            {
                "catboost_version": catboost_version,
                "catboost_parameters": _audit_value(
                    catboost_parameters
                ),
            }
        )

    validated = validate_probability_matrix(
        stage_probabilities,
        n_rows=len(engineered_test),
        n_classes=2,
        name=f"{family} binary-stage predictions",
    )
    audit = {
        "fit_status": "complete",
        "api": "fit_predict_soft_binary_stage",
        "panel": str(panel),
        "model_family": family,
        "model_params": _audit_value(resolved_params),
        "seed": int(seed),
        "rare_min_frequency": int(rare_min_frequency),
        "class_labels": list(class_names),
        "target_probability_semantics": (
            "target_probability is fractional probability of class0"
        ),
        "n_train_records_before_zero_weight_filter": int(
            n_source_train_records
        ),
        "n_zero_weight_records_excluded_before_feature_engineering": (
            excluded_zero_weight_records
        ),
        "zero_weight_records_excluded_before_feature_engineering": True,
        "n_train_records": int(len(engineered_train)),
        "n_test_records": int(len(engineered_test)),
        "total_sample_weight": float(weights.sum()),
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
        "feature_engineering_fit_scope": "supplied_training_rows_only",
        "required_selection_scope": (
            "grouped_inner_oof_within_outer_training_rows"
        ),
        "posterior_draws_expanded": False,
        "class_rows_expanded": False,
        "posterior_argmax_used": False,
        "prediction_probability_audit": probability_matrix_audit(
            validated
        ),
        "stage_audit": stage_audit,
    }
    return validated[:, 0].copy(), audit


class TwoStageSoftCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Two-stage CatBoost fit directly to fractional binary targets.

    CatBoost ``CrossEntropy`` receives ``q_below_high_off`` for the gate.
    The conditional model receives ``q_low / q_below_high_off`` and uses
    ``q_below_high_off`` as its record weight.  CPU training is fixed to one
    thread with no filesystem logging.  A degenerate stage falls back to its
    fold-local soft prior and records that fact in the audit.
    """

    def __init__(
        self,
        *,
        categorical_columns: Sequence[str] = (),
        iterations: int = 200,
        depth: int = 3,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 5.0,
        random_seed: int = 42,
        thread_count: int = 1,
        border_count: int = 32,
        random_strength: float = 0.0,
    ) -> None:
        self.categorical_columns = categorical_columns
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.border_count = border_count
        self.random_strength = random_strength

    def _validated_parameters(self) -> dict[str, Any]:
        if int(self.iterations) < 1:
            raise ValueError("iterations must be positive")
        if int(self.depth) < 1:
            raise ValueError("depth must be positive")
        if not np.isfinite(float(self.learning_rate)) or float(
            self.learning_rate
        ) <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(float(self.l2_leaf_reg)) or float(
            self.l2_leaf_reg
        ) < 0:
            raise ValueError("l2_leaf_reg must be finite and nonnegative")
        if int(self.thread_count) != 1:
            raise ValueError(
                "thread_count is fixed to 1 for deterministic sensitivity fits"
            )
        if int(self.border_count) < 1:
            raise ValueError("border_count must be positive")
        if not np.isfinite(float(self.random_strength)) or float(
            self.random_strength
        ) < 0:
            raise ValueError("random_strength must be finite and nonnegative")
        return {
            "loss_function": "CrossEntropy",
            "eval_metric": "CrossEntropy",
            "iterations": int(self.iterations),
            "depth": int(self.depth),
            "learning_rate": float(self.learning_rate),
            "l2_leaf_reg": float(self.l2_leaf_reg),
            "random_seed": int(self.random_seed),
            "thread_count": 1,
            "border_count": int(self.border_count),
            "random_strength": float(self.random_strength),
            "boosting_type": "Ordered",
            "bootstrap_type": "No",
            "allow_writing_files": False,
            "verbose": False,
            "task_type": "CPU",
        }

    @staticmethod
    def _stage_is_degenerate(
        first_probability: np.ndarray,
        sample_weight: np.ndarray,
    ) -> bool:
        positive = sample_weight > 0
        values = first_probability[positive]
        weights = sample_weight[positive]
        mean = float(np.average(values, weights=weights))
        variance = float(np.average((values - mean) ** 2, weights=weights))
        return variance <= 1.0e-15

    def _fit_stage(
        self,
        *,
        classifier_type: Any,
        frame: pd.DataFrame,
        first_probability: np.ndarray,
        sample_weight: np.ndarray,
        class_names: Sequence[str],
        parameters: Mapping[str, Any],
    ) -> tuple[Any, str, dict[str, Any]]:
        n_records_before_filter = int(len(frame))
        positive_weight = np.asarray(sample_weight, dtype=float) > 0
        n_zero_weight_excluded = int(np.sum(~positive_weight))
        frame = frame.iloc[
            np.flatnonzero(positive_weight)
        ].reset_index(drop=True)
        first_probability = np.asarray(
            first_probability,
            dtype=float,
        )[positive_weight]
        sample_weight = np.asarray(
            sample_weight,
            dtype=float,
        )[positive_weight]
        weighted_mean = float(
            np.average(first_probability, weights=sample_weight)
        )
        if self._stage_is_degenerate(first_probability, sample_weight):
            estimator = FoldLocalBinarySoftPriorClassifier(
                class_names=tuple(class_names)
            ).fit(
                frame,
                first_probability,
                sample_weight=sample_weight,
            )
            backend = "fold_local_binary_soft_prior"
        else:
            estimator = classifier_type(**dict(parameters))
            estimator.fit(
                frame,
                first_probability,
                sample_weight=sample_weight,
                cat_features=list(map(str, self.categorical_columns)),
            )
            backend = "catboost_crossentropy"
        audit = {
            "backend": backend,
            "class_labels": list(map(str, class_names)),
            "first_class_target_mean_weighted": weighted_mean,
            "total_sample_weight": float(sample_weight.sum()),
            "n_positive_weight_records": int(np.sum(sample_weight > 0)),
            "n_records_before_zero_weight_filter": n_records_before_filter,
            "n_zero_weight_records_excluded": n_zero_weight_excluded,
            "zero_weight_records_excluded_before_catboost_fit": True,
            "fractional_binary_target": True,
            "loss_function": (
                "CrossEntropy"
                if backend == "catboost_crossentropy"
                else "fold_local_prior"
            ),
        }
        return estimator, backend, audit

    @staticmethod
    def _predict_first_class(
        estimator: Any,
        backend: str,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        probabilities = validate_probability_matrix(
            estimator.predict_proba(frame),
            n_rows=len(frame),
            n_classes=2,
            name=f"{backend} binary predictions",
        )
        # CatBoost's second column is P(y=1), and y was the probability of
        # the first named stage class.  The fold-local estimator already uses
        # the named [first, second] order.
        return (
            probabilities[:, 1]
            if backend == "catboost_crossentropy"
            else probabilities[:, 0]
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "TwoStageSoftCatBoostClassifier":
        classifier_type, catboost_version = _require_catboost()
        parameters = self._validated_parameters()
        frame = _feature_frame(
            X,
            categorical_columns=self.categorical_columns,
        )
        targets = _validate_soft_targets(
            y,
            n_rows=len(frame),
            n_classes=3,
        )
        outer_weights = _validate_sample_weight(
            sample_weight,
            n_rows=len(frame),
        )
        below_high_mass = targets[:, :2].sum(axis=1)
        gate_first_probability = below_high_mass
        conditional_first_probability = np.divide(
            targets[:, 0],
            below_high_mass,
            out=np.full(len(targets), 0.5, dtype=float),
            where=below_high_mass > 0,
        )
        conditional_weights = outer_weights * below_high_mass
        if not np.any(conditional_weights > 0):
            raise ValueError(
                "Conditional CatBoost stage has zero below-high probability "
                "mass in the supplied training fold"
            )

        (
            self.gate_estimator_,
            self.gate_backend_,
            gate_audit,
        ) = self._fit_stage(
            classifier_type=classifier_type,
            frame=frame,
            first_probability=gate_first_probability,
            sample_weight=outer_weights,
            class_names=SENSITIVITY_GATE_LABELS,
            parameters=parameters,
        )
        (
            self.conditional_estimator_,
            self.conditional_backend_,
            conditional_audit,
        ) = self._fit_stage(
            classifier_type=classifier_type,
            frame=frame,
            first_probability=conditional_first_probability,
            sample_weight=conditional_weights,
            class_names=SENSITIVITY_CONDITIONAL_LABELS,
            parameters=parameters,
        )
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = int(frame.shape[1])
        self.classes_ = np.asarray(SENSITIVITY_CLASS_LABELS, dtype=object)
        self.catboost_parameters_ = dict(parameters)
        self.fit_audit_ = {
            "estimator": "two_stage_soft_catboost",
            "architecture": (
                "below_high_off_gate_then_low60_vs_120_conditional"
            ),
            "class_labels": list(SENSITIVITY_CLASS_LABELS),
            "gate": gate_audit,
            "conditional": conditional_audit,
            "catboost_version": catboost_version,
            "catboost_parameters": dict(parameters),
            "random_seed": int(self.random_seed),
            "thread_count": 1,
            "allow_writing_files": False,
            "feature_names": list(map(str, frame.columns)),
            "categorical_features": list(
                map(str, self.categorical_columns)
            ),
            "n_training_records": int(len(frame)),
            "posterior_draws_expanded": False,
            "class_rows_expanded": False,
            "posterior_argmax_used": False,
            "feature_engineering_fit_scope": (
                "caller_supplied_fold_local_features"
            ),
        }
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "gate_estimator_"):
            raise RuntimeError(
                "TwoStageSoftCatBoostClassifier must be fitted before "
                "predict_proba"
            )
        frame = _feature_frame(
            X,
            expected_columns=list(map(str, self.feature_names_in_)),
            categorical_columns=self.categorical_columns,
        )
        below_high = self._predict_first_class(
            self.gate_estimator_,
            self.gate_backend_,
            frame,
        )
        low_conditional = self._predict_first_class(
            self.conditional_estimator_,
            self.conditional_backend_,
            frame,
        )
        gate = np.column_stack([below_high, 1.0 - below_high])
        conditional = np.column_stack(
            [low_conditional, 1.0 - low_conditional]
        )
        probabilities, combine_audit = combine_two_stage_probabilities(
            gate,
            conditional,
            return_audit=True,
        )
        self.last_prediction_audit_ = {
            "n_prediction_records": int(len(frame)),
            "gate_backend": self.gate_backend_,
            "conditional_backend": self.conditional_backend_,
            "probability_combination": combine_audit,
        }
        return probabilities

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


__all__: Sequence[str] = (
    "SENSITIVITY_CLASS_LABELS",
    "SENSITIVITY_CONDITIONAL_LABELS",
    "SENSITIVITY_GATE_LABELS",
    "SOFT_BINARY_STAGE_MODEL_FAMILIES",
    "ConvexOOFProbabilityStacker",
    "FoldLocalBinarySoftPriorClassifier",
    "TwoStageSoftCatBoostClassifier",
    "combine_two_stage_probabilities",
    "fit_predict_soft_binary_stage",
    "prior_shrink_probabilities",
    "probability_matrix_audit",
    "validate_probability_matrix",
)
