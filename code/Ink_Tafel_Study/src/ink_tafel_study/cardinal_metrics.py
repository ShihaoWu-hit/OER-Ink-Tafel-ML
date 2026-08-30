"""Probability metrics for posterior-aware cardinal-region classification.

The observed target for one record is a probability vector derived from its
posterior slope draws.  Metrics in this module therefore compare probability
vectors directly; they do not convert the target to a hard class first.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


CLASS_LABELS = (
    "low_30_40_like",
    "60_like",
    "120_like",
    "off_cardinal",
)
SOFT_LABEL_COLUMNS = tuple(f"q_{label}" for label in CLASS_LABELS)
PREDICTION_COLUMNS = tuple(f"p_{label}" for label in CLASS_LABELS)
CLASSIC_LABELS = CLASS_LABELS[:-1]


def normalize_probabilities(values: np.ndarray, *, epsilon: float = 1e-15) -> np.ndarray:
    """Return a finite, clipped, row-normalized probability matrix."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Probability values must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError("Probability values contain NaN or infinite entries")
    if (array < -epsilon).any():
        raise ValueError("Probability values contain negative entries")
    clipped = np.maximum(array, 0.0)
    totals = clipped.sum(axis=1, keepdims=True)
    if (totals <= epsilon).any():
        raise ValueError("Each probability row must have positive total mass")
    return clipped / totals


def _validated_pair(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_array = normalize_probabilities(observed)
    predicted_array = normalize_probabilities(predicted)
    if observed_array.shape != predicted_array.shape:
        raise ValueError(
            "Observed and predicted probability matrices have different shapes: "
            f"{observed_array.shape} != {predicted_array.shape}"
        )
    return observed_array, predicted_array


def soft_cross_entropy(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    epsilon: float = 1e-15,
) -> float:
    """Mean cross entropy against posterior soft labels."""

    observed_array, predicted_array = _validated_pair(observed, predicted)
    per_record = -np.sum(
        observed_array * np.log(np.clip(predicted_array, epsilon, 1.0)),
        axis=1,
    )
    return _weighted_mean(per_record, sample_weight)


def multiclass_brier(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Mean sum-of-squares multiclass Brier score.

    The convention is ``mean_i sum_k (p_ik - q_ik)^2`` without division by
    the number of classes.  The convention is persisted in the run registry.
    """

    observed_array, predicted_array = _validated_pair(observed, predicted)
    per_record = np.sum(np.square(predicted_array - observed_array), axis=1)
    return _weighted_mean(per_record, sample_weight)


def _weighted_mean(values: np.ndarray, sample_weight: np.ndarray | None) -> float:
    array = np.asarray(values, dtype=float)
    if sample_weight is None:
        return float(np.mean(array))
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if len(weights) != len(array):
        raise ValueError("sample_weight length does not match metric rows")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample_weight must be finite and nonnegative")
    if float(weights.sum()) <= 0:
        return float("nan")
    return float(np.average(array, weights=weights))


def two_stage_probability_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Score combined, cardinal-gate, and conditional-classic probabilities."""

    observed_array, predicted_array = _validated_pair(observed, predicted)
    observed_cardinal = observed_array[:, :-1].sum(axis=1)
    predicted_cardinal = predicted_array[:, :-1].sum(axis=1)
    observed_gate = np.column_stack([observed_array[:, -1], observed_cardinal])
    predicted_gate = np.column_stack([predicted_array[:, -1], predicted_cardinal])

    conditional_observed = np.divide(
        observed_array[:, :-1],
        observed_cardinal[:, None],
        out=np.full_like(observed_array[:, :-1], 1.0 / len(CLASSIC_LABELS)),
        where=observed_cardinal[:, None] > 0,
    )
    conditional_predicted = np.divide(
        predicted_array[:, :-1],
        predicted_cardinal[:, None],
        out=np.full_like(predicted_array[:, :-1], 1.0 / len(CLASSIC_LABELS)),
        where=predicted_cardinal[:, None] > 0,
    )
    cumulative_observed = np.cumsum(conditional_observed, axis=1)[:, :-1]
    cumulative_predicted = np.cumsum(conditional_predicted, axis=1)[:, :-1]
    per_record_rps = np.sum(
        np.square(cumulative_predicted - cumulative_observed),
        axis=1,
    )
    return {
        "soft_log_loss": soft_cross_entropy(observed_array, predicted_array),
        "multiclass_brier": multiclass_brier(observed_array, predicted_array),
        "gate_soft_log_loss": soft_cross_entropy(observed_gate, predicted_gate),
        "gate_brier": multiclass_brier(observed_gate, predicted_gate),
        "conditional_soft_log_loss": soft_cross_entropy(
            conditional_observed,
            conditional_predicted,
            sample_weight=observed_cardinal,
        ),
        "conditional_brier": multiclass_brier(
            conditional_observed,
            conditional_predicted,
            sample_weight=observed_cardinal,
        ),
        "conditional_rps": _weighted_mean(per_record_rps, observed_cardinal),
        "mean_observed_cardinal_probability": float(np.mean(observed_cardinal)),
        "mean_predicted_cardinal_probability": float(np.mean(predicted_cardinal)),
    }


def collapse_30_and_40(values: np.ndarray) -> np.ndarray:
    """Return low/60/120/off probabilities.

    Four-column inputs already use the hierarchical target and are returned
    unchanged.  Five-column inputs are accepted only as a registry/audit
    compatibility path and have their first two columns collapsed.
    """

    array = normalize_probabilities(values)
    if array.shape[1] == len(CLASS_LABELS):
        return array
    if array.shape[1] != 5:
        raise ValueError(f"Expected four or five classes, found {array.shape[1]}")
    return np.column_stack(
        [
            array[:, 0] + array[:, 1],
            array[:, 2],
            array[:, 3],
            array[:, 4],
        ]
    )


def collapsed_hard_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Secondary hard metrics for low/60/120/off modal labels."""

    observed_hard = np.argmax(collapse_30_and_40(observed), axis=1)
    predicted_hard = np.argmax(collapse_30_and_40(predicted), axis=1)
    return {
        "hierarchical_balanced_accuracy": float(
            balanced_accuracy_score(observed_hard, predicted_hard)
        ),
        "hierarchical_macro_f1": float(
            f1_score(
                observed_hard,
                predicted_hard,
                labels=np.arange(4),
                average="macro",
                zero_division=0,
            )
        ),
    }


def calibration_table(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    class_labels: Sequence[str] = CLASS_LABELS,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Return classwise equal-width calibration bins for soft outcomes."""

    observed_array, predicted_array = _validated_pair(observed, predicted)
    if observed_array.shape[1] != len(class_labels):
        raise ValueError("class_labels length does not match probability columns")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []
    for class_index, class_label in enumerate(class_labels):
        probability = predicted_array[:, class_index]
        outcome = observed_array[:, class_index]
        assignments = np.minimum(
            np.searchsorted(edges, probability, side="right") - 1,
            n_bins - 1,
        )
        assignments = np.maximum(assignments, 0)
        for bin_index in range(n_bins):
            selected = assignments == bin_index
            if not selected.any():
                continue
            rows.append(
                {
                    "class_label": str(class_label),
                    "bin": int(bin_index),
                    "bin_low": float(edges[bin_index]),
                    "bin_high": float(edges[bin_index + 1]),
                    "n_records": int(selected.sum()),
                    "mean_predicted_probability": float(probability[selected].mean()),
                    "mean_observed_soft_probability": float(outcome[selected].mean()),
                    "absolute_calibration_gap": float(
                        abs(probability[selected].mean() - outcome[selected].mean())
                    ),
                }
            )
    return pd.DataFrame(rows)


def expected_confusion_table(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    class_labels: Sequence[str] = CLASS_LABELS,
) -> pd.DataFrame:
    """Return a soft expected confusion matrix normalized by target class mass."""

    observed_array, predicted_array = _validated_pair(observed, predicted)
    if observed_array.shape[1] != len(class_labels):
        raise ValueError("class_labels length does not match probability columns")
    joint = observed_array.T @ predicted_array
    denominators = observed_array.sum(axis=0)
    normalized = np.divide(
        joint,
        denominators[:, None],
        out=np.full_like(joint, np.nan),
        where=denominators[:, None] > 0,
    )
    rows: list[dict[str, object]] = []
    for true_index, true_label in enumerate(class_labels):
        for predicted_index, predicted_label in enumerate(class_labels):
            rows.append(
                {
                    "target_class": str(true_label),
                    "predicted_class": str(predicted_label),
                    "expected_mass": float(joint[true_index, predicted_index]),
                    "row_normalized_probability": float(
                        normalized[true_index, predicted_index]
                    ),
                    "target_effective_mass": float(denominators[true_index]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_repeated_oof(
    oof: pd.DataFrame,
    *,
    key_columns: Sequence[str] = (
        "analysis_id",
        "scope",
        "panel",
        "model_role",
        "model_family",
        "record_id",
        "cv_group_id",
    ),
) -> pd.DataFrame:
    """Average repeated OOF probabilities once per record before inference."""

    if oof.empty:
        return pd.DataFrame()
    required_columns = (
        tuple(key_columns) + SOFT_LABEL_COLUMNS + PREDICTION_COLUMNS
    )
    missing = sorted(set(required_columns) - set(oof.columns))
    if missing:
        raise KeyError(f"OOF table is missing aggregation columns: {missing}")
    aggregation = {
        **{column: "first" for column in SOFT_LABEL_COLUMNS},
        **{column: "mean" for column in PREDICTION_COLUMNS},
    }
    if "panel_role" in oof.columns and "panel_role" not in key_columns:
        aggregation["panel_role"] = "first"
    result = (
        oof.groupby(list(key_columns), dropna=False, as_index=False)
        .agg(aggregation)
    )
    predicted = normalize_probabilities(result.loc[:, PREDICTION_COLUMNS].to_numpy(float))
    result.loc[:, PREDICTION_COLUMNS] = predicted
    counts = (
        oof.groupby(list(key_columns), dropna=False)
        .size()
        .rename("n_repeated_oof")
        .reset_index()
    )
    return result.merge(counts, on=list(key_columns), how="left", validate="one_to_one")
