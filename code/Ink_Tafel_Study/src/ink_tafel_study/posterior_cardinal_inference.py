"""Inference helpers for posterior-aware cardinal classification.

The supervised response is a probability vector, not a posterior-argmax
label.  Soft log loss remains the primary score.  Hard-label quantities in
this module are descriptive diagnostics and must not be used to fit or select
models.

Connected-group resampling operates on record-level deltas.  Sampling a group
therefore copies every record in that group, including repeated copies when a
group is sampled more than once.  The sign-flip randomization uses one shared
Rademacher sign per group and randomization across all feature columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .cardinal_metrics import normalize_probabilities


DEFAULT_BOOTSTRAP_SEED = 20260724
DEFAULT_SIGNFLIP_SEED = 20260725
_BOOTSTRAP_REPLICATE_COLUMN = "bootstrap_replicate"
_RANDOMIZATION_REPLICATE_COLUMN = "randomization_replicate"


def _validated_probability_pair(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_array = normalize_probabilities(observed)
    predicted_array = normalize_probabilities(predicted)
    if observed_array.shape != predicted_array.shape:
        raise ValueError(
            "Observed and predicted probability matrices have different "
            f"shapes: {observed_array.shape} != {predicted_array.shape}"
        )
    if observed_array.shape[0] == 0:
        raise ValueError("Probability metrics require at least one record")
    if observed_array.shape[1] < 2:
        raise ValueError("Probability metrics require at least two classes")
    return observed_array, predicted_array


def _modal_balanced_accuracy(
    observed_class: np.ndarray,
    predicted_class: np.ndarray,
) -> float:
    recalls = [
        float(np.mean(predicted_class[observed_class == class_index] == class_index))
        for class_index in np.unique(observed_class)
    ]
    return float(np.mean(recalls))


def _modal_macro_f1(
    observed_class: np.ndarray,
    predicted_class: np.ndarray,
    *,
    n_classes: int,
) -> float:
    scores: list[float] = []
    for class_index in range(n_classes):
        observed_positive = observed_class == class_index
        predicted_positive = predicted_class == class_index
        true_positive = int(np.sum(observed_positive & predicted_positive))
        false_positive = int(np.sum(~observed_positive & predicted_positive))
        false_negative = int(np.sum(observed_positive & ~predicted_positive))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
    return float(np.mean(scores))


def descriptive_probability_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    entropy_threshold: float = 0.90,
) -> dict[str, float | int]:
    """Return full-coverage soft scores and descriptive hard diagnostics.

    ``target_entropy`` and ``soft_log_loss`` use natural logarithms.  The
    returned ``excess_cross_entropy_KL`` is computed as their difference, so
    the cross-entropy identity

    ``soft_log_loss = target_entropy + excess_cross_entropy_KL``

    is preserved by construction.  ``posterior_expected_hard_accuracy`` is
    the mean target mass assigned to the model's modal prediction.  The oracle
    counterpart is ``mean(max(q_i))`` and quantifies the ceiling for a
    deterministic hard decision when the target remains posterior-distributed.
    """

    threshold = float(entropy_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("entropy_threshold must be finite and in [0, 1]")

    observed_array, predicted_array = _validated_probability_pair(
        observed,
        predicted,
    )
    epsilon = 1e-15
    clipped_predicted = np.clip(predicted_array, epsilon, 1.0)
    clipped_observed = np.clip(observed_array, epsilon, 1.0)

    soft_log_loss = float(
        np.mean(-np.sum(observed_array * np.log(clipped_predicted), axis=1))
    )
    target_entropy = float(
        np.mean(-np.sum(observed_array * np.log(clipped_observed), axis=1))
    )
    excess_cross_entropy = float(soft_log_loss - target_entropy)
    multiclass_brier = float(
        np.mean(np.sum(np.square(predicted_array - observed_array), axis=1))
    )

    observed_class = np.argmax(observed_array, axis=1)
    predicted_class = np.argmax(predicted_array, axis=1)
    modal_accuracy = float(np.mean(observed_class == predicted_class))
    balanced_accuracy = _modal_balanced_accuracy(
        observed_class,
        predicted_class,
    )
    macro_f1 = _modal_macro_f1(
        observed_class,
        predicted_class,
        n_classes=observed_array.shape[1],
    )

    record_positions = np.arange(len(observed_array))
    posterior_expected_hard_accuracy = float(
        np.mean(observed_array[record_positions, predicted_class])
    )
    oracle_posterior_expected_hard_accuracy = float(
        np.mean(np.max(observed_array, axis=1))
    )

    predictive_entropy = -np.sum(
        predicted_array * np.log(np.clip(predicted_array, 1e-300, 1.0)),
        axis=1,
    )
    normalized_predictive_entropy = predictive_entropy / np.log(
        float(predicted_array.shape[1])
    )
    uncertain = normalized_predictive_entropy > threshold

    return {
        "soft_log_loss": soft_log_loss,
        "target_entropy": target_entropy,
        "excess_cross_entropy_KL": excess_cross_entropy,
        "multiclass_brier": multiclass_brier,
        "modal_accuracy": modal_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "posterior_expected_hard_accuracy": (
            posterior_expected_hard_accuracy
        ),
        "oracle_posterior_expected_hard_accuracy": (
            oracle_posterior_expected_hard_accuracy
        ),
        "mean_normalized_predictive_entropy": float(
            np.mean(normalized_predictive_entropy)
        ),
        "uncertain_count": int(np.sum(uncertain)),
        "uncertain_fraction": float(np.mean(uncertain)),
    }


def _validated_resampling_input(
    values: pd.DataFrame,
    metric_columns: Sequence[str],
    *,
    group_col: str,
    reserved_columns: Sequence[str],
) -> tuple[pd.DataFrame, tuple[str, ...], list[np.ndarray]]:
    if not isinstance(values, pd.DataFrame):
        raise TypeError("Resampling input must be a pandas DataFrame")
    if values.empty:
        raise ValueError("Resampling input must contain at least one record")
    if not isinstance(group_col, str) or not group_col:
        raise ValueError("group_col must be a non-empty string")
    columns = (
        (metric_columns,)
        if isinstance(metric_columns, str)
        else tuple(str(column) for column in metric_columns)
    )
    if not columns:
        raise ValueError("At least one metric column is required")
    if len(set(columns)) != len(columns):
        raise ValueError("Metric columns must be unique")
    if any(column in reserved_columns for column in columns):
        raise ValueError("Metric columns conflict with output metadata columns")
    missing = sorted({group_col, *columns} - set(values.columns))
    if missing:
        raise KeyError(f"Resampling input is missing columns: {missing}")

    frame = values.loc[:, [group_col, *columns]].copy().reset_index(drop=True)
    if frame[group_col].isna().any():
        raise ValueError(
            f"{group_col} contains NaN/missing group identifiers"
        )
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(converted.to_numpy(float)).all():
            raise ValueError(f"{column} contains NaN or infinite deltas")
        frame[column] = converted.astype(float)

    grouped = frame.groupby(
        group_col,
        sort=True,
        observed=True,
        dropna=False,
    )
    positions = [
        np.asarray(group_positions, dtype=int)
        for group_positions in grouped.indices.values()
    ]
    if len(positions) < 2:
        raise ValueError("Connected-group inference requires at least two groups")
    return frame, columns, positions


def _validated_random_settings(replicates: int, seed: int) -> tuple[int, int]:
    if isinstance(replicates, bool) or not isinstance(
        replicates, (int, np.integer)
    ):
        raise ValueError("replicates must be a positive integer")
    n_replicates = int(replicates)
    if n_replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return n_replicates, int(seed)


def paired_connected_group_bootstrap(
    deltas: pd.DataFrame,
    metric_columns: Sequence[str],
    group_col: str = "cv_group_id",
    replicates: int = 2000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap paired record deltas by sampling connected groups.

    Each replicate samples ``n_unique_groups`` groups with replacement.  Every
    record belonging to a sampled group is copied, and a group sampled more
    than once contributes all of its records more than once.  Replicate means
    are therefore record-weighted rather than equally weighted across sampled
    group labels.
    """

    n_replicates, random_seed = _validated_random_settings(replicates, seed)
    frame, columns, group_positions = _validated_resampling_input(
        deltas,
        metric_columns,
        group_col=group_col,
        reserved_columns=(
            _BOOTSTRAP_REPLICATE_COLUMN,
            "n_sampled_groups",
            "n_sampled_records",
        ),
    )
    n_groups = len(group_positions)
    metric_values = frame.loc[:, columns].to_numpy(float)
    rng = np.random.default_rng(random_seed)
    bootstrap_values = np.empty((n_replicates, len(columns)), dtype=float)
    sampled_record_counts = np.empty(n_replicates, dtype=int)

    for replicate in range(n_replicates):
        sampled_group_positions = rng.integers(
            0,
            n_groups,
            size=n_groups,
        )
        sampled_records = np.concatenate(
            [group_positions[position] for position in sampled_group_positions]
        )
        sampled_record_counts[replicate] = len(sampled_records)
        bootstrap_values[replicate] = np.mean(
            metric_values[sampled_records],
            axis=0,
        )

    draws = pd.DataFrame(
        bootstrap_values,
        columns=columns,
    )
    draws.insert(0, "n_sampled_records", sampled_record_counts)
    draws.insert(0, "n_sampled_groups", n_groups)
    draws.insert(0, _BOOTSTRAP_REPLICATE_COLUMN, np.arange(n_replicates))

    summary_rows: list[dict[str, float | int | str]] = []
    for column_index, column in enumerate(columns):
        values = bootstrap_values[:, column_index]
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        summary_rows.append(
            {
                "metric": column,
                "observed_mean": float(np.mean(metric_values[:, column_index])),
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_median": float(median),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "probability_below_zero": float(np.mean(values < 0.0)),
                "n_bootstrap": n_replicates,
                "n_records": int(len(frame)),
                "n_connected_groups": n_groups,
            }
        )
    return draws, pd.DataFrame(summary_rows)


def connected_group_signflip_pvalues(
    deltas: pd.DataFrame,
    feature_columns: Sequence[str],
    group_col: str = "cv_group_id",
    replicates: int = 20_000,
    seed: int = DEFAULT_SIGNFLIP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return shared-sign null draws and lower-tail feature p-values.

    For each feature, record deltas are summed within connected group.  Every
    randomization draws one Rademacher sign per group and applies that same
    sign vector to all feature columns.  The random statistic is
    ``sum_g(sign_g * group_sum_g) / N``.  The lower-tail p-value includes ties
    and uses the finite-randomization correction
    ``(1 + count(null <= observed)) / (replicates + 1)``.
    """

    n_replicates, random_seed = _validated_random_settings(replicates, seed)
    frame, columns, group_positions = _validated_resampling_input(
        deltas,
        feature_columns,
        group_col=group_col,
        reserved_columns=(_RANDOMIZATION_REPLICATE_COLUMN,),
    )
    values = frame.loc[:, columns].to_numpy(float)
    group_sums = np.vstack(
        [np.sum(values[positions], axis=0) for positions in group_positions]
    )
    observed = np.sum(group_sums, axis=0) / len(frame)

    rng = np.random.default_rng(random_seed)
    signs = 2 * rng.integers(
        0,
        2,
        size=(n_replicates, len(group_positions)),
        dtype=np.int8,
    ) - 1
    null_values = signs.astype(float) @ group_sums / len(frame)

    draws = pd.DataFrame(null_values, columns=columns)
    draws.insert(
        0,
        _RANDOMIZATION_REPLICATE_COLUMN,
        np.arange(n_replicates),
    )

    summary_rows: list[dict[str, float | int | str]] = []
    for column_index, column in enumerate(columns):
        feature_null = null_values[:, column_index]
        lower_tail_count = int(np.sum(feature_null <= observed[column_index]))
        summary_rows.append(
            {
                "feature": column,
                "observed_mean_delta": float(observed[column_index]),
                "p_value_lower_tail": float(
                    (1 + lower_tail_count) / (n_replicates + 1)
                ),
                "lower_tail_count_including_ties": lower_tail_count,
                "n_randomizations": n_replicates,
                "n_records": int(len(frame)),
                "n_connected_groups": int(len(group_positions)),
                "shared_group_signs_across_features": True,
            }
        )
    return draws, pd.DataFrame(summary_rows)


def holm_one_sided_adjustment(
    p_values: Mapping[str, float],
) -> dict[str, float]:
    """Apply Holm's step-down adjustment to named one-sided p-values."""

    if not isinstance(p_values, Mapping):
        raise TypeError("p_values must be a mapping from name to p-value")
    if not p_values:
        raise ValueError("p_values must be non-empty")

    validated: list[tuple[str, float]] = []
    for name, raw_value in p_values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Holm family names must be non-empty strings")
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Holm p-values must be finite and in [0, 1]")
        validated.append((name, value))

    ordered = sorted(validated, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    family_size = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * value)
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = min(1.0, running_maximum)
    return {name: adjusted[name] for name in p_values}


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_SIGNFLIP_SEED",
    "connected_group_signflip_pvalues",
    "descriptive_probability_metrics",
    "holm_one_sided_adjustment",
    "paired_connected_group_bootstrap",
]
