#!/usr/bin/env python3
"""Outer-test descriptor reliance for probability and High-screening models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


from joblib import Parallel, delayed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ink_tafel_study.cardinal_metrics import (  # noqa: E402
    multiclass_brier,
    normalize_probabilities,
    soft_cross_entropy,
)
from ink_tafel_study.cardinal_models import (  # noqa: E402
    SoftGowerKNeighborsClassifier,
)
from ink_tafel_study.models import GowerKNeighborsRegressor  # noqa: E402
from ink_tafel_study.posterior_cardinal_contiguous_threeclass import (  # noqa: E402
    CLASS_LABELS,
    P_COLUMNS,
    Q_COLUMNS,
)
from ink_tafel_study.posterior_cardinal_contiguous_threeclass_direct_gower import (  # noqa: E402
    DISTANCE_POWER as DIRECT_DISTANCE_POWER,
    EPSILON as DIRECT_EPSILON,
    NEIGHBOR_COUNTS,
)
from ink_tafel_study.posterior_cardinal_contiguous_threeclass_presto_gate import (  # noqa: E402
    L1_SMOOTHING,
    NOVELTY_NEIGHBORS,
    RARE_MIN_FREQUENCY,
    RIDGE,
    _k_smallest_mean,
)
from ink_tafel_study.presto_ordinal import (  # noqa: E402
    SoftFusedContinuationRatioClassifier,
)
from ink_tafel_study.threeclass_sensitivity_models import (  # noqa: E402
    _fold_local_engineered_features,
    probability_matrix_audit,
)


EXPECTED_PROBABILITY_MODEL = "presto_high_gate_direct_gower_conditional"
EXPECTED_POLICY = "presto_policy_only_soft_f05_mean_margin_v2"
EXPECTED_POLICY_SCORE_MODEL = "presto_inspired_policy_only_high_score_v2"
EXPECTED_RECORDS = 97
EXPECTED_GROUPS = 80
EXPECTED_OUTER_ROWS = 970
DEFAULT_PERMUTATIONS = 100
DEFAULT_BOOTSTRAPS = 2000
PERMUTATION_SEED = 20260726
BOOTSTRAP_SEED = 20260726
F_BETA = 0.5
BASELINE_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class FeatureSpec:
    feature: str
    display_name: str
    branch: str
    high_column: str | None
    conditional_column: str | None
    high_block: str | None
    conditional_block: str | None
    source_columns: tuple[str, ...]


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "catalyst_family",
        "Catalyst family",
        "both_branches",
        "catalyst_family",
        "catalyst_family",
        "context",
        "context",
        ("catalyst_active_material_type",),
    ),
    FeatureSpec(
        "catalyst_loading",
        "Catalyst loading",
        "both_branches",
        "log10_catalyst_loading_mg_cm2",
        "log10_catalyst_loading_mg_cm2",
        "context",
        "context",
        ("catalyst_loading_mg_cm2_selected",),
    ),
    FeatureSpec(
        "electrolyte_molarity",
        "Electrolyte molarity",
        "both_branches",
        "log10_electrolyte_molarity_M",
        "log10_electrolyte_molarity_M",
        "context",
        "context",
        ("electrolyte_molarity_M",),
    ),
    FeatureSpec(
        "rotation_rate",
        "Rotation rate",
        "both_branches",
        "log1p_rpm",
        "log1p_rpm",
        "context",
        "context",
        ("rpm",),
    ),
    FeatureSpec(
        "scan_rate",
        "Scan rate",
        "both_branches",
        "log10_scan_rate_mV_s",
        "log10_scan_rate_mV_s",
        "context",
        "context",
        ("scan_rate_mV_s",),
    ),
    FeatureSpec(
        "ir_compensation",
        "iR compensation",
        "both_branches",
        "ir_correction_status",
        "ir_correction_status",
        "context",
        "context",
        ("ir_correction_status",),
    ),
    FeatureSpec(
        "electrolyte_scope",
        "Electrolyte scope",
        "both_branches",
        "electrolyte_scope",
        "electrolyte_scope",
        "context",
        "context",
        ("electrolyte_scope",),
    ),
    FeatureSpec(
        "solvent_polarity_coordinate",
        "Solvent polarity coordinate",
        "high_gate_only",
        "solvent_polarity_coordinate",
        None,
        "solvent_polarity",
        None,
        (
            "solvent_mixture_property_proxy_surface_tension_mN_m",
            "solvent_mixture_property_proxy_density_kg_m3",
            "solvent_mixture_property_proxy_dielectric_constant_best_available",
        ),
    ),
    FeatureSpec(
        "ionomer_to_solvent_ratio",
        "Ionomer / solvent ratio",
        "conditional_only",
        None,
        "log1p_ionomer_to_solvent_ratio",
        None,
        "ionomer_to_solvent_ratio",
        ("ionomer_to_solvent_ratio",),
    ),
)


def _indexed(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    output = frame.copy()
    if "record_id" not in output.columns:
        raise KeyError(f"{name} lacks record_id")
    output["record_id"] = output["record_id"].astype(str)
    if output["record_id"].duplicated().any():
        raise ValueError(f"{name} has duplicate record_id values")
    return output.set_index("record_id", drop=False)






def _gower_weight(
    blocks: Mapping[str, Sequence[str]],
    feature: str | None,
) -> float:
    if feature is None:
        return 0.0
    normalized = {
        str(block): tuple(map(str, members)) for block, members in blocks.items()
    }
    for members in normalized.values():
        if str(feature) in members:
            return float(1.0 / len(normalized) / len(members))
    return 0.0


def _repeat_permuted_frame(
    frame: pd.DataFrame,
    column: str,
    permutations: np.ndarray,
) -> pd.DataFrame:
    repeated = pd.concat(
        [frame] * int(permutations.shape[0]),
        ignore_index=True,
    )
    source = frame[str(column)].to_numpy(copy=True)
    repeated[str(column)] = np.concatenate(
        [source[index] for index in permutations],
        axis=0,
    )
    return repeated


def _fit_presto_estimator(
    novelty_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    penalty: float,
) -> SoftFusedContinuationRatioClassifier:
    estimator = SoftFusedContinuationRatioClassifier(
        numeric_columns=("gower_novelty",),
        categorical_columns=(),
        class_names=CLASS_LABELS,
        penalty=float(penalty),
        ridge=RIDGE,
        l1_smoothing=L1_SMOOTHING,
        max_iter=2000,
        tolerance=1.0e-9,
    ).fit(novelty_train, y_train)
    if estimator.fit_status_ != "complete":
        raise RuntimeError(
            "PRESTO refit did not converge: "
            f"{estimator.optimization_audit_}"
        )
    return estimator


def _fit_high_branch(
    X_train_raw: pd.DataFrame,
    train_groups: Sequence[str],
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    *,
    probability_penalty: float,
    policy_penalty: float,
) -> dict[str, Any]:
    builder, engineered_train, engineered_test, _ = (
        _fold_local_engineered_features(
            X_train_raw,
            X_test_raw,
            panel="context_plus_polarity",
            rare_min_frequency=RARE_MIN_FREQUENCY,
        )
    )
    blocks = builder.get_blocks_out()
    distance_model = GowerKNeighborsRegressor(
        n_neighbors=1,
        distance_power=1.0,
        numeric_features=builder.numeric_columns_out_,
        categorical_features=builder.categorical_columns_out_,
        blocks=blocks,
        epsilon=1.0e-12,
    ).fit(engineered_train, np.zeros(len(engineered_train), dtype=float))
    groups = np.asarray(list(map(str, train_groups)), dtype=object)
    train_distances = distance_model._pairwise_distances(engineered_train)
    train_distances[groups[:, None] == groups[None, :]] = np.inf
    train_score = _k_smallest_mean(train_distances, NOVELTY_NEIGHBORS)
    test_score = _k_smallest_mean(
        distance_model._pairwise_distances(engineered_test),
        NOVELTY_NEIGHBORS,
    )
    novelty_train = pd.DataFrame({"gower_novelty": train_score})
    novelty_test = pd.DataFrame({"gower_novelty": test_score})
    probability_estimator = _fit_presto_estimator(
        novelty_train,
        y_train,
        penalty=float(probability_penalty),
    )
    policy_estimator = _fit_presto_estimator(
        novelty_train,
        y_train,
        penalty=float(policy_penalty),
    )
    probability = probability_estimator.predict_proba(novelty_test)
    policy_score = policy_estimator.predict_proba(novelty_test)[:, 2]
    probability_matrix_audit(probability)
    return {
        "builder": builder,
        "engineered_test": engineered_test,
        "distance_model": distance_model,
        "probability_estimator": probability_estimator,
        "policy_estimator": policy_estimator,
        "baseline_probability": probability,
        "baseline_policy_score": policy_score,
        "blocks": blocks,
    }


def _direct_probabilities_from_distances(
    estimator: SoftGowerKNeighborsClassifier,
    distances: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(distances), len(CLASS_LABELS)), dtype=float)
    n_train = len(estimator.soft_targets_)
    for row_index, row_distances in enumerate(distances):
        components = [
            estimator._knn_probability(  # noqa: SLF001 - frozen implementation
                row_distances,
                min(int(count), n_train),
            )
            for count in NEIGHBOR_COUNTS
        ]
        output[row_index] = np.mean(np.stack(components, axis=0), axis=0)
    return normalize_probabilities(output)


def _fit_conditional_branch(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, Any]:
    builder, engineered_train, engineered_test, _ = (
        _fold_local_engineered_features(
            X_train_raw,
            X_test_raw,
            panel="context_plus_ionomer_ratio",
            rare_min_frequency=RARE_MIN_FREQUENCY,
        )
    )
    blocks = builder.get_blocks_out()
    estimator = SoftGowerKNeighborsClassifier(
        numeric_columns=builder.numeric_columns_out_,
        categorical_columns=builder.categorical_columns_out_,
        class_names=CLASS_LABELS,
        blocks=blocks,
        block_weights=None,
        feature_weights=None,
        n_neighbors=max(NEIGHBOR_COUNTS),
        distance_power=DIRECT_DISTANCE_POWER,
        epsilon=DIRECT_EPSILON,
        neighbor_mode="knn",
        bandwidth=1.0,
    ).fit(engineered_train, y_train)
    distances = estimator.distance_model_._pairwise_distances(engineered_test)
    probability = _direct_probabilities_from_distances(estimator, distances)
    probability_matrix_audit(probability)
    return {
        "builder": builder,
        "engineered_test": engineered_test,
        "estimator": estimator,
        "baseline_probability": probability,
        "blocks": blocks,
    }


def _hybrid_probabilities(
    high_source: np.ndarray,
    direct_source: np.ndarray,
) -> np.ndarray:
    high_probability = np.asarray(high_source, dtype=float)
    direct_probability = np.asarray(direct_source, dtype=float)
    if high_probability.shape != direct_probability.shape:
        raise ValueError("PRESTO and direct-Gower predictions do not align")
    non_high = direct_probability[:, :2].sum(axis=1)
    conditional = np.divide(
        direct_probability[:, :2],
        non_high[:, None],
        out=np.full((len(direct_probability), 2), 0.5, dtype=float),
        where=non_high[:, None] > 0,
    )
    high = high_probability[:, 2]
    return normalize_probabilities(
        np.column_stack(
            [
                (1.0 - high) * conditional[:, 0],
                (1.0 - high) * conditional[:, 1],
                high,
            ]
        )
    )


def _selection_row(
    frame: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
) -> pd.Series:
    selected = frame.loc[
        frame["repeat"].eq(int(repeat)) & frame["fold"].eq(int(fold))
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one selection for repeat={repeat}, fold={fold}")
    return selected.iloc[0]


def _saved_split(
    frame: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    test_ids: Sequence[str],
) -> pd.DataFrame:
    selected = frame.loc[
        frame["repeat"].eq(int(repeat)) & frame["fold"].eq(int(fold))
    ].copy()
    selected["record_id"] = selected["record_id"].astype(str)
    indexed = selected.set_index("record_id")
    ordered = list(map(str, test_ids))
    if set(indexed.index) != set(ordered):
        raise ValueError("Saved split rows do not match outer-test IDs")
    return indexed.loc[ordered].reset_index()


def _run_outer_split(
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    parent_selections: pd.DataFrame,
    policy_selections: pd.DataFrame,
    saved_probability: pd.DataFrame,
    saved_policy: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    test_ids: Sequence[str],
    n_permutations: int,
) -> dict[str, Any]:
    ordered_ids = records_by_id.index.astype(str).tolist()
    test = list(map(str, test_ids))
    test_set = set(test)
    train = [record_id for record_id in ordered_ids if record_id not in test_set]
    train_groups = records_by_id.loc[train, "cv_group_id"].astype(str).tolist()
    test_groups = set(records_by_id.loc[test, "cv_group_id"].astype(str))
    if set(train_groups) & test_groups:
        raise AssertionError("A connected group crosses an outer split")

    probability_selection = _selection_row(
        parent_selections,
        repeat=repeat,
        fold=fold,
    )
    policy_selection = _selection_row(
        policy_selections,
        repeat=repeat,
        fold=fold,
    )
    probability_penalty = float(probability_selection["selected_penalty"])
    policy_penalty = float(policy_selection["selected_policy_penalty"])
    threshold = float(policy_selection["selected_high_threshold"])

    X_train_raw = records_by_id.loc[train].reset_index(drop=True)
    X_test_raw = records_by_id.loc[test].reset_index(drop=True)
    y_train = labels_by_id.loc[train, Q_COLUMNS].to_numpy(float)
    high = _fit_high_branch(
        X_train_raw,
        train_groups,
        X_test_raw,
        y_train,
        probability_penalty=probability_penalty,
        policy_penalty=policy_penalty,
    )
    conditional = _fit_conditional_branch(
        X_train_raw,
        X_test_raw,
        y_train,
    )
    baseline_probability = _hybrid_probabilities(
        high["baseline_probability"],
        conditional["baseline_probability"],
    )
    baseline_policy_score = np.asarray(
        high["baseline_policy_score"], dtype=float
    )
    baseline_margin = baseline_policy_score - threshold

    saved_probability_split = _saved_split(
        saved_probability,
        repeat=repeat,
        fold=fold,
        test_ids=test,
    )
    saved_policy_split = _saved_split(
        saved_policy,
        repeat=repeat,
        fold=fold,
        test_ids=test,
    )
    probability_error = float(
        np.max(
            np.abs(
                baseline_probability
                - saved_probability_split.loc[:, P_COLUMNS].to_numpy(float)
            )
        )
    )
    score_error = float(
        np.max(
            np.abs(
                baseline_policy_score
                - saved_policy_split["high_score"].to_numpy(float)
            )
        )
    )
    threshold_error = float(
        np.max(
            np.abs(
                threshold
                - saved_policy_split["selected_high_threshold"].to_numpy(float)
            )
        )
    )
    if max(probability_error, score_error, threshold_error) > BASELINE_TOLERANCE:
        raise RuntimeError(
            "Unperturbed outer refit does not reproduce the verified OOF: "
            f"probability={probability_error}, score={score_error}, "
            f"threshold={threshold_error}"
        )

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [PERMUTATION_SEED, int(repeat), int(fold)]
        )
    )
    permutation_indices = np.vstack(
        [rng.permutation(len(test)) for _ in range(int(n_permutations))]
    )
    n_test = len(test)
    repeated_high_baseline = np.repeat(
        high["baseline_probability"][None, :, :],
        int(n_permutations),
        axis=0,
    )
    repeated_policy_score = np.repeat(
        baseline_policy_score[None, :],
        int(n_permutations),
        axis=0,
    )
    repeated_direct_baseline = np.repeat(
        conditional["baseline_probability"][None, :, :],
        int(n_permutations),
        axis=0,
    )

    permuted_probability: dict[str, np.ndarray] = {}
    permuted_margin: dict[str, np.ndarray] = {}
    weight_rows: list[dict[str, Any]] = []
    for spec in FEATURES:
        high_probability = repeated_high_baseline
        policy_score = repeated_policy_score
        direct_probability = repeated_direct_baseline

        if spec.high_column is not None and spec.high_column in high[
            "engineered_test"
        ].columns:
            high_frame = _repeat_permuted_frame(
                high["engineered_test"],
                spec.high_column,
                permutation_indices,
            )
            high_score = _k_smallest_mean(
                high["distance_model"]._pairwise_distances(high_frame),
                NOVELTY_NEIGHBORS,
            )
            novelty = pd.DataFrame({"gower_novelty": high_score})
            high_probability = high["probability_estimator"].predict_proba(
                novelty
            ).reshape(int(n_permutations), n_test, len(CLASS_LABELS))
            policy_score = high["policy_estimator"].predict_proba(
                novelty
            )[:, 2].reshape(int(n_permutations), n_test)

        if spec.conditional_column is not None and spec.conditional_column in (
            conditional["engineered_test"].columns
        ):
            conditional_frame = _repeat_permuted_frame(
                conditional["engineered_test"],
                spec.conditional_column,
                permutation_indices,
            )
            distances = conditional[
                "estimator"
            ].distance_model_._pairwise_distances(conditional_frame)
            direct_probability = _direct_probabilities_from_distances(
                conditional["estimator"],
                distances,
            ).reshape(int(n_permutations), n_test, len(CLASS_LABELS))

        flattened_hybrid = _hybrid_probabilities(
            high_probability.reshape(-1, len(CLASS_LABELS)),
            direct_probability.reshape(-1, len(CLASS_LABELS)),
        )
        permuted_probability[spec.feature] = flattened_hybrid.reshape(
            int(n_permutations), n_test, len(CLASS_LABELS)
        )
        permuted_margin[spec.feature] = policy_score - threshold
        weight_rows.append(
            {
                "feature": spec.feature,
                "repeat": int(repeat),
                "fold": int(fold),
                "high_gate_gower_weight": _gower_weight(
                    high["blocks"], spec.high_column
                ),
                "conditional_gower_weight": _gower_weight(
                    conditional["blocks"], spec.conditional_column
                ),
            }
        )

    return {
        "repeat": int(repeat),
        "fold": int(fold),
        "test_ids": test,
        "baseline_probability": baseline_probability,
        "baseline_margin": baseline_margin,
        "permuted_probability": permuted_probability,
        "permuted_margin": permuted_margin,
        "weight_rows": weight_rows,
        "probability_max_abs_error": probability_error,
        "policy_score_max_abs_error": score_error,
        "threshold_max_abs_error": threshold_error,
    }


def _soft_high_metrics(
    q_high: np.ndarray,
    predicted_high: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(q_high, dtype=float)
    predicted = np.asarray(predicted_high, dtype=bool)
    tp = float(np.sum(target * predicted))
    fn = float(np.sum(target * (~predicted)))
    fp = float(np.sum((1.0 - target) * predicted))
    tn = float(np.sum((1.0 - target) * (~predicted)))
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    beta_squared = F_BETA**2
    denominator = beta_squared * precision + recall
    f_beta = (
        (1.0 + beta_squared) * precision * recall / denominator
        if denominator > 0
        else 0.0
    )
    return {
        "soft_f0_5": float(f_beta),
        "soft_high_recall": float(recall),
        "soft_high_precision": float(precision),
        "soft_high_false_positive_rate": float(fpr),
        "predicted_high_count": int(np.sum(predicted)),
    }


def _policy_decisions(
    probability: np.ndarray,
    margin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probability, dtype=float)
    m = np.asarray(margin, dtype=float)
    high = m >= 0.0
    if p.ndim == 2:
        decision = np.argmax(p[:, :2], axis=1)
    elif p.ndim == 3:
        decision = np.argmax(p[:, :, :2], axis=2)
    else:
        raise ValueError("Policy probability array must have 2 or 3 dimensions")
    decision = np.asarray(decision, dtype=int)
    decision[high] = 2
    return decision, high


def _posterior_expected_accuracy(
    q: np.ndarray,
    decision: np.ndarray,
) -> float | np.ndarray:
    targets = np.asarray(q, dtype=float)
    choices = np.asarray(decision, dtype=int)
    if choices.ndim == 1:
        return float(np.mean(targets[np.arange(len(targets)), choices]))
    expanded = np.broadcast_to(
        targets[None, :, :],
        (choices.shape[0], targets.shape[0], targets.shape[1]),
    )
    selected = np.take_along_axis(
        expanded,
        choices[:, :, None],
        axis=2,
    )[:, :, 0]
    return np.mean(selected, axis=1)


def _vector_soft_high_metrics(
    q_high: np.ndarray,
    predicted_high: np.ndarray,
) -> dict[str, np.ndarray]:
    target = np.asarray(q_high, dtype=float)
    predicted = np.asarray(predicted_high, dtype=bool)
    tp = predicted @ target
    fn = (~predicted) @ target
    fp = predicted @ (1.0 - target)
    tn = (~predicted) @ (1.0 - target)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) > 0)
    beta_squared = F_BETA**2
    denominator = beta_squared * precision + recall
    f_beta = np.divide(
        (1.0 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return {
        "soft_f0_5": f_beta,
        "soft_high_recall": recall,
        "soft_high_precision": precision,
        "soft_high_false_positive_rate": fpr,
        "predicted_high_count": np.sum(predicted, axis=1).astype(float),
    }


def _aggregate_outer_results(
    results: Sequence[dict[str, Any]],
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    *,
    n_permutations: int,
) -> dict[str, Any]:
    record_ids = records_by_id.index.astype(str).tolist()
    position = {record_id: index for index, record_id in enumerate(record_ids)}
    counts = np.zeros(len(record_ids), dtype=int)
    baseline_probability_sum = np.zeros((len(record_ids), len(CLASS_LABELS)))
    baseline_margin_sum = np.zeros(len(record_ids), dtype=float)
    permuted_probability_sum = {
        spec.feature: np.zeros(
            (int(n_permutations), len(record_ids), len(CLASS_LABELS)),
            dtype=float,
        )
        for spec in FEATURES
    }
    permuted_margin_sum = {
        spec.feature: np.zeros(
            (int(n_permutations), len(record_ids)), dtype=float
        )
        for spec in FEATURES
    }
    weight_rows: list[dict[str, Any]] = []
    for result in results:
        indices = np.asarray(
            [position[record_id] for record_id in result["test_ids"]],
            dtype=int,
        )
        counts[indices] += 1
        baseline_probability_sum[indices] += result["baseline_probability"]
        baseline_margin_sum[indices] += result["baseline_margin"]
        for spec in FEATURES:
            permuted_probability_sum[spec.feature][:, indices, :] += result[
                "permuted_probability"
            ][spec.feature]
            permuted_margin_sum[spec.feature][:, indices] += result[
                "permuted_margin"
            ][spec.feature]
        weight_rows.extend(result["weight_rows"])

    expected_repeats = 10
    if not np.all(counts == expected_repeats):
        raise AssertionError(
            f"Repeated OOF coverage changed: {np.unique(counts, return_counts=True)}"
        )
    baseline_probability = normalize_probabilities(
        baseline_probability_sum / counts[:, None]
    )
    baseline_margin = baseline_margin_sum / counts
    permuted_probability = {
        feature: values / counts[None, :, None]
        for feature, values in permuted_probability_sum.items()
    }
    for values in permuted_probability.values():
        probability_matrix_audit(values.reshape(-1, len(CLASS_LABELS)))
    permuted_margin = {
        feature: values / counts[None, :]
        for feature, values in permuted_margin_sum.items()
    }
    q = labels_by_id.loc[record_ids, Q_COLUMNS].to_numpy(float)
    groups = records_by_id.loc[record_ids, "cv_group_id"].astype(str).to_numpy()
    return {
        "record_ids": record_ids,
        "groups": groups,
        "q": q,
        "baseline_probability": baseline_probability,
        "baseline_margin": baseline_margin,
        "permuted_probability": permuted_probability,
        "permuted_margin": permuted_margin,
        "weights": pd.DataFrame(weight_rows),
        "maximum_probability_refit_error": max(
            float(result["probability_max_abs_error"]) for result in results
        ),
        "maximum_policy_score_refit_error": max(
            float(result["policy_score_max_abs_error"]) for result in results
        ),
        "maximum_threshold_error": max(
            float(result["threshold_max_abs_error"]) for result in results
        ),
    }


def _observed_and_permutation_metrics(
    aggregated: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    q = np.asarray(aggregated["q"], dtype=float)
    baseline_probability = np.asarray(
        aggregated["baseline_probability"], dtype=float
    )
    baseline_margin = np.asarray(aggregated["baseline_margin"], dtype=float)
    baseline_decision, baseline_high = _policy_decisions(
        baseline_probability,
        baseline_margin,
    )
    baseline_high_metrics = _soft_high_metrics(q[:, 2], baseline_high)
    baseline = {
        "soft_log_loss": soft_cross_entropy(q, baseline_probability),
        "multiclass_brier": multiclass_brier(q, baseline_probability),
        "posterior_expected_hard_accuracy": _posterior_expected_accuracy(
            q, baseline_decision
        ),
        **baseline_high_metrics,
    }

    rows: list[dict[str, Any]] = []
    derived: dict[str, Any] = {
        "baseline_decision": baseline_decision,
        "baseline_high": baseline_high,
        "permuted_decision": {},
        "permuted_high": {},
    }
    for spec in FEATURES:
        probability = np.asarray(
            aggregated["permuted_probability"][spec.feature], dtype=float
        )
        margin = np.asarray(
            aggregated["permuted_margin"][spec.feature], dtype=float
        )
        decision, high = _policy_decisions(probability, margin)
        derived["permuted_decision"][spec.feature] = decision
        derived["permuted_high"][spec.feature] = high
        expected_accuracy = np.asarray(
            _posterior_expected_accuracy(q, decision), dtype=float
        )
        high_metrics = _vector_soft_high_metrics(q[:, 2], high)
        probability_losses = np.asarray(
            [soft_cross_entropy(q, value) for value in probability],
            dtype=float,
        )
        probability_brier = np.asarray(
            [multiclass_brier(q, value) for value in probability],
            dtype=float,
        )
        alert_flips = np.sum(high != baseline_high[None, :], axis=1)
        for permutation in range(len(probability)):
            rows.append(
                {
                    "feature": spec.feature,
                    "display_name": spec.display_name,
                    "branch": spec.branch,
                    "permutation": int(permutation),
                    "soft_log_loss": float(probability_losses[permutation]),
                    "importance_soft_log_loss": float(
                        probability_losses[permutation]
                        - baseline["soft_log_loss"]
                    ),
                    "multiclass_brier": float(probability_brier[permutation]),
                    "importance_multiclass_brier": float(
                        probability_brier[permutation]
                        - baseline["multiclass_brier"]
                    ),
                    "soft_f0_5": float(
                        high_metrics["soft_f0_5"][permutation]
                    ),
                    "importance_soft_f0_5": float(
                        baseline["soft_f0_5"]
                        - high_metrics["soft_f0_5"][permutation]
                    ),
                    "posterior_expected_hard_accuracy": float(
                        expected_accuracy[permutation]
                    ),
                    "importance_posterior_expected_hard_accuracy": float(
                        baseline["posterior_expected_hard_accuracy"]
                        - expected_accuracy[permutation]
                    ),
                    "soft_high_recall": float(
                        high_metrics["soft_high_recall"][permutation]
                    ),
                    "soft_high_precision": float(
                        high_metrics["soft_high_precision"][permutation]
                    ),
                    "soft_high_false_positive_rate": float(
                        high_metrics[
                            "soft_high_false_positive_rate"
                        ][permutation]
                    ),
                    "predicted_high_count": int(
                        high_metrics["predicted_high_count"][permutation]
                    ),
                    "alert_flip_count": int(alert_flips[permutation]),
                }
            )
    return baseline, pd.DataFrame(rows), derived


def _bootstrap_indices(groups: np.ndarray, repetitions: int) -> list[np.ndarray]:
    group_values = np.asarray(groups, dtype=object)
    unique = np.asarray(sorted(set(map(str, group_values))), dtype=object)
    if len(unique) != EXPECTED_GROUPS:
        raise ValueError("Connected-group bootstrap requires 80 groups")
    lookup = {
        str(group): np.flatnonzero(group_values == str(group)) for group in unique
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: list[np.ndarray] = []
    for _ in range(int(repetitions)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        output.append(
            np.concatenate([lookup[str(group)] for group in sampled])
        )
    return output


def _bootstrap_reliance(
    aggregated: Mapping[str, Any],
    derived: Mapping[str, Any],
    *,
    repetitions: int,
) -> pd.DataFrame:
    q = np.asarray(aggregated["q"], dtype=float)
    baseline_probability = np.asarray(
        aggregated["baseline_probability"], dtype=float
    )
    baseline_high = np.asarray(derived["baseline_high"], dtype=bool)
    baseline_decision = np.asarray(derived["baseline_decision"], dtype=int)
    indices_list = _bootstrap_indices(
        np.asarray(aggregated["groups"], dtype=object), repetitions
    )
    rows: list[dict[str, Any]] = []
    for replicate, indices in enumerate(indices_list):
        q_selected = q[indices]
        baseline_probability_selected = baseline_probability[indices]
        baseline_loss = soft_cross_entropy(
            q_selected, baseline_probability_selected
        )
        baseline_brier = multiclass_brier(
            q_selected, baseline_probability_selected
        )
        baseline_high_metrics = _soft_high_metrics(
            q_selected[:, 2], baseline_high[indices]
        )
        baseline_accuracy = float(
            _posterior_expected_accuracy(
                q_selected, baseline_decision[indices]
            )
        )
        for spec in FEATURES:
            probability = np.asarray(
                aggregated["permuted_probability"][spec.feature], dtype=float
            )[:, indices, :]
            high = np.asarray(
                derived["permuted_high"][spec.feature], dtype=bool
            )[:, indices]
            decision = np.asarray(
                derived["permuted_decision"][spec.feature], dtype=int
            )[:, indices]
            probability_loss = np.mean(
                [soft_cross_entropy(q_selected, value) for value in probability]
            )
            probability_brier = np.mean(
                [multiclass_brier(q_selected, value) for value in probability]
            )
            high_metrics = _vector_soft_high_metrics(q_selected[:, 2], high)
            expected_accuracy = np.asarray(
                _posterior_expected_accuracy(q_selected, decision), dtype=float
            )
            rows.append(
                {
                    "feature": spec.feature,
                    "replicate": int(replicate),
                    "importance_soft_log_loss": float(
                        probability_loss - baseline_loss
                    ),
                    "importance_multiclass_brier": float(
                        probability_brier - baseline_brier
                    ),
                    "importance_soft_f0_5": float(
                        baseline_high_metrics["soft_f0_5"]
                        - np.mean(high_metrics["soft_f0_5"])
                    ),
                    "importance_posterior_expected_hard_accuracy": float(
                        baseline_accuracy - np.mean(expected_accuracy)
                    ),
                    "permuted_minus_baseline_soft_high_fpr": float(
                        np.mean(
                            high_metrics["soft_high_false_positive_rate"]
                        )
                        - baseline_high_metrics[
                            "soft_high_false_positive_rate"
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _interval_status(low: float, high: float) -> str:
    if low > 0:
        return "positive_conditional_interval"
    if high < 0:
        return "negative_conditional_interval"
    return "conditional_interval_crosses_zero"


def _feature_summary(
    baseline: Mapping[str, Any],
    permutation_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    n_permutations: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for order, spec in enumerate(FEATURES):
        observed = permutation_metrics.loc[
            permutation_metrics["feature"].eq(spec.feature)
        ].copy()
        draws = bootstrap.loc[bootstrap["feature"].eq(spec.feature)].copy()
        weight = weights.loc[weights["feature"].eq(spec.feature)].copy()
        probability_values = draws["importance_soft_log_loss"].to_numpy(float)
        policy_values = draws["importance_soft_f0_5"].to_numpy(float)
        accuracy_values = draws[
            "importance_posterior_expected_hard_accuracy"
        ].to_numpy(float)
        brier_values = draws["importance_multiclass_brier"].to_numpy(float)
        fpr_values = draws[
            "permuted_minus_baseline_soft_high_fpr"
        ].to_numpy(float)
        probability_low, probability_high = np.quantile(
            probability_values, [0.025, 0.975]
        )
        policy_low, policy_high = np.quantile(
            policy_values, [0.025, 0.975]
        )
        accuracy_low, accuracy_high = np.quantile(
            accuracy_values, [0.025, 0.975]
        )
        rows.append(
            {
                "display_order": int(order),
                "feature": spec.feature,
                "display_name": spec.display_name,
                "branch": spec.branch,
                "source_columns": "|".join(spec.source_columns),
                "high_gate_gower_block": spec.high_block or "not_used",
                "conditional_gower_block": spec.conditional_block or "not_used",
                "mean_high_gate_nominal_gower_weight": float(
                    weight["high_gate_gower_weight"].mean()
                ),
                "mean_conditional_nominal_gower_weight": float(
                    weight["conditional_gower_weight"].mean()
                ),
                "active_high_gate_outer_splits": int(
                    weight["high_gate_gower_weight"].gt(0).sum()
                ),
                "active_conditional_outer_splits": int(
                    weight["conditional_gower_weight"].gt(0).sum()
                ),
                "n_records": EXPECTED_RECORDS,
                "n_connected_groups": EXPECTED_GROUPS,
                "n_outer_splits": 50,
                "n_permutations": int(n_permutations),
                "n_bootstrap": int(n_bootstrap),
                "baseline_soft_log_loss": float(baseline["soft_log_loss"]),
                "permuted_soft_log_loss_mean": float(
                    observed["soft_log_loss"].mean()
                ),
                "importance_soft_log_loss": float(
                    observed["importance_soft_log_loss"].mean()
                ),
                "importance_soft_log_loss_ci95_low": float(probability_low),
                "importance_soft_log_loss_ci95_high": float(probability_high),
                "soft_log_loss_positive_permutation_fraction": float(
                    observed["importance_soft_log_loss"].gt(0).mean()
                ),
                "probability_reliance_status": _interval_status(
                    float(probability_low), float(probability_high)
                ),
                "baseline_multiclass_brier": float(
                    baseline["multiclass_brier"]
                ),
                "importance_multiclass_brier": float(
                    observed["importance_multiclass_brier"].mean()
                ),
                "importance_multiclass_brier_ci95_low": float(
                    np.quantile(brier_values, 0.025)
                ),
                "importance_multiclass_brier_ci95_high": float(
                    np.quantile(brier_values, 0.975)
                ),
                "baseline_soft_f0_5": float(baseline["soft_f0_5"]),
                "permuted_soft_f0_5_mean": float(
                    observed["soft_f0_5"].mean()
                ),
                "importance_soft_f0_5": float(
                    observed["importance_soft_f0_5"].mean()
                ),
                "importance_soft_f0_5_ci95_low": float(policy_low),
                "importance_soft_f0_5_ci95_high": float(policy_high),
                "soft_f0_5_positive_permutation_fraction": float(
                    observed["importance_soft_f0_5"].gt(0).mean()
                ),
                "policy_reliance_status": (
                    "structurally_not_used_by_high_gate"
                    if spec.high_column is None
                    else _interval_status(float(policy_low), float(policy_high))
                ),
                "baseline_posterior_expected_hard_accuracy": float(
                    baseline["posterior_expected_hard_accuracy"]
                ),
                "importance_posterior_expected_hard_accuracy": float(
                    observed[
                        "importance_posterior_expected_hard_accuracy"
                    ].mean()
                ),
                "importance_posterior_expected_hard_accuracy_ci95_low": float(
                    accuracy_low
                ),
                "importance_posterior_expected_hard_accuracy_ci95_high": float(
                    accuracy_high
                ),
                "baseline_soft_high_recall": float(
                    baseline["soft_high_recall"]
                ),
                "permuted_soft_high_recall_mean": float(
                    observed["soft_high_recall"].mean()
                ),
                "baseline_soft_high_precision": float(
                    baseline["soft_high_precision"]
                ),
                "permuted_soft_high_precision_mean": float(
                    observed["soft_high_precision"].mean()
                ),
                "baseline_soft_high_false_positive_rate": float(
                    baseline["soft_high_false_positive_rate"]
                ),
                "permuted_soft_high_false_positive_rate_mean": float(
                    observed["soft_high_false_positive_rate"].mean()
                ),
                "permuted_minus_baseline_soft_high_fpr_ci95_low": float(
                    np.quantile(fpr_values, 0.025)
                ),
                "permuted_minus_baseline_soft_high_fpr_ci95_high": float(
                    np.quantile(fpr_values, 0.975)
                ),
                "baseline_predicted_high_count": int(
                    baseline["predicted_high_count"]
                ),
                "permuted_predicted_high_count_mean": float(
                    observed["predicted_high_count"].mean()
                ),
                "mean_alert_flip_count": float(
                    observed["alert_flip_count"].mean()
                ),
                "interpretation_status": "posthoc_model_reliance_only",
                "feature_claim_allowed": False,
                "causal_claim_allowed": False,
                "mechanistic_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows).sort_values("display_order", kind="stable")
