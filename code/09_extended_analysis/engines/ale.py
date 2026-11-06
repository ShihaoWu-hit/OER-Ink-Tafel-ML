#!/usr/bin/env python3
"""Cross-fitted ALE with training-fold bins and connected-group intervals."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_feature_importance as reliance  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


N_BINS = 5
BIN_EDGES_PERCENTILE = np.linspace(0.0, 1.0, N_BINS + 1)
BIN_CENTERS_PERCENTILE = 100.0 * (
    BIN_EDGES_PERCENTILE[:-1] + BIN_EDGES_PERCENTILE[1:]
) / 2.0
MIN_SUPPORTED_REPEATS_PER_RECORD = 8
MIN_RECORD_COVERAGE = 0.80
MIN_GROUPS_PER_BIN = 8
SUPPORT_QUANTILE = 0.95
DEFAULT_BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20260726
SIMPLEX_TOLERANCE = 1.0e-10


FEATURES: tuple[dict[str, str], ...] = (
    {
        "feature": "solvent_polarity_coordinate",
        "display_name": "Solvent polarity coordinate",
        "branch": "high_gate",
        "column": "solvent_polarity_coordinate",
        "panel_title": "Solvent polarity coordinate (high branch)",
    },
    {
        "feature": "ionomer_to_solvent_ratio",
        "display_name": "Ionomer / solvent ratio",
        "branch": "conditional",
        "column": "log1p_ionomer_to_solvent_ratio",
        "panel_title": "Ionomer / solvent ratio (conditional branch)",
    },
)

CLASS_DISPLAY = {
    "low_60_side": "Low / 60 side",
    "120_side": "120 side",
    "high_off": "High / off",
}


def _training_support_state(
    engineered_train: pd.DataFrame,
    train_groups: Sequence[str],
    distance_model: Any,
) -> dict[str, Any]:
    """Calibrate q95 nearest support outside each connected group."""

    groups = np.asarray(list(map(str, train_groups)), dtype=object)
    if len(engineered_train) != len(groups) or len(engineered_train) < 2:
        raise ValueError("Support calibration rows and groups do not align")
    distances = np.asarray(
        distance_model._pairwise_distances(engineered_train),  # noqa: SLF001
        dtype=float,
    )
    if distances.shape != (len(engineered_train), len(engineered_train)):
        raise ValueError("Unexpected training support-distance shape")
    distances[groups[:, None] == groups[None, :]] = np.inf
    nearest = np.min(distances, axis=1)
    if not np.isfinite(nearest).all():
        raise ValueError(
            "Every outer-training record needs support outside its connected group"
        )
    threshold = float(np.quantile(nearest, SUPPORT_QUANTILE))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid fold-local Gower support threshold")
    return {
        "threshold": threshold,
        "training_nearest": nearest,
        "quantile": SUPPORT_QUANTILE,
    }


def _query_support(
    engineered_query: pd.DataFrame,
    distance_model: Any,
    state: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(
        distance_model._pairwise_distances(engineered_query),  # noqa: SLF001
        dtype=float,
    )
    nearest = np.min(distances, axis=1)
    supported = nearest <= float(state["threshold"]) + 1.0e-12
    return supported.astype(bool), nearest.astype(float)


def _quantile_bins(
    training_values: np.ndarray,
    observed_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training = np.asarray(training_values, dtype=float)
    observed = np.asarray(observed_values, dtype=float)
    if not np.isfinite(training).all() or not np.isfinite(observed).all():
        raise ValueError("ALE feature values must be finite after fold-local fitting")
    edges = np.quantile(training, BIN_EDGES_PERCENTILE)
    if len(np.unique(edges)) != N_BINS + 1 or np.any(np.diff(edges) <= 0):
        raise RuntimeError(
            "Five distinct fold-local ALE bins are not estimable; stopping "
            "rather than silently merging bins"
        )
    # Assign numerically tied boundary values consistently across platforms.
    observed = observed.copy()
    for edge in edges:
        observed[np.isclose(observed, edge, rtol=1.0e-12, atol=1.0e-12)] = edge
    bin_index = np.searchsorted(edges[1:-1], observed, side="right")
    bin_index = np.clip(bin_index, 0, N_BINS - 1).astype(int)
    observed_in_range = (observed >= edges[0]) & (observed <= edges[-1])
    return edges.astype(float), bin_index, observed_in_range.astype(bool)


def _high_probabilities(
    high: Mapping[str, Any],
    engineered: pd.DataFrame,
) -> np.ndarray:
    novelty = reliance._k_smallest_mean(
        high["distance_model"]._pairwise_distances(engineered),  # noqa: SLF001
        reliance.NOVELTY_NEIGHBORS,
    )
    frame = pd.DataFrame({"gower_novelty": novelty})
    probability = high["probability_estimator"].predict_proba(frame)
    reliance.probability_matrix_audit(probability)
    return probability


def _conditional_probabilities(
    conditional: Mapping[str, Any],
    engineered: pd.DataFrame,
) -> np.ndarray:
    distances = conditional[
        "estimator"
    ].distance_model_._pairwise_distances(engineered)  # noqa: SLF001
    probability = reliance._direct_probabilities_from_distances(
        conditional["estimator"], distances
    )
    reliance.probability_matrix_audit(probability)
    return probability


def _endpoint_frames(
    engineered_test: pd.DataFrame,
    column: str,
    edges: np.ndarray,
    bin_index: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    lower_values = edges[bin_index]
    upper_values = edges[bin_index + 1]
    lower = engineered_test.copy()
    upper = engineered_test.copy()
    lower[column] = lower_values
    upper[column] = upper_values
    return lower, upper, lower_values, upper_values


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
) -> dict[str, Any]:
    """Refit one exact outer split and calculate local endpoint differences."""

    ordered_ids = records_by_id.index.astype(str).tolist()
    test = list(map(str, test_ids))
    test_set = set(test)
    train = [record_id for record_id in ordered_ids if record_id not in test_set]
    train_groups = records_by_id.loc[train, "cv_group_id"].astype(str).tolist()
    test_groups = set(records_by_id.loc[test, "cv_group_id"].astype(str))
    if set(train_groups) & test_groups:
        raise AssertionError("A connected group crosses an outer split")

    probability_selection = reliance._selection_row(
        parent_selections, repeat=repeat, fold=fold
    )
    policy_selection = reliance._selection_row(
        policy_selections, repeat=repeat, fold=fold
    )
    probability_penalty = float(probability_selection["selected_penalty"])
    policy_penalty = float(policy_selection["selected_policy_penalty"])

    X_train_raw = records_by_id.loc[train].reset_index(drop=True)
    X_test_raw = records_by_id.loc[test].reset_index(drop=True)
    y_train = labels_by_id.loc[train, reliance.Q_COLUMNS].to_numpy(float)
    high = reliance._fit_high_branch(
        X_train_raw,
        train_groups,
        X_test_raw,
        y_train,
        probability_penalty=probability_penalty,
        policy_penalty=policy_penalty,
    )
    conditional = reliance._fit_conditional_branch(
        X_train_raw, X_test_raw, y_train
    )
    baseline_probability = reliance._hybrid_probabilities(
        high["baseline_probability"], conditional["baseline_probability"]
    )

    saved_probability_split = reliance._saved_split(
        saved_probability,
        repeat=repeat,
        fold=fold,
        test_ids=test,
    )
    probability_error = float(
        np.max(
            np.abs(
                baseline_probability
                - saved_probability_split.loc[
                    :, reliance.P_COLUMNS
                ].to_numpy(float)
            )
        )
    )
    saved_policy_split = reliance._saved_split(
        saved_policy,
        repeat=repeat,
        fold=fold,
        test_ids=test,
    )
    policy_score_error = float(
        np.max(
            np.abs(
                np.asarray(high["baseline_policy_score"], dtype=float)
                - saved_policy_split["high_score"].to_numpy(float)
            )
        )
    )
    if max(probability_error, policy_score_error) > reliance.BASELINE_TOLERANCE:
        raise RuntimeError(
            "Exact outer refit failed verified OOF reproduction: "
            f"probability={probability_error}, policy_score={policy_score_error}"
        )

    high_train = high["builder"].transform(X_train_raw).reset_index(drop=True)
    conditional_train = conditional["builder"].transform(
        X_train_raw
    ).reset_index(drop=True)
    high_test = high["engineered_test"].reset_index(drop=True)
    conditional_test = conditional["engineered_test"].reset_index(drop=True)
    high_support_state = _training_support_state(
        high_train, train_groups, high["distance_model"]
    )
    conditional_support_state = _training_support_state(
        conditional_train,
        train_groups,
        conditional["estimator"].distance_model_,
    )
    high_observed_supported, high_observed_distance = _query_support(
        high_test, high["distance_model"], high_support_state
    )
    conditional_observed_supported, conditional_observed_distance = (
        _query_support(
            conditional_test,
            conditional["estimator"].distance_model_,
            conditional_support_state,
        )
    )

    local_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    identities = records_by_id.loc[test, ["cv_group_id"]].reset_index()
    identities["record_id"] = identities["record_id"].astype(str)

    for spec in FEATURES:
        feature = spec["feature"]
        branch = spec["branch"]
        column = spec["column"]
        if branch == "high_gate":
            train_frame = high_train
            observed_frame = high_test
        elif branch == "conditional":
            train_frame = conditional_train
            observed_frame = conditional_test
        else:
            raise ValueError(f"Unknown ALE branch {branch!r}")
        if column not in train_frame or column not in observed_frame:
            raise RuntimeError(
                f"ALE feature {column!r} was removed in outer split "
                f"repeat={repeat}, fold={fold}"
            )

        edges, bin_index, observed_in_range = _quantile_bins(
            train_frame[column].to_numpy(float),
            observed_frame[column].to_numpy(float),
        )
        lower, upper, lower_values, upper_values = _endpoint_frames(
            observed_frame, column, edges, bin_index
        )

        if branch == "high_gate":
            lower_branch_probability = _high_probabilities(high, lower)
            upper_branch_probability = _high_probabilities(high, upper)
            lower_probability = reliance._hybrid_probabilities(
                lower_branch_probability,
                conditional["baseline_probability"],
            )
            upper_probability = reliance._hybrid_probabilities(
                upper_branch_probability,
                conditional["baseline_probability"],
            )
            lower_supported, lower_distance = _query_support(
                lower, high["distance_model"], high_support_state
            )
            upper_supported, upper_distance = _query_support(
                upper, high["distance_model"], high_support_state
            )
            companion_supported = conditional_observed_supported
            companion_distance = conditional_observed_distance
            main_threshold = float(high_support_state["threshold"])
            companion_threshold = float(
                conditional_support_state["threshold"]
            )
        else:
            lower_branch_probability = _conditional_probabilities(
                conditional, lower
            )
            upper_branch_probability = _conditional_probabilities(
                conditional, upper
            )
            lower_probability = reliance._hybrid_probabilities(
                high["baseline_probability"], lower_branch_probability
            )
            upper_probability = reliance._hybrid_probabilities(
                high["baseline_probability"], upper_branch_probability
            )
            lower_supported, lower_distance = _query_support(
                lower,
                conditional["estimator"].distance_model_,
                conditional_support_state,
            )
            upper_supported, upper_distance = _query_support(
                upper,
                conditional["estimator"].distance_model_,
                conditional_support_state,
            )
            companion_supported = high_observed_supported
            companion_distance = high_observed_distance
            main_threshold = float(conditional_support_state["threshold"])
            companion_threshold = float(high_support_state["threshold"])

        delta = upper_probability - lower_probability
        local_simplex_error = np.abs(delta.sum(axis=1))
        if float(np.max(local_simplex_error)) > SIMPLEX_TOLERANCE:
            raise RuntimeError("Local ALE probability differences violate simplex")
        joint_supported = (
            observed_in_range
            & lower_supported
            & upper_supported
            & companion_supported
        )

        for position, identity in identities.iterrows():
            raw_lower = (
                float(np.expm1(lower_values[position]))
                if feature == "ionomer_to_solvent_ratio"
                else np.nan
            )
            raw_upper = (
                float(np.expm1(upper_values[position]))
                if feature == "ionomer_to_solvent_ratio"
                else np.nan
            )
            support_rows.append(
                {
                    "feature": feature,
                    "branch": branch,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "record_id": str(identity["record_id"]),
                    "cv_group_id": str(identity["cv_group_id"]),
                    "bin_index": int(bin_index[position]),
                    "bin_percentile_lower": float(
                        100.0 * BIN_EDGES_PERCENTILE[bin_index[position]]
                    ),
                    "bin_percentile_center": float(
                        BIN_CENTERS_PERCENTILE[bin_index[position]]
                    ),
                    "bin_percentile_upper": float(
                        100.0 * BIN_EDGES_PERCENTILE[bin_index[position] + 1]
                    ),
                    "engineered_lower_fold": float(lower_values[position]),
                    "engineered_upper_fold": float(upper_values[position]),
                    "raw_lower_fold_diagnostic": raw_lower,
                    "raw_upper_fold_diagnostic": raw_upper,
                    "observed_in_training_range": bool(
                        observed_in_range[position]
                    ),
                    "lower_endpoint_supported": bool(
                        lower_supported[position]
                    ),
                    "upper_endpoint_supported": bool(
                        upper_supported[position]
                    ),
                    "companion_branch_supported": bool(
                        companion_supported[position]
                    ),
                    "joint_supported": bool(joint_supported[position]),
                    "lower_endpoint_nearest_distance": float(
                        lower_distance[position]
                    ),
                    "upper_endpoint_nearest_distance": float(
                        upper_distance[position]
                    ),
                    "companion_nearest_distance": float(
                        companion_distance[position]
                    ),
                    "main_branch_support_threshold": main_threshold,
                    "companion_branch_support_threshold": companion_threshold,
                    "support_threshold_quantile": SUPPORT_QUANTILE,
                }
            )
            for class_index, class_label in enumerate(reliance.CLASS_LABELS):
                local_rows.append(
                    {
                        "feature": feature,
                        "branch": branch,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "record_id": str(identity["record_id"]),
                        "cv_group_id": str(identity["cv_group_id"]),
                        "bin_index": int(bin_index[position]),
                        "class_label": str(class_label),
                        "local_probability_difference": float(
                            delta[position, class_index]
                        ),
                        "joint_supported": bool(joint_supported[position]),
                    }
                )

    return {
        "local": pd.DataFrame(local_rows),
        "support": pd.DataFrame(support_rows),
        "probability_max_abs_error": probability_error,
        "policy_score_max_abs_error": policy_score_error,
    }


def _record_aggregate(
    local: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """Aggregate repeated local effects by record before inference."""

    included: dict[str, set[str]] = {}
    record_support_rows: list[pd.DataFrame] = []
    for spec in FEATURES:
        feature = spec["feature"]
        feature_support = support.loc[support["feature"].eq(feature)].copy()
        counts = (
            feature_support.groupby(
                ["record_id", "cv_group_id"], as_index=False
            )
            .agg(
                n_available_repeats=("repeat", "nunique"),
                n_supported_repeats=("joint_supported", "sum"),
            )
        )
        if not counts["n_available_repeats"].eq(10).all():
            raise ValueError("ALE support does not contain all 10 repeats")
        counts["record_included"] = counts["n_supported_repeats"].ge(
            MIN_SUPPORTED_REPEATS_PER_RECORD
        )
        counts["feature"] = feature
        included[feature] = set(
            counts.loc[counts["record_included"], "record_id"].astype(str)
        )
        record_support_rows.append(counts)
    record_support = pd.concat(record_support_rows, ignore_index=True)

    selected = local.loc[
        local["joint_supported"].astype(bool)
        & local.apply(
            lambda row: str(row["record_id"])
            in included[str(row["feature"])],
            axis=1,
        )
    ].copy()
    if selected.empty:
        raise RuntimeError("No jointly supported ALE local effects remain")
    record_bin = (
        selected.groupby(
            [
                "feature",
                "branch",
                "record_id",
                "cv_group_id",
                "bin_index",
                "class_label",
            ],
            as_index=False,
        )
        .agg(
            local_probability_difference=(
                "local_probability_difference",
                "mean",
            ),
            n_supported_repeats_in_bin=("repeat", "nunique"),
        )
    )
    totals = (
        record_bin.drop_duplicates(
            ["feature", "record_id", "bin_index"]
        )
        .groupby(["feature", "record_id"], as_index=False)
        ["n_supported_repeats_in_bin"]
        .sum()
        .rename(
            columns={
                "n_supported_repeats_in_bin": "n_supported_repeats_total"
            }
        )
    )
    record_bin = record_bin.merge(
        totals,
        on=["feature", "record_id"],
        how="left",
        validate="many_to_one",
    )
    record_bin["record_bin_weight"] = (
        record_bin["n_supported_repeats_in_bin"]
        / record_bin["n_supported_repeats_total"]
    )
    weight_check = (
        record_bin.drop_duplicates(
            ["feature", "record_id", "bin_index"]
        )
        .groupby(["feature", "record_id"])["record_bin_weight"]
        .sum()
    )
    if not np.allclose(weight_check.to_numpy(float), 1.0, atol=1.0e-12):
        raise RuntimeError("Record-level ALE bin weights do not sum to one")
    return record_bin, record_support, included


def _curve_from_record_bins(
    record_bin: pd.DataFrame,
    *,
    feature: str,
    group_multiplicity: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    frame = record_bin.loc[record_bin["feature"].eq(feature)].copy()
    if frame.empty:
        raise RuntimeError(f"No record-level ALE rows for {feature}")
    if group_multiplicity is None:
        frame["bootstrap_multiplier"] = 1.0
    else:
        frame["bootstrap_multiplier"] = frame["cv_group_id"].astype(str).map(
            lambda value: float(group_multiplicity.get(str(value), 0))
        )
        frame = frame.loc[frame["bootstrap_multiplier"].gt(0)].copy()
    frame["analysis_weight"] = (
        frame["record_bin_weight"] * frame["bootstrap_multiplier"]
    )

    bin_base = frame.drop_duplicates(["record_id", "bin_index"])
    bin_weights = (
        bin_base.groupby("bin_index")["analysis_weight"].sum().reindex(
            range(N_BINS), fill_value=0.0
        )
    )
    if np.any(bin_weights.to_numpy(float) <= 0):
        raise RuntimeError(
            f"A connected-group bootstrap replicate emptied an ALE bin for {feature}"
        )
    normalized_bin_weight = bin_weights / float(bin_weights.sum())

    local_delta = np.zeros((N_BINS, len(reliance.CLASS_LABELS)), dtype=float)
    for bin_index in range(N_BINS):
        for class_index, class_label in enumerate(reliance.CLASS_LABELS):
            cell = frame.loc[
                frame["bin_index"].eq(bin_index)
                & frame["class_label"].eq(class_label)
            ]
            if cell.empty or float(cell["analysis_weight"].sum()) <= 0:
                raise RuntimeError(
                    f"Missing ALE class/bin cell: {feature}, {class_label}, "
                    f"bin={bin_index}"
                )
            local_delta[bin_index, class_index] = float(
                np.average(
                    cell["local_probability_difference"].to_numpy(float),
                    weights=cell["analysis_weight"].to_numpy(float),
                )
            )
    if float(np.max(np.abs(local_delta.sum(axis=1)))) > SIMPLEX_TOLERANCE:
        raise RuntimeError("Aggregated ALE local differences violate simplex")

    midpoint = np.cumsum(local_delta, axis=0) - 0.5 * local_delta
    center = np.sum(
        normalized_bin_weight.to_numpy(float)[:, None] * midpoint,
        axis=0,
    )
    ale = midpoint - center[None, :]
    if float(np.max(np.abs(ale.sum(axis=1)))) > SIMPLEX_TOLERANCE:
        raise RuntimeError("Centered ALE curves violate probability simplex")

    rows: list[dict[str, Any]] = []
    for bin_index in range(N_BINS):
        base_cell = bin_base.loc[bin_base["bin_index"].eq(bin_index)]
        for class_index, class_label in enumerate(reliance.CLASS_LABELS):
            rows.append(
                {
                    "feature": feature,
                    "class_label": str(class_label),
                    "bin_index": int(bin_index),
                    "bin_percentile_lower": float(
                        100.0 * BIN_EDGES_PERCENTILE[bin_index]
                    ),
                    "bin_percentile_center": float(
                        BIN_CENTERS_PERCENTILE[bin_index]
                    ),
                    "bin_percentile_upper": float(
                        100.0 * BIN_EDGES_PERCENTILE[bin_index + 1]
                    ),
                    "local_mean_probability_difference": float(
                        local_delta[bin_index, class_index]
                    ),
                    "ale_probability": float(ale[bin_index, class_index]),
                    "empirical_bin_weight": float(
                        normalized_bin_weight.loc[bin_index]
                    ),
                    "effective_record_weight": float(
                        bin_weights.loc[bin_index]
                    ),
                    "n_records": int(base_cell["record_id"].nunique()),
                    "n_connected_groups": int(
                        base_cell.loc[
                            base_cell["analysis_weight"].gt(0), "cv_group_id"
                        ]
                        .astype(str)
                        .nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_curves(
    record_bin: pd.DataFrame,
    all_groups: Sequence[str],
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, int]:
    unique_groups = np.asarray(sorted(set(map(str, all_groups))), dtype=object)
    if len(unique_groups) != reliance.EXPECTED_GROUPS:
        raise ValueError("ALE bootstrap requires all 80 connected groups")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[pd.DataFrame] = []
    invalid = 0
    for replicate in range(int(repetitions)):
        sampled = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        multiplicity = Counter(map(str, sampled))
        replicate_rows: list[pd.DataFrame] = []
        try:
            for spec in FEATURES:
                curve = _curve_from_record_bins(
                    record_bin,
                    feature=spec["feature"],
                    group_multiplicity=multiplicity,
                )
                replicate_rows.append(curve)
        except RuntimeError:
            invalid += 1
            continue
        replicate_frame = pd.concat(replicate_rows, ignore_index=True)
        replicate_frame["bootstrap_replicate"] = int(replicate)
        rows.append(replicate_frame)
    if not rows:
        raise RuntimeError("All connected-group ALE bootstrap replicates failed")
    bootstrap = pd.concat(rows, ignore_index=True)
    valid_repetitions = int(bootstrap["bootstrap_replicate"].nunique())
    if valid_repetitions < int(np.ceil(0.99 * repetitions)):
        raise RuntimeError(
            "More than 1% of connected-group ALE bootstrap replicates lacked "
            f"full bin support: valid={valid_repetitions}, requested={repetitions}"
        )
    return bootstrap, invalid


def _support_summary(
    raw_support: pd.DataFrame,
    record_support: pd.DataFrame,
    record_bin: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in FEATURES:
        feature = spec["feature"]
        raw = raw_support.loc[raw_support["feature"].eq(feature)].copy()
        records = record_support.loc[
            record_support["feature"].eq(feature)
        ].copy()
        bins = record_bin.loc[record_bin["feature"].eq(feature)].copy()
        included_records = records.loc[records["record_included"]].copy()
        coverage = float(len(included_records) / reliance.EXPECTED_RECORDS)
        for bin_index in range(N_BINS):
            raw_bin = raw.loc[raw["bin_index"].eq(bin_index)]
            bin_rows = bins.loc[bins["bin_index"].eq(bin_index)].drop_duplicates(
                ["record_id", "bin_index"]
            )
            effective_weight = float(bin_rows["record_bin_weight"].sum())
            n_groups = int(bin_rows["cv_group_id"].astype(str).nunique())
            n_records = int(bin_rows["record_id"].astype(str).nunique())
            rows.append(
                {
                    "feature": feature,
                    "branch": spec["branch"],
                    "bin_index": int(bin_index),
                    "bin_percentile_lower": float(
                        100.0 * BIN_EDGES_PERCENTILE[bin_index]
                    ),
                    "bin_percentile_center": float(
                        BIN_CENTERS_PERCENTILE[bin_index]
                    ),
                    "bin_percentile_upper": float(
                        100.0 * BIN_EDGES_PERCENTILE[bin_index + 1]
                    ),
                    "n_available_outer_test_rows": int(len(raw_bin)),
                    "n_joint_supported_outer_test_rows": int(
                        raw_bin["joint_supported"].sum()
                    ),
                    "joint_support_fraction": float(
                        raw_bin["joint_supported"].mean()
                    ),
                    "n_included_records": n_records,
                    "effective_record_weight": effective_weight,
                    "n_included_connected_groups": n_groups,
                    "n_feature_included_records": int(len(included_records)),
                    "feature_record_coverage": coverage,
                    "minimum_supported_repeats_among_included_records": int(
                        included_records["n_supported_repeats"].min()
                    ),
                    "main_support_threshold_min": float(
                        raw_bin["main_branch_support_threshold"].min()
                    ),
                    "main_support_threshold_median": float(
                        raw_bin["main_branch_support_threshold"].median()
                    ),
                    "main_support_threshold_max": float(
                        raw_bin["main_branch_support_threshold"].max()
                    ),
                    "companion_support_threshold_median": float(
                        raw_bin["companion_branch_support_threshold"].median()
                    ),
                    "support_gate_pass": bool(
                        coverage >= MIN_RECORD_COVERAGE
                        and effective_weight >= MIN_GROUPS_PER_BIN
                        and n_groups >= MIN_GROUPS_PER_BIN
                    ),
                    "support_rule": (
                        "both local endpoints and unchanged companion branch "
                        "within fold-local connected-group-excluded training "
                        "Gower nearest-distance q95; record retained with >=8/10 "
                        "supported repeats"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_endpoints(
    raw_support: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = (
        raw_support.groupby(["feature", "bin_index"], as_index=False)
        .agg(
            engineered_lower_fold_median=("engineered_lower_fold", "median"),
            engineered_upper_fold_median=("engineered_upper_fold", "median"),
            raw_lower_fold_median_diagnostic=(
                "raw_lower_fold_diagnostic",
                "median",
            ),
            raw_upper_fold_median_diagnostic=(
                "raw_upper_fold_diagnostic",
                "median",
            ),
        )
    )
    return diagnostics
