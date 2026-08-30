"""Gate-controlled adjusted probability curves for the all-97 classifier.

This module implements the curve policy frozen in
``posterior_cardinal_all97_v2``.  It is deliberately separate from model
selection and release testing:

* only a formal continuous ``context + one feature`` panel that has already
  passed every release gate is eligible;
* every prediction comes from the primary two-stage Gaussian-prior MAP model;
* grids and the reference are outer-training-fold percentiles;
* outer-test records retain their empirical context covariates;
* support is calibrated solely within the outer-training fold with
  block-balanced Gower distance; and
* one fixed common record set is used for every grid percentile.

The Gaussian-prior estimator is a deterministic MAP approximation, not a full
Bayesian coefficient posterior.  Curves are adjusted predictive associations,
not causal or mechanistic effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cardinal_metrics import CLASS_LABELS, SOFT_LABEL_COLUMNS
from .cardinal_reduced_models import ReducedPanelTwoStageMAPClassifier
from .feature_association_features import (
    FROZEN_ASSOCIATION_PANELS,
    required_association_source_columns,
)
from .posterior_cardinal_contract import EXPECTED_FORMAL_PANELS


PRIMARY_WIDTH = 0.20
PRIMARY_MODEL_FAMILY = "two_stage_gaussian_prior_MAP"
GRID_PERCENTILES = tuple(round(0.05 * index, 2) for index in range(1, 20))
REFERENCE_PERCENTILE = 0.50
EXPECTED_RECORDS = 97
EXPECTED_REPEATS = 10
EXPECTED_FOLDS = 5
MINIMUM_SUPPORTED_REPEATS = 8
MINIMUM_COMMON_RECORD_COVERAGE = 0.80
SUPPORT_THRESHOLD_QUANTILE = 0.95
SIMPLEX_ROUNDOFF_TOLERANCE = 1.0e-10


FOLD_PREDICTION_COLUMNS = (
    "panel",
    "feature",
    "repeat",
    "fold",
    "record_id",
    "cv_group_id",
    "grid_percentile",
    "grid_value_fold",
    "reference_percentile",
    "reference_value_fold",
    "class_label",
    "probability",
    "reference_probability",
    "probability_difference_from_reference",
    "grid_supported",
    "reference_supported",
    "joint_support",
    "grid_nearest_distance",
    "reference_nearest_distance",
    "joint_support_threshold",
    "joint_support_threshold_quantile",
    "n_outer_training_records",
    "cross_fitted",
    "empirical_standardization",
)

CURVE_COLUMNS = (
    "panel",
    "feature",
    "grid_percentile",
    "reference_percentile",
    "class_label",
    "probability",
    "reference_probability",
    "probability_difference_from_reference",
    "n_common_records",
    "n_common_groups",
    "common_record_coverage",
    "minimum_supported_repeats_per_record",
    "grid_value_fold_mean_diagnostic",
    "x_axis_semantics",
    "joint_support_rule",
    "cross_fitted_empirical_standardization",
    "interpretation",
)

AUDIT_COLUMNS = (
    "panel",
    "feature",
    "status",
    "n_population_records",
    "n_common_records",
    "common_record_coverage",
    "minimum_supported_repeats_per_record",
    "grid_point_count",
    "primary_model_only",
    "release_gate_required",
    "continuous_feature_only",
    "fixed_common_record_set",
    "reason",
)


class CurveNotEstimableError(ValueError):
    """The frozen curve exists conceptually but cannot be estimated."""


@dataclass(frozen=True)
class AdjustedProbabilityCurveResult:
    """Candidate predictions, released curve rows, and support audit."""

    fold_predictions: pd.DataFrame
    curves: pd.DataFrame
    audit: pd.DataFrame


def empty_fold_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=FOLD_PREDICTION_COLUMNS)


def empty_curve_table() -> pd.DataFrame:
    return pd.DataFrame(columns=CURVE_COLUMNS)


def empty_curve_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=AUDIT_COLUMNS)


def validate_adjusted_curve_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed if any curve-defining contract field has changed."""

    policy = contract.get("adjusted_probability_curve")
    if not isinstance(policy, Mapping):
        raise TypeError("adjusted_probability_curve must be a mapping")
    exact = {
        "grid_values": "fold_local_outer_training_percentiles",
        "feature_type": "continuous_only",
        "joint_support_distance": "fold_local_block_balanced_Gower",
        "joint_support_threshold": (
            "outer_training_leave_one_out_nearest_distance_q95"
        ),
        "support_requires_grid_and_reference": True,
        "minimum_common_record_coverage_fraction": 0.80,
        "minimum_supported_repeats_per_common_record_at_every_grid_point": 8,
        "covariate_averaging_estimand": (
            "equal_weight_empirical_standardization_over_common_all97_"
            "outer_test_records_with_observed_context_covariates"
        ),
        "fold_prediction": (
            "predict_each_outer_test_record_at_fold_local_grid_and_reference"
        ),
        "repeat_aggregation": (
            "average_each_records_supported_cross_fitted_probabilities_"
            "across_repeats"
        ),
        "population_aggregation": (
            "equal_weight_mean_over_one_fixed_common_record_set_shared_by_"
            "all_grid_points"
        ),
        "output": (
            "all_four_absolute_probabilities_and_reference_differences"
        ),
        "simplex_policy": (
            "clip_only_machine_roundoff_then_renormalize_each_record_before_"
            "all_averages_and_final_row"
        ),
        "extrapolation_allowed": False,
        "categorical_contrasts": (
            "withheld_without_separately_frozen_common_reference"
        ),
    }
    changed = {
        name: (expected, policy.get(name))
        for name, expected in exact.items()
        if policy.get(name) != expected
    }
    if changed:
        raise ValueError(
            "Frozen adjusted-probability-curve policy changed: "
            f"{changed}"
        )
    observed_grid = tuple(
        float(value) for value in policy.get("grid_percentiles", ())
    )
    if observed_grid != GRID_PERCENTILES:
        raise ValueError(
            "Frozen adjusted-curve percentile grid changed: "
            f"{observed_grid}"
        )
    if float(policy.get("reference_percentile", np.nan)) != (
        REFERENCE_PERCENTILE
    ):
        raise ValueError("Frozen adjusted-curve reference percentile changed")


def _normalize_roundoff_simplex(values: Any) -> np.ndarray:
    """Clip machine roundoff only, then normalize each probability row."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(CLASS_LABELS):
        raise ValueError(
            "Curve predictions must be an n-by-four probability matrix"
        )
    if not np.isfinite(array).all():
        raise ValueError("Curve predictions contain non-finite probabilities")
    if (
        np.any(array < -SIMPLEX_ROUNDOFF_TOLERANCE)
        or np.any(array > 1.0 + SIMPLEX_ROUNDOFF_TOLERANCE)
    ):
        raise ValueError(
            "Curve predictions leave the simplex beyond machine roundoff"
        )
    clipped = np.clip(array, 0.0, 1.0)
    totals = clipped.sum(axis=1, keepdims=True)
    if np.any(totals <= SIMPLEX_ROUNDOFF_TOLERANCE):
        raise ValueError("Curve predictions have zero probability mass")
    if np.any(np.abs(totals - 1.0) > SIMPLEX_ROUNDOFF_TOLERANCE):
        raise ValueError(
            "Curve probability mass differs from one beyond machine roundoff"
        )
    return clipped / totals


def _block_balanced_gower_distances(
    query: pd.DataFrame,
    train: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """Training-range Gower distance, averaged within and then across blocks."""

    query_aligned = query.loc[:, list(train.columns)]
    numerator = np.zeros((len(query_aligned), len(train)), dtype=float)
    denominator = np.zeros_like(numerator)
    for members in blocks.values():
        usable = [name for name in members if name in train.columns]
        block_numerator = np.zeros_like(numerator)
        block_denominator = np.zeros_like(numerator)
        for feature in usable:
            if pd.api.types.is_numeric_dtype(train[feature]):
                training = pd.to_numeric(
                    train[feature], errors="coerce"
                ).to_numpy(float)
                finite = training[np.isfinite(training)]
                scale = float(np.ptp(finite)) if finite.size else np.nan
                if not np.isfinite(scale) or scale <= 0:
                    continue
                candidate = pd.to_numeric(
                    query_aligned[feature], errors="coerce"
                ).to_numpy(float)
                valid = np.isfinite(candidate[:, None]) & np.isfinite(
                    training[None, :]
                )
                contribution = np.minimum(
                    np.abs(candidate[:, None] - training[None, :]) / scale,
                    1.0,
                )
            else:
                candidate = (
                    query_aligned[feature].astype("object").to_numpy()
                )
                training = train[feature].astype("object").to_numpy()
                valid = (~pd.isna(candidate[:, None])) & (
                    ~pd.isna(training[None, :])
                )
                contribution = (
                    candidate[:, None] != training[None, :]
                ).astype(float)
            block_numerator += np.where(valid, contribution, 0.0)
            block_denominator += valid.astype(float)
        valid_block = block_denominator > 0
        block_distance = np.divide(
            block_numerator,
            block_denominator,
            out=np.zeros_like(block_numerator),
            where=valid_block,
        )
        numerator += np.where(valid_block, block_distance, 0.0)
        denominator += valid_block.astype(float)
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0,
    )


def _joint_support_state(
    train_engineered: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if len(train_engineered) < 2:
        raise CurveNotEstimableError(
            "Joint-support calibration needs at least two training records"
        )
    distances = _block_balanced_gower_distances(
        train_engineered, train_engineered, blocks
    )
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if not finite.size:
        raise CurveNotEstimableError(
            "Joint-support calibration has no finite leave-one-out distances"
        )
    return {
        "train": train_engineered,
        "blocks": dict(blocks),
        "threshold": float(
            np.quantile(finite, SUPPORT_THRESHOLD_QUANTILE)
        ),
        "threshold_quantile": SUPPORT_THRESHOLD_QUANTILE,
    }


def _joint_support_query(
    query: pd.DataFrame,
    state: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    distances = _block_balanced_gower_distances(
        query,
        state["train"],
        state["blocks"],
    )
    nearest = np.min(distances, axis=1)
    supported = nearest <= float(state["threshold"]) + 1.0e-12
    return supported.astype(bool), nearest.astype(float)


def _continuous_panel_feature(
    panel_name: str,
    estimator: ReducedPanelTwoStageMAPClassifier,
) -> str:
    if panel_name not in EXPECTED_FORMAL_PANELS:
        raise ValueError(
            "Adjusted curves are restricted to the six formal increment "
            f"panels, received {panel_name!r}"
        )
    panel = FROZEN_ASSOCIATION_PANELS[panel_name]
    if len(panel.added_features) != 1:
        raise AssertionError("A curve panel must add exactly one feature")
    feature = str(panel.added_features[0])
    builder = estimator.feature_builder_
    if feature not in set(map(str, builder.get_feature_names_out())):
        raise CurveNotEstimableError(
            f"Feature {feature!r} was constant and removed in this outer fold"
        )
    if feature not in set(map(str, builder.numeric_columns_out_)):
        raise ValueError(
            f"Categorical feature {feature!r} has no frozen curve contrast"
        )
    return feature


def generate_fold_adjusted_probability_predictions(
    estimator: ReducedPanelTwoStageMAPClassifier,
    train_records: pd.DataFrame,
    test_records: pd.DataFrame,
    *,
    panel_name: str,
    repeat: int,
    fold: int,
) -> pd.DataFrame:
    """Predict one outer-test fold at every fold-local grid and reference.

    ``estimator`` must already be fitted to ``train_records`` only.  The
    function calls its fitted feature builder and two-stage MAP model directly
    so the intervention occurs on the fold-local engineered coordinate while
    all other engineered context values remain those of each test record.
    """

    if not isinstance(estimator, ReducedPanelTwoStageMAPClassifier):
        raise TypeError(
            "Adjusted curves require ReducedPanelTwoStageMAPClassifier"
        )
    if not hasattr(estimator, "feature_builder_"):
        raise RuntimeError("The primary MAP estimator must be fitted first")
    fitted_panel = str(estimator.feature_builder_.panel_.name)
    if fitted_panel != str(panel_name):
        raise ValueError(
            f"Fitted panel {fitted_panel!r} != requested panel {panel_name!r}"
        )
    for name, frame in (
        ("train_records", train_records),
        ("test_records", test_records),
    ):
        missing = sorted({"record_id", "cv_group_id"} - set(frame.columns))
        if missing:
            raise KeyError(f"{name} is missing identity columns: {missing}")
    if set(train_records["record_id"].astype(str)).intersection(
        set(test_records["record_id"].astype(str))
    ):
        raise ValueError("Outer training and test records must be disjoint")

    builder = estimator.feature_builder_
    feature = _continuous_panel_feature(panel_name, estimator)
    train_engineered = builder.transform(train_records).reset_index(drop=True)
    test_engineered = builder.transform(test_records).reset_index(drop=True)
    training_values = pd.to_numeric(
        train_engineered[feature], errors="coerce"
    )
    if not np.isfinite(training_values.to_numpy(float)).all():
        raise CurveNotEstimableError(
            "Fold-local engineered curve feature is non-finite"
        )
    grid_values = {
        percentile: float(training_values.quantile(percentile))
        for percentile in GRID_PERCENTILES
    }
    reference_value = float(
        training_values.quantile(REFERENCE_PERCENTILE)
    )
    if not all(np.isfinite(value) for value in grid_values.values()):
        raise CurveNotEstimableError("Fold-local percentile grid is non-finite")

    support_state = _joint_support_state(
        train_engineered,
        builder.get_blocks_out(),
    )
    reference_frame = test_engineered.copy()
    reference_frame[feature] = reference_value
    reference_probability = _normalize_roundoff_simplex(
        estimator.model_.predict_proba(reference_frame)
    )
    reference_supported, reference_distance = _joint_support_query(
        reference_frame, support_state
    )

    metadata = test_records.loc[
        :, ["record_id", "cv_group_id"]
    ].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for percentile in GRID_PERCENTILES:
        grid_value = grid_values[percentile]
        intervention = test_engineered.copy()
        intervention[feature] = grid_value
        probability = _normalize_roundoff_simplex(
            estimator.model_.predict_proba(intervention)
        )
        grid_supported, grid_distance = _joint_support_query(
            intervention, support_state
        )
        joint_supported = grid_supported & reference_supported
        difference = probability - reference_probability
        for position, identity in metadata.iterrows():
            for class_index, class_label in enumerate(CLASS_LABELS):
                rows.append(
                    {
                        "panel": str(panel_name),
                        "feature": feature,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "record_id": str(identity["record_id"]),
                        "cv_group_id": str(identity["cv_group_id"]),
                        "grid_percentile": float(percentile),
                        "grid_value_fold": grid_value,
                        "reference_percentile": REFERENCE_PERCENTILE,
                        "reference_value_fold": reference_value,
                        "class_label": str(class_label),
                        "probability": float(
                            probability[position, class_index]
                        ),
                        "reference_probability": float(
                            reference_probability[position, class_index]
                        ),
                        "probability_difference_from_reference": float(
                            difference[position, class_index]
                        ),
                        "grid_supported": bool(grid_supported[position]),
                        "reference_supported": bool(
                            reference_supported[position]
                        ),
                        "joint_support": bool(joint_supported[position]),
                        "grid_nearest_distance": float(
                            grid_distance[position]
                        ),
                        "reference_nearest_distance": float(
                            reference_distance[position]
                        ),
                        "joint_support_threshold": float(
                            support_state["threshold"]
                        ),
                        "joint_support_threshold_quantile": (
                            SUPPORT_THRESHOLD_QUANTILE
                        ),
                        "n_outer_training_records": int(
                            len(train_engineered)
                        ),
                        "cross_fitted": True,
                        "empirical_standardization": True,
                    }
                )
    output = pd.DataFrame(rows, columns=FOLD_PREDICTION_COLUMNS)
    expected_rows = (
        len(test_records) * len(GRID_PERCENTILES) * len(CLASS_LABELS)
    )
    if len(output) != expected_rows:
        raise AssertionError("Fold curve predictions lost required rows")
    return output


def _curve_audit_row(
    *,
    panel: str,
    feature: str,
    status: str,
    n_common_records: int,
    common_record_coverage: float,
    minimum_supported_repeats: int,
    reason: str,
    expected_records: int,
) -> dict[str, Any]:
    return {
        "panel": panel,
        "feature": feature,
        "status": status,
        "n_population_records": int(expected_records),
        "n_common_records": int(n_common_records),
        "common_record_coverage": float(common_record_coverage),
        "minimum_supported_repeats_per_record": int(
            minimum_supported_repeats
        ),
        "grid_point_count": len(GRID_PERCENTILES),
        "primary_model_only": True,
        "release_gate_required": True,
        "continuous_feature_only": True,
        "fixed_common_record_set": True,
        "reason": reason,
    }


def aggregate_adjusted_probability_predictions(
    fold_predictions: pd.DataFrame,
    *,
    expected_records: int = EXPECTED_RECORDS,
    expected_repeats: int = EXPECTED_REPEATS,
    minimum_supported_repeats: int = MINIMUM_SUPPORTED_REPEATS,
    minimum_common_record_coverage: float = (
        MINIMUM_COMMON_RECORD_COVERAGE
    ),
) -> AdjustedProbabilityCurveResult:
    """Apply the frozen common-record support and averaging estimand."""

    if fold_predictions.empty:
        return AdjustedProbabilityCurveResult(
            empty_fold_predictions(), empty_curve_table(), empty_curve_audit()
        )
    missing = sorted(set(FOLD_PREDICTION_COLUMNS) - set(fold_predictions))
    if missing:
        raise KeyError(f"Fold curve predictions are missing columns: {missing}")
    if not 0 < minimum_common_record_coverage <= 1:
        raise ValueError("Common-record coverage must lie in (0, 1]")
    if not 1 <= int(minimum_supported_repeats) <= int(expected_repeats):
        raise ValueError("Supported-repeat requirement is invalid")

    all_curves: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for (panel, feature), frame in fold_predictions.groupby(
        ["panel", "feature"], sort=True
    ):
        frame = frame.copy()
        grids = tuple(
            sorted(frame["grid_percentile"].astype(float).unique())
        )
        if grids != GRID_PERCENTILES:
            raise ValueError(
                f"Panel {panel!r} does not contain the frozen grid"
            )
        if set(frame["class_label"].astype(str)) != set(CLASS_LABELS):
            raise ValueError(
                f"Panel {panel!r} does not contain all four classes"
            )
        if not frame["reference_percentile"].astype(float).eq(
            REFERENCE_PERCENTILE
        ).all():
            raise ValueError("Curve reference percentile changed")
        if not frame["cross_fitted"].astype(bool).all():
            raise ValueError("Non-cross-fitted curve predictions are forbidden")
        if not frame["empirical_standardization"].astype(bool).all():
            raise ValueError(
                "Non-empirical-standardization curve rows are forbidden"
            )
        if not frame["joint_support_threshold_quantile"].astype(float).eq(
            SUPPORT_THRESHOLD_QUANTILE
        ).all():
            raise ValueError("Joint-support threshold quantile changed")
        grid_supported = frame["grid_supported"].astype(bool)
        reference_supported = frame["reference_supported"].astype(bool)
        joint_supported = frame["joint_support"].astype(bool)
        if not joint_supported.eq(
            grid_supported & reference_supported
        ).all():
            raise ValueError(
                "Joint support must require both grid and reference support"
            )

        duplicate_key = [
            "record_id",
            "repeat",
            "grid_percentile",
            "class_label",
        ]
        if frame.duplicated(duplicate_key).any():
            raise ValueError("Duplicate record/repeat/grid/class curve rows")
        class_counts = frame.groupby(
            ["record_id", "repeat", "grid_percentile"],
            dropna=False,
        )["class_label"].nunique()
        if not class_counts.eq(len(CLASS_LABELS)).all():
            raise ValueError("A curve cell is missing one or more classes")
        for probability_column in ("probability", "reference_probability"):
            matrices = (
                frame.pivot(
                    index=["record_id", "repeat", "grid_percentile"],
                    columns="class_label",
                    values=probability_column,
                )
                .loc[:, list(CLASS_LABELS)]
                .to_numpy(float)
            )
            _normalize_roundoff_simplex(matrices)

        support_cells = frame.drop_duplicates(
            ["record_id", "repeat", "grid_percentile"]
        )
        observed_records = support_cells["record_id"].astype(str).nunique()
        if observed_records != int(expected_records):
            raise ValueError(
                f"Expected {expected_records} curve records, "
                f"found {observed_records}"
            )
        repeat_support = (
            support_cells.groupby(
                ["record_id", "grid_percentile"], as_index=False
            )
            .agg(
                n_available_repeats=("repeat", "nunique"),
                n_supported_repeats=("joint_support", "sum"),
            )
        )
        repeat_support["cell_supported"] = (
            repeat_support["n_available_repeats"].le(int(expected_repeats))
            & repeat_support["n_supported_repeats"].ge(
                int(minimum_supported_repeats)
            )
        )
        common = (
            repeat_support.groupby("record_id")
            .agg(
                grid_count=("grid_percentile", "nunique"),
                all_supported=("cell_supported", "all"),
                minimum_supported=(
                    "n_supported_repeats",
                    "min",
                ),
            )
            .loc[
                lambda value: value["grid_count"].eq(len(GRID_PERCENTILES))
                & value["all_supported"]
            ]
        )
        common_ids = set(common.index.astype(str))
        n_common = len(common_ids)
        coverage = float(n_common / int(expected_records))
        minimum_common_repeats = (
            int(common["minimum_supported"].min()) if n_common else 0
        )
        if coverage < float(minimum_common_record_coverage):
            audit_rows.append(
                _curve_audit_row(
                    panel=str(panel),
                    feature=str(feature),
                    status="passed_no_estimable_curve",
                    n_common_records=n_common,
                    common_record_coverage=coverage,
                    minimum_supported_repeats=minimum_common_repeats,
                    reason=(
                        "fixed common-record coverage below frozen 0.80 "
                        "threshold"
                    ),
                    expected_records=int(expected_records),
                )
            )
            continue

        selected = frame.loc[
            frame["record_id"].astype(str).isin(common_ids)
            & frame["joint_support"].astype(bool)
        ].copy()
        record_rows: list[dict[str, Any]] = []
        for (record_id, percentile), cell in selected.groupby(
            ["record_id", "grid_percentile"], sort=True
        ):
            probability = (
                cell.pivot(
                    index="repeat",
                    columns="class_label",
                    values="probability",
                )
                .loc[:, list(CLASS_LABELS)]
                .to_numpy(float)
            )
            reference = (
                cell.pivot(
                    index="repeat",
                    columns="class_label",
                    values="reference_probability",
                )
                .loc[:, list(CLASS_LABELS)]
                .to_numpy(float)
            )
            probability = _normalize_roundoff_simplex(probability)
            reference = _normalize_roundoff_simplex(reference)
            mean_probability = _normalize_roundoff_simplex(
                probability.mean(axis=0, keepdims=True)
            )[0]
            mean_reference = _normalize_roundoff_simplex(
                reference.mean(axis=0, keepdims=True)
            )[0]
            identity = cell.iloc[0]
            for class_index, class_label in enumerate(CLASS_LABELS):
                record_rows.append(
                    {
                        "record_id": str(record_id),
                        "cv_group_id": str(identity["cv_group_id"]),
                        "grid_percentile": float(percentile),
                        "class_label": str(class_label),
                        "probability": float(
                            mean_probability[class_index]
                        ),
                        "reference_probability": float(
                            mean_reference[class_index]
                        ),
                        "grid_value_fold_mean_diagnostic": float(
                            cell["grid_value_fold"].astype(float).mean()
                        ),
                        "n_supported_repeats": int(
                            cell["repeat"].nunique()
                        ),
                    }
                )
        record_level = pd.DataFrame(record_rows)
        expected_record_rows = (
            n_common * len(GRID_PERCENTILES) * len(CLASS_LABELS)
        )
        if len(record_level) != expected_record_rows:
            raise AssertionError(
                "The fixed common record set changed across grid points"
            )

        curve_rows: list[dict[str, Any]] = []
        for percentile, cell in record_level.groupby(
            "grid_percentile", sort=True
        ):
            probability = np.asarray(
                [
                    cell.loc[
                        cell["class_label"].eq(class_label), "probability"
                    ].mean()
                    for class_label in CLASS_LABELS
                ],
                dtype=float,
            )
            reference = np.asarray(
                [
                    cell.loc[
                        cell["class_label"].eq(class_label),
                        "reference_probability",
                    ].mean()
                    for class_label in CLASS_LABELS
                ],
                dtype=float,
            )
            probability = _normalize_roundoff_simplex(
                probability[None, :]
            )[0]
            reference = _normalize_roundoff_simplex(reference[None, :])[0]
            difference = probability - reference
            for class_index, class_label in enumerate(CLASS_LABELS):
                curve_rows.append(
                    {
                        "panel": str(panel),
                        "feature": str(feature),
                        "grid_percentile": float(percentile),
                        "reference_percentile": REFERENCE_PERCENTILE,
                        "class_label": str(class_label),
                        "probability": float(probability[class_index]),
                        "reference_probability": float(
                            reference[class_index]
                        ),
                        "probability_difference_from_reference": float(
                            difference[class_index]
                        ),
                        "n_common_records": n_common,
                        "n_common_groups": int(
                            cell["cv_group_id"].astype(str).nunique()
                        ),
                        "common_record_coverage": coverage,
                        "minimum_supported_repeats_per_record": int(
                            cell["n_supported_repeats"].min()
                        ),
                        "grid_value_fold_mean_diagnostic": float(
                            cell[
                                "grid_value_fold_mean_diagnostic"
                            ].mean()
                        ),
                        "x_axis_semantics": (
                            "outer-training-fold percentile; diagnostic "
                            "fold values are not a fixed exposure"
                        ),
                        "joint_support_rule": (
                            "fold-local block-balanced Gower; grid and "
                            "reference nearest distance <= training LOO "
                            "nearest-distance q95; >=8/10 repeats at every "
                            "grid on one fixed common record set"
                        ),
                        "cross_fitted_empirical_standardization": True,
                        "interpretation": (
                            "adjusted predictive association; not causal "
                            "or mechanistic"
                        ),
                    }
                )
        curve = pd.DataFrame(curve_rows, columns=CURVE_COLUMNS)
        all_curves.append(curve)
        audit_rows.append(
            _curve_audit_row(
                panel=str(panel),
                feature=str(feature),
                status="released",
                n_common_records=n_common,
                common_record_coverage=coverage,
                minimum_supported_repeats=minimum_common_repeats,
                reason="all frozen release and common-support gates passed",
                expected_records=int(expected_records),
            )
        )

    curves = (
        pd.concat(all_curves, ignore_index=True)
        if all_curves
        else empty_curve_table()
    )
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return AdjustedProbabilityCurveResult(
        fold_predictions.loc[:, list(FOLD_PREDICTION_COLUMNS)].copy(),
        curves,
        audit,
    )


def _validate_full_outer_manifest(
    records: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    required = {"record_id", "cv_group_id", "repeat", "fold"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"Outer manifest is missing columns: {missing}")
    if records["record_id"].astype(str).nunique() != EXPECTED_RECORDS:
        raise ValueError("Adjusted curves require all 97 records")
    if len(records) != EXPECTED_RECORDS:
        raise ValueError("Adjusted curves require one row per 97 records")
    if records["cv_group_id"].astype(str).nunique() != 80:
        raise ValueError("Adjusted curves require all 80 connected groups")
    if manifest["repeat"].nunique() != EXPECTED_REPEATS:
        raise ValueError("Adjusted curves require all 10 outer repeats")
    counts = manifest.groupby(
        ["repeat", "record_id"], dropna=False
    ).size()
    if not counts.eq(1).all():
        raise ValueError("Every record must be outer test exactly once per repeat")
    record_ids = set(records["record_id"].astype(str))
    manifest_ids = set(manifest["record_id"].astype(str))
    if manifest_ids != record_ids:
        raise ValueError("Outer manifest record IDs do not match all 97 records")
    expected_groups = records.assign(
        record_id=records["record_id"].astype(str),
        cv_group_id=records["cv_group_id"].astype(str),
    ).set_index("record_id")["cv_group_id"]
    observed_groups = (
        manifest.assign(
            record_id=manifest["record_id"].astype(str),
            cv_group_id=manifest["cv_group_id"].astype(str),
        )
        .drop_duplicates(["record_id", "cv_group_id"])
        .set_index("record_id")["cv_group_id"]
    )
    if observed_groups.to_dict() != expected_groups.to_dict():
        raise ValueError("Outer manifest connected-group IDs changed")
    for repeat, selected in manifest.groupby("repeat"):
        if selected["fold"].nunique() != EXPECTED_FOLDS:
            raise ValueError(
                f"Outer repeat {repeat} does not contain five folds"
            )
        group_folds = selected.groupby("cv_group_id")["fold"].nunique()
        if not group_folds.eq(1).all():
            raise ValueError(
                f"Connected group leakage in outer repeat {repeat}"
            )


def _released_panels(decisions: pd.DataFrame, *, smoke: bool) -> tuple[str, ...]:
    authorization_column = (
        "adjusted_probability_curve_execution_authorized"
        if "adjusted_probability_curve_execution_authorized"
        in decisions.columns
        else "adjusted_probability_curve_release_pass"
    )
    required = {
        "panel",
        "role",
        "primary_release_gate_pass",
        "general_cardinal_region_claim_pass",
        authorization_column,
        "width_direction_consistent",
    }
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise KeyError(f"Release decisions are missing columns: {missing}")
    if smoke and decisions[authorization_column].astype(bool).any():
        raise ValueError("Smoke mode cannot authorize adjusted curves")
    selected = decisions.loc[
        decisions[authorization_column].astype(bool)
    ].copy()
    if selected.empty:
        return ()
    invalid = selected.loc[
        ~selected["panel"].astype(str).isin(EXPECTED_FORMAL_PANELS)
        | ~selected["role"].astype(str).eq(
            "formal_single_feature_increment"
        )
        | ~selected["primary_release_gate_pass"].astype(bool)
        | ~selected["general_cardinal_region_claim_pass"].astype(bool)
        | ~selected["width_direction_consistent"].astype(bool)
    ]
    if not invalid.empty:
        raise ValueError(
            "A curve was authorized without every frozen primary release gate"
        )
    if smoke:
        return ()
    panels = tuple(selected["panel"].astype(str))
    if len(set(panels)) != len(panels):
        raise ValueError("Duplicate release-decision panels")
    return tuple(sorted(panels))


def _selected_feature_scale(
    fit_registry: pd.DataFrame,
    *,
    panel: str,
    repeat: int,
    fold: int,
) -> float:
    required = {
        "width",
        "repeat",
        "fold",
        "panel",
        "model_family",
        "candidate_params",
        "fit_status",
    }
    missing = sorted(required - set(fit_registry.columns))
    if missing:
        raise KeyError(f"Model fit registry is missing columns: {missing}")
    selected = fit_registry.loc[
        fit_registry["width"].astype(float).eq(PRIMARY_WIDTH)
        & fit_registry["repeat"].astype(int).eq(int(repeat))
        & fit_registry["fold"].astype(int).eq(int(fold))
        & fit_registry["panel"].astype(str).eq(panel)
        & fit_registry["model_family"].astype(str).eq(PRIMARY_MODEL_FAMILY)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one selected primary MAP fit for "
            f"{panel}/repeat={repeat}/fold={fold}, found {len(selected)}"
        )
    row = selected.iloc[0]
    if str(row["fit_status"]) != "complete":
        raise ValueError("A non-complete outer MAP fit cannot generate curves")
    payload = json.loads(str(row["candidate_params"]))
    scale = float(payload["feature_prior_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Selected feature prior scale is invalid")
    return scale


def execute_released_adjusted_probability_curves(
    records: pd.DataFrame,
    primary_labels: pd.DataFrame,
    outer_manifest: pd.DataFrame,
    fit_registry: pd.DataFrame,
    release_decisions: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> AdjustedProbabilityCurveResult:
    """Refit and execute only release-authorized primary MAP curves.

    Inner model selection is not repeated.  For every outer split this
    function reuses the selected feature-prior scale persisted by the completed
    nested OOF run, deterministically refits the same primary model on the
    complete outer-training fold, and predicts only that fold's held-out
    records.
    """

    validate_adjusted_curve_contract(contract)
    panels = _released_panels(release_decisions, smoke=smoke)
    if not panels:
        audit = pd.DataFrame(
            [
                _curve_audit_row(
                    panel="",
                    feature="",
                    status="no_release_authorized",
                    n_common_records=0,
                    common_record_coverage=0.0,
                    minimum_supported_repeats=0,
                    reason=(
                        "no formal continuous feature passed every frozen "
                        "primary release gate"
                    ),
                    expected_records=EXPECTED_RECORDS,
                )
            ],
            columns=AUDIT_COLUMNS,
        )
        return AdjustedProbabilityCurveResult(
            empty_fold_predictions(), empty_curve_table(), audit
        )

    _validate_full_outer_manifest(records, outer_manifest)
    label_columns = {"record_id", *SOFT_LABEL_COLUMNS}
    missing_labels = sorted(label_columns - set(primary_labels.columns))
    if missing_labels:
        raise KeyError(f"Primary soft labels are missing: {missing_labels}")
    labels = primary_labels.loc[:, list(label_columns)].copy()
    labels["record_id"] = labels["record_id"].astype(str)
    if labels["record_id"].duplicated().any() or len(labels) != EXPECTED_RECORDS:
        raise ValueError("Primary labels must contain one row per 97 records")
    label_lookup = labels.set_index("record_id")

    primary = contract["models"]["primary"]
    preprocessing = contract["preprocessing"]
    all_fold_predictions: list[pd.DataFrame] = []
    fold_failures: list[dict[str, Any]] = []
    for panel in panels:
        source_columns = required_association_source_columns(panel)
        missing_sources = sorted(set(source_columns) - set(records.columns))
        if missing_sources:
            raise KeyError(
                f"Records are missing panel sources: {missing_sources}"
            )
        for (repeat, fold), manifest_fold in outer_manifest.groupby(
            ["repeat", "fold"], sort=True
        ):
            test_ids = set(manifest_fold["record_id"].astype(str))
            is_test = records["record_id"].astype(str).isin(test_ids)
            train = records.loc[~is_test].reset_index(drop=True)
            test = records.loc[is_test].reset_index(drop=True)
            if len(test) != len(manifest_fold):
                raise ValueError("Outer manifest and records disagree")
            y_train = label_lookup.loc[
                train["record_id"].astype(str), list(SOFT_LABEL_COLUMNS)
            ].to_numpy(float)
            scale = _selected_feature_scale(
                fit_registry,
                panel=panel,
                repeat=int(repeat),
                fold=int(fold),
            )
            estimator = ReducedPanelTwoStageMAPClassifier(
                panel=panel,
                rare_min_frequency=int(
                    preprocessing["rare_min_frequency"]
                ),
                feature_prior_scale=scale,
                intercept_prior_scale=float(
                    primary["intercept_prior_scale"]
                ),
                max_iter=int(primary["max_iter"]),
                tolerance=float(primary["tolerance"]),
            ).fit(train.loc[:, list(source_columns)], y_train)
            if estimator.fit_status_ != "complete":
                raise RuntimeError(
                    "Deterministic curve refit did not reproduce a complete "
                    f"primary MAP fit: {panel}/{repeat}/{fold}"
                )
            try:
                fold_predictions = (
                    generate_fold_adjusted_probability_predictions(
                        estimator,
                        train.loc[
                            :,
                            [
                                "record_id",
                                "cv_group_id",
                                *source_columns,
                            ],
                        ],
                        test.loc[
                            :,
                            [
                                "record_id",
                                "cv_group_id",
                                *source_columns,
                            ],
                        ],
                        panel_name=panel,
                        repeat=int(repeat),
                        fold=int(fold),
                    )
                )
            except CurveNotEstimableError as exc:
                fold_failures.append(
                    {
                        "panel": panel,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "error": str(exc),
                    }
                )
                continue
            all_fold_predictions.append(fold_predictions)

    candidates = (
        pd.concat(all_fold_predictions, ignore_index=True)
        if all_fold_predictions
        else empty_fold_predictions()
    )
    if candidates.empty:
        audit = pd.DataFrame(
            [
                _curve_audit_row(
                    panel=panel,
                    feature=str(
                        FROZEN_ASSOCIATION_PANELS[panel].added_features[0]
                    ),
                    status="passed_no_estimable_curve",
                    n_common_records=0,
                    common_record_coverage=0.0,
                    minimum_supported_repeats=0,
                    reason=(
                        "no outer fold retained an estimable continuous "
                        "feature"
                    ),
                    expected_records=EXPECTED_RECORDS,
                )
                for panel in panels
            ],
            columns=AUDIT_COLUMNS,
        )
        return AdjustedProbabilityCurveResult(candidates, empty_curve_table(), audit)

    result = aggregate_adjusted_probability_predictions(candidates)
    audited_panels = set(result.audit["panel"].astype(str))
    missing_audits = [panel for panel in panels if panel not in audited_panels]
    if missing_audits:
        missing_rows = pd.DataFrame(
            [
                _curve_audit_row(
                    panel=panel,
                    feature=str(
                        FROZEN_ASSOCIATION_PANELS[panel].added_features[0]
                    ),
                    status="passed_no_estimable_curve",
                    n_common_records=0,
                    common_record_coverage=0.0,
                    minimum_supported_repeats=0,
                    reason=(
                        "no outer fold retained an estimable continuous "
                        "feature for this authorized panel"
                    ),
                    expected_records=EXPECTED_RECORDS,
                )
                for panel in missing_audits
            ],
            columns=AUDIT_COLUMNS,
        )
        result = AdjustedProbabilityCurveResult(
            result.fold_predictions,
            result.curves,
            pd.concat([result.audit, missing_rows], ignore_index=True),
        )
    if fold_failures and not result.audit.empty:
        failed_by_panel: dict[str, int] = {}
        for failure in fold_failures:
            key = str(failure["panel"])
            failed_by_panel[key] = failed_by_panel.get(key, 0) + 1
        audit = result.audit.copy()
        for index, row in audit.iterrows():
            count = failed_by_panel.get(str(row["panel"]), 0)
            if count:
                audit.loc[index, "reason"] = (
                    f"{row['reason']}; {count} outer folds had a "
                    "constant/non-estimable feature"
                )
        result = AdjustedProbabilityCurveResult(
            result.fold_predictions, result.curves, audit
        )
    return result


__all__ = [
    "AdjustedProbabilityCurveResult",
    "CurveNotEstimableError",
    "GRID_PERCENTILES",
    "REFERENCE_PERCENTILE",
    "aggregate_adjusted_probability_predictions",
    "empty_curve_audit",
    "empty_curve_table",
    "empty_fold_predictions",
    "execute_released_adjusted_probability_curves",
    "generate_fold_adjusted_probability_predictions",
    "validate_adjusted_curve_contract",
]
