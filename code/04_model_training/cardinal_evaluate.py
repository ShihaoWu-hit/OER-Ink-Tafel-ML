"""Nested connected-group evaluation for posterior-soft cardinal classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .cardinal_cv import (
    make_inner_soft_group_splits,
    make_soft_stratified_group_manifest,
)
from .cardinal_metrics import (
    CLASS_LABELS,
    PREDICTION_COLUMNS,
    SOFT_LABEL_COLUMNS,
    aggregate_repeated_oof,
    calibration_table,
    collapsed_hard_metrics,
    expected_confusion_table,
    multiclass_brier,
    soft_cross_entropy,
    two_stage_probability_metrics,
)
from .cardinal_models import (
    EngineeredCardinalClassifier,
    make_cardinal_model_spec,
)
from .evaluate import (
    DEFAULT_RELATION_COLUMNS,
    assert_no_relational_leakage,
    validate_split_manifest,
)
from .features import ResolvedPanel, panel_schema_frame, resolve_feature_panel
from .provenance import write_json


@dataclass(frozen=True)
class CardinalEvaluationConfig:
    outer_folds: int = 5
    outer_repeats: int = 10
    inner_folds: int = 4
    random_seed: int = 42
    outer_seeds: tuple[int, ...] | None = None
    bootstrap_replicates: int = 2000
    calibration_bins: int = 8
    panel_names: tuple[str, ...] = ("context", "context_plus_ink")
    model_names: tuple[str, ...] = (
        "class_prior",
        "soft_ordinal_logistic",
        "soft_multinomial_logistic",
        "soft_gower_knn",
        "soft_extra_trees",
    )
    logistic_C_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    gower_neighbors: tuple[int, ...] = (5, 9, 15)
    gower_distance_powers: tuple[float, ...] = (1.0, 2.0)
    extra_trees_n_estimators: int = 100
    extra_trees_max_depths: tuple[int, ...] = (2, 3)
    extra_trees_min_samples_leaf: tuple[int, ...] = (4, 8)
    reference_panel: str = "context"
    augmented_panel: str = "context_plus_ink"
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("Outer and inner CV require at least two folds")
        if self.outer_repeats < 1:
            raise ValueError("outer_repeats must be positive")
        if self.outer_seeds is not None and len(self.outer_seeds) != self.outer_repeats:
            raise ValueError("outer_seeds must contain one seed per repeat")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates cannot be negative")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least two")
        if not self.panel_names:
            raise ValueError("At least one feature panel is required")
        if not self.model_names:
            raise ValueError("At least one model family is required")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be positive")


@dataclass(frozen=True)
class CandidateDefinition:
    candidate_id: str
    model_family: str
    params: Mapping[str, Any]
    family_complexity_rank: int
    parameter_complexity_rank: int

    @property
    def complexity_key(self) -> tuple[int, int, str]:
        return (
            int(self.family_complexity_rank),
            int(self.parameter_complexity_rank),
            self.candidate_id,
        )


@dataclass
class CardinalEvaluationResult:
    split_manifest: pd.DataFrame
    inner_split_manifest: pd.DataFrame
    outer_fold_class_mass: pd.DataFrame
    inner_fold_class_mass: pd.DataFrame
    panel_schema: pd.DataFrame
    candidate_catalog: pd.DataFrame
    inner_candidate_scores: pd.DataFrame
    outer_selections: pd.DataFrame
    oof_probabilities: pd.DataFrame
    oof_aggregated: pd.DataFrame
    split_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    calibration: pd.DataFrame
    expected_confusion: pd.DataFrame
    paired_panel_deltas: pd.DataFrame
    paired_bootstrap: pd.DataFrame
    paired_bootstrap_summary: pd.DataFrame
    paired_contrast_registry: pd.DataFrame
    model_selection_frequency: pd.DataFrame
    qa: dict[str, Any]


def candidate_definitions(
    config: CardinalEvaluationConfig,
    *,
    smoke: bool = False,
) -> tuple[CandidateDefinition, ...]:
    """Return the finite, predeclared candidate set."""

    rows: list[CandidateDefinition] = []
    if "class_prior" in config.model_names:
        rows.append(
            CandidateDefinition("class_prior", "class_prior", {}, 0, 0)
        )
    if "soft_ordinal_logistic" in config.model_names:
        values = config.logistic_C_grid[:1] if smoke else config.logistic_C_grid
        for rank, value in enumerate(sorted(values)):
            rows.append(
                CandidateDefinition(
                    f"soft_ordinal_logistic__C_{value:g}",
                    "soft_ordinal_logistic",
                    {"C": float(value)},
                    1,
                    rank,
                )
            )
    if "soft_multinomial_logistic" in config.model_names:
        values = config.logistic_C_grid[:1] if smoke else config.logistic_C_grid
        for rank, value in enumerate(sorted(values)):
            rows.append(
                CandidateDefinition(
                    f"soft_multinomial_logistic__C_{value:g}",
                    "soft_multinomial_logistic",
                    {"C": float(value)},
                    2,
                    rank,
                )
            )
    if "soft_gower_knn" in config.model_names:
        neighbors = config.gower_neighbors[:1] if smoke else config.gower_neighbors
        powers = (
            config.gower_distance_powers[:1]
            if smoke
            else config.gower_distance_powers
        )
        rank = 0
        # Larger k is smoother/simpler; lower distance power is simpler.
        for n_neighbors in sorted(neighbors, reverse=True):
            for distance_power in sorted(powers):
                rows.append(
                    CandidateDefinition(
                        f"soft_gower_knn__k_{n_neighbors}__p_{distance_power:g}",
                        "soft_gower_knn",
                        {
                            "n_neighbors": int(n_neighbors),
                            "distance_power": float(distance_power),
                        },
                        3,
                        rank,
                    )
                )
                rank += 1
    if "soft_extra_trees" in config.model_names:
        depths = (
            config.extra_trees_max_depths[:1]
            if smoke
            else config.extra_trees_max_depths
        )
        leaves = (
            config.extra_trees_min_samples_leaf[:1]
            if smoke
            else config.extra_trees_min_samples_leaf
        )
        rank = 0
        for depth in sorted(depths):
            for leaf in sorted(leaves, reverse=True):
                rows.append(
                    CandidateDefinition(
                        f"soft_extra_trees__depth_{depth}__leaf_{leaf}",
                        "soft_extra_trees",
                        {
                            "n_estimators": int(config.extra_trees_n_estimators),
                            "max_depth": int(depth),
                            "min_samples_leaf": int(leaf),
                            "n_jobs": 1,
                        },
                        4,
                        rank,
                    )
                )
                rank += 1
    if not rows:
        raise ValueError("Configured cardinal model list produced no candidates")
    return tuple(rows)


def candidate_catalog_frame(candidates: Sequence[CandidateDefinition]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "model_family": candidate.model_family,
                "params": json.dumps(dict(candidate.params), sort_keys=True),
                "family_complexity_rank": candidate.family_complexity_rank,
                "parameter_complexity_rank": candidate.parameter_complexity_rank,
            }
            for candidate in candidates
        ]
    )


def _candidate_seed(base_seed: int, candidate_id: str, offset: int = 0) -> int:
    digest = int.from_bytes(
        hashlib.sha256(candidate_id.encode("utf-8")).digest()[:4],
        "big",
    )
    return int((base_seed + digest + offset) % (2**31 - 1))


def _make_estimator(
    candidate: CandidateDefinition,
    panel: ResolvedPanel,
    *,
    random_seed: int,
):
    params = dict(candidate.params)
    feature_recipe = params.pop("feature_recipe", None)
    if feature_recipe is not None:
        base_model_family = params.pop("base_model_family", None)
        base_model_params = params.pop("base_model_params", None)
        if params:
            raise TypeError(
                "Unknown engineered cardinal candidate parameters: "
                f"{sorted(params)}"
            )
        if not base_model_family:
            raise ValueError(
                "Engineered cardinal candidates require base_model_family"
            )
        return EngineeredCardinalClassifier(
            feature_recipe=str(feature_recipe),
            base_model_family=str(base_model_family),
            base_model_params=dict(base_model_params or {}),
            random_seed=int(random_seed),
        )
    spec = make_cardinal_model_spec(
        candidate.model_family,
        numeric_columns=panel.numeric_columns,
        categorical_columns=panel.categorical_columns,
        blocks=panel.blocks,
        random_seed=random_seed,
        **params,
    )
    return clone(spec.estimator)


def _resolved_panel_schema_frame(
    panels: Mapping[str, ResolvedPanel],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for panel_name, panel in panels.items():
        block_lookup = {
            column: block
            for block, columns in panel.blocks.items()
            for column in columns
        }
        numeric = set(panel.numeric_columns)
        categorical = set(panel.categorical_columns)
        for logical_name, column in zip(
            panel.logical_names, panel.columns, strict=True
        ):
            rows.append(
                {
                    "panel": panel_name,
                    "logical_feature": logical_name,
                    "selected_column": column,
                    "kind": (
                        "numeric"
                        if column in numeric
                        else "categorical"
                        if column in categorical
                        else "untyped"
                    ),
                    "distance_block": block_lookup.get(column, "unassigned"),
                    "status": "selected",
                }
            )
        for logical_name in panel.missing_logical_features:
            rows.append(
                {
                    "panel": panel_name,
                    "logical_feature": logical_name,
                    "selected_column": "",
                    "kind": "",
                    "distance_block": "",
                    "status": "missing",
                }
            )
    return pd.DataFrame(rows)


def _score_candidate_inner(
    candidate: CandidateDefinition,
    X: pd.DataFrame,
    y: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    panel: ResolvedPanel,
    *,
    base_seed: int,
) -> dict[str, object]:
    losses: list[float] = []
    try:
        for inner_fold, (train_positions, validation_positions) in enumerate(
            inner_splits
        ):
            estimator = _make_estimator(
                candidate,
                panel,
                random_seed=_candidate_seed(
                    base_seed, candidate.candidate_id, inner_fold
                ),
            )
            estimator.fit(X.iloc[train_positions], y[train_positions])
            predicted = estimator.predict_proba(X.iloc[validation_positions])
            losses.append(
                soft_cross_entropy(y[validation_positions], predicted)
            )
    except Exception as error:  # persisted; never silently remove a candidate
        return {
            "candidate_id": candidate.candidate_id,
            "model_family": candidate.model_family,
            "params": json.dumps(dict(candidate.params), sort_keys=True),
            "family_complexity_rank": candidate.family_complexity_rank,
            "parameter_complexity_rank": candidate.parameter_complexity_rank,
            "n_inner_folds": len(inner_splits),
            "inner_soft_log_loss_mean": np.nan,
            "inner_soft_log_loss_sd": np.nan,
            "inner_soft_log_loss_se": np.nan,
            "status": "failed",
            "error": repr(error),
        }
    values = np.asarray(losses, dtype=float)
    return {
        "candidate_id": candidate.candidate_id,
        "model_family": candidate.model_family,
        "params": json.dumps(dict(candidate.params), sort_keys=True),
        "family_complexity_rank": candidate.family_complexity_rank,
        "parameter_complexity_rank": candidate.parameter_complexity_rank,
        "n_inner_folds": len(values),
        "inner_soft_log_loss_mean": float(values.mean()),
        "inner_soft_log_loss_sd": (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ),
        "inner_soft_log_loss_se": (
            float(values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        ),
        "status": "ok",
        "error": "",
    }


def _evaluate_candidates(
    candidates: Sequence[CandidateDefinition],
    X: pd.DataFrame,
    y: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    panel: ResolvedPanel,
    *,
    base_seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    if n_jobs == 1:
        rows = [
            _score_candidate_inner(
                candidate,
                X,
                y,
                inner_splits,
                panel,
                base_seed=base_seed,
            )
            for candidate in candidates
        ]
    else:
        # Threads avoid OS semaphore limits on constrained macOS/runtime
        # environments. Individual Extra Trees candidates keep n_jobs=1, so
        # this remains the single parallelism layer.
        rows = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
            joblib.delayed(_score_candidate_inner)(
                candidate,
                X,
                y,
                inner_splits,
                panel,
                base_seed=base_seed,
            )
            for candidate in candidates
        )
    return pd.DataFrame(rows)


def _one_se_selection(scores: pd.DataFrame) -> pd.Series:
    valid = scores.loc[
        scores["status"].eq("ok")
        & np.isfinite(scores["inner_soft_log_loss_mean"])
    ].copy()
    if valid.empty:
        errors = scores.loc[:, ["candidate_id", "error"]].to_dict("records")
        raise RuntimeError(f"All cardinal candidates failed in inner CV: {errors}")
    best = valid.sort_values(
        ["inner_soft_log_loss_mean", "candidate_id"], kind="stable"
    ).iloc[0]
    threshold = float(
        best["inner_soft_log_loss_mean"] + best["inner_soft_log_loss_se"]
    )
    eligible = valid.loc[
        valid["inner_soft_log_loss_mean"] <= threshold + 1e-15
    ].copy()
    return eligible.sort_values(
        [
            "family_complexity_rank",
            "parameter_complexity_rank",
            "inner_soft_log_loss_mean",
            "candidate_id",
        ],
        kind="stable",
    ).iloc[0]


def _metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        **two_stage_probability_metrics(observed, predicted),
        **collapsed_hard_metrics(observed, predicted),
    }


def _aggregate_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    if oof.empty:
        return pd.DataFrame()
    keys = ["analysis_id", "scope"]
    if "panel_role" in oof.columns:
        keys.append("panel_role")
    keys.extend(["panel", "model_role", "model_family"])
    rows: list[dict[str, object]] = []
    for values, group in oof.groupby(keys, dropna=False):
        observed = group.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
        predicted = group.loc[:, PREDICTION_COLUMNS].to_numpy(float)
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_records": int(len(group)),
                "n_connected_groups": int(group["cv_group_id"].nunique()),
                "min_repeated_oof": int(group["n_repeated_oof"].min()),
                "max_repeated_oof": int(group["n_repeated_oof"].max()),
                **_metric_row(observed, predicted),
            }
        )
    return pd.DataFrame(rows)


def _calibration_and_confusion(
    oof: pd.DataFrame,
    *,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if oof.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["analysis_id", "scope"]
    if "panel_role" in oof.columns:
        keys.append("panel_role")
    keys.extend(["panel", "model_role", "model_family"])
    calibration_rows: list[pd.DataFrame] = []
    confusion_rows: list[pd.DataFrame] = []
    for values, group in oof.groupby(keys, dropna=False):
        metadata = dict(zip(keys, values, strict=True))
        observed = group.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
        predicted = group.loc[:, PREDICTION_COLUMNS].to_numpy(float)
        calibration = calibration_table(
            observed,
            predicted,
            class_labels=CLASS_LABELS,
            n_bins=n_bins,
        )
        confusion = expected_confusion_table(
            observed,
            predicted,
            class_labels=CLASS_LABELS,
        )
        for key, value in reversed(tuple(metadata.items())):
            calibration.insert(0, key, value)
            confusion.insert(0, key, value)
        calibration_rows.append(calibration)
        confusion_rows.append(confusion)
    return (
        pd.concat(calibration_rows, ignore_index=True),
        pd.concat(confusion_rows, ignore_index=True),
    )


def _paired_split_deltas(
    split_metrics: pd.DataFrame,
    *,
    reference_panel: str,
    augmented_panel: str,
) -> pd.DataFrame:
    selected = split_metrics.loc[
        split_metrics["model_role"].eq("joint_selected")
        & split_metrics["model_family"].eq("nested_selected")
    ].copy()
    if not {reference_panel, augmented_panel}.issubset(
        set(selected["panel"].astype(str))
    ):
        return pd.DataFrame()
    keys = ["analysis_id", "scope", "repeat", "fold"]
    metrics = (
        "soft_log_loss",
        "multiclass_brier",
        "gate_soft_log_loss",
        "gate_brier",
        "conditional_soft_log_loss",
        "conditional_brier",
        "conditional_rps",
    )
    rows: list[pd.DataFrame] = []
    for metric in metrics:
        reference = selected.loc[
            selected["panel"].eq(reference_panel), keys + [metric]
        ].rename(columns={metric: "reference_value"})
        augmented = selected.loc[
            selected["panel"].eq(augmented_panel), keys + [metric]
        ].rename(columns={metric: "augmented_value"})
        paired = reference.merge(augmented, on=keys, validate="one_to_one")
        if paired.empty:
            continue
        paired["comparison"] = (
            f"{augmented_panel}_minus_{reference_panel}"
        )
        paired["reference_panel"] = reference_panel
        paired["augmented_panel"] = augmented_panel
        paired["metric"] = metric
        paired["delta"] = paired["augmented_value"] - paired["reference_value"]
        rows.append(paired)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _paired_record_table(
    oof: pd.DataFrame,
    *,
    reference_panel: str,
    augmented_panel: str,
) -> pd.DataFrame:
    selected = oof.loc[
        oof["model_role"].eq("joint_selected")
        & oof["model_family"].eq("nested_selected")
    ].copy()
    keys = [
        "analysis_id",
        "scope",
        "record_id",
        "cv_group_id",
        *SOFT_LABEL_COLUMNS,
    ]
    reference = selected.loc[
        selected["panel"].eq(reference_panel),
        keys + list(PREDICTION_COLUMNS),
    ].rename(
        columns={
            column: f"reference_{column}" for column in PREDICTION_COLUMNS
        }
    )
    augmented = selected.loc[
        selected["panel"].eq(augmented_panel),
        keys + list(PREDICTION_COLUMNS),
    ].rename(
        columns={
            column: f"augmented_{column}" for column in PREDICTION_COLUMNS
        }
    )
    return reference.merge(augmented, on=keys, validate="one_to_one")


def _paired_group_bootstrap(
    oof: pd.DataFrame,
    *,
    reference_panel: str,
    augmented_panel: str,
    replicates: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replicates <= 0:
        return pd.DataFrame(), pd.DataFrame()
    paired = _paired_record_table(
        oof,
        reference_panel=reference_panel,
        augmented_panel=augmented_panel,
    )
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame()
    observed = paired.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
    reference = paired.loc[
        :, [f"reference_{column}" for column in PREDICTION_COLUMNS]
    ].to_numpy(float)
    augmented = paired.loc[
        :, [f"augmented_{column}" for column in PREDICTION_COLUMNS]
    ].to_numpy(float)
    metric_functions = {
        "soft_log_loss": soft_cross_entropy,
        "multiclass_brier": multiclass_brier,
    }
    observed_delta = {
        name: float(function(observed, augmented) - function(observed, reference))
        for name, function in metric_functions.items()
    }
    groups = paired["cv_group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    positions = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        sampled_positions = np.concatenate(
            [positions[group] for group in sampled_groups]
        )
        for metric, function in metric_functions.items():
            delta = float(
                function(observed[sampled_positions], augmented[sampled_positions])
                - function(
                    observed[sampled_positions], reference[sampled_positions]
                )
            )
            rows.append(
                {
                    "analysis_id": str(paired["analysis_id"].iloc[0]),
                    "scope": str(paired["scope"].iloc[0]),
                    "comparison": f"{augmented_panel}_minus_{reference_panel}",
                    "reference_panel": reference_panel,
                    "augmented_panel": augmented_panel,
                    "metric": metric,
                    "replicate": int(replicate),
                    "delta": delta,
                }
            )
    raw = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for metric, group in raw.groupby("metric"):
        values = group["delta"].to_numpy(float)
        summary_rows.append(
            {
                "analysis_id": str(paired["analysis_id"].iloc[0]),
                "scope": str(paired["scope"].iloc[0]),
                "comparison": f"{augmented_panel}_minus_{reference_panel}",
                "reference_panel": reference_panel,
                "augmented_panel": augmented_panel,
                "metric": metric,
                "observed_delta": observed_delta[str(metric)],
                "bootstrap_mean": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_delta_below_zero": float(np.mean(values < 0)),
                "n_bootstrap": int(len(values)),
            }
        )
    return raw, pd.DataFrame(summary_rows)


def evaluate_paired_panel_contrasts(
    oof: pd.DataFrame,
    split_metrics: pd.DataFrame,
    contrasts: Sequence[Mapping[str, str]],
    *,
    analysis_id: str,
    scope: str,
    panel_role: str,
    replicates: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate predeclared cross-panel contrasts on identical saved folds.

    This helper permits sensitivity panels to live in a separate analysis
    directory while still pairing them with their frozen reference panels.
    The caller must supply record-aggregated OOF probabilities and split
    metrics from analyses that used the same population, seeds, and folds.
    """

    delta_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    registry_rows: list[dict[str, object]] = []
    for position, contrast in enumerate(contrasts):
        contrast_id = str(contrast["id"])
        contrast_role = str(contrast["role"])
        reference = str(contrast["reference"])
        augmented = str(contrast["augmented"])
        if reference == augmented:
            raise ValueError(f"Contrast {contrast_id!r} uses the same panel twice")
        available_oof = set(oof["panel"].astype(str))
        available_splits = set(split_metrics["panel"].astype(str))
        missing = sorted(
            {reference, augmented} - available_oof.intersection(available_splits)
        )
        if missing:
            raise ValueError(
                f"Contrast {contrast_id!r} is missing evaluated panels: {missing}"
            )

        paired_oof = oof.loc[
            oof["panel"].astype(str).isin([reference, augmented])
        ].copy()
        paired_splits = split_metrics.loc[
            split_metrics["panel"].astype(str).isin([reference, augmented])
        ].copy()
        paired_oof["analysis_id"] = analysis_id
        paired_oof["scope"] = scope
        paired_splits["analysis_id"] = analysis_id
        paired_splits["scope"] = scope

        deltas = _paired_split_deltas(
            paired_splits,
            reference_panel=reference,
            augmented_panel=augmented,
        )
        raw, summary = _paired_group_bootstrap(
            paired_oof,
            reference_panel=reference,
            augmented_panel=augmented,
            replicates=replicates,
            random_seed=random_seed + position * 10_003,
        )
        if deltas.empty or raw.empty or summary.empty:
            raise RuntimeError(
                f"Frozen panel contrast {contrast_id!r} produced no paired output"
            )
        for frame in (deltas, raw, summary):
            frame.insert(2, "panel_role", panel_role)
            frame.insert(3, "contrast_id", contrast_id)
            frame.insert(4, "contrast_role", contrast_role)
        delta_frames.append(deltas)
        bootstrap_frames.append(raw)
        summary_frames.append(summary)
        registry_rows.append(
            {
                "analysis_id": analysis_id,
                "scope": scope,
                "panel_role": panel_role,
                "contrast_id": contrast_id,
                "contrast_role": contrast_role,
                "reference_panel": reference,
                "augmented_panel": augmented,
                "delta_sign": "augmented_minus_reference",
                "primary_metric": "soft_log_loss",
                "bootstrap_replicates": int(replicates),
            }
        )
    if not registry_rows:
        raise ValueError("At least one paired panel contrast is required")
    return (
        pd.concat(delta_frames, ignore_index=True),
        pd.concat(bootstrap_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
        pd.DataFrame(registry_rows),
    )


def _selection_frequency(selections: pd.DataFrame) -> pd.DataFrame:
    selected = selections.loc[selections["selection_scope"].eq("joint")].copy()
    if selected.empty:
        return pd.DataFrame()
    keys = ["analysis_id", "scope"]
    if "panel_role" in selected.columns:
        keys.append("panel_role")
    keys.extend(["panel", "selected_model_family"])
    result = (
        selected.groupby(keys, dropna=False)
        .size()
        .rename("n_outer_selections")
        .reset_index()
    )
    denominator_keys = ["analysis_id", "scope"]
    if "panel_role" in selected.columns:
        denominator_keys.append("panel_role")
    denominator_keys.append("panel")
    denominators = selected.groupby(denominator_keys, dropna=False).size()
    result["selection_fraction"] = [
        count
        / denominators.loc[
            tuple(getattr(row, key) for key in denominator_keys)
        ]
        for row, count in zip(
            result.itertuples(index=False), result["n_outer_selections"], strict=True
        )
    ]
    return result


def _validated_outer_manifest(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    settings: CardinalEvaluationConfig,
    record_id_col: str,
    group_col: str,
    relation_columns: Sequence[str],
) -> pd.DataFrame:
    """Validate and normalize a precomputed outer-fold assignment."""

    frozen = manifest.copy()
    validate_split_manifest(
        data,
        frozen,
        group_col=group_col,
        record_id_col=record_id_col,
        expected_repeats=settings.outer_repeats,
        expected_folds=settings.outer_folds,
        relation_columns=relation_columns,
    )
    required = {"repeat", "fold", "record_id", "cv_group_id", "split_seed"}
    missing = sorted(required - set(frozen.columns))
    if missing:
        raise ValueError(
            f"Frozen outer manifest is missing required columns: {missing}"
        )
    if "row_position" not in frozen:
        position_lookup = {
            str(record_id): position
            for position, record_id in enumerate(data[record_id_col])
        }
        frozen["row_position"] = frozen["record_id"].astype(str).map(
            position_lookup
        )
    return frozen.sort_values(
        ["repeat", "fold", "row_position"], kind="stable"
    ).reset_index(drop=True)


def _inner_splits_from_frozen_manifest(
    train: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    outer_repeat: int,
    outer_fold: int,
    inner_folds: int,
    record_id_col: str,
    group_col: str,
    relation_columns: Sequence[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, int]:
    """Reconstruct one outer-training inner CV from persisted assignments."""

    required = {
        "outer_repeat",
        "outer_fold",
        "inner_fold",
        "record_id",
        "cv_group_id",
        "inner_seed",
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
    expected_ids = set(train[record_id_col].astype(str))
    observed_ids = set(selected["record_id"].astype(str))
    if observed_ids != expected_ids:
        raise ValueError(
            "Frozen inner manifest record set differs from the current "
            f"outer-training set for repeat={outer_repeat}, fold={outer_fold}"
        )
    counts = selected["record_id"].astype(str).value_counts()
    if not (counts == 1).all():
        raise ValueError(
            "Frozen inner manifest must assign every outer-training record "
            "exactly once"
        )
    expected_groups = train.set_index(
        train[record_id_col].astype(str), drop=False
    )[group_col].astype(str)
    observed_groups = selected.assign(
        _record_key=selected["record_id"].astype(str)
    ).set_index("_record_key")["cv_group_id"].astype(str)
    if any(
        observed_groups.get(record_id) != expected_groups.get(record_id)
        for record_id in expected_ids
    ):
        raise ValueError(
            "Frozen inner manifest connected-group IDs disagree with the "
            "current outer-training registry"
        )
    fold_values = sorted(int(value) for value in selected["inner_fold"].unique())
    if fold_values != list(range(inner_folds)):
        raise ValueError(
            "Frozen inner manifest does not contain the expected inner folds: "
            f"{fold_values} != {list(range(inner_folds))}"
        )
    seed_values = selected["inner_seed"].astype(int).unique()
    if len(seed_values) != 1:
        raise ValueError(
            "Frozen inner manifest must contain one seed per outer fold"
        )
    assignment_lookup = selected.assign(
        _record_key=selected["record_id"].astype(str)
    ).set_index("_record_key")["inner_fold"]
    assignments = np.asarray(
        [
            int(assignment_lookup.loc[str(record_id)])
            for record_id in train[record_id_col]
        ],
        dtype=int,
    )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for inner_fold in range(inner_folds):
        validation_positions = np.flatnonzero(assignments == inner_fold)
        train_positions = np.flatnonzero(assignments != inner_fold)
        if not len(validation_positions):
            raise ValueError(
                f"Frozen inner manifest produced empty fold {inner_fold}"
            )
        assert_no_relational_leakage(
            train.iloc[train_positions],
            train.iloc[validation_positions],
            group_col=group_col,
            relation_columns=relation_columns,
        )
        splits.append((train_positions, validation_positions))
    return splits, assignments, int(seed_values[0])


def run_nested_cardinal_evaluation(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    analysis_id: str,
    scope: str,
    panel_role: str,
    config: CardinalEvaluationConfig,
    smoke: bool = False,
    record_id_col: str = "record_id",
    group_col: str = "cv_group_id",
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
    frozen_outer_manifest: pd.DataFrame | None = None,
    frozen_inner_manifest: pd.DataFrame | None = None,
    candidate_definitions_override: Sequence[CandidateDefinition] | None = None,
    panel_resolutions_override: Mapping[str, ResolvedPanel] | None = None,
) -> CardinalEvaluationResult:
    """Run the complete nested soft-label classification evaluation."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = {
        record_id_col,
        group_col,
        *SOFT_LABEL_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Cardinal evaluation frame is missing columns: {missing}")
    data = frame.reset_index(drop=True).copy()
    if not str(panel_role).strip():
        raise ValueError("panel_role must be a non-empty label")
    if (frozen_outer_manifest is None) != (frozen_inner_manifest is None):
        raise ValueError(
            "Frozen outer and inner manifests must be supplied together"
        )
    if data[record_id_col].astype(str).duplicated().any():
        raise ValueError("Cardinal evaluation requires one row per record_id")

    settings = config
    if smoke:
        settings = CardinalEvaluationConfig(
            **{
                **asdict(config),
                "outer_folds": 2,
                "outer_repeats": 1,
                "inner_folds": 2,
                "outer_seeds": (config.random_seed,),
                "bootstrap_replicates": min(config.bootstrap_replicates, 50),
                "panel_names": tuple(config.panel_names),
                "n_jobs": min(config.n_jobs, 2),
            }
        )
    candidates = (
        tuple(candidate_definitions_override)
        if candidate_definitions_override is not None
        else candidate_definitions(settings, smoke=smoke)
    )
    if not candidates:
        raise ValueError("Cardinal evaluation requires at least one candidate")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Cardinal candidate IDs must be unique")
    unknown_families = sorted(
        {candidate.model_family for candidate in candidates}
        - set(settings.model_names)
    )
    if unknown_families:
        raise ValueError(
            "Candidate families are absent from config.model_names: "
            f"{unknown_families}"
        )
    candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
    candidate_catalog = candidate_catalog_frame(candidates)
    candidate_catalog.insert(0, "analysis_id", analysis_id)
    candidate_catalog.insert(1, "scope", scope)
    candidate_catalog.insert(2, "panel_role", panel_role)
    if panel_resolutions_override is None:
        panel_resolutions = {
            panel_name: resolve_feature_panel(
                data, panel_name, require_all=True
            )
            for panel_name in settings.panel_names
        }
        schema = panel_schema_frame(data, settings.panel_names)
    else:
        panel_resolutions = dict(panel_resolutions_override)
        if set(panel_resolutions) != set(settings.panel_names):
            raise ValueError(
                "Custom panel resolutions must exactly match config.panel_names"
            )
        for panel_name, panel in panel_resolutions.items():
            if panel.name != panel_name:
                raise ValueError(
                    f"Custom panel key {panel_name!r} disagrees with "
                    f"ResolvedPanel.name={panel.name!r}"
                )
            missing_columns = sorted(set(panel.columns) - set(data.columns))
            if missing_columns:
                raise KeyError(
                    f"Custom panel {panel_name!r} is missing source columns: "
                    f"{missing_columns}"
                )
        schema = _resolved_panel_schema_frame(panel_resolutions)
    schema.insert(0, "analysis_id", analysis_id)
    schema.insert(1, "scope", scope)
    schema.insert(2, "panel_role", panel_role)

    if frozen_outer_manifest is None:
        manifest = make_soft_stratified_group_manifest(
            data,
            SOFT_LABEL_COLUMNS,
            group_col=group_col,
            record_id_col=record_id_col,
            folds=settings.outer_folds,
            repeats=settings.outer_repeats,
            random_seed=settings.random_seed,
            repeat_seeds=settings.outer_seeds,
            relation_columns=relation_columns,
        )
    else:
        manifest = _validated_outer_manifest(
            data,
            frozen_outer_manifest,
            settings=settings,
            record_id_col=record_id_col,
            group_col=group_col,
            relation_columns=relation_columns,
        )
    label_lookup = data.loc[:, [record_id_col, *SOFT_LABEL_COLUMNS]].copy()
    label_lookup[record_id_col] = label_lookup[record_id_col].astype(str)
    outer_mass_source = manifest.merge(
        label_lookup,
        left_on="record_id",
        right_on=record_id_col,
        how="left",
        validate="many_to_one",
    )
    outer_fold_class_mass = (
        outer_mass_source.groupby(["repeat", "fold"], as_index=False)
        .agg(
            n_records=("record_id", "size"),
            n_connected_groups=("cv_group_id", "nunique"),
            **{column: (column, "sum") for column in SOFT_LABEL_COLUMNS},
        )
    )
    outer_fold_class_mass.insert(0, "analysis_id", analysis_id)
    outer_fold_class_mass.insert(1, "scope", scope)
    outer_fold_class_mass.insert(2, "panel_role", panel_role)
    if not (
        outer_fold_class_mass.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float) > 0
    ).all():
        raise ValueError(
            "At least one outer validation fold has zero posterior mass for a "
            "frozen cardinal class"
        )
    id_to_position = {
        str(record_id): position
        for position, record_id in enumerate(data[record_id_col])
    }
    inner_manifest_rows: list[dict[str, object]] = []
    inner_score_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    split_metric_rows: list[dict[str, object]] = []

    for repeat in sorted(manifest["repeat"].unique()):
        repeat_manifest = manifest.loc[manifest["repeat"].eq(repeat)]
        for fold in sorted(repeat_manifest["fold"].unique()):
            test_ids = set(
                repeat_manifest.loc[
                    repeat_manifest["fold"].eq(fold), "record_id"
                ].astype(str)
            )
            test_positions = np.asarray(
                sorted(id_to_position[record_id] for record_id in test_ids),
                dtype=int,
            )
            test_set = set(test_positions.tolist())
            train_positions = np.asarray(
                [
                    position
                    for position in range(len(data))
                    if position not in test_set
                ],
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
            if frozen_inner_manifest is None:
                inner_seed = (
                    settings.random_seed
                    + int(repeat) * 10_000
                    + int(fold) * 100
                )
                inner_splits, inner_assignments = make_inner_soft_group_splits(
                    train,
                    SOFT_LABEL_COLUMNS,
                    group_col=group_col,
                    folds=settings.inner_folds,
                    random_seed=inner_seed,
                    relation_columns=relation_columns,
                )
            else:
                (
                    inner_splits,
                    inner_assignments,
                    inner_seed,
                ) = _inner_splits_from_frozen_manifest(
                    train,
                    frozen_inner_manifest,
                    outer_repeat=int(repeat),
                    outer_fold=int(fold),
                    inner_folds=settings.inner_folds,
                    record_id_col=record_id_col,
                    group_col=group_col,
                    relation_columns=relation_columns,
                )
            for position, inner_fold in enumerate(inner_assignments):
                inner_manifest_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "scope": scope,
                        "panel_role": panel_role,
                        "outer_repeat": int(repeat),
                        "outer_fold": int(fold),
                        "inner_fold": int(inner_fold),
                        "record_id": str(train.iloc[position][record_id_col]),
                        "cv_group_id": str(train.iloc[position][group_col]),
                        "inner_seed": int(inner_seed),
                    }
                )
            y_train = train.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
            y_test = test.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)

            for panel_name, panel in panel_resolutions.items():
                X_train = train.loc[:, list(panel.columns)]
                X_test = test.loc[:, list(panel.columns)]
                scores = _evaluate_candidates(
                    candidates,
                    X_train,
                    y_train,
                    inner_splits,
                    panel,
                    base_seed=inner_seed,
                    n_jobs=settings.n_jobs,
                )
                scores.insert(0, "analysis_id", analysis_id)
                scores.insert(1, "scope", scope)
                scores.insert(2, "panel_role", panel_role)
                scores.insert(3, "panel", panel_name)
                scores.insert(4, "outer_repeat", int(repeat))
                scores.insert(5, "outer_fold", int(fold))
                inner_score_rows.extend(scores.to_dict("records"))

                selections: list[tuple[str, str, pd.Series]] = []
                joint = _one_se_selection(scores)
                selections.append(("joint", "nested_selected", joint))
                for family in settings.model_names:
                    family_scores = scores.loc[
                        scores["model_family"].eq(family)
                    ]
                    if family_scores.empty:
                        continue
                    selections.append(
                        ("family", family, _one_se_selection(family_scores))
                    )

                prediction_cache: dict[str, np.ndarray] = {}
                estimator_cache: dict[str, Any] = {}
                for _, _, selection in selections:
                    candidate_id = str(selection["candidate_id"])
                    if candidate_id in prediction_cache:
                        continue
                    candidate = candidate_lookup[candidate_id]
                    estimator = _make_estimator(
                        candidate,
                        panel,
                        random_seed=_candidate_seed(
                            inner_seed, candidate_id, 50_000
                        ),
                    )
                    estimator.fit(X_train, y_train)
                    estimator_cache[candidate_id] = estimator
                    prediction_cache[candidate_id] = estimator.predict_proba(
                        X_test
                    )

                for selection_scope, output_family, selection in selections:
                    candidate_id = str(selection["candidate_id"])
                    candidate = candidate_lookup[candidate_id]
                    fitted_estimator = estimator_cache[candidate_id]
                    predicted = prediction_cache[candidate_id]
                    model_role = (
                        "joint_selected"
                        if selection_scope == "joint"
                        else "family_selected"
                    )
                    selection_rows.append(
                        {
                            "analysis_id": analysis_id,
                            "scope": scope,
                            "panel_role": panel_role,
                            "panel": panel_name,
                            "repeat": int(repeat),
                            "fold": int(fold),
                            "selection_scope": selection_scope,
                            "model_family": output_family,
                            "selected_model_family": candidate.model_family,
                            "selected_candidate_id": candidate_id,
                            "selected_params": json.dumps(
                                dict(candidate.params), sort_keys=True
                            ),
                            "n_engineered_features": (
                                int(
                                    len(
                                        fitted_estimator.engineered_feature_names_
                                    )
                                )
                                if hasattr(
                                    fitted_estimator,
                                    "engineered_feature_names_",
                                )
                                else 0
                            ),
                            "engineered_feature_names": json.dumps(
                                list(
                                    getattr(
                                        fitted_estimator,
                                        "engineered_feature_names_",
                                        (),
                                    )
                                )
                            ),
                            "engineered_blocks": json.dumps(
                                {
                                    key: list(values)
                                    for key, values in getattr(
                                        fitted_estimator,
                                        "engineered_blocks_",
                                        {},
                                    ).items()
                                },
                                sort_keys=True,
                            ),
                            "inner_soft_log_loss_mean": float(
                                selection["inner_soft_log_loss_mean"]
                            ),
                            "inner_soft_log_loss_se": float(
                                selection["inner_soft_log_loss_se"]
                            ),
                            "selection_rule": "one_standard_error_simplest",
                        }
                    )
                    for local_index, row in test.iterrows():
                        output_row: dict[str, object] = {
                            "analysis_id": analysis_id,
                            "scope": scope,
                            "panel_role": panel_role,
                            "panel": panel_name,
                            "model_role": model_role,
                            "model_family": output_family,
                            "selected_model_family": candidate.model_family,
                            "selected_candidate_id": candidate_id,
                            "selected_params": json.dumps(
                                dict(candidate.params), sort_keys=True
                            ),
                            "repeat": int(repeat),
                            "fold": int(fold),
                            "record_id": str(row[record_id_col]),
                            "cv_group_id": str(row[group_col]),
                        }
                        for column, value in zip(
                            SOFT_LABEL_COLUMNS,
                            y_test[local_index],
                            strict=True,
                        ):
                            output_row[column] = float(value)
                        for column, value in zip(
                            PREDICTION_COLUMNS,
                            predicted[local_index],
                            strict=True,
                        ):
                            output_row[column] = float(value)
                        oof_rows.append(output_row)
                    split_metric_rows.append(
                        {
                            "analysis_id": analysis_id,
                            "scope": scope,
                            "panel_role": panel_role,
                            "panel": panel_name,
                            "model_role": model_role,
                            "model_family": output_family,
                            "selected_model_family": candidate.model_family,
                            "selected_candidate_id": candidate_id,
                            "repeat": int(repeat),
                            "fold": int(fold),
                            "n_train": int(len(train)),
                            "n_test": int(len(test)),
                            "n_train_groups": int(train[group_col].nunique()),
                            "n_test_groups": int(test[group_col].nunique()),
                            **_metric_row(y_test, predicted),
                        }
                    )

    inner_manifest = pd.DataFrame(inner_manifest_rows)
    inner_mass_source = inner_manifest.merge(
        label_lookup,
        left_on="record_id",
        right_on=record_id_col,
        how="left",
        validate="many_to_one",
    )
    inner_fold_class_mass = (
        inner_mass_source.groupby(
            ["outer_repeat", "outer_fold", "inner_fold"],
            as_index=False,
        )
        .agg(
            n_records=("record_id", "size"),
            n_connected_groups=("cv_group_id", "nunique"),
            **{column: (column, "sum") for column in SOFT_LABEL_COLUMNS},
        )
    )
    inner_fold_class_mass.insert(0, "analysis_id", analysis_id)
    inner_fold_class_mass.insert(1, "scope", scope)
    inner_fold_class_mass.insert(2, "panel_role", panel_role)
    if not (
        inner_fold_class_mass.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float) > 0
    ).all():
        raise ValueError(
            "At least one inner validation fold has zero posterior mass for a "
            "frozen cardinal class"
        )
    inner_scores = pd.DataFrame(inner_score_rows)
    selections = pd.DataFrame(selection_rows)
    oof = pd.DataFrame(oof_rows)
    split_metrics = pd.DataFrame(split_metric_rows)
    aggregated = aggregate_repeated_oof(oof)
    aggregate_metrics = _aggregate_metrics(aggregated)
    calibration, confusion = _calibration_and_confusion(
        aggregated, n_bins=settings.calibration_bins
    )
    if {settings.reference_panel, settings.augmented_panel}.issubset(
        set(settings.panel_names)
    ):
        (
            paired_deltas,
            paired_bootstrap,
            paired_bootstrap_summary,
            paired_contrast_registry,
        ) = evaluate_paired_panel_contrasts(
            aggregated,
            split_metrics,
            [
                {
                    "id": (
                        f"{settings.augmented_panel}_minus_"
                        f"{settings.reference_panel}"
                    ),
                    "role": (
                        "primary_incremental_ink"
                        if panel_role == "primary"
                        else "population_sensitivity_incremental_ink"
                    ),
                    "reference": settings.reference_panel,
                    "augmented": settings.augmented_panel,
                }
            ],
            analysis_id=analysis_id,
            scope=scope,
            panel_role=panel_role,
            replicates=settings.bootstrap_replicates,
            random_seed=settings.random_seed,
        )
    else:
        paired_deltas = pd.DataFrame()
        paired_bootstrap = pd.DataFrame()
        paired_bootstrap_summary = pd.DataFrame()
        paired_contrast_registry = pd.DataFrame()
    selection_frequency = _selection_frequency(selections)

    expected_repeats = settings.outer_repeats
    counts_ok = bool(
        not aggregated.empty
        and aggregated["n_repeated_oof"].eq(expected_repeats).all()
    )
    qa = {
        "analysis_id": analysis_id,
        "scope": scope,
        "panel_role": panel_role,
        "status": "complete_smoke" if smoke else "complete",
        "n_records": int(len(data)),
        "n_connected_groups": int(data[group_col].nunique()),
        "outer_folds": settings.outer_folds,
        "outer_repeats": settings.outer_repeats,
        "inner_folds": settings.inner_folds,
        "supervised_class_order": list(CLASS_LABELS),
        "frozen_outer_manifest_reused": frozen_outer_manifest is not None,
        "frozen_inner_manifest_reused": frozen_inner_manifest is not None,
        "soft_label_rows_sum_to_one": bool(
            np.allclose(
                data.loc[:, SOFT_LABEL_COLUMNS].sum(axis=1).to_numpy(float),
                1.0,
            )
        ),
        "all_outer_folds_have_positive_mass_for_all_classes": True,
        "all_inner_folds_have_positive_mass_for_all_classes": True,
        "minimum_outer_fold_class_mass": {
            column: float(outer_fold_class_mass[column].min())
            for column in SOFT_LABEL_COLUMNS
        },
        "minimum_inner_fold_class_mass": {
            column: float(inner_fold_class_mass[column].min())
            for column in SOFT_LABEL_COLUMNS
        },
        "repeated_oof_count_matches": counts_ok,
        "candidate_failures": int(inner_scores["status"].ne("ok").sum()),
        "paired_registered_contrast_available": bool(
            not paired_bootstrap_summary.empty
        ),
        "mechanistic_claim_allowed": False,
    }
    if not counts_ok:
        raise AssertionError("Repeated OOF coverage does not match requested repeats")

    artifacts = {
        "split_manifest.csv": manifest,
        "inner_split_manifest.csv": inner_manifest,
        "outer_fold_class_mass.csv": outer_fold_class_mass,
        "inner_fold_class_mass.csv": inner_fold_class_mass,
        "panel_schema.csv": schema,
        "candidate_catalog.csv": candidate_catalog,
        "inner_candidate_scores.csv": inner_scores,
        "outer_selections.csv": selections,
        "oof_probabilities.csv": oof,
        "oof_aggregated.csv": aggregated,
        "split_metrics.csv": split_metrics,
        "aggregate_metrics.csv": aggregate_metrics,
        "calibration_by_class.csv": calibration,
        "expected_confusion.csv": confusion,
        "paired_panel_deltas.csv": paired_deltas,
        "paired_group_bootstrap.csv": paired_bootstrap,
        "paired_group_bootstrap_summary.csv": paired_bootstrap_summary,
        "paired_contrast_registry.csv": paired_contrast_registry,
        "model_selection_frequency.csv": selection_frequency,
    }
    for filename, artifact in artifacts.items():
        artifact.to_csv(output / filename, index=False)
    write_json(output / "analysis_qa.json", qa)

    return CardinalEvaluationResult(
        split_manifest=manifest,
        inner_split_manifest=inner_manifest,
        outer_fold_class_mass=outer_fold_class_mass,
        inner_fold_class_mass=inner_fold_class_mass,
        panel_schema=schema,
        candidate_catalog=candidate_catalog,
        inner_candidate_scores=inner_scores,
        outer_selections=selections,
        oof_probabilities=oof,
        oof_aggregated=aggregated,
        split_metrics=split_metrics,
        aggregate_metrics=aggregate_metrics,
        calibration=calibration,
        expected_confusion=confusion,
        paired_panel_deltas=paired_deltas,
        paired_bootstrap=paired_bootstrap,
        paired_bootstrap_summary=paired_bootstrap_summary,
        paired_contrast_registry=paired_contrast_registry,
        model_selection_frequency=selection_frequency,
        qa=qa,
    )


def fit_final_cardinal_pipeline(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    primary_result: CardinalEvaluationResult,
    config: CardinalEvaluationConfig,
    label_scheme_id: str,
    relative_half_width: float,
    low_anchor_subdivision: Mapping[str, object],
    training_scope: str = "alkaline",
    smoke: bool = False,
) -> dict[str, object]:
    """Fit a full-data exploratory pipeline after nested OOF evaluation."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = config.reference_panel
    augmented = config.augmented_panel
    bootstrap = primary_result.paired_bootstrap_summary
    log_loss = (
        bootstrap.loc[bootstrap["metric"].eq("soft_log_loss")]
        if not bootstrap.empty and "metric" in bootstrap
        else pd.DataFrame()
    )
    ink_gate_passed = bool(
        not log_loss.empty
        and float(log_loss.iloc[0]["observed_delta"]) < 0
        and float(log_loss.iloc[0]["ci95_high"]) < 0
    )
    selected_panel_name = augmented if ink_gate_passed else reference
    panel = resolve_feature_panel(frame, selected_panel_name, require_all=True)
    candidates = candidate_definitions(config, smoke=smoke)
    inner_splits, _ = make_inner_soft_group_splits(
        frame,
        SOFT_LABEL_COLUMNS,
        group_col="cv_group_id",
        folds=2 if smoke else config.inner_folds,
        random_seed=config.random_seed + 900_000,
        relation_columns=DEFAULT_RELATION_COLUMNS,
    )
    X = frame.loc[:, list(panel.columns)]
    y = frame.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
    scores = _evaluate_candidates(
        candidates,
        X,
        y,
        inner_splits,
        panel,
        base_seed=config.random_seed + 900_000,
        n_jobs=min(config.n_jobs, 2) if smoke else config.n_jobs,
    )
    selected = _one_se_selection(scores)
    lookup = {candidate.candidate_id: candidate for candidate in candidates}
    candidate = lookup[str(selected["candidate_id"])]
    estimator = _make_estimator(
        candidate,
        panel,
        random_seed=_candidate_seed(
            config.random_seed + 900_000, candidate.candidate_id
        ),
    )
    estimator.fit(X, y)
    model_path = output / "hierarchical_cardinal_pipeline.joblib"
    joblib.dump(estimator, model_path)
    scores.to_csv(output / "full_data_inner_candidate_scores.csv", index=False)
    specification: dict[str, object] = {
        "status": "smoke_structural_only" if smoke else "exploratory_fitted",
        "performance_source": "nested_record_aggregated_outer_oof_not_training_fit",
        "label_scheme_id": str(label_scheme_id),
        "relative_half_width": float(relative_half_width),
        "supervised_class_order": list(CLASS_LABELS),
        "soft_target_columns": list(SOFT_LABEL_COLUMNS),
        "output_probability_columns": list(PREDICTION_COLUMNS),
        "low_classic_definition": "q_low_30_40_like = q_30_like + q_40_like",
        "low_anchor_subdivision": dict(low_anchor_subdivision),
        "training_scope": str(training_scope),
        "n_training_records": int(len(frame)),
        "panel": selected_panel_name,
        "panel_selection_rule": (
            "use context_plus_ink only when paired group-bootstrap 95% CI for "
            "soft-log-loss delta is entirely below zero; otherwise use context"
        ),
        "ink_panel_gate_passed": ink_gate_passed,
        "model_family": candidate.model_family,
        "candidate_id": candidate.candidate_id,
        "params": dict(candidate.params),
        "feature_columns": list(panel.columns),
        "numeric_columns": list(panel.numeric_columns),
        "categorical_columns": list(panel.categorical_columns),
        "distance_blocks": {
            key: list(values) for key, values in panel.blocks.items()
        },
        "model_path": "hierarchical_cardinal_pipeline.joblib",
        "model_path_relative_to": "exact_final_pipeline_json_parent",
        "mechanistic_claim_allowed": False,
    }
    write_json(output / "exact_final_pipeline.json", specification)
    return specification
