"""Leakage-resistant grouped nested evaluation for continuous Tafel targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from .features import (
    PANEL_FEATURES,
    ResolvedPanel,
    is_target_like_column,
    panel_schema_frame,
    resolve_feature_panel,
)
from .models import (
    gaussian_log_scores,
    lognormal_original_scale_scores,
    make_model_spec,
    predict_log_with_uncertainty,
)
from .interpretation import (
    BLOCK_EFFECT_COLUMNS,
    BLOCK_IMPORTANCE_COLUMNS,
    outer_test_block_perturbations,
)


DEFAULT_RELATION_COLUMNS = ("publication_id", "lab_group_id", "recipe_hash", "sample_id")
DEFAULT_MODELS = ("dummy_mean", "dummy_median", "bayesian_ridge", "gower_knn", "extra_trees")
PROPERTY_PROXY_COLUMNS = (
    "solvent_mixture_property_proxy_viscosity_mPa_s",
    "solvent_mixture_property_proxy_surface_tension_mN_m",
    "solvent_mixture_property_proxy_density_kg_m3",
    "solvent_mixture_property_proxy_dielectric_constant",
    "solvent_mixture_property_proxy_dielectric_constant_estimated",
    "solvent_mixture_property_proxy_dielectric_constant_best_available",
    "solvent_mixture_property_proxy_dielectric_constant_estimated_flag",
)
PROPERTY_PROXY_CORE_COLUMNS = PROPERTY_PROXY_COLUMNS[:3]
PROPERTY_PROXY_SOURCE_BACKED_COLUMNS = PROPERTY_PROXY_COLUMNS[:4]
PROPERTY_PROXY_ESTIMATED_COLUMNS = PROPERTY_PROXY_COLUMNS[:3] + (
    PROPERTY_PROXY_COLUMNS[5],
    PROPERTY_PROXY_COLUMNS[6],
)
PROPERTY_PROXY_INTERPRETATION_BLOCKS: Mapping[str, str] = {
    PROPERTY_PROXY_COLUMNS[0]: "solvent_property_proxy_viscosity",
    PROPERTY_PROXY_COLUMNS[1]: "solvent_property_proxy_surface_tension",
    PROPERTY_PROXY_COLUMNS[2]: "solvent_property_proxy_density",
    PROPERTY_PROXY_COLUMNS[3]: "solvent_property_proxy_dielectric_source_backed",
    PROPERTY_PROXY_COLUMNS[5]: "solvent_property_proxy_dielectric_best_available",
    PROPERTY_PROXY_COLUMNS[6]: "solvent_property_proxy_dielectric_estimation_indicator",
}
PROPERTY_PROXY_PANEL_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "context_plus_property_proxy_core": PROPERTY_PROXY_CORE_COLUMNS,
    "context_plus_property_proxy_4p_source_backed": PROPERTY_PROXY_SOURCE_BACKED_COLUMNS,
    "context_plus_property_proxy_4p_estimated": PROPERTY_PROXY_ESTIMATED_COLUMNS,
}
SOURCE_BACKED_4P_PANEL = "context_plus_property_proxy_4p_source_backed"
INK_AMOUNTS_REFERENCE_PANEL = "context_plus_ink_amounts"
SOURCE_BACKED_4P_REFERENCE_PANEL = (
    "context_plus_ink_amounts_on_4p_source_backed_complete_cases"
)
CORE_PROPERTY_PANEL = "context_plus_property_proxy_core"
MEDIUM_IPA_SURFACE_STATUS_COLUMN = (
    "solvent_mixture_property_proxy_surface_tension_mN_m__model_use_status"
)
MEDIUM_IPA_SURFACE_STATUS = "eligible_source_backed_medium_confidence"
CONTROLLED_IPA_SURFACE_STATUSES = frozenset(
    {
        "eligible_source_backed",
        MEDIUM_IPA_SURFACE_STATUS,
        "held",
    }
)
CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL = (
    "context_plus_property_proxy_core_excluding_medium_ipa_surface"
)
CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL = (
    "context_plus_ink_amounts_on_core_excluding_medium_ipa_surface"
)

PAIRED_PANEL_COMPARISONS = (
    (
        "context_plus_ink_minus_context",
        "context",
        "context_plus_ink",
        "delta_mae_context_plus_ink_minus_context",
    ),
    (
        "property_proxy_core_minus_context_plus_ink_amounts",
        INK_AMOUNTS_REFERENCE_PANEL,
        CORE_PROPERTY_PANEL,
        "delta_mae_context_plus_property_proxy_core_minus_context_plus_ink_amounts",
    ),
    (
        "property_proxy_core_minus_raw_solvent_composition_same_records",
        "context_plus_ink",
        CORE_PROPERTY_PANEL,
        "delta_mae_property_proxy_core_minus_context_plus_ink",
    ),
    (
        "property_proxy_4p_source_backed_minus_context_plus_ink_amounts_complete_cases",
        SOURCE_BACKED_4P_REFERENCE_PANEL,
        SOURCE_BACKED_4P_PANEL,
        "delta_mae_context_plus_property_proxy_4p_source_backed_minus_context_plus_ink_amounts_complete_cases",
    ),
    (
        "property_proxy_core_excluding_medium_ipa_surface_minus_context_plus_ink_amounts_same_records",
        CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL,
        CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL,
        "delta_mae_property_proxy_core_excluding_medium_ipa_surface_minus_context_plus_ink_amounts_same_records",
    ),
)


def _interpretation_blocks(
    panel_name: str, panel: ResolvedPanel
) -> dict[str, tuple[str, ...]]:
    """Return a post-hoc perturbation partition without changing model blocks.

    Gower fitting continues to average the registered property columns inside
    one joint ``solvent_property_proxy`` distance block.  For a property-panel
    interpretation pass only, each proxy is perturbed separately so the saved
    outer-test artifact can answer which proxy carries conditional predictive
    information in the presence of the others.  These singleton perturbations
    are not marginal, causal, or mechanistic property effects.
    """

    blocks = {
        str(block): tuple(str(column) for column in members)
        for block, members in panel.blocks.items()
    }
    property_members = blocks.get("solvent_property_proxy")
    if not property_members:
        return blocks

    unknown = sorted(
        set(property_members) - set(PROPERTY_PROXY_INTERPRETATION_BLOCKS)
    )
    if unknown:
        raise ValueError(
            f"Property interpretation partition for {panel_name!r} has "
            f"unregistered proxy columns: {unknown}"
        )
    del blocks["solvent_property_proxy"]
    for column in property_members:
        blocks[PROPERTY_PROXY_INTERPRETATION_BLOCKS[column]] = (column,)
    return blocks


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for a single target/scope nested-CV evaluation."""

    outer_folds: int = 5
    outer_repeats: int = 10
    inner_folds: int = 4
    random_seed: int = 42
    outer_seeds: tuple[int, ...] | None = None
    bootstrap_replicates: int = 2000
    panel_names: tuple[str, ...] = tuple(PANEL_FEATURES)
    model_names: tuple[str, ...] = DEFAULT_MODELS
    gower_neighbors: tuple[int, ...] = (3, 5, 7, 9, 11, 15, 21)
    gower_distance_powers: tuple[float, ...] = (1.0, 2.0)
    extra_trees_n_estimators: int = 300
    extra_trees_max_depths: tuple[int, ...] = (2, 3, 4)
    extra_trees_min_samples_leaf: tuple[int, ...] = (4, 6, 8)
    interpretation_enabled: bool = True
    importance_panel: str = CORE_PROPERTY_PANEL
    importance_permutation_repeats: int = 3
    require_complete_panels: bool = False
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("Outer and inner grouped CV require at least two folds")
        if self.outer_repeats < 1:
            raise ValueError("outer_repeats must be positive")
        if self.outer_seeds is not None and len(self.outer_seeds) != self.outer_repeats:
            raise ValueError("outer_seeds must contain exactly one seed per outer repeat")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates cannot be negative")
        if not self.gower_neighbors or any(value < 1 for value in self.gower_neighbors):
            raise ValueError("gower_neighbors must contain positive integers")
        if not self.gower_distance_powers or any(value <= 0 for value in self.gower_distance_powers):
            raise ValueError("gower_distance_powers must contain positive values")
        if self.extra_trees_n_estimators < 1:
            raise ValueError("extra_trees_n_estimators must be positive")
        if not self.extra_trees_max_depths or not self.extra_trees_min_samples_leaf:
            raise ValueError("Extra Trees depth and minimum-leaf grids cannot be empty")
        if self.importance_permutation_repeats < 1:
            raise ValueError("importance_permutation_repeats must be positive")


@dataclass
class EvaluationResult:
    split_manifest: pd.DataFrame
    inner_split_manifest: pd.DataFrame
    panel_schema: pd.DataFrame
    panel_eligibility: pd.DataFrame
    oof_predictions: pd.DataFrame
    oof_aggregated: pd.DataFrame
    split_metrics: pd.DataFrame
    metric_summary: pd.DataFrame
    inner_selection: pd.DataFrame
    paired_delta: pd.DataFrame
    paired_delta_bootstrap: pd.DataFrame
    model_selection: pd.DataFrame
    outer_test_block_effects: pd.DataFrame
    outer_test_block_importance: pd.DataFrame


def _validate_medium_ipa_surface_status(
    frame: pd.DataFrame,
    *,
    panel_names: Sequence[str],
    record_id_col: str,
) -> None:
    """Fail closed on the status metadata that triggers the IPA sensitivity."""

    if CORE_PROPERTY_PANEL not in panel_names:
        return
    if MEDIUM_IPA_SURFACE_STATUS_COLUMN not in frame:
        raise KeyError(
            f"Core property panel {CORE_PROPERTY_PANEL!r} requires controlled "
            f"surface-tension status column {MEDIUM_IPA_SURFACE_STATUS_COLUMN!r}; "
            "the medium-confidence IPA exclusion sensitivity cannot be audited without it"
        )

    statuses = frame[MEDIUM_IPA_SURFACE_STATUS_COLUMN]
    invalid = statuses.isna() | ~statuses.astype(str).isin(
        CONTROLLED_IPA_SURFACE_STATUSES
    )
    if invalid.any():
        invalid_statuses = sorted(
            {
                "<missing>" if pd.isna(value) else str(value)
                for value in statuses.loc[invalid]
            }
        )
        invalid_records = sorted(
            frame.loc[invalid, record_id_col].astype(str).tolist()
        )
        raise ValueError(
            f"Core property panel {CORE_PROPERTY_PANEL!r} requires controlled values in "
            f"{MEDIUM_IPA_SURFACE_STATUS_COLUMN!r}; invalid_statuses={invalid_statuses}, "
            f"invalid_record_ids={invalid_records}, "
            f"allowed_statuses={sorted(CONTROLLED_IPA_SURFACE_STATUSES)}"
        )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _is_shareable_relation(value: object) -> bool:
    if pd.isna(value):
        return False
    normalized = str(value).strip().lower()
    if not normalized:
        return False
    return normalized not in {"unknown", "unresolved", "missing", "none", "n/a", "na"} and not normalized.startswith(
        ("unknown_", "lab_unknown_", "unresolved_")
    )


def build_connected_group_ids(
    frame: pd.DataFrame,
    *,
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
    record_id_col: str = "record_id",
) -> pd.Series:
    """Build publication/lab/recipe/sample connected components.

    Relations are namespaced, so an accidental equal string in two different
    fields cannot connect records.  Unresolved identifiers do not connect
    records; curation should give each uncertain lab its own provenance-bearing
    identifier instead of pooling all unknown laboratories.
    """

    if record_id_col not in frame:
        raise KeyError(f"Missing record identifier column {record_id_col!r}")
    if frame[record_id_col].astype(str).duplicated().any():
        duplicates = frame.loc[frame[record_id_col].astype(str).duplicated(False), record_id_col].tolist()
        raise ValueError(f"record_id must be unique before grouping; duplicates={duplicates}")

    union_find = _UnionFind(len(frame))
    for column in relation_columns:
        if column not in frame:
            continue
        first_position: dict[str, int] = {}
        for position, value in enumerate(frame[column].tolist()):
            if not _is_shareable_relation(value):
                continue
            key = str(value).strip()
            if key in first_position:
                union_find.union(position, first_position[key])
            else:
                first_position[key] = position

    components: dict[int, list[int]] = {}
    for position in range(len(frame)):
        components.setdefault(union_find.find(position), []).append(position)

    group_by_position: dict[int, str] = {}
    record_ids = frame[record_id_col].astype(str).tolist()
    for positions in components.values():
        members = sorted(record_ids[position] for position in positions)
        digest = hashlib.sha256("\x1f".join(members).encode("utf-8")).hexdigest()[:16]
        group_id = f"cc_{digest}"
        for position in positions:
            group_by_position[position] = group_id
    return pd.Series(
        [group_by_position[position] for position in range(len(frame))],
        index=frame.index,
        name="cv_group_id",
        dtype="string",
    )


def _relation_values(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    return {str(value).strip() for value in frame[column] if _is_shareable_relation(value)}


def assert_no_relational_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_col: str = "cv_group_id",
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> None:
    """Assert that connected groups and their constituent relations are held out."""

    overlap = set(train[group_col].astype(str)).intersection(test[group_col].astype(str))
    if overlap:
        raise AssertionError(f"Connected CV groups occur in both train and test: {sorted(overlap)}")
    for column in relation_columns:
        shared = _relation_values(train, column).intersection(_relation_values(test, column))
        if shared:
            raise AssertionError(f"Relation {column!r} occurs in both train and test: {sorted(shared)}")


def make_outer_split_manifest(
    frame: pd.DataFrame,
    *,
    group_col: str = "cv_group_id",
    record_id_col: str = "record_id",
    folds: int = 5,
    repeats: int = 10,
    random_seed: int = 42,
    repeat_seeds: Sequence[int] | None = None,
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> pd.DataFrame:
    """Create a compact, persistent repeated shuffled grouped-fold manifest."""

    if group_col not in frame:
        raise KeyError(f"Missing connected group column {group_col!r}")
    n_groups = frame[group_col].nunique(dropna=False)
    if n_groups < folds:
        raise ValueError(f"Need at least {folds} connected groups, found {n_groups}")

    if repeat_seeds is not None and len(repeat_seeds) != repeats:
        raise ValueError("repeat_seeds must contain exactly one seed per requested repeat")
    seeds = tuple(int(value) for value in repeat_seeds) if repeat_seeds is not None else tuple(
        random_seed + repeat for repeat in range(repeats)
    )
    rows: list[dict[str, object]] = []
    groups = frame[group_col].astype(str).to_numpy()
    for repeat, split_seed in enumerate(seeds):
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=split_seed)
        fold_assignment = np.full(len(frame), -1, dtype=int)
        for fold, (train_positions, test_positions) in enumerate(splitter.split(frame, groups=groups)):
            assert_no_relational_leakage(
                frame.iloc[train_positions],
                frame.iloc[test_positions],
                group_col=group_col,
                relation_columns=relation_columns,
            )
            fold_assignment[test_positions] = fold
        if (fold_assignment < 0).any():
            raise AssertionError(f"Repeat {repeat} did not assign every record to an outer test fold")
        for row_position, fold in enumerate(fold_assignment):
            rows.append(
                {
                    "repeat": repeat,
                    "fold": int(fold),
                    "row_position": row_position,
                    "record_id": str(frame.iloc[row_position][record_id_col]),
                    "cv_group_id": groups[row_position],
                    "split_seed": split_seed,
                }
            )
    manifest = pd.DataFrame(rows).sort_values(["repeat", "fold", "row_position"]).reset_index(drop=True)
    validate_split_manifest(
        frame,
        manifest,
        group_col=group_col,
        record_id_col=record_id_col,
        expected_repeats=repeats,
        expected_folds=folds,
        relation_columns=relation_columns,
    )
    return manifest


def validate_split_manifest(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    group_col: str = "cv_group_id",
    record_id_col: str = "record_id",
    expected_repeats: int | None = None,
    expected_folds: int | None = None,
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> None:
    """Validate OOF coverage and all grouping relations in a split manifest."""

    required = {"repeat", "fold", "record_id", "cv_group_id"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Split manifest is missing required columns: {missing_columns}")
    data_ids = set(frame[record_id_col].astype(str))
    manifest_ids = set(manifest["record_id"].astype(str))
    if data_ids != manifest_ids:
        raise ValueError(
            "Split manifest record set differs from evaluation data; "
            f"missing={sorted(data_ids - manifest_ids)}, extra={sorted(manifest_ids - data_ids)}"
        )
    if (manifest.assign(_record_key=manifest["record_id"].astype(str)).groupby("_record_key")["cv_group_id"].nunique() > 1).any():
        raise ValueError("A record changes connected-group ID across manifest repeats")
    expected_groups = frame.assign(_record_key=frame[record_id_col].astype(str)).set_index(
        "_record_key"
    )[group_col].astype(str)
    manifest_groups = (
        manifest.assign(_record_key=manifest["record_id"].astype(str))
        .drop_duplicates("_record_key")
        .set_index("_record_key")["cv_group_id"]
        .astype(str)
    )
    mismatched_groups = sorted(
        record_id
        for record_id in data_ids
        if manifest_groups.get(record_id) != expected_groups.get(record_id)
    )
    if mismatched_groups:
        raise ValueError(
            "Split manifest connected-group IDs disagree with the canonical registry for records: "
            f"{mismatched_groups}"
        )
    repeats = sorted(manifest["repeat"].unique())
    if expected_repeats is not None and len(repeats) != expected_repeats:
        raise ValueError(f"Expected {expected_repeats} repeats, found {len(repeats)}")

    relation_lookup = frame.set_index(frame[record_id_col].astype(str), drop=False)
    for repeat in repeats:
        repeat_manifest = manifest.loc[manifest["repeat"] == repeat].copy()
        counts = repeat_manifest["record_id"].astype(str).value_counts()
        if set(counts.index) != data_ids or not (counts == 1).all():
            raise ValueError(f"Every record must occur exactly once in repeat {repeat}")
        if expected_folds is not None and repeat_manifest["fold"].nunique() != expected_folds:
            raise ValueError(
                f"Expected {expected_folds} folds in repeat {repeat}, found {repeat_manifest['fold'].nunique()}"
            )
        if (repeat_manifest.groupby("cv_group_id")["fold"].nunique() > 1).any():
            raise ValueError(f"Connected group split across outer folds in repeat {repeat}")
        available_relations = [column for column in relation_columns if column in relation_lookup]
        joined = repeat_manifest.assign(_record_key=repeat_manifest["record_id"].astype(str)).join(
            relation_lookup[available_relations].rename_axis("_record_key"), on="_record_key"
        )
        for column in relation_columns:
            if column not in joined:
                continue
            shareable = joined[column].map(_is_shareable_relation)
            if (joined.loc[shareable].groupby(column)["fold"].nunique() > 1).any():
                raise ValueError(f"Relation {column!r} split across outer folds in repeat {repeat}")


def _to_original(log10_values: np.ndarray | pd.Series) -> np.ndarray:
    return np.power(10.0, np.clip(np.asarray(log10_values, dtype=float), -12.0, 12.0))


def negative_original_mae(estimator: BaseEstimator, X: pd.DataFrame, y_log10: np.ndarray) -> float:
    """Grid-search score: negative MAE after returning slopes to mV dec-1."""

    prediction_log10 = np.asarray(estimator.predict(X), dtype=float)
    return -float(mean_absolute_error(_to_original(y_log10), _to_original(prediction_log10)))


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed_array) & np.isfinite(predicted_array)
    observed_array = observed_array[valid]
    predicted_array = predicted_array[valid]
    if not len(observed_array):
        return {"mae": np.nan, "rmse": np.nan, "median_ae": np.nan, "spearman": np.nan, "r2": np.nan}
    correlation = (
        spearmanr(observed_array, predicted_array).statistic
        if len(observed_array) > 1
        and np.ptp(observed_array) > 0
        and np.ptp(predicted_array) > 0
        else np.nan
    )
    return {
        "mae": float(mean_absolute_error(observed_array, predicted_array)),
        "rmse": float(np.sqrt(mean_squared_error(observed_array, predicted_array))),
        "median_ae": float(median_absolute_error(observed_array, predicted_array)),
        "spearman": float(correlation) if np.isfinite(correlation) else np.nan,
        "r2": float(r2_score(observed_array, predicted_array)) if len(observed_array) > 1 else np.nan,
    }


def _inner_splits(
    train: pd.DataFrame,
    *,
    group_col: str,
    inner_folds: int,
    seed: int,
    relation_columns: Sequence[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    groups = train[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    folds = min(inner_folds, len(unique_groups))
    if folds < 2:
        raise ValueError("An outer training fold has fewer than two connected groups for inner CV")
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    assignments = np.full(len(train), -1, dtype=int)
    for fold, (inner_train, inner_validation) in enumerate(splitter.split(train, groups=groups)):
        assert_no_relational_leakage(
            train.iloc[inner_train],
            train.iloc[inner_validation],
            group_col=group_col,
            relation_columns=relation_columns,
        )
        assignments[inner_validation] = fold
        splits.append((inner_train, inner_validation))
    if (assignments < 0).any():
        raise AssertionError("Inner grouped CV did not assign every outer-training record")
    return splits, assignments


def _models_for_panel(panel_name: str, requested: Sequence[str]) -> tuple[str, ...]:
    dummy_models = {"dummy_mean", "dummy_median"}
    if panel_name == "dummy":
        return tuple(model for model in requested if model in dummy_models)
    return tuple(model for model in requested if model not in dummy_models)


def _metric_summary(split_metrics: pd.DataFrame) -> pd.DataFrame:
    if split_metrics.empty:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in (
            "mae",
            "rmse",
            "median_ae",
            "mae_mV_dec",
            "rmse_mV_dec",
            "median_ae_mV_dec",
            "spearman",
            "r2",
            "nlpd_mV_dec",
            "crps_mV_dec",
            "nlpd_log10",
            "crps_log10",
            "coverage90",
            "coverage95",
            "interval90_width",
            "interval95_width",
        )
        if column in split_metrics
    ]
    grouped = split_metrics.groupby(["target", "scope", "panel", "model"], dropna=False)
    rows: list[dict[str, object]] = []
    for keys, group in grouped:
        row: dict[str, object] = dict(zip(("target", "scope", "panel", "model"), keys, strict=True))
        row["n_outer_splits"] = len(group)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_n"] = int(values.notna().sum())
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
            row[f"{metric}_se"] = (
                float(values.std(ddof=1) / np.sqrt(values.notna().sum())) if values.notna().sum() > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_repeated_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated OOF predictions once per record before inference."""

    if oof.empty:
        return pd.DataFrame()
    keys = ["target", "scope", "panel", "model", "record_id", "cv_group_id"]
    aggregation: dict[str, str] = {
        "observed_mV_dec": "first",
        "predicted_mV_dec": "mean",
        "observed_log10": "first",
        "predicted_log10": "mean",
        "predictive_sd_log10": "mean",
        "interval90_low_mV_dec": "mean",
        "interval90_high_mV_dec": "mean",
        "interval95_low_mV_dec": "mean",
        "interval95_high_mV_dec": "mean",
    }
    available = {column: function for column, function in aggregation.items() if column in oof}
    result = oof.groupby(keys, dropna=False, as_index=False).agg(available)
    result["absolute_error_mV_dec"] = np.abs(result["observed_mV_dec"] - result["predicted_mV_dec"])
    counts = oof.groupby(keys, dropna=False).size().rename("n_repeated_oof").reset_index()
    return result.merge(counts, on=keys, how="left")


def paired_panel_deltas(split_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute registered nested-selected paired panel comparisons.

    Fixed-family rows remain in ``fold_metrics.csv`` as sensitivities, but the
    canonical deltas use the family selected solely inside each outer training
    fold.  The strict four-property and medium-confidence surface-tension
    exclusion comparisons use dedicated ``context_plus_ink_amounts``
    companions, so both estimators see exactly the same records.  The separate
    raw-composition comparison uses ``context_plus_ink`` on the same folds.
    """

    keys = ["target", "scope", "model", "repeat", "fold"]
    primary = split_metrics
    if "nested_selected" in set(split_metrics.get("model", pd.Series(dtype=str))):
        primary = split_metrics.loc[split_metrics["model"] == "nested_selected"]
    rows: list[pd.DataFrame] = []
    available_panels = set(primary.get("panel", pd.Series(dtype=str)).astype(str))
    for comparison, reference_panel, augmented_panel, delta_column in PAIRED_PANEL_COMPARISONS:
        if reference_panel not in available_panels or augmented_panel not in available_panels:
            continue
        reference = primary.loc[
            primary["panel"] == reference_panel, keys + ["mae", "n_train", "n_test"]
        ].rename(
            columns={
                "mae": "reference_mae",
                "n_train": "reference_n_train",
                "n_test": "reference_n_test",
            }
        )
        augmented = primary.loc[
            primary["panel"] == augmented_panel, keys + ["mae", "n_train", "n_test"]
        ].rename(
            columns={
                "mae": "augmented_mae",
                "n_train": "augmented_n_train",
                "n_test": "augmented_n_test",
            }
        )
        paired = reference.merge(augmented, on=keys, validate="one_to_one")
        if paired.empty:
            continue
        if not (
            paired["reference_n_train"].eq(paired["augmented_n_train"]).all()
            and paired["reference_n_test"].eq(paired["augmented_n_test"]).all()
        ):
            raise AssertionError(
                f"Paired comparison {comparison!r} does not use identical train/test row counts"
            )
        paired.insert(2, "comparison", comparison)
        paired.insert(3, "reference_panel", reference_panel)
        paired.insert(4, "augmented_panel", augmented_panel)
        paired["delta_mae"] = paired["augmented_mae"] - paired["reference_mae"]
        paired[delta_column] = paired["delta_mae"]
        paired["augmented_improves_this_split"] = paired["delta_mae"] < 0
        if comparison == "context_plus_ink_minus_context":
            paired["context_mae"] = paired["reference_mae"]
            paired["context_plus_ink_mae"] = paired["augmented_mae"]
            paired["ink_improves_this_split"] = paired["augmented_improves_this_split"]
        rows.append(paired)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def paired_delta_group_bootstrap(
    aggregated_oof: pd.DataFrame,
    *,
    replicates: int = 2000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Cluster-bootstrap paired panel error differences by connected group."""

    if replicates <= 0 or aggregated_oof.empty:
        return pd.DataFrame()
    primary = aggregated_oof
    if "nested_selected" in set(aggregated_oof.get("model", pd.Series(dtype=str))):
        primary = aggregated_oof.loc[aggregated_oof["model"] == "nested_selected"]
    keys = ["target", "scope", "model", "record_id", "cv_group_id", "observed_mV_dec"]
    rows: list[dict[str, object]] = []
    available_panels = set(primary.get("panel", pd.Series(dtype=str)).astype(str))
    for comparison, reference_panel, augmented_panel, _ in PAIRED_PANEL_COMPARISONS:
        if reference_panel not in available_panels or augmented_panel not in available_panels:
            continue
        comparison_offset = int.from_bytes(
            hashlib.sha256(comparison.encode("utf-8")).digest()[:4], "big"
        )
        rng = np.random.default_rng(random_seed + comparison_offset)
        reference = primary.loc[
            primary["panel"] == reference_panel, keys + ["predicted_mV_dec"]
        ].rename(columns={"predicted_mV_dec": "reference_prediction"})
        augmented = primary.loc[
            primary["panel"] == augmented_panel, keys + ["predicted_mV_dec"]
        ].rename(columns={"predicted_mV_dec": "augmented_prediction"})
        paired = reference.merge(augmented, on=keys, validate="one_to_one")
        for (target, scope, model), group in paired.groupby(["target", "scope", "model"], dropna=False):
            connected_groups = group["cv_group_id"].astype(str).unique()
            if not len(connected_groups):
                continue
            observed = group["observed_mV_dec"].to_numpy(float)
            delta_errors = np.abs(group["augmented_prediction"].to_numpy(float) - observed) - np.abs(
                group["reference_prediction"].to_numpy(float) - observed
            )
            group_to_positions = {
                connected_group: np.flatnonzero(group["cv_group_id"].astype(str).to_numpy() == connected_group)
                for connected_group in connected_groups
            }
            draws = np.empty(replicates, dtype=float)
            for replicate in range(replicates):
                sampled = rng.choice(connected_groups, size=len(connected_groups), replace=True)
                positions = np.concatenate([group_to_positions[value] for value in sampled])
                draws[replicate] = float(np.mean(delta_errors[positions]))
            rows.append(
                {
                    "comparison": comparison,
                    "reference_panel": reference_panel,
                    "augmented_panel": augmented_panel,
                    "target": target,
                    "scope": scope,
                    "model": model,
                    "n_records": len(group),
                    "n_connected_groups": len(connected_groups),
                    "bootstrap_replicates": replicates,
                    "delta_mae_estimate": float(np.mean(delta_errors)),
                    "delta_mae_ci_low": float(np.quantile(draws, 0.025)),
                    "delta_mae_ci_high": float(np.quantile(draws, 0.975)),
                    "probability_delta_below_zero": float(np.mean(draws < 0)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_nested_grouped_cv(
    data: pd.DataFrame,
    *,
    target_col: str,
    target_name: str | None = None,
    scope: str = "alkaline",
    config: EvaluationConfig | None = None,
    split_manifest: pd.DataFrame | None = None,
    record_id_col: str = "record_id",
    group_col: str = "cv_group_id",
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
    output_dir: str | Path | None = None,
) -> EvaluationResult:
    """Run the fixed model/panel comparison under strict grouped nested CV.

    The response is fitted as ``log10(slope)``.  Selection and reported error
    metrics are evaluated after transforming predictions back to mV dec-1.
    """

    settings = config or EvaluationConfig()
    if target_col not in data:
        raise KeyError(f"Missing target column {target_col!r}")
    if record_id_col not in data:
        raise KeyError(f"Missing record identifier {record_id_col!r}")

    frame = data.copy().reset_index(drop=True)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.loc[np.isfinite(frame[target_col]) & (frame[target_col] > 0)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No finite positive values are available for target {target_col!r}")
    if frame[record_id_col].astype(str).duplicated().any():
        raise ValueError("Nested CV requires exactly one target row per record_id")
    if group_col not in frame:
        frame[group_col] = build_connected_group_ids(
            frame, relation_columns=relation_columns, record_id_col=record_id_col
        )
    frame["_target_log10"] = np.log10(frame[target_col].to_numpy(float))
    effective_target_name = target_name or target_col
    _validate_medium_ipa_surface_status(
        frame,
        panel_names=settings.panel_names,
        record_id_col=record_id_col,
    )

    if split_manifest is None:
        manifest = make_outer_split_manifest(
            frame,
            group_col=group_col,
            record_id_col=record_id_col,
            folds=settings.outer_folds,
            repeats=settings.outer_repeats,
            random_seed=settings.random_seed,
            repeat_seeds=settings.outer_seeds,
            relation_columns=relation_columns,
        )
    else:
        manifest = split_manifest.copy()
        validate_split_manifest(
            frame,
            manifest,
            group_col=group_col,
            record_id_col=record_id_col,
            expected_repeats=settings.outer_repeats,
            expected_folds=settings.outer_folds,
            relation_columns=relation_columns,
        )

    schema = panel_schema_frame(frame, settings.panel_names)
    panel_resolutions: dict[str, ResolvedPanel] = {}
    panel_eligible_record_ids: dict[str, set[str]] = {}
    panel_eligibility_policy: dict[str, str] = {}
    skipped_panels: dict[str, str] = {}
    all_record_ids = set(frame[record_id_col].astype(str))
    for panel_name in settings.panel_names:
        resolved = resolve_feature_panel(
            frame, panel_name, require_all=settings.require_complete_panels
        )
        required_proxy_columns = PROPERTY_PROXY_PANEL_COLUMNS.get(panel_name)
        eligible_ids = all_record_ids
        eligibility_policy = "all_target_eligible_records"
        if required_proxy_columns is not None:
            missing_proxy_columns = [
                column for column in required_proxy_columns if column not in frame
            ]
            if missing_proxy_columns:
                skipped_panels[panel_name] = "no_authoritative_solvent_mixture_property_proxy_columns"
                continue
            numeric_proxy = frame.loc[:, list(required_proxy_columns)].apply(
                pd.to_numeric, errors="coerce"
            )
            complete_mask = np.isfinite(numeric_proxy.to_numpy(float)).all(axis=1)
            complete_ids = set(frame.loc[complete_mask, record_id_col].astype(str))
            if not complete_ids:
                skipped_panels[panel_name] = (
                    "no_complete_property_proxy_records_for="
                    + ",".join(required_proxy_columns)
                )
                continue
            if panel_name == SOURCE_BACKED_4P_PANEL:
                eligible_ids = complete_ids
                eligibility_policy = "complete_cases_all_four_source_backed_properties"
            elif len(complete_ids) != len(frame):
                skipped_panels[panel_name] = (
                    f"registered_complete_property_panel_has_{len(frame) - len(complete_ids)}_"
                    "incomplete_records"
                )
                continue
        panel_resolutions[panel_name] = resolved
        panel_eligible_record_ids[panel_name] = eligible_ids
        panel_eligibility_policy[panel_name] = eligibility_policy

    # The strict four-property sensitivity requires a reference estimator fit
    # on exactly the same complete cases.  Its reference includes every
    # non-solvent ink-amount term already present in the property panel.  Keep
    # the output label distinct from the full-data reference so no downstream
    # aggregation can pair unlike training or test populations by accident.
    if SOURCE_BACKED_4P_PANEL in panel_resolutions:
        strict_reference = resolve_feature_panel(
            frame,
            INK_AMOUNTS_REFERENCE_PANEL,
            require_all=settings.require_complete_panels,
        )
        panel_resolutions[SOURCE_BACKED_4P_REFERENCE_PANEL] = ResolvedPanel(
            name=SOURCE_BACKED_4P_REFERENCE_PANEL,
            columns=strict_reference.columns,
            logical_names=strict_reference.logical_names,
            numeric_columns=strict_reference.numeric_columns,
            categorical_columns=strict_reference.categorical_columns,
            blocks=strict_reference.blocks,
            missing_logical_features=strict_reference.missing_logical_features,
        )
        panel_eligible_record_ids[SOURCE_BACKED_4P_REFERENCE_PANEL] = set(
            panel_eligible_record_ids[SOURCE_BACKED_4P_PANEL]
        )
        panel_eligibility_policy[SOURCE_BACKED_4P_REFERENCE_PANEL] = (
            "same_complete_cases_as_context_plus_property_proxy_4p_source_backed"
        )
        companion_schema = panel_schema_frame(
            frame, (INK_AMOUNTS_REFERENCE_PANEL,)
        ).copy()
        companion_schema["panel"] = SOURCE_BACKED_4P_REFERENCE_PANEL
        schema = pd.concat([schema, companion_schema], ignore_index=True)

    # Two registered IPA/water compositions use unusually wide source
    # brackets for surface tension.  When the core property panel is requested,
    # automatically run its pre-registered exclusion sensitivity.  Both the
    # property estimator and its context reference are restricted to exactly
    # the same records, while retaining the original outer-fold assignment.
    medium_ipa_surface_record_ids: set[str] = set()
    medium_ipa_surface_sensitivity_panels = {
        CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL,
        CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL,
    }
    if CORE_PROPERTY_PANEL in panel_resolutions:
        medium_mask = frame[MEDIUM_IPA_SURFACE_STATUS_COLUMN].astype(str).eq(
            MEDIUM_IPA_SURFACE_STATUS
        )
        medium_ipa_surface_record_ids = set(
            frame.loc[medium_mask, record_id_col].astype(str)
        )
        if medium_ipa_surface_record_ids:
            sensitivity_ids = set(panel_eligible_record_ids[CORE_PROPERTY_PANEL]).difference(
                medium_ipa_surface_record_ids
            )
            core_panel = panel_resolutions[CORE_PROPERTY_PANEL]
            reference_panel = resolve_feature_panel(
                frame,
                INK_AMOUNTS_REFERENCE_PANEL,
                require_all=settings.require_complete_panels,
            )
            panel_resolutions[CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL] = ResolvedPanel(
                name=CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL,
                columns=core_panel.columns,
                logical_names=core_panel.logical_names,
                numeric_columns=core_panel.numeric_columns,
                categorical_columns=core_panel.categorical_columns,
                blocks=core_panel.blocks,
                missing_logical_features=core_panel.missing_logical_features,
            )
            panel_resolutions[CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL] = ResolvedPanel(
                name=CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL,
                columns=reference_panel.columns,
                logical_names=reference_panel.logical_names,
                numeric_columns=reference_panel.numeric_columns,
                categorical_columns=reference_panel.categorical_columns,
                blocks=reference_panel.blocks,
                missing_logical_features=reference_panel.missing_logical_features,
            )
            for panel_name in medium_ipa_surface_sensitivity_panels:
                panel_eligible_record_ids[panel_name] = set(sensitivity_ids)
            panel_eligibility_policy[CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL] = (
                "exclude_medium_confidence_ipa_surface_tension_brackets"
            )
            panel_eligibility_policy[CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL] = (
                "same_records_as_core_excluding_medium_confidence_ipa_surface_tension_brackets"
            )
            core_schema = schema.loc[schema["panel"].eq(CORE_PROPERTY_PANEL)].copy()
            core_schema["panel"] = CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL
            reference_schema = panel_schema_frame(
                frame, (INK_AMOUNTS_REFERENCE_PANEL,)
            ).copy()
            reference_schema["panel"] = (
                CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL
            )
            schema = pd.concat(
                [schema, core_schema, reference_schema], ignore_index=True
            )

    schema.insert(0, "target", effective_target_name)
    schema.insert(1, "scope", scope)
    schema["panel_evaluation_status"] = schema["panel"].map(
        lambda panel_name: f"skipped:{skipped_panels[panel_name]}" if panel_name in skipped_panels else "evaluated"
    )
    schema["eligibility_policy"] = schema["panel"].map(panel_eligibility_policy).fillna(
        "not_evaluated"
    )
    schema["n_eligible_records"] = schema["panel"].map(
        {name: len(values) for name, values in panel_eligible_record_ids.items()}
    ).fillna(0).astype(int)
    schema["n_ineligible_records"] = len(frame) - schema["n_eligible_records"]
    eligibility_rows: list[dict[str, object]] = []
    for panel_name in dict.fromkeys(schema["panel"].astype(str)):
        eligible_ids = panel_eligible_record_ids.get(panel_name, set())
        policy = panel_eligibility_policy.get(panel_name, "not_evaluated")
        skipped_reason = skipped_panels.get(panel_name, "")
        for record_id in frame[record_id_col].astype(str):
            eligible = record_id in eligible_ids
            eligibility_rows.append(
                {
                    "target": effective_target_name,
                    "scope": scope,
                    "panel": panel_name,
                    "record_id": record_id,
                    "eligible": eligible,
                    "eligibility_policy": policy,
                    "ineligibility_reason": (
                        ""
                        if eligible
                        else "medium_confidence_ipa_surface_tension_bracket_excluded"
                        if panel_name in medium_ipa_surface_sensitivity_panels
                        and record_id in medium_ipa_surface_record_ids
                        else f"panel_skipped:{skipped_reason}"
                        if skipped_reason
                        else "missing_one_or_more_required_property_proxy_values"
                    ),
                }
            )
    panel_eligibility = pd.DataFrame(eligibility_rows)

    id_to_position = {str(value): position for position, value in enumerate(frame[record_id_col])}
    oof_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    inner_selection_rows: list[dict[str, object]] = []
    inner_manifest_rows: list[dict[str, object]] = []
    model_selection_rows: list[dict[str, object]] = []
    block_effect_rows: list[dict[str, object]] = []
    block_importance_rows: list[dict[str, object]] = []

    for repeat in sorted(manifest["repeat"].unique()):
        repeat_manifest = manifest.loc[manifest["repeat"] == repeat]
        for fold in sorted(repeat_manifest["fold"].unique()):
            test_ids = set(repeat_manifest.loc[repeat_manifest["fold"] == fold, "record_id"].astype(str))
            test_positions = np.asarray(sorted(id_to_position[value] for value in test_ids), dtype=int)
            train_positions = np.asarray(
                [position for position in range(len(frame)) if position not in set(test_positions)], dtype=int
            )
            outer_train = frame.iloc[train_positions].reset_index(drop=True)
            outer_test = frame.iloc[test_positions].reset_index(drop=True)
            assert_no_relational_leakage(
                outer_train, outer_test, group_col=group_col, relation_columns=relation_columns
            )

            for panel_name, panel in panel_resolutions.items():
                eligible_ids = panel_eligible_record_ids[panel_name]
                train = outer_train.loc[
                    outer_train[record_id_col].astype(str).isin(eligible_ids)
                ].reset_index(drop=True)
                test = outer_test.loc[
                    outer_test[record_id_col].astype(str).isin(eligible_ids)
                ].reset_index(drop=True)
                # Some folds can contain no complete cases for a sensitivity
                # panel.  Such folds have no valid test metric; every eligible
                # record is nevertheless predicted in its assigned fold.
                if test.empty:
                    continue
                if train[group_col].nunique() < 2:
                    raise ValueError(
                        f"Panel {panel_name!r} has fewer than two eligible connected groups "
                        f"in outer train repeat={repeat}, fold={fold}"
                    )
                assert_no_relational_leakage(
                    train, test, group_col=group_col, relation_columns=relation_columns
                )
                inner_seed = settings.random_seed + repeat * 1000 + int(fold)
                inner_cv, inner_assignments = _inner_splits(
                    train,
                    group_col=group_col,
                    inner_folds=settings.inner_folds,
                    seed=inner_seed,
                    relation_columns=relation_columns,
                )
                for inner_position, inner_fold in enumerate(inner_assignments):
                    inner_manifest_rows.append(
                        {
                            "target": effective_target_name,
                            "scope": scope,
                            "panel": panel_name,
                            "eligibility_policy": panel_eligibility_policy[panel_name],
                            "outer_repeat": repeat,
                            "outer_fold": fold,
                            "inner_fold": int(inner_fold),
                            "record_id": str(train.iloc[inner_position][record_id_col]),
                            "cv_group_id": str(train.iloc[inner_position][group_col]),
                            "inner_seed": inner_seed,
                        }
                    )

                y_train_log = train["_target_log10"].to_numpy(float)
                y_test_log = test["_target_log10"].to_numpy(float)
                y_test = test[target_col].to_numpy(float)
                model_names = _models_for_panel(panel_name, settings.model_names)
                if not model_names:
                    continue
                if panel_name == "dummy":
                    X_train = pd.DataFrame({"__constant__": np.ones(len(train))})
                    X_test = pd.DataFrame({"__constant__": np.ones(len(test))})
                else:
                    X_train = train.loc[:, list(panel.columns)]
                    X_test = test.loc[:, list(panel.columns)]

                fold_model_candidates: list[dict[str, object]] = []
                for model_name in model_names:
                    model_seed = inner_seed + sum(map(ord, model_name))
                    model_spec = make_model_spec(
                        model_name,
                        numeric_columns=("__constant__",) if panel_name == "dummy" else panel.numeric_columns,
                        categorical_columns=() if panel_name == "dummy" else panel.categorical_columns,
                        blocks=panel.blocks if panel_name != "dummy" else {"dummy": ("__constant__",)},
                        random_seed=model_seed,
                        gower_neighbors=settings.gower_neighbors,
                        gower_distance_powers=settings.gower_distance_powers,
                        extra_trees_n_estimators=settings.extra_trees_n_estimators,
                        extra_trees_max_depths=settings.extra_trees_max_depths,
                        extra_trees_min_samples_leaf=settings.extra_trees_min_samples_leaf,
                    )
                    search = GridSearchCV(
                        model_spec.estimator,
                        param_grid=dict(model_spec.param_grid) if model_spec.param_grid else [{}],
                        scoring=negative_original_mae,
                        cv=inner_cv,
                        refit=True,
                        n_jobs=settings.n_jobs,
                        error_score="raise",
                        return_train_score=False,
                    )
                    search.fit(X_train, y_train_log, groups=train[group_col].astype(str).to_numpy())
                    predicted_log, predictive_sd = predict_log_with_uncertainty(search.best_estimator_, X_test)
                    predicted = _to_original(predicted_log)

                    probabilistic: dict[str, np.ndarray] = {}
                    if predictive_sd is not None:
                        nlpd, crps = gaussian_log_scores(y_test_log, predicted_log, predictive_sd)
                        nlpd_original, crps_original = lognormal_original_scale_scores(
                            y_test, predicted_log, predictive_sd
                        )
                        probabilistic = {
                            "predictive_sd_log10": predictive_sd,
                            "nlpd_mV_dec": nlpd_original,
                            "crps_mV_dec": crps_original,
                            "nlpd_log10": nlpd,
                            "crps_log10": crps,
                            "interval90_low_mV_dec": _to_original(predicted_log - 1.6448536269514722 * predictive_sd),
                            "interval90_high_mV_dec": _to_original(predicted_log + 1.6448536269514722 * predictive_sd),
                            "interval95_low_mV_dec": _to_original(predicted_log - 1.959963984540054 * predictive_sd),
                            "interval95_high_mV_dec": _to_original(predicted_log + 1.959963984540054 * predictive_sd),
                        }

                    for local_index, row in test.iterrows():
                        output: dict[str, object] = {
                            "target": effective_target_name,
                            "scope": scope,
                            "panel": panel_name,
                            "eligibility_policy": panel_eligibility_policy[panel_name],
                            "model": model_name,
                            "model_role": "fixed_family_sensitivity",
                            "selected_model_family": model_name,
                            "repeat": repeat,
                            "fold": fold,
                            "record_id": str(row[record_id_col]),
                            "cv_group_id": str(row[group_col]),
                            "observed_mV_dec": y_test[local_index],
                            "predicted_mV_dec": predicted[local_index],
                            "observed_log10": y_test_log[local_index],
                            "predicted_log10": predicted_log[local_index],
                            "absolute_error_mV_dec": abs(y_test[local_index] - predicted[local_index]),
                            "selected_params": json.dumps(search.best_params_, sort_keys=True),
                            "model_seed": model_seed,
                        }
                        for name, values in probabilistic.items():
                            output[name] = values[local_index]
                        oof_rows.append(output)

                    metrics = regression_metrics(y_test, predicted)
                    metric_output: dict[str, object] = {
                        "target": effective_target_name,
                        "scope": scope,
                        "panel": panel_name,
                        "eligibility_policy": panel_eligibility_policy[panel_name],
                        "model": model_name,
                        "model_role": "fixed_family_sensitivity",
                        "selected_model_family": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_train_groups": train[group_col].nunique(),
                        "n_test_groups": test[group_col].nunique(),
                        "selected_params": json.dumps(search.best_params_, sort_keys=True),
                        "inner_best_negative_mae": float(search.best_score_),
                        "model_seed": model_seed,
                        **metrics,
                        "mae_mV_dec": metrics["mae"],
                        "rmse_mV_dec": metrics["rmse"],
                        "median_ae_mV_dec": metrics["median_ae"],
                    }
                    if predictive_sd is not None:
                        metric_output.update(
                            {
                                "nlpd_log10": float(np.mean(probabilistic["nlpd_log10"])),
                                "crps_log10": float(np.mean(probabilistic["crps_log10"])),
                                "nlpd_mV_dec": float(np.mean(probabilistic["nlpd_mV_dec"])),
                                "crps_mV_dec": float(np.mean(probabilistic["crps_mV_dec"])),
                                "coverage90": float(
                                    np.mean(
                                        (y_test >= probabilistic["interval90_low_mV_dec"])
                                        & (y_test <= probabilistic["interval90_high_mV_dec"])
                                    )
                                ),
                                "coverage95": float(
                                    np.mean(
                                        (y_test >= probabilistic["interval95_low_mV_dec"])
                                        & (y_test <= probabilistic["interval95_high_mV_dec"])
                                    )
                                ),
                                "interval90_width": float(
                                    np.mean(
                                        probabilistic["interval90_high_mV_dec"]
                                        - probabilistic["interval90_low_mV_dec"]
                                    )
                                ),
                                "interval95_width": float(
                                    np.mean(
                                        probabilistic["interval95_high_mV_dec"]
                                        - probabilistic["interval95_low_mV_dec"]
                                    )
                                ),
                            }
                        )
                    metric_rows.append(metric_output)

                    # These perturbations are deliberately post hoc: the
                    # estimator has already been tuned/fitted using only the
                    # outer training fold, and the results never flow back to
                    # model or hyperparameter selection.  Joint block
                    # permutation preserves relationships inside each mutually
                    # exclusive block (including solvent-composition sums).
                    if settings.interpretation_enabled and panel_name == settings.importance_panel:
                        effects, importance = outer_test_block_perturbations(
                            search.best_estimator_,
                            X_test,
                            y_test,
                            predicted_log,
                            blocks=_interpretation_blocks(panel_name, panel),
                            record_ids=test[record_id_col].astype(str).to_numpy(),
                            cv_group_ids=test[group_col].astype(str).to_numpy(),
                            target=effective_target_name,
                            scope=scope,
                            panel=panel_name,
                            model=model_name,
                            repeat=int(repeat),
                            fold=int(fold),
                            selected_params=json.dumps(search.best_params_, sort_keys=True),
                            permutation_repeats=settings.importance_permutation_repeats,
                            random_seed=model_seed + 100_000,
                        )
                        block_effect_rows.extend(effects.to_dict("records"))
                        block_importance_rows.extend(importance.to_dict("records"))

                    cv_results = pd.DataFrame(search.cv_results_)
                    best_candidate = cv_results.iloc[int(search.best_index_)]
                    split_score_columns = [
                        column
                        for column in cv_results.columns
                        if column.startswith("split") and column.endswith("_test_score")
                    ]
                    best_inner_mae_by_fold = -pd.to_numeric(
                        best_candidate[split_score_columns], errors="coerce"
                    ).to_numpy(float)
                    valid_inner = best_inner_mae_by_fold[np.isfinite(best_inner_mae_by_fold)]
                    best_inner_se = (
                        float(np.std(valid_inner, ddof=1) / np.sqrt(len(valid_inner)))
                        if len(valid_inner) > 1
                        else 0.0
                    )
                    fold_model_candidates.append(
                        {
                            "model": model_name,
                            "complexity_rank": model_spec.complexity_rank,
                            "inner_mae": -float(search.best_score_),
                            "inner_mae_se": best_inner_se,
                            "model_seed": model_seed,
                            "oof_start": len(oof_rows) - len(test),
                            "oof_end": len(oof_rows),
                            "metric_index": len(metric_rows) - 1,
                        }
                    )
                    for candidate_index, candidate in cv_results.iterrows():
                        inner_selection_rows.append(
                            {
                                "target": effective_target_name,
                                "scope": scope,
                                "panel": panel_name,
                                "eligibility_policy": panel_eligibility_policy[panel_name],
                                "model": model_name,
                                "outer_repeat": repeat,
                                "outer_fold": fold,
                                "candidate_index": int(candidate_index),
                                "params": json.dumps(candidate["params"], sort_keys=True),
                                "mean_inner_negative_mae": float(candidate["mean_test_score"]),
                                "sd_inner_negative_mae": float(candidate["std_test_score"]),
                                "rank_inner": int(candidate["rank_test_score"]),
                                "selected": int(candidate["rank_test_score"]) == 1,
                                "model_seed": model_seed,
                            }
                        )

                # Model-family selection is itself inside the outer training
                # fold.  The outer test fold is still untouched here.  Apply a
                # one-standard-error rule to the inner grouped scores and copy
                # the corresponding already-fitted prediction into the
                # canonical ``nested_selected`` stream.
                best_family = min(fold_model_candidates, key=lambda candidate: float(candidate["inner_mae"]))
                family_threshold = float(best_family["inner_mae"]) + float(best_family["inner_mae_se"])
                eligible_families = [
                    candidate
                    for candidate in fold_model_candidates
                    if float(candidate["inner_mae"]) <= family_threshold
                ]
                selected_family = min(
                    eligible_families,
                    key=lambda candidate: (
                        int(candidate["complexity_rank"]),
                        float(candidate["inner_mae"]),
                        str(candidate["model"]),
                    ),
                )
                selected_family_name = str(selected_family["model"])
                for original in oof_rows[
                    int(selected_family["oof_start"]) : int(selected_family["oof_end"])
                ]:
                    nested_row = dict(original)
                    nested_row["model"] = "nested_selected"
                    nested_row["model_role"] = "nested_selected_primary"
                    nested_row["selected_model_family"] = selected_family_name
                    oof_rows.append(nested_row)
                nested_metric = dict(metric_rows[int(selected_family["metric_index"])])
                nested_metric["model"] = "nested_selected"
                nested_metric["model_role"] = "nested_selected_primary"
                nested_metric["selected_model_family"] = selected_family_name
                metric_rows.append(nested_metric)
                for candidate in fold_model_candidates:
                    model_selection_rows.append(
                        {
                            "target": effective_target_name,
                            "scope": scope,
                            "panel": panel_name,
                            "eligibility_policy": panel_eligibility_policy[panel_name],
                            "outer_repeat": repeat,
                            "outer_fold": fold,
                            "candidate_model_family": candidate["model"],
                            "inner_mae": candidate["inner_mae"],
                            "inner_mae_se": candidate["inner_mae_se"],
                            "model_seed": candidate["model_seed"],
                            "best_family": candidate["model"] == best_family["model"],
                            "one_se_threshold": family_threshold,
                            "eligible_within_one_se": candidate in eligible_families,
                            "selected_model_family": selected_family_name,
                            "selected": candidate["model"] == selected_family_name,
                            "selection_data": "outer_training_fold_inner_grouped_cv_only",
                        }
                    )

    oof = pd.DataFrame(oof_rows)
    split_metrics = pd.DataFrame(metric_rows)
    inner_selection = pd.DataFrame(inner_selection_rows)
    inner_manifest = pd.DataFrame(inner_manifest_rows)
    aggregated = aggregate_repeated_oof(oof)
    summary = _metric_summary(split_metrics)
    selection = pd.DataFrame(model_selection_rows)
    block_effects = pd.DataFrame(block_effect_rows, columns=BLOCK_EFFECT_COLUMNS)
    block_importance = pd.DataFrame(block_importance_rows, columns=BLOCK_IMPORTANCE_COLUMNS)
    paired = paired_panel_deltas(split_metrics)
    paired_bootstrap = paired_delta_group_bootstrap(
        aggregated,
        replicates=settings.bootstrap_replicates,
        random_seed=settings.random_seed,
    )

    result = EvaluationResult(
        split_manifest=manifest,
        inner_split_manifest=inner_manifest,
        panel_schema=schema,
        panel_eligibility=panel_eligibility,
        oof_predictions=oof,
        oof_aggregated=aggregated,
        split_metrics=split_metrics,
        metric_summary=summary,
        inner_selection=inner_selection,
        paired_delta=paired,
        paired_delta_bootstrap=paired_bootstrap,
        model_selection=selection,
        outer_test_block_effects=block_effects,
        outer_test_block_importance=block_importance,
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        artifacts: Mapping[str, pd.DataFrame] = {
            "split_manifest.csv": result.split_manifest,
            "inner_split_manifest.csv": result.inner_split_manifest,
            "feature_panel_schema.csv": result.panel_schema,
            "panel_eligibility.csv": result.panel_eligibility,
            "oof_predictions.csv": result.oof_predictions,
            "oof_aggregated.csv": result.oof_aggregated,
            "split_metrics.csv": result.split_metrics,
            "fold_metrics.csv": result.split_metrics,
            "metric_summary.csv": result.metric_summary,
            "inner_selection.csv": result.inner_selection,
            "paired_delta_mae.csv": result.paired_delta,
            "paired_delta_group_bootstrap.csv": result.paired_delta_bootstrap,
            "model_selection_one_se.csv": result.model_selection,
            "inner_model_family_selection.csv": result.model_selection,
            "outer_test_block_effects.csv": result.outer_test_block_effects,
            "outer_test_block_importance.csv": result.outer_test_block_importance,
        }
        for filename, artifact in artifacts.items():
            artifact.to_csv(destination / filename, index=False)
        (destination / "evaluation_config.json").write_text(
            json.dumps(asdict(settings), indent=2, sort_keys=True), encoding="utf-8"
        )
    return result


def _read_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _filter_electrolyte_scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    normalized_scope = scope.strip().lower()
    if normalized_scope in {"all", "all_electrolyte", "all_electrolytes", "all-electrolyte"}:
        return frame.copy()
    if normalized_scope != "alkaline":
        raise ValueError(f"Unsupported electrolyte scope {scope!r}; use 'alkaline' or 'all'")

    if "is_alkaline" in frame:
        values = frame["is_alkaline"]
        if pd.api.types.is_bool_dtype(values):
            mask = values.fillna(False)
        else:
            mask = values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    elif "electrolyte_scope" in frame:
        mask = frame["electrolyte_scope"].astype(str).str.strip().str.lower().eq("alkaline")
    elif "electrolyte_class" in frame:
        mask = frame["electrolyte_class"].astype(str).str.strip().str.lower().eq("alkaline")
    elif "electrolyte_identity" in frame:
        identity = frame["electrolyte_identity"].astype(str).str.strip().str.upper()
        mask = identity.isin({"KOH", "NAOH"})
    else:
        raise KeyError(
            "Alkaline scope requires is_alkaline, electrolyte_scope, electrolyte_class, "
            "or electrolyte_identity"
        )
    return frame.loc[mask].copy()


def run_nested_evaluation(
    records: pd.DataFrame | str | Path,
    targets: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    target_col: str = "b_bayesian_local_line_mV_dec",
    target_name: str | None = None,
    scope: str = "alkaline",
    config: EvaluationConfig | None = None,
    smoke: bool = False,
    record_id_col: str = "record_id",
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> EvaluationResult:
    """Load/merge canonical registries and run one reproducible evaluation.

    This is the integration entry point used by the command-line workflow.  A
    single split manifest is created before any panel/model loop and is reused
    unchanged for every comparison.  ``smoke=True`` changes only computational
    replication, not grouping or leakage rules.
    """

    record_frame = _read_frame(records)
    target_frame = _read_frame(targets)
    if record_id_col not in record_frame or record_id_col not in target_frame:
        raise KeyError(f"Both records and targets must contain {record_id_col!r}")
    if target_col not in target_frame:
        # Also accept the long target-registry representation.
        required_long = {"target_name", "target_value_mV_dec"}
        if required_long.issubset(target_frame.columns):
            selected_name = target_name or target_col
            target_frame = target_frame.loc[target_frame["target_name"].astype(str) == selected_name].copy()
            target_frame[target_col] = target_frame["target_value_mV_dec"]
        else:
            raise KeyError(f"Target registry does not contain {target_col!r}")
    if record_frame[record_id_col].astype(str).duplicated().any():
        raise ValueError("Canonical records must have one row per record_id")
    selected_targets = target_frame.loc[:, [record_id_col, target_col]].copy()
    selected_targets = selected_targets.loc[pd.to_numeric(selected_targets[target_col], errors="coerce").notna()]
    if selected_targets[record_id_col].astype(str).duplicated().any():
        raise ValueError(f"Target {target_col!r} has more than one row per record_id")

    # Compute connected components on the complete registry before scope/target
    # filtering, preserving transitive links through records absent from this
    # particular analysis.
    record_frame["cv_group_id"] = build_connected_group_ids(
        record_frame, relation_columns=relation_columns, record_id_col=record_id_col
    )
    merged = record_frame.merge(
        selected_targets,
        on=record_id_col,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_target_registry"),
    )
    if f"{target_col}_target_registry" in merged:
        if target_col in record_frame:
            # A target must never be silently sourced from the predictor table.
            merged = merged.drop(columns=[target_col])
        merged = merged.rename(columns={f"{target_col}_target_registry": target_col})
    scoped = _filter_electrolyte_scope(merged, scope).reset_index(drop=True)

    settings = config or EvaluationConfig()
    if smoke:
        available_groups = scoped["cv_group_id"].nunique()
        smoke_outer = min(3, available_groups)
        if smoke_outer < 2:
            raise ValueError("Smoke evaluation still requires at least two connected groups")
        settings = replace(
            settings,
            outer_folds=smoke_outer,
            outer_repeats=1,
            outer_seeds=None,
            inner_folds=2,
            bootstrap_replicates=min(settings.bootstrap_replicates, 100),
            gower_neighbors=tuple(value for value in settings.gower_neighbors if value <= 7) or (3,),
            gower_distance_powers=(1.0,),
            extra_trees_n_estimators=min(settings.extra_trees_n_estimators, 100),
            extra_trees_max_depths=(settings.extra_trees_max_depths[0],),
            extra_trees_min_samples_leaf=(settings.extra_trees_min_samples_leaf[0],),
            interpretation_enabled=False,
        )

    destination = Path(output_dir)
    result = evaluate_nested_grouped_cv(
        scoped,
        target_col=target_col,
        target_name=target_name or target_col,
        scope=scope,
        config=settings,
        record_id_col=record_id_col,
        group_col="cv_group_id",
        relation_columns=relation_columns,
        output_dir=destination,
    )

    coverage = (
        result.oof_predictions.groupby(["target", "scope", "panel", "model", "record_id"], dropna=False)
        .size()
        .rename("count")
    )
    coverage_ok = bool((coverage == settings.outer_repeats).all()) if len(coverage) else False
    property_proxy_coverage = {
        column: (
            int(pd.to_numeric(scoped[column], errors="coerce").notna().sum())
            if column in scoped
            else 0
        )
        for column in PROPERTY_PROXY_COLUMNS
    }
    property_values_available = sum(property_proxy_coverage.values())
    property_values_possible = len(scoped) * len(PROPERTY_PROXY_COLUMNS)
    requested_property_panels = [
        panel for panel in settings.panel_names if panel in PROPERTY_PROXY_PANEL_COLUMNS
    ]
    evaluated_panels = sorted(result.oof_predictions["panel"].unique().tolist())
    skipped_panels = sorted(
        result.panel_schema.loc[
            result.panel_schema["panel_evaluation_status"].str.startswith("skipped:"), "panel"
        ].unique().tolist()
    )
    property_proxy_panel_statuses = {
        panel: (
            "evaluated"
            if panel in evaluated_panels
            else "skipped"
            if panel in skipped_panels
            else "not_requested"
        )
        for panel in PROPERTY_PROXY_PANEL_COLUMNS
    }
    property_proxy_panel_status = (
        "not_requested"
        if not requested_property_panels
        else "evaluated"
        if any(panel in evaluated_panels for panel in requested_property_panels)
        else "skipped"
    )
    property_panel_complete_case_records = {}
    for panel, columns in PROPERTY_PROXY_PANEL_COLUMNS.items():
        if any(column not in scoped for column in columns):
            property_panel_complete_case_records[panel] = 0
            continue
        values = scoped.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        property_panel_complete_case_records[panel] = int(
            np.isfinite(values.to_numpy(float)).all(axis=1).sum()
        )
    assignment_columns = ["repeat", "fold", "record_id", "cv_group_id"]
    inner_columns = [
        "outer_repeat",
        "outer_fold",
        "inner_fold",
        "record_id",
        "cv_group_id",
    ]

    def paired_assignment_flags(
        reference_panel: str, augmented_panel: str
    ) -> tuple[bool, bool, bool]:
        if not {reference_panel, augmented_panel}.issubset(evaluated_panels):
            return False, False, False
        reference_oof = result.oof_predictions.loc[
            result.oof_predictions["panel"].eq(reference_panel)
            & result.oof_predictions["model"].eq("nested_selected")
        ]
        augmented_oof = result.oof_predictions.loc[
            result.oof_predictions["panel"].eq(augmented_panel)
            & result.oof_predictions["model"].eq("nested_selected")
        ]
        same_records = set(reference_oof["record_id"].astype(str)) == set(
            augmented_oof["record_id"].astype(str)
        )
        same_outer = set(
            map(tuple, reference_oof[assignment_columns].astype(str).to_numpy())
        ) == set(
            map(tuple, augmented_oof[assignment_columns].astype(str).to_numpy())
        )
        reference_inner = result.inner_split_manifest.loc[
            result.inner_split_manifest["panel"].eq(reference_panel), inner_columns
        ]
        augmented_inner = result.inner_split_manifest.loc[
            result.inner_split_manifest["panel"].eq(augmented_panel), inner_columns
        ]
        same_inner = set(map(tuple, reference_inner.astype(str).to_numpy())) == set(
            map(tuple, augmented_inner.astype(str).to_numpy())
        )
        return same_records, same_outer, same_inner

    core_pair_required = CORE_PROPERTY_PANEL in evaluated_panels
    core_pair_flags = (
        paired_assignment_flags(INK_AMOUNTS_REFERENCE_PANEL, CORE_PROPERTY_PANEL)
        if core_pair_required
        else (True, True, True)
    )
    core_pair_valid = all(core_pair_flags)
    property_representation_pair_flags = (
        paired_assignment_flags("context_plus_ink", CORE_PROPERTY_PANEL)
        if core_pair_required
        else (True, True, True)
    )
    property_representation_pair_valid = all(property_representation_pair_flags)

    strict_pair_flags = (
        paired_assignment_flags(
            SOURCE_BACKED_4P_REFERENCE_PANEL, SOURCE_BACKED_4P_PANEL
        )
        if SOURCE_BACKED_4P_PANEL in evaluated_panels
        else (True, True, True)
    )
    (
        strict_pair_same_records,
        strict_pair_same_outer_assignments,
        strict_pair_same_inner_assignments,
    ) = strict_pair_flags
    strict_pair_valid = all(strict_pair_flags)

    medium_surface_status_available = MEDIUM_IPA_SURFACE_STATUS_COLUMN in scoped
    medium_surface_record_ids = (
        set(
            scoped.loc[
                scoped[MEDIUM_IPA_SURFACE_STATUS_COLUMN].astype(str).eq(
                    MEDIUM_IPA_SURFACE_STATUS
                ),
                record_id_col,
            ].astype(str)
        )
        if medium_surface_status_available
        else set()
    )
    medium_surface_sensitivity_required = bool(
        CORE_PROPERTY_PANEL in settings.panel_names and medium_surface_record_ids
    )
    medium_surface_sensitivity_evaluated = {
        CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL,
        CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL,
    }.issubset(evaluated_panels)
    medium_surface_pair_same_records = not medium_surface_sensitivity_required
    medium_surface_pair_same_outer_assignments = not medium_surface_sensitivity_required
    medium_surface_pair_same_inner_assignments = not medium_surface_sensitivity_required
    if medium_surface_sensitivity_evaluated:
        property_oof = result.oof_predictions.loc[
            result.oof_predictions["panel"].eq(CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL)
            & result.oof_predictions["model"].eq("nested_selected")
        ]
        reference_oof = result.oof_predictions.loc[
            result.oof_predictions["panel"].eq(
                CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL
            )
            & result.oof_predictions["model"].eq("nested_selected")
        ]
        property_records = set(property_oof["record_id"].astype(str))
        reference_records = set(reference_oof["record_id"].astype(str))
        expected_records = set(scoped[record_id_col].astype(str)).difference(
            medium_surface_record_ids
        )
        medium_surface_pair_same_records = (
            property_records == reference_records == expected_records
        )
        _, medium_surface_pair_same_outer_assignments, medium_surface_pair_same_inner_assignments = (
            paired_assignment_flags(
                CORE_EXCLUDING_MEDIUM_IPA_SURFACE_REFERENCE_PANEL,
                CORE_EXCLUDING_MEDIUM_IPA_SURFACE_PANEL,
            )
        )
    medium_surface_pair_valid = (
        (not medium_surface_sensitivity_required)
        or (
            medium_surface_sensitivity_evaluated
            and medium_surface_pair_same_records
            and medium_surface_pair_same_outer_assignments
            and medium_surface_pair_same_inner_assignments
        )
    )
    qa = {
        "status": (
            "pass"
            if coverage_ok
            and core_pair_valid
            and property_representation_pair_valid
            and strict_pair_valid
            and medium_surface_pair_valid
            else "fail"
        ),
        "target": target_name or target_col,
        "scope": scope,
        "n_records_with_target": int(len(scoped)),
        "n_connected_groups": int(scoped["cv_group_id"].nunique()),
        "outer_folds": settings.outer_folds,
        "outer_repeats": settings.outer_repeats,
        "inner_folds": settings.inner_folds,
        "same_outer_manifest_reused_across_panels_and_models": True,
        "evaluated_panels": evaluated_panels,
        "skipped_panels": skipped_panels,
        "property_proxy_coverage": property_proxy_coverage,
        "property_proxy_panel_partial": bool(
            0 < property_values_available < property_values_possible
        ),
        "property_proxy_panel_status": property_proxy_panel_status,
        "property_proxy_panel_statuses": property_proxy_panel_statuses,
        "property_panel_complete_case_records": property_panel_complete_case_records,
        "core_property_incremental_reference_panel": INK_AMOUNTS_REFERENCE_PANEL,
        "core_property_incremental_pair_uses_identical_records": core_pair_flags[0],
        "core_property_incremental_pair_uses_identical_outer_assignments": core_pair_flags[1],
        "core_property_incremental_pair_uses_identical_inner_assignments": core_pair_flags[2],
        "property_representation_raw_composition_reference_panel": "context_plus_ink",
        "property_representation_pair_uses_identical_records": property_representation_pair_flags[0],
        "property_representation_pair_uses_identical_outer_assignments": property_representation_pair_flags[1],
        "property_representation_pair_uses_identical_inner_assignments": property_representation_pair_flags[2],
        "source_backed_4p_ink_amounts_reference_panel": SOURCE_BACKED_4P_REFERENCE_PANEL,
        "source_backed_4p_pair_uses_identical_records": strict_pair_same_records,
        "source_backed_4p_pair_uses_identical_outer_assignments": strict_pair_same_outer_assignments,
        "source_backed_4p_pair_uses_identical_inner_assignments": strict_pair_same_inner_assignments,
        "medium_confidence_ipa_surface_exclusion_required": medium_surface_sensitivity_required,
        "medium_confidence_ipa_surface_excluded_records": len(medium_surface_record_ids),
        "medium_confidence_ipa_surface_excluded_record_ids": sorted(
            medium_surface_record_ids
        ),
        "medium_confidence_ipa_surface_exclusion_status": (
            "evaluated"
            if medium_surface_sensitivity_evaluated
            else "required_but_not_evaluated"
            if medium_surface_sensitivity_required
            else "not_required"
        ),
        "medium_confidence_ipa_surface_pair_uses_identical_records": medium_surface_pair_same_records,
        "medium_confidence_ipa_surface_pair_uses_identical_outer_assignments": medium_surface_pair_same_outer_assignments,
        "medium_confidence_ipa_surface_pair_uses_identical_inner_assignments": medium_surface_pair_same_inner_assignments,
        "oof_predictions_per_record_and_model_panel": settings.outer_repeats,
        "oof_coverage_complete": coverage_ok,
        "relational_group_leakage_checks": "passed",
        "delta_convention": (
            "augmented-panel MAE minus its named paired reference MAE; property-increment "
            "comparisons hold non-solvent ink amounts fixed with context_plus_ink_amounts, "
            "the separate representation comparison contrasts property proxies against raw "
            "solvent composition in context_plus_ink, strict four-property results use "
            "complete cases, and the surface-tension exclusion uses the same retained "
            "records for both estimators; negative favors the augmented property panel"
        ),
        "response_transform": "model log10(slope); error metrics in mV dec^-1",
        "outer_test_interpretation_status": (
            "not_evaluated_smoke_mode"
            if smoke
            else (
                "available_outer_test_posthoc"
                if not result.outer_test_block_importance.empty
                else "not_available"
            )
        ),
        "outer_test_importance_used_for_selection": False,
        "importance_blocks_mutually_exclusive": True,
        "property_proxy_gower_distance_block": "joint_solvent_property_proxy",
        "property_proxy_outer_test_perturbation_partition": (
            "singleton_conditional_prediction_sensitivity"
        ),
        "outer_test_importance_blocks": sorted(
            map(
                str,
                result.outer_test_block_importance.get(
                    "block", pd.Series(dtype=str)
                ).astype(str).unique(),
            )
        ),
        "target_like_columns_present_but_hard_excluded": sorted(
            column for column in scoped.columns if is_target_like_column(column)
        ),
        "target_like_predictors_selected": sorted(
            column
            for column in result.panel_schema["selected_column"].dropna().astype(str).unique()
            if is_target_like_column(column)
        ),
    }
    if qa["target_like_predictors_selected"]:
        qa["status"] = "fail"
    destination.mkdir(parents=True, exist_ok=True)
    qa_path = destination / "evaluation_qa.json"
    qa_path.write_text(
        json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8"
    )
    if qa["status"] != "pass":
        failed_checks: list[str] = []
        if not coverage_ok:
            failed_checks.append("oof_coverage_complete")
        if not core_pair_valid:
            failed_checks.append("core_property_incremental_pair")
        if not property_representation_pair_valid:
            failed_checks.append("property_representation_pair")
        if not strict_pair_valid:
            failed_checks.append("source_backed_4p_pair")
        if not medium_surface_pair_valid:
            failed_checks.append("medium_confidence_ipa_surface_pair")
        if qa["target_like_predictors_selected"]:
            failed_checks.append("target_like_predictors_selected")
        raise RuntimeError(
            "Evaluation QA failed; workflow release is blocked. "
            f"Failed checks: {', '.join(failed_checks) or 'unspecified'}. "
            f"Audit artifact: {qa_path}"
        )
    return result
