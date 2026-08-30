"""Soft-label-aware connected-group cross-validation splits.

The splitters in this module balance posterior class probability mass directly.
They deliberately do not collapse a posterior target to an argmax class before
stratification.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .evaluate import (
    DEFAULT_RELATION_COLUMNS,
    assert_no_relational_leakage,
    validate_split_manifest,
)


def _validate_soft_labels(
    frame: pd.DataFrame,
    soft_label_columns: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    columns = tuple(str(column) for column in soft_label_columns)
    if not columns:
        raise ValueError("soft_label_columns must contain at least one column")
    if len(set(columns)) != len(columns):
        raise ValueError("soft_label_columns must not contain duplicates")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing soft-label columns: {missing}")
    if any(frame.columns.tolist().count(column) > 1 for column in columns):
        raise ValueError("Selected soft-label column names must be unique in the frame")

    try:
        labels = frame.loc[:, columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Soft labels must be numeric") from exc
    if not len(labels):
        raise ValueError("Cannot construct cross-validation folds for an empty frame")
    if not np.isfinite(labels).all():
        raise ValueError("Soft labels must all be finite")
    if (labels < 0.0).any():
        raise ValueError("Soft labels must all be nonnegative")
    row_sums = labels.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-8, atol=1e-8):
        bad_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, rtol=1e-8, atol=1e-8))
        raise ValueError(
            "Soft labels must sum to 1 for every record; "
            f"invalid row positions={bad_rows[:10].tolist()}"
        )
    return columns, labels


def _validated_groups(
    frame: pd.DataFrame,
    *,
    group_col: str,
) -> np.ndarray:
    if group_col not in frame.columns:
        raise KeyError(f"Missing connected group column {group_col!r}")
    if frame[group_col].isna().any():
        raise ValueError(f"Connected group column {group_col!r} contains missing values")
    groups = frame[group_col].astype(str).str.strip().to_numpy()
    if np.any(groups == ""):
        raise ValueError(f"Connected group column {group_col!r} contains blank values")
    return groups


def _validate_fold_request(folds: int, *, n_groups: int) -> int:
    if isinstance(folds, bool) or int(folds) != folds or int(folds) < 2:
        raise ValueError("folds must be an integer of at least 2")
    folds = int(folds)
    if n_groups < folds:
        raise ValueError(f"Need at least {folds} connected groups, found {n_groups}")
    return folds


def _greedy_soft_group_assignment(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Assign whole groups while minimizing class-mass and size imbalance."""

    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("split seeds must be nonnegative integers")
    seed = int(seed)

    group_positions: dict[str, list[int]] = {}
    for position, group in enumerate(groups):
        group_positions.setdefault(str(group), []).append(position)
    group_names = np.asarray(sorted(group_positions), dtype=object)
    group_sizes = np.asarray(
        [len(group_positions[str(group)]) for group in group_names],
        dtype=float,
    )
    group_mass = np.vstack(
        [
            labels[np.asarray(group_positions[str(group)], dtype=int)].sum(axis=0)
            for group in group_names
        ]
    )

    rng = np.random.default_rng(seed)
    # Large connected components are placed first. Random tie ordering makes
    # repeated manifests genuinely repeatable-but-distinct when group sizes
    # permit, without deriving any hard class from the posterior target.
    random_ties = rng.random(len(group_names))
    group_order = np.lexsort((group_names.astype(str), random_ties, -group_sizes))
    fold_tie_order = rng.permutation(folds)
    fold_tie_rank = np.empty(folds, dtype=int)
    fold_tie_rank[fold_tie_order] = np.arange(folds)

    total_mass = labels.sum(axis=0)
    target_mass = total_mass / float(folds)
    target_size = len(labels) / float(folds)
    mass_scale = np.where(target_mass > 1e-12, target_mass, 1.0)
    size_scale = max(target_size, 1.0)

    fold_mass = np.zeros((folds, labels.shape[1]), dtype=float)
    fold_sizes = np.zeros(folds, dtype=float)
    group_to_fold: dict[str, int] = {}

    for order_position, group_index in enumerate(group_order):
        if order_position < folds:
            # Seeding one group per fold guarantees nonempty validation folds.
            selected_fold = int(fold_tie_order[order_position])
        else:
            candidate_scores: list[tuple[float, int, int]] = []
            for fold in range(folds):
                candidate_mass = fold_mass.copy()
                candidate_sizes = fold_sizes.copy()
                candidate_mass[fold] += group_mass[group_index]
                candidate_sizes[fold] += group_sizes[group_index]
                class_imbalance = float(
                    np.mean(np.square((candidate_mass - target_mass) / mass_scale))
                )
                size_imbalance = float(
                    np.mean(np.square((candidate_sizes - target_size) / size_scale))
                )
                score = class_imbalance + 0.25 * size_imbalance
                candidate_scores.append((score, int(fold_tie_rank[fold]), fold))
            selected_fold = min(candidate_scores)[2]

        group_name = str(group_names[group_index])
        group_to_fold[group_name] = selected_fold
        fold_mass[selected_fold] += group_mass[group_index]
        fold_sizes[selected_fold] += group_sizes[group_index]

    assignments = np.asarray([group_to_fold[str(group)] for group in groups], dtype=int)
    if set(assignments.tolist()) != set(range(folds)):
        raise AssertionError("Soft grouped allocation produced an empty fold")
    return assignments


def make_soft_stratified_group_manifest(
    frame: pd.DataFrame,
    soft_label_columns: Sequence[str],
    *,
    group_col: str = "cv_group_id",
    record_id_col: str = "record_id",
    folds: int = 5,
    repeats: int = 10,
    random_seed: int = 42,
    repeat_seeds: Sequence[int] | None = None,
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> pd.DataFrame:
    """Create repeated soft-stratified connected-group outer folds.

    Each record occurs once per repeat as an outer-test observation. The
    returned schema intentionally matches :func:`validate_split_manifest`.
    """

    _, labels = _validate_soft_labels(frame, soft_label_columns)
    groups = _validated_groups(frame, group_col=group_col)
    if record_id_col not in frame.columns:
        raise KeyError(f"Missing record identifier column {record_id_col!r}")
    record_ids = frame[record_id_col].astype(str)
    if record_ids.duplicated().any():
        duplicates = sorted(record_ids.loc[record_ids.duplicated(False)].unique())
        raise ValueError(f"record_id must be unique; duplicates={duplicates}")

    folds = _validate_fold_request(folds, n_groups=len(set(groups)))
    if isinstance(repeats, bool) or int(repeats) != repeats or int(repeats) < 1:
        raise ValueError("repeats must be a positive integer")
    repeats = int(repeats)
    if repeat_seeds is not None and len(repeat_seeds) != repeats:
        raise ValueError("repeat_seeds must contain exactly one seed per requested repeat")
    seeds = (
        tuple(int(value) for value in repeat_seeds)
        if repeat_seeds is not None
        else tuple(int(random_seed) + repeat for repeat in range(repeats))
    )

    rows: list[dict[str, object]] = []
    for repeat, split_seed in enumerate(seeds):
        assignments = _greedy_soft_group_assignment(
            labels,
            groups,
            folds=folds,
            seed=split_seed,
        )
        for fold in range(folds):
            test_positions = np.flatnonzero(assignments == fold)
            train_positions = np.flatnonzero(assignments != fold)
            if not len(test_positions):
                raise AssertionError(f"Repeat {repeat} produced empty outer fold {fold}")
            assert_no_relational_leakage(
                frame.iloc[train_positions],
                frame.iloc[test_positions],
                group_col=group_col,
                relation_columns=relation_columns,
            )
        for row_position, fold in enumerate(assignments):
            rows.append(
                {
                    "repeat": repeat,
                    "fold": int(fold),
                    "row_position": row_position,
                    "record_id": str(record_ids.iloc[row_position]),
                    "cv_group_id": str(groups[row_position]),
                    "split_seed": int(split_seed),
                }
            )

    manifest = (
        pd.DataFrame(rows)
        .sort_values(["repeat", "fold", "row_position"], kind="stable")
        .reset_index(drop=True)
    )
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


def make_inner_soft_group_splits(
    train: pd.DataFrame,
    soft_label_columns: Sequence[str],
    *,
    group_col: str = "cv_group_id",
    folds: int = 4,
    random_seed: int = 42,
    relation_columns: Sequence[str] = DEFAULT_RELATION_COLUMNS,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Create soft-stratified connected-group inner CV positional splits."""

    _, labels = _validate_soft_labels(train, soft_label_columns)
    groups = _validated_groups(train, group_col=group_col)
    folds = _validate_fold_request(folds, n_groups=len(set(groups)))
    assignments = _greedy_soft_group_assignment(
        labels,
        groups,
        folds=folds,
        seed=random_seed,
    )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(folds):
        validation_positions = np.flatnonzero(assignments == fold)
        train_positions = np.flatnonzero(assignments != fold)
        if not len(validation_positions):
            raise AssertionError(f"Soft grouped allocation produced empty inner fold {fold}")
        assert_no_relational_leakage(
            train.iloc[train_positions],
            train.iloc[validation_positions],
            group_col=group_col,
            relation_columns=relation_columns,
        )
        splits.append((train_positions, validation_positions))
    return splits, assignments
