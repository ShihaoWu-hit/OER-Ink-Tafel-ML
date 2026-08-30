from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.cardinal_cv import (
    make_inner_soft_group_splits,
    make_soft_stratified_group_manifest,
)


SOFT_COLUMNS = ("p_low_30_40", "p_60", "p_120", "p_off")


def _soft_group_frame() -> pd.DataFrame:
    patterns = np.asarray(
        [
            [0.60, 0.20, 0.10, 0.10],
            [0.20, 0.55, 0.15, 0.10],
            [0.10, 0.20, 0.60, 0.10],
            [0.10, 0.15, 0.20, 0.55],
        ]
    )
    rows: list[dict[str, object]] = []
    for group_index in range(12):
        for member in range(2):
            pattern = patterns[group_index % len(patterns)].copy()
            # Preserve soft labels while preventing a hidden hard-label
            # stratifier from appearing to solve the same test accidentally.
            delta = 0.01 if member else -0.01
            pattern[0] += delta
            pattern[1] -= delta
            rows.append(
                {
                    "record_id": f"r_{group_index:02d}_{member}",
                    "cv_group_id": f"g_{group_index:02d}",
                    "publication_id": f"pub_{group_index:02d}",
                    "lab_group_id": f"lab_{group_index:02d}",
                    "recipe_hash": f"recipe_{group_index:02d}",
                    "sample_id": f"sample_{group_index:02d}_{member}",
                    **dict(zip(SOFT_COLUMNS, pattern, strict=True)),
                }
            )
    return pd.DataFrame(rows)


def test_outer_soft_manifest_is_deterministic_isolated_and_complete() -> None:
    frame = _soft_group_frame()
    first = make_soft_stratified_group_manifest(
        frame,
        SOFT_COLUMNS,
        folds=3,
        repeats=4,
        random_seed=17,
    )
    second = make_soft_stratified_group_manifest(
        frame,
        SOFT_COLUMNS,
        folds=3,
        repeats=4,
        random_seed=17,
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [
        "repeat",
        "fold",
        "row_position",
        "record_id",
        "cv_group_id",
        "split_seed",
    ]
    assert len(first) == len(frame) * 4
    assert (first.groupby(["repeat", "record_id"]).size() == 1).all()
    assert (first.groupby(["repeat", "cv_group_id"])["fold"].nunique() == 1).all()
    assert (first.groupby(["repeat", "fold"]).size() > 0).all()
    assert first.groupby("repeat")["fold"].nunique().eq(3).all()


def test_outer_soft_manifest_balances_probability_mass_without_hard_labels() -> None:
    frame = _soft_group_frame()
    manifest = make_soft_stratified_group_manifest(
        frame,
        SOFT_COLUMNS,
        folds=3,
        repeats=3,
        random_seed=29,
    )
    labels = frame.set_index("record_id")[list(SOFT_COLUMNS)]
    expected = labels.sum(axis=0) / 3.0

    for repeat in range(3):
        joined = manifest.loc[manifest["repeat"].eq(repeat)].join(
            labels,
            on="record_id",
        )
        observed = joined.groupby("fold")[list(SOFT_COLUMNS)].sum()
        relative_deviation = (observed - expected).abs().div(expected)
        assert float(relative_deviation.to_numpy().max()) <= 0.35 + 1e-12
        fold_sizes = joined.groupby("fold").size()
        assert int(fold_sizes.max() - fold_sizes.min()) <= 2


def test_inner_soft_splits_are_deterministic_cover_rows_and_isolate_groups() -> None:
    frame = _soft_group_frame()
    first_splits, first_assignments = make_inner_soft_group_splits(
        frame,
        SOFT_COLUMNS,
        folds=4,
        random_seed=101,
    )
    second_splits, second_assignments = make_inner_soft_group_splits(
        frame,
        SOFT_COLUMNS,
        folds=4,
        random_seed=101,
    )

    np.testing.assert_array_equal(first_assignments, second_assignments)
    assert len(first_splits) == len(second_splits) == 4
    validation_positions = np.concatenate([validation for _, validation in first_splits])
    np.testing.assert_array_equal(np.sort(validation_positions), np.arange(len(frame)))
    for (train_positions, validation_positions), (
        second_train,
        second_validation,
    ) in zip(first_splits, second_splits, strict=True):
        np.testing.assert_array_equal(train_positions, second_train)
        np.testing.assert_array_equal(validation_positions, second_validation)
        train_groups = set(frame.iloc[train_positions]["cv_group_id"])
        validation_groups = set(frame.iloc[validation_positions]["cv_group_id"])
        assert train_groups.isdisjoint(validation_groups)


@pytest.mark.parametrize(
    "replacement, message",
    [
        ((0.2, 0.2, 0.2, 0.1), "sum to 1"),
        ((-0.1, 0.3, 0.3, 0.5), "nonnegative"),
        ((np.nan, 0.2, 0.2, 0.6), "finite"),
    ],
)
def test_soft_label_validation_rejects_invalid_posteriors(
    replacement: tuple[float, ...],
    message: str,
) -> None:
    frame = _soft_group_frame()
    frame.loc[0, list(SOFT_COLUMNS)] = replacement
    with pytest.raises(ValueError, match=message):
        make_soft_stratified_group_manifest(
            frame,
            SOFT_COLUMNS,
            folds=3,
            repeats=1,
        )
