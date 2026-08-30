"""Posterior-aware labels for proximity to classical Tafel-slope values.

The functions in this module convert each record's posterior slope draws into
soft probabilities for the classical 30, 40, 60, and 120 mV dec-1 values.
They do not infer a reaction mechanism.  The primary operational definition is
an inclusive ``+/- relative_half_width`` band around each classical value,
with draws outside every band assigned to ``off_cardinal``.  Where bands
overlap, the closest anchor on the log-slope scale wins.

This module is deliberately side-effect free: it returns data frames and never
writes analysis artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_ANCHORS: tuple[float, ...] = (30.0, 40.0, 60.0, 120.0)
HIERARCHICAL_CLASS_LABELS: tuple[str, ...] = (
    "low_30_40_like",
    "60_like",
    "120_like",
    "off_cardinal",
)
HIERARCHICAL_SOFT_LABEL_COLUMNS: tuple[str, ...] = tuple(
    f"q_{label}" for label in HIERARCHICAL_CLASS_LABELS
)


def _validated_anchors(anchors: Sequence[float]) -> np.ndarray:
    values = np.asarray(tuple(anchors), dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("anchors must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("anchors must contain only finite positive values")
    if (np.diff(values) <= 0).any():
        raise ValueError("anchors must be unique and strictly increasing")
    return values


def _anchor_token(anchor: float) -> str:
    if float(anchor).is_integer():
        return str(int(anchor))
    return np.format_float_positional(float(anchor), trim="-").replace(".", "p")


def _label_names(anchors: np.ndarray) -> list[str]:
    return [f"{_anchor_token(anchor)}_like" for anchor in anchors] + [
        "off_cardinal"
    ]


def _probability_columns(anchors: np.ndarray) -> list[str]:
    return [f"q_{label}" for label in _label_names(anchors)]


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    terms = np.zeros_like(probabilities, dtype=float)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -terms.sum(axis=1)


def validate_soft_labels(
    labels: pd.DataFrame,
    anchors: Sequence[float] = DEFAULT_ANCHORS,
    *,
    sample_col: str = "sample_record",
    atol: float = 1e-10,
) -> None:
    """Raise if a posterior soft-label table violates its probability contract."""

    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels must be a pandas DataFrame")
    anchor_values = _validated_anchors(anchors)
    probability_columns = _probability_columns(anchor_values)
    required = {
        *probability_columns,
        "n_draws",
        "modal_label",
        "max_probability",
        "entropy",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Soft-label table is missing required columns: {missing}")
    if labels.empty:
        raise ValueError("Soft-label table must contain at least one record")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    if sample_col in labels.columns:
        if labels[sample_col].isna().any():
            raise ValueError(f"{sample_col!r} contains missing values")
        if labels[sample_col].duplicated().any():
            raise ValueError(f"Soft labels are not unique by {sample_col!r}")

    try:
        probabilities = labels[probability_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Soft-label probabilities must be numeric") from error
    if not np.isfinite(probabilities).all():
        raise ValueError("Soft-label probabilities must be finite")
    if (probabilities < -atol).any() or (probabilities > 1.0 + atol).any():
        raise ValueError("Soft-label probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=atol, rtol=0.0):
        raise ValueError("Soft-label probabilities must sum to one per record")

    draw_counts = pd.to_numeric(labels["n_draws"], errors="raise").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(draw_counts).all()
        or (draw_counts <= 0).any()
        or not np.allclose(draw_counts, np.round(draw_counts), atol=atol, rtol=0.0)
    ):
        raise ValueError("n_draws must contain positive integer counts")
    if "n_nonpositive_draws" in labels.columns:
        nonpositive = pd.to_numeric(
            labels["n_nonpositive_draws"], errors="raise"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(nonpositive).all()
            or (nonpositive < 0).any()
            or (nonpositive > draw_counts).any()
            or not np.allclose(
                nonpositive, np.round(nonpositive), atol=atol, rtol=0.0
            )
        ):
            raise ValueError(
                "n_nonpositive_draws must be integer counts between zero and n_draws"
            )

    label_names = np.asarray(_label_names(anchor_values), dtype=object)
    expected_modal = label_names[np.argmax(probabilities, axis=1)]
    observed_modal = labels["modal_label"].astype(str).to_numpy()
    if not np.array_equal(observed_modal, expected_modal):
        raise ValueError(
            "modal_label is inconsistent with the soft-label probabilities"
        )
    observed_max = pd.to_numeric(
        labels["max_probability"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(observed_max).all() or not np.allclose(
        observed_max, probabilities.max(axis=1), atol=atol, rtol=0.0
    ):
        raise ValueError(
            "max_probability is inconsistent with the soft-label probabilities"
        )
    observed_entropy = pd.to_numeric(labels["entropy"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(observed_entropy).all() or not np.allclose(
        observed_entropy, _entropy(probabilities), atol=atol, rtol=0.0
    ):
        raise ValueError("entropy is inconsistent with the soft-label probabilities")


def derive_posterior_soft_labels(
    draws: pd.DataFrame,
    mapping: pd.DataFrame,
    anchors: Sequence[float] = DEFAULT_ANCHORS,
    relative_half_width: float = 0.20,
    value_col: str = "primary_slope_mV_dec",
    sample_col: str = "sample_record",
    *,
    expected_draws_per_record: int | None = None,
) -> pd.DataFrame:
    """Derive one posterior-probability vector per mapped record.

    Parameters
    ----------
    draws:
        Long posterior-draw table.  Every row is one slope draw.
    mapping:
        One-row-per-record identifier/metadata table.  Its row order and all
        of its columns are preserved in the result.
    anchors:
        Positive, strictly increasing classical slope values in mV dec-1.
    relative_half_width:
        Linear relative half-width.  An anchor ``a`` has the inclusive band
        ``[a * (1 - width), a * (1 + width)]``.
    expected_draws_per_record:
        If provided, require exactly this many draws for every mapped record.

    Notes
    -----
    Nonpositive finite draws are assigned to ``off_cardinal`` and counted in
    ``n_nonpositive_draws``.  Nonfinite values fail rather than being silently
    discarded.  Draws in overlapping bands are assigned to the nearest anchor
    in absolute log distance; an exact tie goes to the lower anchor.
    """

    if not isinstance(draws, pd.DataFrame):
        raise TypeError("draws must be a pandas DataFrame")
    if not isinstance(mapping, pd.DataFrame):
        raise TypeError("mapping must be a pandas DataFrame")
    anchor_values = _validated_anchors(anchors)
    if (
        not np.isfinite(relative_half_width)
        or relative_half_width <= 0
        or relative_half_width >= 1
    ):
        raise ValueError("relative_half_width must be finite and lie in (0, 1)")
    missing_draw_columns = sorted({sample_col, value_col} - set(draws.columns))
    if missing_draw_columns:
        raise ValueError(
            f"Posterior draws are missing required columns: {missing_draw_columns}"
        )
    if sample_col not in mapping.columns:
        raise ValueError(f"Mapping is missing required column: {sample_col!r}")
    if mapping.empty:
        raise ValueError("mapping must contain at least one record")
    if draws.empty:
        raise ValueError("draws must contain at least one posterior draw")
    if mapping[sample_col].isna().any():
        raise ValueError(f"Mapping {sample_col!r} contains missing values")
    if mapping[sample_col].duplicated().any():
        raise ValueError(f"Mapping must be unique by {sample_col!r}")
    if draws[sample_col].isna().any():
        raise ValueError(f"Posterior-draw {sample_col!r} contains missing values")

    probability_columns = _probability_columns(anchor_values)
    derived_columns = {
        *probability_columns,
        "n_draws",
        "n_nonpositive_draws",
        "modal_label",
        "max_probability",
        "entropy",
    }
    collisions = sorted(derived_columns & set(mapping.columns))
    if collisions:
        raise ValueError(
            f"Mapping columns collide with derived soft-label fields: {collisions}"
        )

    mapping_index = pd.Index(mapping[sample_col])
    draw_positions = mapping_index.get_indexer(draws[sample_col])
    if (draw_positions < 0).any():
        unexpected = pd.unique(draws.loc[draw_positions < 0, sample_col]).tolist()
        raise ValueError(
            "Posterior draws contain records absent from mapping: "
            f"{unexpected[:10]}"
        )
    observed_positions = np.unique(draw_positions)
    if len(observed_positions) != len(mapping):
        missing_positions = np.setdiff1d(
            np.arange(len(mapping), dtype=int), observed_positions
        )
        missing_records = mapping.iloc[missing_positions][sample_col].tolist()
        raise ValueError(
            "Mapping contains records without posterior draws: "
            f"{missing_records[:10]}"
        )

    try:
        values = pd.to_numeric(draws[value_col], errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Posterior-draw column {value_col!r} must be numeric") from error
    if not np.isfinite(values).all():
        bad_rows = np.flatnonzero(~np.isfinite(values))
        raise ValueError(
            f"Posterior-draw column {value_col!r} contains nonfinite values "
            f"at rows {bad_rows[:10].tolist()}"
        )

    n_classes = len(anchor_values) + 1
    off_cardinal_code = len(anchor_values)
    class_codes = np.full(len(values), off_cardinal_code, dtype=np.intp)
    positive = values > 0
    if positive.any():
        positive_values = values[positive]
        lower = anchor_values * (1.0 - relative_half_width)
        upper = anchor_values * (1.0 + relative_half_width)
        in_band = (positive_values[:, None] >= lower) & (
            positive_values[:, None] <= upper
        )
        has_band = in_band.any(axis=1)
        if has_band.any():
            distances = np.abs(
                np.log(positive_values[has_band, None])
                - np.log(anchor_values)[None, :]
            )
            distances[~in_band[has_band]] = np.inf
            positive_codes = np.full(
                len(positive_values), off_cardinal_code, dtype=np.intp
            )
            positive_codes[has_band] = np.argmin(distances, axis=1)
            class_codes[positive] = positive_codes

    counts = np.zeros((len(mapping), n_classes), dtype=np.int64)
    np.add.at(counts, (draw_positions, class_codes), 1)
    n_draws = counts.sum(axis=1)
    if expected_draws_per_record is not None:
        if (
            isinstance(expected_draws_per_record, bool)
            or not isinstance(expected_draws_per_record, (int, np.integer))
            or expected_draws_per_record <= 0
        ):
            raise ValueError("expected_draws_per_record must be a positive integer")
        mismatched = n_draws != int(expected_draws_per_record)
        if mismatched.any():
            details = [
                (
                    mapping.iloc[position][sample_col],
                    int(n_draws[position]),
                )
                for position in np.flatnonzero(mismatched)[:10]
            ]
            raise ValueError(
                "Posterior draw counts do not equal "
                f"expected_draws_per_record={expected_draws_per_record}: {details}"
            )

    probabilities = counts / n_draws[:, None]
    modal_codes = np.argmax(probabilities, axis=1)
    label_names = np.asarray(_label_names(anchor_values), dtype=object)
    nonpositive_counts = np.zeros(len(mapping), dtype=np.int64)
    np.add.at(nonpositive_counts, draw_positions[~positive], 1)

    result = mapping.reset_index(drop=True).copy()
    for column_position, column in enumerate(probability_columns):
        result[column] = probabilities[:, column_position]
    result["n_draws"] = n_draws
    result["n_nonpositive_draws"] = nonpositive_counts
    result["modal_label"] = label_names[modal_codes]
    result["max_probability"] = probabilities.max(axis=1)
    result["entropy"] = _entropy(probabilities)
    validate_soft_labels(result, anchor_values, sample_col=sample_col)
    return result


def support_summary(
    labels: pd.DataFrame,
    anchors: Sequence[float] = DEFAULT_ANCHORS,
    *,
    scope_col: str | None = "electrolyte_scope",
    group_col: str | None = "cv_group_id",
) -> pd.DataFrame:
    """Summarize effective posterior mass and modal support by population.

    The returned table is long-form with one row per population/class.  It
    always includes ``population == "all"`` and, when ``scope_col`` is
    available, one population for each observed scope value.
    """

    anchor_values = _validated_anchors(anchors)
    validate_soft_labels(labels, anchor_values)
    if scope_col is not None and scope_col in labels.columns:
        if labels[scope_col].isna().any():
            raise ValueError(f"{scope_col!r} contains missing values")
        scope_values = sorted(labels[scope_col].unique(), key=lambda value: str(value))
    else:
        scope_values = []
    if group_col is not None and group_col in labels.columns:
        if labels[group_col].isna().any():
            raise ValueError(f"{group_col!r} contains missing values")
        has_groups = True
    else:
        has_groups = False

    probability_columns = _probability_columns(anchor_values)
    label_names = _label_names(anchor_values)
    populations: list[tuple[object, pd.DataFrame]] = [("all", labels)]
    populations.extend(
        (scope, labels.loc[labels[scope_col].eq(scope)])
        for scope in scope_values
    )
    rows: list[dict[str, object]] = []
    for population, frame in populations:
        common = {
            "population": population,
            "n_records": int(len(frame)),
            "n_connected_groups": (
                int(frame[group_col].nunique()) if has_groups else pd.NA
            ),
            "median_max_probability": float(frame["max_probability"].median()),
            "q25_max_probability": float(frame["max_probability"].quantile(0.25)),
            "q75_max_probability": float(frame["max_probability"].quantile(0.75)),
            "population_n_any_class_probability_ge_0_8": int(
                (frame["max_probability"] >= 0.8).sum()
            ),
            "mean_entropy": float(frame["entropy"].mean()),
        }
        for label_name, probability_column in zip(
            label_names, probability_columns, strict=True
        ):
            modal = frame["modal_label"].eq(label_name)
            rows.append(
                {
                    **common,
                    "class_label": label_name,
                    "probability_column": probability_column,
                    "effective_posterior_mass": float(frame[probability_column].sum()),
                    "modal_record_count": int(modal.sum()),
                    "n_class_probability_ge_0_5": int(
                        (frame[probability_column] >= 0.5).sum()
                    ),
                    "n_class_probability_ge_0_8": int(
                        (frame[probability_column] >= 0.8).sum()
                    ),
                    "max_class_probability": float(
                        frame[probability_column].max()
                    ),
                    "modal_group_count": (
                        int(frame.loc[modal, group_col].nunique())
                        if has_groups
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows)


def derive_hierarchical_soft_labels(
    labels: pd.DataFrame,
    anchors: Sequence[float] = DEFAULT_ANCHORS,
) -> pd.DataFrame:
    """Collapse 30/40 posterior mass for the primary supervised target.

    The original ``q_30_like`` and ``q_40_like`` columns remain untouched in
    the returned registry.  Only the model target is collapsed, because the
    current data do not support a stable stand-alone 30-like training class.
    """

    anchor_values = _validated_anchors(anchors)
    if tuple(anchor_values) != DEFAULT_ANCHORS:
        raise ValueError(
            "The hierarchical target is frozen to the 30/40/60/120 anchors"
        )
    validate_soft_labels(labels, anchor_values)
    result = labels.copy()
    result["q_low_30_40_like"] = (
        result["q_30_like"].to_numpy(float)
        + result["q_40_like"].to_numpy(float)
    )
    probabilities = result.loc[
        :,
        [
            "q_low_30_40_like",
            "q_60_like",
            "q_120_like",
            "q_off_cardinal",
        ],
    ].to_numpy(float)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("Hierarchical soft-label rows do not sum to one")
    labels_array = np.asarray(HIERARCHICAL_CLASS_LABELS, dtype=object)
    result["hierarchical_modal_label"] = labels_array[
        np.argmax(probabilities, axis=1)
    ]
    result["hierarchical_max_probability"] = probabilities.max(axis=1)
    result["hierarchical_entropy"] = _entropy(probabilities)
    validate_hierarchical_soft_labels(result)
    return result


def validate_hierarchical_soft_labels(
    labels: pd.DataFrame,
    *,
    sample_col: str = "sample_record",
    atol: float = 1e-10,
) -> None:
    """Raise if the collapsed four-component model target is invalid."""

    required = {
        *HIERARCHICAL_SOFT_LABEL_COLUMNS,
        "hierarchical_modal_label",
        "hierarchical_max_probability",
        "hierarchical_entropy",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(
            f"Hierarchical soft-label table is missing required columns: {missing}"
        )
    if labels.empty:
        raise ValueError("Hierarchical soft-label table must not be empty")
    if sample_col in labels.columns and labels[sample_col].duplicated().any():
        raise ValueError(f"Hierarchical labels are not unique by {sample_col!r}")
    probabilities = labels.loc[
        :, HIERARCHICAL_SOFT_LABEL_COLUMNS
    ].to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Hierarchical soft-label probabilities must be finite")
    if (probabilities < -atol).any() or (probabilities > 1.0 + atol).any():
        raise ValueError(
            "Hierarchical soft-label probabilities must lie in [0, 1]"
        )
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=atol, rtol=0.0
    ):
        raise ValueError(
            "Hierarchical soft-label probabilities must sum to one per record"
        )
    expected_modal = np.asarray(HIERARCHICAL_CLASS_LABELS, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]
    if not np.array_equal(
        labels["hierarchical_modal_label"].astype(str).to_numpy(),
        expected_modal,
    ):
        raise ValueError(
            "hierarchical_modal_label is inconsistent with probabilities"
        )
    if not np.allclose(
        pd.to_numeric(
            labels["hierarchical_max_probability"], errors="raise"
        ).to_numpy(float),
        probabilities.max(axis=1),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            "hierarchical_max_probability is inconsistent with probabilities"
        )
    if not np.allclose(
        pd.to_numeric(
            labels["hierarchical_entropy"], errors="raise"
        ).to_numpy(float),
        _entropy(probabilities),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            "hierarchical_entropy is inconsistent with probabilities"
        )


def hierarchical_support_summary(
    labels: pd.DataFrame,
    *,
    scope_col: str | None = "electrolyte_scope",
    group_col: str | None = "cv_group_id",
) -> pd.DataFrame:
    """Summarize support for the four-component supervised target."""

    validate_hierarchical_soft_labels(labels)
    if scope_col is not None and scope_col in labels.columns:
        if labels[scope_col].isna().any():
            raise ValueError(f"{scope_col!r} contains missing values")
        scope_values = sorted(
            labels[scope_col].unique(), key=lambda value: str(value)
        )
    else:
        scope_values = []
    has_groups = group_col is not None and group_col in labels.columns
    if has_groups and labels[group_col].isna().any():
        raise ValueError(f"{group_col!r} contains missing values")
    populations: list[tuple[object, pd.DataFrame]] = [("all", labels)]
    populations.extend(
        (scope, labels.loc[labels[scope_col].eq(scope)])
        for scope in scope_values
    )
    rows: list[dict[str, object]] = []
    for population, frame in populations:
        common = {
            "population": population,
            "n_records": int(len(frame)),
            "n_connected_groups": (
                int(frame[group_col].nunique()) if has_groups else pd.NA
            ),
            "median_max_probability": float(
                frame["hierarchical_max_probability"].median()
            ),
            "q25_max_probability": float(
                frame["hierarchical_max_probability"].quantile(0.25)
            ),
            "q75_max_probability": float(
                frame["hierarchical_max_probability"].quantile(0.75)
            ),
            "population_n_any_class_probability_ge_0_8": int(
                (frame["hierarchical_max_probability"] >= 0.8).sum()
            ),
            "mean_entropy": float(frame["hierarchical_entropy"].mean()),
        }
        for class_label, probability_column in zip(
            HIERARCHICAL_CLASS_LABELS,
            HIERARCHICAL_SOFT_LABEL_COLUMNS,
            strict=True,
        ):
            modal = frame["hierarchical_modal_label"].eq(class_label)
            rows.append(
                {
                    **common,
                    "class_label": class_label,
                    "probability_column": probability_column,
                    "effective_posterior_mass": float(
                        frame[probability_column].sum()
                    ),
                    "modal_record_count": int(modal.sum()),
                    "n_class_probability_ge_0_5": int(
                        (frame[probability_column] >= 0.5).sum()
                    ),
                    "n_class_probability_ge_0_8": int(
                        (frame[probability_column] >= 0.8).sum()
                    ),
                    "max_class_probability": float(
                        frame[probability_column].max()
                    ),
                    "modal_group_count": (
                        int(frame.loc[modal, group_col].nunique())
                        if has_groups
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows)


def low_anchor_subdivision_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Preserve conditional 30-vs-40 posterior support without training it."""

    validate_soft_labels(labels)
    low_mass = (
        labels["q_30_like"].to_numpy(float)
        + labels["q_40_like"].to_numpy(float)
    )
    q30 = np.divide(
        labels["q_30_like"].to_numpy(float),
        low_mass,
        out=np.full(len(labels), np.nan, dtype=float),
        where=low_mass > 0,
    )
    q40 = np.divide(
        labels["q_40_like"].to_numpy(float),
        low_mass,
        out=np.full(len(labels), np.nan, dtype=float),
        where=low_mass > 0,
    )
    identity_columns = [
        column
        for column in (
            "record_id",
            "sample_record",
            "curve_id",
            "cv_group_id",
            "electrolyte_scope",
        )
        if column in labels.columns
    ]
    result = labels.loc[:, identity_columns].copy()
    result["q_low_30_40_like"] = low_mass
    result["q_30_given_low"] = q30
    result["q_40_given_low"] = q40
    return result
