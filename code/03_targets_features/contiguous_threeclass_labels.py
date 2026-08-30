"""Posterior-soft labels for the contiguous three-class sensitivity.

This module defines the post-hoc prediction-oriented response requested after
reviewing the completed classic-neighbourhood three-class analysis.  It does
not alter or reinterpret the frozen v1 response.  Every posterior draw is
assigned to exactly one contiguous interval:

``low_60_side``
    ``slope < sqrt(60 * 120)``.
``120_side``
    ``sqrt(60 * 120) <= slope <= 144``.
``high_off``
    ``slope > 144``.

Only the resulting three probabilities are used as one response vector per
record.  Posterior draws are never expanded into ML rows.  Nonpositive finite
draws remain valid members of the user-specified ``slope < midpoint`` interval
and are counted explicitly for audit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


LOW_120_LOG_MIDPOINT_MV_DEC = float(np.sqrt(60.0 * 120.0))
HIGH_OFF_BOUNDARY_MV_DEC = 144.0

CONTIGUOUS_THREE_CLASS_LABELS: tuple[str, ...] = (
    "low_60_side",
    "120_side",
    "high_off",
)
CONTIGUOUS_THREE_CLASS_Q_COLUMNS: tuple[str, ...] = tuple(
    f"q_{label}" for label in CONTIGUOUS_THREE_CLASS_LABELS
)
CONTIGUOUS_THREE_CLASS_P_COLUMNS: tuple[str, ...] = tuple(
    f"p_{label}" for label in CONTIGUOUS_THREE_CLASS_LABELS
)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-300, 1.0)
    values = np.asarray(probabilities, dtype=float)
    return -np.sum(values * np.log(clipped), axis=1)


def validate_contiguous_threeclass_soft_labels(
    labels: pd.DataFrame,
    *,
    atol: float = 1e-12,
) -> None:
    """Fail closed unless ``labels`` is one valid probability row per record."""

    required = {
        "record_id",
        "sample_record",
        "n_draws",
        "n_nonpositive_draws",
        "threeclass_modal_label",
        "threeclass_max_probability",
        "threeclass_entropy",
        *CONTIGUOUS_THREE_CLASS_Q_COLUMNS,
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(
            "Contiguous three-class labels are missing required columns: "
            f"{missing}"
        )
    if labels.empty:
        raise ValueError("Contiguous three-class labels cannot be empty")
    if labels["record_id"].duplicated().any():
        raise ValueError("Contiguous three-class labels must be unique by record_id")
    if labels["sample_record"].duplicated().any():
        raise ValueError(
            "Contiguous three-class labels must be unique by sample_record"
        )

    probabilities = labels.loc[
        :, list(CONTIGUOUS_THREE_CLASS_Q_COLUMNS)
    ].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Contiguous three-class probabilities must be finite")
    if np.any(probabilities < -atol) or np.any(probabilities > 1.0 + atol):
        raise ValueError(
            "Contiguous three-class probabilities must lie in [0, 1]"
        )
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=atol, rtol=0.0
    ):
        raise ValueError(
            "Every contiguous three-class probability row must sum to one"
        )

    modal_index = np.argmax(probabilities, axis=1)
    modal = np.asarray(CONTIGUOUS_THREE_CLASS_LABELS, dtype=object)[modal_index]
    if not np.array_equal(
        modal,
        labels["threeclass_modal_label"].astype(str).to_numpy(dtype=object),
    ):
        raise ValueError(
            "threeclass_modal_label is inconsistent with probability argmax"
        )
    observed_max = pd.to_numeric(
        labels["threeclass_max_probability"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.allclose(
        observed_max, np.max(probabilities, axis=1), atol=atol, rtol=0.0
    ):
        raise ValueError(
            "threeclass_max_probability is inconsistent with probabilities"
        )
    observed_entropy = pd.to_numeric(
        labels["threeclass_entropy"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.allclose(
        observed_entropy, _entropy(probabilities), atol=atol, rtol=0.0
    ):
        raise ValueError(
            "threeclass_entropy is inconsistent with probabilities"
        )


def derive_contiguous_threeclass_soft_labels(
    draws: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    value_col: str = "primary_slope_mV_dec",
    sample_col: str = "sample_record",
    lower_boundary_mV_dec: float = LOW_120_LOG_MIDPOINT_MV_DEC,
    upper_boundary_mV_dec: float = HIGH_OFF_BOUNDARY_MV_DEC,
    expected_draws_per_record: int | None = 8_000,
) -> pd.DataFrame:
    """Assign posterior draws to the three contiguous prediction intervals."""

    if not isinstance(draws, pd.DataFrame):
        raise TypeError("draws must be a pandas DataFrame")
    if not isinstance(mapping, pd.DataFrame):
        raise TypeError("mapping must be a pandas DataFrame")
    required_mapping = {"record_id", sample_col}
    missing_mapping = sorted(required_mapping - set(mapping.columns))
    missing_draws = sorted({sample_col, value_col} - set(draws.columns))
    if missing_mapping:
        raise ValueError(f"Mapping is missing required columns: {missing_mapping}")
    if missing_draws:
        raise ValueError(
            f"Posterior draws are missing required columns: {missing_draws}"
        )
    if mapping.empty or draws.empty:
        raise ValueError("Mapping and posterior draws must both be non-empty")
    if mapping[sample_col].isna().any() or mapping[sample_col].duplicated().any():
        raise ValueError(
            f"Mapping must contain unique, non-missing {sample_col!r} values"
        )
    if draws[sample_col].isna().any():
        raise ValueError(
            f"Posterior draws contain missing {sample_col!r} values"
        )

    lower = float(lower_boundary_mV_dec)
    upper = float(upper_boundary_mV_dec)
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0
        or upper <= lower
    ):
        raise ValueError(
            "Contiguous boundaries must be finite, positive, and increasing"
        )

    mapping_index = pd.Index(mapping[sample_col])
    positions = mapping_index.get_indexer(draws[sample_col])
    if np.any(positions < 0):
        unexpected = pd.unique(
            draws.loc[positions < 0, sample_col]
        ).tolist()
        raise ValueError(
            "Posterior draws contain records absent from mapping: "
            f"{unexpected[:10]}"
        )
    observed = np.unique(positions)
    if len(observed) != len(mapping):
        missing_positions = sorted(set(range(len(mapping))) - set(observed))
        missing_samples = mapping.iloc[missing_positions][sample_col].tolist()
        raise ValueError(
            "Mapping records have no posterior draws: "
            f"{missing_samples[:10]}"
        )

    slope = pd.to_numeric(draws[value_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(slope).all():
        raise ValueError("Posterior slope draws must all be finite")

    n_records = len(mapping)
    counts = np.zeros((n_records, len(CONTIGUOUS_THREE_CLASS_LABELS)), dtype=int)
    class_index = np.where(
        slope < lower,
        0,
        np.where(slope <= upper, 1, 2),
    )
    np.add.at(counts, (positions, class_index), 1)
    n_draws = counts.sum(axis=1)
    n_nonpositive = np.zeros(n_records, dtype=int)
    np.add.at(n_nonpositive, positions, slope <= 0)
    if expected_draws_per_record is not None and not np.all(
        n_draws == int(expected_draws_per_record)
    ):
        bad = mapping.loc[
            n_draws != int(expected_draws_per_record), sample_col
        ].tolist()
        raise ValueError(
            "Every record must have the expected number of posterior draws; "
            f"expected={int(expected_draws_per_record)}, bad={bad[:10]}"
        )

    probabilities = counts / n_draws[:, None]
    output = mapping.copy().reset_index(drop=True)
    collisions = sorted(
        {
            *CONTIGUOUS_THREE_CLASS_Q_COLUMNS,
            "n_draws",
            "n_nonpositive_draws",
            "threeclass_modal_label",
            "threeclass_max_probability",
            "threeclass_entropy",
            "lower_boundary_mV_dec",
            "upper_boundary_mV_dec",
            "label_scheme_id",
        }
        & set(output.columns)
    )
    if collisions:
        raise ValueError(
            "Mapping columns collide with contiguous label outputs: "
            f"{collisions}"
        )
    for index, column in enumerate(CONTIGUOUS_THREE_CLASS_Q_COLUMNS):
        output[column] = probabilities[:, index]
    output["n_draws"] = n_draws
    output["n_nonpositive_draws"] = n_nonpositive
    modal_index = np.argmax(probabilities, axis=1)
    output["threeclass_modal_label"] = np.asarray(
        CONTIGUOUS_THREE_CLASS_LABELS, dtype=object
    )[modal_index]
    output["threeclass_max_probability"] = np.max(probabilities, axis=1)
    output["threeclass_entropy"] = _entropy(probabilities)
    output["lower_boundary_mV_dec"] = lower
    output["upper_boundary_mV_dec"] = upper
    output["label_scheme_id"] = "contiguous_log_midpoint_84p8528_high144_v2"
    validate_contiguous_threeclass_soft_labels(output)
    return output


def contiguous_threeclass_support(labels: pd.DataFrame) -> pd.DataFrame:
    """Summarize effective posterior mass and modal connected-group support."""

    validate_contiguous_threeclass_soft_labels(labels)
    probabilities = labels.loc[
        :, list(CONTIGUOUS_THREE_CLASS_Q_COLUMNS)
    ].to_numpy(dtype=float)
    modal_index = np.argmax(probabilities, axis=1)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(CONTIGUOUS_THREE_CLASS_LABELS):
        selected = modal_index == index
        rows.append(
            {
                "class_label": label,
                "effective_posterior_mass": float(probabilities[:, index].sum()),
                "modal_record_count": int(np.sum(selected)),
                "modal_connected_group_count": int(
                    labels.loc[selected, "cv_group_id"].astype(str).nunique()
                    if "cv_group_id" in labels.columns
                    else np.nan
                ),
                "mean_probability": float(probabilities[:, index].mean()),
            }
        )
    return pd.DataFrame(rows)


__all__: Sequence[str] = (
    "CONTIGUOUS_THREE_CLASS_LABELS",
    "CONTIGUOUS_THREE_CLASS_P_COLUMNS",
    "CONTIGUOUS_THREE_CLASS_Q_COLUMNS",
    "HIGH_OFF_BOUNDARY_MV_DEC",
    "LOW_120_LOG_MIDPOINT_MV_DEC",
    "contiguous_threeclass_support",
    "derive_contiguous_threeclass_soft_labels",
    "validate_contiguous_threeclass_soft_labels",
)
