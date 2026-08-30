from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import pandas as pd

from .plot_style import PALETTE, apply_publication_style, save_figure


_MANIFEST_COLUMNS = (
    "figure_id",
    "figure_family",
    "scope",
    "feature",
    "panel",
    "block",
    "model",
    "format",
    "path",
    "status",
    "gate_status",
    "reason",
    "n_records",
    "n_groups",
)
_CARDINAL_GUIDES = (30.0, 40.0, 60.0, 120.0)
_MARGINAL_BOUNDARY = (
    "Marginal descriptive association; not adjusted, ranked, mechanistic, or causal."
)
_ADJUSTED_BOUNDARY = (
    "Cross-fitted adjusted associational curve; not a causal or mechanistic effect."
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _text_series(
    frame: pd.DataFrame,
    names: Sequence[str],
    *,
    default: str = "",
) -> pd.Series:
    name = _column(frame, names)
    if name is None:
        return pd.Series(default, index=frame.index, dtype="string")
    return frame[name].astype("string").fillna(default).str.strip()


def _numeric_series(
    frame: pd.DataFrame,
    names: Sequence[str],
) -> pd.Series:
    name = _column(frame, names)
    if name is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _bool_series(
    frame: pd.DataFrame,
    names: Sequence[str],
    *,
    default: bool = False,
) -> pd.Series:
    name = _column(frame, names)
    if name is None:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[name]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin(
        {"1", "true", "yes", "y", "pass", "passed", "released", "eligible"}
    )


def _slug(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_")
    return text.lower() or "unknown"


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _scope_frames(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    scope_column = _column(
        frame, ("scope", "electrolyte_scope", "population")
    )
    if frame.empty or scope_column is None:
        return [("", frame)]
    values = frame[scope_column].astype("string").fillna("unspecified")
    return [
        (str(scope), frame.loc[values.eq(scope)].copy())
        for scope in sorted(values.unique())
    ]


def _family_style(value: object) -> tuple[str, str, str]:
    normalized = str(value).strip().lower()
    if "ru" in normalized:
        return PALETTE["teal"], "o", "RuO$_2$"
    if "ir" in normalized:
        return PALETTE["blue_main"], "^", "IrO$_2$ + Ir"
    return PALETTE["neutral_dark"], "s", str(value) if str(value).strip() else "Other"


def _manifest_rows(
    root: Path,
    *,
    figure_id: str,
    figure_family: str,
    paths: Sequence[Path] = (),
    scope: str = "",
    feature: str = "",
    panel: str = "",
    block: str = "",
    model: str = "",
    status: str = "generated",
    gate_status: str = "not_applicable",
    reason: str = "",
    n_records: int | float | None = None,
    n_groups: int | float | None = None,
) -> list[dict[str, object]]:
    common: dict[str, object] = {
        "figure_id": figure_id,
        "figure_family": figure_family,
        "scope": scope,
        "feature": feature,
        "panel": panel,
        "block": block,
        "model": model,
        "status": status,
        "gate_status": gate_status,
        "reason": reason,
        "n_records": "" if n_records is None or pd.isna(n_records) else int(n_records),
        "n_groups": "" if n_groups is None or pd.isna(n_groups) else int(n_groups),
    }
    if not paths:
        return [{**common, "format": "", "path": ""}]
    return [
        {
            **common,
            "format": path.suffix.lower().lstrip("."),
            "path": _relative(path, root),
        }
        for path in paths
    ]


def _support_figure(
    support: pd.DataFrame,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    *,
    scope: str = "",
) -> tuple[list[Path], int, int]:
    feature = _text_series(support, ("feature_name", "feature", "logical_feature"))
    n_total_records = _numeric_series(
        support, ("n_records", "n_population_records")
    )
    n_nonmissing = _numeric_series(
        support, ("n_nonmissing", "n_records_with_data", "n_records")
    )
    n_total_groups = _numeric_series(
        support,
        ("n_connected_groups", "n_groups", "n_groups_with_data"),
    )
    n_variable_groups = _numeric_series(
        support,
        (
            "n_within_group_variable_groups",
            "n_variable_groups",
            "n_nonzero_groups",
        ),
    )
    n_variable_groups = n_variable_groups.where(
        n_variable_groups.notna(), n_total_groups
    )
    valid = feature.ne("") & (
        n_total_records.notna()
        | n_nonmissing.notna()
        | n_total_groups.notna()
        | n_variable_groups.notna()
    )
    frame = pd.DataFrame(
        {
            "feature": feature.loc[valid],
            "n_total_records": n_total_records.loc[valid],
            "n_nonmissing": n_nonmissing.loc[valid],
            "n_total_groups": n_total_groups.loc[valid],
            "n_variable_groups": n_variable_groups.loc[valid],
        }
    )
    if frame.empty:
        return [], 0, 0
    frame = frame.sort_values(
        ["n_variable_groups", "n_nonmissing", "feature"], na_position="first"
    ).reset_index(drop=True)
    y = np.arange(len(frame))
    apply_publication_style(9)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, max(4.2, min(15.0, 0.31 * len(frame) + 2.0))),
        sharey=True,
    )
    axes[0].barh(
        y,
        frame["n_total_records"].fillna(0),
        color=PALETTE["neutral"],
        edgecolor="black",
        linewidth=0.35,
        label="population records",
    )
    axes[0].barh(
        y,
        frame["n_nonmissing"].fillna(0),
        color=PALETTE["blue_secondary"],
        edgecolor="black",
        linewidth=0.35,
        height=0.56,
        label="nonmissing records",
    )
    axes[0].set_yticks(y, frame["feature"].str.replace("_", " ", regex=False))
    axes[0].set_xlabel("Records")
    axes[1].barh(
        y,
        frame["n_total_groups"].fillna(0),
        color=PALETTE["neutral"],
        edgecolor="black",
        linewidth=0.35,
        label="population connected groups",
    )
    axes[1].barh(
        y,
        frame["n_variable_groups"].fillna(0),
        color=PALETTE["teal"],
        edgecolor="black",
        linewidth=0.35,
        height=0.56,
        label="within-group variable groups",
    )
    axes[1].set_xlabel("Connected groups")
    handles: list[object] = []
    labels: list[str] = []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            handles.append(handle)
            labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        fontsize=8,
    )
    scope_suffix = f" — {scope}" if scope else ""
    fig.suptitle(
        f"Feature support before target-blind reduction{scope_suffix}",
        y=1.065,
    )
    paths = save_figure(fig, output_base, formats=formats, dpi=dpi)
    records_max = frame["n_total_records"].max()
    groups_max = frame["n_total_groups"].max()
    return (
        paths,
        0 if pd.isna(records_max) else int(records_max),
        0 if pd.isna(groups_max) else int(groups_max),
    )


def _redundancy_figure(
    redundancy: pd.DataFrame,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    *,
    scope: str = "",
) -> list[Path]:
    first = _text_series(redundancy, ("feature_a", "first_feature"))
    second = _text_series(redundancy, ("feature_b", "second_feature"))
    rho = _numeric_series(
        redundancy, ("spearman_rho", "rho", "correlation", "abs_spearman_rho")
    )
    valid = first.ne("") & second.ne("") & rho.notna()
    if not valid.any():
        return []
    pairs = pd.DataFrame(
        {"feature_a": first.loc[valid], "feature_b": second.loc[valid], "rho": rho.loc[valid]}
    )
    features = sorted(set(pairs["feature_a"]) | set(pairs["feature_b"]))
    if len(features) < 2:
        return []
    matrix = pd.DataFrame(
        np.eye(len(features), dtype=float),
        index=features,
        columns=features,
    )
    matrix = matrix.where(
        pd.DataFrame(
            np.eye(len(features), dtype=bool),
            index=features,
            columns=features,
        )
    )
    for row in pairs.itertuples(index=False):
        value = float(np.clip(row.rho, -1.0, 1.0))
        matrix.loc[row.feature_a, row.feature_b] = value
        matrix.loc[row.feature_b, row.feature_a] = value

    apply_publication_style(8)
    size = min(15.0, max(6.2, 0.43 * len(features) + 2.5))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(
        matrix.to_numpy(float),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="none",
    )
    labels = [feature.replace("_", " ") for feature in features]
    axis.set_xticks(range(len(features)), labels, rotation=55, ha="right")
    axis.set_yticks(range(len(features)), labels)
    for row_index in range(len(features)):
        for column_index in range(len(features)):
            value = matrix.iat[row_index, column_index]
            if np.isfinite(value) and row_index != column_index and abs(value) >= 0.90:
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(value) >= 0.75 else "black",
                )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman correlation")
    scope_suffix = f" — {scope}" if scope else ""
    axis.set_title(f"Numeric-feature redundancy audit{scope_suffix}")
    return save_figure(fig, output_base, formats=formats, dpi=dpi)


def _atlas_columns(
    atlas: pd.DataFrame,
) -> Mapping[str, str | None]:
    return {
        "record": _column(atlas, ("record_id", "sample_id")),
        "group": _column(atlas, ("cv_group_id", "connected_group_id", "group_id")),
        "scope": _column(atlas, ("electrolyte_scope", "scope", "population")),
        "family": _column(
            atlas,
            ("catalyst_family", "catalyst_stratum", "catalyst_type"),
        ),
        "feature": _column(
            atlas, ("feature_name", "feature", "predictor", "input_feature")
        ),
        "kind": _column(atlas, ("kind", "feature_type", "dtype")),
        "numeric": _column(
            atlas,
            (
                "feature_value_numeric",
                "x_numeric",
                "transformed_value",
                "x_value",
            ),
        ),
        "category": _column(
            atlas,
            (
                "feature_value_category",
                "feature_level",
                "category",
                "x_category",
            ),
        ),
        "value": _column(atlas, ("feature_value", "value", "feature_value_raw")),
        "median": _column(
            atlas,
            (
                "target_median_mV_dec",
                "target_value_mV_dec",
                "posterior_median_mV_dec",
                "target_median",
            ),
        ),
        "low": _column(
            atlas,
            (
                "target_q025_mV_dec",
                "target_interval_low_mV_dec",
                "target_ci_low_mV_dec",
                "posterior_q025_mV_dec",
            ),
        ),
        "high": _column(
            atlas,
            (
                "target_q975_mV_dec",
                "target_interval_high_mV_dec",
                "target_ci_high_mV_dec",
                "posterior_q975_mV_dec",
            ),
        ),
    }


def _validate_atlas_grain(atlas: pd.DataFrame, columns: Mapping[str, str | None]) -> None:
    required = ("record", "feature", "median", "low", "high")
    missing = [name for name in required if columns[name] is None]
    if missing:
        raise ValueError(f"Atlas observations lack required fields: {missing}")
    key = [str(columns["record"]), str(columns["feature"])]
    if columns["scope"] is not None:
        key.insert(0, str(columns["scope"]))
    duplicates = atlas.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(
            "atlas_observations must contain one row per scope-record-feature; "
            f"found {int(duplicates.sum())} duplicated rows"
        )
    median = pd.to_numeric(atlas[str(columns["median"])], errors="coerce")
    low = pd.to_numeric(atlas[str(columns["low"])], errors="coerce")
    high = pd.to_numeric(atlas[str(columns["high"])], errors="coerce")
    invalid = (
        ~np.isfinite(median)
        | ~np.isfinite(low)
        | ~np.isfinite(high)
        | (median <= 0)
        | (low <= 0)
        | (high <= 0)
        | (low > median)
        | (median > high)
    )
    if invalid.any():
        raise ValueError(
            "Atlas target summaries must be finite, positive, and ordered "
            "q025 <= median <= q975"
        )


def _atlas_metadata(summary: pd.DataFrame) -> dict[str, dict[str, object]]:
    if summary.empty:
        return {}
    feature_column = _column(summary, ("feature_name", "feature"))
    if feature_column is None:
        return {}
    metadata: dict[str, dict[str, object]] = {}
    for _, row in summary.drop_duplicates(feature_column).iterrows():
        metadata[str(row[feature_column])] = row.to_dict()
    return metadata


def _filter_atlas_by_registered_status(
    atlas: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_column = _column(
        summary, ("atlas_status", "plot_status", "figure_status")
    )
    summary_feature = _column(summary, ("feature_name", "feature"))
    atlas_feature = _column(atlas, ("feature_name", "feature"))
    if (
        summary.empty
        or status_column is None
        or summary_feature is None
        or atlas_feature is None
    ):
        return atlas, pd.DataFrame()
    summary_scope = _column(
        summary, ("scope", "electrolyte_scope", "population")
    )
    atlas_scope = _column(
        atlas, ("scope", "electrolyte_scope", "population")
    )
    registry = summary.copy()
    registry["_feature_key"] = registry[summary_feature].astype(str)
    registry["_scope_key"] = (
        registry[summary_scope].astype(str)
        if summary_scope is not None
        else ""
    )
    status = registry[status_column].astype("string").str.strip().str.lower()
    allowed = status.isin(
        {
            "generated",
            "generated_descriptive",
            "eligible",
            "pass",
            "passed",
        }
    )
    observations = atlas.copy()
    observations["_feature_key"] = observations[atlas_feature].astype(str)
    observations["_scope_key"] = (
        observations[atlas_scope].astype(str)
        if atlas_scope is not None
        else ""
    )
    allowed_keys = registry.loc[
        allowed, ["_scope_key", "_feature_key"]
    ].drop_duplicates()
    filtered = observations.merge(
        allowed_keys,
        on=["_scope_key", "_feature_key"],
        how="inner",
        validate="many_to_one",
    ).drop(columns=["_scope_key", "_feature_key"])
    skipped = registry.loc[
        ~allowed,
        ["_scope_key", "_feature_key", status_column],
    ].drop_duplicates()
    return filtered, skipped


def _symmetric_offsets(count: int, width: float) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width, width, count)


def _numeric_x_with_tie_jitter(
    values: pd.Series,
    record_ids: pd.Series,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        return numeric
    unique = np.unique(finite)
    if len(unique) > 1:
        positive_gaps = np.diff(unique)
        positive_gaps = positive_gaps[positive_gaps > 0]
        width = 0.04 * float(positive_gaps.min()) if len(positive_gaps) else 0.0
    else:
        width = max(abs(float(unique[0])) * 0.015, 0.015)
    jittered = numeric.copy()
    work = pd.DataFrame(
        {"x": numeric, "record": record_ids.astype(str), "position": np.arange(len(numeric))}
    )
    for _, tied in work.loc[np.isfinite(work["x"])].groupby("x", sort=True):
        if len(tied) < 2:
            continue
        tied = tied.sort_values("record")
        jittered[tied["position"].to_numpy(int)] += _symmetric_offsets(
            len(tied), width
        )
    return jittered


def _target_limits(
    frame: pd.DataFrame,
    low_column: str,
    high_column: str,
) -> tuple[float, float]:
    low = pd.to_numeric(frame[low_column], errors="coerce")
    high = pd.to_numeric(frame[high_column], errors="coerce")
    lower = min(float(low.min()), min(_CARDINAL_GUIDES))
    upper = max(float(high.max()), max(_CARDINAL_GUIDES))
    return max(lower * 0.82, np.finfo(float).tiny), upper * 1.18


def _apply_target_axis(axis: plt.Axes, limits: tuple[float, float]) -> None:
    axis.set_yscale("log")
    for guide in _CARDINAL_GUIDES:
        axis.axhline(
            guide,
            color=PALETTE["neutral_dark"],
            linestyle=":" if guide == 30 else "--",
            linewidth=0.75,
            alpha=0.55,
            zorder=0,
        )
    candidates = np.asarray(
        [20, 30, 40, 60, 80, 120, 160, 240, 400, 600, 1000], dtype=float
    )
    ticks = candidates[(candidates >= limits[0]) & (candidates <= limits[1])]
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FixedFormatter([f"{value:g}" for value in ticks]))
    axis.set_ylim(limits)
    axis.set_ylabel("Tafel slope (mV dec$^{-1}$)")


def _plot_numeric_atlas_axis(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    columns: Mapping[str, str | None],
    feature: str,
    metadata: Mapping[str, object],
    limits: tuple[float, float],
) -> None:
    numeric_column = columns["numeric"] or columns["value"]
    if numeric_column is None:
        axis.text(0.5, 0.5, "Numeric values unavailable", ha="center", va="center")
        axis.set_axis_off()
        return
    x = pd.to_numeric(frame[str(numeric_column)], errors="coerce")
    valid = x.notna()
    frame = frame.loc[valid].copy()
    if frame.empty:
        axis.text(0.5, 0.5, "No finite values", ha="center", va="center")
        axis.set_axis_off()
        return
    x = x.loc[valid]
    records = frame[str(columns["record"])].astype(str)
    x_plot = _numeric_x_with_tie_jitter(x, records)
    median = pd.to_numeric(frame[str(columns["median"])], errors="coerce").to_numpy(float)
    low = pd.to_numeric(frame[str(columns["low"])], errors="coerce").to_numpy(float)
    high = pd.to_numeric(frame[str(columns["high"])], errors="coerce").to_numpy(float)
    family = (
        frame[str(columns["family"])].astype(str)
        if columns["family"] is not None
        else pd.Series("all", index=frame.index)
    )
    seen: set[str] = set()
    for position, (_, row) in enumerate(frame.iterrows()):
        color, marker, label = _family_style(family.iloc[position])
        axis.vlines(
            x_plot[position],
            low[position],
            high[position],
            color=color,
            linewidth=0.65,
            alpha=0.28,
            zorder=1,
        )
        axis.scatter(
            x_plot[position],
            median[position],
            color=color,
            marker=marker,
            s=24,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.86,
            label=label if label not in seen else None,
            zorder=2,
        )
        seen.add(label)
    _apply_target_axis(axis, limits)
    x_label = str(metadata.get("x_label", "")).strip() or feature.replace("_", " ")
    axis.set_xlabel(x_label)
    x_scale = str(metadata.get("x_scale", "")).strip().lower()
    if x_scale == "log" and (x > 0).all():
        axis.set_xscale("log")
    n_groups = (
        frame[str(columns["group"])].nunique()
        if columns["group"] is not None
        else len(frame)
    )
    axis.set_title(f"{feature.replace('_', ' ')}\nn={len(frame)}, groups={n_groups}")


def _plot_categorical_atlas_axis(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    columns: Mapping[str, str | None],
    feature: str,
    metadata: Mapping[str, object],
    limits: tuple[float, float],
) -> None:
    category_column = columns["category"] or columns["value"]
    if category_column is None:
        axis.text(0.5, 0.5, "Category values unavailable", ha="center", va="center")
        axis.set_axis_off()
        return
    categories = frame[str(category_column)].astype("string").fillna("missing")
    order = sorted(categories.unique(), key=lambda value: str(value))
    positions = {str(value): index for index, value in enumerate(order)}
    median = pd.to_numeric(frame[str(columns["median"])], errors="coerce")
    low = pd.to_numeric(frame[str(columns["low"])], errors="coerce")
    high = pd.to_numeric(frame[str(columns["high"])], errors="coerce")
    family = (
        frame[str(columns["family"])].astype(str)
        if columns["family"] is not None
        else pd.Series("all", index=frame.index)
    )
    x_plot = np.zeros(len(frame), dtype=float)
    working = pd.DataFrame(
        {
            "category": categories.astype(str).to_numpy(),
            "record": frame[str(columns["record"])].astype(str).to_numpy(),
            "position": np.arange(len(frame)),
        }
    )
    for category, subset in working.groupby("category", sort=True):
        subset = subset.sort_values("record")
        x_plot[subset["position"].to_numpy(int)] = (
            positions[category] + _symmetric_offsets(len(subset), 0.20)
        )
    seen: set[str] = set()
    for position in range(len(frame)):
        color, marker, label = _family_style(family.iloc[position])
        axis.vlines(
            x_plot[position],
            low.iloc[position],
            high.iloc[position],
            color=color,
            linewidth=0.65,
            alpha=0.28,
            zorder=1,
        )
        axis.scatter(
            x_plot[position],
            median.iloc[position],
            color=color,
            marker=marker,
            s=24,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.86,
            label=label if label not in seen else None,
            zorder=2,
        )
        seen.add(label)
    labels: list[str] = []
    for value in order:
        selected = categories.astype(str).eq(str(value))
        n_records = int(selected.sum())
        n_groups = (
            int(frame.loc[selected, str(columns["group"])].nunique())
            if columns["group"] is not None
            else n_records
        )
        labels.append(f"{value}\n{n_records}r/{n_groups}g")
    axis.set_xticks(range(len(order)), labels, rotation=38, ha="right")
    _apply_target_axis(axis, limits)
    x_label = str(metadata.get("x_label", "")).strip() or feature.replace("_", " ")
    axis.set_xlabel(x_label)
    n_groups = (
        frame[str(columns["group"])].nunique()
        if columns["group"] is not None
        else len(frame)
    )
    axis.set_title(f"{feature.replace('_', ' ')}\nn={len(frame)}, groups={n_groups}")


def _atlas_pages(
    atlas: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[tuple[dict[str, object], list[Path]]]:
    if atlas.empty:
        return []
    columns = _atlas_columns(atlas)
    _validate_atlas_grain(atlas, columns)
    feature_column = str(columns["feature"])
    scope = (
        atlas[str(columns["scope"])].astype("string").fillna("unspecified")
        if columns["scope"] is not None
        else pd.Series("all97", index=atlas.index, dtype="string")
    )
    kind = (
        atlas[str(columns["kind"])].astype("string").str.lower()
        if columns["kind"] is not None
        else pd.Series("", index=atlas.index, dtype="string")
    )
    numeric_fallback = (
        pd.to_numeric(
            atlas[str(columns["numeric"] or columns["value"])], errors="coerce"
        ).notna()
        if columns["numeric"] is not None or columns["value"] is not None
        else pd.Series(False, index=atlas.index)
    )
    atlas = atlas.copy()
    atlas["_scope"] = scope.astype(str)
    atlas["_kind"] = np.where(
        kind.str.contains("num|continuous|float|int", regex=True),
        "numeric",
        np.where(
            kind.str.contains("cat|binary|ordinal|string", regex=True),
            "categorical",
            np.where(numeric_fallback, "numeric", "categorical"),
        ),
    )
    metadata = _atlas_metadata(summary)
    results: list[tuple[dict[str, object], list[Path]]] = []
    for (scope_name, kind_name), group in atlas.groupby(
        ["_scope", "_kind"], sort=True
    ):
        features = sorted(group[feature_column].astype(str).unique())
        for page_index in range(math.ceil(len(features) / 6)):
            page_features = features[page_index * 6 : (page_index + 1) * 6]
            page = group.loc[group[feature_column].astype(str).isin(page_features)]
            limits = _target_limits(
                page,
                str(columns["low"]),
                str(columns["high"]),
            )
            apply_publication_style(8.5)
            nrows = 1 if len(page_features) <= 3 else 2
            fig, axes = plt.subplots(
                nrows,
                3,
                figsize=(13.2, 4.25 * nrows + 0.55),
                squeeze=False,
            )
            for axis, feature_name in zip(
                axes.ravel(), page_features, strict=False
            ):
                feature_frame = page.loc[
                    page[feature_column].astype(str).eq(feature_name)
                ].reset_index(drop=True)
                if kind_name == "numeric":
                    _plot_numeric_atlas_axis(
                        axis,
                        feature_frame,
                        columns=columns,
                        feature=feature_name,
                        metadata=metadata.get(feature_name, {}),
                        limits=limits,
                    )
                else:
                    _plot_categorical_atlas_axis(
                        axis,
                        feature_frame,
                        columns=columns,
                        feature=feature_name,
                        metadata=metadata.get(feature_name, {}),
                        limits=limits,
                    )
            for axis in axes.ravel()[len(page_features) :]:
                axis.set_axis_off()
            handles: list[object] = []
            labels: list[str] = []
            for axis in axes.ravel():
                for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.005),
                    ncol=len(labels),
                )
            fig.suptitle(
                f"Posterior-aware feature association atlas — {scope_name} "
                f"({kind_name}, page {page_index + 1})",
                y=1.075,
            )
            fig.text(0.5, -0.025, _MARGINAL_BOUNDARY, ha="center", fontsize=8)
            figure_id = (
                f"{_slug(scope_name)}__{kind_name}__p{page_index + 1:02d}"
            )
            paths = save_figure(
                fig,
                output_dir / figure_id,
                formats=formats,
                dpi=dpi,
            )
            n_groups = (
                page[str(columns["group"])].nunique()
                if columns["group"] is not None
                else page[str(columns["record"])].nunique()
            )
            results.append(
                (
                    {
                        "figure_id": figure_id,
                        "scope": str(scope_name),
                        "feature": "|".join(page_features),
                        "n_records": page[str(columns["record"])].nunique(),
                        "n_groups": n_groups,
                    },
                    paths,
                )
            )
    return results


def _primary_metric_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    primary = _bool_series(
        summary, ("is_primary_metric", "primary_metric", "metric_primary")
    )
    if primary.any():
        return summary.loc[primary].copy()
    role = _text_series(summary, ("metric_role", "role")).str.lower()
    if role.eq("primary").any():
        return summary.loc[role.eq("primary")].copy()
    metric_column = _column(summary, ("metric", "metric_name"))
    if metric_column is None or summary[metric_column].nunique(dropna=True) <= 1:
        return summary.copy()
    metrics = summary[metric_column].astype(str)
    preferences = (
        "posterior_log_score",
        "negative_log_predictive_density",
        "nlpd",
        "mae_mV_dec",
        "mae",
    )
    for preferred in preferences:
        selected = metrics.str.lower().eq(preferred.lower())
        if selected.any():
            return summary.loc[selected].copy()
    first = sorted(metrics.dropna().unique())[0]
    return summary.loc[metrics.eq(first)].copy()


def _block_delta_figure(
    summary: pd.DataFrame,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> tuple[list[Path], str]:
    frame = _primary_metric_rows(summary)
    estimate = _numeric_series(
        frame,
        ("observed_delta", "delta_estimate", "estimate", "delta"),
    )
    low = _numeric_series(
        frame,
        ("ci95_low", "ci_low", "delta_ci_low"),
    )
    high = _numeric_series(
        frame,
        ("ci95_high", "ci_high", "delta_ci_high"),
    )
    valid = estimate.notna() & low.notna() & high.notna()
    frame = frame.loc[valid].copy()
    if frame.empty:
        return [], ""
    frame["_estimate"] = estimate.loc[valid]
    frame["_low"] = low.loc[valid]
    frame["_high"] = high.loc[valid]
    frame["_model"] = _text_series(
        frame, ("model", "model_family", "primary_model"), default="model"
    )
    frame["_block"] = _text_series(
        frame,
        ("block_id", "block", "augmented_panel", "comparison"),
        default="block",
    )
    frame["_metric"] = _text_series(
        frame, ("metric", "metric_name"), default="primary metric"
    )
    frame["_scope"] = _text_series(
        frame, ("scope", "electrolyte_scope", "population")
    )
    frame = frame.sort_values(["_scope", "_block", "_model"]).reset_index(drop=True)
    labels = [
        " | ".join(
            value
            for value in (
                str(row["_scope"]).replace("_", " "),
                str(row["_block"]).replace("_", " "),
                str(row["_model"]).replace("_", " "),
            )
            if value
        )
        for _, row in frame.iterrows()
    ]
    y = np.arange(len(frame))
    colors = [
        PALETTE["teal"] if "gam" in model.lower() else PALETTE["blue_main"]
        for model in frame["_model"]
    ]
    apply_publication_style(9)
    fig, axis = plt.subplots(figsize=(9.2, max(4.0, 0.48 * len(frame) + 2.0)))
    axis.axvline(0, color=PALETTE["neutral_dark"], linewidth=1.0)
    axis.errorbar(
        frame["_estimate"],
        y,
        xerr=[
            frame["_estimate"] - frame["_low"],
            frame["_high"] - frame["_estimate"],
        ],
        fmt="none",
        ecolor=PALETTE["neutral_dark"],
        capsize=3,
        linewidth=1.2,
    )
    axis.scatter(
        frame["_estimate"],
        y,
        color=colors,
        s=54,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    axis.set_yticks(y, labels)
    metric = str(frame["_metric"].iloc[0])
    axis.set_xlabel(
        f"Augmented − reference Δ {metric.replace('_', ' ')} "
        "(negative favors augmented)"
    )
    axis.set_title(
        "Block-wise paired comparison under connected-group holdout\n"
        "Connected-group bootstrap 95% intervals"
    )
    return save_figure(fig, output_base, formats=formats, dpi=dpi), metric


def _release_pass(frame: pd.DataFrame) -> pd.Series:
    explicit = _column(
        frame,
        (
            "block_gate_passed",
            "effect_release_pass",
            "release_pass",
            "gate_pass",
            "overall_pass",
        ),
    )
    if explicit is not None:
        return _bool_series(frame, (explicit,))
    status = _text_series(
        frame, ("effect_curve_status", "release_status", "gate_status", "status")
    ).str.lower()
    return status.isin({"pass", "passed", "released", "eligible"})


def _effect_grid_for_release(
    grid: pd.DataFrame,
    release: pd.Series,
) -> pd.DataFrame:
    selected = grid.copy()
    key_pairs = (
        (("block_id", "block"), ("block_id", "block")),
        (("panel", "augmented_panel"), ("panel", "augmented_panel")),
        (
            ("scope", "electrolyte_scope", "population"),
            ("scope", "electrolyte_scope", "population"),
        ),
    )
    for grid_keys, release_keys in key_pairs:
        grid_key = next((key for key in grid_keys if key in selected), None)
        release_key = next((key for key in release_keys if key in release.index), None)
        if grid_key is not None and release_key is not None:
            value = str(release[release_key]).strip()
            if value:
                selected = selected.loc[selected[grid_key].astype(str).eq(value)]
    model = str(
        release.get("released_model", release.get("primary_model", ""))
    ).strip()
    if model and "model" in selected:
        selected = selected.loc[selected["model"].astype(str).eq(model)]
    return selected


def _effect_curve_figure(
    grid: pd.DataFrame,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    *,
    feature: str,
) -> tuple[list[Path], int | None, int | None]:
    numeric_x_column = _column(
        grid,
        (
            "grid_quantile",
            "grid_value_numeric",
            "x_value",
            "grid_value",
            "feature_value_numeric",
        ),
    )
    category_x_column = _column(
        grid,
        ("grid_value_category", "category", "feature_level"),
    )
    centered_column = _column(
        grid,
        ("predicted_log10_centered", "effect_log10_centered"),
    )
    ratio_column = _column(
        grid,
        ("effect_ratio", "tafel_slope_ratio", "adjusted_ratio", "effect_mV_dec_or_ratio"),
    )
    centered = (
        pd.to_numeric(grid[centered_column], errors="coerce")
        if centered_column is not None
        else pd.Series(np.nan, index=grid.index, dtype=float)
    )
    if ratio_column is not None:
        estimate = pd.to_numeric(grid[ratio_column], errors="coerce")
    elif centered_column is not None:
        estimate = np.power(10.0, centered)
    else:
        return [], None, None
    ratio_low = _column(
        grid,
        ("ratio_ci95_low", "effect_ratio_ci95_low", "ratio_ci_low"),
    )
    ratio_high = _column(
        grid,
        ("ratio_ci95_high", "effect_ratio_ci95_high", "ratio_ci_high"),
    )
    low_log = _column(
        grid,
        ("predicted_log10_ci_low", "effect_log10_ci_low", "log10_ci95_low"),
    )
    high_log = _column(
        grid,
        ("predicted_log10_ci_high", "effect_log10_ci_high", "log10_ci95_high"),
    )
    if ratio_low is not None and ratio_high is not None:
        low = pd.to_numeric(grid[ratio_low], errors="coerce")
        high = pd.to_numeric(grid[ratio_high], errors="coerce")
    elif low_log is not None and high_log is not None:
        low = np.power(10.0, pd.to_numeric(grid[low_log], errors="coerce"))
        high = np.power(10.0, pd.to_numeric(grid[high_log], errors="coerce"))
    else:
        raw_low = _numeric_series(
            grid, ("ci95_low", "ci_low", "effect_ci_low")
        )
        raw_high = _numeric_series(
            grid, ("ci95_high", "ci_high", "effect_ci_high")
        )
        centered_bracket = (
            centered_column is not None
            and (
                (raw_low <= centered)
                & (centered <= raw_high)
                & raw_low.notna()
                & raw_high.notna()
            ).mean()
            >= 0.8
        )
        ratio_bracket = (
            (raw_low <= estimate)
            & (estimate <= raw_high)
            & raw_low.notna()
            & raw_high.notna()
        ).mean()
        if centered_bracket and ratio_bracket < 0.8:
            low = np.power(10.0, raw_low)
            high = np.power(10.0, raw_high)
        else:
            low, high = raw_low, raw_high
    valid = (
        np.isfinite(estimate)
        & np.isfinite(low)
        & np.isfinite(high)
        & (estimate > 0)
        & (low > 0)
        & (high > 0)
        & (low <= estimate)
        & (estimate <= high)
    )
    frame = grid.loc[valid].copy()
    frame["_estimate"] = np.asarray(estimate)[valid]
    frame["_low"] = np.asarray(low)[valid]
    frame["_high"] = np.asarray(high)[valid]
    if frame.empty:
        return [], None, None
    group_column = _column(
        frame, ("catalyst_family", "catalyst_stratum", "family")
    )
    groups = (
        list(frame.groupby(group_column, sort=True))
        if group_column is not None
        else [("pooled", frame)]
    )
    apply_publication_style(10)
    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    categorical = numeric_x_column is None or pd.to_numeric(
        frame[numeric_x_column], errors="coerce"
    ).isna().all()
    for family, subset in groups:
        color, marker, label = _family_style(family)
        if str(family) == "pooled":
            color, marker, label = PALETTE["blue_main"], "o", "adjusted association"
        if categorical:
            if category_x_column is None:
                plt.close(fig)
                return [], None, None
            categories = sorted(
                subset[category_x_column].astype("string").fillna("missing").unique()
            )
            x = np.arange(len(categories))
            indexed = subset.set_index(
                subset[category_x_column].astype("string").fillna("missing")
            ).reindex(categories)
            axis.errorbar(
                x,
                indexed["_estimate"],
                yerr=[
                    indexed["_estimate"] - indexed["_low"],
                    indexed["_high"] - indexed["_estimate"],
                ],
                fmt=marker,
                color=color,
                capsize=3,
                label=label,
            )
            axis.set_xticks(x, categories, rotation=35, ha="right")
        else:
            subset = subset.assign(
                _x=pd.to_numeric(subset[numeric_x_column], errors="coerce")
            ).dropna(subset=["_x"]).sort_values("_x")
            axis.plot(
                subset["_x"],
                subset["_estimate"],
                color=color,
                marker=marker,
                markersize=3.5,
                linewidth=1.8,
                label=label,
            )
            axis.fill_between(
                subset["_x"],
                subset["_low"],
                subset["_high"],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
    axis.axhline(
        1.0,
        color=PALETTE["neutral_dark"],
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_xlabel(
        (
            f"{feature.replace('_', ' ')} (training-fold quantile)"
            if numeric_x_column == "grid_quantile"
            else feature.replace("_", " ")
        )
    )
    axis.set_ylabel("Adjusted Tafel-slope ratio vs registered reference")
    axis.set_title(feature.replace("_", " "))
    axis.legend()
    fig.text(0.5, -0.01, _ADJUSTED_BOUNDARY, ha="center", fontsize=8)
    paths = save_figure(fig, output_base, formats=formats, dpi=dpi)
    n_records = _numeric_series(frame, ("n_records",)).max()
    n_groups = _numeric_series(frame, ("n_groups", "n_connected_groups")).max()
    return (
        paths,
        None if pd.isna(n_records) else int(n_records),
        None if pd.isna(n_groups) else int(n_groups),
    )


def _effect_figures(
    release: pd.DataFrame,
    grid: pd.DataFrame,
    root: Path,
    formats: Iterable[str],
    dpi: int,
    *,
    smoke: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if release.empty:
        return _manifest_rows(
            root,
            figure_id="adjusted_effect_curves",
            figure_family="adjusted_effect",
            status="gated",
            gate_status="no_release_registry",
            reason="effect release registry is unavailable; no effect curve released",
        )
    passes = _release_pass(release)
    for row_index, release_row in release.iterrows():
        block = str(
            release_row.get("block_id", release_row.get("block", f"block_{row_index}"))
        ).strip()
        panel = str(
            release_row.get("augmented_panel", release_row.get("panel", ""))
        ).strip()
        scope = str(
            release_row.get(
                "scope",
                release_row.get(
                    "electrolyte_scope", release_row.get("population", "")
                ),
            )
        ).strip()
        model = str(
            release_row.get(
                "released_model", release_row.get("primary_model", "")
            )
        ).strip()
        reason = str(release_row.get("reason", "")).strip()
        scope_token = f"{_slug(scope)}__" if scope else ""
        base_id = (
            f"adjusted_effect__{scope_token}{_slug(block)}__{_slug(model)}"
        )
        if smoke or not bool(passes.loc[row_index]):
            rows.extend(
                _manifest_rows(
                    root,
                    figure_id=base_id,
                    figure_family="adjusted_effect",
                    scope=scope,
                    panel=panel,
                    block=block,
                    model=model,
                    status="gated",
                    gate_status="smoke_not_scientific" if smoke else "failed",
                    reason=(
                        "smoke mode never releases adjusted effect curves"
                        if smoke
                        else reason or "registered block validation gate did not pass"
                    ),
                )
            )
            continue
        selected = _effect_grid_for_release(grid, release_row)
        feature_column = _column(selected, ("feature_name", "feature"))
        if selected.empty or feature_column is None:
            rows.extend(
                _manifest_rows(
                    root,
                    figure_id=base_id,
                    figure_family="adjusted_effect",
                    scope=scope,
                    panel=panel,
                    block=block,
                    model=model,
                    status="not_generated",
                    gate_status="passed",
                    reason="block passed, but a matching adjusted-effect grid is unavailable",
                )
            )
            continue
        for feature in sorted(selected[feature_column].astype(str).unique()):
            feature_grid = selected.loc[
                selected[feature_column].astype(str).eq(feature)
            ].copy()
            figure_id = f"{base_id}__{_slug(feature)}"
            paths, n_records, n_groups = _effect_curve_figure(
                feature_grid,
                root / "figures" / "effects" / figure_id,
                formats,
                dpi,
                feature=feature,
            )
            rows.extend(
                _manifest_rows(
                    root,
                    figure_id=figure_id,
                    figure_family="adjusted_effect",
                    scope=scope,
                    feature=feature,
                    panel=panel,
                    block=block,
                    model=model,
                    paths=paths,
                    status="generated" if paths else "not_generated",
                    gate_status="passed",
                    reason=(
                        _ADJUSTED_BOUNDARY
                        if paths
                        else "released grid has no finite ordered ratio estimates"
                    ),
                    n_records=n_records,
                    n_groups=n_groups,
                )
            )
    return rows


def generate_feature_association_figures(
    run_dir: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
    smoke: bool = False,
) -> pd.DataFrame:
    """Generate descriptive and gate-controlled feature-association figures."""

    root = Path(run_dir)
    formats = tuple(formats)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    support = _read_csv(root / "audits" / "feature_support_audit.csv")
    redundancy = _read_csv(root / "audits" / "numeric_redundancy_pairs.csv")
    atlas = _read_csv(root / "atlas" / "atlas_observations.csv")
    atlas_summary = _read_csv(root / "atlas" / "atlas_feature_summary.csv")
    block_summary = _read_csv(
        root / "analyses" / "blockwise" / "paired_group_bootstrap_summary.csv"
    )
    release = _read_csv(root / "effects" / "effect_release_registry.csv")
    effect_grid = _read_csv(root / "effects" / "adjusted_effect_grid.csv")

    rows: list[dict[str, object]] = []
    for scope_name, scope_support in _scope_frames(support):
        scope_prefix = f"{_slug(scope_name)}__" if scope_name else ""
        figure_id = f"{scope_prefix}feature_support_records_groups"
        support_paths, support_records, support_groups = _support_figure(
            scope_support,
            figure_dir / "support" / figure_id,
            formats,
            dpi,
            scope=scope_name,
        )
        rows.extend(
            _manifest_rows(
                root,
                figure_id=figure_id,
                figure_family="support",
                scope=scope_name,
                paths=support_paths,
                status="generated" if support_paths else "not_generated",
                reason=(
                    ""
                    if support_paths
                    else "feature support audit is unavailable or empty"
                ),
                n_records=support_records if support_paths else None,
                n_groups=support_groups if support_paths else None,
            )
        )
    for scope_name, scope_redundancy in _scope_frames(redundancy):
        scope_prefix = f"{_slug(scope_name)}__" if scope_name else ""
        figure_id = f"{scope_prefix}numeric_redundancy_heatmap"
        redundancy_paths = _redundancy_figure(
            scope_redundancy,
            figure_dir / "support" / figure_id,
            formats,
            dpi,
            scope=scope_name,
        )
        rows.extend(
            _manifest_rows(
                root,
                figure_id=figure_id,
                figure_family="support",
                scope=scope_name,
                paths=redundancy_paths,
                status="generated" if redundancy_paths else "not_generated",
                reason=(
                    ""
                    if redundancy_paths
                    else "numeric redundancy pairs are unavailable or not estimable"
                ),
            )
        )

    if atlas.empty:
        rows.extend(
            _manifest_rows(
                root,
                figure_id="posterior_aware_atlas",
                figure_family="atlas",
                status="not_generated",
                reason="atlas observations are unavailable",
            )
        )
    else:
        _validate_atlas_grain(atlas, _atlas_columns(atlas))
        atlas_for_plot, skipped_atlas = _filter_atlas_by_registered_status(
            atlas, atlas_summary
        )
        for _, skipped in skipped_atlas.iterrows():
            scope_name = str(skipped["_scope_key"])
            feature_name = str(skipped["_feature_key"])
            status_value = str(
                skipped.iloc[-1]
            )
            rows.extend(
                _manifest_rows(
                    root,
                    figure_id=(
                        f"atlas__{_slug(scope_name)}__{_slug(feature_name)}"
                    ),
                    figure_family="atlas",
                    scope=scope_name,
                    feature=feature_name,
                    status="gated",
                    gate_status="insufficient_support",
                    reason=f"registered atlas status: {status_value}",
                )
            )
        for metadata, paths in _atlas_pages(
            atlas_for_plot,
            atlas_summary,
            figure_dir / "atlas",
            formats,
            dpi,
        ):
            rows.extend(
                _manifest_rows(
                    root,
                    figure_id=str(metadata["figure_id"]),
                    figure_family="atlas",
                    scope=str(metadata["scope"]),
                    feature=str(metadata["feature"]),
                    paths=paths,
                    reason=_MARGINAL_BOUNDARY,
                    n_records=int(metadata["n_records"]),
                    n_groups=int(metadata["n_groups"]),
                )
            )

    if smoke:
        rows.extend(
            _manifest_rows(
                root,
                figure_id="blockwise_primary_delta",
                figure_family="block_comparison",
                status="gated",
                gate_status="smoke_not_scientific",
                reason="smoke mode does not publish block-performance comparisons",
            )
        )
    else:
        block_paths, metric = _block_delta_figure(
            block_summary,
            figure_dir / "comparison" / "blockwise_primary_delta",
            formats,
            dpi,
        )
        rows.extend(
            _manifest_rows(
                root,
                figure_id="blockwise_primary_delta",
                figure_family="block_comparison",
                paths=block_paths,
                status="generated" if block_paths else "not_generated",
                reason=(
                    f"primary metric: {metric}"
                    if block_paths
                    else "paired connected-group bootstrap summary is unavailable"
                ),
            )
        )

    rows.extend(
        _effect_figures(
            release,
            effect_grid,
            root,
            formats,
            dpi,
            smoke=smoke,
        )
    )
    manifest = pd.DataFrame(rows, columns=_MANIFEST_COLUMNS)
    manifest_path = figure_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest
