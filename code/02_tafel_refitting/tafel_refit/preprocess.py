from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class ProcessedCurve:
    curve_id: str
    sample_record: int
    raw_points: pd.DataFrame
    binned_points: pd.DataFrame
    metrics: dict[str, Any]
    fit_eligible: bool
    exclusion_reason: str
    qc_flags: tuple[str, ...]
    bin_width_V: float


def _longest_monotonic_branch(points: pd.DataFrame) -> tuple[pd.DataFrame, str, int]:
    """Select one contiguous acquisition branch without joining a reversal step.

    Direction runs are detected locally.  The first step after a direction
    change is treated as the boundary between branches, rather than allowing a
    large scan-reset jump to inflate the span of the following branch.
    """
    ordered = points.sort_values("excel_row", kind="stable").copy()
    potential = ordered["potential_raw"].to_numpy(dtype=float)
    if len(potential) < 2:
        ordered["branch_selected"] = True
        return ordered, "insufficient_for_direction", 1

    diffs = np.diff(potential)
    nonzero = np.abs(diffs) > 1e-12
    if not nonzero.any():
        ordered["branch_selected"] = True
        return ordered, "constant_potential", 1

    median_step = float(np.median(np.abs(diffs[nonzero])))
    tolerance = max(1e-9, 0.10 * median_step)
    directions = np.where(diffs > tolerance, 1, np.where(diffs < -tolerance, -1, 0))
    directional = np.flatnonzero(directions)
    if not len(directional):
        ordered["branch_selected"] = True
        return ordered, "constant_potential", 1
    first_directional = int(directional[0])
    directions[:first_directional] = directions[first_directional]
    for index in range(first_directional + 1, len(directions)):
        if directions[index] == 0:
            directions[index] = directions[index - 1]

    changes = np.flatnonzero(directions[1:] != directions[:-1]) + 1
    diff_starts = np.r_[0, changes]
    diff_stops = np.r_[changes, len(diffs)]
    candidates: list[tuple[float, int, int, int, int, int]] = []
    for run_index, (diff_start, diff_stop) in enumerate(
        zip(diff_starts, diff_stops, strict=True)
    ):
        # For every run after the first, skip the cross-direction step that
        # entered the run.  This prevents a scan reset from being counted as
        # observed within-branch potential support.
        point_start = int(diff_start) if run_index == 0 else int(diff_start) + 1
        point_stop = int(diff_stop) + 1
        segment = potential[point_start:point_stop]
        span = float(np.ptp(segment)) if len(segment) else 0.0
        orientation = int(directions[int(diff_start)])
        candidates.append(
            (
                span,
                point_stop - point_start,
                -point_start,
                point_start,
                point_stop,
                orientation,
            )
        )
    _, _, _, best_start, best_stop, orientation = max(candidates)

    ordered["branch_selected"] = False
    selected_indices = ordered.index[best_start:best_stop]
    ordered.loc[selected_indices, "branch_selected"] = True
    selected = ordered.loc[selected_indices].copy()
    if orientation < 0:
        selected = selected.iloc[::-1].copy()
    direction = "increasing" if orientation > 0 else "decreasing_reversed"
    return selected, direction, len(candidates)


def _bin_positive_run(
    positive: pd.DataFrame,
    *,
    curve_id: str,
    sample_record: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    if positive.empty:
        return pd.DataFrame(), np.nan
    working = positive.copy()
    working["log10_current"] = np.log10(
        working["current_raw"].to_numpy(dtype=float)
    )
    potential_min = float(working["potential_raw"].min())
    potential_span = float(working["potential_raw"].max() - potential_min)
    bin_width = max(
        float(config["minimum_bin_width_V"]),
        potential_span / int(config["max_effective_bins"])
        if potential_span > 0
        else 0.0,
    )
    working["potential_bin_index"] = np.floor(
        (working["potential_raw"].to_numpy(dtype=float) - potential_min)
        / bin_width
        + 1e-12
    ).astype(int)
    binned = (
        working.groupby("potential_bin_index", sort=True)
        .agg(
            potential_V=("potential_raw", "median"),
            log10_current=("log10_current", "median"),
            n_raw_points=("excel_row", "size"),
            first_excel_row=("excel_row", "min"),
            last_excel_row=("excel_row", "max"),
        )
        .reset_index()
    )
    binned["relative_potential_V"] = (
        binned["potential_V"] - binned["potential_V"].min()
    )
    binned.insert(0, "sample_record", sample_record)
    binned.insert(0, "curve_id", curve_id)
    return binned, bin_width


def _positive_region_quality(
    binned: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    if binned.empty:
        fit_span = 0.0
        log_span = 0.0
        rho = np.nan
    else:
        fit_span = float(binned["potential_V"].max() - binned["potential_V"].min())
        log_span = float(
            binned["log10_current"].max() - binned["log10_current"].min()
        )
        rho = (
            float(
                spearmanr(
                    binned["potential_V"], binned["log10_current"]
                ).statistic
            )
            if len(binned) >= 2
            and binned["potential_V"].nunique() >= 2
            and binned["log10_current"].nunique() >= 2
            else np.nan
        )

    reasons: list[str] = []
    if len(binned) < int(config["minimum_fit_bins"]):
        reasons.append("too_few_effective_bins")
    if fit_span < float(config["minimum_fit_span_V"]):
        reasons.append("insufficient_potential_span")
    if log_span < float(config["minimum_log_current_span_dec"]):
        reasons.append("insufficient_log_current_span")
    if (
        not np.isfinite(rho)
        or rho <= float(config["minimum_spearman_logj_vs_potential"])
    ):
        reasons.append("insufficient_positive_region_monotonicity")
    return {
        "fit_potential_span_V": fit_span,
        "fit_log_current_span_dec": log_span,
        "spearman_logj_vs_potential": rho,
        "n_effective_bins": float(len(binned)),
    }, reasons


def _contiguous_positive_runs(selected: pd.DataFrame) -> list[tuple[int, int]]:
    positive = selected["current_raw"].to_numpy(dtype=float) > 0
    if not positive.any():
        return []
    starts = np.flatnonzero(positive & np.r_[True, ~positive[:-1]])
    stops = np.flatnonzero(positive & np.r_[~positive[1:], True]) + 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def preprocess_curve(
    curve_manifest: pd.Series,
    curve_points: pd.DataFrame,
    config: dict[str, Any],
) -> ProcessedCurve:
    curve_id = str(curve_manifest["curve_id"])
    sample_record = int(curve_manifest["sample_record"])
    points = curve_points.copy().sort_values("excel_row", kind="stable")
    points["branch_selected"] = False
    points["used_for_log_fit"] = False
    points["left_censored_nonpositive"] = False
    points["pre_reliable_onset_positive"] = False
    points["nonpositive_after_first_numeric_positive"] = False
    points["nonpositive_after_first_reliable_positive"] = False
    flags: list[str] = []

    finite = points.loc[points["finite_pair"]].copy()
    if len(finite) != len(points):
        flags.append("nonfinite_pairs_excluded")
    selected, potential_order, branch_count = _longest_monotonic_branch(finite)
    points.loc[selected.index, "branch_selected"] = True
    if branch_count > 1:
        flags.append("multiple_potential_branches")

    positive_runs = _contiguous_positive_runs(selected)
    run_audits: list[dict[str, Any]] = []
    for ordinal, (start, stop) in enumerate(positive_runs, start=1):
        run_points = selected.iloc[start:stop].copy()
        run_binned, run_bin_width = _bin_positive_run(
            run_points,
            curve_id=curve_id,
            sample_record=sample_record,
            config=config,
        )
        run_metrics, run_reasons = _positive_region_quality(run_binned, config)
        run_audits.append(
            {
                "ordinal": ordinal,
                "start": start,
                "stop": stop,
                "points": run_points,
                "binned": run_binned,
                "bin_width": run_bin_width,
                "metrics": run_metrics,
                "reasons": run_reasons,
            }
        )

    reliable = next((audit for audit in run_audits if not audit["reasons"]), None)
    if reliable is not None:
        chosen = reliable
    elif run_audits:
        chosen = max(
            run_audits,
            key=lambda audit: (
                -len(audit["reasons"]),
                audit["metrics"]["fit_potential_span_V"],
                audit["metrics"]["n_effective_bins"],
            ),
        )
    else:
        empty_binned, empty_bin_width = _bin_positive_run(
            selected.iloc[0:0],
            curve_id=curve_id,
            sample_record=sample_record,
            config=config,
        )
        empty_metrics, empty_reasons = _positive_region_quality(empty_binned, config)
        chosen = {
            "ordinal": 0,
            "start": 0,
            "stop": 0,
            "points": selected.iloc[0:0],
            "binned": empty_binned,
            "bin_width": empty_bin_width,
            "metrics": empty_metrics,
            "reasons": empty_reasons,
        }

    positive_selected = chosen["points"]
    binned = chosen["binned"]
    bin_width = float(chosen["bin_width"])
    fit_span = float(chosen["metrics"]["fit_potential_span_V"])
    log_span = float(chosen["metrics"]["fit_log_current_span_dec"])
    rho = float(chosen["metrics"]["spearman_logj_vs_potential"])

    selected_positive_mask = selected["current_raw"].to_numpy(dtype=float) > 0
    numeric_positive_positions = np.flatnonzero(selected_positive_mask)
    first_numeric_position = (
        int(numeric_positive_positions[0]) if len(numeric_positive_positions) else None
    )
    numeric_post_nonpositive = (
        selected.iloc[first_numeric_position:].loc[
            ~selected_positive_mask[first_numeric_position:]
        ]
        if first_numeric_position is not None
        else selected.iloc[0:0]
    )
    points.loc[
        numeric_post_nonpositive.index,
        "nonpositive_after_first_numeric_positive",
    ] = True

    if reliable is not None:
        reliable_start = int(reliable["start"])
        pre_reliable = selected.iloc[:reliable_start]
        post_reliable_nonpositive = selected.iloc[reliable_start:].loc[
            selected.iloc[reliable_start:]["current_raw"] <= 0
        ]
        first_reliable_potential = float(
            selected.iloc[reliable_start]["potential_raw"]
        )
    else:
        reliable_start = None
        pre_reliable = selected
        post_reliable_nonpositive = selected.iloc[0:0]
        first_reliable_potential = np.nan
    leading_censored = pre_reliable.loc[pre_reliable["current_raw"] <= 0]
    pre_reliable_positive = pre_reliable.loc[pre_reliable["current_raw"] > 0]
    points.loc[leading_censored.index, "left_censored_nonpositive"] = True
    points.loc[
        pre_reliable_positive.index,
        "pre_reliable_onset_positive",
    ] = True
    points.loc[
        post_reliable_nonpositive.index,
        "nonpositive_after_first_reliable_positive",
    ] = True

    n_nonpositive = int((selected["current_raw"] <= 0).sum())
    n_leading_censored = int(len(leading_censored))
    n_pre_reliable_positive = int(len(pre_reliable_positive))
    n_post_numeric_nonpositive = int(len(numeric_post_nonpositive))
    n_post_reliable_nonpositive = int(len(post_reliable_nonpositive))
    nonpositive_fraction = n_nonpositive / len(selected) if len(selected) else np.nan
    if n_nonpositive:
        flags.append("nonpositive_current_not_log_transformed")
    if n_leading_censored:
        flags.extend(
            [
                "leading_nonpositive_left_censored_status_only",
                "left_censoring_limit_unverified",
            ]
        )
    if n_pre_reliable_positive:
        flags.append("pre_reliable_onset_positive_not_fit")
    if n_post_numeric_nonpositive:
        flags.append("baseline_crossings_before_reliable_positive_region")
    if n_post_reliable_nonpositive:
        flags.append("nonpositive_after_first_reliable_positive")

    reasons = list(chosen["reasons"]) if reliable is None else []
    if n_post_reliable_nonpositive:
        reasons.append("nonpositive_after_first_reliable_positive")
    if "insufficient_positive_region_monotonicity" in reasons:
        flags.append("insufficient_positive_region_monotonicity")
    manual_review_rho = float(config["manual_review_spearman_below"])
    minimum_rho = float(config["minimum_spearman_logj_vs_potential"])
    if np.isfinite(rho) and minimum_rho < rho < manual_review_rho:
        flags.append("positive_region_monotonicity_manual_review")
    if log_span < 1.0:
        flags.append("low_log_current_coverage")
    if potential_order != "increasing":
        flags.append(f"potential_order_{potential_order}")
    if int(curve_manifest["n_unique_potential"]) < int(curve_manifest["n_finite_pairs"]):
        flags.append("duplicate_potential_values")
    flags.extend(
        [
            "potential_unit_inferred_V_unverified",
            "potential_reference_unknown",
            "ir_correction_unknown",
        ]
    )
    if curve_manifest["current_unit_raw"] == "unknown_current_density":
        flags.append("current_density_unit_unknown")
    fit_eligible = reliable is not None and not reasons
    if fit_eligible:
        points.loc[positive_selected.index, "used_for_log_fit"] = True
    exclusion_reason = ";".join(reasons)

    left_censoring_display_proxy = (
        float(positive_selected["current_raw"].min())
        if len(positive_selected)
        else np.nan
    )
    first_numeric_potential = (
        float(selected.iloc[first_numeric_position]["potential_raw"])
        if first_numeric_position is not None
        else np.nan
    )

    metrics = {
        "potential_order": potential_order,
        "n_potential_branches": branch_count,
        "n_branch_points": len(selected),
        "n_positive_branch_points": int(selected_positive_mask.sum()),
        "n_positive_runs": len(positive_runs),
        "reliable_positive_region_found": reliable is not None,
        "reliable_positive_run_ordinal": (
            int(reliable["ordinal"]) if reliable is not None else np.nan
        ),
        "n_reliable_positive_points": (
            len(positive_selected) if reliable is not None else 0
        ),
        "n_pre_reliable_onset_points": len(pre_reliable),
        "n_pre_reliable_positive_points": n_pre_reliable_positive,
        "n_effective_bins": len(binned),
        "n_nonpositive_selected_branch": n_nonpositive,
        "n_leading_left_censored": n_leading_censored,
        "n_nonpositive_after_first_numeric_positive": n_post_numeric_nonpositive,
        "n_nonpositive_after_first_reliable_positive": n_post_reliable_nonpositive,
        "nonpositive_confined_to_pre_reliable_region": (
            reliable is not None and n_post_reliable_nonpositive == 0
        ),
        "nonpositive_fraction_selected_branch": nonpositive_fraction,
        "first_numeric_positive_potential_raw": first_numeric_potential,
        "first_reliable_positive_potential_raw": first_reliable_potential,
        "left_censoring_display_proxy_raw": left_censoring_display_proxy,
        "left_censoring_limit_status": config["left_censoring_limit_status"],
        "fit_potential_span_V": fit_span,
        "fit_log_current_span_dec": log_span,
        "spearman_logj_vs_potential": rho,
        "potential_bin_width_V": bin_width,
    }
    return ProcessedCurve(
        curve_id=curve_id,
        sample_record=sample_record,
        raw_points=points,
        binned_points=binned,
        metrics=metrics,
        fit_eligible=fit_eligible,
        exclusion_reason=exclusion_reason,
        qc_flags=tuple(sorted(set(flags))),
        bin_width_V=bin_width,
    )
