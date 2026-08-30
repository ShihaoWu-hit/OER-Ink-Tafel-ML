from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from .models import (
    FitResult,
    MODEL_PARAMETER_NAMES,
    PRIMARY_MODEL_NAMES,
    blocked_cross_validation,
    fit_model,
    fit_paper_exact,
    local_slopes_mV_dec,
    model_spec,
    model_structure_fields,
    moving_block_bootstrap,
    predict_log10,
    regime_probabilities,
)
from .preprocess import ProcessedCurve


@dataclass(frozen=True)
class CurveAnalysis:
    summary: dict[str, Any]
    model_fits: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    local_slopes: pd.DataFrame
    fitted_curves: pd.DataFrame
    fit_results: dict[str, FitResult]


def _slope_point(fit: FitResult) -> float:
    if (
        not fit.parameters
        or model_spec(fit.model_name).kinetic_structure != "constant"
    ):
        return np.nan
    return float(1000.0 / np.exp(fit.parameters["log_m"]))


def _quantile(values: pd.Series | np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if len(array) else np.nan


def _plateau_identified(fit: FitResult, y: np.ndarray) -> bool:
    if not model_spec(fit.model_name).has_limiting_current or not fit.parameters:
        return False
    return bool(
        fit.parameters["L"] - float(np.max(y)) <= 2.0
        and not any(hit.startswith("L_") for hit in fit.boundary_hits)
    )


def _scientific_boundary_prefixes(model_name: str) -> tuple[str, ...]:
    try:
        return tuple(f"{name}_" for name in MODEL_PARAMETER_NAMES[model_name])
    except KeyError as exc:
        raise ValueError(f"Unknown model for boundary audit: {model_name}") from exc


def _point_scientifically_identifiable(fit: FitResult) -> bool:
    prefixes = _scientific_boundary_prefixes(fit.model_name)
    return bool(
        fit.success
        and
        fit.parameters
        and not any(hit.startswith(prefixes) for hit in fit.boundary_hits)
    )


def _scientific_boundary_fraction(draws: pd.DataFrame, model_name: str) -> float:
    if draws.empty or "success" not in draws:
        return np.nan
    successful = draws.loc[draws["success"].fillna(False)]
    if successful.empty:
        return np.nan
    prefixes = _scientific_boundary_prefixes(model_name)
    boundary_text = successful["boundary_hits"].fillna("").astype(str)
    return float(
        boundary_text.map(
            lambda text: any(
                token.startswith(prefixes)
                for token in text.split(";")
                if token
            )
        ).mean()
    )


def _segment_supported(
    fit: FitResult, x: np.ndarray, config: dict[str, Any]
) -> bool:
    if (
        model_spec(fit.model_name).kinetic_structure != "potential_dependent"
        or not fit.parameters
    ):
        return False
    c = fit.parameters["c"]
    left = x <= c
    right = x > c
    if left.sum() < int(config["minimum_segment_bins"]):
        return False
    if right.sum() < int(config["minimum_segment_bins"]):
        return False
    if float(np.ptp(x)) < float(config["minimum_segment_total_span_V"]):
        return False
    if float(np.ptp(x[left])) < float(config["minimum_segment_span_V"]):
        return False
    if float(np.ptp(x[right])) < float(config["minimum_segment_span_V"]):
        return False
    return True


def _bootstrap_success(draws: pd.DataFrame) -> float:
    if draws.empty:
        return np.nan
    return float(draws["success"].fillna(False).mean())


def _draw_parameters(model_name: str, row: pd.Series) -> dict[str, float]:
    return {
        name: float(row[name]) for name in MODEL_PARAMETER_NAMES[model_name]
    }


def _spline_local_slope(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(x) < 5:
        return np.full_like(grid, np.nan)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    second_difference = np.diff(y, n=2)
    if len(second_difference):
        median = np.median(second_difference)
        noise = np.median(np.abs(second_difference - median)) / 0.6745 / np.sqrt(6.0)
    else:
        noise = 0.05
    noise = float(np.clip(noise, 0.02, 0.30))
    try:
        spline = UnivariateSpline(x, y, k=min(3, len(x) - 1), s=len(x) * noise * noise)
        derivative = spline.derivative()(grid)
        return np.where(derivative > 1e-6, 1000.0 / derivative, np.nan)
    except Exception:
        return np.full_like(grid, np.nan)


def analyze_curve(
    processed: ProcessedCurve,
    manifest_row: pd.Series,
    config: dict[str, Any],
    *,
    bootstrap_repetitions: int,
    optimizer_starts: int,
    seed: int,
) -> CurveAnalysis:
    base_summary: dict[str, Any] = {
        "curve_id": processed.curve_id,
        "sample_record": processed.sample_record,
        "curve_variant": int(manifest_row["curve_variant"]),
        "sheet_index": int(manifest_row["sheet_index"]),
        "sheet_name": manifest_row["sheet_name"],
        "tafel_slope_source": manifest_row["tafel_slope_source"],
        "publication_id": manifest_row.get("publication_id"),
        "doi": manifest_row.get("doi"),
        "workbook_doi": manifest_row.get("workbook_doi"),
        "existing_tafel_slope_mV_dec": manifest_row["y_tafel_slope_mV_dec"],
        "potential_unit_raw": manifest_row["potential_unit_raw"],
        "potential_unit_assumption": manifest_row["potential_unit_assumption"],
        "current_unit_raw": manifest_row["current_unit_raw"],
        "potential_reference_raw": manifest_row["potential_reference_raw"],
        "ir_status": manifest_row["ir_status"],
        "fit_eligible": processed.fit_eligible,
        "fit_exclusion_reason": processed.exclusion_reason,
        "qc_flags": ";".join(processed.qc_flags),
        **processed.metrics,
    }
    if not processed.fit_eligible:
        base_summary.update(
            fit_status="excluded_by_qc",
            selected_model="none",
            selected_model_display_name="not fit",
            model_family="continuous_tafel",
            kinetic_structure="not_fit",
            observation_parameterization="not_fit",
            low_current_floor_policy="excluded_from_primary",
            model_role="not_fit",
            potential_dependent_supported=False,
            label_eligible=False,
            label_reason="fit_not_eligible",
        )
        return CurveAnalysis(
            base_summary,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )

    binned = processed.binned_points
    x = binned["relative_potential_V"].to_numpy(dtype=float)
    y = binned["log10_current"].to_numpy(dtype=float)
    transition_width = max(
        float(config["transition_width_min_V"]), 2.0 * processed.bin_width_V
    )
    fit_results: dict[str, FitResult] = {}
    cv_results: dict[str, dict[str, float]] = {}
    model_rows: list[dict[str, Any]] = []
    model_names = list(PRIMARY_MODEL_NAMES)
    for model_name in model_names:
        model_seed_offset = model_spec(model_name).analysis_seed_offset
        fit = fit_model(
            model_name,
            x,
            y,
            transition_width_V=transition_width,
            n_starts=optimizer_starts,
            seed=seed + model_seed_offset * 1009,
        )
        fit_results[model_name] = fit
        cv = blocked_cross_validation(
            model_name,
            x,
            y,
            transition_width_V=transition_width,
            n_starts=int(config["optimizer_starts_cv"]),
            seed=seed + model_seed_offset * 2003,
        )
        cv_results[model_name] = cv
        row: dict[str, Any] = {
            "curve_id": processed.curve_id,
            "sample_record": processed.sample_record,
            "model_name": model_name,
            "success": fit.success,
            "nll": fit.nll,
            "bic": fit.bic,
            "mae": fit.mae,
            "boundary_hits": ";".join(fit.boundary_hits),
            "optimizer_message": fit.optimizer_message,
            "plateau_identified": _plateau_identified(fit, y),
            "model_display_name": model_spec(model_name).display_name,
            **model_structure_fields(model_name),
            **cv,
            **fit.parameters,
        }
        if model_spec(model_name).kinetic_structure == "constant":
            row["slope_mV_dec"] = _slope_point(fit)
        else:
            row["slope_low_mV_dec"] = (
                1000.0 / np.exp(fit.parameters["log_m1"])
                if fit.parameters
                else np.nan
            )
            row["slope_high_mV_dec"] = (
                1000.0 / np.exp(fit.parameters["log_m2"])
                if fit.parameters
                else np.nan
            )
            row["change_potential_relative_V"] = (
                fit.parameters["c"] if fit.parameters else np.nan
            )
            row["segment_support_sufficient"] = _segment_supported(fit, x, config)
        model_rows.append(row)

    raw_used = processed.raw_points.loc[processed.raw_points["used_for_log_fit"]].copy()
    raw_x = raw_used["potential_raw"].to_numpy(dtype=float)
    raw_x = raw_x - np.min(raw_x)
    raw_y = np.log10(raw_used["current_raw"].to_numpy(dtype=float))
    paper_exact = fit_paper_exact(
        raw_x,
        raw_y,
        n_starts=int(config["paper_exact_optimizer_starts"]),
        seed=seed + 7919,
    )

    linear = fit_results["linear"]
    paper = fit_results["paper_single_plateau"]
    simple_model_names = ("linear", "paper_single_plateau")
    point_successful_simple_names = [
        name
        for name in simple_model_names
        if fit_results[name].success
        and bool(fit_results[name].parameters)
        and np.isfinite(fit_results[name].bic)
    ]
    if not point_successful_simple_names:
        raise RuntimeError("No primary constant-slope fit converged")
    plateau_supported = bool(
        linear.success
        and bool(linear.parameters)
        and paper.success
        and bool(paper.parameters)
        and _plateau_identified(paper, y)
        and linear.bic - paper.bic >= float(config["bic_improvement_min"])
        and cv_results["paper_single_plateau"]["cv_mae"]
        <= 1.02 * cv_results["linear"]["cv_mae"]
    )
    if plateau_supported:
        best_simple_name = "paper_single_plateau"
    elif "linear" in point_successful_simple_names:
        best_simple_name = "linear"
    else:
        best_simple_name = "paper_single_plateau"
    best_simple = fit_results[best_simple_name]
    cv_valid_simple_names = [
        name
        for name in point_successful_simple_names
        if np.isfinite(cv_results[name]["cv_mae"])
        and cv_results[name]["cv_folds_successful"]
        == cv_results[name]["cv_folds_requested"]
    ]
    best_simple_bic_reference = min(
        fit_results[name].bic for name in point_successful_simple_names
    )
    best_simple_cv_reference = (
        min(cv_results[name]["cv_mae"] for name in cv_valid_simple_names)
        if cv_valid_simple_names
        else np.inf
    )

    complex_model_name = "two_regime_plateau"
    two_regime_selected_form = fit_results[complex_model_name]
    complex_cv = cv_results[complex_model_name]
    segment_support = _segment_supported(two_regime_selected_form, x, config)
    complex_model_point_identifiable = _point_scientifically_identifiable(
        two_regime_selected_form
    )
    folds_ok = (
        complex_cv["cv_folds_successful"] == complex_cv["cv_folds_requested"]
    )
    best_single_bic_gain = best_simple_bic_reference - two_regime_selected_form.bic
    m2_cv_improvement_vs_best_single = (
        (best_simple_cv_reference - complex_cv["cv_mae"])
        / best_simple_cv_reference
        if np.isfinite(best_simple_cv_reference) and best_simple_cv_reference > 0
        else -np.inf
    )
    m2_candidate = bool(
        two_regime_selected_form.success
        and complex_model_point_identifiable
        and segment_support
        and folds_ok
        and best_single_bic_gain >= float(config["bic_improvement_min"])
        and m2_cv_improvement_vs_best_single
        >= float(config["cv_improvement_fraction_min"])
    )
    m2_bic_gain = paper.bic - two_regime_selected_form.bic
    paper_cv = cv_results["paper_single_plateau"]["cv_mae"]
    m2_cv = complex_cv["cv_mae"]
    m2_cv_improvement = (
        (paper_cv - m2_cv) / paper_cv if np.isfinite(paper_cv) and paper_cv > 0 else -np.inf
    )

    block_length = max(
        3,
        int(np.ceil(float(config["bootstrap_block_width_V"]) / processed.bin_width_V)),
    )
    bootstrap_frames: list[pd.DataFrame] = []
    m2_draws = pd.DataFrame()
    conditional_slope_change_probability = np.nan
    if m2_candidate:
        m2_draws = moving_block_bootstrap(
            two_regime_selected_form,
            x,
            y,
            repetitions=bootstrap_repetitions,
            block_length=block_length,
            n_starts=int(config["optimizer_starts_bootstrap"]),
            seed=seed + 100_003,
        )
        if not m2_draws.empty:
            m2_draws.insert(0, "model_name", complex_model_name)
            m2_draws.insert(0, "sample_record", processed.sample_record)
            m2_draws.insert(0, "curve_id", processed.curve_id)
            bootstrap_frames.append(m2_draws)
            successful = m2_draws.loc[m2_draws["success"].fillna(False)]
            if len(successful):
                ratio = successful["slope_high_mV_dec"] / successful["slope_low_mV_dec"]
                threshold = np.log(float(config["slope_ratio_min"]))
                conditional_slope_change_probability = float(
                    np.mean(np.abs(np.log(ratio)) > threshold)
                )

    m2_bootstrap_success = _bootstrap_success(m2_draws)
    complex_model_bootstrap_boundary_fraction = _scientific_boundary_fraction(
        m2_draws, complex_model_name
    )
    potential_dependent = bool(
        m2_candidate
        and complex_model_point_identifiable
        and np.isfinite(conditional_slope_change_probability)
        and conditional_slope_change_probability
        >= float(config["conditional_slope_change_probability_min"])
        and m2_bootstrap_success >= float(config["bootstrap_success_fraction_min"])
        and np.isfinite(complex_model_bootstrap_boundary_fraction)
        and complex_model_bootstrap_boundary_fraction
        <= float(config["bootstrap_boundary_hit_fraction_max"])
    )
    selected_name = complex_model_name if potential_dependent else best_simple_name
    selected = fit_results[selected_name]

    if model_spec(selected_name).kinetic_structure == "potential_dependent":
        selected_draws = m2_draws
    else:
        selected_draws = moving_block_bootstrap(
            selected,
            x,
            y,
            repetitions=bootstrap_repetitions,
            block_length=block_length,
            n_starts=int(config["optimizer_starts_bootstrap"]),
            seed=seed + 200_003,
        )
    if (
        model_spec(selected_name).kinetic_structure == "constant"
        and not selected_draws.empty
    ):
        selected_draws.insert(0, "model_name", selected_name)
        selected_draws.insert(0, "sample_record", processed.sample_record)
        selected_draws.insert(0, "curve_id", processed.curve_id)
        bootstrap_frames.append(selected_draws)

    paper_draws = (
        selected_draws
        if selected_name == "paper_single_plateau"
        else moving_block_bootstrap(
            paper,
            x,
            y,
            repetitions=bootstrap_repetitions,
            block_length=block_length,
            n_starts=int(config["optimizer_starts_bootstrap"]),
            seed=seed + 300_007,
        )
    )
    if selected_name != "paper_single_plateau" and not paper_draws.empty:
        paper_draws.insert(0, "model_name", "paper_single_plateau")
        paper_draws.insert(0, "sample_record", processed.sample_record)
        paper_draws.insert(0, "curve_id", processed.curve_id)
        bootstrap_frames.append(paper_draws)

    all_bootstrap = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames
        else pd.DataFrame()
    )
    selected_success = (
        selected_draws.loc[selected_draws["success"].fillna(False)]
        if not selected_draws.empty
        else pd.DataFrame()
    )
    paper_success = (
        paper_draws.loc[paper_draws["success"].fillna(False)]
        if not paper_draws.empty
        else pd.DataFrame()
    )
    selected_point_identifiable = _point_scientifically_identifiable(selected)
    selected_scientific_boundary_fraction = _scientific_boundary_fraction(
        selected_draws, selected_name
    )

    summary = dict(base_summary)
    summary.update(
        fit_status="success"
        if selected.success and selected.parameters
        else "optimization_failed",
        selected_model=selected_name,
        selected_model_display_name=model_spec(selected_name).display_name,
        paper_exact_raw_points_slope_mV_dec=paper_exact.get("slope_mV_dec", np.nan),
        paper_exact_fit_success=paper_exact.get("success", False),
        paper_robust_binned_slope_point_mV_dec=_slope_point(paper),
        paper_robust_slope_ci_low_mV_dec=_quantile(paper_success.get("slope_mV_dec", []), 0.025),
        paper_robust_slope_ci_high_mV_dec=_quantile(paper_success.get("slope_mV_dec", []), 0.975),
        paper_plateau_identified=_plateau_identified(paper, y),
        selected_plateau_identified=_plateau_identified(selected, y),
        primary_model_count=len(PRIMARY_MODEL_NAMES),
        **model_structure_fields(selected_name),
        potential_dependent_supported=potential_dependent,
        conditional_slope_change_probability=conditional_slope_change_probability,
        potential_dependence_model_gate_passed=m2_candidate,
        leading_left_censored_status_only=bool(
            processed.metrics.get("n_leading_left_censored", 0)
        ),
        complex_model_point_identifiable=complex_model_point_identifiable,
        complex_model_bootstrap_boundary_fraction=complex_model_bootstrap_boundary_fraction,
        m2_bic_gain_vs_paper=m2_bic_gain,
        m2_blocked_cv_improvement_fraction=m2_cv_improvement,
        low_current_floor_primary_policy="excluded_without_independent_blank",
        low_current_floor_sensitivity_implementation_available=True,
        low_current_floor_sensitivity_run=False,
        potential_dependence_floor_ambiguity=(
            "requires_separate_sensitivity_or_independent_blank"
            if potential_dependent
            else "not_triggered"
        ),
        best_single_regime_model=best_simple_name,
        m2_bic_gain_vs_best_single=best_single_bic_gain,
        m2_blocked_cv_improvement_vs_best_single=m2_cv_improvement_vs_best_single,
        potential_dependence_model=complex_model_name,
        segment_support_sufficient=segment_support,
        bootstrap_repetitions_requested=bootstrap_repetitions,
        selected_bootstrap_success_fraction=_bootstrap_success(selected_draws),
        selected_point_identifiable=selected_point_identifiable,
        selected_scientific_boundary_fraction=selected_scientific_boundary_fraction,
        paper_bootstrap_success_fraction=_bootstrap_success(paper_draws),
        bootstrap_block_length_bins=block_length,
        transition_width_V=transition_width,
        selected_boundary_hits=";".join(selected.boundary_hits),
    )

    if model_spec(selected_name).kinetic_structure == "potential_dependent":
        low = selected_success.get("slope_low_mV_dec", pd.Series(dtype=float))
        high = selected_success.get("slope_high_mV_dec", pd.Series(dtype=float))
        summary.update(
            refit_tafel_slope_mV_dec=np.nan,
            slope_low_point_mV_dec=1000.0 / np.exp(selected.parameters["log_m1"]),
            slope_low_ci_low_mV_dec=_quantile(low, 0.025),
            slope_low_ci_high_mV_dec=_quantile(low, 0.975),
            slope_high_point_mV_dec=1000.0 / np.exp(selected.parameters["log_m2"]),
            slope_high_ci_low_mV_dec=_quantile(high, 0.025),
            slope_high_ci_high_mV_dec=_quantile(high, 0.975),
            change_potential_relative_V=selected.parameters["c"],
            label_eligible=False,
            label_reason="potential_dependent_requires_two_labels",
            **{f"low_{key}": value for key, value in regime_probabilities(low).items()},
            **{f"high_{key}": value for key, value in regime_probabilities(high).items()},
        )
    else:
        slopes = selected_success.get("slope_mV_dec", pd.Series(dtype=float))
        probabilities = regime_probabilities(slopes)
        max_probability = max(
            (value for value in probabilities.values() if np.isfinite(value)),
            default=np.nan,
        )
        bootstrap_ok = _bootstrap_success(selected_draws) >= float(
            config["bootstrap_success_fraction_min"]
        )
        label_eligible = bool(
            bootstrap_ok
            and selected_point_identifiable
            and np.isfinite(selected_scientific_boundary_fraction)
            and selected_scientific_boundary_fraction
            <= float(config["bootstrap_boundary_hit_fraction_max"])
            and np.isfinite(max_probability)
            and max_probability >= float(config["label_probability_min"])
        )
        if not bootstrap_ok:
            label_reason = "low_bootstrap_success"
        elif not selected_point_identifiable:
            label_reason = "point_estimate_at_parameter_boundary"
        elif (
            not np.isfinite(selected_scientific_boundary_fraction)
            or selected_scientific_boundary_fraction
            > float(config["bootstrap_boundary_hit_fraction_max"])
        ):
            label_reason = "bootstrap_parameter_boundary_instability"
        else:
            label_reason = "ambiguous_regime_probability"
        summary.update(
            refit_tafel_slope_mV_dec=_quantile(slopes, 0.5)
            if len(slopes)
            else _slope_point(selected),
            refit_slope_ci_low_mV_dec=_quantile(slopes, 0.025),
            refit_slope_ci_high_mV_dec=_quantile(slopes, 0.975),
            label_eligible=label_eligible,
            label_reason="eligible" if label_eligible else label_reason,
            **probabilities,
        )

    grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    fitted_frames: list[pd.DataFrame] = []
    for model_name, fit in fit_results.items():
        predicted = (
            predict_log10(model_name, grid, fit.parameters, transition_width)
            if fit.success and fit.parameters
            else np.full_like(grid, np.nan)
        )
        fitted_frames.append(
            pd.DataFrame(
                {
                    "curve_id": processed.curve_id,
                    "sample_record": processed.sample_record,
                    "model_name": model_name,
                    "model_display_name": model_spec(model_name).display_name,
                    **model_structure_fields(model_name),
                    "relative_potential_V": grid,
                    "predicted_log10_current": predicted,
                }
            )
        )
    fitted_curves = pd.concat(fitted_frames, ignore_index=True)

    kinetic_b, apparent_b = local_slopes_mV_dec(
        selected_name, grid, selected.parameters, transition_width
    )
    local = pd.DataFrame(
        {
            "curve_id": processed.curve_id,
            "sample_record": processed.sample_record,
            "selected_model": selected_name,
            **model_structure_fields(selected_name),
            "relative_potential_V": grid,
            "potential_V": grid + float(binned["potential_V"].min()),
            "kinetic_slope_mV_dec": kinetic_b,
            "apparent_slope_mV_dec": apparent_b,
            "spline_apparent_slope_mV_dec": _spline_local_slope(x, y, grid),
        }
    )
    if not selected_success.empty:
        kinetic_draws = []
        apparent_draws = []
        for _, draw in selected_success.iterrows():
            parameters = _draw_parameters(selected_name, draw)
            kb, ab = local_slopes_mV_dec(selected_name, grid, parameters, transition_width)
            kinetic_draws.append(kb)
            apparent_draws.append(ab)
        for label, values in (
            ("kinetic", np.asarray(kinetic_draws)),
            ("apparent", np.asarray(apparent_draws)),
        ):
            local[f"{label}_slope_ci_low_mV_dec"] = np.nanquantile(values, 0.025, axis=0)
            local[f"{label}_slope_ci_high_mV_dec"] = np.nanquantile(values, 0.975, axis=0)

    return CurveAnalysis(
        summary=summary,
        model_fits=pd.DataFrame(model_rows),
        bootstrap_draws=all_bootstrap,
        local_slopes=local,
        fitted_curves=fitted_curves,
        fit_results=fit_results,
    )
