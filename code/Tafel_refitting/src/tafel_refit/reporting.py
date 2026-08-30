from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIME_COLUMNS = ["q_low_40_60", "q_120_like", "q_high_gt120"]


def _argmax_regime(row: pd.Series) -> str:
    values = {column.removeprefix("q_"): row.get(column, np.nan) for column in REGIME_COLUMNS}
    finite = {key: value for key, value in values.items() if np.isfinite(value)}
    return max(finite, key=finite.get) if finite else "unknown"


def aggregate_sample_candidates(
    master: pd.DataFrame, curve_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for sample_record, group in curve_summary.groupby("sample_record", sort=True):
        group = group.sort_values("curve_variant")
        fit_success = group["fit_status"].eq("success")
        potential_dependent = group["potential_dependent_supported"].fillna(False)
        eligible = group["label_eligible"].fillna(False)
        slopes = pd.to_numeric(group.get("refit_tafel_slope_mV_dec"), errors="coerce")
        candidate = np.nan
        ci_low = np.nan
        ci_high = np.nan
        label = "unknown"
        if not fit_success.all():
            status = "one_or_more_curve_fits_failed"
        elif potential_dependent.any():
            status = "potential_dependent_no_single_scalar"
        elif len(group) == 1 and bool(eligible.iloc[0]) and np.isfinite(slopes.iloc[0]):
            status = "eligible_single_curve"
            candidate = float(slopes.iloc[0])
            ci_low = float(group.iloc[0].get("refit_slope_ci_low_mV_dec", np.nan))
            ci_high = float(group.iloc[0].get("refit_slope_ci_high_mV_dec", np.nan))
            label = _argmax_regime(group.iloc[0])
        elif len(group) > 1 and eligible.all() and slopes.notna().all():
            labels = [_argmax_regime(row) for _, row in group.iterrows()]
            relative_difference = float((slopes.max() - slopes.min()) / np.median(slopes))
            if len(set(labels)) == 1 and relative_difference <= 0.25:
                status = "eligible_multiple_curves_agree"
                candidate = float(np.median(slopes))
                ci_low = float(
                    pd.to_numeric(group["refit_slope_ci_low_mV_dec"], errors="coerce").min()
                )
                ci_high = float(
                    pd.to_numeric(group["refit_slope_ci_high_mV_dec"], errors="coerce").max()
                )
                label = labels[0]
            else:
                status = "multiple_curves_disagree"
        else:
            status = "ambiguous_or_unreliable_single_regime"

        rows.append(
            {
                "sample_record": int(sample_record),
                "n_curves": len(group),
                "curve_ids": ";".join(group["curve_id"].astype(str)),
                "refit_candidate_status": status,
                "refit_candidate_mV_dec": candidate,
                "refit_candidate_ci_low_mV_dec": ci_low,
                "refit_candidate_ci_high_mV_dec": ci_high,
                "refit_candidate_regime": label,
                "any_potential_dependent_curve": bool(potential_dependent.any()),
                "all_curve_fits_successful": bool(fit_success.all()),
                "all_curves_label_eligible": bool(eligible.all()),
            }
        )
    candidates = pd.DataFrame(rows)
    master_copy = master.copy()
    master_copy["sample_record"] = pd.to_numeric(
        master_copy["sample_record"], errors="raise"
    ).astype(int)
    candidate_dataset = master_copy.merge(
        candidates, on="sample_record", how="left", validate="one_to_one"
    )
    return candidates, candidate_dataset


def _fmt_number(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def write_summary_report(
    output_path: Path,
    *,
    run_metadata: dict[str, Any],
    manifest: pd.DataFrame,
    curve_summary: pd.DataFrame,
    sample_candidates: pd.DataFrame,
) -> None:
    n_curves = len(curve_summary)
    n_samples = curve_summary["sample_record"].nunique()
    success = int(curve_summary["fit_status"].eq("success").sum())
    excluded = int(curve_summary["fit_status"].eq("excluded_by_qc").sum())
    errors = n_curves - success - excluded
    potential_dependent = int(
        curve_summary["potential_dependent_supported"].fillna(False).sum()
    )
    plateau_parameterized = int(
        curve_summary.get(
            "observation_parameterization",
            pd.Series("none", index=curve_summary.index),
        )
        .eq("limiting_current_plateau")
        .sum()
    )
    plateau_identified = int(
        curve_summary.get(
            "selected_plateau_identified",
            pd.Series(False, index=curve_summary.index),
        )
        .fillna(False)
        .sum()
    )
    label_eligible = int(curve_summary["label_eligible"].fillna(False).sum())
    leading_censored_counts = pd.to_numeric(
        curve_summary.get(
            "n_leading_left_censored",
            pd.Series(0, index=curve_summary.index),
        ),
        errors="coerce",
    ).fillna(0)
    post_positive_counts = pd.to_numeric(
        curve_summary.get(
            "n_nonpositive_after_first_reliable_positive",
            pd.Series(0, index=curve_summary.index),
        ),
        errors="coerce",
    ).fillna(0)
    post_numeric_counts = pd.to_numeric(
        curve_summary.get(
            "n_nonpositive_after_first_numeric_positive",
            pd.Series(0, index=curve_summary.index),
        ),
        errors="coerce",
    ).fillna(0)
    pre_reliable_positive_counts = pd.to_numeric(
        curve_summary.get(
            "n_pre_reliable_positive_points",
            pd.Series(0, index=curve_summary.index),
        ),
        errors="coerce",
    ).fillna(0)
    leading_censored_curves = int(leading_censored_counts.gt(0).sum())
    leading_censored_points = int(leading_censored_counts.sum())
    post_positive_failures = int(post_positive_counts.gt(0).sum())
    pre_reliable_crossing_curves = int(post_numeric_counts.gt(0).sum())
    pre_reliable_positive_points = int(pre_reliable_positive_counts.sum())
    monotonicity_failures = int(
        curve_summary["fit_exclusion_reason"]
        .fillna("")
        .astype(str)
        .str.contains("insufficient_positive_region_monotonicity", regex=False)
        .sum()
    )
    monotonicity_reviews = int(
        curve_summary["qc_flags"]
        .fillna("")
        .astype(str)
        .str.contains("positive_region_monotonicity_manual_review", regex=False)
        .sum()
    )
    candidate_eligible = int(sample_candidates["refit_candidate_mV_dec"].notna().sum())
    fitted_structures = curve_summary.loc[
        curve_summary["fit_status"].eq("success"),
        ["kinetic_structure", "observation_parameterization"],
    ].copy()
    structure_counts = (
        fitted_structures.value_counts(sort=False).sort_index()
        if len(fitted_structures)
        else pd.Series(dtype=int)
    )
    nonpositive_curves = int(
        ((manifest["n_zero_current"] + manifest["n_negative_current"]) > 0).sum()
    )
    nonpositive_points = int(
        (manifest["n_zero_current"] + manifest["n_negative_current"]).sum()
    )
    nonfinite_points = int(manifest["n_nonfinite_pairs"].sum())
    source_path = run_metadata["inputs"]["workbook"]
    master_path = run_metadata["inputs"]["canonical_csv"]
    if run_metadata.get("input_hash_verification_bypassed", False):
        hash_verification_note = (
            "Warning: this run explicitly bypassed the pre-registered input SHA256 gate; "
            "the result is for mapping checks only and is not a formal refitting result."
        )
    else:
        hash_verification_note = (
            "Input mapping is executed only when input SHA256 values match the pre-registered hashes; "
            "this run did not bypass that gate."
        )

    lines = [
        "# OER Tafel refitting technical report",
        "",
        "## Technical summary",
        "",
        f"This run analyzed **{n_curves} curves / {n_samples} samples**.",
        f"{success} fits succeeded, {excluded} were excluded by QC, and {errors} returned errors.",
        "",
        f"**{potential_dependent} curves** passed BIC, contiguous-potential blocked cross-validation, and "
        "moving-block bootstrap while being flagged as incompatible with the best constant-slope structure.",
        f"This is exploratory screening evidence without multiplicity/null calibration across {n_curves} curves; "
        "it does not prove a mechanistic transition and does not exclude an unmodeled low-current floor.",
        "",
        f"**{label_eligible} curves** met the single-scalar label-reliability gate; at the sample level, "
        f"{candidate_eligible}/{n_samples} samples produced a refit candidate for manual review.",
        "The original `y_tafel_slope_mV_dec` values were not overwritten.",
        "",
        f"The selected branch contains **{leading_censored_curves} curves / "
        f"{leading_censored_points} leading low-potential non-positive points** flagged as possible left censoring.",
        "This state alone no longer downgrades a curve by point-count proportion.",
        "",
        f"**{post_positive_failures} curves** were excluded because a non-positive value reappeared after the first reliable positive-current region; "
        f"**{monotonicity_failures} curves** were excluded because the positive-current region had non-positive Spearman direction.",
        f"{monotonicity_reviews} weakly positive curves were marked for manual review rather than automatically rejected.",
        "",
        f"{pre_reliable_crossing_curves} curves showed positive/non-positive crossings before the reliable positive region was established, "
        f"with {pre_reliable_positive_points} leading small positive points; these belong to the low-potential onset/quantization region, "
        "are excluded from formal log-current fitting, and do not reject a curve solely because of sign jitter.",
        "",
        "## Data granularity and quality",
        "",
        f"The input is `{source_path}` and the curve-level key is `curve_id`; sr042_v1 and sr042_v2 "
        "are bound to canonical records 42 and 43, respectively.",
        f"Sample metadata came from `{master_path}`. {hash_verification_note}",
        "",
        f"There were {nonpositive_points} non-positive current points across {nonpositive_curves} curves, "
        f"plus {nonfinite_points} non-finite points. Absolute values and epsilon offsets were not used.",
        "Non-positive values before the reliable positive region may be recorded as possible left censoring and removed from log-current fitting; "
        "a curve-level QC failure occurs only when a non-positive value reappears after a contiguous positive run passes the count, span, log-span, and monotonicity gates.",
        "",
        "The data have no measured per-curve detection limits, so only left-censoring status is recorded and the smallest positive current is used as a plotting-position proxy; "
        "this is not an instrument-validated LOD and is not used in a formal censored likelihood. The former 35% potential-position gate was removed; "
        "the 20% non-positive-point proportion is retained as a descriptive metric only.",
        "",
        "Dense digitized points are first reduced to one monotonic potential branch in acquisition order and then aggregated into equal-potential-width bins. "
        "Each bin uses the median log-current with equal weight to reduce pseudo-replication caused by digitization density.",
        "",
        "## Model comparison",
        "",
    ]
    structure_labels = {
        ("constant", "none"): "constant slope; no plateau parameterization",
        (
            "constant",
            "limiting_current_plateau",
        ): "constant slope + limiting-current plateau parameterization",
        (
            "potential_dependent",
            "limiting_current_plateau",
        ): "potential-dependent slope + limiting-current plateau parameterization",
    }
    for structure, count in structure_counts.items():
        kinetic_structure, observation_parameterization = structure
        label = structure_labels.get(
            (str(kinetic_structure), str(observation_parameterization)),
            f"{kinetic_structure}; {observation_parameterization}",
        )
        lines.append(f"- {label}: {int(count)} curves")
    lines.extend(
        [
            "",
            "The primary analysis uses one continuous Tafel model family, two kinetic structures (constant slope or potential-dependent slope), "
            "and two observation parameterizations (without or with a limiting-current plateau); three combinations actually enter model selection.",
            "They use the same effective bins and Student-t error model. The paper-compatible raw-point Gaussian/SSE results are retained for reproduction comparison but do not enter formal model selection.",
            "",
            f"{plateau_parameterized} curves selected a plateau parameterization, of which {plateau_identified} passed the current plateau-identification gate; "
            "parameterization alone does not mean that a plateau was identified by the data.",
            "",
            "A low-current constant floor is not part of the primary analysis because no independent blank/background measurement is available. "
            "Curve fitting alone cannot identify low-current curvature as physical background current. The legacy technical key is retained only for historical-result reading and pre-declared sensitivity checks.",
            "",
            "The potential-dependent structure must outperform the strongest constant-slope BIC and blocked-CV baselines. It is selected only when all of the following hold:",
            "Delta BIC >= 6, blocked-CV MAE improves by at least 10%, both sides of the change point have sufficient coverage, the bootstrap slope-ratio change evidence is >= 0.90 under that structure, and bootstrap success is >= 0.80.",
            "This probability is not a model-selection probability recomputed after selecting a model in every bootstrap replicate.",
            "Key parameters including slope, change point, plateau, and error scale must not hit optimization boundaries; the corresponding bootstrap boundary-hit rate must not exceed 20%.",
            "",
            "## KNN usage boundaries",
            "",
            "`sample_refit_candidates.csv` contains candidate labels for review and does not replace the active training CSV.",
            "Potential-dependent, bootstrap-unstable, or regime-probability-below-0.80 curves do not receive a single hard label.",
            "",
            "For two-region curves, use low/high slopes and their respective class probabilities; if the legacy KNN requires one scalar, retain only the paper-compatible slope for sensitivity analysis and explicitly exclude it as an error-free ground truth.",
            "",
            "## Important limitations",
            "",
            "- The workbook potential header has no unit. Based on the 1.02–2.06 range and the OER plot context, this workflow assumes V; mV/dec remains to be verified against each original plot axis.",
            "- The workbook has no potential reference, explicit current-density unit, or iR-correction state. Constant current-unit scaling does not affect slope, but it limits cross-paper comparison of absolute change potential, intercept, and limiting current.",
            "- Digitized continuous curves are not independent observations; block bootstrap is an approximation that preserves local correlation under the available information.",
            "- Potential dependence may still arise from iR effects, bubbles, film transport, background, or surface reconstruction and cannot be named a mechanistic change from fitting alone.",
            f"- The current null test is a regression test on fixed synthetic curves; parametric-bootstrap false-positive calibration and multiplicity correction have not been performed for {n_curves} complex-model screens.",
            "Therefore, positive counts must be described as exploratory screening flags.",
            "- Bootstrap intervals in this report are frequentist uncertainty intervals, not posterior credible intervals; formal Bayesian confirmation should use `tafel-refit bayes` and inspect R-hat, ESS, divergences, and PPC.",
            "",
            "## Run information",
            "",
            f"- Mode: `{run_metadata['mode']}`",
            f"- Random seed: `{run_metadata['random_seed']}`",
            f"- Bootstrap repetitions: `{run_metadata['bootstrap_repetitions']}`",
            f"- Workbook SHA256: `{run_metadata['inputs']['workbook_sha256']}`",
            f"- Canonical CSV SHA256: `{run_metadata['inputs']['canonical_sha256']}`",
            f"- Git commit: `{run_metadata.get('git', {}).get('commit') or 'unavailable'}`",
            f"- Git dirty: `{run_metadata.get('git', {}).get('dirty')}`; workflow tracked: `{run_metadata.get('git', {}).get('workflow_files_tracked')}`",
            f"- Workflow tree SHA256: `{run_metadata.get('workflow_tree_sha256')}`",
            f"- Workflow files changed during run: `{run_metadata.get('workflow_files_changed_during_run')}`",
            f"- Input files changed during run: `{run_metadata.get('input_files_changed_during_run')}`",
            f"- Hash verification bypassed: `{run_metadata.get('input_hash_verification_bypassed')}`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
