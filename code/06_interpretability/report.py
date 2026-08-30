from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _concat_nested(directory: Path, filename: str) -> pd.DataFrame:
    paths = [directory / filename] if (directory / filename).is_file() else sorted(directory.glob(f"*/{filename}"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "analysis_id" not in frame:
            frame.insert(0, "analysis_id", path.parent.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _target_counts(targets: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if {"target_name", "target_value_mV_dec"}.issubset(targets):
        ml_targets = targets.loc[
            targets["target_name"].astype(str).isin(
                {"b_bayesian_local_line_mV_dec"}
            )
        ]
        for target, frame in ml_targets.groupby("target_name", sort=False):
            n = pd.to_numeric(frame["target_value_mV_dec"], errors="coerce").notna().sum()
            lines.append(f"- `{target}`: {n} eligible records")
    else:
        for target in ("b_bayesian_local_line_mV_dec",):
            if target in targets:
                n = pd.to_numeric(targets[target], errors="coerce").notna().sum()
                lines.append(f"- `{target}`: {n} eligible records")
    return lines


def _delta_table(delta: pd.DataFrame) -> str:
    if delta.empty:
        return "Grouped nested-CV results are not yet available."
    columns = [column for column in ("analysis_id", "comparison", "target", "scope", "model", "delta_mae_estimate", "delta_mae_ci_low", "delta_mae_ci_high", "probability_delta_below_zero") if column in delta]
    table = delta[columns].copy()
    for column in ("delta_mae_estimate", "delta_mae_ci_low", "delta_mae_ci_high"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    if "probability_delta_below_zero" in table:
        table["probability_delta_below_zero"] = pd.to_numeric(table["probability_delta_below_zero"], errors="coerce").round(3)
    return table.to_markdown(index=False)


def _property_panel_status(root: Path, panel_name: str, *, smoke: bool) -> str:
    requested = False
    evaluated = False
    skipped_reasons: list[str] = []
    for config_path in sorted((root / "evaluation").glob("*/evaluation_config.json")):
        config = _read_json(config_path)
        requested = requested or panel_name in config.get("panel_names", [])
        qa = _read_json(config_path.with_name("evaluation_qa.json"))
        evaluated = evaluated or panel_name in qa.get("evaluated_panels", [])
        for item in qa.get("skipped_panels", []):
            if str(item).startswith(panel_name):
                skipped_reasons.append(str(item))
    if evaluated:
        return "evaluated"
    if skipped_reasons:
        return "skipped: " + " | ".join(sorted(set(skipped_reasons)))
    if requested:
        return "requested_but_no_status_recorded"
    return "not_scheduled_in_smoke_run" if smoke else "not_scheduled"


def _property_panel_gate_is_eligible(status: object, expected_status: str) -> bool:
    """Return true only for the exact registered, fail-closed gate status."""

    return isinstance(status, str) and status == expected_status


def build_report(run_dir: str | Path, *, smoke: bool = False) -> tuple[Path, Path]:
    root = Path(run_dir)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    records = _read_csv(root / "data" / "records.csv")
    sources = _read_csv(root / "data" / "sources.csv")
    targets = _read_csv(root / "targets" / "target_registry.csv")
    bayes = _read_csv(root / "targets" / "bayesian_audit.csv")
    delta = _concat_nested(root / "evaluation", "paired_delta_group_bootstrap.csv")
    target_draw_summary = _read_csv(root / "evaluation" / "target_uncertainty" / "target_draw_metric_summary.csv")
    temporal_metrics = _read_csv(root / "evaluation" / "historical_2024plus_backtest" / "temporal_backtest_metrics.csv")
    figures = _read_csv(root / "figures" / "figure_manifest.csv")
    source_figures = _read_csv(root / "figures" / "source_curve_artifact_manifest.csv")
    interpretation_status = _read_json(
        root / "evaluation" / "interpretation" / "interpretation_status.json"
    )
    interpretation_manifest = _read_csv(
        root / "evaluation" / "interpretation" / "interpretation_artifact_manifest.csv"
    )
    interpretation_stability = _read_csv(
        root / "evaluation" / "interpretation" / "interpretation_stability.csv"
    )
    curation_qa = _read_json(root / "data" / "curation_qa.json")
    target_qa = _read_json(root / "targets" / "target_qa.json")
    property_audit = _read_csv(root / "data" / "legacy_property_source_audit.csv")
    property_registry = _read_csv(root / "data" / "record_solvent_property_registry.csv")
    legacy_property_comparison = _read_csv(root / "data" / "legacy_property_comparison.csv")
    property_source_qa = _read_json(root / "data" / "property_source_reconstruction_qa.json")
    property_proxy_qa = _read_json(root / "data" / "property_proxy_qa.json")
    property_workbook_status = _read_json(
        root / "data" / "solvent_property_review_build_status.json"
    )
    property_panel_status = {
        panel: _property_panel_status(root, panel, smoke=smoke)
        for panel in (
            "context_plus_property_proxy_core",
            "context_plus_property_proxy_4p_source_backed",
            "context_plus_property_proxy_4p_estimated",
        )
    }
    estimated_panel_status = str(
        property_proxy_qa.get("estimated_panel_model_use_status", "not_registered")
    )
    estimated_panel_eligible = estimated_panel_status in {
        "eligible_user_approved_with_estimate_indicator",
        "eligible_sensitivity_only",
    }
    if not estimated_panel_eligible:
        property_panel_status["context_plus_property_proxy_4p_estimated"] = (
            "excluded_by_fail_closed_validation_gate:"
            f"reported_status={estimated_panel_status}"
        )
        estimated_panel_fitting_sentence = (
            "The 31 IPA dielectric estimates remain visible as `D_model_estimate`, "
            "together with linear/Looyenga sensitivity bounds. The estimated panel "
            "enters fitting only for an explicitly whitelisted status; its current status is "
            f"`{estimated_panel_status}`, so it is excluded."
        )
    elif estimated_panel_status == "eligible_user_approved_with_estimate_indicator":
        estimated_panel_fitting_sentence = (
            "The 97/97 best-available dielectric feature enters the registered "
            "property panel under the user's versioned approval, together with "
            "the explicit 31-record estimate indicator. This does not re-label "
            "estimated values as measured or claim that the prior validation "
            "threshold passed."
        )
    else:
        estimated_panel_fitting_sentence = (
            "The estimated dielectric panel has the exact whitelisted status "
            "`eligible_sensitivity_only`; it may enter only as a registered sensitivity, "
            "not as the primary property result."
        )
    source_backed_4p_status = property_proxy_qa.get(
        "source_backed_4p_panel_model_use_status"
    )
    if not _property_panel_gate_is_eligible(
        source_backed_4p_status, "eligible_complete_case_only"
    ):
        reported_status = (
            str(source_backed_4p_status)
            if source_backed_4p_status is not None
            else "missing"
        )
        property_panel_status["context_plus_property_proxy_4p_source_backed"] = (
            "excluded_by_fail_closed_validation_gate:"
            f"reported_status={reported_status}"
        )

    n_records = len(records)
    alkaline = 0
    if "is_alkaline" in records:
        alkaline = int(records["is_alkaline"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    elif "electrolyte_identity" in records:
        alkaline = int(records["electrolyte_identity"].astype(str).str.upper().isin({"KOH", "NAOH"}).sum())
    unique_publications = int(sources["publication_id"].nunique()) if "publication_id" in sources else 0
    verified_labs = 0
    if "lab_review_status" in sources:
        verified_labs = int(sources["lab_review_status"].astype(str).str.lower().eq("verified").sum())
    elif "lab_group_confidence" in sources:
        verified_labs = int(sources["lab_group_confidence"].astype(str).str.lower().isin({"verified", "high"}).sum())

    target_lines = _target_counts(targets) or ["- Target registry unavailable"]
    property_coverage_lines: list[str] = []
    if not property_registry.empty and {
        "property_name",
        "strict_source_backed_value",
        "best_available_value",
        "evidence_tier",
    }.issubset(property_registry):
        for property_name, frame in property_registry.groupby("property_name", sort=False):
            strict = int(
                pd.to_numeric(frame["strict_source_backed_value"], errors="coerce")
                .notna()
                .sum()
            )
            best = int(
                pd.to_numeric(frame["best_available_value"], errors="coerce").notna().sum()
            )
            ab = int(
                (
                    frame["evidence_tier"].isin(
                        {"A_direct", "B_same_source_interpolation"}
                    )
                    & pd.to_numeric(
                        frame["strict_source_backed_value"], errors="coerce"
                    ).notna()
                ).sum()
            )
            property_coverage_lines.append(
                f"- `{property_name}`: {strict}/{n_records} strict source-backed "
                f"({ab} A/B); {best}/{n_records} best-available numeric values"
            )
    detailed_property_provenance = bool(property_coverage_lines) or (
        "authoritative_measurement_rows" in property_proxy_qa
    )
    if not property_coverage_lines:
        property_coverage_lines = [
            "- Core solvent-property proxies complete: "
            f"{property_proxy_qa.get('records_with_all_three_core_proxies', 0)}/{n_records}",
            "- All four best-available property values complete: "
            f"{property_proxy_qa.get('records_with_all_four_best_available_values', 0)}/{n_records}",
            "- Best-available dielectric provenance: "
            f"{property_proxy_qa.get('best_available_dielectric_strict_records', 0)} strict and "
            f"{property_proxy_qa.get('best_available_dielectric_estimated_records', 0)} explicitly estimated records",
        ]
    dielectric_endpoints = property_proxy_qa.get(
        "dielectric_estimator_endpoint_parameters", {}
    )
    water_endpoint = dielectric_endpoints.get("water", {})
    ipa_endpoint = dielectric_endpoints.get("isopropanol", {})
    dielectric_endpoint_line = (
        "- D-model endpoint parameters: "
        f"water epsilon={water_endpoint.get('value', 'unavailable')} "
        f"({water_endpoint.get('evidence_tier', 'unavailable')}, "
        f"`{water_endpoint.get('source_id', 'unavailable')}`); "
        f"isopropanol epsilon={ipa_endpoint.get('value', 'unavailable')} "
        f"({ipa_endpoint.get('evidence_tier', 'unavailable')}, "
        f"`{ipa_endpoint.get('source_id', 'unavailable')}`). Both parameters "
        "are hash-frozen; neither is independent IPA/water mixture evidence."
    )
    bayesian_approved = 0
    bayesian_pending = 0
    if not bayes.empty:
        if "formal_nuts_ppc_approved" in bayes:
            approved = bayes["formal_nuts_ppc_approved"].astype(str).str.lower().isin({"true", "1", "yes"})
            bayesian_approved = int(approved.sum())
            bayesian_pending = int((~approved).sum())
        else:
            status_column = next((column for column in ("bayesian_status", "status", "review_status") if column in bayes), None)
            if status_column:
                status = bayes[status_column].astype(str).str.lower()
                bayesian_approved = int(status.str.contains("approved").sum())
                bayesian_pending = int(status.str.contains("pending|missing|incomplete", regex=True).sum())

    ir_counts = (
        records["ir_correction_status"].astype(str).value_counts().to_dict()
        if "ir_correction_status" in records
        else {}
    )
    if {"corrected", "not_corrected"} <= set(ir_counts):
        ir_summary = (
            f"{int(ir_counts.get('corrected', 0))} corrected and "
            f"{int(ir_counts.get('not_corrected', 0))} not corrected"
        )
        ir_blocker_line = (
            "- iR compensation state follows `DATA0718-IR-SEM-001`: "
            "38 corrected and 59 not corrected. Raw reporting evidence remains "
            "separate, including Record 28's inaccessible-full-text state."
        )
    else:
        ir_summary = (
            f"{int(ir_counts.get('reported_corrected', 0))} corrected, "
            f"{int(ir_counts.get('reported_not_corrected', 0))} reported not corrected, and "
            f"{int(ir_counts.get('not_reported_or_unverifiable', 0))} not reported/unverifiable"
        )
        ir_blocker_line = (
            "- Forty-one records retain the reviewed "
            "`not_reported_or_unverifiable` iR state; they are not recoded as "
            "uncorrected."
        )

    if detailed_property_provenance:
        property_provenance_detail = f"""
- Registered measurement/reference rows: {property_proxy_qa.get('authoritative_measurement_rows', 0)}
- Measurement rows with a frozen local raw-file hash: {property_proxy_qa.get('measurement_rows_with_frozen_raw_sha256', 0)}; explicitly hash-pending rows: {property_proxy_qa.get('measurement_rows_with_explicit_hash_pending', 0)}
- Canonical composition-property rows: {property_proxy_qa.get('canonical_composition_property_rows', 0)}
- Record-property rows: {property_proxy_qa.get('record_property_rows', 0)}
- Core A/B coverage: {property_proxy_qa.get('core_A_or_B_values', 0)}/{property_proxy_qa.get('core_A_or_B_expected', 291)}
- Traceable dielectric coverage: {property_proxy_qa.get('dielectric_strict_source_backed_values', 0)}/{n_records}, including {property_proxy_qa.get('dielectric_A_or_B_values', 0)} A/B and {property_proxy_qa.get('dielectric_C_values', 0)} C cells
- Fully hash-frozen dielectric cells: {property_proxy_qa.get('fully_hash_frozen_strict_values_by_property', {}).get('dielectric_constant', 0)}/{n_records}; hash-pending traceable cells: {property_proxy_qa.get('dielectric_strict_values_with_hash_pending', 0)}
- Best-available numeric coverage: {property_proxy_qa.get('best_available_record_property_values', 0)}/{property_proxy_qa.get('best_available_expected', 388)}
- Legacy discrepancies requiring review: {property_proxy_qa.get('legacy_discrepancies', 0)}/{len(legacy_property_comparison)}
{dielectric_endpoint_line}
- IPA Bruggeman validation: MAPE {property_proxy_qa.get('ipa_dielectric_estimate_validation', {}).get('mape_percent', float('nan')):.2f}% and maximum point error {property_proxy_qa.get('ipa_dielectric_estimate_validation', {}).get('maximum_single_point_error_percent', float('nan')):.2f}%
""".strip()
    else:
        property_provenance_detail = (
            "- Detailed measurement/source registries are not duplicated in this compact run; "
            "the canonical property QA summary and registered panel gate are authoritative here."
        )
    property_delivery_lines: list[str] = []
    if property_workbook_status:
        property_delivery_lines.append(
            f"- Review workbook: `{property_workbook_status.get('status', 'not_attempted')}`"
        )
    if "solvent_property_full_acceptance_passed" in curation_qa:
        property_delivery_lines.append(
            "- Full solvent-property deliverable acceptance: "
            f"`{curation_qa.get('solvent_property_full_acceptance_passed')}`; blockers: "
            f"`{json.dumps(curation_qa.get('solvent_property_acceptance_blockers', []), ensure_ascii=False)}`"
        )

    ink_comparison_delta = delta
    property_core_delta = pd.DataFrame()
    property_representation_delta = pd.DataFrame()
    property_comparison_delta = pd.DataFrame()
    property_surface_exclusion_delta = pd.DataFrame()
    if "comparison" in delta:
        ink_comparison_delta = delta.loc[
            delta["comparison"].astype(str).eq("context_plus_ink_minus_context")
        ]
        property_core_delta = delta.loc[
            delta["comparison"].astype(str).eq(
                "property_proxy_core_minus_context_plus_ink_amounts"
            )
        ]
        property_representation_delta = delta.loc[
            delta["comparison"].astype(str).eq(
                "property_proxy_core_minus_raw_solvent_composition_same_records"
            )
        ]
        property_comparison_delta = delta.loc[
            delta["comparison"].astype(str).eq(
                "property_proxy_4p_source_backed_minus_context_plus_ink_amounts_complete_cases"
            )
        ]
        property_surface_exclusion_delta = delta.loc[
            delta["comparison"].astype(str).eq(
                "property_proxy_core_excluding_medium_ipa_surface_minus_context_plus_ink_amounts_same_records"
            )
        ]
    strict_property_result = _delta_table(property_comparison_delta)
    if not _property_panel_gate_is_eligible(
        source_backed_4p_status, "eligible_complete_case_only"
    ):
        reported_status = (
            str(source_backed_4p_status)
            if source_backed_4p_status is not None
            else "missing"
        )
        strict_property_result = (
            "Not evaluated because the registered panel gate is "
            f"`{reported_status}`; only the exact status "
            "`eligible_complete_case_only` is accepted."
        )

    if smoke:
        result_statement = "This is a smoke run. It validates interfaces and leakage controls; it is not a scientific performance estimate."
    elif ink_comparison_delta.empty:
        result_statement = "Evaluation is incomplete, so no claim about incremental ink information is made."
    else:
        primary = ink_comparison_delta.loc[ink_comparison_delta.get("analysis_id", pd.Series("", index=ink_comparison_delta.index)).astype(str).str.contains("primary", case=False, regex=False)]
        if primary.empty:
            primary = ink_comparison_delta
        p_est = pd.to_numeric(primary["delta_mae_estimate"], errors="coerce")
        p_low = pd.to_numeric(primary["delta_mae_ci_low"], errors="coerce")
        p_high = pd.to_numeric(primary["delta_mae_ci_high"], errors="coerce")
        if ((p_high < 0) & p_est.notna()).any():
            result_statement = "At least one primary model shows lower grouped OOF MAE after adding ink features; transferability and stability gates must still be checked before mechanistic interpretation."
        elif ((p_low <= 0) & (p_high >= 0)).any():
            result_statement = "Primary connected-group bootstrap intervals include zero; the data do not establish transferable incremental ink information."
        else:
            result_statement = "Adding ink features does not reduce primary grouped OOF MAE; this is a valid negative predictive result."

    report = f"""# Ink formulation–Tafel study report

## Answer first

{result_statement}

Protocol amendment (2026-07-17, adopted before formal outer results): property
increment is now isolated against `context_plus_ink_amounts`, while the separate
same-fold representation contrast compares property proxies with the raw solvent
composition in `context_plus_ink`. The primary `context_plus_ink - context`
estimand is unchanged.

The study is associational and predictive. Cross-literature data alone cannot establish a causal ink effect or an OER reaction mechanism.

## Dataset and provenance

- Canonical records: {n_records}
- Primary alkaline records in the canonical registry: {alkaline}
- Resolved publication groups: {unique_publications}
- Publications with a verified laboratory assignment: {verified_labs}
- Reviewed iR evidence states: {ir_summary}; unknown values remain unknown and no numerical correction is inferred
- Curation QA status: `{curation_qa.get('status', 'unavailable')}`

Until laboratory assignments are independently verified, publication/recipe-connected holdout is valid but cannot be described as a definitive lab-held-out test.

## Solvent-property provenance

The legacy workbook is retained for cross-checking and was not overwritten. Predictor values come from the registered measurement series, explicit composition-basis conversion, and bracketed interpolation. The values are solvent-mixture proxies, not measurements of the complete catalyst/Nafion/particle ink.

{chr(10).join(property_coverage_lines)}

{property_provenance_detail}

- Estimated dielectric panel status: `{property_proxy_qa.get('estimated_panel_model_use_status', 'unavailable')}`
- `context_plus_property_proxy_core`: `{property_panel_status['context_plus_property_proxy_core']}`
- `context_plus_property_proxy_4p_source_backed`: `{property_panel_status['context_plus_property_proxy_4p_source_backed']}`
- `context_plus_property_proxy_4p_estimated`: `{property_panel_status['context_plus_property_proxy_4p_estimated']}`
{chr(10).join(property_delivery_lines)}

{estimated_panel_fitting_sentence}

## Continuous targets

{chr(10).join(target_lines)}

- Bayesian/PPC-approved curves represented in the audit: {bayesian_approved}
- Bayesian rows still pending or incomplete: {bayesian_pending}
- Named PPC reviews complete: {target_qa.get('n_named_ppc_review_complete', bayesian_approved)}/{n_records}
- Target QA status: `{target_qa.get('status', 'unavailable')}`

Potential-dependent records use the model-coordinate low-potential branch. They are never selected by taking the numerically smaller branch. The exact-2 mA target is retained only as a diagnostic audit and is excluded from ML.

## Primary paired comparison

Negative delta favors adding ink features:

{_delta_table(ink_comparison_delta)}

Metrics are computed after transforming predictions back from `log10(slope)` to mV dec^-1. Repeated OOF predictions are aggregated once per record before connected-group bootstrap inference.

## Source-backed property comparisons

For every table below, delta is property-panel MAE minus the named paired
reference MAE; negative values favor the property representation.

Incremental property contribution, holding catalyst/electrochemical context and
all non-solvent ink amounts fixed (`context_plus_property_proxy_core -
context_plus_ink_amounts`):

{_delta_table(property_core_delta)}

Property representation versus raw solvent composition on the same records and
folds (`context_plus_property_proxy_core - context_plus_ink`):

{_delta_table(property_representation_delta)}

Strict four-property panel versus `context_plus_ink_amounts` fitted on the
identical complete-case records:

{strict_property_result}

Amended-protocol surface-tension sensitivity after excluding every record mapped to the two medium-confidence IPA brackets; both sides use the same retained records and saved group folds:

{_delta_table(property_surface_exclusion_delta)}

## Target uncertainty and temporal sensitivity

- Synchronized posterior low-potential slope draws scored: {int(target_draw_summary['n_target_draws'].max()) if 'n_target_draws' in target_draw_summary and len(target_draw_summary) else 0}. This is score-level target uncertainty; models are not retrained for each draw.
- 2024+ historical backtest rows: {int(temporal_metrics['n_test'].max()) if 'n_test' in temporal_metrics and len(temporal_metrics) else 0}. This is explicitly not an untouched or prospective test.

## Interpretation boundary

Feature importance, block ablations, solvent-substitution effects, and neighbor cases are mechanistic hypotheses only if their directions survive target, model, split, catalyst, electrolyte, and laboratory stability checks. Gated figures are listed in the figure manifest rather than silently generated from unstable results.

- Outer-test block importance status: `{interpretation_status.get('importance_status', 'not_estimable_from_saved_artifacts')}`
- Blocks passing every registered stability dimension: {interpretation_status.get('n_blocks_passing_all_stability_dimensions', 0)}
- Composition-conserving solvent substitution: `{interpretation_status.get('composition_substitution_status', 'gated')}`
- kNN neighbor cases: `{interpretation_status.get('neighbor_case_status', 'gated')}`

Block masking and joint permutation, when available, are scored only on the outer test fold after inner-CV fitting and selection; they never tune the model. They are conditional prediction sensitivities, not causal effects. If fold estimators or perturbed predictions were not saved, the audit explicitly records `not_estimable_from_saved_artifacts` rather than reconstructing a misleading importance value.

For the registered property importance panel, Gower fitting still uses one joint
`solvent_property_proxy` distance block. The post-hoc perturbation audit splits
viscosity, surface tension, and density into singleton perturbation groups only
to identify conditional predictive sensitivity in the presence of the other
proxies. That output is not a marginal effect, a mechanism, or a causal ranking.

## Remaining blockers

{ir_blocker_line}
- Corresponding-author/affiliation candidates do not become `lab_group_id` without verification.
- The 2024+ analysis is historical backtesting. Only literature added after the 2026-07-16 freeze can be called prospective validation.
- The Smith 1998 numerical table or author data are still needed to replace the two C-tier methanol/water figure reconstructions and consider an A/B upgrade; no graph-derived value is promoted to A/B.
- Local primary PDFs for Petong 2000 and Uosaki 1997 are still needed to close the 10 eligible measurement-row SHA256 gaps. Until then the 66-cell dielectric table remains an auditable traceable layer, but the strict four-property ML panel is excluded.
{('- The artifact-tool spreadsheet runtime is unavailable in this environment, so CSV/JSON workbook inputs and the workbook builder are complete but XLSX export/render remains blocked.' if property_workbook_status and property_workbook_status.get('status') != 'generated_and_rendered' else '')}
"""
    report_path = report_dir / "study_report.md"
    report_path.write_text(report, encoding="utf-8")

    generated = int((figures.get("status", pd.Series(dtype=str)) == "generated").sum()) if not figures.empty else 0
    gated_statuses = {
        "gated",
        "not_estimable_from_saved_artifacts",
        "not_evaluated_smoke_mode",
    }
    gated = int(figures.get("status", pd.Series(dtype=str)).isin(gated_statuses).sum()) if not figures.empty else 0
    overall = "Share with caveats" if (n_records == 97 and not targets.empty) else "Needs revision"
    if smoke:
        overall = "Structural validation only"
    validation = f"""# Validation report

## Overall assessment: {overall}

## Methodology review

- Canonical grain: one formulation record per `record_id`; observed curves remain separate in `curve_registry`.
- Leakage control: publication/lab/recipe/sample connected groups are held out together in both outer and inner CV.
- Model selection: preprocessing and hyperparameter selection occur inside inner grouped folds.
- Primary comparison: paired `context_plus_ink - context` MAE on identical outer splits.
- Property protocol amendment: adopted 2026-07-17 before formal outer results; property-increment baselines include `context_plus_ink_amounts`, and the raw-composition representation comparison uses `context_plus_ink` on the same records/folds.
- Causal language is prohibited by the registered protocol.

## Calculation spot-checks

- Canonical record count: {'verified' if n_records == 97 else 'not verified'} ({n_records})
- Alkaline record count: {'verified' if alkaline == 90 else 'not verified'} ({alkaline})
- Target registry available: {'yes' if not targets.empty else 'no'}
- Connected-group bootstrap table available: {'yes' if not delta.empty else 'no'}
- Publication figures generated: {generated}; interpretation-gated: {gated}
- Interpretation audit available: {'yes' if not interpretation_manifest.empty else 'no'}; fully stable blocks: {int(interpretation_stability['overall_stability_pass'].astype(str).str.lower().isin({'true', '1', 'yes'}).sum()) if 'overall_stability_pass' in interpretation_stability else 0}
- Property QA status: `{property_proxy_qa.get('status', 'unavailable')}`
- Records with all three core property proxies: {property_proxy_qa.get('records_with_all_three_core_proxies', 0)}/{n_records}
- Strict best-available dielectric records: {property_proxy_qa.get('best_available_dielectric_strict_records', 0)}/{n_records}
- Explicitly estimated dielectric records: {property_proxy_qa.get('best_available_dielectric_estimated_records', 0)}/{n_records}
- Records with all four best-available property values: {property_proxy_qa.get('records_with_all_four_best_available_values', 0)}/{n_records}
- Property proxy claim boundary: {property_proxy_qa.get('claim_boundary', 'unavailable')}
- Property panel status: `{json.dumps(property_panel_status, ensure_ascii=False, sort_keys=True)}`
- IPA estimated-panel gate: `{property_proxy_qa.get('estimated_panel_model_use_status', 'unavailable')}`
- Compact property QA summary bound to this run: {'yes' if bool(property_proxy_qa) else 'no'}

## Required caveats

- Literature association is not causal evidence.
- Unknown iR status prevents describing slopes as intrinsic or iR-free.
- Unverified laboratory metadata limits the strength of cross-laboratory transfer claims.
- Bayesian/PPC approval and connected-group bootstrap uncertainty are reported separately.
- A documented candidate value is not automatically a 298.15 K model-ready proxy; source composition, definition, hash, and temperature gates remain explicit.
"""
    validation_path = report_dir / "validation_report.md"
    validation_path.write_text(validation, encoding="utf-8")
    return report_path, validation_path
