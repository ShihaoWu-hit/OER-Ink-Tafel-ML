"""Answer-first report for an independent cardinal-classification run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_first_csv(paths: Sequence[Path]) -> pd.DataFrame:
    for path in paths:
        frame = _read_csv(path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _format_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "Not available."
    selected = [column for column in columns if column in frame]
    if not selected:
        return "Not available."
    return frame.loc[:, selected].to_markdown(index=False, floatfmt=".4f")


def _records_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, list):
        records = [row for row in value if isinstance(row, Mapping)]
        return pd.DataFrame(records)
    return pd.DataFrame()


def _paired_log_loss_conclusion(
    bootstrap: pd.DataFrame,
    *,
    label: str,
    smoke: bool,
) -> str:
    if smoke:
        return f"{label}: smoke output; no performance conclusion is allowed."
    if bootstrap.empty or "metric" not in bootstrap:
        return f"{label}: paired soft-log-loss result is unavailable."
    selected = bootstrap.loc[bootstrap["metric"].eq("soft_log_loss")]
    if selected.empty:
        return f"{label}: paired soft-log-loss result is unavailable."
    row = selected.iloc[0]
    delta = float(row["observed_delta"])
    low = float(row["ci95_low"])
    high = float(row["ci95_high"])
    if high < 0:
        interpretation = (
            "exploratory evidence favors the augmented ink panel; prospective "
            "confirmation is still required"
        )
    elif low > 0:
        interpretation = "the paired result favors the context-only reference"
    else:
        interpretation = "the paired result is inconclusive"
    return (
        f"{label}: {interpretation} "
        f"(soft-log-loss Δ={delta:.4f}, group-bootstrap 95% CI "
        f"[{low:.4f}, {high:.4f}])."
    )


def _first_bool(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> bool | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            return value
    for value in payload.values():
        if isinstance(value, Mapping):
            found = _first_bool(value, keys)
            if found is not None:
                return found
    return None


def _low_subdivision_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    required = {
        "q_low_30_40_like",
        "q_30_given_low",
        "q_40_given_low",
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    populations: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    if "electrolyte_scope" in frame.columns:
        populations.extend(
            (
                str(scope),
                frame.loc[frame["electrolyte_scope"].astype(str).eq(str(scope))],
            )
            for scope in sorted(frame["electrolyte_scope"].dropna().unique())
        )
    rows: list[dict[str, object]] = []
    for population, selected in populations:
        low = pd.to_numeric(
            selected["q_low_30_40_like"], errors="coerce"
        ).to_numpy(float)
        q30 = pd.to_numeric(
            selected["q_30_given_low"], errors="coerce"
        ).to_numpy(float)
        q40 = pd.to_numeric(
            selected["q_40_given_low"], errors="coerce"
        ).to_numpy(float)
        finite30 = np.isfinite(q30)
        finite40 = np.isfinite(q40)
        rows.append(
            {
                "population": population,
                "n_records": int(len(selected)),
                "effective_low_mass": float(np.nansum(low)),
                "max_q30_given_low": (
                    float(np.max(q30[finite30])) if finite30.any() else np.nan
                ),
                "n_q30_given_low_ge_0_5": int(
                    np.sum(finite30 & (q30 >= 0.5))
                ),
                "max_q40_given_low": (
                    float(np.max(q40[finite40])) if finite40.any() else np.nan
                ),
                "n_q40_given_low_ge_0_5": int(
                    np.sum(finite40 & (q40 >= 0.5))
                ),
            }
        )
    return pd.DataFrame(rows)


def _low_gate_table(payload: Mapping[str, Any]) -> pd.DataFrame:
    support = payload.get("support")
    if not isinstance(support, list):
        return pd.DataFrame()
    common = {
        "status": payload.get("status"),
        "relative_half_width": payload.get("relative_half_width"),
        "training_attempted": payload.get("training_attempted"),
        "support_gate_passed": payload.get("support_gate_passed"),
        "minimum_effective_mass_per_subclass": payload.get(
            "minimum_effective_mass_per_subclass"
        ),
        "minimum_conditional_majority_groups_per_subclass": payload.get(
            "minimum_conditional_majority_groups_per_subclass"
        ),
    }
    return pd.DataFrame(
        [
            {**common, **dict(row)}
            for row in support
            if isinstance(row, Mapping)
        ]
    )


def _pipeline_summary(
    exact_pipeline: Mapping[str, Any],
    *,
    smoke: bool,
) -> list[str]:
    if not exact_pipeline:
        return ["Final-pipeline specification is unavailable."]
    width = exact_pipeline.get(
        "relative_half_width",
        exact_pipeline.get(
            "label_relative_half_width",
            exact_pipeline.get("primary_relative_half_width", 0.20),
        ),
    )
    return [
        f"- Status: `{exact_pipeline.get('status', 'unknown')}`",
        f"- Primary label half-width: `{width}` (±20% is the only primary definition)",
        f"- Selected panel: `{exact_pipeline.get('panel', 'unknown')}`",
        f"- Selected family: `{exact_pipeline.get('model_family', 'unknown')}`",
        f"- Selected candidate: `{exact_pipeline.get('candidate_id', 'unknown')}`",
        (
            "- Performance claim: smoke/structural only."
            if smoke
            else "- Performance source: nested record-aggregated outer OOF; "
            "the full-data fit is not a performance estimate."
        ),
    ]


def _analysis_width(path: Path, qa: Mapping[str, Any]) -> float | str:
    configured = qa.get(
        "relative_half_width",
        qa.get("label_relative_half_width"),
    )
    if configured is not None:
        return float(configured)
    for part in path.parts:
        if part.startswith("width_"):
            try:
                return int(part.removeprefix("width_")) / 100.0
            except ValueError:
                continue
    return 0.20


def build_cardinal_report(
    run_dir: str | Path,
    *,
    smoke: bool = False,
) -> Path:
    """Build a self-contained Markdown report from saved classification artifacts."""

    root = Path(run_dir).expanduser().resolve()
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    decision = _read_json(root / "cardinal_classification_decision.json")
    exact_pipeline = _read_json(
        root / "final_pipeline" / "exact_final_pipeline.json"
    )
    decision_pipeline = decision.get("final_pipeline")
    if (
        exact_pipeline
        and isinstance(decision_pipeline, Mapping)
        and dict(decision_pipeline) != exact_pipeline
    ):
        raise ValueError(
            "Decision final_pipeline does not match exact_final_pipeline.json"
        )

    definition = _read_json(
        root / "registry" / "cardinal_label_definition.json"
    )
    if not definition:
        definition = _read_json(
            root
            / "registry"
            / "label_widths"
            / "width_20"
            / "cardinal_label_definition.json"
        )
    hierarchical_support = _read_first_csv(
        (
            root / "registry" / "hierarchical_class_support.csv",
            root
            / "registry"
            / "label_widths"
            / "width_20"
            / "hierarchical_class_support.csv",
        )
    )
    anchor_support = _read_first_csv(
        (
            root / "registry" / "class_support.csv",
            root
            / "registry"
            / "label_widths"
            / "width_20"
            / "class_support.csv",
        )
    )
    low_subdivision = _read_first_csv(
        (
            root / "registry" / "low_anchor_subdivision.csv",
            root
            / "registry"
            / "label_widths"
            / "width_20"
            / "low_anchor_subdivision.csv",
        )
    )
    low_gate = _read_json(
        root / "registry" / "low_anchor_subdivision_gate.json"
    )
    if not low_gate:
        low_gate = _read_json(
            root
            / "registry"
            / "label_widths"
            / "width_20"
            / "low_anchor_subdivision_gate.json"
        )
    low_gate_passed = _first_bool(
        low_gate,
        (
            "support_gate_passed",
            "gate_passed",
            "eligible",
            "training_allowed",
            "support_sufficient",
        ),
    )
    if low_gate_passed is True:
        low_gate_statement = (
            "The frozen 30-versus-40 support gate passed. Any resulting "
            "low-subclass analysis remains secondary and cannot alter the "
            "four-class primary result."
        )
    elif low_gate_passed is False:
        low_gate_statement = (
            "The frozen 30-versus-40 support gate did not pass, so no separate "
            "30-versus-40 model was trained. The original q30 and q40 values "
            "remain descriptive posterior quantities only."
        )
    else:
        low_gate_statement = (
            "The 30-versus-40 gate status is unavailable. No separate "
            "low-subclass performance claim is allowed."
        )

    width_summary = _read_csv(
        root / "label_width_sensitivity_summary.csv"
    )
    if width_summary.empty:
        width_summary = _records_frame(
            decision.get("label_width_sensitivity")
        )
    primary_bootstrap = _records_frame(
        decision.get("primary_paired_bootstrap")
    )
    if primary_bootstrap.empty:
        primary_bootstrap = _read_csv(
            root
            / "analyses"
            / "primary_alkaline"
            / "paired_group_bootstrap_summary.csv"
        )
    population_bootstrap = _records_frame(
        decision.get("population_sensitivity_paired_bootstrap")
    )
    if population_bootstrap.empty:
        population_bootstrap = _read_csv(
            root
            / "analyses"
            / "population_sensitivity_all"
            / "paired_group_bootstrap_summary.csv"
        )

    analyses_root = root / "analyses"
    analysis_dirs = sorted(
        {
            qa_path.parent
            for qa_path in analyses_root.rglob("analysis_qa.json")
        },
        key=lambda path: (
            "label_width_sensitivity"
            in path.relative_to(analyses_root).parts,
            str(path.relative_to(analyses_root)),
        ),
    )
    analysis_sections: list[str] = []
    for analysis_dir in analysis_dirs:
        metrics = _read_csv(analysis_dir / "aggregate_metrics.csv")
        bootstrap = _read_csv(
            analysis_dir / "paired_group_bootstrap_summary.csv"
        )
        frequency = _read_csv(
            analysis_dir / "model_selection_frequency.csv"
        )
        qa = _read_json(analysis_dir / "analysis_qa.json")
        relative_analysis_path = analysis_dir.relative_to(analyses_root)
        analysis_width = _analysis_width(relative_analysis_path, qa)
        section = [
            f"## {relative_analysis_path}",
            "",
            f"Panel role: `{qa.get('panel_role', 'unknown')}`; "
            f"label half-width: `{analysis_width}`.",
            "",
            "### Record-aggregated OOF probability scores",
            "",
            _format_table(
                metrics,
                [
                    "relative_half_width",
                    "panel_role",
                    "panel",
                    "model_role",
                    "model_family",
                    "n_records",
                    "soft_log_loss",
                    "multiclass_brier",
                    "gate_soft_log_loss",
                    "conditional_soft_log_loss",
                    "conditional_rps",
                    "hierarchical_balanced_accuracy",
                    "hierarchical_macro_f1",
                ],
            ),
            "",
            "### Paired panel contrasts",
            "",
            _format_table(
                bootstrap,
                [
                    "relative_half_width",
                    "contrast_id",
                    "contrast_role",
                    "comparison",
                    "metric",
                    "observed_delta",
                    "bootstrap_mean",
                    "ci95_low",
                    "ci95_high",
                    "probability_delta_below_zero",
                    "n_bootstrap",
                ],
            ),
            "",
            "The delta is augmented minus reference; negative values favor the "
            "augmented panel for the named within-width contrast.",
            "",
            "### Outer-fold model selection frequency",
            "",
            _format_table(
                frequency,
                [
                    "relative_half_width",
                    "panel_role",
                    "panel",
                    "selected_model_family",
                    "n_outer_selections",
                    "selection_fraction",
                ],
            ),
        ]
        analysis_sections.append("\n".join(section))

    status = (
        "Smoke/structural validation only; no scientific performance conclusion is allowed."
        if smoke
        else "Exploratory post-target-review analysis; it does not retroactively replace the registered continuous-response evaluation."
    )
    lines = [
        "# Posterior-aware hierarchical cardinal-region classification",
        "",
        status,
        "",
        "## Result at a glance",
        "",
        _paired_log_loss_conclusion(
            primary_bootstrap,
            label="Primary ±20% alkaline context-plus-ink contrast",
            smoke=smoke,
        ),
        "",
        _paired_log_loss_conclusion(
            population_bootstrap,
            label="Primary-definition all-97 population sensitivity",
            smoke=smoke,
        ),
        "",
        low_gate_statement,
        "",
        "The ±15% and ±25% results test label-definition robustness. Only ±20% "
        "is primary and may control the final-panel gate. Process and property "
        "panels also remain ±20% sensitivities and cannot override the primary "
        "contrast.",
        "",
        "## Complete label-width sensitivity",
        "",
        _format_table(
            width_summary,
            [
                "relative_half_width",
                "width_percent",
                "scheme_role",
                "analysis_population",
                "scope",
                "analysis_id",
                "comparison",
                "metric",
                "observed_delta",
                "ci95_low",
                "ci95_high",
                "probability_delta_below_zero",
                "n_bootstrap",
                "absolute_log_loss_comparable_across_widths",
                "can_override_primary_width_decision",
            ],
        ),
        "",
        "Each width receives complete nested training, OOF prediction, paired "
        "panel scoring, and connected-group bootstrap. Absolute soft log loss "
        "must not be compared across widths because changing the bands changes "
        "the target entropy and class mass. Interpret only the within-width "
        "`context_plus_ink - context` deltas and intervals.",
        "",
        "## Final exploratory pipeline",
        "",
        *_pipeline_summary(exact_pipeline, smoke=smoke),
        "",
        "## Question",
        "",
        "Can experimental context and ink-formulation descriptors predict the "
        "posterior four-region distribution `low_30_40_like`, `60_like`, "
        "`120_like`, and `off_cardinal`this",
        "",
        "Target q probabilities summarize slope-fit posterior support; OOF model "
        "p probabilities predict that target distribution. Neither is a "
        "reaction-mechanism probability or evidence for a unique "
        "rate-determining step.",
        "",
        "## Frozen label definition",
        "",
        f"- Scheme: `{definition.get('scheme_id', 'classic_hierarchical_band20_v2')}`",
        f"- Anchors (mV dec⁻¹): `{definition.get('anchors_mV_dec', [30, 40, 60, 120])}`",
        f"- Primary relative half-width: `{definition.get('relative_half_width', 0.20)}`",
        "- Every draw is first assigned to the nearest eligible classic anchor "
        "in log space; all other and nonpositive draws are `off_cardinal`.",
        "- The supervised low probability is exactly "
        "`q_30_like + q_40_like`; raw q30 and q40 are retained without being "
        "forced into separate supervised classes.",
        "- Each experimental record remains one ML sample. Its 8,000 draws "
        "produce one four-component supervised soft target.",
        "",
        "## Four-class supervised-target support at ±20%",
        "",
        _format_table(
            hierarchical_support,
            [
                "population",
                "class_label",
                "effective_posterior_mass",
                "modal_record_count",
                "modal_group_count",
                "n_class_probability_ge_0_5",
                "n_class_probability_ge_0_8",
                "max_class_probability",
            ],
        ),
        "",
        "## Descriptive anchor support at ±20%",
        "",
        _format_table(
            anchor_support,
            [
                "population",
                "class_label",
                "effective_posterior_mass",
                "modal_record_count",
                "modal_group_count",
                "n_class_probability_ge_0_5",
                "n_class_probability_ge_0_8",
                "max_class_probability",
            ],
        ),
        "",
        "Raw 30-like and 40-like support is reported for description and audit; "
        "it is not the primary supervised target.",
        "",
        "## 30-versus-40 subdivision gate",
        "",
        low_gate_statement,
        "",
        _format_table(
            _low_gate_table(low_gate),
            [
                "relative_half_width",
                "population",
                "status",
                "training_attempted",
                "support_gate_passed",
                "minimum_effective_mass_per_subclass",
                "minimum_conditional_majority_groups_per_subclass",
                "q_30_effective_posterior_mass",
                "q_40_effective_posterior_mass",
                "q_30_conditional_modal_group_count",
                "q_40_conditional_modal_group_count",
            ],
        ),
        "",
        _format_table(
            _low_subdivision_summary(low_subdivision),
            [
                "population",
                "n_records",
                "effective_low_mass",
                "max_q30_given_low",
                "n_q30_given_low_ge_0_5",
                "max_q40_given_low",
                "n_q40_given_low_ge_0_5",
            ],
        ),
        "",
        *analysis_sections,
        "",
        "## Interpretation limits",
        "",
        "- Primary comparison: identical connected-group folds for `context` and "
        "`context_plus_ink` within each width.",
        "- Feature preprocessing and model/hyperparameter selection occur inside "
        "outer-training data only.",
        "- Posterior-soft ordinal logistic is a candidate only for the "
        "conditional ordered `low/60/120` stage. `off_cardinal` is heterogeneous "
        "and not ordinal.",
        "- Property proxies replace solvent fractions in their panels; the two "
        "representations are not combined.",
        "- Process/property panels are explicitly ±20% sensitivity analyses.",
        "- Absolute soft log loss includes irreducible posterior-target entropy; "
        "cross-width absolute comparisons are invalid.",
        "- `hierarchical_balanced_accuracy` and `hierarchical_macro_f1` use modal "
        "four-region labels and are secondary readability diagnostics. Soft log "
        "loss remains primary.",
        "- None of these classes identifies a reaction mechanism or a unique "
        "rate-determining step.",
        "",
    ]
    output = report_dir / "cardinal_classification_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
