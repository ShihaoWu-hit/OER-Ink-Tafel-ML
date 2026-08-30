from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import PALETTE, apply_publication_style, save_figure


def _read_if_present(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _read_json_if_present(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})


_CONDITIONAL_ASSOCIATIONAL_BOUNDARY = (
    "conditional associational sensitivity only; not a feature ranking or causal effect"
)


def _released_panel_blocks(
    stability: pd.DataFrame,
    interpretation_status: dict[str, object],
) -> tuple[set[tuple[str, str]], str]:
    """Resolve stability-passing panel/block keys without inferring a release."""

    if stability.empty:
        return set(), f"interpretation stability audit is unavailable; {_CONDITIONAL_ASSOCIATIONAL_BOUNDARY}"
    if not {"block", "overall_stability_pass"}.issubset(stability):
        return set(), (
            "interpretation stability audit lacks block or overall_stability_pass; "
            f"{_CONDITIONAL_ASSOCIATIONAL_BOUNDARY}"
        )
    passed = stability.loc[_normalize_bool(stability["overall_stability_pass"])].copy()
    if passed.empty:
        return set(), (
            "no predictor block passed every registered interpretation stability dimension; "
            f"{_CONDITIONAL_ASSOCIATIONAL_BOUNDARY}"
        )

    if "panel" in passed:
        panel_values = passed["panel"].astype("string").str.strip()
    else:
        config = interpretation_status.get("config", {})
        panel = str(config.get("importance_panel", "")).strip() if isinstance(config, dict) else ""
        if not panel:
            return set(), (
                "passing stability rows cannot be paired to a registered predictor panel; "
                f"{_CONDITIONAL_ASSOCIATIONAL_BOUNDARY}"
            )
        panel_values = pd.Series(panel, index=passed.index, dtype="string")

    block_values = passed["block"].astype("string").str.strip()
    valid = panel_values.notna() & panel_values.ne("") & block_values.notna() & block_values.ne("")
    pairs = {
        (str(panel), str(block))
        for panel, block in zip(panel_values.loc[valid], block_values.loc[valid], strict=True)
    }
    if not pairs:
        return set(), (
            "no valid panel/block key survived the interpretation stability audit; "
            f"{_CONDITIONAL_ASSOCIATIONAL_BOUNDARY}"
        )
    released = ", ".join(f"{panel}+{block}" for panel, block in sorted(pairs))
    return pairs, (
        f"stability-released panel/block pairs: {released}; "
        f"{_CONDITIONAL_ASSOCIATIONAL_BOUNDARY}"
    )


def _filter_released_panel_blocks(
    importance: pd.DataFrame,
    released_panel_blocks: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    pairs = {(str(panel), str(block)) for panel, block in released_panel_blocks}
    if importance.empty or not pairs or not {"panel", "block"}.issubset(importance):
        return importance.iloc[0:0].copy()
    allowed = pd.DataFrame(sorted(pairs), columns=["panel", "block"])
    frame = importance.copy()
    frame["panel"] = frame["panel"].astype(str)
    frame["block"] = frame["block"].astype(str)
    return frame.merge(allowed, on=["panel", "block"], how="inner", validate="many_to_one")


def _evaluation_tables(evaluation_dir: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    direct = evaluation_dir / filename
    paths = [direct] if direct.is_file() else sorted(evaluation_dir.glob(f"*/{filename}"))
    for path in paths:
        frame = pd.read_csv(path)
        if "analysis_id" not in frame:
            frame.insert(0, "analysis_id", path.parent.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _missingness_columns(records: pd.DataFrame) -> list[str]:
    candidates = [
        "electrolyte_identity",
        "electrolyte_molarity_M",
        "electrode_type",
        "catalyst_loading_mg_cm2",
        "rotation_rate_rpm",
        "rpm",
        "scan_rate_mV_s",
        "sonication_time_min",
        "ultrasonication_time_min",
        "drying_method",
        "lab_group_id",
        "ir_correction_status",
        "uncompensated_resistance_ohm",
    ]
    return [column for column in candidates if column in records]


def plot_data_coverage(records: pd.DataFrame, output_base: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    apply_publication_style(11)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    electrolyte = (
        records.get("electrolyte_identity", pd.Series("unknown", index=records.index))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .sort_values(ascending=True)
    )
    electrolyte.index = [
        {"HCLO4": "HClO$_4$", "H2SO4": "H$_2$SO$_4$", "KHCO3": "KHCO$_3$", "NAOH": "NaOH"}.get(label.upper(), label)
        for label in electrolyte.index
    ]
    axes[0].barh(electrolyte.index, electrolyte.values, color=PALETTE["blue_secondary"], edgecolor="black", linewidth=0.7)
    axes[0].set_xlabel("Records")
    axes[0].set_title("Electrolyte coverage")

    year_column = next((column for column in ("publication_year", "source_year") if column in records), None)
    if year_column and pd.to_numeric(records[year_column], errors="coerce").notna().any():
        years = pd.to_numeric(records[year_column], errors="coerce").dropna()
        bins = np.arange(int(years.min()), int(years.max()) + 2) - 0.5
        axes[1].hist(years, bins=bins, color=PALETTE["teal"], edgecolor="white")
        axes[1].set_xlabel("Publication year")
        axes[1].set_ylabel("Records")
    else:
        axes[1].text(0.5, 0.5, "Publication year\nnot yet curated", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    axes[1].set_title("Temporal coverage")

    columns = _missingness_columns(records)
    if columns:
        missing_rates: dict[str, float] = {}
        for column in columns:
            values = records[column]
            text = values.astype("string").str.strip().str.lower()
            unknown = text.isna() | text.eq("") | text.eq("unknown") | text.str.startswith(
                ("unknown_", "lab_unknown_", "lab-unknown", "unresolved_"), na=False
            )
            if column == "lab_group_id" and "lab_group_confidence" in records:
                confidence = records["lab_group_confidence"].astype("string").str.strip().str.lower()
                unknown = unknown | confidence.isna() | confidence.isin({"", "unknown", "unresolved"})
            missing_rates[column] = float(unknown.mean())
        missing = pd.Series(missing_rates).sort_values()
        axes[2].barh(missing.index, missing.values * 100.0, color=PALETTE["red_2"], edgecolor="black", linewidth=0.5)
        axes[2].set_xlim(0, 100)
        axes[2].set_xlabel("Missing / unknown (%)")
    else:
        axes[2].text(0.5, 0.5, "Registered fields unavailable", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    axes[2].set_title("Protocol and provenance gaps")
    return save_figure(fig, output_base, formats=formats, dpi=dpi)


def plot_target_coverage(targets: pd.DataFrame, output_base: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    apply_publication_style(11)
    if {"target_name", "target_value_mV_dec"}.issubset(targets):
        target_names = [
            name
            for name in ("b_bayesian_local_line_mV_dec",)
            if targets["target_name"].astype(str).eq(name).any()
        ]
        values = {
            name: pd.to_numeric(
                targets.loc[targets["target_name"].astype(str) == name, "target_value_mV_dec"],
                errors="coerce",
            ).dropna()
            for name in target_names
        }
    else:
        target_names = [
            column
            for column in ("b_bayesian_local_line_mV_dec",)
            if column in targets
        ]
        values = {name: pd.to_numeric(targets[name], errors="coerce").dropna() for name in target_names}
    if not target_names:
        return []
    fig, axes = plt.subplots(1, len(target_names), figsize=(4.5 * len(target_names), 4.0), squeeze=False)
    for axis, name in zip(axes.ravel(), target_names, strict=True):
        series = values[name]
        axis.hist(series, bins=min(14, max(5, int(np.sqrt(max(len(series), 1))))), color=PALETTE["blue_main"], alpha=0.85, edgecolor="white")
        axis.axvline(90, color=PALETTE["neutral_dark"], linestyle="--", linewidth=1)
        axis.axvline(122, color=PALETTE["red_strong"], linestyle="--", linewidth=1)
        axis.set_xlabel("Tafel slope (mV dec$^{-1}$)")
        axis.set_ylabel("Records")
        display_name = {
            "b_bayesian_local_line_mV_dec": "Bayesian local-line Tafel slope",
            "b_single_regime_mV_dec": "Single-regime only",
        }.get(name, name)
        axis.set_title(f"{display_name}\neligible n={len(series)}")
    return save_figure(fig, output_base, formats=formats, dpi=dpi)


def plot_paired_delta(delta: pd.DataFrame, output_base: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    if delta.empty:
        return []
    required = {"delta_mae_estimate", "delta_mae_ci_low", "delta_mae_ci_high"}
    if not required.issubset(delta):
        return []
    frame = delta.copy()
    if "comparison" in frame:
        frame = frame.loc[
            frame["comparison"].astype(str).eq("context_plus_ink_minus_context")
        ].copy()
    if frame.empty:
        return []
    def compact_label(row: pd.Series) -> str:
        analysis = str(row.get("analysis_id", row.get("target", "analysis")))
        analysis = analysis.replace("primary_low_", "low-potential, ").replace("sensitivity_", "sensitivity: ")
        model = str(row.get("model", "model")).replace("_", " ")
        return f"{analysis} | {model}"

    frame["label"] = frame.apply(compact_label, axis=1)
    frame = frame.sort_values("delta_mae_estimate").reset_index(drop=True)
    y = np.arange(len(frame))
    x = pd.to_numeric(frame["delta_mae_estimate"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(frame["delta_mae_ci_low"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(frame["delta_mae_ci_high"], errors="coerce").to_numpy(float)
    colors = [PALETTE["green_3"] if value < 0 else PALETTE["red_2"] for value in x]
    apply_publication_style(10)
    fig, axis = plt.subplots(figsize=(8.0, max(3.6, 0.42 * len(frame) + 1.8)))
    axis.axvline(0, color=PALETTE["neutral_dark"], linewidth=1.2)
    axis.errorbar(x, y, xerr=[x - low, high - x], fmt="none", ecolor=PALETTE["neutral_dark"], capsize=3, linewidth=1.3)
    axis.scatter(x, y, c=colors, edgecolor="black", linewidth=0.6, s=55, zorder=3)
    axis.set_yticks(y, frame["label"])
    axis.set_xlabel(r"Paired $\Delta$MAE: context + ink $-$ context (mV dec$^{-1}$; negative favors ink)")
    axis.set_title("Incremental ink information under connected-group holdout\nConnected-group bootstrap 95% intervals")
    return save_figure(fig, output_base, formats=formats, dpi=dpi)


def plot_oof_primary(oof: pd.DataFrame, output_base: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    if oof.empty or not {"observed_mV_dec", "predicted_mV_dec", "panel", "model"}.issubset(oof):
        return []
    frame = oof.copy()
    if "target" in frame:
        preferred = frame["target"].astype(str).str.contains("2mA", case=False, regex=False)
        if preferred.any():
            frame = frame.loc[preferred]
    if "scope" in frame and frame["scope"].astype(str).eq("alkaline").any():
        frame = frame.loc[frame["scope"].astype(str).eq("alkaline")]
    if frame["model"].astype(str).eq("nested_selected").any():
        frame = frame.loc[frame["model"].astype(str).eq("nested_selected")]
    elif frame["model"].astype(str).eq("bayesian_ridge").any():
        frame = frame.loc[frame["model"].astype(str).eq("bayesian_ridge")]
    frame = frame.loc[frame["panel"].isin(["context", "context_plus_ink"])]
    if frame.empty:
        return []
    panels = [panel for panel in ("context", "context_plus_ink") if panel in set(frame["panel"])]
    apply_publication_style(11)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
    limits = [
        float(np.nanmin(frame[["observed_mV_dec", "predicted_mV_dec"]].to_numpy(float))),
        float(np.nanmax(frame[["observed_mV_dec", "predicted_mV_dec"]].to_numpy(float))),
    ]
    padding = max(5.0, 0.05 * (limits[1] - limits[0]))
    limits = [limits[0] - padding, limits[1] + padding]
    for axis, panel in zip(axes.ravel(), panels, strict=True):
        subset = frame.loc[frame["panel"] == panel]
        axis.scatter(subset["observed_mV_dec"], subset["predicted_mV_dec"], color=PALETTE["blue_main"], alpha=0.75, edgecolor="white", linewidth=0.4)
        axis.plot(limits, limits, color=PALETTE["neutral_dark"], linestyle="--", linewidth=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel("Observed (mV dec$^{-1}$)")
        axis.set_ylabel("Repeated-OOF aggregate (mV dec$^{-1}$)")
        mae = np.mean(np.abs(subset["observed_mV_dec"] - subset["predicted_mV_dec"]))
        axis.set_title(f"{panel}\nMAE={mae:.1f}, n={len(subset)}")
    return save_figure(fig, output_base, formats=formats, dpi=dpi)


def plot_residual_diagnostics(oof: pd.DataFrame, output_base: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    required = {"observed_mV_dec", "predicted_mV_dec", "panel", "model"}
    if oof.empty or not required.issubset(oof):
        return []
    frame = oof.copy()
    if "target" in frame:
        preferred = frame["target"].astype(str).eq("b_bayesian_local_line_mV_dec")
        if preferred.any():
            frame = frame.loc[preferred]
    if "scope" in frame and frame["scope"].astype(str).eq("alkaline").any():
        frame = frame.loc[frame["scope"].astype(str).eq("alkaline")]
    if frame["model"].astype(str).eq("nested_selected").any():
        frame = frame.loc[frame["model"].astype(str).eq("nested_selected")]
    if frame["panel"].astype(str).eq("context_plus_ink").any():
        frame = frame.loc[frame["panel"].astype(str).eq("context_plus_ink")]
    if frame.empty:
        return []
    residual = frame["observed_mV_dec"].to_numpy(float) - frame["predicted_mV_dec"].to_numpy(float)
    absolute = np.sort(np.abs(residual))
    ecdf = np.arange(1, len(absolute) + 1) / len(absolute)
    apply_publication_style(11)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].axhline(0, color=PALETTE["neutral_dark"], linewidth=1)
    axes[0].scatter(frame["predicted_mV_dec"], residual, color=PALETTE["blue_main"], alpha=0.75, edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Predicted (mV dec$^{-1}$)")
    axes[0].set_ylabel("Observed − predicted (mV dec$^{-1}$)")
    axes[0].set_title("Grouped OOF residuals")
    axes[1].step(absolute, ecdf, where="post", color=PALETTE["teal"], linewidth=2)
    axes[1].set_xlabel("Absolute error (mV dec$^{-1}$)")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Absolute-error distribution")
    return save_figure(fig, output_base, formats=formats, dpi=dpi)


def plot_outer_test_block_importance(
    importance: pd.DataFrame,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    *,
    released_panel_blocks: Iterable[tuple[str, str]],
) -> list[Path]:
    """Plot gate-released conditional associational sensitivity without CIs."""

    required = {
        "target",
        "scope",
        "panel",
        "model",
        "repeat",
        "fold",
        "block",
        "perturbation_method",
        "delta_mae_mV_dec",
    }
    if importance.empty or not required.issubset(importance):
        return []
    frame = _filter_released_panel_blocks(importance, released_panel_blocks)
    frame = frame.loc[
        frame["target"].astype(str).eq("b_bayesian_local_line_mV_dec")
        & frame["scope"].astype(str).eq("alkaline")
    ].copy()
    if frame.empty:
        return []
    split_level = (
        frame.groupby(
            ["model", "panel", "block", "perturbation_method", "repeat", "fold"],
            as_index=False,
        )["delta_mae_mV_dec"]
        .mean()
    )
    models = [
        model
        for model in ("bayesian_ridge", "gower_knn", "extra_trees")
        if model in set(split_level["model"].astype(str))
    ]
    panel_blocks = sorted(
        set(zip(split_level["panel"].astype(str), split_level["block"].astype(str), strict=True))
    )
    if not models or not panel_blocks:
        return []

    apply_publication_style(10)
    fig, axes = plt.subplots(1, len(models), figsize=(5.3 * len(models), 5.3), squeeze=False, sharey=True)
    rng = np.random.default_rng(20260716)
    method_style = {
        "joint_block_permutation": (PALETTE["blue_main"], "o", "joint permutation"),
        "block_masking_to_fitted_missingness_policy": (
            PALETTE["red_strong"],
            "^",
            "mask using fitted missingness policy",
        ),
    }
    for axis, model in zip(axes.ravel(), models, strict=True):
        model_rows = split_level.loc[split_level["model"].astype(str).eq(model)]
        for method_index, (method, (color, marker, label)) in enumerate(method_style.items()):
            offset = -0.13 if method_index == 0 else 0.13
            for block_index, (panel, block) in enumerate(panel_blocks):
                values = pd.to_numeric(
                    model_rows.loc[
                        model_rows["panel"].astype(str).eq(panel)
                        & model_rows["block"].astype(str).eq(block)
                        & model_rows["perturbation_method"].astype(str).eq(method),
                        "delta_mae_mV_dec",
                    ],
                    errors="coerce",
                ).dropna().to_numpy(float)
                if not len(values):
                    continue
                jitter = rng.normal(0.0, 0.025, size=len(values))
                axis.scatter(
                    block_index + offset + jitter,
                    values,
                    s=14,
                    alpha=0.23,
                    color=color,
                    edgecolor="none",
                )
                axis.scatter(
                    [block_index + offset],
                    [float(np.median(values))],
                    s=52,
                    color=color,
                    marker=marker,
                    edgecolor="black",
                    linewidth=0.6,
                    label=label if block_index == 0 else None,
                    zorder=3,
                )
        axis.axhline(0, color=PALETTE["neutral_dark"], linewidth=1.0, linestyle="--")
        axis.set_xticks(
            range(len(panel_blocks)),
            [
                f"{panel.replace('_', ' ')}\n{block.replace('_', ' ')}"
                for panel, block in panel_blocks
            ],
            rotation=42,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_title(model.replace("_", " "))
    axes[0, 0].set_ylabel(r"Conditional outer-test $\Delta$MAE after perturbation (mV dec$^{-1}$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, fontsize=9)
    fig.supxlabel("Stability-released panel + mutually exclusive predictor block", y=0.01)
    fig.suptitle(
        "Outer-test conditional associational block sensitivity\n"
        "Stability-released pairs only; not a feature ranking or causal effect\n"
        "Points are repeated grouped splits; markers are medians (not independent confidence intervals)",
        y=1.04,
    )
    return save_figure(fig, output_base, formats=formats, dpi=dpi)


def generate_figures(
    run_dir: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
    smoke: bool = False,
) -> pd.DataFrame:
    root = Path(run_dir)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = _read_if_present(root / "data" / "records.csv")
    targets = _read_if_present(root / "targets" / "target_registry.csv")
    target_artifacts = _read_if_present(root / "targets" / "target_artifact_manifest.csv")
    delta = _evaluation_tables(root / "evaluation", "paired_delta_group_bootstrap.csv")
    oof = _evaluation_tables(root / "evaluation", "oof_aggregated.csv")
    importance = _evaluation_tables(root / "evaluation", "outer_test_block_importance.csv")
    interpretation_dir = root / "evaluation" / "interpretation"
    interpretation_manifest = _read_if_present(interpretation_dir / "interpretation_artifact_manifest.csv")
    interpretation_stability = _read_if_present(interpretation_dir / "interpretation_stability.csv")
    interpretation_status = _read_json_if_present(interpretation_dir / "interpretation_status.json")

    rows: list[dict[str, object]] = []

    def register(
        name: str,
        paths: list[Path],
        reason_if_missing: str,
        *,
        status_if_missing: str = "not_generated",
        reason_if_generated: str = "",
    ) -> None:
        if paths:
            for path in paths:
                rows.append(
                    {
                        "figure": name,
                        "path": str(path.relative_to(root)),
                        "status": "generated",
                        "reason": reason_if_generated,
                    }
                )
        else:
            rows.append(
                {
                    "figure": name,
                    "path": "",
                    "status": status_if_missing,
                    "reason": reason_if_missing,
                }
            )

    register(
        "data_coverage",
        plot_data_coverage(records, figure_dir / "data_coverage", formats, dpi) if not records.empty else [],
        "canonical records are unavailable",
    )
    register(
        "target_coverage",
        plot_target_coverage(targets, figure_dir / "target_coverage", formats, dpi) if not targets.empty else [],
        "target registry is unavailable",
    )
    register(
        "paired_delta_mae",
        [] if smoke else plot_paired_delta(delta, figure_dir / "paired_delta_mae", formats, dpi),
        (
            "smoke mode does not generate performance figures"
            if smoke
            else "grouped nested evaluation has not produced paired bootstrap intervals"
        ),
    )
    register(
        "oof_observed_vs_predicted",
        [] if smoke else plot_oof_primary(oof, figure_dir / "oof_observed_vs_predicted", formats, dpi),
        "smoke mode does not generate performance figures" if smoke else "aggregated OOF predictions are unavailable",
    )
    register(
        "oof_residual_diagnostics",
        [] if smoke else plot_residual_diagnostics(oof, figure_dir / "oof_residual_diagnostics", formats, dpi),
        "smoke mode does not generate performance figures" if smoke else "aggregated OOF predictions are unavailable",
    )

    released_panel_blocks, release_reason = _released_panel_blocks(
        interpretation_stability,
        interpretation_status,
    )
    released_importance = _filter_released_panel_blocks(importance, released_panel_blocks)
    block_figure_paths = (
        []
        if smoke or not released_panel_blocks
        else plot_outer_test_block_importance(
            released_importance,
            figure_dir / "outer_test_block_ablation_permutation",
            formats,
            dpi,
            released_panel_blocks=released_panel_blocks,
        )
    )
    register(
        "outer_test_block_ablation_permutation",
        block_figure_paths,
        (
            "smoke mode intentionally skips outer-test conditional associational sensitivity; "
            "not a feature ranking or causal effect"
            if smoke
            else (
                release_reason
                if not released_panel_blocks
                else (
                    "stability passed, but matching primary alkaline outer-test perturbations are unavailable; "
                    f"{_CONDITIONAL_ASSOCIATIONAL_BOUNDARY}"
                )
            )
        ),
        status_if_missing="not_generated" if smoke else "gated",
        reason_if_generated=release_reason,
    )

    # These are intentionally conditional interpretation products.  Absence is
    # recorded rather than replaced by an unstable or misleading plot.
    interpretation_by_artifact = (
        interpretation_manifest.set_index("artifact").to_dict("index")
        if not interpretation_manifest.empty and "artifact" in interpretation_manifest
        else {}
    )
    substitution = interpretation_by_artifact.get("composition_conserving_solvent_substitution", {})
    neighbors = interpretation_by_artifact.get("knn_neighbor_case_studies", {})
    rows.extend(
        [
            {
                "figure": "composition_conserving_substitution",
                "path": "",
                "status": str(substitution.get("status", "not_estimable_from_saved_artifacts")),
                "reason": str(
                    substitution.get(
                        "reason",
                        "requires a passed stability gate and retained outer-fold estimator state",
                    )
                ),
            },
            {
                "figure": "knn_neighbor_cases",
                "path": "",
                "status": str(neighbors.get("status", "not_estimable_from_saved_artifacts")),
                "reason": str(
                    neighbors.get(
                        "reason",
                        "requires kNN to win every registered one-SE fold and retained neighbour state",
                    )
                ),
            },
            {
                "figure": "two_regime_nuts_ppc",
                "path": "",
                "status": "gated",
                "reason": "selected-model NUTS plus named PPC review is pending for potential-dependent curves",
            },
        ]
    )
    source_rows: list[dict[str, object]] = []
    if not target_artifacts.empty and {"artifact_name", "path", "sha256", "size_bytes"}.issubset(target_artifacts):
        fit_source = target_artifacts.loc[target_artifacts["artifact_name"].astype(str).eq("fit_summary")]
        if not fit_source.empty:
            refit_root = Path(str(fit_source.iloc[0]["path"])).parent.parent
            refit_manifest_path = refit_root / "csv" / "figure_manifest.csv"
            if refit_manifest_path.is_file():
                refit_manifest = pd.read_csv(refit_manifest_path)
                for _, item in refit_manifest.iterrows():
                    source_rows.append(
                        {
                            "artifact_type": "deterministic_curve_qc",
                            "path": str(refit_root / str(item["relative_path"])),
                            "sha256": item["sha256"],
                            "size_bytes": item["size_bytes"],
                            "review_status": item.get("status", "generated"),
                        }
                    )
        ppc = target_artifacts.loc[target_artifacts["path"].astype(str).str.endswith("_posterior_predictive.png")]
        for _, item in ppc.iterrows():
            source_rows.append(
                {
                    "artifact_type": "single_regime_bayesian_ppc",
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                    "review_status": "sampler_pass_named_ppc_review_pending",
                }
            )
    source_manifest = pd.DataFrame(source_rows)
    source_manifest_path = figure_dir / "source_curve_artifact_manifest.csv"
    source_manifest.to_csv(source_manifest_path, index=False)
    if not source_manifest.empty:
        rows.append(
            {
                "figure": "curve_qc_ppc_source_artifacts",
                "path": str(source_manifest_path.relative_to(root)),
                "status": "linked_manifest",
                "reason": f"{len(source_manifest)} hashed source figures; PPC review status retained",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(figure_dir / "figure_manifest.csv", index=False)
    return manifest
