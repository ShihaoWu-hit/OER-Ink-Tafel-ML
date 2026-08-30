from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import CurveAnalysis
from .models import model_spec
from .preprocess import ProcessedCurve


MODEL_COLORS = {
    "linear": "#4C78A8",
    "paper_single_plateau": "#F58518",
    "two_regime_plateau": "#54A24B",
}


def _figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_curve(
    processed: ProcessedCurve,
    analysis: CurveAnalysis,
    output_path: Path,
) -> None:
    _figure_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    raw = processed.raw_points
    selected = raw.loc[raw["branch_selected"] & raw["finite_pair"]].copy()
    pre_reliable_positive = selected.loc[
        selected.get(
            "pre_reliable_onset_positive",
            pd.Series(False, index=selected.index),
        ).astype(bool)
    ].copy()
    positive = selected.loc[
        (selected["current_raw"] > 0)
        & ~selected.index.isin(pre_reliable_positive.index)
    ].copy()
    if len(processed.binned_points):
        potential_origin = float(processed.binned_points["potential_V"].min())
    elif len(positive):
        potential_origin = float(positive["potential_raw"].min())
    else:
        potential_origin = 0.0
    if len(positive):
        x_raw = positive["potential_raw"] - potential_origin
        axes[0].scatter(
            x_raw,
            np.log10(positive["current_raw"]),
            s=8,
            alpha=0.22,
            color="#7F7F7F",
            linewidths=0,
            label="reliable-region positive points",
        )
    if len(pre_reliable_positive):
        axes[0].scatter(
            pre_reliable_positive["potential_raw"] - potential_origin,
            np.log10(pre_reliable_positive["current_raw"]),
            s=18,
            marker="o",
            facecolors="none",
            edgecolors="#F58518",
            linewidths=0.8,
            label="pre-reliable onset positive (not fit)",
            zorder=3,
        )
    display_proxy = float(
        processed.metrics.get("left_censoring_display_proxy_raw", np.nan)
    )
    if np.isfinite(display_proxy) and display_proxy > 0:
        proxy_log = float(np.log10(display_proxy))
        if "left_censored_nonpositive" in selected:
            censored = selected.loc[selected["left_censored_nonpositive"]]
            if len(censored):
                axes[0].scatter(
                    censored["potential_raw"] - potential_origin,
                    np.full(len(censored), proxy_log),
                    s=22,
                    marker="v",
                    facecolors="none",
                    edgecolors="#4C78A8",
                    linewidths=0.9,
                    label="left-censored status (display proxy)",
                    zorder=3,
                )
        if "nonpositive_after_first_reliable_positive" in selected:
            invalid = selected.loc[
                selected["nonpositive_after_first_reliable_positive"]
            ]
            if len(invalid):
                axes[0].scatter(
                    invalid["potential_raw"] - potential_origin,
                    np.full(len(invalid), proxy_log),
                    s=24,
                    marker="x",
                    color="#E45756",
                    linewidths=1.1,
                    label="nonpositive after reliable region (QC fail)",
                    zorder=5,
                )
    if len(processed.binned_points):
        binned = processed.binned_points
        axes[0].scatter(
            binned["relative_potential_V"],
            binned["log10_current"],
            s=16,
            color="black",
            label="equal-potential bins",
            zorder=4,
        )
    if len(analysis.fitted_curves):
        for model_name, group in analysis.fitted_curves.groupby("model_name", sort=False):
            selected_model = analysis.summary.get("selected_model") == model_name
            axes[0].plot(
                group["relative_potential_V"],
                group["predicted_log10_current"],
                color=MODEL_COLORS.get(model_name, "#7F7F7F"),
                linewidth=2.2 if selected_model else 1.1,
                alpha=1.0 if selected_model else 0.65,
                label=(
                    f"{model_spec(model_name).display_name}"
                    f"{' (selected)' if selected_model else ''}"
                ),
            )
    axes[0].set_xlabel("Relative potential / V")
    axes[0].set_ylabel(r"$\log_{10}(j)$")
    axes[0].set_title("Curve and primary structural variants")
    axes[0].legend(frameon=False, loc="best")

    local = analysis.local_slopes
    if len(local):
        x = local["relative_potential_V"].to_numpy(dtype=float)
        axes[1].plot(
            x,
            local["kinetic_slope_mV_dec"],
            color="#4C78A8",
            linewidth=2,
            label="model kinetic slope",
        )
        axes[1].plot(
            x,
            local["apparent_slope_mV_dec"],
            color="#E45756",
            linewidth=1.8,
            label="model apparent slope",
        )
        if "kinetic_slope_ci_low_mV_dec" in local:
            axes[1].fill_between(
                x,
                local["kinetic_slope_ci_low_mV_dec"],
                local["kinetic_slope_ci_high_mV_dec"],
                color="#4C78A8",
                alpha=0.15,
                linewidth=0,
            )
        spline = local["spline_apparent_slope_mV_dec"].to_numpy(dtype=float)
        finite_spline = np.isfinite(spline) & (spline < 2000)
        if finite_spline.any():
            axes[1].plot(
                x[finite_spline],
                spline[finite_spline],
                color="#B279A2",
                linestyle="--",
                linewidth=1.2,
                label="spline diagnostic",
            )
        finite_values = np.concatenate(
            [
                local["kinetic_slope_mV_dec"].to_numpy(dtype=float),
                local["apparent_slope_mV_dec"].to_numpy(dtype=float),
            ]
        )
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values):
            upper = min(1000.0, max(150.0, float(np.quantile(finite_values, 0.98) * 1.15)))
            axes[1].set_ylim(0, upper)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Not fit: " + str(analysis.summary.get("fit_exclusion_reason", "unknown")),
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            wrap=True,
        )
    axes[1].axhline(40, color="#BBBBBB", linewidth=0.7, linestyle=":")
    axes[1].axhline(60, color="#BBBBBB", linewidth=0.7, linestyle=":")
    axes[1].axhline(120, color="#BBBBBB", linewidth=0.7, linestyle=":")
    axes[1].set_xlabel("Relative potential / V")
    axes[1].set_ylabel(r"Tafel slope / mV dec$^{-1}$")
    axes[1].set_title("Potential-dependent slope diagnostic")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, frameon=False, loc="best")
    figure.suptitle(
        f"{processed.curve_id} | kinetics="
        f"{analysis.summary.get('kinetic_structure', 'not_fit')} | "
        f"observation={analysis.summary.get('observation_parameterization', 'not_fit')} | "
        f"QC={analysis.summary.get('fit_status', 'unknown')}",
        fontsize=11,
    )
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_summary(curve_summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    _figure_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    counts = curve_summary["kinetic_structure"].fillna("not_fit").value_counts()
    labels = {
        "constant": "constant slope",
        "potential_dependent": "potential-dependent slope",
        "not_fit": "not fit",
    }
    display_index = [labels.get(str(value), str(value)) for value in counts.index]
    figure, axis = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    axis.bar(display_index, counts.values, color="#4C78A8")
    axis.set_ylabel("Number of curves")
    axis.set_title("Selected kinetic structure after blocked validation")
    for index, value in enumerate(counts.values):
        axis.text(index, value + 0.3, str(value), ha="center", va="bottom")
    path = figure_dir / "selected_kinetic_structure_counts.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    usable = curve_summary.loc[
        np.isfinite(pd.to_numeric(curve_summary.get("refit_tafel_slope_mV_dec"), errors="coerce"))
        & np.isfinite(pd.to_numeric(curve_summary["existing_tafel_slope_mV_dec"], errors="coerce"))
    ].copy()
    if len(usable):
        figure, axis = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
        color = np.where(usable["label_eligible"].fillna(False), "#54A24B", "#E45756")
        axis.scatter(
            usable["existing_tafel_slope_mV_dec"],
            usable["refit_tafel_slope_mV_dec"],
            c=color,
            alpha=0.75,
            s=28,
        )
        limit = float(
            np.nanmax(
                [
                    usable["existing_tafel_slope_mV_dec"].max(),
                    usable["refit_tafel_slope_mV_dec"].max(),
                ]
            )
        )
        axis.plot([0, limit], [0, limit], color="#777777", linestyle="--", linewidth=1)
        axis.set_xlim(0, limit * 1.04)
        axis.set_ylim(0, limit * 1.04)
        axis.set_xlabel(r"Existing slope / mV dec$^{-1}$")
        axis.set_ylabel(r"Refit single-regime slope / mV dec$^{-1}$")
        axis.set_title("Existing versus refit scalar slopes")
        path = figure_dir / "existing_vs_refit_slopes.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    return paths


def build_figure_manifest(paths: Iterable[Path], run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "relative_path": str(path.relative_to(run_dir)),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "status": "generated",
            }
        )
    return pd.DataFrame(rows)
