#!/usr/bin/env python3
"""Fit Limaye Equation 2 inside frozen strict contiguous-increase windows.

This is a paired model-form sensitivity.  It reuses one representative window
from the corrected contiguous-increase candidate for each supported curve and
changes only the Bayesian observation/model specification to the Limaye et al.
released Equation-2 model:

* one kinetic slope plus a limiting-current term;
* per-curve shifted log10 current;
* fixed Gaussian sigma of 0.10 log10 units;
* Uniform(0, 10 * theta_star) priors after data preconditioning.

The output is sensitivity evidence only.  It never updates RESULTS, canonical
data, ML exports, or human PPC decisions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code/Tafel_refitting/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import uniform_equation2_targets as eq2


SOURCE_POLICY_ID = "bayesian_first_supported_linear_eta015_contiguous_increase_v3"
POLICY_ID = "bayesian_limaye_eq2_eta015_contiguous_increase_representative_v1"
TARGET_ID = "contiguous_increase_limaye_eq2"
TARGET_NAME = "b_contiguous_increase_limaye_eq2_mV_dec"
TARGET_ROLE = "paired_model_form_sensitivity_not_canonical"
DEFAULT_CANDIDATE_DIR = (
    ROOT
    / "Data/03_outputs/tafel_refitting/outputs/"
    "20260723_first_supported_contiguous_increase_dry_run_v2"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "Data/03_outputs/tafel_refitting/outputs/"
    "20260723_contiguous_increase_limaye_eq2_sensitivity_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-records", type=int, nargs="+")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def read_candidate(
    candidate_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_path = candidate_dir / "target_summary.csv"
    points_path = candidate_dir / "selected_region_points.csv.gz"
    qa_path = candidate_dir / "qa.json"
    for path in (summary_path, points_path, qa_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = pd.read_csv(summary_path)
    points = pd.read_csv(points_path)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if len(summary) != 97 or summary["sample_record"].nunique() != 97:
        raise ValueError("The source candidate must contain 97 unique records")
    if set(summary["policy_id"].dropna()) != {SOURCE_POLICY_ID}:
        raise ValueError("Unexpected source target policy")
    eligible = summary["target_structurally_eligible"].astype(bool)
    if int(eligible.sum()) != 95:
        raise ValueError("Expected 95 source-supported records")
    unsupported = summary.loc[~eligible, "sample_record"].astype(int).tolist()
    if unsupported != [23, 74]:
        raise ValueError(f"Unexpected unsupported records: {unsupported}")
    if not qa.get("all_supported_windows_strictly_increasing", False):
        raise ValueError("Source candidate did not pass strict-progression QA")
    source = {
        "candidate_dir": str(candidate_dir.resolve()),
        "target_summary_path": str(summary_path.resolve()),
        "target_summary_sha256": eq2.sha256_file(summary_path),
        "selected_region_points_path": str(points_path.resolve()),
        "selected_region_points_sha256": eq2.sha256_file(points_path),
        "qa_path": str(qa_path.resolve()),
        "qa_sha256": eq2.sha256_file(qa_path),
        "source_policy_id": SOURCE_POLICY_ID,
    }
    return summary, points, source


def choose_representative_window(
    summary_row: pd.Series, record_points: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    candidates: list[tuple[float, int]] = []
    target_start = float(summary_row["window_start_eta_median_V"])
    target_end = float(summary_row["window_end_eta_median_V"])
    for replicate, group in record_points.groupby("selection_replicate", sort=True):
        observed_eta = pd.to_numeric(group["observed_eta_V"], errors="raise")
        distance = abs(float(observed_eta.min()) - target_start) + abs(
            float(observed_eta.max()) - target_end
        )
        candidates.append((distance, int(replicate)))
    if not candidates:
        raise ValueError(f"Record {int(summary_row['sample_record'])} has no windows")
    replicate = min(candidates)[1]
    selected = record_points.loc[
        record_points["selection_replicate"].astype(int).eq(replicate)
    ].copy()
    selected = selected.sort_values(
        ["observed_potential_V", "potential_bin_index"], kind="mergesort"
    ).reset_index(drop=True)
    observed_logj = pd.to_numeric(
        selected["observed_log10_current"], errors="raise"
    ).to_numpy(dtype=float)
    observed_potential = pd.to_numeric(
        selected["observed_potential_V"], errors="raise"
    ).to_numpy(dtype=float)
    if len(selected) < 6 or np.any(np.diff(observed_logj) <= 0):
        raise ValueError("Representative window violates strict progression")
    if np.any(np.diff(observed_potential) <= 0):
        raise ValueError("Representative window potential must strictly increase")
    if float(np.ptp(observed_logj)) < 0.5 - 1e-12:
        raise ValueError("Representative window has insufficient current span")

    selected["target_id"] = TARGET_ID
    selected["target_name"] = TARGET_NAME
    selected["potential_V"] = observed_potential
    selected["eta_V"] = observed_potential - eq2.OER_EQUILIBRIUM_POTENTIAL_V_RHE
    selected["fit_relative_potential_V"] = observed_potential - observed_potential.min()
    selected["unshifted_log10_current"] = observed_logj
    selected["fit_log10_current"] = observed_logj - observed_logj.min()
    selected["observation_id"] = pd.to_numeric(
        selected["potential_bin_index"], errors="raise"
    ).astype(int)
    selected["representative_selection_replicate"] = replicate
    return selected, replicate


def build_manifest_and_points(
    summary: pd.DataFrame, points: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for _, row in summary.sort_values("sample_record").iterrows():
        if not bool(row["target_structurally_eligible"]):
            continue
        record = int(row["sample_record"])
        curve_id = str(row["curve_id"])
        record_points = points.loc[
            points["sample_record_canonical"].astype(int).eq(record)
            & points["curve_id_canonical"].astype(str).eq(curve_id)
        ].copy()
        selected, replicate = choose_representative_window(row, record_points)
        x = selected["fit_relative_potential_V"].to_numpy(dtype=float)
        y = selected["fit_log10_current"].to_numpy(dtype=float)
        candidate_slope = float(row["posterior_target_median_mV_dec"])
        alpha_guess = float(1000.0 / (eq2.LOG10E * candidate_slope))
        preconditioner = eq2.optimize_paper_preconditioner(
            x,
            y,
            project_initial=(alpha_guess, 2.0, 15.0),
        )
        manifest_rows.append(
            {
                "target_id": TARGET_ID,
                "target_name": TARGET_NAME,
                "target_role": TARGET_ROLE,
                "curve_id": curve_id,
                "sample_record": record,
                "model_name": eq2.MODEL_NAME,
                "equation": "Limaye_et_al_Equation_2_single_regime_plus_plateau",
                "input_kind": "representative_unperturbed_strict_contiguous_increase_bins",
                "selection_rule": (
                    "frozen_representative_window_from_contiguous_increase_v3"
                ),
                "source_policy_id": SOURCE_POLICY_ID,
                "representative_selection_replicate": replicate,
                "eta_min_V": eq2.ETA_MIN_V,
                "start_eta_V": float(selected["eta_V"].min()),
                "end_eta_V": float(selected["eta_V"].max()),
                "n_observations": int(len(selected)),
                "log_current_span_dec": float(np.ptp(y)),
                "all_adjacent_observed_current_steps_positive": bool(
                    np.all(np.diff(selected["unshifted_log10_current"]) > 0)
                ),
                "likelihood": "iid_gaussian_fixed_sigma_0p10_log10",
                "prior_policy": "limaye_uniform_0_to_10x_theta_star",
                "paper_log_current_shift_dec": float(
                    selected["unshifted_log10_current"].min()
                ),
                "paper_fixed_sigma_log10": eq2.PAPER_FIXED_SIGMA_LOG10,
                "paper_prior_multiplier": eq2.PAPER_PRIOR_MULTIPLIER,
                **preconditioner,
            }
        )
        selected_frames.append(selected)
    manifest = pd.DataFrame(manifest_rows).sort_values("sample_record").reset_index(
        drop=True
    )
    selected_points = pd.concat(selected_frames, ignore_index=True, sort=False)
    if len(manifest) != 95 or manifest["sample_record"].nunique() != 95:
        raise ValueError("Expected 95 representative strict windows")
    if not manifest["all_adjacent_observed_current_steps_positive"].all():
        raise ValueError("Strict progression audit failed")
    return manifest, selected_points


def load_completed(
    fit_dir: Path,
    *,
    row: pd.Series,
    seed: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    selected: pd.DataFrame,
    configuration_sha256: str,
) -> dict[str, Any] | None:
    return eq2._load_completed_fit(
        fit_dir,
        target_id=TARGET_ID,
        curve_id=str(row["curve_id"]),
        sample_record=int(row["sample_record"]),
        seed=seed,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        points_sha256=eq2.dataframe_sha256(
            selected.sort_values(
                ["potential_V", "observation_id"], kind="mergesort"
            ).reset_index(drop=True)
        ),
        configuration_sha256=configuration_sha256,
    )


def write_aggregates(
    output_dir: Path,
    candidate_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    completed: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    fit_results = pd.DataFrame(completed + failures)
    fitted = manifest.merge(
        fit_results,
        on=["target_id", "curve_id", "sample_record"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_fit"),
    )
    base_columns = [
        "sample_record",
        "curve_id",
        "target_structurally_eligible",
        "posterior_target_median_mV_dec",
        "posterior_target_hdi_low_mV_dec",
        "posterior_target_hdi_high_mV_dec",
        "window_start_eta_median_V",
        "window_end_eta_median_V",
    ]
    result = candidate_summary[base_columns].rename(
        columns={
            "target_structurally_eligible": "source_window_supported",
            "posterior_target_median_mV_dec": "local_linear_median_mV_dec",
            "posterior_target_hdi_low_mV_dec": "local_linear_hdi_low_mV_dec",
            "posterior_target_hdi_high_mV_dec": "local_linear_hdi_high_mV_dec",
        }
    )
    result = result.merge(
        fitted,
        on=["sample_record", "curve_id"],
        how="left",
        validate="one_to_one",
    )
    unsupported = ~result["source_window_supported"].astype(bool)
    result.loc[unsupported, "fit_status"] = "unsupported_no_strict_window"
    result.insert(0, "policy_id", POLICY_ID)
    if "target_name" in result:
        result["target_name"] = TARGET_NAME
    else:
        result.insert(1, "target_name", TARGET_NAME)
    result["eq2_minus_local_linear_mV_dec"] = (
        pd.to_numeric(result["posterior_slope_median_mV_dec"], errors="coerce")
        - pd.to_numeric(result["local_linear_median_mV_dec"], errors="coerce")
    )
    eq2._atomic_csv(result, output_dir / "target_summary.csv")
    complete = result["fit_status"].eq(eq2.COMPLETE_STATUS)
    draw_frames = [
        pd.read_csv(path)
        for path in result.loc[complete, "slope_draws_path"].dropna().map(Path)
        if path.is_file()
    ]
    if draw_frames:
        eq2._atomic_csv(
            pd.concat(draw_frames, ignore_index=True),
            output_dir / "posterior_slope_draws.csv.gz",
            compression="gzip",
        )
    return result


def positive_kde(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_values = np.log10(values)
    grid = np.geomspace(
        10 ** (log_values.min() - 0.22),
        10 ** (log_values.max() + 0.22),
        900,
    )
    density = gaussian_kde(log_values, bw_method="scott")(np.log10(grid))
    density /= grid * np.log(10.0)
    return grid, density


def write_comparison(output_dir: Path, result: pd.DataFrame) -> dict[str, Any]:
    complete = result["fit_status"].eq(eq2.COMPLETE_STATUS)
    paired = result.loc[
        complete,
        [
            "sample_record",
            "curve_id",
            "local_linear_median_mV_dec",
            "posterior_slope_median_mV_dec",
            "posterior_slope_hdi_low_mV_dec",
            "posterior_slope_hdi_high_mV_dec",
            "eq2_minus_local_linear_mV_dec",
            "sampler_diagnostics_pass",
            "posterior_prior_boundary_warning",
        ],
    ].copy()
    eq2._atomic_csv(paired, output_dir / "paired_comparison.csv")
    local = paired["local_linear_median_mV_dec"].to_numpy(dtype=float)
    limaye = paired["posterior_slope_median_mV_dec"].to_numpy(dtype=float)
    delta = limaye - local
    comparison = {
        "paired_record_count": int(len(paired)),
        "local_linear_median_mV_dec": float(np.median(local)),
        "limaye_eq2_median_mV_dec": float(np.median(limaye)),
        "median_paired_delta_eq2_minus_local_mV_dec": float(np.median(delta)),
        "median_absolute_paired_delta_mV_dec": float(np.median(np.abs(delta))),
        "spearman_rho": float(spearmanr(local, limaye).statistic),
        "local_linear_above_500_count": int(np.sum(local > 500)),
        "limaye_eq2_above_500_count": int(np.sum(limaye > 500)),
        "sampler_diagnostics_pass_count": int(
            paired["sampler_diagnostics_pass"].astype(bool).sum()
        ),
        "prior_boundary_warning_count": int(
            paired["posterior_prior_boundary_warning"].astype(bool).sum()
        ),
    }
    eq2._atomic_json(output_dir / "comparison_summary.json", comparison)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    local_grid, local_density = positive_kde(local)
    limaye_grid, limaye_density = positive_kde(limaye)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(local_grid, local_density, color="#0F4D92", lw=2.5, label="Bayesian local line")
    ax.fill_between(local_grid, local_density, color="#3775BA", alpha=0.14)
    ax.plot(limaye_grid, limaye_density, color="#B64342", lw=2.5, label="Limaye Equation 2")
    ax.fill_between(limaye_grid, limaye_density, color="#E9A6A1", alpha=0.16)
    ax.set_xscale("log")
    ax.set_xlabel(r"Bayesian Tafel slope (mV dec$^{-1}$)")
    ax.set_ylabel("Probability density")
    ax.set_title(
        "Same strict contiguous-increase windows, different Bayesian model",
        loc="left",
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout(pad=1.6)
    fig.savefig(output_dir / "local_linear_vs_limaye_eq2_kde.png", dpi=300)
    fig.savefig(output_dir / "local_linear_vs_limaye_eq2_kde.pdf")
    plt.close(fig)

    limit = float(max(local.max(), limaye.max()) * 1.06)
    fig, ax = plt.subplots(figsize=(6.1, 5.9))
    ax.scatter(local, limaye, s=38, alpha=0.76, color="#0F4D92")
    ax.plot([0, limit], [0, limit], color="#767676", ls="--", lw=1.5)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Bayesian local-line slope (mV dec$^{-1}$)")
    ax.set_ylabel(r"Limaye Equation-2 slope (mV dec$^{-1}$)")
    ax.set_title(
        f"Paired model-form sensitivity (n = {len(paired)})",
        loc="left",
        fontweight="bold",
    )
    fig.tight_layout(pad=1.6)
    fig.savefig(output_dir / "local_linear_vs_limaye_eq2_scatter.png", dpi=300)
    fig.savefig(output_dir / "local_linear_vs_limaye_eq2_scatter.pdf")
    plt.close(fig)
    return comparison


def write_readme(output_dir: Path, status: dict[str, Any]) -> None:
    text = f"""# Strict contiguous-increase + Limaye Equation-2 sensitivity

- Source window policy: `{SOURCE_POLICY_ID}`
- Sensitivity policy: `{POLICY_ID}`
- Target: `{TARGET_NAME}`
- Analysis role: `{TARGET_ROLE}`
- Source records: 97
- Strict-window support: 95
- Unsupported without fallback: Records 23 and 74
- Completed Bayesian fits: {status['bayesian_sampling_complete_count']}
- Sampler diagnostics pass: {status['sampler_diagnostics_pass_count']}

This is a paired model-form sensitivity. Each supported record uses the
representative unperturbed window from the corrected strict
contiguous-increase candidate. Only the Bayesian curve model changes: the
local Student-t line is replaced by Limaye et al. Equation 2 with the released
fixed-sigma likelihood and data-preconditioned priors. Window selection is not
rerun under Equation 2.

This package does not update `Data/04_results/RESULTS/`, canonical data, ML exports, or human
PPC decisions. The Equation-2 slope remains a sensitivity target, not the
registered ML response.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.smoke:
        draws, tune, chains = 40, 40, 2
        requested_records = set(args.sample_records or [1, 89, 97])
    else:
        draws, tune, chains = args.draws, args.tune, args.chains
        requested_records = set(args.sample_records or range(1, 98))
    if draws < 1 or tune < 0 or chains < 2:
        raise ValueError("draws>=1, tune>=0, chains>=2 are required")
    if not 0.5 < args.target_accept < 1:
        raise ValueError("target_accept must lie between 0.5 and 1")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_summary, source_points, source = read_candidate(
        args.candidate_dir.resolve()
    )
    manifest, selected_points = build_manifest_and_points(
        candidate_summary, source_points
    )
    requested = manifest.loc[
        manifest["sample_record"].astype(int).isin(requested_records)
    ].copy()
    configuration = {
        "policy_id": POLICY_ID,
        "target_id": TARGET_ID,
        "target_name": TARGET_NAME,
        "target_role": TARGET_ROLE,
        "source_policy_id": SOURCE_POLICY_ID,
        "model_variant": eq2.TARGET_EQ2_PAPER,
        "equation": "Limaye_et_al_Equation_2_single_regime_plus_plateau",
        "window_policy": "frozen_representative_unperturbed_strict_window",
        "window_selection_uncertainty_propagated": False,
        "measurement_jitter_propagated": False,
        "paper_fixed_sigma_log10": eq2.PAPER_FIXED_SIGMA_LOG10,
        "paper_prior_multiplier": eq2.PAPER_PRIOR_MULTIPLIER,
        "draws_per_chain": draws,
        "tune_per_chain": tune,
        "chains": chains,
        "target_accept": args.target_accept,
        "base_seed": args.seed,
        "smoke": bool(args.smoke),
        "requested_records": sorted(requested_records),
        "manifest_sha256": eq2.dataframe_sha256(manifest),
        "selected_points_sha256": eq2.dataframe_sha256(selected_points),
        "source": source,
    }
    config_path = output_dir / "run_configuration.json"
    manifest_path = output_dir / "target_manifest.csv"
    points_path = output_dir / "selected_target_points.csv.gz"
    if args.resume:
        if not all(path.is_file() for path in (config_path, manifest_path, points_path)):
            raise FileNotFoundError("--resume requires configuration and input files")
        if json.loads(config_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Resume configuration differs from the existing run")
    else:
        eq2._atomic_csv(manifest, manifest_path)
        eq2._atomic_csv(selected_points, points_path, compression="gzip")
        eq2._atomic_json(config_path, configuration)
    config_hash = eq2.sha256_file(config_path)

    lookup = {
        str(curve): frame.copy()
        for curve, frame in selected_points.groupby("curve_id", sort=False)
    }
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(
        requested.sort_values("sample_record").iterrows(), start=1
    ):
        record = int(row["sample_record"])
        curve = str(row["curve_id"])
        seed = int(args.seed + record * 10_000)
        frame = lookup[curve].sort_values(
            ["potential_V", "observation_id"], kind="mergesort"
        ).reset_index(drop=True)
        fit_dir = output_dir / "bayes" / f"record_{record:03d}__curve_{curve}"
        previous = (
            load_completed(
                fit_dir,
                row=row,
                seed=seed,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=args.target_accept,
                selected=frame,
                configuration_sha256=config_hash,
            )
            if args.resume
            else None
        )
        if previous is not None:
            completed.append(previous)
            print(f"[{index:03d}/{len(requested):03d}] resume Record {record:03d}")
            continue
        print(f"[{index:03d}/{len(requested):03d}] fit Record {record:03d} {curve}")
        try:
            completed.append(
                eq2.fit_one(
                    frame,
                    row,
                    final_dir=fit_dir,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=args.target_accept,
                    seed=seed,
                    progressbar=not args.no_progress,
                    configuration_sha256=config_hash,
                    model_variant=eq2.TARGET_EQ2_PAPER,
                    analysis_role=TARGET_ROLE,
                    target_name=TARGET_NAME,
                )
            )
        except Exception as exc:
            failure = {
                "fit_status": eq2.FAILED_STATUS,
                "target_id": TARGET_ID,
                "target_name": TARGET_NAME,
                "curve_id": curve,
                "sample_record": record,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failure_dir = output_dir / "failures"
            failure_dir.mkdir(exist_ok=True)
            eq2._atomic_json(
                failure_dir / f"record_{record:03d}__curve_{curve}.json", failure
            )
            failures.append(failure)
            if args.fail_fast:
                raise

    result = write_aggregates(
        output_dir, candidate_summary, manifest, completed, failures
    )
    requested_supported = int(
        manifest["sample_record"].astype(int).isin(requested_records).sum()
    )
    complete_count = len(completed)
    diagnostics_pass_count = int(
        sum(bool(item.get("sampler_diagnostics_pass")) for item in completed)
    )
    status = {
        "completed_at_utc": eq2.utc_now(),
        "policy_id": POLICY_ID,
        "analysis_role": TARGET_ROLE,
        "canonical_target_modified": False,
        "mode": "smoke" if args.smoke else "full",
        "requested_supported_fits": requested_supported,
        "bayesian_sampling_complete_count": complete_count,
        "bayesian_sampling_failed_count": len(failures),
        "sampler_diagnostics_pass_count": diagnostics_pass_count,
        "sampler_diagnostics_not_required_for_short_smoke": bool(args.smoke),
        "all_requested_sampling_complete": bool(
            complete_count == requested_supported and not failures
        ),
        "analysis_ready": bool(
            complete_count == requested_supported
            and not failures
            and (args.smoke or diagnostics_pass_count == requested_supported)
        ),
    }
    eq2._atomic_json(output_dir / "completion_status.json", status)
    if complete_count:
        comparison = write_comparison(output_dir, result)
        status["comparison"] = comparison
        eq2._atomic_json(output_dir / "completion_status.json", status)
    write_readme(output_dir, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["analysis_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
