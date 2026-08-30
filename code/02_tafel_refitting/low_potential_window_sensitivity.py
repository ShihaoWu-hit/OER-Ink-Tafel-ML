#!/usr/bin/env python3
"""Bayesian low-potential Tafel-slope sensitivity at 1.5 and 2.0 decades.

This is deliberately separate from the formal 97-curve workflow. It uses a
common Bayesian window-local linear estimand in both low-potential windows so
that the comparison changes the likelihood window, not the slope definition.
The formal selected-model posterior is carried through as a reference only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "Data/03_outputs/tafel_refitting/outputs/20260721_112453"
DEFAULT_BASELINE_INVENTORY = (
    ROOT
    / "Data/03_outputs/tafel_refitting/outputs/20260721_112453_v5_low_potential/"
    "bayesian_batch/bayesian_artifact_inventory.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "Data/03_outputs/tafel_refitting/outputs/"
    "20260721_112453_low_potential_window_sensitivity"
)
WINDOWS = (1.5, 2.0)
WINDOW_MODEL = "linear"
WINDOW_SLOPE_VARIABLE = "slope_mV_dec"
METHOD = "bayesian_student_t_nu4_local_linear"
ANALYSIS_ROLE = "sensitivity_only_not_primary_target"
STABILITY_RATIO_THRESHOLD = 1.25
SAMPLING_COMPLETE_STATUSES = frozenset({"bayesian_sampling_complete"})
SAMPLING_FAILED_STATUSES = frozenset({"bayesian_sampling_failed"})


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def window_token(window_decades: float) -> str:
    return f"{float(window_decades):.1f}".replace(".", "p")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )


def _initial_equal_value_count(values: np.ndarray) -> int:
    if not len(values):
        return 0
    first = float(values[0])
    count = 0
    for value in values:
        if not np.isclose(float(value), first, rtol=0.0, atol=1e-12):
            break
        count += 1
    return count


def select_low_potential_window(
    curve_points: pd.DataFrame,
    window_decades: float,
    *,
    min_bins: int = 12,
    potential_span_flag_V: float = 0.08,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select the shortest continuous low-potential prefix reaching a span.

    The prefix is ordered by relative potential.  Its log-current span is the
    running maximum minus running minimum; this remains correct when the first
    bin is not the minimum-current bin.  The boundary-crossing observed bin is
    included, and no interpolation or extrapolation is used.
    """

    required = {
        "curve_id",
        "sample_record",
        "potential_bin_index",
        "potential_V",
        "relative_potential_V",
        "log10_current",
    }
    missing = sorted(required.difference(curve_points.columns))
    if missing:
        raise ValueError(f"Window input is missing columns: {missing}")
    if not np.isfinite(window_decades) or float(window_decades) <= 0:
        raise ValueError("window_decades must be finite and positive")
    if min_bins < 3:
        raise ValueError("min_bins must be at least 3")
    if len(curve_points) < 1:
        raise ValueError("curve_points must not be empty")
    curve_ids = curve_points["curve_id"].astype(str).unique()
    records = pd.to_numeric(curve_points["sample_record"], errors="raise").unique()
    if len(curve_ids) != 1 or len(records) != 1:
        raise ValueError("curve_points must describe exactly one curve and record")

    ordered = curve_points.sort_values(
        ["relative_potential_V", "potential_bin_index"], kind="mergesort"
    ).reset_index(drop=True)
    numeric_columns = [
        "potential_V",
        "relative_potential_V",
        "log10_current",
    ]
    numeric = ordered[numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite binned values for {curve_ids[0]}")
    if ordered["relative_potential_V"].duplicated().any():
        raise ValueError(f"Duplicate binned relative potentials for {curve_ids[0]}")

    y = ordered["log10_current"].to_numpy(dtype=float)
    x = ordered["relative_potential_V"].to_numpy(dtype=float)
    running_span = np.maximum.accumulate(y) - np.minimum.accumulate(y)
    full_span = float(np.max(y) - np.min(y))
    crossing = np.flatnonzero(running_span >= float(window_decades) - 1e-12)
    span_supported = bool(len(crossing))

    if span_supported:
        stop = int(crossing[0]) + 1
        selected = ordered.iloc[:stop].copy()
        achieved_span = float(running_span[stop - 1])
        potential_span = float(np.ptp(x[:stop]))
        spearman = float(
            pd.Series(x[:stop]).corr(pd.Series(y[:stop]), method="spearman")
        )
        n_unique_log_current = int(pd.Series(y[:stop]).nunique(dropna=True))
        initial_equal_bins = _initial_equal_value_count(y[:stop])
    else:
        selected = ordered.iloc[0:0].copy()
        achieved_span = np.nan
        potential_span = np.nan
        spearman = np.nan
        n_unique_log_current = 0
        initial_equal_bins = 0

    n_window_bins = int(len(selected))
    min_bins_pass = bool(span_supported and n_window_bins >= min_bins)
    positive_spearman_pass = bool(span_supported and np.isfinite(spearman) and spearman > 0)
    fit_eligible = bool(span_supported and min_bins_pass and positive_spearman_pass)
    reasons: list[str] = []
    if not span_supported:
        reasons.append("observed_log_current_span_below_window")
    if span_supported and not min_bins_pass:
        reasons.append(f"window_has_fewer_than_{min_bins}_bins")
    if span_supported and not positive_spearman_pass:
        reasons.append("window_spearman_not_positive")

    row: dict[str, Any] = {
        "curve_id": str(curve_ids[0]),
        "sample_record": int(records[0]),
        "window_decades_requested": float(window_decades),
        "window_definition": "shortest_continuous_low_potential_prefix_first_crossing",
        "no_interpolation": True,
        "no_extrapolation": True,
        "full_n_bins": int(len(ordered)),
        "full_log_current_span_dec": full_span,
        "span_supported": span_supported,
        "n_window_bins": n_window_bins,
        "minimum_bins_required": int(min_bins),
        "minimum_bins_pass": min_bins_pass,
        "window_log_current_span_dec": achieved_span,
        "window_span_overshoot_dec": (
            achieved_span - float(window_decades) if span_supported else np.nan
        ),
        "window_potential_span_V": potential_span,
        "potential_span_0p08V_flag_pass": bool(
            span_supported and potential_span >= potential_span_flag_V
        ),
        "spearman_logj_vs_potential": spearman,
        "positive_spearman_pass": positive_spearman_pass,
        "n_unique_log_current": n_unique_log_current,
        "n_initial_equal_log_current_bins": initial_equal_bins,
        "window_start_relative_potential_V": (
            float(selected["relative_potential_V"].iloc[0])
            if span_supported
            else np.nan
        ),
        "window_end_relative_potential_V": (
            float(selected["relative_potential_V"].iloc[-1])
            if span_supported
            else np.nan
        ),
        "window_start_potential_V": (
            float(selected["potential_V"].iloc[0]) if span_supported else np.nan
        ),
        "window_end_potential_V": (
            float(selected["potential_V"].iloc[-1]) if span_supported else np.nan
        ),
        "bayesian_fit_eligible": fit_eligible,
        "ineligibility_reason": ";".join(reasons),
        "analysis_role": ANALYSIS_ROLE,
        "window_model": WINDOW_MODEL,
        "window_likelihood": "student_t_nu4_equal_potential_bins",
    }
    return row, selected


def load_and_validate_inputs(
    run_dir: Path,
    baseline_inventory_path: Path,
    *,
    expected_curves: int = 97,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = run_dir.resolve()
    baseline_inventory_path = baseline_inventory_path.resolve()
    points_path = run_dir / "csv/binned_curve_points.csv"
    summary_path = run_dir / "csv/curve_fit_summary.csv"
    status_path = run_dir / "run_status.json"
    model_config_path = run_dir / "model_config.json"
    for path in (
        points_path,
        summary_path,
        status_path,
        model_config_path,
        baseline_inventory_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Sensitivity requires a complete deterministic source run")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    expected_hashes = model_config.get("deterministic_artifacts_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Source run lacks deterministic_artifacts_sha256")
    for relative_path, path in (
        ("csv/binned_curve_points.csv", points_path),
        ("csv/curve_fit_summary.csv", summary_path),
    ):
        if expected_hashes.get(relative_path) != sha256_file(path):
            raise ValueError(
                f"Source deterministic artifact differs from model_config.json: {path}"
            )

    points = pd.read_csv(points_path)
    summary = pd.read_csv(summary_path)
    baseline = pd.read_csv(baseline_inventory_path)
    if summary["curve_id"].astype(str).duplicated().any():
        raise ValueError("curve_fit_summary.csv has duplicate curve_id values")
    if baseline["curve_id"].astype(str).duplicated().any():
        raise ValueError("Bayesian baseline inventory has duplicate curve_id values")
    curve_sets = {
        "points": set(points["curve_id"].astype(str)),
        "summary": set(summary["curve_id"].astype(str)),
        "baseline": set(baseline["curve_id"].astype(str)),
    }
    if any(len(values) != expected_curves for values in curve_sets.values()):
        raise ValueError(
            f"Expected {expected_curves} curves in every input; found "
            + ", ".join(f"{key}={len(value)}" for key, value in curve_sets.items())
        )
    if not (curve_sets["points"] == curve_sets["summary"] == curve_sets["baseline"]):
        raise ValueError("Curve IDs differ across points, summary, and baseline inventory")
    if not summary["fit_status"].astype(str).eq("success").all():
        raise ValueError("All source curves must have successful deterministic fits")
    if not _as_bool(baseline["sampler_diagnostics_pass"]).all():
        raise ValueError("The formal Bayesian baseline contains failed sampler diagnostics")
    if not _as_bool(baseline["formal_confirmation_provenance_eligible"]).all():
        raise ValueError("The formal Bayesian baseline is not provenance eligible")

    binding = summary[["curve_id", "selected_model"]].merge(
        baseline[["curve_id", "model_name", "posterior_slope_variable"]],
        on="curve_id",
        validate="one_to_one",
    )
    if not binding["selected_model"].astype(str).eq(binding["model_name"].astype(str)).all():
        raise ValueError("Formal Bayesian model does not match deterministic selection")
    expected_variable = np.where(
        binding["selected_model"].astype(str).eq("two_regime_plateau"),
        "slope_low_mV_dec",
        "slope_mV_dec",
    )
    if not binding["posterior_slope_variable"].astype(str).eq(expected_variable).all():
        raise ValueError("Formal Bayesian baseline does not use the registered low branch")
    baseline_median = pd.to_numeric(
        baseline["posterior_slope_median_mV_dec"], errors="raise"
    )
    if not np.isfinite(baseline_median).all() or not (baseline_median > 0).all():
        raise ValueError("Formal Bayesian slope medians must be finite and positive")

    source = {
        "run_dir": str(run_dir),
        "binned_points_path": str(points_path),
        "binned_points_sha256": sha256_file(points_path),
        "curve_summary_path": str(summary_path),
        "curve_summary_sha256": sha256_file(summary_path),
        "run_status_path": str(status_path),
        "run_status_sha256": sha256_file(status_path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "deterministic_artifact_manifest_verified": True,
        "baseline_inventory_path": str(baseline_inventory_path),
        "baseline_inventory_sha256": sha256_file(baseline_inventory_path),
    }
    return points, summary, baseline, source


def build_window_manifest(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    windows: Iterable[float] = WINDOWS,
    min_bins: int = 12,
    potential_span_flag_V: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_lookup = summary.set_index("curve_id")
    if not summary_lookup.index.is_unique:
        raise ValueError("Summary curve_id values must be unique")
    selection_rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    for curve_id, curve_points in points.groupby("curve_id", sort=False):
        curve_id = str(curve_id)
        if curve_id not in summary_lookup.index:
            raise ValueError(f"Missing summary row for {curve_id}")
        source_row = summary_lookup.loc[curve_id]
        for window in windows:
            row, selected = select_low_potential_window(
                curve_points,
                float(window),
                min_bins=min_bins,
                potential_span_flag_V=potential_span_flag_V,
            )
            row.update(
                {
                    "formal_selected_model": str(source_row["selected_model"]),
                    "formal_transition_width_V": float(source_row["transition_width_V"]),
                    "source_qc_flags": str(source_row.get("qc_flags", "")),
                    "known_digitization_floor_finding": curve_id == "sr064",
                    "digitization_floor_note": (
                        "user-confirmed pixel-resolution/source-axis lower-limit artifact; "
                        "not an experimental background-current correction"
                        if curve_id == "sr064"
                        else ""
                    ),
                }
            )
            selection_rows.append(row)
            if len(selected):
                selected = selected.copy()
                selected.insert(2, "window_decades_requested", float(window))
                selected["bayesian_fit_eligible"] = bool(row["bayesian_fit_eligible"])
                point_frames.append(selected)
    selection = pd.DataFrame(selection_rows).sort_values(
        ["sample_record", "window_decades_requested"]
    ).reset_index(drop=True)
    selected_points = (
        pd.concat(point_frames, ignore_index=True)
        if point_frames
        else pd.DataFrame()
    )
    return selection, selected_points


def configure_runtime_cache(output_dir: Path) -> None:
    cache_root = Path(tempfile.gettempdir()) / "tafel-window-sensitivity-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("PYTENSOR_FLAGS", f"base_compiledir={cache_root / 'pytensor'}")
    output_dir.mkdir(parents=True, exist_ok=True)


def render_relative_potential_ppc(
    *,
    curve_id: str,
    window_decades: float,
    relative_potential_V: np.ndarray,
    observed_log10_current: np.ndarray,
    posterior_predictive_draws: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    """Render a sensitivity PPC without asserting an unverified reference scale."""

    import matplotlib.pyplot as plt

    x = np.asarray(relative_potential_V, dtype=float)
    y = np.asarray(observed_log10_current, dtype=float)
    draws = np.asarray(posterior_predictive_draws, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("PPC coordinates and observations must be equal-length vectors")
    if draws.ndim != 2 or draws.shape[1] != len(y):
        raise ValueError("PPC draws must have shape (draw, observation)")
    median = np.median(draws, axis=0)
    low = np.quantile(draws, 0.025, axis=0)
    high = np.quantile(draws, 0.975, axis=0)
    rmse = float(np.sqrt(np.mean(np.square(y - median))))
    coverage = float(np.mean((y >= low) & (y <= high)))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        }
    )
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.fill_between(
        x[order],
        low[order],
        high[order],
        color="#4472C4",
        alpha=0.22,
        linewidth=0,
        label="95% posterior predictive interval",
    )
    ax.plot(
        x[order],
        median[order],
        color="#4472C4",
        linewidth=2.0,
        label="Posterior predictive median",
    )
    ax.scatter(
        x,
        y,
        s=22,
        facecolor="white",
        edgecolor="#222222",
        linewidth=0.8,
        zorder=3,
        label="Observed bins",
    )
    ax.set_title(f"{curve_id}: {window_decades:.1f}-decade Bayesian window PPC")
    ax.set_xlabel("Relative potential from first fitted bin (V)")
    ax.set_ylabel(r"$\log_{10}$(current)")
    ax.text(
        0.01,
        0.99,
        f"Bayesian local-linear model; 95% point coverage = {coverage:.1%}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#333333",
        fontsize=9,
    )
    ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "posterior_predictive_rmse_log10": rmse,
        "posterior_predictive_95_coverage_fraction": coverage,
        "relative_potential_min_V": float(np.min(x)),
        "relative_potential_max_V": float(np.max(x)),
    }


def _finite_dataset_values(dataset: Any) -> np.ndarray:
    arrays = [
        np.asarray(dataset[name].values, dtype=float).reshape(-1)
        for name in dataset.data_vars
    ]
    values = np.concatenate(arrays) if arrays else np.array([], dtype=float)
    return values[np.isfinite(values)]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression=compression,
        lineterminator="\n",
    )
    temporary.replace(path)


def _fit_artifact_paths(fit_dir: Path) -> dict[str, Path]:
    return {
        "netcdf": fit_dir / "posterior.nc",
        "posterior_summary": fit_dir / "posterior_summary.csv",
        "slope_draws": fit_dir / "posterior_slope_draws.csv.gz",
        "diagnostics": fit_dir / "diagnostics.json",
        "ppc": fit_dir / "posterior_predictive.png",
        "result": fit_dir / "result.json",
    }


def _load_completed_fit(
    fit_dir: Path,
    *,
    curve_id: str,
    sample_record: int,
    window_decades: float,
    sampling_seed: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    window_points_sha256: str,
    provenance: dict[str, Any],
) -> dict[str, Any] | None:
    paths = _fit_artifact_paths(fit_dir)
    existing = [key for key, path in paths.items() if path.exists()]
    if not existing:
        return None
    if len(existing) != len(paths):
        raise RuntimeError(
            f"Partial Bayesian artifact set in {fit_dir}; refusing to infer completion: "
            + ", ".join(existing)
        )
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    expected_identity = {
        "curve_id": str(curve_id),
        "sample_record": int(sample_record),
        "window_decades_requested": float(window_decades),
        "sampling_seed": int(sampling_seed),
        "draws_per_chain": int(draws),
        "tune_per_chain": int(tune),
        "chains": int(chains),
        "target_accept": float(target_accept),
        "window_points_sha256": str(window_points_sha256),
        "run_configuration_sha256": str(provenance["run_configuration_sha256"]),
        "source_binned_points_sha256": str(
            provenance["source_binned_points_sha256"]
        ),
    }
    mismatches = {
        key: (result.get(key), expected)
        for key, expected in expected_identity.items()
        if result.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Completed-fit identity mismatch in {fit_dir}: {mismatches}"
        )
    hashes = result.get("artifact_sha256", {})
    for key in ("netcdf", "posterior_summary", "slope_draws", "diagnostics", "ppc"):
        if hashes.get(key) != sha256_file(paths[key]):
            raise RuntimeError(f"Artifact hash mismatch for {paths[key]}")
    return result


def _fit_bayesian_local_linear_into_directory(
    window_points: pd.DataFrame,
    selection_row: pd.Series,
    *,
    fit_dir: Path,
    published_fit_dir: Path,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    progressbar: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Fit one Bayesian Student-t local-linear sensitivity model."""

    configure_runtime_cache(fit_dir)
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    if draws < 1 or tune < 0 or chains < 2:
        raise ValueError("draws>=1, tune>=0, and chains>=2 are required")
    if not 0.5 < target_accept < 1.0:
        raise ValueError("target_accept must be between 0.5 and 1.0")
    fit_dir.mkdir(parents=True, exist_ok=True)
    paths = _fit_artifact_paths(fit_dir)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to overwrite Bayesian artifacts in {fit_dir}")

    ordered = window_points.sort_values(
        ["relative_potential_V", "potential_bin_index"], kind="mergesort"
    )
    x = ordered["relative_potential_V"].to_numpy(dtype=float)
    y = ordered["log10_current"].to_numpy(dtype=float)
    point_hash = dataframe_sha256(ordered.reset_index(drop=True))
    variable_names = [
        "a",
        "log_m",
        "m_dec_per_V",
        WINDOW_SLOPE_VARIABLE,
        "log_sigma",
        "sigma_log10_current",
    ]

    observation_ids = ordered["potential_bin_index"].to_numpy(dtype=int)
    with pm.Model(coords={"observation": observation_ids}):
        a = pm.Uniform("a", lower=float(np.min(y) - 6.0), upper=float(np.max(y) + 2.0))
        log_m = pm.Uniform("log_m", lower=float(np.log(1.0)), upper=float(np.log(100.0)))
        m = pm.Deterministic("m_dec_per_V", pt.exp(log_m))
        pm.Deterministic(WINDOW_SLOPE_VARIABLE, 1000.0 / m)
        x_data = pm.Data("relative_potential_V", x, dims="observation")
        mean = a + m * x_data
        log_sigma = pm.Uniform(
            "log_sigma", lower=float(np.log(0.005)), upper=float(np.log(0.5))
        )
        sigma = pm.Deterministic("sigma_log10_current", pt.exp(log_sigma))
        pm.StudentT(
            "observed_log10_current",
            nu=4.0,
            mu=mean,
            sigma=sigma,
            observed=y,
            dims="observation",
        )
        inference_data = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            progressbar=progressbar,
        )
        posterior_predictive = pm.sample_posterior_predictive(
            inference_data,
            var_names=["observed_log10_current"],
            random_seed=seed + 1,
            progressbar=progressbar,
        )
        inference_data.extend(posterior_predictive)
    inference_data.attrs.update(
        {
            "analysis_role": ANALYSIS_ROLE,
            "curve_id": str(selection_row["curve_id"]),
            "sample_record": int(selection_row["sample_record"]),
            "window_decades_requested": float(
                selection_row["window_decades_requested"]
            ),
            "window_points_sha256": point_hash,
            "run_configuration_sha256": str(
                provenance["run_configuration_sha256"]
            ),
            "source_binned_points_sha256": str(
                provenance["source_binned_points_sha256"]
            ),
            "sampling_seed": int(seed),
            "model_name": WINDOW_MODEL,
            "likelihood": "student_t_nu4_equal_potential_bins",
        }
    )

    temporary_netcdf = paths["netcdf"].with_suffix(".nc.tmp")
    inference_data.to_netcdf(temporary_netcdf)
    temporary_netcdf.replace(paths["netcdf"])
    netcdf_sha256 = sha256_file(paths["netcdf"])
    posterior_summary = az.summary(
        inference_data, var_names=variable_names, kind="all", hdi_prob=0.95
    )
    _atomic_csv(posterior_summary.reset_index(names="variable"), paths["posterior_summary"])

    slope_array = np.asarray(
        inference_data.posterior[WINDOW_SLOPE_VARIABLE].values, dtype=float
    )
    chains_observed, draws_observed = slope_array.shape[:2]
    slope_draws = pd.DataFrame(
        {
            "curve_id": str(selection_row["curve_id"]),
            "sample_record": int(selection_row["sample_record"]),
            "window_decades_requested": float(selection_row["window_decades_requested"]),
            "chain": np.repeat(np.arange(chains_observed), draws_observed),
            "draw": np.tile(np.arange(draws_observed), chains_observed),
            "posterior_slope_variable": WINDOW_SLOPE_VARIABLE,
            "slope_mV_dec": slope_array.reshape(-1),
            "source_netcdf_sha256": netcdf_sha256,
            "window_points_sha256": point_hash,
        }
    )
    _atomic_csv(slope_draws, paths["slope_draws"], compression="gzip")

    r_hat_values = _finite_dataset_values(
        az.rhat(inference_data, var_names=variable_names, method="rank")
    )
    ess_bulk_values = _finite_dataset_values(
        az.ess(inference_data, var_names=variable_names, method="bulk")
    )
    divergences = int(inference_data.sample_stats["diverging"].sum().item())
    slopes = slope_array.reshape(-1)
    slope_hdi = np.asarray(az.hdi(slopes, hdi_prob=0.95), dtype=float)
    ppc_array = np.asarray(
        inference_data.posterior_predictive["observed_log10_current"].values,
        dtype=float,
    ).reshape(-1, len(y))
    ppc_median = np.median(ppc_array, axis=0)
    ppc_low = np.quantile(ppc_array, 0.025, axis=0)
    ppc_high = np.quantile(ppc_array, 0.975, axis=0)
    ppc_rmse = float(np.sqrt(np.mean(np.square(y - ppc_median))))
    ppc_coverage = float(np.mean((y >= ppc_low) & (y <= ppc_high)))
    sampler_pass = bool(
        len(r_hat_values)
        and float(np.max(r_hat_values)) <= 1.01
        and len(ess_bulk_values)
        and float(np.min(ess_bulk_values)) >= 400
        and divergences == 0
    )
    prior_boundary_fraction = float(np.mean((slopes <= 10.5) | (slopes >= 990.0)))

    ppc_metrics = render_relative_potential_ppc(
        curve_id=str(selection_row["curve_id"]),
        window_decades=float(selection_row["window_decades_requested"]),
        relative_potential_V=x,
        observed_log10_current=y,
        posterior_predictive_draws=ppc_array,
        output_path=paths["ppc"],
    )
    diagnostics = {
        "created_at_utc": utc_now(),
        "analysis_role": ANALYSIS_ROLE,
        "formal_confirmation_provenance_eligible": False,
        "formal_confirmation_ready_for_ppc_review": False,
        "confirmation_scope": "sensitivity_only_not_formal_confirmation",
        "curve_id": str(selection_row["curve_id"]),
        "sample_record": int(selection_row["sample_record"]),
        "window_decades_requested": float(selection_row["window_decades_requested"]),
        "window_log_current_span_dec": float(selection_row["window_log_current_span_dec"]),
        "n_window_bins": int(selection_row["n_window_bins"]),
        "model_name": WINDOW_MODEL,
        "slope_estimand": "Bayesian window-local linear Tafel slope of observed bins",
        "posterior_slope_variable": WINDOW_SLOPE_VARIABLE,
        "window_points_sha256": point_hash,
        "run_configuration_sha256": provenance["run_configuration_sha256"],
        "source_binned_points_sha256": provenance["source_binned_points_sha256"],
        "likelihood": "student_t_nu4_equal_potential_bins",
        "draws_per_chain": int(draws),
        "tune_per_chain": int(tune),
        "chains": int(chains),
        "target_accept": float(target_accept),
        "sampling_seed": int(seed),
        "posterior_predictive_seed": int(seed + 1),
        "max_r_hat": float(np.max(r_hat_values)) if len(r_hat_values) else None,
        "min_ess_bulk": float(np.min(ess_bulk_values)) if len(ess_bulk_values) else None,
        "divergences": divergences,
        "sampler_diagnostics_pass": sampler_pass,
        "posterior_slope_prior_boundary_fraction": prior_boundary_fraction,
        "posterior_slope_prior_boundary_warning": prior_boundary_fraction > 0.05,
        "posterior_predictive_rmse_log10": ppc_rmse,
        "posterior_predictive_95_coverage_fraction": ppc_coverage,
        "ppc_coverage_is_diagnostic_not_gate": True,
        "ppc_x_axis": "relative_potential_V",
        "ppc_does_not_assert_potential_reference": True,
        "note": (
            "This sensitivity fit is Bayesian, not OLS. PPC coverage is descriptive; "
            "sampler pass requires max rank R-hat <= 1.01, min bulk ESS >= 400, "
            "and zero divergences."
        ),
        **ppc_metrics,
    }
    _atomic_json(paths["diagnostics"], diagnostics)
    artifact_hashes = {
        key: sha256_file(paths[key])
        for key in ("netcdf", "posterior_summary", "slope_draws", "diagnostics", "ppc")
    }
    result: dict[str, Any] = {
        "fit_status": "bayesian_sampling_complete",
        "curve_id": str(selection_row["curve_id"]),
        "sample_record": int(selection_row["sample_record"]),
        "window_decades_requested": float(selection_row["window_decades_requested"]),
        "bayesian_method": METHOD,
        "window_model": WINDOW_MODEL,
        "posterior_slope_variable": WINDOW_SLOPE_VARIABLE,
        "window_points_sha256": point_hash,
        "run_configuration_sha256": provenance["run_configuration_sha256"],
        "source_binned_points_sha256": provenance["source_binned_points_sha256"],
        "posterior_slope_median_mV_dec": float(np.median(slopes)),
        "posterior_slope_mean_mV_dec": float(np.mean(slopes)),
        "posterior_slope_hdi_low_mV_dec": float(slope_hdi[0]),
        "posterior_slope_hdi_high_mV_dec": float(slope_hdi[1]),
        "posterior_slope_total_draws": int(len(slopes)),
        "draws_per_chain": int(draws),
        "tune_per_chain": int(tune),
        "chains": int(chains),
        "target_accept": float(target_accept),
        "max_r_hat": diagnostics["max_r_hat"],
        "min_ess_bulk": diagnostics["min_ess_bulk"],
        "divergences": divergences,
        "sampler_diagnostics_pass": sampler_pass,
        "posterior_slope_prior_boundary_fraction": prior_boundary_fraction,
        "posterior_slope_prior_boundary_warning": prior_boundary_fraction > 0.05,
        "posterior_predictive_rmse_log10": ppc_rmse,
        "posterior_predictive_95_coverage_fraction": ppc_coverage,
        "sampling_seed": int(seed),
        "netcdf_path": str(_fit_artifact_paths(published_fit_dir)["netcdf"]),
        "posterior_summary_path": str(
            _fit_artifact_paths(published_fit_dir)["posterior_summary"]
        ),
        "slope_draws_path": str(_fit_artifact_paths(published_fit_dir)["slope_draws"]),
        "diagnostics_path": str(_fit_artifact_paths(published_fit_dir)["diagnostics"]),
        "ppc_plot_path": str(_fit_artifact_paths(published_fit_dir)["ppc"]),
        "artifact_sha256": artifact_hashes,
        "analysis_role": ANALYSIS_ROLE,
    }
    _atomic_json(paths["result"], result)
    return result


def fit_bayesian_local_linear(
    window_points: pd.DataFrame,
    selection_row: pd.Series,
    *,
    fit_dir: Path,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    progressbar: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Fit into a staging directory and atomically publish a complete artifact set."""

    final_dir = fit_dir.resolve()
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Bayesian fit directory: {final_dir}")
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.inprogress-", dir=final_dir.parent)
    )
    try:
        result = _fit_bayesian_local_linear_into_directory(
            window_points,
            selection_row,
            fit_dir=staging_dir,
            published_fit_dir=final_dir,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            seed=seed,
            progressbar=progressbar,
            provenance=provenance,
        )
        staging_dir.replace(final_dir)
        return result
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _symmetric_ratio(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right) or left <= 0 or right <= 0:
        return np.nan
    return float(max(left, right) / min(left, right))


def build_stability_table(
    selection: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    requested_records: set[int],
    ratio_threshold: float = STABILITY_RATIO_THRESHOLD,
) -> pd.DataFrame:
    baseline_columns = [
        "curve_id",
        "model_name",
        "posterior_slope_variable",
        "posterior_slope_median_mV_dec",
        "posterior_slope_draw_hdi_low_mV_dec",
        "posterior_slope_draw_hdi_high_mV_dec",
        "sampler_diagnostics_pass",
    ]
    base = baseline[baseline_columns].copy().rename(
        columns={
            "model_name": "full_formal_model",
            "posterior_slope_variable": "full_formal_slope_variable",
            "posterior_slope_median_mV_dec": "full_formal_slope_median_mV_dec",
            "posterior_slope_draw_hdi_low_mV_dec": "full_formal_slope_hdi_low_mV_dec",
            "posterior_slope_draw_hdi_high_mV_dec": "full_formal_slope_hdi_high_mV_dec",
            "sampler_diagnostics_pass": "full_formal_sampler_diagnostics_pass",
        }
    )
    records = summary[["curve_id", "sample_record"]].merge(
        base, on="curve_id", validate="one_to_one"
    )
    records["fit_requested"] = records["sample_record"].astype(int).isin(requested_records)

    result_lookup: dict[tuple[str, float], pd.Series] = {}
    if len(results):
        for _, row in results.iterrows():
            result_lookup[(str(row["curve_id"]), float(row["window_decades_requested"]))] = row
    selection_lookup = {
        (str(row["curve_id"]), float(row["window_decades_requested"])): row
        for _, row in selection.iterrows()
    }
    output: list[dict[str, Any]] = []
    for _, record in records.sort_values("sample_record").iterrows():
        row = record.to_dict()
        curve_id = str(record["curve_id"])
        window_values: dict[float, dict[str, Any]] = {}
        for window in WINDOWS:
            token = window_token(window)
            selected = selection_lookup[(curve_id, window)]
            fitted = result_lookup.get((curve_id, window))
            row[f"window_{token}_span_supported"] = bool(selected["span_supported"])
            row[f"window_{token}_fit_eligible"] = bool(selected["bayesian_fit_eligible"])
            row[f"window_{token}_n_bins"] = int(selected["n_window_bins"])
            row[f"window_{token}_achieved_span_dec"] = selected[
                "window_log_current_span_dec"
            ]
            row[f"window_{token}_ineligibility_reason"] = selected["ineligibility_reason"]
            if fitted is None:
                row[f"window_{token}_fit_status"] = (
                    "not_requested"
                    if not bool(record["fit_requested"])
                    else "window_ineligible"
                    if not bool(selected["bayesian_fit_eligible"])
                    else "missing_requested_fit"
                )
                for field in (
                    "slope_median_mV_dec",
                    "slope_hdi_low_mV_dec",
                    "slope_hdi_high_mV_dec",
                    "max_r_hat",
                    "min_ess_bulk",
                    "divergences",
                    "sampler_diagnostics_pass",
                    "prior_boundary_warning",
                ):
                    row[f"window_{token}_{field}"] = np.nan
                window_values[window] = {"success": False}
            else:
                success = str(fitted.get("fit_status")) in SAMPLING_COMPLETE_STATUSES
                row[f"window_{token}_fit_status"] = fitted.get("fit_status")
                mapping = {
                    "slope_median_mV_dec": "posterior_slope_median_mV_dec",
                    "slope_hdi_low_mV_dec": "posterior_slope_hdi_low_mV_dec",
                    "slope_hdi_high_mV_dec": "posterior_slope_hdi_high_mV_dec",
                    "max_r_hat": "max_r_hat",
                    "min_ess_bulk": "min_ess_bulk",
                    "divergences": "divergences",
                    "sampler_diagnostics_pass": "sampler_diagnostics_pass",
                    "prior_boundary_warning": (
                        "posterior_slope_prior_boundary_warning"
                    ),
                }
                for output_name, input_name in mapping.items():
                    row[f"window_{token}_{output_name}"] = fitted.get(input_name, np.nan)
                window_values[window] = {
                    "success": success,
                    "median": float(fitted["posterior_slope_median_mV_dec"])
                    if success
                    else np.nan,
                    "diagnostics": bool(fitted.get("sampler_diagnostics_pass", False)),
                    "prior_boundary_warning": bool(
                        fitted.get("posterior_slope_prior_boundary_warning", False)
                    ),
                    "hdi_low": float(fitted["posterior_slope_hdi_low_mV_dec"])
                    if success
                    else np.nan,
                    "hdi_high": float(fitted["posterior_slope_hdi_high_mV_dec"])
                    if success
                    else np.nan,
                }

            window_median = row.get(f"window_{token}_slope_median_mV_dec", np.nan)
            full_median = float(row["full_formal_slope_median_mV_dec"])
            row[f"window_{token}_minus_full_reference_mV_dec"] = (
                float(window_median) - full_median
                if np.isfinite(window_median)
                else np.nan
            )
            row[f"window_{token}_vs_full_reference_fold_ratio"] = _symmetric_ratio(
                float(window_median), full_median
            )

        left = window_values[1.5]
        right = window_values[2.0]
        if not bool(record["fit_requested"]):
            evaluable = False
            reason = "not_requested"
        elif not bool(selection_lookup[(curve_id, 1.5)]["bayesian_fit_eligible"]):
            evaluable = False
            reason = "window_1p5_ineligible"
        elif not bool(selection_lookup[(curve_id, 2.0)]["bayesian_fit_eligible"]):
            evaluable = False
            reason = "window_2p0_ineligible"
        elif not left.get("success") or not right.get("success"):
            evaluable = False
            reason = "bayesian_sampling_failed_or_missing"
        elif not left.get("diagnostics") or not right.get("diagnostics"):
            evaluable = False
            reason = "sampler_diagnostics_failed"
        elif left.get("prior_boundary_warning") or right.get("prior_boundary_warning"):
            evaluable = False
            reason = "posterior_slope_prior_boundary_warning"
        else:
            evaluable = True
            reason = "evaluable"
        row["stability_evaluable"] = evaluable
        row["stability_status_reason"] = reason
        if left.get("success") and right.get("success"):
            left_median = float(left["median"])
            right_median = float(right["median"])
            fold_ratio = _symmetric_ratio(left_median, right_median)
            row["slope_delta_2p0_minus_1p5_mV_dec"] = right_median - left_median
            row["slope_relative_change_2p0_vs_1p5"] = (
                (right_median - left_median) / left_median
            )
            row["slope_abs_log_ratio_2p0_vs_1p5"] = abs(np.log(right_median / left_median))
            row["slope_fold_ratio_1p5_vs_2p0"] = fold_ratio
            row["posterior_hdi_overlap_1p5_vs_2p0"] = bool(
                max(float(left["hdi_low"]), float(right["hdi_low"]))
                <= min(float(left["hdi_high"]), float(right["hdi_high"]))
            )
            row["window_1p5_relative_hdi_width"] = (
                float(left["hdi_high"]) - float(left["hdi_low"])
            ) / left_median
            row["window_2p0_relative_hdi_width"] = (
                float(right["hdi_high"]) - float(right["hdi_low"])
            ) / right_median
        else:
            fold_ratio = np.nan
            row["slope_delta_2p0_minus_1p5_mV_dec"] = np.nan
            row["slope_relative_change_2p0_vs_1p5"] = np.nan
            row["slope_abs_log_ratio_2p0_vs_1p5"] = np.nan
            row["slope_fold_ratio_1p5_vs_2p0"] = np.nan
            row["posterior_hdi_overlap_1p5_vs_2p0"] = pd.NA
            row["window_1p5_relative_hdi_width"] = np.nan
            row["window_2p0_relative_hdi_width"] = np.nan
        row["stability_ratio_threshold"] = float(ratio_threshold)
        row["window_local_posterior_median_ratio_within_threshold"] = (
            bool(fold_ratio <= ratio_threshold) if evaluable else pd.NA
        )
        formal_model_is_linear = str(row["full_formal_model"]) == "linear"
        if not formal_model_is_linear:
            formal_evaluable = False
            formal_reason = "cross_estimand_formal_model_not_linear"
        elif not evaluable:
            formal_evaluable = False
            formal_reason = f"window_comparison_{reason}"
        else:
            formal_evaluable = True
            formal_reason = "evaluable_same_linear_model"
        row["formal_target_window_stability_evaluable"] = formal_evaluable
        row["formal_target_window_stability_reason"] = formal_reason
        if formal_evaluable:
            ratios_to_full = [
                float(row["window_1p5_vs_full_reference_fold_ratio"]),
                float(row["window_2p0_vs_full_reference_fold_ratio"]),
                float(fold_ratio),
            ]
            row["formal_linear_target_stable_within_ratio_threshold"] = bool(
                max(ratios_to_full) <= ratio_threshold
            )
        else:
            row["formal_linear_target_stable_within_ratio_threshold"] = pd.NA
        row["window_comparison_estimand"] = (
            "Bayesian window-local linear Tafel slope of observed bins"
        )
        row["full_reference_comparison_role"] = (
            "context_only_cross_estimand_unless_full_formal_model_is_linear"
        )
        row["analysis_role"] = ANALYSIS_ROLE
        output.append(row)
    return pd.DataFrame(output)


def build_cohort_summary(stability: pd.DataFrame) -> pd.DataFrame:
    frames: list[tuple[str, pd.DataFrame]] = [("all", stability)]
    for model_name, subset in stability.groupby("full_formal_model", sort=True):
        frames.append((f"formal_model={model_name}", subset))
    rows: list[dict[str, Any]] = []
    for stratum, subset in frames:
        requested = subset.loc[_as_bool(subset["fit_requested"])]
        evaluable = requested.loc[_as_bool(requested["stability_evaluable"])]
        stable = _as_bool(
            evaluable["window_local_posterior_median_ratio_within_threshold"]
        )
        formal_evaluable = requested.loc[
            _as_bool(requested["formal_target_window_stability_evaluable"])
        ]
        formal_stable = _as_bool(
            formal_evaluable["formal_linear_target_stable_within_ratio_threshold"]
        )
        ratios = pd.to_numeric(evaluable["slope_fold_ratio_1p5_vs_2p0"], errors="coerce")
        rows.append(
            {
                "stratum": stratum,
                "n_curves": int(len(subset)),
                "n_fit_requested": int(len(requested)),
                "n_1p5_span_supported": int(_as_bool(requested["window_1p5_span_supported"]).sum()),
                "n_2p0_span_supported": int(_as_bool(requested["window_2p0_span_supported"]).sum()),
                "n_1p5_fit_eligible": int(_as_bool(requested["window_1p5_fit_eligible"]).sum()),
                "n_2p0_fit_eligible": int(_as_bool(requested["window_2p0_fit_eligible"]).sum()),
                "n_window_local_stability_evaluable": int(len(evaluable)),
                "n_window_local_median_ratio_stable": int(stable.sum()),
                "window_local_median_ratio_stable_fraction_among_evaluable": (
                    float(stable.mean()) if len(evaluable) else np.nan
                ),
                "n_formal_target_stability_evaluable": int(len(formal_evaluable)),
                "n_formal_linear_target_stable": int(formal_stable.sum()),
                "formal_linear_target_stable_fraction_among_evaluable": (
                    float(formal_stable.mean()) if len(formal_evaluable) else np.nan
                ),
                "median_symmetric_fold_ratio": (
                    float(ratios.median()) if len(ratios) else np.nan
                ),
                "q90_symmetric_fold_ratio": (
                    float(ratios.quantile(0.90)) if len(ratios) else np.nan
                ),
                "stability_ratio_threshold": STABILITY_RATIO_THRESHOLD,
                "unsupported_or_failed_not_counted_as_unstable": True,
                "analysis_role": ANALYSIS_ROLE,
            }
        )
    return pd.DataFrame(rows)


def _merge_results_with_manifest(
    selection: pd.DataFrame,
    completed: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    requested_records: set[int],
) -> pd.DataFrame:
    fit_rows: list[dict[str, Any]] = []
    for source_row in completed + failures:
        row = dict(source_row)
        if isinstance(row.get("artifact_sha256"), dict):
            row["artifact_sha256_json"] = json.dumps(
                row.pop("artifact_sha256"), sort_keys=True
            )
        fit_rows.append(row)
    fitted = pd.DataFrame(fit_rows)
    manifest = selection.copy()
    manifest["fit_requested"] = manifest["sample_record"].astype(int).isin(requested_records)
    if len(fitted):
        manifest = manifest.merge(
            fitted,
            on=["curve_id", "sample_record", "window_decades_requested"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_fit"),
        )
    if "fit_status" not in manifest:
        manifest["fit_status"] = np.nan
    missing = manifest["fit_status"].isna()
    manifest.loc[missing & ~manifest["fit_requested"], "fit_status"] = "not_requested"
    manifest.loc[
        missing & manifest["fit_requested"] & ~_as_bool(manifest["bayesian_fit_eligible"]),
        "fit_status",
    ] = "window_ineligible"
    manifest.loc[
        missing & manifest["fit_requested"] & _as_bool(manifest["bayesian_fit_eligible"]),
        "fit_status",
    ] = "missing_requested_fit"
    return manifest.sort_values(["sample_record", "window_decades_requested"]).reset_index(
        drop=True
    )


def _write_output_readme(
    output_dir: Path,
    selection: pd.DataFrame,
    result_manifest: pd.DataFrame,
    stability: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    smoke: bool,
    requested_count: int,
) -> None:
    counts = selection.groupby("window_decades_requested").agg(
        span_supported=("span_supported", "sum"),
        fit_eligible=("bayesian_fit_eligible", "sum"),
    )
    lines = [
        "# Bayesian low-potential window sensitivity",
        "",
        f"Analysis role: `{ANALYSIS_ROLE}`.",
        "",
        "Both 1.5- and 2.0-decade slopes are newly sampled Bayesian estimates,",
        "not OLS fits. Each fit uses the same window-local linear model with a",
        "Student-t likelihood (nu=4) and unknown residual scale on the approved",
        "equal-potential binned observations.",
        "",
        "The window is the shortest continuous prefix from the low-potential end",
        "whose observed log-current range first reaches the requested number of",
        "decades. The crossing bin is retained; no interpolation or extrapolation",
        "is used. At least 12 bins and positive Spearman association are required.",
        "The 0.08 V span rule is reported as a flag, not used as a sensitivity gate.",
        "",
        f"Run mode: `{'smoke' if smoke else 'full'}`; requested records: {requested_count}.",
        "",
    ]
    for window, row in counts.iterrows():
        lines.append(
            f"- {float(window):.1f} dec: {int(row['span_supported'])}/97 span-supported; "
            f"{int(row['fit_eligible'])}/97 fit-eligible."
        )
    if smoke:
        lines.extend(
            [
                "",
                "Smoke chains are intentionally too short for scientific inference;",
                "only structural artifact completion is evaluated in smoke mode.",
            ]
        )
    else:
        overall = cohort.loc[cohort["stratum"].eq("all")].iloc[0]
        linear = cohort.loc[cohort["stratum"].eq("formal_model=linear")].iloc[0]
        complete = result_manifest.loc[
            result_manifest["fit_status"].isin(SAMPLING_COMPLETE_STATUSES)
        ]
        diagnostics_pass = int(
            _as_bool(complete["sampler_diagnostics_pass"]).sum()
        )
        boundary_warnings = int(
            _as_bool(complete["posterior_slope_prior_boundary_warning"]).sum()
        )
        lines.extend(
            [
                "",
                "## Results",
                "",
                f"- Bayesian sampling completed for {len(complete)} eligible windows; "
                f"{diagnostics_pass}/{len(complete)} passed the sampler gate and "
                f"{boundary_warnings} had a slope-prior boundary warning.",
                "- Window-local comparison: "
                f"{int(overall['n_window_local_median_ratio_stable'])}/"
                f"{int(overall['n_window_local_stability_evaluable'])} "
                "diagnostics-pass pairs had posterior-median fold ratio <= "
                f"{STABILITY_RATIO_THRESHOLD:.2f} "
                f"({float(overall['window_local_median_ratio_stable_fraction_among_evaluable']):.1%}).",
                "- Formal-linear stratum: "
                f"{int(linear['n_window_local_median_ratio_stable'])}/"
                f"{int(linear['n_window_local_stability_evaluable'])} "
                "had a stable 1.5-vs-2.0 window-local median; only "
                f"{int(linear['n_formal_linear_target_stable'])}/"
                f"{int(linear['n_formal_target_stability_evaluable'])} "
                "also kept both windows within the same ratio threshold of the "
                "formal full-range slope.",
                "- Plateau and two-regime strata remain cross-estimand diagnostics;",
                "  their window-local ratios do not establish stability or instability",
                "  of the registered latent low-potential branch.",
            ]
        )
    lines.extend(
        [
            "",
            "`slope_stability.csv` compares the two posterior medians using the",
            f"descriptive symmetric ratio threshold {STABILITY_RATIO_THRESHOLD:.2f}.",
            "A missing window or failed sampler diagnostics is `indeterminate`, not",
            "unstable. Independent NUTS runs are not subtracted draw-by-draw.",
            "",
            "The formal full-range selected-model posterior is copied in as context.",
            "It is a different estimand whenever the formal model is not `linear` and",
            "formal-target window stability is therefore reported only for the",
            "formal-linear stratum. Other strata remain window-local diagnostics.",
            "This sensitivity does not redefine the primary target, canonical snapshot,",
            "PPC review, or ML inputs.",
            "Sensitivity PPCs use relative potential because the source potential",
            "reference is not verified; they do not claim an OER-overpotential axis.",
            "",
            "Files:",
            "",
            "- `window_selection.csv`: support, discrete overshoot, QC, and eligibility.",
            "- `selected_window_points.csv.gz`: exact observed bins entering each window.",
            "- `window_posterior_results.csv`: fit status, posterior summaries, diagnostics.",
            "- `slope_stability.csv`: per-record 1.5-vs-2.0 comparison and full reference.",
            "- `cohort_summary.csv`: overall and formal-model-stratified summaries.",
            "- `bayes/`: NetCDF, posterior draws/summary, diagnostics, and PPC per fit.",
            "",
            "PPC coverage is diagnostic and is never an automatic approval criterion.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--baseline-inventory", type=Path, default=DEFAULT_BASELINE_INVENTORY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--window-decades", type=float, nargs="+", default=list(WINDOWS)
    )
    parser.add_argument("--min-bins", type=int, default=12)
    parser.add_argument("--potential-span-flag-V", type=float, default=0.08)
    parser.add_argument("--expected-curves", type=int, default=97)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--sample-records", type=int, nargs="+")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fit Records 1, 2, and 4 (one per formal model stratum) with "
            "50 tune + 50 draws and two chains."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    windows = tuple(sorted(float(value) for value in args.window_decades))
    if windows != WINDOWS:
        raise ValueError("This registered sensitivity requires exactly 1.5 and 2.0 decades")
    if args.min_bins < 3:
        raise ValueError("--min-bins must be at least 3")
    if args.smoke:
        draws, tune, chains = 50, 50, 2
        sample_records = set(args.sample_records or [1, 2, 4])
    else:
        draws, tune, chains = args.draws, args.tune, args.chains
        sample_records = set(args.sample_records or range(1, args.expected_curves + 1))
    if any(record < 1 or record > args.expected_curves for record in sample_records):
        raise ValueError("--sample-records contains a record outside the active 1..97 set")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --resume or a new directory."
        )
    configure_runtime_cache(output_dir)

    points, summary, baseline, source = load_and_validate_inputs(
        args.run_dir, args.baseline_inventory, expected_curves=args.expected_curves
    )
    selection, selected_points = build_window_manifest(
        points,
        summary,
        windows=windows,
        min_bins=args.min_bins,
        potential_span_flag_V=args.potential_span_flag_V,
    )
    configuration = {
        "analysis_role": ANALYSIS_ROLE,
        "method": METHOD,
        "window_model": WINDOW_MODEL,
        "window_slope_variable": WINDOW_SLOPE_VARIABLE,
        "window_decades": list(windows),
        "window_definition": "shortest_continuous_low_potential_prefix_first_crossing",
        "minimum_bins": int(args.min_bins),
        "positive_spearman_required": True,
        "potential_span_flag_V": float(args.potential_span_flag_V),
        "potential_span_is_gate": False,
        "draws_per_chain": int(draws),
        "tune_per_chain": int(tune),
        "chains": int(chains),
        "target_accept": float(args.target_accept),
        "base_seed": int(args.seed),
        "sample_records": sorted(sample_records),
        "smoke": bool(args.smoke),
        "expected_curves": int(args.expected_curves),
        "stability_ratio_threshold": STABILITY_RATIO_THRESHOLD,
        "window_selection_sha256": dataframe_sha256(selection),
        "selected_window_points_sha256": dataframe_sha256(selected_points),
        "source": source,
    }
    config_path = output_dir / "run_configuration.json"
    selection_path = output_dir / "window_selection.csv"
    selected_points_path = output_dir / "selected_window_points.csv.gz"
    if args.resume:
        for path in (config_path, selection_path, selected_points_path):
            if not path.is_file():
                raise FileNotFoundError(f"--resume requires {path}")
        configuration["window_selection_file_sha256"] = sha256_file(selection_path)
        configuration["selected_window_points_file_sha256"] = sha256_file(
            selected_points_path
        )
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise ValueError(
                "Resume configuration or saved window artifacts differ from the existing run"
            )
    else:
        _atomic_csv(selection, selection_path)
        _atomic_csv(
            selected_points,
            selected_points_path,
            compression="gzip",
        )
        configuration["window_selection_file_sha256"] = sha256_file(selection_path)
        configuration["selected_window_points_file_sha256"] = sha256_file(
            selected_points_path
        )
        _atomic_json(config_path, configuration)
    provenance = {
        "run_configuration_sha256": sha256_file(config_path),
        "source_binned_points_sha256": source["binned_points_sha256"],
    }

    point_lookup = {
        (str(curve_id), float(window)): subset.copy()
        for (curve_id, window), subset in selected_points.groupby(
            ["curve_id", "window_decades_requested"], sort=False
        )
    }
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    eligible = selection.loc[
        selection["sample_record"].astype(int).isin(sample_records)
        & _as_bool(selection["bayesian_fit_eligible"])
    ].sort_values(["sample_record", "window_decades_requested"])
    for _, selected in eligible.iterrows():
        curve_id = str(selected["curve_id"])
        record = int(selected["sample_record"])
        window = float(selected["window_decades_requested"])
        token = window_token(window)
        fit_dir = (
            output_dir
            / "bayes"
            / f"record_{record:03d}__curve_{curve_id}__window_{token}dec"
        )
        seed = int(args.seed + record * 10_000 + round(window * 100))
        window_frame = point_lookup[(curve_id, window)].sort_values(
            ["relative_potential_V", "potential_bin_index"], kind="mergesort"
        )
        window_point_hash = dataframe_sha256(window_frame.reset_index(drop=True))
        previous = (
            _load_completed_fit(
                fit_dir,
                curve_id=curve_id,
                sample_record=record,
                window_decades=window,
                sampling_seed=seed,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=args.target_accept,
                window_points_sha256=window_point_hash,
                provenance=provenance,
            )
            if args.resume
            else None
        )
        if previous is not None:
            completed.append(previous)
            print(f"resume: {curve_id} {window:.1f} dec already complete", flush=True)
            continue
        print(f"fit: {curve_id} {window:.1f} dec (seed {seed})", flush=True)
        try:
            result = fit_bayesian_local_linear(
                point_lookup[(curve_id, window)],
                selected,
                fit_dir=fit_dir,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=args.target_accept,
                seed=seed,
                progressbar=not args.no_progress,
                provenance=provenance,
            )
            completed.append(result)
        except Exception as exc:  # batch must preserve other completed fits
            failure = {
                "fit_status": "bayesian_sampling_failed",
                "curve_id": curve_id,
                "sample_record": record,
                "window_decades_requested": window,
                "sampling_seed": seed,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "analysis_role": ANALYSIS_ROLE,
            }
            failure_dir = output_dir / "failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                failure_dir
                / f"record_{record:03d}__curve_{curve_id}__window_{token}dec.json",
                failure,
            )
            failures.append(failure)
            print(f"FAILED: {curve_id} {window:.1f} dec: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise

        interim = _merge_results_with_manifest(
            selection, completed, failures, sample_records
        )
        _atomic_csv(interim, output_dir / "window_posterior_results.csv")

    result_manifest = _merge_results_with_manifest(
        selection, completed, failures, sample_records
    )
    _atomic_csv(result_manifest, output_dir / "window_posterior_results.csv")
    fit_results = result_manifest.loc[
        result_manifest["fit_status"].isin(
            SAMPLING_COMPLETE_STATUSES | SAMPLING_FAILED_STATUSES
        )
    ].copy()
    stability = build_stability_table(
        selection,
        fit_results,
        summary,
        baseline,
        requested_records=sample_records,
    )
    cohort = build_cohort_summary(stability)
    _atomic_csv(stability, output_dir / "slope_stability.csv")
    _atomic_csv(cohort, output_dir / "cohort_summary.csv")
    _write_output_readme(
        output_dir,
        selection,
        result_manifest,
        stability,
        cohort,
        smoke=args.smoke,
        requested_count=len(sample_records),
    )

    requested_eligible = int(len(eligible))
    complete_mask = result_manifest["fit_status"].isin(SAMPLING_COMPLETE_STATUSES)
    failed_mask = result_manifest["fit_status"].isin(SAMPLING_FAILED_STATUSES)
    successful = int(complete_mask.sum())
    failed = int(failed_mask.sum())
    expected_keys = {
        (str(row["curve_id"]), float(row["window_decades_requested"]))
        for _, row in eligible.iterrows()
    }
    completed_keys = {
        (str(row["curve_id"]), float(row["window_decades_requested"]))
        for _, row in result_manifest.loc[complete_mask].iterrows()
    }
    structural_pass = bool(completed_keys == expected_keys and failed == 0)
    diagnostics_pass_count = int(
        _as_bool(result_manifest.loc[complete_mask, "sampler_diagnostics_pass"]).sum()
    ) if "sampler_diagnostics_pass" in result_manifest else 0
    boundary_warning_count = int(
        _as_bool(
            result_manifest.loc[
                complete_mask, "posterior_slope_prior_boundary_warning"
            ]
        ).sum()
    ) if "posterior_slope_prior_boundary_warning" in result_manifest else 0
    analysis_ready = bool(
        structural_pass
        and diagnostics_pass_count == requested_eligible
        and boundary_warning_count == 0
    )
    completion = {
        "completed_at_utc": utc_now(),
        "analysis_role": ANALYSIS_ROLE,
        "mode": "smoke" if args.smoke else "full",
        "requested_records": len(sample_records),
        "requested_eligible_window_fits": requested_eligible,
        "bayesian_sampling_complete_count": successful,
        "bayesian_sampling_failed_count": failed,
        "structural_smoke_pass": structural_pass if args.smoke else None,
        "all_requested_eligible_sampling_complete": structural_pass,
        "sampler_diagnostics_pass_count": diagnostics_pass_count,
        "posterior_slope_prior_boundary_warning_count": boundary_warning_count,
        "analysis_ready": analysis_ready,
        "sampler_diagnostics_are_not_relaxed_for_smoke": True,
        "formal_outputs_modified": False,
    }
    _atomic_json(output_dir / "completion_status.json", completion)
    print(json.dumps(completion, indent=2, ensure_ascii=False), flush=True)
    return 0 if (structural_pass if args.smoke else analysis_ready) else 1


if __name__ == "__main__":
    raise SystemExit(main())
