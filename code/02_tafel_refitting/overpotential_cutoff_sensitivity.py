#!/usr/bin/env python3
"""Selected-model Bayesian sensitivity after OER-overpotential cutoffs.

For each of the 97 active curves and each registered lower cutoff
``eta_min in {0, 0.10, 0.15, 0.20} V``, retain every equal-potential bin at or
above the cutoff and refit that curve's already selected Bayesian model.  Each
curve uses its own remaining observed range: there is no common current-span
or decade requirement, no interpolation, and no extrapolation.

This is a sensitivity workflow.  It never modifies the frozen v5 targets,
canonical snapshot, formal posterior evidence, or named PPC decisions.
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
    "20260721_112453_overpotential_cutoff_sensitivity"
)
ETA_CUTOFFS_V = (0.0, 0.10, 0.15, 0.20)
OER_EQUILIBRIUM_POTENTIAL_V_RHE = 1.23
MINIMUM_BINS = 3
ANALYSIS_ROLE = "sensitivity_only_not_primary_target"
METHOD = "selected_model_bayesian_student_t_after_eta_min_cutoff"
COMPLETE_STATUS = "bayesian_sampling_complete"
FAILED_STATUS = "bayesian_sampling_failed"


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression, lineterminator="\n")
    temporary.replace(path)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )


def cutoff_token(eta_min_V: float) -> str:
    return f"{float(eta_min_V):.2f}".replace(".", "p")


def expected_slope_variable(model_name: str) -> str:
    if model_name == "two_regime_plateau":
        return "slope_low_mV_dec"
    if model_name in {"linear", "paper_single_plateau"}:
        return "slope_mV_dec"
    raise ValueError(f"Unsupported selected model: {model_name}")


def configure_runtime_cache(output_dir: Path) -> None:
    cache_root = Path(tempfile.gettempdir()) / "tafel-eta-cutoff-sensitivity-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("PYTENSOR_FLAGS", f"base_compiledir={cache_root / 'pytensor'}")
    output_dir.mkdir(parents=True, exist_ok=True)


def select_points_after_eta_cutoff(
    curve_points: pd.DataFrame,
    eta_min_V: float,
    *,
    min_bins: int = MINIMUM_BINS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Retain all observed bins at or above one absolute OER-overpotential cutoff."""

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
        raise ValueError(f"Cutoff input is missing columns: {missing}")
    if not np.isfinite(eta_min_V):
        raise ValueError("eta_min_V must be finite")
    if min_bins < 3:
        raise ValueError("min_bins must be at least 3")
    curve_ids = curve_points["curve_id"].astype(str).unique()
    records = pd.to_numeric(curve_points["sample_record"], errors="raise").unique()
    if len(curve_ids) != 1 or len(records) != 1:
        raise ValueError("curve_points must describe exactly one curve and record")

    ordered = curve_points.sort_values(
        ["potential_V", "potential_bin_index"], kind="mergesort"
    ).reset_index(drop=True)
    numeric = ordered[["potential_V", "relative_potential_V", "log10_current"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite binned values for {curve_ids[0]}")

    ordered = ordered.copy()
    ordered["eta_V"] = ordered["potential_V"].astype(float) - OER_EQUILIBRIUM_POTENTIAL_V_RHE
    selected = ordered.loc[ordered["eta_V"] >= float(eta_min_V) - 1e-12].copy()
    selected.reset_index(drop=True, inplace=True)
    if len(selected):
        selected["cutoff_relative_potential_V"] = (
            selected["potential_V"].astype(float) - float(selected["potential_V"].min())
        )
    else:
        selected["cutoff_relative_potential_V"] = pd.Series(dtype=float)

    n_bins = int(len(selected))
    eligible = bool(n_bins >= min_bins)
    current_span = (
        float(selected["log10_current"].max() - selected["log10_current"].min())
        if n_bins
        else np.nan
    )
    row: dict[str, Any] = {
        "curve_id": str(curve_ids[0]),
        "sample_record": int(records[0]),
        "eta_min_V": float(eta_min_V),
        "selection_rule": "all_observed_equal_potential_bins_at_or_above_eta_min",
        "potential_reference": "RHE_user_confirmed_2026-07-22",
        "oer_overpotential_definition": "eta_V=E_RHE-1.23V",
        "common_current_span_required": False,
        "common_decade_span_required": False,
        "no_interpolation": True,
        "no_extrapolation": True,
        "full_n_bins": int(len(ordered)),
        "retained_n_bins": n_bins,
        "minimum_bins_required": int(min_bins),
        "bayesian_fit_eligible": eligible,
        "ineligibility_reason": "" if eligible else f"fewer_than_{min_bins}_bins_after_cutoff",
        "retained_log_current_span_dec": current_span,
        "retained_start_potential_RHE_V": (
            float(selected["potential_V"].iloc[0]) if n_bins else np.nan
        ),
        "retained_end_potential_RHE_V": (
            float(selected["potential_V"].iloc[-1]) if n_bins else np.nan
        ),
        "retained_start_eta_V": float(selected["eta_V"].iloc[0]) if n_bins else np.nan,
        "retained_end_eta_V": float(selected["eta_V"].iloc[-1]) if n_bins else np.nan,
        "analysis_role": ANALYSIS_ROLE,
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
    for path in (points_path, summary_path, status_path, model_config_path, baseline_inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Sensitivity requires a complete deterministic source run")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    expected_hashes = model_config.get("deterministic_artifacts_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Source run lacks deterministic_artifacts_sha256")
    for relative, path in (
        ("csv/binned_curve_points.csv", points_path),
        ("csv/curve_fit_summary.csv", summary_path),
    ):
        if expected_hashes.get(relative) != sha256_file(path):
            raise ValueError(f"Source deterministic artifact differs from manifest: {path}")

    points = pd.read_csv(points_path)
    summary = pd.read_csv(summary_path)
    baseline = pd.read_csv(baseline_inventory_path)
    sets = {
        "points": set(points["curve_id"].astype(str)),
        "summary": set(summary["curve_id"].astype(str)),
        "baseline": set(baseline["curve_id"].astype(str)),
    }
    if any(len(values) != expected_curves for values in sets.values()):
        raise ValueError(
            f"Expected {expected_curves} curves in every input; found "
            + ", ".join(f"{key}={len(value)}" for key, value in sets.items())
        )
    if not (sets["points"] == sets["summary"] == sets["baseline"]):
        raise ValueError("Curve IDs differ across source artifacts")
    if summary["curve_id"].astype(str).duplicated().any() or baseline["curve_id"].astype(str).duplicated().any():
        raise ValueError("Source summary/inventory contains duplicate curve IDs")
    if not summary["fit_status"].astype(str).eq("success").all():
        raise ValueError("Every source deterministic fit must be successful")
    if not _as_bool(baseline["sampler_diagnostics_pass"]).all():
        raise ValueError("Formal baseline contains failed sampler diagnostics")

    binding = summary[["curve_id", "selected_model"]].merge(
        baseline[["curve_id", "model_name", "posterior_slope_variable"]],
        on="curve_id",
        validate="one_to_one",
    )
    if not binding["selected_model"].astype(str).eq(binding["model_name"].astype(str)).all():
        raise ValueError("Formal Bayesian model does not match deterministic selection")
    expected_variables = binding["selected_model"].astype(str).map(expected_slope_variable)
    if not binding["posterior_slope_variable"].astype(str).eq(expected_variables).all():
        raise ValueError("Formal baseline slope variable does not match selected model")

    source = {
        "run_dir": str(run_dir),
        "binned_points_path": str(points_path),
        "binned_points_sha256": sha256_file(points_path),
        "curve_summary_path": str(summary_path),
        "curve_summary_sha256": sha256_file(summary_path),
        "baseline_inventory_path": str(baseline_inventory_path),
        "baseline_inventory_sha256": sha256_file(baseline_inventory_path),
        "potential_reference": "RHE_user_confirmed_2026-07-22",
    }
    return points, summary, baseline, source


def build_cutoff_manifest(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    cutoffs: Iterable[float] = ETA_CUTOFFS_V,
    min_bins: int = MINIMUM_BINS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_lookup = summary.set_index("curve_id")
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for curve_id, curve_points in points.groupby("curve_id", sort=False):
        source_row = summary_lookup.loc[str(curve_id)]
        for cutoff in cutoffs:
            row, selected = select_points_after_eta_cutoff(
                curve_points, float(cutoff), min_bins=min_bins
            )
            model_name = str(source_row["selected_model"])
            row.update(
                {
                    "selected_model_fixed_from_source": model_name,
                    "posterior_slope_variable": expected_slope_variable(model_name),
                    "transition_width_V_fixed_from_source": float(source_row["transition_width_V"]),
                    "source_qc_flags": str(source_row.get("qc_flags", "")),
                    "known_digitization_floor_finding": str(curve_id) == "sr064",
                }
            )
            rows.append(row)
            if len(selected):
                selected = selected.copy()
                selected.insert(2, "eta_min_V", float(cutoff))
                selected["selected_model_fixed_from_source"] = model_name
                frames.append(selected)
    manifest = pd.DataFrame(rows).sort_values(["sample_record", "eta_min_V"]).reset_index(drop=True)
    selected_points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return manifest, selected_points


def _artifact_paths(fit_dir: Path) -> dict[str, Path]:
    return {
        "netcdf": fit_dir / "posterior.nc",
        "posterior_summary": fit_dir / "posterior_summary.csv",
        "slope_draws": fit_dir / "posterior_slope_draws.csv.gz",
        "diagnostics": fit_dir / "diagnostics.json",
        "ppc": fit_dir / "posterior_predictive.png",
        "result": fit_dir / "result.json",
    }


def _finite_dataset_values(dataset: Any) -> np.ndarray:
    arrays = [
        np.asarray(dataset[name].values, dtype=float).reshape(-1)
        for name in dataset.data_vars
    ]
    values = np.concatenate(arrays) if arrays else np.array([], dtype=float)
    return values[np.isfinite(values)]


def _load_completed_fit(
    fit_dir: Path,
    *,
    curve_id: str,
    sample_record: int,
    eta_min_V: float,
    model_name: str,
    seed: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    selected_points_sha256: str,
    run_configuration_sha256: str,
) -> dict[str, Any] | None:
    paths = _artifact_paths(fit_dir)
    existing = [key for key, path in paths.items() if path.exists()]
    if not existing:
        return None
    if len(existing) != len(paths):
        raise RuntimeError(f"Partial Bayesian artifact set in {fit_dir}: {existing}")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    expected = {
        "curve_id": curve_id,
        "sample_record": sample_record,
        "eta_min_V": eta_min_V,
        "model_name": model_name,
        "sampling_seed": seed,
        "draws_per_chain": draws,
        "tune_per_chain": tune,
        "chains": chains,
        "target_accept": target_accept,
        "selected_points_sha256": selected_points_sha256,
        "run_configuration_sha256": run_configuration_sha256,
    }
    mismatches = {key: (result.get(key), value) for key, value in expected.items() if result.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Completed-fit identity mismatch in {fit_dir}: {mismatches}")
    for key in ("netcdf", "posterior_summary", "slope_draws", "diagnostics", "ppc"):
        if result.get("artifact_sha256", {}).get(key) != sha256_file(paths[key]):
            raise RuntimeError(f"Artifact hash mismatch for {paths[key]}")
    return result


def _fit_into_directory(
    selected_points: pd.DataFrame,
    manifest_row: pd.Series,
    *,
    fit_dir: Path,
    published_fit_dir: Path,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    progressbar: bool,
    provenance: dict[str, str],
) -> dict[str, Any]:
    configure_runtime_cache(fit_dir)
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    from tafel_refit.bayes import _bayes_mean, render_posterior_predictive_plot

    if draws < 1 or tune < 0 or chains < 2:
        raise ValueError("draws>=1, tune>=0, and chains>=2 are required")
    if not 0.5 < target_accept < 1.0:
        raise ValueError("target_accept must be between 0.5 and 1.0")
    fit_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(fit_dir)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to overwrite Bayesian artifacts in {fit_dir}")

    ordered = selected_points.sort_values(
        ["potential_V", "potential_bin_index"], kind="mergesort"
    ).reset_index(drop=True)
    potential_rhe = ordered["potential_V"].to_numpy(dtype=float)
    x = ordered["cutoff_relative_potential_V"].to_numpy(dtype=float)
    y = ordered["log10_current"].to_numpy(dtype=float)
    model_name = str(manifest_row["selected_model_fixed_from_source"])
    slope_variable = expected_slope_variable(model_name)
    transition_width = float(manifest_row["transition_width_V_fixed_from_source"])
    point_hash = dataframe_sha256(ordered)

    observation_ids = ordered["potential_bin_index"].to_numpy(dtype=int)
    with pm.Model(coords={"observation": observation_ids}):
        mean, variable_names = _bayes_mean(model_name, x, y, transition_width)
        log_sigma = pm.Uniform("log_sigma", lower=float(np.log(0.005)), upper=float(np.log(0.5)))
        sigma = pm.Deterministic("sigma_log10_current", pt.exp(log_sigma))
        variable_names.extend(["log_sigma", "sigma_log10_current"])
        pm.StudentT(
            "observed_log10_current",
            nu=4.0,
            mu=mean,
            sigma=sigma,
            observed=y,
            dims="observation",
        )
        idata = pm.sample(
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
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["observed_log10_current"],
            random_seed=seed + 1,
            progressbar=progressbar,
        )
        idata.extend(ppc)

    idata.attrs.update(
        {
            "analysis_role": ANALYSIS_ROLE,
            "method": METHOD,
            "curve_id": str(manifest_row["curve_id"]),
            "sample_record": int(manifest_row["sample_record"]),
            "eta_min_V": float(manifest_row["eta_min_V"]),
            "potential_reference": "RHE_user_confirmed_2026-07-22",
            "model_name": model_name,
            "posterior_slope_variable": slope_variable,
            "selected_points_sha256": point_hash,
            "run_configuration_sha256": provenance["run_configuration_sha256"],
            "sampling_seed": int(seed),
        }
    )
    temporary_netcdf = paths["netcdf"].with_suffix(".nc.tmp")
    idata.to_netcdf(temporary_netcdf)
    temporary_netcdf.replace(paths["netcdf"])
    netcdf_hash = sha256_file(paths["netcdf"])

    posterior_summary = az.summary(idata, var_names=variable_names, kind="all", hdi_prob=0.95)
    _atomic_csv(posterior_summary.reset_index(names="variable"), paths["posterior_summary"])
    slope_array = np.asarray(idata.posterior[slope_variable].values, dtype=float)
    n_chains, n_draws = slope_array.shape[:2]
    slopes = slope_array.reshape(-1)
    slope_hdi = np.asarray(az.hdi(slopes, hdi_prob=0.95), dtype=float)
    slope_draws = pd.DataFrame(
        {
            "curve_id": str(manifest_row["curve_id"]),
            "sample_record": int(manifest_row["sample_record"]),
            "eta_min_V": float(manifest_row["eta_min_V"]),
            "model_name": model_name,
            "posterior_slope_variable": slope_variable,
            "chain": np.repeat(np.arange(n_chains), n_draws),
            "draw": np.tile(np.arange(n_draws), n_chains),
            "slope_mV_dec": slopes,
            "source_netcdf_sha256": netcdf_hash,
            "selected_points_sha256": point_hash,
        }
    )
    _atomic_csv(slope_draws, paths["slope_draws"], compression="gzip")

    rhat = _finite_dataset_values(az.rhat(idata, var_names=variable_names, method="rank"))
    ess = _finite_dataset_values(az.ess(idata, var_names=variable_names, method="bulk"))
    divergences = int(idata.sample_stats["diverging"].sum().item())
    sampler_pass = bool(
        len(rhat) and np.max(rhat) <= 1.01 and len(ess) and np.min(ess) >= 400 and divergences == 0
    )
    boundary_fraction = float(np.mean((slopes <= 10.5) | (slopes >= 990.0)))
    ppc_array = np.asarray(
        idata.posterior_predictive["observed_log10_current"].values, dtype=float
    ).reshape(-1, len(y))
    ppc_metrics = render_posterior_predictive_plot(
        curve_id=str(manifest_row["curve_id"]),
        model_name=model_name,
        potential_rhe_V=potential_rhe,
        observed_log10_current=y,
        posterior_predictive_draws=ppc_array,
        output_path=paths["ppc"],
    )
    diagnostics = {
        "created_at_utc": utc_now(),
        "analysis_role": ANALYSIS_ROLE,
        "formal_confirmation_provenance_eligible": False,
        "formal_outputs_modified": False,
        "curve_id": str(manifest_row["curve_id"]),
        "sample_record": int(manifest_row["sample_record"]),
        "eta_min_V": float(manifest_row["eta_min_V"]),
        "potential_reference": "RHE_user_confirmed_2026-07-22",
        "model_name": model_name,
        "posterior_slope_variable": slope_variable,
        "selected_model_is_fixed_from_source": True,
        "common_current_span_required": False,
        "retained_n_bins": int(len(ordered)),
        "retained_log_current_span_dec": float(np.ptp(y)),
        "selected_points_sha256": point_hash,
        "run_configuration_sha256": provenance["run_configuration_sha256"],
        "draws_per_chain": int(draws),
        "tune_per_chain": int(tune),
        "chains": int(chains),
        "target_accept": float(target_accept),
        "sampling_seed": int(seed),
        "max_r_hat": float(np.max(rhat)) if len(rhat) else None,
        "min_ess_bulk": float(np.min(ess)) if len(ess) else None,
        "divergences": divergences,
        "sampler_diagnostics_pass": sampler_pass,
        "posterior_slope_prior_boundary_fraction": boundary_fraction,
        "posterior_slope_prior_boundary_warning": boundary_fraction > 0.05,
        **ppc_metrics,
    }
    _atomic_json(paths["diagnostics"], diagnostics)
    artifact_hashes = {
        key: sha256_file(paths[key])
        for key in ("netcdf", "posterior_summary", "slope_draws", "diagnostics", "ppc")
    }
    result = {
        "fit_status": COMPLETE_STATUS,
        "curve_id": str(manifest_row["curve_id"]),
        "sample_record": int(manifest_row["sample_record"]),
        "eta_min_V": float(manifest_row["eta_min_V"]),
        "model_name": model_name,
        "posterior_slope_variable": slope_variable,
        "selected_points_sha256": point_hash,
        "run_configuration_sha256": provenance["run_configuration_sha256"],
        "posterior_slope_median_mV_dec": float(np.median(slopes)),
        "posterior_slope_mean_mV_dec": float(np.mean(slopes)),
        "posterior_slope_hdi_low_mV_dec": float(slope_hdi[0]),
        "posterior_slope_hdi_high_mV_dec": float(slope_hdi[1]),
        "posterior_slope_total_draws": int(len(slopes)),
        "draws_per_chain": int(draws),
        "tune_per_chain": int(tune),
        "chains": int(chains),
        "target_accept": float(target_accept),
        "sampling_seed": int(seed),
        "max_r_hat": diagnostics["max_r_hat"],
        "min_ess_bulk": diagnostics["min_ess_bulk"],
        "divergences": divergences,
        "sampler_diagnostics_pass": sampler_pass,
        "posterior_slope_prior_boundary_fraction": boundary_fraction,
        "posterior_slope_prior_boundary_warning": boundary_fraction > 0.05,
        "netcdf_path": str(_artifact_paths(published_fit_dir)["netcdf"]),
        "posterior_summary_path": str(_artifact_paths(published_fit_dir)["posterior_summary"]),
        "slope_draws_path": str(_artifact_paths(published_fit_dir)["slope_draws"]),
        "diagnostics_path": str(_artifact_paths(published_fit_dir)["diagnostics"]),
        "ppc_plot_path": str(_artifact_paths(published_fit_dir)["ppc"]),
        "artifact_sha256": artifact_hashes,
        "analysis_role": ANALYSIS_ROLE,
    }
    _atomic_json(paths["result"], result)
    return result


def fit_selected_model(
    selected_points: pd.DataFrame,
    manifest_row: pd.Series,
    *,
    fit_dir: Path,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    progressbar: bool,
    provenance: dict[str, str],
) -> dict[str, Any]:
    final_dir = fit_dir.resolve()
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Bayesian fit directory: {final_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.inprogress-", dir=final_dir.parent))
    try:
        result = _fit_into_directory(
            selected_points,
            manifest_row,
            fit_dir=staging,
            published_fit_dir=final_dir,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            seed=seed,
            progressbar=progressbar,
            provenance=provenance,
        )
        staging.replace(final_dir)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def merge_results(
    manifest: pd.DataFrame,
    completed: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    requested_records: set[int],
) -> pd.DataFrame:
    output = manifest.loc[manifest["sample_record"].astype(int).isin(requested_records)].copy()
    results = pd.DataFrame(completed + failures)
    if len(results):
        output = output.merge(
            results,
            on=["curve_id", "sample_record", "eta_min_V"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_fit"),
        )
    if "fit_status" not in output:
        output["fit_status"] = np.nan
    requested = output["bayesian_fit_eligible"].astype(bool)
    output.loc[~requested, "fit_status"] = "cutoff_ineligible"
    output.loc[requested & output["fit_status"].isna(), "fit_status"] = "not_requested_or_pending"
    return output.sort_values(["sample_record", "eta_min_V"]).reset_index(drop=True)


def build_slope_sensitivity(
    results: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    base = baseline[[
        "curve_id",
        "model_name",
        "posterior_slope_variable",
        "posterior_slope_median_mV_dec",
        "posterior_slope_draw_hdi_low_mV_dec",
        "posterior_slope_draw_hdi_high_mV_dec",
    ]].rename(
        columns={
            "model_name": "formal_model",
            "posterior_slope_variable": "formal_slope_variable",
            "posterior_slope_median_mV_dec": "formal_slope_median_mV_dec",
            "posterior_slope_draw_hdi_low_mV_dec": "formal_slope_hdi_low_mV_dec",
            "posterior_slope_draw_hdi_high_mV_dec": "formal_slope_hdi_high_mV_dec",
        }
    )
    keep = [
        "curve_id",
        "sample_record",
        "eta_min_V",
        "retained_n_bins",
        "retained_log_current_span_dec",
        "retained_start_eta_V",
        "retained_end_eta_V",
        "fit_status",
        "posterior_slope_median_mV_dec",
        "posterior_slope_hdi_low_mV_dec",
        "posterior_slope_hdi_high_mV_dec",
        "sampler_diagnostics_pass",
        "posterior_slope_prior_boundary_warning",
    ]
    available = [column for column in keep if column in results.columns]
    long = results[available].merge(base, on="curve_id", how="left", validate="many_to_one")
    long["cutoff_minus_formal_mV_dec"] = (
        long["posterior_slope_median_mV_dec"] - long["formal_slope_median_mV_dec"]
    )
    long["cutoff_to_formal_ratio"] = (
        long["posterior_slope_median_mV_dec"] / long["formal_slope_median_mV_dec"]
    )
    long["analysis_role"] = ANALYSIS_ROLE
    return long.sort_values(["sample_record", "eta_min_V"]).reset_index(drop=True)


def write_readme(output_dir: Path, manifest: pd.DataFrame, *, smoke: bool) -> None:
    counts = manifest.groupby("eta_min_V").agg(
        eligible_curves=("bayesian_fit_eligible", "sum"),
        minimum_retained_bins=("retained_n_bins", "min"),
        median_retained_current_span_dec=("retained_log_current_span_dec", "median"),
    )
    count_lines = [
        "| eta_min / V | eligible curves | minimum retained bins | median retained current span / dec |",
        "|---:|---:|---:|---:|",
    ]
    for cutoff, row in counts.iterrows():
        count_lines.append(
            f"| {float(cutoff):.2f} | {int(row['eligible_curves'])} | "
            f"{int(row['minimum_retained_bins'])} | "
            f"{float(row['median_retained_current_span_dec']):.3f} |"
        )
    lines = [
        "# Bayesian OER-overpotential cutoff sensitivity",
        "",
        "This is a sensitivity-only refit of the fixed selected model for each curve.",
        "All potential values are treated as RHE by the user's 2026-07-22 confirmation,",
        "and `eta_V = E_RHE - 1.23 V`.",
        "",
        "For each cutoff (0, 0.10, 0.15, 0.20 V), every observed equal-potential bin",
        "at or above the cutoff is retained. Each curve uses its own remaining range;",
        "there is no common current-decade requirement, interpolation, or extrapolation.",
        "The source selected model is fixed; model selection is not repeated after cutoff.",
        "",
        f"Mode: {'smoke' if smoke else 'full'}",
        "",
        "## Structural support before sampling",
        "",
        *count_lines,
        "",
        "This output does not replace the frozen v5 target, canonical snapshot, formal",
        "PPC review, or ML input. A future target amendment requires registration and",
        "a new controlled evidence publication.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--baseline-inventory", type=Path, default=DEFAULT_BASELINE_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eta-min-V", type=float, nargs="+", default=list(ETA_CUTOFFS_V))
    parser.add_argument("--min-bins", type=int, default=MINIMUM_BINS)
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
        help="Fit Records 3, 17, and 20 at all four cutoffs with 20 tune/draws and two chains.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cutoffs = tuple(sorted(float(value) for value in args.eta_min_V))
    if cutoffs != ETA_CUTOFFS_V:
        raise ValueError("This registered sensitivity requires eta_min = 0, 0.10, 0.15, 0.20 V")
    if args.min_bins < 3:
        raise ValueError("--min-bins must be at least 3")
    if args.smoke:
        draws, tune, chains = 20, 20, 2
        requested_records = set(args.sample_records or [3, 17, 20])
    else:
        draws, tune, chains = args.draws, args.tune, args.chains
        requested_records = set(args.sample_records or range(1, args.expected_curves + 1))
    if any(record < 1 or record > args.expected_curves for record in requested_records):
        raise ValueError("--sample-records contains a record outside active 1..97")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --resume")
    configure_runtime_cache(output_dir)
    points, summary, baseline, source = load_and_validate_inputs(
        args.run_dir, args.baseline_inventory, expected_curves=args.expected_curves
    )
    manifest, selected_points = build_cutoff_manifest(
        points, summary, cutoffs=cutoffs, min_bins=args.min_bins
    )
    if args.expected_curves == 97 and int(manifest["bayesian_fit_eligible"].sum()) != 97 * len(cutoffs):
        raise ValueError("The registered 97-curve cutoff sensitivity must retain all 97 curves at every cutoff")

    configuration = {
        "analysis_role": ANALYSIS_ROLE,
        "method": METHOD,
        "eta_min_V": list(cutoffs),
        "potential_reference": "RHE_user_confirmed_2026-07-22",
        "oer_overpotential_definition": "eta_V=E_RHE-1.23V",
        "selection_rule": "all_observed_equal_potential_bins_at_or_above_eta_min",
        "common_current_span_required": False,
        "common_decade_span_required": False,
        "selected_model_fixed_from_source": True,
        "minimum_bins": int(args.min_bins),
        "draws_per_chain": int(draws),
        "tune_per_chain": int(tune),
        "chains": int(chains),
        "target_accept": float(args.target_accept),
        "base_seed": int(args.seed),
        "sample_records": sorted(requested_records),
        "smoke": bool(args.smoke),
        "expected_curves": int(args.expected_curves),
        "cutoff_manifest_sha256": dataframe_sha256(manifest),
        "selected_cutoff_points_sha256": dataframe_sha256(selected_points),
        "source": source,
    }
    config_path = output_dir / "run_configuration.json"
    manifest_path = output_dir / "cutoff_manifest.csv"
    points_path = output_dir / "selected_cutoff_points.csv.gz"
    if args.resume:
        for path in (config_path, manifest_path, points_path):
            if not path.is_file():
                raise FileNotFoundError(f"--resume requires {path}")
        configuration["cutoff_manifest_file_sha256"] = sha256_file(manifest_path)
        configuration["selected_cutoff_points_file_sha256"] = sha256_file(points_path)
        if json.loads(config_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Resume configuration differs from the existing run")
    else:
        _atomic_csv(manifest, manifest_path)
        _atomic_csv(selected_points, points_path, compression="gzip")
        configuration["cutoff_manifest_file_sha256"] = sha256_file(manifest_path)
        configuration["selected_cutoff_points_file_sha256"] = sha256_file(points_path)
        _atomic_json(config_path, configuration)
    provenance = {"run_configuration_sha256": sha256_file(config_path)}

    point_lookup = {
        (str(curve_id), float(cutoff)): frame.copy()
        for (curve_id, cutoff), frame in selected_points.groupby(["curve_id", "eta_min_V"], sort=False)
    }
    eligible = manifest.loc[
        manifest["sample_record"].astype(int).isin(requested_records)
        & manifest["bayesian_fit_eligible"].astype(bool)
    ].sort_values(["sample_record", "eta_min_V"])
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        curve_id = str(row["curve_id"])
        record = int(row["sample_record"])
        cutoff = float(row["eta_min_V"])
        model_name = str(row["selected_model_fixed_from_source"])
        token = cutoff_token(cutoff)
        fit_dir = output_dir / "bayes" / f"record_{record:03d}__curve_{curve_id}__eta_min_{token}V"
        seed = int(args.seed + record * 10_000 + round(cutoff * 1000))
        frame = point_lookup[(curve_id, cutoff)].sort_values(
            ["potential_V", "potential_bin_index"], kind="mergesort"
        ).reset_index(drop=True)
        point_hash = dataframe_sha256(frame)
        previous = (
            _load_completed_fit(
                fit_dir,
                curve_id=curve_id,
                sample_record=record,
                eta_min_V=cutoff,
                model_name=model_name,
                seed=seed,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=args.target_accept,
                selected_points_sha256=point_hash,
                run_configuration_sha256=provenance["run_configuration_sha256"],
            )
            if args.resume
            else None
        )
        if previous is not None:
            completed.append(previous)
            print(f"resume: {curve_id} eta>={cutoff:.2f} V already complete", flush=True)
            continue
        print(f"fit: {curve_id} {model_name} eta>={cutoff:.2f} V (seed {seed})", flush=True)
        try:
            completed.append(
                fit_selected_model(
                    frame,
                    row,
                    fit_dir=fit_dir,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=args.target_accept,
                    seed=seed,
                    progressbar=not args.no_progress,
                    provenance=provenance,
                )
            )
        except Exception as exc:
            failure = {
                "fit_status": FAILED_STATUS,
                "curve_id": curve_id,
                "sample_record": record,
                "eta_min_V": cutoff,
                "model_name": model_name,
                "sampling_seed": seed,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "analysis_role": ANALYSIS_ROLE,
            }
            failure_dir = output_dir / "failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(failure_dir / f"record_{record:03d}__curve_{curve_id}__eta_min_{token}V.json", failure)
            failures.append(failure)
            print(f"FAILED: {curve_id} eta>={cutoff:.2f}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
        _atomic_csv(
            merge_results(manifest, completed, failures, requested_records),
            output_dir / "cutoff_posterior_results.csv",
        )

    results = merge_results(manifest, completed, failures, requested_records)
    _atomic_csv(results, output_dir / "cutoff_posterior_results.csv")
    sensitivity = build_slope_sensitivity(results, baseline)
    _atomic_csv(sensitivity, output_dir / "slope_sensitivity.csv")
    write_readme(output_dir, manifest, smoke=args.smoke)

    complete = results["fit_status"].eq(COMPLETE_STATUS)
    failed = results["fit_status"].eq(FAILED_STATUS)
    expected_count = int(len(eligible))
    structural_pass = bool(int(complete.sum()) == expected_count and not failed.any())
    diagnostics_pass_count = int(
        _as_bool(results.loc[complete, "sampler_diagnostics_pass"]).sum()
    ) if "sampler_diagnostics_pass" in results else 0
    status = {
        "completed_at_utc": utc_now(),
        "analysis_role": ANALYSIS_ROLE,
        "mode": "smoke" if args.smoke else "full",
        "requested_records": len(requested_records),
        "requested_eligible_fits": expected_count,
        "bayesian_sampling_complete_count": int(complete.sum()),
        "bayesian_sampling_failed_count": int(failed.sum()),
        "structural_smoke_pass": structural_pass if args.smoke else None,
        "all_requested_eligible_sampling_complete": structural_pass,
        "sampler_diagnostics_pass_count": diagnostics_pass_count,
        "sampler_diagnostics_not_required_for_short_smoke": bool(args.smoke),
        "analysis_ready": bool(structural_pass and (args.smoke or diagnostics_pass_count == expected_count)),
        "formal_outputs_modified": False,
    }
    _atomic_json(output_dir / "completion_status.json", status)
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    return 0 if status["analysis_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
