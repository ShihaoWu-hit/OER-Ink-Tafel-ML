#!/usr/bin/env python3
"""Fit two non-canonical uniform Equation-2 Bayesian Tafel targets.

The two registered sensitivity targets are:

``uniform_eq2_eta015``
    Every observed equal-potential bin at eta >= 0.15 V, the project's robust
    Student-t likelihood, and the project's bounded log-slope/plateau priors.

``uniform_eq2_paper_replication``
    Every positive raw point accepted by deterministic preprocessing, the
    Limaye et al. shifted-log-current parameterization, fixed sigma=0.10
    Gaussian likelihood, and data-preconditioned Uniform(0, 10 * theta_star)
    priors from the authors' released ``julius`` implementation.

Both targets use the same one-kinetic-regime plus limiting-current Equation 2.
They are sensitivity evidence and never modify the active canonical target.
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
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "Data/03_outputs/tafel_refitting/outputs/20260721_112453"
DEFAULT_OUTPUT_DIR = (
    ROOT / "Data/03_outputs/tafel_refitting/outputs/20260723_uniform_equation2_dual_target_v1"
)
TARGET_EQ2_ETA015 = "uniform_eq2_eta015"
TARGET_EQ2_PAPER = "uniform_eq2_paper_replication"
TARGET_IDS = (TARGET_EQ2_ETA015, TARGET_EQ2_PAPER)
TARGET_NAMES = {
    TARGET_EQ2_ETA015: "b_uniform_eq2_eta015_mV_dec",
    TARGET_EQ2_PAPER: "b_uniform_eq2_paper_replication_mV_dec",
}
ANALYSIS_ROLE = "sensitivity_target_not_canonical"
MODEL_NAME = "paper_single_plateau"
OER_EQUILIBRIUM_POTENTIAL_V_RHE = 1.23
ETA_MIN_V = 0.15
PAPER_FIXED_SIGMA_LOG10 = 0.10
PAPER_PRIOR_MULTIPLIER = 10.0
LOG10E = np.log10(np.e)
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
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(
    frame: pd.DataFrame, path: Path, *, compression: str | None = None
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary, index=False, compression=compression, lineterminator="\n"
    )
    temporary.replace(path)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )


def configure_runtime_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "tafel-uniform-equation2-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("PYTENSOR_FLAGS", f"base_compiledir={cache_root / 'pytensor'}")


def paper_equation2_log10(
    relative_potential_V: np.ndarray,
    alpha_per_V: float,
    b_log_current_natural: float,
    ilim_ratio: float,
) -> np.ndarray:
    """Authors' released series-resistances Equation-2 parameterization."""

    x = np.asarray(relative_potential_V, dtype=float)
    if alpha_per_V <= 0 or b_log_current_natural <= 0 or ilim_ratio <= 0:
        return np.full_like(x, np.nan, dtype=float)
    return LOG10E * (
        b_log_current_natural
        + alpha_per_V * x
        - np.logaddexp(np.log(ilim_ratio), alpha_per_V * x)
    )


def paper_alpha_to_slope_mV_dec(alpha_per_V: np.ndarray | float) -> np.ndarray:
    alpha = np.asarray(alpha_per_V, dtype=float)
    return 1000.0 / (LOG10E * alpha)


def optimize_paper_preconditioner(
    x: np.ndarray,
    shifted_log10_current: np.ndarray,
    *,
    project_initial: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Find positive theta_star for the paper's Uniform(0, 10*theta_star) priors."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(shifted_log10_current, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("Paper preconditioner needs at least three paired points")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Paper preconditioner inputs must be finite")

    def objective(theta: np.ndarray) -> float:
        prediction = paper_equation2_log10(x, *theta)
        if not np.isfinite(prediction).all():
            return np.inf
        return float(np.sum(np.square(y - prediction)))

    starts = [np.asarray((15.0, 2.0, 15.0), dtype=float)]
    if project_initial is not None:
        candidate = np.asarray(project_initial, dtype=float)
        if np.isfinite(candidate).all() and (candidate > 0).all():
            starts.append(candidate)
    bounds = ((1e-6, 1e4), (1e-6, 1e4), (1e-8, 1e12))
    fits = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        for start in starts
    ]
    finite = [fit for fit in fits if np.isfinite(fit.fun) and (fit.x > 0).all()]
    if not finite:
        raise RuntimeError("Could not obtain positive Equation-2 preconditioner")
    best = min(finite, key=lambda fit: float(fit.fun))
    theta = np.asarray(best.x, dtype=float)
    upper = PAPER_PRIOR_MULTIPLIER * theta
    return {
        "paper_theta_star_alpha_per_V": float(theta[0]),
        "paper_theta_star_b_natural": float(theta[1]),
        "paper_theta_star_ilim_ratio": float(theta[2]),
        "paper_prior_alpha_upper_per_V": float(upper[0]),
        "paper_prior_b_upper_natural": float(upper[1]),
        "paper_prior_ilim_ratio_upper": float(upper[2]),
        "paper_preconditioner_sse": float(best.fun),
        "paper_preconditioner_success": bool(best.success),
        "paper_preconditioner_message": str(best.message),
        "paper_preconditioner_optimizer": "positive_L-BFGS-B_modernization",
        "paper_released_code_initial_guess": "alpha=15,b=2,ilim=15",
    }


def select_eta015_binned(curve: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "curve_id",
        "sample_record",
        "potential_bin_index",
        "potential_V",
        "log10_current",
    }
    if missing := required.difference(curve.columns):
        raise ValueError(f"Binned input missing columns: {sorted(missing)}")
    ordered = curve.sort_values(
        ["potential_V", "potential_bin_index"], kind="mergesort"
    ).reset_index(drop=True)
    selected = ordered.loc[
        pd.to_numeric(ordered["potential_V"]) - OER_EQUILIBRIUM_POTENTIAL_V_RHE
        >= ETA_MIN_V - 1e-12
    ].copy()
    if len(selected) < 3:
        raise ValueError("eta015 Equation-2 target requires at least three bins")
    selected["target_id"] = TARGET_EQ2_ETA015
    selected["eta_V"] = (
        pd.to_numeric(selected["potential_V"])
        - OER_EQUILIBRIUM_POTENTIAL_V_RHE
    )
    selected["fit_relative_potential_V"] = (
        pd.to_numeric(selected["potential_V"])
        - float(pd.to_numeric(selected["potential_V"]).min())
    )
    selected["fit_log10_current"] = pd.to_numeric(selected["log10_current"])
    selected["observation_id"] = pd.to_numeric(
        selected["potential_bin_index"], errors="raise"
    ).astype(int)
    audit = {
        "curve_id": str(selected["curve_id"].iloc[0]),
        "sample_record": int(selected["sample_record"].iloc[0]),
        "target_id": TARGET_EQ2_ETA015,
        "target_name": TARGET_NAMES[TARGET_EQ2_ETA015],
        "model_name": MODEL_NAME,
        "input_kind": "observed_equal_potential_bins",
        "selection_rule": "all_equal_potential_bins_at_or_above_eta_0p15V",
        "eta_min_V": ETA_MIN_V,
        "likelihood": "student_t_nu4_estimated_sigma",
        "prior_policy": "project_log_slope_and_identifiable_plateau_bounds",
        "n_observations": int(len(selected)),
        "log_current_span_dec": float(np.ptp(selected["fit_log10_current"])),
        "start_eta_V": float(selected["eta_V"].min()),
        "end_eta_V": float(selected["eta_V"].max()),
        "common_current_span_required": False,
        "all_available_points_within_curve_rule": False,
    }
    return audit, selected


def select_paper_raw(
    curve: pd.DataFrame, summary_row: pd.Series
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "curve_id",
        "sample_record",
        "excel_row",
        "potential_raw",
        "current_raw",
        "used_for_log_fit",
    }
    if missing := required.difference(curve.columns):
        raise ValueError(f"Raw-point input missing columns: {sorted(missing)}")
    selected = curve.loc[_as_bool(curve["used_for_log_fit"])].copy()
    selected = selected.sort_values(
        ["potential_raw", "excel_row"], kind="mergesort"
    ).reset_index(drop=True)
    if len(selected) < 3:
        raise ValueError("Paper-replication target requires at least three raw points")
    if (pd.to_numeric(selected["current_raw"]) <= 0).any():
        raise ValueError("Paper-replication selected points must have positive current")
    selected["target_id"] = TARGET_EQ2_PAPER
    selected["potential_V"] = pd.to_numeric(selected["potential_raw"])
    selected["eta_V"] = selected["potential_V"] - OER_EQUILIBRIUM_POTENTIAL_V_RHE
    selected["fit_relative_potential_V"] = (
        selected["potential_V"] - float(selected["potential_V"].min())
    )
    selected["unshifted_log10_current"] = np.log10(
        pd.to_numeric(selected["current_raw"]).to_numpy(dtype=float)
    )
    selected["fit_log10_current"] = (
        selected["unshifted_log10_current"]
        - float(selected["unshifted_log10_current"].min())
    )
    selected["observation_id"] = pd.to_numeric(
        selected["excel_row"], errors="raise"
    ).astype(int)

    log_m = float(summary_row.get("log_m", np.nan))
    a = float(summary_row.get("a", np.nan))
    plateau = float(summary_row.get("L", np.nan))
    shift = float(selected["unshifted_log10_current"].min())
    mapped_initial = (
        float(np.log(10.0) * np.exp(log_m)),
        float(np.log(10.0) * (plateau - shift)),
        float(10.0 ** (plateau - a)),
    )
    preconditioner = optimize_paper_preconditioner(
        selected["fit_relative_potential_V"].to_numpy(dtype=float),
        selected["fit_log10_current"].to_numpy(dtype=float),
        project_initial=mapped_initial,
    )
    audit = {
        "curve_id": str(selected["curve_id"].iloc[0]),
        "sample_record": int(selected["sample_record"].iloc[0]),
        "target_id": TARGET_EQ2_PAPER,
        "target_name": TARGET_NAMES[TARGET_EQ2_PAPER],
        "model_name": MODEL_NAME,
        "input_kind": "all_positive_raw_points_accepted_by_preprocessing",
        "selection_rule": "all_used_for_log_fit_raw_points",
        "eta_min_V": np.nan,
        "likelihood": "iid_gaussian_fixed_sigma_0p10_log10",
        "prior_policy": "limaye_uniform_0_to_10x_theta_star",
        "n_observations": int(len(selected)),
        "log_current_span_dec": float(np.ptp(selected["fit_log10_current"])),
        "start_eta_V": float(selected["eta_V"].min()),
        "end_eta_V": float(selected["eta_V"].max()),
        "common_current_span_required": False,
        "all_available_points_within_curve_rule": True,
        "paper_log_current_shift_dec": shift,
        "paper_fixed_sigma_log10": PAPER_FIXED_SIGMA_LOG10,
        "paper_prior_multiplier": PAPER_PRIOR_MULTIPLIER,
        **preconditioner,
    }
    return audit, selected


def load_inputs(run_dir: Path, *, expected_curves: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = run_dir.resolve()
    paths = {
        "binned": run_dir / "csv/binned_curve_points.csv",
        "raw": run_dir / "csv/curve_points_audited.csv",
        "summary": run_dir / "csv/curve_fit_summary.csv",
        "status": run_dir / "run_status.json",
        "config": run_dir / "model_config.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Uniform Equation-2 targets require a complete source run")
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    expected_hashes = config.get("deterministic_artifacts_sha256", {})
    for relative, key in (
        ("csv/binned_curve_points.csv", "binned"),
        ("csv/curve_points_audited.csv", "raw"),
        ("csv/curve_fit_summary.csv", "summary"),
    ):
        if expected_hashes.get(relative) != sha256_file(paths[key]):
            raise ValueError(f"Source artifact differs from frozen manifest: {paths[key]}")

    binned = pd.read_csv(paths["binned"])
    raw = pd.read_csv(paths["raw"])
    summary = pd.read_csv(paths["summary"])
    sets = {
        "binned": set(binned["curve_id"].astype(str)),
        "raw": set(raw["curve_id"].astype(str)),
        "summary": set(summary["curve_id"].astype(str)),
    }
    if any(len(values) != expected_curves for values in sets.values()):
        raise ValueError(f"Expected {expected_curves} curves in every source table: {sets}")
    if not (sets["binned"] == sets["raw"] == sets["summary"]):
        raise ValueError("Curve IDs differ across source inputs")
    if summary["curve_id"].duplicated().any():
        raise ValueError("Source summary has duplicate curve IDs")
    if not summary["fit_status"].astype(str).eq("success").all():
        raise ValueError("All source deterministic fits must be successful")
    source = {
        "deterministic_run_dir": str(run_dir),
        **{f"{key}_path": str(path) for key, path in paths.items()},
        **{
            f"{key}_sha256": sha256_file(path)
            for key, path in paths.items()
        },
    }
    return binned, raw, summary, source


def build_manifest_and_points(
    binned: pd.DataFrame, raw: pd.DataFrame, summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_lookup = summary.set_index("curve_id")
    manifest_rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    for curve_id in summary.sort_values("sample_record")["curve_id"].astype(str):
        binned_audit, binned_selected = select_eta015_binned(
            binned.loc[binned["curve_id"].astype(str).eq(curve_id)]
        )
        raw_audit, raw_selected = select_paper_raw(
            raw.loc[raw["curve_id"].astype(str).eq(curve_id)],
            summary_lookup.loc[curve_id],
        )
        manifest_rows.extend((binned_audit, raw_audit))
        point_frames.extend((binned_selected, raw_selected))
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["sample_record", "target_id"]
    ).reset_index(drop=True)
    points = pd.concat(point_frames, ignore_index=True, sort=False)
    return manifest, points


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
    target_id: str,
    curve_id: str,
    sample_record: int,
    seed: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    points_sha256: str,
    configuration_sha256: str,
) -> dict[str, Any] | None:
    paths = _artifact_paths(fit_dir)
    existing = [key for key, path in paths.items() if path.exists()]
    if not existing:
        return None
    if len(existing) != len(paths):
        raise RuntimeError(f"Partial artifact set in {fit_dir}: {existing}")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    expected = {
        "target_id": target_id,
        "curve_id": curve_id,
        "sample_record": sample_record,
        "sampling_seed": seed,
        "draws_per_chain": draws,
        "tune_per_chain": tune,
        "chains": chains,
        "target_accept": target_accept,
        "selected_points_sha256": points_sha256,
        "run_configuration_sha256": configuration_sha256,
    }
    mismatches = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Completed-fit identity mismatch in {fit_dir}: {mismatches}")
    for key in ("netcdf", "posterior_summary", "slope_draws", "diagnostics", "ppc"):
        if result["artifact_sha256"].get(key) != sha256_file(paths[key]):
            raise RuntimeError(f"Artifact hash mismatch for {paths[key]}")
    return result


def _fit_model(
    selected: pd.DataFrame,
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
    configuration_sha256: str,
    model_variant: str | None = None,
    analysis_role: str = ANALYSIS_ROLE,
    target_name: str | None = None,
) -> dict[str, Any]:
    configure_runtime_cache()
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    from tafel_refit.bayes import render_posterior_predictive_plot
    from tafel_refit.models import LOG10, harmonic_plateau_log10, latent_plateau_bounds

    target_id = str(manifest_row["target_id"])
    resolved_variant = model_variant or target_id
    resolved_target_name = target_name or TARGET_NAMES[target_id]
    curve_id = str(manifest_row["curve_id"])
    record = int(manifest_row["sample_record"])
    ordered = selected.sort_values(
        ["potential_V", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)
    x = ordered["fit_relative_potential_V"].to_numpy(dtype=float)
    y = ordered["fit_log10_current"].to_numpy(dtype=float)
    potential_rhe = ordered["potential_V"].to_numpy(dtype=float)
    observation_ids = ordered["observation_id"].to_numpy(dtype=int)
    points_hash = dataframe_sha256(ordered)
    paths = _artifact_paths(fit_dir)
    fit_dir.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to overwrite artifacts in {fit_dir}")

    with pm.Model(coords={"observation": observation_ids}):
        x_tensor = pt.as_tensor_variable(x)
        if resolved_variant == TARGET_EQ2_ETA015:
            a = pm.Uniform("a", lower=float(np.min(y) - 6), upper=float(np.max(y) + 2))
            log_m = pm.Uniform("log_m", lower=float(np.log(1)), upper=float(np.log(100)))
            m = pm.Deterministic("m_dec_per_V", pt.exp(log_m))
            slope = pm.Deterministic("slope_mV_dec", 1000.0 / m)
            kinetic = a + m * x_tensor
            plateau_lower, plateau_upper = latent_plateau_bounds(y)
            plateau = pm.Uniform(
                "L_log10_current", lower=plateau_lower, upper=plateau_upper
            )
            mean = kinetic + plateau - pt.logaddexp(
                kinetic * LOG10, plateau * LOG10
            ) / LOG10
            log_sigma = pm.Uniform(
                "log_sigma", lower=float(np.log(0.005)), upper=float(np.log(0.5))
            )
            sigma = pm.Deterministic("sigma_log10_current", pt.exp(log_sigma))
            variable_names = [
                "a",
                "log_m",
                "m_dec_per_V",
                "slope_mV_dec",
                "L_log10_current",
                "log_sigma",
                "sigma_log10_current",
            ]
            pm.StudentT(
                "observed_log10_current",
                nu=4.0,
                mu=mean,
                sigma=sigma,
                observed=y,
                dims="observation",
            )
            prior_boundary_description = "slope outside 12--900 mV_dec"
        elif resolved_variant == TARGET_EQ2_PAPER:
            alpha_upper = float(manifest_row["paper_prior_alpha_upper_per_V"])
            b_upper = float(manifest_row["paper_prior_b_upper_natural"])
            ilim_upper = float(manifest_row["paper_prior_ilim_ratio_upper"])
            alpha = pm.Uniform("alpha_per_V", lower=0.0, upper=alpha_upper)
            b_natural = pm.Uniform("b_log_current_natural", lower=0.0, upper=b_upper)
            ilim_ratio = pm.Uniform("ilim_ratio", lower=0.0, upper=ilim_upper)
            slope = pm.Deterministic("slope_mV_dec", 1000.0 / (LOG10E * alpha))
            mean = LOG10E * (
                b_natural
                + alpha * x_tensor
                - pt.logaddexp(pt.log(ilim_ratio), alpha * x_tensor)
            )
            variable_names = [
                "alpha_per_V",
                "b_log_current_natural",
                "ilim_ratio",
                "slope_mV_dec",
            ]
            pm.Normal(
                "observed_log10_current",
                mu=mean,
                sigma=PAPER_FIXED_SIGMA_LOG10,
                observed=y,
                dims="observation",
            )
            prior_boundary_description = "alpha in outer 5% of data-conditioned prior"
        else:
            raise ValueError(f"Unknown Equation-2 model variant: {resolved_variant}")

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
        posterior_predictive = pm.sample_posterior_predictive(
            idata,
            var_names=["observed_log10_current"],
            random_seed=seed + 1,
            progressbar=progressbar,
        )
        idata.extend(posterior_predictive)

    idata.attrs.update(
        {
            "analysis_role": analysis_role,
            "target_id": target_id,
            "target_name": resolved_target_name,
            "curve_id": curve_id,
            "sample_record": record,
            "model_name": MODEL_NAME,
            "equation": "Limaye_et_al_Equation_2_single_regime_plus_plateau",
            "selected_points_sha256": points_hash,
            "run_configuration_sha256": configuration_sha256,
            "sampling_seed": seed,
        }
    )
    temporary_netcdf = paths["netcdf"].with_suffix(".nc.tmp")
    idata.to_netcdf(temporary_netcdf)
    temporary_netcdf.replace(paths["netcdf"])
    netcdf_hash = sha256_file(paths["netcdf"])
    posterior_summary = az.summary(
        idata, var_names=variable_names, kind="all", hdi_prob=0.95
    )
    _atomic_csv(
        posterior_summary.reset_index(names="variable"), paths["posterior_summary"]
    )

    slope_array = np.asarray(idata.posterior["slope_mV_dec"].values, dtype=float)
    n_chains, n_draws = slope_array.shape[:2]
    slopes = slope_array.reshape(-1)
    slope_hdi = np.asarray(az.hdi(slopes, hdi_prob=0.95), dtype=float)
    slope_draws = pd.DataFrame(
        {
            "target_id": target_id,
            "target_name": resolved_target_name,
            "curve_id": curve_id,
            "sample_record": record,
            "chain": np.repeat(np.arange(n_chains), n_draws),
            "draw": np.tile(np.arange(n_draws), n_chains),
            "slope_mV_dec": slopes,
            "source_netcdf_sha256": netcdf_hash,
            "selected_points_sha256": points_hash,
        }
    )
    _atomic_csv(slope_draws, paths["slope_draws"], compression="gzip")
    rhat = _finite_dataset_values(
        az.rhat(idata, var_names=variable_names, method="rank")
    )
    ess = _finite_dataset_values(
        az.ess(idata, var_names=variable_names, method="bulk")
    )
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    sampler_pass = bool(
        len(rhat)
        and np.max(rhat) <= 1.01
        and len(ess)
        and np.min(ess) >= 400
        and divergences == 0
    )
    if resolved_variant == TARGET_EQ2_ETA015:
        boundary_fraction = float(np.mean((slopes <= 12.0) | (slopes >= 900.0)))
    else:
        alpha_draws = np.asarray(
            idata.posterior["alpha_per_V"].values, dtype=float
        ).reshape(-1)
        alpha_upper = float(manifest_row["paper_prior_alpha_upper_per_V"])
        boundary_fraction = float(
            np.mean((alpha_draws <= 0.05 * alpha_upper) | (alpha_draws >= 0.95 * alpha_upper))
        )
    ppc_array = np.asarray(
        idata.posterior_predictive["observed_log10_current"].values, dtype=float
    ).reshape(-1, len(y))
    ppc_metrics = render_posterior_predictive_plot(
        curve_id=curve_id,
        model_name=MODEL_NAME,
        potential_rhe_V=potential_rhe,
        observed_log10_current=y,
        posterior_predictive_draws=ppc_array,
        output_path=paths["ppc"],
    )
    diagnostics = {
        "created_at_utc": utc_now(),
        "analysis_role": analysis_role,
        "canonical_target_modified": False,
        "target_id": target_id,
        "target_name": resolved_target_name,
        "curve_id": curve_id,
        "sample_record": record,
        "model_name": MODEL_NAME,
        "n_observations": int(len(y)),
        "selected_points_sha256": points_hash,
        "run_configuration_sha256": configuration_sha256,
        "draws_per_chain": draws,
        "tune_per_chain": tune,
        "chains": chains,
        "target_accept": target_accept,
        "sampling_seed": seed,
        "max_r_hat": float(np.max(rhat)) if len(rhat) else None,
        "min_ess_bulk": float(np.min(ess)) if len(ess) else None,
        "divergences": divergences,
        "sampler_diagnostics_pass": sampler_pass,
        "posterior_prior_boundary_fraction": boundary_fraction,
        "posterior_prior_boundary_warning": boundary_fraction > 0.05,
        "prior_boundary_description": prior_boundary_description,
        **ppc_metrics,
    }
    _atomic_json(paths["diagnostics"], diagnostics)
    artifact_hashes = {
        key: sha256_file(paths[key])
        for key in ("netcdf", "posterior_summary", "slope_draws", "diagnostics", "ppc")
    }
    published_paths = _artifact_paths(published_fit_dir)
    result = {
        "fit_status": COMPLETE_STATUS,
        "analysis_role": analysis_role,
        "canonical_target_modified": False,
        "target_id": target_id,
        "target_name": resolved_target_name,
        "curve_id": curve_id,
        "sample_record": record,
        "model_name": MODEL_NAME,
        "selected_points_sha256": points_hash,
        "run_configuration_sha256": configuration_sha256,
        "posterior_slope_median_mV_dec": float(np.median(slopes)),
        "posterior_slope_mean_mV_dec": float(np.mean(slopes)),
        "posterior_slope_hdi_low_mV_dec": float(slope_hdi[0]),
        "posterior_slope_hdi_high_mV_dec": float(slope_hdi[1]),
        "posterior_slope_total_draws": int(len(slopes)),
        "draws_per_chain": draws,
        "tune_per_chain": tune,
        "chains": chains,
        "target_accept": target_accept,
        "sampling_seed": seed,
        "max_r_hat": diagnostics["max_r_hat"],
        "min_ess_bulk": diagnostics["min_ess_bulk"],
        "divergences": divergences,
        "sampler_diagnostics_pass": sampler_pass,
        "posterior_prior_boundary_fraction": boundary_fraction,
        "posterior_prior_boundary_warning": boundary_fraction > 0.05,
        "netcdf_path": str(published_paths["netcdf"]),
        "posterior_summary_path": str(published_paths["posterior_summary"]),
        "slope_draws_path": str(published_paths["slope_draws"]),
        "diagnostics_path": str(published_paths["diagnostics"]),
        "ppc_plot_path": str(published_paths["ppc"]),
        "artifact_sha256": artifact_hashes,
    }
    _atomic_json(paths["result"], result)
    return result


def fit_one(
    selected: pd.DataFrame,
    manifest_row: pd.Series,
    *,
    final_dir: Path,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    progressbar: bool,
    configuration_sha256: str,
    model_variant: str | None = None,
    analysis_role: str = ANALYSIS_ROLE,
    target_name: str | None = None,
) -> dict[str, Any]:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fit directory: {final_dir}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.inprogress-", dir=final_dir.parent)
    )
    try:
        result = _fit_model(
            selected,
            manifest_row,
            fit_dir=staging,
            published_fit_dir=final_dir,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            seed=seed,
            progressbar=progressbar,
            configuration_sha256=configuration_sha256,
            model_variant=model_variant,
            analysis_role=analysis_role,
            target_name=target_name,
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
    requested_targets: set[str],
) -> pd.DataFrame:
    output = manifest.loc[
        manifest["sample_record"].astype(int).isin(requested_records)
        & manifest["target_id"].isin(requested_targets)
    ].copy()
    results = pd.DataFrame(completed + failures)
    if len(results):
        output = output.merge(
            results,
            on=["target_id", "curve_id", "sample_record"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_fit"),
        )
    if "fit_status" not in output:
        output["fit_status"] = "not_requested_or_pending"
    else:
        output["fit_status"] = output["fit_status"].fillna("not_requested_or_pending")
    return output.sort_values(["target_id", "sample_record"]).reset_index(drop=True)


def write_aggregates(output_dir: Path, results: pd.DataFrame) -> None:
    complete = results.loc[results["fit_status"].eq(COMPLETE_STATUS)].copy()
    for target_id in TARGET_IDS:
        target = complete.loc[complete["target_id"].eq(target_id)].copy()
        _atomic_csv(target, output_dir / f"{target_id}_target_summary.csv")
        draw_frames = [
            pd.read_csv(path)
            for path in target["slope_draws_path"].map(Path)
            if path.is_file()
        ]
        if draw_frames:
            _atomic_csv(
                pd.concat(draw_frames, ignore_index=True),
                output_dir / f"{target_id}_posterior_draws.csv.gz",
                compression="gzip",
            )


def write_readme(output_dir: Path, *, smoke: bool) -> None:
    text = f"""# Uniform Equation-2 dual Bayesian targets

Analysis role: `{ANALYSIS_ROLE}`. Canonical target modified: **no**.

## Targets

- `{TARGET_EQ2_ETA015}` / `{TARGET_NAMES[TARGET_EQ2_ETA015]}`: all observed
  equal-potential bins at `eta >= 0.15 V`, project Student-t(nu=4) likelihood,
  estimated noise, and project slope/identifiable-plateau priors.
- `{TARGET_EQ2_PAPER}` / `{TARGET_NAMES[TARGET_EQ2_PAPER]}`: all positive raw
  points accepted by deterministic preprocessing, per-curve log-current shift,
  fixed `sigma=0.10 log10`, and `Uniform(0, 10*theta_star)` priors matching the
  released Limaye et al. `julius` model. Positive bounded optimization is a
  numerical modernization of their unconstrained preconditioning step.

Both use one constant kinetic slope plus the Equation-2 limiting-current term.
There is no manual fit-window selection or common current-span requirement.

Mode: {'smoke' if smoke else 'full'}.

The paper used 10,000 retained draws from each of four chains. This run records
its own explicit sampling budget and is a model/likelihood replication, not a
bitwise reproduction of the original PyMC3 execution.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="+", choices=TARGET_IDS, default=list(TARGET_IDS))
    parser.add_argument("--sample-records", type=int, nargs="+")
    parser.add_argument("--expected-curves", type=int, default=97)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        draws, tune, chains = 30, 30, 2
        requested_records = set(args.sample_records or [3, 25, 79])
    else:
        draws, tune, chains = args.draws, args.tune, args.chains
        requested_records = set(args.sample_records or range(1, args.expected_curves + 1))
    requested_targets = set(args.targets)
    if any(record < 1 or record > args.expected_curves for record in requested_records):
        raise ValueError("Requested record is outside the active range")
    if draws < 1 or tune < 0 or chains < 2:
        raise ValueError("draws>=1, tune>=0, and chains>=2 are required")
    if not 0.5 < args.target_accept < 1.0:
        raise ValueError("target_accept must lie between 0.5 and 1.0")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime_cache()
    binned, raw, summary, source = load_inputs(
        args.run_dir, expected_curves=args.expected_curves
    )
    manifest, selected_points = build_manifest_and_points(binned, raw, summary)
    if len(manifest) != args.expected_curves * len(TARGET_IDS):
        raise ValueError("Dual-target manifest is incomplete")
    if manifest.groupby("target_id")["sample_record"].nunique().to_dict() != {
        target_id: args.expected_curves for target_id in TARGET_IDS
    }:
        raise ValueError("Every target must contain every active record")

    configuration = {
        "analysis_role": ANALYSIS_ROLE,
        "canonical_target_modified": False,
        "model_name": MODEL_NAME,
        "equation": "Limaye_et_al_Equation_2_single_regime_plus_plateau",
        "nature_communications_doi": "10.1038/s41467-021-20924-y",
        "zenodo_doi": "10.5281/zenodo.3995021",
        "target_ids": sorted(requested_targets),
        "target_names": {target: TARGET_NAMES[target] for target in sorted(requested_targets)},
        "sample_records": sorted(requested_records),
        "draws_per_chain": draws,
        "tune_per_chain": tune,
        "chains": chains,
        "target_accept": args.target_accept,
        "base_seed": args.seed,
        "smoke": bool(args.smoke),
        "expected_curves": args.expected_curves,
        "eta015_cutoff_V": ETA_MIN_V,
        "paper_fixed_sigma_log10": PAPER_FIXED_SIGMA_LOG10,
        "paper_prior_multiplier": PAPER_PRIOR_MULTIPLIER,
        "paper_sampling_budget_is_adapted_from_original": True,
        "manifest_sha256": dataframe_sha256(manifest),
        "selected_points_sha256": dataframe_sha256(selected_points),
        "source": source,
    }
    config_path = output_dir / "run_configuration.json"
    manifest_path = output_dir / "target_manifest.csv"
    points_path = output_dir / "selected_target_points.csv.gz"
    if args.resume:
        for path in (config_path, manifest_path, points_path):
            if not path.is_file():
                raise FileNotFoundError(f"--resume requires {path}")
        configuration["target_manifest_file_sha256"] = sha256_file(manifest_path)
        configuration["selected_target_points_file_sha256"] = sha256_file(points_path)
        if json.loads(config_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Resume configuration differs from the existing run")
    else:
        _atomic_csv(manifest, manifest_path)
        _atomic_csv(selected_points, points_path, compression="gzip")
        configuration["target_manifest_file_sha256"] = sha256_file(manifest_path)
        configuration["selected_target_points_file_sha256"] = sha256_file(points_path)
        _atomic_json(config_path, configuration)
    configuration_sha256 = sha256_file(config_path)

    lookup = {
        (str(target_id), str(curve_id)): frame.copy()
        for (target_id, curve_id), frame in selected_points.groupby(
            ["target_id", "curve_id"], sort=False
        )
    }
    requested = manifest.loc[
        manifest["sample_record"].astype(int).isin(requested_records)
        & manifest["target_id"].isin(requested_targets)
    ].sort_values(["target_id", "sample_record"])
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _, row in requested.iterrows():
        target_id = str(row["target_id"])
        curve_id = str(row["curve_id"])
        record = int(row["sample_record"])
        target_offset = 0 if target_id == TARGET_EQ2_ETA015 else 5_000_000
        seed = int(args.seed + target_offset + record * 10_000)
        final_dir = (
            output_dir
            / "bayes"
            / target_id
            / f"record_{record:03d}__curve_{curve_id}"
        )
        frame = lookup[(target_id, curve_id)].sort_values(
            ["potential_V", "observation_id"], kind="mergesort"
        ).reset_index(drop=True)
        points_hash = dataframe_sha256(frame)
        previous = (
            _load_completed_fit(
                final_dir,
                target_id=target_id,
                curve_id=curve_id,
                sample_record=record,
                seed=seed,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=args.target_accept,
                points_sha256=points_hash,
                configuration_sha256=configuration_sha256,
            )
            if args.resume
            else None
        )
        if previous is not None:
            completed.append(previous)
            print(f"resume: {target_id} {curve_id} complete", flush=True)
            continue
        print(f"fit: {target_id} {curve_id} (seed {seed})", flush=True)
        try:
            completed.append(
                fit_one(
                    frame,
                    row,
                    final_dir=final_dir,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=args.target_accept,
                    seed=seed,
                    progressbar=not args.no_progress,
                    configuration_sha256=configuration_sha256,
                )
            )
        except Exception as exc:
            failure = {
                "fit_status": FAILED_STATUS,
                "analysis_role": ANALYSIS_ROLE,
                "target_id": target_id,
                "target_name": TARGET_NAMES[target_id],
                "curve_id": curve_id,
                "sample_record": record,
                "sampling_seed": seed,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failure_dir = output_dir / "failures" / target_id
            failure_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                failure_dir / f"record_{record:03d}__curve_{curve_id}.json", failure
            )
            failures.append(failure)
            print(f"FAILED: {target_id} {curve_id}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
        _atomic_csv(
            merge_results(
                manifest,
                completed,
                failures,
                requested_records,
                requested_targets,
            ),
            output_dir / "target_results.csv",
        )

    results = merge_results(
        manifest, completed, failures, requested_records, requested_targets
    )
    _atomic_csv(results, output_dir / "target_results.csv")
    write_aggregates(output_dir, results)
    write_readme(output_dir, smoke=args.smoke)
    complete = results["fit_status"].eq(COMPLETE_STATUS)
    failed = results["fit_status"].eq(FAILED_STATUS)
    expected_count = len(requested)
    complete_count = int(complete.sum())
    diagnostics_pass_count = int(
        _as_bool(results.loc[complete, "sampler_diagnostics_pass"]).sum()
    )
    structural_pass = bool(complete_count == expected_count and not failed.any())
    status = {
        "completed_at_utc": utc_now(),
        "analysis_role": ANALYSIS_ROLE,
        "canonical_target_modified": False,
        "mode": "smoke" if args.smoke else "full",
        "requested_records": len(requested_records),
        "requested_targets": sorted(requested_targets),
        "requested_fits": expected_count,
        "bayesian_sampling_complete_count": complete_count,
        "bayesian_sampling_failed_count": int(failed.sum()),
        "sampler_diagnostics_pass_count": diagnostics_pass_count,
        "sampler_diagnostics_not_required_for_short_smoke": bool(args.smoke),
        "all_requested_sampling_complete": structural_pass,
        "analysis_ready": bool(
            structural_pass
            and (args.smoke or diagnostics_pass_count == expected_count)
        ),
    }
    _atomic_json(output_dir / "completion_status.json", status)
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    return 0 if status["analysis_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
