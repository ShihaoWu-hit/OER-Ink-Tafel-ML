from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .data import sha256_file
from .models import (
    latent_low_current_floor_bounds,
    latent_plateau_bounds,
    model_spec,
    model_structure_fields,
)
from .pipeline import workflow_hashes


LOG10 = np.log(10.0)
OER_EQUILIBRIUM_POTENTIAL_V_RHE = 1.23


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
        }
    )


def oer_overpotential_V(potential_rhe_V: np.ndarray) -> np.ndarray:
    """Convert an absolute RHE potential axis to OER overpotential."""

    potential = np.asarray(potential_rhe_V, dtype=float)
    if potential.ndim != 1 or not len(potential) or not np.isfinite(potential).all():
        raise ValueError("PPC RHE potentials must be a non-empty finite 1D array")
    return potential - OER_EQUILIBRIUM_POTENTIAL_V_RHE


def render_posterior_predictive_plot(
    *,
    curve_id: str,
    model_name: str,
    potential_rhe_V: np.ndarray,
    observed_log10_current: np.ndarray,
    posterior_predictive_draws: np.ndarray,
    output_path: Path,
    paper_exact: bool = False,
) -> dict[str, float]:
    """Render a PPC against OER overpotential without changing model coordinates."""

    potential = np.asarray(potential_rhe_V, dtype=float)
    y = np.asarray(observed_log10_current, dtype=float)
    ppc_array = np.asarray(posterior_predictive_draws, dtype=float)
    if ppc_array.ndim != 2 or ppc_array.shape[1] != len(y):
        raise ValueError("Posterior-predictive draws must have shape (draw, observation)")
    if len(potential) != len(y):
        raise ValueError("PPC potential and observation arrays must have equal length")

    eta = oer_overpotential_V(potential)
    ppc_median = np.median(ppc_array, axis=0)
    ppc_low = np.quantile(ppc_array, 0.025, axis=0)
    ppc_high = np.quantile(ppc_array, 0.975, axis=0)
    ppc_rmse = float(np.sqrt(np.mean(np.square(y - ppc_median))))
    ppc_coverage = float(np.mean((y >= ppc_low) & (y <= ppc_high)))

    _figure_style()
    order = np.argsort(eta)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.fill_between(
        eta[order],
        ppc_low[order],
        ppc_high[order],
        color="#4472C4",
        alpha=0.22,
        linewidth=0,
        label="95% posterior predictive interval",
    )
    ax.plot(
        eta[order],
        ppc_median[order],
        color="#4472C4",
        linewidth=2.0,
        label="Posterior predictive median",
    )
    ax.scatter(
        eta,
        y,
        s=22,
        facecolor="white",
        edgecolor="#222222",
        linewidth=0.8,
        zorder=3,
        label="Observed bins" if not paper_exact else "Observed raw points",
    )
    ax.set_title(f"{curve_id}: posterior predictive check")
    ax.set_xlabel(r"OER overpotential, $\eta = E_{\mathrm{RHE}} - 1.23$ V (V)")
    ax.set_ylabel(r"$\log_{10}$(current)")
    ax.text(
        0.01,
        0.99,
        f"{model_spec(model_name).display_name}; 95% point coverage = {ppc_coverage:.1%}",
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
        "posterior_predictive_rmse_log10": ppc_rmse,
        "posterior_predictive_95_coverage_fraction": ppc_coverage,
        "overpotential_min_V": float(np.min(eta)),
        "overpotential_max_V": float(np.max(eta)),
    }


def _bayes_mean(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    transition_width_V: float,
) -> tuple[Any, list[str]]:
    spec = model_spec(model_name)
    a = pm.Uniform("a", lower=float(np.min(y) - 6.0), upper=float(np.max(y) + 2.0))
    variable_names = ["a"]
    x_tensor = pt.as_tensor_variable(x)
    if spec.kinetic_structure == "constant":
        log_m = pm.Uniform("log_m", lower=float(np.log(1.0)), upper=float(np.log(100.0)))
        m = pm.Deterministic("m_dec_per_V", pt.exp(log_m))
        pm.Deterministic("slope_mV_dec", 1000.0 / m)
        kinetic = a + m * x_tensor
        variable_names.extend(["log_m", "m_dec_per_V", "slope_mV_dec"])
    elif spec.kinetic_structure == "potential_dependent":
        log_m1 = pm.Uniform(
            "log_m1", lower=float(np.log(1.0)), upper=float(np.log(100.0))
        )
        log_m2 = pm.Uniform(
            "log_m2", lower=float(np.log(1.0)), upper=float(np.log(100.0))
        )
        m1 = pm.Deterministic("m1_dec_per_V", pt.exp(log_m1))
        m2 = pm.Deterministic("m2_dec_per_V", pt.exp(log_m2))
        pm.Deterministic("slope_low_mV_dec", 1000.0 / m1)
        pm.Deterministic("slope_high_mV_dec", 1000.0 / m2)
        c = pm.Uniform(
            "change_potential_relative_V",
            lower=float(np.quantile(x, 0.20)),
            upper=float(np.quantile(x, 0.80)),
        )
        tau = float(transition_width_V)
        integrated_transition = tau * (
            pt.logaddexp(0.0, (x_tensor - c) / tau) - pt.logaddexp(0.0, -c / tau)
        )
        kinetic = a + m1 * x_tensor + (m2 - m1) * integrated_transition
        variable_names.extend(
            [
                "log_m1",
                "log_m2",
                "m1_dec_per_V",
                "m2_dec_per_V",
                "slope_low_mV_dec",
                "slope_high_mV_dec",
                "change_potential_relative_V",
            ]
        )
    else:
        raise ValueError(f"Unknown Bayesian model: {model_name}")

    if not spec.has_limiting_current:
        return kinetic, variable_names
    plateau_lower, plateau_upper = latent_plateau_bounds(y)
    plateau = pm.Uniform(
        "L_log10_current",
        lower=plateau_lower,
        upper=plateau_upper,
    )
    signal = kinetic + plateau - pt.logaddexp(kinetic * LOG10, plateau * LOG10) / LOG10
    variable_names.append("L_log10_current")
    if spec.has_low_current_floor:
        floor_lower, floor_upper = latent_low_current_floor_bounds(y)
        # The variable name is a compatibility key for existing posterior
        # artifacts.  This optional term is a low-current floor sensitivity,
        # not an identified physical background current.
        low_current_floor = pm.Uniform(
            "B_log10_background",
            lower=floor_lower,
            upper=floor_upper,
        )
        variable_names.append("B_log10_background")
        mean = pt.logaddexp(low_current_floor * LOG10, signal * LOG10) / LOG10
        return mean, variable_names
    return signal, variable_names


def run_bayesian_confirmation(
    *,
    run_dir: Path,
    curve_id: str,
    model_name: str | None,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    paper_exact: bool,
    sampling_attempt: int = 1,
    allow_workflow_mismatch: bool = False,
) -> tuple[Path, Path, Path, Path]:
    if sampling_attempt < 1:
        raise ValueError("sampling_attempt must be >= 1")
    run_dir = run_dir.resolve()
    summary_path = run_dir / "csv" / "curve_fit_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing deterministic run summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    deterministic_config_path = run_dir / "model_config.json"
    deterministic_config = (
        json.loads(deterministic_config_path.read_text(encoding="utf-8"))
        if deterministic_config_path.is_file()
        else {}
    )
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"Missing deterministic run status: {status_path}")
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    if run_status.get("status") != "complete":
        raise ValueError(
            "Bayesian confirmation requires run_status='complete'; "
            f"found {run_status.get('status')!r}"
        )

    expected_artifacts = deterministic_config.get("deterministic_artifacts_sha256")
    if not isinstance(expected_artifacts, dict) or not expected_artifacts:
        raise ValueError(
            "Deterministic run has no artifact-hash manifest; rerun the deterministic workflow"
        )

    def artifact_hashes() -> dict[str, str]:
        hashes: dict[str, str] = {}
        for relative_path in sorted(expected_artifacts):
            path = run_dir / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Missing deterministic artifact: {path}")
            hashes[relative_path] = sha256_file(path)
        return hashes

    artifact_hashes_at_start = artifact_hashes()
    artifact_hashes_match_manifest = artifact_hashes_at_start == expected_artifacts
    if not artifact_hashes_match_manifest:
        raise ValueError(
            "A deterministic artifact hash differs from model_config.json; "
            "Bayesian confirmation aborted"
        )
    project_root = Path(__file__).resolve().parents[2]
    _, workflow_tree_hash_at_bayes = workflow_hashes(project_root)
    deterministic_workflow_hash = deterministic_config.get("workflow_tree_sha256")
    workflow_matches_deterministic_run = bool(
        deterministic_workflow_hash
        and deterministic_workflow_hash == workflow_tree_hash_at_bayes
    )
    if not workflow_matches_deterministic_run and not allow_workflow_mismatch:
        raise ValueError(
            "Current workflow hash differs from the deterministic run. "
            "Use --allow-workflow-mismatch only for a sensitivity run."
        )
    selected_row = summary.loc[summary["curve_id"].eq(curve_id)]
    if len(selected_row) != 1:
        raise ValueError(f"curve_id must match exactly one row: {curve_id}")
    if model_name is None:
        model_name = str(selected_row.iloc[0]["selected_model"])
    if model_name == "none":
        raise ValueError(f"Curve {curve_id} was not eligible for deterministic fitting")
    selected_model = str(selected_row.iloc[0]["selected_model"])

    if paper_exact:
        model_name = "paper_single_plateau"
        points = pd.read_csv(run_dir / "csv" / "curve_points_audited.csv")
        points = points.loc[
            points["curve_id"].eq(curve_id) & points["used_for_log_fit"].astype(bool)
        ].copy()
        potential_rhe_V = points["potential_raw"].to_numpy(dtype=float)
        x = potential_rhe_V - np.min(potential_rhe_V)
        y = np.log10(points["current_raw"].to_numpy(dtype=float))
        transition_width = 0.01
        likelihood_name = "iid_normal_fixed_sigma_0p10_raw_points"
    else:
        points = pd.read_csv(run_dir / "csv" / "binned_curve_points.csv")
        points = points.loc[points["curve_id"].eq(curve_id)].copy()
        potential_rhe_V = points["potential_V"].to_numpy(dtype=float)
        x = points["relative_potential_V"].to_numpy(dtype=float)
        y = points["log10_current"].to_numpy(dtype=float)
        transition_width = float(selected_row.iloc[0].get("transition_width_V", 0.01))
        likelihood_name = "student_t_nu4_equal_potential_bins"
    if len(x) < 3:
        raise ValueError(f"Curve {curve_id} has too few points for Bayesian fitting")

    output_dir = run_dir / "bayes"
    output_dir.mkdir(exist_ok=True)
    suffix = "paper_exact" if paper_exact else model_name
    suffix = f"{suffix}_seed{seed}_attempt{sampling_attempt:02d}"
    netcdf_path = output_dir / f"{curve_id}_{suffix}.nc"
    summary_csv = output_dir / f"{curve_id}_{suffix}_posterior_summary.csv"
    diagnostics_json = output_dir / f"{curve_id}_{suffix}_diagnostics.json"
    ppc_plot = output_dir / f"{curve_id}_{suffix}_posterior_predictive.png"
    existing_outputs = [
        path
        for path in (netcdf_path, summary_csv, diagnostics_json, ppc_plot)
        if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing Bayesian output(s): "
            + ", ".join(str(path) for path in existing_outputs)
        )

    with pm.Model():
        mean, variable_names = _bayes_mean(model_name, x, y, transition_width)
        if paper_exact:
            pm.Normal("observed_log10_current", mu=mean, sigma=0.10, observed=y)
        else:
            log_sigma = pm.Uniform(
                "log_sigma", lower=float(np.log(0.005)), upper=float(np.log(0.5))
            )
            sigma = pm.Deterministic("sigma_log10_current", pt.exp(log_sigma))
            variable_names.extend(["log_sigma", "sigma_log10_current"])
            pm.StudentT("observed_log10_current", nu=4.0, mu=mean, sigma=sigma, observed=y)
        inference_data = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
        posterior_predictive = pm.sample_posterior_predictive(
            inference_data,
            var_names=["observed_log10_current"],
            random_seed=seed + 1,
        )
        inference_data.extend(posterior_predictive)

    inference_data.to_netcdf(netcdf_path)
    posterior_summary = az.summary(
        inference_data,
        var_names=variable_names,
        kind="all",
        hdi_prob=0.95,
    )
    posterior_summary.to_csv(summary_csv)

    r_hat_dataset = az.rhat(
        inference_data, var_names=variable_names, method="rank"
    )
    ess_bulk_dataset = az.ess(
        inference_data, var_names=variable_names, method="bulk"
    )

    def finite_dataset_values(dataset: Any) -> np.ndarray:
        arrays = [
            np.asarray(dataset[name].values, dtype=float).reshape(-1)
            for name in dataset.data_vars
        ]
        values = np.concatenate(arrays) if arrays else np.array([], dtype=float)
        return values[np.isfinite(values)]

    r_hat_values = finite_dataset_values(r_hat_dataset)
    ess_bulk_values = finite_dataset_values(ess_bulk_dataset)
    divergences = int(inference_data.sample_stats["diverging"].sum().item())
    ppc_array = np.asarray(
        inference_data.posterior_predictive["observed_log10_current"].values,
        dtype=float,
    ).reshape(-1, len(y))
    ppc_median = np.median(ppc_array, axis=0)
    ppc_low = np.quantile(ppc_array, 0.025, axis=0)
    ppc_high = np.quantile(ppc_array, 0.975, axis=0)
    ppc_rmse = float(np.sqrt(np.mean(np.square(y - ppc_median))))
    ppc_coverage = float(np.mean((y >= ppc_low) & (y <= ppc_high)))
    sampler_diagnostics_pass = bool(
        len(r_hat_values)
        and float(np.max(r_hat_values)) <= 1.01
        and len(ess_bulk_values)
        and float(np.min(ess_bulk_values)) >= 400
        and divergences == 0
    )
    _, workflow_tree_hash_at_end = workflow_hashes(project_root)
    artifact_hashes_at_end = artifact_hashes()
    workflow_changed_during_bayes = bool(
        workflow_tree_hash_at_end != workflow_tree_hash_at_bayes
    )
    deterministic_artifacts_changed_during_bayes = bool(
        artifact_hashes_at_end != artifact_hashes_at_start
    )
    model_matches_deterministic_selection = bool(
        not paper_exact and model_name == selected_model
    )
    formal_confirmation_provenance_eligible = bool(
        workflow_matches_deterministic_run
        and not workflow_changed_during_bayes
        and artifact_hashes_match_manifest
        and not deterministic_artifacts_changed_during_bayes
        and not deterministic_config.get("input_hash_verification_bypassed", False)
        and model_matches_deterministic_selection
    )
    formal_confirmation_ready_for_ppc_review = bool(
        formal_confirmation_provenance_eligible and sampler_diagnostics_pass
    )
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "curve_id": curve_id,
        "model_name": model_name,
        "model_display_name": model_spec(model_name).display_name,
        **model_structure_fields(model_name),
        "likelihood": likelihood_name,
        "draws_per_chain": draws,
        "tune_per_chain": tune,
        "chains": chains,
        "target_accept": target_accept,
        "sampling_seed": seed,
        "posterior_predictive_seed": seed + 1,
        "sampling_attempt": int(sampling_attempt),
        "max_r_hat": float(np.max(r_hat_values)) if len(r_hat_values) else None,
        "min_ess_bulk": float(np.min(ess_bulk_values))
        if len(ess_bulk_values)
        else None,
        "divergences": divergences,
        "posterior_predictive_rmse_log10": ppc_rmse,
        "posterior_predictive_95_coverage_fraction": ppc_coverage,
        "ppc_x_axis": "oer_overpotential_V",
        "ppc_x_axis_formula": "eta_V = potential_V_vs_RHE - 1.23",
        "oer_equilibrium_potential_V_vs_RHE": OER_EQUILIBRIUM_POTENTIAL_V_RHE,
        "sampler_diagnostics_pass": sampler_diagnostics_pass,
        "model_fit_review_required": True,
        "deterministic_run_status": run_status.get("status"),
        "deterministic_artifacts_sha256": artifact_hashes_at_start,
        "deterministic_artifacts_match_manifest": artifact_hashes_match_manifest,
        "deterministic_artifacts_changed_during_bayes": (
            deterministic_artifacts_changed_during_bayes
        ),
        "deterministic_run_workflow_tree_sha256": deterministic_workflow_hash,
        "workflow_tree_sha256_at_bayes": workflow_tree_hash_at_bayes,
        "workflow_tree_sha256_at_bayes_end": workflow_tree_hash_at_end,
        "workflow_matches_deterministic_run": workflow_matches_deterministic_run,
        "workflow_changed_during_bayes": workflow_changed_during_bayes,
        "allow_workflow_mismatch": bool(allow_workflow_mismatch),
        "model_matches_deterministic_selection": model_matches_deterministic_selection,
        "formal_confirmation_provenance_eligible": (
            formal_confirmation_provenance_eligible
        ),
        "formal_confirmation_ready_for_ppc_review": (
            formal_confirmation_ready_for_ppc_review
        ),
        "confirmation_scope": (
            "same_workflow_selected_model_check"
            if formal_confirmation_provenance_eligible
            else "sensitivity_only_not_formal_confirmation"
        ),
        "note": (
            "Sampler pass covers R-hat, bulk ESS and divergences only. "
            "Bayesian confirmation uses binned Student-t data unless paper_exact is true; "
            "inspect the posterior-predictive plot before judging model adequacy."
        ),
    }
    diagnostics_json.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    render_posterior_predictive_plot(
        curve_id=curve_id,
        model_name=model_name,
        potential_rhe_V=potential_rhe_V,
        observed_log10_current=y,
        posterior_predictive_draws=ppc_array,
        output_path=ppc_plot,
        paper_exact=paper_exact,
    )
    return netcdf_path, summary_csv, diagnostics_json, ppc_plot
