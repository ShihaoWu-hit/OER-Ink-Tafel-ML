"""Chain/draw-synchronous target sidecars from selected-model NetCDFs.

This module is executed with the independently locked ``Tafel_refitting``
environment, which supplies ArviZ/xarray for NetCDF reads.  The numerical
target definition itself lives in :mod:`ink_tafel_study.targets` so bootstrap
and posterior draws use the same complete-model inversion followed by local
latent-kinetic-slope semantics.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .provenance import sha256_file
from .targets import fixed_current_slope


FUNCTIONAL_SCHEMA_VERSION = "posterior-functional-target-v1"
FUNCTIONAL_SUFFIX = "_functional_targets_2mA.csv.gz"
SLOPE_SCHEMA_VERSION = "posterior-branch-slope-v1"
SLOPE_SUFFIX = "_posterior_slope_draws.csv.gz"

POSTERIOR_PARAMETER_COLUMNS: dict[str, dict[str, str]] = {
    "linear": {"a": "a", "log_m": "log_m"},
    "paper_single_plateau": {
        "a": "a",
        "log_m": "log_m",
        "L": "L_log10_current",
    },
    "single_background_plateau": {
        "B": "B_log10_background",
        "a": "a",
        "log_m": "log_m",
        "L": "L_log10_current",
    },
    "two_regime_plateau": {
        "a": "a",
        "log_m1": "log_m1",
        "log_m2": "log_m2",
        "c": "change_potential_relative_V",
        "L": "L_log10_current",
    },
    "two_regime_background_plateau": {
        "B": "B_log10_background",
        "a": "a",
        "log_m1": "log_m1",
        "log_m2": "log_m2",
        "c": "change_potential_relative_V",
        "L": "L_log10_current",
    },
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write byte-reproducible gzip CSV (no wall-clock gzip header)."""

    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)


def evaluate_posterior_draw_frame(
    posterior: pd.DataFrame,
    *,
    curve_id: str,
    sample_record: int,
    model_name: str,
    transition_width_V: float,
    potential_min_V: float,
    potential_max_V: float,
    observed_current_min_mA_cm2: float,
    observed_current_max_mA_cm2: float,
    target_current_mA_cm2: float = 2.0,
) -> pd.DataFrame:
    """Evaluate every posterior chain/draw without marginalizing parameters."""

    if model_name not in POSTERIOR_PARAMETER_COLUMNS:
        raise ValueError(f"Unsupported posterior model: {model_name}")
    mapping = POSTERIOR_PARAMETER_COLUMNS[model_name]
    required = {"chain", "draw", *mapping.values()}
    missing = sorted(required - set(posterior.columns))
    if missing:
        raise KeyError(f"Posterior frame is missing variables: {missing}")
    if posterior.duplicated(["chain", "draw"]).any():
        raise ValueError("Posterior chain/draw coordinates must be unique")

    rows: list[dict[str, Any]] = []
    ordered = posterior.sort_values(["chain", "draw"]).reset_index(drop=True)
    for posterior_index, item in enumerate(ordered.to_dict("records")):
        parameters = {
            parameter: float(item[posterior_name])
            for parameter, posterior_name in mapping.items()
        }
        try:
            result = fixed_current_slope(
                model_name=model_name,
                parameters=parameters,
                transition_width_V=float(transition_width_V),
                potential_min_V=float(potential_min_V),
                potential_max_V=float(potential_max_V),
                observed_current_min_mA_cm2=float(observed_current_min_mA_cm2),
                observed_current_max_mA_cm2=float(observed_current_max_mA_cm2),
                target_current_mA_cm2=float(target_current_mA_cm2),
            )
        except (KeyError, TypeError, ValueError, FloatingPointError) as error:
            result = {
                "eligible": False,
                "slope_mV_dec": np.nan,
                "kinetic_slope_mV_dec": np.nan,
                "apparent_slope_mV_dec": np.nan,
                "relative_potential_at_current_V": np.nan,
                "observed_support": False,
                "extrapolated": False,
                "reason": f"functional_evaluation_error:{type(error).__name__}",
            }
        rows.append(
            {
                "functional_schema_version": FUNCTIONAL_SCHEMA_VERSION,
                "curve_id": str(curve_id),
                "sample_record": int(sample_record),
                "model_name": model_name,
                "chain": int(item["chain"]),
                "draw": int(item["draw"]),
                "posterior_draw_index": int(posterior_index),
                "target_current_density_mA_cm2": float(target_current_mA_cm2),
                "relative_potential_at_current_V": result.get(
                    "relative_potential_at_current_V", np.nan
                ),
                "local_slope_mV_dec": result.get("slope_mV_dec", np.nan),
                "kinetic_slope_mV_dec": result.get(
                    "kinetic_slope_mV_dec", np.nan
                ),
                "apparent_slope_mV_dec": result.get(
                    "apparent_slope_mV_dec", np.nan
                ),
                "observed_support": _as_bool(result.get("observed_support")),
                "eligible": _as_bool(result.get("eligible")),
                "extrapolated": _as_bool(result.get("extrapolated")),
                "exclusion_reason": str(result.get("reason", "unknown")),
                "potential_min_V": float(potential_min_V),
                "potential_max_V": float(potential_max_V),
                "observed_current_min_mA_cm2": float(
                    observed_current_min_mA_cm2
                ),
                "observed_current_max_mA_cm2": float(
                    observed_current_max_mA_cm2
                ),
                "transition_width_V": float(transition_width_V),
                "target_slope_semantics": (
                    "local_kinetic_slope_at_complete_model_2mA_potential"
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_posterior_slope_draw_frame(
    posterior: pd.DataFrame,
    *,
    curve_id: str,
    sample_record: int,
    model_name: str,
) -> pd.DataFrame:
    """Return the registered Bayesian low-potential slope for every draw.

    Constant-kinetic-slope models contribute ``slope_mV_dec``.  Two-regime
    models contribute the model-coordinate low-potential branch
    ``slope_low_mV_dec``; the numerical minimum of the two branches is never
    substituted.  The high-potential branch and change-point coordinate are
    retained as diagnostics.
    """

    required = {"chain", "draw"}
    if model_name in {"linear", "paper_single_plateau", "single_background_plateau"}:
        required.add("log_m")
    elif model_name in {"two_regime_plateau", "two_regime_background_plateau"}:
        required.update({"log_m1", "log_m2", "change_potential_relative_V"})
    else:
        raise ValueError(f"Unsupported posterior model: {model_name}")
    missing = sorted(required - set(posterior.columns))
    if missing:
        raise KeyError(f"Posterior frame is missing branch-slope variables: {missing}")
    if posterior.duplicated(["chain", "draw"]).any():
        raise ValueError("Posterior chain/draw coordinates must be unique")

    ordered = posterior.sort_values(["chain", "draw"]).reset_index(drop=True)
    if "log_m" in required:
        primary = 1000.0 / np.exp(ordered["log_m"].to_numpy(dtype=float))
        high = primary.copy()
        change = np.full(len(ordered), np.nan)
        semantics = "single_regime_bayesian_slope"
        source_variable = "slope_mV_dec"
    else:
        primary = 1000.0 / np.exp(ordered["log_m1"].to_numpy(dtype=float))
        high = 1000.0 / np.exp(ordered["log_m2"].to_numpy(dtype=float))
        change = ordered["change_potential_relative_V"].to_numpy(dtype=float)
        semantics = "two_regime_low_potential_branch"
        source_variable = "slope_low_mV_dec"
    eligible = np.isfinite(primary) & (primary > 0)
    return pd.DataFrame(
        {
            "slope_schema_version": SLOPE_SCHEMA_VERSION,
            "curve_id": str(curve_id),
            "sample_record": int(sample_record),
            "model_name": model_name,
            "chain": ordered["chain"].astype(int),
            "draw": ordered["draw"].astype(int),
            "posterior_draw_index": np.arange(len(ordered), dtype=int),
            "target_name": "b_low_potential_mV_dec",
            "primary_slope_mV_dec": primary,
            "slope_low_mV_dec": primary,
            "slope_high_mV_dec": high,
            "change_potential_relative_V": change,
            "eligible": eligible,
            "branch_semantics": semantics,
            "posterior_source_variable": source_variable,
        }
    )


def extract_functional_target_artifact(
    *,
    run_dir: Path,
    diagnostics_path: Path,
    target_current_mA_cm2: float = 2.0,
) -> Path:
    """Create a provenance-bound functional-draw sidecar for one NetCDF."""

    # Import lazily so the lightweight study environment can inspect the
    # module and run unit tests without installing the Bayesian stack.
    try:
        import arviz as az
    except ImportError as error:  # pragma: no cover - exercised in refit env
        raise RuntimeError(
            "Functional posterior extraction requires the Tafel_refitting environment"
        ) from error

    run_dir = Path(run_dir).resolve()
    diagnostics_path = Path(diagnostics_path).resolve()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    evidence_id = diagnostics_path.name.removesuffix("_diagnostics.json")
    netcdf_path = diagnostics_path.parent / f"{evidence_id}.nc"
    if not netcdf_path.is_file():
        raise FileNotFoundError(netcdf_path)
    curve_id = str(diagnostics["curve_id"])
    model_name = str(diagnostics["model_name"])
    if model_name not in POSTERIOR_PARAMETER_COLUMNS:
        raise ValueError(f"Unsupported selected model in diagnostics: {model_name}")

    fit_summary = pd.read_csv(run_dir / "csv" / "curve_fit_summary.csv")
    fit = fit_summary.loc[fit_summary["curve_id"].astype(str).eq(curve_id)]
    if len(fit) != 1:
        raise ValueError(f"Expected one deterministic summary row for {curve_id}")
    fit_row = fit.iloc[0]
    if str(fit_row["selected_model"]) != model_name:
        raise ValueError(
            f"Posterior model {model_name} is not selected model "
            f"{fit_row['selected_model']} for {curve_id}"
        )

    points = pd.read_csv(run_dir / "csv" / "binned_curve_points.csv")
    points = points.loc[points["curve_id"].astype(str).eq(curve_id)]
    if len(points) < 2:
        raise ValueError(f"Insufficient binned support metadata for {curve_id}")
    potential_min = float(points["relative_potential_V"].min())
    potential_max = float(points["relative_potential_V"].max())
    current_min = float(10.0 ** points["log10_current"].min())
    current_max = float(10.0 ** points["log10_current"].max())

    inference_data = az.from_netcdf(netcdf_path)
    posterior_names = list(POSTERIOR_PARAMETER_COLUMNS[model_name].values())
    missing_variables = sorted(set(posterior_names) - set(inference_data.posterior.data_vars))
    if missing_variables:
        raise KeyError(f"NetCDF posterior is missing variables: {missing_variables}")
    posterior = (
        inference_data.posterior[posterior_names]
        .to_dataframe()
        .reset_index()
    )
    draws = evaluate_posterior_draw_frame(
        posterior,
        curve_id=curve_id,
        sample_record=int(fit_row["sample_record"]),
        model_name=model_name,
        transition_width_V=float(fit_row["transition_width_V"]),
        potential_min_V=potential_min,
        potential_max_V=potential_max,
        observed_current_min_mA_cm2=current_min,
        observed_current_max_mA_cm2=current_max,
        target_current_mA_cm2=target_current_mA_cm2,
    )
    draws["evidence_id"] = evidence_id
    draws["source_netcdf_sha256"] = sha256_file(netcdf_path)
    draws["source_diagnostics_sha256"] = sha256_file(diagnostics_path)
    draws["deterministic_workflow_tree_sha256"] = str(
        diagnostics.get("deterministic_run_workflow_tree_sha256", "")
    )
    draws["extractor_sha256"] = sha256_file(Path(__file__).resolve())

    output = diagnostics_path.parent / f"{evidence_id}{FUNCTIONAL_SUFFIX}"
    if not output.is_file():
        _write_gzip_csv(draws, output)

    slope_draws = evaluate_posterior_slope_draw_frame(
        posterior,
        curve_id=curve_id,
        sample_record=int(fit_row["sample_record"]),
        model_name=model_name,
    )
    slope_draws["evidence_id"] = evidence_id
    slope_draws["source_netcdf_sha256"] = sha256_file(netcdf_path)
    slope_draws["source_diagnostics_sha256"] = sha256_file(diagnostics_path)
    slope_draws["deterministic_workflow_tree_sha256"] = str(
        diagnostics.get("deterministic_run_workflow_tree_sha256", "")
    )
    slope_draws["extractor_sha256"] = sha256_file(Path(__file__).resolve())
    slope_output = diagnostics_path.parent / f"{evidence_id}{SLOPE_SUFFIX}"
    if not slope_output.is_file():
        _write_gzip_csv(slope_draws, slope_output)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract chain/draw-synchronous Bayesian branch slopes and the "
            "audit-only 2 mA functional target"
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--target-current", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = extract_functional_target_artifact(
        run_dir=args.run_dir,
        diagnostics_path=args.diagnostics,
        target_current_mA_cm2=args.target_current,
    )
    print(output)


if __name__ == "__main__":
    main()
