"""Discovery and human-review binding for selected-model Bayesian evidence.

Bayesian sampling is intentionally performed in an isolated copy of the
deterministic refit run.  This module makes those sidecars discoverable by the
active study without copying anything back into the deterministic run or the
read-only legacy archive.  Human PPC decisions are stored in the study run and
are cryptographically bound to the four files that were reviewed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .provenance import sha256_file


EVIDENCE_SUFFIXES = {
    "diagnostics": "_diagnostics.json",
    "posterior_summary": "_posterior_summary.csv",
    "ppc_plot": "_posterior_predictive.png",
    "netcdf": ".nc",
    "functional_draws": "_functional_targets_2mA.csv.gz",
    "slope_draws": "_posterior_slope_draws.csv.gz",
}

REVIEW_COLUMNS = [
    "curve_id",
    "sample_record",
    "selected_model",
    "evidence_id",
    "diagnostics_sha256",
    "netcdf_sha256",
    "posterior_summary_sha256",
    "ppc_plot_sha256",
    "slope_draws_sha256",
    "functional_draws_sha256",
    "ppc_approved",
    "ppc_reviewer",
    "ppc_review_date",
    "ppc_review_note",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _summary_stat(path: Path, variable: str, column: str) -> float:
    """Read one ArviZ summary statistic without importing ArviZ/xarray."""

    if not path.is_file():
        return np.nan
    frame = pd.read_csv(path, index_col=0)
    if variable not in frame.index or column not in frame.columns:
        return np.nan
    return _finite_float(frame.loc[variable, column])


def _summary_interval_columns(path: Path) -> tuple[str | None, str | None]:
    if not path.is_file():
        return None, None
    columns = list(pd.read_csv(path, nrows=0).columns)
    low = next((name for name in ("hdi_2.5%", "hdi_3%") if name in columns), None)
    high = next((name for name in ("hdi_97.5%", "hdi_97%") if name in columns), None)
    return low, high


def _posterior_slope_stats(path: Path, model_name: str) -> dict[str, Any]:
    variable = (
        "slope_low_mV_dec"
        if model_name in {"two_regime_plateau", "two_regime_background_plateau"}
        else "slope_mV_dec"
    )
    low_column, high_column = _summary_interval_columns(path)
    return {
        "posterior_slope_variable": variable,
        "posterior_slope_mean_mV_dec": _summary_stat(path, variable, "mean"),
        "posterior_slope_interval_low_mV_dec": (
            _summary_stat(path, variable, low_column) if low_column else np.nan
        ),
        "posterior_slope_interval_high_mV_dec": (
            _summary_stat(path, variable, high_column) if high_column else np.nan
        ),
    }


def _posterior_slope_draw_stats(
    path: Path,
    *,
    diagnostics: dict[str, Any],
    diagnostics_sha256: str,
    netcdf_sha256: str,
) -> dict[str, Any]:
    defaults = {
        "slope_draws_artifact_valid": False,
        "slope_draws_validation": "missing_posterior_slope_draws",
        "posterior_slope_total_draws": 0,
        "posterior_slope_eligible_draws": 0,
        "posterior_slope_median_mV_dec": np.nan,
        "posterior_slope_draw_mean_mV_dec": np.nan,
        "posterior_slope_draw_hdi_low_mV_dec": np.nan,
        "posterior_slope_draw_hdi_high_mV_dec": np.nan,
    }
    if not path.is_file():
        return defaults
    try:
        frame = pd.read_csv(path)
    except Exception as error:  # pragma: no cover - defensive corruption path
        return {**defaults, "slope_draws_validation": f"unreadable:{type(error).__name__}"}
    required = {
        "slope_schema_version",
        "curve_id",
        "model_name",
        "chain",
        "draw",
        "posterior_draw_index",
        "target_name",
        "primary_slope_mV_dec",
        "eligible",
        "branch_semantics",
        "posterior_source_variable",
        "evidence_id",
        "source_netcdf_sha256",
        "source_diagnostics_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return {**defaults, "slope_draws_validation": f"missing_columns:{'|'.join(missing)}"}
    expected_draws = int(diagnostics.get("draws_per_chain", 0) or 0) * int(
        diagnostics.get("chains", 0) or 0
    )
    curve_id = str(diagnostics.get("curve_id", ""))
    model_name = str(diagnostics.get("model_name", ""))
    evidence_id = path.name.removesuffix("_posterior_slope_draws.csv.gz")
    eligible = frame["eligible"].map(_as_bool)
    slopes = pd.to_numeric(frame["primary_slope_mV_dec"], errors="coerce")
    expected_variable = (
        "slope_low_mV_dec"
        if model_name in {"two_regime_plateau", "two_regime_background_plateau"}
        else "slope_mV_dec"
    )
    valid = bool(
        len(frame) == expected_draws
        and expected_draws > 0
        and not frame.duplicated(["chain", "draw"]).any()
        and frame["slope_schema_version"].astype(str).eq(
            "posterior-branch-slope-v1"
        ).all()
        and frame["curve_id"].astype(str).eq(curve_id).all()
        and frame["model_name"].astype(str).eq(model_name).all()
        and frame["target_name"].astype(str).eq(
            "b_low_potential_mV_dec"
        ).all()
        and frame["posterior_source_variable"].astype(str).eq(
            expected_variable
        ).all()
        and frame["evidence_id"].astype(str).eq(evidence_id).all()
        and frame["source_netcdf_sha256"].astype(str).eq(netcdf_sha256).all()
        and frame["source_diagnostics_sha256"].astype(str).eq(
            diagnostics_sha256
        ).all()
        and eligible.all()
        and np.isfinite(slopes).all()
        and (slopes > 0).all()
    )
    values = slopes.loc[eligible & np.isfinite(slopes) & (slopes > 0)].to_numpy()
    low, high = _empirical_hdi(values)
    return {
        "slope_draws_artifact_valid": valid,
        "slope_draws_validation": (
            "chain_draw_synchronous_hash_bound_low_potential_branch"
            if valid
            else "posterior_slope_draw_contract_failed"
        ),
        "posterior_slope_total_draws": int(len(frame)),
        "posterior_slope_eligible_draws": int(len(values)),
        "posterior_slope_median_mV_dec": (
            float(np.median(values)) if len(values) else np.nan
        ),
        "posterior_slope_draw_mean_mV_dec": (
            float(np.mean(values)) if len(values) else np.nan
        ),
        "posterior_slope_draw_hdi_low_mV_dec": low,
        "posterior_slope_draw_hdi_high_mV_dec": high,
    }


def _empirical_hdi(values: np.ndarray, probability: float = 0.95) -> tuple[float, float]:
    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not len(finite):
        return np.nan, np.nan
    width = max(1, int(np.ceil(probability * len(finite))))
    if width >= len(finite):
        return float(finite[0]), float(finite[-1])
    spans = finite[width - 1 :] - finite[: len(finite) - width + 1]
    start = int(np.argmin(spans))
    return float(finite[start]), float(finite[start + width - 1])


def _functional_target_stats(
    path: Path,
    *,
    diagnostics: dict[str, Any],
    diagnostics_sha256: str,
    netcdf_sha256: str,
) -> dict[str, Any]:
    defaults = {
        "functional_target_artifact_valid": False,
        "functional_target_validation": "missing_functional_target_artifact",
        "functional_target_total_draws": 0,
        "functional_target_eligible_draws": 0,
        "functional_target_support_fraction": 0.0,
        "functional_target_supported": False,
        "functional_target_median_mV_dec": np.nan,
        "functional_target_mean_mV_dec": np.nan,
        "functional_target_hdi_low_mV_dec": np.nan,
        "functional_target_hdi_high_mV_dec": np.nan,
    }
    if not path.is_file():
        return defaults
    try:
        frame = pd.read_csv(path)
    except Exception as error:  # pragma: no cover - defensive corruption path
        return {
            **defaults,
            "functional_target_validation": f"unreadable:{type(error).__name__}",
        }
    required = {
        "functional_schema_version",
        "curve_id",
        "model_name",
        "chain",
        "draw",
        "target_current_density_mA_cm2",
        "local_slope_mV_dec",
        "kinetic_slope_mV_dec",
        "apparent_slope_mV_dec",
        "target_slope_semantics",
        "eligible",
        "extrapolated",
        "evidence_id",
        "source_netcdf_sha256",
        "source_diagnostics_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return {
            **defaults,
            "functional_target_validation": f"missing_columns:{'|'.join(missing)}",
        }
    expected_draws = int(diagnostics.get("draws_per_chain", 0) or 0) * int(
        diagnostics.get("chains", 0) or 0
    )
    curve_id = str(diagnostics.get("curve_id", ""))
    model_name = str(diagnostics.get("model_name", ""))
    evidence_id = path.name.removesuffix("_functional_targets_2mA.csv.gz")
    eligible = frame["eligible"].map(_as_bool)
    extrapolated = frame["extrapolated"].map(_as_bool)
    slopes = pd.to_numeric(frame["local_slope_mV_dec"], errors="coerce")
    kinetic_slopes = pd.to_numeric(
        frame["kinetic_slope_mV_dec"], errors="coerce"
    )
    apparent_slopes = pd.to_numeric(
        frame["apparent_slope_mV_dec"], errors="coerce"
    )
    valid = bool(
        len(frame) == expected_draws
        and expected_draws > 0
        and not frame.duplicated(["chain", "draw"]).any()
        and frame["functional_schema_version"].astype(str).eq(
            "posterior-functional-target-v1"
        ).all()
        and frame["curve_id"].astype(str).eq(curve_id).all()
        and frame["model_name"].astype(str).eq(model_name).all()
        and frame["evidence_id"].astype(str).eq(evidence_id).all()
        and frame["target_slope_semantics"].astype(str).eq(
            "local_kinetic_slope_at_complete_model_2mA_potential"
        ).all()
        and np.allclose(
            pd.to_numeric(frame["target_current_density_mA_cm2"], errors="coerce"),
            2.0,
        )
        and frame["source_netcdf_sha256"].astype(str).eq(netcdf_sha256).all()
        and frame["source_diagnostics_sha256"].astype(str).eq(
            diagnostics_sha256
        ).all()
        and not extrapolated.any()
        and np.isfinite(slopes.loc[eligible]).all()
        and (slopes.loc[eligible] > 0).all()
        and np.allclose(
            slopes.loc[eligible], kinetic_slopes.loc[eligible], rtol=1e-10, atol=1e-10
        )
        and np.isfinite(apparent_slopes.loc[eligible]).all()
        and (
            apparent_slopes.loc[eligible] + 1e-9
            >= kinetic_slopes.loc[eligible]
        ).all()
    )
    eligible_values = slopes.loc[eligible & np.isfinite(slopes) & (slopes > 0)].to_numpy()
    support_fraction = float(len(eligible_values) / len(frame)) if len(frame) else 0.0
    low, high = _empirical_hdi(eligible_values)
    return {
        "functional_target_artifact_valid": valid,
        "functional_target_validation": (
            "chain_draw_synchronous_hash_bound_no_extrapolation"
            if valid
            else "functional_target_contract_failed"
        ),
        "functional_target_total_draws": int(len(frame)),
        "functional_target_eligible_draws": int(len(eligible_values)),
        "functional_target_support_fraction": support_fraction,
        "functional_target_supported": bool(valid and support_fraction >= 0.80),
        "functional_target_median_mV_dec": (
            float(np.median(eligible_values)) if len(eligible_values) else np.nan
        ),
        "functional_target_mean_mV_dec": (
            float(np.mean(eligible_values)) if len(eligible_values) else np.nan
        ),
        "functional_target_hdi_low_mV_dec": low,
        "functional_target_hdi_high_mV_dec": high,
    }


def discover_bayesian_evidence(
    bayes_dirs: Iterable[Path],
    *,
    current_deterministic_config: Path | None = None,
) -> pd.DataFrame:
    """Return one row per diagnostics prefix with companion-file hashes.

    ``formal_sampling_candidate`` is deliberately strict.  It requires the
    current deterministic artifact/workflow hashes, the selected-model
    provenance flags written by ``Tafel_refitting``, four-chain production
    sampling, and every required companion artifact, including the registered
    branch-slope draws.  The exact-2 mA functional sidecar is audit-only and
    does not gate formal slope evidence.  Smoke draws remain useful for plumbing
    checks but can never satisfy this flag.
    """

    expected_artifacts: dict[str, str] | None = None
    expected_workflow: str | None = None
    if current_deterministic_config is not None and Path(current_deterministic_config).is_file():
        payload = json.loads(Path(current_deterministic_config).read_text(encoding="utf-8"))
        candidate = payload.get("deterministic_artifacts_sha256")
        expected_artifacts = candidate if isinstance(candidate, dict) else None
        expected_workflow = payload.get("workflow_tree_sha256")

    rows: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    for raw_dir in bayes_dirs:
        directory = Path(raw_dir).resolve()
        if directory in seen_dirs or not directory.is_dir():
            continue
        seen_dirs.add(directory)
        for diagnostics_path in sorted(directory.glob("*_diagnostics.json")):
            try:
                diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                rows.append(
                    {
                        "curve_id": "",
                        "model_name": "",
                        "evidence_id": diagnostics_path.name.removesuffix("_diagnostics.json"),
                        "evidence_dir": str(directory),
                        "diagnostics_path": str(diagnostics_path),
                        "discovery_error": f"invalid_diagnostics_json:{type(exc).__name__}",
                        "formal_sampling_candidate": False,
                    }
                )
                continue

            evidence_id = diagnostics_path.name.removesuffix("_diagnostics.json")
            companions = {
                "diagnostics": diagnostics_path,
                "posterior_summary": directory / f"{evidence_id}_posterior_summary.csv",
                "ppc_plot": directory / f"{evidence_id}_posterior_predictive.png",
                "netcdf": directory / f"{evidence_id}.nc",
                "functional_draws": (
                    directory / f"{evidence_id}_functional_targets_2mA.csv.gz"
                ),
                "slope_draws": (
                    directory / f"{evidence_id}_posterior_slope_draws.csv.gz"
                ),
            }
            required_companions = {
                name: path
                for name, path in companions.items()
                if name != "functional_draws"
            }
            companion_complete = all(
                path.is_file() for path in required_companions.values()
            )
            companion_hashes = {
                name: sha256_file(path) if path.is_file() else ""
                for name, path in companions.items()
            }
            artifacts_match = bool(
                expected_artifacts
                and diagnostics.get("deterministic_artifacts_sha256") == expected_artifacts
            )
            workflow_match = bool(
                expected_workflow
                and diagnostics.get("deterministic_run_workflow_tree_sha256")
                == expected_workflow
                and diagnostics.get("workflow_tree_sha256_at_bayes") == expected_workflow
                and diagnostics.get("workflow_tree_sha256_at_bayes_end") == expected_workflow
            )
            production_depth = bool(
                int(diagnostics.get("draws_per_chain", 0) or 0) >= 2000
                and int(diagnostics.get("tune_per_chain", 0) or 0) >= 1000
                and int(diagnostics.get("chains", 0) or 0) >= 4
            )
            provenance_eligible = bool(
                _as_bool(diagnostics.get("formal_confirmation_provenance_eligible"))
                and _as_bool(diagnostics.get("model_matches_deterministic_selection"))
                and not _as_bool(diagnostics.get("allow_workflow_mismatch"))
            )
            sampler_pass = _as_bool(diagnostics.get("sampler_diagnostics_pass"))
            model_name = str(diagnostics.get("model_name", ""))
            summary_path = companions["posterior_summary"]
            functional_stats = _functional_target_stats(
                companions["functional_draws"],
                diagnostics=diagnostics,
                diagnostics_sha256=companion_hashes["diagnostics"],
                netcdf_sha256=companion_hashes["netcdf"],
            )
            slope_draw_stats = _posterior_slope_draw_stats(
                companions["slope_draws"],
                diagnostics=diagnostics,
                diagnostics_sha256=companion_hashes["diagnostics"],
                netcdf_sha256=companion_hashes["netcdf"],
            )
            row: dict[str, Any] = {
                "curve_id": str(diagnostics.get("curve_id", "")),
                "model_name": model_name,
                "evidence_id": evidence_id,
                "evidence_dir": str(directory),
                "created_at_utc": diagnostics.get("created_at_utc", ""),
                "draws_per_chain": int(diagnostics.get("draws_per_chain", 0) or 0),
                "tune_per_chain": int(diagnostics.get("tune_per_chain", 0) or 0),
                "chains": int(diagnostics.get("chains", 0) or 0),
                "sampling_attempt": int(diagnostics.get("sampling_attempt", 1) or 1),
                "sampling_seed": int(diagnostics.get("sampling_seed", 0) or 0),
                "posterior_predictive_seed": int(
                    diagnostics.get("posterior_predictive_seed", 0) or 0
                ),
                "max_r_hat": _finite_float(diagnostics.get("max_r_hat")),
                "min_ess_bulk": _finite_float(diagnostics.get("min_ess_bulk")),
                "divergences": _finite_float(diagnostics.get("divergences")),
                "sampler_diagnostics_pass": sampler_pass,
                "companion_artifacts_complete": companion_complete,
                "deterministic_artifacts_match_current": artifacts_match,
                "workflow_hash_matches_current": workflow_match,
                "production_sampling_depth": production_depth,
                "formal_confirmation_provenance_eligible": provenance_eligible,
                "formal_sampling_candidate": bool(
                    companion_complete
                    and slope_draw_stats["slope_draws_artifact_valid"]
                    and artifacts_match
                    and workflow_match
                    and production_depth
                    and provenance_eligible
                    and sampler_pass
                ),
                "discovery_error": "",
            }
            for name, path in companions.items():
                row[f"{name}_path"] = str(path.resolve()) if path.is_file() else ""
                row[f"{name}_sha256"] = companion_hashes[name]
            row.update(_posterior_slope_stats(summary_path, model_name))
            row.update(slope_draw_stats)
            row.update(functional_stats)
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "curve_id",
                "model_name",
                "evidence_id",
                "evidence_dir",
                "formal_sampling_candidate",
            ]
        )
    return frame.sort_values(
        ["curve_id", "formal_sampling_candidate", "created_at_utc"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def _blank_review_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "curve_id": item.get("curve_id", ""),
        "sample_record": item.get("sample_record", ""),
        "selected_model": item.get("selected_model", item.get("model_name", "")),
        "evidence_id": item.get("evidence_id", ""),
        "diagnostics_sha256": item.get("diagnostics_sha256", ""),
        "netcdf_sha256": item.get("netcdf_sha256", ""),
        "posterior_summary_sha256": item.get("posterior_summary_sha256", ""),
        "ppc_plot_sha256": item.get("ppc_plot_sha256", ""),
        "slope_draws_sha256": item.get("slope_draws_sha256", ""),
        "functional_draws_sha256": item.get("functional_draws_sha256", ""),
        "ppc_approved": "",
        "ppc_reviewer": "",
        "ppc_review_date": "",
        "ppc_review_note": "",
    }


def sync_manual_ppc_review(
    review_path: Path,
    evidence: pd.DataFrame,
    *,
    sample_records: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add reviewable evidence without erasing completed human decisions.

    A prior decision is retained only when every evidence hash is unchanged.
    If a new posterior replaces any reviewed artifact, the human fields are
    cleared so the new PPC must be inspected explicitly.
    """

    path = Path(review_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path, dtype=str).fillna("") if path.is_file() else pd.DataFrame()
    existing_map = (
        existing.set_index("curve_id").to_dict("index")
        if not existing.empty and "curve_id" in existing.columns
        else {}
    )
    records = sample_records or {}
    formal = evidence.loc[evidence["formal_sampling_candidate"].map(_as_bool)].copy()
    formal = formal.drop_duplicates("curve_id", keep="first")
    rows: list[dict[str, Any]] = []
    hash_columns = [
        "diagnostics_sha256",
        "netcdf_sha256",
        "posterior_summary_sha256",
        "ppc_plot_sha256",
        "slope_draws_sha256",
        "functional_draws_sha256",
    ]
    for item in formal.to_dict("records"):
        item["sample_record"] = records.get(str(item.get("curve_id", "")), "")
        blank = _blank_review_row(item)
        previous = existing_map.get(str(item.get("curve_id", "")))
        hashes_match = bool(
            previous
            and previous.get("evidence_id", "") == blank["evidence_id"]
            and all(previous.get(column, "") == blank[column] for column in hash_columns)
        )
        if hashes_match:
            for column in ("ppc_approved", "ppc_reviewer", "ppc_review_date", "ppc_review_note"):
                blank[column] = previous.get(column, "")
        rows.append(blank)
    result = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    result.to_csv(path, index=False)
    return result


def validate_bound_ppc_review(
    review: dict[str, Any], evidence: dict[str, Any]
) -> tuple[bool, str]:
    """Validate a named approval and its binding to the current evidence."""

    if not review:
        return False, "missing_manual_review"
    if not _as_bool(review.get("ppc_approved")):
        return False, "ppc_not_approved"
    if not str(review.get("ppc_reviewer", "")).strip():
        return False, "missing_named_reviewer"
    if not str(review.get("ppc_review_date", "")).strip():
        return False, "missing_review_date"
    try:
        datetime.fromisoformat(str(review["ppc_review_date"]).strip())
    except ValueError:
        return False, "invalid_review_date_use_iso8601"
    if review.get("evidence_id", "") != evidence.get("evidence_id", ""):
        return False, "review_evidence_id_mismatch"
    for column in (
        "diagnostics_sha256",
        "netcdf_sha256",
        "posterior_summary_sha256",
        "ppc_plot_sha256",
        "slope_draws_sha256",
    ):
        if not review.get(column) or review.get(column) != evidence.get(column):
            return False, f"review_{column}_mismatch"
    return True, "named_ppc_approval_bound_to_current_artifacts"


def write_evidence_inventory(path: Path, evidence: pd.DataFrame) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(destination, index=False)
    return destination
