from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .provenance import sha256_file


SUPPORT_SCHEMA_VERSION = "current-support-from-refit-v2"


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def derive_current_support_manifest(
    fit_summary: pd.DataFrame,
    sample_candidates: pd.DataFrame,
    binned_points: pd.DataFrame,
) -> pd.DataFrame:
    required_fit = {
        "curve_id",
        "sample_record",
        "fit_status",
        "fit_eligible",
        "selected_model",
        "potential_dependent_supported",
        "current_unit_raw",
        "transition_width_V",
    }
    required_candidates = {"sample_record", "n_curves", "refit_candidate_status"}
    required_points = {
        "curve_id",
        "potential_V",
        "relative_potential_V",
        "log10_current",
        "n_raw_points",
    }
    for frame, required, name in (
        (fit_summary, required_fit, "fit summary"),
        (sample_candidates, required_candidates, "sample candidates"),
        (binned_points, required_points, "binned points"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    if fit_summary["curve_id"].astype(str).duplicated().any():
        raise ValueError("fit summary contains duplicated curve_id")
    candidate_lookup = sample_candidates.set_index("sample_record")
    point_groups = {
        str(curve_id): group.sort_values("relative_potential_V", kind="stable")
        for curve_id, group in binned_points.groupby("curve_id", sort=False)
    }

    rows: list[dict[str, Any]] = []
    for fit in fit_summary.sort_values(["sample_record", "curve_id"]).to_dict("records"):
        curve_id = str(fit["curve_id"])
        sample_record = int(fit["sample_record"])
        if sample_record not in candidate_lookup.index:
            raise ValueError(f"Missing sample candidate row for sample {sample_record}")
        candidate = candidate_lookup.loc[sample_record]
        if isinstance(candidate, pd.DataFrame):
            raise ValueError(f"Duplicated sample candidate rows for sample {sample_record}")
        points = point_groups.get(curve_id, pd.DataFrame())
        fit_success = bool(
            _as_bool(fit.get("fit_eligible")) and str(fit.get("fit_status")) == "success"
        )
        multiple_variants = int(candidate["n_curves"]) != 1
        potential_dependent = _as_bool(fit.get("potential_dependent_supported"))
        candidate_status = str(candidate.get("refit_candidate_status", ""))
        if multiple_variants:
            cohort = "variant_resolution_required"
        elif potential_dependent:
            cohort = "potential_dependent"
        elif candidate_status == "eligible_single_curve":
            cohort = "reliable_single_regime"
        else:
            cohort = "target_model_not_supported"

        raw_unit = str(fit.get("current_unit_raw", ""))
        if raw_unit == "unknown_current_density":
            interpreted_unit = "mA cm^-2"
            scale_group = "current_density_mA_cm^-2_user_confirmed"
            interpretation_source = "user_confirmed_2026-07-15_not_encoded_in_workbook"
            unit_in_scope = True
        else:
            interpreted_unit = raw_unit
            scale_group = "mass_specific_or_unrecognized"
            interpretation_source = "source_header"
            unit_in_scope = False

        cohort_in_scope = cohort in {
            "potential_dependent",
            "reliable_single_regime",
        }
        model_supported = bool(cohort_in_scope and fit_success and len(points))
        included = bool(model_supported and unit_in_scope)
        if included:
            exclusion_reason = "included_primary"
        elif multiple_variants:
            exclusion_reason = "multiple_curve_variants_require_resolution"
        elif not fit_success:
            exclusion_reason = "fit_not_successful"
        elif not cohort_in_scope:
            exclusion_reason = "target_model_not_supported"
        elif not unit_in_scope:
            exclusion_reason = "incompatible_current_unit"
        else:
            exclusion_reason = "missing_binned_current_support"

        log_min = float(points["log10_current"].min()) if len(points) else np.nan
        log_max = float(points["log10_current"].max()) if len(points) else np.nan
        rows.append(
            {
                "support_schema_version": SUPPORT_SCHEMA_VERSION,
                "curve_id": curve_id,
                "sample_record": sample_record,
                "cohort": cohort,
                "selected_model": str(fit.get("selected_model", "")),
                "potential_dependent_supported": potential_dependent,
                "refit_candidate_status": candidate_status,
                "current_header_raw": (
                    ""
                    if pd.isna(fit.get("current_header_raw", ""))
                    else str(fit.get("current_header_raw", ""))
                ),
                "current_unit_raw": raw_unit,
                "current_scale_group": scale_group,
                "current_unit_interpreted": interpreted_unit,
                "current_unit_interpretation_source": interpretation_source,
                "log10_current_min_binned": log_min,
                "log10_current_max_binned": log_max,
                "current_min_source_numeric": 10.0**log_min if np.isfinite(log_min) else np.nan,
                "current_max_source_numeric": 10.0**log_max if np.isfinite(log_max) else np.nan,
                "full_log_current_span_dec": log_max - log_min
                if np.isfinite(log_min) and np.isfinite(log_max)
                else np.nan,
                "transition_width_V": fit.get("transition_width_V", np.nan),
                "n_effective_bins_recomputed": int(len(points)),
                "n_raw_points_in_bins": int(points["n_raw_points"].sum())
                if len(points)
                else 0,
                "fit_success": fit_success,
                "cohort_in_scope": cohort_in_scope,
                "unit_in_scope": unit_in_scope,
                "model_supported": model_supported,
                "included_primary": included,
                "exclusion_reason": exclusion_reason,
            }
        )
    return pd.DataFrame(rows)


def build_current_support_artifacts(
    *,
    refit_run: Path,
    legacy_grid_summary: Path,
    output_dir: Path,
) -> dict[str, Path]:
    refit_run = Path(refit_run).resolve()
    csv_dir = refit_run / "csv"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_inclusion_manifest.csv"
    grid_path = output_dir / "current_conditioned_target_summary.csv"
    status_path = output_dir / "build_status.json"
    if any(path.exists() for path in (manifest_path, grid_path, status_path)):
        raise FileExistsError(
            f"Refusing to overwrite frozen current-support artifacts in {output_dir}"
        )

    inputs = {
        "fit_summary": csv_dir / "curve_fit_summary.csv",
        "sample_candidates": csv_dir / "sample_refit_candidates.csv",
        "binned_points": csv_dir / "binned_curve_points.csv",
        "curve_manifest": csv_dir / "curve_manifest.csv",
        "legacy_grid_summary": Path(legacy_grid_summary).resolve(),
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing current-support input(s): " + ", ".join(missing))
    fit_summary = pd.read_csv(inputs["fit_summary"])
    curve_manifest = pd.read_csv(inputs["curve_manifest"])
    manifest_columns = {"curve_id", "current_header_raw"}
    missing_manifest_columns = sorted(manifest_columns - set(curve_manifest.columns))
    if missing_manifest_columns:
        raise ValueError(
            "curve manifest is missing required columns: "
            f"{missing_manifest_columns}"
        )
    curve_headers = curve_manifest[["curve_id", "current_header_raw"]].copy()
    curve_headers["curve_id"] = curve_headers["curve_id"].astype(str)
    if curve_headers["curve_id"].duplicated().any():
        raise ValueError("curve manifest contains duplicated curve_id")
    fit_summary["curve_id"] = fit_summary["curve_id"].astype(str)
    fit_summary = fit_summary.drop(columns=["current_header_raw"], errors="ignore").merge(
        curve_headers,
        on="curve_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_headers = fit_summary.loc[
        fit_summary["_merge"].ne("both"), "curve_id"
    ].tolist()
    if missing_headers:
        raise ValueError(
            "curve manifest is missing fit-summary curve(s): "
            + ", ".join(missing_headers)
        )
    fit_summary = fit_summary.drop(columns="_merge")
    manifest = derive_current_support_manifest(
        fit_summary,
        pd.read_csv(inputs["sample_candidates"]),
        pd.read_csv(inputs["binned_points"]),
    )
    manifest.to_csv(manifest_path, index=False)
    shutil.copy2(inputs["legacy_grid_summary"], grid_path)
    status = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "support_schema_version": SUPPORT_SCHEMA_VERSION,
        "source_refit_run": str(refit_run),
        "inputs_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "outputs_sha256": {
            "sample_inclusion_manifest": sha256_file(manifest_path),
            "legacy_grid_summary_copy": sha256_file(grid_path),
        },
        "rows": int(len(manifest)),
        "included_primary": int(manifest["included_primary"].sum()),
        "cohort_counts": {
            str(key): int(value)
            for key, value in manifest["cohort"].value_counts().to_dict().items()
        },
    }
    status_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "manifest": manifest_path,
        "legacy_grid_summary": grid_path,
        "status": status_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build active current-support inputs from one deterministic refit run."
    )
    parser.add_argument("--refit-run", type=Path, required=True)
    parser.add_argument("--legacy-grid-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = build_current_support_artifacts(
        refit_run=args.refit_run,
        legacy_grid_summary=args.legacy_grid_summary,
        output_dir=args.output_dir,
    )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
