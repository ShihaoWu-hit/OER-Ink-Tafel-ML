#!/usr/bin/env python3
"""Publish the calibrated Bayesian local-line target and archive superseded targets.

The formal target run must already be complete and calibration-passing.  This
publisher stages a new canonical dataset, PPC package, target-comparison
figures, and exploratory CSVs; verifies them; archives the prior active target
state and candidate target runs; then renames the stages into place.  It never
marks a PPC as approved.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr


REPO = Path(__file__).resolve().parents[2]
TARGET = "b_bayesian_local_line_mV_dec"
POLICY = "bayesian_local_line_eta015_strict_then_pava_fallback_v2"
SOURCE_POLICY = POLICY
EXPECTED_RECORDS = 97
EXPECTED_ELIGIBLE = 97
EXPECTED_DRAWS_PER_ELIGIBLE_CURVE = 8000
UNSUPPORTED_RECORDS: set[int] = set()
NOT_APPLICABLE_DRAW_HASH = "not_applicable_no_supported_local_line_window"
FALLBACK_RECORDS = {23, 74}
ARCHIVE_NAME = "target_state_pre_bayesian_local_line_flag_repair_20260723"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_gzip(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False, lineterminator="\n")


def flat_manifest(directory: Path) -> None:
    files = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha(path)})
    write_json(directory / "artifact_manifest.json", {"files": files})


def recursive_manifest(directory: Path, name: str, extra: dict[str, Any] | None = None) -> None:
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == name:
            continue
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha(path),
            }
        )
    payload = {"created_at_utc": now(), "files": files}
    if extra:
        payload.update(extra)
    write_json(directory / name, payload)


def curve_draw_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def build_identity(summary: pd.DataFrame, source_draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.sort_values("sample_record").itertuples(index=False):
        curve = str(row.curve_id)
        subset = source_draws.loc[source_draws["curve_id"].astype(str).eq(curve)].copy()
        eligible = bool(row.target_structurally_eligible)
        expected = EXPECTED_DRAWS_PER_ELIGIBLE_CURVE if eligible else 0
        if len(subset) != expected:
            raise ValueError(f"Expected {expected} pooled draws for {curve}, found {len(subset)}")
        draws_hash = curve_draw_hash(subset) if eligible else NOT_APPLICABLE_DRAW_HASH
        evidence_id = f"{curve}_{POLICY}_{draws_hash[:12]}"
        rows.append(
            {
                "sample_record": int(row.sample_record),
                "curve_id": curve,
                "evidence_id": evidence_id,
                "slope_draws_sha256": draws_hash,
            }
        )
    return pd.DataFrame(rows)


def _rewrite_json_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_json_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_json_strings(item) for item in value]
    if value == SOURCE_POLICY:
        return POLICY
    if value == "b_first_supported_apparent_tafel_mV_dec":
        return TARGET
    return value


def build_formal_stage(source: Path, stage: Path) -> None:
    """Create a named, self-contained formal evidence package from the frozen run."""

    shutil.copytree(source, stage)
    summary = pd.read_csv(stage / "target_summary.csv")
    summary["policy_id"] = POLICY
    summary["target_name"] = TARGET
    summary["target_role"] = "primary_ml_response_pending_human_ppc"
    summary["quantization_aware_fallback_used"] = (
        summary["quantization_aware_fallback_used"].fillna(False).astype(bool)
    )
    summary.to_csv(stage / "target_summary.csv", index=False)

    draws = pd.read_csv(stage / "posterior_target_draws.csv.gz", low_memory=False)
    draws["target_name"] = TARGET
    write_gzip(draws, stage / "posterior_target_draws.csv.gz")

    review = pd.read_csv(stage / "ppc_review.csv")
    review["policy_id"] = POLICY
    review["target_name"] = TARGET
    for old_path in sorted((stage / "ppc").glob("*contiguous_increase_regional_ppc.png")):
        new_path = old_path.with_name(
            old_path.name.replace(
                "contiguous_increase_regional_ppc", "bayesian_local_line_regional_ppc"
            )
        )
        old_path.rename(new_path)
    review["ppc_image"] = review["ppc_image"].map(
        lambda value: str(value).replace(
            "contiguous_increase_regional_ppc", "bayesian_local_line_regional_ppc"
        )
    )
    review["ppc_sha256"] = review["ppc_image"].map(lambda value: sha(stage / str(value)))
    review.to_csv(stage / "ppc_review.csv", index=False)

    for json_name in ("qa.json", "run_status.json", "run_config.json"):
        path = stage / json_name
        payload = _rewrite_json_strings(json.loads(path.read_text(encoding="utf-8")))
        payload["candidate_only_not_published"] = False
        payload["publication_boundary"] = "formal_evidence_source_pending_named_human_ppc"
        payload["source_candidate_run"] = str(source.resolve())
        if json_name in {"qa.json", "run_status.json"}:
            payload["status"] = "formal_evidence_complete_pending_named_human_ppc"
            payload["formal_evaluation_ready"] = False
            payload["formal_evaluation_blockers"] = ["named_ppc_review_incomplete"]
        write_json(path, payload)

    figures = stage / "figures"
    for path in sorted(figures.glob("contiguous_increase_bayesian_tafel_slope_kde*")):
        path.rename(
            path.with_name(
                path.name.replace(
                    "contiguous_increase_bayesian_tafel_slope_kde",
                    "bayesian_local_line_tafel_slope_kde",
                )
            )
        )
    metadata = figures / "bayesian_local_line_tafel_slope_kde_metadata.json"
    if metadata.is_file():
        write_json(
            metadata,
            _rewrite_json_strings(json.loads(metadata.read_text(encoding="utf-8"))),
        )
    (stage / "README.md").write_text(
        f"""# Formal Bayesian local-line target evidence

This is the formal evidence source for `{TARGET}` under policy `{POLICY}`.
All 97 canonical records have 8,000 pooled posterior draws. Ninety-five use
the strict observed-progression selector. Records 23 and 74 use the registered
quantization-aware PAVA fallback after the strict selector finds no supported
window.

The robust Bayesian local-linear model propagates parameter, digitisation, and
window-selection uncertainty through an equal-weight mixture over 16
coordinate-perturbed selections. The fallback keeps the same local-line
estimand and does not substitute a selected-model or Equation-2 target. The
method does not use slope magnitude, literature slopes, or ML performance to
choose a window. Formal ML remains blocked until all 97 PPCs receive a new
named human review.
""",
        encoding="utf-8",
    )
    recursive_manifest(
        stage,
        "artifact_manifest.json",
        {
            "target_name": TARGET,
            "policy_id": POLICY,
            "record_count": EXPECTED_RECORDS,
            "structurally_eligible_count": EXPECTED_ELIGIBLE,
            "quantization_aware_fallback_records": sorted(FALLBACK_RECORDS),
        },
    )


def build_ppc_stage(formal_dir: Path, stage: Path, summary: pd.DataFrame, identity: pd.DataFrame) -> None:
    stage.mkdir(parents=True)
    images = sorted((formal_dir / "ppc").glob("*.png"))
    if len(images) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} formal PPC images, found {len(images)}")
    for image in images:
        name = image.name.replace("contiguous_increase_regional_ppc", "bayesian_local_line_regional_ppc")
        shutil.copy2(image, stage / name)
    review = pd.read_csv(formal_dir / "ppc_review.csv")
    review["policy_id"] = POLICY
    review["target_name"] = TARGET
    review["ppc_image"] = review["ppc_image"].map(
        lambda value: Path(str(value)).name.replace(
            "contiguous_increase_regional_ppc", "bayesian_local_line_regional_ppc"
        )
    )
    review = review.drop(columns=["ppc_sha256"], errors="ignore").merge(
        identity, on=["sample_record", "curve_id"], validate="one_to_one"
    )
    review["target_value_mV_dec"] = review["curve_id"].map(
        summary.set_index("curve_id")["posterior_target_median_mV_dec"]
    )
    review["ppc_sha256"] = review["ppc_image"].map(lambda value: sha(stage / value))
    review["ppc_approved"] = ""
    review["ppc_reviewer"] = ""
    review["ppc_review_date"] = ""
    review["ppc_review_note"] = ""
    review.to_csv(stage / "ppc_review.csv", index=False)
    review[["sample_record", "curve_id", "ppc_image", "target_structurally_eligible"]].to_csv(
        stage / "PPC_SELECTION.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "target_name": TARGET,
                "total_records": EXPECTED_RECORDS,
                "structurally_eligible": EXPECTED_ELIGIBLE,
                "quantization_aware_fallback": len(FALLBACK_RECORDS),
                "human_review_complete": 0,
                "approved": 0,
                "status": "pending_named_human_review",
            }
        ]
    ).to_csv(stage / "STATUS.csv", index=False)
    (stage / "README.md").write_text(
        f"""# Bayesian local-line PPC review

This is the only active PPC package. It contains 97 regional diagnostics for
`{TARGET}`. Records 1--97 remain in the review table and all require a new
named human review. Records 23 and 74 use the registered quantization-aware
PAVA fallback after the strict observed-progression selector finds no
supported window.

The fitted line is a robust Bayesian local-linear apparent Tafel slope. Shaded
window-boundary bands and the posterior interval include the 16-replicate
coordinate/window-selection mixture. The fallback fits the same Bayesian
local-line estimand; it does not substitute a selected-model or Equation-2
value. All review fields are intentionally blank; no old PPC decision was
transferred. Edit only `ppc_approved`,
`ppc_reviewer`, `ppc_review_date`, and `ppc_review_note`.
""",
        encoding="utf-8",
    )
    recursive_manifest(
        stage,
        "publication_manifest.json",
        {
            "source_formal_target": "Data/03_outputs/tafel_refitting/outputs/20260723_bayesian_local_line_formal_target_v2",
            "target_name": TARGET,
            "human_review_transferred": False,
        },
    )


def build_canonical_stage(
    base: Path,
    stage: Path,
    formal_dir: Path,
    ppc_stage: Path,
    summary: pd.DataFrame,
    source_draws: pd.DataFrame,
    identity: pd.DataFrame,
) -> None:
    shutil.copytree(base, stage)
    master = pd.read_csv(stage / "master.csv", low_memory=False)
    joined = summary.merge(identity, on=["sample_record", "curve_id"], validate="one_to_one")
    by_curve = joined.set_index("curve_id")
    master["y_tafel_bayesian_primary_mV_dec"] = master["curve_id"].map(by_curve["posterior_target_median_mV_dec"])
    master["y_tafel_bayesian_primary_interval_low_mV_dec"] = master["curve_id"].map(by_curve["posterior_target_hdi_low_mV_dec"])
    master["y_tafel_bayesian_primary_interval_high_mV_dec"] = master["curve_id"].map(by_curve["posterior_target_hdi_high_mV_dec"])
    master["bayesian_primary_provisional_eligible"] = master["curve_id"].map(
        by_curve["target_structurally_eligible"]
    ).fillna(False).astype(bool)
    master["bayesian_primary_formal_eligible"] = False
    master["named_ppc_review_complete"] = False
    master["formal_nuts_ppc_approved"] = False
    master["posterior_slope_variable"] = "bayesian_local_line_slope_mV_dec"
    master["posterior_slope_median_mV_dec"] = master["y_tafel_bayesian_primary_mV_dec"]
    master["posterior_slope_draw_hdi_low_mV_dec"] = master["y_tafel_bayesian_primary_interval_low_mV_dec"]
    master["posterior_slope_draw_hdi_high_mV_dec"] = master["y_tafel_bayesian_primary_interval_high_mV_dec"]
    master["target_fit_eta_min_V"] = 0.15
    master["target_fit_retained_start_eta_V"] = master["curve_id"].map(by_curve["window_start_eta_median_V"])
    master["target_fit_retained_end_eta_V"] = master["curve_id"].map(by_curve["window_end_eta_median_V"])
    master["target_fit_retained_n_bins"] = np.nan
    master["target_fit_retained_log_current_span_dec"] = np.nan
    master = master.drop(
        columns=[
            "first_supported_selection_success_fraction",
            "first_supported_window_start_eta_low_V",
            "first_supported_window_start_eta_high_V",
            "first_supported_window_end_eta_low_V",
            "first_supported_window_end_eta_high_V",
        ],
        errors="ignore",
    )
    master["local_line_selection_success_fraction"] = master["curve_id"].map(by_curve["selection_success_fraction"])
    master["local_line_window_start_eta_low_V"] = master["curve_id"].map(by_curve["window_start_eta_low_V"])
    master["local_line_window_start_eta_high_V"] = master["curve_id"].map(by_curve["window_start_eta_high_V"])
    master["local_line_window_end_eta_low_V"] = master["curve_id"].map(by_curve["window_end_eta_low_V"])
    master["local_line_window_end_eta_high_V"] = master["curve_id"].map(by_curve["window_end_eta_high_V"])
    master["local_line_quantization_aware_fallback_used"] = (
        master["curve_id"]
        .map(by_curve["quantization_aware_fallback_used"])
        .fillna(False)
        .astype(bool)
    )
    master.to_csv(stage / "master.csv", index=False)
    master.to_parquet(stage / "master.parquet", index=False)

    old_registry = pd.read_csv(base / "target_registry.csv", low_memory=False)
    template = old_registry.loc[old_registry["target_role"].eq("primary")].copy()
    if len(template) != 97:
        template = old_registry.loc[old_registry["target_name"].eq("b_low_potential_mV_dec")].copy()
    registry = template.sort_values("record_id").reset_index(drop=True)
    registry["target_name"] = TARGET
    registry["target_role"] = "primary"
    registry["selected_model"] = "common_bayesian_local_linear"
    registry["posterior_evidence_id"] = registry["curve_id"].map(identity.set_index("curve_id")["evidence_id"])
    registry["slope_draws_sha256"] = registry["curve_id"].map(identity.set_index("curve_id")["slope_draws_sha256"])
    registry["target_value_mV_dec"] = registry["curve_id"].map(by_curve["posterior_target_median_mV_dec"])
    registry["target_interval_low_mV_dec"] = registry["curve_id"].map(by_curve["posterior_target_hdi_low_mV_dec"])
    registry["target_interval_high_mV_dec"] = registry["curve_id"].map(by_curve["posterior_target_hdi_high_mV_dec"])
    registry["target_uncertainty_source"] = "bayesian_parameter_plus_digitisation_plus_window_selection_mixture_hdi"
    eligible_map = by_curve["target_structurally_eligible"].astype(bool)
    registry["target_structurally_eligible"] = registry["curve_id"].map(eligible_map).fillna(False).astype(bool)
    registry["target_eligible"] = registry["target_structurally_eligible"]
    registry["posterior_target_supported"] = registry["target_structurally_eligible"]
    registry["formal_target_approved"] = False
    registry["ppc_status"] = np.where(
        registry["target_structurally_eligible"],
        "pending_named_review",
        "not_applicable_structurally_unsupported",
    )
    registry["exclusion_reason"] = np.where(
        registry["target_structurally_eligible"],
        "eligible_pending_named_human_ppc_review",
        "no_supported_strict_contiguous_increase_local_line_window",
    )
    registry["posterior_target_gap_reason"] = np.where(
        registry["target_structurally_eligible"],
        "complete",
        "no_supported_strict_contiguous_increase_local_line_window",
    )
    registry["cohort"] = "bayesian_local_line"
    registry["branch_rule"] = "not_applicable_common_local_linear_estimand"
    registry["branch_selection_rule"] = (
        "strict_contiguous_increase_else_quantization_aware_pava_fallback"
    )
    registry["branch_semantics"] = "apparent_deta_dlog10_current_in_lowest_supported_strictly_increasing_region"
    registry["calculation_method"] = POLICY
    registry["target_current_density_mA_cm2"] = np.nan
    registry["observed_support"] = registry["target_structurally_eligible"]
    registry["extrapolated"] = False
    registry["interpolation_used"] = False
    registry["target_fit_eta_min_V"] = 0.15
    registry["target_fit_retained_start_eta_V"] = registry["curve_id"].map(by_curve["window_start_eta_median_V"])
    registry["target_fit_retained_end_eta_V"] = registry["curve_id"].map(by_curve["window_end_eta_median_V"])
    registry["quantization_aware_fallback_used"] = (
        registry["curve_id"]
        .map(by_curve["quantization_aware_fallback_used"])
        .fillna(False)
        .astype(bool)
    )
    registry.to_csv(stage / "target_registry.csv", index=False)

    canonical_draws = source_draws.copy()
    canonical_draws = canonical_draws.rename(columns={"mixture_draw": "posterior_draw_index", "slope_mV_dec": "primary_slope_mV_dec"})
    canonical_draws["slope_schema_version"] = "posterior-bayesian-local-line-window-mixture-v1"
    canonical_draws["target_name"] = TARGET
    canonical_draws["eligible"] = True
    canonical_draws["posterior_source_variable"] = "bayesian_local_line_slope_mV_dec"
    canonical_draws["branch_semantics"] = "common_local_linear_apparent_slope"
    canonical_draws["evidence_id"] = canonical_draws["curve_id"].map(identity.set_index("curve_id")["evidence_id"])
    front = ["slope_schema_version", "curve_id", "sample_record", "posterior_draw_index", "target_name", "primary_slope_mV_dec"]
    canonical_draws = canonical_draws[front + [c for c in canonical_draws if c not in front]]
    write_gzip(canonical_draws, stage / "posterior_slope_draws.csv.gz")

    review = pd.read_csv(ppc_stage / "ppc_review.csv")
    audit = joined.merge(
        review[["curve_id", "ppc_image", "ppc_sha256"]], on="curve_id", validate="one_to_one"
    )
    audit["quantization_aware_fallback_used"] = (
        audit["quantization_aware_fallback_used"].fillna(False).astype(bool)
    )
    audit["record_id"] = audit["sample_record"]
    audit["evidence_schema_version"] = "bayesian-local-line-primary-v1"
    audit["policy_id"] = POLICY
    audit["target_name"] = TARGET
    audit["selected_model"] = "common_bayesian_local_linear"
    audit["posterior_slope_variable"] = "bayesian_local_line_slope_mV_dec"
    audit["posterior_slope_median_mV_dec"] = audit["posterior_target_median_mV_dec"]
    audit["posterior_slope_draw_hdi_low_mV_dec"] = audit["posterior_target_hdi_low_mV_dec"]
    audit["posterior_slope_draw_hdi_high_mV_dec"] = audit["posterior_target_hdi_high_mV_dec"]
    audit["posterior_slope_eligible_draws"] = np.where(
        audit["target_structurally_eligible"], EXPECTED_DRAWS_PER_ELIGIBLE_CURVE, 0
    )
    audit["slope_draws_artifact_valid"] = audit["target_structurally_eligible"]
    audit["ppc_plot_sha256"] = audit["ppc_sha256"]
    audit["sampler_diagnostics_pass"] = audit["target_structurally_eligible"]
    audit["calibration_pass"] = True
    audit["selection_uncertainty_propagated"] = True
    audit["named_ppc_review_complete"] = False
    audit["formal_target_approved"] = False
    audit["next_action"] = np.where(
        audit["target_structurally_eligible"],
        "perform_ppc_review",
        "retain_record_with_missing_target_no_fallback",
    )
    audit.to_csv(stage / "bayesian_audit.csv", index=False)
    joined.to_csv(stage / "bayesian_targets.csv", index=False)
    joined.to_csv(stage / "low_potential_target_audit.csv", index=False)

    ml = pd.read_csv(stage / "ml_ready.csv", low_memory=False)
    drop = [c for c in ml.columns if c.startswith("y_tafel_bayesian")]
    ml = ml.drop(columns=drop, errors="ignore")
    ml["target_source"] = "calibrated_bayesian_local_line_eta015_contiguous_increase_window_mixture"
    ml["y_tafel_bayesian_local_line_mV_dec"] = ml["curve_id"].map(by_curve["posterior_target_median_mV_dec"])
    ml["y_tafel_bayesian_local_line_interval_low_mV_dec"] = ml["curve_id"].map(by_curve["posterior_target_hdi_low_mV_dec"])
    ml["y_tafel_bayesian_local_line_interval_high_mV_dec"] = ml["curve_id"].map(by_curve["posterior_target_hdi_high_mV_dec"])
    ml.to_csv(stage / "ml_ready.csv", index=False)

    qa = json.loads((formal_dir / "qa.json").read_text(encoding="utf-8"))
    qa.update(
        {
            "status": "canonical_complete_pending_named_human_ppc_review",
            "target_name": TARGET,
            "canonical_record_count": 97,
            "canonical_target_structurally_eligible_count": EXPECTED_ELIGIBLE,
            "canonical_quantization_aware_fallback_count": len(FALLBACK_RECORDS),
            "quantization_aware_fallback_records": sorted(FALLBACK_RECORDS),
            "formal_evaluation_ready": False,
            "formal_evaluation_blockers": ["named_ppc_review_incomplete"],
            "archived_target_methods_excluded_from_active_registry": True,
        }
    )
    write_json(stage / "qa.json", qa)
    (stage / "README.md").write_text(
        f"""# Canonical 97-record ML dataset — Bayesian local-line target

The only active ML response is `{TARGET}`. The registry and data tables contain
97 calibrated values. Ninety-five use strict observed-current progression.
Records 23 and 74 use the registered quantization-aware PAVA fallback after
the strict selector finds no supported interval.

The target is a robust Bayesian local-linear apparent slope in the
lowest-potential supported interval. Its posterior mixture propagates
digitisation and window-selection uncertainty. The fallback keeps the same
Bayesian local-line estimand. Formal ML remains blocked until all 97 regional
PPCs receive a hash-bound named human review.
Superseded targets are retained under `archive/` and are not active.
""",
        encoding="utf-8",
    )
    flat_manifest(stage)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 14,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def build_results_stage(stage: Path, summary: pd.DataFrame, master: pd.DataFrame) -> None:
    stage.mkdir(parents=True)
    values = summary.sort_values("sample_record").copy()
    values["policy_id"] = POLICY
    values["target_name"] = TARGET
    values["target_role"] = "primary_ml_response_pending_human_ppc"
    values.to_csv(stage / "bayesian_local_line_target.csv", index=False)
    style()
    slopes = pd.to_numeric(values["posterior_target_median_mV_dec"], errors="coerce").dropna().to_numpy()
    if len(slopes) != EXPECTED_ELIGIBLE or np.any(slopes <= 0):
        raise ValueError("Bayesian local-line KDE requires 97 positive targets")
    log_slopes = np.log10(slopes)
    log_grid = np.linspace(log_slopes.min() - 0.12, log_slopes.max() + 0.12, 500)
    density = gaussian_kde(log_slopes)(log_grid)
    grid = 10**log_grid
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    ax.fill_between(grid, density, color="#3775BA", alpha=0.28)
    ax.plot(grid, density, color="#0F4D92", lw=2.5)
    ax.axvline(np.median(slopes), color="#B64342", ls="--", lw=2, label=f"Median = {np.median(slopes):.1f}")
    ax.set_xscale("log")
    normal_ticks = [30, 40, 60, 100, 200, 400, 600]
    ax.xaxis.set_major_locator(FixedLocator(normal_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Bayesian apparent Tafel slope (mV dec$^{-1}$)")
    ax.set_ylabel("Density in log-slope space")
    ax.set_title(f"Bayesian local-line target (n = {EXPECTED_ELIGIBLE})")
    ax.legend()
    fig.tight_layout(pad=1.6)
    fig.savefig(stage / "bayesian_local_line_tafel_slope_kde.png", dpi=300)
    fig.savefig(stage / "bayesian_local_line_tafel_slope_kde.pdf")
    plt.close(fig)

    literature = pd.to_numeric(
        master.get("tafel_slope_article_reported_only_mV_dec"), errors="coerce"
    )
    assignment = pd.Series(
        np.where(literature.notna(), "record_specific_article_reported", ""),
        index=master.index,
        dtype="object",
    )
    # Records 42 and 43 are two retained curves from legacy record 42.  The
    # article reports one 95 mV dec-1 comparator for that shared source record,
    # so bind it to both curve records rather than treating Record 43 as a
    # thirtieth unreported record.
    shared_reported = (
        master.loc[
            literature.notna(),
            ["legacy_record_id", "tafel_slope_article_reported_only_mV_dec"],
        ]
        .drop_duplicates("legacy_record_id")
        .set_index("legacy_record_id")["tafel_slope_article_reported_only_mV_dec"]
    )
    inherited_mask = literature.isna() & master["legacy_record_id"].isin(
        shared_reported.index
    )
    literature.loc[inherited_mask] = master.loc[
        inherited_mask, "legacy_record_id"
    ].map(shared_reported)
    assignment.loc[inherited_mask] = "shared_legacy_record_article_reported"
    comparison = pd.DataFrame(
        {
            "record_id": master["record_id"],
            "curve_id": master["curve_id"],
            "literature_reported_mV_dec": literature,
            "reported_value_assignment": assignment,
        }
    ).merge(
        values[["sample_record", "posterior_target_median_mV_dec"]],
        left_on="record_id",
        right_on="sample_record",
        validate="one_to_one",
    )
    comparison = comparison.loc[
        comparison["literature_reported_mV_dec"].notna()
    ].copy()
    comparison["relative_error_fraction"] = (
        comparison["posterior_target_median_mV_dec"]
        - comparison["literature_reported_mV_dec"]
    ).abs() / comparison["literature_reported_mV_dec"].abs()
    comparison["within_10pct_relative_error"] = comparison[
        "relative_error_fraction"
    ].le(0.10)
    comparison["within_20pct_relative_error"] = comparison[
        "relative_error_fraction"
    ].le(0.20)
    comparison.to_csv(stage / "bayesian_local_line_vs_literature.csv", index=False)
    mask = comparison["posterior_target_median_mV_dec"].notna()
    x = comparison.loc[mask, "literature_reported_mV_dec"].to_numpy()
    y = comparison.loc[mask, "posterior_target_median_mV_dec"].to_numpy()
    rho = float(spearmanr(x, y).statistic)
    limit = max(np.max(x), np.max(y)) * 1.06
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    error_grid = np.linspace(0.0, limit, 500)
    outer_band = ax.fill_between(
        error_grid,
        0.80 * error_grid,
        1.20 * error_grid,
        color="#D95F59",
        alpha=0.12,
        linewidth=0,
        label="±20% relative error",
        zorder=1,
    )
    inner_band = ax.fill_between(
        error_grid,
        0.90 * error_grid,
        1.10 * error_grid,
        color="#C43C39",
        alpha=0.22,
        linewidth=0,
        label="±10% relative error",
        zorder=2,
    )
    agreement_line, = ax.plot(
        [0, limit],
        [0, limit],
        color="#B22222",
        ls="-",
        lw=2.2,
        label="Perfect agreement",
        zorder=3,
    )
    points = ax.scatter(
        comparison.loc[mask, "literature_reported_mV_dec"],
        comparison.loc[mask, "posterior_target_median_mV_dec"],
        s=32,
        alpha=0.82,
        color="#0F4D92",
        edgecolor="white",
        linewidth=0.35,
        marker="o",
        label=f"Bayesian local-line refit (n = {int(mask.sum())})",
        zorder=4,
    )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"Literature-reported Tafel slope (mV dec$^{-1}$)")
    ax.set_ylabel(r"Bayesian local-line target (mV dec$^{-1}$)")
    ax.legend(
        [points, agreement_line, inner_band, outer_band],
        [
            f"Bayesian local-line refit (n = {int(mask.sum())})",
            "Perfect agreement",
            "±10% relative error",
            "±20% relative error",
        ],
        fontsize=9,
        loc="lower right",
    )
    fig.tight_layout(pad=1.6)
    fig.savefig(stage / "bayesian_local_line_vs_literature.png", dpi=300)
    fig.savefig(stage / "bayesian_local_line_vs_literature.pdf")
    plt.close(fig)
    write_json(
        stage / "summary.json",
        {
            "target_name": TARGET,
            "record_count": EXPECTED_RECORDS,
            "supported_target_count": EXPECTED_ELIGIBLE,
            "quantization_aware_fallback_records": sorted(FALLBACK_RECORDS),
            "median_mV_dec": float(np.median(slopes)),
            "min_mV_dec": float(slopes.min()),
            "max_mV_dec": float(slopes.max()),
            "above_500_count": int((slopes > 500).sum()),
            "literature_comparison_n": int(mask.sum()),
            "literature_spearman": rho,
            "within_10pct_relative_error_count": int(
                comparison.loc[mask, "within_10pct_relative_error"].sum()
            ),
            "within_20pct_relative_error_count": int(
                comparison.loc[mask, "within_20pct_relative_error"].sum()
            ),
            "records_without_literature_reported_value": int(
                EXPECTED_RECORDS - mask.sum()
            ),
            "shared_reported_value_records": comparison.loc[
                comparison["reported_value_assignment"].eq(
                    "shared_legacy_record_article_reported"
                ),
                "record_id",
            ]
            .astype(int)
            .tolist(),
        },
    )
    (stage / "README.md").write_text(
        f"""# Active Bayesian local-line target figures

Only `{TARGET}` is active. The KDE uses all 97 target values. The literature
comparison uses the 68 records with an article-reported Tafel slope; the 29
records without a reported value are excluded rather than replaced by an
earlier reconstruction. Records 42 and 43 are two retained curves bound to
legacy record 42 and therefore share its article-reported 95 mV dec-1
comparator. The scatter follows the information design of Limaye et al.
Figure 3A: a red perfect-agreement line with nested ±10% and ±20% relative
error bands. Unlike the paper's 0--200 mV dec-1 display restriction, this
project retains all 68 comparison records and the full observed range.
Design reference: Limaye et al., Nature Communications 12, 703 (2021),
https://doi.org/10.1038/s41467-021-20924-y.
Records 23 and 74 use the registered quantization-aware PAVA fallback; their
strict observed-progression selector had no supported interval.

Superseded target comparisons are archived. PNG files are 300 dpi and PDF
files preserve vector text.
""",
        encoding="utf-8",
    )
    recursive_manifest(stage, "artifact_manifest.json", {"target_name": TARGET})


def build_data_stage(stage: Path, base: Path, summary: pd.DataFrame) -> None:
    shutil.copytree(base, stage)
    target_map = summary.set_index("curve_id")["posterior_target_median_mV_dec"]
    for path in stage.glob("ml_dataset_*.csv"):
        frame = pd.read_csv(path)
        old_targets = [column for column in frame if column.startswith("b_")]
        frame = frame.drop(columns=old_targets, errors="ignore")
        frame[TARGET] = frame["curve_id"].map(target_map)
        frame.to_csv(path, index=False)
    (stage / "README.md").write_text(
        f"""# Exploratory ML datasets

The only output column is `{TARGET}`. Both CSV files contain all 97 records and
all 97 targets. Records 23 and 74 use the registered quantization-aware PAVA
fallback after strict observed progression has no supported interval.

Formal ML is blocked pending named review of all 97 PPCs. Any
ordinary random-split benchmark remains exploratory and potentially
optimistic.
""",
        encoding="utf-8",
    )
    recursive_manifest(stage, "artifact_manifest.json", {"target_name": TARGET})


def archive_and_publish(stages: dict[Path, Path], archive_root: Path) -> None:
    if archive_root.exists():
        raise FileExistsError(f"Archive destination already exists: {archive_root}")
    archive_root.mkdir(parents=True)
    active_destinations = list(stages)
    candidate_paths: list[Path] = [
        REPO
        / "Data/03_outputs/tafel_refitting/outputs/20260723_bayesian_local_line_formal_target_v1"
    ]
    moved = []
    for active in active_destinations:
        if active.exists():
            relative = active.relative_to(REPO)
            destination = archive_root / "active_state" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(active), str(destination))
            moved.append({"source": relative.as_posix(), "archive": destination.relative_to(archive_root).as_posix(), "kind": "superseded_active_state"})
    for source in candidate_paths:
        if source.exists():
            destination = archive_root / "candidate_target_runs" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append({"source": source.relative_to(REPO).as_posix(), "archive": destination.relative_to(archive_root).as_posix(), "kind": "candidate_or_superseded_target_run"})
    for active, stage in stages.items():
        os.replace(stage, active)
    write_json(
        archive_root / "archive_manifest.json",
        {
            "created_at_utc": now(),
            "reason": "97-label Bayesian local-line target with calibrated PAVA fallback replaced the 95-label no-fallback publication",
            "recoverable": True,
            "moved": moved,
            "retained_source_target": "Data/03_outputs/tafel_refitting/outputs/20260723_bayesian_local_line_strict_plus_fallback_merged_v2",
        },
    )
    (archive_root / "README.md").write_text(
        "# Superseded active target state\n\nThis recoverable archive preserves the active canonical, PPC, comparison, and exploratory-dataset publication that was replaced by the Bayesian local-line target. It is audit-only. Restore paths are recorded in `archive_manifest.json`; no files were deleted.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO
        / "Data/03_outputs/tafel_refitting/outputs/20260723_bayesian_local_line_strict_plus_fallback_merged_v2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    qa = json.loads((source_dir / "qa.json").read_text(encoding="utf-8"))
    if not qa.get("calibration_pass") or not qa.get("structural_qa_pass"):
        raise ValueError("Formal target must pass calibration and structural QA")
    if qa.get("policy_id") != SOURCE_POLICY:
        raise ValueError(f"Unexpected source policy: {qa.get('policy_id')}")
    if qa.get("record_count") != EXPECTED_RECORDS or qa.get("structurally_eligible_count") != EXPECTED_ELIGIBLE:
        raise ValueError("Source target must contain 97 registered records and 97 supported targets")
    if set(qa.get("unsupported_records", [])) != UNSUPPORTED_RECORDS:
        raise ValueError("Source target must not contain unsupported records")
    nonce = f".bayesian_local_line_stage_{os.getpid()}"
    formal_active = REPO / "Data/03_outputs/tafel_refitting/outputs/20260723_bayesian_local_line_formal_target_v2"
    canonical_active = REPO / "Data/02_inputs/canonical_dataset_ir_corrected_v2"
    ppc_active = REPO / "Data/04_results/RESULTS/01_PPC_REVIEW"
    results_active = REPO / "Data/04_results/RESULTS/03_TARGET_COMPARISON"
    data_active = REPO / "Data/04_results/RESULTS/08_DATA_EXPORTS"
    stages = {
        formal_active: formal_active.with_name(formal_active.name + nonce),
        canonical_active: canonical_active.with_name(canonical_active.name + nonce),
        ppc_active: ppc_active.with_name(ppc_active.name + nonce),
        results_active: results_active.with_name(results_active.name + nonce),
        data_active: data_active.with_name(data_active.name + nonce),
    }
    if any(path.exists() for path in stages.values()):
        raise FileExistsError("One or more publication staging directories already exist")
    try:
        build_formal_stage(source_dir, stages[formal_active])
        summary = pd.read_csv(stages[formal_active] / "target_summary.csv")
        source_draws = pd.read_csv(
            stages[formal_active] / "posterior_target_draws.csv.gz", low_memory=False
        )
        identity = build_identity(summary, source_draws)
        build_ppc_stage(stages[formal_active], stages[ppc_active], summary, identity)
        build_canonical_stage(
            canonical_active,
            stages[canonical_active],
            stages[formal_active],
            stages[ppc_active],
            summary,
            source_draws,
            identity,
        )
        master = pd.read_csv(stages[canonical_active] / "master.csv", low_memory=False)
        build_results_stage(stages[results_active], summary, master)
        build_data_stage(stages[data_active], data_active, summary)
        staged_registry = pd.read_csv(stages[canonical_active] / "target_registry.csv")
        staged_review = pd.read_csv(stages[ppc_active] / "ppc_review.csv")
        checks = {
            "one_active_target": set(staged_registry["target_name"]) == {TARGET},
            "registry_rows_97": len(staged_registry) == EXPECTED_RECORDS,
            "structural_eligible_97": int(staged_registry["target_structurally_eligible"].sum()) == EXPECTED_ELIGIBLE,
            "unsupported_records_empty": set(
                staged_registry.loc[~staged_registry["target_structurally_eligible"], "record_id"].astype(int)
            ) == UNSUPPORTED_RECORDS,
            "posterior_draw_rows_776000": len(source_draws)
            == EXPECTED_ELIGIBLE * EXPECTED_DRAWS_PER_ELIGIBLE_CURVE,
            "ppc_rows_97": len(staged_review) == EXPECTED_RECORDS,
            "ppc_images_97": len(list(stages[ppc_active].glob("*.png"))) == EXPECTED_RECORDS,
            "human_approvals_zero": staged_review["ppc_reviewer"].fillna("").eq("").all(),
            "canonical_manifest_present": (stages[canonical_active] / "artifact_manifest.json").is_file(),
            "formal_evidence_manifest_present": (
                stages[formal_active] / "artifact_manifest.json"
            ).is_file(),
        }
        if not all(checks.values()):
            raise ValueError(f"Staged publication checks failed: {checks}")
        archive_and_publish(stages, REPO / "archive" / ARCHIVE_NAME)
    except Exception:
        for path in stages.values():
            if path.exists():
                shutil.rmtree(path)
        raise
    print(
        json.dumps(
            {
                "status": "published_pending_human_ppc",
                "target": TARGET,
                "registered_records": EXPECTED_RECORDS,
                "supported_targets": EXPECTED_ELIGIBLE,
                "quantization_aware_fallback_records": sorted(FALLBACK_RECORDS),
                "checks": checks,
            },
            indent=2,
            default=lambda value: bool(value) if isinstance(value, np.bool_) else str(value),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
