"""Adapter from the canonical 97-record dataset to the ML workflow.

The canonical dataset is deliberately treated as a frozen input snapshot.
This module never edits the source workbook or Bayesian evidence.  It creates
the ``data/records.csv`` and ``targets/target_registry.csv`` contracts used by
the existing grouped nested-CV evaluator, while preserving the named PPC gate
for non-smoke evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .evaluate import build_connected_group_ids
from .features import is_target_like_column, panel_schema_frame
from .provenance import sha256_file, write_json


PRIMARY_TARGET = "b_bayesian_local_line_mV_dec"
EXACT_FIXED_CURRENT_TARGET = "b_refit_at_2mA_mV_dec"
SINGLE_REGIME_TARGET = "b_single_regime_mV_dec"
BEST_DIELECTRIC_SOURCE = (
    "solvent_mixture_property_proxy_dielectric_constant__best_available"
)
BEST_DIELECTRIC_MODEL = (
    "solvent_mixture_property_proxy_dielectric_constant_best_available"
)
BEST_DIELECTRIC_ESTIMATED_FLAG = (
    "solvent_mixture_property_proxy_dielectric_constant__best_available_is_estimated"
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_canonical_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Validate the canonical dataset manifest before it becomes input."""

    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Integration artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Integration artifact manifest contains no files")
    failures: list[str] = []
    for row in rows:
        path = dataset_dir / str(row.get("path", ""))
        if not path.is_file():
            failures.append(f"missing:{path.name}")
            continue
        observed = _sha256(path)
        if observed != str(row.get("sha256", "")):
            failures.append(f"hash_mismatch:{path.name}")
    if failures:
        raise ValueError(f"Integration artifact validation failed: {failures}")
    return {
        "status": "pass",
        "dataset_dir": str(dataset_dir),
        "artifact_manifest_sha256": _sha256(manifest_path),
        "verified_files": len(rows),
    }


def _drop_outcome_and_fit_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    explicit = {
        "overpotential_10mA_reported_mV",
        "overpotential_100mA_reported_mV",
        "cdl_reported_mF_cm2",
        "ecsa_reported_m2_g",
        "stability_100mA_reported_h",
        "selected_model",
        "sampler_diagnostics_pass",
        "named_ppc_review_complete",
        "formal_nuts_ppc_approved",
        "evidence_id",
        "functional_draws_sha256",
    }
    prefixes = (
        "bayesian_",
        "functional_",
        "posterior_",
        "legacy_target_",
    )
    dropped = sorted(
        column
        for column in frame.columns
        if column in explicit
        or column.startswith(prefixes)
        or is_target_like_column(column)
    )
    return frame.drop(columns=dropped, errors="ignore"), dropped


def _prepare_records(master: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    records = master.copy()
    records["sample_record"] = records["record_id"].astype(int)
    records["publication_year"] = pd.to_numeric(
        records.get("publication_year_legacy_unverified"), errors="coerce"
    )
    records["publication_year_provenance"] = "legacy_metadata_unverified"
    records["publication_id_provenance"] = "Data_0718_DOI_group"
    records["catalyst_source"] = "not_reported"
    records["catalyst_support"] = records.get("catalyst_support_type", pd.Series(index=records.index)).fillna("none")
    records["lab_group_id"] = records["publication_id"].map(
        lambda value: f"lab_unknown_{value}"
    )
    records["lab_group_confidence"] = "unverified"
    records["lab_group_provenance"] = "publication_specific_unknown_not_pooled"
    records["catalyst_loading_mg_cm2"] = pd.to_numeric(
        records["catalyst_loading_mg_cm2_selected"], errors="coerce"
    )
    records["dropwise_addition_uL"] = pd.to_numeric(
        records["drop_volume_uL"], errors="coerce"
    )
    records["solvent_ratio_sum_uL_per_mgcat"] = pd.to_numeric(
        records["total_solvent_only_volume_uL_per_mgcat"], errors="coerce"
    )
    records["total_solvent_volume_uL"] = pd.to_numeric(
        records["total_solvent_only_volume_uL"], errors="coerce"
    )
    records[BEST_DIELECTRIC_MODEL] = pd.to_numeric(
        records[BEST_DIELECTRIC_SOURCE], errors="coerce"
    )
    records["solvent_mixture_property_proxy_dielectric_constant_estimated"] = pd.to_numeric(
        records["solvent_mixture_property_proxy_dielectric_constant__estimated"],
        errors="coerce",
    )
    records["solvent_mixture_property_proxy_dielectric_constant_estimated_flag"] = pd.to_numeric(
        records[BEST_DIELECTRIC_ESTIMATED_FLAG], errors="raise"
    ).astype(int)
    records, dropped = _drop_outcome_and_fit_columns(records)
    records["cv_group_id"] = build_connected_group_ids(records)
    return records, dropped


def build_integrated_curation(
    *,
    dataset_dir: Path,
    output_dir: Path,
    panel_names: Iterable[str],
) -> dict[str, Path]:
    """Create the evaluator's canonical-record contract from Data_0718."""

    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_validation = validate_canonical_dataset(dataset_dir)

    master = pd.read_csv(dataset_dir / "master.csv")
    feature_registry = pd.read_csv(dataset_dir / "feature_registry.csv")
    source_qa = json.loads(
        (dataset_dir / "qa.json").read_text(encoding="utf-8")
    )
    required = {
        "record_id",
        "record_uid",
        "curve_id",
        "publication_id",
        "doi",
        "sample_id",
        "recipe_hash",
        "electrolyte_scope",
        "ir_correction_status",
        "row_data_quality_pass",
        BEST_DIELECTRIC_SOURCE,
        BEST_DIELECTRIC_ESTIMATED_FLAG,
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Integrated master is missing required columns: {missing}")
    if len(master) != 97 or master["record_id"].nunique() != 97:
        raise ValueError("Integrated master must contain 97 unique records")
    if master["curve_id"].nunique() != 97:
        raise ValueError("Canonical master must contain 97 unique curves")
    if not master["row_data_quality_pass"].map(_as_bool).all():
        failed = master.loc[
            ~master["row_data_quality_pass"].map(_as_bool), "record_id"
        ].tolist()
        raise ValueError(f"Records still failing row QA: {failed}")

    records, dropped = _prepare_records(master)
    panels = tuple(str(value) for value in panel_names)
    schema = panel_schema_frame(records, panels)
    selected = schema.loc[schema["status"].eq("present"), "selected_column"].dropna()
    leaking = sorted(column for column in selected.astype(str) if is_target_like_column(column))
    if leaking:
        raise ValueError(f"Target-like predictors survived panel resolution: {leaking}")

    best = pd.to_numeric(records[BEST_DIELECTRIC_MODEL], errors="coerce")
    estimate_flag = pd.to_numeric(
        records["solvent_mixture_property_proxy_dielectric_constant_estimated_flag"],
        errors="coerce",
    )
    ir_counts = {
        str(key): int(value)
        for key, value in records["ir_correction_status"].value_counts().to_dict().items()
    }
    expected_ir_counts = source_qa.get(
        "ir_status_counts",
        {
            "not_reported_or_unverifiable": 41,
            "reported_corrected": 38,
            "reported_not_corrected": 18,
        },
    )
    expected_ir_counts = {
        str(key): int(value) for key, value in expected_ir_counts.items()
    }
    curve_registry = records[["record_id", "curve_id"]].copy()
    hard_checks = {
        "records_are_1_to_97": records["record_id"].astype(int).tolist()
        == list(range(1, 98)),
        "record_ids_unique": not records["record_id"].duplicated().any(),
        "record_uids_unique": not records["record_uid"].duplicated().any(),
        "curves_unique": not records["curve_id"].duplicated().any(),
        "canonical_curve_registry_has_97_rows": len(curve_registry) == 97,
        "canonical_curve_registry_is_one_to_one": bool(
            not curve_registry["record_id"].duplicated().any()
            and not curve_registry["curve_id"].duplicated().any()
        ),
        "row_qa_all_pass": bool(records["row_data_quality_pass"].map(_as_bool).all()),
        "best_dielectric_97_of_97": int(best.notna().sum()) == 97,
        "dielectric_estimate_flag_count_31": int(estimate_flag.sum()) == 31,
        "ir_status_counts_match_canonical_qa": ir_counts == expected_ir_counts,
        "no_target_like_selected_predictors": not leaking,
        "connected_groups_available": records["cv_group_id"].nunique() >= 5,
    }
    if not all(hard_checks.values()):
        raise ValueError(f"Integrated curation hard checks failed: {hard_checks}")

    property_qa = {
        "status": "ready_with_user_approved_best_available_dielectric",
        "canonical_records": 97,
        "records_with_all_three_core_proxies": int(
            records[
                [
                    "solvent_mixture_property_proxy_viscosity_mPa_s",
                    "solvent_mixture_property_proxy_surface_tension_mN_m",
                    "solvent_mixture_property_proxy_density_kg_m3",
                ]
            ]
            .notna()
            .all(axis=1)
            .sum()
        ),
        "records_with_all_four_best_available_values": int(
            records[
                [
                    "solvent_mixture_property_proxy_viscosity_mPa_s",
                    "solvent_mixture_property_proxy_surface_tension_mN_m",
                    "solvent_mixture_property_proxy_density_kg_m3",
                    BEST_DIELECTRIC_MODEL,
                ]
            ]
            .notna()
            .all(axis=1)
            .sum()
        ),
        "best_available_dielectric_estimated_records": int(estimate_flag.sum()),
        "best_available_dielectric_strict_records": int((estimate_flag == 0).sum()),
        "estimated_panel_model_use_status": (
            "eligible_user_approved_with_estimate_indicator"
        ),
        "estimated_panel_user_override": {
            "date": "2026-07-18",
            "decision": "use_97_of_97_best_available_dielectric_with_estimate_flag",
            "strict_values_retained_for_audit": True,
        },
        "source_backed_4p_panel_model_use_status": "excluded_pending_primary_file_hashes",
        "medium_confidence_ipa_surface_compositions": 2,
        "medium_confidence_ipa_surface_sensitivity_required": True,
        "property_composition_coordinates_coexist_in_registered_panel": False,
        "claim_boundary": (
            "solvent-mixture property proxy only; 31 dielectric values are explicit "
            "model estimates, not measured catalyst/Nafion ink properties"
        ),
    }
    curation_qa = {
        "status": "pass",
        "source_mode": "canonical_97",
        "record_count": len(records),
        "curve_count": records["curve_id"].nunique(),
        "alkaline_record_count": int(records["electrolyte_scope"].eq("alkaline").sum()),
        "acidic_record_count": int(records["electrolyte_scope"].eq("acidic").sum()),
        "neutral_record_count": int(records["electrolyte_scope"].eq("neutral").sum()),
        "ir_status_counts": {str(key): int(value) for key, value in ir_counts.items()},
        "n_connected_groups": int(records["cv_group_id"].nunique()),
        "outcome_and_fit_columns_removed_from_records": dropped,
        "lab_held_out_validation_ready": False,
        "lab_validation_boundary": (
            "Unknown labs remain publication-specific and unpooled; results are not "
            "definitive verified-lab-held-out validation."
        ),
        "target_binding_ready": True,
        "formal_training_gate_source": "targets/target_qa.json",
        "missingness_policy": {
            "primary_drop_when_ratio_gt": 0.40,
            "sensitivity_only_when_ratio_gt": 0.20,
            "dropped_primary_features": [
                "drying_process",
                "drying_temperature_C",
                "drying_time_min",
            ],
            "sensitivity_only_features": [
                "sonication_time_min",
                "ir_compensation_fraction",
            ],
            "imputation_location": "inside_inner_grouped_CV",
        },
        "dataset_validation": dataset_validation,
        "source_data_quality_status": source_qa.get("status"),
        "hard_checks": hard_checks,
    }

    paths = {
        "records": output_dir / "records.csv",
        "feature_registry": output_dir / "feature_registry.csv",
        "sources": output_dir / "sources.csv",
        "curve_registry": output_dir / "curve_registry.csv",
        "panel_schema": output_dir / "registered_panel_schema.csv",
        "property_proxy_qa": output_dir / "property_proxy_qa.json",
        "curation_qa": output_dir / "curation_qa.json",
        "source_data_quality_qa": output_dir / "source_data_quality_qa.json",
        "integration_input_manifest": output_dir / "integration_input_manifest.json",
    }
    records.to_csv(paths["records"], index=False)
    feature_registry.to_csv(paths["feature_registry"], index=False)
    sources = (
        records[
            [
                "publication_id",
                "doi",
                "paper_name",
                "publication_year",
                "publication_year_provenance",
            ]
        ]
        .drop_duplicates("publication_id")
        .sort_values("publication_id")
    )
    sources.to_csv(paths["sources"], index=False)
    curve_registry.to_csv(paths["curve_registry"], index=False)
    schema.to_csv(paths["panel_schema"], index=False)
    write_json(paths["property_proxy_qa"], property_qa)
    write_json(paths["curation_qa"], curation_qa)
    write_json(paths["source_data_quality_qa"], source_qa)
    write_json(
        paths["integration_input_manifest"],
        {
            "dataset_dir": str(dataset_dir),
            "dataset_artifact_manifest_sha256": dataset_validation[
                "artifact_manifest_sha256"
            ],
            "source_files": {
                name: {
                    "path": str((dataset_dir / filename).resolve()),
                    "sha256": sha256_file(dataset_dir / filename),
                }
                for name, filename in {
                    "master": "master.csv",
                    "feature_registry": "feature_registry.csv",
                    "data_quality_qa": "qa.json",
                }.items()
            },
        },
    )
    return paths


def _review_complete(row: pd.Series) -> bool:
    decision = row.get("ppc_approved")
    decision_present = not (
        decision is None or (isinstance(decision, float) and np.isnan(decision))
    )
    reviewer = _clean_text(row.get("ppc_reviewer"))
    review_date = _clean_text(row.get("ppc_review_date"))
    return bool(decision_present and reviewer and review_date)


def _build_bayesian_local_line_targets_passthrough(
    *,
    dataset_dir: Path,
    records: pd.DataFrame,
    review: pd.DataFrame,
    source_registry: pd.DataFrame,
    source_audit: pd.DataFrame,
    ppc_review_path: Path,
    output_dir: Path,
    expected_primary_eligible: int,
    expected_primary_alkaline_eligible: int,
) -> dict[str, Path]:
    """Bind the calibrated Bayesian local-line target without target fallback."""

    required_registry = {
        "record_id",
        "sample_record",
        "curve_id",
        "target_name",
        "target_value_mV_dec",
        "target_interval_low_mV_dec",
        "target_interval_high_mV_dec",
        "target_structurally_eligible",
        "posterior_target_supported",
        "posterior_evidence_id",
        "slope_draws_sha256",
    }
    required_audit = {
        "curve_id",
        "evidence_id",
        "slope_draws_sha256",
        "ppc_plot_sha256",
        "target_structurally_eligible",
        "selection_uncertainty_propagated",
        "calibration_pass",
    }
    required_review = {
        "curve_id",
        "evidence_id",
        "slope_draws_sha256",
        "ppc_sha256",
        "ppc_approved",
        "ppc_reviewer",
        "ppc_review_date",
    }
    for label, frame, required in (
        ("target registry", source_registry, required_registry),
        ("Bayesian audit", source_audit, required_audit),
        ("PPC review", review, required_review),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Bayesian local-line {label} is missing columns: {missing}")
    if (
        len(source_registry) != 97
        or len(source_audit) != 97
        or len(review) != 97
        or source_registry["curve_id"].duplicated().any()
        or source_audit["curve_id"].duplicated().any()
        or review["curve_id"].duplicated().any()
    ):
        raise ValueError("Bayesian local-line target, audit, and PPC review must each contain 97 unique curves")
    if set(source_registry["target_name"].astype(str)) != {PRIMARY_TARGET}:
        raise ValueError("Active target registry must contain only the Bayesian local-line primary target")

    bound = source_audit.merge(
        review,
        on="curve_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_audit", "_review"),
    )
    binding_checks = {
        "evidence_id": bound["evidence_id_audit"].eq(bound["evidence_id_review"]),
        "slope_draws_sha256": bound["slope_draws_sha256_audit"].eq(
            bound["slope_draws_sha256_review"]
        ),
        "ppc_sha256": bound["ppc_plot_sha256"].eq(
            bound.get("ppc_sha256_review", bound.get("ppc_sha256"))
        ),
    }
    for label, matches in binding_checks.items():
        if not matches.all():
            bad = bound.loc[~matches, "curve_id"].tolist()
            raise ValueError(f"Bayesian local-line PPC {label} binding failed: {bad}")
    bound["named_ppc_review_complete"] = bound.apply(_review_complete, axis=1)
    bound["ppc_approved_bool"] = bound["ppc_approved"].map(_as_bool)
    bound["formal_target_approved_bound"] = (
        bound.get(
            "target_structurally_eligible_audit",
            bound.get("target_structurally_eligible"),
        ).map(_as_bool)
        & bound["named_ppc_review_complete"]
        & bound["ppc_approved_bool"]
    )
    bound["ppc_status_bound"] = np.where(
        bound["formal_target_approved_bound"],
        "named_approved",
        np.where(
            bound["named_ppc_review_complete"],
            "named_not_approved",
            "pending_named_review",
        ),
    )

    record_meta = records.set_index("curve_id")
    registry = source_registry.copy()
    if set(registry["curve_id"].astype(str)) != set(records["curve_id"].astype(str)):
        raise ValueError("Bayesian local-line registry curve set differs from canonical records")
    for column in ("record_id", "sample_record", "cv_group_id", "electrolyte_scope", "publication_id", "doi"):
        source_column = "record_id" if column == "sample_record" else column
        registry[column] = registry["curve_id"].map(record_meta[source_column])
    registry["target_structurally_eligible"] = registry[
        "target_structurally_eligible"
    ].map(_as_bool)
    registry["posterior_target_supported"] = registry[
        "posterior_target_supported"
    ].map(_as_bool)
    registry["target_eligible"] = (
        registry["target_structurally_eligible"]
        & registry["posterior_target_supported"]
        & pd.to_numeric(registry["target_value_mV_dec"], errors="coerce").notna()
    )
    formal_map = bound.set_index("curve_id")["formal_target_approved_bound"]
    named_map = bound.set_index("curve_id")["named_ppc_review_complete"]
    registry["formal_target_approved"] = registry["curve_id"].map(formal_map).map(_as_bool)
    registry["ppc_status"] = np.where(
        registry["formal_target_approved"],
        "named_approved",
        np.where(
            ~registry["target_structurally_eligible"],
            "not_applicable_structurally_unsupported",
            np.where(
                registry["curve_id"].map(named_map),
                "named_not_approved",
                "pending_named_review",
            ),
        ),
    )
    registry["target_role"] = "primary"
    registry = registry.sort_values("record_id").reset_index(drop=True)

    eligible_count = int(registry["target_eligible"].sum())
    alkaline_count = int(
        (registry["target_eligible"] & registry["electrolyte_scope"].eq("alkaline")).sum()
    )
    eligible_bound = bound.get(
        "target_structurally_eligible_audit",
        bound.get("target_structurally_eligible"),
    ).map(_as_bool)
    named_count = int((bound["named_ppc_review_complete"] & eligible_bound).sum())
    approved_count = int(bound["formal_target_approved_bound"].sum())
    hard_checks = {
        "registry_has_97_rows": len(registry) == 97,
        "one_active_target_only": set(registry["target_name"]) == {PRIMARY_TARGET},
        "primary_target_count_matches_registered": eligible_count == expected_primary_eligible,
        "alkaline_primary_count_matches_registered": alkaline_count
        == expected_primary_alkaline_eligible,
        "all_selection_uncertainty_propagated": bool(
            source_audit["selection_uncertainty_propagated"].map(_as_bool).all()
        ),
        "all_calibration_bound": bool(source_audit["calibration_pass"].map(_as_bool).all()),
        "binding_checks_pass": all(matches.all() for matches in binding_checks.values()),
    }
    if not all(hard_checks.values()):
        raise ValueError(f"Bayesian local-line target hard checks failed: {hard_checks}")
    formal_ready = bool(approved_count == eligible_count == expected_primary_eligible)
    qa = {
        "status": "formal_evaluation_ready" if formal_ready else "complete_with_formal_bayesian_review_pending",
        "source_mode": "canonical_97_bayesian_local_line_target",
        "primary_target": PRIMARY_TARGET,
        "primary_eligible_all_scopes": eligible_count,
        "primary_eligible_alkaline": alkaline_count,
        "n_named_ppc_review_complete": named_count,
        "n_formal_ppc_approved": approved_count,
        "formal_evaluation_ready": formal_ready,
        "formal_evaluation_blockers": [] if formal_ready else ["named_ppc_review_incomplete"],
        "only_remaining_formal_gate": None if formal_ready else "named_manual_PPC_review",
        "target_eligible_counts": {PRIMARY_TARGET: eligible_count},
        "formal_approved_target_counts": {PRIMARY_TARGET: approved_count},
        "primary_target_policy": {
            "common_estimand": "robust Bayesian local-linear apparent slope",
            "eta_min_V": 0.15,
            "window_rule": "strict observed progression; quantization-aware PAVA fallback only if unsupported",
            "window_uncertainty": "posterior mixture over interval-perturbed selections",
            "fallback": "same Bayesian local-line estimand on equal-weight unperturbed PAVA quantization blocks",
        },
        "ppc_review_source": {
            "path": str(Path(ppc_review_path).resolve()),
            "sha256": sha256_file(Path(ppc_review_path)),
        },
        "hard_checks": hard_checks,
    }
    paths = {
        "target_registry": output_dir / "target_registry.csv",
        "targets_wide": output_dir / "targets_wide.csv",
        "target_qa": output_dir / "target_qa.json",
        "ppc_binding_audit": output_dir / "ppc_binding_audit.csv",
        "bayesian_audit": output_dir / "bayesian_audit.csv",
        "bayesian_pending_queue": output_dir / "bayesian_pending_queue.csv",
        "posterior_functional_draws": output_dir / "posterior_functional_draws_at_2mA.csv.gz",
        "posterior_slope_draws": output_dir / "posterior_slope_draws.csv.gz",
    }
    registry.to_csv(paths["target_registry"], index=False)
    wide = registry.pivot(
        index="record_id", columns="target_name", values="target_value_mV_dec"
    ).reset_index().merge(
        records[["record_id", "sample_record", "cv_group_id", "electrolyte_scope", "curve_id"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    wide.to_csv(paths["targets_wide"], index=False)
    bound.to_csv(paths["ppc_binding_audit"], index=False)
    remapped_audit = source_audit.copy()
    remapped_audit["named_ppc_review_complete"] = remapped_audit["curve_id"].map(
        named_map
    ).map(_as_bool)
    remapped_audit["formal_target_approved"] = remapped_audit["curve_id"].map(
        formal_map
    ).map(_as_bool)
    remapped_audit["ppc_status"] = np.where(
        remapped_audit["formal_target_approved"],
        "named_approved",
        np.where(
            remapped_audit["named_ppc_review_complete"],
            "named_not_approved",
            "pending_named_review",
        ),
    )
    remapped_audit["next_action"] = np.where(
        ~remapped_audit["target_structurally_eligible"].map(_as_bool),
        "retain_record_with_missing_target_no_fallback",
        np.where(
            remapped_audit["formal_target_approved"],
            "none_formal_evidence_ready",
            "perform_ppc_review",
        ),
    )
    remapped_audit.to_csv(paths["bayesian_audit"], index=False)
    remapped_audit.loc[~remapped_audit["formal_target_approved"]].to_csv(
        paths["bayesian_pending_queue"], index=False
    )
    source_draws_path = dataset_dir / "posterior_slope_draws.csv.gz"
    draws = pd.read_csv(source_draws_path, low_memory=False)
    draw_checks = {
        "rows_match_eligible_times_8000": len(draws)
        == expected_primary_eligible * 8000,
        "curves_match_eligible": draws["curve_id"].nunique()
        == expected_primary_eligible,
        "one_target_only": set(draws["target_name"].astype(str)) == {PRIMARY_TARGET},
        "no_draw_duplicates": not draws.duplicated(["curve_id", "posterior_draw_index"]).any(),
    }
    if not all(draw_checks.values()):
        raise ValueError(f"Bayesian local-line posterior draw checks failed: {draw_checks}")
    draws.to_csv(paths["posterior_slope_draws"], index=False, compression="gzip")
    qa["posterior_slope_draw_aggregate"] = {
        "source_aggregate_path": str(source_draws_path.resolve()),
        "source_aggregate_sha256": sha256_file(source_draws_path),
        "hard_checks": draw_checks,
    }
    write_json(paths["target_qa"], qa)
    return paths


def build_integrated_targets(
    *,
    dataset_dir: Path,
    records_path: Path,
    ppc_review_path: Path,
    output_dir: Path,
    expected_primary_eligible: int = 97,
    expected_primary_alkaline_eligible: int = 90,
    expected_low_branch_eligible: int = 97,
    expected_single_regime_eligible: int = 72,
) -> dict[str, Path]:
    """Rebuild all registered Tafel targets on the canonical 97-record grain."""

    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_canonical_dataset(dataset_dir)
    records = pd.read_csv(records_path)
    review = pd.read_csv(ppc_review_path)
    source_registry = pd.read_csv(dataset_dir / "target_registry.csv")
    source_audit = pd.read_csv(dataset_dir / "bayesian_audit.csv")
    if len(records) != 97 or records["record_id"].nunique() != 97:
        raise ValueError("Target build requires the corrected 97-record registry")
    if len(source_audit) != 97 or source_audit["curve_id"].nunique() != 97:
        raise ValueError("Canonical Bayesian evidence audit must contain 97 curves")
    if review["curve_id"].duplicated().any():
        raise ValueError("PPC review must be unique by curve_id")
    if set(source_registry["target_name"].astype(str)) == {PRIMARY_TARGET}:
        return _build_bayesian_local_line_targets_passthrough(
            dataset_dir=dataset_dir,
            records=records,
            review=review,
            source_registry=source_registry,
            source_audit=source_audit,
            ppc_review_path=ppc_review_path,
            output_dir=output_dir,
            expected_primary_eligible=expected_primary_eligible,
            expected_primary_alkaline_eligible=expected_primary_alkaline_eligible,
        )

    required_slope_evidence = {
        "curve_id",
        "selected_model",
        "evidence_id",
        "sampler_diagnostics_pass",
        "posterior_slope_variable",
        "posterior_slope_median_mV_dec",
        "posterior_slope_draw_hdi_low_mV_dec",
        "posterior_slope_draw_hdi_high_mV_dec",
        "slope_draws_artifact_valid",
        "slope_draws_sha256",
    }
    missing_slope_evidence = sorted(required_slope_evidence - set(source_audit))
    if missing_slope_evidence:
        raise ValueError(
            "Canonical dataset predates the Bayesian low-potential primary-target "
            "policy and must be rebuilt; missing Bayesian audit columns: "
            f"{missing_slope_evidence}"
        )

    bound = source_audit.merge(
        review,
        on="curve_id",
        how="left",
        validate="one_to_one",
        suffixes=("_audit", "_review"),
    )
    if len(bound) != 97 or bound["evidence_id_review"].isna().any():
        raise ValueError("Every retained curve must bind to one PPC review row")
    evidence_match = bound["evidence_id_audit"].eq(bound["evidence_id_review"])
    hash_match = bound["slope_draws_sha256_audit"].eq(
        bound["slope_draws_sha256_review"]
    )
    if not evidence_match.all() or not hash_match.all():
        bad = bound.loc[~(evidence_match & hash_match), "curve_id"].tolist()
        raise ValueError(f"PPC evidence binding failed for curves: {bad}")
    model_match = bound["selected_model_audit"].eq(bound["selected_model_review"])
    if not model_match.all():
        bad = bound.loc[~model_match, "curve_id"].tolist()
        raise ValueError(f"PPC selected-model binding failed for curves: {bad}")

    bound["named_ppc_review_complete"] = bound.apply(_review_complete, axis=1)
    bound["ppc_approved_bool"] = bound["ppc_approved"].map(_as_bool)
    bound["formal_nuts_ppc_approved_bound"] = (
        bound["sampler_diagnostics_pass"].map(_as_bool)
        & bound["named_ppc_review_complete"]
        & bound["ppc_approved_bool"]
    )
    retained = records[["curve_id", "record_id"]].rename(
        columns={"record_id": "new_record_id"}
    )
    retained_set = set(retained["curve_id"].astype(str))
    if len(retained) != 97 or retained["new_record_id"].nunique() != 97:
        raise ValueError("Canonical records must map 97 curves to 97 records")

    registry = source_registry.loc[
        source_registry["curve_id"].astype(str).isin(retained_set)
    ].copy()
    registry = registry.merge(retained, on="curve_id", how="inner", validate="many_to_one")
    expected_targets = {
        EXACT_FIXED_CURRENT_TARGET,
        PRIMARY_TARGET,
        SINGLE_REGIME_TARGET,
    }
    if (
        len(registry) != 291
        or set(registry["target_name"].astype(str)) != expected_targets
        or registry.groupby("curve_id").size().ne(3).any()
    ):
        raise ValueError("Remapped target registry must contain three targets for 97 curves")
    registry["record_id"] = registry["new_record_id"]
    registry["sample_record"] = registry["new_record_id"]
    registry = registry.drop(columns=["new_record_id"])

    record_meta = records.set_index("record_id")
    curve_binding = bound.set_index("curve_id")
    audit_binding = source_audit.set_index("curve_id")
    for column in ("cv_group_id", "electrolyte_scope", "publication_id", "doi"):
        registry[column] = registry["record_id"].map(record_meta[column])
    registry["ir_status"] = registry["record_id"].map(
        record_meta["ir_correction_status"]
    )
    registry["selected_model"] = registry["curve_id"].map(
        audit_binding["selected_model"]
    )
    registry["posterior_evidence_id"] = registry["curve_id"].map(
        audit_binding["evidence_id"]
    )
    registry["slope_draws_sha256"] = registry["curve_id"].map(
        audit_binding["slope_draws_sha256"]
    )
    registry["functional_draws_sha256"] = registry["curve_id"].map(
        audit_binding.get("functional_draws_sha256", pd.Series(dtype=object))
    )
    registry["fit_status"] = "production_sampler_pass"
    registry["fit_eligible"] = True
    registry["n_curves"] = 1
    registry["curve_ids"] = registry["curve_id"]

    exact_mask = registry["target_name"].eq(EXACT_FIXED_CURRENT_TARGET)
    registry.loc[exact_mask, "target_role"] = "diagnostic"
    exact_curves = registry.loc[exact_mask, "curve_id"]
    exact_valid = (
        exact_curves.map(audit_binding["functional_target_supported"]).map(_as_bool)
        & exact_curves.map(audit_binding["functional_target_artifact_valid"]).map(
            _as_bool
        )
        & exact_curves.map(audit_binding["sampler_diagnostics_pass"]).map(_as_bool)
    )
    registry.loc[exact_mask, "target_value_mV_dec"] = exact_curves.map(
        audit_binding["functional_target_median_mV_dec"]
    ).where(exact_valid).to_numpy()
    registry.loc[exact_mask, "target_interval_low_mV_dec"] = exact_curves.map(
        audit_binding["functional_target_hdi_low_mV_dec"]
    ).where(exact_valid).to_numpy()
    registry.loc[exact_mask, "target_interval_high_mV_dec"] = exact_curves.map(
        audit_binding["functional_target_hdi_high_mV_dec"]
    ).where(exact_valid).to_numpy()
    registry.loc[exact_mask, "target_structurally_eligible"] = exact_valid.to_numpy()
    registry.loc[exact_mask, "target_eligible"] = exact_valid.to_numpy()
    registry.loc[exact_mask, "posterior_target_supported"] = exact_valid.to_numpy()
    registry.loc[exact_mask, "observed_support"] = exact_valid.to_numpy()
    registry.loc[exact_mask, "extrapolated"] = False
    registry.loc[exact_mask, "exclusion_reason"] = np.where(
        exact_valid, "diagnostic_supported", "diagnostic_exact_2mA_unsupported"
    )
    registry.loc[exact_mask, "target_uncertainty_source"] = (
        "chain_draw_synchronous_functional_hdi_diagnostic_only"
    )
    registry.loc[exact_mask, "target_current_density_mA_cm2"] = 2.0
    registry.loc[exact_mask, "calculation_method"] = (
        "diagnostic_complete_model_inversion_local_kinetic_slope"
    )
    registry.loc[exact_mask, "formal_target_approved"] = False

    primary_mask = registry["target_name"].eq(PRIMARY_TARGET)
    primary_curves = registry.loc[primary_mask, "curve_id"]
    primary_values = primary_curves.map(audit_binding["posterior_slope_median_mV_dec"])
    primary_low = primary_curves.map(
        audit_binding["posterior_slope_draw_hdi_low_mV_dec"]
    )
    primary_high = primary_curves.map(
        audit_binding["posterior_slope_draw_hdi_high_mV_dec"]
    )
    primary_variable = primary_curves.map(audit_binding["posterior_slope_variable"])
    primary_valid = (
        primary_curves.map(audit_binding["slope_draws_artifact_valid"]).map(_as_bool)
        & primary_curves.map(audit_binding["sampler_diagnostics_pass"]).map(_as_bool)
        & primary_variable.isin({"slope_mV_dec", "slope_low_mV_dec"})
        & pd.to_numeric(primary_values, errors="coerce").notna()
        & pd.to_numeric(primary_low, errors="coerce").notna()
        & pd.to_numeric(primary_high, errors="coerce").notna()
    )
    registry.loc[primary_mask, "target_role"] = "primary"
    registry.loc[primary_mask, "target_value_mV_dec"] = primary_values.to_numpy()
    registry.loc[primary_mask, "target_interval_low_mV_dec"] = primary_low.to_numpy()
    registry.loc[primary_mask, "target_interval_high_mV_dec"] = primary_high.to_numpy()
    registry.loc[primary_mask, "target_structurally_eligible"] = primary_valid.to_numpy()
    registry.loc[primary_mask, "target_eligible"] = primary_valid.to_numpy()
    registry.loc[primary_mask, "posterior_target_supported"] = primary_valid.to_numpy()
    registry.loc[primary_mask, "exclusion_reason"] = np.where(
        primary_valid, "eligible", "missing_or_invalid_posterior_branch_slope"
    )
    registry.loc[primary_mask, "sample_candidate_gate_passed"] = primary_valid.to_numpy()
    registry.loc[primary_mask, "sample_candidate_gate_reason"] = np.where(
        primary_valid,
        "selected_model_has_posterior_low_potential_slope",
        "selected_model_lacks_valid_posterior_low_potential_slope",
    )
    registry.loc[primary_mask, "target_uncertainty_source"] = (
        "chain_draw_synchronous_selected_model_slope_hdi"
    )
    registry.loc[primary_mask, "target_current_density_mA_cm2"] = np.nan
    registry.loc[primary_mask, "observed_support"] = True
    registry.loc[primary_mask, "extrapolated"] = False
    registry.loc[primary_mask, "calculation_method"] = (
        "selected_model_posterior_slope_single_or_low_potential_branch"
    )
    registry.loc[primary_mask, "cohort"] = np.where(
        primary_variable.eq("slope_low_mV_dec"),
        "selected_two_regime_low_potential_branch",
        "selected_single_regime",
    )
    registry.loc[primary_mask, "branch_rule"] = np.where(
        primary_variable.eq("slope_low_mV_dec"),
        "model_coordinate_low_potential_branch",
        "single_constant_slope",
    )
    registry.loc[primary_mask, "posterior_target_gap_reason"] = np.where(
        primary_valid, "complete", "invalid_posterior_slope_artifact"
    )

    single_mask = registry["target_name"].eq(SINGLE_REGIME_TARGET)
    single_curves = registry.loc[single_mask, "curve_id"]
    single_variable = single_curves.map(audit_binding["posterior_slope_variable"])
    single_valid = (
        single_curves.map(audit_binding["slope_draws_artifact_valid"]).map(_as_bool)
        & single_curves.map(audit_binding["sampler_diagnostics_pass"]).map(_as_bool)
        & single_variable.eq("slope_mV_dec")
    )
    registry.loc[single_mask, "target_role"] = "sensitivity_a"
    registry.loc[single_mask, "target_value_mV_dec"] = single_curves.map(
        audit_binding["posterior_slope_median_mV_dec"]
    ).where(single_valid).to_numpy()
    registry.loc[single_mask, "target_interval_low_mV_dec"] = single_curves.map(
        audit_binding["posterior_slope_draw_hdi_low_mV_dec"]
    ).where(single_valid).to_numpy()
    registry.loc[single_mask, "target_interval_high_mV_dec"] = single_curves.map(
        audit_binding["posterior_slope_draw_hdi_high_mV_dec"]
    ).where(single_valid).to_numpy()
    registry.loc[single_mask, "target_structurally_eligible"] = single_valid.to_numpy()
    registry.loc[single_mask, "target_eligible"] = single_valid.to_numpy()
    registry.loc[single_mask, "posterior_target_supported"] = single_valid.to_numpy()
    registry.loc[single_mask, "exclusion_reason"] = np.where(
        single_valid, "eligible", "selected_model_is_two_regime"
    )
    registry.loc[single_mask, "sample_candidate_gate_passed"] = single_valid.to_numpy()
    registry.loc[single_mask, "sample_candidate_gate_reason"] = np.where(
        single_valid, "selected_single_regime_model", "selected_two_regime_model"
    )
    registry.loc[single_mask, "target_uncertainty_source"] = (
        "chain_draw_synchronous_selected_model_slope_hdi"
    )
    registry.loc[single_mask, "target_current_density_mA_cm2"] = np.nan
    registry.loc[single_mask, "observed_support"] = single_valid.to_numpy()
    registry.loc[single_mask, "extrapolated"] = False
    registry.loc[single_mask, "calculation_method"] = "selected_single_regime_posterior_slope"
    registry.loc[single_mask, "cohort"] = "selected_single_regime"

    registry["target_structurally_eligible"] = registry[
        "target_structurally_eligible"
    ].map(_as_bool)
    registry["target_eligible"] = registry["target_eligible"].map(_as_bool)
    registry["posterior_target_supported"] = registry[
        "posterior_target_supported"
    ].map(_as_bool)
    registry["bayesian_status"] = "production_sampler_pass"
    curve_formal = registry["curve_id"].map(
        curve_binding["formal_nuts_ppc_approved_bound"]
    ).map(_as_bool)
    curve_named = registry["curve_id"].map(
        curve_binding["named_ppc_review_complete"]
    ).map(_as_bool)
    registry["ppc_status"] = np.where(
        curve_formal,
        "named_approved",
        np.where(curve_named, "named_not_approved", "pending_named_review"),
    )
    ml_target_mask = registry["target_name"].isin({PRIMARY_TARGET, SINGLE_REGIME_TARGET})
    registry["formal_target_approved"] = (
        ml_target_mask & registry["target_eligible"] & curve_formal
    )
    registry = registry.sort_values(
        ["target_role", "target_name", "record_id"], kind="stable"
    ).reset_index(drop=True)
    front = ["record_id", "sample_record", "cv_group_id", "electrolyte_scope"]
    registry = registry[front + [column for column in registry if column not in front]]

    named_count = int(bound["named_ppc_review_complete"].sum())
    approved_count = int(bound["formal_nuts_ppc_approved_bound"].sum())
    eligible_counts = {
        str(key): int(value)
        for key, value in registry.groupby("target_name")["target_eligible"].sum().items()
    }
    eligible_count = eligible_counts[PRIMARY_TARGET]
    formal_ready = bool(
        named_count == 97
        and approved_count == 97
        and registry.loc[
            ml_target_mask & registry["target_eligible"], "formal_target_approved"
        ].all()
    )
    blockers = []
    if named_count != 97:
        blockers.append("named_ppc_review_incomplete")
    if approved_count != 97:
        blockers.append("one_or_more_retained_curves_not_ppc_approved")
    if not registry.loc[
        ml_target_mask & registry["target_eligible"], "formal_target_approved"
    ].all():
        blockers.append("one_or_more_eligible_targets_lack_bound_ppc_approval")
    qa = {
        "status": (
            "formal_evaluation_ready"
            if formal_ready
            else "complete_with_formal_bayesian_review_pending"
        ),
        "source_mode": "canonical_97_tafel_targets",
        "n_evidence_curves": 97,
        "n_curves_in_registry": 97,
        "n_records_in_registry": 97,
        "n_fit_eligible_curves": 97,
        "n_sampler_diagnostics_pass": int(
            bound["sampler_diagnostics_pass"].map(_as_bool).sum()
        ),
        "n_named_ppc_review_complete": named_count,
        "n_formal_nuts_ppc_approved": approved_count,
        "target_eligible_counts": eligible_counts,
        "posterior_supported_target_counts": {
            target_name: int(
                (
                    registry["target_name"].eq(target_name)
                    & registry["target_eligible"]
                    & registry["posterior_target_supported"]
                ).sum()
            )
            for target_name in sorted(expected_targets)
        },
        "formal_approved_target_counts": {
            str(key): int(value)
            for key, value in registry.groupby("target_name")[
                "formal_target_approved"
            ].sum().items()
        },
        "primary_eligible_all_scopes": eligible_count,
        "primary_eligible_alkaline": int(
            (
                registry["target_name"].eq(PRIMARY_TARGET)
                & registry["target_eligible"]
                & registry["electrolyte_scope"].eq("alkaline")
            ).sum()
        ),
        "formal_evaluation_ready": formal_ready,
        "formal_evaluation_blockers": blockers,
        "only_remaining_formal_gate": (
            "named_manual_PPC_review" if blockers and all("ppc" in item for item in blockers) else None
        ),
        "primary_target_policy": {
            "target_name": PRIMARY_TARGET,
            "single_regime": "posterior constant slope",
            "two_regime": "posterior model-coordinate low-potential branch slope",
            "summary": "posterior median with 95% draw interval",
            "exact_2mA_role": "audit_only_excluded_from_ml",
        },
        "exact_2mA_policy": {
            "role": "diagnostic_audit_only",
            "included_in_ml": False,
            "included_in_formal_readiness": False,
        },
        "ppc_review_source": {
            "path": str(Path(ppc_review_path).resolve()),
            "sha256": sha256_file(Path(ppc_review_path)),
        },
        "hard_checks": {
            "registry_has_291_rows": len(registry) == 291,
            "registry_has_97_record_ids": registry["record_id"].nunique() == 97,
            "registry_has_97_curve_ids": registry["curve_id"].nunique() == 97,
            "one_row_per_record_target": not registry.duplicated(
                ["record_id", "target_name"]
            ).any(),
            "evidence_ids_bound": bool(evidence_match.all()),
            "slope_draw_hashes_bound": bool(hash_match.all()),
            "all_sampler_diagnostics_pass": bool(
                bound["sampler_diagnostics_pass"].map(_as_bool).all()
            ),
            "primary_target_count_matches_registered": eligible_count
            == expected_primary_eligible,
            "low_target_count_matches_registered": eligible_counts[
                "b_low_potential_mV_dec"
            ]
            == expected_low_branch_eligible,
            "single_target_count_matches_registered": eligible_counts[
                "b_single_regime_mV_dec"
            ]
            == expected_single_regime_eligible,
            "all_eligible_targets_have_posterior_support": bool(
                registry.loc[
                    ml_target_mask & registry["target_eligible"],
                    "posterior_target_supported",
                ].all()
            ),
            "alkaline_primary_target_count_matches_registered": int(
                (
                    registry["target_name"].eq(PRIMARY_TARGET)
                    & registry["target_eligible"]
                    & registry["electrolyte_scope"].eq("alkaline")
                ).sum()
            )
            == expected_primary_alkaline_eligible,
        },
    }
    if not all(qa["hard_checks"].values()):
        raise ValueError(f"Integrated target hard checks failed: {qa['hard_checks']}")

    paths = {
        "target_registry": output_dir / "target_registry.csv",
        "targets_wide": output_dir / "targets_wide.csv",
        "target_qa": output_dir / "target_qa.json",
        "ppc_binding_audit": output_dir / "ppc_binding_audit.csv",
        "bayesian_audit": output_dir / "bayesian_audit.csv",
        "bayesian_pending_queue": output_dir / "bayesian_pending_queue.csv",
        "posterior_functional_draws": output_dir
        / "posterior_functional_draws_at_2mA.csv.gz",
        "posterior_slope_draws": output_dir / "posterior_slope_draws.csv.gz",
    }
    registry.to_csv(paths["target_registry"], index=False)
    wide = (
        registry.pivot(
            index="record_id", columns="target_name", values="target_value_mV_dec"
        )
        .reset_index()
        .merge(
            records[
                [
                    "record_id",
                    "sample_record",
                    "cv_group_id",
                    "electrolyte_scope",
                    "curve_id",
                ]
            ],
            on="record_id",
            how="left",
            validate="one_to_one",
        )
    )
    wide.to_csv(paths["targets_wide"], index=False)
    write_json(paths["target_qa"], qa)
    bound[
        [
            "record_id",
            "curve_id",
            "evidence_id_audit",
            "slope_draws_sha256_audit",
            "sampler_diagnostics_pass",
            "named_ppc_review_complete",
            "ppc_approved_bool",
            "formal_nuts_ppc_approved_bound",
        ]
    ].to_csv(paths["ppc_binding_audit"], index=False)

    remapped_audit = source_audit.loc[
        source_audit["curve_id"].astype(str).isin(retained_set)
    ].merge(retained, on="curve_id", how="inner", validate="one_to_one")
    remapped_audit["sample_record"] = remapped_audit["new_record_id"].astype(int)
    remapped_audit = remapped_audit.drop(columns=["new_record_id"])
    audit_formal_map = curve_binding["formal_nuts_ppc_approved_bound"]
    audit_named_map = curve_binding["named_ppc_review_complete"]
    remapped_audit["named_ppc_review_complete"] = remapped_audit["curve_id"].map(
        audit_named_map
    ).map(_as_bool)
    remapped_audit["formal_nuts_ppc_approved"] = remapped_audit["curve_id"].map(
        audit_formal_map
    ).map(_as_bool)
    remapped_audit["next_action"] = np.where(
        remapped_audit["formal_nuts_ppc_approved"],
        "none_formal_evidence_ready",
        "perform_ppc_review",
    )
    remapped_audit.to_csv(paths["bayesian_audit"], index=False)
    remapped_audit.loc[
        ~remapped_audit["formal_nuts_ppc_approved"]
    ].to_csv(paths["bayesian_pending_queue"], index=False)

    source_draws_path = dataset_dir / "posterior_slope_draws.csv.gz"
    source_draws = pd.read_csv(source_draws_path, low_memory=False)
    primary_curves = set(
        registry.loc[
            registry["target_name"].eq(PRIMARY_TARGET)
            & registry["target_eligible"],
            "curve_id",
        ].astype(str)
    )
    posterior_draws = source_draws.loc[
        source_draws["curve_id"].astype(str).isin(primary_curves)
    ].copy()
    record_by_curve = retained.set_index("curve_id")["new_record_id"]
    posterior_draws["sample_record"] = posterior_draws["curve_id"].map(
        record_by_curve
    ).astype(int)
    draw_keys = ["sample_record", "posterior_draw_index"]
    expected_draw_rows = int(
        pd.to_numeric(
            audit_binding.loc[sorted(primary_curves), "posterior_slope_eligible_draws"],
            errors="raise",
        ).sum()
    )
    draw_hard_checks = {
        "rows_match_supported_draw_count": len(posterior_draws) == expected_draw_rows,
        "curves_match_primary_target_count": posterior_draws["curve_id"].nunique()
        == expected_primary_eligible,
        "records_match_primary_target_count": posterior_draws["sample_record"].nunique()
        == expected_primary_eligible,
        "no_draw_key_duplicates": not posterior_draws.duplicated(draw_keys).any(),
        "eligible_curve_set_exact": set(posterior_draws["curve_id"].astype(str))
        == primary_curves,
    }
    if not all(draw_hard_checks.values()):
        raise ValueError(f"Remapped posterior draw checks failed: {draw_hard_checks}")
    posterior_draws.to_csv(
        paths["posterior_slope_draws"], index=False, compression="gzip"
    )
    qa["posterior_slope_draw_aggregate"] = {
        "source_aggregate_path": str(
            source_draws_path.resolve()
        ),
        "source_aggregate_sha256": sha256_file(
            source_draws_path
        ),
        "hard_checks": draw_hard_checks,
        "expected_rows": expected_draw_rows,
        "mapping_policy": "canonical curve_id-to-record_id binding",
    }
    exact_draws_path = dataset_dir / "posterior_draws_at_2mA.csv.gz"
    if exact_draws_path.is_file():
        exact_draws = pd.read_csv(exact_draws_path, low_memory=False)
        exact_draws["sample_record"] = exact_draws["curve_id"].map(record_by_curve)
        exact_draws.to_csv(
            paths["posterior_functional_draws"], index=False, compression="gzip"
        )
        qa["exact_2mA_audit_draw_aggregate"] = {
            "role": "diagnostic_audit_only",
            "included_in_ml": False,
            "source_aggregate_path": str(exact_draws_path.resolve()),
            "source_aggregate_sha256": sha256_file(exact_draws_path),
        }
    write_json(paths["target_qa"], qa)
    return paths
