"""Data contracts and QA gates for orthogonal kinetic evidence.

This module does not infer unavailable measurements.  It distinguishes
source-verified condition covariates from auxiliary kinetic outcomes and from
features that would leak the target into cardinal classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .provenance import sha256_file, write_json


EXPECTED_BINNED_POINTS_SHA256 = (
    "a33bd30158dbac07957166ed7b55c161fc2b30a06f0221a3b307cfc849990ea4"
)

ALLOWED_FEATURE_ROLES = frozenset(
    {
        "context_predictor",
        "orthogonal_kinetic_covariate",
        "auxiliary_outcome",
        "forbidden_target_derived",
    }
)
ALLOWED_PH_ORIGINS = frozenset(
    {
        "measured",
        "reported",
        "calculated_activity_model",
        "nominal_ideal_proxy",
        "missing",
    }
)

TABLE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "orthogonal_kinetic_series.csv": (
        "series_id",
        "publication_id",
        "doi",
        "source_locator",
        "source_file_sha256",
        "catalyst_material_id",
        "catalyst_batch_id",
        "formulation_id",
        "electrode_id",
        "varied_factor",
        "held_constant_fields_sha256",
        "temperature_C",
        "gas_condition",
        "electrolyte_identity",
        "reference_electrode",
        "potential_reference_reported",
        "rhe_conversion_formula",
        "ir_correction_status",
        "rotation_rpm",
        "measurement_protocol",
        "value_origin",
        "extractor",
        "reviewer",
        "review_date",
        "verification_status",
        "feature_role",
    ),
    "orthogonal_kinetic_conditions.csv": (
        "condition_id",
        "series_id",
        "record_id",
        "curve_id",
        "replicate_id",
        "pH_value",
        "pH_origin",
        "pH_measurement_method",
        "pH_temperature_C",
        "reactant_species",
        "reactant_concentration_M",
        "reactant_activity",
        "activity_model",
        "ionic_strength_M",
        "condition_level",
        "source_locator",
        "verification_status",
        "feature_role",
    ),
    "orthogonal_kinetic_points.csv": (
        "condition_id",
        "curve_id",
        "point_index",
        "potential_raw",
        "potential_unit",
        "potential_reference_raw",
        "potential_RHE_V",
        "eta_V",
        "current_raw",
        "current_unit",
        "current_normalization",
        "current_density_A_cm2",
        "digitization_resolution_potential_V",
        "digitization_resolution_log_current_dec",
        "source_locator",
        "source_file_sha256",
    ),
    "apparent_reaction_orders.csv": (
        "reaction_order_id",
        "series_id",
        "reactant_species",
        "eta_RHE_V",
        "apparent_reaction_order",
        "ci_low",
        "ci_high",
        "n_activity_levels",
        "log_activity_span_dec",
        "fit_method",
        "verification_status",
        "feature_role",
    ),
    "controlled_formulation_registry.csv": (
        "controlled_series_id",
        "record_id",
        "series_id",
        "formulation_level",
        "catalyst_material_id",
        "catalyst_batch_id",
        "ink_batch_id",
        "ionomer_lot",
        "solvent_fraction_acetone",
        "solvent_fraction_water",
        "solvent_fraction_ethanol",
        "solvent_fraction_isopropanol",
        "solvent_fraction_methanol",
        "sonication_time_min",
        "drying_protocol",
        "deposition_protocol",
        "catalyst_loading_mg_cm2",
        "randomization_protocol",
        "independent_ink_preparation_id",
        "independent_electrode_id",
        "varied_formulation_field",
        "held_constant_fields_sha256",
        "verification_status",
        "feature_role",
    ),
}


@dataclass
class OrthogonalKineticsAudit:
    tables: dict[str, pd.DataFrame]
    table_coverage: pd.DataFrame
    predictor_eligibility: pd.DataFrame
    data_quality_issues: pd.DataFrame
    feature_roles: pd.DataFrame
    acquisition_queue: pd.DataFrame
    qa: dict[str, object]


def orthogonal_feature_role_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_or_measurement": "source_verified_pH_value",
                "feature_role": "orthogonal_kinetic_covariate",
                "cardinal_classifier_use": "coverage_and_provenance_gated",
                "reason": "independently reported experimental condition",
            },
            {
                "field_or_measurement": "nominal_ideal_pH_proxy",
                "feature_role": "context_predictor",
                "cardinal_classifier_use": "sensitivity_only",
                "reason": (
                    "deterministic transform of electrolyte identity and "
                    "molarity; not an independent pH measurement"
                ),
            },
            {
                "field_or_measurement": "pH_dependence",
                "feature_role": "auxiliary_outcome",
                "cardinal_classifier_use": "forbidden_same_experiment",
                "reason": "kinetic response across controlled pH levels",
            },
            {
                "field_or_measurement": "apparent_reaction_order",
                "feature_role": "auxiliary_outcome",
                "cardinal_classifier_use": "forbidden_same_experiment",
                "reason": "orthogonal kinetic response, not formulation context",
            },
            {
                "field_or_measurement": "local_slope_vs_potential",
                "feature_role": "forbidden_target_derived",
                "cardinal_classifier_use": "forbidden",
                "reason": (
                    "derived from the same electrochemical curve as the "
                    "cardinal target"
                ),
            },
            {
                "field_or_measurement": "controlled_formulation_descriptor",
                "feature_role": "context_predictor",
                "cardinal_classifier_use": "separate_controlled_series_analysis",
                "reason": (
                    "eligible only after catalyst/ink batch and independent "
                    "replicate gates pass"
                ),
            },
        ]
    )


def _read_contract_tables(registry_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for filename, expected in TABLE_COLUMNS.items():
        path = registry_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing orthogonal registry table: {path}")
        frame = pd.read_csv(path)
        missing = sorted(set(expected) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(expected))
        if missing or extra:
            raise ValueError(
                f"{filename} schema mismatch: missing={missing}, extra={extra}"
            )
        tables[filename] = frame.loc[:, list(expected)].copy()
    return tables


def _append_issue(
    rows: list[dict[str, object]],
    *,
    table: str,
    row_identifier: str,
    field: str,
    severity: str,
    issue: str,
) -> None:
    rows.append(
        {
            "table": table,
            "row_identifier": row_identifier,
            "field": field,
            "severity": severity,
            "issue": issue,
        }
    )


def _validate_tables(
    tables: Mapping[str, pd.DataFrame],
    records: pd.DataFrame,
) -> pd.DataFrame:
    issues: list[dict[str, object]] = []
    identifiers = {
        "orthogonal_kinetic_series.csv": "series_id",
        "orthogonal_kinetic_conditions.csv": "condition_id",
        "apparent_reaction_orders.csv": "reaction_order_id",
    }
    for filename, id_column in identifiers.items():
        frame = tables[filename]
        if not frame.empty and (
            frame[id_column].isna().any()
            or frame[id_column].astype(str).duplicated().any()
        ):
            _append_issue(
                issues,
                table=filename,
                row_identifier="*",
                field=id_column,
                severity="error",
                issue="identifier must be nonmissing and unique",
            )

    series = tables["orthogonal_kinetic_series.csv"]
    conditions = tables["orthogonal_kinetic_conditions.csv"]
    reaction = tables["apparent_reaction_orders.csv"]
    controlled = tables["controlled_formulation_registry.csv"]
    valid_series = set(series["series_id"].dropna().astype(str))
    for filename, frame in (
        ("orthogonal_kinetic_conditions.csv", conditions),
        ("apparent_reaction_orders.csv", reaction),
    ):
        for value in sorted(
            set(frame["series_id"].dropna().astype(str)) - valid_series
        ):
            _append_issue(
                issues,
                table=filename,
                row_identifier=value,
                field="series_id",
                severity="error",
                issue="foreign key is absent from orthogonal_kinetic_series.csv",
            )

    valid_records = set(records["record_id"].astype(str))
    for filename, frame in (
        ("orthogonal_kinetic_conditions.csv", conditions),
        ("controlled_formulation_registry.csv", controlled),
    ):
        linked = set(frame["record_id"].dropna().astype(str))
        for value in sorted(linked - valid_records):
            _append_issue(
                issues,
                table=filename,
                row_identifier=value,
                field="record_id",
                severity="error",
                issue="record_id is absent from the canonical 97-record registry",
            )

    for filename, frame in tables.items():
        if "feature_role" in frame:
            invalid_roles = sorted(
                set(frame["feature_role"].dropna().astype(str))
                - ALLOWED_FEATURE_ROLES
            )
            for value in invalid_roles:
                _append_issue(
                    issues,
                    table=filename,
                    row_identifier="*",
                    field="feature_role",
                    severity="error",
                    issue=f"unsupported feature role: {value}",
                )

    if not conditions.empty:
        invalid_origins = sorted(
            set(conditions["pH_origin"].dropna().astype(str))
            - ALLOWED_PH_ORIGINS
        )
        for value in invalid_origins:
            _append_issue(
                issues,
                table="orthogonal_kinetic_conditions.csv",
                row_identifier="*",
                field="pH_origin",
                severity="error",
                issue=f"unsupported pH origin: {value}",
            )

    verified_series = series.loc[
        series["verification_status"].astype(str).eq("verified")
    ]
    required_provenance = (
        "doi",
        "source_locator",
        "source_file_sha256",
        "extractor",
        "reviewer",
        "review_date",
    )
    for _, row in verified_series.iterrows():
        for field in required_provenance:
            if pd.isna(row[field]) or not str(row[field]).strip():
                _append_issue(
                    issues,
                    table="orthogonal_kinetic_series.csv",
                    row_identifier=str(row["series_id"]),
                    field=field,
                    severity="error",
                    issue="verified series is missing required provenance",
                )
    return pd.DataFrame(
        issues,
        columns=[
            "table",
            "row_identifier",
            "field",
            "severity",
            "issue",
        ],
    )


def _pH_series_gate(
    conditions: pd.DataFrame,
) -> tuple[int, int, list[str]]:
    verified = conditions.loc[
        conditions["verification_status"].astype(str).eq("verified")
        & conditions["pH_origin"].astype(str).isin(
            {"measured", "reported", "calculated_activity_model"}
        )
    ].copy()
    verified["pH_value"] = pd.to_numeric(
        verified["pH_value"], errors="coerce"
    )
    verified = verified.loc[np.isfinite(verified["pH_value"])]
    eligible: list[str] = []
    for series_id, frame in verified.groupby("series_id"):
        values = np.sort(frame["pH_value"].unique())
        if len(values) >= 3 and float(values[-1] - values[0]) >= 1.0:
            eligible.append(str(series_id))
    return len(verified), int(verified["series_id"].nunique()), eligible


def _reaction_order_gate(
    reaction: pd.DataFrame,
) -> pd.DataFrame:
    if reaction.empty:
        return reaction.copy()
    levels = pd.to_numeric(reaction["n_activity_levels"], errors="coerce")
    span = pd.to_numeric(reaction["log_activity_span_dec"], errors="coerce")
    estimate = pd.to_numeric(
        reaction["apparent_reaction_order"], errors="coerce"
    )
    return reaction.loc[
        reaction["verification_status"].astype(str).eq("verified")
        & levels.ge(3)
        & span.ge(0.75)
        & np.isfinite(estimate)
    ].copy()


def _controlled_series_gate(
    controlled: pd.DataFrame,
) -> list[str]:
    verified = controlled.loc[
        controlled["verification_status"].astype(str).eq("verified")
    ].copy()
    eligible: list[str] = []
    for series_id, frame in verified.groupby("controlled_series_id"):
        levels = frame["formulation_level"].dropna().astype(str).nunique()
        preparations = (
            frame.groupby("formulation_level")[
                "independent_ink_preparation_id"
            ].nunique()
        )
        electrodes = (
            frame.groupby("formulation_level")[
                "independent_electrode_id"
            ].nunique()
        )
        required_text = (
            "catalyst_batch_id",
            "ink_batch_id",
            "varied_formulation_field",
            "held_constant_fields_sha256",
        )
        complete = all(frame[column].notna().all() for column in required_text)
        if (
            levels >= 4
            and not preparations.empty
            and preparations.ge(3).all()
            and electrodes.ge(3).all()
            and complete
        ):
            eligible.append(str(series_id))
    return eligible


def audit_orthogonal_kinetics(
    records: pd.DataFrame,
    registry_dir: str | Path,
    binned_points_path: str | Path,
    output_dir: str | Path,
) -> OrthogonalKineticsAudit:
    """Validate available evidence and persist explicit use/skip decisions."""

    registry = Path(registry_dir).expanduser().resolve()
    binned_path = Path(binned_points_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = _read_contract_tables(registry)
    issues = _validate_tables(tables, records)
    if not issues.empty and issues["severity"].eq("error").any():
        issues.to_csv(output / "data_quality_issues.csv", index=False)
        write_json(
            output / "orthogonal_kinetics_qa.json",
            {
                "status": "blocked_data_quality",
                "n_blocking_data_quality_issues": int(
                    issues["severity"].eq("error").sum()
                ),
                "missing_values_inferred": False,
                "mechanistic_claim_allowed": False,
            },
        )
        raise ValueError(
            "Orthogonal kinetic registry contains blocking QA errors; see "
            f"{output / 'data_quality_issues.csv'}"
        )

    conditions = tables["orthogonal_kinetic_conditions.csv"]
    reaction = tables["apparent_reaction_orders.csv"]
    controlled = tables["controlled_formulation_registry.csv"]
    n_verified_pH, n_pH_series, eligible_pH_series = _pH_series_gate(
        conditions
    )
    eligible_reaction = _reaction_order_gate(reaction)
    eligible_controlled = _controlled_series_gate(controlled)

    binned_exists = binned_path.is_file()
    binned_hash = sha256_file(binned_path) if binned_exists else ""
    binned_hash_matches = bool(
        binned_exists and binned_hash == EXPECTED_BINNED_POINTS_SHA256
    )
    if binned_hash_matches:
        binned = pd.read_csv(binned_path)
        n_binned_points = int(len(binned))
        n_binned_curves = int(binned["curve_id"].nunique())
    else:
        n_binned_points = 0
        n_binned_curves = 0

    n_records = int(len(records))
    linked_verified_pH_records = int(
        conditions.loc[
            conditions["verification_status"].astype(str).eq("verified")
            & conditions["pH_origin"].astype(str).isin(
                {"measured", "reported", "calculated_activity_model"}
            ),
            "record_id",
        ]
        .dropna()
        .astype(str)
        .nunique()
    )
    pH_predictor_gate = bool(
        linked_verified_pH_records >= int(np.ceil(0.70 * n_records))
        and len(eligible_pH_series) > 0
    )
    coverage = pd.DataFrame(
        [
            {
                "evidence_stream": "source_verified_pH",
                "n_rows": n_verified_pH,
                "n_series": n_pH_series,
                "n_linked_records": linked_verified_pH_records,
                "status": (
                    "eligible_covariate"
                    if pH_predictor_gate
                    else "not_eligible_insufficient_verified_coverage"
                ),
            },
            {
                "evidence_stream": "apparent_reaction_order",
                "n_rows": int(len(eligible_reaction)),
                "n_series": int(
                    eligible_reaction["series_id"].nunique()
                    if not eligible_reaction.empty
                    else 0
                ),
                "n_linked_records": 0,
                "status": (
                    "eligible_auxiliary_outcome"
                    if len(eligible_reaction)
                    else "not_available"
                ),
            },
            {
                "evidence_stream": "local_slope_vs_potential_source",
                "n_rows": n_binned_points,
                "n_series": n_binned_curves,
                "n_linked_records": n_binned_curves,
                "status": (
                    "available_auxiliary_outcome_forbidden_as_predictor"
                    if binned_hash_matches and n_binned_curves == 97
                    else "blocked_source_hash_or_curve_count"
                ),
            },
            {
                "evidence_stream": "controlled_formulation_series",
                "n_rows": int(len(controlled)),
                "n_series": int(
                    controlled["controlled_series_id"].nunique()
                    if not controlled.empty
                    else 0
                ),
                "n_linked_records": int(
                    controlled["record_id"].dropna().astype(str).nunique()
                ),
                "status": (
                    "eligible_separate_analysis"
                    if eligible_controlled
                    else "not_available_or_support_gate_failed"
                ),
            },
        ]
    )
    eligibility = pd.DataFrame(
        [
            {
                "field_or_stream": "source_verified_pH_value",
                "eligible_for_cardinal_predictor": pH_predictor_gate,
                "eligible_for_auxiliary_analysis": bool(
                    eligible_pH_series
                ),
                "decision_reason": (
                    "requires source-verified values linked to >=70% of records "
                    "and at least one controlled three-level pH series"
                ),
            },
            {
                "field_or_stream": "apparent_reaction_order",
                "eligible_for_cardinal_predictor": False,
                "eligible_for_auxiliary_analysis": bool(
                    len(eligible_reaction)
                ),
                "decision_reason": (
                    "kinetic auxiliary outcome; never a same-experiment "
                    "formulation predictor"
                ),
            },
            {
                "field_or_stream": "local_slope_vs_potential",
                "eligible_for_cardinal_predictor": False,
                "eligible_for_auxiliary_analysis": bool(
                    binned_hash_matches and n_binned_curves == 97
                ),
                "decision_reason": (
                    "same-curve target-derived information is forbidden in "
                    "cardinal classification"
                ),
            },
            {
                "field_or_stream": "controlled_formulation_descriptors",
                "eligible_for_cardinal_predictor": False,
                "eligible_for_auxiliary_analysis": bool(
                    eligible_controlled
                ),
                "decision_reason": (
                    "reserved for a separate controlled-series analysis after "
                    "batch and independent-replicate gates"
                ),
            },
        ]
    )
    roles = orthogonal_feature_role_registry()
    acquisition_queue = pd.DataFrame(
        [
            {
                "priority": 1,
                "evidence_stream": "controlled_formulation_series",
                "candidate": "new factorial formulation experiments",
                "required_action": (
                    "collect >=4 formulation levels with >=3 independent ink "
                    "preparations/electrodes per level"
                ),
            },
            {
                "priority": 2,
                "evidence_stream": "pH_dependence",
                "candidate": "Record 59 source supplementary pH series",
                "required_action": (
                    "verify catalyst identity, import every level/curve, and "
                    "bind page/figure plus source hash"
                ),
            },
            {
                "priority": 3,
                "evidence_stream": "loading_control",
                "candidate": "Record 97 source 0.5/1 mg cm^-2 curves",
                "required_action": (
                    "digitize both source curves and verify all held constants"
                ),
            },
            {
                "priority": 4,
                "evidence_stream": "local_slope_vs_potential",
                "candidate": "frozen 97-curve binned coordinates",
                "required_action": (
                    "run a separately registered multi-window Bayesian "
                    "auxiliary-outcome analysis; do not use as a predictor"
                ),
            },
        ]
    )
    qa = {
        "status": "pass",
        "n_canonical_records": n_records,
        "template_registry_path": str(registry),
        "n_registry_rows": {
            filename: int(len(frame)) for filename, frame in tables.items()
        },
        "n_blocking_data_quality_issues": int(
            issues["severity"].eq("error").sum() if not issues.empty else 0
        ),
        "eligible_pH_series": eligible_pH_series,
        "eligible_reaction_order_rows": int(len(eligible_reaction)),
        "eligible_controlled_formulation_series": eligible_controlled,
        "eligible_predictor_columns": (
            ["pH_value"] if pH_predictor_gate else []
        ),
        "binned_points_path": str(binned_path),
        "binned_points_sha256": binned_hash,
        "binned_points_expected_sha256": EXPECTED_BINNED_POINTS_SHA256,
        "binned_points_hash_matches": binned_hash_matches,
        "n_binned_points": n_binned_points,
        "n_binned_curves": n_binned_curves,
        "slope_profile_role": "forbidden_target_derived",
        "mechanistic_claim_allowed": False,
        "missing_values_inferred": False,
    }

    coverage.to_csv(output / "evidence_coverage.csv", index=False)
    eligibility.to_csv(output / "predictor_eligibility.csv", index=False)
    issues.to_csv(output / "data_quality_issues.csv", index=False)
    roles.to_csv(output / "feature_role_registry.csv", index=False)
    acquisition_queue.to_csv(output / "acquisition_queue.csv", index=False)
    write_json(output / "orthogonal_kinetics_qa.json", qa)
    return OrthogonalKineticsAudit(
        tables=dict(tables),
        table_coverage=coverage,
        predictor_eligibility=eligibility,
        data_quality_issues=issues,
        feature_roles=roles,
        acquisition_queue=acquisition_queue,
        qa=qa,
    )


__all__ = [
    "EXPECTED_BINNED_POINTS_SHA256",
    "OrthogonalKineticsAudit",
    "audit_orthogonal_kinetics",
    "orthogonal_feature_role_registry",
]
