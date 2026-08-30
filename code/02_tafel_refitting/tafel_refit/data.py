from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .models import PRIMARY_MODEL_NAMES


class InputValidationError(RuntimeError):
    """Raised when a positional mapping cannot be proven safe."""


@dataclass(frozen=True)
class InputBundle:
    config: dict[str, Any]
    manifest: pd.DataFrame
    raw_points: pd.DataFrame
    master: pd.DataFrame
    workbook_sha256: str
    master_sha256: str
    mapping_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 2:
        raise InputValidationError(
            f"Unsupported config schema_version={config.get('schema_version')!r}"
        )
    if config.get("model_family") != "continuous_tafel":
        raise InputValidationError("Config must register model_family='continuous_tafel'")
    if config.get("primary_model_count") != len(PRIMARY_MODEL_NAMES):
        raise InputValidationError(
            "Config primary_model_count does not match the primary model registry"
        )
    if (
        config.get("primary_low_current_floor_policy")
        != "excluded_without_independent_blank"
    ):
        raise InputValidationError(
            "Primary low-current-floor policy must exclude the term without an "
            "independent blank"
        )
    return config


def normalize_doi(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip()


def _curve_id(sample_record: int, variant: int, variants: int) -> str:
    if variants == 1:
        return f"sr{sample_record:03d}"
    return f"sr{sample_record:03d}_v{variant}"


def _expected_mapping(master: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required = {
        "sample_record",
        "tafel_slope_source",
        "publication_id",
        "doi",
        "paper_name",
        "source_year",
        "y_tafel_slope_mV_dec",
    }
    missing = sorted(required.difference(master.columns))
    if missing:
        raise InputValidationError(f"Master CSV is missing required columns: {missing}")

    sample_numeric = pd.to_numeric(master["sample_record"], errors="coerce")
    if sample_numeric.isna().any():
        raise InputValidationError("sample_record contains non-numeric or missing values")
    master = master.copy()
    master["sample_record"] = sample_numeric.astype(int)

    if len(master) != 97 or not master["sample_record"].is_unique:
        raise InputValidationError(
            f"Expected 97 unique master rows; got rows={len(master)}, "
            f"unique={master['sample_record'].nunique()}"
        )
    if set(master["sample_record"]) != set(range(1, 98)):
        raise InputValidationError("sample_record must be exactly 1..97")

    source_counts = master["tafel_slope_source"].value_counts().to_dict()
    if source_counts != {"reported": 68, "reconstructed": 29}:
        raise InputValidationError(
            "Unexpected tafel_slope_source counts: " f"{source_counts}"
        )

    extra = {
        int(key): int(value)
        for key, value in config.get("extra_curve_sample_records", {}).items()
    }
    entries: list[dict[str, Any]] = []
    for source in ("reconstructed", "reported"):
        subset = master.loc[master["tafel_slope_source"].eq(source)].sort_values(
            "sample_record"
        )
        for _, row in subset.iterrows():
            sample_record = int(row["sample_record"])
            variants = extra.get(sample_record, 1)
            for variant in range(1, variants + 1):
                record = row.to_dict()
                record.update(
                    curve_id=_curve_id(sample_record, variant, variants),
                    curve_variant=variant,
                    curve_variants_for_sample=variants,
                    mapping_partition=source,
                )
                entries.append(record)

    mapping = pd.DataFrame(entries)
    if len(mapping) != int(config["source_sheet_count"]):
        raise InputValidationError(
            f"Expected mapping has {len(mapping)} rows, not "
            f"{config['source_sheet_count']}"
        )
    if not mapping["curve_id"].is_unique:
        raise InputValidationError("Generated curve_id values are not unique")
    return mapping.reset_index(drop=True)


def _coerce_finite_number(value: Any) -> tuple[float, bool]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan, False
    return number, bool(np.isfinite(number))


def load_inputs(
    workbook_path: Path,
    mapping_csv_path: Path,
    canonical_csv_path: Path,
    config_path: Path,
    *,
    allow_unverified_hashes: bool = False,
) -> InputBundle:
    workbook_path = workbook_path.resolve()
    mapping_csv_path = mapping_csv_path.resolve()
    canonical_csv_path = canonical_csv_path.resolve()
    config_path = config_path.resolve()
    for path in (workbook_path, mapping_csv_path, canonical_csv_path, config_path):
        if not path.is_file():
            raise InputValidationError(f"Required input does not exist: {path}")

    config = load_config(config_path)
    workbook_hash = sha256_file(workbook_path)
    mapping_hash = sha256_file(mapping_csv_path)
    canonical_hash = sha256_file(canonical_csv_path)
    expected_workbook_hash = config["expected_workbook_sha256"]
    expected_mapping_hash = config["expected_mapping_sha256"]
    expected_canonical_hash = config["expected_canonical_sha256"]
    if not allow_unverified_hashes:
        if workbook_hash != expected_workbook_hash:
            raise InputValidationError(
                "Workbook hash changed; positional mapping is unsafe. "
                f"Expected {expected_workbook_hash}, got {workbook_hash}."
            )
        if mapping_hash != expected_mapping_hash:
            raise InputValidationError(
                "Mapping CSV hash changed; positional mapping is unsafe. "
                f"Expected {expected_mapping_hash}, got {mapping_hash}."
            )
        if canonical_hash != expected_canonical_hash:
            raise InputValidationError(
                "Canonical CSV hash changed; record binding is unsafe. "
                f"Expected {expected_canonical_hash}, got {canonical_hash}."
            )

    mapping_master = pd.read_csv(mapping_csv_path)
    mapping = _expected_mapping(mapping_master, config)
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    expected_names = [
        f"Sheet{i}" for i in range(1, int(config["source_sheet_count"]) + 1)
    ]
    if workbook.sheetnames != expected_names:
        raise InputValidationError(
            "Workbook sheet names/order changed; positional mapping is unsafe"
        )

    manifest_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for offset, sheet_name in enumerate(workbook.sheetnames):
        sheet_index = offset + 1
        sheet = workbook[sheet_name]
        expected = mapping.iloc[offset].to_dict()
        potential_header = sheet["A1"].value
        current_header = sheet["B1"].value
        workbook_doi = sheet["G1"].value
        if potential_header != "Potential":
            raise InputValidationError(
                f"{sheet_name}!A1 changed: expected 'Potential', got {potential_header!r}"
            )
        if current_header not in {"Current density", "Current (A g^-1_Ir)"}:
            raise InputValidationError(
                f"{sheet_name}!B1 has an unreviewed current header: {current_header!r}"
            )

        master_doi = normalize_doi(expected.get("doi"))
        sheet_doi = normalize_doi(workbook_doi)
        mapping_flags: list[str] = []
        if master_doi and sheet_doi != master_doi:
            raise InputValidationError(
                f"DOI mismatch at {sheet_name}/{expected['curve_id']}: "
                f"master={master_doi!r}, workbook={sheet_doi!r}"
            )
        if not master_doi and sheet_doi:
            mapping_flags.append("doi_only_in_curve_workbook")
        if int(expected["sample_record"]) == 16:
            mapping_flags.append("missing_from_0619_refitting_table")
        if int(expected["sample_record"]) == 42:
            mapping_flags.append("two_curves_for_one_sample")

        finite_pairs = 0
        nonfinite_pairs = 0
        positive = 0
        zero = 0
        negative = 0
        potentials: list[float] = []
        currents: list[float] = []
        for excel_row, values in enumerate(
            sheet.iter_rows(min_row=2, max_col=2, values_only=True), start=2
        ):
            potential, potential_finite = _coerce_finite_number(values[0])
            current, current_finite = _coerce_finite_number(values[1])
            finite = potential_finite and current_finite
            if finite:
                finite_pairs += 1
                potentials.append(potential)
                currents.append(current)
                if current > 0:
                    positive += 1
                elif current == 0:
                    zero += 1
                else:
                    negative += 1
            else:
                nonfinite_pairs += 1
            point_rows.append(
                {
                    "curve_id": expected["curve_id"],
                    "sample_record": int(expected["sample_record"]),
                    "sheet_index": sheet_index,
                    "sheet_name": sheet_name,
                    "excel_row": excel_row,
                    "potential_raw": potential,
                    "current_raw": current,
                    "finite_pair": finite,
                    "positive_current": bool(finite and current > 0),
                }
            )

        unique_potential = len(np.unique(potentials)) if potentials else 0
        if potentials:
            p_min = float(np.min(potentials))
            p_max = float(np.max(potentials))
            p_span = p_max - p_min
            c_min = float(np.min(currents))
            c_max = float(np.max(currents))
        else:
            p_min = p_max = p_span = c_min = c_max = np.nan

        manifest_rows.append(
            {
                **expected,
                "sheet_index": sheet_index,
                "sheet_name": sheet_name,
                "source_file": str(workbook_path),
                "source_sha256": workbook_hash,
                "mapping_rule_version": config["mapping_rule_version"],
                "mapping_confidence": "verified_positional_and_doi"
                if master_doi
                else "verified_positional_only",
                "mapping_flags": ";".join(mapping_flags),
                "workbook_doi": sheet_doi,
                "potential_header_raw": potential_header,
                "potential_unit_raw": "unknown",
                "potential_unit_assumption": "V_inferred_from_numeric_scale_not_header_verified",
                "potential_reference_raw": "unknown",
                "current_header_raw": current_header,
                "current_unit_raw": "A g^-1_Ir"
                if current_header == "Current (A g^-1_Ir)"
                else "unknown_current_density",
                "ir_status": "unknown",
                "n_rows_raw": max(sheet.max_row - 1, 0),
                "n_finite_pairs": finite_pairs,
                "n_nonfinite_pairs": nonfinite_pairs,
                "n_positive_current": positive,
                "n_zero_current": zero,
                "n_negative_current": negative,
                "n_unique_potential": unique_potential,
                "potential_min_raw": p_min,
                "potential_max_raw": p_max,
                "potential_span_raw": p_span,
                "current_min_raw": c_min,
                "current_max_raw": c_max,
            }
        )

    workbook.close()
    manifest = pd.DataFrame(manifest_rows)
    raw_points = pd.DataFrame(point_rows)

    excluded = set(map(str, config.get("excluded_curve_ids", [])))
    manifest = manifest.loc[~manifest["curve_id"].astype(str).isin(excluded)].copy()
    raw_points = raw_points.loc[
        ~raw_points["curve_id"].astype(str).isin(excluded)
    ].copy()
    canonical = pd.read_csv(canonical_csv_path)
    if (
        len(canonical) != int(config["expected_curve_count"])
        or canonical["record_id"].nunique() != int(config["expected_curve_count"])
        or canonical["curve_id"].nunique() != int(config["expected_curve_count"])
    ):
        raise InputValidationError("Canonical registry must contain 97 one-to-one records")
    record_by_curve = canonical.set_index("curve_id")["record_id"].astype(int)
    if set(manifest["curve_id"].astype(str)) != set(record_by_curve.index.astype(str)):
        raise InputValidationError("Curve workbook and canonical registry differ")
    manifest["sample_record"] = manifest["curve_id"].map(record_by_curve).astype(int)
    manifest["curve_variant"] = 1
    manifest["curve_variants_for_sample"] = 1
    raw_points["sample_record"] = raw_points["curve_id"].map(record_by_curve).astype(int)
    canonical = canonical.copy()
    canonical["sample_record"] = canonical["record_id"].astype(int)

    header_counts = manifest["current_header_raw"].value_counts().to_dict()
    expected_headers = {"Current density": 93, "Current (A g^-1_Ir)": 4}
    if header_counts != expected_headers:
        raise InputValidationError(
            f"Current-header counts changed: expected {expected_headers}, got {header_counts}"
        )
    expected_finite_pairs = int(config["expected_finite_pair_count"])
    if manifest["n_finite_pairs"].sum() != expected_finite_pairs:
        raise InputValidationError(
            "Finite-pair count changed; input requires a new audit before positional "
            f"mapping (expected {expected_finite_pairs})"
        )
    expected_nonfinite_pairs = int(config["expected_nonfinite_pair_count"])
    if manifest["n_nonfinite_pairs"].sum() != expected_nonfinite_pairs:
        raise InputValidationError(
            "Non-finite-pair count changed; input requires a new audit "
            f"(expected {expected_nonfinite_pairs})"
        )
    expected_nonpositive = int(config["expected_nonpositive_current_count"])
    observed_nonpositive = int(
        (manifest["n_zero_current"] + manifest["n_negative_current"]).sum()
    )
    if observed_nonpositive != expected_nonpositive:
        raise InputValidationError(
            "Non-positive-current count changed; input requires a new audit "
            f"(expected {expected_nonpositive}, got {observed_nonpositive})"
        )

    return InputBundle(
        config=config,
        manifest=manifest,
        raw_points=raw_points,
        master=canonical,
        workbook_sha256=workbook_hash,
        master_sha256=canonical_hash,
        mapping_sha256=mapping_hash,
    )
