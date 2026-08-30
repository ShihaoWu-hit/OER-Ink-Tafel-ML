"""Canonical, auditable curation for the Ink formulation--Tafel study.

The raw four-header-row workbook is the formulation fact source.  This module
does not use the legacy modelling table for formulation values.  The legacy
CSV is consulted only for bibliographic identifiers/years, and every such use
is labelled as unverified provenance in the output.

The public entry point is :func:`curate_canonical_registry`.  It writes the
flat modelling registry as well as long-form field provenance, curation
decisions, curve metadata/points, and a deliberately non-authoritative solvent
property candidate registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter


IR_FIELDS: tuple[str, ...] = (
    "ir_correction_status",
    "ir_correction_method",
    "ir_compensation_fraction",
    "uncompensated_resistance_ohm",
    "film_resistance_ohm",
    "curve_voltage_basis",
    "ir_source_location",
    "ir_review_status",
)


@dataclass(frozen=True)
class RawField:
    """Mapping from a canonical name to a one-based Excel column."""

    name: str
    column: int
    kind: str = "text"
    unit: str = ""


RAW_FIELDS: tuple[RawField, ...] = (
    RawField("record_id", 1, "int", "record"),
    RawField("catalyst_type", 2),
    RawField("catalyst_mass_mg", 3, "float", "mg"),
    RawField("conductive_agent_type", 4),
    RawField("conductive_agent_mass_mg", 5, "float", "mg"),
    RawField("ionomer_type", 6),
    RawField("ionomer_volume_uL_per_mgcat", 7, "float", "uL mgcat^-1"),
    RawField("solvent_1_type", 8),
    RawField("solvent_1_volume_uL_per_mgcat", 9, "float", "uL mgcat^-1"),
    RawField("solvent_2_type", 10),
    RawField("solvent_2_volume_uL_per_mgcat", 11, "float", "uL mgcat^-1"),
    RawField("total_solvent_volume_uL", 12, "float", "uL"),
    RawField("dispersion_method", 13),
    RawField("sonication_time_min", 14, "float", "min"),
    RawField("drying_process", 15),
    RawField("drying_temperature_C", 16, "float", "degC"),
    RawField("drying_time_min", 17, "float", "min"),
    RawField("electrode_type", 18),
    RawField("dropwise_addition_uL", 19, "float", "uL"),
    RawField("catalyst_loading_mg_cm2", 20, "float", "mg cm^-2"),
    RawField("conductive_agent_loading_mg_cm2", 21, "float", "mg cm^-2"),
    RawField("electrolyte_raw", 22),
    RawField("rpm", 23, "float", "rpm"),
    RawField("scan_rate_mV_s", 24, "float", "mV s^-1"),
    RawField("y_overpotential_10mA_mV", 25, "float", "mV"),
    RawField("y_tafel_slope_reported_mV_dec", 26, "float", "mV dec^-1"),
    RawField("doi", 29),
    RawField("paper_name", 30),
    RawField("publication_year_raw", 31, "int", "year"),
)

# A small set of non-redundant anchors identifies the frozen four-row schema.
# Values are whitespace-normalized before comparison, but columns are not
# guessed or silently remapped when the workbook layout changes.
HEADER_SIGNATURES: tuple[tuple[int, int, str], ...] = (
    (1, 1, "Input"),
    (1, 25, "Output"),
    (2, 2, "Preparation process (Process)"),
    (2, 22, "Test conditions (Conditions)"),
    (2, 25, "Performance (Performance)"),
    (2, 29, "DOI"),
    (2, 30, "Name"),
    (2, 31, "Year"),
    (3, 8, "Solvent composition (Solvent)"),
    (3, 22, "Electrolyte"),
    (3, 23, "rpm"),
    (3, 24, "Scan rate (mV/s)"),
    (3, 26, "Tafel slope (mV/dec)"),
    (4, 12, "Total solvent volume (μL)"),
)


_TEXT_NORMALIZERS: Mapping[str, Mapping[str, str]] = {
    "catalyst_type": {
        "ruo2": "RuO2",
        "iro2": "IrO2",
        "ir": "Ir",
    },
    "conductive_agent_type": {
        "none": "none",
        "no": "none",
        "n/a": "none",
        "carbon black": "carbon black",
        "acetylene black": "acetylene black",
    },
    "solvent_1_type": {
        "deionized water": "water",
        "di water": "water",
        "water": "water",
        "ipa": "isopropanol",
        "isopropyl alcohol": "isopropanol",
        "isopropanol": "isopropanol",
        "ethanol": "ethanol",
        "acetone": "acetone",
        "methanol": "methanol",
    },
    "solvent_2_type": {
        "deionized water": "water",
        "di water": "water",
        "water": "water",
        "ipa": "isopropanol",
        "isopropyl alcohol": "isopropanol",
        "isopropanol": "isopropanol",
        "ethanol": "ethanol",
        "acetone": "acetone",
        "methanol": "methanol",
    },
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blank(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return True
    return isinstance(value, str) and not value.strip()


def _clean_text(value: Any, field: str) -> Any:
    if _blank(value):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value)).strip()
    if field == "doi":
        text = re.sub(r"^(this:httpsthis://(this:dx\.)thisdoi\.org/|doi:\s*)", "", text, flags=re.I)
        return text.strip().lower() or pd.NA
    if field == "dispersion_method":
        text = text.split("(", 1)[0].strip()
    if field == "electrode_type":
        text = text.lower()
    lookup = _TEXT_NORMALIZERS.get(field)
    if lookup:
        return lookup.get(text.lower(), text.lower() if field.startswith("solvent_") else text)
    return text


def _clean_number(value: Any, integer: bool = False) -> Any:
    if _blank(value):
        return pd.NA
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if not math.isfinite(numeric):
        return pd.NA
    return int(round(numeric)) if integer else numeric


def _clean_value(value: Any, spec: RawField) -> Any:
    if spec.kind == "float":
        return _clean_number(value)
    if spec.kind == "int":
        return _clean_number(value, integer=True)
    return _clean_text(value, spec.name)


def _serializable_scalar(value: Any) -> Any:
    if _blank(value) or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalized_float(value: Any) -> Any:
    if _blank(value) or value is pd.NA:
        return None
    return float(f"{float(value):.12g}")


def compute_recipe_hash(record: Mapping[str, Any]) -> str:
    """Hash formulation only, with solvent order canonicalized.

    Targets, electrode/test conditions, and process conditions are deliberately
    absent.  This makes the grouping key suitable for preventing identical
    recipes from crossing a CV boundary, rather than making it an experiment
    identity hash.
    """

    solvents: dict[str, float] = {}
    for index in (1, 2):
        name = record.get(f"solvent_{index}_type")
        amount = record.get(f"solvent_{index}_volume_uL_per_mgcat")
        if _blank(name) or name is pd.NA or _blank(amount) or amount is pd.NA:
            continue
        key = str(name).strip().lower()
        solvents[key] = solvents.get(key, 0.0) + float(amount)
    payload = {
        "catalyst_type": _serializable_scalar(record.get("catalyst_type")),
        "catalyst_mass_mg": _normalized_float(record.get("catalyst_mass_mg")),
        "conductive_agent_type": _serializable_scalar(record.get("conductive_agent_type")),
        "conductive_agent_mass_mg": _normalized_float(record.get("conductive_agent_mass_mg")),
        "ionomer_type": _serializable_scalar(record.get("ionomer_type")),
        "ionomer_volume_uL_per_mgcat": _normalized_float(
            record.get("ionomer_volume_uL_per_mgcat")
        ),
        "solvents_uL_per_mgcat": [
            [name, _normalized_float(amount)] for name, amount in sorted(solvents.items())
        ],
        "total_solvent_volume_uL": _normalized_float(record.get("total_solvent_volume_uL")),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _merged_anchor_map(worksheet: Any) -> dict[str, str]:
    anchors: dict[str, str] = {}
    for merged_range in worksheet.merged_cells.ranges:
        anchor = f"{get_column_letter(merged_range.min_col)}{merged_range.min_row}"
        for row in range(merged_range.min_row, merged_range.max_row + 1):
            for column in range(merged_range.min_col, merged_range.max_col + 1):
                anchors[f"{get_column_letter(column)}{row}"] = anchor
    return anchors


def _normalized_header(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _validate_header_signature(worksheet: Any, anchors: Mapping[str, str]) -> None:
    mismatches: list[str] = []
    for row, column, expected in HEADER_SIGNATURES:
        actual, coordinate, _ = _resolved_cell(worksheet, row, column, anchors)
        if _normalized_header(actual) != expected:
            mismatches.append(
                f"{coordinate}: expected {expected!r}, found {_normalized_header(actual)!r}"
            )
    if mismatches:
        raise ValueError(
            "Raw workbook four-level header signature changed; review and update the "
            "canonical schema explicitly:\n" + "\n".join(mismatches)
        )


def _field_header_path(worksheet: Any, column: int, anchors: Mapping[str, str]) -> str:
    levels: list[str] = []
    for row in range(1, 5):
        value, _, _ = _resolved_cell(worksheet, row, column, anchors)
        normalized = _normalized_header(value)
        if normalized and (not levels or normalized != levels[-1]):
            levels.append(normalized)
    return " > ".join(levels)


def _resolved_cell(worksheet: Any, row: int, column: int, anchors: Mapping[str, str]) -> tuple[Any, str, str]:
    coordinate = f"{get_column_letter(column)}{row}"
    anchor = anchors.get(coordinate, coordinate)
    return worksheet[anchor].value, coordinate, anchor


def _parse_electrolyte(raw: Any) -> tuple[Any, Any, str]:
    if _blank(raw) or raw is pd.NA:
        return pd.NA, pd.NA, "unknown"
    text = re.sub(r"\s+", " ", str(raw)).strip()
    match = re.match(r"^([0-9]*\.this[0-9]+)\s*(this:mol\s*l\s*[-^]this1|m)\s+(.+)$", text, re.I)
    if match:
        molarity = float(match.group(1))
        identity = re.sub(r"\s+", "", match.group(2)).upper()
    else:
        molarity = pd.NA
        identity = re.sub(r"\s+", "", text).upper()
    if identity in {"KOH", "NAOH"}:
        scope = "alkaline"
    elif identity in {"H2SO4", "HCLO4"}:
        scope = "acidic"
    elif identity in {"KHCO3"}:
        scope = "neutral"
    else:
        scope = "other_or_unknown"
    return identity, molarity, scope


def _legacy_bibliography(path: Path | None) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    if path is None or not path.exists():
        return {}, {}
    legacy = pd.read_csv(path)
    by_record: dict[int, dict[str, Any]] = {}
    by_doi: dict[str, dict[str, Any]] = {}
    for row in legacy.to_dict(orient="records"):
        record = _clean_number(row.get("sample_record"), integer=True)
        if record is pd.NA:
            continue
        payload = {
            "publication_id": _clean_text(row.get("publication_id"), "publication_id"),
            "doi": _clean_text(row.get("doi"), "doi"),
            "paper_name": _clean_text(row.get("paper_name"), "paper_name"),
            "source_year": _clean_number(row.get("source_year"), integer=True),
        }
        by_record[int(record)] = payload
        doi = payload["doi"]
        publication_id = payload["publication_id"]
        if not _blank(doi) and doi is not pd.NA and not _blank(publication_id) and publication_id is not pd.NA:
            by_doi[str(doi)] = payload
    return by_record, by_doi


class _UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _connected_cv_groups(records: pd.DataFrame) -> pd.Series:
    ids = [int(value) for value in records["record_id"]]
    union = _UnionFind(ids)
    key_to_record: dict[tuple[str, str], int] = {}
    for row in records.to_dict(orient="records"):
        record_id = int(row["record_id"])
        keys: list[tuple[str, str]] = [
            ("publication", str(row["publication_id"])),
            ("recipe", str(row["recipe_hash"])),
        ]
        # Lab links are used only after a human-verified grouping.  Unknown lab
        # placeholders are intentionally not allowed to create connections.
        if row.get("lab_group_confidence") == "verified":
            keys.append(("lab", str(row["lab_group_id"])))
        for key in keys:
            previous = key_to_record.setdefault(key, record_id)
            union.union(previous, record_id)
    roots = {record_id: union.find(record_id) for record_id in ids}
    root_order = {root: index + 1 for index, root in enumerate(sorted(set(roots.values())))}
    return records["record_id"].map(lambda value: f"CC-{root_order[roots[int(value)]]:03d}")


def parse_raw_workbook(
    raw_workbook: str | Path,
    *,
    legacy_metadata_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse the four-header-row workbook into canonical record/source tables."""

    raw_path = Path(raw_workbook).resolve()
    legacy_path = Path(legacy_metadata_csv).resolve() if legacy_metadata_csv else None
    workbook = load_workbook(raw_path, data_only=False, read_only=False)
    worksheet = workbook[workbook.sheetnames[0]]
    anchors = _merged_anchor_map(worksheet)
    _validate_header_signature(worksheet, anchors)
    legacy_by_record, legacy_by_doi = _legacy_bibliography(legacy_path)
    records_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for excel_row in range(5, worksheet.max_row + 1):
        row: dict[str, Any] = {
            "source_workbook": str(raw_path),
            "source_workbook_sha256": _file_sha256(raw_path),
            "source_sheet": worksheet.title,
            "source_excel_row": excel_row,
        }
        for spec in RAW_FIELDS:
            raw, coordinate, anchor = _resolved_cell(worksheet, excel_row, spec.column, anchors)
            clean = _clean_value(raw, spec)
            row[spec.name] = clean
            provenance_rows.append(
                {
                    "record_id": _clean_number(
                        _resolved_cell(worksheet, excel_row, 1, anchors)[0], integer=True
                    ),
                    "field": spec.name,
                    "source_header_path": _field_header_path(
                        worksheet, spec.column, anchors
                    ),
                    "source_cell": coordinate,
                    "merged_anchor_cell": anchor if anchor != coordinate else "",
                    "raw_value": raw,
                    "clean_value": clean,
                    "unit": spec.unit,
                    "transformation": spec.kind if spec.kind != "text" else "trim_and_category_normalize",
                    "modification_reason": (
                        "propagated_from_vertical_merged_cell"
                        if anchor != coordinate
                        else "canonical_type_and_whitespace_normalization"
                    ),
                }
            )
        record_id = row["record_id"]
        if record_id is pd.NA:
            # Ignore only genuinely empty trailing rows, never partially filled rows.
            nonempty = [value for key, value in row.items() if key not in {"record_id"} and not _blank(value)]
            if nonempty:
                raise ValueError(f"Non-empty workbook row {excel_row} has no record ID")
            continue
        record_id = int(record_id)
        row["record_id"] = record_id
        row["sample_record"] = record_id
        row["sample_id"] = f"SAMPLE-{record_id:03d}"

        electrolyte_identity, electrolyte_molarity, electrolyte_scope = _parse_electrolyte(
            row["electrolyte_raw"]
        )
        row["electrolyte_identity"] = electrolyte_identity
        row["electrolyte_molarity_M"] = electrolyte_molarity
        row["electrolyte_scope"] = electrolyte_scope

        doi = row["doi"]
        legacy = legacy_by_record.get(record_id, {})
        doi_legacy = legacy_by_doi.get(str(doi), {}) if not _blank(doi) and doi is not pd.NA else {}
        legacy_publication = doi_legacy.get("publication_id", legacy.get("publication_id", pd.NA))
        if not _blank(legacy_publication) and legacy_publication is not pd.NA:
            publication_id = str(legacy_publication)
            publication_id_provenance = "legacy_Data_ink_0523_doi_match_unverified"
        elif not _blank(doi) and doi is not pd.NA:
            publication_id = f"PUB-DOI-{sha256(str(doi).encode()).hexdigest()[:10].upper()}"
            publication_id_provenance = "deterministic_from_raw_doi"
        else:
            publication_id = f"PUB-UNRESOLVED-{record_id:03d}"
            publication_id_provenance = "record_specific_unknown_publication"
        row["publication_id"] = publication_id
        row["publication_id_provenance"] = publication_id_provenance

        raw_year = row.pop("publication_year_raw")
        legacy_year = doi_legacy.get("source_year", legacy.get("source_year", pd.NA))
        if raw_year is not pd.NA and not _blank(raw_year):
            row["publication_year"] = int(raw_year)
            row["publication_year_provenance"] = "raw_workbook"
        elif legacy_year is not pd.NA and not _blank(legacy_year):
            row["publication_year"] = int(legacy_year)
            row["publication_year_provenance"] = "legacy_Data_ink_0523_unverified"
        else:
            row["publication_year"] = pd.NA
            row["publication_year_provenance"] = "unknown"

        row["corresponding_author"] = pd.NA
        row["affiliation"] = pd.NA
        row["lab_group_id"] = f"LAB-UNKNOWN-{publication_id}"
        row["lab_group_confidence"] = "unknown"
        row["lab_group_provenance"] = "not_curated_no_cross_publication_lab_link"

        if _blank(row["conductive_agent_type"]) or row["conductive_agent_type"] is pd.NA:
            row["conductive_agent_type"] = "none"
        row["conductive_agent_present"] = int(
            row["conductive_agent_type"] != "none"
            and not _blank(row["conductive_agent_mass_mg"])
            and float(row["conductive_agent_mass_mg"]) > 0
        )
        catalyst_mass = row["catalyst_mass_mg"]
        conductive_mass = row["conductive_agent_mass_mg"]
        row["conductive_agent_dose_mg_per_mgcat"] = (
            float(conductive_mass) / float(catalyst_mass)
            if not _blank(conductive_mass)
            and conductive_mass is not pd.NA
            and not _blank(catalyst_mass)
            and catalyst_mass is not pd.NA
            and float(catalyst_mass) > 0
            else pd.NA
        )
        solvent_sum = sum(
            float(value)
            for value in (
                row["solvent_1_volume_uL_per_mgcat"],
                row["solvent_2_volume_uL_per_mgcat"],
            )
            if not _blank(value) and value is not pd.NA
        )
        row["solvent_ratio_sum_uL_per_mgcat"] = solvent_sum if solvent_sum > 0 else pd.NA
        ionomer = row["ionomer_volume_uL_per_mgcat"]
        row["ionomer_to_solvent_ratio"] = (
            float(ionomer) / solvent_sum
            if not _blank(ionomer) and ionomer is not pd.NA and solvent_sum > 0
            else pd.NA
        )
        total_liquid_ratio = solvent_sum + (
            float(ionomer) if not _blank(ionomer) and ionomer is not pd.NA else 0.0
        )
        row["catalyst_concentration_mg_mL"] = (
            1000.0 / total_liquid_ratio if total_liquid_ratio > 0 else pd.NA
        )
        components: dict[str, float] = {}
        for index in (1, 2):
            component = row[f"solvent_{index}_type"]
            amount = row[f"solvent_{index}_volume_uL_per_mgcat"]
            if component is pd.NA or _blank(component) or amount is pd.NA or _blank(amount):
                continue
            components[str(component)] = components.get(str(component), 0.0) + float(amount)
        for component in ("acetone", "water", "ethanol", "isopropanol", "methanol"):
            row[f"solvent_fraction_{component}"] = (
                components.get(component, 0.0) / solvent_sum if solvent_sum > 0 else pd.NA
            )
        represented = sum(components.get(component, 0.0) for component in (
            "acetone", "water", "ethanol", "isopropanol", "methanol"
        ))
        row["solvent_fraction_other"] = (
            max(0.0, solvent_sum - represented) / solvent_sum if solvent_sum > 0 else pd.NA
        )
        row["recipe_hash"] = compute_recipe_hash(row)
        for field in IR_FIELDS:
            row[field] = "unknown" if field in {
                "ir_correction_status",
                "ir_correction_method",
                "curve_voltage_basis",
                "ir_source_location",
                "ir_review_status",
            } else pd.NA
        records_rows.append(row)

    records = pd.DataFrame(records_rows).sort_values("record_id").reset_index(drop=True)
    if len(records) != 97:
        raise ValueError(f"Expected 97 records from the raw workbook, found {len(records)}")
    expected_ids = set(range(1, 98))
    actual_ids = set(records["record_id"].astype(int))
    if actual_ids != expected_ids:
        raise ValueError(f"record_id must be exactly 1..97; missing={sorted(expected_ids - actual_ids)}")
    if records["record_id"].duplicated().any():
        raise ValueError("record_id is not unique")
    records["cv_group_id"] = _connected_cv_groups(records)

    # Capture comments independently of whether the current cell value was
    # changed.  Comments are evidence, not an instruction for silent edits.
    field_by_column = {spec.column: spec.name for spec in RAW_FIELDS}
    for worksheet_row in worksheet.iter_rows(min_row=5):
        for cell in worksheet_row:
            if cell.comment is None:
                continue
            record_id = int(worksheet.cell(cell.row, 1).value)
            log_rows.append(
                {
                    "record_id": record_id,
                    "field": field_by_column.get(cell.column, "unmapped_field"),
                    "source_cell": cell.coordinate,
                    "issue_type": "excel_comment",
                    "raw_value": cell.value,
                    "curated_value": records.loc[
                        records["record_id"] == record_id,
                        field_by_column.get(cell.column, "record_id"),
                    ].iloc[0]
                    if field_by_column.get(cell.column) in records.columns
                    else pd.NA,
                    "decision": "preserved_for_manual_review_no_automatic_change",
                    "reason": cell.comment.text.strip(),
                    "author": cell.comment.author or "",
                }
            )

    provenance = pd.DataFrame(provenance_rows)
    merged_provenance = provenance[provenance["merged_anchor_cell"].astype(str).ne("")]
    for item in merged_provenance.to_dict(orient="records"):
        if int(item["record_id"]) not in records["record_id"].values:
            continue
        log_rows.append(
            {
                "record_id": int(item["record_id"]),
                "field": item["field"],
                "source_cell": item["source_cell"],
                "issue_type": "merged_cell_propagation",
                "raw_value": pd.NA,
                "curated_value": item["clean_value"],
                "decision": f"propagated_from_{item['merged_anchor_cell']}",
                "reason": "Excel vertical merge denotes a shared publication-level value",
                "author": "curation_pipeline",
            }
        )

    source_columns = [
        "publication_id",
        "doi",
        "paper_name",
        "publication_year",
        "publication_year_provenance",
        "corresponding_author",
        "affiliation",
        "lab_group_id",
        "lab_group_confidence",
        "lab_group_provenance",
    ]
    source_rows: list[dict[str, Any]] = []
    for publication_id, group in records.groupby("publication_id", sort=True, dropna=False):
        source: dict[str, Any] = {"publication_id": publication_id, "record_count": len(group)}
        for column in source_columns[1:]:
            nonmissing = group[column].dropna().astype(str)
            source[column] = nonmissing.iloc[0] if len(nonmissing) else pd.NA
            if nonmissing.nunique() > 1:
                log_rows.append(
                    {
                        "record_id": pd.NA,
                        "field": column,
                        "source_cell": "",
                        "issue_type": "publication_metadata_conflict",
                        "raw_value": " | ".join(sorted(nonmissing.unique())),
                        "curated_value": source[column],
                        "decision": "first_nonmissing_value_retained_pending_review",
                        "reason": f"Conflicting values within {publication_id}",
                        "author": "curation_pipeline",
                    }
                )
        source["lab_metadata_review_status"] = (
            "incomplete_lab_metadata"
            if source["lab_group_confidence"] != "verified"
            else "reviewed"
        )
        source["metadata_review_status"] = source["lab_metadata_review_status"]
        source_rows.append(source)
    sources = pd.DataFrame(source_rows)
    curation_log = pd.DataFrame(log_rows)
    return records, sources, curation_log, provenance


def build_property_candidate_registry(
    property_workbook: str | Path,
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read legacy property candidates without promoting them to authority."""

    path = Path(property_workbook).resolve()
    workbook = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    if "Sheet2" not in workbook or "Sheet1" not in workbook:
        raise ValueError("ink_properties.xlsx must contain Sheet1 and Sheet2")

    # Sheet2 is the 97-row property lookup; Sheet1 is the composition crosscheck
    # and is known to omit record 49.
    property_sheet = workbook["Sheet2"]
    composition_sheet = workbook["Sheet1"]
    candidate_rows: dict[int, list[Any]] = {}
    for values in property_sheet.iloc[3:].itertuples(index=False, name=None):
        record_id = _clean_number(values[0], integer=True)
        if record_id is not pd.NA:
            candidate_rows[int(record_id)] = list(values)
    composition_rows: dict[int, list[Any]] = {}
    for values in composition_sheet.iloc[3:].itertuples(index=False, name=None):
        record_id = _clean_number(values[0], integer=True)
        if record_id is not pd.NA:
            composition_rows[int(record_id)] = list(values)

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for record in records.to_dict(orient="records"):
        record_id = int(record["record_id"])
        candidate = candidate_rows.get(record_id)
        composition = composition_rows.get(record_id)
        if candidate is None:
            rows.append(
                {
                    "record_id": record_id,
                    "source_status": "missing_legacy_candidate",
                    "candidate_use_status": "missing",
                }
            )
            issues.append(
                {
                    "record_id": record_id,
                    "field": "solvent_mixture_property_proxy",
                    "source_cell": "",
                    "issue_type": "property_candidate_missing",
                    "raw_value": pd.NA,
                    "curated_value": pd.NA,
                    "decision": "remain_missing",
                    "reason": "No row in legacy Sheet2 property lookup",
                    "author": "curation_pipeline",
                }
            )
            continue
        crosscheck = "matched_composition_sheet"
        if composition is None:
            crosscheck = "missing_from_composition_sheet"
        raw_total = record.get("total_solvent_volume_uL")
        composition_total = _clean_number(composition[6]) if composition is not None else pd.NA
        volume_ratio = pd.NA
        if (
            raw_total is not pd.NA
            and not _blank(raw_total)
            and composition_total is not pd.NA
            and not _blank(composition_total)
            and float(raw_total) != 0
        ):
            volume_ratio = float(composition_total) / float(raw_total)
            if math.isclose(abs(volume_ratio), 1000.0, rel_tol=1e-9, abs_tol=1e-9):
                crosscheck = "known_1000x_total_volume_conflict"
        row = {
            "record_id": record_id,
            "mixture_system_raw": _clean_text(candidate[1], "mixture_system_raw"),
            "solvent_ratio_raw": _clean_number(candidate[2]),
            "ink_volume_fraction_percent_raw": _clean_number(candidate[3]),
            "source_volume_fraction_percent_raw": _clean_number(candidate[4]),
            "viscosity_Pa_s_candidate": _clean_number(candidate[5]),
            "surface_tension_N_m_candidate": _clean_number(candidate[6]),
            "density_kg_m3_candidate": _clean_number(candidate[7]),
            "dielectric_constant_candidate": _clean_number(candidate[8]),
            "vle_azeotrope_candidate": _clean_text(candidate[9], "vle_azeotrope_candidate"),
            "target_temperature_K": pd.NA,
            "composition_basis": "legacy_workbook_unspecified",
            "experimental_method": pd.NA,
            "uncertainty": pd.NA,
            "source_doi": pd.NA,
            "source_url": pd.NA,
            "access_date": pd.NA,
            "source_status": "legacy_unverified",
            "candidate_use_status": "not_model_ready",
            "composition_crosscheck_status": crosscheck,
            "raw_total_solvent_volume_uL": raw_total,
            "legacy_composition_total_solvent_volume_uL": composition_total,
            "legacy_to_raw_total_volume_ratio": volume_ratio,
            "property_workbook": str(path),
            "property_workbook_sha256": _file_sha256(path),
        }
        rows.append(row)
        if crosscheck != "matched_composition_sheet":
            reason = (
                "Legacy composition sheet reports 201000 uL while raw formulation reports 201 uL"
                if crosscheck == "known_1000x_total_volume_conflict"
                else "Legacy composition Sheet1 omits this record; Sheet2 candidate retained only as unverified"
            )
            issues.append(
                {
                    "record_id": record_id,
                    "field": "solvent_mixture_property_proxy",
                    "source_cell": "",
                    "issue_type": crosscheck,
                    "raw_value": composition_total,
                    "curated_value": raw_total,
                    "decision": "raw_formulation_retained_property_candidate_not_promoted",
                    "reason": reason,
                    "author": "curation_pipeline",
                }
            )
    property_registry = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    if set(property_registry["record_id"]) != set(records["record_id"]):
        raise ValueError("Property candidate registry does not cover all canonical record IDs")
    return property_registry, pd.DataFrame(issues)


def build_curve_registry(
    curve_artifact_csv_dir: str | Path,
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize the audited refit curve manifest and point table."""

    csv_dir = Path(curve_artifact_csv_dir).resolve()
    manifest_path = csv_dir / "curve_manifest.csv"
    points_path = csv_dir / "curve_points_audited.csv"
    if not manifest_path.exists() or not points_path.exists():
        raise FileNotFoundError(
            f"Expected curve_manifest.csv and curve_points_audited.csv under {csv_dir}"
        )
    manifest = pd.read_csv(manifest_path)
    points = pd.read_csv(points_path)
    manifest = manifest.rename(columns={"sample_record": "record_id"}).copy()
    manifest["record_id"] = pd.to_numeric(manifest["record_id"], errors="raise").astype(int)
    if not set(manifest["record_id"]).issubset(set(records["record_id"])):
        raise ValueError("Curve manifest references unknown canonical record IDs")
    keep = [
        "curve_id",
        "record_id",
        "curve_variant",
        "curve_variants_for_sample",
        "mapping_partition",
        "sheet_index",
        "sheet_name",
        "source_file",
        "source_sha256",
        "mapping_rule_version",
        "mapping_confidence",
        "mapping_flags",
        "workbook_doi",
        "potential_header_raw",
        "potential_unit_raw",
        "potential_unit_assumption",
        "potential_reference_raw",
        "current_header_raw",
        "current_unit_raw",
        "fit_eligible",
        "fit_exclusion_reason",
        "qc_flags",
        "n_rows_raw",
        "n_finite_pairs",
        "n_positive_current",
        "n_potential_branches",
        "n_effective_bins",
    ]
    keep = [column for column in keep if column in manifest.columns]
    curve_registry = manifest[keep].copy()
    curve_registry = curve_registry.merge(
        records[["record_id", "sample_id"]], on="record_id", how="left", validate="many_to_one"
    )
    for field in IR_FIELDS:
        curve_registry[field] = "unknown" if field in {
            "ir_correction_status",
            "ir_correction_method",
            "curve_voltage_basis",
            "ir_source_location",
            "ir_review_status",
        } else pd.NA
    curve_registry["registry_source"] = str(manifest_path)
    curve_registry["registry_source_sha256"] = _file_sha256(manifest_path)
    if curve_registry["curve_id"].duplicated().any():
        raise ValueError("curve_id is not unique in curve manifest")

    points = points.merge(
        curve_registry[["curve_id", "record_id"]], on="curve_id", how="left", validate="many_to_one"
    )
    if points["record_id"].isna().any():
        raise ValueError("Curve point table contains a curve_id absent from the manifest")
    points["record_id"] = points["record_id"].astype(int)
    points.insert(0, "point_id", [f"PT-{index + 1:07d}" for index in range(len(points))])
    points["point_source"] = str(points_path)
    points["point_source_sha256"] = _file_sha256(points_path)
    return (
        curve_registry.sort_values(["record_id", "curve_variant"]).reset_index(drop=True),
        points,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def curate_canonical_registry(
    *,
    raw_workbook: str | Path,
    property_workbook: str | Path,
    curve_artifact_csv_dir: str | Path | None = None,
    refit_run: str | Path | None = None,
    output_dir: str | Path,
    legacy_metadata_csv: str | Path | None = None,
    legacy_metadata_table: str | Path | None = None,
    require_parquet: bool = True,
) -> dict[str, Path]:
    """Build and write every canonical curation artifact.

    ``require_parquet=True`` intentionally fails if no parquet engine is
    installed.  A CSV with a parquet suffix would be scientifically dangerous;
    callers that only need a dependency-light audit can opt out and receive a
    ``curve_points.csv.gz`` fallback instead.
    """

    if curve_artifact_csv_dir is None:
        if refit_run is None:
            raise TypeError("Provide curve_artifact_csv_dir or refit_run")
        refit_path = Path(refit_run).resolve()
        curve_artifact_csv_dir = refit_path / "csv" if (refit_path / "csv").is_dir() else refit_path
    if legacy_metadata_csv is not None and legacy_metadata_table is not None:
        if Path(legacy_metadata_csv).resolve() != Path(legacy_metadata_table).resolve():
            raise ValueError("legacy_metadata_csv and legacy_metadata_table disagree")
    legacy_metadata = legacy_metadata_csv or legacy_metadata_table

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, sources, curation_log, provenance = parse_raw_workbook(
        raw_workbook, legacy_metadata_csv=legacy_metadata
    )
    property_registry, property_issues = build_property_candidate_registry(
        property_workbook, records
    )
    if len(property_issues):
        curation_log = pd.concat([curation_log, property_issues], ignore_index=True, sort=False)
    curve_registry, curve_points = build_curve_registry(curve_artifact_csv_dir, records)

    paths = {
        "records": output / "records.csv",
        "sources": output / "sources.csv",
        "curation_log": output / "curation_log.csv",
        "field_provenance": output / "field_provenance.csv",
        "curve_registry": output / "curve_registry.csv",
        "curve_points": output / "curve_points.parquet",
        "property_registry": output / "property_registry.csv",
        "lab_metadata_review_queue": output / "lab_metadata_review_queue.csv",
        "curation_qa": output / "curation_qa.json",
    }
    _write_csv(records, paths["records"])
    _write_csv(sources, paths["sources"])
    _write_csv(curation_log, paths["curation_log"])
    _write_csv(provenance, paths["field_provenance"])
    _write_csv(curve_registry, paths["curve_registry"])
    _write_csv(property_registry, paths["property_registry"])
    review_queue = sources[
        [
            "publication_id",
            "doi",
            "publication_year",
            "corresponding_author",
            "affiliation",
            "lab_group_id",
            "lab_group_confidence",
            "lab_group_provenance",
            "lab_metadata_review_status",
        ]
    ].copy()
    review_queue["corresponding_author_candidate"] = pd.NA
    review_queue["affiliation_candidate"] = pd.NA
    review_queue["candidate_provenance"] = pd.NA
    review_queue["candidate_confidence"] = "unknown"
    _write_csv(review_queue, paths["lab_metadata_review_queue"])
    try:
        curve_points.to_parquet(paths["curve_points"], index=False)
    except (ImportError, ModuleNotFoundError) as error:
        if require_parquet:
            raise RuntimeError(
                "Writing curve_points.parquet requires pyarrow or fastparquet; no disguised fallback was written"
            ) from error
        fallback = output / "curve_points.csv.gz"
        curve_points.to_csv(fallback, index=False, compression="gzip")
        paths["curve_points"] = fallback

    qa_checks = {
        "four_level_header_signature_validated": True,
        "exactly_97_unique_records": bool(
            len(records) == 97
            and records["record_id"].is_unique
            and set(records["record_id"].astype(int)) == set(range(1, 98))
        ),
        "merged_publication_cells_propagated": bool(
            provenance["merged_anchor_cell"].fillna("").astype(str).ne("").sum() >= 6
            and records["doi"].notna().sum() == 95
        ),
        "excel_comments_preserved": bool(
            (curation_log["issue_type"] == "excel_comment").sum() == 2
        ),
        "electrolyte_rpm_scan_rate_retained": bool(
            "electrolyte_identity" in records
            and records["rpm"].notna().sum() == 95
            and records["scan_rate_mV_s"].notna().sum() == 95
        ),
        "all_ir_status_unknown": bool(
            (records["ir_correction_status"] == "unknown").all()
            and (curve_registry["ir_correction_status"] == "unknown").all()
        ),
        "expected_98_curves": bool(
            len(curve_registry) == 98 and curve_registry["curve_id"].is_unique
        ),
        "id35_1000x_property_conflict_flagged": bool(
            (
                (property_registry["record_id"] == 35)
                & (
                    property_registry["composition_crosscheck_status"]
                    == "known_1000x_total_volume_conflict"
                )
            ).any()
        ),
        "id49_property_gap_flagged": bool(
            (
                (property_registry["record_id"] == 49)
                & (
                    property_registry["composition_crosscheck_status"]
                    == "missing_from_composition_sheet"
                )
            ).any()
        ),
    }
    qa = {
        "status": "pass" if all(qa_checks.values()) else "fail",
        "checks": qa_checks,
        "record_count": int(len(records)),
        "unique_record_count": int(records["record_id"].nunique()),
        "publication_count": int(records["publication_id"].nunique()),
        "doi_nonmissing_count": int(records["doi"].notna().sum()),
        "alkaline_record_count": int((records["electrolyte_scope"] == "alkaline").sum()),
        "acidic_record_count": int((records["electrolyte_scope"] == "acidic").sum()),
        "neutral_record_count": int((records["electrolyte_scope"] == "neutral").sum()),
        "rpm_nonmissing_count": int(records["rpm"].notna().sum()),
        "scan_rate_nonmissing_count": int(records["scan_rate_mV_s"].notna().sum()),
        "curve_count": int(len(curve_registry)),
        "curve_point_count": int(len(curve_points)),
        "merged_cell_provenance_count": int(
            provenance["merged_anchor_cell"].fillna("").astype(str).ne("").sum()
        ),
        "excel_comment_count": int((curation_log["issue_type"] == "excel_comment").sum()),
        "all_ir_status_unknown": qa_checks["all_ir_status_unknown"],
        "known_id35_property_conflict_flagged": qa_checks[
            "id35_1000x_property_conflict_flagged"
        ],
        "known_id49_property_gap_flagged": qa_checks["id49_property_gap_flagged"],
        "lab_metadata_review_status": "incomplete",
        "lab_held_out_validation_ready": bool(
            (sources["lab_group_confidence"] == "verified").all()
        ),
        "lab_validation_boundary": (
            "Lab-held-out claims are prohibited until lab_group_id is human verified; "
            "unknown lab placeholders are publication-specific and are not connected."
        ),
        "input_files": {
            "raw_workbook": {
                "path": str(Path(raw_workbook).resolve()),
                "sha256": _file_sha256(Path(raw_workbook).resolve()),
            },
            "property_workbook": {
                "path": str(Path(property_workbook).resolve()),
                "sha256": _file_sha256(Path(property_workbook).resolve()),
            },
        },
    }
    paths["curation_qa"].write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    row_counts = {
        "records": len(records),
        "sources": len(sources),
        "curation_log": len(curation_log),
        "field_provenance": len(provenance),
        "curve_registry": len(curve_registry),
        "curve_points": len(curve_points),
        "property_registry": len(property_registry),
        "lab_metadata_review_queue": len(review_queue),
        "curation_qa": 1,
    }
    manifest_rows = []
    for artifact, path in paths.items():
        manifest_rows.append(
            {
                "artifact": artifact,
                "path": str(path),
                "sha256": _file_sha256(path),
                "rows": row_counts[artifact],
            }
        )
    manifest_path = output / "curation_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    paths["curation_manifest"] = manifest_path
    return paths


__all__ = [
    "IR_FIELDS",
    "build_curve_registry",
    "build_property_candidate_registry",
    "compute_recipe_hash",
    "curate_canonical_registry",
    "parse_raw_workbook",
]
