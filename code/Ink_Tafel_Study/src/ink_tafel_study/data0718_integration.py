from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from openpyxl import load_workbook


SHEET1_COLUMNS = [
    "record_id",
    "catalyst_type_raw",
    "catalyst_mass_mg",
    "conductive_agent_type_raw",
    "conductive_agent_mass_mg",
    "ionomer_type_raw",
    "ionomer_volume_uL",
    "ionomer_ratio_workbook",
    "solvent_1_type_raw",
    "solvent_1_volume_uL",
    "solvent_1_ratio_workbook",
    "solvent_2_type_raw",
    "solvent_2_volume_uL",
    "solvent_2_ratio_workbook",
    "total_liquid_volume_workbook_uL",
    "dispersion_method_raw",
    "sonication_time_raw",
    "drying_process_raw",
    "drying_temperature_C",
    "drying_time_min",
    "electrode_type_raw",
    "electrode_diameter_cm",
    "electrode_area_workbook_cm2",
    "drop_volume_uL",
    "catalyst_loading_workbook_mg_cm2",
    "catalyst_loading_reported_mg_cm2",
    "conductive_agent_loading_workbook_mg_cm2",
    "electrolyte_raw",
    "rpm_raw",
    "scan_rate_mV_s",
    "overpotential_10mA_reported_mV",
    "overpotential_100mA_reported_mV",
    "tafel_slope_reported_mV_dec",
    "cdl_reported_mF_cm2",
    "ecsa_reported_m2_g",
    "stability_100mA_reported_h",
    "doi",
    "paper_name",
]

SOLVENT_ORDER = ("acetone", "water", "ethanol", "isopropanol", "methanol", "other")

PROPERTY_NAMES = {
    "viscosity": "solvent_mixture_property_proxy_viscosity_mPa_s",
    "surface_tension": "solvent_mixture_property_proxy_surface_tension_mN_m",
    "density": "solvent_mixture_property_proxy_density_kg_m3",
    "dielectric_constant": "solvent_mixture_property_proxy_dielectric_constant",
}

DIELECTRIC_BEST_AVAILABLE = (
    f"{PROPERTY_NAMES['dielectric_constant']}__best_available"
)
DIELECTRIC_BEST_AVAILABLE_IS_ESTIMATED = (
    f"{PROPERTY_NAMES['dielectric_constant']}__best_available_is_estimated"
)

PRIMARY_TARGET = "b_bayesian_local_line_mV_dec"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _clean_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _normalize_doi(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    text = re.sub(r"^httpsthis://(this:dx\.)thisdoi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().lower()


def _numeric(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        result = float(value)
        return result if np.isfinite(result) else float("nan")
    text = str(value).strip()
    if not text or text.startswith("#"):
        return float("nan")
    match = re.search(r"[-+]this\d+(this:\.\d+)this(this:[eE][-+]this\d+)this", text.replace(",", ""))
    return float(match.group(0)) if match else float("nan")


def _positive(value: Any) -> float:
    number = _numeric(value)
    return number if np.isfinite(number) and number > 0 else float("nan")


def _safe_divide(numerator: Any, denominator: Any) -> float:
    top = _numeric(numerator)
    bottom = _numeric(denominator)
    if not np.isfinite(top) or not np.isfinite(bottom) or bottom <= 0:
        return float("nan")
    return top / bottom


def _normalize_solvent(value: Any) -> str | None:
    text = (_clean_text(value) or "").lower()
    if text in {"", "0", "none", "nan"}:
        return None
    if text in {"deionized water", "di water", "water", "h2o"}:
        return "water"
    if "isoprop" in text or text in {"ipa", "2-propanol"}:
        return "isopropanol"
    if "methanol" in text:
        return "methanol"
    if "ethanol" in text:
        return "ethanol"
    if "acetone" in text:
        return "acetone"
    return "other"


def _normalize_conductive_agent(value: Any, mass: Any) -> tuple[str, str]:
    text = (_clean_text(value) or "").lower()
    mass_value = _numeric(mass)
    if text in {"", "0", "none", "nan"} or (np.isfinite(mass_value) and mass_value == 0):
        return "none", "none"
    text = text.strip()
    if "black" in text:
        return text, "carbon_black"
    if any(token in text for token in ("carbon", "vulcan", "ketjen", "c powder")):
        return text, "carbon_other"
    return text, "other"


def _normalize_electrode(value: Any) -> str | None:
    text = (_clean_text(value) or "").lower()
    if not text:
        return None
    if text == "gce" or "glassy carbon" in text:
        return "glassy_carbon"
    if "gold" in text:
        return "gold"
    return text.replace(" ", "_")


def _normalize_drying(value: Any) -> str | None:
    text = (_clean_text(value) or "").lower()
    if not text:
        return None
    if text in {"air-dried", "air dry", "air dried"}:
        return "air_dried"
    if text in {"ir lamp", "infrared lamp"}:
        return "infrared_lamp"
    return text.replace(" ", "_")


def _normalize_dispersion(value: Any) -> str | None:
    text = (_clean_text(value) or "").lower()
    if not text:
        return None
    if "finger" in text or "homogenizer" in text:
        return "ultrasonic_probe_or_homogenizer"
    if "ultra" in text:
        return "ultrasonication"
    return text.replace(" ", "_")


def _parse_ionomer(value: Any) -> tuple[str | None, float]:
    text = (_clean_text(value) or "").replace(" ", "")
    if not text:
        return None, float("nan")
    concentration = _numeric(text.split("%", maxsplit=1)[0]) if "%" in text else float("nan")
    polymer = "Nafion" if "nafion" in text.lower() else text
    return polymer, concentration


def _parse_electrolyte(value: Any) -> tuple[str | None, float, str | None]:
    text = (_clean_text(value) or "").replace(" ", "")
    match = re.fullmatch(r"(thisP<molarity>\d+(this:\.\d+)this)M(thisP<identity>KOH|NaOH|HClO4|KHCO3|H2SO4)", text, re.IGNORECASE)
    if match is None:
        return None, float("nan"), None
    identity_map = {
        "koh": "KOH",
        "naoh": "NaOH",
        "hclo4": "HClO4",
        "khco3": "KHCO3",
        "h2so4": "H2SO4",
    }
    identity = identity_map[match.group("identity").lower()]
    scope = "alkaline" if identity in {"KOH", "NaOH"} else "neutral" if identity == "KHCO3" else "acidic"
    return identity, float(match.group("molarity")), scope


def _composition_key(fractions: dict[str, float], precision: int = 6) -> str:
    parts = []
    for solvent in SOLVENT_ORDER:
        value = float(fractions.get(solvent, 0.0))
        if np.isfinite(value) and value > 10 ** (-precision):
            parts.append(f"{solvent}={value:.{precision}f}")
    return "|".join(parts) if parts else "unknown"


def _recipe_hash(row: pd.Series) -> str:
    fields = [
        "catalyst_type",
        "catalyst_mass_mg",
        "conductive_agent_type",
        "conductive_agent_mass_mg",
        "ionomer_type",
        "ionomer_volume_uL_per_mgcat",
        "solvent_composition_key",
        "total_solvent_only_volume_uL_per_mgcat",
    ]
    payload = "|".join(f"{field}={row.get(field)}" for field in fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_manual_corrections(path: Path) -> pd.DataFrame:
    corrections = pd.read_csv(path)
    required = {
        "correction_id",
        "record_id",
        "source_excel_row",
        "field_name",
        "source_value",
        "corrected_value",
        "unit",
        "status",
        "decision_basis",
        "confirmed_by",
        "confirmed_date",
        "source_reference",
        "notes",
    }
    missing = required.difference(corrections.columns)
    if missing:
        raise ValueError(f"Manual correction registry is missing columns: {sorted(missing)}")
    approved = corrections.loc[corrections["status"].eq("approved")].copy()
    if approved.empty:
        return approved
    approved["record_id"] = pd.to_numeric(approved["record_id"], errors="raise").astype(int)
    approved["source_excel_row"] = pd.to_numeric(
        approved["source_excel_row"], errors="raise"
    ).astype(int)
    if approved["correction_id"].duplicated().any():
        raise ValueError("Manual correction IDs must be unique")
    if approved.duplicated(["record_id", "field_name"]).any():
        raise ValueError("Only one approved correction is allowed per record and field")
    unsupported = sorted(set(approved["field_name"]) - {"ionomer_volume_uL"})
    if unsupported:
        raise ValueError(f"Unsupported manual correction fields: {unsupported}")
    return approved


def read_data0718(
    path: Path, manual_corrections: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    formula_book = load_workbook(path, data_only=False, read_only=False)
    value_book = load_workbook(path, data_only=True, read_only=False)
    formula_sheet = formula_book["Sheet1"]
    value_sheet = value_book["Sheet1"]

    raw_rows: list[dict[str, Any]] = []
    formula_error_cells: list[str] = []
    for excel_row in range(5, 102):
        values = [value_sheet.cell(excel_row, column).value for column in range(1, 39)]
        row = dict(zip(SHEET1_COLUMNS, values, strict=True))
        row["source_excel_row"] = excel_row
        raw_rows.append(row)
        for column in range(1, 39):
            cached = value_sheet.cell(excel_row, column).value
            if isinstance(cached, str) and cached.startswith("#"):
                formula_error_cells.append(value_sheet.cell(excel_row, column).coordinate)

    trailing_nonrecord_cells: list[str] = []
    for row_index in range(102, formula_sheet.max_row + 1):
        if formula_sheet.cell(row_index, 1).value not in (None, ""):
            continue
        for column in range(1, formula_sheet.max_column + 1):
            if formula_sheet.cell(row_index, column).value not in (None, ""):
                trailing_nonrecord_cells.append(formula_sheet.cell(row_index, column).coordinate)

    frame = pd.DataFrame(raw_rows)
    frame["record_id"] = pd.to_numeric(frame["record_id"], errors="coerce").astype("Int64")
    if frame["record_id"].isna().any():
        raise ValueError("Data_0718 Sheet1 rows 5:101 contain missing or non-numeric record IDs")
    frame["record_id"] = frame["record_id"].astype(int)
    if frame["record_id"].tolist() != list(range(1, 98)):
        raise ValueError("Data_0718 record IDs must be exactly 1..97 in order")

    correction_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    if manual_corrections is not None:
        correction_lookup = {
            (int(row["record_id"]), str(row["field_name"])): row
            for row in manual_corrections.to_dict(orient="records")
        }

    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        record_id = int(raw["record_id"])
        catalyst_mass = _numeric(raw["catalyst_mass_mg"])
        conductive_mass = _numeric(raw["conductive_agent_mass_mg"])
        conductive_type, conductive_class = _normalize_conductive_agent(
            raw["conductive_agent_type_raw"], conductive_mass
        )
        ionomer_type, ionomer_concentration = _parse_ionomer(raw["ionomer_type_raw"])
        ionomer_volume_source_raw = _numeric(raw["ionomer_volume_uL"])
        correction = correction_lookup.get((record_id, "ionomer_volume_uL"))
        ionomer_volume_corrected = ionomer_volume_source_raw
        if correction is not None:
            if int(correction["source_excel_row"]) != int(raw["source_excel_row"]):
                raise ValueError(
                    f"Correction {correction['correction_id']} Excel row does not match source"
                )
            expected_source = _numeric(correction["source_value"])
            if not np.isclose(
                ionomer_volume_source_raw, expected_source, rtol=0, atol=1e-12
            ):
                raise ValueError(
                    f"Correction {correction['correction_id']} source value does not match workbook"
                )
            ionomer_volume_corrected = _numeric(correction["corrected_value"])
            if not np.isfinite(ionomer_volume_corrected) or ionomer_volume_corrected <= 0:
                raise ValueError(
                    f"Correction {correction['correction_id']} corrected value must be positive"
                )
        solvent_1_volume_source_raw = _numeric(raw["solvent_1_volume_uL"])
        solvent_2_volume_source_raw = _numeric(raw["solvent_2_volume_uL"])
        solvent_1_ratio_workbook = _numeric(raw["solvent_1_ratio_workbook"])
        solvent_2_ratio_workbook = _numeric(raw["solvent_2_ratio_workbook"])
        if np.isfinite(solvent_1_volume_source_raw):
            solvent_1_volume = solvent_1_volume_source_raw
            solvent_1_volume_source = "primitive_volume"
        elif np.isfinite(solvent_1_ratio_workbook) and np.isfinite(catalyst_mass):
            solvent_1_volume = solvent_1_ratio_workbook * catalyst_mass
            solvent_1_volume_source = "workbook_ratio_fallback"
        else:
            solvent_1_volume = float("nan")
            solvent_1_volume_source = "missing"
        if np.isfinite(solvent_2_volume_source_raw):
            solvent_2_volume = solvent_2_volume_source_raw
            solvent_2_volume_source = "primitive_volume"
        elif np.isfinite(solvent_2_ratio_workbook) and np.isfinite(catalyst_mass):
            solvent_2_volume = solvent_2_ratio_workbook * catalyst_mass
            solvent_2_volume_source = "workbook_ratio_fallback"
        else:
            solvent_2_volume = float("nan")
            solvent_2_volume_source = "missing"
        solvent_1_type = _normalize_solvent(raw["solvent_1_type_raw"])
        solvent_2_type = _normalize_solvent(raw["solvent_2_type_raw"])

        fractions = {solvent: 0.0 for solvent in SOLVENT_ORDER}
        solvent_total = sum(
            value for value in (solvent_1_volume, solvent_2_volume) if np.isfinite(value) and value > 0
        )
        if solvent_total > 0:
            for solvent_type, volume in (
                (solvent_1_type, solvent_1_volume),
                (solvent_2_type, solvent_2_volume),
            ):
                if solvent_type is not None and np.isfinite(volume) and volume > 0:
                    fractions[solvent_type] += volume / solvent_total

        ionomer_ratio_source_raw = _safe_divide(ionomer_volume_source_raw, catalyst_mass)
        ionomer_ratio_corrected = _safe_divide(ionomer_volume_corrected, catalyst_mass)
        source_unit_outlier = bool(
            (np.isfinite(ionomer_volume_source_raw) and ionomer_volume_source_raw > 10_000)
            or (np.isfinite(ionomer_ratio_source_raw) and ionomer_ratio_source_raw > 10_000)
        )
        critical_unit_outlier = bool(
            (np.isfinite(ionomer_volume_corrected) and ionomer_volume_corrected > 10_000)
            or (np.isfinite(ionomer_ratio_corrected) and ionomer_ratio_corrected > 10_000)
        )
        # Preserve workbook values in source-raw audit fields. Approved corrections
        # affect only curated features and their downstream derived quantities.
        ionomer_volume = (
            float("nan") if critical_unit_outlier else ionomer_volume_corrected
        )
        ionomer_ratio = float("nan") if critical_unit_outlier else ionomer_ratio_corrected
        solvent_1_ratio = _safe_divide(solvent_1_volume, catalyst_mass)
        solvent_2_ratio = _safe_divide(solvent_2_volume, catalyst_mass)
        solvent_ratio_sum = sum(
            value for value in (solvent_1_ratio, solvent_2_ratio) if np.isfinite(value)
        )
        total_liquid_volume_source_raw = sum(
            value
            for value in (ionomer_volume_source_raw, solvent_1_volume, solvent_2_volume)
            if np.isfinite(value)
        )
        total_liquid_volume_corrected = sum(
            value
            for value in (ionomer_volume_corrected, solvent_1_volume, solvent_2_volume)
            if np.isfinite(value)
        )
        total_liquid_volume = float("nan") if critical_unit_outlier else total_liquid_volume_corrected
        total_liquid_ratio = _safe_divide(total_liquid_volume, catalyst_mass)

        diameter = _positive(raw["electrode_diameter_cm"])
        workbook_area = _positive(raw["electrode_area_workbook_cm2"])
        if np.isfinite(diameter):
            electrode_area = math.pi * (diameter / 2.0) ** 2
            electrode_area_source = "recomputed_from_diameter"
        elif np.isfinite(workbook_area):
            electrode_area = workbook_area
            electrode_area_source = "workbook_direct_value"
        else:
            electrode_area = float("nan")
            electrode_area_source = "missing"

        drop_volume = _positive(raw["drop_volume_uL"])
        catalyst_loading_recomputed = float("nan")
        conductive_loading_recomputed = float("nan")
        if all(
            np.isfinite(value) and value > 0
            for value in (catalyst_mass, total_liquid_volume, drop_volume, electrode_area)
        ):
            catalyst_loading_recomputed = (
                catalyst_mass * drop_volume / total_liquid_volume / electrode_area
            )
            if np.isfinite(conductive_mass) and conductive_mass >= 0:
                conductive_loading_recomputed = (
                    conductive_mass * drop_volume / total_liquid_volume / electrode_area
                )

        workbook_loading = _positive(raw["catalyst_loading_workbook_mg_cm2"])
        reported_loading = _positive(raw["catalyst_loading_reported_mg_cm2"])
        if np.isfinite(reported_loading):
            selected_loading = reported_loading
            loading_source = "literature_reported"
        elif np.isfinite(catalyst_loading_recomputed):
            selected_loading = catalyst_loading_recomputed
            loading_source = "recomputed_from_primitives"
        elif np.isfinite(workbook_loading):
            selected_loading = workbook_loading
            loading_source = "workbook_manual_derived_unverified"
        else:
            selected_loading = float("nan")
            loading_source = "missing"

        loading_relative_difference = float("nan")
        if np.isfinite(reported_loading) and np.isfinite(catalyst_loading_recomputed):
            loading_relative_difference = abs(catalyst_loading_recomputed - reported_loading) / reported_loading

        electrolyte_identity, electrolyte_molarity, electrolyte_scope = _parse_electrolyte(
            raw["electrolyte_raw"]
        )
        sonication_raw = _clean_text(raw["sonication_time_raw"])
        sonication_time = _numeric(raw["sonication_time_raw"])
        sonication_category = (
            "overnight" if sonication_raw and "overnight" in sonication_raw.lower() else None
        )
        rpm = _numeric(raw["rpm_raw"])

        row: dict[str, Any] = {
            "record_id": record_id,
            "source_excel_row": int(raw["source_excel_row"]),
            "doi": _normalize_doi(raw["doi"]),
            "paper_name": _clean_text(raw["paper_name"]),
            "catalyst_type": _clean_text(raw["catalyst_type_raw"]),
            "catalyst_mass_mg": catalyst_mass,
            "conductive_agent_type": conductive_type,
            "conductive_agent_class": conductive_class,
            "conductive_agent_mass_mg": conductive_mass,
            "conductive_agent_present": int(
                np.isfinite(conductive_mass) and conductive_mass > 0 and conductive_class != "none"
            ),
            "conductive_agent_dose_mg_per_mgcat": _safe_divide(
                conductive_mass, catalyst_mass
            ),
            "ionomer_type": ionomer_type,
            "ionomer_concentration_wt_percent": ionomer_concentration,
            "ionomer_volume_uL_source_raw": ionomer_volume_source_raw,
            "ionomer_volume_uL_per_mgcat_source_raw": ionomer_ratio_source_raw,
            "ionomer_volume_uL": ionomer_volume,
            "ionomer_volume_uL_per_mgcat": ionomer_ratio,
            "manual_correction_applied": correction is not None,
            "manual_correction_id": (
                str(correction["correction_id"]) if correction is not None else None
            ),
            "manual_correction_decision_basis": (
                str(correction["decision_basis"]) if correction is not None else None
            ),
            "solvent_1_type": solvent_1_type or "none",
            "solvent_1_volume_uL_source_raw": solvent_1_volume_source_raw,
            "solvent_1_volume_uL": solvent_1_volume,
            "solvent_1_volume_source": solvent_1_volume_source,
            "solvent_1_volume_uL_per_mgcat": solvent_1_ratio,
            "solvent_2_type": solvent_2_type or "none",
            "solvent_2_volume_uL_source_raw": solvent_2_volume_source_raw,
            "solvent_2_volume_uL": solvent_2_volume,
            "solvent_2_volume_source": solvent_2_volume_source,
            "solvent_2_volume_uL_per_mgcat": solvent_2_ratio,
            "total_solvent_only_volume_uL": solvent_total,
            "total_solvent_only_volume_uL_per_mgcat": solvent_ratio_sum,
            "total_liquid_volume_uL_source_raw": total_liquid_volume_source_raw,
            "total_liquid_volume_uL": total_liquid_volume,
            "total_liquid_volume_uL_per_mgcat": total_liquid_ratio,
            "ionomer_to_solvent_ratio": _safe_divide(ionomer_ratio, solvent_ratio_sum),
            "catalyst_concentration_mg_mL": _safe_divide(
                catalyst_mass, total_liquid_volume / 1000.0 if total_liquid_volume else np.nan
            ),
            "dispersion_method": _normalize_dispersion(raw["dispersion_method_raw"]),
            "sonication_time_min": sonication_time,
            "sonication_duration_category": sonication_category,
            "drying_process": _normalize_drying(raw["drying_process_raw"]),
            "drying_temperature_C": _numeric(raw["drying_temperature_C"]),
            "drying_time_min": _numeric(raw["drying_time_min"]),
            "electrode_type": _normalize_electrode(raw["electrode_type_raw"]),
            "electrode_diameter_cm": diameter,
            "electrode_area_cm2": electrode_area,
            "electrode_area_source": electrode_area_source,
            "drop_volume_uL": drop_volume,
            "catalyst_loading_recomputed_mg_cm2": catalyst_loading_recomputed,
            "catalyst_loading_workbook_mg_cm2": workbook_loading,
            "catalyst_loading_reported_mg_cm2": reported_loading,
            "catalyst_loading_mg_cm2_selected": selected_loading,
            "catalyst_loading_selected_source": loading_source,
            "catalyst_loading_reported_vs_recomputed_relative_difference": loading_relative_difference,
            "catalyst_loading_conflict_gt25pct": bool(
                np.isfinite(loading_relative_difference) and loading_relative_difference > 0.25
            ),
            "conductive_agent_loading_recomputed_mg_cm2": conductive_loading_recomputed,
            "conductive_agent_loading_workbook_mg_cm2": _positive(
                raw["conductive_agent_loading_workbook_mg_cm2"]
            ),
            "electrolyte_raw": _clean_text(raw["electrolyte_raw"]),
            "electrolyte_identity": electrolyte_identity,
            "electrolyte_molarity_M": electrolyte_molarity,
            "electrolyte_scope": electrolyte_scope,
            "rpm": rpm,
            "scan_rate_mV_s": _numeric(raw["scan_rate_mV_s"]),
            "overpotential_10mA_reported_mV": _numeric(
                raw["overpotential_10mA_reported_mV"]
            ),
            "overpotential_100mA_reported_mV": _numeric(
                raw["overpotential_100mA_reported_mV"]
            ),
            "tafel_slope_sheet1_legacy_value_mV_dec": _numeric(
                raw["tafel_slope_reported_mV_dec"]
            ),
            "cdl_reported_mF_cm2": _numeric(raw["cdl_reported_mF_cm2"]),
            "ecsa_reported_m2_g": _numeric(raw["ecsa_reported_m2_g"]),
            "stability_100mA_reported_h": _numeric(raw["stability_100mA_reported_h"]),
            "qa_critical_unit_outlier": critical_unit_outlier,
            "qa_source_unit_outlier_corrected": bool(
                source_unit_outlier and correction is not None and not critical_unit_outlier
            ),
        }
        for solvent in SOLVENT_ORDER:
            row[f"solvent_fraction_{solvent}"] = fractions[solvent]
        row["solvent_composition_key"] = _composition_key(fractions)
        records.append(row)

    result = pd.DataFrame(records)
    result["recipe_hash"] = result.apply(_recipe_hash, axis=1)
    doi_order = list(dict.fromkeys(result["doi"].tolist()))
    publication_ids = {doi: f"PUB0718-{index:03d}" for index, doi in enumerate(doi_order, 1)}
    result["publication_id"] = result["doi"].map(publication_ids)
    result["sample_id"] = result["record_id"].map(lambda value: f"SAMPLE0718-{value:03d}")

    qa = {
        "formula_error_cells": formula_error_cells,
        "trailing_nonrecord_cells": trailing_nonrecord_cells,
        "n_formula_error_cells": len(formula_error_cells),
        "n_trailing_nonrecord_cells": len(trailing_nonrecord_cells),
    }
    return result, qa


def _normalize_ir_status(value: Any) -> tuple[str, str]:
    """Return compensation state and independent reporting-evidence state.

    DATA0718-IR-SEM-001 records the user-confirmed data-dictionary meaning:
    Sheet2 ``Unreported*`` denotes no iR compensation.  The raw wording remains
    available in ``ir_status_raw`` and is not rewritten as a paper report.
    """

    text = _clean_text(value) or ""
    if text.startswith("Yes"):
        return "corrected", "reported"
    if text.startswith("No"):
        return "not_corrected", "reported"
    if "Unverifiable" in text:
        return "not_corrected", "unverifiable"
    return "not_corrected", "unreported"


def _parse_compensation_fraction(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    number = _numeric(text)
    if not np.isfinite(number):
        return float("nan")
    if "%" in text or number > 1:
        number /= 100.0
    return number if 0 <= number <= 1 else float("nan")


def read_ir_sheet(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet2", header=0, dtype=object)
    frame = frame.iloc[:97, :8].copy()
    frame.columns = [
        "record_id",
        "ir_doi",
        "ir_paper_name",
        "ir_status_raw",
        "ir_compensation_fraction_raw",
        "ir_resistance_text",
        "ir_original_wording",
        "ir_screenshot_note",
    ]
    frame["record_id"] = pd.to_numeric(frame["record_id"], errors="coerce").astype("Int64")
    if frame["record_id"].isna().any() or set(frame["record_id"].astype(int)) != set(range(1, 98)):
        raise ValueError("Data_0718 Sheet2 IDs must coerce to exactly 1..97")
    frame["record_id"] = frame["record_id"].astype(int)
    frame["ir_doi"] = frame["ir_doi"].map(_normalize_doi)
    statuses = frame["ir_status_raw"].map(_normalize_ir_status)
    frame["ir_correction_status"] = statuses.map(lambda item: item[0])
    frame["ir_reporting_evidence_state"] = statuses.map(lambda item: item[1])
    frame["ir_compensation_fraction"] = frame["ir_compensation_fraction_raw"].map(
        _parse_compensation_fraction
    )
    for column in (
        "ir_status_raw",
        "ir_resistance_text",
        "ir_original_wording",
        "ir_screenshot_note",
    ):
        frame[column] = frame[column].map(_clean_text)
    screenshot_usable = frame["ir_screenshot_note"].fillna("").map(
        lambda value: bool(value) and "No verifiable" not in value
    )
    frame["ir_direct_evidence_available"] = (
        frame["ir_original_wording"].notna() | screenshot_usable
    )
    frame["ir_compensation_corrected_binary"] = frame[
        "ir_correction_status"
    ].map({"corrected": 1.0, "not_corrected": 0.0})
    frame["ir_reported_corrected_binary"] = frame[
        "ir_compensation_corrected_binary"
    ].where(frame["ir_reporting_evidence_state"].eq("reported"))
    frame["ir_reporting_known_binary"] = frame[
        "ir_reporting_evidence_state"
    ].eq("reported").astype(int)
    frame["ir_compensation_assignment_basis"] = np.select(
        [
            frame["ir_correction_status"].eq("corrected"),
            frame["ir_reporting_evidence_state"].eq("reported")
            & frame["ir_correction_status"].eq("not_corrected"),
            frame["ir_reporting_evidence_state"].eq("unreported"),
        ],
        [
            "source_reported_corrected",
            "source_reported_not_corrected",
            "user_confirmed_unreported_means_not_corrected",
        ],
        default="user_confirmed_unverifiable_record_means_not_corrected",
    )
    frame["ir_compensation_state_user_confirmed"] = frame[
        "ir_reporting_evidence_state"
    ].isin(["unreported", "unverifiable"])
    frame["ir_compensation_state_revision_id"] = "DATA0718-IR-SEM-001"
    return frame


def build_crosswalk(curve_registry: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for curve in curve_registry.to_dict(orient="records"):
        curve_id = str(curve["curve_id"])
        legacy_record_id = int(curve["record_id"])
        if curve_id == "sr042_v1":
            new_record_id = 42
            rule = "user_confirmed_sr042_v1_to_new42"
            disposition = "retain"
        elif curve_id == "sr042_v2":
            new_record_id = 43
            rule = "user_confirmed_sr042_v2_to_new43"
            disposition = "retain"
        elif curve_id == "sr050":
            new_record_id = pd.NA
            rule = "duplicate_of_sr041_removed_from_canonical"
            disposition = "exclude_duplicate"
        else:
            match = re.fullmatch(r"sr(\d{3})", curve_id)
            if match is None:
                raise ValueError(f"Unexpected curve ID in registry: {curve_id}")
            old_id = int(match.group(1))
            if 1 <= old_id <= 41:
                new_record_id = old_id
                rule = "identity_before_split"
            elif 43 <= old_id <= 49:
                new_record_id = old_id + 1
                rule = "shift_plus_one_after_new43"
            elif 51 <= old_id <= 97:
                new_record_id = old_id
                rule = "identity_after_removed_old50"
            else:
                raise ValueError(f"Curve ID has no approved 0718 mapping: {curve_id}")
            disposition = "retain"
        rows.append(
            {
                "curve_id": curve_id,
                "legacy_record_id": legacy_record_id,
                "new_record_id": new_record_id,
                "disposition": disposition,
                "mapping_rule": rule,
                "mapping_confirmation": "user_confirmed_2026-07-18",
                "legacy_curve_variant": int(curve.get("curve_variant", 1)),
                "legacy_curve_variants_for_sample": int(
                    curve.get("curve_variants_for_sample", 1)
                ),
                "new_curve_variant": 1 if disposition == "retain" else pd.NA,
                "new_curve_variants_for_sample": 1 if disposition == "retain" else pd.NA,
                "workbook_doi": _normalize_doi(curve.get("workbook_doi")),
                "source_curve_sha256": curve.get("source_sha256"),
                "duplicate_of_curve_id": "sr041" if curve_id == "sr050" else None,
            }
        )
    crosswalk = pd.DataFrame(rows)
    retained = crosswalk.loc[crosswalk["disposition"].eq("retain")].copy()
    retained["new_record_id"] = retained["new_record_id"].astype(int)
    doi_map = records.set_index("record_id")["doi"]
    retained["new_record_doi"] = retained["new_record_id"].map(doi_map)
    retained["doi_match"] = retained["workbook_doi"].eq(retained["new_record_doi"])
    crosswalk = crosswalk.merge(
        retained[["curve_id", "new_record_doi", "doi_match"]], on="curve_id", how="left"
    )
    if len(crosswalk) != 98 or len(retained) != 97:
        raise ValueError("Approved crosswalk must contain 98 legacy curves and 97 retained curves")
    if retained["new_record_id"].duplicated().any() or set(retained["new_record_id"]) != set(
        range(1, 98)
    ):
        raise ValueError("Retained curves must map one-to-one onto new record IDs 1..97")
    if not retained["doi_match"].all():
        bad = retained.loc[~retained["doi_match"], ["curve_id", "new_record_id"]]
        raise ValueError(f"Curve-to-record DOI mismatch after approved mapping: {bad.to_dict('records')}")
    return crosswalk.sort_values("curve_id").reset_index(drop=True)


def attach_crosswalk(records: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    retained = crosswalk.loc[crosswalk["disposition"].eq("retain")].copy()
    retained["new_record_id"] = retained["new_record_id"].astype(int)
    mapping = retained[
        [
            "new_record_id",
            "curve_id",
            "legacy_record_id",
            "mapping_rule",
            "new_curve_variant",
            "new_curve_variants_for_sample",
        ]
    ].rename(columns={"new_record_id": "record_id"})
    result = records.merge(mapping, on="record_id", how="left", validate="one_to_one")
    if result["curve_id"].isna().any():
        raise ValueError("Every Data_0718 record must have one retained curve")
    result["record_uid"] = result.apply(
        lambda row: "REC-"
        + hashlib.sha256(f"{row['doi']}|{row['curve_id']}".encode("utf-8")).hexdigest()[:16],
        axis=1,
    )
    return result


def attach_record43_composite_review(records: pd.DataFrame, review_path: Path) -> pd.DataFrame:
    review = pd.read_excel(review_path, sheet_name="42-2_IrO2C", header=3, dtype=object)
    required = {"Parameter", "Value / formula", "Evidence status"}
    if not required.issubset(review.columns):
        raise ValueError(
            f"Record-43 composite review is missing columns: {sorted(required - set(review))}"
        )
    parameter_values = {
        str(row["Parameter"]).strip(): row["Value / formula"]
        for _, row in review.dropna(subset=["Parameter"]).iterrows()
    }
    expected_doi = _normalize_doi(parameter_values.get("DOI retained from dataset"))
    record43 = records.loc[records["record_id"].eq(43)]
    if len(record43) != 1 or record43["doi"].iloc[0] != expected_doi:
        raise ValueError("Record-43 composite review DOI does not match Data_0718 new record 43")
    if _clean_text(parameter_values.get("Corrected catalyst type")) != "IrO2/C":
        raise ValueError("Record-43 composite review must identify the catalyst as IrO2/C")

    result = records.copy()
    result["catalyst_active_material_type"] = result["catalyst_type"]
    result["catalyst_composite_type"] = result["catalyst_type"]
    result["catalyst_support_type"] = None
    result["catalyst_support_review_state"] = "not_curated"
    result["extra_conductive_agent_type"] = result["conductive_agent_type"]
    result["extra_conductive_agent_mass_mg"] = result["conductive_agent_mass_mg"]
    result["catalyst_mass_interpretation"] = "nominal_catalyst_mass"
    result["catalyst_loading_interpretation"] = "nominal_or_reported_catalyst_loading"
    result["theoretical_active_material_mass_fraction"] = np.nan
    result["theoretical_support_mass_fraction"] = np.nan
    result["theoretical_active_material_loading_mg_cm2"] = np.nan
    result["theoretical_support_loading_mg_cm2"] = np.nan
    result["theoretical_component_estimate_status"] = None

    mask = result["record_id"].eq(43)
    result.loc[mask, "catalyst_active_material_type"] = "IrO2"
    result.loc[mask, "catalyst_composite_type"] = "IrO2/C"
    result.loc[mask, "catalyst_support_type"] = "carbon"
    result.loc[mask, "catalyst_support_review_state"] = "source_interpretation_reviewed"
    result.loc[mask, "extra_conductive_agent_type"] = "none"
    result.loc[mask, "extra_conductive_agent_mass_mg"] = _numeric(
        parameter_values.get("Extra conductive-agent mass")
    )
    result.loc[mask, "catalyst_mass_interpretation"] = "total_IrO2_C_composite_mass"
    result.loc[mask, "catalyst_loading_interpretation"] = "total_IrO2_C_composite_loading"
    result.loc[mask, "theoretical_active_material_mass_fraction"] = _numeric(
        parameter_values.get("Theoretical IrO₂ mass fraction")
    )
    result.loc[mask, "theoretical_support_mass_fraction"] = _numeric(
        parameter_values.get("Theoretical carbon-support mass fraction")
    )
    result.loc[mask, "theoretical_active_material_loading_mg_cm2"] = _numeric(
        parameter_values.get("Theoretical IrO₂ loading")
    )
    result.loc[mask, "theoretical_support_loading_mg_cm2"] = _numeric(
        parameter_values.get("Theoretical carbon-support loading")
    )
    result.loc[mask, "theoretical_component_estimate_status"] = (
        "theoretical_stoichiometric_estimate_not_measured"
    )
    return result


def attach_ir(records: pd.DataFrame, ir: pd.DataFrame) -> pd.DataFrame:
    result = records.merge(ir, on="record_id", how="left", validate="one_to_one")
    if result["ir_correction_status"].isna().any():
        raise ValueError("Every record must have one iR status")
    if not result["doi"].eq(result["ir_doi"]).all():
        raise ValueError("Sheet1 and Sheet2 DOI values do not match after numeric ID coercion")
    return result


def attach_properties(records: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    required = {"solvent_composition_key", "property_name", "proxy_value"}
    if not required.issubset(canonical.columns):
        raise ValueError(f"Canonical property table is missing {sorted(required - set(canonical))}")
    property_rows: list[dict[str, Any]] = []
    for composition_key, group in canonical.groupby("solvent_composition_key", sort=False):
        row: dict[str, Any] = {"solvent_composition_key": composition_key}
        for property_name, prefix in PROPERTY_NAMES.items():
            candidates = group.loc[group["property_name"].eq(property_name)]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one {property_name} row for composition {composition_key}; got {len(candidates)}"
                )
            source = candidates.iloc[0]
            row[prefix] = pd.to_numeric(pd.Series([source.get("proxy_value")]), errors="coerce").iloc[0]
            row[f"{prefix}__evidence_tier"] = source.get("evidence_tier")
            row[f"{prefix}__model_use_status"] = source.get("model_use_status")
            row[f"{prefix}__source_ids"] = source.get("source_ids")
            row[f"{prefix}__series_id"] = source.get("series_id")
            row[f"{prefix}__raw_file_sha256"] = source.get("raw_file_sha256")
            row[f"{prefix}__confidence_flag"] = source.get("confidence_flag")
            row[f"{prefix}__estimated"] = pd.to_numeric(
                pd.Series([source.get("estimated_value")]), errors="coerce"
            ).iloc[0]
            row[f"{prefix}__estimated_model_use_status"] = source.get(
                "estimated_model_use_status"
            )
            row[f"{prefix}__best_available"] = pd.to_numeric(
                pd.Series([source.get("best_available_value")]), errors="coerce"
            ).iloc[0]
            row[f"{prefix}__best_available_model_use_status"] = source.get(
                "best_available_model_use_status"
            )
        property_rows.append(row)
    property_wide = pd.DataFrame(property_rows)
    result = records.merge(
        property_wide, on="solvent_composition_key", how="left", validate="many_to_one"
    )
    core = [PROPERTY_NAMES[name] for name in ("viscosity", "surface_tension", "density")]
    if result[core].isna().any().any():
        raise ValueError("All records must match source-backed core solvent property proxies")
    result["solvent_mixture_property_proxy_dielectric_constant_missing"] = (
        result[PROPERTY_NAMES["dielectric_constant"]].isna().astype(int)
    )
    result[DIELECTRIC_BEST_AVAILABLE_IS_ESTIMATED] = (
        result[PROPERTY_NAMES["dielectric_constant"]].isna()
        & result[DIELECTRIC_BEST_AVAILABLE].notna()
    ).astype(int)
    if result[DIELECTRIC_BEST_AVAILABLE].isna().any():
        raise ValueError("Best-available dielectric proxy must cover every record")
    return result


def attach_targets(
    records: pd.DataFrame,
    bayesian_audit: pd.DataFrame,
    target_registry: pd.DataFrame,
) -> pd.DataFrame:
    audit_columns = [
        "curve_id",
        "selected_model",
        "sampler_diagnostics_pass",
        "named_ppc_review_complete",
        "formal_nuts_ppc_approved",
        "posterior_slope_variable",
        "posterior_slope_median_mV_dec",
        "posterior_slope_draw_hdi_low_mV_dec",
        "posterior_slope_draw_hdi_high_mV_dec",
        "slope_draws_artifact_valid",
        "evidence_id",
        "slope_draws_sha256",
    ]
    audit = bayesian_audit[audit_columns].copy()
    if audit["curve_id"].duplicated().any():
        raise ValueError("Bayesian audit must have one row per curve")
    primary = target_registry.loc[target_registry["target_name"].eq(PRIMARY_TARGET)].copy()
    primary = primary[
        [
            "curve_id",
            "target_eligible",
            "posterior_target_supported",
            "formal_target_approved",
            "exclusion_reason",
            "sample_candidate_gate_reason",
            "target_value_mV_dec",
            "target_interval_low_mV_dec",
            "target_interval_high_mV_dec",
        ]
    ].rename(
        columns={
            "target_eligible": "legacy_target_eligible",
            "posterior_target_supported": "legacy_posterior_target_supported",
            "formal_target_approved": "legacy_formal_target_approved",
            "exclusion_reason": "legacy_target_exclusion_reason",
            "sample_candidate_gate_reason": "legacy_sample_candidate_gate_reason",
            "target_value_mV_dec": "registered_primary_value_mV_dec",
            "target_interval_low_mV_dec": "registered_primary_interval_low_mV_dec",
            "target_interval_high_mV_dec": "registered_primary_interval_high_mV_dec",
        }
    )
    result = records.merge(audit, on="curve_id", how="left", validate="one_to_one")
    result = result.merge(primary, on="curve_id", how="left", validate="one_to_one")
    if result["selected_model"].isna().any():
        raise ValueError("Every retained curve must have Bayesian evidence")

    result["bayesian_primary_provisional_eligible"] = (
        result["legacy_target_eligible"].fillna(False).astype(bool)
        & result["legacy_posterior_target_supported"].fillna(False).astype(bool)
        & result["slope_draws_artifact_valid"].fillna(False).astype(bool)
    )
    result["bayesian_primary_formal_eligible"] = (
        result["bayesian_primary_provisional_eligible"]
        & result["named_ppc_review_complete"].fillna(False).astype(bool)
        & result["formal_nuts_ppc_approved"].fillna(False).astype(bool)
    )
    result["bayesian_primary_eligibility_reason"] = result[
        "legacy_target_exclusion_reason"
    ].fillna("missing_legacy_target_row")
    result.loc[
        result["bayesian_primary_provisional_eligible"]
        ,
        "bayesian_primary_eligibility_reason",
    ] = "eligible_ppc_pending"
    result["y_tafel_bayesian_primary_mV_dec"] = result[
        "registered_primary_value_mV_dec"
    ].where(result["bayesian_primary_provisional_eligible"])
    result["y_tafel_bayesian_primary_interval_low_mV_dec"] = result[
        "registered_primary_interval_low_mV_dec"
    ].where(result["bayesian_primary_provisional_eligible"])
    result["y_tafel_bayesian_primary_interval_high_mV_dec"] = result[
        "registered_primary_interval_high_mV_dec"
    ].where(result["bayesian_primary_provisional_eligible"])
    return result


def attach_legacy_year(records: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    legacy_year = legacy[
        ["sample_record", "source_year", "publication_id", "tafel_slope_source"]
    ].rename(
        columns={
            "sample_record": "legacy_record_id",
            "source_year": "publication_year_legacy_unverified",
            "publication_id": "publication_id_legacy_unverified",
            "tafel_slope_source": "tafel_slope_provenance_legacy",
        }
    )
    result = records.merge(legacy_year, on="legacy_record_id", how="left", validate="many_to_one")
    result.loc[result["curve_id"].eq("sr042_v2"), "tafel_slope_provenance_legacy"] = (
        "missing_new_split_record"
    )
    result["tafel_slope_article_reported_only_mV_dec"] = result[
        "tafel_slope_sheet1_legacy_value_mV_dec"
    ].where(result["tafel_slope_provenance_legacy"].eq("reported"))
    result["tafel_slope_reconstructed_only_mV_dec"] = result[
        "tafel_slope_sheet1_legacy_value_mV_dec"
    ].where(result["tafel_slope_provenance_legacy"].eq("reconstructed"))
    return result


def _values_equal(left: Any, right: Any) -> tuple[bool | None, float, float]:
    numeric_types = (int, float, np.integer, np.floating)
    left_is_numeric = isinstance(left, numeric_types) and not isinstance(left, bool)
    right_is_numeric = isinstance(right, numeric_types) and not isinstance(right, bool)
    left_number = float(left) if left_is_numeric else float("nan")
    right_number = float(right) if right_is_numeric else float("nan")
    if np.isfinite(left_number) and np.isfinite(right_number):
        difference = abs(left_number - right_number)
        denominator = max(abs(left_number), abs(right_number), 1e-12)
        return bool(difference <= 1e-9 * denominator), difference, difference / denominator
    left_text = (_clean_text(left) or "").lower()
    right_text = (_clean_text(right) or "").lower()
    if left_text or right_text:
        return left_text == right_text, float("nan"), float("nan")
    return None, float("nan"), float("nan")


def build_legacy_comparison(
    records: pd.DataFrame, legacy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy_index = legacy.set_index("sample_record", drop=False)
    field_pairs = [
        ("doi", "doi", "Data_0718"),
        ("catalyst_type", "catalyst_type", "Data_0718"),
        ("catalyst_mass_mg", "catalyst_mass_mg", "Data_0718"),
        ("conductive_agent_mass_mg", "conductive_agent_mass_mg", "Data_0718"),
        (
            "ionomer_volume_uL_per_mgcat",
            "nafion_solution_volume_ratio_uL_mgcat",
            "Data_0718_recomputed_from_primitives",
        ),
        ("total_liquid_volume_uL", "total_solvent_volume_uL", "Data_0718_recomputed"),
        ("sonication_time_min", "ultrasonication_time_min", "Data_0718"),
        ("drop_volume_uL", "dropwise_addition_uL", "Data_0718"),
        (
            "catalyst_loading_mg_cm2_selected",
            "catalyst_loading_mg_cm2",
            "Data_0718_reported_then_recomputed",
        ),
        ("rpm", "rpm", "Data_0718"),
        ("scan_rate_mV_s", "scan_rate_mV_s", "Data_0718"),
        ("solvent_fraction_acetone", "F_acetone", "Data_0718_recomputed"),
        ("solvent_fraction_water", "F_deionized_water", "Data_0718_recomputed"),
        ("solvent_fraction_ethanol", "F_ethanol", "Data_0718_recomputed"),
        ("solvent_fraction_isopropanol", "F_isopropanol", "Data_0718_recomputed"),
        ("solvent_fraction_methanol", "F_methanol", "Data_0718_recomputed"),
        (
            "tafel_slope_sheet1_legacy_value_mV_dec",
            "y_tafel_slope_mV_dec",
            "Data_0718_legacy_outcome_audit",
        ),
        (
            PROPERTY_NAMES["viscosity"],
            "physics_viscosity_Pa_s",
            "source_backed_property_registry",
        ),
        (
            PROPERTY_NAMES["surface_tension"],
            "physics_surface_tension_N_m",
            "source_backed_property_registry",
        ),
        (PROPERTY_NAMES["density"], "physics_density_kg_m3", "source_backed_property_registry"),
        (
            PROPERTY_NAMES["dielectric_constant"],
            "physics_dielectric_constant",
            "source_backed_property_registry_strict_only",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for record in records.to_dict(orient="records"):
        legacy_id = int(record["legacy_record_id"])
        legacy_row = legacy_index.loc[legacy_id]
        for new_field, legacy_field, authority in field_pairs:
            new_value = record.get(new_field)
            legacy_value = legacy_row.get(legacy_field)
            if legacy_field == "physics_viscosity_Pa_s" and np.isfinite(_numeric(legacy_value)):
                legacy_value = _numeric(legacy_value) * 1000.0
            if legacy_field == "physics_surface_tension_N_m" and np.isfinite(_numeric(legacy_value)):
                legacy_value = _numeric(legacy_value) * 1000.0
            equal, absolute_difference, relative_difference = _values_equal(new_value, legacy_value)
            rows.append(
                {
                    "record_id": int(record["record_id"]),
                    "legacy_record_id": legacy_id,
                    "mapping_rule": record["mapping_rule"],
                    "field_name": new_field,
                    "legacy_field_name": legacy_field,
                    "new_value": new_value,
                    "legacy_value": legacy_value,
                    "normalized_equal": equal,
                    "absolute_difference": absolute_difference,
                    "relative_difference": relative_difference,
                    "authoritative_source_for_new_master": authority,
                    "legacy_model_use_allowed": False,
                }
            )
    comparison = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for field_name, group in comparison.groupby("field_name", sort=False):
        numeric_relative = pd.to_numeric(group["relative_difference"], errors="coerce")
        comparable = numeric_relative.notna()
        equality = group["normalized_equal"].dropna().astype(bool)
        summary_rows.append(
            {
                "field_name": field_name,
                "legacy_field_name": group["legacy_field_name"].iloc[0],
                "authoritative_source_for_new_master": group[
                    "authoritative_source_for_new_master"
                ].iloc[0],
                "n_records": len(group),
                "n_comparable_numeric": int(comparable.sum()),
                "n_equal_after_normalization": int(equality.sum()),
                "median_relative_difference": (
                    float(numeric_relative.loc[comparable].median()) if comparable.any() else np.nan
                ),
                "n_relative_difference_gt5pct": int((numeric_relative > 0.05).sum()),
                "n_relative_difference_gt20pct": int((numeric_relative > 0.20).sum()),
                "legacy_model_use_allowed": False,
            }
        )
    return comparison, pd.DataFrame(summary_rows)


def source_decisions() -> pd.DataFrame:
    rows = [
        (
            "record identity and DOI",
            "formulation_source.xlsx Sheet1 + approved curve crosswalk",
            "canonical",
            "numeric IDs changed after the 42 split and removal of old 50",
        ),
        (
            "primitive ink recipe and measurement context",
            "formulation_source.xlsx Sheet1",
            "canonical_after_QA",
            "latest manually checked literature extraction",
        ),
        (
            "new43 IrO2/C composite and carbon-support semantics",
            "Data_ink_0619_42-2_IrO2C_revised.xlsx review sheet",
            "canonical_semantic_override_with_theoretical_components_audit_only",
            "1 mg is total IrO2/C composite; carbon support is not an extra conductive agent",
        ),
        (
            "derived ratios, fractions, areas, and loadings",
            "recomputed from Data_0718 primitive inputs",
            "canonical_after_QA",
            "do not trust cached Excel formulas as the source of truth",
        ),
        (
            "iR correction state",
            "formulation_source.xlsx Sheet2 + DATA0718-IR-SEM-001",
            "canonical_compensation_state_with_separate_reporting_evidence",
            "corrected / not_corrected / unverifiable",
        ),
        (
            "iR compensation fraction",
            "formulation_source.xlsx Sheet2",
            "audit_or_sensitivity_only",
            "too sparse and partly nonnumeric",
        ),
        (
            "electrolyte identity, molarity, and scope",
            "parsed from Data_0718 Sheet1 electrolyte text",
            "canonical_context",
            "all 97 strings parse into identity, molarity, and acid/base scope",
        ),
        (
            "legacy article/reconstructed Tafel value",
            "formulation_source.xlsx Sheet1 + Data_ink_0523 provenance",
            "comparison_only",
            "68 reported and 29 reconstructed in the legacy source; never a predictor",
        ),
        (
            "primary Tafel target",
            "Bayesian local-line slope target",
            "provisional_until_named_PPC",
            "eta >= 0.15 V strict observed-current progression, with calibrated quantization-aware PAVA fallback only for Records 23 and 74; posterior median and 95% interval",
        ),
        (
            "solvent-mixture property proxies",
            "source-backed canonical property registry at 298.15 K",
            "primary_property_panel",
            "viscosity, surface tension, density, best-available dielectric, and dielectric-estimated indicator are primary-panel candidates",
        ),
        (
            "legacy physics_* properties",
            "ink_provenance.csv",
            "audit_only_excluded_from_ML",
            "legacy_unverified and materially discrepant from source-backed proxies",
        ),
        (
            "overpotential, Cdl, ECSA, and stability",
            "formulation_source.xlsx Sheet1",
            "descriptive_only",
            "performance scope is currently restricted to Tafel slope",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["data_domain", "controlling_source", "model_use", "decision_reason"],
    )


def _feature_metadata() -> list[dict[str, Any]]:
    def feature(
        column: str,
        panel: str,
        source: str,
        representation_group: str,
        *,
        special_status: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        return {
            "column_name": column,
            "semantic_role": "predictor_candidate",
            "panel": panel,
            "source_layer": source,
            "representation_group": representation_group,
            "special_status": special_status,
            "note": note,
        }

    return [
        feature("catalyst_type", "context", "Data_0718", "catalyst_identity"),
        feature("electrolyte_identity", "context", "Data_0718_derived", "electrolyte"),
        feature("electrolyte_molarity_M", "context", "Data_0718_derived", "electrolyte"),
        feature("electrolyte_scope", "context", "Data_0718_derived", "electrolyte"),
        feature("rpm", "context", "Data_0718", "measurement"),
        feature("scan_rate_mV_s", "context", "Data_0718", "measurement"),
        feature("electrode_type", "context", "Data_0718_derived", "measurement"),
        feature("electrode_area_cm2", "context", "Data_0718_derived", "measurement"),
        feature("drop_volume_uL", "context", "Data_0718", "measurement"),
        feature(
            "catalyst_loading_mg_cm2_selected",
            "context",
            "Data_0718_reported_then_recomputed",
            "measurement",
        ),
        feature("ir_correction_status", "context", "Data_0718_Sheet2", "ir"),
        feature(
            "ir_compensation_fraction",
            "context_sensitivity",
            "Data_0718_Sheet2",
            "ir",
            special_status="sensitivity_only_sparse_numeric",
            note="Only parseable where an explicit compensation percentage was reported.",
        ),
        feature("catalyst_mass_mg", "raw_ink", "Data_0718", "ink_recipe"),
        feature(
            "conductive_agent_class", "raw_ink", "Data_0718_derived", "ink_recipe"
        ),
        feature("conductive_agent_mass_mg", "raw_ink", "Data_0718", "ink_recipe"),
        feature(
            "conductive_agent_present", "raw_ink", "Data_0718_derived", "ink_recipe"
        ),
        feature(
            "conductive_agent_dose_mg_per_mgcat",
            "derived_ink",
            "Data_0718_derived",
            "ink_recipe",
        ),
        feature(
            "ionomer_concentration_wt_percent", "raw_ink", "Data_0718_derived", "ink_recipe"
        ),
        feature(
            "ionomer_volume_uL_per_mgcat", "raw_ink", "Data_0718_derived", "ink_recipe"
        ),
        feature("solvent_1_type", "raw_ink", "Data_0718_derived", "solvent_raw"),
        feature(
            "solvent_1_volume_uL_per_mgcat",
            "raw_ink",
            "Data_0718_derived",
            "solvent_raw",
        ),
        feature("solvent_2_type", "raw_ink", "Data_0718_derived", "solvent_raw"),
        feature(
            "solvent_2_volume_uL_per_mgcat",
            "raw_ink",
            "Data_0718_derived",
            "solvent_raw",
        ),
        feature(
            "total_solvent_only_volume_uL_per_mgcat",
            "derived_ink",
            "Data_0718_derived",
            "solvent_amount",
        ),
        feature(
            "total_liquid_volume_uL_per_mgcat",
            "derived_ink",
            "Data_0718_derived",
            "ink_recipe",
        ),
        feature(
            "ionomer_to_solvent_ratio", "derived_ink", "Data_0718_derived", "ink_recipe"
        ),
        feature(
            "catalyst_concentration_mg_mL",
            "derived_ink",
            "Data_0718_derived",
            "ink_recipe",
        ),
        *[
            feature(
                f"solvent_fraction_{solvent}",
                "derived_ink",
                "Data_0718_derived",
                "solvent_fraction_representation",
            )
            for solvent in SOLVENT_ORDER
        ],
        feature("dispersion_method", "process", "Data_0718", "process"),
        feature("sonication_time_min", "process", "Data_0718", "process"),
        feature("drying_process", "process", "Data_0718", "process"),
        feature("drying_temperature_C", "process", "Data_0718", "process"),
        feature("drying_time_min", "process", "Data_0718", "process"),
        feature(
            PROPERTY_NAMES["viscosity"],
            "property_core",
            "source_backed_property_registry",
            "property_proxy_representation",
        ),
        feature(
            PROPERTY_NAMES["surface_tension"],
            "property_core",
            "source_backed_property_registry",
            "property_proxy_representation",
        ),
        feature(
            PROPERTY_NAMES["density"],
            "property_core",
            "source_backed_property_registry",
            "property_proxy_representation",
        ),
        feature(
            DIELECTRIC_BEST_AVAILABLE,
            "property_core",
            "source_backed_property_registry",
            "property_proxy_representation",
            note="User-approved 97/97 best-available feature; 31 estimated values are explicitly flagged.",
        ),
        feature(
            DIELECTRIC_BEST_AVAILABLE_IS_ESTIMATED,
            "property_core",
            "Data_0718_derived_from_source_backed_property_registry",
            "property_proxy_representation",
            note="1 for estimated best-available dielectric values; 0 for strict source-backed values.",
        ),
    ]


def build_feature_registry(
    master: pd.DataFrame,
    *,
    drop_threshold: float,
    sensitivity_threshold: float,
) -> pd.DataFrame:
    primary = master.loc[master["electrolyte_scope"].eq("alkaline")].copy()
    rows: list[dict[str, Any]] = []
    for metadata in _feature_metadata():
        column = metadata["column_name"]
        if column not in master:
            raise ValueError(f"Registered feature is missing from master dataset: {column}")
        series_all = master[column]
        series_primary = primary[column]
        missing_all = int(series_all.isna().sum())
        missing_primary = int(series_primary.isna().sum())
        distinct_primary = int(series_primary.dropna().nunique())
        missing_ratio_primary = missing_primary / len(primary)
        special_status = metadata.pop("special_status")
        if distinct_primary <= 1:
            status = "drop_zero_variance"
            reason = "zero or one observed value in the 90-record primary scope"
        elif special_status is not None:
            status = special_status
            reason = metadata.get("note") or "pre-registered scientific/provenance restriction"
        elif missing_ratio_primary > drop_threshold:
            status = "drop_primary_missing_gt_threshold"
            reason = f"primary-scope missing ratio exceeds {drop_threshold:.0%}"
        elif missing_ratio_primary > sensitivity_threshold:
            status = "sensitivity_only_missing_gt_threshold"
            reason = f"primary-scope missing ratio exceeds {sensitivity_threshold:.0%}"
        else:
            status = "include_primary_candidate"
            reason = "passes missingness and variance gates"
        rows.append(
            {
                **metadata,
                "n_all_records": len(master),
                "missing_count_all": missing_all,
                "missing_ratio_all": missing_all / len(master),
                "n_primary_alkaline_records": len(primary),
                "missing_count_primary": missing_primary,
                "missing_ratio_primary": missing_ratio_primary,
                "distinct_nonmissing_primary": distinct_primary,
                "primary_model_status": status,
                "decision_reason": reason,
                "imputation_rule": (
                    "inside_inner_grouped_CV_if_panel_is_run"
                    if (
                        status == "include_primary_candidate"
                        or status.startswith("sensitivity_only")
                    )
                    and missing_primary
                    else "none"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_modeling_candidates(master: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    features = registry.loc[
        registry["primary_model_status"].eq("include_primary_candidate"), "column_name"
    ].tolist()
    identifiers = [
        "record_uid",
        "record_id",
        "sample_id",
        "curve_id",
        "publication_id",
        "doi",
        "recipe_hash",
        "electrolyte_scope",
        "row_data_quality_pass",
        "bayesian_primary_provisional_eligible",
        "bayesian_primary_formal_eligible",
        "y_tafel_bayesian_primary_mV_dec",
        "y_tafel_bayesian_primary_interval_low_mV_dec",
        "y_tafel_bayesian_primary_interval_high_mV_dec",
    ]
    selected = master.loc[
        master["electrolyte_scope"].eq("alkaline")
        & master["row_data_quality_pass"]
        & master["bayesian_primary_provisional_eligible"]
    ].copy()
    return selected[identifiers + features].reset_index(drop=True)


def _reorder_master(master: pd.DataFrame) -> pd.DataFrame:
    leading = [
        "record_uid",
        "record_id",
        "sample_id",
        "curve_id",
        "publication_id",
        "doi",
        "paper_name",
        "legacy_record_id",
        "mapping_rule",
        "recipe_hash",
        "electrolyte_scope",
        "row_data_quality_pass",
        "row_data_quality_issues",
        "ir_correction_status",
        "bayesian_primary_provisional_eligible",
        "bayesian_primary_formal_eligible",
        "y_tafel_bayesian_primary_mV_dec",
        "y_tafel_bayesian_primary_interval_low_mV_dec",
        "y_tafel_bayesian_primary_interval_high_mV_dec",
        "tafel_slope_sheet1_legacy_value_mV_dec",
        "tafel_slope_provenance_legacy",
    ]
    remaining = [column for column in master.columns if column not in leading]
    return master[leading + remaining]


def build_package(
    *,
    data0718: Path,
    legacy_csv: Path,
    record43_review: Path,
    manual_corrections: Path,
    formal_run: Path,
    output_dir: Path,
    drop_threshold: float = 0.40,
    sensitivity_threshold: float = 0.20,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = formal_run / "data"
    target_dir = formal_run / "targets"
    inputs = {
        "data0718": data0718,
        "legacy_csv": legacy_csv,
        "record43_composite_review": record43_review,
        "manual_corrections": manual_corrections,
        "curve_registry": data_dir / "curve_registry.csv",
        "canonical_properties": data_dir / "canonical_solvent_properties.csv",
        "property_proxy_qa": data_dir / "property_proxy_qa.json",
        "bayesian_audit": target_dir / "bayesian_audit.csv",
        "target_registry": target_dir / "target_registry.csv",
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required {name}: {path}")
    input_hashes_before = {name: _sha256(path) for name, path in inputs.items()}

    corrections = read_manual_corrections(manual_corrections)
    records, workbook_qa = read_data0718(data0718, corrections)
    ir = read_ir_sheet(data0718)
    curve_registry = pd.read_csv(inputs["curve_registry"])
    crosswalk = build_crosswalk(curve_registry, records)
    master = attach_crosswalk(records, crosswalk)
    master = attach_record43_composite_review(master, record43_review)
    master = attach_ir(master, ir)
    canonical = pd.read_csv(inputs["canonical_properties"])
    master = attach_properties(master, canonical)
    bayesian_audit = pd.read_csv(inputs["bayesian_audit"])
    target_registry = pd.read_csv(inputs["target_registry"])
    master = attach_targets(master, bayesian_audit, target_registry)
    legacy = pd.read_csv(legacy_csv)
    master = attach_legacy_year(master, legacy)

    master["row_data_quality_issues"] = np.where(
        master["qa_critical_unit_outlier"],
        "critical_ionomer_volume_or_ratio_unit_outlier",
        None,
    )
    master["row_data_quality_pass"] = ~master["qa_critical_unit_outlier"]
    master = _reorder_master(master)

    legacy_comparison, comparison_summary = build_legacy_comparison(master, legacy)
    feature_registry = build_feature_registry(
        master,
        drop_threshold=drop_threshold,
        sensitivity_threshold=sensitivity_threshold,
    )
    modeling = build_modeling_candidates(master, feature_registry)
    bayesian_targets = master[
        [
            "record_uid",
            "record_id",
            "curve_id",
            "publication_id",
            "doi",
            "electrolyte_scope",
            "selected_model",
            "posterior_slope_variable",
            "posterior_slope_median_mV_dec",
            "posterior_slope_draw_hdi_low_mV_dec",
            "posterior_slope_draw_hdi_high_mV_dec",
            "slope_draws_artifact_valid",
            "bayesian_primary_provisional_eligible",
            "bayesian_primary_formal_eligible",
            "bayesian_primary_eligibility_reason",
            "y_tafel_bayesian_primary_mV_dec",
            "y_tafel_bayesian_primary_interval_low_mV_dec",
            "y_tafel_bayesian_primary_interval_high_mV_dec",
            "sampler_diagnostics_pass",
            "named_ppc_review_complete",
            "formal_nuts_ppc_approved",
            "evidence_id",
            "slope_draws_sha256",
        ]
    ].copy()

    master_path = output_dir / "data0718_complete_master.csv"
    parquet_path = output_dir / "data0718_complete_master.parquet"
    crosswalk_path = output_dir / "record_curve_crosswalk.csv"
    feature_registry_path = output_dir / "feature_registry.csv"
    modeling_path = output_dir / "modeling_candidates_provisional.csv"
    bayesian_targets_path = output_dir / "bayesian_tafel_targets_provisional.csv"
    legacy_comparison_path = output_dir / "legacy_0523_comparison_long.csv"
    comparison_summary_path = output_dir / "legacy_0523_comparison_summary.csv"
    source_decisions_path = output_dir / "source_decisions.csv"
    corrections_path = output_dir / "manual_corrections_applied.csv"

    master.to_csv(master_path, index=False)
    master.to_parquet(parquet_path, index=False)
    crosswalk.to_csv(crosswalk_path, index=False)
    feature_registry.to_csv(feature_registry_path, index=False)
    modeling.to_csv(modeling_path, index=False)
    bayesian_targets.to_csv(bayesian_targets_path, index=False)
    legacy_comparison.to_csv(legacy_comparison_path, index=False)
    comparison_summary.to_csv(comparison_summary_path, index=False)
    source_decisions().to_csv(source_decisions_path, index=False)
    corrections.to_csv(corrections_path, index=False)

    property_qa = json.loads(inputs["property_proxy_qa"].read_text(encoding="utf-8"))
    ir_counts = master["ir_correction_status"].value_counts().to_dict()
    retained_crosswalk = crosswalk["disposition"].eq("retain")
    qa = {
        "status": "review_required_not_formal_ml_ready",
        "grain": "one canonical record and one retained curve per row",
        "n_records": len(master),
        "n_unique_record_ids": int(master["record_id"].nunique()),
        "n_unique_record_uids": int(master["record_uid"].nunique()),
        "n_unique_curves": int(master["curve_id"].nunique()),
        "n_unique_dois": int(master["doi"].nunique()),
        "crosswalk": {
            "n_legacy_curves": len(crosswalk),
            "n_retained": int(retained_crosswalk.sum()),
            "n_excluded_duplicate": int(crosswalk["disposition"].eq("exclude_duplicate").sum()),
            "all_retained_doi_match": bool(
                crosswalk.loc[retained_crosswalk, "doi_match"].fillna(False).all()
            ),
            "excluded_curve_ids": crosswalk.loc[
                ~retained_crosswalk, "curve_id"
            ].tolist(),
        },
        "ir_status_counts": {str(key): int(value) for key, value in ir_counts.items()},
        "ir_compensation_fraction_parseable": int(master["ir_compensation_fraction"].notna().sum()),
        "ir_direct_evidence_available": int(master["ir_direct_evidence_available"].sum()),
        "electrolyte_scope_counts": {
            str(key): int(value)
            for key, value in master["electrolyte_scope"].value_counts().to_dict().items()
        },
        "properties": {
            "unique_composition_keys": int(master["solvent_composition_key"].nunique()),
            "core_property_complete_records": int(
                master[
                    [
                        PROPERTY_NAMES["viscosity"],
                        PROPERTY_NAMES["surface_tension"],
                        PROPERTY_NAMES["density"],
                    ]
                ]
                .notna()
                .all(axis=1)
                .sum()
            ),
            "strict_dielectric_complete_records": int(
                master[PROPERTY_NAMES["dielectric_constant"]].notna().sum()
            ),
            "best_available_dielectric_complete_records": int(
                master[DIELECTRIC_BEST_AVAILABLE].notna().sum()
            ),
            "best_available_dielectric_estimated_records": int(
                master[DIELECTRIC_BEST_AVAILABLE_IS_ESTIMATED].sum()
            ),
            "upstream_property_qa_status": property_qa.get("status"),
        },
        "bayesian_target": {
            "primary_provisional_eligible_all_scopes": int(
                master["bayesian_primary_provisional_eligible"].sum()
            ),
            "primary_provisional_eligible_alkaline": int(
                (
                    master["bayesian_primary_provisional_eligible"]
                    & master["electrolyte_scope"].eq("alkaline")
                ).sum()
            ),
            "primary_formal_eligible": int(master["bayesian_primary_formal_eligible"].sum()),
            "new42_median_mV_dec": float(
                master.loc[master["record_id"].eq(42), "y_tafel_bayesian_primary_mV_dec"].iloc[0]
            ),
            "new43_median_mV_dec": float(
                master.loc[master["record_id"].eq(43), "y_tafel_bayesian_primary_mV_dec"].iloc[0]
            ),
            "new50_primary_is_present": bool(
                master.loc[
                    master["record_id"].eq(50), "y_tafel_bayesian_primary_mV_dec"
                ].notna().iloc[0]
            ),
        },
        "data_quality": {
            "critical_unit_outlier_record_ids": master.loc[
                master["qa_critical_unit_outlier"], "record_id"
            ].tolist(),
            "source_unit_outlier_corrected_record_ids": master.loc[
                master["qa_source_unit_outlier_corrected"], "record_id"
            ].tolist(),
            "manual_corrections_applied": int(master["manual_correction_applied"].sum()),
            "catalyst_loading_conflict_gt25pct_records": master.loc[
                master["catalyst_loading_conflict_gt25pct"], "record_id"
            ].tolist(),
            "formula_error_cells_in_source": workbook_qa["formula_error_cells"],
            "trailing_nonrecord_cells_in_source": workbook_qa["trailing_nonrecord_cells"],
        },
        "feature_filter": {
            "primary_scope_n": int(master["electrolyte_scope"].eq("alkaline").sum()),
            "drop_missing_threshold": drop_threshold,
            "sensitivity_missing_threshold": sensitivity_threshold,
            "status_counts": {
                str(key): int(value)
                for key, value in feature_registry["primary_model_status"]
                .value_counts()
                .to_dict()
                .items()
            },
        },
        "modeling_candidates": {
            "n_rows_provisional": len(modeling),
            "n_predictor_columns": int(
                feature_registry["primary_model_status"].eq("include_primary_candidate").sum()
            ),
            "formal_use_allowed": False,
            "reason": "named Bayesian PPC review and formal approval are incomplete",
        },
        "hard_checks": {
            "records_are_1_to_97": master["record_id"].tolist() == list(range(1, 98)),
            "records_unique": not master["record_id"].duplicated().any(),
            "retained_curves_unique": not master["curve_id"].duplicated().any(),
            "sheet1_sheet2_doi_match": bool(master["doi"].eq(master["ir_doi"]).all()),
            "sr050_absent_from_master": "sr050" not in set(master["curve_id"]),
            "new42_curve_correct": master.loc[
                master["record_id"].eq(42), "curve_id"
            ].iloc[0]
            == "sr042_v1",
            "new43_curve_correct": master.loc[
                master["record_id"].eq(43), "curve_id"
            ].iloc[0]
            == "sr042_v2",
            "new43_composite_semantics_correct": (
                master.loc[master["record_id"].eq(43), "catalyst_composite_type"].iloc[0]
                == "IrO2/C"
                and master.loc[
                    master["record_id"].eq(43), "extra_conductive_agent_type"
                ].iloc[0]
                == "none"
            ),
            "record35_manual_correction_is_exact": bool(
                master.loc[master["record_id"].eq(35), "manual_correction_id"].iloc[0]
                == "DATA0718-CORR-001"
                and np.isclose(
                    master.loc[master["record_id"].eq(35), "ionomer_volume_uL_source_raw"].iloc[0],
                    200000.0,
                )
                and np.isclose(
                    master.loc[master["record_id"].eq(35), "ionomer_volume_uL"].iloc[0],
                    200.0,
                )
                and np.isclose(
                    master.loc[
                        master["record_id"].eq(35), "ionomer_volume_uL_per_mgcat"
                    ].iloc[0],
                    100.0,
                )
                and np.isclose(
                    master.loc[master["record_id"].eq(35), "total_liquid_volume_uL"].iloc[0],
                    1200.0,
                )
                and not bool(
                    master.loc[master["record_id"].eq(35), "qa_critical_unit_outlier"].iloc[0]
                )
            ),
            "composition_key_count_is_33": master["solvent_composition_key"].nunique()
            == 33,
            "best_available_dielectric_is_97_of_97": bool(
                master[DIELECTRIC_BEST_AVAILABLE].notna().all()
            ),
            "best_available_dielectric_estimated_count_is_31": int(
                master[DIELECTRIC_BEST_AVAILABLE_IS_ESTIMATED].sum()
            )
            == 31,
            "dielectric_estimated_flag_matches_strict_missingness": bool(
                master[DIELECTRIC_BEST_AVAILABLE_IS_ESTIMATED]
                .eq(master[PROPERTY_NAMES["dielectric_constant"]].isna().astype(int))
                .all()
            ),
            "ir_compensation_state_counts_match": ir_counts
            == {
                "corrected": 38,
                "not_corrected": 59,
            },
            "primary_provisional_target_count_is_97": int(
                master["bayesian_primary_provisional_eligible"].sum()
            )
            == 97,
            "alkaline_primary_provisional_target_count_is_90": int(
                (
                    master["bayesian_primary_provisional_eligible"]
                    & master["electrolyte_scope"].eq("alkaline")
                ).sum()
            )
            == 90,
            "new42_primary_target_present": bool(
                master.loc[
                    master["record_id"].eq(42), "y_tafel_bayesian_primary_mV_dec"
                ].notna().iloc[0]
            ),
            "new43_primary_target_present": bool(
                master.loc[
                    master["record_id"].eq(43), "y_tafel_bayesian_primary_mV_dec"
                ].notna().iloc[0]
            ),
            "new50_primary_target_present": bool(
                master.loc[
                    master["record_id"].eq(50), "y_tafel_bayesian_primary_mV_dec"
                ].notna().iloc[0]
            ),
        },
    }
    if not all(qa["hard_checks"].values()):
        raise ValueError(f"One or more hard QA checks failed: {qa['hard_checks']}")

    input_hashes_after = {name: _sha256(path) for name, path in inputs.items()}
    if input_hashes_before != input_hashes_after:
        changed = sorted(
            name
            for name in input_hashes_before
            if input_hashes_before[name] != input_hashes_after[name]
        )
        raise RuntimeError(f"One or more inputs changed while the package was built: {changed}")
    input_manifest = {
        "inputs": [
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": input_hashes_before[name],
            }
            for name, path in inputs.items()
        ],
        "formal_run": str(formal_run.resolve()),
    }
    _write_json(output_dir / "input_manifest.json", input_manifest)
    _write_json(output_dir / "data_quality_qa.json", qa)

    readme = f"""# Data 0718 integration package

This package is a non-destructive, provisional data-layer rebuild. It does not modify the two source datasets or the formal Bayesian run.

## Canonical grain

- 97 canonical records, 97 retained curves, and 88 DOI groups.
- `sr042_v1 -> new 42` and `sr042_v2 -> new 43` are user-confirmed.
- `sr050` is retained only in `record_curve_crosswalk.csv` as an excluded duplicate of `sr041`.

## Source precedence

- Latest primitive recipe, electrolyte, measurement, and reported performance fields: `formulation_source.xlsx`.
- Derived formulation fields: recomputed from primitive fields, not copied from cached Excel formulas.
- iR: three-state categorical field from Sheet2.
- Solvent properties: source-backed canonical proxy registry at 298.15 K. Legacy `physics_*` fields are audit-only.
- Primary response: Bayesian local-line posterior median from eta >= 0.15 V. Records 23 and 74 use the calibrated quantization-aware PAVA fallback after the strict observed-current progression selector has no support. It remains provisional until named PPC approval.
- Approved manual corrections are applied from a hash-frozen registry while workbook values remain in `*_source_raw` audit columns.

## Missingness rule

- Primary-scope missing ratio `> {drop_threshold:.0%}`: dropped from the provisional primary feature set.
- Missing ratio `> {sensitivity_threshold:.0%}` and `<= {drop_threshold:.0%}`: sensitivity-only.
- Missing values in retained features are not imputed here; future imputation must remain inside inner grouped CV.
- All fields remain in `data0718_complete_master.csv`; filtering only affects `modeling_candidates_provisional.csv`.
- `modeling_candidates_provisional.csv` is a feature superset, not a final single model matrix. Raw solvent/fraction and property-proxy representations must be evaluated in separate registered panels.

## Known blockers

- Record 35 workbook value `200000 uL` is retained for audit; the user-confirmed curated value is `200 uL`, so the record now passes row QA.
- Named Bayesian PPC review/formal approval is still incomplete, so no row is formally ML-ready.
- The primary dielectric feature is the 97/97 best-available proxy. Its 31 estimated values are marked by `solvent_mixture_property_proxy_dielectric_constant__best_available_is_estimated = 1`; strict values remain in the master table for audit and sensitivity checks.

## Files

- `data0718_complete_master.csv` / `.parquet`: complete canonical record table.
- `modeling_candidates_provisional.csv`: alkaline, target-supported, row-QA-passing candidates after feature gates; not for formal inference.
- `bayesian_tafel_targets_provisional.csv`: low-potential posterior target summaries, PPC state, and slope-draw evidence hashes.
- `record_curve_crosswalk.csv`: complete 98-curve legacy-to-new mapping and the excluded duplicate.
- `feature_registry.csv`: missingness, panel, provenance, and include/drop decisions.
- `legacy_0523_comparison_long.csv` and summary: source comparison; legacy values are never model predictors.
- `source_decisions.csv`: source-of-truth decisions.
- `manual_corrections_applied.csv`: approved corrections, source values, curated values, and decision provenance.
- `data_quality_qa.json`, `input_manifest.json`, and `artifact_manifest.json`: reproducibility and QA.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    artifact_paths = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ]
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return qa


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the non-destructive Data_0718 data-integration package."
    )
    parser.add_argument("--data0718", type=Path, required=True)
    parser.add_argument("--legacy-csv", type=Path, required=True)
    parser.add_argument("--record43-review", type=Path, required=True)
    parser.add_argument("--manual-corrections", type=Path, required=True)
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drop-threshold", type=float, default=0.40)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.20)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if not 0 <= args.sensitivity_threshold <= args.drop_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= sensitivity <= drop <= 1")
    qa = build_package(
        data0718=args.data0718,
        legacy_csv=args.legacy_csv,
        record43_review=args.record43_review,
        manual_corrections=args.manual_corrections,
        formal_run=args.formal_run,
        output_dir=args.output_dir,
        drop_threshold=args.drop_threshold,
        sensitivity_threshold=args.sensitivity_threshold,
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
