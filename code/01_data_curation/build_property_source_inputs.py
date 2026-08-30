#!/usr/bin/env python3
"""Build hash-frozen solvent-property source registries.

This script deliberately stops at primary/source-archive measurements.  It does
not interpolate to an ink recipe, estimate dielectric constants, or overwrite
the legacy ink_properties.xlsx workbook.  The generated CSV files are the
auditable inputs consumed by the property-reconstruction stage.

The NIST MDS2-2962 flat files and the three NIST ThermoML JSON records are not
copied into the repository.  Instead, every retained row records its raw-file
SHA256 and exact line/JSON locator, and the selected rows are frozen in
``config/property_measurement_series.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ACCESS_DATE = "2026-07-16"
TARGET_TEMPERATURE_K = 298.15
TEMPERATURE_TOLERANCE_K = 0.05

COMPOUND_IDS = {
    "ethanol": "trc4mit_485",
    "methanol": "trc4mit_494",
    "isopropanol": "trc4mit_495",
    "acetone": "trc4mit_496",
    "water": "trc4mit_562",
}

MDS_SERIES = [
    {
        "series_id": "viscosity_ethanol_water_gonzalez_2007",
        "property_name": "viscosity",
        "organic": "ethanol",
        "reference_id": "trc4mitref_1456",
        "source_id": "gonzalez_2007_jct",
        "source_file_id": "nist_mds2_2962_visc",
        "file_name": "visc.dat",
        "expected_rows": 12,
        "priority": 1,
    },
    {
        "series_id": "viscosity_methanol_water_gonzalez_2007",
        "property_name": "viscosity",
        "organic": "methanol",
        "reference_id": "trc4mitref_1456",
        "source_id": "gonzalez_2007_jct",
        "source_file_id": "nist_mds2_2962_visc",
        "file_name": "visc.dat",
        "expected_rows": 13,
        "priority": 1,
    },
    {
        "series_id": "viscosity_isopropanol_water_pang_2007",
        "property_name": "viscosity",
        "organic": "isopropanol",
        "reference_id": "trc4mitref_1440",
        "source_id": "pang_2007_jml",
        "source_file_id": "nist_mds2_2962_visc",
        "file_name": "visc.dat",
        "expected_rows": 15,
        "priority": 2,
    },
    {
        "series_id": "viscosity_acetone_water_duereh_2017",
        "property_name": "viscosity",
        "organic": "acetone",
        "reference_id": "trc4mitref_2695",
        "source_id": "duereh_2017_jpcb",
        "source_file_id": "nist_mds2_2962_visc",
        "file_name": "visc.dat",
        "expected_rows": 22,
        "priority": 3,
    },
    {
        "series_id": "surface_tension_ethanol_water_vazquez_1995",
        "property_name": "surface_tension",
        "organic": "ethanol",
        "reference_id": "trc4mitref_792",
        "source_id": "vazquez_1995_jced",
        "source_file_id": "nist_mds2_2962_surften",
        "file_name": "surften.dat",
        "expected_rows": 14,
        "priority": 1,
    },
    {
        "series_id": "surface_tension_methanol_water_vazquez_1995",
        "property_name": "surface_tension",
        "organic": "methanol",
        "reference_id": "trc4mitref_792",
        "source_id": "vazquez_1995_jced",
        "source_file_id": "nist_mds2_2962_surften",
        "file_name": "surften.dat",
        "expected_rows": 14,
        "priority": 1,
    },
    {
        "series_id": "surface_tension_isopropanol_water_vazquez_1995",
        "property_name": "surface_tension",
        "organic": "isopropanol",
        "reference_id": "trc4mitref_792",
        "source_id": "vazquez_1995_jced",
        "source_file_id": "nist_mds2_2962_surften",
        "file_name": "surften.dat",
        "expected_rows": 14,
        "priority": 1,
    },
    {
        "series_id": "surface_tension_isopropanol_water_rafati_2010",
        "property_name": "surface_tension",
        "organic": "isopropanol",
        "reference_id": "trc4mitref_1760",
        "source_id": "rafati_2010_jced",
        "source_file_id": "nist_mds2_2962_surften",
        "file_name": "surften.dat",
        "expected_rows": 11,
        "priority": 2,
        "preferred_use": "supplement_for_narrower_high_isopropanol_brackets",
    },
    {
        "series_id": "surface_tension_acetone_water_enders_2007",
        "property_name": "surface_tension",
        "organic": "acetone",
        "reference_id": "trc4mitref_1459",
        "source_id": "enders_2007_jced",
        "source_file_id": "nist_mds2_2962_surften",
        "file_name": "surften.dat",
        "expected_rows": 14,
        "priority": 2,
    },
]

THERMOML_DENSITY_SERIES = [
    {
        "series_id": "density_methanol_water_gonzalez_2007",
        "organic": "methanol",
        "source_id": "gonzalez_2007_jct",
        "source_file_id": "nist_thermoml_jct_2007",
        "file_name": "thermoml_jct_2007_05_004.json",
        "block_index": 9,
        "expected_rows": 13,
        "priority": 1,
    },
    {
        "series_id": "density_ethanol_water_gonzalez_2007",
        "organic": "ethanol",
        "source_id": "gonzalez_2007_jct",
        "source_file_id": "nist_thermoml_jct_2007",
        "file_name": "thermoml_jct_2007_05_004.json",
        "block_index": 11,
        "expected_rows": 12,
        "priority": 1,
    },
    {
        "series_id": "density_isopropanol_water_navarro_2012",
        "organic": "isopropanol",
        "source_id": "navarro_2012_jced",
        "source_file_id": "nist_thermoml_je201010s",
        "file_name": "thermoml_je201010s.json",
        "block_index": 7,
        "expected_rows": 17,
        "priority": 1,
    },
    {
        "series_id": "density_acetone_water_estrada_baltazar_2003",
        "organic": "acetone",
        "source_id": "estrada_baltazar_2003_jced",
        "source_file_id": "nist_thermoml_je030102f",
        "file_name": "thermoml_je030102f.json",
        "block_index": 4,
        "expected_rows": 11,
        "priority": 1,
    },
]

MEASUREMENT_FIELDS = [
    "measurement_id",
    "series_id",
    "series_priority",
    "property_name",
    "evidence_tier",
    "measurement_status",
    "model_use_status",
    "component_1",
    "component_2",
    "composition_component",
    "composition_value",
    "composition_basis",
    "temperature_K",
    "target_temperature_K",
    "temperature_match_status",
    "pressure_kPa",
    "frequency_Hz",
    "static_low_frequency",
    "raw_value",
    "raw_unit",
    "value",
    "unit",
    "reported_uncertainty",
    "uncertainty_unit",
    "uncertainty_type",
    "uncertainty_coverage_percent",
    "experimental_method",
    "phase",
    "source_id",
    "source_file_id",
    "source_reference_id",
    "source_row_locator",
    "raw_file_sha256",
    "access_date",
    "notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def nested_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                found.append(item_value)
            found.extend(nested_values(item_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(nested_values(item, key))
    return found


def refs_map(path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        reference_id, citation = line.split("\t", 1)
        refs[reference_id] = citation.strip()
    return refs


def temperature_status(temperature: float) -> str:
    if math.isclose(temperature, TARGET_TEMPERATURE_K, rel_tol=0.0, abs_tol=1e-9):
        return "exact_target_temperature"
    return "within_0.05_K_target_tolerance"


def mds_measurements(raw_dir: Path, references: dict[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_file: dict[str, list[tuple[int, list[str]]]] = {}
    for file_name in {str(item["file_name"]) for item in MDS_SERIES}:
        path = raw_dir / file_name
        parsed: list[tuple[int, list[str]]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip() or line.startswith("#"):
                continue
            parsed.append((line_number, line.split("\t")))
        by_file[file_name] = parsed

    for specification in MDS_SERIES:
        property_name = str(specification["property_name"])
        organic = str(specification["organic"])
        organic_id = COMPOUND_IDS[organic]
        water_id = COMPOUND_IDS["water"]
        path = raw_dir / str(specification["file_name"])
        file_hash = sha256(path)
        matches: list[dict[str, Any]] = []
        for line_number, columns in by_file[path.name]:
            if property_name == "viscosity":
                if len(columns) != 8:
                    raise ValueError(f"Unexpected visc.dat column count at line {line_number}")
                c1, c2, x1, temperature, pressure, raw_value, raw_uncertainty, reference_id = columns
            else:
                if len(columns) != 7:
                    raise ValueError(f"Unexpected surften.dat column count at line {line_number}")
                c1, c2, x1, temperature, raw_value, raw_uncertainty, reference_id = columns
                pressure = ""
            if {c1, c2} != {organic_id, water_id}:
                continue
            if reference_id != specification["reference_id"]:
                continue
            temperature_value = float(temperature)
            if abs(temperature_value - TARGET_TEMPERATURE_K) > TEMPERATURE_TOLERANCE_K:
                continue
            x_organic = float(x1) if c1 == organic_id else 1.0 - float(x1)
            raw_numeric = float(raw_value)
            raw_uncertainty_numeric = float(raw_uncertainty)
            if property_name == "viscosity":
                value = raw_numeric * 1000.0
                uncertainty = raw_uncertainty_numeric * 1000.0
                raw_unit = "Pa s"
                unit = "mPa s"
            else:
                value = raw_numeric * 1000.0
                uncertainty = raw_uncertainty_numeric * 1000.0
                raw_unit = "N m-1"
                unit = "mN m-1"
            matches.append(
                {
                    "series_id": specification["series_id"],
                    "series_priority": specification["priority"],
                    "property_name": property_name,
                    "evidence_tier": "A_direct",
                    "measurement_status": "direct_experimental_row_archived_by_NIST",
                    "model_use_status": "eligible_source_backed",
                    "component_1": organic,
                    "component_2": "water",
                    "composition_component": organic,
                    "composition_value": f"{x_organic:.10g}",
                    "composition_basis": "mole_fraction",
                    "temperature_K": f"{temperature_value:.8g}",
                    "target_temperature_K": TARGET_TEMPERATURE_K,
                    "temperature_match_status": temperature_status(temperature_value),
                    "pressure_kPa": pressure,
                    "frequency_Hz": "",
                    "static_low_frequency": "",
                    "raw_value": raw_value,
                    "raw_unit": raw_unit,
                    "value": f"{value:.12g}",
                    "unit": unit,
                    "reported_uncertainty": f"{uncertainty:.12g}",
                    "uncertainty_unit": unit,
                    "uncertainty_type": "reported_uncertainty_coverage_not_encoded_in_MDS2_flat_file",
                    "uncertainty_coverage_percent": "",
                    "experimental_method": "not_encoded_in_MDS2_flat_file_consult_original_article",
                    "phase": "liquid",
                    "source_id": specification["source_id"],
                    "source_file_id": specification["source_file_id"],
                    "source_reference_id": reference_id,
                    "source_row_locator": f"{path.name}:line_{line_number}",
                    "raw_file_sha256": file_hash,
                    "access_date": ACCESS_DATE,
                    "notes": (
                        references.get(reference_id, "")
                        + ("; " + str(specification.get("preferred_use")) if specification.get("preferred_use") else "")
                    ),
                    "_sort_x": x_organic,
                    "_sort_locator": line_number,
                }
            )
        matches.sort(key=lambda row: (float(row["_sort_x"]), int(row["_sort_locator"])))
        if len(matches) != int(specification["expected_rows"]):
            raise ValueError(
                f"{specification['series_id']} expected {specification['expected_rows']} rows, found {len(matches)}"
            )
        for index, row in enumerate(matches, start=1):
            row["measurement_id"] = f"{specification['series_id']}__{index:03d}"
            row.pop("_sort_x")
            row.pop("_sort_locator")
        output.extend(matches)
    return output


def thermoml_density_measurements(raw_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for specification in THERMOML_DENSITY_SERIES:
        path = raw_dir / str(specification["file_name"])
        document = json.loads(path.read_text(encoding="utf-8"))
        compounds = {
            int(item["RegNum"]["nOrgNum"]): (
                item["sCommonName"][0]
                if isinstance(item.get("sCommonName"), list)
                else item.get("sCommonName")
            )
            for item in document["Compound"]
        }
        aliases = {"propan-2-ol": "isopropanol"}
        block_index = int(specification["block_index"])
        block = document["PureOrMixtureData"][block_index]
        property_metadata = block["Property"][0]
        property_names = nested_values(property_metadata, "ePropName")
        if not property_names or "Mass density" not in str(property_names[0]):
            raise ValueError(f"ThermoML block {block_index} is not a density block")
        methods = nested_values(property_metadata, "sMethodName") + nested_values(
            property_metadata, "eMethodName"
        )
        method = str(methods[0]) if methods else "method_not_encoded"
        coverage_values = nested_values(property_metadata, "nCombUncertLevOfConfid")
        coverage = coverage_values[0] if coverage_values else ""

        constraint_temperature = ""
        constraint_pressure = ""
        for constraint in block.get("Constraint", []):
            constraint_text = json.dumps(constraint.get("ConstraintID", {}))
            if "Temperature" in constraint_text:
                constraint_temperature = float(constraint["nConstraintValue"])
            elif "Pressure" in constraint_text:
                constraint_pressure = float(constraint["nConstraintValue"])

        variables: dict[int, dict[str, Any]] = {}
        for variable in block.get("Variable", []):
            variable_id = variable["VariableID"]
            variable_type = variable_id["VariableType"]
            number = int(variable["nVarNumber"])
            if "eTemperature" in variable_type:
                variables[number] = {"kind": "temperature"}
            elif "ePressure" in variable_type:
                variables[number] = {"kind": "pressure"}
            elif "eComponentComposition" in variable_type:
                component_number = int(variable_id["RegNum"]["nOrgNum"])
                variables[number] = {
                    "kind": "composition",
                    "basis": str(variable_type["eComponentComposition"]).lower().replace(" ", "_"),
                    "component": aliases.get(str(compounds[component_number]), str(compounds[component_number])),
                }

        matches: list[dict[str, Any]] = []
        for row_index, row in enumerate(block["NumValues"]):
            values = {int(item["nVarNumber"]): float(item["nVarValue"]) for item in row["VariableValue"]}
            temperature = float(constraint_temperature) if constraint_temperature != "" else math.nan
            pressure = float(constraint_pressure) if constraint_pressure != "" else math.nan
            raw_composition_component = ""
            raw_composition_value = math.nan
            composition_basis = ""
            for number, metadata in variables.items():
                if metadata["kind"] == "temperature":
                    temperature = values[number]
                elif metadata["kind"] == "pressure":
                    pressure = values[number]
                elif metadata["kind"] == "composition":
                    raw_composition_component = str(metadata["component"])
                    raw_composition_value = values[number]
                    composition_basis = str(metadata["basis"])
            if not math.isfinite(temperature):
                raise ValueError(f"No temperature in ThermoML block {block_index} row {row_index}")
            if abs(temperature - TARGET_TEMPERATURE_K) > TEMPERATURE_TOLERANCE_K:
                continue
            organic = str(specification["organic"])
            if raw_composition_component == organic:
                x_organic = raw_composition_value
            elif raw_composition_component == "water":
                x_organic = 1.0 - raw_composition_value
            else:
                raise ValueError(
                    f"Unexpected composition component {raw_composition_component!r} in {specification['series_id']}"
                )
            property_value = row["PropertyValue"][0]
            numeric = float(property_value["nPropValue"])
            uncertainty_values = nested_values(property_value, "nCombExpandUncertValue")
            uncertainty = float(uncertainty_values[0]) if uncertainty_values else ""
            matches.append(
                {
                    "series_id": specification["series_id"],
                    "series_priority": specification["priority"],
                    "property_name": "density",
                    "evidence_tier": "A_direct",
                    "measurement_status": "direct_experimental_row_archived_by_NIST_ThermoML",
                    "model_use_status": "eligible_source_backed",
                    "component_1": organic,
                    "component_2": "water",
                    "composition_component": organic,
                    "composition_value": f"{x_organic:.10g}",
                    "composition_basis": composition_basis,
                    "temperature_K": f"{temperature:.8g}",
                    "target_temperature_K": TARGET_TEMPERATURE_K,
                    "temperature_match_status": temperature_status(temperature),
                    "pressure_kPa": "" if not math.isfinite(pressure) else f"{pressure:.12g}",
                    "frequency_Hz": "",
                    "static_low_frequency": "",
                    "raw_value": f"{numeric:.12g}",
                    "raw_unit": "kg m-3",
                    "value": f"{numeric:.12g}",
                    "unit": "kg m-3",
                    "reported_uncertainty": uncertainty,
                    "uncertainty_unit": "kg m-3" if uncertainty != "" else "",
                    "uncertainty_type": "combined_expanded_uncertainty" if uncertainty != "" else "not_reported",
                    "uncertainty_coverage_percent": coverage,
                    "experimental_method": method,
                    "phase": "liquid",
                    "source_id": specification["source_id"],
                    "source_file_id": specification["source_file_id"],
                    "source_reference_id": document["Citation"].get("sDOI", ""),
                    "source_row_locator": f"ThermoML:PureOrMixtureData[{block_index}]:NumValues[{row_index}]",
                    "raw_file_sha256": sha256(path),
                    "access_date": ACCESS_DATE,
                    "notes": f"Raw composition variable was {raw_composition_component} {composition_basis}; normalized here to {organic} fraction.",
                    "_sort_x": x_organic,
                    "_sort_locator": row_index,
                }
            )
        matches.sort(key=lambda row: (float(row["_sort_x"]), int(row["_sort_locator"])))
        if len(matches) != int(specification["expected_rows"]):
            raise ValueError(
                f"{specification['series_id']} expected {specification['expected_rows']} rows, found {len(matches)}"
            )
        for index, row in enumerate(matches, start=1):
            row["measurement_id"] = f"{specification['series_id']}__{index:03d}"
            row.pop("_sort_x")
            row.pop("_sort_locator")
        output.extend(matches)
    return output


def pure_component_reference_measurements() -> list[dict[str, Any]]:
    """Return exact-state official reference values with explicit priority.

    Pure water appears as an endpoint in several binary experimental series.
    Those endpoints differ slightly and must not be selected by a lexical
    series-name tie-break.  The study plan pre-registers IAPWS as the density
    reference, so this exact-state row has priority zero for the pure-water
    composition only.  Its evaluated-equation origin remains explicit in the
    method/status fields.
    """

    return [
        {
            "measurement_id": "density_water_iapws95_2018__001",
            "series_id": "density_water_iapws95_2018",
            "series_priority": 0,
            "property_name": "density",
            "evidence_tier": "A_direct",
            "measurement_status": "official_evaluated_reference_exact_target_state",
            "model_use_status": "eligible_source_backed",
            "component_1": "water",
            "component_2": "",
            "composition_component": "water",
            "composition_value": "1",
            "composition_basis": "volume_fraction",
            "temperature_K": "298.15",
            "target_temperature_K": TARGET_TEMPERATURE_K,
            "temperature_match_status": "exact_target_temperature",
            "pressure_kPa": "101.325",
            "frequency_Hz": "",
            "static_low_frequency": "",
            "raw_value": "997.0474",
            "raw_unit": "kg m-3",
            "value": "997.0474",
            "unit": "kg m-3",
            "reported_uncertainty": "",
            "uncertainty_unit": "",
            "uncertainty_type": "IAPWS_evaluated_reference_uncertainty_not_transcribed",
            "uncertainty_coverage_percent": "",
            "experimental_method": "IAPWS95_evaluated_equation",
            "phase": "liquid",
            "source_id": "iapws95_2018",
            "source_file_id": "iapws95_2018_pdf",
            "source_reference_id": "IAPWS95-2018",
            "source_row_locator": "IAPWS-95 evaluated equation at T=298.15 K and p=0.101325 MPa",
            "raw_file_sha256": "fbaa0abc4cbf071152a5cbe230a586bd045db6c16f0cb2c872caf4e8888515dd",
            "access_date": ACCESS_DATE,
            "notes": (
                "Exact target-state official reference value. A_direct here means no "
                "temperature/composition interpolation; the evaluated-equation origin is "
                "not represented as a new experimental measurement."
            ),
        }
    ]


def dielectric_measurements() -> list[dict[str, Any]]:
    """Return verified static-limit dielectric rows.

    Petong Table 3 was manually transcribed from the indexed primary PDF.  The
    remote PDF host was unavailable to the command-line downloader during this
    build, so its file hash is intentionally blank and the limitation is made
    machine-readable.  The x=0 water value is an IAPWS evaluated reference and
    is kept as a separate series/source; downstream code must not describe an
    interpolation spanning that point as same-source Petong data.

    Smith Figure 2 is retained as an explicitly digitized C-tier series.  Its
    local publisher-image hash, pixel calibration, and conservative reading
    uncertainty are frozen below.  These rows are source-backed, but they must
    never be counted as A/B table measurements.
    """

    rows: list[dict[str, Any]] = []
    petong_values = [(0.22, 54.4), (0.36, 44.4), (0.54, 35.4), (0.76, 28.8), (1.0, 24.5)]
    for index, (mole_fraction, epsilon) in enumerate(petong_values, start=1):
        rows.append(
            {
                "measurement_id": f"dielectric_ethanol_water_petong_2000__{index:03d}",
                "series_id": "dielectric_ethanol_water_petong_2000",
                "series_priority": 1,
                "property_name": "dielectric_constant",
                "evidence_tier": "A_direct",
                "measurement_status": "direct_primary_table_manual_transcription_remote_file_hash_pending",
                "model_use_status": "eligible_source_backed",
                "component_1": "ethanol",
                "component_2": "water",
                "composition_component": "ethanol",
                "composition_value": f"{mole_fraction:.10g}",
                "composition_basis": "mole_fraction",
                "temperature_K": "298.15",
                "target_temperature_K": TARGET_TEMPERATURE_K,
                "temperature_match_status": "exact_target_temperature",
                "pressure_kPa": "",
                "frequency_Hz": "0",
                "static_low_frequency": "true",
                "raw_value": f"{epsilon:.10g}",
                "raw_unit": "dimensionless",
                "value": f"{epsilon:.10g}",
                "unit": "dimensionless",
                "reported_uncertainty": f"{epsilon * 0.02:.10g}",
                "uncertainty_unit": "dimensionless",
                "uncertainty_type": "reported_relative_uncertainty_2_percent",
                "uncertainty_coverage_percent": "",
                "experimental_method": "frequency_domain_dielectric_spectroscopy_extrapolated_low_frequency_limit",
                "phase": "liquid",
                "source_id": "petong_2000_jpca",
                "source_file_id": "petong_2000_primary_pdf_indexed",
                "source_reference_id": "10.1021/jp001393r",
                "source_row_locator": f"Table 3; 25 degC; x_ethanol={mole_fraction:g}",
                "raw_file_sha256": "",
                "access_date": ACCESS_DATE,
                "notes": "Primary-table value; source states epsilon(0) uncertainty +/-2% and temperature uncertainty +/-0.05 degC. Remote PDF hash remains pending and is not fabricated.",
            }
        )
    rows.append(
        {
            "measurement_id": "dielectric_water_iapws_2011__001",
            "series_id": "dielectric_water_iapws_2011",
            "series_priority": 1,
            "property_name": "dielectric_constant",
            "evidence_tier": "C_secondary_reconstruction",
            "measurement_status": "official_evaluated_reference_value",
            "model_use_status": "eligible_pure_component_anchor_cross_source_sensitivity_only",
            "component_1": "ethanol",
            "component_2": "water",
            "composition_component": "ethanol",
            "composition_value": "0",
            "composition_basis": "mole_fraction",
            "temperature_K": "298.15",
            "target_temperature_K": TARGET_TEMPERATURE_K,
            "temperature_match_status": "exact_target_temperature",
            "pressure_kPa": "100",
            "frequency_Hz": "0",
            "static_low_frequency": "true",
            "raw_value": "78.375",
            "raw_unit": "dimensionless",
            "value": "78.375",
            "unit": "dimensionless",
            "reported_uncertainty": "",
            "uncertainty_unit": "",
            "uncertainty_type": "IAPWS_evaluated_reference_uncertainty_not_transcribed",
            "uncertainty_coverage_percent": "",
            "experimental_method": "IAPWS_SR6_08_2011_evaluated_correlation",
            "phase": "liquid",
            "source_id": "iapws_liquid_water_2011",
            "source_file_id": "iapws_liquid_water_2011_pdf",
            "source_reference_id": "IAPWS SR6-08(2011)",
            "source_row_locator": "Table at T=298.15 K; static dielectric constant=78.375",
            "raw_file_sha256": "04bd37dd7f4c9c535230af4c51d2a803912d1d41c1656e20527eae598bc55ff4",
            "access_date": ACCESS_DATE,
            "notes": "Pure-water evaluated anchor. It is not a Petong same-source measurement and may not be silently relabelled A_direct.",
        }
    )
    uosaki_values = [(0.2, 53.56), (0.4, 38.76), (0.6, 30.07), (0.8, 24.63), (1.0, 20.74)]
    for index, (mole_fraction, epsilon) in enumerate(uosaki_values, start=1):
        rows.append(
            {
                "measurement_id": f"dielectric_acetone_water_uosaki_1997__{index:03d}",
                "series_id": "dielectric_acetone_water_uosaki_1997",
                "series_priority": 1,
                "property_name": "dielectric_constant",
                "evidence_tier": "C_secondary_reconstruction",
                "measurement_status": "primary_table_manual_transcription_remote_file_hash_pending",
                "model_use_status": "eligible_source_backed_with_C_tier_flag",
                "component_1": "acetone",
                "component_2": "water",
                "composition_component": "acetone",
                "composition_value": f"{mole_fraction:.10g}",
                "composition_basis": "mole_fraction",
                "temperature_K": "298.15",
                "target_temperature_K": TARGET_TEMPERATURE_K,
                "temperature_match_status": "exact_target_temperature",
                "pressure_kPa": "100",
                "frequency_Hz": "0",
                "static_low_frequency": "true",
                "raw_value": f"{epsilon:.10g}",
                "raw_unit": "dimensionless",
                "value": f"{epsilon:.10g}",
                "unit": "dimensionless",
                "reported_uncertainty": f"{epsilon * 0.001:.10g}",
                "uncertainty_unit": "dimensionless",
                "uncertainty_type": "reported_relative_uncertainty_0.1_percent",
                "uncertainty_coverage_percent": "",
                "experimental_method": "static_relative_permittivity_measurement_method_consult_primary_article",
                "phase": "liquid",
                "source_id": "uosaki_1997_jced",
                "source_file_id": "uosaki_1997_primary_pdf_indexed",
                "source_reference_id": "10.1021/je960390y",
                "source_row_locator": f"Table 1; P=0.1 MPa; x_acetone={mole_fraction:g}",
                "raw_file_sha256": "",
                "access_date": ACCESS_DATE,
                "notes": "Primary-table transcription recovered from the indexed PDF; evidence is conservatively C until the PDF is locally hashed and independently rechecked.",
            }
        )
    smith_values = [
        (297.0, 0.6, 54.000),
        (297.0, 0.7, 59.000),
        (333.0, 0.6, 43.385),
        (333.0, 0.7, 48.308),
    ]
    for index, (temperature_K, water_mole_fraction, epsilon) in enumerate(smith_values, start=1):
        rows.append(
            {
                "measurement_id": f"dielectric_methanol_water_smith_1998_figure_digitized__{index:03d}",
                "series_id": "dielectric_methanol_water_smith_1998_figure_digitized",
                "series_priority": 1,
                "property_name": "dielectric_constant",
                "evidence_tier": "C_secondary_reconstruction",
                "measurement_status": "primary_figure_digitized_with_local_publisher_image_hash",
                "model_use_status": "eligible_source_backed_with_C_tier_flag",
                "component_1": "methanol",
                "component_2": "water",
                "composition_component": "water",
                "composition_value": f"{water_mole_fraction:.10g}",
                "composition_basis": "mole_fraction",
                "temperature_K": f"{temperature_K:.10g}",
                "target_temperature_K": TARGET_TEMPERATURE_K,
                "temperature_match_status": "temperature_bracketing_source_point",
                "pressure_kPa": "",
                "frequency_Hz": "0",
                "static_low_frequency": "true",
                "raw_value": f"{epsilon:.10g}",
                "raw_unit": "dimensionless",
                "value": f"{epsilon:.10g}",
                "unit": "dimensionless",
                "reported_uncertainty": "0.7",
                "uncertainty_unit": "dimensionless",
                "uncertainty_type": "conservative_figure_digitization_uncertainty_not_standardized",
                "uncertainty_coverage_percent": "",
                "experimental_method": "frequency_domain_reflection_100_MHz_to_20_GHz_static_limit_fit",
                "phase": "liquid",
                "source_id": "smith_1998_fpe",
                "source_file_id": "smith_1998_figure2",
                "source_reference_id": "10.1016/S0378-3812(97)00275-6",
                "source_row_locator": (
                    f"Figure 2; digitized marker; T={temperature_K:g} K; "
                    f"x_water={water_mole_fraction:g}; plot calibration "
                    "x=(px-32)/(331-32), epsilon=80-(py-4)*60/(199-4)"
                ),
                "raw_file_sha256": "2972be5c0e560321f9ab2cad2d4bcdf2ff7503bace3dc4252f02efb093f4765f",
                "access_date": ACCESS_DATE,
                "notes": (
                    "Digitized from the publisher-hosted Figure 2 preview and retained only as "
                    "C_secondary_reconstruction. Reading uncertainty is conservatively +/-0.7; "
                    "the primary measurement uncertainty is unavailable from the preview."
                ),
            }
        )
    rows.append(
        {
            "measurement_id": "dielectric_water_iapws_2011_acetone_system__001",
            "series_id": "dielectric_water_iapws_2011_acetone_system",
            "series_priority": 1,
            "property_name": "dielectric_constant",
            "evidence_tier": "C_secondary_reconstruction",
            "measurement_status": "official_evaluated_reference_value",
            "model_use_status": "eligible_pure_component_anchor_cross_source_sensitivity_only",
            "component_1": "acetone",
            "component_2": "water",
            "composition_component": "acetone",
            "composition_value": "0",
            "composition_basis": "mole_fraction",
            "temperature_K": "298.15",
            "target_temperature_K": TARGET_TEMPERATURE_K,
            "temperature_match_status": "exact_target_temperature",
            "pressure_kPa": "100",
            "frequency_Hz": "0",
            "static_low_frequency": "true",
            "raw_value": "78.375",
            "raw_unit": "dimensionless",
            "value": "78.375",
            "unit": "dimensionless",
            "reported_uncertainty": "",
            "uncertainty_unit": "",
            "uncertainty_type": "IAPWS_evaluated_reference_uncertainty_not_transcribed",
            "uncertainty_coverage_percent": "",
            "experimental_method": "IAPWS_SR6_08_2011_evaluated_correlation",
            "phase": "liquid",
            "source_id": "iapws_liquid_water_2011",
            "source_file_id": "iapws_liquid_water_2011_pdf",
            "source_reference_id": "IAPWS SR6-08(2011)",
            "source_row_locator": "Table at T=298.15 K; static dielectric constant=78.375",
            "raw_file_sha256": "04bd37dd7f4c9c535230af4c51d2a803912d1d41c1656e20527eae598bc55ff4",
            "access_date": ACCESS_DATE,
            "notes": "Pure-water evaluated anchor. It is not a Uosaki same-source measurement.",
        }
    )
    return rows


def source_catalog() -> list[dict[str, str]]:
    rows = [
        ("local_property_source_note", "local_source_note", "user-curated source note", "source_notes.docx", "../../Data/source_notes.docx", "", "registered_provenance", "Identifies candidate websites; it is not a measurement table."),
        ("nist_mds2_2962", "official_dataset", "NIST", "Experimental data sets for viscosity and surface tension of binary mixtures at 293.15-323.15 K", "https://doi.org/10.18434/mds2-2962", "10.18434/mds2-2962", "authoritative_machine_source", "Raw unevaluated literature data extracted from NIST SRD 103b."),
        ("nist_mds2_2962_visc", "official_dataset_resource", "NIST", "visc.dat", "https://data.nist.gov/od/ds/ark:/88434/mds2-2962/visc.dat", "10.18434/mds2-2962", "authoritative_machine_source", "Viscosity in Pa s; published SHA256 is verified by the build script."),
        ("nist_mds2_2962_surften", "official_dataset_resource", "NIST", "surften.dat", "https://data.nist.gov/od/ds/ark:/88434/mds2-2962/surften.dat", "10.18434/mds2-2962", "authoritative_machine_source", "Surface tension in N m-1; published SHA256 is verified by the build script."),
        ("nist_mds2_2962_refs", "official_dataset_resource", "NIST", "refs.dat", "https://data.nist.gov/od/ds/ark:/88434/mds2-2962/refs.dat", "10.18434/mds2-2962", "authoritative_reference_map", "Maps trc4mitref IDs to original literature citations."),
        ("nist_mds2_2962_compounds", "official_dataset_resource", "NIST", "compounds.sdf", "https://data.nist.gov/od/ds/ark:/88434/mds2-2962/compounds.sdf", "10.18434/mds2-2962", "authoritative_compound_map", "Maps trc4mit compound IDs to molecular structures."),
        ("nist_thermoml_jct_2007", "official_archive_record", "NIST Thermodynamics Research Center", "Density and dynamic viscosity of methanol or ethanol with water", "https://trc.nist.gov/ThermoML/10.1016/j.jct.2007.05.004.json", "10.1016/j.jct.2007.05.004", "authoritative_machine_source", "Machine-readable original-paper record with composition, method, and uncertainty metadata."),
        ("nist_thermoml_je201010s", "official_archive_record", "NIST Thermodynamics Research Center", "Physical properties of 2-propanol + water mixtures", "https://trc.nist.gov/ThermoML/10.1021/je201010s.json", "10.1021/je201010s", "authoritative_machine_source", "Machine-readable article record; binary density block is used."),
        ("nist_thermoml_je030102f", "official_archive_record", "NIST Thermodynamics Research Center", "Experimental densities of acetone + water", "https://trc.nist.gov/ThermoML/10.1021/je030102f.json", "10.1021/je030102f", "authoritative_machine_source", "Machine-readable article record; binary density block is used."),
        ("gonzalez_2007_jct", "primary_article", "Journal of Chemical Thermodynamics", "Density, dynamic viscosity, and derived properties of binary mixtures of methanol or ethanol with water, ethyl acetate, and methyl acetate", "https://doi.org/10.1016/j.jct.2007.05.004", "10.1016/j.jct.2007.05.004", "authoritative_primary_source", "Selected primary source for ethanol/water and methanol/water density and viscosity."),
        ("pang_2007_jml", "primary_article", "Journal of Molecular Liquids", "Densities and viscosities of aqueous solutions of 1-propanol and 2-propanol at temperatures from 293.15 K to 333.15 K", "https://doi.org/10.1016/j.molliq.2007.01.003", "10.1016/j.molliq.2007.01.003", "authoritative_primary_source", "Selected primary source for isopropanol/water viscosity; rows are archived in NIST MDS2-2962."),
        ("duereh_2017_jpcb", "primary_article", "Journal of Physical Chemistry B", "Does Synergism in Microscopic Polarity Correlate with Extrema in Macroscopic Properties for Aqueous Mixtures of Dipolar Aprotic Solventsthis", "https://doi.org/10.1021/acs.jpcb.7b03446", "10.1021/acs.jpcb.7b03446", "authoritative_primary_source", "Selected primary source for acetone/water viscosity; rows are archived in NIST MDS2-2962."),
        ("vazquez_1995_jced", "primary_article", "Journal of Chemical & Engineering Data", "Surface Tension of Alcohol + Water from 20 to 50 degrees C", "https://doi.org/10.1021/je00019a016", "10.1021/je00019a016", "authoritative_primary_source", "Selected common source for methanol, ethanol, and isopropanol aqueous surface tension."),
        ("enders_2007_jced", "primary_article", "Journal of Chemical & Engineering Data", "Surface Tension of the Ternary System Water + Acetone + Toluene", "https://doi.org/10.1021/je7000182", "10.1021/je7000182", "authoritative_primary_source", "Selected source for acetone/water surface tension; binary rows are archived in NIST MDS2-2962."),
        ("rafati_2010_jced", "primary_article", "Journal of Chemical & Engineering Data", "Experimental Data and Correlation of Surface Tensions of the Binary and Ternary Systems of Water + Acetonitrile + 2-Propanol at 298.15 K", "https://doi.org/10.1021/je1001329", "10.1021/je1001329", "authoritative_primary_source_supplement", "Supplemental isopropanol/water series used only when it gives a materially narrower same-source bracket."),
        ("navarro_2012_jced", "primary_article", "Journal of Chemical & Engineering Data", "Physical Properties of Binary and Ternary Mixtures of 2-Propanol, Water, and 1-Butyl-3-methylimidazolium Tetrafluoroborate Ionic Liquid", "https://doi.org/10.1021/je201010s", "10.1021/je201010s", "authoritative_primary_source", "Selected primary source for isopropanol/water density."),
        ("estrada_baltazar_2003_jced", "primary_article", "Journal of Chemical & Engineering Data", "Experimental Densities and Excess Volumes for Binary Mixtures Containing Propionic Acid, Acetone, and Water from 283.15 K to 323.15 K", "https://doi.org/10.1021/je030102f", "10.1021/je030102f", "authoritative_primary_source", "Selected primary source for acetone/water density."),
        ("petong_2000_jpca", "primary_article", "Journal of Physical Chemistry A", "Water-Ethanol Mixtures at Different Compositions and Temperatures: A Dielectric Relaxation Study", "https://doi.org/10.1021/jp001393r", "10.1021/jp001393r", "primary_table_transcribed_local_hash_required", "Five Table 3 static-limit values are transcribed; current local-file hash status is recorded separately in the source-file registry."),
        ("smith_1998_fpe", "primary_article", "Fluid Phase Equilibria", "Relative permittivity and dielectric relaxation in aqueous alcohol solutions", "https://doi.org/10.1016/S0378-3812(97)00275-6", "10.1016/S0378-3812(97)00275-6", "primary_figure_digitized_C_tier_with_local_hash", "Methanol/water static-limit values are reconstructed from Figure 2 with a frozen publisher-image hash; they remain C tier pending an original numerical table."),
        ("uosaki_1997_jced", "primary_article", "Journal of Chemical & Engineering Data", "Static Relative Permittivities of Water + Acetone and Water + Dimethyl Sulfoxide under Pressures up to 300 MPa at 298.15 K", "https://doi.org/10.1021/je960390y", "10.1021/je960390y", "primary_table_transcription_C_tier_local_hash_required", "Five Table 1 acetone/water values remain C_secondary_reconstruction; current local-file hash status is recorded separately in the source-file registry."),
        ("iapws95_2018", "official_evaluated_reference", "IAPWS", "Revised Release on the IAPWS Formulation 1995 for the Thermodynamic Properties of Ordinary Water Substance", "https://iapws.org/relguide/IAPWS95-2018.pdf", "", "authoritative_exact_state_and_conversion_reference", "Used for the exact-state pure-water density proxy and the matching composition-conversion constant at 298.15 K and 0.101325 MPa."),
        ("iapws_liquid_water_2011", "official_evaluated_reference", "IAPWS", "Revised Supplementary Release on Properties of Liquid Water at 0.1 MPa", "https://www.iapws.org/relguide/LiquidWater.pdf", "", "authoritative_pure_component_reference", "Provides the evaluated static dielectric constant 78.375 at 298.15 K and 0.1 MPa."),
        ("deyhimi_karimzadeh_2010_jsc", "secondary_reconstruction_article", "Journal of Solution Chemistry", "Pitzer and Pitzer-Simonson-Clegg Modeling Approaches: Ternary HCl + 2-Propanol + Water Electrolyte System", "https://doi.org/10.1007/s10953-010-9497-x", "10.1007/s10953-010-9497-x", "estimate_validation_crosscheck_only", "Table 2 reproduces static dielectric constants for 10-50 mass% 2-propanol from its reference 35; values are C_secondary_reconstruction and are used only to gate the D_model_estimate layer."),
        ("wang_2023_fpe", "secondary_parameter_source_article", "Fluid Phase Equilibria", "Experimental and theoretical study on ion association in [Hmim][halide] + water/isopropanol mixtures", "https://doi.org/10.1016/j.fluid.2022.113680", "10.1016/j.fluid.2022.113680", "D_model_parameter_source_C_tier", "Table 2 reports the 298.15 K pure-isopropanol static relative permittivity 19.387 from cited literature; used only as a C-tier endpoint parameter for D-model estimates."),
        ("nist_webbook_water", "official_database_record", "NIST", "Water", "https://webbook.nist.gov/cgi/cbook.cgithisID=C7732185", "", "authoritative_conversion_reference", "Molecular formula and molar mass provenance."),
        ("nist_webbook_methanol", "official_database_record", "NIST", "Methanol", "https://webbook.nist.gov/cgi/cbook.cgithisID=C67561", "", "authoritative_conversion_reference", "Molecular formula and molar mass provenance."),
        ("nist_webbook_ethanol", "official_database_record", "NIST", "Ethanol", "https://webbook.nist.gov/cgi/cbook.cgithisID=C64175", "", "authoritative_conversion_reference", "Molecular formula and molar mass provenance."),
        ("nist_webbook_isopropanol", "official_database_record", "NIST", "2-Propanol", "https://webbook.nist.gov/cgi/cbook.cgithisID=C67630", "", "authoritative_conversion_reference", "Molecular formula and molar mass provenance."),
        ("nist_webbook_acetone", "official_database_record", "NIST", "Acetone", "https://webbook.nist.gov/cgi/cbook.cgithisID=C67641", "", "authoritative_conversion_reference", "Molecular formula and molar mass provenance."),
        ("nist_webbook", "legacy_source_alias", "NIST", "NIST Chemistry WebBook candidate rows", "https://webbook.nist.gov/", "", "legacy_crosscheck_only", "Compatibility source ID used by the frozen legacy value map; no value is promoted without an exact compound row, temperature, and original-reference binding."),
        ("pubchem_ethanol", "secondary_database_record", "PubChem", "Ethanol compound record", "https://pubchem.ncbi.nlm.nih.gov/compound/702", "", "legacy_crosscheck_only", "Compatibility source ID used by the frozen legacy value map; PubChem is not used as an authoritative measurement series."),
        ("pubchem_isopropanol", "secondary_database_record", "PubChem", "2-Propanol compound record", "https://pubchem.ncbi.nlm.nih.gov/compound/3776", "", "legacy_crosscheck_only", "Compatibility source ID used by the frozen legacy value map; PubChem is not used as an authoritative measurement series."),
        ("engineering_toolbox_dielectric", "secondary_table", "Engineering ToolBox", "Liquid dielectric constants", "https://www.engineeringtoolbox.com/liquid-dielectric-constants-d_1263.html", "", "legacy_crosscheck_only", "Not authoritative and never promoted to a primary measurement."),
        ("ddbst", "commercial_database_entry", "DDBST", "Dortmund Data Bank", "https://www.ddbst.com/ddb.html", "", "candidate_not_used", "No value is used without an export containing composition, temperature, method, and citation."),
        ("legacy_ink_properties", "legacy_candidate", "local workbook", "ink_properties.xlsx", "../Data/ink_properties.xlsx", "", "legacy_crosscheck_only", "Preserved without overwrite and excluded from authoritative measurement rows."),
    ]
    reference_aliases = {
        "gonzalez_2007_jct": ["trc4mitref_1456"],
        "pang_2007_jml": ["trc4mitref_1440"],
        "duereh_2017_jpcb": ["trc4mitref_2695"],
        "vazquez_1995_jced": ["trc4mitref_792"],
        "enders_2007_jced": ["trc4mitref_1459"],
        "rafati_2010_jced": ["trc4mitref_1760"],
        "iapws95_2018": ["IAPWS95-2018"],
        "iapws_liquid_water_2011": ["IAPWS SR6-08(2011)"],
    }
    return [
        {
            "source_id": source_id,
            "source_type": source_type,
            "authority": authority,
            "title": title,
            "url": url,
            "doi": doi,
            "registered_reference_ids": "|".join(
                [item for item in [doi, *reference_aliases.get(source_id, [])] if item]
            ),
            "accessed_on": ACCESS_DATE,
            "usage_status": usage_status,
            "notes": notes,
        }
        for source_id, source_type, authority, title, url, doi, usage_status, notes in rows
    ]


def source_files(raw_dir: Path) -> list[dict[str, str]]:
    specifications = [
        ("nist_mds2_2962_readme", "nist_mds2_2962", "2962_README.txt", "https://data.nist.gov/od/ds/ark:/88434/mds2-2962/2962_README.txt", ""),
        ("nist_mds2_2962_compounds", "nist_mds2_2962_compounds", "compounds.sdf", "https://data.nist.gov/od/ds/ark:/88434/mds2-2962/compounds.sdf", ""),
        ("nist_mds2_2962_refs", "nist_mds2_2962_refs", "refs.dat", "https://data.nist.gov/od/ds/ark:/88434/mds2-2962/refs.dat", ""),
        ("nist_mds2_2962_surften", "nist_mds2_2962_surften", "surften.dat", "https://data.nist.gov/od/ds/ark:/88434/mds2-2962/surften.dat", "c518579caa7b67d6894639e8a7d4c64cd50141046365044032dff50a16873887"),
        ("nist_mds2_2962_visc", "nist_mds2_2962_visc", "visc.dat", "https://data.nist.gov/od/ds/ark:/88434/mds2-2962/visc.dat", "4ca3797064e7f98c672d7b39c6b1b2a823a0a8d4519d97b83c8dd463d427dffd"),
        ("nist_thermoml_jct_2007", "nist_thermoml_jct_2007", "thermoml_jct_2007_05_004.json", "https://trc.nist.gov/ThermoML/10.1016/j.jct.2007.05.004.json", ""),
        ("nist_thermoml_je201010s", "nist_thermoml_je201010s", "thermoml_je201010s.json", "https://trc.nist.gov/ThermoML/10.1021/je201010s.json", ""),
        ("nist_thermoml_je030102f", "nist_thermoml_je030102f", "thermoml_je030102f.json", "https://trc.nist.gov/ThermoML/10.1021/je030102f.json", ""),
        ("iapws95_2018_pdf", "iapws95_2018", "IAPWS95-2018.pdf", "https://iapws.org/relguide/IAPWS95-2018.pdf", ""),
        ("iapws_liquid_water_2011_pdf", "iapws_liquid_water_2011", "LiquidWater.pdf", "https://www.iapws.org/relguide/LiquidWater.pdf", ""),
        ("smith_1998_figure2", "smith_1998_fpe", "smith_1998_fig2.gif", "https://ars.els-cdn.com/content/image/1-s2.0-S0378381297002756-gr2.gif", "2972be5c0e560321f9ab2cad2d4bcdf2ff7503bace3dc4252f02efb093f4765f"),
        ("wang_2023_fpe_pdf", "wang_2023_fpe", "wang_2023_fpe.pdf", "https://www.diva-portal.org/smash/get/diva2%3A1714684/FULLTEXT01.pdf", "a77bd0def1238f45e9112c703fe876c3f29931255bf552d2e01e889ee8ae0921"),
    ]
    rows: list[dict[str, str]] = []
    for file_id, source_id, file_name, url, expected in specifications:
        path = raw_dir / file_name
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if expected and actual != expected:
            raise ValueError(f"Published SHA256 mismatch for {file_name}: {actual} != {expected}")
        rows.append(
            {
                "source_file_id": file_id,
                "source_id": source_id,
                "file_name": file_name,
                "resource_url": url,
                "local_source_path_at_build": str(path),
                "sha256": actual,
                "expected_sha256": expected,
                "hash_status": (
                    "verified_against_publisher_resource_download"
                    if file_id == "smith_1998_figure2"
                    else "locally_hash_frozen_open_access_repository_download"
                    if file_id == "wang_2023_fpe_pdf"
                    else "verified_against_NIST_published_sha256"
                    if expected
                    else "locally_hash_frozen"
                ),
                "accessed_on": ACCESS_DATE,
                "parser": "build_property_source_inputs.py",
                "notes": "Raw file is external to the repository; selected rows and locators are frozen in config/property_measurement_series.csv.",
            }
        )
    indexed_primary_files = [
        {
                "source_file_id": "petong_2000_primary_pdf_indexed",
                "source_id": "petong_2000_jpca",
                "file_name": "jp001393r.pdf",
                "resource_url": "https://www.electronicsandbooks.com/edt/manual/Magazine/J/Journal%20of%20Physical%20Chemistry/a104/jp001393r.pdf",
                "expected_sha256": "",
                "accessed_on": ACCESS_DATE,
                "parser": "manual_table_transcription_reviewed_by_root_agent",
                "pending_notes": "Table 3 was recoverable from the indexed primary PDF, but the local primary file is unavailable; the missing hash is explicit.",
            },
            {
                "source_file_id": "uosaki_1997_primary_pdf_indexed",
                "source_id": "uosaki_1997_jced",
                "file_name": "je960390y.pdf",
                "resource_url": "https://www.electronicsandbooks.com/edt/manual/Magazine/J/Journal%20of%20Chemical%20%26%20Engineering%20Data/40-43/je960390y.pdf",
                "expected_sha256": "",
                "accessed_on": ACCESS_DATE,
                "parser": "manual_table_transcription_reviewed_by_root_agent",
                "pending_notes": "Table 1 was recoverable from the indexed primary PDF, but the local primary file is unavailable; rows remain C tier and the missing hash is explicit.",
            },
        ]
    for item in indexed_primary_files:
        local_path = raw_dir / item["file_name"]
        local_file_available = local_path.is_file() and local_path.stat().st_size > 0
        item["local_source_path_at_build"] = (
            str(local_path) if local_file_available else ""
        )
        item["sha256"] = sha256(local_path) if local_file_available else ""
        item["hash_status"] = (
            "locally_hash_frozen_primary_pdf"
            if local_file_available
            else "remote_indexed_primary_pdf_local_hash_pending"
        )
        item["notes"] = (
            "Locally hash-frozen primary PDF; table transcription remains independently reviewable."
            if local_file_available
            else item.pop("pending_notes")
        )
        item.pop("pending_notes", None)
        rows.append(item)
    contained_reference_ids = {
        "nist_mds2_2962_surften": (
            "trc4mitref_792|trc4mitref_1459|trc4mitref_1760"
        ),
        "nist_mds2_2962_visc": (
            "trc4mitref_1440|trc4mitref_1456|trc4mitref_2695"
        ),
        "nist_thermoml_jct_2007": "10.1016/j.jct.2007.05.004",
        "nist_thermoml_je201010s": "10.1021/je201010s",
        "nist_thermoml_je030102f": "10.1021/je030102f",
        "iapws95_2018_pdf": "IAPWS95-2018",
        "iapws_liquid_water_2011_pdf": "IAPWS SR6-08(2011)",
        "smith_1998_figure2": "10.1016/S0378-3812(97)00275-6",
        "wang_2023_fpe_pdf": "10.1016/j.fluid.2022.113680",
        "petong_2000_primary_pdf_indexed": "10.1021/jp001393r",
        "uosaki_1997_primary_pdf_indexed": "10.1021/je960390y",
    }
    for row in rows:
        row["contained_reference_ids"] = contained_reference_ids.get(
            row["source_file_id"], ""
        )
    return rows


def component_constants() -> list[dict[str, Any]]:
    rows = [
        {
            "component": "water",
            "cas_number": "7732-18-5",
            "molecular_formula": "H2O",
            "molar_mass_g_mol": 18.01528,
            "molar_mass_source_id": "nist_webbook_water",
            "pure_density_kg_m3": 997.0474,
            "pure_density_temperature_K": 298.15,
            "pure_density_pressure_kPa": 101.325,
            "pure_density_source_id": "iapws95_2018",
            "pure_density_source_file_id": "iapws95_2018_pdf",
            "pure_density_source_file_sha256": "fbaa0abc4cbf071152a5cbe230a586bd045db6c16f0cb2c872caf4e8888515dd",
            "pure_density_evidence_tier": "A_direct",
            "pure_density_uncertainty_kg_m3": "",
            "source_locator": "IAPWS-95 evaluated equation at T=298.15 K and p=0.101325 MPa",
            "notes": "Exact target-state official evaluated reference used only for volume-to-mole/mass conversion; A_direct denotes no temperature/composition interpolation and does not turn it into an ink-property measurement.",
        },
        {
            "component": "methanol",
            "cas_number": "67-56-1",
            "molecular_formula": "CH4O",
            "molar_mass_g_mol": 32.04186,
            "molar_mass_source_id": "nist_webbook_methanol",
            "pure_density_kg_m3": 787.2,
            "pure_density_temperature_K": 298.15,
            "pure_density_pressure_kPa": 101,
            "pure_density_source_id": "gonzalez_2007_jct",
            "pure_density_source_file_id": "nist_thermoml_jct_2007",
            "pure_density_source_file_sha256": "4719c84b7d855c457936408224879f8335712447c5731d8f108792f9e06c5914",
            "pure_density_evidence_tier": "A_direct",
            "pure_density_uncertainty_kg_m3": 0.46,
            "source_locator": "ThermoML:PureOrMixtureData[1]:NumValues[1]",
            "notes": "Reported combined expanded uncertainty at 95% confidence.",
        },
        {
            "component": "ethanol",
            "cas_number": "64-17-5",
            "molecular_formula": "C2H6O",
            "molar_mass_g_mol": 46.06844,
            "molar_mass_source_id": "nist_webbook_ethanol",
            "pure_density_kg_m3": 785.46,
            "pure_density_temperature_K": 298.15,
            "pure_density_pressure_kPa": 101,
            "pure_density_source_id": "gonzalez_2007_jct",
            "pure_density_source_file_id": "nist_thermoml_jct_2007",
            "pure_density_source_file_sha256": "4719c84b7d855c457936408224879f8335712447c5731d8f108792f9e06c5914",
            "pure_density_evidence_tier": "A_direct",
            "pure_density_uncertainty_kg_m3": 0.47,
            "source_locator": "ThermoML:PureOrMixtureData[3]:NumValues[1]",
            "notes": "Reported combined expanded uncertainty at 95% confidence.",
        },
        {
            "component": "isopropanol",
            "cas_number": "67-63-0",
            "molecular_formula": "C3H8O",
            "molar_mass_g_mol": 60.09502,
            "molar_mass_source_id": "nist_webbook_isopropanol",
            "pure_density_kg_m3": 780.82,
            "pure_density_temperature_K": 298.15,
            "pure_density_pressure_kPa": 101,
            "pure_density_source_id": "navarro_2012_jced",
            "pure_density_source_file_id": "nist_thermoml_je201010s",
            "pure_density_source_file_sha256": "5e3a3f3a46a25369c89ae1c30719ad7fc494846828b59e98f0a1ea4c67bfade0",
            "pure_density_evidence_tier": "A_direct",
            "pure_density_uncertainty_kg_m3": 1.01,
            "source_locator": "ThermoML:PureOrMixtureData[1]:NumValues[0]",
            "notes": "Reported combined expanded uncertainty at 95% confidence.",
        },
        {
            "component": "acetone",
            "cas_number": "67-64-1",
            "molecular_formula": "C3H6O",
            "molar_mass_g_mol": 58.07914,
            "molar_mass_source_id": "nist_webbook_acetone",
            "pure_density_kg_m3": 784.638,
            "pure_density_temperature_K": 298.146,
            "pure_density_pressure_kPa": 101.325,
            "pure_density_source_id": "estrada_baltazar_2003_jced",
            "pure_density_source_file_id": "nist_thermoml_je030102f",
            "pure_density_source_file_sha256": "bdbc814a51e1647e8bc1517736832d15ba688334ef0a5c063f825267317aa884",
            "pure_density_evidence_tier": "A_direct",
            "pure_density_uncertainty_kg_m3": 0.144,
            "source_locator": "ThermoML:PureOrMixtureData[1]:NumValues[3]",
            "notes": "T differs from 298.15 K by 0.004 K; reported combined expanded uncertainty at 95% confidence.",
        },
    ]
    dielectric_parameters: dict[str, dict[str, Any]] = {
        "water": {
            "static_dielectric_constant": 78.375,
            "static_dielectric_temperature_K": 298.15,
            "static_dielectric_pressure_kPa": 100.0,
            "static_dielectric_definition": "static_low_frequency_limit",
            "static_dielectric_source_id": "iapws_liquid_water_2011",
            "static_dielectric_source_file_id": "iapws_liquid_water_2011_pdf",
            "static_dielectric_source_file_sha256": "04bd37dd7f4c9c535230af4c51d2a803912d1d41c1656e20527eae598bc55ff4",
            "static_dielectric_evidence_tier": "A_direct",
            "static_dielectric_source_locator": "IAPWS LiquidWater table at T=298.15 K and p=0.1 MPa",
            "static_dielectric_notes": "Exact-state official evaluated reference endpoint for the D-model parameter registry.",
        },
        "isopropanol": {
            "static_dielectric_constant": 19.387,
            "static_dielectric_temperature_K": 298.15,
            "static_dielectric_pressure_kPa": "",
            "static_dielectric_definition": "static_low_frequency_limit",
            "static_dielectric_source_id": "wang_2023_fpe",
            "static_dielectric_source_file_id": "wang_2023_fpe_pdf",
            "static_dielectric_source_file_sha256": "a77bd0def1238f45e9112c703fe876c3f29931255bf552d2e01e889ee8ae0921",
            "static_dielectric_evidence_tier": "C_secondary_reconstruction",
            "static_dielectric_source_locator": "Table 2; T=298.15 K; pure isopropanol; epsilon=19.387",
            "static_dielectric_notes": "Secondary literature compilation used only as a Bruggeman/linear/Looyenga endpoint for D_model_estimate values; not independent mixture evidence.",
        },
    }
    parameter_fields = (
        "static_dielectric_constant",
        "static_dielectric_temperature_K",
        "static_dielectric_pressure_kPa",
        "static_dielectric_definition",
        "static_dielectric_source_id",
        "static_dielectric_source_file_id",
        "static_dielectric_source_file_sha256",
        "static_dielectric_evidence_tier",
        "static_dielectric_source_locator",
        "static_dielectric_notes",
    )
    for row in rows:
        row.update({field: "" for field in parameter_fields})
        row.update(dielectric_parameters.get(str(row["component"]), {}))
    return rows


def source_gaps() -> list[dict[str, str]]:
    return [
        {"property_name": "dielectric_constant", "mixture_system": "ethanol/water", "registered_source_id": "petong_2000_jpca", "status": "five_primary_table_rows_transcribed_x_0.22_to_1.0", "model_use_status": "partial_source_backed", "reason": "Petong Table 3 is available at five ethanol mole fractions; the pure-water endpoint is an IAPWS evaluated value, not a same-source Petong row.", "next_action": "Hash a local copy of the primary PDF; do not call x<0.22 interpolation same-source unless a Petong water endpoint is verified."},
        {"property_name": "dielectric_constant", "mixture_system": "methanol/water", "registered_source_id": "smith_1998_fpe", "status": "primary_figure_digitized_C_tier", "model_use_status": "source_backed_C_tier_sensitivity", "reason": "The publisher-hosted Figure 2 is locally hash-frozen and digitized with explicit axis calibration; the absence of an original numerical table prevents A/B classification.", "next_action": "Obtain the original numerical table or author data to replace and potentially upgrade the C-tier reconstruction."},
        {"property_name": "dielectric_constant", "mixture_system": "acetone/water", "registered_source_id": "uosaki_1997_jced", "status": "five_primary_table_rows_transcribed_as_C_tier", "model_use_status": "source_backed_C_tier_sensitivity", "reason": "Table 1 values are explicit, but the remote PDF could not be locally hashed in this build.", "next_action": "Hash and independently recheck the primary PDF before upgrading these rows to A_direct."},
        {"property_name": "dielectric_constant", "mixture_system": "isopropanol/water", "registered_source_id": "wang_2023_fpe", "status": "no_accepted_complete_primary_mixture_series_endpoint_parameter_hash_frozen", "model_use_status": "estimated_layer_only", "reason": "No complete, verified static-limit mixture table is available locally; the pure-isopropanol epsilon=19.387 endpoint is a hash-frozen C-tier secondary parameter, not mixture evidence.", "next_action": "Keep Bruggeman/Looyenga/linear values in the D_model_estimate layer and replace them only after a complete primary static-mixture series is verified."},
    ]


def ipa_dielectric_validation_rows() -> list[dict[str, Any]]:
    return [
        {
            "validation_id": "ipa_dielectric_validation__000",
            "isopropanol_mass_fraction": 0.0,
            "temperature_K": 298.15,
            "dielectric_constant": 78.375,
            "evidence_tier": "C_secondary_reconstruction",
            "source_id": "iapws_liquid_water_2011",
            "source_locator": "IAPWS LiquidWater table at 298.15 K and 0.1 MPa",
            "model_use_status": "validation_only",
        },
        *[
            {
                "validation_id": f"ipa_dielectric_validation__{index:03d}",
                "isopropanol_mass_fraction": mass_percent / 100.0,
                "temperature_K": 298.15,
                "dielectric_constant": dielectric,
                "evidence_tier": "C_secondary_reconstruction",
                "source_id": "deyhimi_karimzadeh_2010_jsc",
                "source_locator": f"Table 2; 2-propanol mass fraction {mass_percent}%",
                "model_use_status": "validation_only_not_authoritative_measurement",
            }
            for index, (mass_percent, dielectric) in enumerate(
                ((10, 71.40), (20, 64.10), (30, 56.90), (40, 49.70)),
                start=1,
            )
        ],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/private/tmp/ink_property_doc_review"),
        help="Directory containing the downloaded NIST raw resources.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config",
        help="Output configuration directory.",
    )
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()
    config_dir = args.config_dir.resolve()

    file_rows = source_files(raw_dir)
    references = refs_map(raw_dir / "refs.dat")
    measurement_rows = mds_measurements(raw_dir, references)
    measurement_rows.extend(thermoml_density_measurements(raw_dir))
    measurement_rows.extend(pure_component_reference_measurements())
    measurement_rows.extend(dielectric_measurements())
    registered_file_hashes = {
        str(row["source_file_id"]): str(row.get("sha256", ""))
        for row in file_rows
    }
    for row in measurement_rows:
        if row["source_file_id"] in registered_file_hashes:
            row["raw_file_sha256"] = registered_file_hashes[row["source_file_id"]]
    measurement_rows.sort(
        key=lambda row: (
            str(row["property_name"]),
            str(row["component_1"]),
            int(row["series_priority"]),
            str(row["series_id"]),
            float(row["composition_value"]),
        )
    )
    if len(measurement_rows) != 199:
        raise ValueError(f"Expected 199 selected measurement/reference rows, found {len(measurement_rows)}")

    write_csv(
        config_dir / "property_sources.csv",
        ["source_id", "source_type", "authority", "title", "url", "doi", "registered_reference_ids", "accessed_on", "usage_status", "notes"],
        source_catalog(),
    )
    write_csv(
        config_dir / "property_source_files.csv",
        ["source_file_id", "source_id", "file_name", "resource_url", "local_source_path_at_build", "sha256", "expected_sha256", "contained_reference_ids", "hash_status", "accessed_on", "parser", "notes"],
        file_rows,
    )
    write_csv(config_dir / "property_measurement_series.csv", MEASUREMENT_FIELDS, measurement_rows)
    # The authoritative table is the accepted source-backed subset.  Derived
    # B-level recipe interpolations and D-level estimates belong downstream.
    write_csv(
        config_dir / "authoritative_property_measurements.csv",
        MEASUREMENT_FIELDS,
        [row for row in measurement_rows if not str(row["model_use_status"]).startswith("held")],
    )
    write_csv(
        config_dir / "property_component_constants.csv",
        [
            "component",
            "cas_number",
            "molecular_formula",
            "molar_mass_g_mol",
            "molar_mass_source_id",
            "pure_density_kg_m3",
            "pure_density_temperature_K",
            "pure_density_pressure_kPa",
            "pure_density_source_id",
            "pure_density_source_file_id",
            "pure_density_source_file_sha256",
            "pure_density_evidence_tier",
            "pure_density_uncertainty_kg_m3",
            "static_dielectric_constant",
            "static_dielectric_temperature_K",
            "static_dielectric_pressure_kPa",
            "static_dielectric_definition",
            "static_dielectric_source_id",
            "static_dielectric_source_file_id",
            "static_dielectric_source_file_sha256",
            "static_dielectric_evidence_tier",
            "static_dielectric_source_locator",
            "static_dielectric_notes",
            "source_locator",
            "notes",
        ],
        component_constants(),
    )
    write_csv(
        config_dir / "property_source_gaps.csv",
        ["property_name", "mixture_system", "registered_source_id", "status", "model_use_status", "reason", "next_action"],
        source_gaps(),
    )
    write_csv(
        config_dir / "property_dielectric_validation.csv",
        [
            "validation_id",
            "isopropanol_mass_fraction",
            "temperature_K",
            "dielectric_constant",
            "evidence_tier",
            "source_id",
            "source_locator",
            "model_use_status",
        ],
        ipa_dielectric_validation_rows(),
    )

    counts: dict[str, int] = {}
    for row in measurement_rows:
        counts[str(row["property_name"])] = counts.get(str(row["property_name"]), 0) + 1
    print(json.dumps({"measurement_rows": len(measurement_rows), "by_property": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
