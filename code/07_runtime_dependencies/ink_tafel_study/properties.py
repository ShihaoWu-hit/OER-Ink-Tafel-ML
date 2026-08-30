from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROPERTY_UNITS = {
    "viscosity": "mPa s",
    "surface_tension": "mN m-1",
    "density": "kg m-3",
    "dielectric_constant": "dimensionless",
}
PROPERTY_OUTPUT_COLUMNS = {
    "viscosity": "solvent_mixture_property_proxy_viscosity_mPa_s",
    "surface_tension": "solvent_mixture_property_proxy_surface_tension_mN_m",
    "density": "solvent_mixture_property_proxy_density_kg_m3",
    "dielectric_constant": "solvent_mixture_property_proxy_dielectric_constant",
}
SUPPORTED_COMPOSITION_BASES = {
    "volume_fraction",
    "mole_fraction",
    "mass_fraction",
}
SOLVENT_FRACTIONS = {
    "acetone": "solvent_fraction_acetone",
    "water": "solvent_fraction_water",
    "ethanol": "solvent_fraction_ethanol",
    "isopropanol": "solvent_fraction_isopropanol",
    "methanol": "solvent_fraction_methanol",
    "other": "solvent_fraction_other",
}
LEGACY_PROPERTY_COLUMNS = {
    "viscosity": ("viscosity_Pa_s_candidate", "Pa s", 1000.0, "mPa s"),
    "surface_tension": ("surface_tension_N_m_candidate", "N m-1", 1000.0, "mN m-1"),
    "density": ("density_kg_m3_candidate", "kg m-3", 1.0, "kg m-3"),
    "dielectric_constant": (
        "dielectric_constant_candidate",
        "dimensionless",
        1.0,
        "dimensionless",
    ),
}


def record_composition_key(row: pd.Series, *, precision: int = 6) -> str:
    parts: list[str] = []
    for solvent, column in SOLVENT_FRACTIONS.items():
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if np.isfinite(value) and value > 10 ** (-precision):
            parts.append(f"{solvent}={float(value):.{precision}f}")
    return "|".join(parts) if parts else "unknown"


def _validate_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "solvent_composition_key",
        "property_name",
        "value",
        "unit",
        "temperature_K",
        "composition_basis",
        "experimental_method",
        "source_doi",
        "source_url",
        "access_date",
        "source_id",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Authoritative property table is missing columns: {missing}")
    out = frame.copy()
    out = out.loc[out.astype("string").apply(lambda column: column.str.strip()).ne("").any(axis=1)].copy()
    if out.empty:
        return out
    out["property_name"] = out["property_name"].astype(str).str.strip().str.lower()
    unknown = sorted(set(out["property_name"]) - set(PROPERTY_UNITS))
    if unknown:
        raise ValueError(f"Unsupported property names: {unknown}")
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    out["temperature_K"] = pd.to_numeric(out["temperature_K"], errors="raise")
    out["composition_basis"] = (
        out["composition_basis"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )
    unsupported_bases = sorted(set(out["composition_basis"]) - SUPPORTED_COMPOSITION_BASES)
    if unsupported_bases:
        raise ValueError(f"Unsupported composition bases: {unsupported_bases}")
    if (~np.isfinite(out["value"]) | ~np.isfinite(out["temperature_K"])).any():
        raise ValueError("Property values and temperatures must be finite")
    for _, row in out.iterrows():
        expected_unit = PROPERTY_UNITS[str(row["property_name"])]
        if str(row["unit"]).strip() != expected_unit:
            raise ValueError(
                f"{row['property_name']} must use unit {expected_unit!r}, found {row['unit']!r}"
            )
        if not str(row["composition_basis"]).strip() or not str(row["experimental_method"]).strip():
            raise ValueError("composition_basis and experimental_method are required")
        if not str(row.get("source_doi", "")).strip() and not str(row.get("source_url", "")).strip():
            raise ValueError("Every authoritative property row needs a DOI or URL")
        if not str(row["access_date"]).strip():
            raise ValueError("Every authoritative property row needs an access_date")
        if not str(row["source_id"]).strip():
            raise ValueError("Every authoritative property row needs a source_id")
    return out


def _validate_source_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"source_id", "source_type", "authority", "title", "usage_status"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Property source catalogue is missing columns: {missing}")
    out = frame.copy()
    out["source_id"] = out["source_id"].astype("string").str.strip()
    if out["source_id"].isna().any() or out["source_id"].eq("").any():
        raise ValueError("Property source catalogue contains a blank source_id")
    if out["source_id"].duplicated().any():
        duplicates = sorted(out.loc[out["source_id"].duplicated(False), "source_id"].unique())
        raise ValueError(f"Property source catalogue contains duplicate source_id values: {duplicates}")
    return out


def build_legacy_property_source_audit(
    candidate_registry: pd.DataFrame | str | Path,
    value_source_map: pd.DataFrame | str | Path,
    source_catalog: pd.DataFrame | str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bind every legacy candidate to an explicit provenance tier.

    This audit does not promote a value into the predictor table.  It separates
    numerical documentation from the stricter composition/temperature gates
    used by :func:`attach_authoritative_property_proxies`.
    """

    candidates = (
        candidate_registry.copy()
        if isinstance(candidate_registry, pd.DataFrame)
        else pd.read_csv(candidate_registry)
    )
    mapping = (
        value_source_map.copy()
        if isinstance(value_source_map, pd.DataFrame)
        else pd.read_csv(value_source_map)
    )
    sources = (
        source_catalog.copy()
        if isinstance(source_catalog, pd.DataFrame)
        else pd.read_csv(source_catalog)
    )
    sources = _validate_source_catalog(sources)
    required_candidate = {"record_id", "mixture_system_raw", "source_volume_fraction_percent_raw"}
    missing_candidate = sorted(required_candidate - set(candidates))
    if missing_candidate:
        raise ValueError(f"Legacy property registry is missing columns: {missing_candidate}")
    required_mapping = {
        "mixture_system",
        "source_volume_fraction_percent",
        "property_name",
        "legacy_value",
        "legacy_unit",
        "source_id",
        "source_match_status",
        "temperature_eligibility",
        "model_use_status",
    }
    missing_mapping = sorted(required_mapping - set(mapping))
    if missing_mapping:
        raise ValueError(f"Property value source map is missing columns: {missing_mapping}")
    unknown_source_ids = sorted(set(mapping["source_id"].astype(str)) - set(sources["source_id"].astype(str)))
    if unknown_source_ids:
        raise ValueError(f"Property value source map references unknown source_id values: {unknown_source_ids}")
    duplicate_key = mapping.duplicated(
        ["mixture_system", "source_volume_fraction_percent", "property_name", "legacy_value"],
        keep=False,
    )
    if duplicate_key.any():
        raise ValueError("Property value source map contains duplicate value-level keys")

    mapping = mapping.copy()
    mapping["mixture_system"] = mapping["mixture_system"].astype("string").str.strip().str.lower()
    mapping["source_volume_fraction_percent"] = pd.to_numeric(
        mapping["source_volume_fraction_percent"], errors="raise"
    )
    mapping["legacy_value"] = pd.to_numeric(mapping["legacy_value"], errors="raise")
    mapping["property_name"] = mapping["property_name"].astype("string").str.strip().str.lower()

    audit_rows: list[dict[str, Any]] = []
    mapping_columns = [column for column in mapping.columns if column not in {"legacy_value", "legacy_unit"}]
    for candidate in candidates.to_dict(orient="records"):
        mixture = str(candidate.get("mixture_system_raw", "")).strip().lower()
        source_volume = pd.to_numeric(
            pd.Series([candidate.get("source_volume_fraction_percent_raw")]), errors="coerce"
        ).iloc[0]
        for property_name, (candidate_column, raw_unit, factor, model_unit) in LEGACY_PROPERTY_COLUMNS.items():
            raw_value = pd.to_numeric(
                pd.Series([candidate.get(candidate_column)]), errors="coerce"
            ).iloc[0]
            base = {
                "record_id": int(candidate["record_id"]),
                "mixture_system": mixture,
                "ink_volume_fraction_percent": pd.to_numeric(
                    pd.Series([candidate.get("ink_volume_fraction_percent_raw")]),
                    errors="coerce",
                ).iloc[0],
                "source_volume_fraction_percent": source_volume,
                "property_name": property_name,
                "legacy_candidate_value": raw_value,
                "legacy_unit": raw_unit,
                "model_unit": model_unit,
                "unit_conversion_factor": factor,
                "converted_candidate_value": float(raw_value) * factor if np.isfinite(raw_value) else np.nan,
            }
            if not np.isfinite(raw_value):
                audit_rows.append(
                    {
                        **base,
                        "source_match_status": "missing",
                        "temperature_eligibility": "missing",
                        "model_use_status": "missing",
                        "source_measurement_row_ready": False,
                    }
                )
                continue
            eligible = mapping.loc[
                mapping["mixture_system"].eq(mixture)
                & mapping["property_name"].eq(property_name)
                & np.isclose(
                    mapping["source_volume_fraction_percent"],
                    float(source_volume),
                    rtol=0.0,
                    atol=5e-6,
                )
                & np.isclose(mapping["legacy_value"], float(raw_value), rtol=1e-9, atol=1e-12)
            ]
            if len(eligible) != 1:
                raise ValueError(
                    "Every non-missing legacy property candidate must have exactly one value-level "
                    f"source-map row; record={candidate['record_id']}, property={property_name}, "
                    f"matches={len(eligible)}"
                )
            source_row = eligible.iloc[0]
            mapped = {column: source_row.get(column) for column in mapping_columns}
            status = str(source_row["model_use_status"])
            audit_rows.append(
                {
                    **base,
                    **mapped,
                    "source_measurement_row_ready": status.startswith("eligible_source_row"),
                }
            )

    audit = pd.DataFrame(audit_rows).sort_values(["record_id", "property_name"]).reset_index(drop=True)
    nonmissing = audit["legacy_candidate_value"].notna()
    exact = audit["source_match_status"].isin({"exact_nist_row", "multiple_exact_nist_rows"})
    ready = audit["source_measurement_row_ready"].astype(bool)
    qa = {
        "status": "source_reconstruction_complete_with_use_gates",
        "record_property_rows": len(audit),
        "legacy_candidate_values_documented": int(nonmissing.sum()),
        "exact_nist_record_property_matches": int((nonmissing & exact).sum()),
        "source_measurement_rows_ready": int((nonmissing & ready).sum()),
        "off_target_temperature_rows": int(
            audit["model_use_status"].astype(str).eq("hold_off_target_temperature").sum()
        ),
        "no_exact_raw_anchor_rows": int(
            audit["model_use_status"].astype(str).eq("hold_no_exact_raw_anchor").sum()
        ),
        "pure_source_rows_pending": int(
            audit["model_use_status"].astype(str).eq("hold_pure_source_row_pending").sum()
        ),
        "source_catalog_rows": len(sources),
        "value_level_source_map_rows": len(mapping),
        "unit_conversions": {"viscosity": "Pa s * 1000 -> mPa s", "surface_tension": "N m-1 * 1000 -> mN m-1"},
        "claim_boundary": (
            "The legacy values are numerically documented candidates. Only rows passing separate "
            "composition and temperature gates may populate solvent-mixture property proxies."
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    audit.to_csv(destination / "legacy_property_source_audit.csv", index=False)
    mapping.to_csv(destination / "property_value_source_map_snapshot.csv", index=False)
    sources.to_csv(destination / "property_sources_snapshot.csv", index=False)
    (destination / "property_source_reconstruction_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit, qa


def attach_source_reconstructed_property_proxies(
    records: pd.DataFrame,
    source_audit: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    base_qa: dict[str, Any] | None = None,
    max_abs_composition_delta_percent: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach only source-verified rows that pass the registered temperature gate.

    The source measurement may be at a nearby composition.  That approximation
    is explicit and bounded in volume-percentage points; it is never labelled
    measured-exact at the canonical recipe composition.
    """

    audit = (
        source_audit.copy()
        if isinstance(source_audit, pd.DataFrame)
        else pd.read_csv(source_audit)
    )
    required = {
        "record_id",
        "property_name",
        "converted_candidate_value",
        "ink_volume_fraction_percent",
        "source_volume_fraction_percent",
        "source_measurement_row_ready",
        "source_match_status",
        "model_use_status",
    }
    missing = sorted(required - set(audit))
    if missing:
        raise ValueError(f"Legacy property source audit is missing columns: {missing}")
    if max_abs_composition_delta_percent < 0:
        raise ValueError("max_abs_composition_delta_percent must be non-negative")
    registry = audit.copy()
    registry["source_target_composition_delta_percent"] = (
        pd.to_numeric(registry["source_volume_fraction_percent"], errors="coerce")
        - pd.to_numeric(registry["ink_volume_fraction_percent"], errors="coerce")
    )
    source_ready = (
        registry["source_measurement_row_ready"]
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    composition_ready = (
        registry["source_target_composition_delta_percent"].abs()
        <= float(max_abs_composition_delta_percent)
    )
    value_ready = pd.to_numeric(registry["converted_candidate_value"], errors="coerce").notna()
    ready = source_ready & composition_ready & value_ready
    registry["proxy_status"] = np.select(
        [
            ready,
            source_ready & ~composition_ready,
            registry["model_use_status"].astype(str).eq("hold_off_target_temperature"),
            registry["model_use_status"].astype(str).eq("hold_no_exact_raw_anchor"),
            registry["model_use_status"].astype(str).eq("hold_pure_source_row_pending"),
        ],
        [
            "measured_source_row_within_composition_tolerance",
            "held_source_target_composition_delta",
            "held_off_target_temperature",
            "held_no_exact_raw_anchor",
            "held_pure_source_row_pending",
        ],
        default="missing",
    )
    registry["proxy_value"] = np.where(
        ready,
        pd.to_numeric(registry["converted_candidate_value"], errors="coerce"),
        np.nan,
    )
    registry["composition_extrapolation_used"] = False
    registry["ink_measurement_claim_allowed"] = False

    result = records.copy()
    if result["record_id"].duplicated().any():
        raise ValueError("Source-reconstructed property attachment requires unique record_id values")
    for property_name, output_column in PROPERTY_OUTPUT_COLUMNS.items():
        values = registry.loc[
            registry["property_name"].astype(str).eq(property_name),
            ["record_id", "proxy_value"],
        ]
        if values["record_id"].duplicated().any():
            raise ValueError(f"Duplicate source-reconstructed rows for {property_name}")
        incoming = values.rename(columns={"proxy_value": f"{output_column}__reconstructed"})
        result = result.merge(incoming, on="record_id", how="left", validate="one_to_one")
        reconstructed_column = f"{output_column}__reconstructed"
        if output_column in result:
            # A separately curated exact-composition authoritative value takes
            # precedence over the bounded near-composition reconstruction.
            result[output_column] = pd.to_numeric(result[output_column], errors="coerce").combine_first(
                pd.to_numeric(result[reconstructed_column], errors="coerce")
            )
        else:
            result[output_column] = pd.to_numeric(result[reconstructed_column], errors="coerce")
        result = result.drop(columns=[reconstructed_column])

    property_counts = {
        property_name: int(
            result[output_column].notna().sum()
        )
        for property_name, output_column in PROPERTY_OUTPUT_COLUMNS.items()
    }
    inherited = dict(base_qa or {})
    qa = {
        **inherited,
        "status": "ready_partial_source_reconstruction" if ready.any() else inherited.get(
            "status", "unavailable_no_authoritative_values"
        ),
        "record_property_values_ready": int(
            result[list(PROPERTY_OUTPUT_COLUMNS.values())].notna().sum().sum()
        ),
        "record_property_values_ready_by_property": property_counts,
        "source_reconstructed_values_attached": int(ready.sum()),
        "records_with_all_four_proxies": int(
            result[list(PROPERTY_OUTPUT_COLUMNS.values())].notna().all(axis=1).sum()
        ),
        "max_abs_source_target_volume_fraction_delta_percent": float(
            max_abs_composition_delta_percent
        ),
        "maximum_attached_abs_composition_delta_percent": float(
            registry.loc[ready, "source_target_composition_delta_percent"].abs().max()
        )
        if ready.any()
        else np.nan,
        "composition_tolerance_attachment_used": bool(ready.any()),
        "composition_extrapolation_used": False,
        "temperature_extrapolation_used": False,
        "legacy_candidate_values_used": False,
        "source_reconstructed_values_used": bool(ready.any()),
        "claim_boundary": (
            "source-backed solvent-mixture property proxy at a bounded nearby composition; "
            "not a measured catalyst/Nafion ink property"
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    registry.to_csv(destination / "property_proxy_registry.csv", index=False)
    (destination / "property_proxy_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result, registry, qa


def _at_target_temperature(group: pd.DataFrame, target_temperature_K: float) -> dict[str, Any]:
    group = group.sort_values("temperature_K")
    exact = group.loc[np.abs(group["temperature_K"] - target_temperature_K) <= 0.05]
    if not exact.empty:
        return {
            "proxy_value": float(exact["value"].mean()),
            "proxy_status": "measured_exact" if len(exact) == 1 else "measured_exact_replicate_mean",
            "temperature_low_K": float(exact["temperature_K"].min()),
            "temperature_high_K": float(exact["temperature_K"].max()),
            "value_low": float(exact["value"].min()),
            "value_high": float(exact["value"].max()),
            "n_source_points": len(exact),
        }
    lower = group.loc[group["temperature_K"] < target_temperature_K].tail(1)
    upper = group.loc[group["temperature_K"] > target_temperature_K].head(1)
    if lower.empty or upper.empty:
        return {
            "proxy_value": np.nan,
            "proxy_status": "missing_no_temperature_bracket",
            "temperature_low_K": float(lower["temperature_K"].iloc[0]) if not lower.empty else np.nan,
            "temperature_high_K": float(upper["temperature_K"].iloc[0]) if not upper.empty else np.nan,
            "value_low": float(lower["value"].iloc[0]) if not lower.empty else np.nan,
            "value_high": float(upper["value"].iloc[0]) if not upper.empty else np.nan,
            "n_source_points": len(group),
        }
    low = lower.iloc[0]
    high = upper.iloc[0]
    fraction = (target_temperature_K - float(low["temperature_K"])) / (
        float(high["temperature_K"]) - float(low["temperature_K"])
    )
    value = float(low["value"]) + fraction * (float(high["value"]) - float(low["value"]))
    return {
        "proxy_value": value,
        "proxy_status": "interpolated_bracketed",
        "temperature_low_K": float(low["temperature_K"]),
        "temperature_high_K": float(high["temperature_K"]),
        "value_low": float(low["value"]),
        "value_high": float(high["value"]),
        "n_source_points": 2,
    }


def _joined_source_values(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = {
        str(value).strip()
        for value in frame[column]
        if str(value).strip() and str(value).strip().lower() not in {"nan", "none", "<na>"}
    }
    return " | ".join(sorted(values))


def attach_authoritative_property_proxies(
    records: pd.DataFrame,
    measurements: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    source_catalog: pd.DataFrame | str | Path | None = None,
    target_temperature_K: float = 298.15,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach only exact-composition measured/bracket-interpolated proxies."""

    raw = measurements.copy() if isinstance(measurements, pd.DataFrame) else pd.read_csv(measurements)
    validated = _validate_measurements(raw)
    sources = pd.DataFrame()
    if source_catalog is not None:
        raw_sources = (
            source_catalog.copy()
            if isinstance(source_catalog, pd.DataFrame)
            else pd.read_csv(source_catalog)
        )
        sources = _validate_source_catalog(raw_sources)
        unknown_source_ids = sorted(
            set(validated.get("source_id", pd.Series(dtype="string")).dropna().astype(str))
            - set(sources["source_id"].astype(str))
        )
        if unknown_source_ids:
            raise ValueError(
                "Authoritative property rows reference unknown source_id values: "
                f"{unknown_source_ids}"
            )
    result = records.copy()
    result["solvent_composition_key"] = result.apply(record_composition_key, axis=1)
    registry_rows: list[dict[str, Any]] = []
    for _, record in result.iterrows():
        key = str(record["solvent_composition_key"])
        for property_name, output_column in PROPERTY_OUTPUT_COLUMNS.items():
            same_key = validated.loc[
                validated["solvent_composition_key"].astype(str).eq(key)
                & validated["property_name"].astype(str).eq(property_name)
            ]
            # Canonical recipe coordinates are solvent volume fractions.  A
            # numerically identical mole- or mass-fraction key is not the same
            # mixture and must never be attached without a separately audited
            # basis conversion.
            group = same_key.loc[same_key["composition_basis"].eq("volume_fraction")]
            if group.empty:
                evaluated = {
                    "proxy_value": np.nan,
                    "proxy_status": (
                        "missing_composition_basis_not_volume_fraction"
                        if not same_key.empty
                        else "missing_no_exact_composition_data"
                    ),
                    "temperature_low_K": np.nan,
                    "temperature_high_K": np.nan,
                    "value_low": np.nan,
                    "value_high": np.nan,
                    "n_source_points": 0,
                }
                source_frame = same_key if not same_key.empty else group
            else:
                evaluated = _at_target_temperature(group, target_temperature_K)
                source_frame = group
            source_dois = _joined_source_values(source_frame, "source_doi")
            source_urls = _joined_source_values(source_frame, "source_url")
            registry_rows.append(
                {
                    "record_id": record["record_id"],
                    "solvent_composition_key": key,
                    "property_name": property_name,
                    "unit": PROPERTY_UNITS[property_name],
                    "target_temperature_K": target_temperature_K,
                    **evaluated,
                    "source_composition_bases": _joined_source_values(
                        source_frame, "composition_basis"
                    ),
                    "source_experimental_methods": _joined_source_values(
                        source_frame, "experimental_method"
                    ),
                    "source_reported_uncertainties": _joined_source_values(
                        source_frame, "uncertainty"
                    ),
                    "source_pressure_kPa": _joined_source_values(
                        source_frame, "pressure_kPa"
                    ),
                    "source_dois": source_dois,
                    "source_urls": source_urls,
                    "source_access_dates": _joined_source_values(
                        source_frame, "access_date"
                    ),
                    "source_ids": _joined_source_values(source_frame, "source_id"),
                    "source_provenance": " | ".join(
                        value for value in (source_dois, source_urls) if value
                    ),
                    "proxy_column": output_column,
                    "ink_measurement_claim_allowed": False,
                }
            )
    registry = pd.DataFrame(registry_rows)
    for property_name, output_column in PROPERTY_OUTPUT_COLUMNS.items():
        values = registry.loc[registry["property_name"].eq(property_name), ["record_id", "proxy_value"]]
        result = result.merge(values.rename(columns={"proxy_value": output_column}), on="record_id", how="left", validate="one_to_one")
    ready = registry["proxy_status"].isin({"measured_exact", "measured_exact_replicate_mean", "interpolated_bracketed"})
    qa = {
        "status": "ready" if ready.any() else "unavailable_no_authoritative_values",
        "target_temperature_K": target_temperature_K,
        "authoritative_measurement_rows": len(validated),
        "registered_property_sources": len(sources),
        "measurement_source_foreign_keys_valid": bool(source_catalog is not None),
        "record_property_values_ready": int(ready.sum()),
        "records_with_all_four_proxies": int(
            result[list(PROPERTY_OUTPUT_COLUMNS.values())].notna().all(axis=1).sum()
        ),
        "temperature_extrapolation_used": False,
        "composition_extrapolation_used": False,
        "legacy_candidate_values_used": False,
        "claim_boundary": "solvent-mixture property proxy; not a measured catalyst/Nafion ink property",
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    registry.to_csv(destination / "authoritative_property_registry.csv", index=False)
    result[["record_id", "solvent_composition_key"]].to_csv(
        destination / "record_property_composition_keys.csv", index=False
    )
    (destination / "property_proxy_qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return result, registry, qa
