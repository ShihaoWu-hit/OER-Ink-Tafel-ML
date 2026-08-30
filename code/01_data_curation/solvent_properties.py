"""Source-backed solvent-mixture property reconstruction.

The canonical recipes store *dosed solvent volume fractions*.  Literature
series commonly use mole or mass fractions, so this module performs the basis
conversion explicitly from registered pure-component densities and molar
masses before any interpolation.  Only adjacent points from one source series
and one property definition may be interpolated.  Neither composition nor
temperature extrapolation is allowed.

The resulting quantities are solvent-mixture property proxies.  They are not
measurements of the catalyst/ionomer/particle ink.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .properties import PROPERTY_OUTPUT_COLUMNS, PROPERTY_UNITS, SOLVENT_FRACTIONS, record_composition_key


EVIDENCE_TIERS = {
    "A_direct",
    "B_same_source_interpolation",
    "C_secondary_reconstruction",
    "D_model_estimate",
    "X_held",
}
CORE_PROPERTIES = ("viscosity", "surface_tension", "density")
ALL_PROPERTIES = CORE_PROPERTIES + ("dielectric_constant",)
ESTIMATED_DIELECTRIC_COLUMN = (
    "solvent_mixture_property_proxy_dielectric_constant_estimated"
)
BEST_AVAILABLE_DIELECTRIC_COLUMN = (
    "solvent_mixture_property_proxy_dielectric_constant_best_available"
)
ESTIMATED_DIELECTRIC_STATUS_COLUMN = (
    "solvent_mixture_property_proxy_dielectric_constant_estimated_model_use_status"
)

MEASUREMENT_REQUIRED = {
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
    "value",
    "unit",
    "source_id",
    "source_file_id",
    "source_reference_id",
    "source_row_locator",
    "raw_file_sha256",
}
CONSTANT_REQUIRED = {
    "component",
    "molar_mass_g_mol",
    "molar_mass_source_id",
    "pure_density_kg_m3",
    "pure_density_temperature_K",
    "pure_density_source_id",
    "pure_density_source_file_id",
    "pure_density_source_file_sha256",
    "pure_density_evidence_tier",
    "static_dielectric_constant",
    "static_dielectric_temperature_K",
    "static_dielectric_definition",
    "static_dielectric_source_id",
    "static_dielectric_source_file_id",
    "static_dielectric_source_file_sha256",
    "static_dielectric_evidence_tier",
    "static_dielectric_source_locator",
}
SOURCE_REQUIRED = {
    "source_id",
    "title",
    "usage_status",
    "registered_reference_ids",
}
SOURCE_FILE_REQUIRED = {
    "source_file_id",
    "source_id",
    "sha256",
    "resource_url",
    "contained_reference_ids",
}


def _load_csv(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _normalise_reference_id(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return " ".join(text.split())


def _reference_id_set(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        normalised
        for item in str(value).split("|")
        if (normalised := _normalise_reference_id(item))
    }


def _validate_source_file_reference_crosswalk(
    *,
    source_id: Any,
    source_file_id: Any,
    sources: pd.DataFrame,
    source_files: pd.DataFrame,
    context: str,
) -> None:
    source_row = sources.set_index("source_id").loc[str(source_id)]
    file_row = source_files.set_index("source_file_id").loc[str(source_file_id)]
    source_references = _reference_id_set(source_row["registered_reference_ids"])
    file_references = _reference_id_set(file_row["contained_reference_ids"])
    if not source_references or not file_references or not (
        source_references & file_references
    ):
        raise ValueError(
            f"{context}: source/file original-reference crosswalk is missing or inconsistent"
        )


def _validate_sources(
    sources: pd.DataFrame, source_files: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(SOURCE_REQUIRED - set(sources))
    if missing:
        raise ValueError(f"Property source catalogue is missing columns: {missing}")
    missing_files = sorted(SOURCE_FILE_REQUIRED - set(source_files))
    if missing_files:
        raise ValueError(f"Property source-file registry is missing columns: {missing_files}")
    out = sources.copy()
    out["source_id"] = out["source_id"].astype("string").str.strip()
    if out["source_id"].isna().any() or out["source_id"].eq("").any():
        raise ValueError("Property source catalogue contains blank source_id values")
    if out["source_id"].duplicated().any():
        raise ValueError("Property source catalogue contains duplicate source_id values")
    files = source_files.copy()
    files["source_file_id"] = files["source_file_id"].astype("string").str.strip()
    if files["source_file_id"].duplicated().any():
        raise ValueError("Property source-file registry contains duplicate source_file_id values")
    unknown = sorted(set(files["source_id"].astype(str)) - set(out["source_id"].astype(str)))
    if unknown:
        raise ValueError(f"Property source-file registry references unknown sources: {unknown}")
    hashes = files["sha256"].fillna("").astype(str).str.strip().str.lower()
    explicitly_pending = files.get(
        "hash_status", pd.Series("", index=files.index)
    ).astype(str).str.contains("pending", case=False, regex=False)
    bad_hash = ~hashes.str.fullmatch(r"[0-9a-f]{64}") & ~(
        hashes.eq("") & explicitly_pending
    )
    if bad_hash.any():
        identifiers = files.loc[bad_hash, "source_file_id"].astype(str).tolist()
        raise ValueError(f"Property source files have invalid SHA256 values: {identifiers}")
    files["sha256"] = hashes
    return out, files


def _validate_constants(
    constants: pd.DataFrame,
    sources: pd.DataFrame,
    source_files: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(CONSTANT_REQUIRED - set(constants))
    if missing:
        raise ValueError(f"Property component registry is missing columns: {missing}")
    out = constants.copy()
    out["component"] = out["component"].astype("string").str.strip().str.lower()
    if out["component"].duplicated().any():
        raise ValueError("Property component registry contains duplicate components")
    for column in (
        "molar_mass_g_mol",
        "pure_density_kg_m3",
        "pure_density_temperature_K",
    ):
        out[column] = pd.to_numeric(out[column], errors="raise")
        if (~np.isfinite(out[column]) | (out[column] <= 0)).any():
            raise ValueError(f"Component registry column {column} must be positive and finite")
    known_sources = set(sources["source_id"].astype(str))
    for column in ("molar_mass_source_id", "pure_density_source_id"):
        unknown = sorted(set(out[column].dropna().astype(str)) - known_sources)
        if unknown:
            raise ValueError(f"Component registry {column} references unknown sources: {unknown}")
    if not out["pure_density_evidence_tier"].astype(str).isin(EVIDENCE_TIERS).all():
        raise ValueError("Component density registry contains an unsupported evidence tier")
    file_lookup = source_files.set_index("source_file_id")
    density_file_ids = out["pure_density_source_file_id"].fillna("").astype(str).str.strip()
    unknown_density_files = sorted(set(density_file_ids) - set(file_lookup.index.astype(str)))
    if unknown_density_files:
        raise ValueError(
            f"Component density registry references unknown source files: {unknown_density_files}"
        )
    for row in out.itertuples(index=False):
        expected_hash = str(file_lookup.loc[str(row.pure_density_source_file_id), "sha256"])
        registered_hash = str(row.pure_density_source_file_sha256).strip().lower()
        if registered_hash != expected_hash:
            raise ValueError(
                f"{row.component}: pure-density source-file SHA256 does not match the source-file registry"
            )
        _validate_source_file_reference_crosswalk(
            source_id=row.pure_density_source_id,
            source_file_id=row.pure_density_source_file_id,
            sources=sources,
            source_files=source_files,
            context=f"{row.component} pure density",
        )

    static_values = pd.to_numeric(out["static_dielectric_constant"], errors="coerce")
    static_temperatures = pd.to_numeric(
        out["static_dielectric_temperature_K"], errors="coerce"
    )
    registered_static = static_values.notna()
    registered_components = set(out.loc[registered_static, "component"].astype(str))
    if registered_components != {"water", "isopropanol"}:
        raise ValueError(
            "Static dielectric parameter registry must contain exactly water and isopropanol"
        )
    if (
        (~np.isfinite(static_values.loc[registered_static]))
        | (static_values.loc[registered_static] <= 0)
        | (~np.isfinite(static_temperatures.loc[registered_static]))
        | ((static_temperatures.loc[registered_static] - 298.15).abs() > 0.05)
    ).any():
        raise ValueError("Static dielectric endpoint parameters must be positive at 298.15 K")
    if not out.loc[registered_static, "static_dielectric_definition"].astype(str).eq(
        "static_low_frequency_limit"
    ).all():
        raise ValueError("Static dielectric endpoint parameters must use the static limit")
    parameter_source_ids = out.loc[registered_static, "static_dielectric_source_id"].astype(str)
    unknown_parameter_sources = sorted(set(parameter_source_ids) - known_sources)
    if unknown_parameter_sources:
        raise ValueError(
            f"Static dielectric parameters reference unknown sources: {unknown_parameter_sources}"
        )
    parameter_file_ids = out.loc[
        registered_static, "static_dielectric_source_file_id"
    ].astype(str)
    unknown_parameter_files = sorted(
        set(parameter_file_ids) - set(file_lookup.index.astype(str))
    )
    if unknown_parameter_files:
        raise ValueError(
            f"Static dielectric parameters reference unknown source files: {unknown_parameter_files}"
        )
    parameter_tiers = out.loc[
        registered_static, "static_dielectric_evidence_tier"
    ].astype(str)
    if not parameter_tiers.isin({"A_direct", "C_secondary_reconstruction"}).all():
        raise ValueError("Static dielectric endpoint parameters require A or C evidence")
    for row in out.loc[registered_static].itertuples(index=False):
        expected_hash = str(
            file_lookup.loc[str(row.static_dielectric_source_file_id), "sha256"]
        )
        registered_hash = str(row.static_dielectric_source_file_sha256).strip().lower()
        if registered_hash != expected_hash:
            raise ValueError(
                f"{row.component}: static-dielectric source-file SHA256 does not match the source-file registry"
            )
        _validate_source_file_reference_crosswalk(
            source_id=row.static_dielectric_source_id,
            source_file_id=row.static_dielectric_source_file_id,
            sources=sources,
            source_files=source_files,
            context=f"{row.component} static dielectric",
        )
    return out


def _validate_measurements(
    measurements: pd.DataFrame,
    sources: pd.DataFrame,
    source_files: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(MEASUREMENT_REQUIRED - set(measurements))
    if missing:
        raise ValueError(f"Property measurement registry is missing columns: {missing}")
    out = measurements.copy()
    out = out.loc[
        out.astype("string").apply(lambda column: column.str.strip()).ne("").any(axis=1)
    ].copy()
    if out.empty:
        raise ValueError("Property measurement registry is empty")
    out["measurement_id"] = out["measurement_id"].astype("string").str.strip()
    if out["measurement_id"].duplicated().any():
        raise ValueError("Property measurement registry contains duplicate measurement_id values")
    out["property_name"] = out["property_name"].astype("string").str.strip().str.lower()
    unknown_properties = sorted(set(out["property_name"].astype(str)) - set(ALL_PROPERTIES))
    if unknown_properties:
        raise ValueError(f"Unsupported solvent properties: {unknown_properties}")
    out["evidence_tier"] = out["evidence_tier"].astype("string").str.strip()
    unknown_tiers = sorted(set(out["evidence_tier"].astype(str)) - EVIDENCE_TIERS)
    if unknown_tiers:
        raise ValueError(f"Unsupported property evidence tiers: {unknown_tiers}")
    for column in ("value", "temperature_K", "composition_value", "series_priority"):
        out[column] = pd.to_numeric(out[column], errors="raise")
        if (~np.isfinite(out[column])).any():
            raise ValueError(f"Measurement column {column} must be finite")
    out["composition_basis"] = (
        out["composition_basis"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )
    allowed_bases = {"volume_fraction", "mole_fraction", "mass_fraction"}
    unsupported_bases = sorted(set(out["composition_basis"].astype(str)) - allowed_bases)
    if unsupported_bases:
        raise ValueError(f"Unsupported measurement composition bases: {unsupported_bases}")
    if ((out["composition_value"] < -1e-12) | (out["composition_value"] > 1 + 1e-12)).any():
        raise ValueError("Measurement composition fractions must be between zero and one")
    out["component_1"] = out["component_1"].astype("string").str.strip().str.lower()
    out["component_2"] = out["component_2"].fillna("").astype("string").str.strip().str.lower()
    out["composition_component"] = (
        out["composition_component"].astype("string").str.strip().str.lower()
    )
    for row in out.itertuples(index=False):
        if row.unit != PROPERTY_UNITS[str(row.property_name)]:
            raise ValueError(
                f"{row.measurement_id}: {row.property_name} must use "
                f"{PROPERTY_UNITS[str(row.property_name)]!r}, found {row.unit!r}"
            )
        if row.composition_component not in {row.component_1, row.component_2}:
            raise ValueError(
                f"{row.measurement_id}: composition_component is not part of the mixture"
            )
    dielectric = out["property_name"].eq("dielectric_constant")
    dielectric_static = (
        out.loc[dielectric, "static_low_frequency"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    dielectric_frequency = pd.to_numeric(
        out.loc[dielectric, "frequency_Hz"], errors="coerce"
    )
    if (not dielectric_static.all()) or (
        not np.isclose(dielectric_frequency.to_numpy(float), 0.0, atol=0.0).all()
    ):
        raise ValueError(
            "Dielectric measurements must be static/low-frequency-limit values with frequency_Hz=0"
        )
    invariant_columns = (
        "property_name",
        "unit",
        "component_1",
        "component_2",
        "composition_component",
        "composition_basis",
        "static_low_frequency",
        "frequency_Hz",
        "pressure_kPa",
        "phase",
        "source_id",
        "source_file_id",
        "experimental_method",
    )
    for series_id, series in out.groupby("series_id", sort=False):
        mixed = [
            column
            for column in invariant_columns
            if series[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .nunique(dropna=False)
            > 1
        ]
        if mixed:
            raise ValueError(
                f"Measurement series {series_id} mixes property definitions: {mixed}"
            )
        point_key = pd.DataFrame(
            {
                "composition_value": series["composition_value"],
                "temperature_K": series["temperature_K"],
                "pressure_kPa": series["pressure_kPa"].fillna("").astype(str).str.strip(),
                "frequency_Hz": series["frequency_Hz"].fillna("").astype(str).str.strip(),
            }
        )
        if point_key.duplicated().any():
            raise ValueError(
                f"Measurement series {series_id} contains duplicate state points"
            )
    known_sources = set(sources["source_id"].astype(str))
    unknown_sources = sorted(set(out["source_id"].astype(str)) - known_sources)
    if unknown_sources:
        raise ValueError(f"Measurements reference unknown source_id values: {unknown_sources}")
    known_files = set(source_files["source_file_id"].astype(str))
    unknown_files = sorted(set(out["source_file_id"].astype(str)) - known_files)
    if unknown_files:
        raise ValueError(f"Measurements reference unknown source_file_id values: {unknown_files}")
    source_reference_lookup = {
        str(row.source_id): _reference_id_set(row.registered_reference_ids)
        for row in sources.itertuples(index=False)
    }
    file_reference_lookup = {
        str(row.source_file_id): _reference_id_set(row.contained_reference_ids)
        for row in source_files.itertuples(index=False)
    }
    bad_source_reference: list[str] = []
    bad_file_reference: list[str] = []
    for row in out.itertuples(index=False):
        reference_id = _normalise_reference_id(row.source_reference_id)
        if (
            not reference_id
            or reference_id not in source_reference_lookup[str(row.source_id)]
        ):
            bad_source_reference.append(str(row.measurement_id))
        if (
            not reference_id
            or reference_id not in file_reference_lookup[str(row.source_file_id)]
        ):
            bad_file_reference.append(str(row.measurement_id))
    if bad_source_reference:
        raise ValueError(
            "Measurement source_reference_id is not registered for source_id: "
            f"{bad_source_reference}"
        )
    if bad_file_reference:
        raise ValueError(
            "Measurement source_reference_id is not present in the source-file "
            f"crosswalk: {bad_file_reference}"
        )
    file_lookup = source_files.set_index("source_file_id")
    expected_hash = file_lookup["sha256"].fillna("").astype(str).to_dict()
    pending_file = file_lookup.get(
        "hash_status", pd.Series("", index=file_lookup.index)
    ).astype(str).str.contains("pending", case=False, regex=False).to_dict()
    mismatch = out.apply(
        lambda row: (
            (
                ""
                if pd.isna(row["raw_file_sha256"])
                else str(row["raw_file_sha256"]).strip().lower()
            )
            != expected_hash[str(row["source_file_id"])]
        )
        or (
            expected_hash[str(row["source_file_id"])] == ""
            and not pending_file[str(row["source_file_id"])]
        ),
        axis=1,
    )
    if mismatch.any():
        ids = out.loc[mismatch, "measurement_id"].astype(str).tolist()
        raise ValueError(f"Measurement raw-file SHA256 does not match source registry: {ids}")
    eligible = out["model_use_status"].astype(str).str.startswith("eligible_")
    held = out["evidence_tier"].astype(str).eq("X_held")
    if (eligible & held).any():
        raise ValueError("X_held measurements cannot be eligible_source_backed")
    return out


def _constant_map(constants: pd.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in constants.itertuples(index=False):
        values = {
            "molar_mass_g_mol": float(row.molar_mass_g_mol),
            "pure_density_kg_m3": float(row.pure_density_kg_m3),
        }
        static_value = pd.to_numeric(
            pd.Series([getattr(row, "static_dielectric_constant", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if np.isfinite(static_value):
            values["static_dielectric_constant"] = float(static_value)
        output[str(row.component)] = values
    return output


def convert_dosed_volume_fractions(
    volume_fractions: dict[str, float],
    component_constants: dict[str, dict[str, float]],
    *,
    basis: str,
) -> dict[str, float]:
    """Convert pre-mix dosing volume fractions to mass or mole fractions.

    The calculation uses ``m_i = phi_i rho_i`` and
    ``n_i = phi_i rho_i / M_i``.  Mixture volume contraction is deliberately
    not introduced into the pre-mix mole count.
    """

    clean = {
        str(component): float(value)
        for component, value in volume_fractions.items()
        if np.isfinite(value) and value > 1e-12
    }
    total_volume = sum(clean.values())
    if total_volume <= 0:
        raise ValueError("At least one positive solvent volume fraction is required")
    clean = {component: value / total_volume for component, value in clean.items()}
    missing = sorted(set(clean) - set(component_constants))
    if missing:
        raise ValueError(f"No conversion constants registered for components: {missing}")
    if basis == "volume_fraction":
        weights = clean
    elif basis == "mass_fraction":
        weights = {
            component: phi * component_constants[component]["pure_density_kg_m3"]
            for component, phi in clean.items()
        }
    elif basis == "mole_fraction":
        weights = {
            component: phi
            * component_constants[component]["pure_density_kg_m3"]
            / component_constants[component]["molar_mass_g_mol"]
            for component, phi in clean.items()
        }
    else:
        raise ValueError(f"Unsupported composition basis: {basis}")
    total = sum(weights.values())
    return {component: value / total for component, value in weights.items()}


def _record_volume_fractions(row: pd.Series) -> dict[str, float]:
    fractions: dict[str, float] = {}
    for component, column in SOLVENT_FRACTIONS.items():
        if component == "other":
            continue
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if np.isfinite(value) and float(value) > 1e-12:
            fractions[component] = float(value)
    return fractions


def _series_pair(frame: pd.DataFrame) -> frozenset[str]:
    components: set[str] = set(frame["component_1"].dropna().astype(str))
    components.update(value for value in frame["component_2"].dropna().astype(str) if value)
    return frozenset(components)


def _standard_uncertainty(row: pd.Series) -> float:
    reported = pd.to_numeric(pd.Series([row.get("reported_uncertainty")]), errors="coerce").iloc[0]
    if not np.isfinite(reported):
        return np.nan
    coverage_factor = pd.to_numeric(
        pd.Series([row.get("uncertainty_coverage_factor")]), errors="coerce"
    ).iloc[0]
    uncertainty_type = str(row.get("uncertainty_type", "")).lower()
    if np.isfinite(coverage_factor) and float(coverage_factor) > 0:
        return float(reported) / float(coverage_factor)
    if uncertainty_type in {"standard_uncertainty", "standard_deviation", "one_sigma"}:
        return float(reported)
    return np.nan


def _temperature_collapsed_series(
    series: pd.DataFrame,
    *,
    target_temperature_K: float,
    exact_temperature_tolerance_K: float,
) -> pd.DataFrame:
    """Return one row per source composition at the target temperature."""

    rows: list[dict[str, Any]] = []
    for composition, group in series.groupby("composition_value", sort=True):
        group = group.sort_values("temperature_K")
        exact = group.loc[
            (group["temperature_K"] - target_temperature_K).abs()
            <= exact_temperature_tolerance_K
        ]
        if not exact.empty:
            chosen = exact.iloc[(exact["temperature_K"] - target_temperature_K).abs().argmin()]
            rows.append(
                {
                    **chosen.to_dict(),
                    "composition_value": float(composition),
                    "target_temperature_value": float(chosen["value"]),
                    "temperature_interpolation_used": False,
                    "temperature_low_K": float(chosen["temperature_K"]),
                    "temperature_high_K": float(chosen["temperature_K"]),
                    "temperature_weight_high": 0.0,
                    "temperature_endpoint_ids": str(chosen["measurement_id"]),
                    "endpoint_standard_uncertainty": _standard_uncertainty(chosen),
                }
            )
            continue
        lower = group.loc[group["temperature_K"] < target_temperature_K].tail(1)
        upper = group.loc[group["temperature_K"] > target_temperature_K].head(1)
        if lower.empty or upper.empty:
            continue
        low = lower.iloc[0]
        high = upper.iloc[0]
        weight = (target_temperature_K - float(low["temperature_K"])) / (
            float(high["temperature_K"]) - float(low["temperature_K"])
        )
        value = float(low["value"]) + weight * (float(high["value"]) - float(low["value"]))
        u_low = _standard_uncertainty(low)
        u_high = _standard_uncertainty(high)
        endpoint_u = (
            math.sqrt(((1 - weight) * u_low) ** 2 + (weight * u_high) ** 2)
            if np.isfinite(u_low) and np.isfinite(u_high)
            else np.nan
        )
        rows.append(
            {
                **low.to_dict(),
                "composition_value": float(composition),
                "target_temperature_value": value,
                "temperature_interpolation_used": True,
                "temperature_low_K": float(low["temperature_K"]),
                "temperature_high_K": float(high["temperature_K"]),
                "temperature_weight_high": float(weight),
                "temperature_endpoint_ids": f"{low['measurement_id']}|{high['measurement_id']}",
                "endpoint_standard_uncertainty": endpoint_u,
            }
        )
    return pd.DataFrame(rows).sort_values("composition_value").reset_index(drop=True)


def _loo_linear_rmse(curve: pd.DataFrame) -> float:
    if len(curve) < 3:
        return np.nan
    ordered = curve.sort_values("composition_value").reset_index(drop=True)
    errors: list[float] = []
    for index in range(1, len(ordered) - 1):
        low = ordered.iloc[index - 1]
        point = ordered.iloc[index]
        high = ordered.iloc[index + 1]
        width = float(high["composition_value"]) - float(low["composition_value"])
        if width <= 0:
            continue
        weight = (float(point["composition_value"]) - float(low["composition_value"])) / width
        prediction = float(low["target_temperature_value"]) + weight * (
            float(high["target_temperature_value"]) - float(low["target_temperature_value"])
        )
        errors.append(prediction - float(point["target_temperature_value"]))
    return float(math.sqrt(np.mean(np.square(errors)))) if errors else np.nan


def _candidate_from_series(
    series: pd.DataFrame,
    *,
    target_coordinate: float,
    target_temperature_K: float,
    exact_temperature_tolerance_K: float,
) -> dict[str, Any] | None:
    curve = _temperature_collapsed_series(
        series,
        target_temperature_K=target_temperature_K,
        exact_temperature_tolerance_K=exact_temperature_tolerance_K,
    )
    if curve.empty:
        return None
    exact = curve.loc[np.isclose(curve["composition_value"], target_coordinate, atol=1e-10)]
    loo_rmse = _loo_linear_rmse(curve)
    if not exact.empty:
        point = exact.iloc[0]
        temperature_interpolated = bool(point["temperature_interpolation_used"])
        source_tier = str(point["evidence_tier"])
        if source_tier == "A_direct" and not temperature_interpolated:
            tier = "A_direct"
        elif source_tier in {"A_direct", "B_same_source_interpolation"}:
            tier = "B_same_source_interpolation"
        else:
            tier = source_tier
        endpoint_u = float(point["endpoint_standard_uncertainty"])
        # A direct-composition point is not a composition interpolation.  Its
        # standard uncertainty therefore comes only from standardized source
        # uncertainty (including propagated temperature endpoints when
        # applicable), never from leave-one-out composition interpolation
        # error.  LOO remains a separate series diagnostic below.
        uncertainty = endpoint_u if np.isfinite(endpoint_u) else np.nan
        return {
            "proxy_value": float(point["target_temperature_value"]),
            "evidence_tier": tier,
            "proxy_status": (
                "direct_target_composition_temperature_interpolated"
                if temperature_interpolated
                else "direct_target_composition"
            ),
            "composition_low": float(point["composition_value"]),
            "composition_high": float(point["composition_value"]),
            "composition_weight_high": 0.0,
            "composition_interval_width": 0.0,
            "composition_endpoint_ids": str(point["temperature_endpoint_ids"]),
            "temperature_interpolation_used": temperature_interpolated,
            "temperature_low_K": float(point["temperature_low_K"]),
            "temperature_high_K": float(point["temperature_high_K"]),
            "loo_interpolation_rmse": loo_rmse,
            "proxy_standard_uncertainty": uncertainty,
            "uncertainty_status": (
                "temperature_interpolation_endpoint_standard_uncertainty_only"
                if temperature_interpolated and np.isfinite(endpoint_u)
                else "source_standard_uncertainty_only"
                if np.isfinite(endpoint_u)
                else "unavailable_temperature_interpolation_model_uncertainty_not_estimated"
                if temperature_interpolated
                else "unavailable_source_uncertainty_not_standardized"
            ),
            "source_ids": str(point["source_id"]),
            "source_file_ids": str(point["source_file_id"]),
            "source_row_locators": str(point["source_row_locator"]),
            "raw_file_sha256": str(point["raw_file_sha256"]),
            "series_id": str(point["series_id"]),
            "series_priority": int(point["series_priority"]),
            "source_composition_basis": str(point["composition_basis"]),
            "composition_component": str(point["composition_component"]),
        }
    lower = curve.loc[curve["composition_value"] < target_coordinate].tail(1)
    upper = curve.loc[curve["composition_value"] > target_coordinate].head(1)
    if lower.empty or upper.empty:
        return None
    low = lower.iloc[0]
    high = upper.iloc[0]
    width = float(high["composition_value"]) - float(low["composition_value"])
    if width <= 0:
        return None
    weight = (target_coordinate - float(low["composition_value"])) / width
    value = float(low["target_temperature_value"]) + weight * (
        float(high["target_temperature_value"]) - float(low["target_temperature_value"])
    )
    u_low = float(low["endpoint_standard_uncertainty"])
    u_high = float(high["endpoint_standard_uncertainty"])
    endpoint_u = (
        math.sqrt(((1 - weight) * u_low) ** 2 + (weight * u_high) ** 2)
        if np.isfinite(u_low) and np.isfinite(u_high)
        else np.nan
    )
    uncertainty = (
        math.sqrt(endpoint_u**2 + loo_rmse**2)
        if np.isfinite(endpoint_u) and np.isfinite(loo_rmse)
        else endpoint_u
        if np.isfinite(endpoint_u)
        else loo_rmse
    )
    endpoint_tiers = {str(low["evidence_tier"]), str(high["evidence_tier"])}
    tier = (
        "B_same_source_interpolation"
        if endpoint_tiers.issubset({"A_direct", "B_same_source_interpolation"})
        else "C_secondary_reconstruction"
        if "C_secondary_reconstruction" in endpoint_tiers
        else "X_held"
    )
    return {
        "proxy_value": value,
        "evidence_tier": tier,
        "proxy_status": "same_source_bracket_interpolation",
        "composition_low": float(low["composition_value"]),
        "composition_high": float(high["composition_value"]),
        "composition_weight_high": float(weight),
        "composition_interval_width": width,
        "composition_endpoint_ids": (
            f"{low['temperature_endpoint_ids']}|{high['temperature_endpoint_ids']}"
        ),
        "temperature_interpolation_used": bool(low["temperature_interpolation_used"])
        or bool(high["temperature_interpolation_used"]),
        "temperature_low_K": min(float(low["temperature_low_K"]), float(high["temperature_low_K"])),
        "temperature_high_K": max(float(low["temperature_high_K"]), float(high["temperature_high_K"])),
        "loo_interpolation_rmse": loo_rmse,
        "proxy_standard_uncertainty": uncertainty,
        "uncertainty_status": (
            "source_standard_uncertainty_plus_loo_rmse"
            if np.isfinite(endpoint_u) and np.isfinite(loo_rmse)
            else "source_standard_uncertainty_only"
            if np.isfinite(endpoint_u)
            else "loo_rmse_only_source_uncertainty_not_standardized"
            if np.isfinite(loo_rmse)
            else "unavailable_source_uncertainty_not_standardized"
        ),
        "source_ids": str(low["source_id"]),
        "source_file_ids": str(low["source_file_id"]),
        "source_row_locators": f"{low['source_row_locator']}|{high['source_row_locator']}",
        "raw_file_sha256": str(low["raw_file_sha256"]),
        "series_id": str(low["series_id"]),
        "series_priority": int(low["series_priority"]),
        "source_composition_basis": str(low["composition_basis"]),
        "composition_component": str(low["composition_component"]),
    }


def _select_candidate(
    candidates: list[dict[str, Any]],
    *,
    property_name: str,
    components: frozenset[str],
    target_coordinate: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    # Rafati is registered only as a high-IPA supplement.  At high IPA mole
    # fraction the materially narrower bracket takes precedence; elsewhere the
    # pre-registered source priority controls selection.
    high_ipa = (
        property_name == "surface_tension"
        and components == frozenset({"isopropanol", "water"})
        and target_coordinate >= 0.70
    )
    if high_ipa:
        ordered = sorted(
            candidates,
            key=lambda row: (
                float(row["composition_interval_width"]),
                int(row["series_priority"]),
                str(row["series_id"]),
            ),
        )
    else:
        ordered = sorted(
            candidates,
            key=lambda row: (
                int(row["series_priority"]),
                float(row["composition_interval_width"]),
                str(row["series_id"]),
            ),
        )
    return ordered[0]


def _cross_source_water_anchor_candidates(
    property_measurements: pd.DataFrame,
    volume_fractions: dict[str, float],
    components: frozenset[str],
    constants: dict[str, dict[str, float]],
    *,
    target_temperature_K: float,
    exact_temperature_tolerance_K: float,
) -> list[dict[str, Any]]:
    """Construct explicitly C-tier low-concentration dielectric brackets.

    Petong and Uosaki do not report the pure-water endpoint in their selected
    tables.  The registered IAPWS evaluated water value is therefore allowed
    as a *cross-source* anchor, but the resulting value is never relabelled as
    same-source B evidence.
    """

    if property_measurements.empty or "water" not in components:
        return []
    anchors = property_measurements.loc[
        property_measurements["measurement_status"].astype(str).eq(
            "official_evaluated_reference_value"
        )
        & np.isclose(property_measurements["composition_value"], 0.0, atol=1e-12)
    ]
    if anchors.empty:
        return []
    output: list[dict[str, Any]] = []
    for series_id, primary in property_measurements.groupby("series_id", sort=False):
        if primary["measurement_status"].astype(str).eq(
            "official_evaluated_reference_value"
        ).all():
            continue
        pair = _series_pair(primary)
        if pair != components:
            continue
        basis_values = set(primary["composition_basis"].astype(str))
        coordinate_components = set(primary["composition_component"].astype(str))
        if len(basis_values) != 1 or len(coordinate_components) != 1:
            continue
        basis = next(iter(basis_values))
        coordinate_component = next(iter(coordinate_components))
        target_coordinate = convert_dosed_volume_fractions(
            volume_fractions, constants, basis=basis
        ).get(coordinate_component, 0.0)
        minimum_primary = float(primary["composition_value"].min())
        if not (0 < target_coordinate < minimum_primary):
            continue
        compatible_anchors = anchors.loc[
            anchors["composition_basis"].astype(str).eq(basis)
            & anchors["composition_component"].astype(str).eq(coordinate_component)
        ]
        if compatible_anchors.empty:
            continue
        anchor = compatible_anchors.iloc[[0]].copy()
        combined = pd.concat([anchor, primary], ignore_index=True, sort=False)
        candidate = _candidate_from_series(
            combined,
            target_coordinate=float(target_coordinate),
            target_temperature_K=target_temperature_K,
            exact_temperature_tolerance_K=exact_temperature_tolerance_K,
        )
        if candidate is None:
            continue
        first_primary = primary.sort_values("composition_value").iloc[0]
        anchor_row = anchor.iloc[0]
        candidate.update(
            {
                "evidence_tier": "C_secondary_reconstruction",
                "proxy_status": "cross_source_iapws_water_anchor_interpolation",
                "source_ids": f"{anchor_row['source_id']}|{first_primary['source_id']}",
                "source_file_ids": (
                    f"{anchor_row['source_file_id']}|{first_primary['source_file_id']}"
                ),
                "source_row_locators": (
                    f"{anchor_row['source_row_locator']}|{first_primary['source_row_locator']}"
                ),
                "raw_file_sha256": (
                    f"{anchor_row['raw_file_sha256']}|{first_primary['raw_file_sha256']}"
                ),
                "series_id": f"cross_source_iapws_anchor__{series_id}",
                "series_priority": int(primary["series_priority"].min()),
                "source_composition_basis": basis,
                "composition_component": coordinate_component,
                "target_composition_coordinate": float(target_coordinate),
                "target_composition_basis": basis,
                "cross_source_interpolation_used": True,
            }
        )
        output.append(candidate)
    return output


def bruggeman_permittivity(
    organic_volume_fraction: float, organic_epsilon: float, water_epsilon: float
) -> float:
    fraction = float(organic_volume_fraction)
    if not 0 <= fraction <= 1:
        raise ValueError("Bruggeman volume fraction must be between zero and one")
    coefficient = fraction * (2 * organic_epsilon - water_epsilon) + (1 - fraction) * (
        2 * water_epsilon - organic_epsilon
    )
    discriminant = coefficient**2 + 8 * organic_epsilon * water_epsilon
    return float((coefficient + math.sqrt(discriminant)) / 4)


def looyenga_permittivity(
    organic_volume_fraction: float, organic_epsilon: float, water_epsilon: float
) -> float:
    fraction = float(organic_volume_fraction)
    return float(
        (
            fraction * organic_epsilon ** (1 / 3)
            + (1 - fraction) * water_epsilon ** (1 / 3)
        )
        ** 3
    )


def linear_permittivity(
    organic_volume_fraction: float, organic_epsilon: float, water_epsilon: float
) -> float:
    fraction = float(organic_volume_fraction)
    return float(fraction * organic_epsilon + (1 - fraction) * water_epsilon)


def _validate_ipa_estimate(
    validation: pd.DataFrame,
    constants: dict[str, dict[str, float]],
    known_source_ids: set[str],
    *,
    water_epsilon: float,
    ipa_epsilon: float,
    target_temperature_K: float,
    mape_threshold_percent: float,
    max_error_threshold_percent: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "validation_id",
        "isopropanol_mass_fraction",
        "temperature_K",
        "dielectric_constant",
        "evidence_tier",
        "source_id",
    }
    missing = sorted(required - set(validation))
    if missing:
        raise ValueError(f"IPA dielectric validation table is missing columns: {missing}")
    frame = validation.copy()
    frame["isopropanol_mass_fraction"] = pd.to_numeric(
        frame["isopropanol_mass_fraction"], errors="raise"
    )
    frame["dielectric_constant"] = pd.to_numeric(frame["dielectric_constant"], errors="raise")
    frame["temperature_K"] = pd.to_numeric(frame["temperature_K"], errors="raise")
    if frame["validation_id"].astype(str).duplicated().any():
        raise ValueError("IPA dielectric validation table has duplicate validation_id values")
    if frame["isopropanol_mass_fraction"].duplicated().any():
        raise ValueError("IPA dielectric validation compositions must be unique")
    expected_mass_fractions = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4])
    observed_mass_fractions = np.sort(
        frame["isopropanol_mass_fraction"].to_numpy(float)
    )
    if len(observed_mass_fractions) != len(expected_mass_fractions) or not np.allclose(
        observed_mass_fractions, expected_mass_fractions, atol=1e-12, rtol=0
    ):
        raise ValueError(
            "IPA dielectric validation must contain exactly the registered 0-40 mass% grid"
        )
    if (
        (~np.isfinite(frame["dielectric_constant"]))
        | (frame["dielectric_constant"] <= 0)
    ).any():
        raise ValueError("IPA dielectric validation values must be positive and finite")
    if (
        (~np.isfinite(frame["temperature_K"]))
        | ((frame["temperature_K"] - target_temperature_K).abs() > 0.05)
    ).any():
        raise ValueError("IPA dielectric validation rows must be at 298.15 K")
    unknown_sources = sorted(set(frame["source_id"].astype(str)) - known_source_ids)
    if unknown_sources:
        raise ValueError(
            f"IPA dielectric validation references unknown sources: {unknown_sources}"
        )
    if not frame["evidence_tier"].astype(str).eq(
        "C_secondary_reconstruction"
    ).all():
        raise ValueError("IPA dielectric validation rows must remain C-tier cross-checks")
    rho_ipa = constants["isopropanol"]["pure_density_kg_m3"]
    rho_water = constants["water"]["pure_density_kg_m3"]
    volume_fractions: list[float] = []
    predictions: list[float] = []
    for row in frame.itertuples(index=False):
        weight = float(row.isopropanol_mass_fraction)
        volume = (weight / rho_ipa) / ((weight / rho_ipa) + ((1 - weight) / rho_water)) if weight else 0.0
        volume_fractions.append(volume)
        predictions.append(bruggeman_permittivity(volume, ipa_epsilon, water_epsilon))
    frame["isopropanol_volume_fraction_converted"] = volume_fractions
    frame["bruggeman_prediction"] = predictions
    frame["absolute_percent_error"] = (
        (frame["bruggeman_prediction"] - frame["dielectric_constant"]).abs()
        / frame["dielectric_constant"].abs()
        * 100
    )
    mape = float(frame["absolute_percent_error"].mean())
    maximum = float(frame["absolute_percent_error"].max())
    passed = mape <= mape_threshold_percent and maximum <= max_error_threshold_percent
    qa = {
        "validation_rows": len(frame),
        "mape_percent": mape,
        "maximum_single_point_error_percent": maximum,
        "mape_threshold_percent": float(mape_threshold_percent),
        "maximum_single_point_error_threshold_percent": float(max_error_threshold_percent),
        "gate_passed": bool(passed),
        "estimated_panel_model_use_status": "eligible_sensitivity_only" if passed else "excluded",
    }
    return frame, qa


def _canonical_compositions(records: pd.DataFrame) -> pd.DataFrame:
    if "record_id" not in records:
        raise ValueError("Canonical records require record_id")
    if records["record_id"].duplicated().any():
        raise ValueError("Canonical records require unique record_id values")
    frame = records.copy()
    calculated = frame.apply(record_composition_key, axis=1)
    if "solvent_composition_key" in frame:
        mismatch = frame["solvent_composition_key"].astype(str).ne(calculated.astype(str))
        if mismatch.any():
            raise ValueError("Stored solvent_composition_key does not match canonical fractions")
    frame["solvent_composition_key"] = calculated
    fraction_columns = [column for component, column in SOLVENT_FRACTIONS.items() if component != "other"]
    other = pd.to_numeric(frame.get("solvent_fraction_other", 0), errors="coerce").fillna(0)
    if (other.abs() > 1e-12).any():
        raise ValueError("Unregistered 'other' solvent cannot enter property reconstruction")
    return (
        frame[["solvent_composition_key", *fraction_columns]]
        .drop_duplicates()
        .sort_values("solvent_composition_key")
        .reset_index(drop=True)
    )


def _property_rows_for_composition(
    composition: pd.Series,
    measurements: pd.DataFrame,
    constants: dict[str, dict[str, float]],
    *,
    target_temperature_K: float,
    exact_temperature_tolerance_K: float,
) -> list[dict[str, Any]]:
    volume_fractions = _record_volume_fractions(composition)
    components = frozenset(volume_fractions)
    rows: list[dict[str, Any]] = []
    eligible_measurements = measurements.loc[
        measurements["model_use_status"].astype(str).str.startswith("eligible_")
    ]
    for property_name in ALL_PROPERTIES:
        candidates: list[dict[str, Any]] = []
        property_measurements = eligible_measurements.loc[
            eligible_measurements["property_name"].eq(property_name)
        ]
        for _, series in property_measurements.groupby("series_id", sort=False):
            pair = _series_pair(series)
            # A pure component may use the endpoint of a binary series.  A
            # binary recipe must match the complete binary component set.
            if len(components) == 1:
                if not components.issubset(pair):
                    continue
            elif pair != components:
                continue
            basis_values = set(series["composition_basis"].astype(str))
            composition_components = set(series["composition_component"].astype(str))
            if len(basis_values) != 1 or len(composition_components) != 1:
                raise ValueError(
                    f"Series {series['series_id'].iloc[0]} mixes composition definitions"
                )
            basis = next(iter(basis_values))
            coordinate_component = next(iter(composition_components))
            converted = convert_dosed_volume_fractions(
                volume_fractions, constants, basis=basis
            )
            target_coordinate = float(converted.get(coordinate_component, 0.0))
            candidate = _candidate_from_series(
                series,
                target_coordinate=target_coordinate,
                target_temperature_K=target_temperature_K,
                exact_temperature_tolerance_K=exact_temperature_tolerance_K,
            )
            if candidate is not None:
                candidate["target_composition_coordinate"] = target_coordinate
                candidate["target_composition_basis"] = basis
                candidates.append(candidate)
        if property_name == "dielectric_constant":
            candidates.extend(
                _cross_source_water_anchor_candidates(
                    property_measurements,
                    volume_fractions,
                    components,
                    constants,
                    target_temperature_K=target_temperature_K,
                    exact_temperature_tolerance_K=exact_temperature_tolerance_K,
                )
            )
        # The target coordinate differs across bases, but registered competing
        # series for a property/system currently share one basis.  Selection
        # records the chosen series coordinate explicitly.
        selected = _select_candidate(
            candidates,
            property_name=property_name,
            components=components,
            target_coordinate=(
                float(candidates[0]["target_composition_coordinate"])
                if candidates
                else np.nan
            ),
        )
        base: dict[str, Any] = {
            "solvent_composition_key": composition["solvent_composition_key"],
            "property_name": property_name,
            "unit": PROPERTY_UNITS[property_name],
            "target_temperature_K": target_temperature_K,
            "recipe_volume_fractions": json.dumps(volume_fractions, sort_keys=True),
            "mixture_components": "|".join(sorted(components)),
            "composition_extrapolation_used": False,
            "temperature_extrapolation_used": False,
            "ink_measurement_claim_allowed": False,
        }
        if selected is None:
            rows.append(
                {
                    **base,
                    "strict_source_backed_value": np.nan,
                    "evidence_tier": "X_held",
                    "proxy_status": "missing_no_same_source_bracket",
                    "model_use_status": "held",
                }
            )
            continue
        medium_confidence = (
            property_name == "surface_tension"
            and components == frozenset({"isopropanol", "water"})
            and (
                abs(float(selected["target_composition_coordinate"]) - 0.5561) <= 0.003
                or abs(float(selected["target_composition_coordinate"]) - 0.8087) <= 0.003
            )
        )
        rows.append(
            {
                **base,
                **selected,
                "strict_source_backed_value": selected["proxy_value"],
                "model_use_status": (
                    "eligible_source_backed_medium_confidence"
                    if medium_confidence
                    else "eligible_source_backed"
                ),
                "confidence_flag": "medium_wide_composition_bracket" if medium_confidence else "",
            }
        )
    return rows


def _attach_estimated_dielectric(
    canonical: pd.DataFrame,
    *,
    validation_qa: dict[str, Any],
    water_epsilon: float,
    ipa_epsilon: float,
) -> pd.DataFrame:
    frame = canonical.copy()
    for column in (
        "estimated_value",
        "estimated_linear_value",
        "estimated_looyenga_value",
        "estimated_model_interval_low",
        "estimated_model_interval_high",
    ):
        frame[column] = np.nan
    frame["estimated_evidence_tier"] = ""
    frame["estimated_model_use_status"] = "not_applicable"
    dielectric = frame["property_name"].eq("dielectric_constant")
    ipa_system = frame["mixture_components"].astype(str).apply(
        lambda value: set(value.split("|")) in (
            {"isopropanol", "water"},
            {"isopropanol"},
        )
    )
    mask = dielectric & ipa_system & frame["strict_source_backed_value"].isna()
    for index in frame.index[mask]:
        fractions = json.loads(str(frame.at[index, "recipe_volume_fractions"]))
        ipa_fraction = float(fractions.get("isopropanol", 0.0))
        bruggeman = bruggeman_permittivity(ipa_fraction, ipa_epsilon, water_epsilon)
        linear = linear_permittivity(ipa_fraction, ipa_epsilon, water_epsilon)
        looyenga = looyenga_permittivity(ipa_fraction, ipa_epsilon, water_epsilon)
        frame.at[index, "estimated_value"] = bruggeman
        frame.at[index, "estimated_linear_value"] = linear
        frame.at[index, "estimated_looyenga_value"] = looyenga
        frame.at[index, "estimated_model_interval_low"] = min(bruggeman, linear, looyenga)
        frame.at[index, "estimated_model_interval_high"] = max(bruggeman, linear, looyenga)
        frame.at[index, "estimated_evidence_tier"] = "D_model_estimate"
        frame.at[index, "estimated_model_use_status"] = (
            "eligible_sensitivity_only"
            if validation_qa["gate_passed"]
            else "excluded_validation_gate_failed"
        )
    frame["best_available_value"] = frame["strict_source_backed_value"].combine_first(
        frame["estimated_value"]
    )
    frame["best_available_evidence_tier"] = np.where(
        frame["strict_source_backed_value"].notna(),
        frame["evidence_tier"],
        frame["estimated_evidence_tier"],
    )
    frame["best_available_model_use_status"] = np.where(
        frame["strict_source_backed_value"].notna(),
        frame["model_use_status"],
        frame["estimated_model_use_status"],
    )
    return frame


def _legacy_comparison(
    record_registry: pd.DataFrame,
    legacy_audit: pd.DataFrame | None,
    *,
    relative_threshold: float,
) -> pd.DataFrame:
    columns = [
        "record_id",
        "property_name",
        "legacy_value",
        "new_best_available_value",
        "absolute_difference",
        "relative_difference",
        "new_standard_uncertainty",
        "legacy_standard_uncertainty",
        "combined_standard_uncertainty",
        "discrepancy_over_5_percent",
        "discrepancy_over_two_combined_uncertainties",
        "uncertainty_comparison_status",
        "discrepancy_flag",
        "comparison_status",
    ]
    if legacy_audit is None or legacy_audit.empty:
        return pd.DataFrame(columns=columns)
    audit = legacy_audit.copy()
    legacy_column = next(
        (
            column
            for column in ("converted_candidate_value", "legacy_candidate_value", "legacy_value")
            if column in audit
        ),
        None,
    )
    if legacy_column is None:
        raise ValueError("Legacy audit has no comparable value column")
    legacy_uncertainty_column = next(
        (
            column
            for column in (
                "legacy_standard_uncertainty",
                "converted_candidate_standard_uncertainty",
                "legacy_candidate_standard_uncertainty",
            )
            if column in audit
        ),
        None,
    )
    legacy_columns = ["record_id", "property_name", legacy_column]
    if legacy_uncertainty_column is not None:
        legacy_columns.append(legacy_uncertainty_column)
    legacy = audit[legacy_columns].rename(columns={legacy_column: "legacy_value"})
    if legacy_uncertainty_column is None:
        legacy["legacy_standard_uncertainty"] = np.nan
    elif legacy_uncertainty_column != "legacy_standard_uncertainty":
        legacy = legacy.rename(
            columns={legacy_uncertainty_column: "legacy_standard_uncertainty"}
        )
    merged = record_registry.merge(
        legacy, on=["record_id", "property_name"], how="left", validate="one_to_one"
    )
    merged["new_best_available_value"] = merged["best_available_value"]
    merged["legacy_value"] = pd.to_numeric(merged["legacy_value"], errors="coerce")
    merged["absolute_difference"] = (
        merged["new_best_available_value"] - merged["legacy_value"]
    ).abs()
    denominator = merged["new_best_available_value"].abs().replace(0, np.nan)
    merged["relative_difference"] = merged["absolute_difference"] / denominator
    merged["discrepancy_over_5_percent"] = merged["relative_difference"] > relative_threshold
    proxy_u = pd.to_numeric(merged.get("proxy_standard_uncertainty"), errors="coerce")
    legacy_u = pd.to_numeric(merged["legacy_standard_uncertainty"], errors="coerce")
    merged["new_standard_uncertainty"] = proxy_u
    merged["legacy_standard_uncertainty"] = legacy_u
    merged["combined_standard_uncertainty"] = np.sqrt(proxy_u**2 + legacy_u**2)
    uncertainty_comparable = merged["combined_standard_uncertainty"].notna()
    merged["discrepancy_over_two_combined_uncertainties"] = np.where(
        uncertainty_comparable,
        merged["absolute_difference"] > 2 * merged["combined_standard_uncertainty"],
        False,
    )
    merged["uncertainty_comparison_status"] = np.where(
        uncertainty_comparable,
        "both_standard_uncertainties_available",
        "not_evaluated_missing_legacy_or_new_uncertainty",
    )
    merged["discrepancy_flag"] = (
        merged["discrepancy_over_5_percent"]
        | merged["discrepancy_over_two_combined_uncertainties"]
    )
    merged["comparison_status"] = np.select(
        [
            merged["legacy_value"].isna(),
            merged["new_best_available_value"].isna(),
            merged["discrepancy_flag"],
        ],
        ["legacy_missing", "new_value_missing", "discrepancy_review_required"],
        default="within_registered_tolerance",
    )
    return merged[columns]


def reconstruct_solvent_property_registries(
    records: pd.DataFrame,
    measurements: pd.DataFrame | str | Path,
    component_constants: pd.DataFrame | str | Path,
    source_catalog: pd.DataFrame | str | Path,
    source_files: pd.DataFrame | str | Path,
    ipa_dielectric_validation: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    legacy_audit: pd.DataFrame | str | Path | None = None,
    target_temperature_K: float = 298.15,
    exact_temperature_tolerance_K: float = 0.05,
    expected_records: int = 97,
    expected_compositions: int = 33,
    expected_dielectric_strict_source_backed_values: int = 66,
    expected_dielectric_A_or_B_values: int = 40,
    expected_dielectric_C_values: int = 26,
    expected_dielectric_D_estimates: int = 31,
    water_dielectric_constant: float | None = None,
    ipa_dielectric_constant: float | None = None,
    ipa_validation_mape_threshold_percent: float = 5.0,
    ipa_validation_max_error_threshold_percent: float = 10.0,
    legacy_relative_discrepancy_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build canonical and record-level solvent-property registries."""

    sources, files = _validate_sources(_load_csv(source_catalog), _load_csv(source_files))
    constants_frame = _validate_constants(
        _load_csv(component_constants), sources, files
    )
    measurement_frame = _validate_measurements(
        _load_csv(measurements), sources, files
    )
    constants = _constant_map(constants_frame)
    registered_water_epsilon = constants["water"].get("static_dielectric_constant")
    registered_ipa_epsilon = constants["isopropanol"].get(
        "static_dielectric_constant"
    )
    if registered_water_epsilon is None or registered_ipa_epsilon is None:
        raise ValueError("Registered water/isopropanol dielectric endpoint parameters are missing")
    if water_dielectric_constant is not None and not np.isclose(
        water_dielectric_constant, registered_water_epsilon, atol=1e-12, rtol=0
    ):
        raise ValueError("Configured water dielectric endpoint differs from the source registry")
    if ipa_dielectric_constant is not None and not np.isclose(
        ipa_dielectric_constant, registered_ipa_epsilon, atol=1e-12, rtol=0
    ):
        raise ValueError(
            "Configured isopropanol dielectric endpoint differs from the source registry"
        )
    water_dielectric_constant = float(registered_water_epsilon)
    ipa_dielectric_constant = float(registered_ipa_epsilon)
    validation_frame, validation_qa = _validate_ipa_estimate(
        _load_csv(ipa_dielectric_validation),
        constants,
        set(sources["source_id"].astype(str)),
        water_epsilon=water_dielectric_constant,
        ipa_epsilon=ipa_dielectric_constant,
        target_temperature_K=target_temperature_K,
        mape_threshold_percent=ipa_validation_mape_threshold_percent,
        max_error_threshold_percent=ipa_validation_max_error_threshold_percent,
    )
    compositions = _canonical_compositions(records)
    if len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} records, found {len(records)}")
    if len(compositions) != expected_compositions:
        raise ValueError(
            f"Expected {expected_compositions} canonical solvent compositions, found {len(compositions)}"
        )

    canonical_rows: list[dict[str, Any]] = []
    for _, composition in compositions.iterrows():
        canonical_rows.extend(
            _property_rows_for_composition(
                composition,
                measurement_frame,
                constants,
                target_temperature_K=target_temperature_K,
                exact_temperature_tolerance_K=exact_temperature_tolerance_K,
            )
        )
    canonical = pd.DataFrame(canonical_rows).sort_values(
        ["solvent_composition_key", "property_name"]
    ).reset_index(drop=True)
    if len(canonical) != expected_compositions * len(ALL_PROPERTIES):
        raise AssertionError("Canonical composition-property registry has the wrong grain")
    if canonical.duplicated(["solvent_composition_key", "property_name"]).any():
        raise AssertionError("Canonical composition-property registry contains duplicate keys")
    canonical = _attach_estimated_dielectric(
        canonical,
        validation_qa=validation_qa,
        water_epsilon=water_dielectric_constant,
        ipa_epsilon=ipa_dielectric_constant,
    )

    record_keys = records[["record_id"]].copy()
    record_keys["solvent_composition_key"] = records.apply(record_composition_key, axis=1)
    record_registry = record_keys.merge(
        canonical, on="solvent_composition_key", how="left", validate="many_to_many"
    ).sort_values(["record_id", "property_name"]).reset_index(drop=True)
    if len(record_registry) != expected_records * len(ALL_PROPERTIES):
        raise AssertionError("Record-property registry has the wrong grain")
    if record_registry.duplicated(["record_id", "property_name"]).any():
        raise AssertionError("Record-property registry contains duplicate keys")

    result = records.copy()
    result["solvent_composition_key"] = result.apply(record_composition_key, axis=1)
    record_wide = result[["record_id", "solvent_composition_key"]].copy()
    for property_name, output_column in PROPERTY_OUTPUT_COLUMNS.items():
        subset = record_registry.loc[
            record_registry["property_name"].eq(property_name),
            [
                "record_id",
                "strict_source_backed_value",
                "evidence_tier",
                "model_use_status",
                "source_ids",
                "series_id",
            ],
        ].rename(
            columns={
                "strict_source_backed_value": output_column,
                "evidence_tier": f"{output_column}__evidence_tier",
                "model_use_status": f"{output_column}__model_use_status",
                "source_ids": f"{output_column}__source_ids",
                "series_id": f"{output_column}__series_id",
            }
        )
        record_wide = record_wide.merge(subset, on="record_id", how="left", validate="one_to_one")
    dielectric = record_registry.loc[
        record_registry["property_name"].eq("dielectric_constant"),
        [
            "record_id",
            "estimated_value",
            "estimated_evidence_tier",
            "estimated_model_use_status",
            "best_available_value",
            "best_available_evidence_tier",
            "best_available_model_use_status",
            "estimated_linear_value",
            "estimated_looyenga_value",
            "estimated_model_interval_low",
            "estimated_model_interval_high",
        ],
    ].rename(
        columns={
            "estimated_value": ESTIMATED_DIELECTRIC_COLUMN,
            "estimated_evidence_tier": f"{ESTIMATED_DIELECTRIC_COLUMN}__evidence_tier",
            "estimated_model_use_status": ESTIMATED_DIELECTRIC_STATUS_COLUMN,
            "best_available_value": BEST_AVAILABLE_DIELECTRIC_COLUMN,
            "best_available_evidence_tier": f"{BEST_AVAILABLE_DIELECTRIC_COLUMN}__evidence_tier",
            "best_available_model_use_status": f"{BEST_AVAILABLE_DIELECTRIC_COLUMN}__model_use_status",
            "estimated_linear_value": f"{ESTIMATED_DIELECTRIC_COLUMN}__linear_sensitivity",
            "estimated_looyenga_value": f"{ESTIMATED_DIELECTRIC_COLUMN}__looyenga_sensitivity",
            "estimated_model_interval_low": f"{ESTIMATED_DIELECTRIC_COLUMN}__model_interval_low",
            "estimated_model_interval_high": f"{ESTIMATED_DIELECTRIC_COLUMN}__model_interval_high",
        }
    )
    record_wide = record_wide.merge(dielectric, on="record_id", how="left", validate="one_to_one")
    drop_existing = [column for column in record_wide.columns if column in result and column != "record_id"]
    result = result.drop(columns=drop_existing).merge(record_wide, on="record_id", how="left", validate="one_to_one")

    legacy_frame = None
    if legacy_audit is not None:
        legacy_frame = _load_csv(legacy_audit)
    legacy_comparison = _legacy_comparison(
        record_registry,
        legacy_frame,
        relative_threshold=legacy_relative_discrepancy_threshold,
    )

    strict_counts = {
        property_name: int(
            record_registry.loc[
                record_registry["property_name"].eq(property_name),
                "strict_source_backed_value",
            ].notna().sum()
        )
        for property_name in ALL_PROPERTIES
    }
    ab_counts = {
        property_name: int(
            record_registry.loc[
                record_registry["property_name"].eq(property_name)
                & record_registry["evidence_tier"].isin(
                    {"A_direct", "B_same_source_interpolation"}
                ),
                "strict_source_backed_value",
            ].notna().sum()
        )
        for property_name in ALL_PROPERTIES
    }
    c_counts = {
        property_name: int(
            record_registry.loc[
                record_registry["property_name"].eq(property_name)
                & record_registry["evidence_tier"].eq(
                    "C_secondary_reconstruction"
                ),
                "strict_source_backed_value",
            ].notna().sum()
        )
        for property_name in ALL_PROPERTIES
    }
    pending_source_file_ids = set(
        files.loc[
            files["sha256"].fillna("").astype(str).str.strip().eq(""),
            "source_file_id",
        ].astype(str)
    )
    strict_has_pending_file = record_registry["source_file_ids"].fillna("").apply(
        lambda value: bool(
            set(str(value).split("|")) & pending_source_file_ids
        )
    )
    fully_hash_frozen_strict_counts = {
        property_name: int(
            (
                record_registry["property_name"].eq(property_name)
                & record_registry["strict_source_backed_value"].notna()
                & ~strict_has_pending_file
            ).sum()
        )
        for property_name in ALL_PROPERTIES
    }
    dielectric_D_estimates = int(
        (
            record_registry["property_name"].eq("dielectric_constant")
            & record_registry["estimated_value"].notna()
            & record_registry["estimated_evidence_tier"].eq("D_model_estimate")
        ).sum()
    )
    measurement_file_hash_pending = measurement_frame["raw_file_sha256"].fillna(
        ""
    ).astype(str).str.strip().eq("")
    eligible_measurement_rows_with_hash_pending = int(
        (
            measurement_file_hash_pending
            & measurement_frame["model_use_status"].astype(str).str.startswith(
                "eligible_"
            )
        ).sum()
    )
    best_available_count = int(record_registry["best_available_value"].notna().sum())
    interpolation_rows = canonical["proxy_status"].astype(str).eq(
        "same_source_bracket_interpolation"
    )
    invalid_brackets = interpolation_rows & (
        canonical["composition_low"].ge(canonical["target_composition_coordinate"])
        | canonical["composition_high"].le(canonical["target_composition_coordinate"])
    )
    medium = canonical["confidence_flag"].fillna("").astype(str).eq(
        "medium_wide_composition_bracket"
    )
    dielectric_endpoint_parameters: dict[str, dict[str, Any]] = {}
    for component in ("water", "isopropanol"):
        parameter = constants_frame.loc[
            constants_frame["component"].astype(str).eq(component)
        ].iloc[0]
        dielectric_endpoint_parameters[component] = {
            "value": float(parameter["static_dielectric_constant"]),
            "temperature_K": float(parameter["static_dielectric_temperature_K"]),
            "definition": str(parameter["static_dielectric_definition"]),
            "source_id": str(parameter["static_dielectric_source_id"]),
            "source_file_id": str(parameter["static_dielectric_source_file_id"]),
            "source_file_sha256": str(
                parameter["static_dielectric_source_file_sha256"]
            ),
            "evidence_tier": str(parameter["static_dielectric_evidence_tier"]),
            "source_locator": str(parameter["static_dielectric_source_locator"]),
        }
    qa: dict[str, Any] = {
        "status": "ready_with_registered_estimate_gate",
        "target_temperature_K": float(target_temperature_K),
        "canonical_records": len(records),
        "canonical_compositions": len(compositions),
        "canonical_composition_property_rows": len(canonical),
        "record_property_rows": len(record_registry),
        "authoritative_measurement_rows": len(measurement_frame),
        "registered_property_sources": len(sources),
        "registered_source_files": len(files),
        "measurement_source_foreign_keys_valid": True,
        "measurement_source_file_hashes_valid": True,
        "measurement_rows_with_frozen_raw_sha256": int(
            measurement_frame["raw_file_sha256"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
        "measurement_rows_with_explicit_hash_pending": int(
            measurement_frame["raw_file_sha256"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),
        "record_property_values_ready": int(
            result[list(PROPERTY_OUTPUT_COLUMNS.values())].notna().sum().sum()
        ),
        "strict_source_backed_values_by_property": strict_counts,
        "A_or_B_values_by_property": ab_counts,
        "C_values_by_property": c_counts,
        "fully_hash_frozen_strict_values_by_property": (
            fully_hash_frozen_strict_counts
        ),
        "core_A_or_B_values": int(sum(ab_counts[name] for name in CORE_PROPERTIES)),
        "core_A_or_B_expected": expected_records * len(CORE_PROPERTIES),
        "dielectric_A_or_B_values": ab_counts["dielectric_constant"],
        "dielectric_A_or_B_expected": int(expected_dielectric_A_or_B_values),
        "dielectric_C_values": c_counts["dielectric_constant"],
        "dielectric_C_expected": int(expected_dielectric_C_values),
        "dielectric_D_estimates": dielectric_D_estimates,
        "dielectric_D_estimates_expected": int(expected_dielectric_D_estimates),
        "dielectric_strict_source_backed_values": strict_counts["dielectric_constant"],
        "dielectric_strict_source_backed_expected": int(
            expected_dielectric_strict_source_backed_values
        ),
        "best_available_record_property_values": best_available_count,
        "best_available_expected": expected_records * len(ALL_PROPERTIES),
        "eligible_measurement_rows_with_hash_pending": (
            eligible_measurement_rows_with_hash_pending
        ),
        "dielectric_strict_values_with_hash_pending": int(
            (
                record_registry["property_name"].eq("dielectric_constant")
                & record_registry["strict_source_backed_value"].notna()
                & strict_has_pending_file
            ).sum()
        ),
        "full_source_file_hash_acceptance_passed": (
            eligible_measurement_rows_with_hash_pending == 0
        ),
        "source_backed_4p_panel_model_use_status": (
            "eligible_complete_case_only"
            if eligible_measurement_rows_with_hash_pending == 0
            else "excluded_pending_primary_file_hashes"
        ),
        "records_with_all_three_core_proxies": int(
            result[[PROPERTY_OUTPUT_COLUMNS[name] for name in CORE_PROPERTIES]].notna().all(axis=1).sum()
        ),
        "records_with_all_four_strict_proxies": int(
            result[list(PROPERTY_OUTPUT_COLUMNS.values())].notna().all(axis=1).sum()
        ),
        "records_with_all_four_best_available_values": int(
            result[[
                *[PROPERTY_OUTPUT_COLUMNS[name] for name in CORE_PROPERTIES],
                BEST_AVAILABLE_DIELECTRIC_COLUMN,
            ]].notna().all(axis=1).sum()
        ),
        "composition_interpolation_rows": int(interpolation_rows.sum()),
        "invalid_same_source_brackets": int(invalid_brackets.sum()),
        "composition_extrapolation_used": False,
        "temperature_extrapolation_used": False,
        "medium_confidence_ipa_surface_compositions": int(medium.sum()),
        "medium_confidence_ipa_surface_sensitivity_required": bool(medium.any()),
        "legacy_comparison_rows": len(legacy_comparison),
        "legacy_discrepancies": int(legacy_comparison.get("discrepancy_flag", pd.Series(dtype=bool)).sum()),
        "ipa_dielectric_estimate_validation": validation_qa,
        "dielectric_estimator_endpoint_parameters": dielectric_endpoint_parameters,
        "estimated_panel_model_use_status": validation_qa["estimated_panel_model_use_status"],
        "legacy_candidate_values_used_as_authoritative": False,
        "property_composition_coordinates_coexist_in_registered_panel": False,
        "claim_boundary": (
            "solvent-mixture property proxy only; not a measured catalyst/Nafion/particle ink property"
        ),
    }
    if invalid_brackets.any():
        raise AssertionError("A same-source interpolation lacks two strict bracketing endpoints")
    if qa["core_A_or_B_values"] != qa["core_A_or_B_expected"]:
        qa["status"] = "incomplete_core_source_backed_coverage"
    elif (
        qa["dielectric_strict_source_backed_values"]
        != qa["dielectric_strict_source_backed_expected"]
    ):
        qa["status"] = "unexpected_dielectric_source_backed_coverage"
    elif (
        qa["dielectric_A_or_B_values"] != qa["dielectric_A_or_B_expected"]
        or qa["dielectric_C_values"] != qa["dielectric_C_expected"]
        or qa["dielectric_D_estimates"] != qa["dielectric_D_estimates_expected"]
    ):
        qa["status"] = "unexpected_dielectric_evidence_composition"
    elif best_available_count != qa["best_available_expected"]:
        qa["status"] = "core_ready_best_available_incomplete"

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_review = sources.rename(
        columns={
            "url": "source_url",
            "accessed_on": "source_accessed_on",
            "notes": "source_notes",
        }
    ).merge(
        files.rename(
            columns={
                "accessed_on": "file_accessed_on",
                "notes": "source_file_notes",
            }
        ),
        on="source_id",
        how="left",
        validate="one_to_many",
    )
    canonical.to_csv(destination / "canonical_solvent_properties.csv", index=False)
    record_registry.to_csv(destination / "record_solvent_property_registry.csv", index=False)
    record_wide.to_csv(destination / "record_solvent_properties.csv", index=False)
    # Backwards-compatible audit names point to the new, stricter registries.
    record_registry.to_csv(destination / "property_proxy_registry.csv", index=False)
    record_registry.to_csv(destination / "authoritative_property_registry.csv", index=False)
    measurement_frame.to_csv(
        destination / "authoritative_property_measurements_snapshot.csv", index=False
    )
    sources.to_csv(destination / "property_sources_snapshot.csv", index=False)
    files.to_csv(destination / "property_source_files_snapshot.csv", index=False)
    source_review.to_csv(destination / "property_sources_review.csv", index=False)
    constants_frame.to_csv(destination / "property_component_constants_snapshot.csv", index=False)
    validation_frame.to_csv(destination / "ipa_dielectric_validation_results.csv", index=False)
    legacy_comparison.to_csv(destination / "legacy_property_comparison.csv", index=False)
    (destination / "property_proxy_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False, default=_json_scalar) + "\n",
        encoding="utf-8",
    )
    artifacts = [
        "canonical_solvent_properties.csv",
        "record_solvent_property_registry.csv",
        "record_solvent_properties.csv",
        "property_proxy_registry.csv",
        "authoritative_property_registry.csv",
        "authoritative_property_measurements_snapshot.csv",
        "property_sources_snapshot.csv",
        "property_source_files_snapshot.csv",
        "property_sources_review.csv",
        "property_component_constants_snapshot.csv",
        "ipa_dielectric_validation_results.csv",
        "legacy_property_comparison.csv",
        "property_proxy_qa.json",
    ]
    artifact_hashes = {
        name: _sha256(destination / name)
        for name in artifacts
        if (destination / name).is_file()
    }
    (destination / "property_artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result, canonical, record_registry, qa
