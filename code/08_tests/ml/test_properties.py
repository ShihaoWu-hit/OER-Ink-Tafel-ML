from pathlib import Path

import numpy as np
import pandas as pd

from ink_tafel_study.properties import (
    attach_authoritative_property_proxies,
    attach_source_reconstructed_property_proxies,
    build_legacy_property_source_audit,
    record_composition_key,
)


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [1],
            "solvent_fraction_acetone": [0.0],
            "solvent_fraction_water": [0.5],
            "solvent_fraction_ethanol": [0.5],
            "solvent_fraction_isopropanol": [0.0],
            "solvent_fraction_methanol": [0.0],
            "solvent_fraction_other": [0.0],
        }
    )


def _row(key: str, temperature: float, value: float) -> dict[str, object]:
    return {
        "solvent_composition_key": key,
        "property_name": "viscosity",
        "value": value,
        "unit": "mPa s",
        "temperature_K": temperature,
        "composition_basis": "volume_fraction",
        "experimental_method": "capillary viscometer",
        "source_doi": "10.0000/example",
        "source_url": "",
        "access_date": "2026-07-16",
        "source_id": "example_source",
    }


def _sources() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "example_source",
                "source_type": "test",
                "authority": "test",
                "title": "test source",
                "usage_status": "authoritative_machine_source",
            }
        ]
    )


def test_bracketed_interpolation_without_extrapolation(tmp_path: Path) -> None:
    records = _records()
    key = record_composition_key(records.iloc[0])
    measurements = pd.DataFrame([_row(key, 298.0, 2.0), _row(key, 299.0, 3.0)])
    result, registry, qa = attach_authoritative_property_proxies(
        records, measurements, tmp_path, source_catalog=_sources()
    )
    expected = 2.0 + 0.15 * (3.0 - 2.0)
    assert np.isclose(result["solvent_mixture_property_proxy_viscosity_mPa_s"].iloc[0], expected)
    assert registry.loc[registry.property_name.eq("viscosity"), "proxy_status"].item() == "interpolated_bracketed"
    viscosity = registry.loc[registry.property_name.eq("viscosity")].iloc[0]
    assert viscosity["source_composition_bases"] == "volume_fraction"
    assert viscosity["source_experimental_methods"] == "capillary viscometer"
    assert viscosity["source_access_dates"] == "2026-07-16"
    assert viscosity["source_dois"] == "10.0000/example"
    assert qa["temperature_extrapolation_used"] is False
    assert qa["measurement_source_foreign_keys_valid"] is True


def test_one_sided_temperature_data_remain_missing(tmp_path: Path) -> None:
    records = _records()
    key = record_composition_key(records.iloc[0])
    measurements = pd.DataFrame([_row(key, 290.0, 2.0)])
    result, registry, qa = attach_authoritative_property_proxies(
        records, measurements, tmp_path, source_catalog=_sources()
    )
    assert pd.isna(result["solvent_mixture_property_proxy_viscosity_mPa_s"].iloc[0])
    assert registry.loc[registry.property_name.eq("viscosity"), "proxy_status"].item() == "missing_no_temperature_bracket"
    assert qa["legacy_candidate_values_used"] is False


def test_mole_fraction_key_is_not_attached_to_volume_fraction_recipe(tmp_path: Path) -> None:
    records = _records()
    key = record_composition_key(records.iloc[0])
    row = _row(key, 298.15, 2.0)
    row["composition_basis"] = "mole_fraction"
    result, registry, qa = attach_authoritative_property_proxies(
        records, pd.DataFrame([row]), tmp_path, source_catalog=_sources()
    )
    assert pd.isna(result["solvent_mixture_property_proxy_viscosity_mPa_s"].iloc[0])
    assert (
        registry.loc[registry.property_name.eq("viscosity"), "proxy_status"].item()
        == "missing_composition_basis_not_volume_fraction"
    )
    assert qa["composition_extrapolation_used"] is False


def test_unknown_property_source_id_is_rejected(tmp_path: Path) -> None:
    records = _records()
    key = record_composition_key(records.iloc[0])
    row = _row(key, 298.15, 2.0)
    row["source_id"] = "not_registered"
    with np.testing.assert_raises_regex(ValueError, "unknown source_id"):
        attach_authoritative_property_proxies(
            records, pd.DataFrame([row]), tmp_path, source_catalog=_sources()
        )


def test_legacy_source_audit_converts_units_without_promoting(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        [
            {
                "record_id": 1,
                "mixture_system_raw": "ethanol/water",
                "source_volume_fraction_percent_raw": 40.0,
                "viscosity_Pa_s_candidate": 0.002,
                "surface_tension_N_m_candidate": np.nan,
                "density_kg_m3_candidate": np.nan,
                "dielectric_constant_candidate": np.nan,
            }
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "mixture_system": "ethanol/water",
                "source_volume_fraction_percent": 40.0,
                "property_name": "viscosity",
                "legacy_value": 0.002,
                "legacy_unit": "Pa s",
                "source_id": "example_source",
                "source_match_status": "exact_nist_row",
                "temperature_eligibility": "exact_target",
                "model_use_status": "eligible_source_row",
            }
        ]
    )
    audit, qa = build_legacy_property_source_audit(
        candidates, mapping, _sources(), tmp_path
    )
    viscosity = audit.loc[audit["property_name"].eq("viscosity")].iloc[0]
    assert viscosity["converted_candidate_value"] == 2.0
    assert viscosity["model_unit"] == "mPa s"
    assert bool(viscosity["source_measurement_row_ready"])
    assert qa["legacy_candidate_values_documented"] == 1
    assert qa["source_measurement_rows_ready"] == 1


def test_source_reconstructed_proxy_uses_bounded_composition_match(tmp_path: Path) -> None:
    records = _records()
    audit = pd.DataFrame(
        [
            {
                "record_id": 1,
                "property_name": "viscosity",
                "converted_candidate_value": 2.271,
                "ink_volume_fraction_percent": 40.0,
                "source_volume_fraction_percent": 40.111659,
                "source_measurement_row_ready": True,
                "source_match_status": "exact_nist_row",
                "model_use_status": "eligible_source_row",
            }
        ]
    )
    result, registry, qa = attach_source_reconstructed_property_proxies(
        records, audit, tmp_path, max_abs_composition_delta_percent=1.0
    )
    assert np.isclose(
        result["solvent_mixture_property_proxy_viscosity_mPa_s"].iloc[0], 2.271
    )
    assert (
        registry.loc[registry["property_name"].eq("viscosity"), "proxy_status"].item()
        == "measured_source_row_within_composition_tolerance"
    )
    assert qa["source_reconstructed_values_attached"] == 1
    assert qa["composition_extrapolation_used"] is False


def test_source_reconstructed_proxy_holds_large_composition_delta(tmp_path: Path) -> None:
    records = _records()
    audit = pd.DataFrame(
        [
            {
                "record_id": 1,
                "property_name": "viscosity",
                "converted_candidate_value": 2.271,
                "ink_volume_fraction_percent": 40.0,
                "source_volume_fraction_percent": 45.0,
                "source_measurement_row_ready": True,
                "source_match_status": "exact_nist_row",
                "model_use_status": "eligible_source_row",
            }
        ]
    )
    result, registry, qa = attach_source_reconstructed_property_proxies(
        records, audit, tmp_path, max_abs_composition_delta_percent=1.0
    )
    assert pd.isna(result["solvent_mixture_property_proxy_viscosity_mPa_s"].iloc[0])
    assert registry["proxy_status"].item() == "held_source_target_composition_delta"
    assert qa["source_reconstructed_values_attached"] == 0
