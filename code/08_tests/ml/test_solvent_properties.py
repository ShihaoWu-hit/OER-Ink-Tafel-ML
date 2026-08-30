from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "code" / "Ink_Tafel_Study" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ink_tafel_study.curation import parse_raw_workbook  # noqa: E402
from ink_tafel_study.solvent_properties import (  # noqa: E402
    BEST_AVAILABLE_DIELECTRIC_COLUMN,
    ESTIMATED_DIELECTRIC_COLUMN,
    ESTIMATED_DIELECTRIC_STATUS_COLUMN,
    bruggeman_permittivity,
    convert_dosed_volume_fractions,
    linear_permittivity,
    looyenga_permittivity,
    reconstruct_solvent_property_registries,
)


RAW = REPO_ROOT / "Data" / "Data_ink_cleansed_0520_v3.xlsx"
LEGACY = REPO_ROOT / "Data" / "ink_provenance.csv"
CONFIG = REPO_ROOT / "code" / "Ink_Tafel_Study" / "config"
requires_legacy_raw_workbook = pytest.mark.skipif(
    not RAW.is_file(),
    reason="superseded v2 raw workbook was intentionally retired in the canonical-97 cleanup",
)


def test_dosed_volume_basis_conversion_does_not_use_mixture_contraction() -> None:
    constants = {
        "water": {"pure_density_kg_m3": 997.0474, "molar_mass_g_mol": 18.01528},
        "methanol": {"pure_density_kg_m3": 787.2, "molar_mass_g_mol": 32.04186},
    }
    volume = {"water": 0.5, "methanol": 0.5}
    mass = convert_dosed_volume_fractions(volume, constants, basis="mass_fraction")
    mole = convert_dosed_volume_fractions(volume, constants, basis="mole_fraction")
    assert np.isclose(mass["methanol"], 787.2 / (787.2 + 997.0474))
    expected_moles = (787.2 / 32.04186) / (
        (787.2 / 32.04186) + (997.0474 / 18.01528)
    )
    assert np.isclose(mole["methanol"], expected_moles)
    assert np.isclose(sum(mass.values()), 1.0)
    assert np.isclose(sum(mole.values()), 1.0)


def test_dielectric_mixing_models_have_exact_endpoints() -> None:
    water = 78.375
    ipa = 19.387
    for model in (bruggeman_permittivity, looyenga_permittivity, linear_permittivity):
        assert np.isclose(model(0.0, ipa, water), water)
        assert np.isclose(model(1.0, ipa, water), ipa)


@requires_legacy_raw_workbook
def test_registered_property_reconstruction_has_auditable_grain_and_gates(
    tmp_path: Path,
) -> None:
    records, _, _, _ = parse_raw_workbook(RAW, legacy_metadata_csv=LEGACY)
    result, canonical, registry, qa = reconstruct_solvent_property_registries(
        records,
        CONFIG / "authoritative_property_measurements.csv",
        CONFIG / "property_component_constants.csv",
        CONFIG / "property_sources.csv",
        CONFIG / "property_source_files.csv",
        CONFIG / "property_dielectric_validation.csv",
        tmp_path,
    )
    assert len(result) == 97
    assert result["record_id"].is_unique
    assert canonical["solvent_composition_key"].nunique() == 33
    assert len(canonical) == 132
    assert len(registry) == 388
    assert not registry.duplicated(["record_id", "property_name"]).any()
    assert qa["core_A_or_B_values"] == 291
    assert qa["strict_source_backed_values_by_property"] == {
        "viscosity": 97,
        "surface_tension": 97,
        "density": 97,
        "dielectric_constant": 66,
    }
    assert qa["dielectric_A_or_B_values"] == 40
    assert qa["dielectric_C_values"] == 26
    assert qa["dielectric_D_estimates"] == 31
    assert qa["fully_hash_frozen_strict_values_by_property"] == {
        "viscosity": 97,
        "surface_tension": 97,
        "density": 97,
        "dielectric_constant": 8,
    }
    assert qa["measurement_rows_with_frozen_raw_sha256"] == 189
    assert qa["measurement_rows_with_explicit_hash_pending"] == 10
    assert qa["eligible_measurement_rows_with_hash_pending"] == 10
    assert qa["dielectric_strict_values_with_hash_pending"] == 58
    assert not qa["full_source_file_hash_acceptance_passed"]
    assert qa["source_backed_4p_panel_model_use_status"] == (
        "excluded_pending_primary_file_hashes"
    )
    endpoints = qa["dielectric_estimator_endpoint_parameters"]
    assert np.isclose(endpoints["water"]["value"], 78.375)
    assert endpoints["water"]["evidence_tier"] == "A_direct"
    assert endpoints["water"]["source_id"] == "iapws_liquid_water_2011"
    assert endpoints["water"]["source_file_sha256"] == (
        "04bd37dd7f4c9c535230af4c51d2a803912d1d41c1656e20527eae598bc55ff4"
    )
    assert np.isclose(endpoints["isopropanol"]["value"], 19.387)
    assert endpoints["isopropanol"]["evidence_tier"] == (
        "C_secondary_reconstruction"
    )
    assert endpoints["isopropanol"]["source_id"] == "wang_2023_fpe"
    assert endpoints["isopropanol"]["source_file_sha256"] == (
        "a77bd0def1238f45e9112c703fe876c3f29931255bf552d2e01e889ee8ae0921"
    )
    pure_water = canonical.loc[canonical["mixture_components"].eq("water")]
    water_density = pure_water.loc[pure_water["property_name"].eq("density")].iloc[0]
    water_viscosity = pure_water.loc[pure_water["property_name"].eq("viscosity")].iloc[0]
    water_surface = pure_water.loc[
        pure_water["property_name"].eq("surface_tension")
    ].iloc[0]
    assert np.isclose(water_density["strict_source_backed_value"], 997.0474)
    assert water_density["series_id"] == "density_water_iapws95_2018"
    assert np.isclose(water_viscosity["strict_source_backed_value"], 0.89)
    assert "gonzalez_2007" in water_viscosity["series_id"]
    assert np.isclose(water_surface["strict_source_backed_value"], 72.01)
    assert "vazquez_1995" in water_surface["series_id"]
    # The direct water-viscosity endpoint has a finite composition-series LOO
    # diagnostic, but its source uncertainty is not standardized.  LOO must
    # not be relabelled as the direct point's standard uncertainty.
    assert np.isfinite(float(water_viscosity["loo_interpolation_rmse"]))
    assert np.isnan(float(water_viscosity["proxy_standard_uncertainty"]))
    # The two methanol/water records use the hash-frozen Smith Figure 2
    # reconstruction.  They close numeric coverage but remain explicitly C,
    # never A/B, until an original numerical table is obtained.
    methanol_dielectric = registry.loc[
        registry["property_name"].eq("dielectric_constant")
        & registry["source_ids"].astype(str).str.contains("smith_1998_fpe", regex=False)
    ]
    assert len(methanol_dielectric) == 2
    assert methanol_dielectric["evidence_tier"].eq("C_secondary_reconstruction").all()
    assert np.allclose(methanol_dielectric["strict_source_backed_value"], 58.30, atol=0.15)
    assert qa["best_available_record_property_values"] == 388
    assert qa["records_with_all_four_best_available_values"] == 97
    assert qa["medium_confidence_ipa_surface_compositions"] == 2
    medium_surface = canonical.loc[
        canonical["confidence_flag"].fillna("").eq(
            "medium_wide_composition_bracket"
        )
    ]
    assert medium_surface["property_name"].eq("surface_tension").all()
    assert np.allclose(
        np.sort(medium_surface["target_composition_coordinate"].to_numpy(float)),
        [0.555968, 0.808642],
        atol=1e-6,
    )
    assert medium_surface["model_use_status"].eq(
        "eligible_source_backed_medium_confidence"
    ).all()
    medium_status_column = (
        "solvent_mixture_property_proxy_surface_tension_mN_m__model_use_status"
    )
    medium_records = result.loc[
        result[medium_status_column].eq("eligible_source_backed_medium_confidence"),
        "record_id",
    ]
    assert set(medium_records.astype(str)) == {"36", "97"}
    assert qa["invalid_same_source_brackets"] == 0
    assert not qa["composition_extrapolation_used"]
    assert not qa["temperature_extrapolation_used"]
    assert not qa["ipa_dielectric_estimate_validation"]["gate_passed"]
    assert np.isclose(
        qa["ipa_dielectric_estimate_validation"]["mape_percent"],
        5.692405541233109,
    )
    assert np.isclose(
        qa["ipa_dielectric_estimate_validation"][
            "maximum_single_point_error_percent"
        ],
        9.636134168416442,
    )
    assert qa["estimated_panel_model_use_status"] == "excluded"
    assert result[ESTIMATED_DIELECTRIC_COLUMN].notna().sum() == 31
    assert result[BEST_AVAILABLE_DIELECTRIC_COLUMN].notna().sum() == 97
    excluded = result.loc[
        result[ESTIMATED_DIELECTRIC_COLUMN].notna(), ESTIMATED_DIELECTRIC_STATUS_COLUMN
    ]
    assert excluded.eq("excluded_validation_gate_failed").all()
    assert result["ir_correction_status"].eq("unknown").all()
    assert result["ir_review_status"].eq("unknown").all()

    saved_qa = json.loads((tmp_path / "property_proxy_qa.json").read_text())
    assert saved_qa["record_property_rows"] == 388
    assert (tmp_path / "property_artifact_hashes.json").is_file()


@requires_legacy_raw_workbook
def test_dielectric_registry_rejects_nonstatic_values(tmp_path: Path) -> None:
    records, _, _, _ = parse_raw_workbook(RAW, legacy_metadata_csv=LEGACY)
    measurements = pd.read_csv(CONFIG / "authoritative_property_measurements.csv")
    first_dielectric = measurements.index[
        measurements["property_name"].eq("dielectric_constant")
    ][0]
    measurements.loc[first_dielectric, "static_low_frequency"] = "false"
    with pytest.raises(ValueError, match="static/low-frequency-limit"):
        reconstruct_solvent_property_registries(
            records,
            measurements,
            CONFIG / "property_component_constants.csv",
            CONFIG / "property_sources.csv",
            CONFIG / "property_source_files.csv",
            CONFIG / "property_dielectric_validation.csv",
            tmp_path,
        )


@requires_legacy_raw_workbook
def test_registered_dielectric_coverage_shortfall_is_not_ready(tmp_path: Path) -> None:
    records, _, _, _ = parse_raw_workbook(RAW, legacy_metadata_csv=LEGACY)
    measurements = pd.read_csv(CONFIG / "authoritative_property_measurements.csv")
    measurements = measurements.loc[
        ~measurements["series_id"].eq(
            "dielectric_methanol_water_smith_1998_figure_digitized"
        )
    ].copy()
    _, _, _, qa = reconstruct_solvent_property_registries(
        records,
        measurements,
        CONFIG / "property_component_constants.csv",
        CONFIG / "property_sources.csv",
        CONFIG / "property_source_files.csv",
        CONFIG / "property_dielectric_validation.csv",
        tmp_path,
    )
    assert qa["dielectric_strict_source_backed_values"] == 64
    assert qa["status"] == "unexpected_dielectric_source_backed_coverage"


@requires_legacy_raw_workbook
def test_ipa_dielectric_validation_grid_fails_closed(tmp_path: Path) -> None:
    records, _, _, _ = parse_raw_workbook(RAW, legacy_metadata_csv=LEGACY)
    validation = pd.read_csv(CONFIG / "property_dielectric_validation.csv")
    validation.loc[0, "temperature_K"] = 303.15
    with pytest.raises(ValueError, match="must be at 298.15 K"):
        reconstruct_solvent_property_registries(
            records,
            CONFIG / "authoritative_property_measurements.csv",
            CONFIG / "property_component_constants.csv",
            CONFIG / "property_sources.csv",
            CONFIG / "property_source_files.csv",
            validation,
            tmp_path,
        )


@requires_legacy_raw_workbook
def test_dielectric_endpoint_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    records, _, _, _ = parse_raw_workbook(RAW, legacy_metadata_csv=LEGACY)
    constants = pd.read_csv(CONFIG / "property_component_constants.csv")
    constants.loc[
        constants["component"].eq("isopropanol"),
        "static_dielectric_source_file_sha256",
    ] = "0" * 64
    with pytest.raises(ValueError, match="static-dielectric source-file SHA256"):
        reconstruct_solvent_property_registries(
            records,
            CONFIG / "authoritative_property_measurements.csv",
            constants,
            CONFIG / "property_sources.csv",
            CONFIG / "property_source_files.csv",
            CONFIG / "property_dielectric_validation.csv",
            tmp_path,
        )


@requires_legacy_raw_workbook
def test_measurement_original_reference_crosswalk_fails_closed(tmp_path: Path) -> None:
    records, _, _, _ = parse_raw_workbook(RAW, legacy_metadata_csv=LEGACY)
    measurements = pd.read_csv(CONFIG / "authoritative_property_measurements.csv")
    measurements.loc[
        measurements["series_id"].eq("density_methanol_water_gonzalez_2007"),
        "source_id",
    ] = "pubchem_ethanol"
    with pytest.raises(ValueError, match="source_reference_id is not registered"):
        reconstruct_solvent_property_registries(
            records,
            measurements,
            CONFIG / "property_component_constants.csv",
            CONFIG / "property_sources.csv",
            CONFIG / "property_source_files.csv",
            CONFIG / "property_dielectric_validation.csv",
            tmp_path,
        )


@requires_legacy_raw_workbook
def test_dielectric_evidence_composition_is_fail_closed(tmp_path: Path) -> None:
    records, _, _, _ = parse_raw_workbook(RAW, legacy_metadata_csv=LEGACY)
    measurements = pd.read_csv(CONFIG / "authoritative_property_measurements.csv")
    measurements.loc[
        measurements["series_id"].eq(
            "dielectric_methanol_water_smith_1998_figure_digitized"
        ),
        "evidence_tier",
    ] = "A_direct"
    _, _, _, qa = reconstruct_solvent_property_registries(
        records,
        measurements,
        CONFIG / "property_component_constants.csv",
        CONFIG / "property_sources.csv",
        CONFIG / "property_source_files.csv",
        CONFIG / "property_dielectric_validation.csv",
        tmp_path,
    )
    assert qa["dielectric_A_or_B_values"] == 42
    assert qa["dielectric_C_values"] == 24
    assert qa["status"] == "unexpected_dielectric_evidence_composition"
