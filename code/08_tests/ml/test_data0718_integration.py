from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ink_tafel_study.data0718_integration import (
    DIELECTRIC_BEST_AVAILABLE,
    DIELECTRIC_BEST_AVAILABLE_IS_ESTIMATED,
    _normalize_ir_status,
    _normalize_solvent,
    _parse_compensation_fraction,
    attach_properties,
    build_crosswalk,
    read_data0718,
    read_manual_corrections,
)
from ink_tafel_study.best_available_export import build_best_available_export


LEGACY_EXPORT = Path("Data/Data_ink_Tafel_ML_ready_0719_v3")
requires_legacy_export = pytest.mark.skipif(
    not LEGACY_EXPORT.is_dir(),
    reason="superseded ML-ready exports were intentionally retired in favor of Data/02_inputs/canonical_dataset_ir_corrected_v2",
)


def _synthetic_curve_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.DataFrame(
        {
            "record_id": list(range(1, 98)),
            "doi": [f"10.test/{record_id}" for record_id in range(1, 98)],
        }
    )
    rows = []
    for old_id in range(1, 98):
        if old_id == 42:
            curve_ids = ["sr042_v1", "sr042_v2"]
        else:
            curve_ids = [f"sr{old_id:03d}"]
        for variant, curve_id in enumerate(curve_ids, 1):
            if curve_id == "sr042_v1":
                new_id = 42
            elif curve_id == "sr042_v2":
                new_id = 43
            elif curve_id == "sr050":
                new_id = 41
            elif 43 <= old_id <= 49:
                new_id = old_id + 1
            else:
                new_id = old_id
            rows.append(
                {
                    "curve_id": curve_id,
                    "record_id": old_id,
                    "curve_variant": variant,
                    "curve_variants_for_sample": len(curve_ids),
                    "workbook_doi": f"10.test/{new_id}",
                    "source_sha256": "a" * 64,
                }
            )
    return pd.DataFrame(rows), records


def test_approved_crosswalk_is_one_to_one_after_duplicate_exclusion() -> None:
    registry, records = _synthetic_curve_registry()
    crosswalk = build_crosswalk(registry, records)
    retained = crosswalk.loc[crosswalk["disposition"].eq("retain")]

    assert len(crosswalk) == 98
    assert len(retained) == 97
    assert retained["new_record_id"].astype(int).tolist() != []
    assert set(retained["new_record_id"].astype(int)) == set(range(1, 98))
    assert crosswalk.loc[crosswalk["curve_id"].eq("sr042_v1"), "new_record_id"].iloc[0] == 42
    assert crosswalk.loc[crosswalk["curve_id"].eq("sr042_v2"), "new_record_id"].iloc[0] == 43
    assert crosswalk.loc[crosswalk["curve_id"].eq("sr049"), "new_record_id"].iloc[0] == 50
    assert crosswalk.loc[crosswalk["curve_id"].eq("sr050"), "disposition"].iloc[0] == "exclude_duplicate"
    assert retained["doi_match"].all()


def test_ir_compensation_state_and_fraction_parser() -> None:
    assert _normalize_ir_status("Yes (explicit)") == ("corrected", "reported")
    assert _normalize_ir_status("No") == ("not_corrected", "reported")
    assert _normalize_ir_status("Unreported") == ("not_corrected", "unreported")
    assert _normalize_ir_status("Unverifiable (full text inaccessible)") == (
        "not_corrected",
        "unverifiable",
    )
    assert _parse_compensation_fraction("95%") == 0.95
    assert pd.isna(_parse_compensation_fraction("Reference calibration used"))


def test_methanol_is_not_misclassified_as_ethanol() -> None:
    assert _normalize_solvent("methanol") == "methanol"
    assert _normalize_solvent("ethanol") == "ethanol"


def test_record35_manual_correction_preserves_raw_and_updates_curated_values() -> None:
    corrections = read_manual_corrections(
        Path("code/Ink_Tafel_Study/config/data0718_manual_corrections.csv")
    )
    records, _ = read_data0718(Path("Data/01_raw/formulation_source.xlsx"), corrections)
    row = records.loc[records["record_id"].eq(35)].iloc[0]

    assert row["ionomer_volume_uL_source_raw"] == 200000
    assert row["ionomer_volume_uL_per_mgcat_source_raw"] == 100000
    assert row["ionomer_volume_uL"] == 200
    assert row["ionomer_volume_uL_per_mgcat"] == 100
    assert row["total_liquid_volume_uL_source_raw"] == 201000
    assert row["total_liquid_volume_uL"] == 1200
    assert row["ionomer_to_solvent_ratio"] == 0.2
    assert row["manual_correction_id"] == "DATA0718-CORR-001"
    assert bool(row["qa_source_unit_outlier_corrected"])
    assert not bool(row["qa_critical_unit_outlier"])


def test_best_available_dielectric_has_explicit_estimated_indicator() -> None:
    records = pd.DataFrame({"solvent_composition_key": ["a", "b"]})
    rows = []
    for key in ("a", "b"):
        for property_name, value in (
            ("viscosity", 1.0),
            ("surface_tension", 2.0),
            ("density", 3.0),
            ("dielectric_constant", 4.0 if key == "a" else float("nan")),
        ):
            rows.append(
                {
                    "solvent_composition_key": key,
                    "property_name": property_name,
                    "proxy_value": value,
                    "estimated_value": 5.0 if property_name == "dielectric_constant" and key == "b" else float("nan"),
                    "best_available_value": (
                        5.0
                        if property_name == "dielectric_constant" and key == "b"
                        else value
                    ),
                }
            )
    result = attach_properties(records, pd.DataFrame(rows))
    assert result["solvent_mixture_property_proxy_dielectric_constant_missing"].tolist() == [0, 1]
    assert result[DIELECTRIC_BEST_AVAILABLE].tolist() == [4.0, 5.0]
    assert result[DIELECTRIC_BEST_AVAILABLE_IS_ESTIMATED].tolist() == [0, 1]


@requires_legacy_export
def test_best_available_export_preserves_exact_target_and_flags_fallback_reasons() -> None:
    source = Path("Data/Data_ink_Tafel_ML_ready_0719_v3")
    training, audit, qa = build_best_available_export(
        pd.read_csv(source / "records.csv"),
        pd.read_csv(source / "target_registry.csv"),
        pd.read_csv("Data/Data_ink_Tafel_ML_ready_0718/primary_alkaline_flat.csv"),
    )

    assert len(training) == 97
    assert qa["electrolyte_scope_counts"] == {
        "alkaline": 90,
        "acidic": 5,
        "neutral": 2,
    }
    assert training.loc[
        training["single_regime_fallback_used"].eq(1), "record_id"
    ].tolist() == [23, 45, 46, 47, 48, 50, 74, 91]
    assert training.loc[
        training["target_extrapolation_required"].eq(1), "record_id"
    ].tolist() == [23, 50, 74, 91]
    assert training.loc[
        training["current_unit_conversion_required"].eq(1), "record_id"
    ].tolist() == [45, 46, 47, 48]
    assert audit.loc[
        ~audit["ml_training_eligible_best_available"], "record_id"
    ].tolist() == []
