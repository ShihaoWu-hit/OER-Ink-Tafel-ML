from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "code" / "Ink_Tafel_Study" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ink_tafel_study.curation import (  # noqa: E402
    IR_FIELDS,
    build_curve_registry,
    build_property_candidate_registry,
    compute_recipe_hash,
    curate_canonical_registry,
    parse_raw_workbook,
)


RAW = REPO_ROOT / "Data" / "Data_ink_cleansed_0520_v3.xlsx"
PROPERTIES = REPO_ROOT / "Data" / "ink_properties.xlsx"
LEGACY = REPO_ROOT / "Data" / "ink_provenance.csv"
REFIT_RUN = REPO_ROOT / "Tafel_refitting" / "outputs" / "20260714_040837"
LEGACY_INPUTS_AVAILABLE = RAW.is_file() and PROPERTIES.is_file() and REFIT_RUN.is_dir()
requires_legacy_inputs = pytest.mark.skipif(
    not LEGACY_INPUTS_AVAILABLE,
    reason="superseded v2 curation inputs were intentionally retired in the canonical-97 cleanup",
)


def _parsed():
    return parse_raw_workbook(RAW, legacy_metadata_csv=LEGACY)


@requires_legacy_inputs
def test_raw_workbook_has_exactly_97_unique_records_and_retains_conditions() -> None:
    records, _, _, _ = _parsed()
    assert records["record_id"].tolist() == list(range(1, 98))
    assert records["record_id"].is_unique
    assert records["rpm"].notna().sum() == 95
    assert records["scan_rate_mV_s"].notna().sum() == 95
    assert (records["electrolyte_scope"] == "alkaline").sum() == 90
    assert (records["electrolyte_scope"] == "acidic").sum() == 5
    assert (records["electrolyte_scope"] == "neutral").sum() == 2
    for field in (
        "electrode_type",
        "catalyst_loading_mg_cm2",
        "sonication_time_min",
        "drying_process",
        "drying_temperature_C",
        "drying_time_min",
    ):
        assert field in records.columns


@requires_legacy_inputs
def test_vertical_merged_publication_cells_are_propagated_with_provenance() -> None:
    records, _, log, provenance = _parsed()
    shared = records.set_index("record_id").loc[[44, 45, 46, 47]]
    assert shared["doi"].notna().all()
    assert shared["doi"].nunique() == 1
    assert shared["paper_name"].nunique() == 1
    assert shared["publication_id"].nunique() == 1
    propagated = provenance[
        provenance["record_id"].isin([45, 46, 47])
        & provenance["field"].isin(["doi", "paper_name"])
    ]
    assert propagated["merged_anchor_cell"].astype(str).ne("").all()
    assert (
        (log["record_id"].isin([45, 46, 47]))
        & (log["issue_type"] == "merged_cell_propagation")
    ).any()
    header_paths = provenance.groupby("field")["source_header_path"].first()
    assert "Input" in header_paths.loc["solvent_1_type"]
    assert "Solvent composition" in header_paths.loc["solvent_1_type"]
    assert "Output" in header_paths.loc["y_tafel_slope_reported_mV_dec"]


@requires_legacy_inputs
def test_comments_are_evidence_not_silent_edits() -> None:
    records, _, log, _ = _parsed()
    comments = log[log["issue_type"] == "excel_comment"].set_index("record_id")
    assert set(comments.index) == {16, 35}
    assert "Overnight" in comments.loc[16, "reason"]
    assert "typo" in comments.loc[35, "reason"]
    assert comments["decision"].eq("preserved_for_manual_review_no_automatic_change").all()
    # The source value is retained; the comment is not treated as permission to
    # invent a replacement.
    assert records.set_index("record_id").loc[35, "ionomer_volume_uL_per_mgcat"] == 100.0


def test_recipe_hash_excludes_targets_process_and_test_conditions() -> None:
    base = {
        "catalyst_type": "RuO2",
        "catalyst_mass_mg": 5.0,
        "conductive_agent_type": "none",
        "conductive_agent_mass_mg": 0.0,
        "ionomer_type": "5%Nafion",
        "ionomer_volume_uL_per_mgcat": 10.0,
        "solvent_1_type": "water",
        "solvent_1_volume_uL_per_mgcat": 100.0,
        "solvent_2_type": "ethanol",
        "solvent_2_volume_uL_per_mgcat": 20.0,
        "total_solvent_volume_uL": 600.0,
        "rpm": 1600,
        "electrolyte_identity": "KOH",
        "sonication_time_min": 30,
        "y_tafel_slope_reported_mV_dec": 60,
    }
    changed_context = {
        **base,
        "rpm": 0,
        "electrolyte_identity": "H2SO4",
        "sonication_time_min": 720,
        "y_tafel_slope_reported_mV_dec": 180,
    }
    swapped_solvents = {
        **base,
        "solvent_1_type": "ethanol",
        "solvent_1_volume_uL_per_mgcat": 20.0,
        "solvent_2_type": "water",
        "solvent_2_volume_uL_per_mgcat": 100.0,
    }
    changed_recipe = {**base, "ionomer_volume_uL_per_mgcat": 11.0}
    assert compute_recipe_hash(base) == compute_recipe_hash(changed_context)
    assert compute_recipe_hash(base) == compute_recipe_hash(swapped_solvents)
    assert compute_recipe_hash(base) != compute_recipe_hash(changed_recipe)


@requires_legacy_inputs
def test_ir_fields_are_reserved_unknown_and_unknown_labs_do_not_globally_merge() -> None:
    records, sources, _, _ = _parsed()
    for field in IR_FIELDS:
        assert field in records.columns
    assert records["ir_correction_status"].eq("unknown").all()
    assert records["ir_review_status"].eq("unknown").all()
    assert sources["lab_metadata_review_status"].eq("incomplete_lab_metadata").all()
    assert sources["lab_group_id"].nunique() == sources["publication_id"].nunique()
    # Same publication connects; unknown labs from different publications do not.
    by_record = records.set_index("record_id")
    assert by_record.loc[44, "cv_group_id"] == by_record.loc[47, "cv_group_id"]


@requires_legacy_inputs
def test_legacy_property_anomalies_are_explicit_and_not_promoted() -> None:
    records, _, _, _ = _parsed()
    registry, issues = build_property_candidate_registry(PROPERTIES, records)
    by_id = registry.set_index("record_id")
    assert len(registry) == 97
    assert by_id.loc[35, "composition_crosscheck_status"] == "known_1000x_total_volume_conflict"
    assert by_id.loc[35, "legacy_to_raw_total_volume_ratio"] == 1000.0
    assert by_id.loc[49, "composition_crosscheck_status"] == "missing_from_composition_sheet"
    assert registry["source_status"].eq("legacy_unverified").all()
    assert registry["candidate_use_status"].eq("not_model_ready").all()
    assert set(issues["record_id"].astype(int)) == {35, 49}


@requires_legacy_inputs
def test_curve_registry_maps_98_curves_to_canonical_records() -> None:
    records, _, _, _ = _parsed()
    registry, points = build_curve_registry(REFIT_RUN / "csv", records)
    assert len(registry) == 98
    assert registry["curve_id"].is_unique
    assert set(registry["record_id"]) == set(records["record_id"])
    assert len(points) > 0
    assert points["record_id"].notna().all()
    assert registry["ir_correction_status"].eq("unknown").all()


@requires_legacy_inputs
def test_full_writer_emits_qa_and_dependency_honest_point_file(tmp_path: Path) -> None:
    paths = curate_canonical_registry(
        raw_workbook=RAW,
        property_workbook=PROPERTIES,
        refit_run=REFIT_RUN,
        output_dir=tmp_path,
        legacy_metadata_table=LEGACY,
        require_parquet=False,
    )
    for required in (
        "records",
        "sources",
        "curation_log",
        "curve_registry",
        "curve_points",
        "property_registry",
        "curation_qa",
        "lab_metadata_review_queue",
        "curation_manifest",
    ):
        assert paths[required].exists()
    qa = json.loads(paths["curation_qa"].read_text(encoding="utf-8"))
    assert qa["status"] == "pass"
    assert qa["record_count"] == 97
    assert qa["curve_count"] == 98
    assert qa["known_id35_property_conflict_flagged"]
    assert qa["known_id49_property_gap_flagged"]
    assert qa["checks"]["four_level_header_signature_validated"]
    assert not qa["lab_held_out_validation_ready"]
    assert paths["curve_points"].suffix in {".parquet", ".gz"}
