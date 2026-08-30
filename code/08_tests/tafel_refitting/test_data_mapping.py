from __future__ import annotations

from pathlib import Path

import pytest

from tafel_refit.data import load_inputs
from tafel_refit.preprocess import preprocess_curve


WORKSPACE = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not (WORKSPACE / "Data/01_raw/tafel_refitting/tafel_curve_source.xlsx").exists(),
    reason="local refitting workbook is intentionally not part of the public repository",
)
def test_audited_workbook_mapping_and_qc_counts() -> None:
    bundle = load_inputs(
        WORKSPACE / "Data/01_raw/tafel_refitting/tafel_curve_source.xlsx",
        WORKSPACE / "Data/01_raw/ink_provenance.csv",
        WORKSPACE / "Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv",
        WORKSPACE / "code/Tafel_refitting/config/default.json",
    )
    assert len(bundle.manifest) == 97
    assert bundle.manifest["sample_record"].nunique() == 97
    assert bundle.manifest.loc[bundle.manifest["sample_record"].eq(42), "curve_id"].tolist() == ["sr042_v1"]
    assert bundle.manifest.loc[bundle.manifest["sample_record"].eq(43), "curve_id"].tolist() == ["sr042_v2"]
    assert int(bundle.manifest["n_finite_pairs"].sum()) == 11415
    assert int(bundle.manifest["n_nonfinite_pairs"].sum()) == 15
    assert int((bundle.manifest["n_zero_current"] + bundle.manifest["n_negative_current"]).sum()) == 376
    sr080 = bundle.manifest.loc[bundle.manifest["curve_id"].eq("sr080")].iloc[0]
    assert int(sr080["n_zero_current"]) == 7
    assert float(sr080["current_min_raw"]) == 0.0

    processed = []
    for _, manifest_row in bundle.manifest.iterrows():
        points = bundle.raw_points.loc[
            bundle.raw_points["curve_id"].eq(manifest_row["curve_id"])
        ]
        processed.append(preprocess_curve(manifest_row, points, bundle.config))
    eligible = {curve.curve_id for curve in processed if curve.fit_eligible}
    excluded = {curve.curve_id for curve in processed if not curve.fit_eligible}
    assert len(eligible) == 97
    assert excluded == set()
    assert {"sr013", "sr029", "sr068", "sr073"}.issubset(eligible)

    by_id = {curve.curve_id: curve for curve in processed}
    assert all(
        by_id[curve_id].metrics["nonpositive_confined_to_pre_reliable_region"]
        for curve_id in ("sr013", "sr029", "sr068", "sr073")
    )
    assert by_id["sr060"].metrics["n_branch_points"] == 160
    assert by_id["sr060"].metrics["spearman_logj_vs_potential"] > 0.99
    assert all(
        by_id[curve_id].metrics["n_pre_reliable_positive_points"] > 0
        for curve_id in ("sr006", "sr036", "sr038", "sr085", "sr091")
    )
    assert "positive_region_monotonicity_manual_review" in by_id["sr083"].qc_flags
