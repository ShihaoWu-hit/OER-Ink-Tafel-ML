import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.current_support import (
    build_current_support_artifacts,
    derive_current_support_manifest,
)


def _support_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_summary = pd.DataFrame(
        [
            {
                "curve_id": "sr001",
                "sample_record": 1,
                "fit_status": "success",
                "fit_eligible": True,
                "selected_model": "paper_single",
                "potential_dependent_supported": False,
                "current_unit_raw": "unknown_current_density",
                "transition_width_V": 0.01,
            },
            {
                "curve_id": "sr002",
                "sample_record": 2,
                "fit_status": "success",
                "fit_eligible": True,
                "selected_model": "two_regime_plateau",
                "potential_dependent_supported": True,
                "current_unit_raw": "unknown_current_density",
                "transition_width_V": 0.02,
            },
            {
                "curve_id": "sr003",
                "sample_record": 3,
                "fit_status": "success",
                "fit_eligible": True,
                "selected_model": "paper_single",
                "potential_dependent_supported": False,
                "current_unit_raw": "A g^-1_Ir",
                "transition_width_V": 0.01,
            },
            {
                "curve_id": "sr004_v1",
                "sample_record": 4,
                "fit_status": "success",
                "fit_eligible": True,
                "selected_model": "paper_single",
                "potential_dependent_supported": False,
                "current_unit_raw": "unknown_current_density",
                "transition_width_V": 0.01,
            },
            {
                "curve_id": "sr004_v2",
                "sample_record": 4,
                "fit_status": "success",
                "fit_eligible": True,
                "selected_model": "single_background",
                "potential_dependent_supported": False,
                "current_unit_raw": "unknown_current_density",
                "transition_width_V": 0.01,
            },
        ]
    )
    sample_candidates = pd.DataFrame(
        [
            {
                "sample_record": 1,
                "n_curves": 1,
                "refit_candidate_status": "eligible_single_curve",
            },
            {
                "sample_record": 2,
                "n_curves": 1,
                "refit_candidate_status": "potential_dependent_no_single_scalar",
            },
            {
                "sample_record": 3,
                "n_curves": 1,
                "refit_candidate_status": "eligible_single_curve",
            },
            {
                "sample_record": 4,
                "n_curves": 2,
                "refit_candidate_status": "multiple_curves_disagree",
            },
        ]
    )
    binned_points = pd.DataFrame(
        [
            {
                "curve_id": curve_id,
                "potential_V": 1.2 + 0.1 * point_index,
                "relative_potential_V": 0.1 * point_index,
                "log10_current": log_current,
                "n_raw_points": point_index + 1,
            }
            for curve_id in fit_summary["curve_id"]
            for point_index, log_current in enumerate((-1.0, 0.0, 1.0))
        ]
    )
    return fit_summary, sample_candidates, binned_points


def test_current_support_classifies_registered_cohorts_and_units() -> None:
    fit_summary, sample_candidates, binned_points = _support_inputs()

    manifest = derive_current_support_manifest(
        fit_summary, sample_candidates, binned_points
    ).set_index("curve_id")

    reliable = manifest.loc["sr001"]
    assert reliable["cohort"] == "reliable_single_regime"
    assert bool(reliable["included_primary"])
    assert reliable["exclusion_reason"] == "included_primary"
    assert reliable["current_header_raw"] == ""
    assert reliable["current_unit_interpreted"] == "mA cm^-2"
    assert reliable["current_min_source_numeric"] == pytest.approx(0.1)
    assert reliable["current_max_source_numeric"] == pytest.approx(10.0)
    assert reliable["full_log_current_span_dec"] == pytest.approx(2.0)
    assert reliable["n_effective_bins_recomputed"] == 3
    assert reliable["n_raw_points_in_bins"] == 6

    potential_dependent = manifest.loc["sr002"]
    assert potential_dependent["cohort"] == "potential_dependent"
    assert bool(potential_dependent["included_primary"])

    incompatible = manifest.loc["sr003"]
    assert incompatible["cohort"] == "reliable_single_regime"
    assert bool(incompatible["model_supported"])
    assert not bool(incompatible["unit_in_scope"])
    assert not bool(incompatible["included_primary"])
    assert incompatible["exclusion_reason"] == "incompatible_current_unit"

    for curve_id in ("sr004_v1", "sr004_v2"):
        multiple = manifest.loc[curve_id]
        assert multiple["cohort"] == "variant_resolution_required"
        assert not bool(multiple["cohort_in_scope"])
        assert not bool(multiple["included_primary"])
        assert (
            multiple["exclusion_reason"]
            == "multiple_curve_variants_require_resolution"
        )

    # Target construction derives these columns from binned points. Keeping them
    # out of the manifest prevents pandas from creating unusable _x/_y columns.
    assert "potential_min_V" not in manifest.columns
    assert "potential_max_V" not in manifest.columns
    point_spans = (
        binned_points.groupby("curve_id", as_index=False)["relative_potential_V"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "potential_min_V", "max": "potential_max_V"})
    )
    merged = manifest.reset_index().merge(
        point_spans, on="curve_id", how="left", validate="one_to_one"
    )
    assert np.isfinite(merged["potential_min_V"]).all()
    assert np.isfinite(merged["potential_max_V"]).all()
    assert not any(column.endswith(("_x", "_y")) for column in merged.columns)


def test_build_artifacts_uses_curve_manifest_header_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    fit_summary, sample_candidates, binned_points = _support_inputs()
    refit_run = tmp_path / "refit"
    csv_dir = refit_run / "csv"
    csv_dir.mkdir(parents=True)
    fit_summary.to_csv(csv_dir / "curve_fit_summary.csv", index=False)
    sample_candidates.to_csv(csv_dir / "sample_refit_candidates.csv", index=False)
    binned_points.to_csv(csv_dir / "binned_curve_points.csv", index=False)
    pd.DataFrame(
        {
            "curve_id": fit_summary["curve_id"],
            "current_header_raw": ["Current density"] * len(fit_summary),
        }
    ).to_csv(csv_dir / "curve_manifest.csv", index=False)
    legacy_grid = tmp_path / "legacy_grid.csv"
    legacy_grid.write_text("target,value\nlegacy,1\n", encoding="utf-8")
    output_dir = tmp_path / "current_support"

    paths = build_current_support_artifacts(
        refit_run=refit_run,
        legacy_grid_summary=legacy_grid,
        output_dir=output_dir,
    )

    manifest = pd.read_csv(paths["manifest"])
    assert set(manifest["current_header_raw"]) == {"Current density"}
    assert int(manifest["included_primary"].sum()) == 2
    assert paths["legacy_grid_summary"].read_bytes() == legacy_grid.read_bytes()
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    assert status["rows"] == 5
    assert status["included_primary"] == 2
    assert "curve_manifest" in status["inputs_sha256"]

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_current_support_artifacts(
            refit_run=refit_run,
            legacy_grid_summary=legacy_grid,
            output_dir=output_dir,
        )
