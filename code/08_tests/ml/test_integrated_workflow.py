from __future__ import annotations

import json

import pandas as pd

from ink_tafel_study.config import load_config
from ink_tafel_study.integrated_workflow import (
    PRIMARY_TARGET,
    build_integrated_curation,
    build_integrated_targets,
)


def test_data0718_contract_tracks_registered_named_ppc_state(tmp_path) -> None:
    config = load_config()
    curated = build_integrated_curation(
        dataset_dir=config.resolve_path("canonical_dataset", must_exist=True),
        output_dir=tmp_path / "data",
        panel_names=tuple(config.value("feature_panels.enabled")),
    )
    records = pd.read_csv(curated["records"])
    curation_qa = json.loads(curated["curation_qa"].read_text())

    assert len(records) == records["record_id"].nunique() == 97
    assert records["sample_record"].astype(int).tolist() == list(range(1, 98))
    assert records["curve_id"].nunique() == 97
    assert "sr050" not in set(records["curve_id"])
    assert records["ir_correction_status"].value_counts().to_dict() == {
        "not_corrected": 59,
        "corrected": 38,
    }
    record_28 = records.loc[records["record_id"].eq(28)].iloc[0]
    assert record_28["ir_correction_status"] == "not_corrected"
    assert record_28["ir_reporting_evidence_state"] == "unverifiable"
    assert record_28["ir_status_raw"] == "Unverifiable (full text inaccessible)"
    assert (
        records[
            "solvent_mixture_property_proxy_dielectric_constant_estimated_flag"
        ].sum()
        == 31
    )
    assert curation_qa["target_binding_ready"] is True
    assert curation_qa["formal_training_gate_source"] == "targets/target_qa.json"

    schema = pd.read_csv(curated["panel_schema"])
    assert not schema["status"].eq("missing").any()
    selected = set(schema["selected_column"].dropna())
    assert "ir_correction_status" in selected
    assert "catalyst_loading_mg_cm2_selected" in selected
    assert (
        "solvent_mixture_property_proxy_dielectric_constant_best_available"
        in selected
    )
    assert (
        "solvent_mixture_property_proxy_dielectric_constant_estimated_flag"
        in selected
    )

    assert PRIMARY_TARGET == "b_bayesian_local_line_mV_dec"
    built = build_integrated_targets(
        dataset_dir=config.resolve_path("canonical_dataset", must_exist=True),
        records_path=curated["records"],
        ppc_review_path=config.resolve_path("ppc_review_registry", must_exist=True),
        output_dir=tmp_path / "targets",
        expected_primary_eligible=config.value("targets.expected_primary_eligible"),
        expected_primary_alkaline_eligible=config.value(
            "targets.expected_primary_alkaline_eligible"
        ),
    )
    target_qa = json.loads(built["target_qa"].read_text())
    assert target_qa["formal_evaluation_ready"] is True
    assert target_qa["n_named_ppc_review_complete"] == 97
    assert target_qa["n_formal_ppc_approved"] == 97
    assert target_qa["only_remaining_formal_gate"] is None
    assert target_qa["target_eligible_counts"] == {
        "b_bayesian_local_line_mV_dec": 97,
    }
