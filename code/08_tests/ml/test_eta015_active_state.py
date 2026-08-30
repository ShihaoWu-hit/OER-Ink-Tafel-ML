from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_active_canonical_uses_calibrated_bayesian_local_line_mixture() -> None:
    audit = pd.read_csv(REPO_ROOT / "Data/02_inputs/canonical_dataset_ir_corrected_v2/bayesian_audit.csv")
    assert len(audit) == audit["curve_id"].nunique() == 97
    assert audit["policy_id"].eq(
        "bayesian_local_line_eta015_strict_then_pava_fallback_v2"
    ).all()
    assert audit["sampler_diagnostics_pass"].astype(bool).all()
    assert audit["calibration_pass"].astype(bool).all()
    assert audit["selection_uncertainty_propagated"].astype(bool).all()
    assert audit["target_structurally_eligible"].astype(bool).all()
    assert (
        audit.loc[
            audit["sample_record"].eq(26), "selection_success_fraction"
        ].iloc[0]
        >= 0.8
    )
    assert (audit["posterior_target_median_mV_dec"] < 500).all()
    assert set(
        audit.loc[
            audit["quantization_aware_fallback_used"].astype(bool),
            "sample_record",
        ].astype(int)
    ) == {23, 74}

    targets = pd.read_csv(REPO_ROOT / "Data/02_inputs/canonical_dataset_ir_corrected_v2/target_registry.csv")
    primary = targets.loc[
        targets["target_name"].eq("b_bayesian_local_line_mV_dec")
    ]
    assert len(primary) == primary["record_id"].nunique() == 97
    assert len(targets) == 97
    assert primary["target_eligible"].astype(bool).all()
    assert np.allclose(primary["target_fit_eta_min_V"], 0.15)
    assert (primary["target_fit_retained_end_eta_V"] >= primary["target_fit_retained_start_eta_V"]).all()
    assert not targets["target_name"].eq("b_refit_at_2mA_mV_dec").any()
    assert not (REPO_ROOT / "Data/02_inputs/canonical_dataset_ir_corrected_v2/posterior_draws_at_2mA.csv.gz").exists()


def test_bayesian_local_line_review_is_complete_and_only_active_review() -> None:
    results_root = REPO_ROOT / "Data" / "04_results" / "RESULTS"
    review_dirs = sorted(path.name for path in results_root.glob("01_PPC_REVIEW*"))
    assert review_dirs == ["01_PPC_REVIEW"]

    review_root = results_root / "01_PPC_REVIEW"
    review = pd.read_csv(review_root / "ppc_review.csv")
    assert len(review) == review["curve_id"].nunique() == 97
    assert review["ppc_approved"].astype(bool).all()
    assert review["ppc_reviewer"].eq("Mo").all()
    assert review["ppc_review_date"].astype(str).eq("2026-07-23").all()
    assert review["ppc_review_note"].notna().all()
    assert review["evidence_id"].str.contains(
        "bayesian_local_line_eta015_strict_then_pava_fallback_v2", regex=False
    ).all()
    assert len(list(review_root.glob("*.png"))) == 97

    qa = json.loads(
        (REPO_ROOT / "Data/03_outputs/study/outputs/current_97/targets/target_qa.json").read_text()
    )
    assert qa["n_named_ppc_review_complete"] == 97
    assert qa["n_formal_ppc_approved"] == 97
    assert qa["formal_evaluation_ready"] is True
    assert qa["formal_evaluation_blockers"] == []
    assert qa["primary_target"] == "b_bayesian_local_line_mV_dec"


def test_prior_target_and_downstream_results_are_archived() -> None:
    archive = (
        REPO_ROOT
        / "archive/target_state_pre_bayesian_local_line_fallback_20260723"
    )
    assert (archive / "README.md").is_file()
    assert (archive / "archive_manifest.json").is_file()
    assert (archive / "active_state/Data/02_inputs/canonical_dataset_ir_corrected_v2").is_dir()
    assert (archive / "active_state/Data/04_results/RESULTS/01_PPC_REVIEW").is_dir()
    assert (
        archive
        / "candidate_target_runs/20260723_bayesian_local_line_formal_target_v1"
    ).is_dir()
