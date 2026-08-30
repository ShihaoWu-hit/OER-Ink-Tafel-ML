from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ink_tafel_study.targets as target_module

from ink_tafel_study.targets import (
    LOW_BRANCH_TARGET,
    PRIMARY_TARGET,
    SINGLE_REGIME_TARGET,
    TargetBuildConfig,
    build_target_registry,
    fixed_current_slope,
    marginal_slope_is_fixed_current_target,
    select_low_potential_branch,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_TARGET_INPUTS = (
    REPO_ROOT / "Data/03_outputs/tafel_refitting/outputs/20260714_040837"
)
requires_legacy_target_inputs = pytest.mark.skipif(
    not LEGACY_TARGET_INPUTS.is_dir(),
    reason="superseded v2 target evidence was intentionally retired in the canonical-97 cleanup",
)


def _target_config(*, curves: int, records: int) -> TargetBuildConfig:
    return TargetBuildConfig(
        expected_curve_count=curves,
        expected_record_count=records,
        expected_primary_eligible=0,
        expected_low_branch_eligible=0,
        expected_single_regime_eligible=0,
    )


def test_structural_target_gate_prefers_current_posterior_and_blocks_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_summary = pd.DataFrame(
        [
            {
                "curve_id": curve_id,
                "sample_record": sample_record,
                "publication_id": "PUB",
                "doi": "10.example/test",
                "selected_model": "linear",
                "ir_status": "unknown",
                "fit_status": "success",
                "fit_eligible": True,
                "potential_dependent_supported": False,
            }
            for curve_id, sample_record in (
                ("sr010", 10),
                ("sr011", 11),
                ("sr042_v1", 42),
                ("sr042_v2", 42),
            )
        ]
    )
    sample_candidates = pd.DataFrame(
        [
            {
                "sample_record": 10,
                "n_curves": 1,
                "curve_ids": "sr010",
                "refit_candidate_status": "eligible_single_curve",
            },
            {
                "sample_record": 11,
                "n_curves": 1,
                "curve_ids": "sr011",
                "refit_candidate_status": "eligible_single_curve",
            },
            {
                "sample_record": 42,
                "n_curves": 2,
                "curve_ids": "sr042_v1;sr042_v2",
                "refit_candidate_status": "multiple_curves_disagree",
            },
        ]
    )
    legacy = pd.DataFrame(
        [
            {
                "curve_ids": "sr011",
                "y_tafel_slope_mV_dec": 90.0,
                "y_tafel_slope_interval_low_mV_dec": 80.0,
                "y_tafel_slope_interval_high_mV_dec": 100.0,
                "bayesian_ppc_approved": True,
            }
        ]
    )
    exact = pd.DataFrame(
        [
            {
                "curve_id": curve_id,
                "cohort": "reliable_single_regime",
                "target_value_mV_dec": 100.0,
                "target_interval_low_mV_dec": 90.0,
                "target_interval_high_mV_dec": 110.0,
                "target_eligible": True,
                "exclusion_reason": "eligible",
                "target_current_density_mA_cm2": 2.0,
                "current_min_observed_mA_cm2": 0.1,
                "current_max_observed_mA_cm2": 10.0,
                "observed_support": True,
                "extrapolated": False,
                "calculation_method": "synthetic",
                "interpolation_used": False,
            }
            for curve_id in fit_summary["curve_id"]
        ]
    )
    audit = pd.DataFrame(
        [
            {
                "curve_id": curve_id,
                "formal_nuts_evidence_present": curve_id != "sr011",
                "named_ppc_review_complete": True,
                "formal_nuts_ppc_approved": True,
                "posterior_slope_mean_mV_dec": 123.0,
                "posterior_slope_interval_low_mV_dec": 115.0,
                "posterior_slope_interval_high_mV_dec": 131.0,
                "evidence_id": f"current-{curve_id}",
                "posterior_summary_path": f"/{curve_id}.csv",
            }
            for curve_id in fit_summary["curve_id"]
        ]
    )
    frames = {
        "fit_summary": fit_summary,
        "sample_candidates": sample_candidates,
        "bayesian_targets": legacy,
    }

    registry = target_module._make_target_registry(
        frames, exact, audit, _target_config(curves=4, records=3)
    )

    promoted = registry.loc[
        registry["curve_id"].eq("sr010")
        & registry["target_name"].isin([LOW_BRANCH_TARGET, SINGLE_REGIME_TARGET])
    ]
    assert promoted["target_structurally_eligible"].all()
    assert promoted["target_eligible"].all()
    assert promoted["posterior_target_supported"].all()
    assert promoted["formal_target_approved"].all()
    assert np.allclose(promoted["target_value_mV_dec"], 123.0)

    historical = registry.loc[
        registry["curve_id"].eq("sr011")
        & registry["target_name"].eq(SINGLE_REGIME_TARGET)
    ].iloc[0]
    assert historical["target_structurally_eligible"]
    assert historical["target_eligible"]
    assert not historical["posterior_target_supported"]
    assert not historical["formal_target_approved"]
    assert historical["ppc_status"] == "historical_unbound_not_formal"
    assert historical["target_uncertainty_source"].startswith("legacy_historical")

    variants = registry.loc[registry["sample_record"].eq(42)]
    assert len(variants) == 6
    assert not variants["sample_candidate_gate_passed"].any()
    assert not variants["target_structurally_eligible"].any()
    assert not variants["target_eligible"].any()
    assert set(variants["sample_candidate_gate_reason"]) == {
        "multiple_curve_variants_require_resolution"
    }

    class _Paths:
        deterministic_config = None
        bayes_dir = tmp_path
        ppc_review_overlay = None

        @staticmethod
        def all_bayes_dirs():
            return ()

    monkeypatch.setattr(
        target_module,
        "discover_bayesian_evidence",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    _, queue, _ = target_module._bayesian_coverage_audit(fit_summary, _Paths())
    assert set(queue.loc[queue["sample_record"].eq(42), "curve_id"]) == {
        "sr042_v1",
        "sr042_v2",
    }


def test_low_potential_branch_is_not_numeric_minimum() -> None:
    selected = select_low_potential_branch(
        {
            "slope_low_point_mV_dec": 530.0,
            "slope_low_ci_low_mV_dec": 400.0,
            "slope_low_ci_high_mV_dec": 650.0,
            "slope_high_point_mV_dec": 80.0,
        }
    )

    assert selected["target_value_mV_dec"] == 530.0
    assert selected["target_value_mV_dec"] != min(530.0, 80.0)
    assert selected["branch_semantics"] == "low_potential_low_current_branch"
    assert selected["branch_selection_rule"] == "model_coordinate_not_numeric_minimum"
    assert selected["numeric_minimum_branch_selected"] is False


def test_fixed_current_requires_observed_support_and_never_extrapolates() -> None:
    parameters = {"a": -1.0, "log_m": float(np.log(10.0))}
    supported = fixed_current_slope(
        model_name="linear",
        parameters=parameters,
        transition_width_V=0.01,
        potential_min_V=0.0,
        potential_max_V=0.5,
        observed_current_min_mA_cm2=0.5,
        observed_current_max_mA_cm2=10.0,
        target_current_mA_cm2=2.0,
    )
    unsupported = fixed_current_slope(
        model_name="linear",
        parameters=parameters,
        transition_width_V=0.01,
        potential_min_V=0.0,
        potential_max_V=0.5,
        observed_current_min_mA_cm2=3.0,
        observed_current_max_mA_cm2=10.0,
        target_current_mA_cm2=2.0,
    )

    assert supported["eligible"] is True
    assert np.isclose(supported["slope_mV_dec"], 100.0)
    assert np.isclose(
        supported["slope_mV_dec"], supported["kinetic_slope_mV_dec"]
    )
    assert np.isclose(
        supported["apparent_slope_mV_dec"], supported["kinetic_slope_mV_dec"]
    )
    assert supported["observed_support"] is True
    assert supported["extrapolated"] is False
    assert unsupported["eligible"] is False
    assert np.isnan(unsupported["slope_mV_dec"])
    assert unsupported["reason"] == "fixed_current_outside_observed_support"
    assert unsupported["extrapolated"] is False


def test_two_regime_fixed_current_is_evaluated_at_exact_coordinate() -> None:
    parameters = {
        "a": -1.0,
        "log_m1": float(np.log(5.0)),
        "log_m2": float(np.log(10.0)),
        "c": 0.2,
        "L": 4.0,
    }
    result = fixed_current_slope(
        model_name="two_regime_plateau",
        parameters=parameters,
        transition_width_V=0.02,
        potential_min_V=0.0,
        potential_max_V=0.5,
        observed_current_min_mA_cm2=0.1,
        observed_current_max_mA_cm2=100.0,
        target_current_mA_cm2=2.0,
        inverse_grid_points=1025,
    )

    assert result["eligible"] is True
    assert result["observed_support"] is True
    assert np.isfinite(result["relative_potential_at_current_V"])
    assert 100.0 < result["slope_mV_dec"] < 200.0
    assert np.isclose(result["slope_mV_dec"], result["kinetic_slope_mV_dec"])
    assert result["apparent_slope_mV_dec"] >= result["kinetic_slope_mV_dec"]
    assert result["extrapolated"] is False


def test_marginal_slope_only_equals_fixed_current_target_for_linear_model() -> None:
    assert marginal_slope_is_fixed_current_target("linear") is True
    assert marginal_slope_is_fixed_current_target("paper_single_plateau") is False
    assert marginal_slope_is_fixed_current_target("single_background_plateau") is False
    assert marginal_slope_is_fixed_current_target("two_regime_plateau") is False


@requires_legacy_target_inputs
def test_frozen_artifact_parity_and_canonical_outputs(tmp_path: Path) -> None:
    refit_run = REPO_ROOT / "Data/03_outputs/tafel_refitting/outputs/20260714_040837"
    knn_root = REPO_ROOT / "KNN_refit_58"
    current_root = REPO_ROOT / "Current_conditioned_Tafel"
    if not knn_root.exists():
        knn_root = REPO_ROOT / "archive/legacy_20260716/workspace/KNN_refit_58"
    if not current_root.exists():
        current_root = REPO_ROOT / "archive/legacy_20260716/workspace/Current_conditioned_Tafel"

    outputs = build_target_registry(
        records_path=None,
        refit_run=refit_run,
        bayesian_table=knn_root / "data/Data_ink_refit_58_bayesian.csv",
        bayesian_manifest=(
            knn_root / "data/Data_ink_refit_58_bayesian_manifest.json"
        ),
        current_conditioned_dir=current_root,
        output_dir=tmp_path,
        config=TargetBuildConfig(
            expected_curve_count=98,
            expected_record_count=97,
            expected_primary_eligible=74,
            expected_low_branch_eligible=79,
            expected_single_regime_eligible=58,
        ),
    )

    assert outputs["target_registry"].name == "target_registry.csv"
    assert outputs["bayesian_audit"].name == "bayesian_audit.csv"
    assert outputs["bayesian_pending_queue"].name == "bayesian_pending_queue.csv"
    assert outputs["posterior_functional_draws"].name == (
        "posterior_functional_draws_at_2mA.csv.gz"
    )
    assert outputs["target_qa"].name == "target_qa.json"

    registry = pd.read_csv(outputs["target_registry"])
    eligible = (
        registry.loc[registry["target_eligible"]]
        .groupby("target_name")
        .size()
        .to_dict()
    )
    assert eligible == {
        PRIMARY_TARGET: 74,
        LOW_BRANCH_TARGET: 79,
        SINGLE_REGIME_TARGET: 58,
    }
    primary = registry.loc[
        registry["target_name"].eq(PRIMARY_TARGET) & registry["target_eligible"]
    ]
    assert np.allclose(primary["target_current_density_mA_cm2"], 2.0)
    assert primary["observed_support"].all()
    assert not primary["extrapolated"].any()
    assert not primary["interpolation_used"].any()
    fixed_draws = pd.read_csv(outputs["fixed_current_draws"])
    assert {
        "local_slope_mV_dec",
        "kinetic_slope_mV_dec",
        "apparent_slope_mV_dec",
    }.issubset(fixed_draws)
    assert np.allclose(
        fixed_draws["local_slope_mV_dec"],
        fixed_draws["kinetic_slope_mV_dec"],
    )
    assert (
        fixed_draws["apparent_slope_mV_dec"] + 1e-9
        >= fixed_draws["kinetic_slope_mV_dec"]
    ).all()

    # Current frozen evidence has no formal functional-posterior sidecar, so
    # the primary registry is the median of kinetic bootstrap draws—not the
    # median apparent derivative.
    kinetic_medians = fixed_draws.groupby("curve_id")[
        "kinetic_slope_mV_dec"
    ].median()
    primary_by_curve = primary.set_index("curve_id")["target_value_mV_dec"]
    common = primary_by_curve.index.intersection(kinetic_medians.index)
    assert len(common) == len(primary_by_curve)
    assert np.allclose(primary_by_curve.loc[common], kinetic_medians.loc[common])

    # sr025 is a real counterexample to the tempting but incorrect "take the
    # smaller number" rule: its low-potential branch is much steeper.
    sr025 = registry.loc[
        registry["target_name"].eq(LOW_BRANCH_TARGET)
        & registry["curve_id"].eq("sr025")
    ].iloc[0]
    fit_sr025 = pd.read_csv(refit_run / "csv/curve_fit_summary.csv").set_index(
        "curve_id"
    ).loc["sr025"]
    assert np.isclose(sr025["target_value_mV_dec"], fit_sr025["slope_low_point_mV_dec"])
    assert sr025["target_value_mV_dec"] > fit_sr025["slope_high_point_mV_dec"]

    audit = pd.read_csv(outputs["bayesian_audit"])
    queue = pd.read_csv(outputs["bayesian_pending_queue"])
    # Historical sidecars remain discoverable, while only evidence bound to
    # the current deterministic artifact/workflow hashes is production-ready.
    assert audit["nuts_diagnostics_present"].sum() >= 58
    assert audit["sampler_diagnostics_pass"].sum() >= 58
    # No historical run contains the new chain/draw-synchronous functional
    # target sidecar, so none can be promoted as current formal evidence.
    assert audit["formal_nuts_evidence_present"].sum() == 0
    assert audit["ppc_approved_declared"].sum() == 58
    assert audit["named_ppc_review_complete"].sum() == 0
    assert len(queue) == 94

    qa = pd.read_json(outputs["target_qa"], typ="series")
    assert qa["formal_evaluation_ready"] is False
    assert any(
        "posterior_target_coverage_incomplete" in blocker
        for blocker in qa["formal_evaluation_blockers"]
    )
