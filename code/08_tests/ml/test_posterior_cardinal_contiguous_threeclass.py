from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ink_tafel_study.cli as cli
from ink_tafel_study.config import StudyConfig, load_config
from ink_tafel_study.posterior_cardinal_classification import IDENTITY_COLUMNS
from ink_tafel_study.posterior_cardinal_contiguous_threeclass import (
    CLASS_LABELS,
    MODEL_FAMILIES,
    P_COLUMNS,
    PRIOR_MODEL,
    Q_COLUMNS,
    STACK_MODEL,
    _default_enqueued_candidates,
    _load_contract,
    _primary_frame,
    build_contiguous_inner_manifest,
    build_contiguous_outer_manifest,
    run_posterior_cardinal_contiguous_threeclass,
)
from ink_tafel_study.contiguous_threeclass_labels import (
    derive_contiguous_threeclass_soft_labels,
)


def test_contiguous_threeclass_contract_is_config_hash_bound() -> None:
    config = load_config()
    contract, path, digest = _load_contract(config)

    assert contract["contract_id"] == (
        "posterior_cardinal_contiguous_threeclass_v2"
    )
    assert path.name == "posterior_cardinal_contiguous_threeclass_v2.yaml"
    assert digest == config.value(
        "posterior_cardinal_contiguous_threeclass.contract_sha256"
    )

    raw = deepcopy(config.raw)
    raw["posterior_cardinal_contiguous_threeclass"][
        "contract_sha256"
    ] = "0" * 64
    drifted = StudyConfig(
        path=config.path,
        project_root=config.project_root,
        repository_root=config.repository_root,
        raw=raw,
    )
    with pytest.raises(ValueError, match="contract hash mismatch"):
        _load_contract(drifted)


def test_cli_accepts_contiguous_threeclass_stage() -> None:
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-contiguous-threeclass",
            "--run-dir",
            "Data/04_results/RESULTS/05_WORKBOOKS/contiguous_threeclass_v2",
            "--source-run",
            "Data/03_outputs/study/outputs/current_97",
            "--n-jobs",
            "4",
            "--optuna-trials",
            "4",
            "--smoke",
        ]
    )

    assert args.stage == "posterior-cardinal-contiguous-threeclass"
    assert args.n_jobs == 4
    assert args.optuna_trials == 4


def test_full_defaults_guarantee_gower_kernel_and_block_coverage() -> None:
    candidates = _default_enqueued_candidates(smoke=False)

    assert len(candidates) == (
        len(MODEL_FAMILIES) * 7 + 4
    )
    structural = candidates[-4:]
    assert {row["neighbor_mode"] for row in structural} == {
        "exponential_kernel",
        "rbf_kernel",
    }
    assert {
        row.get("block_weight_profile")
        for row in structural
        if row["panel"] != "context_core"
    } == {"context_2x", "formulation_2x"}


def test_cli_rejects_output_inside_classic_parent_before_writing() -> None:
    config = load_config()
    classic_root = (
        config.repository_root
        / "RESULTS"
        / "05_WORKBOOKS"
        / "posterior_cardinal_threeclass_optuna_v1_full_20260725"
    )
    forbidden = classic_root / "__forbidden_contiguous_cli_child__"
    assert not forbidden.exists()
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-contiguous-threeclass",
            "--run-dir",
            str(forbidden),
            "--smoke",
        ]
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        cli._resolve_run(config, args)

    assert not forbidden.exists()


def test_frozen_real_input_contiguous_threeclass_smoke(
    tmp_path: Path,
) -> None:
    config = load_config()
    destination = tmp_path / "contiguous_threeclass_smoke"

    result = run_posterior_cardinal_contiguous_threeclass(
        config,
        destination,
        source_run=config.resolve_path("active_source_run"),
        smoke=True,
        n_jobs=1,
        optuna_trials=4,
    )

    qa = json.loads(
        (destination / "analyses" / "analysis_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "complete_smoke"
    assert qa["status"] == "complete_smoke"
    assert qa["scientifically_releasable"] is False
    assert qa["formal_v4_result_modified"] is False
    assert qa["classic_threeclass_v1_result_modified"] is False
    assert qa["n_records"] == 97
    assert qa["n_connected_groups"] == 80
    assert qa["n_posterior_draws"] == 776_000
    assert qa["draws_per_record"] == 8_000
    assert qa["class_order"] == list(CLASS_LABELS)
    assert qa["modal_record_counts"] == {
        "low_60_side": 64,
        "120_side": 25,
        "high_off": 8,
    }
    assert qa["total_nonpositive_finite_draws"] == 1_760
    assert np.isclose(
        qa["posterior_expected_oracle"],
        0.8579329896907216,
    )
    assert qa["outer_repeats"] == 1
    assert qa["outer_folds"] == 2
    assert qa["inner_folds"] == 2
    assert qa["n_optuna_trials_total"] == 16
    assert qa["n_optuna_trials_complete"] == 16
    assert qa["gate_and_conditional_studies_independent"] is True
    assert qa["prior_shrinkage_selected_on_grouped_inner_oof_only"] is True
    assert qa["stage_stack_weights_selected_on_grouped_inner_oof_only"] is True
    assert qa["outer_test_used_for_selection_shrinkage_or_stacking"] is False
    assert qa["posterior_argmax_used_for_training_or_selection"] is False
    assert qa["formal_source_run_unchanged"] is True
    assert all(qa["immutable_parent_runs_unchanged"].values())
    assert qa["immutable_parent_artifact_manifests_verified"] is True
    assert qa["classic_v1_frozen_fold_manifests_verified"] is True

    labels = pd.read_csv(
        destination
        / "registry"
        / "contiguous_three_class_soft_labels.csv"
    )
    assert len(labels) == 97
    assert np.allclose(labels.loc[:, Q_COLUMNS].sum(axis=1), 1.0)
    assert np.allclose(
        labels.loc[:, Q_COLUMNS].sum(axis=0),
        [62.221625, 25.73475, 9.043625],
    )

    trials = pd.read_csv(destination / "analyses" / "optuna_trials.csv")
    assert trials["state"].eq("complete").all()
    assert set(trials["stage"]) == {"gate", "conditional"}
    assert set(trials["model_family"]) == set(MODEL_FAMILIES)
    assert (
        trials.groupby(["outer_repeat", "outer_fold", "stage"])[
            "model_family"
        ]
        .nunique()
        .eq(len(MODEL_FAMILIES))
        .all()
    )

    alpha = pd.read_csv(
        destination / "analyses" / "stage_alpha_selections.csv"
    )
    assert len(alpha) == 2 * 2 * len(MODEL_FAMILIES)
    assert alpha["alpha"].between(0, 1, inclusive="both").all()
    weights = pd.read_csv(
        destination / "analyses" / "stage_stack_weights.csv"
    )
    assert set(weights["candidate"]) == {"prior", *MODEL_FAMILIES}
    assert np.allclose(
        weights.groupby(
            ["outer_repeat", "outer_fold", "stage"]
        )["weight"].sum(),
        1.0,
    )
    selection_summary = pd.read_csv(
        destination / "analyses" / "stage_selection_summary.csv"
    )
    assert set(selection_summary["stage"]) == {"gate", "conditional"}
    assert set(selection_summary["model_family"]) == set(MODEL_FAMILIES)
    assert selection_summary[
        "selection_fraction_within_family"
    ].between(0, 1, inclusive="both").all()

    selections = pd.read_csv(
        destination / "analyses" / "outer_selections.csv"
    )
    assert not selections["outer_test_used_for_selection"].astype(bool).any()
    assert {
        "gate_selected_configs_json",
        "conditional_selected_configs_json",
        "gate_stack_weights_json",
        "conditional_stack_weights_json",
    }.issubset(selections.columns)

    oof = pd.read_csv(destination / "analyses" / "oof_probabilities.csv")
    assert len(oof) == 2 * 97
    assert set(oof["evaluation_model"]) == {PRIOR_MODEL, STACK_MODEL}
    assert np.isfinite(oof.loc[:, P_COLUMNS].to_numpy(float)).all()
    assert (oof.loc[:, P_COLUMNS].to_numpy(float) >= 0).all()
    assert np.allclose(oof.loc[:, P_COLUMNS].sum(axis=1), 1.0)

    contract, _, _ = _load_contract(config)
    records = pd.read_csv(destination / "inputs" / "records.csv")
    draws = pd.read_csv(
        destination / "inputs" / "posterior_slope_draws.csv.gz"
    )
    full_labels = derive_contiguous_threeclass_soft_labels(
        draws, records.loc[:, IDENTITY_COLUMNS]
    )
    full_frame = _primary_frame(records, full_labels)
    full_outer = build_contiguous_outer_manifest(
        full_frame, contract, smoke=False
    )
    full_inner = build_contiguous_inner_manifest(
        full_frame, full_outer, contract, smoke=False
    )
    assert hashlib.sha256(
        full_outer.to_csv(index=False).encode()
    ).hexdigest() == contract["outer_validation"][
        "expected_full_manifest_sha256"
    ]
    assert hashlib.sha256(
        full_inner.to_csv(index=False).encode()
    ).hexdigest() == contract["inner_validation"][
        "expected_full_manifest_sha256"
    ]

    required = (
        "registry/contract_snapshot.yaml",
        "registry/contiguous_three_class_support.csv",
        "analyses/stage_component_oof_predictions.csv",
        "analyses/stage_alpha_selections.csv",
        "analyses/stage_selection_summary.csv",
        "analyses/stage_stack_weights.csv",
        "analyses/model_fit_registry.csv",
        "analyses/aggregate_metrics.csv",
        "analyses/connected_group_bootstrap_summary.csv",
        "provenance/contract_binding.json",
        "provenance/fold_manifest_binding.json",
        "provenance/parent_contract_bindings.json",
        "report/posterior_cardinal_contiguous_threeclass_report.md",
        "report/validation_report.json",
    )
    assert all((destination / relative).is_file() for relative in required)


def test_contiguous_full_run_rejects_trial_budget_deviation(
    tmp_path: Path,
) -> None:
    config = load_config()
    with pytest.raises(
        ValueError,
        match="hash-bound 36 Optuna trials per stage",
    ):
        run_posterior_cardinal_contiguous_threeclass(
            config,
            tmp_path / "full_budget_deviation",
            source_run=config.resolve_path("active_source_run"),
            smoke=False,
            optuna_trials=28,
        )
