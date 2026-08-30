from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ink_tafel_study.posterior_cardinal_contiguous_threeclass_direct_gower import (
    CONTRACT_ID,
    CONTRACT_SHA256,
    DIRECT_MODEL,
    NEIGHBOR_COUNTS,
    fit_predict_direct_multiscale_gower,
    load_direct_gower_contract,
    run_direct_multiscale_gower_sensitivity,
)
from ink_tafel_study.provenance import (
    sha256_file,
    verify_artifact_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    REPOSITORY_ROOT
    / "RESULTS"
    / "05_WORKBOOKS"
    / "posterior_cardinal_contiguous_threeclass_v2_20260725"
)


def test_direct_gower_contract_is_source_hash_bound() -> None:
    contract, path, digest = load_direct_gower_contract()

    assert contract["contract_id"] == CONTRACT_ID
    assert digest == CONTRACT_SHA256
    assert sha256_file(path) == CONTRACT_SHA256
    assert contract["model"]["gower"]["n_neighbors"] == list(
        NEIGHBOR_COUNTS
    )
    assert contract["model"]["optuna"]["enabled"] is False


def test_direct_multiscale_prediction_is_exact_component_mean() -> None:
    records = pd.read_csv(PARENT / "inputs" / "records.csv")
    labels = pd.read_csv(
        PARENT / "registry" / "contiguous_three_class_soft_labels.csv"
    )
    train = records.iloc[:72].reset_index(drop=True)
    test = records.iloc[72:82].reset_index(drop=True)
    targets = labels.iloc[:72][
        ["q_low_60_side", "q_120_side", "q_high_off"]
    ].to_numpy(float)

    prediction, components, audit = fit_predict_direct_multiscale_gower(
        train,
        targets,
        test,
    )
    expected = np.mean(
        np.stack(
            [components[count] for count in NEIGHBOR_COUNTS],
            axis=0,
        ),
        axis=0,
    )

    assert tuple(components) == NEIGHBOR_COUNTS
    assert np.allclose(prediction, expected)
    assert np.allclose(prediction.sum(axis=1), 1.0)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()
    assert audit["posterior_draws_expanded"] is False
    assert audit["class_rows_expanded"] is False
    assert audit["posterior_argmax_used"] is False
    assert audit["outer_test_used_for_selection"] is False


def test_direct_gower_smoke_is_disjoint_and_audited(
    tmp_path: Path,
) -> None:
    parent_before = verify_artifact_manifest(PARENT)
    destination = tmp_path / "direct_gower_smoke"

    result = run_direct_multiscale_gower_sensitivity(
        destination,
        parent_run=PARENT,
        n_jobs=2,
        smoke=True,
    )

    qa = json.loads(
        (destination / "analyses" / "analysis_qa.json").read_text(
            encoding="utf-8"
        )
    )
    oof = pd.read_csv(destination / "analyses" / "oof_probabilities.csv")
    components = pd.read_csv(
        destination
        / "analyses"
        / "gower_component_oof_probabilities.csv"
    )
    parent_after = verify_artifact_manifest(PARENT)

    assert result["status"] == "complete_smoke"
    assert qa["status"] == "complete_smoke"
    assert qa["scientifically_releasable"] is False
    assert qa["outer_test_used_for_model_or_feature_selection"] is False
    assert qa["posterior_argmax_used_for_training_or_selection"] is False
    assert qa["optuna_used"] is False
    assert qa["parent_tree_unchanged"] is True
    assert set(oof["evaluation_model"]) == {
        DIRECT_MODEL,
        "fold_local_contiguous_threeclass_soft_prior",
        "stagewise_inner_oof_convex_stack",
    }
    assert len(components) == (
        oof.loc[oof["evaluation_model"].eq(DIRECT_MODEL)].shape[0]
        * len(NEIGHBOR_COUNTS)
    )
    assert (
        verify_artifact_manifest(destination)["status"] == "pass"
    )
    assert parent_before["actual_tree_sha256"] == (
        parent_after["actual_tree_sha256"]
    )
