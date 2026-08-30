from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ink_tafel_study.posterior_cardinal_contiguous_threeclass_presto_gate import (
    CONTRACT_SHA256,
    DIRECT_POLICY,
    PRESTO_POLICY,
    _decision_metrics,
    _hybrid_probabilities,
    _threshold_table,
    load_presto_gate_contract,
    run_presto_gate_sensitivity,
)
from ink_tafel_study.provenance import verify_artifact_manifest


def test_presto_contract_hash_and_identity() -> None:
    contract, path, digest = load_presto_gate_contract()
    assert digest == CONTRACT_SHA256
    assert contract["model"]["exact_presto_replication"] is False
    assert contract["high_recall_policy"][
        "baseline_receives_identical_protocol"
    ] is True
    assert path.is_file()


def test_hybrid_preserves_presto_high_and_gower_conditional() -> None:
    presto = np.asarray(
        [[0.6, 0.25, 0.15], [0.3, 0.4, 0.3]],
        dtype=float,
    )
    direct = np.asarray(
        [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]],
        dtype=float,
    )
    hybrid = _hybrid_probabilities(presto, direct)
    assert np.allclose(hybrid[:, 2], presto[:, 2], atol=0, rtol=0)
    assert np.allclose(
        hybrid[:, :2] / hybrid[:, :2].sum(axis=1, keepdims=True),
        direct[:, :2] / direct[:, :2].sum(axis=1, keepdims=True),
        atol=1.0e-15,
        rtol=0,
    )
    assert np.allclose(hybrid.sum(axis=1), 1.0)


def test_soft_threshold_protocol_and_hard_special_case() -> None:
    q_high = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=float)
    p_high = np.asarray([0.02, 0.08, 0.12, 0.30], dtype=float)
    folds = np.asarray([0, 0, 1, 1], dtype=int)
    table, selected = _threshold_table(
        q_high,
        p_high,
        folds,
        model=PRESTO_POLICY,
    )
    chosen = table.loc[table["selected"]].iloc[0]
    assert float(chosen["soft_high_false_positive_rate"]) <= 0.20
    threshold = float(selected["threshold"])
    decision = np.where(p_high >= threshold, 2, 0)
    q = np.column_stack([1.0 - q_high, np.zeros(4), q_high])
    metrics = _decision_metrics(q, decision)
    assert np.isclose(
        metrics["modal_high_recall"],
        chosen["soft_high_recall"],
    )
    assert np.isclose(
        metrics["modal_high_precision"],
        chosen["soft_high_precision"],
    )


def test_presto_smoke_artifact_is_self_consistent(tmp_path: Path) -> None:
    destination = tmp_path / "presto-smoke"
    result = run_presto_gate_sensitivity(
        destination,
        n_jobs=2,
        smoke=True,
    )
    assert result["status"] == "complete_smoke"
    verification = verify_artifact_manifest(destination)
    assert verification["status"] == "pass"
    qa = json.loads(
        (destination / "analyses" / "analysis_qa.json").read_text()
    )
    assert qa["outer_test_used_for_model_penalty_or_threshold_selection"] is False
    assert qa["posterior_draws_expanded"] is False
    policy = pd.read_csv(
        destination / "analyses" / "policy_aggregate_metrics.csv"
    )
    assert set(policy["decision_policy"]) == {
        PRESTO_POLICY,
        DIRECT_POLICY,
    }
