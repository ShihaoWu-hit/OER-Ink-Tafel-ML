from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ink_tafel_study.threeclass_sensitivity_models as sensitivity
from ink_tafel_study.threeclass_sensitivity_models import (
    SENSITIVITY_CLASS_LABELS,
    ConvexOOFProbabilityStacker,
    FoldLocalBinarySoftPriorClassifier,
    TwoStageSoftCatBoostClassifier,
    combine_two_stage_probabilities,
    fit_predict_soft_binary_stage,
    prior_shrink_probabilities,
    validate_probability_matrix,
)


def _source_records() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    return pd.read_csv(
        project_root
        / "outputs"
        / "current_97_ir_compensation_corrected_v2_20260724"
        / "data"
        / "records.csv"
    )


def test_fold_local_binary_soft_prior_is_weighted_and_fold_constant() -> None:
    X_train = np.arange(8, dtype=float).reshape(4, 2)
    target = np.asarray([0.1, 0.3, 0.7, 0.9])
    weights = np.asarray([1.0, 1.0, 2.0, 2.0])
    estimator = FoldLocalBinarySoftPriorClassifier(
        class_names=("class0", "class1")
    ).fit(X_train, target, sample_weight=weights)

    expected = float(np.average(target, weights=weights))
    probabilities = estimator.predict_proba(np.zeros((3, 2)))

    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities[:, 0], expected)
    assert np.allclose(probabilities[:, 1], 1.0 - expected)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert estimator.fit_audit_["fit_scope"] == (
        "supplied_training_rows_only"
    )
    assert estimator.fit_audit_["posterior_draws_expanded"] is False


def test_prior_shrinkage_exactly_reproduces_both_alpha_endpoints() -> None:
    model = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    prior = np.asarray([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2]])

    at_prior, prior_audit = prior_shrink_probabilities(
        model,
        prior,
        0.0,
        return_audit=True,
    )
    at_model, model_audit = prior_shrink_probabilities(
        model,
        prior,
        1.0,
        return_audit=True,
    )
    interior = prior_shrink_probabilities(model, prior, 0.25)

    assert np.array_equal(at_prior, prior)
    assert np.array_equal(at_model, model)
    assert np.allclose(interior, 0.25 * model + 0.75 * prior)
    assert prior_audit["endpoint_behavior"] == "exact_prior"
    assert model_audit["endpoint_behavior"] == "exact_model"
    assert prior_audit["normalization_applied"] is False


def test_probability_validation_fails_closed_instead_of_repairing() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        validate_probability_matrix([[0.2, 0.7]])
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_probability_matrix([[1.1, -0.1]])
    with pytest.raises(ValueError, match="alpha"):
        prior_shrink_probabilities(
            [[0.5, 0.5]],
            [[0.4, 0.6]],
            1.01,
        )


def test_convex_inner_oof_stacker_has_simplex_weights_and_beats_vertices() -> None:
    candidate_a = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    )
    candidate_b = np.asarray(
        [[0.6, 0.4], [0.4, 0.6], [0.7, 0.3], [0.3, 0.7]]
    )
    prior = np.repeat([[0.5, 0.5]], 4, axis=0)
    target = 0.65 * candidate_a + 0.35 * candidate_b
    predictions = {
        "model_b": candidate_b,
        "prior": prior,
        "model_a": candidate_a,
    }

    stacker = ConvexOOFProbabilityStacker().fit(predictions, target)
    stacked = stacker.predict_proba(predictions)

    assert tuple(stacker.candidate_names_) == (
        "model_a",
        "model_b",
        "prior",
    )
    assert np.all(stacker.weights_ >= 0)
    assert np.isclose(stacker.weights_.sum(), 1.0)
    assert np.allclose(stacked.sum(axis=1), 1.0)
    assert stacker.fit_audit_["prior_candidate_present"] is True
    assert "best_single_candidate" in {
        row["start_name"]
        for row in stacker.fit_audit_["solver_runs"]
    }
    assert stacker.fit_audit_["stack_soft_log_loss"] <= (
        stacker.fit_audit_["best_single_soft_log_loss"] + 1.0e-10
    )
    assert stacker.fit_audit_["outer_test_predictions_permitted_for_fit"] is (
        False
    )


def test_convex_stacker_has_deterministic_prior_tie_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_minimize(
        objective: object,
        start: np.ndarray,
        **kwargs: object,
    ) -> SimpleNamespace:
        del objective, kwargs
        return SimpleNamespace(
            success=False,
            status=9,
            message="forced failure",
            nit=0,
            x=np.asarray(start, dtype=float),
        )

    monkeypatch.setattr(sensitivity, "minimize", failed_minimize)
    identical = np.repeat([[0.7, 0.3]], 3, axis=0)
    predictions = {"z_model": identical, "prior": identical}
    target = identical.copy()

    stacker = ConvexOOFProbabilityStacker().fit(predictions, target)

    assert stacker.fit_audit_["fallback_used"] is True
    assert stacker.fit_audit_["fallback_reason"] == (
        "no_feasible_optimizer_solution"
    )
    assert stacker.fit_audit_["best_single_candidate"] == "prior"
    assert stacker.fit_audit_["candidate_weights"] == {
        "prior": 1.0,
        "z_model": 0.0,
    }


def test_convex_stacker_contract_fallback_prefers_prior_even_when_worse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_minimize(
        objective: object,
        start: np.ndarray,
        **kwargs: object,
    ) -> SimpleNamespace:
        del objective, kwargs
        return SimpleNamespace(
            success=False,
            status=9,
            message="forced failure",
            nit=0,
            x=np.asarray(start, dtype=float),
        )

    monkeypatch.setattr(sensitivity, "minimize", failed_minimize)
    target = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    good_model = target.copy()
    worse_prior = np.repeat([[0.5, 0.5]], 2, axis=0)

    stacker = ConvexOOFProbabilityStacker().fit(
        {"model": good_model, "prior": worse_prior},
        target,
    )

    assert stacker.fit_audit_["best_single_candidate"] == "model"
    assert stacker.fit_audit_["fallback_candidate_requested"] == "prior"
    assert stacker.fit_audit_["fallback_candidate_resolved"] == "prior"
    assert stacker.fit_audit_["candidate_weights"] == {
        "model": 0.0,
        "prior": 1.0,
    }
    assert stacker.fit_audit_["stack_soft_log_loss"] > (
        stacker.fit_audit_["best_single_soft_log_loss"]
    )


def test_two_stage_probability_combination_preserves_declared_order() -> None:
    gate = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    conditional = np.asarray([[0.25, 0.75], [0.6, 0.4]])

    combined, audit = combine_two_stage_probabilities(
        gate,
        conditional,
        return_audit=True,
    )

    assert SENSITIVITY_CLASS_LABELS == (
        "low_60_side",
        "120_side",
        "high_off",
    )
    assert np.allclose(combined, [[0.2, 0.6, 0.2], [0.18, 0.12, 0.7]])
    assert audit["output_class_labels"] == list(
        SENSITIVITY_CLASS_LABELS
    )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("gaussian_prior_MAP", "gaussian_map"),
        ("soft_gower", "gower"),
        ("soft_extra_trees", "extra_trees"),
        ("soft_catboost", "catboost"),
        ("gaussian_map", "gaussian_map"),
        ("gower", "gower"),
        ("extra_trees", "extra_trees"),
        ("catboost", "catboost"),
    ],
)
def test_binary_stage_accepts_contract_and_short_model_names(
    alias: str,
    canonical: str,
) -> None:
    assert (
        sensitivity._canonical_soft_binary_stage_family(alias)
        == canonical
    )


@pytest.mark.parametrize(
    ("family", "params"),
    [
        (
            "gaussian_map",
            {
                "feature_prior_scale": 1.0,
                "max_iter": 500,
                "tolerance": 1.0e-8,
            },
        ),
        (
            "gower",
            {
                "neighbor_mode": "rbf_kernel",
                "bandwidth": 0.3,
                "block_weight_profile": "equal",
            },
        ),
        (
            "extra_trees",
            {
                "n_estimators": 20,
                "max_depth": 2,
                "min_samples_leaf": 2,
                "max_features": 0.7,
            },
        ),
        (
            "catboost",
            {
                "iterations": 12,
                "depth": 2,
                "learning_rate": 0.05,
            },
        ),
    ],
)
def test_fold_local_binary_stage_api_supports_fixed_model_library(
    family: str,
    params: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    records = _source_records().iloc[:72].reset_index(drop=True)
    train = records.iloc[:64].copy()
    test = records.iloc[64:].copy()
    target = np.linspace(0.15, 0.85, len(train))
    sample_weight = np.ones(len(train), dtype=float)
    sample_weight[0] = 0.0

    probability, audit = fit_predict_soft_binary_stage(
        train,
        target,
        test,
        "context_core",
        family,
        params,
        seed=713,
        sample_weight=sample_weight,
    )

    assert probability.shape == (len(test),)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert audit["model_family"] == family
    assert audit["feature_engineering_fit_scope"] == (
        "supplied_training_rows_only"
    )
    assert audit["posterior_argmax_used"] is False
    assert audit["n_train_records_before_zero_weight_filter"] == len(train)
    assert (
        audit["n_zero_weight_records_excluded_before_feature_engineering"]
        == 1
    )
    assert audit["n_train_records"] == len(train) - 1
    assert not (tmp_path / "catboost_info").exists()


def test_gower_block_weight_profile_maps_context_and_increment() -> None:
    records = _source_records().iloc[:72].reset_index(drop=True)
    train = records.iloc[:64].copy()
    test = records.iloc[64:].copy()
    target = np.linspace(0.15, 0.85, len(train))

    _, audit = fit_predict_soft_binary_stage(
        train,
        target,
        test,
        "context_plus_ionomer_ratio",
        "gower",
        {
            "neighbor_mode": "exponential_kernel",
            "bandwidth": 0.25,
            "block_weight_profile": "formulation_2x",
        },
        seed=41,
    )

    profile = audit["stage_audit"]["block_weight_profile"]
    assert audit["stage_audit"]["neighbor_scope"] == "all_training_records"
    assert audit["stage_audit"]["n_neighbors_effective"] == len(train)
    assert audit["stage_audit"]["distance_power"] is None
    assert audit["stage_audit"]["bandwidth"] == 0.25
    assert profile["context_block"] == "context"
    assert profile["increment_block"] == "ionomer_to_solvent_ratio"
    assert profile["resolved_block_weights"] == {
        "context": 1.0,
        "ionomer_to_solvent_ratio": 2.0,
    }
    assert profile["mapping_status"] == "complete"


def test_non_equal_gower_profile_fails_closed_without_increment() -> None:
    records = _source_records().iloc[:72].reset_index(drop=True)
    train = records.iloc[:64].copy()
    test = records.iloc[64:].copy()
    target = np.linspace(0.15, 0.85, len(train))

    with pytest.raises(ValueError, match="exactly one surviving increment"):
        fit_predict_soft_binary_stage(
            train,
            target,
            test,
            "context_core",
            "gower",
            {
                "neighbor_mode": "knn",
                "block_weight_profile": "context_2x",
            },
            seed=41,
        )


def test_two_stage_soft_catboost_uses_fractional_targets_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    X = pd.DataFrame(
        {
            "numeric": np.linspace(-1.0, 1.0, 18),
            "category": ["a", "b", "c"] * 6,
        }
    )
    q_high = np.linspace(0.05, 0.35, len(X))
    q_high[0] = 1.0
    q_low_conditional = np.linspace(0.2, 0.8, len(X))
    y = np.column_stack(
        [
            (1.0 - q_high) * q_low_conditional,
            (1.0 - q_high) * (1.0 - q_low_conditional),
            q_high,
        ]
    )
    estimator = TwoStageSoftCatBoostClassifier(
        categorical_columns=("category",),
        iterations=15,
        depth=2,
        learning_rate=0.05,
        random_seed=97,
    ).fit(X, y)

    first = estimator.predict_proba(X.iloc[:5])
    second = estimator.predict_proba(X.iloc[:5])

    assert first.shape == (5, 3)
    assert np.allclose(first, second)
    assert np.allclose(first.sum(axis=1), 1.0)
    assert estimator.fit_audit_["catboost_parameters"]["loss_function"] == (
        "CrossEntropy"
    )
    assert estimator.fit_audit_["thread_count"] == 1
    assert estimator.fit_audit_["allow_writing_files"] is False
    assert estimator.fit_audit_["gate"]["fractional_binary_target"] is True
    assert estimator.fit_audit_["conditional"][
        "fractional_binary_target"
    ] is True
    assert estimator.fit_audit_["conditional"]["total_sample_weight"] == (
        pytest.approx(float((1.0 - q_high).sum()))
    )
    assert estimator.fit_audit_["conditional"][
        "n_zero_weight_records_excluded"
    ] == 1
    assert not (tmp_path / "catboost_info").exists()


def test_catboost_missing_dependency_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> tuple[object, str]:
        raise RuntimeError("optional dependency missing; fails closed")

    monkeypatch.setattr(sensitivity, "_require_catboost", missing)
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = np.asarray(
        [[0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    )

    with pytest.raises(RuntimeError, match="fails closed"):
        TwoStageSoftCatBoostClassifier(iterations=2).fit(X, y)
