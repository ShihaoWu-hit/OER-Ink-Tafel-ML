from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.presto_ordinal import (
    SoftFusedContinuationRatioClassifier,
)


def _synthetic_soft_ordinal() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20260726)
    numeric = rng.normal(size=72)
    category = np.where(numeric > 0, "positive", "negative")
    first = 1.0 / (1.0 + np.exp(-(0.8 * numeric - 0.1)))
    second = 1.0 / (1.0 + np.exp(-(0.7 * numeric - 1.0)))
    q = np.column_stack(
        [
            1.0 - first,
            first * (1.0 - second),
            first * second,
        ]
    )
    return pd.DataFrame({"numeric": numeric, "category": category}), q


def _estimator(penalty: float) -> SoftFusedContinuationRatioClassifier:
    return SoftFusedContinuationRatioClassifier(
        numeric_columns=("numeric",),
        categorical_columns=("category",),
        penalty=penalty,
        max_iter=3000,
        tolerance=1.0e-10,
    )


def test_soft_presto_gradient_matches_central_difference() -> None:
    X, q = _synthetic_soft_ordinal()
    fitted = _estimator(0.03).fit(X, q)
    transformed = np.asarray(
        fitted.preprocessor_.transform(X),
        dtype=float,
    )
    rng = np.random.default_rng(17)
    theta = rng.normal(scale=0.1, size=2 * fitted.design_width_)
    objective, gradient = fitted._objective_and_gradient(
        theta,
        transformed,
        q,
        np.ones(len(q), dtype=float),
    )
    assert np.isfinite(objective)
    step = 1.0e-6
    numerical = np.empty_like(theta)
    for index in range(len(theta)):
        upper = theta.copy()
        lower = theta.copy()
        upper[index] += step
        lower[index] -= step
        numerical[index] = (
            fitted._objective_and_gradient(
                upper,
                transformed,
                q,
                np.ones(len(q), dtype=float),
            )[0]
            - fitted._objective_and_gradient(
                lower,
                transformed,
                q,
                np.ones(len(q), dtype=float),
            )[0]
        ) / (2.0 * step)
    assert np.max(np.abs(numerical - gradient)) < 2.0e-6


def test_soft_presto_probabilities_are_exactly_valid() -> None:
    X, q = _synthetic_soft_ordinal()
    fitted = _estimator(0.03).fit(X, q)
    prediction = fitted.predict_proba(
        pd.DataFrame(
            {
                "numeric": [np.nan, -1.0, 1.0],
                "category": ["unseen", "negative", "positive"],
            }
        )
    )
    transition = fitted.predict_transition_proba(
        pd.DataFrame(
            {
                "numeric": [np.nan, -1.0, 1.0],
                "category": ["unseen", "negative", "positive"],
            }
        )
    )
    assert np.isfinite(prediction).all()
    assert np.min(prediction) >= 0
    assert np.max(np.abs(prediction.sum(axis=1) - 1.0)) < 1.0e-12
    assert np.allclose(
        prediction[:, 2],
        transition[:, 0] * transition[:, 1],
        atol=1.0e-15,
        rtol=0,
    )
    assert fitted.posterior_draws_expanded_ is False
    assert fitted.class_rows_expanded_ is False


def test_fusion_penalty_shrinks_adjacent_slopes() -> None:
    X, q = _synthetic_soft_ordinal()
    unpenalized = _estimator(0.0).fit(X, q)
    fused = _estimator(0.3).fit(X, q)
    assert fused.slope_difference_l1_ < unpenalized.slope_difference_l1_


def test_zero_second_transition_mass_fails_closed() -> None:
    X, _ = _synthetic_soft_ordinal()
    q = np.column_stack(
        [
            np.ones(len(X), dtype=float),
            np.zeros(len(X), dtype=float),
            np.zeros(len(X), dtype=float),
        ]
    )
    with pytest.raises(
        ValueError,
        match="second ordinal transition has zero training mass",
    ):
        _estimator(0.03).fit(X, q)
