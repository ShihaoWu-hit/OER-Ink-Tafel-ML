from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import ink_tafel_study.cardinal_reduced_models as reduced_models
from ink_tafel_study.cardinal_models import (
    CARDINAL_CLASSES,
    SoftGowerKNeighborsClassifier,
)
from ink_tafel_study.cardinal_reduced_models import (
    GaussianMAPSoftmaxClassifier,
    ReducedPanelGowerClassifier,
    ReducedPanelTwoStageMAPClassifier,
)


def _raw_records(n_rows: int = 18) -> pd.DataFrame:
    index = np.arange(n_rows)
    return pd.DataFrame(
        {
            "catalyst_active_material_type": np.where(
                index % 3 == 0,
                "RuO2",
                np.where(index % 3 == 1, "IrO2", "Ir"),
            ),
            "catalyst_loading_mg_cm2_selected": 0.10 + 0.01 * index,
            "electrolyte_molarity_M": 0.5 + 0.1 * (index % 5),
            "rpm": 100.0 * (index % 6),
            "scan_rate_mV_s": 5.0 + (index % 4),
            "ir_correction_status": np.where(
                index % 2 == 0, "corrected", "not_reported"
            ),
            "electrolyte_scope": np.where(
                index % 5 == 0, "acidic", "alkaline"
            ),
            "catalyst_concentration_mg_mL": 1.0 + 0.25 * index,
        }
    )


def _cardinal_soft_targets(n_rows: int = 18) -> pd.DataFrame:
    x = np.linspace(-1.5, 1.5, n_rows)
    logits = np.column_stack(
        [
            1.2 - 0.8 * x,
            1.0 - 0.15 * np.square(x),
            0.7 + 0.9 * x,
            0.4 + 0.25 * np.cos(2.0 * x),
        ]
    )
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return pd.DataFrame(probabilities, columns=CARDINAL_CLASSES)


def _assert_simplex(
    probabilities: np.ndarray, n_rows: int, n_classes: int
) -> None:
    assert probabilities.shape == (n_rows, n_classes)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-12
    )


def test_gaussian_map_uses_one_soft_vector_per_record_and_is_cloneable() -> None:
    X = pd.DataFrame(
        {
            "x": np.linspace(-2.0, 2.0, 12),
            "kind": ["a", "a", "b", "b"] * 3,
        }
    )
    y = pd.DataFrame(
        np.column_stack(
            [
                np.linspace(0.70, 0.15, len(X)),
                np.full(len(X), 0.20),
                np.linspace(0.10, 0.65, len(X)),
            ]
        ),
        columns=["q_low", "q_middle", "q_high"],
    )
    estimator = GaussianMAPSoftmaxClassifier(
        numeric_columns=("x",),
        categorical_columns=("kind",),
        class_names=("low", "middle", "high"),
        max_iter=500,
        tolerance=1e-9,
    )
    fitted = clone(estimator).fit(X, y)

    assert fitted.n_input_records_ == len(X)
    assert fitted.n_training_records_ == len(X)
    assert fitted.objective_row_count_ == len(X)
    assert fitted.soft_target_shape_ == (len(X), 3)
    assert fitted.class_rows_expanded_ is False
    assert fitted.posterior_draws_expanded_ is False
    assert fitted.inference_method_ == "gaussian_prior_MAP_scipy_L-BFGS-B"
    assert fitted.optimization_audit_["method"] == "L-BFGS-B"
    _assert_simplex(fitted.predict_proba(X.iloc[:4]), 4, 3)

    cloned = clone(fitted)
    assert cloned.get_params(deep=True) == estimator.get_params(deep=True)
    assert not hasattr(cloned, "preprocessor_")


def test_strong_gaussian_feature_prior_shrinks_map_coefficients() -> None:
    x = np.linspace(-3.0, 3.0, 60)
    X = pd.DataFrame({"x": x})
    probability = 0.05 + 0.90 / (1.0 + np.exp(-2.5 * x))
    y = np.column_stack([probability, 1.0 - probability])
    weak = GaussianMAPSoftmaxClassifier(
        numeric_columns=("x",),
        class_names=("positive", "negative"),
        feature_prior_scale=100.0,
        intercept_prior_scale=100.0,
        tolerance=1e-10,
    ).fit(X, y)
    strong = GaussianMAPSoftmaxClassifier(
        numeric_columns=("x",),
        class_names=("positive", "negative"),
        feature_prior_scale=0.01,
        intercept_prior_scale=100.0,
        tolerance=1e-10,
    ).fit(X, y)

    weak_norm = float(np.linalg.norm(weak.coef_))
    strong_norm = float(np.linalg.norm(strong.coef_))
    assert strong_norm < 0.1 * weak_norm
    _assert_simplex(strong.predict_proba(X), len(X), 2)


def test_nonconverged_finite_optimizer_result_is_auditable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_minimize(
        function: object,
        initial: np.ndarray,
        args: tuple[object, ...],
        **kwargs: object,
    ) -> SimpleNamespace:
        del function, args, kwargs
        return SimpleNamespace(
            x=np.asarray(initial, dtype=float),
            fun=12.5,
            success=False,
            status=1,
            message="iteration limit reached",
            nit=1,
            nfev=2,
            njev=2,
        )

    monkeypatch.setattr(reduced_models, "minimize", fake_minimize)
    X = pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
    y = np.asarray([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
    fitted = GaussianMAPSoftmaxClassifier(
        numeric_columns=("x",),
        class_names=("a", "b"),
    ).fit(X, y)

    assert fitted.optimizer_success_ is False
    assert fitted.fit_status_ == "optimizer_nonconverged_finite_solution"
    assert fitted.optimization_audit_["status"] == 1
    assert "iteration limit" in fitted.optimization_audit_["message"]
    _assert_simplex(fitted.predict_proba(X), len(X), 2)


def test_reduced_panel_map_fits_transform_and_cardinal_weighted_stages() -> None:
    X = _raw_records()
    y = _cardinal_soft_targets()
    weights = np.linspace(0.5, 1.5, len(X))
    estimator = ReducedPanelTwoStageMAPClassifier(
        panel="context_plus_catalyst_concentration",
        feature_prior_scale=1.0,
        intercept_prior_scale=3.0,
        max_iter=500,
        tolerance=1e-9,
    )
    fitted = clone(estimator).fit(X, y, sample_weight=weights)

    assert fitted.feature_builder_.panel_.name == (
        "context_plus_catalyst_concentration"
    )
    assert "log10_catalyst_concentration_mg_mL" in set(
        fitted.engineered_feature_names_
    )
    assert isinstance(
        fitted.gate_estimator_, GaussianMAPSoftmaxClassifier
    )
    assert isinstance(
        fitted.conditional_estimator_, GaussianMAPSoftmaxClassifier
    )
    assert fitted.gate_estimator_.n_training_records_ == len(X)
    cardinal_mass = y.loc[:, CARDINAL_CLASSES[:-1]].sum(axis=1)
    assert fitted.conditional_estimator_.total_fractional_weight_ == (
        pytest.approx(float(np.sum(weights * cardinal_mass)))
    )
    assert fitted.posterior_draws_expanded_ is False
    _assert_simplex(
        fitted.predict_proba(X.iloc[:5]),
        5,
        len(CARDINAL_CLASSES),
    )

    cloned = clone(fitted)
    assert cloned.panel == estimator.panel
    assert cloned.feature_prior_scale == estimator.feature_prior_scale
    assert not hasattr(cloned, "feature_builder_")


def test_reduced_panel_gower_uses_fold_local_panel_k_p_and_simplex() -> None:
    X = _raw_records()
    y = _cardinal_soft_targets()
    estimator = ReducedPanelGowerClassifier(
        panel="context_plus_catalyst_concentration",
        k=3,
        p=2.0,
    )
    fitted = clone(estimator).fit(X, y)

    assert fitted.feature_builder_.panel_.name == estimator.panel
    assert isinstance(
        fitted.gate_estimator_, SoftGowerKNeighborsClassifier
    )
    assert isinstance(
        fitted.conditional_estimator_, SoftGowerKNeighborsClassifier
    )
    assert fitted.gate_estimator_.n_neighbors == 3
    assert fitted.gate_estimator_.distance_power == pytest.approx(2.0)
    assert set(fitted.blocks_) == {
        "context",
        "catalyst_concentration",
    }
    _assert_simplex(
        fitted.predict_proba(X.iloc[:6]),
        6,
        len(CARDINAL_CLASSES),
    )

    cloned = clone(fitted)
    assert cloned.k == 3
    assert cloned.p == pytest.approx(2.0)
    assert not hasattr(cloned, "feature_builder_")
