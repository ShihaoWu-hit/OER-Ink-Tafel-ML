from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ink_tafel_study.feature_association_models import (
    MODEL_FAMILIES,
    EmpiricalBayesAssociationModel,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_loading": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.nan, 0.5],
            "few_levels": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, np.nan, 1.0],
            "catalyst_family": [
                "Ir",
                "Ir",
                "Ir",
                "Ru",
                "Ru",
                "Ru",
                "singleton",
                None,
            ],
            "ir_corrected": [True, True, False, False, True, False, True, False],
        }
    )


def _response() -> np.ndarray:
    return np.log10(np.array([42.0, 48.0, 56.0, 65.0, 76.0, 91.0, 60.0, 70.0]))


def test_model_catalog_and_estimators_are_cloneable() -> None:
    assert MODEL_FAMILIES == ("bayesian_linear", "gam_low_df")
    for family in MODEL_FAMILIES:
        original = EmpiricalBayesAssociationModel(family)
        copied = clone(original)
        assert copied.get_params() == original.get_params()
        fitted = copied.fit(_features(), _response())
        prediction, std = fitted.predict(_features().iloc[:2], return_std=True)
        assert prediction.shape == (2,)
        assert std.shape == (2,)
        assert np.isfinite(prediction).all()
        assert np.isfinite(std).all()
        assert (std > 0).all()


def test_gam_splines_only_supported_numeric_columns_and_reports_low_df_design() -> None:
    fitted = EmpiricalBayesAssociationModel("gam_low_df").fit(
        _features(), _response()
    )
    assert fitted.numeric_columns_ == ("log_loading", "few_levels")
    assert fitted.categorical_columns_ == ("catalyst_family", "ir_corrected")
    assert fitted.spline_columns_ == ("log_loading",)
    assert fitted.linear_fallback_columns_ == ("few_levels",)
    assert fitted.design_column_count_ == len(fitted.design_feature_names_)
    assert 1.0 <= fitted.empirical_edf_ <= 1.0 + fitted.design_column_count_
    assert any("log_loading_sp_" in name for name in fitted.design_feature_names_)
    assert not any(
        ":" in str(name) or "*" in str(name)
        for name in fitted.design_feature_names_
    )

    definition = fitted.model_definition
    assert definition["scientific_name"] == (
        "empirical-Bayes shrinkage additive spline model"
    )
    assert definition["full_bayesian_posterior"] is False
    assert definition["interactions_included"] is False
    assert definition["gam_spline"]["n_knots"] == 3
    assert definition["gam_spline"]["degree"] == 1
    assert definition["gam_spline_columns"] == ["log_loading"]
    assert definition["gam_linear_fallback_columns"] == ["few_levels"]


def test_linear_model_standardizes_numeric_without_spline_terms() -> None:
    fitted = EmpiricalBayesAssociationModel("bayesian_linear").fit(
        _features(), _response()
    )
    assert fitted.spline_columns_ == ()
    assert fitted.linear_fallback_columns_ == ()
    assert not any("_sp_" in name for name in fitted.design_feature_names_)
    assert fitted.model_definition["scientific_name"] == (
        "empirical-Bayes shrinkage linear model"
    )


def test_missing_rare_and_unseen_categories_are_fold_local_and_safe() -> None:
    train = _features().iloc[:6].copy()
    fitted = EmpiricalBayesAssociationModel(
        "bayesian_linear", rare_min_frequency=3
    ).fit(train, _response()[:6])
    definition_before = fitted.model_definition
    levels = definition_before["categorical_levels_after_pooling"][
        "catalyst_family"
    ]
    assert levels[:2] == ["Ir", "Ru"]
    assert "__MISSING__" in levels
    assert "__RARE__" in levels

    test = train.iloc[:3].copy()
    test["catalyst_family"] = ["unseen", None, "Ir"]
    prediction = fitted.predict(test)
    assert np.isfinite(prediction).all()
    assert fitted.model_definition == definition_before


def test_heldout_extreme_does_not_change_training_median_knots_or_scaling() -> None:
    train = _features().iloc[:6].copy()
    fitted = EmpiricalBayesAssociationModel("gam_low_df").fit(
        train, _response()[:6]
    )
    transformer = fitted.preprocessor_.named_transformers_["numeric_000"]
    imputer_before = transformer.named_steps["imputer"].statistics_.copy()
    knots_before = transformer.named_steps["spline"].bsplines_[0].t.copy()
    scale_before = transformer.named_steps["scaler"].scale_.copy()

    heldout = train.iloc[[0]].copy()
    heldout["log_loading"] = 1_000_000.0
    fitted.predict(heldout)

    np.testing.assert_allclose(
        transformer.named_steps["imputer"].statistics_, imputer_before
    )
    np.testing.assert_allclose(
        transformer.named_steps["spline"].bsplines_[0].t, knots_before
    )
    np.testing.assert_allclose(transformer.named_steps["scaler"].scale_, scale_before)


@pytest.mark.parametrize(
    "column",
    [
        "target",
        "y_tafel_bayesian_local_line_mV_dec",
        "posterior_slope_median_mV_dec",
        "overpotential_10mA_reported_mV",
    ],
)
def test_target_or_curve_derived_predictors_are_rejected(column: str) -> None:
    X = _features().copy()
    X[column] = np.arange(len(X), dtype=float)
    with pytest.raises(ValueError, match="forbidden predictors"):
        EmpiricalBayesAssociationModel("bayesian_linear").fit(X, _response())


def test_prediction_requires_exact_fitted_schema_and_rejects_target_extras() -> None:
    fitted = EmpiricalBayesAssociationModel("bayesian_linear").fit(
        _features(), _response()
    )
    missing = _features().drop(columns="few_levels")
    with pytest.raises(ValueError, match="schema differs"):
        fitted.predict(missing)

    extra = _features().assign(target=np.arange(len(_features())))
    with pytest.raises(ValueError, match="forbidden predictors"):
        fitted.predict(extra)


def test_invalid_family_response_and_entirely_missing_numeric_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model_family"):
        EmpiricalBayesAssociationModel("random_forest").fit(
            _features(), _response()
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        EmpiricalBayesAssociationModel("bayesian_linear").fit(
            _features(), _response()[:, None]
        )
    X = _features().copy()
    X["all_missing"] = np.nan
    with pytest.raises(ValueError, match="no finite training values"):
        EmpiricalBayesAssociationModel("bayesian_linear").fit(X, _response())


def test_model_definition_property_is_a_defensive_copy() -> None:
    fitted = EmpiricalBayesAssociationModel("bayesian_linear").fit(
        _features(), _response()
    )
    definition = fitted.model_definition
    definition["numeric_columns"].append("tampered")
    assert "tampered" not in fitted.model_definition["numeric_columns"]

