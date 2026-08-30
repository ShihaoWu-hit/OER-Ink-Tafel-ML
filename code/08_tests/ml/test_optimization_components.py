from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge

from ink_tafel_study.optimization_components import (
    FEATURE_RECIPES,
    PROPERTY_CORE,
    PROPERTY_DIELECTRIC_BEST,
    PROPERTY_DIELECTRIC_FLAG,
    SOLVENT_FRACTIONS,
    RecipeFeatureBuilder,
    TargetTransformRegressor,
    make_candidate_regressor_specs,
    make_optimization_pipeline,
    normalize_optimization_columns,
    required_source_columns,
)


def _feature_frame(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    fractions = rng.dirichlet(np.ones(5), size=n)
    fractions[::4, 0] = 0.0
    fractions = fractions / fractions.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        {
            "catalyst_loading_mg_cm2_selected": rng.uniform(0.05, 1.0, n),
            "electrolyte_molarity_M": rng.choice([0.1, 0.5, 1.0], n),
            "rpm": rng.choice([0.0, 400.0, 900.0, 1600.0], n),
            "scan_rate_mV_s": rng.choice([2.0, 5.0, 10.0, 20.0], n),
            "electrolyte_identity": rng.choice(["KOH", "NaOH"], n),
            "ir_correction_status": rng.choice(
                [
                    "reported_corrected",
                    "reported_not_corrected",
                    "not_reported_or_unverifiable",
                ],
                n,
            ),
            "catalyst_active_material_type": rng.choice(["IrO2", "RuO2"], n),
            "catalyst_support": rng.choice(["none", "carbon"], n),
            "catalyst_concentration_mg_mL": rng.uniform(0.2, 10.0, n),
            "ionomer_to_solvent_ratio": rng.uniform(0.0, 0.2, n),
            "ionomer_concentration_wt_percent": rng.choice([0.0, 5.0, 20.0], n),
            "conductive_agent_dose_mg_per_mgcat": rng.uniform(0.0, 0.5, n),
            "conductive_agent_present": rng.choice(["yes", "no"], n),
            PROPERTY_CORE[0]: rng.uniform(0.7, 3.0, n),
            PROPERTY_CORE[1]: rng.uniform(20.0, 72.0, n),
            PROPERTY_CORE[2]: rng.uniform(780.0, 1030.0, n),
            PROPERTY_DIELECTRIC_BEST: rng.uniform(18.0, 78.0, n),
            PROPERTY_DIELECTRIC_FLAG: rng.choice([0, 1], n),
            "solvent_mixture_property_proxy_dielectric_constant": rng.uniform(
                18.0, 78.0, n
            ),
            "sonication_time_min": rng.uniform(5.0, 60.0, n),
            "drop_volume_uL": rng.uniform(2.0, 20.0, n),
            "electrode_area_cm2": rng.uniform(0.1, 1.0, n),
            "dispersion_method": rng.choice(["bath", "probe", "manual"], n),
        }
    )
    for index, column in enumerate(SOLVENT_FRACTIONS):
        frame[column] = fractions[:, index]
    frame.loc[0, "catalyst_loading_mg_cm2_selected"] = np.nan
    frame.loc[1, "dispersion_method"] = None
    return frame


def test_every_feature_recipe_is_cloneable_fold_local_and_finite() -> None:
    frame = _feature_frame()
    for recipe_name in FEATURE_RECIPES:
        builder = RecipeFeatureBuilder(recipe_name, numeric_transform="log")
        transformed = clone(builder).fit_transform(frame.iloc[:22])
        held_out = builder.fit(frame.iloc[:22]).transform(frame.iloc[22:])
        assert list(transformed.columns) == list(held_out.columns)
        assert np.isfinite(transformed.select_dtypes(include=[np.number])).all().all()
        assert np.isfinite(held_out.select_dtypes(include=[np.number])).all().all()


def test_recipe_allowlists_exclude_ids_targets_and_fit_fields() -> None:
    forbidden_tokens = (
        "record_id",
        "curve_id",
        "publication_id",
        "recipe_hash",
        "target",
        "tafel",
        "slope",
        "posterior",
        "selected_model",
    )
    for recipe in FEATURE_RECIPES.values():
        columns = required_source_columns(recipe)
        assert not any(
            token in column.lower() for column in columns for token in forbidden_tokens
        )


def test_ilr_is_four_dimensional_finite_and_closure_invariant() -> None:
    frame = _feature_frame()
    builder = RecipeFeatureBuilder("context_ink_ilr", numeric_transform="raw").fit(
        frame.iloc[:20]
    )
    original = builder.transform(frame.iloc[20:])
    scaled = frame.iloc[20:].copy()
    scaled.loc[:, list(SOLVENT_FRACTIONS)] *= 7.0
    transformed_scaled = builder.transform(scaled)
    ilr_columns = [column for column in original if column.startswith("solvent_ilr_")]
    assert len(ilr_columns) == 4
    np.testing.assert_allclose(
        original[ilr_columns].to_numpy(),
        transformed_scaled[ilr_columns].to_numpy(),
        atol=1e-12,
    )


def test_property_pca_is_fitted_only_on_training_rows() -> None:
    frame = _feature_frame()
    train = frame.iloc[:20].copy()
    held_out = frame.iloc[20:].copy()
    held_out.loc[:, list(PROPERTY_CORE) + [PROPERTY_DIELECTRIC_BEST]] *= 1_000.0
    builder = RecipeFeatureBuilder(
        "context_ink_properties_pca2", numeric_transform="raw"
    ).fit(train)
    components_before = builder.property_pca_.components_.copy()
    medians_before = builder.property_medians_.copy()
    builder.transform(held_out)
    np.testing.assert_allclose(builder.property_pca_.components_, components_before)
    np.testing.assert_allclose(builder.property_medians_, medians_before)
    refit_with_extreme = RecipeFeatureBuilder(
        "context_ink_properties_pca2", numeric_transform="raw"
    ).fit(pd.concat([train, held_out], ignore_index=True))
    assert not np.allclose(
        refit_with_extreme.property_scaler_.mean_, builder.property_scaler_.mean_
    )


def test_target_transform_wrapper_predicts_on_original_scale() -> None:
    X = np.arange(30, dtype=float).reshape(-1, 1)
    y = np.power(10.0, 1.0 + 0.02 * X[:, 0])
    for transform in ("identity", "log10"):
        estimator = TargetTransformRegressor(Ridge(alpha=0.1), transform=transform)
        fitted = clone(estimator).fit(X, y)
        prediction = fitted.predict(X)
        assert prediction.shape == y.shape
        assert np.isfinite(prediction).all()
        if transform == "log10":
            assert (prediction > 0).all()
    with pytest.raises(ValueError, match="strictly positive"):
        TargetTransformRegressor(Ridge(), transform="log10").fit(X, y - y.max())


def test_model_pipelines_clone_and_optional_catboost_can_be_skipped(monkeypatch) -> None:
    frame = _feature_frame()
    y = 80.0 + 30.0 * frame["electrolyte_molarity_M"].to_numpy(float)
    specs, _ = make_candidate_regressor_specs(include_catboost=False)
    for model_name, spec in specs.items():
        recipe = None if model_name.startswith("dummy_") else "context_ink_properties_pca2"
        pipeline = make_optimization_pipeline(
            recipe,
            spec,
            target_transform="log10",
            numeric_transform="log",
        )
        clone(pipeline)
        fitted = pipeline.fit(frame.iloc[:24], y[:24])
        assert np.isfinite(fitted.predict(frame.iloc[24:])).all()

    original = importlib.util.find_spec

    def no_catboost(name: str, *args, **kwargs):
        return None if name == "catboost" else original(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", no_catboost)
    without_catboost, skipped = make_candidate_regressor_specs(include_catboost=True)
    assert "catboost" not in without_catboost
    assert "catboost" in skipped


def test_canonical_alias_normalization_is_deterministic() -> None:
    frame = _feature_frame(4).drop(columns=[PROPERTY_DIELECTRIC_BEST])
    frame["solvent_mixture_property_proxy_dielectric_constant__best_available"] = [
        20.0,
        30.0,
        40.0,
        50.0,
    ]
    normalized = normalize_optimization_columns(frame)
    assert normalized[PROPERTY_DIELECTRIC_BEST].tolist() == [20.0, 30.0, 40.0, 50.0]

