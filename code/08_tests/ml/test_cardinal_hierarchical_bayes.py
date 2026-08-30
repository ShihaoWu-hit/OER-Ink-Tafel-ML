from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from ink_tafel_study.cardinal_feature_engineering import (
    required_cardinal_feature_columns,
)
from ink_tafel_study.cardinal_hierarchical_bayes import (
    HIERARCHICAL_FAMILIES,
    HIERARCHICAL_FAMILY_COLUMN,
    HierarchicalGaussianMAPSoftmaxClassifier,
    HierarchicalMAPCandidate,
    HierarchicalMAPCardinalClassifier,
    canonical_catalyst_family,
    run_hierarchical_map_nested_oof,
)
from ink_tafel_study.cardinal_models import CARDINAL_CLASSES


def _core_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    n_per_family = 14
    family = np.repeat(HIERARCHICAL_FAMILIES, n_per_family)
    x = np.tile(np.linspace(-1.5, 1.5, n_per_family), 2)
    x += rng.normal(0.0, 0.05, len(x))
    family_shift = np.where(family == "RuO2", 0.8, -0.8)
    logits = np.column_stack(
        [
            0.7 * x + family_shift,
            -0.4 * x - 0.5 * family_shift,
            np.zeros(len(x)),
        ]
    )
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    # The responses remain fractional; they are not sampled hard classes.
    probability = 0.9 * probability + 0.1 / probability.shape[1]
    frame = pd.DataFrame(
        {
            "x": x,
            "electrolyte": np.where(x > 0, "KOH", "NaOH"),
            HIERARCHICAL_FAMILY_COLUMN: family,
        }
    )
    return frame, probability


def _raw_nested_frame(n_per_family: int = 12) -> pd.DataFrame:
    n = 2 * n_per_family
    family = np.repeat(["IrO2", "RuO2"], n_per_family)
    x = np.tile(np.linspace(0.1, 1.0, n_per_family), 2)
    family_sign = np.where(family == "RuO2", 1.0, -1.0)
    logits = np.column_stack(
        [
            0.7 * x + 0.5 * family_sign,
            0.2 - 0.3 * x,
            -0.2 + 0.2 * x - 0.3 * family_sign,
            np.zeros(n),
        ]
    )
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        {
            "record_id": [f"r{index:03d}" for index in range(n)],
            "cv_group_id": [f"g{index:03d}" for index in range(n)],
            "catalyst_type": family,
            "catalyst_loading_mg_cm2_selected": 0.1 + x,
            "electrolyte_molarity_M": 0.5 + x,
            "rpm": 1000.0 + 200.0 * np.arange(n),
            "scan_rate_mV_s": 5.0 + x,
            "electrolyte_identity": np.where(
                np.arange(n) % 2, "KOH", "NaOH"
            ),
            "ir_correction_status": np.where(
                np.arange(n) % 3, "corrected", "not_reported"
            ),
        }
    )
    for index, class_name in enumerate(CARDINAL_CLASSES):
        frame[f"q_{class_name}"] = probability[:, index]
    return frame


def test_hierarchical_map_uses_one_fractional_response_per_record() -> None:
    X, y = _core_data()
    estimator = HierarchicalGaussianMAPSoftmaxClassifier(
        numeric_columns=("x",),
        categorical_columns=("electrolyte",),
        family_column=HIERARCHICAL_FAMILY_COLUMN,
        expected_families=HIERARCHICAL_FAMILIES,
        class_names=("a", "b", "c"),
        family_feature_scale=0.5,
        family_intercept_scale=1.0,
        max_iter=500,
    )
    fitted = clone(estimator).fit(X, y)
    predicted = fitted.predict_proba(X.iloc[:5])
    assert predicted.shape == (5, 3)
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(predicted.sum(axis=1), 1.0, atol=1e-12)
    assert fitted.n_training_records_ == len(X)
    assert fitted.total_fractional_weight_ == len(X)
    assert fitted.posterior_draws_expanded_ is False
    assert "MAP" in fitted.inference_method_
    # The contrast basis enforces a training-mass-weighted zero mean.
    np.testing.assert_allclose(
        np.einsum(
            "g,gpk->pk",
            fitted.family_weights_,
            fitted.family_coef_deviation_,
        ),
        0.0,
        atol=1e-10,
    )


def test_stronger_family_prior_shrinks_family_departures() -> None:
    X, y = _core_data()
    common = {
        "numeric_columns": ("x",),
        "categorical_columns": ("electrolyte",),
        "family_column": HIERARCHICAL_FAMILY_COLUMN,
        "expected_families": HIERARCHICAL_FAMILIES,
        "class_names": ("a", "b", "c"),
        "global_feature_scale": 2.0,
        "global_intercept_scale": 5.0,
        "max_iter": 500,
    }
    strong = HierarchicalGaussianMAPSoftmaxClassifier(
        **common,
        family_feature_scale=0.05,
        family_intercept_scale=0.05,
    ).fit(X, y)
    weak = HierarchicalGaussianMAPSoftmaxClassifier(
        **common,
        family_feature_scale=5.0,
        family_intercept_scale=5.0,
    ).fit(X, y)
    strong_norm = np.linalg.norm(strong.family_coef_deviation_) + np.linalg.norm(
        strong.family_intercept_deviation_
    )
    weak_norm = np.linalg.norm(weak.family_coef_deviation_) + np.linalg.norm(
        weak.family_intercept_deviation_
    )
    assert strong_norm < weak_norm


def test_two_stage_wrapper_maps_metallic_ir_into_ir_family() -> None:
    records = pd.read_csv(
        "Data/03_outputs/study/outputs/current_97/data/records.csv"
    )
    selected = pd.concat(
        [
            records.loc[records["catalyst_type"].eq("RuO2")].head(10),
            records.loc[records["catalyst_type"].eq("IrO2")].head(9),
            records.loc[records["catalyst_type"].eq("Ir")].head(1),
        ],
        ignore_index=True,
    )
    columns = required_cardinal_feature_columns("context_log")
    X = selected.loc[:, columns]
    targets = np.tile(
        np.asarray([[0.2, 0.4, 0.15, 0.25]]), (len(X), 1)
    )
    fitted = clone(
        HierarchicalMAPCardinalClassifier(
            feature_recipe="context_log",
            family_feature_scale=0.25,
            family_intercept_scale=0.5,
            max_iter=300,
        )
    ).fit(X, targets)
    probability = fitted.predict_proba(X.iloc[:4])
    assert probability.shape == (4, 4)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-12)
    assert fitted.gate_estimator_.n_training_records_ == len(X)
    assert fitted.posterior_draws_expanded_ is False
    assert canonical_catalyst_family(["Ir"]).tolist() == [
        "IrO2_plus_Ir"
    ]


def test_nested_hierarchical_map_keeps_connected_groups_out_of_fold(
    tmp_path: Path,
) -> None:
    frame = _raw_nested_frame()
    candidate = HierarchicalMAPCandidate(
        "fixed_moderate", 0.5, 1.0, complexity_rank=0
    )
    output = tmp_path / "hierarchical"
    result = run_hierarchical_map_nested_oof(
        frame,
        feature_recipe="context_log",
        output_dir=output,
        outer_folds=2,
        outer_repeats=1,
        inner_folds=2,
        candidates=(candidate,),
        max_iter=250,
    )
    assert len(result.oof_aggregated) == len(frame)
    assert result.oof_aggregated["n_repeated_oof"].eq(1).all()
    assert result.inner_candidate_scores["status"].eq("ok").all()
    for _, repeat in result.split_manifest.groupby("repeat"):
        assert repeat.groupby("cv_group_id")["fold"].nunique().eq(1).all()
    definition = json.loads(
        (output / "model_definition.json").read_text(encoding="utf-8")
    )
    assert definition["posterior_draws_expanded"] is False
    assert definition["one_fractional_response_per_record"] is True
    assert "no NUTS/MCMC" in definition["approximation"]

