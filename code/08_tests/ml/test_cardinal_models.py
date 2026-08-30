from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ink_tafel_study.cardinal_models import (
    CARDINAL_CLASSES,
    CARDINAL_MODEL_NAMES,
    CLASSIC_CARDINAL_CLASSES,
    CardinalModelSpec,
    ClassPriorSoftClassifier,
    SoftExtraTreesClassifier,
    SoftGowerKNeighborsClassifier,
    SoftMultinomialLogisticClassifier,
    SoftOrdinalLogisticClassifier,
    TwoStageCardinalClassifier,
    cardinal_model_catalog,
    make_cardinal_model_spec,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fraction": [0.0, 0.1, 0.3, 0.6, 0.8, 1.0, np.nan, 0.4],
            "electrolyte": [
                "KOH",
                "KOH",
                "NaOH",
                "NaOH",
                "KOH",
                "NaOH",
                "KOH",
                None,
            ],
        }
    )


def _soft_targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.80, 0.10, 0.05, 0.05],
            [0.65, 0.20, 0.05, 0.10],
            [0.35, 0.45, 0.10, 0.10],
            [0.15, 0.55, 0.20, 0.10],
            [0.10, 0.25, 0.50, 0.15],
            [0.05, 0.10, 0.65, 0.20],
            [0.15, 0.20, 0.20, 0.45],
            [0.25, 0.25, 0.15, 0.35],
        ],
        columns=CARDINAL_CLASSES,
    )


def _assert_valid_probabilities(probabilities: np.ndarray, n_rows: int) -> None:
    assert probabilities.shape == (n_rows, len(CARDINAL_CLASSES))
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_class_prior_uses_training_record_weights_and_soft_mass() -> None:
    X = pd.DataFrame({"x": [0.0, 1.0]})
    y = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        columns=CARDINAL_CLASSES,
    )
    model = ClassPriorSoftClassifier(prior_smoothing=0.0).fit(
        X, y, sample_weight=np.array([1.0, 3.0])
    )
    probabilities = model.predict_proba(pd.DataFrame({"x": [10.0, 20.0]}))
    np.testing.assert_allclose(probabilities[0], [0.25, 0.0, 0.0, 0.75])
    np.testing.assert_allclose(probabilities[1], probabilities[0])


def test_models_accept_posterior_q_prefixed_soft_label_columns() -> None:
    X = _features()
    q_targets = _soft_targets().rename(
        columns={name: f"q_{name}" for name in CARDINAL_CLASSES}
    )
    model = ClassPriorSoftClassifier().fit(X, q_targets)
    _assert_valid_probabilities(model.predict_proba(X.iloc[:2]), 2)


@pytest.mark.parametrize(
    "model",
    [
        SoftMultinomialLogisticClassifier(
            numeric_columns=("fraction",),
            categorical_columns=("electrolyte",),
            C=1.0,
        ),
        SoftExtraTreesClassifier(
            numeric_columns=("fraction",),
            categorical_columns=("electrolyte",),
            n_estimators=20,
            max_depth=2,
            min_samples_leaf=2,
            random_state=7,
        ),
    ],
)
def test_tabular_soft_models_fit_targets_with_fold_local_preprocessing(
    model: object,
) -> None:
    X = _features()
    y = _soft_targets()
    fitted = clone(model).fit(X, y, sample_weight=np.arange(1, len(X) + 1))
    probabilities = fitted.predict_proba(
        pd.DataFrame({"fraction": [0.2, np.nan], "electrolyte": ["LiOH", "KOH"]})
    )
    _assert_valid_probabilities(probabilities, 2)
    assert hasattr(fitted, "preprocessor_")
    assert list(fitted.feature_names_in_) == list(X.columns)


def test_soft_ordinal_logistic_models_only_the_ordered_cardinal_subclasses() -> None:
    X = _features()
    cardinal_mass = _soft_targets().loc[:, CLASSIC_CARDINAL_CLASSES].sum(axis=1)
    conditional = _soft_targets().loc[:, CLASSIC_CARDINAL_CLASSES].div(
        cardinal_mass, axis=0
    )
    model = SoftOrdinalLogisticClassifier(
        numeric_columns=("fraction",),
        categorical_columns=("electrolyte",),
        class_names=CLASSIC_CARDINAL_CLASSES,
        C=1.0,
    ).fit(X, conditional, sample_weight=cardinal_mass)

    probabilities = model.predict_proba(X.iloc[:3])
    assert probabilities.shape == (3, len(CLASSIC_CARDINAL_CLASSES))
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert tuple(model.classes_) == CLASSIC_CARDINAL_CLASSES
    assert "off_cardinal" not in set(model.classes_)


def test_soft_gower_knn_exact_neighbors_use_record_and_soft_label_weights() -> None:
    X = pd.DataFrame({"fraction": [0.0, 0.0, 1.0]})
    y = pd.DataFrame(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        columns=CLASSIC_CARDINAL_CLASSES,
    )
    model = SoftGowerKNeighborsClassifier(
        numeric_columns=("fraction",),
        categorical_columns=(),
        class_names=CLASSIC_CARDINAL_CLASSES,
        n_neighbors=3,
    ).fit(X, y, sample_weight=np.array([1.0, 3.0, 1.0]))
    probability = model.predict_proba(pd.DataFrame({"fraction": [0.0]}))
    np.testing.assert_allclose(probability[0], [0.25, 0.75, 0.0])


def test_soft_gower_knn_uses_all_exact_matches_independent_of_training_order() -> None:
    X = pd.DataFrame({"fraction": [0.0, 0.0, 0.0, 1.0]})
    y = pd.DataFrame(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        columns=CLASSIC_CARDINAL_CLASSES,
    )
    weights = np.array([1.0, 2.0, 3.0, 1.0])
    query = pd.DataFrame({"fraction": [0.0]})
    permutation = np.array([2, 3, 0, 1])

    original = SoftGowerKNeighborsClassifier(
        numeric_columns=("fraction",),
        categorical_columns=(),
        class_names=CLASSIC_CARDINAL_CLASSES,
        n_neighbors=2,
    ).fit(X, y, sample_weight=weights)
    shuffled = SoftGowerKNeighborsClassifier(
        numeric_columns=("fraction",),
        categorical_columns=(),
        class_names=CLASSIC_CARDINAL_CLASSES,
        n_neighbors=2,
    ).fit(
        X.iloc[permutation].reset_index(drop=True),
        y.iloc[permutation].reset_index(drop=True),
        sample_weight=weights[permutation],
    )

    expected = np.array([[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]])
    np.testing.assert_allclose(original.predict_proba(query), expected)
    np.testing.assert_allclose(shuffled.predict_proba(query), expected)


def test_soft_gower_knn_expands_equal_kth_distance_independent_of_order() -> None:
    X = pd.DataFrame({"fraction": [0.0, 1.0]})
    y = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        columns=CLASSIC_CARDINAL_CLASSES,
    )
    query = pd.DataFrame({"fraction": [0.5]})

    original = SoftGowerKNeighborsClassifier(
        numeric_columns=("fraction",),
        categorical_columns=(),
        class_names=CLASSIC_CARDINAL_CLASSES,
        n_neighbors=1,
    ).fit(X, y)
    shuffled = SoftGowerKNeighborsClassifier(
        numeric_columns=("fraction",),
        categorical_columns=(),
        class_names=CLASSIC_CARDINAL_CLASSES,
        n_neighbors=1,
    ).fit(
        X.iloc[::-1].reset_index(drop=True),
        y.iloc[::-1].reset_index(drop=True),
    )

    expected = np.array([[0.5, 0.0, 0.5]])
    np.testing.assert_allclose(original.predict_proba(query), expected)
    np.testing.assert_allclose(shuffled.predict_proba(query), expected)


@pytest.mark.parametrize(
    "neighbor_mode",
    ["exponential_kernel", "rbf_kernel"],
)
def test_soft_gower_full_neighbor_kernels_return_probability_simplex(
    neighbor_mode: str,
) -> None:
    X = _features()
    y = _soft_targets()
    model = SoftGowerKNeighborsClassifier(
        numeric_columns=("fraction",),
        categorical_columns=("electrolyte",),
        blocks={
            "formulation": ("fraction",),
            "context": ("electrolyte",),
        },
        block_weights={"formulation": 2.0, "context": 0.5},
        feature_weights={"fraction": 1.5},
        neighbor_mode=neighbor_mode,
        bandwidth=0.25,
    ).fit(X, y, sample_weight=np.arange(1, len(X) + 1, dtype=float))

    probabilities = model.predict_proba(
        pd.DataFrame(
            {
                "fraction": [0.2, np.nan, 0.9],
                "electrolyte": ["KOH", "LiOH", None],
            }
        )
    )
    _assert_valid_probabilities(probabilities, 3)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("neighbor_mode", "triangular", "neighbor_mode"),
        ("bandwidth", 0.0, "bandwidth"),
    ],
)
def test_soft_gower_rejects_invalid_kernel_parameters(
    parameter: str,
    value: object,
    message: str,
) -> None:
    kwargs = {parameter: value}
    with pytest.raises(ValueError, match=message):
        SoftGowerKNeighborsClassifier(
            numeric_columns=("fraction",),
            categorical_columns=("electrolyte",),
            **kwargs,
        ).fit(_features(), _soft_targets())


def test_soft_extra_trees_leaf_constraint_counts_original_records() -> None:
    X = _features()
    y = _soft_targets()
    minimum_records = 3
    model = SoftExtraTreesClassifier(
        numeric_columns=("fraction",),
        categorical_columns=("electrolyte",),
        n_estimators=20,
        max_depth=3,
        min_samples_leaf=minimum_records,
        random_state=11,
    ).fit(X, y)

    transformed = model.preprocessor_.transform(X)
    leaf_assignments = model.model_.apply(transformed)
    for tree_position in range(leaf_assignments.shape[1]):
        _, counts = np.unique(
            leaf_assignments[:, tree_position], return_counts=True
        )
        assert int(counts.min()) >= minimum_records


def test_two_stage_prior_reconstructs_global_hierarchical_soft_prior() -> None:
    X = _features()
    y = _soft_targets()
    weights = np.arange(1, len(X) + 1, dtype=float)
    model = TwoStageCardinalClassifier(
        gate_estimator=ClassPriorSoftClassifier(
            class_names=("cardinal", "off_cardinal"),
            prior_smoothing=0.0,
        ),
        conditional_estimator=ClassPriorSoftClassifier(
            class_names=CLASSIC_CARDINAL_CLASSES,
            prior_smoothing=0.0,
        ),
    ).fit(X, y, sample_weight=weights)

    probability = model.predict_proba(X.iloc[:1])[0]
    expected = np.average(y.to_numpy(), axis=0, weights=weights)
    np.testing.assert_allclose(probability, expected, atol=1e-12)
    assert list(model.classes_) == list(CARDINAL_CLASSES)
    assert hasattr(model, "gate_estimator_")
    assert hasattr(model, "conditional_estimator_")


def test_two_stage_returns_uniform_conditional_if_fold_is_entirely_off_cardinal() -> None:
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = np.zeros((3, len(CARDINAL_CLASSES)), dtype=float)
    y[:, -1] = 1.0
    model = TwoStageCardinalClassifier().fit(X, y)
    probabilities = model.predict_proba(X)
    _assert_valid_probabilities(probabilities, len(X))
    assert (probabilities[:, -1] > 1.0 - 1e-8).all()
    np.testing.assert_allclose(
        probabilities[:, :-1],
        np.repeat(
            probabilities[:, [0]], len(CLASSIC_CARDINAL_CLASSES), axis=1
        ),
    )


def test_catalog_has_ordered_finite_small_grids_and_cloneable_estimators() -> None:
    specs = cardinal_model_catalog(
        numeric_columns=("fraction",),
        categorical_columns=("electrolyte",),
        blocks={"formulation": ("fraction",), "context": ("electrolyte",)},
        random_seed=9,
    )
    assert tuple(spec.name for spec in specs) == CARDINAL_MODEL_NAMES
    assert tuple(spec.complexity_rank for spec in specs) == (0, 1, 2, 3, 4)
    assert all(isinstance(spec, CardinalModelSpec) for spec in specs)
    assert all(
        all(0 < len(tuple(values)) <= 2 for values in spec.param_grid.values())
        for spec in specs
    )
    for spec in specs:
        cloned = clone(spec.estimator)
        fitted = cloned.fit(_features(), _soft_targets())
        _assert_valid_probabilities(fitted.predict_proba(_features().iloc[:2]), 2)


def test_factory_rejects_unknown_models_and_parameters() -> None:
    with pytest.raises(KeyError, match="Unknown cardinal model"):
        make_cardinal_model_spec(
            "not_a_model",
            numeric_columns=("x",),
        )
    with pytest.raises(TypeError, match="Unknown"):
        make_cardinal_model_spec(
            "soft_multinomial_logistic",
            numeric_columns=("x",),
            accidental_option=True,
        )

    ordinal = make_cardinal_model_spec(
        "soft_ordinal_logistic",
        numeric_columns=("x",),
        C=0.5,
    )
    assert ordinal.name == "soft_ordinal_logistic"
    assert ordinal.complexity_rank < make_cardinal_model_spec(
        "soft_multinomial_logistic",
        numeric_columns=("x",),
    ).complexity_rank


def test_soft_target_validation_rejects_hard_vector_and_negative_mass() -> None:
    X = pd.DataFrame({"x": [0.0, 1.0]})
    model = ClassPriorSoftClassifier()
    with pytest.raises(ValueError, match="shape"):
        model.fit(X, np.array([0, 1]))
    bad = np.full((2, len(CARDINAL_CLASSES)), 1.0 / len(CARDINAL_CLASSES))
    bad[0, 0] = -0.1
    with pytest.raises(ValueError, match="negative"):
        model.fit(X, bad)
