from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.models import (
    GowerKNeighborsRegressor,
    gaussian_log_scores,
    lognormal_original_scale_scores,
)


def test_gower_distance_blocks_must_be_mutually_exclusive_and_exhaustive() -> None:
    X = pd.DataFrame({"amount": [0.0, 1.0], "solvent": ["water", "ipa"]})
    y = np.log10([40.0, 120.0])
    overlapping = GowerKNeighborsRegressor(
        numeric_features=("amount",),
        categorical_features=("solvent",),
        blocks={"ink": ("amount", "solvent"), "solvent": ("solvent",)},
    )
    with pytest.raises(ValueError, match="overlap"):
        overlapping.fit(X, y)

    incomplete = GowerKNeighborsRegressor(
        numeric_features=("amount",),
        categorical_features=("solvent",),
        blocks={"ink": ("amount",)},
    )
    with pytest.raises(ValueError, match="exhaustive"):
        incomplete.fit(X, y)


def test_gower_knn_exact_neighbor_and_missing_aware_prediction() -> None:
    X = pd.DataFrame(
        {
            "amount": [0.0, 0.5, 1.0],
            "solvent": ["water", "mix", "ipa"],
        }
    )
    y = np.log10([40.0, 80.0, 120.0])
    model = GowerKNeighborsRegressor(
        n_neighbors=2,
        numeric_features=("amount",),
        categorical_features=("solvent",),
        blocks={"amount": ("amount",), "composition": ("solvent",)},
    ).fit(X, y)

    exact = model.predict(pd.DataFrame({"amount": [0.5], "solvent": ["mix"]}))
    assert exact[0] == pytest.approx(np.log10(80.0))

    # Missing numeric input uses the still-comparable categorical block.
    missing = model.predict(pd.DataFrame({"amount": [np.nan], "solvent": ["water"]}))
    assert np.isfinite(missing[0])
    assert 10**missing[0] == pytest.approx(40.0)


def test_gower_training_ranges_do_not_use_prediction_data() -> None:
    X = pd.DataFrame({"amount": [0.0, 1.0]})
    y = np.log10([50.0, 100.0])
    model = GowerKNeighborsRegressor(
        n_neighbors=1,
        numeric_features=("amount",),
        categorical_features=(),
        blocks={"ink": ("amount",)},
    ).fit(X, y)
    assert model.numeric_ranges_["amount"] == pytest.approx(1.0)
    model.predict(pd.DataFrame({"amount": [1000.0]}))
    assert model.numeric_ranges_["amount"] == pytest.approx(1.0)


def test_gaussian_scores_are_finite_and_best_at_observation() -> None:
    observed = np.log10(np.array([60.0, 120.0]))
    sd = np.array([0.1, 0.1])
    nlpd_at_observed, crps_at_observed = gaussian_log_scores(observed, observed, sd)
    nlpd_shifted, crps_shifted = gaussian_log_scores(observed, observed + 0.5, sd)
    assert np.isfinite(nlpd_at_observed).all()
    assert np.isfinite(crps_at_observed).all()
    assert (nlpd_at_observed < nlpd_shifted).all()
    assert (crps_at_observed < crps_shifted).all()

    nlpd_original, crps_original = lognormal_original_scale_scores(
        10**observed, observed, sd
    )
    _, crps_original_shifted = lognormal_original_scale_scores(
        10**observed, observed + 0.5, sd
    )
    assert np.isfinite(nlpd_original).all()
    assert np.isfinite(crps_original).all()
    assert (crps_original < crps_original_shifted).all()
