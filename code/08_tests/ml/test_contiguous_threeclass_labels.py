from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.contiguous_threeclass_labels import (
    CONTIGUOUS_THREE_CLASS_Q_COLUMNS,
    HIGH_OFF_BOUNDARY_MV_DEC,
    LOW_120_LOG_MIDPOINT_MV_DEC,
    derive_contiguous_threeclass_soft_labels,
    validate_contiguous_threeclass_soft_labels,
)


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["r1", "r2"],
            "sample_record": [1, 2],
            "curve_id": ["c1", "c2"],
            "cv_group_id": ["g1", "g2"],
            "electrolyte_scope": ["alkaline", "acidic"],
        }
    )


def test_contiguous_boundaries_assign_every_draw_once() -> None:
    lower = LOW_120_LOG_MIDPOINT_MV_DEC
    upper = HIGH_OFF_BOUNDARY_MV_DEC
    draws = pd.DataFrame(
        {
            "sample_record": [1, 1, 1, 1, 2, 2, 2, 2],
            "primary_slope_mV_dec": [
                lower - 1e-9,
                lower,
                upper,
                upper + 1e-9,
                60.0,
                84.0,
                120.0,
                200.0,
            ],
        }
    )
    labels = derive_contiguous_threeclass_soft_labels(
        draws,
        _mapping(),
        expected_draws_per_record=4,
    )

    assert labels.loc[0, list(CONTIGUOUS_THREE_CLASS_Q_COLUMNS)].tolist() == [
        0.25,
        0.50,
        0.25,
    ]
    assert labels.loc[1, list(CONTIGUOUS_THREE_CLASS_Q_COLUMNS)].tolist() == [
        0.50,
        0.25,
        0.25,
    ]
    assert np.allclose(
        labels.loc[:, CONTIGUOUS_THREE_CLASS_Q_COLUMNS].sum(axis=1), 1.0
    )
    validate_contiguous_threeclass_soft_labels(labels)


def test_contiguous_labels_preserve_mapping_order() -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [2, 1, 2, 1],
            "primary_slope_mV_dec": [200.0, 60.0, 200.0, 60.0],
        }
    )
    labels = derive_contiguous_threeclass_soft_labels(
        draws,
        _mapping(),
        expected_draws_per_record=2,
    )
    assert labels["record_id"].tolist() == ["r1", "r2"]
    assert labels["threeclass_modal_label"].tolist() == [
        "low_60_side",
        "high_off",
    ]


def test_nonpositive_finite_draws_are_low_side_and_audited() -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [1, 1, 2, 2],
            "primary_slope_mV_dec": [0.0, -1.0, 120.0, 200.0],
        }
    )
    labels = derive_contiguous_threeclass_soft_labels(
        draws,
        _mapping(),
        expected_draws_per_record=2,
    )
    assert labels.loc[0, "q_low_60_side"] == 1.0
    assert labels.loc[0, "n_nonpositive_draws"] == 2
    assert labels.loc[1, "n_nonpositive_draws"] == 0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_contiguous_labels_fail_closed_on_invalid_draws(
    bad_value: float,
) -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [1, 2],
            "primary_slope_mV_dec": [60.0, bad_value],
        }
    )
    with pytest.raises(ValueError):
        derive_contiguous_threeclass_soft_labels(
            draws,
            _mapping(),
            expected_draws_per_record=1,
        )
