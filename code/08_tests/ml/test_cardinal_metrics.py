import numpy as np
import pandas as pd

from ink_tafel_study.cardinal_metrics import (
    CLASS_LABELS,
    PREDICTION_COLUMNS,
    SOFT_LABEL_COLUMNS,
    aggregate_repeated_oof,
    calibration_table,
    collapse_30_and_40,
    collapsed_hard_metrics,
    expected_confusion_table,
    multiclass_brier,
    soft_cross_entropy,
    two_stage_probability_metrics,
)


def _targets() -> np.ndarray:
    return np.asarray(
        [
            [0.30, 0.40, 0.10, 0.20],
            [0.20, 0.25, 0.35, 0.20],
            [0.10, 0.20, 0.10, 0.60],
        ],
        dtype=float,
    )


def test_probability_metrics_use_soft_targets_directly() -> None:
    observed = _targets()
    uniform = np.full_like(observed, 0.25)

    assert multiclass_brier(observed, observed) == 0.0
    assert soft_cross_entropy(observed, observed) < soft_cross_entropy(
        observed, uniform
    )
    metrics = two_stage_probability_metrics(observed, observed)
    assert metrics["multiclass_brier"] == 0.0
    assert metrics["gate_brier"] == 0.0
    assert metrics["conditional_brier"] == 0.0
    assert metrics["conditional_rps"] == 0.0
    hard = collapsed_hard_metrics(observed, observed)
    assert hard["hierarchical_balanced_accuracy"] == 1.0
    assert hard["hierarchical_macro_f1"] == 0.75


def test_calibration_and_expected_confusion_preserve_all_classes() -> None:
    observed = _targets()
    calibration = calibration_table(observed, observed, n_bins=4)
    confusion = expected_confusion_table(observed, observed)

    assert set(calibration["class_label"]) == set(CLASS_LABELS)
    assert calibration.groupby("class_label")["n_records"].sum().eq(3).all()
    assert len(confusion) == len(CLASS_LABELS) ** 2
    assert np.allclose(
        confusion.groupby("target_class")["row_normalized_probability"].sum(),
        1.0,
    )


def test_repeated_oof_is_averaged_once_per_record() -> None:
    observed = _targets()[:2]
    rows = []
    for repeat in (0, 1):
        for record_position in range(2):
            row = {
                "analysis_id": "a",
                "scope": "alkaline",
                "panel": "context",
                "model_role": "joint_selected",
                "model_family": "nested_selected",
                "record_id": str(record_position + 1),
                "cv_group_id": f"g{record_position + 1}",
            }
            row.update(
                dict(
                    zip(
                        SOFT_LABEL_COLUMNS,
                        observed[record_position],
                        strict=True,
                    )
                )
            )
            prediction = (
                observed[record_position]
                if repeat == 0
                else np.full(4, 0.25)
            )
            row.update(
                dict(
                    zip(PREDICTION_COLUMNS, prediction, strict=True)
                )
            )
            rows.append(row)
    aggregated = aggregate_repeated_oof(pd.DataFrame(rows))

    assert len(aggregated) == 2
    assert aggregated["n_repeated_oof"].eq(2).all()
    assert np.allclose(
        aggregated.loc[:, PREDICTION_COLUMNS].sum(axis=1),
        1.0,
    )
    assert collapse_30_and_40(
        aggregated.loc[:, PREDICTION_COLUMNS].to_numpy(float)
    ).shape == (2, 4)


def test_registry_five_class_probabilities_collapse_exactly_to_primary_hierarchy() -> None:
    five_class = np.asarray(
        [
            [0.10, 0.20, 0.40, 0.10, 0.20],
            [0.05, 0.15, 0.25, 0.35, 0.20],
        ]
    )
    collapsed = collapse_30_and_40(five_class)

    np.testing.assert_allclose(
        collapsed,
        np.asarray(
            [
                [0.30, 0.40, 0.10, 0.20],
                [0.20, 0.25, 0.35, 0.20],
            ]
        ),
    )
    assert tuple(CLASS_LABELS) == (
        "low_30_40_like",
        "60_like",
        "120_like",
        "off_cardinal",
    )
