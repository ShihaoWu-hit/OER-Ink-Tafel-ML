from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ink_tafel_study.cardinal_metrics import CLASS_LABELS
from ink_tafel_study.cardinal_reduced_models import (
    ReducedPanelGowerClassifier,
    ReducedPanelTwoStageMAPClassifier,
)
from ink_tafel_study.posterior_cardinal_curves import (
    FOLD_PREDICTION_COLUMNS,
    GRID_PERCENTILES,
    REFERENCE_PERCENTILE,
    aggregate_adjusted_probability_predictions,
    execute_released_adjusted_probability_curves,
    generate_fold_adjusted_probability_predictions,
    validate_adjusted_curve_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "contracts" / "posterior_cardinal_all97_v2.yaml"
)


def _contract() -> dict[str, object]:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def _raw_records(n_rows: int = 30) -> pd.DataFrame:
    index = np.arange(n_rows)
    return pd.DataFrame(
        {
            "record_id": [f"r{value:03d}" for value in index],
            "cv_group_id": [f"g{value:03d}" for value in index],
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
                index % 7 == 0, "acidic", "alkaline"
            ),
            "catalyst_concentration_mg_mL": 1.0 + 0.25 * index,
        }
    )


def _soft_targets(n_rows: int) -> np.ndarray:
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
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def test_adjusted_curve_contract_is_exactly_locked() -> None:
    contract = _contract()
    validate_adjusted_curve_contract(contract)

    changed = deepcopy(contract)
    changed["adjusted_probability_curve"][
        "minimum_supported_repeats_per_common_record_at_every_grid_point"
    ] = 7
    with pytest.raises(ValueError, match="policy changed"):
        validate_adjusted_curve_contract(changed)


def test_fold_predictions_use_training_percentiles_and_preserve_simplex() -> None:
    records = _raw_records()
    train = records.iloc[:22].reset_index(drop=True)
    test = records.iloc[22:].reset_index(drop=True)
    source_columns = [
        name
        for name in train.columns
        if name not in {"record_id", "cv_group_id"}
    ]
    estimator = ReducedPanelTwoStageMAPClassifier(
        panel="context_plus_catalyst_concentration",
        feature_prior_scale=0.5,
    ).fit(
        train.loc[:, source_columns],
        _soft_targets(len(train)),
    )

    predictions = generate_fold_adjusted_probability_predictions(
        estimator,
        train,
        test,
        panel_name="context_plus_catalyst_concentration",
        repeat=2,
        fold=3,
    )

    assert tuple(predictions.columns) == FOLD_PREDICTION_COLUMNS
    assert len(predictions) == len(test) * 19 * 4
    assert predictions["cross_fitted"].all()
    assert predictions["empirical_standardization"].all()
    assert tuple(
        sorted(predictions["grid_percentile"].unique())
    ) == GRID_PERCENTILES
    assert predictions["reference_percentile"].eq(
        REFERENCE_PERCENTILE
    ).all()
    probability_sums = predictions.groupby(
        ["record_id", "grid_percentile"]
    )["probability"].sum()
    reference_sums = predictions.groupby(
        ["record_id", "grid_percentile"]
    )["reference_probability"].sum()
    np.testing.assert_allclose(probability_sums, 1.0, atol=1e-12)
    np.testing.assert_allclose(reference_sums, 1.0, atol=1e-12)
    reference_rows = predictions.loc[
        predictions["grid_percentile"].eq(REFERENCE_PERCENTILE)
    ]
    np.testing.assert_allclose(
        reference_rows["probability_difference_from_reference"],
        0.0,
        atol=1e-12,
    )

    engineered = estimator.feature_builder_.transform(train)
    feature = "log10_catalyst_concentration_mg_mL"
    expected = float(engineered[feature].quantile(0.05))
    observed = predictions.loc[
        predictions["grid_percentile"].eq(0.05), "grid_value_fold"
    ].unique()
    np.testing.assert_allclose(observed, [expected], atol=1e-12)
    assert predictions["joint_support_threshold_quantile"].eq(0.95).all()


def test_fold_predictions_reject_non_primary_estimator() -> None:
    records = _raw_records()
    train = records.iloc[:22].reset_index(drop=True)
    test = records.iloc[22:].reset_index(drop=True)
    source_columns = [
        name
        for name in train.columns
        if name not in {"record_id", "cv_group_id"}
    ]
    estimator = ReducedPanelGowerClassifier(
        panel="context_plus_catalyst_concentration",
        k=5,
    ).fit(train.loc[:, source_columns], _soft_targets(len(train)))
    with pytest.raises(TypeError, match="ReducedPanelTwoStageMAPClassifier"):
        generate_fold_adjusted_probability_predictions(
            estimator,  # type: ignore[arg-type]
            train,
            test,
            panel_name="context_plus_catalyst_concentration",
            repeat=0,
            fold=0,
        )


def _synthetic_fold_predictions(
    *,
    excluded_records: int,
    unsupported_repeats: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record_index in range(97):
        record_id = f"r{record_index:03d}"
        group_id = f"g{min(record_index, 79):03d}"
        for repeat in range(10):
            for percentile in GRID_PERCENTILES:
                shift = 0.10 * (percentile - REFERENCE_PERCENTILE)
                probability = np.asarray(
                    [
                        0.25 + shift,
                        0.25 - 0.5 * shift,
                        0.25,
                        0.25 - 0.5 * shift,
                    ]
                )
                reference = np.full(4, 0.25)
                supported = not (
                    record_index < excluded_records
                    and percentile == GRID_PERCENTILES[0]
                    and repeat < unsupported_repeats
                )
                for class_index, class_label in enumerate(CLASS_LABELS):
                    rows.append(
                        {
                            "panel": (
                                "context_plus_catalyst_concentration"
                            ),
                            "feature": (
                                "log10_catalyst_concentration_mg_mL"
                            ),
                            "repeat": repeat,
                            "fold": record_index % 5,
                            "record_id": record_id,
                            "cv_group_id": group_id,
                            "grid_percentile": percentile,
                            "grid_value_fold": percentile + repeat / 1000,
                            "reference_percentile": 0.50,
                            "reference_value_fold": 0.50 + repeat / 1000,
                            "class_label": class_label,
                            "probability": probability[class_index],
                            "reference_probability": reference[class_index],
                            "probability_difference_from_reference": (
                                probability[class_index]
                                - reference[class_index]
                            ),
                            "grid_supported": supported,
                            "reference_supported": True,
                            "joint_support": supported,
                            "grid_nearest_distance": 0.1,
                            "reference_nearest_distance": 0.1,
                            "joint_support_threshold": 0.2,
                            "joint_support_threshold_quantile": 0.95,
                            "n_outer_training_records": 77,
                            "cross_fitted": True,
                            "empirical_standardization": True,
                        }
                    )
    return pd.DataFrame(rows, columns=FOLD_PREDICTION_COLUMNS)


def test_common_record_curve_uses_eight_repeats_and_one_fixed_set() -> None:
    candidates = _synthetic_fold_predictions(excluded_records=19)
    result = aggregate_adjusted_probability_predictions(candidates)

    assert len(result.curves) == 19 * 4
    assert result.audit.loc[0, "status"] == "released"
    assert result.audit.loc[0, "n_common_records"] == 78
    assert result.audit.loc[0, "common_record_coverage"] == pytest.approx(
        78 / 97
    )
    assert result.audit.loc[
        0, "minimum_supported_repeats_per_record"
    ] == 10
    assert result.curves["n_common_records"].eq(78).all()
    assert result.curves["minimum_supported_repeats_per_record"].min() >= 8
    probability_sums = result.curves.groupby(
        "grid_percentile"
    )["probability"].sum()
    reference_sums = result.curves.groupby(
        "grid_percentile"
    )["reference_probability"].sum()
    difference_sums = result.curves.groupby(
        "grid_percentile"
    )["probability_difference_from_reference"].sum()
    np.testing.assert_allclose(probability_sums, 1.0, atol=1e-12)
    np.testing.assert_allclose(reference_sums, 1.0, atol=1e-12)
    np.testing.assert_allclose(difference_sums, 0.0, atol=1e-12)
    reference = result.curves.loc[
        result.curves["grid_percentile"].eq(REFERENCE_PERCENTILE)
    ]
    np.testing.assert_allclose(
        reference["probability_difference_from_reference"],
        0.0,
        atol=1e-12,
    )


def test_exactly_eight_supported_repeats_remain_in_common_set() -> None:
    candidates = _synthetic_fold_predictions(
        excluded_records=97,
        unsupported_repeats=2,
    )
    result = aggregate_adjusted_probability_predictions(candidates)

    assert result.audit.loc[0, "status"] == "released"
    assert result.audit.loc[0, "n_common_records"] == 97
    assert result.audit.loc[
        0, "minimum_supported_repeats_per_record"
    ] == 8
    assert result.curves["n_common_records"].eq(97).all()
    assert result.curves[
        "minimum_supported_repeats_per_record"
    ].min() == 8


def test_common_record_coverage_below_eighty_percent_withholds_curve() -> None:
    candidates = _synthetic_fold_predictions(excluded_records=20)
    result = aggregate_adjusted_probability_predictions(candidates)

    assert result.curves.empty
    assert result.audit.loc[0, "status"] == "passed_no_estimable_curve"
    assert result.audit.loc[0, "n_common_records"] == 77
    assert result.audit.loc[0, "common_record_coverage"] == pytest.approx(
        77 / 97
    )


def test_executor_does_not_fit_when_no_release_is_authorized() -> None:
    decisions = pd.DataFrame(
        {
            "panel": ["context_plus_catalyst_concentration"],
            "role": ["formal_single_feature_increment"],
            "primary_release_gate_pass": [False],
            "general_cardinal_region_claim_pass": [False],
            "adjusted_probability_curve_release_pass": [False],
            "width_direction_consistent": [False],
        }
    )
    result = execute_released_adjusted_probability_curves(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        decisions,
        _contract(),
    )

    assert result.fold_predictions.empty
    assert result.curves.empty
    assert result.audit.loc[0, "status"] == "no_release_authorized"
