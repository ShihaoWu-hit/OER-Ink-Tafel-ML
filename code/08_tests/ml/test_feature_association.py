from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.config import load_config
from ink_tafel_study.feature_association import (
    TARGET_NAME,
    FeatureAssociationEvaluationConfig,
    _bootstrap_effect_grid,
    _holm_adjust,
    _joint_support_query,
    _joint_support_state,
    _validate_frozen_settings,
    build_blockwise_comparison,
    build_posterior_association_atlas,
    build_posterior_target_summary,
    score_grouped_predictions,
)


def _formal_targets(record_ids: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": list(record_ids),
            "sample_record": [f"sr{value}" for value in record_ids],
            "curve_id": [f"curve_{value}" for value in record_ids],
            "target_name": TARGET_NAME,
            "target_value_mV_dec": [40.0 + 20.0 * i for i in range(len(record_ids))],
            "target_interval_low_mV_dec": [
                35.0 + 20.0 * i for i in range(len(record_ids))
            ],
            "target_interval_high_mV_dec": [
                45.0 + 20.0 * i for i in range(len(record_ids))
            ],
        }
    )


def _posterior_draws(record_ids: tuple[str, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for i, record_id in enumerate(record_ids):
        values = np.linspace(35.0 + 20.0 * i, 45.0 + 20.0 * i, 8_000)
        if i == 0:
            # The first draw is deliberately selected by the deterministic
            # score grid and therefore exercises both scale-specific policies.
            values[0] = -1.0
        rows.append(
            pd.DataFrame(
                {
                    "sample_record": f"sr{record_id}",
                    "posterior_draw_index": np.arange(8_000),
                    "target_name": TARGET_NAME,
                    "primary_slope_mV_dec": values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_posterior_target_summary_keeps_one_training_row_per_record() -> None:
    targets = _formal_targets(("1", "2"))
    summary, score_draws, qa = build_posterior_target_summary(
        targets,
        _posterior_draws(("1", "2")),
        draw_repetitions=2,
    )

    assert len(summary) == 2
    assert summary["record_id"].is_unique
    assert summary["n_posterior_draws"].eq(8_000).all()
    assert len(score_draws) == 4
    assert score_draws.groupby("record_id").size().eq(2).all()
    assert qa["posterior_draws_expanded_as_ml_samples"] is False
    assert qa["training_rows_per_record"] == 1

    nonpositive = score_draws.loc[score_draws["target_draw_mV_dec"].le(0)].iloc[0]
    assert nonpositive["target_draw_mV_dec"] == -1.0
    assert np.isnan(nonpositive["target_draw_log10"])
    assert not bool(nonpositive["positive_for_log_metrics"])
    assert qa["nonpositive_draw_policy_original_scale"] == "retained"
    assert (
        qa["nonpositive_draw_policy_log_metrics"]
        == "excluded_and_mass_reported"
    )


def test_posterior_aware_atlas_has_one_row_per_scope_record_feature() -> None:
    record_ids = ("1", "2", "3")
    targets = _formal_targets(record_ids)
    target_summary, score_draws, _ = build_posterior_target_summary(
        targets,
        _posterior_draws(record_ids),
        draw_repetitions=2,
    )
    frame = target_summary.assign(
        cv_group_id=["g1", "g2", "g3"],
        catalyst_family=["IrO2_plus_Ir", "RuO2", "RuO2"],
        electrolyte_scope=["alkaline", "acidic", "neutral"],
        numeric_feature=[1.0, 2.0, 3.0],
        categorical_feature=["a", "b", "a"],
    )
    candidate_registry = pd.DataFrame(
        {
            "selected_column": ["numeric_feature", "categorical_feature"],
            "logical_feature": ["numeric", "categorical"],
            "kind": ["numeric", "categorical"],
        }
    )
    feature_support = pd.DataFrame(
        [
            {
                "scope": scope,
                "feature_name": feature,
                "support_status": "supported",
                "atlas_status": "eligible",
            }
            for scope in ("all97",)
            for feature in ("numeric_feature", "categorical_feature")
        ]
    )

    observations, summary, draw_statistics, qa = (
        build_posterior_association_atlas(
            frame,
            score_draws,
            candidate_registry,
            feature_support,
        )
    )

    keys = ["scope", "record_id", "feature_name"]
    assert not observations.duplicated(keys).any()
    assert set(observations["scope"]) == {"all97"}
    assert len(observations) == len(record_ids) * 2
    assert len(summary) == 2
    assert len(draw_statistics) == 2 * 2
    assert observations["target_q025_mV_dec"].notna().all()
    assert observations["target_q975_mV_dec"].notna().all()
    assert not summary["fitted_smoother_used"].any()
    assert qa["posterior_draws_used_as_independent_ml_samples"] is False
    assert qa["fitted_smoother_used"] is False


def test_holm_adjustment_is_monotone_and_preserves_missing_values() -> None:
    adjusted = _holm_adjust(
        {"block_a": 0.01, "block_b": 0.04, "block_c": 0.03, "missing": np.nan}
    )

    assert adjusted["block_a"] == 0.03
    assert adjusted["block_c"] == 0.06
    assert adjusted["block_b"] == 0.06
    assert np.isnan(adjusted["missing"])
    assert adjusted["block_a"] <= adjusted["block_c"] <= adjusted["block_b"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("models", "gam", "spline_knots"), 4),
        (("validation", "outer_repeats"), 9),
        (("posterior_policy", "score_level_target_draw_repetitions"), 50),
        (("effects", "extrapolation_allowed"), True),
        (("release_gate", "gam_only_rescue_allowed"), True),
        (
            (
                "feature_reduction",
                "outcome_driven_feature_selection_used",
            ),
            True,
        ),
    ],
)
def test_scientific_feature_association_settings_are_exactly_locked(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    settings = deepcopy(load_config().section("feature_association"))
    target = settings
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(ValueError, match="Frozen feature-association config"):
        _validate_frozen_settings(settings)


def test_joint_support_uses_training_only_block_balanced_gower() -> None:
    train = pd.DataFrame(
        {
            "context": [0.0, 0.0, 1.0, 1.0],
            "amount": [0.0, 0.1, 0.9, 1.0],
        }
    )
    state = _joint_support_state(
        train,
        {"context": ("context",), "amount": ("amount",)},
        threshold_quantile=0.95,
    )
    supported, distances = _joint_support_query(
        pd.DataFrame(
            {
                "context": [0.0, 0.5],
                "amount": [0.05, 0.5],
            }
        ),
        state,
    )

    assert supported.tolist() == [True, False]
    assert distances[0] <= state["threshold"]
    assert distances[1] > state["threshold"]


def _comparison_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panels = {
        "prior": 10.0,
        "context_core": 9.0,
        "context_plus_catalyst_concentration": 8.0,
        "context_plus_ionomer_ratio": 8.1,
        "context_plus_water_organic": 7.8,
        "context_plus_ethanol_isopropanol": 7.7,
        "context_plus_viscosity": 8.3,
        "context_plus_polarity": 8.2,
        "context_plus_conductive_sensitivity": 8.5,
    }
    scored_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    for scope in ("all97",):
        for model in ("bayesian_linear", "gam_low_df"):
            for panel, metric in panels.items():
                for record_id, group in (("1", "g1"), ("2", "g2"), ("3", "g2")):
                    scored_rows.append(
                        {
                            "scope": scope,
                            "model": model,
                            "panel": panel,
                            "record_id": record_id,
                            "cv_group_id": group,
                            "posterior_expected_mae_mV_dec": metric,
                        }
                    )
                for repeat in (0, 1):
                    split_rows.append(
                        {
                            "scope": scope,
                            "model": model,
                            "panel": panel,
                            "repeat": repeat,
                            "posterior_expected_mae_mV_dec": metric,
                        }
                    )
                for repetition in (0, 1):
                    draw_rows.append(
                        {
                            "scope": scope,
                            "model": model,
                            "panel": panel,
                            "target_draw_repetition": repetition,
                            "posterior_draw_mae_mV_dec": metric,
                        }
                    )
    return (
        pd.DataFrame(scored_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(draw_rows),
    )


def test_smoke_gate_never_releases_or_emits_adjusted_effects() -> None:
    scored, split_metrics, draw_metrics = _comparison_inputs()
    config = FeatureAssociationEvaluationConfig(
        outer_folds=2,
        outer_repeats=2,
        outer_seeds=(1, 2),
        bootstrap_replicates=5,
        target_draw_repetitions=2,
        smoke=True,
    )
    _, summary, decisions = build_blockwise_comparison(
        scored,
        split_metrics,
        draw_metrics,
        config=config,
    )

    assert decisions["gate_status"].eq("smoke_not_scientific").all()
    assert not decisions["block_gate_passed"].any()
    assert not decisions["effect_release_pass"].any()
    context = summary.loc[
        summary["scope"].eq("all97")
        & summary["contrast_id"].eq("context_minus_prior")
        & summary["model"].eq("bayesian_linear")
    ].sort_values("comparison_kind")
    assert len(context) == 2
    assert context["observed_delta"].nunique() == 1
    assert context["ci95_low"].nunique() == 1
    assert context["ci95_high"].nunique() == 1

    candidates = pd.DataFrame(
        {
            "scope": ["all97"],
            "panel": ["context_plus_catalyst_concentration"],
            "block_id": ["catalyst_concentration"],
            "model": ["gam_low_df"],
        }
    )
    grid, release_registry = _bootstrap_effect_grid(
        candidates,
        decisions,
        config=config,
    )
    assert grid.empty
    assert not release_registry["effect_release_pass"].any()
    assert release_registry["effect_curve_status"].eq("gated").all()


def test_failed_block_does_not_emit_adjusted_effect_grid() -> None:
    config = FeatureAssociationEvaluationConfig(
        outer_folds=2,
        outer_repeats=1,
        outer_seeds=(1,),
        bootstrap_replicates=5,
        target_draw_repetitions=2,
    )
    decisions = pd.DataFrame(
        {
            "contrast_id": ["catalyst_concentration_minus_context"],
            "block_id": ["catalyst_concentration"],
            "augmented_panel": ["context_plus_catalyst_concentration"],
            "effect_release_pass": [False],
        }
    )
    candidates = pd.DataFrame(
        {
            "scope": ["all97"],
            "panel": ["context_plus_catalyst_concentration"],
            "block_id": ["catalyst_concentration"],
            "model": ["gam_low_df"],
            "record_id": ["1"],
        }
    )

    grid, release_registry = _bootstrap_effect_grid(
        candidates,
        decisions,
        config=config,
    )

    assert grid.empty
    assert release_registry.loc[0, "effect_curve_status"] == "gated"


def test_passed_block_without_supported_numeric_curve_is_not_released() -> None:
    config = FeatureAssociationEvaluationConfig(
        outer_folds=2,
        outer_repeats=1,
        outer_seeds=(1,),
        bootstrap_replicates=5,
        target_draw_repetitions=2,
    )
    decisions = pd.DataFrame(
        {
            "contrast_id": ["catalyst_concentration_minus_context"],
            "block_id": ["catalyst_concentration"],
            "augmented_panel": ["context_plus_catalyst_concentration"],
            "effect_release_pass": [True],
            "reason": ["all registered block checks passed"],
        }
    )

    grid, release_registry = _bootstrap_effect_grid(
        pd.DataFrame(),
        decisions,
        config=config,
    )

    assert grid.empty
    assert not bool(release_registry.loc[0, "effect_release_pass"])
    assert (
        release_registry.loc[0, "effect_curve_status"]
        == "passed_no_estimable_curve"
    )
    assert set(grid.columns) >= {
        "cross_fitted_g_computation",
        "effect_interpretation",
        "joint_support_common_record_coverage",
    }


def test_repeated_oof_aggregation_scores_one_row_per_record() -> None:
    predictions = pd.DataFrame(
        [
            {
                "scope": "all97",
                "repeat": repeat,
                "record_id": record_id,
                "cv_group_id": group,
                "catalyst_family": "RuO2",
                "panel": "context_core",
                "model": "bayesian_linear",
                "prediction_log10_mean": mean,
                "prediction_log10_std": 0.10,
                "prediction_median_mV_dec": 10.0**mean,
                "target_median_mV_dec": target,
            }
            for record_id, group, target, means in (
                ("1", "g1", 40.0, (1.55, 1.65)),
                ("2", "g2", 60.0, (1.75, 1.85)),
            )
            for repeat, mean in enumerate(means)
        ]
    )
    score_draws = pd.DataFrame(
        [
            {
                "record_id": record_id,
                "target_draw_repetition": repetition,
                "target_draw_mV_dec": value,
                "target_draw_log10": np.log10(value),
                "positive_for_log_metrics": True,
            }
            for record_id, values in (("1", (38.0, 42.0)), ("2", (58.0, 62.0)))
            for repetition, value in enumerate(values)
        ]
    )

    outputs = score_grouped_predictions(predictions, score_draws)
    aggregated = outputs["aggregated_predictions"]

    assert len(aggregated) == 2
    assert aggregated["record_id"].is_unique
    assert aggregated["n_oof_repeats"].eq(2).all()
    assert np.isclose(
        aggregated.loc[aggregated["record_id"].eq("1"), "prediction_log10_mean"].iloc[0],
        1.60,
    )
    assert len(outputs["target_draw_metrics"]) == 2
    assert outputs["aggregate_metrics"].loc[0, "n_records"] == 2
