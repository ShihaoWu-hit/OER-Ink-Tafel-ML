import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.posterior_cardinal_inference import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_SIGNFLIP_SEED,
    connected_group_signflip_pvalues,
    descriptive_probability_metrics,
    holm_one_sided_adjustment,
    paired_connected_group_bootstrap,
)


def test_inference_seeds_match_frozen_contract() -> None:
    assert DEFAULT_BOOTSTRAP_SEED == 20260724
    assert DEFAULT_SIGNFLIP_SEED == 20260725


def _probability_examples() -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(
        [
            [0.70, 0.10, 0.10, 0.10],
            [0.10, 0.60, 0.20, 0.10],
            [0.10, 0.20, 0.60, 0.10],
            [0.10, 0.20, 0.10, 0.60],
        ],
        dtype=float,
    )
    predicted = np.asarray(
        [
            [0.60, 0.20, 0.10, 0.10],
            [0.15, 0.55, 0.20, 0.10],
            [0.05, 0.15, 0.70, 0.10],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=float,
    )
    return observed, predicted


def test_descriptive_probability_metrics_preserve_cross_entropy_identity() -> None:
    observed, predicted = _probability_examples()
    metrics = descriptive_probability_metrics(
        observed,
        predicted,
        entropy_threshold=0.90,
    )

    assert metrics["soft_log_loss"] == pytest.approx(
        metrics["target_entropy"] + metrics["excess_cross_entropy_KL"],
        abs=1e-15,
    )
    direct_mean_kl = np.mean(
        np.sum(observed * np.log(observed / predicted), axis=1)
    )
    assert metrics["excess_cross_entropy_KL"] == pytest.approx(
        direct_mean_kl
    )
    assert metrics["modal_accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["posterior_expected_hard_accuracy"] == pytest.approx(
        (0.70 + 0.60 + 0.60 + 0.10) / 4
    )
    assert metrics["oracle_posterior_expected_hard_accuracy"] == pytest.approx(
        (0.70 + 0.60 + 0.60 + 0.60) / 4
    )
    assert metrics["uncertain_count"] == 1
    assert metrics["uncertain_fraction"] == pytest.approx(0.25)
    assert 0.0 <= metrics["mean_normalized_predictive_entropy"] <= 1.0


def test_connected_group_bootstrap_repeats_all_records_with_sampled_group() -> None:
    deltas = pd.DataFrame(
        {
            "cv_group_id": ["g1", "g1", "g2"],
            "delta_log_loss": [0.0, 2.0, 10.0],
            "delta_brier": [0.0, 4.0, 20.0],
        }
    )
    replicates = 12
    seed = 17
    draws, summary = paired_connected_group_bootstrap(
        deltas,
        ["delta_log_loss", "delta_brier"],
        replicates=replicates,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    group_records = (
        np.asarray([0, 1], dtype=int),
        np.asarray([2], dtype=int),
    )
    expected = []
    expected_record_counts = []
    values = deltas[["delta_log_loss", "delta_brier"]].to_numpy(float)
    for _ in range(replicates):
        sampled_groups = rng.integers(0, 2, size=2)
        sampled_records = np.concatenate(
            [group_records[position] for position in sampled_groups]
        )
        expected.append(values[sampled_records].mean(axis=0))
        expected_record_counts.append(len(sampled_records))

    np.testing.assert_allclose(
        draws[["delta_log_loss", "delta_brier"]],
        np.asarray(expected),
    )
    np.testing.assert_allclose(
        draws["delta_brier"],
        2.0 * draws["delta_log_loss"],
    )
    np.testing.assert_array_equal(
        draws["n_sampled_records"],
        expected_record_counts,
    )
    assert draws["n_sampled_groups"].eq(2).all()
    assert summary["n_connected_groups"].eq(2).all()
    assert summary["n_records"].eq(3).all()


def test_connected_group_bootstrap_samples_eighty_groups_and_is_deterministic() -> None:
    deltas = pd.DataFrame(
        {
            "cv_group_id": [f"g{index:02d}" for index in range(80)],
            "delta": np.linspace(-1.0, 1.0, 80),
        }
    )
    first_draws, first_summary = paired_connected_group_bootstrap(
        deltas,
        ["delta"],
        replicates=25,
        seed=101,
    )
    second_draws, second_summary = paired_connected_group_bootstrap(
        deltas,
        ["delta"],
        replicates=25,
        seed=101,
    )

    pd.testing.assert_frame_equal(first_draws, second_draws)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert first_draws["n_sampled_groups"].eq(80).all()
    assert first_draws["n_sampled_records"].eq(80).all()


def test_signflip_uses_shared_group_signs_and_lower_tail_direction() -> None:
    base = np.asarray([-1.4, -0.9, -1.2, -0.8, -1.1, -1.0, -0.7, -1.3])
    deltas = pd.DataFrame(
        {
            "cv_group_id": [f"g{index}" for index in range(len(base))],
            "negative_feature": base,
            "scaled_negative_feature": 2.0 * base,
            "positive_feature": -base,
        }
    )
    draws, summary = connected_group_signflip_pvalues(
        deltas,
        [
            "negative_feature",
            "scaled_negative_feature",
            "positive_feature",
        ],
        replicates=4096,
        seed=314,
    )

    np.testing.assert_allclose(
        draws["scaled_negative_feature"],
        2.0 * draws["negative_feature"],
        atol=0.0,
        rtol=0.0,
    )
    indexed = summary.set_index("feature")
    assert indexed.loc["negative_feature", "p_value_lower_tail"] < 0.05
    assert indexed.loc["scaled_negative_feature", "p_value_lower_tail"] < 0.05
    assert indexed.loc["positive_feature", "p_value_lower_tail"] > 0.95
    assert summary["shared_group_signs_across_features"].all()


def test_signflip_includes_ties_and_is_deterministic() -> None:
    deltas = pd.DataFrame(
        {
            "cv_group_id": ["g1", "g2", "g3"],
            "zero_delta": [0.0, 0.0, 0.0],
        }
    )
    first_draws, first_summary = connected_group_signflip_pvalues(
        deltas,
        ["zero_delta"],
        replicates=31,
        seed=11,
    )
    second_draws, second_summary = connected_group_signflip_pvalues(
        deltas,
        ["zero_delta"],
        replicates=31,
        seed=11,
    )

    pd.testing.assert_frame_equal(first_draws, second_draws)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    row = first_summary.iloc[0]
    assert row["lower_tail_count_including_ties"] == 31
    assert row["p_value_lower_tail"] == 1.0


def test_holm_one_sided_adjustment_is_step_down_and_monotone() -> None:
    adjusted = holm_one_sided_adjustment(
        {
            "feature_a": 0.01,
            "feature_b": 0.04,
            "feature_c": 0.03,
        }
    )

    assert adjusted == pytest.approx(
        {
            "feature_a": 0.03,
            "feature_b": 0.06,
            "feature_c": 0.06,
        }
    )


@pytest.mark.parametrize(
    ("callable_name", "factory"),
    [
        (
            "descriptive observed",
            lambda: descriptive_probability_metrics(
                np.asarray([[np.nan, 0.5, 0.5, 0.0]]),
                np.asarray([[0.25, 0.25, 0.25, 0.25]]),
            ),
        ),
        (
            "descriptive predicted",
            lambda: descriptive_probability_metrics(
                np.asarray([[0.25, 0.25, 0.25, 0.25]]),
                np.asarray([[np.nan, 0.5, 0.5, 0.0]]),
            ),
        ),
        (
            "descriptive threshold",
            lambda: descriptive_probability_metrics(
                np.asarray([[0.25, 0.25, 0.25, 0.25]]),
                np.asarray([[0.25, 0.25, 0.25, 0.25]]),
                entropy_threshold=np.nan,
            ),
        ),
        (
            "bootstrap",
            lambda: paired_connected_group_bootstrap(
                pd.DataFrame(
                    {
                        "cv_group_id": ["g1", "g2"],
                        "delta": [0.0, np.nan],
                    }
                ),
                ["delta"],
                replicates=2,
            ),
        ),
        (
            "bootstrap group",
            lambda: paired_connected_group_bootstrap(
                pd.DataFrame(
                    {
                        "cv_group_id": ["g1", np.nan],
                        "delta": [0.0, 1.0],
                    }
                ),
                ["delta"],
                replicates=2,
            ),
        ),
        (
            "signflip",
            lambda: connected_group_signflip_pvalues(
                pd.DataFrame(
                    {
                        "cv_group_id": ["g1", "g2"],
                        "feature": [0.0, np.nan],
                    }
                ),
                ["feature"],
                replicates=2,
            ),
        ),
        (
            "holm",
            lambda: holm_one_sided_adjustment(
                {"feature_a": 0.01, "feature_b": np.nan}
            ),
        ),
    ],
)
def test_nan_inputs_fail_closed(callable_name: str, factory: object) -> None:
    del callable_name
    with pytest.raises(ValueError, match="NaN|finite"):
        factory()
