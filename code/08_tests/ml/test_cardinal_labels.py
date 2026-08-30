from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.cardinal_labels import (
    HIERARCHICAL_CLASS_LABELS,
    HIERARCHICAL_SOFT_LABEL_COLUMNS,
    derive_hierarchical_soft_labels,
    derive_posterior_soft_labels,
    hierarchical_support_summary,
    low_anchor_subdivision_table,
    support_summary,
    validate_hierarchical_soft_labels,
    validate_soft_labels,
)


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_record": [1, 2],
            "record_id": [1, 2],
            "curve_id": ["sr001", "sr002"],
            "cv_group_id": ["g1", "g2"],
            "electrolyte_scope": ["alkaline", "acidic"],
        }
    )


def test_primary_bands_overlap_resolution_and_off_cardinal() -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [1] * 8 + [2] * 8,
            "primary_slope_mV_dec": [
                24.0,  # inclusive 30 lower bound
                32.0,  # 30/40 overlap; closer to 30 on log scale
                np.sqrt(30.0 * 40.0),  # exact log-distance tie -> lower anchor
                36.0,  # 30/40 overlap; closer to 40 on log scale
                48.0,  # inclusive 40/60 boundary; closer to 40
                72.0,  # inclusive 60 upper bound
                96.0,  # inclusive 120 lower bound
                0.0,  # nonpositive -> off
                -1.0,  # nonpositive -> off
                23.999,
                48.001,
                72.001,
                95.999,
                144.0,  # inclusive 120 upper bound
                144.001,
                60.0,
            ],
        }
    )
    result = derive_posterior_soft_labels(
        draws, _mapping(), expected_draws_per_record=8
    )

    first = result.set_index("sample_record").loc[1]
    assert first["q_30_like"] == pytest.approx(3 / 8)
    assert first["q_40_like"] == pytest.approx(2 / 8)
    assert first["q_60_like"] == pytest.approx(1 / 8)
    assert first["q_120_like"] == pytest.approx(1 / 8)
    assert first["q_off_cardinal"] == pytest.approx(1 / 8)
    assert first["n_nonpositive_draws"] == 1
    assert first["modal_label"] == "30_like"

    second = result.set_index("sample_record").loc[2]
    assert second["q_30_like"] == 0
    assert second["q_40_like"] == 0
    assert second["q_60_like"] == pytest.approx(2 / 8)
    assert second["q_120_like"] == pytest.approx(1 / 8)
    assert second["q_off_cardinal"] == pytest.approx(5 / 8)
    assert second["n_nonpositive_draws"] == 1
    assert second["modal_label"] == "off_cardinal"
    assert list(result.columns[:5]) == list(_mapping().columns)
    validate_soft_labels(result)


def test_nonfinite_draws_fail_instead_of_being_dropped() -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [1, 2],
            "primary_slope_mV_dec": [40.0, np.nan],
        }
    )
    with pytest.raises(ValueError, match="nonfinite"):
        derive_posterior_soft_labels(draws, _mapping())


def test_mapping_and_expected_draw_counts_are_strict() -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [1, 1, 2],
            "primary_slope_mV_dec": [40.0, 60.0, 120.0],
        }
    )
    with pytest.raises(ValueError, match="expected_draws_per_record=2"):
        derive_posterior_soft_labels(
            draws, _mapping(), expected_draws_per_record=2
        )
    with pytest.raises(ValueError, match="absent from mapping"):
        derive_posterior_soft_labels(
            pd.DataFrame(
                {
                    "sample_record": [1, 2, 3],
                    "primary_slope_mV_dec": [40.0, 60.0, 120.0],
                }
            ),
            _mapping(),
        )


def test_validate_soft_labels_rejects_inconsistent_derived_fields() -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [1, 2],
            "primary_slope_mV_dec": [40.0, 60.0],
        }
    )
    result = derive_posterior_soft_labels(draws, _mapping())
    broken = result.copy()
    broken.loc[0, "q_30_like"] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        validate_soft_labels(broken)

    broken = result.copy()
    broken.loc[0, "modal_label"] = "off_cardinal"
    with pytest.raises(ValueError, match="modal_label"):
        validate_soft_labels(broken)


def test_support_summary_reports_effective_mass_and_modal_groups() -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [1, 1, 2, 2],
            "primary_slope_mV_dec": [40.0, 60.0, 120.0, 200.0],
        }
    )
    labels = derive_posterior_soft_labels(
        draws, _mapping(), expected_draws_per_record=2
    )
    summary = support_summary(labels)

    assert set(summary["population"]) == {"all", "alkaline", "acidic"}
    all_rows = summary.loc[summary["population"].eq("all")].set_index(
        "class_label"
    )
    assert all_rows.loc["40_like", "effective_posterior_mass"] == pytest.approx(
        0.5
    )
    assert all_rows.loc["60_like", "effective_posterior_mass"] == pytest.approx(
        0.5
    )
    assert all_rows.loc["120_like", "effective_posterior_mass"] == pytest.approx(
        0.5
    )
    assert all_rows.loc[
        "off_cardinal", "effective_posterior_mass"
    ] == pytest.approx(0.5)
    assert all_rows.loc["40_like", "modal_record_count"] == 1
    assert all_rows.loc["120_like", "modal_record_count"] == 1
    assert all_rows.loc["40_like", "modal_group_count"] == 1


def test_hierarchical_target_collapses_30_and_40_without_losing_audit_mass() -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [1, 1, 1, 1, 2, 2, 2, 2],
            "primary_slope_mV_dec": [
                30.0,
                40.0,
                60.0,
                200.0,
                30.0,
                40.0,
                120.0,
                0.0,
            ],
        }
    )
    five_class = derive_posterior_soft_labels(
        draws, _mapping(), expected_draws_per_record=4
    )
    hierarchical = derive_hierarchical_soft_labels(five_class)

    np.testing.assert_allclose(
        hierarchical["q_low_30_40_like"],
        hierarchical["q_30_like"] + hierarchical["q_40_like"],
    )
    np.testing.assert_allclose(
        hierarchical.loc[:, HIERARCHICAL_SOFT_LABEL_COLUMNS].sum(axis=1),
        1.0,
    )
    assert tuple(HIERARCHICAL_CLASS_LABELS) == (
        "low_30_40_like",
        "60_like",
        "120_like",
        "off_cardinal",
    )
    assert {"q_30_like", "q_40_like"}.issubset(hierarchical.columns)
    validate_hierarchical_soft_labels(hierarchical)

    support = hierarchical_support_summary(hierarchical)
    all_support = support.loc[support["population"].eq("all")].set_index(
        "class_label"
    )
    assert all_support.loc[
        "low_30_40_like", "effective_posterior_mass"
    ] == pytest.approx(1.0)


def test_low_anchor_subdivision_is_audit_only_conditional_posterior() -> None:
    draws = pd.DataFrame(
        {
            "sample_record": [1, 1, 2, 2],
            "primary_slope_mV_dec": [30.0, 40.0, 60.0, 120.0],
        }
    )
    labels = derive_posterior_soft_labels(
        draws, _mapping(), expected_draws_per_record=2
    )
    subdivision = low_anchor_subdivision_table(labels).set_index("sample_record")

    assert subdivision.loc[1, "q_low_30_40_like"] == pytest.approx(1.0)
    assert subdivision.loc[1, "q_30_given_low"] == pytest.approx(0.5)
    assert subdivision.loc[1, "q_40_given_low"] == pytest.approx(0.5)
    assert subdivision.loc[2, "q_low_30_40_like"] == 0.0
    assert np.isnan(subdivision.loc[2, "q_30_given_low"])
    assert np.isnan(subdivision.loc[2, "q_40_given_low"])


def test_current_97_golden_effective_mass_and_modal_counts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    draws_path = repo_root / "Data/02_inputs/canonical_dataset_ir_corrected_v2/posterior_slope_draws.csv.gz"
    mapping_path = (
        repo_root
        / "Data/03_outputs/study/outputs/current_97/targets/targets_wide.csv"
    )
    draws = pd.read_csv(
        draws_path, usecols=["sample_record", "primary_slope_mV_dec"]
    )
    mapping = pd.read_csv(
        mapping_path,
        usecols=[
            "record_id",
            "sample_record",
            "curve_id",
            "cv_group_id",
            "electrolyte_scope",
        ],
    )
    labels = derive_posterior_soft_labels(
        draws, mapping, expected_draws_per_record=8_000
    )
    summary = support_summary(labels)
    all_rows = summary.loc[summary["population"].eq("all")].set_index(
        "class_label"
    )

    expected_mass = {
        "30_like": 4.487,
        "40_like": 12.075,
        "60_like": 30.189,
        "120_like": 16.809,
        "off_cardinal": 33.440,
    }
    expected_modal = {
        "30_like": 0,
        "40_like": 8,
        "60_like": 47,
        "120_like": 16,
        "off_cardinal": 26,
    }
    assert all_rows["effective_posterior_mass"].to_dict() == pytest.approx(
        expected_mass, abs=1e-3
    )
    assert all_rows["modal_record_count"].to_dict() == expected_modal
    assert all_rows.loc["30_like", "n_class_probability_ge_0_5"] == 0
    assert all_rows.loc["30_like", "n_class_probability_ge_0_8"] == 0
    assert all_rows.loc["30_like", "max_class_probability"] == pytest.approx(
        0.3359, abs=1e-4
    )
    assert labels["n_nonpositive_draws"].sum() == 1_760
    assert labels["max_probability"].median() == pytest.approx(0.513, abs=1e-3)

    hierarchical = derive_hierarchical_soft_labels(labels)
    hierarchical_support = hierarchical_support_summary(hierarchical)
    hierarchical_all = hierarchical_support.loc[
        hierarchical_support["population"].eq("all")
    ].set_index("class_label")
    assert hierarchical_all["effective_posterior_mass"].to_dict() == pytest.approx(
        {
            "low_30_40_like": 16.562,
            "60_like": 30.189,
            "120_like": 16.809,
            "off_cardinal": 33.440,
        },
        abs=1e-3,
    )
