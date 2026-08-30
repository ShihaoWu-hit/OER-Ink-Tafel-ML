from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ink_tafel_study.feature_association_features import (
    COMPOSITION_SOURCE_SET,
    DIRECT_CONTEXT_SOURCES,
    DIRECT_PRIMARY_FEATURE_SOURCES,
    EXPECTED_REGISTERED_CANDIDATE_COUNT,
    FROZEN_ASSOCIATION_PANELS,
    HARD_EXCLUDED_SUPERVISED_SOURCES,
    POOLED_COMPOSITION_SOURCES,
    POOLED_PROPERTY_SOURCES,
    PROPERTY_SOURCE_SET,
    AssociationFeatureBuilder,
    association_engineered_feature_definitions,
    association_panel_registry,
    audit_feature_support_and_redundancy,
    derive_candidate_predictor_allowlist,
    registered_candidate_registry,
    required_association_source_columns,
)


def _records() -> pd.DataFrame:
    return pd.read_csv(
        "Data/03_outputs/study/outputs/current_97/data/records.csv"
    )


def _schema() -> pd.DataFrame:
    return pd.read_csv(
        "Data/03_outputs/study/outputs/current_97/data/"
        "registered_panel_schema.csv"
    )


def _all97() -> pd.DataFrame:
    records = _records().reset_index(drop=True)
    assert len(records) == 97
    return records


def test_registered_candidate_allowlist_is_exact_deduplicated_and_safe() -> None:
    schema = _schema()
    candidates = derive_candidate_predictor_allowlist(
        schema, expected_count=EXPECTED_REGISTERED_CANDIDATE_COUNT
    )
    assert len(candidates) == EXPECTED_REGISTERED_CANDIDATE_COUNT == 28
    assert len(candidates) == len(set(candidates))
    assert candidates[0] == "catalyst_active_material_type"
    assert "solvent_fraction_methanol" in candidates
    assert not any(
        token in name.lower()
        for name in candidates
        for token in ("target", "posterior", "slope", "tafel")
    )

    duplicated = pd.concat(
        [
            schema,
            schema.loc[
                schema["selected_column"].eq(
                    "catalyst_active_material_type"
                )
            ].head(1),
        ],
        ignore_index=True,
    )
    assert (
        derive_candidate_predictor_allowlist(duplicated)
        == candidates
    )


def test_registered_candidate_allowlist_rejects_target_like_selection() -> None:
    schema = pd.DataFrame(
        {
            "status": ["present"],
            "selected_column": ["target_value_mV_dec"],
            "logical_feature": ["bad"],
            "kind": ["numeric"],
            "distance_block": ["bad"],
        }
    )
    with pytest.raises(ValueError, match="forbidden predictors"):
        registered_candidate_registry(schema)


def test_frozen_panels_are_context_plus_one_and_representations_are_exclusive() -> None:
    assert tuple(FROZEN_ASSOCIATION_PANELS) == (
        "context_core",
        "context_plus_catalyst_concentration",
        "context_plus_ionomer_ratio",
        "context_plus_water_organic",
        "context_plus_ethanol_isopropanol",
        "context_plus_viscosity",
        "context_plus_polarity",
        "context_plus_conductive_sensitivity",
    )
    registry = association_panel_registry().set_index("name")
    augmented = registry.drop(index="context_core")
    assert augmented["parent"].eq("context_core").all()
    assert augmented["added_features"].map(len).eq(1).all()
    assert (
        bool(
            registry.loc[
                "context_plus_conductive_sensitivity", "effect_eligible"
            ]
        )
        is False
    )

    context = set(required_association_source_columns("context_core"))
    concentration = set(
        required_association_source_columns(
            "context_plus_catalyst_concentration"
        )
    )
    ionomer = set(
        required_association_source_columns("context_plus_ionomer_ratio")
    )
    water = set(
        required_association_source_columns("context_plus_water_organic")
    )
    ethanol_ipa = set(
        required_association_source_columns(
            "context_plus_ethanol_isopropanol"
        )
    )
    viscosity = set(
        required_association_source_columns("context_plus_viscosity")
    )
    polarity = set(
        required_association_source_columns("context_plus_polarity")
    )
    assert concentration - context == {"catalyst_concentration_mg_mL"}
    assert ionomer - context == {"ionomer_to_solvent_ratio"}
    assert water - context == COMPOSITION_SOURCE_SET
    assert ethanol_ipa - context == COMPOSITION_SOURCE_SET
    assert viscosity - context == {
        "solvent_mixture_property_proxy_viscosity_mPa_s"
    }
    assert polarity - context == PROPERTY_SOURCE_SET - {
        "solvent_mixture_property_proxy_viscosity_mPa_s"
    }
    assert water.isdisjoint(PROPERTY_SOURCE_SET)
    assert ethanol_ipa.isdisjoint(PROPERTY_SOURCE_SET)
    assert viscosity.isdisjoint(COMPOSITION_SOURCE_SET)
    assert polarity.isdisjoint(COMPOSITION_SOURCE_SET)
    definitions = association_engineered_feature_definitions()
    assert definitions["feature_name"].is_unique
    assert not definitions["target_derived"].astype(bool).any()
    assert {
        "log10_water_to_organic_balance",
        "solvent_polarity_coordinate",
    }.issubset(set(definitions["feature_name"]))


def test_feature_builder_generates_only_the_frozen_compact_coordinates() -> None:
    records = _all97()
    outputs: dict[str, pd.DataFrame] = {}
    builders: dict[str, AssociationFeatureBuilder] = {}
    for panel in FROZEN_ASSOCIATION_PANELS:
        builder = clone(AssociationFeatureBuilder(panel)).fit(records)
        transformed = builder.transform(records)
        builders[panel] = builder
        outputs[panel] = transformed
        numeric = transformed.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy(float)).all()
        assert "electrolyte_scope" in transformed
        assert not any(
            token in name.lower()
            for name in transformed
            for token in ("target", "posterior", "slope", "tafel")
        )
        blocks = builder.get_blocks_out()
        flattened = [
            feature for values in blocks.values() for feature in values
        ]
        assert set(flattened) == set(transformed.columns)
        assert len(flattened) == len(set(flattened))

    catalyst_counts = outputs["context_core"]["catalyst_family"].value_counts()
    assert catalyst_counts.to_dict() == {
        "RuO2": 49,
        "IrO2_plus_Ir": 48,
    }
    context_columns = set(outputs["context_core"])
    for panel, transformed in outputs.items():
        if panel == "context_core":
            continue
        assert len(set(transformed) - context_columns) == 1
        assert context_columns < set(transformed)
    assert set(
        builders["context_plus_water_organic"]
        .get_blocks_out()["water_to_organic_balance"]
    ) == {"log10_water_to_organic_balance"}
    assert set(
        builders["context_plus_ethanol_isopropanol"]
        .get_blocks_out()["ethanol_to_isopropanol_balance"]
    ) == {"log10_ethanol_to_isopropanol_balance"}
    assert "solvent_entropy" not in outputs["context_plus_water_organic"]
    assert set(
        builders["context_plus_viscosity"]
        .get_blocks_out()["solvent_viscosity"]
    ) == {"log10_solvent_viscosity_mPa_s"}
    assert set(
        builders["context_plus_polarity"]
        .get_blocks_out()["solvent_polarity"]
    ) == {"solvent_polarity_coordinate"}
    assert set(
        builders["context_plus_conductive_sensitivity"]
        .get_blocks_out()["conductive_agent"]
    ) == {"conductive_agent_present"}


def test_composition_and_property_state_are_fitted_once_on_training_fold() -> None:
    records = _all97()
    train = records.iloc[:60].reset_index(drop=True)
    heldout = records.iloc[60:].reset_index(drop=True)

    composition = AssociationFeatureBuilder(
        "context_plus_water_organic"
    ).fit(train)
    delta_before = float(composition.composition_delta_)
    extreme_composition = heldout.copy()
    extreme_composition.loc[:, "solvent_fraction_water"] = 1.0
    for column in (
        "solvent_fraction_acetone",
        "solvent_fraction_ethanol",
        "solvent_fraction_isopropanol",
        "solvent_fraction_methanol",
    ):
        extreme_composition.loc[:, column] = 0.0
    transformed = composition.transform(extreme_composition)
    assert np.isfinite(
        transformed.select_dtypes(include=[np.number]).to_numpy(float)
    ).all()
    assert composition.composition_delta_ == pytest.approx(delta_before)

    properties = AssociationFeatureBuilder(
        "context_plus_polarity"
    ).fit(train)
    components_before = properties.property_pca_.components_.copy()
    medians_before = properties.property_medians_.copy()
    scale_before = properties.property_scaler_.scale_.copy()
    extreme_properties = heldout.copy()
    extreme_properties.loc[
        :,
        [
            "solvent_mixture_property_proxy_surface_tension_mN_m",
            "solvent_mixture_property_proxy_density_kg_m3",
            (
                "solvent_mixture_property_proxy_dielectric_constant_"
                "best_available"
            ),
        ],
    ] = 1.0e9
    properties.transform(extreme_properties)
    np.testing.assert_allclose(
        properties.property_pca_.components_, components_before
    )
    np.testing.assert_allclose(properties.property_medians_, medians_before)
    np.testing.assert_allclose(
        properties.property_scaler_.scale_, scale_before
    )
    assert (
        properties.property_pca_.components_[0, -1]
        * properties.property_pca_sign_
        >= 0
    )


def test_support_audit_finds_known_thin_features_and_redundancy() -> None:
    result = audit_feature_support_and_redundancy(_records(), _schema())
    support = result["feature_support"].set_index(["scope", "feature_name"])
    assert support.shape[0] == 28
    assert set(result["feature_support"]["scope"]) == {"all97"}
    assert (
        support.loc[
            ("all97", "solvent_fraction_acetone"), "support_status"
        ]
        == "thin_nonzero_group_support"
    )
    assert int(
        support.loc[
            ("all97", "solvent_fraction_acetone"),
            "n_nonzero_groups",
        ]
    ) == 1
    assert int(
        support.loc[
            ("all97", "solvent_fraction_methanol"),
            "n_nonzero_groups",
        ]
    ) == 2
    assert int(
        support.loc[
            ("all97", "conductive_agent_present"),
            "minimum_category_groups",
        ]
    ) == 14

    pairs = result["numeric_redundancy_pairs"]
    property_pair = pairs.loc[
        pairs["scope"].eq("all97")
        & pairs["feature_a"].eq(
            "solvent_mixture_property_proxy_surface_tension_mN_m"
        )
        & pairs["feature_b"].eq(
            "solvent_mixture_property_proxy_density_kg_m3"
        )
    ]
    assert len(property_pair) == 1
    assert float(property_pair.iloc[0]["abs_spearman_rho"]) > 0.98
    strict_best = pairs.loc[
        pairs["scope"].eq("all97")
        & pairs["feature_a"].eq(
            "solvent_mixture_property_proxy_dielectric_constant"
        )
        & pairs["feature_b"].eq(
            (
                "solvent_mixture_property_proxy_dielectric_constant_"
                "best_available"
            )
        )
    ]
    assert len(strict_best) == 1
    assert float(strict_best.iloc[0]["abs_spearman_rho"]) == pytest.approx(
        1.0
    )

    closure = result["redundancy_clusters"].loc[
        lambda frame: frame["scope"].eq("all97")
        & frame["cluster_id"].eq("structural_solvent_closure")
    ]
    assert len(closure) == 1
    assert set(closure.iloc[0]["members"].split("|")) == set(
        COMPOSITION_SOURCE_SET
    )
    geometry = result["data_geometry"].set_index("scope")
    assert int(
        geometry.loc[
            "all97", "n_registered_candidate_predictors"
        ]
    ) == 28
    assert int(
        geometry.loc[
            "all97",
            "n_candidates_with_any_within_group_variation",
        ]
    ) < 28
    assert int(geometry.loc["all97", "n_records"]) == 97
    assert int(geometry.loc["all97", "n_connected_groups"]) == 80


def test_target_blind_reduction_roles_are_exact_and_raw_sources_do_not_reenter() -> None:
    result = audit_feature_support_and_redundancy(_records(), _schema())
    decisions = result["feature_decisions"].set_index("feature_name")
    assert set(
        decisions.index[
            decisions["supervised_role"].eq(
                "excluded_from_supervised_models"
            )
        ]
    ) == set(HARD_EXCLUDED_SUPERVISED_SOURCES)
    assert set(
        decisions.index[
            decisions["supervised_role"].eq(
                "pooled_composition_coordinate_input_only"
            )
        ]
    ) == set(POOLED_COMPOSITION_SOURCES)
    assert set(
        decisions.index[
            decisions["supervised_role"].eq(
                "pooled_property_coordinate_input_only"
            )
        ]
    ) == set(POOLED_PROPERTY_SOURCES)
    assert set(
        decisions.index[
            decisions["supervised_role"].eq("context_adjustment")
        ]
    ) == set(DIRECT_CONTEXT_SOURCES)
    assert set(
        decisions.index[
            decisions["supervised_role"].eq(
                "direct_single_feature_input"
            )
        ]
    ) == set(DIRECT_PRIMARY_FEATURE_SOURCES)

    supervised_outputs: set[str] = set()
    for panel in FROZEN_ASSOCIATION_PANELS:
        transformed = AssociationFeatureBuilder(panel).fit_transform(_all97())
        supervised_outputs.update(transformed.columns.astype(str))
    assert not set(HARD_EXCLUDED_SUPERVISED_SOURCES).intersection(
        supervised_outputs
    )
    assert "solvent_fraction_acetone" not in supervised_outputs
    assert "solvent_fraction_methanol" not in supervised_outputs


def test_support_and_redundancy_audit_is_target_blind() -> None:
    records = _records()
    first = audit_feature_support_and_redundancy(records, _schema())
    contaminated = records.copy()
    contaminated["target_value_mV_dec"] = np.linspace(
        -1.0e12, 1.0e12, len(contaminated)
    )
    contaminated["posterior_secret"] = np.arange(len(contaminated))[::-1]
    second = audit_feature_support_and_redundancy(
        contaminated, _schema()
    )
    for key in (
        "feature_support",
        "categorical_levels",
        "numeric_redundancy_pairs",
        "redundancy_clusters",
        "feature_decisions",
        "data_geometry",
    ):
        pd.testing.assert_frame_equal(first[key], second[key])
    assert second["summary"]["target_columns_read"] is False
    assert second["summary"]["unregistered_records_columns_scanned"] is False
