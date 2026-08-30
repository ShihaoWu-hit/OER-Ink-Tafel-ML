from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import ink_tafel_study.evaluate as evaluate_module
from ink_tafel_study.evaluate import (
    EvaluationConfig,
    EvaluationResult,
    build_connected_group_ids,
    make_outer_split_manifest,
    run_nested_evaluation,
    validate_split_manifest,
)


def test_connected_components_capture_transitive_publication_lab_recipe_links() -> None:
    frame = pd.DataFrame(
        {
            "record_id": [f"r{index}" for index in range(1, 9)],
            "publication_id": ["p1", "p1", "p3", "p4", "p5", "p6", "p7", "p8"],
            "lab_group_id": ["l1", "l2", "l3", "l4", "l4", "unknown", "unknown", "l8"],
            "recipe_hash": ["a", "b", "b", "d", "e", "f", "g", "h"],
            "sample_id": [f"s{index}" for index in range(1, 9)],
        }
    )
    groups = build_connected_group_ids(frame)
    # r1--r2 share publication and r2--r3 share recipe: all three belong to
    # one connected component even though r1 and r3 share no direct key.
    assert groups.iloc[0] == groups.iloc[1] == groups.iloc[2]
    assert groups.iloc[3] == groups.iloc[4]
    # Generic unknown laboratory labels must never pool unrelated records.
    assert groups.iloc[5] != groups.iloc[6]


def test_outer_manifest_holds_every_relation_together_and_detects_tampering() -> None:
    frame = pd.DataFrame(
        {
            "record_id": [f"r{index:02d}" for index in range(12)],
            "publication_id": [f"p{index // 2:02d}" for index in range(12)],
            "lab_group_id": [f"l{index // 2:02d}" for index in range(12)],
            "recipe_hash": [f"h{index // 2:02d}" for index in range(12)],
            "sample_id": [f"s{index:02d}" for index in range(12)],
        }
    )
    frame["cv_group_id"] = build_connected_group_ids(frame)
    manifest = make_outer_split_manifest(frame, folds=3, repeats=2, random_seed=9)

    assert len(manifest) == 24
    assert (manifest.groupby(["repeat", "record_id"]).size() == 1).all()
    assert (manifest.groupby(["repeat", "cv_group_id"])["fold"].nunique() == 1).all()

    tampered = manifest.copy()
    second_same_publication = tampered.index[
        (tampered["repeat"] == 0) & (tampered["record_id"] == "r01")
    ][0]
    tampered.loc[second_same_publication, "fold"] = (
        int(tampered.loc[second_same_publication, "fold"]) + 1
    ) % 3
    with pytest.raises(ValueError, match="split across outer folds"):
        validate_split_manifest(frame, tampered, expected_repeats=2, expected_folds=3)


def _evaluation_tables(n: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = np.arange(n)
    ink_signal = (index % 6) / 5.0
    records = pd.DataFrame(
        {
            "record_id": [f"record_{value:03d}" for value in index],
            "publication_id": [f"pub_{value:03d}" for value in index],
            "lab_group_id": [f"lab_{value:03d}" for value in index],
            "recipe_hash": [f"recipe_{value:03d}" for value in index],
            "sample_id": [f"sample_{value:03d}" for value in index],
            "catalyst_type": np.where(index % 2, "RuO2", "IrO2"),
            "catalyst_support": "none",
            "electrode_type": "RDE",
            "catalyst_loading_mg_cm2": 0.2 + 0.01 * (index % 3),
            "electrolyte_identity": "KOH",
            "electrolyte_molarity_M": 1.0,
            "rotation_rate_rpm": 1600.0,
            "scan_rate_mV_s": 5.0,
            "ir_correction_status": np.where(
                index % 2, "reported_corrected", "not_reported_or_unverifiable"
            ),
            "catalyst_concentration_mg_mL": 1.0 + ink_signal,
            "nafion_to_solvent_ratio": 0.05 + 0.1 * ink_signal,
            "ionomer_concentration_wt_percent": 5.0,
            "conductive_agent_present": np.where(index % 3, "no", "yes"),
            "conductive_agent_dose_mg_mgcat": np.where(index % 3, 0.0, 0.1),
            "solvent_fraction_acetone": 0.0,
            "solvent_fraction_water": 0.2 + 0.2 * ink_signal,
            "solvent_fraction_ethanol": 0.0,
            "solvent_fraction_isopropanol": 0.8 - 0.2 * ink_signal,
            # These raw outcomes are intentionally present and must not be
            # selected as predictors.
            "y_tafel_slope_mV_dec": 999.0,
            "y_overpotential_10mA_mV": 999.0,
        }
    )
    targets = pd.DataFrame(
        {
            "record_id": records["record_id"],
            "b_refit_at_2mA_mV_dec": 50.0 + 35.0 * ink_signal + (index % 2) * 4.0,
        }
    )
    return records, targets


def test_evaluation_config_defaults_interpretation_to_core_property_panel() -> None:
    assert (
        EvaluationConfig().importance_panel
        == "context_plus_property_proxy_core"
    )


def test_core_property_panel_requires_controlled_surface_statuses(tmp_path) -> None:
    records, targets = _evaluation_tables()
    config = EvaluationConfig(
        outer_folds=3,
        outer_repeats=1,
        inner_folds=2,
        bootstrap_replicates=0,
        panel_names=("context_plus_property_proxy_core",),
        model_names=("bayesian_ridge",),
        interpretation_enabled=False,
        n_jobs=1,
    )

    with pytest.raises(
        KeyError,
        match="requires controlled surface-tension status column",
    ):
        run_nested_evaluation(
            records,
            targets,
            tmp_path / "missing_status",
            target_col="b_refit_at_2mA_mV_dec",
            scope="alkaline",
            config=config,
        )

    records[
        "solvent_mixture_property_proxy_surface_tension_mN_m__model_use_status"
    ] = "eligible_source_backed"
    records.loc[
        records["record_id"].eq("record_003"),
        "solvent_mixture_property_proxy_surface_tension_mN_m__model_use_status",
    ] = "unreviewed_future_status"
    with pytest.raises(
        ValueError,
        match=r"invalid_statuses=.*unreviewed_future_status",
    ):
        run_nested_evaluation(
            records,
            targets,
            tmp_path / "unknown_status",
            target_col="b_refit_at_2mA_mV_dec",
            scope="alkaline",
            config=config,
        )


def test_failed_evaluation_qa_is_written_then_aborts(
    tmp_path, monkeypatch
) -> None:
    records, targets = _evaluation_tables()
    incomplete_oof = pd.DataFrame(
        [
            {
                "target": "b_refit_at_2mA_mV_dec",
                "scope": "alkaline",
                "panel": "context",
                "model": "nested_selected",
                "record_id": "record_000",
                "cv_group_id": "component_000",
                "repeat": 0,
                "fold": 0,
            }
        ]
    )
    empty = pd.DataFrame()
    fake_result = EvaluationResult(
        split_manifest=empty,
        inner_split_manifest=pd.DataFrame(
            columns=[
                "panel",
                "outer_repeat",
                "outer_fold",
                "inner_fold",
                "record_id",
                "cv_group_id",
            ]
        ),
        panel_schema=pd.DataFrame(
            {
                "panel": ["context"],
                "panel_evaluation_status": ["evaluated"],
                "selected_column": ["catalyst_type"],
            }
        ),
        panel_eligibility=empty,
        oof_predictions=incomplete_oof,
        oof_aggregated=empty,
        split_metrics=empty,
        metric_summary=empty,
        inner_selection=empty,
        paired_delta=empty,
        paired_delta_bootstrap=empty,
        model_selection=empty,
        outer_test_block_effects=empty,
        outer_test_block_importance=empty,
    )
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_nested_grouped_cv",
        lambda *args, **kwargs: fake_result,
    )
    config = EvaluationConfig(
        outer_folds=2,
        outer_repeats=2,
        inner_folds=2,
        bootstrap_replicates=0,
        panel_names=("context",),
        model_names=("bayesian_ridge",),
        interpretation_enabled=False,
        n_jobs=1,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Evaluation QA failed.*oof_coverage_complete",
    ):
        run_nested_evaluation(
            records,
            targets,
            tmp_path,
            target_col="b_refit_at_2mA_mV_dec",
            scope="alkaline",
            config=config,
        )

    qa_path = tmp_path / "evaluation_qa.json"
    assert qa_path.is_file()
    qa = json.loads(qa_path.read_text())
    assert qa["status"] == "fail"
    assert qa["oof_coverage_complete"] is False


def test_nested_evaluation_reuses_outer_splits_and_writes_auditable_outputs(tmp_path) -> None:
    records, targets = _evaluation_tables()
    config = EvaluationConfig(
        outer_folds=3,
        outer_repeats=1,
        inner_folds=2,
        bootstrap_replicates=20,
        panel_names=("dummy", "context", "context_plus_ink"),
        model_names=("dummy_mean", "bayesian_ridge"),
        importance_panel="context_plus_ink",
        n_jobs=1,
    )
    result = run_nested_evaluation(
        records,
        targets,
        tmp_path,
        target_col="b_refit_at_2mA_mV_dec",
        scope="alkaline",
        config=config,
    )

    assert set(result.oof_predictions["panel"]) == {"dummy", "context", "context_plus_ink"}
    coverage = result.oof_predictions.groupby(["panel", "model", "record_id"]).size()
    assert (coverage == 1).all()
    assert len(result.paired_delta) == 3
    assert "delta_mae_context_plus_ink_minus_context" in result.paired_delta
    assert set(result.paired_delta["model"]) == {"nested_selected"}
    assert set(result.model_selection["selection_data"]) == {
        "outer_training_fold_inner_grouped_cv_only"
    }
    assert (
        result.model_selection.groupby(["panel", "outer_repeat", "outer_fold"])["selected"]
        .sum()
        .eq(1)
        .all()
    )
    # The same connected group is assigned to only one validation fold inside
    # each outer training set.
    assert (
        result.inner_split_manifest.groupby(
            ["panel", "outer_repeat", "outer_fold", "cv_group_id"]
        )["inner_fold"]
        .nunique()
        .eq(1)
        .all()
    )
    assert not result.panel_schema["selected_column"].astype(str).str.contains("tafel|overpotential", case=False).any()
    assert not result.outer_test_block_importance.empty
    assert result.outer_test_block_importance["outer_test_only"].all()
    assert set(result.outer_test_block_importance["selection_role"]) == {
        "posthoc_metric_only_not_model_selection"
    }

    for filename in (
        "split_manifest.csv",
        "panel_eligibility.csv",
        "inner_selection.csv",
        "oof_predictions.csv",
        "fold_metrics.csv",
        "paired_delta_mae.csv",
        "outer_test_block_effects.csv",
        "outer_test_block_importance.csv",
        "evaluation_qa.json",
    ):
        assert (tmp_path / filename).is_file()
    qa = json.loads((tmp_path / "evaluation_qa.json").read_text())
    assert qa["status"] == "pass"
    assert qa["same_outer_manifest_reused_across_panels_and_models"] is True
    assert qa["delta_convention"].startswith(
        "augmented-panel MAE minus its named paired reference MAE"
    )


def test_registered_property_panels_and_source_backed_complete_case_pairing(
    tmp_path,
) -> None:
    records, targets = _evaluation_tables()
    index = np.arange(len(records))
    strict_dielectric = np.where(index % 4 == 0, np.nan, 25.0 + 0.2 * index)
    estimated_dielectric = np.where(np.isnan(strict_dielectric), 24.5 + 0.2 * index, np.nan)
    best_available_dielectric = np.where(
        np.isnan(strict_dielectric), estimated_dielectric, strict_dielectric
    )
    property_columns = {
        "solvent_mixture_property_proxy_viscosity_mPa_s": 1.0 + 0.05 * index,
        "solvent_mixture_property_proxy_surface_tension_mN_m": 30.0 + 0.1 * index,
        "solvent_mixture_property_proxy_density_kg_m3": 900.0 - index,
        "solvent_mixture_property_proxy_dielectric_constant": strict_dielectric,
        "solvent_mixture_property_proxy_dielectric_constant_estimated": estimated_dielectric,
        "solvent_mixture_property_proxy_dielectric_constant_best_available": best_available_dielectric,
        "solvent_mixture_property_proxy_dielectric_constant_estimated_flag": np.isnan(
            strict_dielectric
        ).astype(int),
    }
    records = records.assign(**property_columns)
    medium_surface_records = {"record_003", "record_017"}
    records[
        "solvent_mixture_property_proxy_surface_tension_mN_m__model_use_status"
    ] = np.where(
        records["record_id"].isin(medium_surface_records),
        "eligible_source_backed_medium_confidence",
        "eligible_source_backed",
    )
    config = EvaluationConfig(
        outer_folds=3,
        outer_repeats=1,
        inner_folds=2,
        bootstrap_replicates=0,
        panel_names=(
            "context",
            "context_plus_ink_amounts",
            "context_plus_ink",
            "context_plus_property_proxy_core",
            "context_plus_property_proxy_4p_source_backed",
            "context_plus_property_proxy_4p_estimated",
        ),
        model_names=("bayesian_ridge",),
        interpretation_enabled=False,
        n_jobs=1,
    )

    result = run_nested_evaluation(
        records,
        targets,
        tmp_path,
        target_col="b_refit_at_2mA_mV_dec",
        scope="alkaline",
        config=config,
    )

    expected_panels = {
        "context",
        "context_plus_ink_amounts",
        "context_plus_ink",
        "context_plus_property_proxy_core",
        "context_plus_property_proxy_4p_source_backed",
        "context_plus_property_proxy_4p_estimated",
        "context_plus_ink_amounts_on_4p_source_backed_complete_cases",
        "context_plus_property_proxy_core_excluding_medium_ipa_surface",
        "context_plus_ink_amounts_on_core_excluding_medium_ipa_surface",
    }
    assert set(result.oof_predictions["panel"]) == expected_panels
    panel_status = result.panel_schema.loc[
        result.panel_schema["panel"].isin(expected_panels), "panel_evaluation_status"
    ]
    assert set(panel_status) == {"evaluated"}

    selected = result.oof_predictions.loc[result.oof_predictions["model"].eq("nested_selected")]
    full_records = set(records["record_id"].astype(str))
    strict_records = set(records.loc[np.isfinite(strict_dielectric), "record_id"].astype(str))
    assert set(
        selected.loc[
            selected["panel"].eq("context_plus_property_proxy_core"), "record_id"
        ].astype(str)
    ) == full_records
    assert set(
        selected.loc[
            selected["panel"].eq("context_plus_property_proxy_4p_estimated"), "record_id"
        ].astype(str)
    ) == full_records
    for panel in (
        "context_plus_property_proxy_4p_source_backed",
        "context_plus_ink_amounts_on_4p_source_backed_complete_cases",
    ):
        assert set(selected.loc[selected["panel"].eq(panel), "record_id"].astype(str)) == strict_records
        manifest = result.panel_eligibility.loc[result.panel_eligibility["panel"].eq(panel)]
        assert set(manifest.loc[manifest["eligible"], "record_id"].astype(str)) == strict_records

    medium_sensitivity_records = full_records - medium_surface_records
    for panel in (
        "context_plus_property_proxy_core_excluding_medium_ipa_surface",
        "context_plus_ink_amounts_on_core_excluding_medium_ipa_surface",
    ):
        assert set(selected.loc[selected["panel"].eq(panel), "record_id"].astype(str)) == (
            medium_sensitivity_records
        )
        manifest = result.panel_eligibility.loc[result.panel_eligibility["panel"].eq(panel)]
        assert set(manifest.loc[manifest["eligible"], "record_id"].astype(str)) == (
            medium_sensitivity_records
        )
        excluded = manifest.loc[~manifest["eligible"]]
        assert set(excluded["record_id"].astype(str)) == medium_surface_records
        assert set(excluded["ineligibility_reason"]) == {
            "medium_confidence_ipa_surface_tension_bracket_excluded"
        }

    strict_delta = result.paired_delta.loc[
        result.paired_delta["comparison"].eq(
            "property_proxy_4p_source_backed_minus_context_plus_ink_amounts_complete_cases"
        )
    ]
    assert len(strict_delta) == 3
    assert set(strict_delta["reference_panel"]) == {
        "context_plus_ink_amounts_on_4p_source_backed_complete_cases"
    }
    assert strict_delta["reference_n_train"].eq(strict_delta["augmented_n_train"]).all()
    assert strict_delta["reference_n_test"].eq(strict_delta["augmented_n_test"]).all()

    medium_surface_delta = result.paired_delta.loc[
        result.paired_delta["comparison"].eq(
            "property_proxy_core_excluding_medium_ipa_surface_minus_context_plus_ink_amounts_same_records"
        )
    ]
    assert len(medium_surface_delta) == 3
    assert set(medium_surface_delta["reference_panel"]) == {
        "context_plus_ink_amounts_on_core_excluding_medium_ipa_surface"
    }
    assert medium_surface_delta["reference_n_train"].eq(
        medium_surface_delta["augmented_n_train"]
    ).all()
    assert medium_surface_delta["reference_n_test"].eq(
        medium_surface_delta["augmented_n_test"]
    ).all()

    full_core_delta = result.paired_delta.loc[
        result.paired_delta["comparison"].eq(
            "property_proxy_core_minus_context_plus_ink_amounts"
        )
    ]
    assert len(full_core_delta) == 3
    assert set(full_core_delta["reference_panel"]) == {"context_plus_ink_amounts"}
    assert full_core_delta["reference_n_train"].eq(
        full_core_delta["augmented_n_train"]
    ).all()
    assert full_core_delta["reference_n_test"].eq(
        full_core_delta["augmented_n_test"]
    ).all()

    representation_delta = result.paired_delta.loc[
        result.paired_delta["comparison"].eq(
            "property_proxy_core_minus_raw_solvent_composition_same_records"
        )
    ]
    assert len(representation_delta) == 3
    assert set(representation_delta["reference_panel"]) == {"context_plus_ink"}
    assert representation_delta["reference_n_train"].eq(
        representation_delta["augmented_n_train"]
    ).all()
    assert representation_delta["reference_n_test"].eq(
        representation_delta["augmented_n_test"]
    ).all()

    qa = json.loads((tmp_path / "evaluation_qa.json").read_text())
    assert qa["property_proxy_coverage"] == {
        column: int(np.isfinite(values).sum())
        for column, values in property_columns.items()
    }
    assert qa["property_panel_complete_case_records"] == {
        "context_plus_property_proxy_core": len(records),
        "context_plus_property_proxy_4p_source_backed": int(np.isfinite(strict_dielectric).sum()),
        "context_plus_property_proxy_4p_estimated": len(records),
    }
    assert qa["source_backed_4p_pair_uses_identical_records"] is True
    assert qa["source_backed_4p_pair_uses_identical_outer_assignments"] is True
    assert qa["source_backed_4p_pair_uses_identical_inner_assignments"] is True
    assert qa["core_property_incremental_reference_panel"] == "context_plus_ink_amounts"
    assert qa["core_property_incremental_pair_uses_identical_records"] is True
    assert qa["core_property_incremental_pair_uses_identical_outer_assignments"] is True
    assert qa["core_property_incremental_pair_uses_identical_inner_assignments"] is True
    assert qa["property_representation_raw_composition_reference_panel"] == "context_plus_ink"
    assert qa["property_representation_pair_uses_identical_records"] is True
    assert qa["property_representation_pair_uses_identical_outer_assignments"] is True
    assert qa["property_representation_pair_uses_identical_inner_assignments"] is True
    assert qa["medium_confidence_ipa_surface_exclusion_required"] is True
    assert qa["medium_confidence_ipa_surface_exclusion_status"] == "evaluated"
    assert qa["medium_confidence_ipa_surface_excluded_records"] == 2
    assert set(qa["medium_confidence_ipa_surface_excluded_record_ids"]) == (
        medium_surface_records
    )
    assert qa["medium_confidence_ipa_surface_pair_uses_identical_records"] is True
    assert qa["medium_confidence_ipa_surface_pair_uses_identical_outer_assignments"] is True
    assert qa["medium_confidence_ipa_surface_pair_uses_identical_inner_assignments"] is True
    assert qa["property_proxy_panel_status"] == "evaluated"
    assert set(qa["evaluated_panels"]) == expected_panels
    assert not qa["skipped_panels"]
