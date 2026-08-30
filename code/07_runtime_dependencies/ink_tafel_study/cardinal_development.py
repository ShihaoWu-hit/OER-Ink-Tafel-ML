"""Feature-engineered and catalyst-stratified cardinal ML development.

This is an independent exploratory stage.  It does not alter the frozen
cardinal-classification analysis or the registered continuous-response
evaluation.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cardinal_abstention import (
    ABSTENTION_POLICY_ID,
    DEFAULT_COVERAGE_CURVE_THRESHOLDS,
    DEFAULT_NORMALIZED_ENTROPY_THRESHOLD,
    FIXED_THRESHOLD_SOURCE,
    write_entropy_abstention_artifacts,
)
from .cardinal_evaluate import (
    CandidateDefinition,
    CardinalEvaluationConfig,
    CardinalEvaluationResult,
    evaluate_paired_panel_contrasts,
    run_nested_cardinal_evaluation,
)
from .cardinal_feature_engineering import (
    CARDINAL_FEATURE_RECIPES,
    PRIMARY_CARDINAL_FEATURE_RECIPES,
    SENSITIVITY_CARDINAL_FEATURE_RECIPES,
    cardinal_engineered_feature_definitions,
    cardinal_feature_recipe_registry,
    required_cardinal_feature_columns,
)
from .cardinal_hierarchical_bayes import (
    HierarchicalMAPCandidate,
    run_hierarchical_map_nested_oof,
)
from .cardinal_hierarchical_bayes import (
    HierarchicalMAPCandidate,
    run_hierarchical_map_nested_oof,
)
from .cardinal_inputs import (
    EXPECTED_DRAWS_PER_RECORD,
    assert_disjoint_cardinal_paths,
    snapshot_cardinal_inputs,
    verify_cardinal_input_snapshot,
    verify_source_run_unchanged,
)
from .cardinal_labels import (
    DEFAULT_ANCHORS,
    derive_hierarchical_soft_labels,
    derive_posterior_soft_labels,
    hierarchical_support_summary,
    support_summary,
    validate_hierarchical_soft_labels,
    validate_soft_labels,
)
from .cardinal_metrics import (
    PREDICTION_COLUMNS,
    SOFT_LABEL_COLUMNS,
    collapsed_hard_metrics,
    two_stage_probability_metrics,
)
from .cardinal_stacking import (
    DEFAULT_FIXED_RECIPES,
    run_fixed_recipe_stacking_evaluation,
)
from .config import StudyConfig
from .features import ResolvedPanel
from .orthogonal_kinetics import audit_orthogonal_kinetics
from .provenance import sha256_file, write_json


FEATURE_SEARCH_PANEL = "fold_local_feature_recipe_search"
USER_REQUESTED_CATALYST_STRATA = {
    "RuO2": ("RuO2",),
    "IrO2_plus_Ir": ("IrO2", "Ir"),
}


def _tuple_setting(
    settings: Mapping[str, Any],
    field: str,
    default: Sequence[Any],
    *,
    cast,
) -> tuple[Any, ...]:
    value = settings.get(field, default)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"cardinal_development.{field} must be non-empty")
    return tuple(cast(item) for item in value)


def _validate_settings(settings: Mapping[str, Any]) -> None:
    abstention = settings.get("abstention", {})
    ir_fixed = settings.get("ir_fixed_comparison", {})
    hierarchical = settings.get("hierarchical_partial_pooling", {})
    fixed_recipes = (
        str(ir_fixed.get("reference_recipe")),
        str(ir_fixed.get("augmented_recipe")),
    )
    hierarchical_recipes = tuple(
        map(str, hierarchical.get("fixed_feature_recipes", ()))
    )
    unknown = sorted(
        set((*fixed_recipes, *hierarchical_recipes))
        - set(CARDINAL_FEATURE_RECIPES)
    )
    exclusions = settings.get("excluded_from_mainline", {})
    checks = {
        "status": str(settings.get("status")) == "exploratory_posthoc",
        "active_design": (
            str(settings.get("active_design"))
            == "fixed_ir_stack_plus_ru_ir_hierarchical_map"
        ),
        "source_target": (
            str(settings.get("source_target"))
            == "b_bayesian_local_line_mV_dec"
        ),
        "mechanistic_claim_allowed": (
            settings.get("mechanistic_claim_allowed") is False
        ),
        "retired_recipe_search_disabled": (
            settings.get("retired_recipe_search", {}).get("enabled")
            is False
        ),
        "metal_Ir_merge": tuple(
            settings.get("catalyst_strata", {}).get(
                "IrO2_plus_Ir", ()
            )
        )
        == ("IrO2", "Ir"),
        "primary_width": float(
            settings.get("relative_half_width", np.nan)
        )
        == 0.20,
        "abstention_enabled": abstention.get("enabled") is True,
        "abstention_policy": (
            str(abstention.get("policy_id")) == ABSTENTION_POLICY_ID
        ),
        "abstention_threshold": float(
            abstention.get("normalized_entropy_threshold", np.nan)
        )
        == DEFAULT_NORMALIZED_ENTROPY_THRESHOLD,
        "abstention_threshold_source": (
            str(abstention.get("threshold_source"))
            == FIXED_THRESHOLD_SOURCE
        ),
        "abstention_curve": tuple(
            float(value)
            for value in abstention.get("coverage_curve_thresholds", ())
        )
        == DEFAULT_COVERAGE_CURVE_THRESHOLDS,
        "abstention_keeps_full_coverage_primary": (
            abstention.get("full_coverage_primary_metrics_unchanged")
            is True
        ),
        "ir_fixed_recipes": (
            str(ir_fixed.get("reference_recipe")),
            str(ir_fixed.get("augmented_recipe")),
        )
        == DEFAULT_FIXED_RECIPES,
        "ir_no_joint_recipe_selector": (
            str(ir_fixed.get("recipe_selection"))
            == "fixed_no_joint_selector"
        ),
        "ir_fixed_components": tuple(
            map(str, ir_fixed.get("base_components", ()))
        )
        == ("soft_gower_knn", "soft_multinomial_logistic"),
        "ir_stacking_source": (
            str(ir_fixed.get("combination"))
            == "convex_weight_from_outer_training_inner_group_oof"
        ),
        "ir_optuna_disabled": ir_fixed.get("optuna_used") is False,
        "hierarchical_enabled": hierarchical.get("enabled") is True,
        "hierarchical_implementation": (
            str(hierarchical.get("implementation"))
            == "two_stage_hierarchical_gaussian_prior_MAP"
        ),
        "hierarchical_approximation_disclosed": (
            str(hierarchical.get("approximation"))
            == "posterior_mode_not_NUTS_or_MCMC"
        ),
        "hierarchical_families": tuple(
            map(str, hierarchical.get("catalyst_families", ()))
        )
        == ("RuO2", "IrO2_plus_Ir"),
        "hierarchical_metal_Ir_mapping": (
            str(hierarchical.get("metallic_Ir_mapping"))
            == "IrO2_plus_Ir"
        ),
        "hierarchical_fixed_recipes": tuple(
            map(str, hierarchical.get("fixed_feature_recipes", ()))
        )
        == DEFAULT_FIXED_RECIPES,
        "hierarchical_no_recipe_search": (
            str(hierarchical.get("recipe_selection"))
            == "fixed_no_joint_selector"
        ),
        "hierarchical_shared_manifests": (
            hierarchical.get(
                "shared_outer_and_inner_manifests_across_recipes"
            )
            is True
        ),
        "hierarchical_draws_not_expanded": (
            hierarchical.get("posterior_draws_expanded") is False
        ),
        "hierarchical_optuna_disabled": (
            hierarchical.get("optuna_used") is False
        ),
        "clustering_not_mainline": (
            str(exclusions.get("clustering"))
            == "exploratory_formulation_archetypes_only"
        ),
        "smote_not_mainline": exclusions.get("SMOTE") is True,
        "deep_learning_not_mainline": (
            exclusions.get("deep_learning") is True
        ),
        "large_boosting_search_not_mainline": (
            exclusions.get("large_CatBoost_XGBoost_search") is True
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if unknown or failures:
        raise ValueError(
            "Cardinal-development config does not match the executable "
            f"contract: unknown_recipes={unknown}, failed={failures}"
        )


def _recipe_panel(
    frame: pd.DataFrame,
    recipe_names: Sequence[str],
) -> ResolvedPanel:
    columns = required_cardinal_feature_columns(tuple(recipe_names))
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(
            f"Cardinal feature-search sources are missing: {missing}"
        )
    categorical_sources = {
        "catalyst_type",
        "electrolyte_identity",
        "ir_correction_status",
        "conductive_agent_class",
    }
    numeric = tuple(
        name for name in columns if name not in categorical_sources
    )
    categorical = tuple(
        name for name in columns if name in categorical_sources
    )
    return ResolvedPanel(
        name=FEATURE_SEARCH_PANEL,
        columns=columns,
        logical_names=columns,
        numeric_columns=numeric,
        categorical_columns=categorical,
        blocks={"raw_source_allowlist": columns},
        missing_logical_features=(),
    )


def _write_development_abstention(
    oof_aggregated: pd.DataFrame,
    output_dir: Path,
    settings: Mapping[str, Any],
) -> dict[str, object]:
    """Apply the frozen reporting policy without changing primary scoring."""

    policy = settings.get("abstention", {})
    if policy.get("enabled") is not True:
        raise ValueError("The registered cardinal abstention policy is disabled")
    return write_entropy_abstention_artifacts(
        oof_aggregated,
        output_dir,
        normalized_entropy_threshold=float(
            policy["normalized_entropy_threshold"]
        ),
        coverage_curve_thresholds=tuple(
            float(value)
            for value in policy["coverage_curve_thresholds"]
        ),
        threshold_source=str(policy["threshold_source"]),
    )


def _candidate_catalog(
    recipe_names: Sequence[str],
    *,
    smoke: bool,
) -> tuple[CandidateDefinition, ...]:
    """Finite joint recipe/model catalogue; class prior appears once."""

    rows = [
        CandidateDefinition(
            candidate_id="class_prior",
            model_family="class_prior",
            params={},
            family_complexity_rank=0,
            parameter_complexity_rank=0,
        )
    ]
    base_models: tuple[tuple[str, dict[str, object]], ...] = (
        ("soft_ordinal_logistic", {"C": 1.0}),
        ("soft_multinomial_logistic", {"C": 1.0}),
        (
            "soft_gower_knn",
            {"n_neighbors": 15, "distance_power": 1.0},
        ),
        (
            "soft_extra_trees",
            {
                "n_estimators": 40 if smoke else 100,
                "max_depth": 2,
                "min_samples_leaf": 8,
                "n_jobs": 1,
            },
        ),
    )
    for recipe_name in recipe_names:
        recipe = CARDINAL_FEATURE_RECIPES[recipe_name]
        for model_rank, (base_model, base_params) in enumerate(base_models, 1):
            rows.append(
                CandidateDefinition(
                    candidate_id=f"{recipe_name}__{base_model}",
                    model_family=recipe_name,
                    params={
                        "feature_recipe": recipe_name,
                        "base_model_family": base_model,
                        "base_model_params": base_params,
                    },
                    family_complexity_rank=int(recipe.complexity_rank),
                    parameter_complexity_rank=model_rank,
                )
            )
    return tuple(rows)


def _evaluation_config(
    settings: Mapping[str, Any],
    recipe_names: Sequence[str],
    *,
    n_jobs: int,
) -> CardinalEvaluationConfig:
    validation = settings.get("validation", {})
    seeds = tuple(
        int(value)
        for value in validation.get(
            "outer_seeds",
            [42, 142, 242, 342, 442, 542, 642, 742, 842, 942],
        )
    )
    return CardinalEvaluationConfig(
        outer_folds=int(validation.get("outer_folds", 5)),
        outer_repeats=int(validation.get("outer_repeats", 10)),
        inner_folds=int(validation.get("inner_folds", 4)),
        random_seed=int(seeds[0]),
        outer_seeds=seeds,
        bootstrap_replicates=int(
            validation.get("group_bootstrap_repetitions", 2000)
        ),
        calibration_bins=int(validation.get("calibration_bins", 8)),
        panel_names=(FEATURE_SEARCH_PANEL,),
        model_names=("class_prior", *tuple(recipe_names)),
        reference_panel="__not_a_registered_panel__",
        augmented_panel="__not_a_registered_panel__",
        n_jobs=int(n_jobs),
    )


def _ir_fixed_stacking_config(
    settings: Mapping[str, Any],
    *,
    n_jobs: int,
) -> CardinalEvaluationConfig:
    validation = settings.get("validation", {})
    seeds = tuple(
        int(value)
        for value in validation.get(
            "outer_seeds",
            [42, 142, 242, 342, 442, 542, 642, 742, 842, 942],
        )
    )
    return CardinalEvaluationConfig(
        outer_folds=int(validation.get("outer_folds", 5)),
        outer_repeats=int(validation.get("outer_repeats", 10)),
        inner_folds=int(validation.get("inner_folds", 4)),
        random_seed=int(seeds[0]),
        outer_seeds=seeds,
        bootstrap_replicates=int(
            validation.get("group_bootstrap_repetitions", 2000)
        ),
        calibration_bins=int(validation.get("calibration_bins", 8)),
        panel_names=DEFAULT_FIXED_RECIPES,
        model_names=DEFAULT_FIXED_RECIPES,
        reference_panel=DEFAULT_FIXED_RECIPES[0],
        augmented_panel=DEFAULT_FIXED_RECIPES[1],
        n_jobs=int(n_jobs),
    )


def _hierarchical_candidates(
    settings: Mapping[str, Any],
) -> tuple[HierarchicalMAPCandidate, ...]:
    rows = settings["hierarchical_partial_pooling"].get(
        "shrinkage_candidates", ()
    )
    candidates = tuple(
        HierarchicalMAPCandidate(
            candidate_id=str(row["id"]),
            family_feature_scale=float(row["family_feature_scale"]),
            family_intercept_scale=float(row["family_intercept_scale"]),
            complexity_rank=int(position),
        )
        for position, row in enumerate(rows)
    )
    if not candidates:
        raise ValueError(
            "cardinal_development.hierarchical_partial_pooling requires "
            "at least one preregistered shrinkage candidate"
        )
    if len({row.candidate_id for row in candidates}) != len(candidates):
        raise ValueError("Hierarchical shrinkage candidate IDs must be unique")
    return candidates


def _hierarchical_split_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (panel, repeat, fold), frame in oof.groupby(
        ["panel", "repeat", "fold"], dropna=False, sort=True
    ):
        observed = frame.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
        predicted = frame.loc[:, PREDICTION_COLUMNS].to_numpy(float)
        rows.append(
            {
                "analysis_id": "hierarchical_map_ru_ir_fixed_recipes",
                "scope": "alkaline_RuO2_and_IrO2_plus_Ir",
                "panel_role": "fixed_hierarchical_recipe_comparison",
                "panel": str(panel),
                # These two values are used only by the generic paired-
                # contrast helper. The persisted OOF tables retain the honest
                # fixed_hierarchical_MAP model identity.
                "model_role": "joint_selected",
                "model_family": "nested_selected",
                "repeat": int(repeat),
                "fold": int(fold),
                "n_test": int(len(frame)),
                "n_test_groups": int(frame["cv_group_id"].nunique()),
                **two_stage_probability_metrics(observed, predicted),
                **collapsed_hard_metrics(observed, predicted),
            }
        )
    return pd.DataFrame(rows)


def _run_hierarchical_fixed_comparison(
    alkaline: pd.DataFrame,
    output_dir: Path,
    settings: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, object]:
    hierarchy = settings["hierarchical_partial_pooling"]
    validation = settings["validation"]
    recipes = tuple(map(str, hierarchy["fixed_feature_recipes"]))
    if recipes != DEFAULT_FIXED_RECIPES:
        raise ValueError(
            "The hierarchical comparison must use the same two fixed recipes "
            "as the Ir stacking comparison"
        )
    seeds = tuple(int(value) for value in validation["outer_seeds"])
    outer_folds = 2 if smoke else int(validation["outer_folds"])
    outer_repeats = 1 if smoke else int(validation["outer_repeats"])
    inner_folds = 2 if smoke else int(validation["inner_folds"])
    outer_seeds = seeds[:outer_repeats]
    bootstrap_replicates = (
        min(50, int(validation["group_bootstrap_repetitions"]))
        if smoke
        else int(validation["group_bootstrap_repetitions"])
    )
    candidates = _hierarchical_candidates(settings)
    analysis_id = "hierarchical_map_ru_ir_fixed_recipes"
    scope = "alkaline_RuO2_and_IrO2_plus_Ir"
    results: dict[str, Any] = {}
    for position, recipe in enumerate(recipes):
        frozen_outer = (
            None
            if position == 0
            else results[recipes[0]].split_manifest
        )
        frozen_inner = (
            None
            if position == 0
            else results[recipes[0]].inner_split_manifest
        )
        results[recipe] = run_hierarchical_map_nested_oof(
            alkaline,
            feature_recipe=recipe,
            output_dir=output_dir / recipe,
            outer_folds=outer_folds,
            outer_repeats=outer_repeats,
            inner_folds=inner_folds,
            random_seed=int(seeds[0]),
            outer_seeds=outer_seeds,
            candidates=candidates,
            global_feature_scale=float(
                hierarchy["global_feature_scale"]
            ),
            global_intercept_scale=float(
                hierarchy["global_intercept_scale"]
            ),
            max_iter=int(hierarchy["max_iter"]),
            tolerance=float(hierarchy["tolerance"]),
            frozen_outer_manifest=frozen_outer,
            frozen_inner_manifest=frozen_inner,
            analysis_id=analysis_id,
            scope=scope,
        )

    reference = results[recipes[0]]
    augmented = results[recipes[1]]
    manifest_columns = ["repeat", "fold", "record_id", "cv_group_id"]
    outer_manifests_match = bool(
        reference.split_manifest.loc[:, manifest_columns]
        .sort_values(manifest_columns, kind="stable")
        .reset_index(drop=True)
        .equals(
            augmented.split_manifest.loc[:, manifest_columns]
            .sort_values(manifest_columns, kind="stable")
            .reset_index(drop=True)
        )
    )
    inner_manifest_columns = [
        "outer_repeat",
        "outer_fold",
        "inner_fold",
        "record_id",
        "cv_group_id",
    ]
    inner_manifests_match = bool(
        reference.inner_split_manifest.loc[:, inner_manifest_columns]
        .sort_values(inner_manifest_columns, kind="stable")
        .reset_index(drop=True)
        .equals(
            augmented.inner_split_manifest.loc[:, inner_manifest_columns]
            .sort_values(inner_manifest_columns, kind="stable")
            .reset_index(drop=True)
        )
    )
    if not outer_manifests_match or not inner_manifests_match:
        raise AssertionError(
            "Hierarchical fixed recipes did not use identical connected-group "
            "outer and inner folds"
        )

    oof = pd.concat(
        [results[recipe].oof_probabilities for recipe in recipes],
        ignore_index=True,
    )
    aggregated = pd.concat(
        [results[recipe].oof_aggregated for recipe in recipes],
        ignore_index=True,
    )
    split_metrics = _hierarchical_split_metrics(oof)
    contrast_oof = aggregated.copy()
    contrast_oof["analysis_id"] = analysis_id
    contrast_oof["scope"] = scope
    contrast_oof["panel_role"] = "fixed_hierarchical_recipe_comparison"
    contrast_oof["model_role"] = "joint_selected"
    contrast_oof["model_family"] = "nested_selected"
    deltas, bootstrap, summary, registry = evaluate_paired_panel_contrasts(
        contrast_oof,
        split_metrics,
        [
            {
                "id": "legacy_safe_compact_minus_context_log",
                "role": "fixed_hierarchical_recipe_comparison",
                "reference": recipes[0],
                "augmented": recipes[1],
            }
        ],
        analysis_id=analysis_id,
        scope=scope,
        panel_role="fixed_hierarchical_recipe_comparison",
        replicates=bootstrap_replicates,
        random_seed=int(seeds[0]) + 400_000,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output_dir / "oof_probabilities_all_recipes.csv", index=False)
    aggregated.to_csv(
        output_dir / "oof_aggregated_all_recipes.csv", index=False
    )
    split_metrics.to_csv(output_dir / "split_metrics.csv", index=False)
    deltas.to_csv(output_dir / "paired_split_deltas.csv", index=False)
    bootstrap.to_csv(
        output_dir / "paired_group_bootstrap.csv", index=False
    )
    summary.to_csv(
        output_dir / "paired_group_bootstrap_summary.csv", index=False
    )
    registry.to_csv(
        output_dir / "paired_contrast_registry.csv", index=False
    )
    selections = pd.concat(
        [
            results[recipe].outer_selections.assign(panel=recipe)
            for recipe in recipes
        ],
        ignore_index=True,
    )
    all_optimizers_succeeded = bool(
        selections[
            ["gate_optimizer_success", "conditional_optimizer_success"]
        ]
        .astype(bool)
        .all(axis=None)
    )
    qa = {
        "status": "complete_smoke" if smoke else "complete",
        "implementation": "two_stage_hierarchical_gaussian_prior_MAP",
        "approximation": "posterior_mode_not_NUTS_or_MCMC",
        "n_records": int(len(alkaline)),
        "n_connected_groups": int(alkaline["cv_group_id"].nunique()),
        "catalyst_families": ["RuO2", "IrO2_plus_Ir"],
        "metallic_Ir_mapping": "IrO2_plus_Ir",
        "fixed_feature_recipes": list(recipes),
        "recipe_selector_used": False,
        "optuna_used": False,
        "posterior_draws_expanded": False,
        "one_fractional_soft_response_per_record": True,
        "shared_outer_manifests_across_recipes": outer_manifests_match,
        "shared_inner_manifests_across_recipes": inner_manifests_match,
        "all_final_MAP_optimizers_succeeded": all_optimizers_succeeded,
        "full_coverage_primary_metrics": [
            "soft_log_loss",
            "multiclass_brier",
        ],
        "mechanistic_claim_allowed": False,
    }
    write_json(output_dir / "hierarchical_comparison_qa.json", qa)
    abstention = _write_development_abstention(
        aggregated,
        output_dir / "abstention",
        settings,
    )
    return {
        "results": results,
        "oof_aggregated": aggregated,
        "paired_bootstrap_summary": summary,
        "qa": qa,
        "abstention": abstention,
    }


def _derive_labels(
    draws: pd.DataFrame,
    records: pd.DataFrame,
    output_dir: Path,
    *,
    relative_half_width: float,
) -> pd.DataFrame:
    mapping_columns = [
        name
        for name in (
            "record_id",
            "sample_record",
            "curve_id",
            "cv_group_id",
            "electrolyte_scope",
            "publication_id",
            "lab_group_id",
            "recipe_hash",
            "sample_id",
        )
        if name in records.columns
    ]
    raw = derive_posterior_soft_labels(
        draws,
        records.loc[:, mapping_columns],
        anchors=DEFAULT_ANCHORS,
        relative_half_width=float(relative_half_width),
        expected_draws_per_record=EXPECTED_DRAWS_PER_RECORD,
    )
    validate_soft_labels(raw, anchors=DEFAULT_ANCHORS)
    hierarchical = derive_hierarchical_soft_labels(
        raw, anchors=DEFAULT_ANCHORS
    )
    validate_hierarchical_soft_labels(hierarchical)
    raw.to_csv(output_dir / "posterior_soft_labels.csv", index=False)
    hierarchical.to_csv(
        output_dir / "hierarchical_soft_labels.csv", index=False
    )
    support_summary(raw, anchors=DEFAULT_ANCHORS).to_csv(
        output_dir / "raw_anchor_support.csv", index=False
    )
    hierarchical_support_summary(hierarchical).to_csv(
        output_dir / "hierarchical_class_support.csv", index=False
    )
    write_json(
        output_dir / "label_definition.json",
        {
            "id": "classic_hierarchical_band20_v2",
            "anchors_mV_dec": list(DEFAULT_ANCHORS),
            "relative_half_width": float(relative_half_width),
            "supervised_soft_label_columns": list(SOFT_LABEL_COLUMNS),
            "posterior_draws_are_not_independent_samples": True,
            "mechanistic_claim_allowed": False,
        },
    )
    return hierarchical


def _merge_labels(
    records: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        name
        for name in labels.columns
        if name == "record_id" or name not in records.columns
    ]
    merged = records.merge(
        labels.loc[:, columns],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 97:
        raise ValueError(
            f"Cardinal development requires 97 records, found {len(merged)}"
        )
    return merged


def _as_contrast_view(
    result: CardinalEvaluationResult,
    families: Sequence[str],
    *,
    joint_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_rows: list[pd.DataFrame] = []
    split_rows: list[pd.DataFrame] = []
    for family in families:
        if family == joint_label:
            oof = result.oof_aggregated.loc[
                result.oof_aggregated["model_role"].eq("joint_selected")
                & result.oof_aggregated["model_family"].eq(
                    "nested_selected"
                )
            ].copy()
            splits = result.split_metrics.loc[
                result.split_metrics["model_role"].eq("joint_selected")
                & result.split_metrics["model_family"].eq(
                    "nested_selected"
                )
            ].copy()
        else:
            oof = result.oof_aggregated.loc[
                result.oof_aggregated["model_role"].eq("family_selected")
                & result.oof_aggregated["model_family"].eq(family)
            ].copy()
            splits = result.split_metrics.loc[
                result.split_metrics["model_role"].eq("family_selected")
                & result.split_metrics["model_family"].eq(family)
            ].copy()
        if oof.empty or splits.empty:
            raise RuntimeError(
                f"No nested OOF predictions are available for {family!r}"
            )
        for frame in (oof, splits):
            frame["panel"] = family
            frame["model_role"] = "joint_selected"
            frame["model_family"] = "nested_selected"
        oof_rows.append(oof)
        split_rows.append(splits)
    return (
        pd.concat(oof_rows, ignore_index=True),
        pd.concat(split_rows, ignore_index=True),
    )


def _recipe_contrasts(
    result: CardinalEvaluationResult,
    output_dir: Path,
    *,
    recipe_names: Sequence[str],
    joint_label: str,
    analysis_id: str,
    scope: str,
    panel_role: str,
    bootstrap_replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    families = tuple(dict.fromkeys((*recipe_names, joint_label)))
    oof, splits = _as_contrast_view(
        result, families, joint_label=joint_label
    )
    comparisons = [
        {
            "id": f"{family}_minus_context_log",
            "role": (
                "nested_recipe_and_model_search"
                if family == joint_label
                else "fixed_recipe_family_comparison"
            ),
            "reference": "context_log",
            "augmented": family,
        }
        for family in families
        if family != "context_log"
    ]
    deltas, bootstrap, summary, registry = evaluate_paired_panel_contrasts(
        oof,
        splits,
        comparisons,
        analysis_id=analysis_id,
        scope=scope,
        panel_role=panel_role,
        replicates=int(bootstrap_replicates),
        random_seed=int(random_seed),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(output_dir / "paired_split_deltas.csv", index=False)
    bootstrap.to_csv(
        output_dir / "paired_group_bootstrap.csv", index=False
    )
    summary.to_csv(
        output_dir / "paired_group_bootstrap_summary.csv", index=False
    )
    registry.to_csv(output_dir / "contrast_registry.csv", index=False)
    return summary


def _catalyst_strata_registry(
    alkaline: pd.DataFrame,
    *,
    minimum_records: int,
    minimum_groups: int,
    minimum_class_mass: float,
    minimum_groups_with_mass: int,
    group_mass_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    assigned = pd.Series("", index=alkaline.index, dtype=object)
    for stratum, raw_types in USER_REQUESTED_CATALYST_STRATA.items():
        mask = alkaline["catalyst_type"].astype(str).isin(raw_types)
        assigned.loc[mask] = stratum
        frame = alkaline.loc[mask]
        masses = {
            column: float(frame[column].sum())
            for column in SOFT_LABEL_COLUMNS
        }
        groups_with_mass = {
            column: int(
                frame.loc[
                    frame[column].ge(group_mass_threshold), "cv_group_id"
                ].nunique()
            )
            for column in SOFT_LABEL_COLUMNS
        }
        passed = bool(
            len(frame) >= minimum_records
            and frame["cv_group_id"].nunique() >= minimum_groups
            and all(
                value >= minimum_class_mass for value in masses.values()
            )
            and all(
                value >= minimum_groups_with_mass
                for value in groups_with_mass.values()
            )
        )
        rows.append(
            {
                "catalyst_stratum": stratum,
                "source_catalyst_types": "|".join(raw_types),
                "metal_Ir_merged_into_IrO2": stratum == "IrO2_plus_Ir",
                "n_records": int(len(frame)),
                "n_connected_groups": int(frame["cv_group_id"].nunique()),
                **{
                    f"{column}_effective_mass": value
                    for column, value in masses.items()
                },
                **{
                    f"{column}_groups_ge_{group_mass_threshold:g}": value
                    for column, value in groups_with_mass.items()
                },
                "support_gate_passed": passed,
                "training_status": (
                    "eligible" if passed else "skipped_underpowered"
                ),
            }
        )
    unassigned = alkaline.loc[assigned.eq("")]
    if not unassigned.empty:
        rows.append(
            {
                "catalyst_stratum": "unassigned",
                "source_catalyst_types": "|".join(
                    sorted(unassigned["catalyst_type"].astype(str).unique())
                ),
                "metal_Ir_merged_into_IrO2": False,
                "n_records": int(len(unassigned)),
                "n_connected_groups": int(
                    unassigned["cv_group_id"].nunique()
                ),
                "support_gate_passed": False,
                "training_status": "skipped_not_in_user_mapping",
            }
        )
    return pd.DataFrame(rows)


def _copy_orthogonal_inputs(
    config: StudyConfig,
    destination: Path,
    settings: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, object]]:
    source_registry = config.repository_root / str(
        settings.get(
            "orthogonal_registry_dir", "Data/orthogonal_kinetics"
        )
    )
    source_binned = config.repository_root / str(
        settings.get(
            "frozen_binned_points",
            (
                "Data/03_outputs/tafel_refitting/outputs/20260721_112453/csv/"
                "binned_curve_points.csv"
            ),
        )
    )
    output = destination / "inputs" / "orthogonal_kinetics"
    output.mkdir(parents=True, exist_ok=True)
    copied: dict[str, object] = {}
    for path in sorted(source_registry.iterdir()):
        if not path.is_file():
            continue
        target = output / path.name
        shutil.copy2(path, target)
        copied[path.name] = {
            "source_path": str(path.resolve()),
            "snapshot_path": str(target.resolve()),
            "sha256": sha256_file(target),
        }
    binned_target = output / "binned_curve_points.csv"
    shutil.copy2(source_binned, binned_target)
    copied["binned_curve_points.csv"] = {
        "source_path": str(source_binned.resolve()),
        "snapshot_path": str(binned_target.resolve()),
        "sha256": sha256_file(binned_target),
    }
    return output, binned_target, copied


def _write_report(
    destination: Path,
    *,
    ir_summary: pd.DataFrame,
    hierarchical_summary: pd.DataFrame,
    ir_abstention_summary: pd.DataFrame,
    hierarchical_abstention_summary: pd.DataFrame,
    catalyst_registry: pd.DataFrame,
    orthogonal_qa: Mapping[str, object],
    smoke: bool,
) -> Path:
    lines = [
        "# Fixed-stack and partially pooled cardinal ML development",
        "",
        (
            "This is a structural smoke run, not a scientific estimate."
            if smoke
            else (
                "All performance values come from nested connected-group OOF "
                "predictions aggregated once per record."
            )
        ),
        "",
        "## IrO2 + metallic Ir: fixed recipe comparison",
        "",
        (
            "`context_log` and `legacy_safe_compact` were fixed before "
            "evaluation. Each recipe uses exactly two base classifiers "
            "(soft Gower kNN and soft multinomial logistic); their convex "
            "probability weight is learned from connected-group inner OOF "
            "predictions inside each outer-training fold. No recipe selector "
            "or Optuna study is used."
        ),
        "",
    ]
    for label, summary in (
        ("Ir fixed stack", ir_summary),
        ("Pooled Ru/Ir hierarchy", hierarchical_summary),
    ):
        selected = summary.loc[summary["metric"].eq("soft_log_loss")]
        if selected.empty:
            lines.append(f"- {label}: no soft-log-loss contrast was produced.")
            continue
        row = selected.iloc[0]
        lines.append(
            f"- {label}: `legacy_safe_compact - context_log` = "
            f"{float(row['observed_delta']):.4f}, 95% connected-group "
            f"bootstrap CI [{float(row['ci95_low']):.4f}, "
            f"{float(row['ci95_high']):.4f}]."
        )
    lines.extend(
        [
            "",
            "Negative deltas favor `legacy_safe_compact`.",
            "",
            "## RuO2 and IrO2 + Ir: partial pooling",
            "",
            (
                "The hierarchical soft classifier shares global coefficients "
                "between RuO2 and `IrO2_plus_Ir` while allowing "
                "family-specific intercept and coefficient deviations under "
                "Gaussian shrinkage. It is a SciPy L-BFGS-B posterior-mode "
                "(MAP) approximation, not NUTS, MCMC, or a full Bayesian "
                "posterior."
            ),
            "",
            (
                "Each experimental record contributes one fractional soft "
                "response vector; the 8,000 slope draws are not expanded into "
                "pseudo-observations. Both fixed recipes use identical "
                "outer/inner connected-group folds."
            ),
            "",
            "## Abstention",
            "",
            (
                "Predictions with normalized Shannon entropy above 0.90 are "
                "reported as `uncertain`. This is a fixed reporting policy; "
                "it does not change training or the full-coverage log-loss "
                "and Brier scores, which remain primary."
            ),
            "",
            "| Analysis | Panel | Coverage | Uncertain |",
            "|---|---|---:|---:|",
        ]
    )
    for label, summary in (
        ("Ir fixed stack", ir_abstention_summary),
        ("Pooled Ru/Ir hierarchy", hierarchical_abstention_summary),
    ):
        for _, row in summary.iterrows():
            lines.append(
                f"| {label} | {row['panel']} | "
                f"{float(row['coverage']):.3f} | "
                f"{float(row['uncertain_rate']):.3f} |"
            )
    lines.extend(
        [
            "",
            (
                "Metallic Ir is mapped into `IrO2_plus_Ir` for modelling; its "
                "original material label remains in the record registry. This "
                "is a user-directed catalyst-family grouping, not evidence "
                "that the catalysts are the same batch or mechanism."
            ),
            "",
            "## Orthogonal kinetic evidence",
            "",
            (
                "Source-verified pH, reaction order, and controlled "
                "formulation rows currently remain unavailable. They were not "
                "imputed. The frozen 97-curve binned coordinates passed the "
                "source-hash audit and are eligible only for a separate "
                "slope-versus-potential auxiliary-outcome analysis."
                if orthogonal_qa.get("binned_points_hash_matches")
                else (
                    "The frozen slope-profile coordinate source did not pass "
                    "its hash audit and is blocked."
                )
            ),
            "",
            (
                "Clustering, SMOTE, deep learning, and larger CatBoost/XGBoost "
                "searches are not part of this mainline. Clustering may only "
                "be used separately to explore formulation archetypes."
            ),
            "",
            "No result in this workflow is mechanistic or causal proof.",
            "",
        ]
    )
    report = destination / "report" / "cardinal_development_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_cardinal_development(
    config: StudyConfig,
    run_dir: str | Path,
    *,
    source_run: str | Path,
    smoke: bool = False,
    n_jobs: int = 1,
) -> dict[str, object]:
    """Run the fixed-stack and Ru/Ir partial-pooling development design."""

    source, destination = assert_disjoint_cardinal_paths(
        source_run, run_dir
    )
    if not source.is_dir():
        raise FileNotFoundError(f"Formal source run does not exist: {source}")
    settings = config.section("cardinal_development")
    classification_settings = config.section("cardinal_classification")
    _validate_settings(settings)
    for name in (
        "inputs",
        "registry",
        "analyses",
        "orthogonal_kinetics",
        "report",
        "provenance",
    ):
        (destination / name).mkdir(parents=True, exist_ok=True)

    records, targets, draws, input_binding = snapshot_cardinal_inputs(
        source,
        destination / "inputs",
        canonical_dataset=config.resolve_path("canonical_dataset"),
        expected_source_hashes=classification_settings.get(
            "source_snapshot_sha256", {}
        ),
        require_formal_approval=not smoke,
    )
    labels = _derive_labels(
        draws,
        records,
        destination / "registry",
        relative_half_width=float(settings["relative_half_width"]),
    )
    frame = _merge_labels(records, labels)
    alkaline = frame.loc[
        frame["electrolyte_scope"].astype(str).eq("alkaline")
    ].reset_index(drop=True)
    if len(alkaline) != 90:
        raise ValueError(f"Expected 90 alkaline records, found {len(alkaline)}")
    expected_catalysts = {"RuO2": 47, "IrO2": 42, "Ir": 1}
    observed_catalysts = (
        alkaline["catalyst_type"].astype(str).value_counts().to_dict()
    )
    if observed_catalysts != expected_catalysts:
        raise ValueError(
            "Unexpected alkaline catalyst-family composition: "
            f"{observed_catalysts}"
        )

    cardinal_feature_recipe_registry().to_csv(
        destination / "registry" / "feature_recipe_registry.csv",
        index=False,
    )
    cardinal_engineered_feature_definitions().to_csv(
        destination / "registry" / "engineered_feature_definitions.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "analysis": "IrO2_plus_Ir_fixed_stack",
                "population": "alkaline IrO2 plus metallic Ir",
                "feature_recipes": "|".join(DEFAULT_FIXED_RECIPES),
                "model": (
                    "soft_gower_knn_plus_soft_multinomial_logistic_"
                    "inner_OOF_convex_stack"
                ),
                "recipe_selector_used": False,
                "optuna_used": False,
            },
            {
                "analysis": "Ru_Ir_hierarchical_MAP",
                "population": "alkaline RuO2 and IrO2_plus_Ir",
                "feature_recipes": "|".join(DEFAULT_FIXED_RECIPES),
                "model": "two_stage_hierarchical_gaussian_prior_MAP",
                "recipe_selector_used": False,
                "optuna_used": False,
            },
        ]
    ).to_csv(destination / "registry" / "active_model_registry.csv", index=False)

    gate_settings = settings["catalyst_support_gate"]
    catalyst_registry = _catalyst_strata_registry(
        alkaline,
        minimum_records=int(gate_settings["minimum_records"]),
        minimum_groups=int(gate_settings["minimum_connected_groups"]),
        minimum_class_mass=float(gate_settings["minimum_soft_class_mass"]),
        minimum_groups_with_mass=int(
            gate_settings["minimum_groups_with_q_ge_threshold"]
        ),
        group_mass_threshold=float(
            gate_settings["group_probability_threshold"]
        ),
    )
    catalyst_registry["active_analysis_role"] = catalyst_registry[
        "catalyst_stratum"
    ].map(
        {
            "IrO2_plus_Ir": (
                "fixed_stack_and_joint_hierarchical_partial_pooling"
            ),
            "RuO2": "joint_hierarchical_partial_pooling_only",
        }
    ).fillna("not_in_active_model")
    catalyst_registry["independent_model_fitted"] = False
    catalyst_registry.to_csv(
        destination / "registry" / "catalyst_strata_eligibility.csv",
        index=False,
    )

    ir_types = USER_REQUESTED_CATALYST_STRATA["IrO2_plus_Ir"]
    ir_frame = alkaline.loc[
        alkaline["catalyst_type"].astype(str).isin(ir_types)
    ].reset_index(drop=True)
    if len(ir_frame) != 43:
        raise ValueError(f"Expected 43 Ir-family records, found {len(ir_frame)}")
    ir_dir = (
        destination / "analyses" / "catalyst_type_strata" / "IrO2_plus_Ir"
    )
    ir_dir.mkdir(parents=True, exist_ok=True)
    ir_frame.loc[
        :, ["record_id", "catalyst_type", "cv_group_id"]
    ].rename(columns={"catalyst_type": "catalyst_type_original"}).to_csv(
        ir_dir / "stratum_record_registry.csv", index=False
    )
    ir_model_frame = ir_frame.copy()
    ir_model_frame["catalyst_type"] = "IrO2_plus_Ir"
    fixed_settings = settings["ir_fixed_comparison"]
    fixed_config = _ir_fixed_stacking_config(settings, n_jobs=n_jobs)
    ir_result = run_fixed_recipe_stacking_evaluation(
        ir_model_frame,
        ir_dir,
        analysis_id="IrO2_plus_Ir_fixed_stack_band20",
        scope="alkaline_IrO2_plus_Ir",
        panel_role="fixed_recipe_probability_stacking",
        config=fixed_config,
        recipes=DEFAULT_FIXED_RECIPES,
        reference_recipe=str(fixed_settings["reference_recipe"]),
        augmented_recipe=str(fixed_settings["augmented_recipe"]),
        gower_params=dict(fixed_settings["soft_gower_knn"]),
        logistic_params=dict(fixed_settings["soft_multinomial_logistic"]),
        smoke=smoke,
    )
    ir_abstention = _write_development_abstention(
        ir_result.oof_aggregated,
        ir_dir / "abstention",
        settings,
    )
    write_json(
        ir_dir / "stratum_qa.json",
        {
            **ir_result.qa,
            "source_catalyst_types": list(ir_types),
            "model_catalyst_label": "IrO2_plus_Ir",
            "original_catalyst_identity_preserved": True,
            "metal_Ir_merged_into_IrO2": True,
            "analysis_mode": "fixed_recipe_convex_probability_stacking",
            "abstention": dict(ir_abstention["qa"]),
        },
    )

    hierarchical = _run_hierarchical_fixed_comparison(
        alkaline,
        destination / "analyses" / "hierarchical_map",
        settings,
        smoke=smoke,
    )

    orthogonal_input, binned_input, orthogonal_binding = (
        _copy_orthogonal_inputs(config, destination, settings)
    )
    orthogonal_audit = audit_orthogonal_kinetics(
        records,
        orthogonal_input,
        binned_input,
        destination / "orthogonal_kinetics",
    )
    write_json(
        destination / "provenance" / "orthogonal_input_binding.json",
        orthogonal_binding,
    )

    def feature_gate(summary: pd.DataFrame) -> bool:
        selected = summary.loc[summary["metric"].eq("soft_log_loss")]
        return bool(
            not selected.empty
            and float(selected.iloc[0]["observed_delta"]) < 0.0
            and float(selected.iloc[0]["ci95_high"]) < 0.0
        )

    decision = {
        "status": (
            "smoke_structural_only" if smoke else "exploratory_complete"
        ),
        "active_design": settings["active_design"],
        "primary_metric": "full_coverage_soft_log_loss",
        "secondary_full_coverage_metric": "multiclass_brier",
        "delta_sign": (
            "legacy_safe_compact_minus_context_log; negative favors "
            "legacy_safe_compact"
        ),
        "IrO2_plus_Ir_fixed_stack": {
            "feature_gate_passed": feature_gate(
                ir_result.paired_bootstrap_summary
            ),
            "paired_contrasts": (
                ir_result.paired_bootstrap_summary.to_dict("records")
            ),
            "joint_recipe_selector_used": False,
            "optuna_used": False,
            "stacking_weight_target_scope": (
                "outer_training_inner_connected_group_oof_only"
            ),
        },
        "Ru_Ir_hierarchical_MAP": {
            "feature_gate_passed": feature_gate(
                hierarchical["paired_bootstrap_summary"]
            ),
            "paired_contrasts": hierarchical[
                "paired_bootstrap_summary"
            ].to_dict("records"),
            **hierarchical["qa"],
        },
        "abstention": {
            "policy": settings["abstention"],
            "changes_primary_scores": False,
            "IrO2_plus_Ir": dict(ir_abstention["qa"]),
            "Ru_Ir_hierarchical_MAP": dict(
                hierarchical["abstention"]["qa"]
            ),
        },
        "retired_recipe_search_executed": False,
        "independent_Ru_model_fitted": False,
        "excluded_from_mainline": settings["excluded_from_mainline"],
        "metal_Ir_merge_user_directed": True,
        "orthogonal_predictor_eligibility": (
            orthogonal_audit.predictor_eligibility.to_dict("records")
        ),
        "slope_vs_potential_is_predictor": False,
        "controlled_formulation_analysis_fitted": bool(
            orthogonal_audit.qa["eligible_controlled_formulation_series"]
        ),
        "mechanistic_claim_allowed": False,
    }
    write_json(
        destination / "cardinal_development_decision.json", decision
    )

    final_snapshot_verification = verify_cardinal_input_snapshot(
        destination / "inputs",
        classification_settings.get("source_snapshot_sha256", {}),
    )
    source_verification = verify_source_run_unchanged(
        source,
        expected_tree_sha256=str(
            input_binding["source_run_tree_sha256_before"]
        ),
    )
    write_json(
        destination / "provenance" / "cardinal_development_binding.json",
        {
            **input_binding,
            "final_snapshot_verification": final_snapshot_verification,
            "final_source_verification": source_verification,
        },
    )
    write_json(
        destination / "provenance" / "resolved_cardinal_development_config.json",
        {
            "settings": settings,
            "IrO2_plus_Ir_fixed_stacking_config": asdict(fixed_config),
            "hierarchical_shrinkage_candidates": [
                asdict(row) for row in _hierarchical_candidates(settings)
            ],
            "smoke": bool(smoke),
            "n_jobs": int(n_jobs),
        },
    )
    report = _write_report(
        destination,
        ir_summary=ir_result.paired_bootstrap_summary,
        hierarchical_summary=hierarchical["paired_bootstrap_summary"],
        ir_abstention_summary=ir_abstention["summary"],
        hierarchical_abstention_summary=hierarchical["abstention"]["summary"],
        catalyst_registry=catalyst_registry,
        orthogonal_qa=orthogonal_audit.qa,
        smoke=smoke,
    )
    qa = {
        "status": "complete_smoke" if smoke else "complete",
        "source_run_unchanged": True,
        "input_snapshot_unchanged_during_modeling": True,
        "n_records_all": int(len(frame)),
        "n_records_alkaline": int(len(alkaline)),
        "n_records_IrO2_plus_Ir": int(len(ir_frame)),
        "n_records_RuO2": int(
            alkaline["catalyst_type"].astype(str).eq("RuO2").sum()
        ),
        "n_posterior_draws": int(len(draws)),
        "active_fixed_recipes": list(DEFAULT_FIXED_RECIPES),
        "retired_recipe_search_executed": False,
        "independent_catalyst_models_fitted": False,
        "IrO2_plus_Ir_fixed_stack": ir_result.qa,
        "hierarchical_partial_pooling": hierarchical["qa"],
        "abstention": {
            "IrO2_plus_Ir_fixed_stack": dict(ir_abstention["qa"]),
            "hierarchical_partial_pooling": dict(
                hierarchical["abstention"]["qa"]
            ),
            "full_coverage_primary_metrics_unchanged": True,
        },
        "metal_Ir_merged_into_IrO2": True,
        "orthogonal_kinetics": orthogonal_audit.qa,
        "missing_orthogonal_values_inferred": False,
        "excluded_from_mainline": settings["excluded_from_mainline"],
        "report_created": report.is_file(),
        "mechanistic_claim_allowed": False,
    }
    write_json(destination / "cardinal_development_qa.json", qa)
    return {
        "run_dir": str(destination),
        "report": str(report),
        "qa": str(destination / "cardinal_development_qa.json"),
        "decision": str(
            destination / "cardinal_development_decision.json"
        ),
        "status": qa["status"],
    }


__all__ = ["run_cardinal_development"]
