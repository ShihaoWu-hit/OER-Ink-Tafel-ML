"""Independent end-to-end posterior-aware cardinal classification workflow."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .cardinal_evaluate import (
    CardinalEvaluationConfig,
    evaluate_paired_panel_contrasts,
    fit_final_cardinal_pipeline,
    run_nested_cardinal_evaluation,
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
from .cardinal_metrics import SOFT_LABEL_COLUMNS
from .cardinal_report import build_cardinal_report
from .config import StudyConfig
from .features import panel_schema_frame, resolve_feature_panel
from .provenance import write_json


FROZEN_PRIMARY_PANELS = ("context", "context_plus_ink")
FROZEN_PANEL_SENSITIVITY_PANELS = (
    "context_plus_ink_amounts",
    "context_plus_ink_process",
    "context_plus_property_proxy_core",
    "context_plus_property_proxy_4p_estimated",
)
FROZEN_POPULATION_SENSITIVITY_PANELS = ("context", "context_plus_ink")
FROZEN_TRAINING_HALF_WIDTHS = (0.15, 0.20, 0.25)
FROZEN_PANEL_CONTRASTS = (
    (
        "process_minus_ink",
        "process_increment",
        "context_plus_ink",
        "context_plus_ink_process",
    ),
    (
        "property_core_minus_ink_amounts",
        "property_increment",
        "context_plus_ink_amounts",
        "context_plus_property_proxy_core",
    ),
    (
        "property_estimated_minus_ink_amounts",
        "property_increment",
        "context_plus_ink_amounts",
        "context_plus_property_proxy_4p_estimated",
    ),
    (
        "property_core_minus_fractions",
        "property_representation",
        "context_plus_ink",
        "context_plus_property_proxy_core",
    ),
    (
        "property_estimated_minus_fractions",
        "property_representation",
        "context_plus_ink",
        "context_plus_property_proxy_4p_estimated",
    ),
)


def _tuple_values(
    value: Any,
    *,
    cast,
    field: str,
) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty sequence")
    return tuple(cast(item) for item in value)


def _validate_frozen_cardinal_settings(settings: Mapping[str, Any]) -> None:
    """Fail if declarative provenance drifts from hard-coded scientific rules."""

    label = settings.get("label_scheme", {})
    validation = settings.get("validation", {})
    comparison = settings.get("primary_comparison", {})
    final_pipeline = settings.get("final_pipeline", {})
    checks = {
        "source_target": (
            str(settings.get("source_target"))
            == "b_bayesian_local_line_mV_dec"
        ),
        "label_scheme.id": (
            str(label.get("id")) == "classic_hierarchical_band20_v2"
        ),
        "label_scheme.anchors_mV_dec": tuple(
            float(value) for value in label.get("anchors_mV_dec", ())
        )
        == DEFAULT_ANCHORS,
        "label_scheme.relative_half_width": float(
            label.get("relative_half_width", float("nan"))
        )
        == 0.20,
        "label_scheme.training_sensitivity_half_widths": tuple(
            float(value)
            for value in label.get("training_sensitivity_half_widths", ())
        )
        == FROZEN_TRAINING_HALF_WIDTHS,
        "label_scheme.supervised_class_order": tuple(
            str(value) for value in label.get("supervised_class_order", ())
        )
        == HIERARCHICAL_CLASS_LABELS,
        "label_scheme.low_classic_definition": str(
            label.get("low_classic_definition")
        )
        == "q_30_like_plus_q_40_like",
        "label_scheme.overlap_policy": str(label.get("overlap_policy"))
        == "nearest_anchor_in_log_space",
        "label_scheme.nonpositive_draw_policy": str(
            label.get("nonpositive_draw_policy")
        )
        == "off_cardinal",
        "label_scheme.nonfinite_draw_policy": str(
            label.get("nonfinite_draw_policy")
        )
        == "fail",
        "label_scheme.posterior_draws_are_not_independent_samples": (
            label.get("posterior_draws_are_not_independent_samples") is True
        ),
        "label_scheme.low_anchor_subdivision.affects_primary_selection": (
            label.get("low_anchor_subdivision", {}).get(
                "affects_primary_selection"
            )
            is False
        ),
        "models.soft_ordinal_logistic_enabled": (
            "soft_ordinal_logistic"
            in tuple(settings.get("models", {}).get("enabled", ()))
        ),
        "validation.primary_metric": str(validation.get("primary_metric"))
        == "soft_log_loss",
        "validation.model_selection_rule": str(
            validation.get("model_selection_rule")
        )
        == "one_standard_error_simplest",
        "primary_comparison.delta_sign": str(comparison.get("delta_sign"))
        == "augmented_minus_reference",
        "primary_panels": tuple(settings.get("primary_panels", ()))
        == FROZEN_PRIMARY_PANELS,
        "panel_sensitivity_panels": tuple(
            settings.get("panel_sensitivity_panels", ())
        )
        == FROZEN_PANEL_SENSITIVITY_PANELS,
        "population_sensitivity_panels": tuple(
            settings.get("population_sensitivity_panels", ())
        )
        == FROZEN_POPULATION_SENSITIVITY_PANELS,
        "final_pipeline.ink_panel_requires_bootstrap_ci_below_zero": (
            final_pipeline.get(
                "ink_panel_requires_bootstrap_ci_below_zero"
            )
            is True
        ),
        "final_pipeline.otherwise_use_reference_panel": (
            final_pipeline.get("otherwise_use_reference_panel") is True
        ),
    }
    try:
        configured_contrasts = tuple(
            (
                row["id"],
                row["role"],
                row["reference"],
                row["augmented"],
            )
            for row in _panel_contrasts(settings)
        )
    except (KeyError, TypeError, ValueError):
        configured_contrasts = ()
    checks["panel_sensitivity_contrasts"] = (
        configured_contrasts == FROZEN_PANEL_CONTRASTS
    )
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise ValueError(
            "Frozen cardinal-classification config disagrees with executable "
            f"policy: {failures}"
        )


def _panel_contrasts(settings: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    raw = settings.get("panel_sensitivity_contrasts")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "cardinal_classification.panel_sensitivity_contrasts must be a "
            "non-empty list"
        )
    rows: list[dict[str, str]] = []
    for position, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Panel contrast at position {position} must be a mapping"
            )
        missing = sorted(
            {"id", "role", "reference", "augmented"} - set(value)
        )
        if missing:
            raise ValueError(
                f"Panel contrast at position {position} is missing: {missing}"
            )
        rows.append(
            {
                key: str(value[key])
                for key in ("id", "role", "reference", "augmented")
            }
        )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Panel sensitivity contrast IDs must be unique")
    return tuple(rows)


def _validate_registered_panel_schema(
    input_dir: Path,
    generated: pd.DataFrame,
    panel_names: tuple[str, ...],
) -> None:
    registered = pd.read_csv(input_dir / "registered_panel_schema.csv")
    columns = [
        "panel",
        "logical_feature",
        "selected_column",
        "kind",
        "distance_block",
        "status",
    ]
    missing = sorted(set(columns) - set(registered.columns))
    if missing:
        raise KeyError(f"Registered panel schema is missing columns: {missing}")
    expected = (
        registered.loc[registered["panel"].isin(panel_names), columns]
        .sort_values(["panel", "logical_feature"], kind="stable")
        .reset_index(drop=True)
    )
    observed = (
        generated.loc[:, columns]
        .sort_values(["panel", "logical_feature"], kind="stable")
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as error:
        raise ValueError(
            "Resolved classification feature panels differ from the "
            "hash-bound registered panel schema"
        ) from error


def _assert_identical_outer_assignments(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> None:
    columns = [
        "repeat",
        "fold",
        "record_id",
        "cv_group_id",
        "split_seed",
    ]
    left_ordered = left.loc[:, columns].sort_values(columns, kind="stable").reset_index(
        drop=True
    )
    right_ordered = right.loc[:, columns].sort_values(
        columns, kind="stable"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left_ordered, right_ordered)
    except AssertionError as error:
        raise ValueError(
            "Primary and panel-sensitivity analyses do not use identical "
            "connected-group outer folds"
        ) from error


def _evaluation_config(
    settings: Mapping[str, Any],
    *,
    panels: tuple[str, ...],
    n_jobs: int,
) -> CardinalEvaluationConfig:
    validation = settings.get("validation", {})
    models = settings.get("models", {})
    logistic = models.get("soft_multinomial_logistic", {})
    ordinal = models.get("soft_ordinal_logistic", {})
    gower = models.get("soft_gower_knn", {})
    trees = models.get("soft_extra_trees", {})
    comparison = settings.get("primary_comparison", {})
    outer_seeds = _tuple_values(
        validation.get("outer_seeds"),
        cast=int,
        field="cardinal_classification.validation.outer_seeds",
    )
    logistic_C_grid = _tuple_values(
        logistic.get("C_grid"),
        cast=float,
        field="cardinal_classification.models.soft_multinomial_logistic.C_grid",
    )
    ordinal_C_grid = _tuple_values(
        ordinal.get("C_grid"),
        cast=float,
        field="cardinal_classification.models.soft_ordinal_logistic.C_grid",
    )
    if ordinal_C_grid != logistic_C_grid:
        raise ValueError(
            "Ordinal and multinomial logistic candidates must use the same "
            "predeclared C grid"
        )
    return CardinalEvaluationConfig(
        outer_folds=int(validation.get("outer_folds", 5)),
        outer_repeats=int(validation.get("outer_repeats", 10)),
        inner_folds=int(validation.get("inner_folds", 4)),
        random_seed=int(outer_seeds[0]),
        outer_seeds=outer_seeds,
        bootstrap_replicates=int(
            validation.get("group_bootstrap_repetitions", 2000)
        ),
        calibration_bins=int(validation.get("calibration_bins", 8)),
        panel_names=panels,
        model_names=_tuple_values(
            models.get("enabled"),
            cast=str,
            field="cardinal_classification.models.enabled",
        ),
        logistic_C_grid=logistic_C_grid,
        gower_neighbors=_tuple_values(
            gower.get("k_grid"),
            cast=int,
            field="cardinal_classification.models.soft_gower_knn.k_grid",
        ),
        gower_distance_powers=_tuple_values(
            gower.get("distance_power_grid"),
            cast=float,
            field="cardinal_classification.models.soft_gower_knn.distance_power_grid",
        ),
        extra_trees_n_estimators=int(trees.get("n_estimators", 100)),
        extra_trees_max_depths=_tuple_values(
            trees.get("max_depth_grid"),
            cast=int,
            field="cardinal_classification.models.soft_extra_trees.max_depth_grid",
        ),
        extra_trees_min_samples_leaf=_tuple_values(
            trees.get("min_samples_leaf_grid"),
            cast=int,
            field="cardinal_classification.models.soft_extra_trees.min_samples_leaf_grid",
        ),
        reference_panel=str(comparison.get("reference", "context")),
        augmented_panel=str(comparison.get("augmented", "context_plus_ink")),
        n_jobs=int(n_jobs),
    )


def _label_definition(
    settings: Mapping[str, Any],
    *,
    relative_half_width: float | None = None,
) -> dict[str, Any]:
    label = settings.get("label_scheme", {})
    anchors = _tuple_values(
        label.get("anchors_mV_dec", DEFAULT_ANCHORS),
        cast=float,
        field="cardinal_classification.label_scheme.anchors_mV_dec",
    )
    primary_width = float(label.get("relative_half_width", 0.20))
    width = (
        primary_width
        if relative_half_width is None
        else float(relative_half_width)
    )
    return {
        "scheme_id": str(
            label.get("id", "classic_hierarchical_band20_v2")
        ),
        "scheme_instance_id": (
            f"classic_hierarchical_band{int(round(width * 100)):02d}_v2"
        ),
        "scheme_role": "primary" if width == primary_width else "sensitivity",
        "target": str(
            settings.get("source_target", "b_bayesian_local_line_mV_dec")
        ),
        "anchors_mV_dec": list(anchors),
        "relative_half_width": width,
        "bands_mV_dec": {
            f"{int(anchor) if anchor.is_integer() else anchor:g}_like": [
                anchor * (1.0 - width),
                anchor * (1.0 + width),
            ]
            for anchor in anchors
        },
        "raw_anchor_class_order": [
            *[
                f"{int(anchor) if anchor.is_integer() else anchor:g}_like"
                for anchor in anchors
            ],
            "off_cardinal",
        ],
        "supervised_class_order": list(HIERARCHICAL_CLASS_LABELS),
        "supervised_soft_target_columns": list(
            HIERARCHICAL_SOFT_LABEL_COLUMNS
        ),
        "low_classic_definition": "q_30_like + q_40_like",
        "low_anchor_subdivision": dict(
            label.get("low_anchor_subdivision", {})
        ),
        "overlap_policy": str(
            label.get("overlap_policy", "nearest_anchor_in_log_space")
        ),
        "nonpositive_draw_policy": str(
            label.get("nonpositive_draw_policy", "off_cardinal")
        ),
        "nonfinite_draw_policy": str(label.get("nonfinite_draw_policy", "fail")),
        "posterior_draws_are_not_independent_samples": True,
        "mechanistic_claim_allowed": False,
    }


def _validate_property_panel_status(
    input_dir: Path,
    panel_names: tuple[str, ...],
) -> dict[str, object]:
    qa_path = input_dir / "property_proxy_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    estimated_panel = "context_plus_property_proxy_4p_estimated"
    status = str(qa.get("estimated_panel_model_use_status", "not_registered"))
    allowed = {
        "eligible_user_approved_with_estimate_indicator",
        "eligible_sensitivity_only",
    }
    if estimated_panel in panel_names and status not in allowed:
        raise ValueError(
            f"Property panel {estimated_panel!r} is not eligible: {status!r}"
        )
    return {
        "estimated_panel_status": status,
        "estimated_panel_included": estimated_panel in panel_names,
        "eligible_statuses": sorted(allowed),
        "decision": (
            "included"
            if estimated_panel in panel_names
            else "not_requested"
        ),
    }


def _width_tag(relative_half_width: float) -> str:
    return f"width_{int(round(float(relative_half_width) * 100)):02d}"


def _low_anchor_subdivision_gate(
    labels: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    relative_half_width: float,
) -> dict[str, object]:
    """Decide whether a separate 30-vs-40 learner has adequate support."""

    policy = settings.get("label_scheme", {}).get(
        "low_anchor_subdivision", {}
    )
    minimum_mass = float(policy.get("min_effective_mass_per_subclass", 5.0))
    minimum_groups = int(
        policy.get("min_conditional_majority_groups_per_subclass", 5)
    )
    rows: list[dict[str, object]] = []
    populations = [("all", labels)]
    if "electrolyte_scope" in labels:
        populations.append(
            (
                "alkaline",
                labels.loc[
                    labels["electrolyte_scope"].astype(str).eq("alkaline")
                ],
            )
        )
    for population, frame in populations:
        low_mass = frame["q_30_like"] + frame["q_40_like"]
        q30_is_conditional_modal = frame["q_30_like"] >= frame["q_40_like"]
        q40_is_conditional_modal = frame["q_40_like"] > frame["q_30_like"]
        has_low_mass = low_mass > 0
        group_col = (
            "cv_group_id" if "cv_group_id" in frame.columns else "record_id"
        )
        rows.append(
            {
                "population": population,
                "n_records": int(len(frame)),
                "q_30_effective_posterior_mass": float(
                    frame["q_30_like"].sum()
                ),
                "q_40_effective_posterior_mass": float(
                    frame["q_40_like"].sum()
                ),
                "q_30_conditional_modal_group_count": int(
                    frame.loc[
                        has_low_mass & q30_is_conditional_modal, group_col
                    ].nunique()
                ),
                "q_40_conditional_modal_group_count": int(
                    frame.loc[
                        has_low_mass & q40_is_conditional_modal, group_col
                    ].nunique()
                ),
            }
        )
    primary = next(row for row in rows if row["population"] == "alkaline")
    passed = bool(
        primary["q_30_effective_posterior_mass"] >= minimum_mass
        and primary["q_40_effective_posterior_mass"] >= minimum_mass
        and primary["q_30_conditional_modal_group_count"] >= minimum_groups
        and primary["q_40_conditional_modal_group_count"] >= minimum_groups
    )
    status = (
        "eligible_but_not_fitted_by_frozen_exploratory_workflow"
        if passed
        else "not_fitted_insufficient_30_support"
    )
    return {
        "status": status,
        "relative_half_width": float(relative_half_width),
        "training_attempted": False,
        "affects_primary_selection": False,
        "assessment_population": "alkaline",
        "minimum_effective_mass_per_subclass": minimum_mass,
        "minimum_conditional_majority_groups_per_subclass": minimum_groups,
        "support_gate_passed": passed,
        "support": rows,
        "reason": (
            "A separate 30-vs-40 predictor is outside the frozen primary "
            "workflow even if future support becomes adequate."
            if passed
            else "The 30-like posterior mass and/or connected-group support "
            "does not meet the predeclared minimum."
        ),
    }


def _write_width_label_registry(
    *,
    draws: pd.DataFrame,
    mapping: pd.DataFrame,
    settings: Mapping[str, Any],
    anchors: tuple[float, ...],
    relative_half_width: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Derive and persist raw-anchor plus hierarchical labels for one width."""

    output_dir.mkdir(parents=True, exist_ok=True)
    definition = _label_definition(
        settings, relative_half_width=relative_half_width
    )
    raw = derive_posterior_soft_labels(
        draws,
        mapping,
        anchors=anchors,
        relative_half_width=float(relative_half_width),
        expected_draws_per_record=EXPECTED_DRAWS_PER_RECORD,
    )
    validate_soft_labels(raw, anchors=anchors)
    hierarchical = derive_hierarchical_soft_labels(raw, anchors=anchors)
    validate_hierarchical_soft_labels(hierarchical)
    raw_support = support_summary(raw, anchors=anchors)
    hierarchical_support = hierarchical_support_summary(hierarchical)
    subdivision = low_anchor_subdivision_table(raw)
    gate = _low_anchor_subdivision_gate(
        raw,
        settings,
        relative_half_width=relative_half_width,
    )
    write_json(output_dir / "cardinal_label_definition.json", definition)
    raw.to_csv(output_dir / "posterior_soft_labels.csv", index=False)
    hierarchical.to_csv(
        output_dir / "hierarchical_soft_labels.csv", index=False
    )
    raw_support.to_csv(output_dir / "class_support.csv", index=False)
    hierarchical_support.to_csv(
        output_dir / "hierarchical_class_support.csv", index=False
    )
    subdivision.to_csv(
        output_dir / "low_anchor_subdivision.csv", index=False
    )
    write_json(output_dir / "low_anchor_subdivision_gate.json", gate)
    return hierarchical, gate, raw_support, hierarchical_support


def _merge_model_frame(
    records: pd.DataFrame,
    hierarchical_labels: pd.DataFrame,
) -> pd.DataFrame:
    derived_columns = [
        column
        for column in hierarchical_labels.columns
        if column == "record_id" or column not in records.columns
    ]
    frame = records.merge(
        hierarchical_labels.loc[:, derived_columns],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != 97:
        raise ValueError(
            f"Cardinal model frame requires 97 records, found {len(frame)}"
        )
    if tuple(SOFT_LABEL_COLUMNS) != tuple(HIERARCHICAL_SOFT_LABEL_COLUMNS):
        raise AssertionError(
            "Evaluation soft-label columns disagree with the hierarchical "
            "registry contract"
        )
    return frame


def _tag_result_frame(
    frame: pd.DataFrame,
    *,
    relative_half_width: float,
    scheme_role: str,
) -> pd.DataFrame:
    tagged = frame.copy()
    tagged.insert(0, "relative_half_width", float(relative_half_width))
    tagged.insert(0, "scheme_role", str(scheme_role))
    return tagged


def run_cardinal_classification(
    config: StudyConfig,
    run_dir: str | Path,
    *,
    source_run: str | Path,
    smoke: bool = False,
    n_jobs: int = 1,
) -> dict[str, object]:
    """Run classification without mutating the formal continuous-response run."""

    source, destination = assert_disjoint_cardinal_paths(source_run, run_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Formal source run does not exist: {source}")
    settings = config.section("cardinal_classification")
    _validate_frozen_cardinal_settings(settings)
    if str(settings.get("status")) != "exploratory_posthoc":
        raise ValueError(
            "cardinal_classification.status must remain exploratory_posthoc "
            "until a prospective protocol amendment is approved"
        )
    if settings.get("mechanistic_claim_allowed") is not False:
        raise ValueError("Cardinal classification cannot allow mechanistic claims")

    for name in (
        "inputs",
        "registry",
        "analyses",
        "final_pipeline",
        "report",
        "provenance",
    ):
        (destination / name).mkdir(parents=True, exist_ok=True)

    records, targets, draws, input_binding = snapshot_cardinal_inputs(
        source,
        destination / "inputs",
        canonical_dataset=config.resolve_path("canonical_dataset"),
        expected_source_hashes=settings.get("source_snapshot_sha256", {}),
        require_formal_approval=not smoke,
    )
    primary_definition = _label_definition(settings)
    anchors = tuple(
        float(value) for value in primary_definition["anchors_mV_dec"]
    )
    primary_width = float(primary_definition["relative_half_width"])
    widths = _tuple_values(
        settings.get("label_scheme", {}).get(
            "training_sensitivity_half_widths"
        ),
        cast=float,
        field=(
            "cardinal_classification.label_scheme."
            "training_sensitivity_half_widths"
        ),
    )
    mapping_columns = [
        column
        for column in (
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
        if column in records.columns
    ]
    labels_by_width: dict[float, pd.DataFrame] = {}
    gates_by_width: dict[float, dict[str, object]] = {}
    support_rows: list[pd.DataFrame] = []
    width_root = destination / "registry" / "label_widths"
    mapping = records.loc[:, mapping_columns]
    for relative_half_width in widths:
        width_dir = width_root / _width_tag(relative_half_width)
        (
            hierarchical,
            subdivision_gate,
            raw_support,
            hierarchical_support,
        ) = _write_width_label_registry(
            draws=draws,
            mapping=mapping,
            settings=settings,
            anchors=anchors,
            relative_half_width=relative_half_width,
            output_dir=width_dir,
        )
        labels_by_width[float(relative_half_width)] = hierarchical
        gates_by_width[float(relative_half_width)] = subdivision_gate
        for target_layer, frame in (
            ("raw_anchor_registry", raw_support),
            ("hierarchical_supervised", hierarchical_support),
        ):
            tagged = frame.copy()
            tagged.insert(0, "target_layer", target_layer)
            tagged.insert(
                0,
                "scheme_role",
                (
                    "primary"
                    if relative_half_width == primary_width
                    else "sensitivity"
                ),
            )
            tagged.insert(
                0, "relative_half_width", float(relative_half_width)
            )
            support_rows.append(tagged)
    if set(labels_by_width) != set(FROZEN_TRAINING_HALF_WIDTHS):
        raise AssertionError("Not all frozen label widths were derived")
    label_width_support = pd.concat(support_rows, ignore_index=True)
    label_width_support.to_csv(
        destination / "registry" / "label_width_support_summary.csv",
        index=False,
    )

    labels = labels_by_width[primary_width]
    primary_width_dir = width_root / _width_tag(primary_width)
    write_json(
        destination / "registry" / "cardinal_label_definition.json",
        primary_definition,
    )
    labels.to_csv(
        destination / "registry" / "hierarchical_soft_labels.csv",
        index=False,
    )
    pd.read_csv(
        primary_width_dir / "posterior_soft_labels.csv"
    ).to_csv(
        destination / "registry" / "posterior_soft_labels.csv", index=False
    )
    pd.read_csv(primary_width_dir / "class_support.csv").to_csv(
        destination / "registry" / "class_support.csv", index=False
    )
    pd.read_csv(
        primary_width_dir / "hierarchical_class_support.csv"
    ).to_csv(
        destination / "registry" / "hierarchical_class_support.csv",
        index=False,
    )
    pd.read_csv(
        primary_width_dir / "low_anchor_subdivision.csv"
    ).to_csv(
        destination / "registry" / "low_anchor_subdivision.csv",
        index=False,
    )
    write_json(
        destination / "registry" / "low_anchor_subdivision_gate.json",
        gates_by_width[primary_width],
    )
    model_frame = _merge_model_frame(records, labels)

    primary_panels = _tuple_values(
        settings.get("primary_panels"),
        cast=str,
        field="cardinal_classification.primary_panels",
    )
    panel_sensitivity_panels = _tuple_values(
        settings.get("panel_sensitivity_panels"),
        cast=str,
        field="cardinal_classification.panel_sensitivity_panels",
    )
    population_sensitivity_panels = _tuple_values(
        settings.get("population_sensitivity_panels"),
        cast=str,
        field="cardinal_classification.population_sensitivity_panels",
    )
    all_panels = tuple(
        dict.fromkeys(
            primary_panels
            + panel_sensitivity_panels
            + population_sensitivity_panels
        )
    )
    for panel_name in all_panels:
        resolve_feature_panel(model_frame, panel_name, require_all=True)
    generated_panel_schema = panel_schema_frame(model_frame, all_panels)
    _validate_registered_panel_schema(
        destination / "inputs",
        generated_panel_schema,
        all_panels,
    )
    generated_panel_schema.to_csv(
        destination / "registry" / "feature_panel_schema.csv",
        index=False,
    )
    property_gate = _validate_property_panel_status(
        destination / "inputs", all_panels
    )
    write_json(
        destination / "registry" / "property_panel_gate.json", property_gate
    )

    primary_scope = str(settings.get("primary_scope", "alkaline"))
    sensitivity_scope = str(settings.get("sensitivity_scope", "all"))
    if primary_scope != "alkaline" or sensitivity_scope != "all":
        raise ValueError(
            "The frozen classification populations must be alkaline primary and all-record sensitivity"
        )
    primary_frame = model_frame.loc[
        model_frame["electrolyte_scope"].astype(str).eq("alkaline")
    ].reset_index(drop=True)
    if len(primary_frame) != 90:
        raise ValueError(f"Expected 90 alkaline records, found {len(primary_frame)}")
    sensitivity_frame = model_frame.reset_index(drop=True)

    primary_config = _evaluation_config(
        settings, panels=primary_panels, n_jobs=n_jobs
    )
    panel_sensitivity_config = _evaluation_config(
        settings, panels=panel_sensitivity_panels, n_jobs=n_jobs
    )
    population_sensitivity_config = _evaluation_config(
        settings, panels=population_sensitivity_panels, n_jobs=n_jobs
    )
    primary_result = run_nested_cardinal_evaluation(
        primary_frame,
        destination / "analyses" / "primary_alkaline",
        analysis_id="primary_alkaline_classic_hierarchical_band20",
        scope="alkaline",
        panel_role="primary",
        config=primary_config,
        smoke=smoke,
    )
    panel_sensitivity_result = run_nested_cardinal_evaluation(
        primary_frame,
        destination / "analyses" / "panel_sensitivity_alkaline",
        analysis_id="panel_sensitivity_alkaline_hierarchical_band20",
        scope="alkaline",
        panel_role="panel_sensitivity",
        config=panel_sensitivity_config,
        smoke=smoke,
        frozen_outer_manifest=primary_result.split_manifest,
        frozen_inner_manifest=primary_result.inner_split_manifest,
    )
    population_sensitivity_result = run_nested_cardinal_evaluation(
        sensitivity_frame,
        destination / "analyses" / "population_sensitivity_all",
        analysis_id="population_sensitivity_all_hierarchical_band20",
        scope="all",
        panel_role="population_sensitivity",
        config=population_sensitivity_config,
        smoke=smoke,
    )
    _assert_identical_outer_assignments(
        primary_result.split_manifest,
        panel_sensitivity_result.split_manifest,
    )

    width_results: dict[
        float, dict[str, Any]
    ] = {
        primary_width: {
            "primary": primary_result,
            "population": population_sensitivity_result,
        }
    }
    for relative_half_width in widths:
        if relative_half_width == primary_width:
            continue
        sensitivity_model_frame = _merge_model_frame(
            records, labels_by_width[relative_half_width]
        )
        width_primary_frame = sensitivity_model_frame.loc[
            sensitivity_model_frame["electrolyte_scope"]
            .astype(str)
            .eq("alkaline")
        ].reset_index(drop=True)
        width_population_frame = sensitivity_model_frame.reset_index(drop=True)
        width_analysis_root = (
            destination
            / "analyses"
            / "label_width_sensitivity"
            / _width_tag(relative_half_width)
        )
        width_primary_result = run_nested_cardinal_evaluation(
            width_primary_frame,
            width_analysis_root / "primary_alkaline",
            analysis_id=(
                "label_width_sensitivity_alkaline_"
                f"band{int(round(relative_half_width * 100)):02d}"
            ),
            scope="alkaline",
            panel_role="label_width_sensitivity_primary",
            config=primary_config,
            smoke=smoke,
            frozen_outer_manifest=primary_result.split_manifest,
            frozen_inner_manifest=primary_result.inner_split_manifest,
        )
        width_population_result = run_nested_cardinal_evaluation(
            width_population_frame,
            width_analysis_root / "population_sensitivity_all",
            analysis_id=(
                "label_width_sensitivity_all_"
                f"band{int(round(relative_half_width * 100)):02d}"
            ),
            scope="all",
            panel_role="label_width_sensitivity_population",
            config=population_sensitivity_config,
            smoke=smoke,
            frozen_outer_manifest=population_sensitivity_result.split_manifest,
            frozen_inner_manifest=(
                population_sensitivity_result.inner_split_manifest
            ),
        )
        _assert_identical_outer_assignments(
            primary_result.split_manifest,
            width_primary_result.split_manifest,
        )
        _assert_identical_outer_assignments(
            population_sensitivity_result.split_manifest,
            width_population_result.split_manifest,
        )
        width_primary_result.qa.update(
            {
                "outer_and_inner_assignments_reused_from_primary_width": True,
                "relative_half_width": float(relative_half_width),
                "scheme_role": "sensitivity",
            }
        )
        width_population_result.qa.update(
            {
                "outer_and_inner_assignments_reused_from_primary_width": True,
                "relative_half_width": float(relative_half_width),
                "scheme_role": "sensitivity",
            }
        )
        write_json(
            width_analysis_root / "primary_alkaline" / "analysis_qa.json",
            width_primary_result.qa,
        )
        write_json(
            width_analysis_root
            / "population_sensitivity_all"
            / "analysis_qa.json",
            width_population_result.qa,
        )
        width_results[float(relative_half_width)] = {
            "primary": width_primary_result,
            "population": width_population_result,
        }

    width_summary_rows: list[pd.DataFrame] = []
    for relative_half_width, result_pair in sorted(width_results.items()):
        for analysis_population, result in result_pair.items():
            summary = result.paired_bootstrap_summary.copy()
            if summary.empty:
                continue
            summary.insert(0, "analysis_population", analysis_population)
            summary.insert(
                0,
                "scheme_role",
                (
                    "primary"
                    if relative_half_width == primary_width
                    else "sensitivity"
                ),
            )
            summary.insert(
                0, "relative_half_width", float(relative_half_width)
            )
            width_summary_rows.append(summary)
    label_width_sensitivity_summary = pd.concat(
        width_summary_rows, ignore_index=True
    )
    label_width_sensitivity_summary[
        "absolute_log_loss_comparable_across_widths"
    ] = False
    label_width_sensitivity_summary[
        "can_override_primary_width_decision"
    ] = label_width_sensitivity_summary["relative_half_width"].eq(
        primary_width
    )
    label_width_sensitivity_summary.to_csv(
        destination / "label_width_sensitivity_summary.csv", index=False
    )

    panel_contrasts = _panel_contrasts(settings)
    (
        panel_contrast_deltas,
        panel_contrast_bootstrap,
        panel_contrast_summary,
        panel_contrast_registry,
    ) = evaluate_paired_panel_contrasts(
        pd.concat(
            [
                primary_result.oof_aggregated,
                panel_sensitivity_result.oof_aggregated,
            ],
            ignore_index=True,
        ),
        pd.concat(
            [
                primary_result.split_metrics,
                panel_sensitivity_result.split_metrics,
            ],
            ignore_index=True,
        ),
        panel_contrasts,
        analysis_id="panel_sensitivity_alkaline_contrasts",
        scope="alkaline",
        panel_role="panel_sensitivity",
        replicates=(
            min(primary_config.bootstrap_replicates, 50)
            if smoke
            else primary_config.bootstrap_replicates
        ),
        random_seed=primary_config.random_seed + 700_000,
    )
    panel_sensitivity_result.paired_panel_deltas = panel_contrast_deltas
    panel_sensitivity_result.paired_bootstrap = panel_contrast_bootstrap
    panel_sensitivity_result.paired_bootstrap_summary = panel_contrast_summary
    panel_sensitivity_result.paired_contrast_registry = panel_contrast_registry
    panel_sensitivity_dir = (
        destination / "analyses" / "panel_sensitivity_alkaline"
    )
    panel_contrast_deltas.to_csv(
        panel_sensitivity_dir / "paired_panel_deltas.csv", index=False
    )
    panel_contrast_bootstrap.to_csv(
        panel_sensitivity_dir / "paired_group_bootstrap.csv", index=False
    )
    panel_contrast_summary.to_csv(
        panel_sensitivity_dir / "paired_group_bootstrap_summary.csv",
        index=False,
    )
    panel_contrast_registry.to_csv(
        panel_sensitivity_dir / "paired_contrast_registry.csv", index=False
    )
    panel_sensitivity_result.qa.update(
        {
            "paired_panel_sensitivity_contrasts_available": True,
            "n_paired_panel_sensitivity_contrasts": int(
                len(panel_contrast_registry)
            ),
            "outer_assignments_identical_to_primary": True,
        }
    )
    write_json(
        panel_sensitivity_dir / "analysis_qa.json",
        panel_sensitivity_result.qa,
    )

    final_pipeline = fit_final_cardinal_pipeline(
        primary_frame,
        destination / "final_pipeline",
        primary_result=primary_result,
        config=primary_config,
        label_scheme_id=str(primary_definition["scheme_id"]),
        relative_half_width=primary_width,
        low_anchor_subdivision=gates_by_width[primary_width],
        training_scope="alkaline",
        smoke=smoke,
    )
    metric_frames = [
        _tag_result_frame(
            primary_result.aggregate_metrics,
            relative_half_width=primary_width,
            scheme_role="primary",
        ),
        _tag_result_frame(
            panel_sensitivity_result.aggregate_metrics,
            relative_half_width=primary_width,
            scheme_role="primary_panel_sensitivity",
        ),
        _tag_result_frame(
            population_sensitivity_result.aggregate_metrics,
            relative_half_width=primary_width,
            scheme_role="primary_population_sensitivity",
        ),
    ]
    frequency_frames = [
        _tag_result_frame(
            primary_result.model_selection_frequency,
            relative_half_width=primary_width,
            scheme_role="primary",
        ),
        _tag_result_frame(
            panel_sensitivity_result.model_selection_frequency,
            relative_half_width=primary_width,
            scheme_role="primary_panel_sensitivity",
        ),
        _tag_result_frame(
            population_sensitivity_result.model_selection_frequency,
            relative_half_width=primary_width,
            scheme_role="primary_population_sensitivity",
        ),
    ]
    for relative_half_width, result_pair in sorted(width_results.items()):
        if relative_half_width == primary_width:
            continue
        for result in result_pair.values():
            metric_frames.append(
                _tag_result_frame(
                    result.aggregate_metrics,
                    relative_half_width=relative_half_width,
                    scheme_role="label_width_sensitivity",
                )
            )
            frequency_frames.append(
                _tag_result_frame(
                    result.model_selection_frequency,
                    relative_half_width=relative_half_width,
                    scheme_role="label_width_sensitivity",
                )
            )
    combined_metrics = pd.concat(metric_frames, ignore_index=True)
    combined_metrics.to_csv(
        destination / "all_analysis_metrics.csv", index=False
    )
    combined_frequency = pd.concat(frequency_frames, ignore_index=True)
    combined_frequency.to_csv(
        destination / "all_selection_frequencies.csv", index=False
    )

    final_snapshot_verification = verify_cardinal_input_snapshot(
        destination / "inputs",
        settings.get("source_snapshot_sha256", {}),
    )
    source_verification = verify_source_run_unchanged(
        source,
        expected_tree_sha256=str(
            input_binding["source_run_tree_sha256_before"]
        ),
    )
    write_json(
        destination / "provenance" / "cardinal_input_binding.json",
        {
            **input_binding,
            "final_snapshot_verification": final_snapshot_verification,
            "final_source_verification": source_verification,
        },
    )
    write_json(
        destination / "provenance" / "resolved_cardinal_config.json",
        {
            "cardinal_classification": settings,
            "primary_evaluation_config": asdict(primary_config),
            "panel_sensitivity_evaluation_config": asdict(
                panel_sensitivity_config
            ),
            "population_sensitivity_evaluation_config": asdict(
                population_sensitivity_config
            ),
            "label_width_evaluation_policy": {
                "primary_relative_half_width": primary_width,
                "fully_trained_relative_half_widths": list(widths),
                "outer_and_inner_assignments_reused_across_widths": True,
                "absolute_log_loss_comparable_across_widths": False,
                "sensitivity_widths_can_override_primary_decision": False,
            },
        },
    )

    decision = {
        "status": "smoke_structural_only" if smoke else "exploratory_complete",
        "primary_question": (
            "whether ink descriptors improve posterior cardinal-region "
            "probability prediction beyond context"
        ),
        "primary_metric": "soft_log_loss",
        "delta_sign": "context_plus_ink_minus_context; negative favors ink",
        "label_scheme_id": str(primary_definition["scheme_id"]),
        "primary_relative_half_width": primary_width,
        "supervised_class_order": list(HIERARCHICAL_CLASS_LABELS),
        "low_classic_definition": "q_30_like + q_40_like",
        "low_anchor_subdivision": gates_by_width[primary_width],
        "primary_paired_bootstrap": primary_result.paired_bootstrap_summary.to_dict(
            "records"
        ),
        "panel_sensitivity_paired_bootstrap": (
            panel_contrast_summary.to_dict("records")
        ),
        "population_sensitivity_paired_bootstrap": (
            population_sensitivity_result.paired_bootstrap_summary.to_dict(
                "records"
            )
        ),
        "label_width_sensitivity": (
            label_width_sensitivity_summary.to_dict("records")
        ),
        "label_width_interpretation_policy": (
            "Only paired context_plus_ink-minus-context deltas are interpreted "
            "within each width. Absolute log loss is not compared across "
            "widths, and 15%/25% cannot override the 20% decision."
        ),
        "final_pipeline": final_pipeline,
        "continuous_regression_replaced": False,
        "mechanistic_claim_allowed": False,
    }
    write_json(
        destination / "cardinal_classification_decision.json", decision
    )
    qa = {
        "status": "complete_smoke" if smoke else "complete",
        "source_run_unchanged": True,
        "input_snapshot_unchanged_during_modeling": True,
        "n_records_all": int(len(model_frame)),
        "n_records_primary_alkaline": int(len(primary_frame)),
        "n_posterior_draws": int(len(draws)),
        "draws_per_record": EXPECTED_DRAWS_PER_RECORD,
        "n_nonpositive_draws": int(labels["n_nonpositive_draws"].sum()),
        "soft_labels_valid": True,
        "supervised_soft_label_columns": list(SOFT_LABEL_COLUMNS),
        "primary_relative_half_width": primary_width,
        "fully_trained_relative_half_widths": list(widths),
        "label_width_outer_and_inner_assignments_identical": True,
        "absolute_log_loss_comparable_across_widths": False,
        "low_anchor_subdivision": gates_by_width[primary_width],
        "primary_analysis": primary_result.qa,
        "panel_sensitivity_analysis": panel_sensitivity_result.qa,
        "population_sensitivity_analysis": population_sensitivity_result.qa,
        "primary_and_panel_sensitivity_outer_assignments_identical": True,
        "n_panel_sensitivity_contrasts": int(len(panel_contrast_registry)),
        "final_pipeline_created": bool(
            (
                destination
                / "final_pipeline"
                / "hierarchical_cardinal_pipeline.joblib"
            ).is_file()
        ),
        "report_created": False,
        "property_panel_gate": property_gate,
        "continuous_regression_outputs_modified": False,
        "mechanistic_claim_allowed": False,
    }
    write_json(destination / "cardinal_classification_qa.json", qa)
    report = build_cardinal_report(destination, smoke=smoke)
    qa["report_created"] = report.is_file()
    write_json(destination / "cardinal_classification_qa.json", qa)
    return {
        "run_dir": str(destination),
        "report": str(report),
        "qa": str(destination / "cardinal_classification_qa.json"),
        "decision": str(
            destination / "cardinal_classification_decision.json"
        ),
        "status": qa["status"],
    }
