"""Posterior-aware fixed-data feature-association workflow.

This stage is deliberately independent from the registered regression and
cardinal-classification analyses.  It answers a narrower question: whether a
small, target-blind predictor block adds reproducible out-of-connected-group
information, and only then whether a cross-fitted adjusted associational curve
may be released.  It does not identify causal or mechanistic effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.utils.parallel import Parallel, delayed

from .cardinal_inputs import (
    EXPECTED_DRAWS_PER_RECORD,
    snapshot_cardinal_inputs,
    verify_cardinal_input_snapshot,
    verify_source_run_unchanged,
)
from .config import StudyConfig
from .evaluate import make_outer_split_manifest, validate_split_manifest
from .feature_association_features import (
    EXPECTED_REGISTERED_CANDIDATE_COUNT,
    FROZEN_ASSOCIATION_PANELS,
    HARD_EXCLUDED_SUPERVISED_SOURCES,
    POOLED_COMPOSITION_SOURCES,
    POOLED_PROPERTY_SOURCES,
    AssociationFeatureBuilder,
    association_engineered_feature_definitions,
    association_panel_registry,
    audit_feature_support_and_redundancy,
    registered_candidate_registry,
)
from .feature_association_figures import generate_feature_association_figures
from .feature_association_models import (
    MODEL_FAMILIES,
    EmpiricalBayesAssociationModel,
)
from .provenance import write_json


TARGET_NAME = "b_bayesian_local_line_mV_dec"
PRIMARY_SCOPE = "all97"
ANALYSIS_SCOPES = (PRIMARY_SCOPE,)
PRIOR_PANEL = "prior"
PRIOR_MODEL = "bayesian_intercept"
FROZEN_PANELS = (
    "context_core",
    "context_plus_catalyst_concentration",
    "context_plus_ionomer_ratio",
    "context_plus_water_organic",
    "context_plus_ethanol_isopropanol",
    "context_plus_viscosity",
    "context_plus_polarity",
    "context_plus_conductive_sensitivity",
)
FROZEN_MODELS = ("bayesian_linear", "gam_low_df")
FROZEN_CONTRASTS = (
    (
        "context_minus_prior",
        "context",
        "prior",
        "context_core",
        "baseline_information",
    ),
    (
        "catalyst_concentration_minus_context",
        "catalyst_concentration",
        "context_core",
        "context_plus_catalyst_concentration",
        "primary_single_feature_increment",
    ),
    (
        "ionomer_ratio_minus_context",
        "ionomer_to_solvent_ratio",
        "context_core",
        "context_plus_ionomer_ratio",
        "primary_single_feature_increment",
    ),
    (
        "water_organic_minus_context",
        "water_to_organic_balance",
        "context_core",
        "context_plus_water_organic",
        "primary_single_feature_increment",
    ),
    (
        "ethanol_isopropanol_minus_context",
        "ethanol_to_isopropanol_balance",
        "context_core",
        "context_plus_ethanol_isopropanol",
        "primary_single_feature_increment",
    ),
    (
        "viscosity_minus_context",
        "solvent_viscosity",
        "context_core",
        "context_plus_viscosity",
        "alternative_property_single_feature",
    ),
    (
        "polarity_minus_context",
        "solvent_polarity",
        "context_core",
        "context_plus_polarity",
        "alternative_property_single_feature",
    ),
    (
        "conductive_minus_context",
        "conductive_agent",
        "context_core",
        "context_plus_conductive_sensitivity",
        "sensitivity_increment",
    ),
)
PRIMARY_MULTIPLICITY_CONTRASTS = (
    "catalyst_concentration_minus_context",
    "ionomer_ratio_minus_context",
    "water_organic_minus_context",
    "ethanol_isopropanol_minus_context",
    "viscosity_minus_context",
    "polarity_minus_context",
)


def _stable_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


@dataclass(frozen=True)
class FeatureAssociationEvaluationConfig:
    outer_folds: int = 5
    outer_repeats: int = 10
    outer_seeds: tuple[int, ...] = (
        42,
        142,
        242,
        342,
        442,
        542,
        642,
        742,
        842,
        942,
    )
    bootstrap_replicates: int = 2000
    target_draw_repetitions: int = 100
    panel_names: tuple[str, ...] = FROZEN_PANELS
    model_names: tuple[str, ...] = FROZEN_MODELS
    model_max_iter: int = 1000
    model_tolerance: float = 1.0e-6
    gam_n_knots: int = 3
    gam_degree: int = 1
    rare_min_frequency: int = 3
    effect_grid_quantiles: tuple[float, ...] = tuple(
        float(value) for value in np.linspace(0.05, 0.95, 19)
    )
    effect_reference_quantile: float = 0.50
    effect_joint_support_threshold_quantile: float = 0.95
    effect_minimum_common_record_coverage: float = 0.80
    minimum_repeat_improvement_fraction: float = 0.80
    minimum_draw_improvement_fraction: float = 0.90
    adjusted_alpha: float = 0.05
    n_jobs: int = 1
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least 2")
        if self.outer_repeats < 1:
            raise ValueError("outer_repeats must be positive")
        if len(self.outer_seeds) != self.outer_repeats:
            raise ValueError("outer_seeds must match outer_repeats")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if self.target_draw_repetitions < 1:
            raise ValueError("target_draw_repetitions must be positive")
        if tuple(self.panel_names) != FROZEN_PANELS:
            raise ValueError("feature-association panels are frozen")
        if tuple(self.model_names) != FROZEN_MODELS:
            raise ValueError("feature-association model families are frozen")
        if tuple(self.model_names) != tuple(MODEL_FAMILIES):
            raise ValueError("Model implementation and frozen config disagree")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be positive")
        quantiles = np.asarray(self.effect_grid_quantiles, dtype=float)
        if (
            quantiles.ndim != 1
            or not len(quantiles)
            or not np.isfinite(quantiles).all()
            or (quantiles <= 0).any()
            or (quantiles >= 1).any()
            or np.any(np.diff(quantiles) <= 0)
        ):
            raise ValueError("effect_grid_quantiles must increase inside (0, 1)")
        if not np.any(
            np.isclose(quantiles, self.effect_reference_quantile, atol=1e-12)
        ):
            raise ValueError("effect_reference_quantile must occur in the grid")
        if not 0 < self.effect_joint_support_threshold_quantile <= 1:
            raise ValueError(
                "effect_joint_support_threshold_quantile must lie in (0, 1]"
            )
        if not 0 < self.effect_minimum_common_record_coverage <= 1:
            raise ValueError(
                "effect_minimum_common_record_coverage must lie in (0, 1]"
            )


def _settings_to_config(
    settings: Mapping[str, Any],
    *,
    smoke: bool,
    n_jobs: int,
) -> FeatureAssociationEvaluationConfig:
    validation = settings.get("validation", {})
    posterior = settings.get("posterior_policy", {})
    models = settings.get("models", {})
    ridge = models.get("bayesian_ridge", {})
    gam = models.get("gam", {})
    effects = settings.get("effects", {})
    gate = settings.get("release_gate", {})
    outer_repeats = int(validation.get("outer_repeats", 10))
    outer_folds = int(validation.get("outer_folds", 5))
    seeds = tuple(int(value) for value in validation.get("outer_seeds", ()))
    bootstrap = int(validation.get("group_bootstrap_repetitions", 2000))
    draw_repetitions = int(
        posterior.get("score_level_target_draw_repetitions", 100)
    )
    if smoke:
        outer_repeats = min(2, outer_repeats)
        outer_folds = min(3, outer_folds)
        seeds = seeds[:outer_repeats]
        bootstrap = min(50, bootstrap)
        draw_repetitions = min(12, draw_repetitions)
    return FeatureAssociationEvaluationConfig(
        outer_folds=outer_folds,
        outer_repeats=outer_repeats,
        outer_seeds=seeds,
        bootstrap_replicates=bootstrap,
        target_draw_repetitions=draw_repetitions,
        panel_names=tuple(str(value) for value in settings.get("frozen_panels", ())),
        model_names=tuple(str(value) for value in models.get("enabled", ())),
        model_max_iter=int(ridge.get("max_iter", 1000)),
        model_tolerance=float(ridge.get("tolerance", 1.0e-6)),
        gam_n_knots=int(gam.get("spline_knots", 3)),
        gam_degree=int(gam.get("spline_degree", 1)),
        effect_grid_quantiles=tuple(
            float(value)
            for value in effects.get(
                "grid_quantiles", np.linspace(0.05, 0.95, 19)
            )
        ),
        effect_reference_quantile=float(
            effects.get("reference_quantile", 0.50)
        ),
        effect_joint_support_threshold_quantile=0.95,
        effect_minimum_common_record_coverage=float(
            effects.get(
                "minimum_common_record_joint_support_coverage", 0.80
            )
        ),
        minimum_repeat_improvement_fraction=float(
            gate.get("minimum_repeat_improvement_fraction", 0.80)
        ),
        minimum_draw_improvement_fraction=float(
            gate.get("minimum_posterior_draw_improvement_fraction", 0.90)
        ),
        adjusted_alpha=float(gate.get("adjusted_alpha", 0.05)),
        n_jobs=int(n_jobs),
        smoke=bool(smoke),
    )


def _configured_contrasts(
    settings: Mapping[str, Any],
) -> tuple[tuple[str, str, str, str, str], ...]:
    rows = settings.get("frozen_contrasts", ())
    if not isinstance(rows, (list, tuple)):
        raise TypeError("feature_association.frozen_contrasts must be a sequence")
    return tuple(
        (
            str(row["id"]),
            str(row["block"]),
            str(row["reference"]),
            str(row["augmented"]),
            str(row["role"]),
        )
        for row in rows
    )


def _validate_frozen_settings(settings: Mapping[str, Any]) -> None:
    posterior = settings.get("posterior_policy", {})
    models = settings.get("models", {})
    gam = models.get("gam", {})
    gate = settings.get("release_gate", {})
    validation = settings.get("validation", {})
    effects = settings.get("effects", {})
    ridge = models.get("bayesian_ridge", {})
    atlas = settings.get("atlas", {})
    reduction = settings.get("feature_reduction", {})
    expected_grid = tuple(
        float(value) for value in np.linspace(0.05, 0.95, 19)
    )
    checks = {
        "source_target": str(settings.get("source_target")) == TARGET_NAME,
        "primary_scope": str(settings.get("primary_scope")) == PRIMARY_SCOPE,
        "analysis_scopes": tuple(settings.get("analysis_scopes", ()))
        == ANALYSIS_SCOPES,
        "no_sensitivity_scope": settings.get("sensitivity_scope") in {
            None,
            "",
        },
        "atlas_scopes": tuple(atlas.get("scopes", ())) == ANALYSIS_SCOPES,
        "fixed_context_plus_one": str(reduction.get("selection_policy"))
        == "fixed_context_plus_one_target_blind",
        "one_feature_per_augmented_panel": reduction.get(
            "one_feature_per_augmented_panel"
        )
        is True,
        "outcome_driven_feature_selection_disabled": reduction.get(
            "outcome_driven_feature_selection_used"
        )
        is False,
        "composition_property_coordinates_exclusive": reduction.get(
            "composition_property_coordinates_never_cooccur"
        )
        is True,
        "hard_exclusions": set(reduction.get("hard_exclusions", ()))
        == set(HARD_EXCLUDED_SUPERVISED_SOURCES),
        "pooled_only_sources": set(reduction.get("pooled_only_sources", ()))
        == set(POOLED_COMPOSITION_SOURCES | POOLED_PROPERTY_SOURCES),
        "mechanistic_claim_allowed": settings.get("mechanistic_claim_allowed")
        is False,
        "causal_claim_allowed": settings.get("causal_claim_allowed") is False,
        "posterior_draws_not_expanded": posterior.get(
            "posterior_draws_expanded_as_samples"
        )
        is False,
        "posterior_training_target": str(
            posterior.get("training_target")
        )
        == "log10_formal_posterior_median",
        "training_records_equally_weighted": posterior.get(
            "training_records_are_equally_weighted"
        )
        is True,
        "posterior_draw_repetitions": int(
            posterior.get("score_level_target_draw_repetitions", -1)
        )
        == 100,
        "original_nonpositive_policy": str(
            posterior.get("nonpositive_draw_policy_for_original_scale")
        )
        == "retained",
        "log_nonpositive_policy": str(
            posterior.get("nonpositive_draw_policy_for_log_metrics")
        )
        == "excluded_and_mass_reported",
        "panels": tuple(settings.get("frozen_panels", ())) == FROZEN_PANELS,
        "models": tuple(models.get("enabled", ())) == FROZEN_MODELS,
        "primary_gate_model": str(models.get("primary_gate_model"))
        == "bayesian_linear",
        "corroborating_model": str(models.get("corroborating_model"))
        == "gam_low_df",
        "effect_curve_model": str(models.get("effect_curve_model"))
        == "gam_low_df",
        "ridge_max_iter": int(ridge.get("max_iter", -1)) == 1000,
        "ridge_tolerance": float(ridge.get("tolerance", np.nan))
        == 1.0e-6,
        "gam_knots": int(gam.get("spline_knots", -1)) == 3,
        "gam_degree": int(gam.get("spline_degree", -1)) == 1,
        "gam_minimum_unique": int(
            gam.get("minimum_unique_values_for_spline", -1)
        )
        == 4,
        "gam_interactions": gam.get("automatic_interactions") is False,
        "outer_folds": int(validation.get("outer_folds", -1)) == 5,
        "outer_repeats": int(validation.get("outer_repeats", -1)) == 10,
        "outer_seeds": tuple(validation.get("outer_seeds", ()))
        == (42, 142, 242, 342, 442, 542, 642, 742, 842, 942),
        "group_bootstrap_repetitions": int(
            validation.get("group_bootstrap_repetitions", -1)
        )
        == 2000,
        "models_and_complexity_fixed": validation.get(
            "models_and_spline_complexity_are_fixed"
        )
        is True,
        "inner_cv_not_required": validation.get("inner_cv_required") is False,
        "primary_metric": str(validation.get("primary_metric"))
        == "posterior_expected_mae_mV_dec",
        "gate_scope": str(gate.get("applies_to_scope")) == PRIMARY_SCOPE,
        "gate_delta_sign": str(gate.get("primary_delta_sign"))
        == "augmented_minus_reference",
        "gate_primary_estimate": gate.get(
            "require_primary_estimate_below_zero"
        )
        is True,
        "gate_primary_ci": gate.get(
            "require_primary_group_bootstrap_ci95_high_below_zero"
        )
        is True,
        "gate_prior_ci": gate.get(
            "require_augmented_vs_prior_ci95_high_below_zero"
        )
        is True,
        "gate_repeat_fraction": float(
            gate.get("minimum_repeat_improvement_fraction", np.nan)
        )
        == 0.80,
        "gate_draw_fraction": float(
            gate.get(
                "minimum_posterior_draw_improvement_fraction", np.nan
            )
        )
        == 0.90,
        "gate_corroborating_ci": gate.get(
            "require_corroborating_model_ci95_high_below_zero"
        )
        is True,
        "gate_multiplicity": str(gate.get("multiplicity_adjustment"))
        == "holm_one_sided_primary_blocks",
        "gate_alpha": float(gate.get("adjusted_alpha", np.nan)) == 0.05,
        "gam_only_rescue": gate.get("gam_only_rescue_allowed") is False,
        "failed_blocks_emit_effects": gate.get(
            "failed_blocks_emit_effect_curves"
        )
        is False,
        "effect_grid": bool(
            len(effects.get("grid_quantiles", ())) == len(expected_grid)
            and np.allclose(
                np.asarray(effects.get("grid_quantiles", ()), dtype=float),
                np.asarray(expected_grid, dtype=float),
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "effect_reference": float(
            effects.get("reference_quantile", np.nan)
        )
        == 0.50,
        "effect_cross_fitted": effects.get("cross_fitted_g_computation")
        is True,
        "effect_group_bootstrap": effects.get(
            "connected_group_bootstrap"
        )
        is True,
        "effect_no_extrapolation": effects.get("extrapolation_allowed")
        is False,
        "effect_joint_support_distance": str(
            effects.get("joint_support_distance")
        )
        == "fold_local_block_balanced_gower",
        "effect_joint_support_threshold": str(
            effects.get("joint_support_threshold")
        )
        == "training_leave_one_out_nearest_distance_q95",
        "effect_minimum_common_coverage": float(
            effects.get(
                "minimum_common_record_joint_support_coverage", np.nan
            )
        )
        == 0.80,
        "effect_categorical_reference_policy": str(
            effects.get("categorical_adjusted_contrasts")
        )
        == "not_released_without_prespecified_common_reference",
        "contrasts": _configured_contrasts(settings) == FROZEN_CONTRASTS,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise ValueError(
            "Frozen feature-association config disagrees with executable policy: "
            f"{failures}"
        )


def build_posterior_target_summary(
    targets: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    draw_repetitions: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize one posterior per record without expanding ML sample size."""

    target_required = {
        "record_id",
        "sample_record",
        "curve_id",
        "target_name",
        "target_value_mV_dec",
        "target_interval_low_mV_dec",
        "target_interval_high_mV_dec",
    }
    draw_required = {
        "sample_record",
        "posterior_draw_index",
        "target_name",
        "primary_slope_mV_dec",
    }
    missing_targets = sorted(target_required - set(targets.columns))
    missing_draws = sorted(draw_required - set(draws.columns))
    if missing_targets:
        raise KeyError(f"Target registry is missing: {missing_targets}")
    if missing_draws:
        raise KeyError(f"Posterior draws are missing: {missing_draws}")
    selected_targets = targets.loc[
        targets["target_name"].astype(str).eq(TARGET_NAME)
    ].copy()
    selected_draws = draws.loc[
        draws["target_name"].astype(str).eq(TARGET_NAME)
    ].copy()
    if selected_targets.empty or selected_draws.empty:
        raise ValueError("Active posterior target evidence is empty")
    selected_targets["record_id"] = selected_targets["record_id"].map(_stable_id)
    selected_targets["sample_record"] = selected_targets["sample_record"].map(
        _stable_id
    )
    selected_draws["sample_record"] = selected_draws["sample_record"].map(
        _stable_id
    )
    if selected_targets["record_id"].duplicated().any():
        raise ValueError("Target registry must have one active row per record")
    sample_to_record = selected_targets.set_index("sample_record")[
        "record_id"
    ].to_dict()
    selected_draws["record_id"] = selected_draws["sample_record"].map(
        sample_to_record
    )
    if selected_draws["record_id"].isna().any():
        raise ValueError("Posterior draws contain unmapped sample_record values")

    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for record_id, frame in selected_draws.groupby("record_id", sort=True):
        frame = frame.sort_values("posterior_draw_index", kind="stable")
        values = pd.to_numeric(
            frame["primary_slope_mV_dec"], errors="coerce"
        ).to_numpy(float)
        if len(values) != EXPECTED_DRAWS_PER_RECORD:
            raise ValueError(
                f"Record {record_id} has {len(values)} posterior draws; "
                f"expected {EXPECTED_DRAWS_PER_RECORD}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Record {record_id} has nonfinite posterior draws")
        positive = values[values > 0]
        if not len(positive):
            raise ValueError(f"Record {record_id} has no positive slope draws")
        log_positive = np.log10(positive)
        target_row = selected_targets.loc[
            selected_targets["record_id"].eq(record_id)
        ].iloc[0]
        median = float(target_row["target_value_mV_dec"])
        low = float(target_row["target_interval_low_mV_dec"])
        high = float(target_row["target_interval_high_mV_dec"])
        if not (np.isfinite([median, low, high]).all() and 0 < low <= median <= high):
            raise ValueError(f"Record {record_id} has invalid formal target interval")
        summary_rows.append(
            {
                "record_id": record_id,
                "sample_record": _stable_id(target_row["sample_record"]),
                "curve_id": str(target_row["curve_id"]),
                "target_median_mV_dec": median,
                "target_interval_low_mV_dec": low,
                "target_interval_high_mV_dec": high,
                "training_target_log10_median": float(np.log10(median)),
                "positive_log10_draw_mean": float(np.mean(log_positive)),
                "positive_log10_draw_variance": float(
                    np.var(log_positive, ddof=1)
                ),
                "n_posterior_draws": int(len(values)),
                "n_nonpositive_draws": int(np.sum(values <= 0)),
                "nonpositive_draw_fraction": float(np.mean(values <= 0)),
                "positive_draw_fraction": float(np.mean(values > 0)),
            }
        )
        positions = np.linspace(
            0,
            len(frame) - 1,
            num=min(int(draw_repetitions), len(frame)),
            dtype=int,
        )
        selected = frame.iloc[positions]
        for repetition, row in enumerate(selected.itertuples(index=False)):
            value = float(row.primary_slope_mV_dec)
            score_rows.append(
                {
                    "record_id": record_id,
                    "target_draw_repetition": int(repetition),
                    "posterior_draw_index": int(row.posterior_draw_index),
                    "target_draw_mV_dec": value,
                    "target_draw_log10": (
                        float(np.log10(value)) if value > 0 else np.nan
                    ),
                    "positive_for_log_metrics": bool(value > 0),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        "record_id", kind="stable"
    ).reset_index(drop=True)
    score_draws = pd.DataFrame(score_rows).sort_values(
        ["record_id", "target_draw_repetition"], kind="stable"
    ).reset_index(drop=True)
    qa = {
        "target_name": TARGET_NAME,
        "n_records": int(len(summary)),
        "n_posterior_draws": int(summary["n_posterior_draws"].sum()),
        "draws_per_record": EXPECTED_DRAWS_PER_RECORD,
        "posterior_draws_expanded_as_ml_samples": False,
        "training_rows_per_record": 1,
        "training_target": "log10 formal posterior median",
        "score_level_draw_repetitions": int(
            score_draws["target_draw_repetition"].nunique()
        ),
        "n_nonpositive_draws": int(summary["n_nonpositive_draws"].sum()),
        "nonpositive_draw_fraction": float(
            summary["n_nonpositive_draws"].sum()
            / summary["n_posterior_draws"].sum()
        ),
        "minimum_positive_draw_fraction_per_record": float(
            summary["positive_draw_fraction"].min()
        ),
        "nonpositive_draw_policy_original_scale": "retained",
        "nonpositive_draw_policy_log_metrics": "excluded_and_mass_reported",
    }
    return summary, score_draws, qa


def _catalyst_family(values: pd.Series) -> pd.Series:
    source = values.astype("string").fillna("__MISSING__").str.strip()
    return source.map(
        lambda value: (
            "IrO2_plus_Ir"
            if value in {"Ir", "IrO2"}
            else "RuO2"
            if value == "RuO2"
            else value
        )
    )


def _merge_records_and_targets(
    records: pd.DataFrame,
    target_summary: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "record_id",
        "sample_record",
        "cv_group_id",
        "electrolyte_scope",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise KeyError(f"Formal records table is missing: {missing}")
    frame = records.copy()
    frame["record_id"] = frame["record_id"].map(_stable_id)
    frame["sample_record"] = frame["sample_record"].map(_stable_id)
    if frame["record_id"].duplicated().any():
        raise ValueError("Formal records must contain one row per record_id")
    target_columns = [
        column
        for column in target_summary.columns
        if column == "record_id" or column not in frame.columns
    ]
    merged = frame.merge(
        target_summary.loc[:, target_columns],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 97:
        raise ValueError(
            f"Expected 97 records after target binding, found {len(merged)}"
        )
    catalyst_column = (
        "catalyst_active_material_type"
        if "catalyst_active_material_type" in merged
        else "catalyst_type"
    )
    if catalyst_column not in merged:
        raise KeyError("No catalyst identity column is available")
    merged["catalyst_family"] = _catalyst_family(merged[catalyst_column])
    return merged.reset_index(drop=True)


def _scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope != PRIMARY_SCOPE:
        raise ValueError(f"Unknown feature-association scope: {scope!r}")
    selected = frame.copy()
    if selected.empty:
        raise ValueError(f"Feature-association scope {scope!r} is empty")
    selected = selected.reset_index(drop=True)
    selected["record_id"] = selected["record_id"].astype(str)
    selected["cv_group_id"] = selected["cv_group_id"].astype(str)
    return selected


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    frame = pd.DataFrame(
        {
            "category": categories.astype("string").fillna("__MISSING__"),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna(subset=["value"])
    if len(frame) < 3 or frame["category"].nunique() < 2:
        return np.nan
    grand = float(frame["value"].mean())
    denominator = float(np.square(frame["value"] - grand).sum())
    if denominator <= 0:
        return np.nan
    numerator = 0.0
    for _, subset in frame.groupby("category", dropna=False):
        numerator += len(subset) * float(subset["value"].mean() - grand) ** 2
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def build_posterior_association_atlas(
    frame: pd.DataFrame,
    score_draws: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    feature_support: pd.DataFrame,
    *,
    scopes: Sequence[str] = ANALYSIS_SCOPES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one-row-per-record-feature atlas plus draw-aware summaries.

    Posterior draws are used only to propagate response uncertainty through
    descriptive association statistics.  They are never treated as
    independent records and no fitted smoother is used in the atlas.
    """

    required_registry = {"selected_column", "logical_feature", "kind"}
    missing = sorted(required_registry - set(candidate_registry.columns))
    if missing:
        raise KeyError(f"Candidate registry is missing: {missing}")
    draw_table = score_draws.copy()
    draw_table["record_id"] = draw_table["record_id"].astype(str)
    observation_rows: list[dict[str, Any]] = []
    draw_stat_rows: list[dict[str, Any]] = []
    support_lookup = (
        feature_support.set_index(["scope", "feature_name"])
        if not feature_support.empty
        else pd.DataFrame()
    )

    for scope in scopes:
        selected = _scope_frame(frame, str(scope))
        selected_draws = draw_table.loc[
            draw_table["record_id"].isin(selected["record_id"])
        ].copy()
        for candidate in candidate_registry.itertuples(index=False):
            feature = str(candidate.selected_column)
            logical = str(candidate.logical_feature)
            kind = str(candidate.kind).strip().lower()
            if feature not in selected:
                raise KeyError(f"Atlas predictor {feature!r} is missing")
            values = selected[feature]
            is_numeric = kind == "numeric"
            support_status = ""
            atlas_status = ""
            if not isinstance(support_lookup, pd.DataFrame) or not support_lookup.empty:
                try:
                    support_row = support_lookup.loc[(str(scope), feature)]
                    support_status = str(support_row["support_status"])
                    atlas_status = str(support_row["atlas_status"])
                except KeyError:
                    pass
            for position, row in selected.iterrows():
                raw_value = values.iloc[position]
                numeric_value = (
                    float(pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0])
                    if is_numeric
                    and pd.notna(
                        pd.to_numeric(
                            pd.Series([raw_value]), errors="coerce"
                        ).iloc[0]
                    )
                    else np.nan
                )
                category_value = (
                    ""
                    if is_numeric
                    else (
                        "__MISSING__"
                        if pd.isna(raw_value) or str(raw_value).strip() == ""
                        else str(raw_value).strip()
                    )
                )
                observation_rows.append(
                    {
                        "scope": str(scope),
                        "record_id": str(row["record_id"]),
                        "cv_group_id": str(row["cv_group_id"]),
                        "catalyst_family": str(row["catalyst_family"]),
                        "feature_name": feature,
                        "logical_feature": logical,
                        "kind": "numeric" if is_numeric else "categorical",
                        "feature_value_numeric": numeric_value,
                        "feature_value_category": category_value,
                        "feature_value_raw": (
                            "" if pd.isna(raw_value) else str(raw_value)
                        ),
                        "target_median_mV_dec": float(
                            row["target_median_mV_dec"]
                        ),
                        "target_q025_mV_dec": float(
                            row["target_interval_low_mV_dec"]
                        ),
                        "target_q975_mV_dec": float(
                            row["target_interval_high_mV_dec"]
                        ),
                        "support_status": support_status,
                        "atlas_status": atlas_status,
                        "interpretation_boundary": (
                            "marginal descriptive association; not adjusted, "
                            "causal, mechanistic, or feature-ranked"
                        ),
                    }
                )

            feature_frame = selected.loc[
                :,
                ["record_id", "cv_group_id", feature],
            ].merge(
                selected_draws,
                on="record_id",
                how="inner",
                validate="one_to_many",
            )
            for repetition, repetition_frame in feature_frame.groupby(
                "target_draw_repetition", sort=True
            ):
                if is_numeric:
                    x = pd.to_numeric(
                        repetition_frame[feature], errors="coerce"
                    )
                    y = pd.to_numeric(
                        repetition_frame["target_draw_mV_dec"], errors="coerce"
                    )
                    valid = np.isfinite(x) & np.isfinite(y)
                    value = (
                        float(spearmanr(x.loc[valid], y.loc[valid]).statistic)
                        if int(valid.sum()) >= 3
                        and x.loc[valid].nunique() >= 2
                        and y.loc[valid].nunique() >= 2
                        else np.nan
                    )
                    statistic = "spearman_rho"
                else:
                    positive = repetition_frame[
                        "positive_for_log_metrics"
                    ].astype(bool)
                    value = _correlation_ratio(
                        repetition_frame.loc[positive, feature],
                        repetition_frame.loc[positive, "target_draw_log10"],
                    )
                    statistic = "log10_correlation_ratio_eta_squared"
                draw_stat_rows.append(
                    {
                        "scope": str(scope),
                        "feature_name": feature,
                        "logical_feature": logical,
                        "kind": "numeric" if is_numeric else "categorical",
                        "target_draw_repetition": int(repetition),
                        "association_statistic": statistic,
                        "association_value": value,
                        "n_records": int(len(selected)),
                        "n_connected_groups": int(
                            selected["cv_group_id"].nunique()
                        ),
                        "inferential_p_value_reported": False,
                    }
                )

    observations = pd.DataFrame(observation_rows)
    draw_statistics = pd.DataFrame(draw_stat_rows)
    summary_rows: list[dict[str, Any]] = []
    for (scope, feature), subset in draw_statistics.groupby(
        ["scope", "feature_name"], sort=True
    ):
        finite = pd.to_numeric(
            subset["association_value"], errors="coerce"
        ).dropna()
        support = feature_support.loc[
            feature_support["scope"].astype(str).eq(str(scope))
            & feature_support["feature_name"].astype(str).eq(str(feature))
        ]
        support_row = support.iloc[0].to_dict() if len(support) else {}
        summary_rows.append(
            {
                "scope": str(scope),
                "feature_name": str(feature),
                "logical_feature": str(subset["logical_feature"].iloc[0]),
                "kind": str(subset["kind"].iloc[0]),
                "association_statistic": str(
                    subset["association_statistic"].iloc[0]
                ),
                "posterior_association_median": (
                    float(finite.median()) if len(finite) else np.nan
                ),
                "posterior_association_q025": (
                    float(finite.quantile(0.025)) if len(finite) else np.nan
                ),
                "posterior_association_q975": (
                    float(finite.quantile(0.975)) if len(finite) else np.nan
                ),
                "posterior_sign_stability": (
                    float(max((finite > 0).mean(), (finite < 0).mean()))
                    if len(finite)
                    and str(subset["kind"].iloc[0]) == "numeric"
                    else np.nan
                ),
                "n_target_draw_repetitions_estimable": int(len(finite)),
                "n_records": int(subset["n_records"].iloc[0]),
                "n_connected_groups": int(
                    subset["n_connected_groups"].iloc[0]
                ),
                "support_status": str(
                    support_row.get("support_status", "")
                ),
                "atlas_status": str(support_row.get("atlas_status", "")),
                "x_label": str(feature).replace("_", " "),
                "x_scale": "linear",
                "fitted_smoother_used": False,
                "interpretation_boundary": (
                    "posterior-aware marginal description only"
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    qa = {
        "status": "pass",
        "n_scopes": int(observations["scope"].nunique()),
        "n_features": int(observations["feature_name"].nunique()),
        "n_scope_record_feature_rows": int(len(observations)),
        "n_draw_level_association_rows": int(len(draw_statistics)),
        "posterior_draws_used_as_independent_ml_samples": False,
        "fitted_smoother_used": False,
        "target_interval_displayed": "formal posterior 95% interval",
        "interpretation": (
            "marginal descriptive association; not adjusted, causal, "
            "mechanistic, or a feature-selection ranking"
        ),
    }
    return observations, summary, draw_statistics, qa


def _fit_intercept_prior(
    y_train: np.ndarray,
    n_test: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    response = np.asarray(y_train, dtype=float)
    if response.ndim != 1 or len(response) < 2 or not np.isfinite(response).all():
        raise ValueError("Intercept prior requires at least two finite responses")
    mean = float(np.mean(response))
    residual_sd = float(np.std(response, ddof=1))
    predictive_sd = max(
        residual_sd * float(np.sqrt(1.0 + 1.0 / len(response))),
        1.0e-6,
    )
    definition = {
        "schema_version": "feature-association-model-v1",
        "model_family": PRIOR_MODEL,
        "scientific_name": "training-fold Gaussian intercept prior",
        "training_response": "log10 formal posterior median",
        "training_records_equally_weighted": True,
        "n_training_records": int(len(response)),
        "predictive_mean_log10": mean,
        "predictive_sd_log10": predictive_sd,
        "design_column_count": 1,
        "empirical_edf": 1.0,
        "full_bayesian_posterior": False,
    }
    return (
        np.full(int(n_test), mean, dtype=float),
        np.full(int(n_test), predictive_sd, dtype=float),
        definition,
    )


def _block_balanced_gower_distances(
    query: pd.DataFrame,
    train: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """Training-scaled mixed-type distances, averaged within then across blocks."""

    if tuple(query.columns) != tuple(train.columns):
        query = query.loc[:, list(train.columns)]
    numerator = np.zeros((len(query), len(train)), dtype=float)
    denominator = np.zeros_like(numerator)
    for members in blocks.values():
        usable = [name for name in members if name in train.columns]
        block_numerator = np.zeros_like(numerator)
        block_denominator = np.zeros_like(numerator)
        for feature in usable:
            if pd.api.types.is_numeric_dtype(train[feature]):
                training = pd.to_numeric(
                    train[feature], errors="coerce"
                ).to_numpy(float)
                finite = training[np.isfinite(training)]
                scale = float(np.ptp(finite)) if len(finite) else np.nan
                if not np.isfinite(scale) or scale <= 0:
                    continue
                candidate = pd.to_numeric(
                    query[feature], errors="coerce"
                ).to_numpy(float)
                valid = np.isfinite(candidate[:, None]) & np.isfinite(
                    training[None, :]
                )
                contribution = np.minimum(
                    np.abs(candidate[:, None] - training[None, :]) / scale,
                    1.0,
                )
            else:
                candidate = query[feature].astype("object").to_numpy()
                training = train[feature].astype("object").to_numpy()
                valid = (~pd.isna(candidate[:, None])) & (
                    ~pd.isna(training[None, :])
                )
                contribution = (
                    candidate[:, None] != training[None, :]
                ).astype(float)
            block_numerator += np.where(valid, contribution, 0.0)
            block_denominator += valid.astype(float)
        valid_block = block_denominator > 0
        block_distance = np.divide(
            block_numerator,
            block_denominator,
            out=np.zeros_like(block_numerator),
            where=valid_block,
        )
        numerator += np.where(valid_block, block_distance, 0.0)
        denominator += valid_block.astype(float)
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0,
    )


def _joint_support_state(
    train_engineered: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
    *,
    threshold_quantile: float,
) -> dict[str, Any]:
    distances = _block_balanced_gower_distances(
        train_engineered, train_engineered, blocks
    )
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if not len(finite):
        raise ValueError("Joint-support calibration has no finite LOO distances")
    return {
        "train": train_engineered,
        "blocks": dict(blocks),
        "threshold": float(np.quantile(finite, threshold_quantile)),
        "threshold_quantile": float(threshold_quantile),
        "training_loo_nearest_median": float(np.median(finite)),
        "training_loo_nearest_max": float(np.max(finite)),
    }


def _joint_support_query(
    query: pd.DataFrame,
    state: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    distances = _block_balanced_gower_distances(
        query,
        state["train"],
        state["blocks"],
    )
    nearest = np.min(distances, axis=1)
    supported = nearest <= float(state["threshold"]) + 1.0e-12
    return supported.astype(bool), nearest.astype(float)


def _effect_candidate_rows(
    *,
    scope: str,
    repeat: int,
    fold: int,
    panel_name: str,
    builder: AssociationFeatureBuilder,
    model: EmpiricalBayesAssociationModel,
    train_engineered: pd.DataFrame,
    test_engineered: pd.DataFrame,
    test_metadata: pd.DataFrame,
    config: FeatureAssociationEvaluationConfig,
) -> list[dict[str, Any]]:
    panel = FROZEN_ASSOCIATION_PANELS[panel_name]
    if (
        scope != PRIMARY_SCOPE
        or model.model_family != "gam_low_df"
        or not panel.effect_eligible
    ):
        return []
    added_features = tuple(
        feature
        for feature in panel.added_features
        if feature in set(builder.get_feature_names_out().astype(str))
    )
    support_state = _joint_support_state(
        train_engineered,
        builder.get_blocks_out(),
        threshold_quantile=config.effect_joint_support_threshold_quantile,
    )
    rows: list[dict[str, Any]] = []
    for feature in added_features:
        reference_frame = test_engineered.copy()
        if feature in builder.numeric_columns_out_:
            training_values = pd.to_numeric(
                train_engineered[feature], errors="coerce"
            )
            # A continuous curve needs enough training support for the fixed
            # low-df spline. Two- or three-level variables remain in the model
            # but are described only in the marginal atlas.
            if training_values.nunique(dropna=True) < 4:
                continue
            grid = [
                (
                    float(quantile),
                    float(training_values.quantile(float(quantile))),
                    "",
                )
                for quantile in config.effect_grid_quantiles
            ]
            reference_value = float(
                training_values.quantile(config.effect_reference_quantile)
            )
        else:
            # No cross-fold categorical reference was preregistered. Releasing
            # fold-mode contrasts could silently mix different comparisons.
            continue
        reference_frame[feature] = reference_value
        reference_prediction = np.asarray(
            model.predict(reference_frame), dtype=float
        )
        reference_supported, reference_distance = _joint_support_query(
            reference_frame, support_state
        )
        for grid_quantile, grid_numeric, grid_category in grid:
            intervention = test_engineered.copy()
            intervention[feature] = (
                grid_numeric
                if np.isfinite(grid_numeric)
                else grid_category
            )
            prediction = np.asarray(model.predict(intervention), dtype=float)
            delta = prediction - reference_prediction
            intervention_supported, intervention_distance = (
                _joint_support_query(intervention, support_state)
            )
            joint_supported = reference_supported & intervention_supported
            for position, metadata in enumerate(
                test_metadata.itertuples(index=False)
            ):
                rows.append(
                    {
                        "scope": scope,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "record_id": str(metadata.record_id),
                        "cv_group_id": str(metadata.cv_group_id),
                        "catalyst_family": str(metadata.catalyst_family),
                        "panel": panel_name,
                        "block_id": panel.added_block,
                        "model": model.model_family,
                        "feature_name": feature,
                        "grid_quantile": grid_quantile,
                        "grid_value_numeric": grid_numeric,
                        "grid_value_category": grid_category,
                        "reference_quantile": (
                            config.effect_reference_quantile
                            if np.isfinite(grid_quantile)
                            else np.nan
                        ),
                        "reference_value_numeric": (
                            float(reference_value)
                            if isinstance(reference_value, (float, np.floating))
                            else np.nan
                        ),
                        "reference_value_category": (
                            str(reference_value)
                            if isinstance(reference_value, str)
                            else ""
                        ),
                        "predicted_log10_centered": float(delta[position]),
                        "joint_support": bool(joint_supported[position]),
                        "joint_support_nearest_distance": float(
                            max(
                                reference_distance[position],
                                intervention_distance[position],
                            )
                        ),
                        "joint_support_threshold": float(
                            support_state["threshold"]
                        ),
                        "joint_support_threshold_quantile": float(
                            support_state["threshold_quantile"]
                        ),
                        "cross_fitted": True,
                        "training_fold_grid": True,
                    }
                )
    return rows


def _evaluate_outer_fold(
    frame: pd.DataFrame,
    *,
    scope: str,
    repeat: int,
    fold: int,
    test_positions: np.ndarray,
    config: FeatureAssociationEvaluationConfig,
) -> dict[str, list[dict[str, Any]]]:
    all_positions = np.arange(len(frame), dtype=int)
    train_positions = np.setdiff1d(
        all_positions, np.asarray(test_positions, dtype=int), assume_unique=True
    )
    train = frame.iloc[train_positions].reset_index(drop=True)
    test = frame.iloc[test_positions].reset_index(drop=True)
    train_groups = set(train["cv_group_id"].astype(str))
    test_groups = set(test["cv_group_id"].astype(str))
    if train_groups.intersection(test_groups):
        raise AssertionError("Connected groups crossed an outer fold")
    y_train = pd.to_numeric(
        train["training_target_log10_median"], errors="coerce"
    ).to_numpy(float)
    prediction_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    metadata_columns = ["record_id", "cv_group_id", "catalyst_family"]
    test_metadata = test.loc[:, metadata_columns].reset_index(drop=True)

    prior_mean, prior_std, prior_definition = _fit_intercept_prior(
        y_train, len(test)
    )
    for model_name in config.model_names:
        for position, metadata in enumerate(
            test_metadata.itertuples(index=False)
        ):
            prediction_rows.append(
                {
                    "scope": scope,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "record_id": str(metadata.record_id),
                    "cv_group_id": str(metadata.cv_group_id),
                    "catalyst_family": str(metadata.catalyst_family),
                    "panel": PRIOR_PANEL,
                    "model": model_name,
                    "prediction_log10_mean": float(prior_mean[position]),
                    "prediction_log10_std": float(prior_std[position]),
                    "prediction_median_mV_dec": float(
                        10.0 ** prior_mean[position]
                    ),
                    "target_median_mV_dec": float(
                        test.loc[position, "target_median_mV_dec"]
                    ),
                }
            )
        model_rows.append(
            {
                "scope": scope,
                "repeat": int(repeat),
                "fold": int(fold),
                "panel": PRIOR_PANEL,
                "model": model_name,
                "n_train_records": int(len(train)),
                "n_test_records": int(len(test)),
                "n_train_groups": int(len(train_groups)),
                "n_test_groups": int(len(test_groups)),
                "engineered_feature_count": 0,
                "design_column_count": 1,
                "empirical_edf": 1.0,
                "feature_names": "",
                "model_definition_json": json.dumps(
                    prior_definition, sort_keys=True
                ),
            }
        )

    for panel_name in config.panel_names:
        builder = AssociationFeatureBuilder(
            panel_name,
            rare_min_frequency=config.rare_min_frequency,
        )
        train_engineered = builder.fit_transform(train)
        test_engineered = builder.transform(test)
        for model_name in config.model_names:
            model = EmpiricalBayesAssociationModel(
                model_name,
                gam_n_knots=config.gam_n_knots,
                gam_degree=config.gam_degree,
                rare_min_frequency=config.rare_min_frequency,
                max_iter=config.model_max_iter,
                tol=config.model_tolerance,
            )
            model.fit(train_engineered, y_train)
            prediction_mean, prediction_std = model.predict(
                test_engineered, return_std=True
            )
            prediction_std = np.maximum(
                np.asarray(prediction_std, dtype=float), 1.0e-6
            )
            definition = model.model_definition
            for position, metadata in enumerate(
                test_metadata.itertuples(index=False)
            ):
                prediction_rows.append(
                    {
                        "scope": scope,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "record_id": str(metadata.record_id),
                        "cv_group_id": str(metadata.cv_group_id),
                        "catalyst_family": str(metadata.catalyst_family),
                        "panel": panel_name,
                        "model": model_name,
                        "prediction_log10_mean": float(
                            prediction_mean[position]
                        ),
                        "prediction_log10_std": float(
                            prediction_std[position]
                        ),
                        "prediction_median_mV_dec": float(
                            10.0 ** prediction_mean[position]
                        ),
                        "target_median_mV_dec": float(
                            test.loc[position, "target_median_mV_dec"]
                        ),
                    }
                )
            model_rows.append(
                {
                    "scope": scope,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "panel": panel_name,
                    "model": model_name,
                    "n_train_records": int(len(train)),
                    "n_test_records": int(len(test)),
                    "n_train_groups": int(len(train_groups)),
                    "n_test_groups": int(len(test_groups)),
                    "engineered_feature_count": int(
                        len(builder.get_feature_names_out())
                    ),
                    "design_column_count": int(
                        definition["design_column_count"]
                    ),
                    "empirical_edf": float(definition["empirical_edf"]),
                    "feature_names": "|".join(
                        builder.get_feature_names_out().astype(str)
                    ),
                    "model_definition_json": json.dumps(
                        definition, sort_keys=True
                    ),
                }
            )
            effect_rows.extend(
                _effect_candidate_rows(
                    scope=scope,
                    repeat=repeat,
                    fold=fold,
                    panel_name=panel_name,
                    builder=builder,
                    model=model,
                    train_engineered=train_engineered,
                    test_engineered=test_engineered,
                    test_metadata=test_metadata,
                    config=config,
                )
            )
    return {
        "predictions": prediction_rows,
        "model_fits": model_rows,
        "effect_candidates": effect_rows,
    }


def run_grouped_outer_predictions(
    frame: pd.DataFrame,
    *,
    scope: str,
    config: FeatureAssociationEvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = _scope_frame(frame, scope)
    manifest = make_outer_split_manifest(
        selected,
        folds=config.outer_folds,
        repeats=config.outer_repeats,
        repeat_seeds=config.outer_seeds,
    )
    validate_split_manifest(
        selected,
        manifest,
        expected_repeats=config.outer_repeats,
        expected_folds=config.outer_folds,
    )
    tasks: list[tuple[int, int, np.ndarray]] = []
    for (repeat, fold), subset in manifest.groupby(
        ["repeat", "fold"], sort=True
    ):
        tasks.append(
            (
                int(repeat),
                int(fold),
                subset["row_position"].to_numpy(int),
            )
        )
    # The fits are small and NumPy/scikit-learn release the GIL for their
    # numerical kernels.  Threads avoid platform-specific process/semaphore
    # failures while preserving the sklearn configuration in every worker.
    outputs = Parallel(n_jobs=config.n_jobs, backend="threading")(
        delayed(_evaluate_outer_fold)(
            selected,
            scope=scope,
            repeat=repeat,
            fold=fold,
            test_positions=test_positions,
            config=config,
        )
        for repeat, fold, test_positions in tasks
    )
    predictions = pd.DataFrame(
        [
            row
            for output in outputs
            for row in output["predictions"]
        ]
    )
    model_fits = pd.DataFrame(
        [
            row
            for output in outputs
            for row in output["model_fits"]
        ]
    )
    effect_candidates = pd.DataFrame(
        [
            row
            for output in outputs
            for row in output["effect_candidates"]
        ]
    )
    expected_predictions = (
        len(selected)
        * config.outer_repeats
        * (len(config.panel_names) + 1)
        * len(config.model_names)
    )
    if len(predictions) != expected_predictions:
        raise AssertionError(
            "OOF prediction coverage is incomplete: "
            f"{len(predictions)} != {expected_predictions}"
        )
    key = ["repeat", "record_id", "panel", "model"]
    if predictions.duplicated(key).any():
        raise AssertionError("OOF predictions contain duplicate record cells")
    tagged_manifest = manifest.copy()
    tagged_manifest.insert(0, "scope", scope)
    return predictions, model_fits, effect_candidates, tagged_manifest


def _posterior_draw_lookup(
    score_draws: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for record_id, subset in score_draws.groupby("record_id", sort=False):
        ordered = subset.sort_values("target_draw_repetition").reset_index(
            drop=True
        )
        lookup[str(record_id)] = ordered
    return lookup


def _score_prediction_rows(
    predictions: pd.DataFrame,
    score_draws: pd.DataFrame,
) -> pd.DataFrame:
    lookup = _posterior_draw_lookup(score_draws)
    rows: list[dict[str, Any]] = []
    for prediction in predictions.itertuples(index=False):
        draws = lookup[str(prediction.record_id)]
        target_mV = pd.to_numeric(
            draws["target_draw_mV_dec"], errors="coerce"
        ).to_numpy(float)
        target_log = pd.to_numeric(
            draws["target_draw_log10"], errors="coerce"
        ).to_numpy(float)
        positive = np.isfinite(target_log)
        predicted_mV = float(prediction.prediction_median_mV_dec)
        predicted_log = float(prediction.prediction_log10_mean)
        predicted_std = max(float(prediction.prediction_log10_std), 1.0e-6)
        log_errors = (
            np.abs(target_log[positive] - predicted_log)
            if positive.any()
            else np.asarray([], dtype=float)
        )
        log_nll = (
            -norm.logpdf(
                target_log[positive],
                loc=predicted_log,
                scale=predicted_std,
            )
            if positive.any()
            else np.asarray([], dtype=float)
        )
        rows.append(
            {
                **prediction._asdict(),
                "posterior_expected_mae_mV_dec": float(
                    np.mean(np.abs(target_mV - predicted_mV))
                ),
                "posterior_expected_mae_log10": (
                    float(np.mean(log_errors)) if len(log_errors) else np.nan
                ),
                "posterior_expected_nll_log10": (
                    float(np.mean(log_nll)) if len(log_nll) else np.nan
                ),
                "median_target_absolute_error_mV_dec": float(
                    abs(
                        float(prediction.target_median_mV_dec)
                        - predicted_mV
                    )
                ),
                "median_target_squared_error_mV_dec2": float(
                    (
                        float(prediction.target_median_mV_dec)
                        - predicted_mV
                    )
                    ** 2
                ),
                "n_target_draws_original_scale": int(len(target_mV)),
                "n_target_draws_log_scale": int(positive.sum()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_repeated_oof(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "scope",
        "record_id",
        "cv_group_id",
        "catalyst_family",
        "panel",
        "model",
    ]
    rows: list[dict[str, Any]] = []
    for key, subset in predictions.groupby(keys, sort=True, dropna=False):
        means = pd.to_numeric(
            subset["prediction_log10_mean"], errors="coerce"
        ).to_numpy(float)
        stds = pd.to_numeric(
            subset["prediction_log10_std"], errors="coerce"
        ).to_numpy(float)
        if not (np.isfinite(means).all() and np.isfinite(stds).all()):
            raise ValueError("OOF predictions contain nonfinite values")
        mixture_mean = float(np.mean(means))
        mixture_variance = float(
            max(np.mean(np.square(stds) + np.square(means)) - mixture_mean**2, 0)
        )
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "prediction_log10_mean": mixture_mean,
                "prediction_log10_std": max(
                    float(np.sqrt(mixture_variance)), 1.0e-6
                ),
                "prediction_median_mV_dec": float(10.0**mixture_mean),
                "target_median_mV_dec": float(
                    subset["target_median_mV_dec"].iloc[0]
                ),
                "n_oof_repeats": int(len(subset)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_summary(
    scored: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, subset in scored.groupby(
        list(group_columns), sort=True, dropna=False
    ):
        key_tuple = key if isinstance(key, tuple) else (key,)
        actual = pd.to_numeric(
            subset["target_median_mV_dec"], errors="coerce"
        ).to_numpy(float)
        predicted = pd.to_numeric(
            subset["prediction_median_mV_dec"], errors="coerce"
        ).to_numpy(float)
        spearman = (
            float(spearmanr(actual, predicted).statistic)
            if len(actual) >= 3
            and np.unique(actual).size >= 2
            and np.unique(predicted).size >= 2
            else np.nan
        )
        denominator = float(np.square(actual - np.mean(actual)).sum())
        r2 = (
            float(1.0 - np.square(actual - predicted).sum() / denominator)
            if denominator > 0
            else np.nan
        )
        row = dict(zip(group_columns, key_tuple, strict=True))
        row.update(
            {
                "posterior_expected_mae_mV_dec": float(
                    subset["posterior_expected_mae_mV_dec"].mean()
                ),
                "posterior_expected_mae_log10": float(
                    subset["posterior_expected_mae_log10"].mean()
                ),
                "posterior_expected_nll_log10": float(
                    subset["posterior_expected_nll_log10"].mean()
                ),
                "median_target_mae_mV_dec": float(
                    subset["median_target_absolute_error_mV_dec"].mean()
                ),
                "rmse_mV_dec": float(
                    np.sqrt(
                        subset[
                            "median_target_squared_error_mV_dec2"
                        ].mean()
                    )
                ),
                "spearman": spearman,
                "r2": r2,
                "n_records": int(len(subset)),
                "n_connected_groups": int(
                    subset["cv_group_id"].astype(str).nunique()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _target_draw_metrics(
    aggregated_predictions: pd.DataFrame,
    score_draws: pd.DataFrame,
) -> pd.DataFrame:
    lookup = _posterior_draw_lookup(score_draws)
    rows: list[dict[str, Any]] = []
    for (scope, panel, model), subset in aggregated_predictions.groupby(
        ["scope", "panel", "model"], sort=True
    ):
        repetitions = sorted(
            {
                int(value)
                for record_id in subset["record_id"].astype(str)
                for value in lookup[record_id][
                    "target_draw_repetition"
                ].tolist()
            }
        )
        for repetition in repetitions:
            original_errors: list[float] = []
            log_errors: list[float] = []
            log_nll: list[float] = []
            n_nonpositive = 0
            for prediction in subset.itertuples(index=False):
                draw = lookup[str(prediction.record_id)].loc[
                    lambda value: value["target_draw_repetition"].eq(
                        repetition
                    )
                ]
                if len(draw) != 1:
                    raise AssertionError(
                        "Each record must have one selected posterior draw "
                        "per target-draw repetition"
                    )
                target_mV = float(draw["target_draw_mV_dec"].iloc[0])
                original_errors.append(
                    abs(target_mV - float(prediction.prediction_median_mV_dec))
                )
                target_log = float(draw["target_draw_log10"].iloc[0])
                if np.isfinite(target_log):
                    mean = float(prediction.prediction_log10_mean)
                    std = max(
                        float(prediction.prediction_log10_std), 1.0e-6
                    )
                    log_errors.append(abs(target_log - mean))
                    log_nll.append(
                        float(-norm.logpdf(target_log, loc=mean, scale=std))
                    )
                else:
                    n_nonpositive += 1
            rows.append(
                {
                    "scope": scope,
                    "panel": panel,
                    "model": model,
                    "target_draw_repetition": int(repetition),
                    "posterior_draw_mae_mV_dec": float(
                        np.mean(original_errors)
                    ),
                    "posterior_draw_mae_log10": (
                        float(np.mean(log_errors)) if log_errors else np.nan
                    ),
                    "posterior_draw_nll_log10": (
                        float(np.mean(log_nll)) if log_nll else np.nan
                    ),
                    "n_records": int(len(subset)),
                    "n_nonpositive_draws_excluded_from_log_metrics": int(
                        n_nonpositive
                    ),
                }
            )
    return pd.DataFrame(rows)


def score_grouped_predictions(
    predictions: pd.DataFrame,
    score_draws: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    scored_repeated = _score_prediction_rows(predictions, score_draws)
    aggregated_predictions = _aggregate_repeated_oof(predictions)
    scored_aggregated = _score_prediction_rows(
        aggregated_predictions, score_draws
    )
    split_metrics = _metric_summary(
        scored_repeated,
        group_columns=("scope", "repeat", "panel", "model"),
    )
    aggregate_metrics = _metric_summary(
        scored_aggregated,
        group_columns=("scope", "panel", "model"),
    )
    draw_metrics = _target_draw_metrics(
        aggregated_predictions, score_draws
    )
    return {
        "scored_repeated": scored_repeated,
        "aggregated_predictions": aggregated_predictions,
        "scored_aggregated": scored_aggregated,
        "split_metrics": split_metrics,
        "aggregate_metrics": aggregate_metrics,
        "target_draw_metrics": draw_metrics,
    }


def _group_bootstrap_delta(
    record_deltas: pd.DataFrame,
    *,
    replicates: int,
    seed_key: str,
) -> tuple[float, np.ndarray]:
    required = {"cv_group_id", "delta"}
    missing = sorted(required - set(record_deltas.columns))
    if missing:
        raise KeyError(f"Paired bootstrap input is missing: {missing}")
    frame = record_deltas.loc[
        np.isfinite(pd.to_numeric(record_deltas["delta"], errors="coerce"))
    ].copy()
    frame["delta"] = pd.to_numeric(frame["delta"], errors="coerce")
    if frame.empty:
        raise ValueError("Paired bootstrap has no finite record deltas")
    groups = sorted(frame["cv_group_id"].astype(str).unique())
    if len(groups) < 2:
        raise ValueError("Paired bootstrap requires at least two groups")
    group_values = {
        group: frame.loc[
            frame["cv_group_id"].astype(str).eq(group), "delta"
        ].to_numpy(float)
        for group in groups
    }
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(int(replicates), dtype=float)
    group_array = np.asarray(groups, dtype=object)
    for repetition in range(int(replicates)):
        sampled = rng.choice(group_array, size=len(group_array), replace=True)
        values = np.concatenate([group_values[str(group)] for group in sampled])
        bootstrap[repetition] = float(np.mean(values))
    return float(frame["delta"].mean()), bootstrap


def _cluster_signflip_p_value(
    record_deltas: pd.DataFrame,
    *,
    replicates: int,
    seed_key: str,
) -> float:
    """One-sided cluster-level Rademacher randomization p-value.

    The null distribution is explicitly centered at zero. Each connected
    group's complete record contribution receives one shared random sign, so
    within-group dependence and unequal group sizes are preserved.
    """

    frame = record_deltas.copy()
    frame["delta"] = pd.to_numeric(frame["delta"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["delta"])].copy()
    if frame.empty:
        raise ValueError("Cluster sign-flip test has no finite deltas")
    group_sums = (
        frame.groupby(frame["cv_group_id"].astype(str))["delta"]
        .sum()
        .to_numpy(float)
    )
    observed = float(frame["delta"].mean())
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    null = np.empty(int(replicates), dtype=float)
    for repetition in range(int(replicates)):
        signs = rng.choice(
            np.asarray([-1.0, 1.0]), size=len(group_sums), replace=True
        )
        null[repetition] = float(
            np.sum(signs * group_sums) / len(frame)
        )
    # Alternative: augmented-minus-reference mean is below zero.
    return float((1 + np.sum(null <= observed)) / (len(null) + 1))


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    finite = sorted(
        (
            (str(name), float(value))
            for name, value in p_values.items()
            if np.isfinite(value)
        ),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: dict[str, float] = {
        str(name): np.nan for name in p_values
    }
    running = 0.0
    total = len(finite)
    for rank, (name, value) in enumerate(finite):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = min(1.0, running)
    return adjusted


def _comparison_record_deltas(
    scored_aggregated: pd.DataFrame,
    *,
    scope: str,
    model: str,
    reference: str,
    augmented: str,
) -> pd.DataFrame:
    selected = scored_aggregated.loc[
        scored_aggregated["scope"].astype(str).eq(scope)
        & scored_aggregated["model"].astype(str).eq(model)
        & scored_aggregated["panel"].astype(str).isin(
            [reference, augmented]
        )
    ].copy()
    pivot = selected.pivot(
        index=["record_id", "cv_group_id"],
        columns="panel",
        values="posterior_expected_mae_mV_dec",
    )
    if reference not in pivot or augmented not in pivot:
        raise ValueError(
            f"Missing paired panels for {augmented} minus {reference}"
        )
    pivot = pivot.dropna(subset=[reference, augmented]).reset_index()
    pivot["delta"] = pivot[augmented] - pivot[reference]
    return pivot


def _improvement_fraction(
    metrics: pd.DataFrame,
    *,
    scope: str,
    model: str,
    reference: str,
    augmented: str,
    repetition_column: str,
    metric_column: str,
) -> float:
    selected = metrics.loc[
        metrics["scope"].astype(str).eq(scope)
        & metrics["model"].astype(str).eq(model)
        & metrics["panel"].astype(str).isin([reference, augmented])
    ]
    pivot = selected.pivot(
        index=repetition_column,
        columns="panel",
        values=metric_column,
    )
    if reference not in pivot or augmented not in pivot:
        return np.nan
    complete = pivot.dropna(subset=[reference, augmented])
    return (
        float((complete[augmented] < complete[reference]).mean())
        if len(complete)
        else np.nan
    )


def build_blockwise_comparison(
    scored_aggregated: pd.DataFrame,
    split_metrics: pd.DataFrame,
    target_draw_metrics: pd.DataFrame,
    *,
    config: FeatureAssociationEvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bootstrap_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scope in ANALYSIS_SCOPES:
        for (
            contrast_id,
            block,
            reference,
            augmented,
            role,
        ) in FROZEN_CONTRASTS:
            comparisons = (
                ("registered_contrast", reference),
                ("augmented_vs_prior", PRIOR_PANEL),
            )
            for model in FROZEN_MODELS:
                for comparison_kind, comparison_reference in comparisons:
                    deltas = _comparison_record_deltas(
                        scored_aggregated,
                        scope=scope,
                        model=model,
                        reference=comparison_reference,
                        augmented=augmented,
                    )
                    estimate, bootstrap = _group_bootstrap_delta(
                        deltas,
                        replicates=config.bootstrap_replicates,
                        seed_key=(
                            f"{scope}|{model}|{augmented}|"
                            f"{comparison_reference}|20260724"
                        ),
                    )
                    low, high = np.quantile(bootstrap, [0.025, 0.975])
                    p_one_sided = _cluster_signflip_p_value(
                        deltas,
                        replicates=config.bootstrap_replicates,
                        seed_key=(
                            f"signflip|{scope}|{model}|{augmented}|"
                            f"{comparison_reference}|20260724"
                        ),
                    )
                    for repetition, value in enumerate(bootstrap):
                        bootstrap_rows.append(
                            {
                                "scope": scope,
                                "contrast_id": contrast_id,
                                "block_id": block,
                                "role": role,
                                "reference_panel": comparison_reference,
                                "augmented_panel": augmented,
                                "model": model,
                                "comparison_kind": comparison_kind,
                                "metric": (
                                    "posterior_expected_mae_mV_dec"
                                ),
                                "bootstrap_repetition": int(repetition),
                                "delta": float(value),
                            }
                        )
                    repeat_fraction = (
                        _improvement_fraction(
                            split_metrics,
                            scope=scope,
                            model=model,
                            reference=reference,
                            augmented=augmented,
                            repetition_column="repeat",
                            metric_column=(
                                "posterior_expected_mae_mV_dec"
                            ),
                        )
                        if comparison_kind == "registered_contrast"
                        else np.nan
                    )
                    draw_fraction = (
                        _improvement_fraction(
                            target_draw_metrics,
                            scope=scope,
                            model=model,
                            reference=reference,
                            augmented=augmented,
                            repetition_column="target_draw_repetition",
                            metric_column="posterior_draw_mae_mV_dec",
                        )
                        if comparison_kind == "registered_contrast"
                        else np.nan
                    )
                    summary_rows.append(
                        {
                            "scope": scope,
                            "contrast_id": contrast_id,
                            "block_id": block,
                            "role": role,
                            "reference_panel": comparison_reference,
                            "augmented_panel": augmented,
                            "model": model,
                            "comparison_kind": comparison_kind,
                            "metric": (
                                "posterior_expected_mae_mV_dec"
                            ),
                            "metric_role": "primary",
                            "is_primary_metric": True,
                            "delta_sign": "augmented_minus_reference",
                            "observed_delta": estimate,
                            "ci95_low": float(low),
                            "ci95_high": float(high),
                            "one_sided_p_unadjusted": p_one_sided,
                            "one_sided_p_method": (
                                "connected_group_rademacher_sign_flip_null"
                            ),
                            "repeat_improvement_fraction": repeat_fraction,
                            "posterior_draw_improvement_fraction": draw_fraction,
                            "n_records": int(len(deltas)),
                            "n_connected_groups": int(
                                deltas["cv_group_id"].astype(str).nunique()
                            ),
                            "bootstrap_replicates": int(len(bootstrap)),
                        }
                    )
    summary = pd.DataFrame(summary_rows)
    primary_mask = (
        summary["scope"].astype(str).eq(PRIMARY_SCOPE)
        & summary["model"].astype(str).eq("bayesian_linear")
        & summary["comparison_kind"].astype(str).eq(
            "registered_contrast"
        )
        & summary["contrast_id"].astype(str).isin(
            PRIMARY_MULTIPLICITY_CONTRASTS
        )
    )
    adjusted = _holm_adjust(
        summary.loc[primary_mask].set_index("contrast_id")[
            "one_sided_p_unadjusted"
        ].to_dict()
    )
    summary["one_sided_p_holm"] = np.nan
    for contrast_id, value in adjusted.items():
        summary.loc[
            primary_mask
            & summary["contrast_id"].astype(str).eq(contrast_id),
            "one_sided_p_holm",
        ] = value

    decision_rows: list[dict[str, Any]] = []
    if config.smoke:
        gate_status = "smoke_not_scientific"
    else:
        gate_status = "evaluated"
    for (
        contrast_id,
        block,
        reference,
        augmented,
        role,
    ) in FROZEN_CONTRASTS:
        primary = summary.loc[
            summary["scope"].astype(str).eq(PRIMARY_SCOPE)
            & summary["contrast_id"].astype(str).eq(contrast_id)
            & summary["model"].astype(str).eq("bayesian_linear")
            & summary["comparison_kind"].astype(str).eq(
                "registered_contrast"
            )
        ].iloc[0]
        corroborating = summary.loc[
            summary["scope"].astype(str).eq(PRIMARY_SCOPE)
            & summary["contrast_id"].astype(str).eq(contrast_id)
            & summary["model"].astype(str).eq("gam_low_df")
            & summary["comparison_kind"].astype(str).eq(
                "registered_contrast"
            )
        ].iloc[0]
        prior = summary.loc[
            summary["scope"].astype(str).eq(PRIMARY_SCOPE)
            & summary["contrast_id"].astype(str).eq(contrast_id)
            & summary["model"].astype(str).eq("bayesian_linear")
            & summary["comparison_kind"].astype(str).eq(
                "augmented_vs_prior"
            )
        ].iloc[0]
        in_holm_family = contrast_id in PRIMARY_MULTIPLICITY_CONTRASTS
        checks = {
            "primary_estimate_below_zero": float(
                primary["observed_delta"]
            )
            < 0,
            "primary_ci95_high_below_zero": float(primary["ci95_high"])
            < 0,
            "augmented_vs_prior_ci95_high_below_zero": float(
                prior["ci95_high"]
            )
            < 0,
            "minimum_repeat_improvement_fraction": float(
                primary["repeat_improvement_fraction"]
            )
            >= config.minimum_repeat_improvement_fraction,
            "minimum_posterior_draw_improvement_fraction": float(
                primary["posterior_draw_improvement_fraction"]
            )
            >= config.minimum_draw_improvement_fraction,
            "corroborating_gam_ci95_high_below_zero": float(
                corroborating["ci95_high"]
            )
            < 0,
            "holm_one_sided_primary_blocks": (
                float(primary["one_sided_p_holm"])
                < config.adjusted_alpha
                if in_holm_family
                else True
            ),
        }
        passed = bool(
            not config.smoke
            and all(bool(value) for value in checks.values())
        )
        failed = [name for name, value in checks.items() if not value]
        panel_effect_eligible = bool(
            FROZEN_ASSOCIATION_PANELS[augmented].effect_eligible
        )
        release = bool(passed and panel_effect_eligible)
        reason = (
            "smoke mode never validates or releases an adjusted effect curve"
            if config.smoke
            else (
                "all registered validation checks passed"
                if passed
                else "failed: " + "|".join(failed)
            )
        )
        if passed and not panel_effect_eligible:
            reason += "; frozen panel is not effect-eligible"
        decision_rows.append(
            {
                "scope": PRIMARY_SCOPE,
                "contrast_id": contrast_id,
                "block_id": block,
                "role": role,
                "reference_panel": reference,
                "augmented_panel": augmented,
                "primary_model": "bayesian_linear",
                "corroborating_model": "gam_low_df",
                "effect_curve_model": "gam_low_df",
                "gate_status": gate_status,
                **{f"check_{name}": value for name, value in checks.items()},
                "block_gate_passed": passed,
                "panel_effect_eligible": panel_effect_eligible,
                "effect_release_pass": release,
                "failed_check_count": int(len(failed)),
                "reason": reason,
                "mechanistic_claim_allowed": False,
                "causal_claim_allowed": False,
            }
        )
    return (
        pd.DataFrame(bootstrap_rows),
        summary,
        pd.DataFrame(decision_rows),
    )


def _bootstrap_effect_grid(
    effect_candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    config: FeatureAssociationEvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_columns = [
        "scope",
        "block_id",
        "panel",
        "model",
        "feature_name",
        "catalyst_family",
        "grid_quantile",
        "grid_value_numeric",
        "grid_value_category",
        "predicted_log10_centered",
        "predicted_log10_ci_low",
        "predicted_log10_ci_high",
        "effect_ratio",
        "ci95_low",
        "ci95_high",
        "n_records",
        "n_groups",
        "minimum_repeats_per_record",
        "joint_support_common_record_coverage",
        "joint_support_rule",
        "effect_x_semantics",
        "cross_fitted_g_computation",
        "effect_interpretation",
    ]
    release_registry = decisions.copy()
    release_registry["released_model"] = "gam_low_df"
    release_registry["effect_curve_status"] = np.where(
        release_registry["effect_release_pass"].astype(bool),
        "pending_support_check",
        "gated",
    )
    released = release_registry.loc[
        release_registry["effect_release_pass"].astype(bool)
    ]
    output_rows: list[dict[str, Any]] = []
    if not effect_candidates.empty and not released.empty:
        for release in released.itertuples(index=False):
            selected = effect_candidates.loc[
                effect_candidates["scope"].astype(str).eq(PRIMARY_SCOPE)
                & effect_candidates["panel"].astype(str).eq(
                    str(release.augmented_panel)
                )
                & effect_candidates["block_id"].astype(str).eq(
                    str(release.block_id)
                )
                & effect_candidates["model"].astype(str).eq("gam_low_df")
            ].copy()
            if selected.empty:
                continue
            selected["grid_value_category"] = selected[
                "grid_value_category"
            ].fillna("")
            for feature, feature_frame in selected.groupby(
                "feature_name", sort=True
            ):
                cell_keys = [
                    "record_id",
                    "grid_quantile",
                    "grid_value_category",
                ]
                support_cells = (
                    feature_frame.groupby(
                        cell_keys, as_index=False, dropna=False
                    )
                    .agg(
                        n_repeats=("repeat", "nunique"),
                        n_supported_repeats=("joint_support", "sum"),
                    )
                )
                minimum_supported_repeats = max(
                    1,
                    int(
                        np.ceil(
                            config.effect_minimum_common_record_coverage
                            * config.outer_repeats
                        )
                    ),
                )
                support_cells["cell_supported"] = (
                    support_cells["n_repeats"].eq(config.outer_repeats)
                    & support_cells["n_supported_repeats"].ge(
                        minimum_supported_repeats
                    )
                )
                grid_count = int(
                    support_cells[
                        ["grid_quantile", "grid_value_category"]
                    ]
                    .drop_duplicates()
                    .shape[0]
                )
                common_records = (
                    support_cells.groupby("record_id")["cell_supported"]
                    .agg(["sum", "count"])
                    .loc[
                        lambda value: value["count"].eq(grid_count)
                        & value["sum"].eq(grid_count)
                    ]
                    .index.astype(str)
                )
                scope_record_count = int(
                    effect_candidates.loc[
                        effect_candidates["scope"]
                        .astype(str)
                        .eq(PRIMARY_SCOPE),
                        "record_id",
                    ]
                    .astype(str)
                    .nunique()
                )
                if scope_record_count != 97:
                    raise AssertionError(
                        "All-97 effect support must be calibrated on exactly "
                        f"97 records, found {scope_record_count}"
                    )
                common_coverage = float(
                    len(common_records) / scope_record_count
                )
                if (
                    common_coverage
                    < config.effect_minimum_common_record_coverage
                ):
                    continue
                feature_frame = feature_frame.loc[
                    feature_frame["record_id"].astype(str).isin(
                        set(common_records)
                    )
                    & feature_frame["joint_support"].astype(bool)
                ].copy()
                grid_keys = ["grid_quantile", "grid_value_category"]
                for grid_key, grid_frame in feature_frame.groupby(
                    grid_keys, sort=True, dropna=False
                ):
                    grid_quantile, grid_category = grid_key
                    record_level = (
                        grid_frame.groupby(
                            ["record_id", "cv_group_id"], as_index=False
                        )
                        .agg(
                            delta=("predicted_log10_centered", "mean"),
                            grid_value_numeric=(
                                "grid_value_numeric",
                                "mean",
                            ),
                            n_repeats=("repeat", "nunique"),
                        )
                    )
                    if len(record_level) != len(common_records):
                        continue
                    estimate, bootstrap = _group_bootstrap_delta(
                        record_level,
                        replicates=config.bootstrap_replicates,
                        seed_key=(
                            f"effect|{release.contrast_id}|{feature}|"
                            f"{grid_quantile}|{grid_category}|20260724"
                        ),
                    )
                    low, high = np.quantile(
                        bootstrap, [0.025, 0.975]
                    )
                    numeric_values = pd.to_numeric(
                        record_level["grid_value_numeric"], errors="coerce"
                    )
                    output_rows.append(
                        {
                            "scope": PRIMARY_SCOPE,
                            "block_id": str(release.block_id),
                            "panel": str(release.augmented_panel),
                            "model": "gam_low_df",
                            "feature_name": str(feature),
                            "catalyst_family": "pooled",
                            "grid_quantile": grid_quantile,
                            "grid_value_numeric": (
                                float(numeric_values.mean())
                                if numeric_values.notna().any()
                                else np.nan
                            ),
                            "grid_value_category": str(grid_category),
                            "predicted_log10_centered": estimate,
                            "predicted_log10_ci_low": float(low),
                            "predicted_log10_ci_high": float(high),
                            "effect_ratio": float(10.0**estimate),
                            "ci95_low": float(10.0**low),
                            "ci95_high": float(10.0**high),
                            "n_records": int(len(record_level)),
                            "n_groups": int(
                                record_level["cv_group_id"]
                                .astype(str)
                                .nunique()
                            ),
                            "minimum_repeats_per_record": int(
                                record_level["n_repeats"].min()
                            ),
                            "joint_support_common_record_coverage": (
                                common_coverage
                            ),
                            "joint_support_rule": (
                                "fold-local block-balanced Gower; common "
                                "records supported in >=80% repeats at "
                                "every grid quantile"
                            ),
                            "effect_x_semantics": (
                                "training-fold quantile; fold-specific "
                                "engineered value and median reference"
                            ),
                            "cross_fitted_g_computation": True,
                            "effect_interpretation": (
                                "joint-support-restricted adjusted "
                                "associational ratio; not causal or "
                                "mechanistic"
                            ),
                        }
                    )
    grid = pd.DataFrame(output_rows, columns=grid_columns)
    for row_index, release in release_registry.loc[
        release_registry["effect_release_pass"].astype(bool)
    ].iterrows():
        has_curve = bool(
            not grid.empty
            and (
                grid["block_id"].astype(str).eq(
                    str(release["block_id"])
                )
                & grid["panel"].astype(str).eq(
                    str(release["augmented_panel"])
                )
            ).any()
        )
        if has_curve:
            release_registry.loc[
                row_index, "effect_curve_status"
            ] = "released"
        else:
            release_registry.loc[
                row_index, "effect_release_pass"
            ] = False
            release_registry.loc[
                row_index, "effect_curve_status"
            ] = "passed_no_estimable_curve"
            release_registry.loc[row_index, "reason"] = (
                str(release.get("reason", ""))
                + "; block passed but no numeric feature satisfied the "
                "registered continuous-support and joint-support curve gate"
            ).lstrip("; ")
    return grid, release_registry


def _contrast_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contrast_id": contrast_id,
                "block_id": block,
                "reference_panel": reference,
                "augmented_panel": augmented,
                "role": role,
                "delta_sign": "augmented_minus_reference",
                "primary_scope": PRIMARY_SCOPE,
                "primary_model": "bayesian_linear",
                "corroborating_model": "gam_low_df",
                "effect_curve_model": "gam_low_df",
            }
            for contrast_id, block, reference, augmented, role in FROZEN_CONTRASTS
        ]
    )


def _model_registry(config: FeatureAssociationEvaluationConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": PRIOR_MODEL,
                "scientific_name": "training-fold Gaussian intercept prior",
                "role": "no-feature baseline",
                "fixed_before_outcome_comparison": True,
                "inner_tuning_used": False,
                "full_bayesian_posterior": False,
            },
            {
                "model": "bayesian_linear",
                "scientific_name": "empirical-Bayes shrinkage linear model",
                "role": "primary block-validation model",
                "fixed_before_outcome_comparison": True,
                "inner_tuning_used": False,
                "full_bayesian_posterior": False,
            },
            {
                "model": "gam_low_df",
                "scientific_name": (
                    "empirical-Bayes shrinkage low-df additive spline model"
                ),
                "role": (
                    "corroborating validation model and gated effect model"
                ),
                "fixed_before_outcome_comparison": True,
                "inner_tuning_used": False,
                "full_bayesian_posterior": False,
                "spline_knots": config.gam_n_knots,
                "spline_degree": config.gam_degree,
                "automatic_interactions": False,
            },
        ]
    )


def _write_workflow_report(
    run_dir: Path,
    *,
    audit: Mapping[str, Any],
    atlas_summary: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    block_summary: pd.DataFrame,
    decisions: pd.DataFrame,
    target_qa: Mapping[str, Any],
    config: FeatureAssociationEvaluationConfig,
) -> Path:
    primary_metrics = aggregate_metrics.loc[
        aggregate_metrics["scope"].astype(str).eq(PRIMARY_SCOPE)
    ].sort_values(["panel", "model"])
    support = audit["feature_support"]
    primary_support = support.loc[
        support["scope"].astype(str).eq(PRIMARY_SCOPE)
    ]
    redundancy = audit["numeric_redundancy_pairs"]
    primary_redundancy = redundancy.loc[
        redundancy["scope"].astype(str).eq(PRIMARY_SCOPE)
    ].sort_values("abs_spearman_rho", ascending=False)
    registered = block_summary.loc[
        block_summary["scope"].astype(str).eq(PRIMARY_SCOPE)
        & block_summary["comparison_kind"].astype(str).eq(
            "registered_contrast"
        )
    ].sort_values(["contrast_id", "model"])

    lines = [
        "# Posterior-aware fixed-data feature association",
        "",
        "## Bottom line",
        "",
        (
            "This is a post-hoc, fixed-dataset associational analysis of "
            "all 97 records as one primary population. It does not identify "
            "causal or mechanistic effects, and it does not publish a separate "
            "90-record alkaline result."
        ),
        "",
        (
            f"The audit was restricted to "
            f"{audit['summary']['n_registered_candidate_predictors']} "
            "hash-bound registered predictors. "
            f"{int(primary_support['support_status'].astype(str).eq('supported_descriptive').sum())} "
            "passed the X-only descriptive support rule in the all-97 scope; "
            f"{int((~primary_support['support_status'].astype(str).eq('supported_descriptive')).sum())} "
            "were thin, constant, or otherwise unsupported."
        ),
        "",
    ]
    released = decisions.loc[
        decisions["effect_release_pass"].astype(bool)
    ]
    if config.smoke:
        lines.append(
            "Smoke mode was used. No validation decision or adjusted effect "
            "curve is scientifically releasable."
        )
    elif released.empty:
        lines.append(
            "No block passed every registered validation gate; therefore no "
            "adjusted effect curve was released."
        )
    else:
        blocks = ", ".join(released["block_id"].astype(str))
        lines.append(
            f"Validated and effect-eligible blocks: {blocks}. Adjusted curves "
            "are associational ratios from cross-fitted g-computation."
        )
    lines.extend(
        [
            "",
            "## Feature support and redundancy",
            "",
            primary_support.loc[
                :,
                [
                    "feature_name",
                    "kind",
                    "n_nonmissing",
                    "n_distinct",
                    "n_nonzero_groups",
                    "n_within_group_variable_groups",
                    "support_status",
                ],
            ].to_markdown(index=False),
            "",
        ]
    )
    if len(primary_redundancy):
        lines.extend(
            [
                "Strong numeric redundancy pairs (all-97 scope):",
                "",
                primary_redundancy.head(12).loc[
                    :,
                    [
                        "feature_a",
                        "feature_b",
                        "spearman_rho",
                        "redundancy_status",
                    ],
                ].to_markdown(index=False),
                "",
            ]
        )
    lines.extend(
        [
            "## Posterior-aware marginal atlas",
            "",
            (
                f"The atlas contains {len(atlas_summary)} scope-feature "
                "summaries. Each plot shows one point and the formal posterior "
                "95% interval per record. Numeric Spearman rho or categorical "
                "log-scale eta-squared is recomputed across selected posterior "
                "draws. No smoother, p-value ranking, or target-driven feature "
                "selection is used."
            ),
            "",
            "## Fixed block-wise grouped comparison",
            "",
            (
                f"Validation used {config.outer_folds}-fold connected-group "
                f"holdout repeated {config.outer_repeats} times. All "
                "preprocessing and feature transformations were learned inside "
                "each training fold."
            ),
            "",
            registered.loc[
                :,
                [
                    "contrast_id",
                    "model",
                    "observed_delta",
                    "ci95_low",
                    "ci95_high",
                    "repeat_improvement_fraction",
                    "posterior_draw_improvement_fraction",
                ],
            ].to_markdown(index=False),
            "",
            "Primary-scope aggregate metrics:",
            "",
            primary_metrics.loc[
                :,
                [
                    "panel",
                    "model",
                    "posterior_expected_mae_mV_dec",
                    "posterior_expected_mae_log10",
                    "posterior_expected_nll_log10",
                    "rmse_mV_dec",
                    "spearman",
                    "r2",
                ],
            ].to_markdown(index=False),
            "",
            "## Release decisions",
            "",
            decisions.loc[
                :,
                [
                    "contrast_id",
                    "block_id",
                    "block_gate_passed",
                    "panel_effect_eligible",
                    "effect_release_pass",
                    "reason",
                ],
            ].to_markdown(index=False),
            "",
            "## Posterior policy",
            "",
            (
                f"Training used one equally weighted row per record and the "
                "log10 formal posterior median. Posterior draws were used only "
                f"for scoring/uncertainty propagation; "
                f"{100 * float(target_qa['nonpositive_draw_fraction']):.3f}% "
                "of draws were nonpositive, retained for original-scale MAE "
                "and excluded from log-scale metrics."
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "A passed block means reproducible out-of-connected-group "
                "incremental predictive association under this frozen "
                "workflow. A released curve is an adjusted associational "
                "description indexed by training-fold feature quantile, "
                "restricted to the preregistered joint-support common-record "
                "set. It is not a fixed physical-dose intervention. Neither "
                "result proves a reaction mechanism, rate-determining step, "
                "or causal formulation effect."
            ),
            "",
        ]
    )
    path = run_dir / "report" / "feature_association_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_feature_association(
    config: StudyConfig,
    run_dir: str | Path,
    *,
    source_run: str | Path,
    smoke: bool = False,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Run the complete fixed-data posterior-aware association workflow."""

    destination = Path(run_dir).expanduser().resolve()
    source = Path(source_run).expanduser().resolve()
    settings = config.section("feature_association")
    _validate_frozen_settings(settings)
    if str(settings.get("status")) != "exploratory_posthoc_fixed_dataset":
        raise ValueError(
            "feature_association.status must remain "
            "exploratory_posthoc_fixed_dataset"
        )
    evaluation_config = _settings_to_config(
        settings, smoke=smoke, n_jobs=n_jobs
    )
    for name in (
        "inputs",
        "audits",
        "registry",
        "atlas",
        "analyses",
        "analyses/blockwise",
        "effects",
        "figures",
        "report",
        "provenance",
    ):
        (destination / name).mkdir(parents=True, exist_ok=True)

    source_hashes = config.section("cardinal_classification").get(
        "source_snapshot_sha256", {}
    )
    records, targets, draws, input_binding = snapshot_cardinal_inputs(
        source,
        destination / "inputs",
        canonical_dataset=config.resolve_path("canonical_dataset"),
        expected_source_hashes=source_hashes,
        require_formal_approval=not smoke,
    )
    registered_schema_path = (
        destination / "inputs" / "registered_panel_schema.csv"
    )

    audit = audit_feature_support_and_redundancy(
        records,
        registered_schema_path,
        scopes=ANALYSIS_SCOPES,
    )
    if (
        int(audit["summary"]["n_registered_candidate_predictors"])
        != EXPECTED_REGISTERED_CANDIDATE_COUNT
    ):
        raise AssertionError("Registered predictor count changed")
    audit_file_map = {
        "candidate_registry": "candidate_registry.csv",
        "feature_support": "feature_support_audit.csv",
        "categorical_levels": "categorical_level_support.csv",
        "numeric_redundancy_pairs": "numeric_redundancy_pairs.csv",
        "redundancy_clusters": "redundancy_clusters.csv",
        "feature_decisions": "feature_exclusion_decisions.csv",
        "data_geometry": "data_geometry.csv",
    }
    for key, filename in audit_file_map.items():
        audit[key].to_csv(destination / "audits" / filename, index=False)
    write_json(
        destination / "audits" / "feature_support_redundancy_qa.json",
        audit["summary"],
    )

    target_summary, score_draws, target_qa = build_posterior_target_summary(
        targets,
        draws,
        draw_repetitions=evaluation_config.target_draw_repetitions,
    )
    target_summary.to_csv(
        destination / "registry" / "posterior_target_summary.csv",
        index=False,
    )
    score_draws.to_csv(
        destination / "registry" / "posterior_score_draws.csv",
        index=False,
    )
    write_json(
        destination / "registry" / "posterior_target_policy_qa.json",
        target_qa,
    )
    frame = _merge_records_and_targets(records, target_summary)
    if len(_scope_frame(frame, PRIMARY_SCOPE)) != 97:
        raise ValueError("Primary scope must contain exactly all 97 records")

    candidate_registry = registered_candidate_registry(
        registered_schema_path,
        expected_count=EXPECTED_REGISTERED_CANDIDATE_COUNT,
    )
    (
        atlas_observations,
        atlas_summary,
        atlas_draw_statistics,
        atlas_qa,
    ) = build_posterior_association_atlas(
        frame,
        score_draws,
        candidate_registry,
        audit["feature_support"],
    )
    atlas_observations.to_csv(
        destination / "atlas" / "atlas_observations.csv", index=False
    )
    atlas_summary.to_csv(
        destination / "atlas" / "atlas_feature_summary.csv", index=False
    )
    atlas_draw_statistics.to_csv(
        destination / "atlas" / "atlas_posterior_associations.csv",
        index=False,
    )
    write_json(destination / "atlas" / "atlas_qa.json", atlas_qa)

    association_panel_registry().to_csv(
        destination / "registry" / "frozen_panel_registry.csv", index=False
    )
    association_engineered_feature_definitions().to_csv(
        destination / "registry" / "engineered_feature_definitions.csv",
        index=False,
    )
    _contrast_registry().to_csv(
        destination / "registry" / "frozen_contrast_registry.csv",
        index=False,
    )
    _model_registry(evaluation_config).to_csv(
        destination / "registry" / "fixed_model_registry.csv", index=False
    )
    write_json(
        destination / "registry" / "evaluation_config.json",
        _jsonable(asdict(evaluation_config)),
    )

    prediction_frames: list[pd.DataFrame] = []
    model_fit_frames: list[pd.DataFrame] = []
    effect_candidate_frames: list[pd.DataFrame] = []
    manifest_frames: list[pd.DataFrame] = []
    for scope in ANALYSIS_SCOPES:
        (
            predictions,
            model_fits,
            effect_candidates,
            split_manifest,
        ) = run_grouped_outer_predictions(
            frame,
            scope=scope,
            config=evaluation_config,
        )
        prediction_frames.append(predictions)
        model_fit_frames.append(model_fits)
        if not effect_candidates.empty:
            effect_candidate_frames.append(effect_candidates)
        manifest_frames.append(split_manifest)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    model_fits = pd.concat(model_fit_frames, ignore_index=True)
    effect_candidates = (
        pd.concat(effect_candidate_frames, ignore_index=True)
        if effect_candidate_frames
        else pd.DataFrame()
    )
    split_manifest = pd.concat(manifest_frames, ignore_index=True)
    score_outputs = score_grouped_predictions(oof_predictions, score_draws)
    (
        bootstrap_draws,
        bootstrap_summary,
        validation_decisions,
    ) = build_blockwise_comparison(
        score_outputs["scored_aggregated"],
        score_outputs["split_metrics"],
        score_outputs["target_draw_metrics"],
        config=evaluation_config,
    )
    adjusted_effect_grid, release_registry = _bootstrap_effect_grid(
        effect_candidates,
        validation_decisions,
        config=evaluation_config,
    )
    failed_blocks = set(
        release_registry.loc[
            ~release_registry["effect_release_pass"].astype(bool),
            "block_id",
        ].astype(str)
    )
    if (
        not adjusted_effect_grid.empty
        and set(adjusted_effect_grid["block_id"].astype(str)).intersection(
            failed_blocks
        )
    ):
        raise AssertionError("A failed block emitted an adjusted effect curve")
    if smoke and not adjusted_effect_grid.empty:
        raise AssertionError("Smoke mode emitted an adjusted effect curve")

    split_manifest.to_csv(
        destination / "analyses" / "outer_split_manifest.csv", index=False
    )
    oof_predictions.to_csv(
        destination / "analyses" / "oof_predictions.csv", index=False
    )
    score_outputs["aggregated_predictions"].to_csv(
        destination / "analyses" / "oof_aggregated.csv", index=False
    )
    score_outputs["scored_repeated"].to_csv(
        destination / "analyses" / "record_level_repeated_scores.csv",
        index=False,
    )
    score_outputs["scored_aggregated"].to_csv(
        destination / "analyses" / "record_level_aggregated_scores.csv",
        index=False,
    )
    score_outputs["split_metrics"].to_csv(
        destination / "analyses" / "split_metrics.csv", index=False
    )
    score_outputs["aggregate_metrics"].to_csv(
        destination / "analyses" / "aggregate_metrics.csv", index=False
    )
    score_outputs["target_draw_metrics"].to_csv(
        destination / "analyses" / "target_draw_metrics.csv", index=False
    )
    model_fits.to_csv(
        destination / "analyses" / "model_fit_registry.csv", index=False
    )
    bootstrap_draws.to_csv(
        destination
        / "analyses"
        / "blockwise"
        / "all_paired_group_bootstrap.csv",
        index=False,
    )
    bootstrap_summary.to_csv(
        destination
        / "analyses"
        / "blockwise"
        / "all_paired_group_bootstrap_summary.csv",
        index=False,
    )
    registered_summary = bootstrap_summary.loc[
        bootstrap_summary["comparison_kind"].astype(str).eq(
            "registered_contrast"
        )
    ].copy()
    registered_summary.to_csv(
        destination
        / "analyses"
        / "blockwise"
        / "paired_group_bootstrap_summary.csv",
        index=False,
    )
    validation_decisions.to_csv(
        destination
        / "analyses"
        / "blockwise"
        / "block_validation_decisions.csv",
        index=False,
    )
    release_registry.to_csv(
        destination / "effects" / "effect_release_registry.csv",
        index=False,
    )
    adjusted_effect_grid.to_csv(
        destination / "effects" / "adjusted_effect_grid.csv",
        index=False,
    )

    snapshot_verification = verify_cardinal_input_snapshot(
        destination / "inputs", source_hashes
    )
    source_verification = verify_source_run_unchanged(
        source,
        expected_tree_sha256=input_binding[
            "source_run_tree_sha256_before"
        ],
    )
    write_json(
        destination / "provenance" / "feature_association_input_binding.json",
        input_binding,
    )
    write_json(
        destination / "provenance" / "input_snapshot_verification.json",
        snapshot_verification,
    )
    write_json(
        destination / "provenance" / "source_run_verification.json",
        source_verification,
    )

    qa = {
        "status": "pass",
        "smoke": bool(smoke),
        "scientifically_releasable": not smoke,
        "primary_scope": PRIMARY_SCOPE,
        "analysis_scopes": list(ANALYSIS_SCOPES),
        "n_records_primary_all97": int(
            len(_scope_frame(frame, PRIMARY_SCOPE))
        ),
        "n_registered_candidate_predictors": int(
            len(candidate_registry)
        ),
        "n_frozen_panels": int(len(FROZEN_PANELS)),
        "n_fixed_models": int(len(FROZEN_MODELS)),
        "n_outer_prediction_rows": int(len(oof_predictions)),
        "n_validated_blocks": int(
            validation_decisions["block_gate_passed"].astype(bool).sum()
        ),
        "n_released_effect_blocks": int(
            release_registry["effect_release_pass"].astype(bool).sum()
        ),
        "n_adjusted_effect_grid_rows": int(len(adjusted_effect_grid)),
        "failed_blocks_emitted_adjusted_effects": False,
        "posterior_draws_expanded_as_ml_samples": False,
        "all_preprocessing_fit_inside_training_fold": True,
        "connected_group_holdout": True,
        "models_and_panels_fixed": True,
        "outcome_driven_feature_selection_used": False,
        "one_feature_per_augmented_panel": True,
        "target_blind_support_redundancy_audit": True,
        "atlas_fitted_smoother_used": False,
        "mechanistic_claim_allowed": False,
        "causal_claim_allowed": False,
        "input_snapshot_verified": snapshot_verification["status"] == "pass",
        "formal_source_run_unchanged": source_verification["unchanged"],
    }
    write_json(destination / "analyses" / "analysis_qa.json", qa)
    write_json(
        destination / "effects" / "effects_qa.json",
        {
            "status": "pass",
            "smoke": bool(smoke),
            "release_requires_all_registered_block_checks": True,
            "failed_blocks_emit_effect_curves": False,
            "n_released_blocks": qa["n_released_effect_blocks"],
            "n_adjusted_effect_grid_rows": qa[
                "n_adjusted_effect_grid_rows"
            ],
        },
    )

    report_path = _write_workflow_report(
        destination,
        audit=audit,
        atlas_summary=atlas_summary,
        aggregate_metrics=score_outputs["aggregate_metrics"],
        block_summary=bootstrap_summary,
        decisions=release_registry,
        target_qa=target_qa,
        config=evaluation_config,
    )
    reporting = config.section("reporting")
    figure_manifest = generate_feature_association_figures(
        destination,
        formats=tuple(reporting.get("figure_formats", ("png", "pdf"))),
        dpi=int(reporting.get("figure_dpi", 300)),
        smoke=bool(smoke),
    )
    validation_report = {
        "status": "pass",
        "workflow_order": [
            "feature_support_and_redundancy_audit",
            "posterior_aware_feature_association_atlas",
            "freeze_low_dimensional_panels",
            "blockwise_grouped_model_comparison",
            "gate_controlled_adjusted_effect_curves",
        ],
        "qa": qa,
        "target_qa": target_qa,
        "atlas_qa": atlas_qa,
        "block_decisions": release_registry.to_dict(orient="records"),
        "figure_manifest_rows": int(len(figure_manifest)),
        "report": str(report_path),
    }
    write_json(
        destination / "report" / "validation_report.json",
        _jsonable(validation_report),
    )
    return _jsonable(
        {
            "status": "complete_smoke" if smoke else "complete",
            "run_dir": str(destination),
            "source_run": str(source),
            "qa": qa,
            "validated_blocks": validation_decisions.loc[
                validation_decisions["block_gate_passed"].astype(bool),
                "block_id",
            ].astype(str).tolist(),
            "released_effect_blocks": release_registry.loc[
                release_registry["effect_release_pass"].astype(bool),
                "block_id",
            ].astype(str).tolist(),
            "report": str(report_path),
            "figure_manifest": str(
                destination / "figures" / "figure_manifest.csv"
            ),
        }
    )
