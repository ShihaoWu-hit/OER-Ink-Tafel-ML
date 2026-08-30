"""Outer-test interpretation diagnostics and conservative release gates.

The functions in this module never participate in model or hyperparameter
selection.  Block perturbations are evaluated only after an estimator has been
selected and fitted using an outer-training fold.  The outer-test response is
used only to score the already-defined perturbations.

Saved OOF predictions alone cannot identify feature perturbation effects,
solvent substitutions, or kNN neighbours.  The artifact audit therefore uses
an explicit ``not_estimable_from_saved_artifacts`` status whenever the required
fold estimator state or perturbed predictions were not retained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error


BLOCK_EFFECT_COLUMNS = (
    "target",
    "scope",
    "panel",
    "model",
    "repeat",
    "fold",
    "record_id",
    "cv_group_id",
    "block",
    "perturbation_method",
    "perturbation_repeat",
    "observed_mV_dec",
    "baseline_prediction_mV_dec",
    "perturbed_prediction_mV_dec",
    "baseline_absolute_error_mV_dec",
    "perturbed_absolute_error_mV_dec",
    "delta_absolute_error_mV_dec",
    "importance_seed",
    "selected_params",
    "outer_test_only",
    "selection_role",
)

BLOCK_IMPORTANCE_COLUMNS = (
    "target",
    "scope",
    "panel",
    "model",
    "repeat",
    "fold",
    "block",
    "perturbation_method",
    "perturbation_repeat",
    "n_test",
    "baseline_mae_mV_dec",
    "perturbed_mae_mV_dec",
    "delta_mae_mV_dec",
    "importance_seed",
    "selected_params",
    "outer_test_only",
    "selection_role",
)

STABILITY_COLUMNS = (
    "panel",
    "block",
    "n_required_targets_present",
    "n_required_targets",
    "target_status",
    "target_details",
    "n_required_models_present",
    "n_required_models",
    "model_status",
    "model_details",
    "n_target_model_cells",
    "minimum_splits_per_target_model_cell",
    "positive_split_fraction_minimum",
    "positive_split_fraction_observed_minimum",
    "split_status",
    "split_details",
    "ablation_direction_median_delta_mae_mV_dec",
    "perturbation_method_status",
    "lab_status",
    "catalyst_status",
    "electrolyte_status",
    "lab_details",
    "catalyst_details",
    "electrolyte_details",
    "overall_stability_pass",
    "interpretation_label",
)

INTERPRETATION_SELECTION_ROLE = "posthoc_metric_only_not_model_selection"

_ARTIFACT_BASE_GRID_COLUMNS = (
    "analysis_id",
    "target",
    "scope",
    "panel",
    "model",
    "repeat",
    "fold",
    "block",
)

_ARTIFACT_PERTURBATION_GRID_COLUMNS = (
    *_ARTIFACT_BASE_GRID_COLUMNS,
    "perturbation_method",
    "perturbation_repeat",
)


@dataclass(frozen=True)
class InterpretationConfig:
    """Pre-registered conservative thresholds for exploratory interpretation."""

    importance_panel: str = "context_plus_property_proxy_core"
    permutation_repeats: int = 3
    required_scope: str = "alkaline"
    required_targets: tuple[str, ...] = (
        "b_bayesian_local_line_mV_dec",
        "b_single_regime_mV_dec",
    )
    required_model_families: tuple[str, ...] = (
        "bayesian_ridge",
        "gower_knn",
        "extra_trees",
    )
    positive_split_fraction_minimum: float = 0.70
    minimum_outer_splits_per_cell: int = 25
    minimum_records_per_stratum: int = 5
    minimum_strata: int = 2

    def __post_init__(self) -> None:
        if self.permutation_repeats < 1:
            raise ValueError("permutation_repeats must be positive")
        if not 0.5 <= self.positive_split_fraction_minimum <= 1.0:
            raise ValueError("positive_split_fraction_minimum must be between 0.5 and 1")
        if self.minimum_outer_splits_per_cell < 1:
            raise ValueError("minimum_outer_splits_per_cell must be positive")
        if self.minimum_records_per_stratum < 1 or self.minimum_strata < 2:
            raise ValueError("Stratum stability requires positive record support and at least two strata")


def _to_original(log10_values: np.ndarray | pd.Series) -> np.ndarray:
    return np.power(10.0, np.clip(np.asarray(log10_values, dtype=float), -12.0, 12.0))


def validate_mutually_exclusive_blocks(
    columns: Sequence[str], blocks: Mapping[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """Require an exhaustive, mutually exclusive block partition."""

    normalized = {
        str(block): tuple(str(column) for column in members)
        for block, members in blocks.items()
        if tuple(members)
    }
    assigned = [column for members in normalized.values() for column in members]
    duplicated = sorted({column for column in assigned if assigned.count(column) > 1})
    expected = set(map(str, columns))
    assigned_set = set(assigned)
    missing = sorted(expected - assigned_set)
    unknown = sorted(assigned_set - expected)
    if duplicated or missing or unknown:
        raise ValueError(
            "Interpretation blocks must be mutually exclusive and exhaustive; "
            f"duplicated={duplicated}, missing={missing}, unknown={unknown}"
        )
    return normalized


def outer_test_block_perturbations(
    fitted_estimator: BaseEstimator,
    X_test: pd.DataFrame,
    y_test_mV_dec: Sequence[float],
    baseline_prediction_log10: Sequence[float],
    *,
    blocks: Mapping[str, Sequence[str]],
    record_ids: Sequence[object],
    cv_group_ids: Sequence[object],
    target: str,
    scope: str,
    panel: str,
    model: str,
    repeat: int,
    fold: int,
    selected_params: str,
    permutation_repeats: int = 3,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score block masking and joint permutation on one untouched outer test fold.

    The estimator must already be fitted on the corresponding outer training
    fold.  Every block is jointly permuted, preserving within-row relationships
    inside that block (including the solvent-fraction sum).  No perturbation
    result is returned to the selection code.
    """

    if permutation_repeats < 1:
        raise ValueError("permutation_repeats must be positive")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame with named predictors")
    partition = validate_mutually_exclusive_blocks(tuple(X_test.columns), blocks)
    observed = np.asarray(y_test_mV_dec, dtype=float).reshape(-1)
    baseline_log = np.asarray(baseline_prediction_log10, dtype=float).reshape(-1)
    ids = np.asarray(record_ids, dtype=object).reshape(-1)
    groups = np.asarray(cv_group_ids, dtype=object).reshape(-1)
    if not (len(X_test) == len(observed) == len(baseline_log) == len(ids) == len(groups)):
        raise ValueError("Outer-test perturbation inputs must have identical row counts")
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Outer-test observed slopes must be finite and positive")
    if not np.isfinite(baseline_log).all():
        raise ValueError("Baseline outer-test predictions must be finite")

    baseline = _to_original(baseline_log)
    baseline_error = np.abs(observed - baseline)
    effect_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(random_seed))

    for block_index, (block, members) in enumerate(partition.items()):
        perturbations: list[tuple[str, int, pd.DataFrame, int]] = []

        masked = X_test.copy()
        for column in members:
            masked[column] = np.nan
        perturbations.append(
            ("block_masking_to_fitted_missingness_policy", -1, masked, int(random_seed) + block_index)
        )

        if len(X_test) > 1:
            identity = np.arange(len(X_test))
            for permutation_repeat in range(permutation_repeats):
                permutation = rng.permutation(len(X_test))
                if np.array_equal(permutation, identity):
                    permutation = np.roll(identity, 1)
                permuted = X_test.copy()
                # One row permutation is shared by every member, so a
                # compositional block remains composition-conserving.
                permuted.loc[:, list(members)] = X_test.iloc[permutation][list(members)].to_numpy()
                perturbations.append(
                    (
                        "joint_block_permutation",
                        permutation_repeat,
                        permuted,
                        int(random_seed) + block_index * 10_000 + permutation_repeat + 1,
                    )
                )

        for method, perturbation_repeat, perturbed_X, importance_seed in perturbations:
            perturbed_log = np.asarray(fitted_estimator.predict(perturbed_X), dtype=float)
            perturbed = _to_original(perturbed_log)
            perturbed_error = np.abs(observed - perturbed)
            for position in range(len(X_test)):
                effect_rows.append(
                    {
                        "target": target,
                        "scope": scope,
                        "panel": panel,
                        "model": model,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "record_id": str(ids[position]),
                        "cv_group_id": str(groups[position]),
                        "block": block,
                        "perturbation_method": method,
                        "perturbation_repeat": int(perturbation_repeat),
                        "observed_mV_dec": float(observed[position]),
                        "baseline_prediction_mV_dec": float(baseline[position]),
                        "perturbed_prediction_mV_dec": float(perturbed[position]),
                        "baseline_absolute_error_mV_dec": float(baseline_error[position]),
                        "perturbed_absolute_error_mV_dec": float(perturbed_error[position]),
                        "delta_absolute_error_mV_dec": float(
                            perturbed_error[position] - baseline_error[position]
                        ),
                        "importance_seed": importance_seed,
                        "selected_params": selected_params,
                        "outer_test_only": True,
                        "selection_role": INTERPRETATION_SELECTION_ROLE,
                    }
                )
            baseline_mae = float(mean_absolute_error(observed, baseline))
            perturbed_mae = float(mean_absolute_error(observed, perturbed))
            importance_rows.append(
                {
                    "target": target,
                    "scope": scope,
                    "panel": panel,
                    "model": model,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "block": block,
                    "perturbation_method": method,
                    "perturbation_repeat": int(perturbation_repeat),
                    "n_test": len(X_test),
                    "baseline_mae_mV_dec": baseline_mae,
                    "perturbed_mae_mV_dec": perturbed_mae,
                    "delta_mae_mV_dec": perturbed_mae - baseline_mae,
                    "importance_seed": importance_seed,
                    "selected_params": selected_params,
                    "outer_test_only": True,
                    "selection_role": INTERPRETATION_SELECTION_ROLE,
                }
            )

    return (
        pd.DataFrame(effect_rows, columns=BLOCK_EFFECT_COLUMNS),
        pd.DataFrame(importance_rows, columns=BLOCK_IMPORTANCE_COLUMNS),
    )


def _read_nested(evaluation_dir: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    direct = evaluation_dir / filename
    paths = [direct] if direct.is_file() else sorted(evaluation_dir.glob(f"*/{filename}"))
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        if "analysis_id" not in frame:
            frame.insert(0, "analysis_id", path.parent.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _normalize_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _valid_stratum(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    return ~(
        normalized.isna()
        | normalized.eq("")
        | normalized.isin({"unknown", "unresolved", "missing", "none", "n/a", "na"})
        | normalized.str.startswith(("unknown_", "lab_unknown_", "lab-unknown", "unresolved_"), na=False)
    )


def _stratum_stability(
    effects: pd.DataFrame,
    records: pd.DataFrame,
    *,
    column: str,
    config: InterpretationConfig,
    require_verified_lab: bool = False,
) -> tuple[str, dict[str, Any]]:
    if column not in records:
        return "not_estimable_missing_column", {"column": column}
    metadata_columns = ["record_id", column]
    if require_verified_lab and "lab_group_confidence" in records:
        metadata_columns.append("lab_group_confidence")
    metadata = records.loc[:, metadata_columns].copy()
    metadata["record_id"] = metadata["record_id"].astype(str)
    effects_for_merge = effects.copy()
    effects_for_merge["record_id"] = effects_for_merge["record_id"].astype(str)
    merged = effects_for_merge.merge(metadata, on="record_id", how="left", validate="many_to_one")
    valid = _valid_stratum(merged[column])
    if require_verified_lab:
        if "lab_group_confidence" not in merged:
            return "not_estimable_unverified_lab_metadata", {
                "reason": "lab_group_confidence is unavailable"
            }
        verified = merged["lab_group_confidence"].astype("string").str.strip().str.lower().isin(
            {"verified", "high"}
        )
        valid &= verified
    merged = merged.loc[valid].copy()
    if merged.empty:
        status = "not_estimable_unverified_lab_metadata" if require_verified_lab else "not_estimable_no_valid_strata"
        return status, {"eligible_strata": 0, "eligible_records": 0}

    # One record receives many repeat/model/target perturbations.  Collapse to
    # one record-level effect before checking whether a stratum drives the sign.
    per_record = (
        merged.groupby(["record_id", column], dropna=False, as_index=False)["delta_absolute_error_mV_dec"]
        .mean()
    )
    counts = per_record.groupby(column)["record_id"].nunique()
    eligible_names = counts.loc[counts >= config.minimum_records_per_stratum].index
    eligible = per_record.loc[per_record[column].isin(eligible_names)].copy()
    if len(eligible_names) < config.minimum_strata:
        return "not_estimable_insufficient_supported_strata", {
            "eligible_strata": int(len(eligible_names)),
            "required_strata": config.minimum_strata,
            "records_per_stratum": {str(key): int(value) for key, value in counts.items()},
        }

    within = eligible.groupby(column)["delta_absolute_error_mV_dec"].mean()
    leave_one_out: dict[str, float] = {}
    for stratum in eligible_names:
        remainder = eligible.loc[eligible[column] != stratum, "delta_absolute_error_mV_dec"]
        leave_one_out[str(stratum)] = float(remainder.mean()) if len(remainder) else np.nan
    pass_within = bool((within > 0).all())
    pass_leave_out = bool(all(np.isfinite(value) and value > 0 for value in leave_one_out.values()))
    details = {
        "eligible_strata": int(len(eligible_names)),
        "eligible_records": int(eligible["record_id"].nunique()),
        "mean_effect_by_stratum": {str(key): float(value) for key, value in within.items()},
        "leave_one_stratum_out_mean_effect": leave_one_out,
        "minimum_records_per_stratum": config.minimum_records_per_stratum,
    }
    return ("pass" if pass_within and pass_leave_out else "fail_direction_unstable"), details


def _expected_split_count(evaluation_dir: Path, target: str, scope: str) -> int | None:
    candidates: list[Path] = []
    for path in evaluation_dir.glob("*/evaluation_config.json"):
        qa_path = path.parent / "evaluation_qa.json"
        if not qa_path.is_file():
            continue
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if str(qa.get("target")) == target and str(qa.get("scope")) == scope:
            candidates.append(path)
    if not candidates:
        return None
    config = json.loads(candidates[0].read_text(encoding="utf-8"))
    return int(config["outer_folds"]) * int(config["outer_repeats"])


def _registered_artifact_rows(
    frame: pd.DataFrame, config: InterpretationConfig
) -> pd.DataFrame:
    """Return only the pre-registered interpretation panel and analysis grid."""

    return frame.loc[
        frame["panel"].astype(str).eq(config.importance_panel)
        & frame["scope"].astype(str).eq(config.required_scope)
        & frame["target"].astype(str).isin(config.required_targets)
        & frame["model"].astype(str).isin(config.required_model_families)
    ].copy()


def _audit_outer_test_artifacts(
    importance: pd.DataFrame,
    effects: pd.DataFrame,
    config: InterpretationConfig,
) -> dict[str, Any]:
    """Fail closed unless saved post-hoc perturbations form a complete grid.

    Importance has one row per perturbation.  Effects has one row per test
    record and perturbation.  The audit verifies the immutable post-hoc
    provenance declaration, the one-to-many relationship between those two
    artifacts, and exact numerical reconciliation of their saved metrics
    before any stability result can be released.
    """

    required_columns = {
        "importance": set(_ARTIFACT_PERTURBATION_GRID_COLUMNS)
        | {
            "n_test",
            "delta_mae_mV_dec",
            "outer_test_only",
            "selection_role",
        },
        "effects": set(_ARTIFACT_PERTURBATION_GRID_COLUMNS)
        | {
            "record_id",
            "cv_group_id",
            "delta_absolute_error_mV_dec",
            "outer_test_only",
            "selection_role",
        },
    }
    issues: list[str] = []
    for name, frame in (("importance", importance), ("effects", effects)):
        missing = sorted(required_columns[name] - set(frame.columns))
        if missing:
            issues.append(f"{name} missing required columns: {missing}")

    if issues:
        return {
            "status": "fail_closed_invalid_outer_test_artifacts",
            "n_issues": len(issues),
            "issues": issues,
        }

    for name, frame in (("importance", importance), ("effects", effects)):
        outer_test = _normalize_bool(frame["outer_test_only"])
        if not bool(outer_test.all()):
            issues.append(
                f"{name} contains {int((~outer_test).sum())} rows not declared outer_test_only=True"
            )
        selection_role = frame["selection_role"].astype("string").str.strip()
        invalid_role = ~selection_role.eq(INTERPRETATION_SELECTION_ROLE).fillna(False)
        if bool(invalid_role.any()):
            issues.append(
                f"{name} contains {int(invalid_role.sum())} rows with an invalid selection_role"
            )

    registered_importance = _registered_artifact_rows(importance, config)
    registered_effects = _registered_artifact_rows(effects, config)
    if registered_importance.empty:
        issues.append(
            "importance has no rows for the registered panel/scope/targets/model families"
        )
    if registered_effects.empty:
        issues.append(
            "effects has no rows for the registered panel/scope/targets/model families"
        )
    if registered_importance.empty or registered_effects.empty:
        return {
            "status": "fail_closed_invalid_outer_test_artifacts",
            "n_issues": len(issues),
            "issues": issues,
        }

    expected_permutation_repeats = set(range(config.permutation_repeats))
    permitted_methods = {
        "joint_block_permutation",
        "block_masking_to_fitted_missingness_policy",
    }
    # Every saved row must satisfy the grid contract, including rows for
    # non-release panels.  The registered subset above determines whether the
    # requested stability audit is estimable; it does not excuse malformed
    # auxiliary artifacts.
    for base_key, group in importance.groupby(
        list(_ARTIFACT_BASE_GRID_COLUMNS), dropna=False, sort=True
    ):
        label = dict(zip(_ARTIFACT_BASE_GRID_COLUMNS, map(str, base_key)))
        methods = set(group["perturbation_method"].astype(str))
        unknown_methods = sorted(methods - permitted_methods)
        if unknown_methods:
            issues.append(f"importance grid {label} has unsupported methods {unknown_methods}")

        joint = group.loc[
            group["perturbation_method"].astype(str).eq("joint_block_permutation")
        ]
        joint_repeats = pd.to_numeric(joint["perturbation_repeat"], errors="coerce")
        observed_joint_repeats = {
            int(value) for value in joint_repeats if np.isfinite(value) and float(value).is_integer()
        }
        if (
            len(joint) != config.permutation_repeats
            or not joint_repeats.notna().all()
            or observed_joint_repeats != expected_permutation_repeats
        ):
            issues.append(
                f"importance grid {label} requires exactly {config.permutation_repeats} "
                f"joint permutations with repeats {sorted(expected_permutation_repeats)}"
            )

        masking = group.loc[
            group["perturbation_method"]
            .astype(str)
            .eq("block_masking_to_fitted_missingness_policy")
        ]
        masking_repeats = pd.to_numeric(masking["perturbation_repeat"], errors="coerce")
        if (
            len(masking) != 1
            or not masking_repeats.notna().all()
            or not masking_repeats.eq(-1).all()
        ):
            issues.append(f"importance grid {label} requires exactly one masking row with repeat -1")

    grid_columns = list(_ARTIFACT_PERTURBATION_GRID_COLUMNS)
    duplicate_importance = importance.duplicated(grid_columns, keep=False)
    if bool(duplicate_importance.any()):
        issues.append(
            f"importance contains {int(duplicate_importance.sum())} duplicate perturbation-grid rows"
        )
    duplicate_effects = effects.duplicated(
        [*grid_columns, "record_id"], keep=False
    )
    if bool(duplicate_effects.any()):
        issues.append(
            f"effects contains {int(duplicate_effects.sum())} duplicate record/perturbation rows"
        )

    importance_grid = importance.loc[:, grid_columns].drop_duplicates()
    effects_grid = effects.loc[:, grid_columns].drop_duplicates()
    grid_comparison = importance_grid.merge(
        effects_grid, on=grid_columns, how="outer", indicator=True
    )
    if not grid_comparison["_merge"].eq("both").all():
        issues.append(
            "importance/effects perturbation grids differ: "
            f"importance_only={int(grid_comparison['_merge'].eq('left_only').sum())}, "
            f"effects_only={int(grid_comparison['_merge'].eq('right_only').sum())}"
        )

    effect_counts = (
        effects.groupby(grid_columns, dropna=False, as_index=False)
        .agg(effect_rows=("record_id", "size"), unique_records=("record_id", "nunique"))
    )
    # Duplicates are already an audit issue.  De-duplicate here solely to keep
    # the diagnostic path fail-closed instead of raising a MergeError before a
    # status artifact can be written.
    importance_for_counts = importance.drop_duplicates(grid_columns, keep="first")
    count_audit = importance_for_counts.merge(
        effect_counts, on=grid_columns, how="left", validate="one_to_one"
    )
    expected_n_test = pd.to_numeric(count_audit["n_test"], errors="coerce")
    observed_rows = pd.to_numeric(count_audit["effect_rows"], errors="coerce")
    unique_records = pd.to_numeric(count_audit["unique_records"], errors="coerce")
    valid_n_test = (
        expected_n_test.notna()
        & expected_n_test.gt(0)
        & expected_n_test.mod(1).eq(0)
        & observed_rows.eq(expected_n_test)
        & unique_records.eq(expected_n_test)
    )
    if not bool(valid_n_test.all()):
        issues.append(
            f"{int((~valid_n_test).sum())} importance rows do not map to exactly n_test unique effect rows"
        )

    for base_key, group in effects.groupby(
        list(_ARTIFACT_BASE_GRID_COLUMNS), dropna=False, sort=True
    ):
        record_sets = group.groupby(
            ["perturbation_method", "perturbation_repeat"], dropna=False
        ).apply(
            lambda rows: frozenset(
                zip(rows["record_id"].astype(str), rows["cv_group_id"].astype(str))
            ),
            include_groups=False,
        )
        if len(set(record_sets)) != 1:
            label = dict(zip(_ARTIFACT_BASE_GRID_COLUMNS, map(str, base_key)))
            issues.append(
                f"effects grid {label} does not reuse the same test records for every perturbation"
            )

    # Reconcile every saved perturbation cell, including auxiliary panels.  A
    # finite mean of the record-level error deltas must equal the corresponding
    # importance delta.  This prevents a structurally valid but numerically
    # stale/tampered importance artifact from reaching the stability gate.
    importance_numeric = importance.drop_duplicates(grid_columns, keep="first").copy()
    effects_numeric = effects.copy()
    importance_delta = pd.to_numeric(
        importance_numeric["delta_mae_mV_dec"], errors="coerce"
    )
    effect_delta = pd.to_numeric(
        effects_numeric["delta_absolute_error_mV_dec"], errors="coerce"
    )
    nonfinite_importance_delta = ~np.isfinite(importance_delta.to_numpy(dtype=float))
    nonfinite_effect_delta = ~np.isfinite(effect_delta.to_numpy(dtype=float))
    if bool(nonfinite_importance_delta.any()):
        issues.append(
            f"importance contains {int(nonfinite_importance_delta.sum())} nonfinite delta_mae_mV_dec values"
        )
    if bool(nonfinite_effect_delta.any()):
        issues.append(
            f"effects contains {int(nonfinite_effect_delta.sum())} nonfinite delta_absolute_error_mV_dec values"
        )

    effects_numeric["_numeric_delta_absolute_error"] = effect_delta
    mean_effect_delta = (
        effects_numeric.groupby(grid_columns, dropna=False, as_index=False)
        .agg(mean_effect_delta_mV_dec=("_numeric_delta_absolute_error", "mean"))
    )
    delta_reconciliation = importance_numeric.loc[:, grid_columns].copy()
    delta_reconciliation["importance_delta_mV_dec"] = importance_delta
    delta_reconciliation = delta_reconciliation.merge(
        mean_effect_delta, on=grid_columns, how="left", validate="one_to_one"
    )
    left_delta = pd.to_numeric(
        delta_reconciliation["importance_delta_mV_dec"], errors="coerce"
    ).to_numpy(dtype=float)
    right_delta = pd.to_numeric(
        delta_reconciliation["mean_effect_delta_mV_dec"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_delta_pairs = np.isfinite(left_delta) & np.isfinite(right_delta)
    mismatched_delta = finite_delta_pairs & ~np.isclose(
        left_delta, right_delta, rtol=1e-9, atol=1e-9
    )
    if bool(mismatched_delta.any()):
        issues.append(
            f"{int(mismatched_delta.sum())} perturbation cells have delta_mae_mV_dec "
            "inconsistent with the mean record-level delta_absolute_error_mV_dec"
        )

    importance_error_columns = {
        "baseline_mae_mV_dec",
        "perturbed_mae_mV_dec",
    }
    effects_error_columns = {
        "baseline_absolute_error_mV_dec",
        "perturbed_absolute_error_mV_dec",
    }
    present_importance_error_columns = importance_error_columns & set(importance.columns)
    present_effects_error_columns = effects_error_columns & set(effects.columns)
    if present_importance_error_columns and present_importance_error_columns != importance_error_columns:
        issues.append(
            "importance must provide baseline_mae_mV_dec and perturbed_mae_mV_dec together"
        )
    if present_effects_error_columns and present_effects_error_columns != effects_error_columns:
        issues.append(
            "effects must provide baseline_absolute_error_mV_dec and "
            "perturbed_absolute_error_mV_dec together"
        )

    if present_importance_error_columns == importance_error_columns:
        baseline_mae = pd.to_numeric(
            importance_numeric["baseline_mae_mV_dec"], errors="coerce"
        ).to_numpy(dtype=float)
        perturbed_mae = pd.to_numeric(
            importance_numeric["perturbed_mae_mV_dec"], errors="coerce"
        ).to_numpy(dtype=float)
        nonfinite_importance_mae = ~(
            np.isfinite(baseline_mae) & np.isfinite(perturbed_mae)
        )
        if bool(nonfinite_importance_mae.any()):
            issues.append(
                f"importance contains {int(nonfinite_importance_mae.sum())} rows with nonfinite baseline/perturbed MAE"
            )
        finite_importance_mae = ~nonfinite_importance_mae & np.isfinite(
            importance_delta.to_numpy(dtype=float)
        )
        inconsistent_importance_delta = finite_importance_mae & ~np.isclose(
            perturbed_mae - baseline_mae,
            importance_delta.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
        )
        if bool(inconsistent_importance_delta.any()):
            issues.append(
                f"{int(inconsistent_importance_delta.sum())} importance rows have delta_mae_mV_dec "
                "inconsistent with perturbed_mae_mV_dec - baseline_mae_mV_dec"
            )

    if present_effects_error_columns == effects_error_columns:
        baseline_error = pd.to_numeric(
            effects_numeric["baseline_absolute_error_mV_dec"], errors="coerce"
        ).to_numpy(dtype=float)
        perturbed_error = pd.to_numeric(
            effects_numeric["perturbed_absolute_error_mV_dec"], errors="coerce"
        ).to_numpy(dtype=float)
        effect_delta_array = effect_delta.to_numpy(dtype=float)
        nonfinite_effect_errors = ~(
            np.isfinite(baseline_error) & np.isfinite(perturbed_error)
        )
        if bool(nonfinite_effect_errors.any()):
            issues.append(
                f"effects contains {int(nonfinite_effect_errors.sum())} rows with nonfinite baseline/perturbed absolute errors"
            )
        finite_effect_errors = ~nonfinite_effect_errors & np.isfinite(effect_delta_array)
        inconsistent_effect_delta = finite_effect_errors & ~np.isclose(
            perturbed_error - baseline_error,
            effect_delta_array,
            rtol=1e-9,
            atol=1e-9,
        )
        if bool(inconsistent_effect_delta.any()):
            issues.append(
                f"{int(inconsistent_effect_delta.sum())} effects rows have delta_absolute_error_mV_dec "
                "inconsistent with perturbed - baseline absolute error"
            )

        # When both artifacts retain their baseline/perturbed metrics, compare
        # each saved importance MAE with the corresponding record-level mean.
        if present_importance_error_columns == importance_error_columns:
            effects_numeric["_numeric_baseline_error"] = baseline_error
            effects_numeric["_numeric_perturbed_error"] = perturbed_error
            mean_effect_errors = (
                effects_numeric.groupby(grid_columns, dropna=False, as_index=False)
                .agg(
                    mean_baseline_error_mV_dec=("_numeric_baseline_error", "mean"),
                    mean_perturbed_error_mV_dec=("_numeric_perturbed_error", "mean"),
                )
            )
            mae_reconciliation = importance_numeric.loc[:, grid_columns].copy()
            mae_reconciliation["importance_baseline_mae_mV_dec"] = baseline_mae
            mae_reconciliation["importance_perturbed_mae_mV_dec"] = perturbed_mae
            mae_reconciliation = mae_reconciliation.merge(
                mean_effect_errors, on=grid_columns, how="left", validate="one_to_one"
            )
            for importance_column, effect_column, label in (
                (
                    "importance_baseline_mae_mV_dec",
                    "mean_baseline_error_mV_dec",
                    "baseline",
                ),
                (
                    "importance_perturbed_mae_mV_dec",
                    "mean_perturbed_error_mV_dec",
                    "perturbed",
                ),
            ):
                saved = pd.to_numeric(
                    mae_reconciliation[importance_column], errors="coerce"
                ).to_numpy(dtype=float)
                recomputed = pd.to_numeric(
                    mae_reconciliation[effect_column], errors="coerce"
                ).to_numpy(dtype=float)
                finite_mae_pairs = np.isfinite(saved) & np.isfinite(recomputed)
                mismatched_mae = finite_mae_pairs & ~np.isclose(
                    saved, recomputed, rtol=1e-9, atol=1e-9
                )
                if bool(mismatched_mae.any()):
                    issues.append(
                        f"{int(mismatched_mae.sum())} perturbation cells have {label} MAE "
                        "inconsistent with the mean saved record-level absolute error"
                    )

    return {
        "status": (
            "pass_complete_outer_test_posthoc_grid"
            if not issues
            else "fail_closed_invalid_outer_test_artifacts"
        ),
        "n_issues": len(issues),
        "issues": issues,
        "registered_importance_rows": int(len(registered_importance)),
        "registered_effect_rows": int(len(registered_effects)),
        "audited_perturbation_cells": int(len(importance_grid)),
    }


def _stability_table(
    importance: pd.DataFrame,
    effects: pd.DataFrame,
    records: pd.DataFrame,
    evaluation_dir: Path,
    config: InterpretationConfig,
) -> pd.DataFrame:
    primary_method = importance.loc[
        importance["perturbation_method"].astype(str).eq("joint_block_permutation")
        & importance["panel"].astype(str).eq(config.importance_panel)
        & importance["scope"].astype(str).eq(config.required_scope)
        & importance["target"].astype(str).isin(config.required_targets)
        & importance["model"].astype(str).isin(config.required_model_families)
    ].copy()
    effect_primary = effects.loc[
        effects["perturbation_method"].astype(str).eq("joint_block_permutation")
        & effects["panel"].astype(str).eq(config.importance_panel)
        & effects["scope"].astype(str).eq(config.required_scope)
        & effects["target"].astype(str).isin(config.required_targets)
        & effects["model"].astype(str).isin(config.required_model_families)
    ].copy()
    masking = importance.loc[
        importance["perturbation_method"].astype(str).eq("block_masking_to_fitted_missingness_policy")
        & importance["panel"].astype(str).eq(config.importance_panel)
        & importance["scope"].astype(str).eq(config.required_scope)
        & importance["target"].astype(str).isin(config.required_targets)
        & importance["model"].astype(str).isin(config.required_model_families)
    ].copy()

    rows: list[dict[str, Any]] = []
    blocks = sorted(set(primary_method.get("block", pd.Series(dtype=str)).astype(str)))
    for block in blocks:
        block_rows = primary_method.loc[primary_method["block"].astype(str).eq(block)].copy()
        split_level = (
            block_rows.groupby(["target", "model", "repeat", "fold"], as_index=False)["delta_mae_mV_dec"]
            .mean()
        )
        targets_present = set(split_level["target"].astype(str))
        models_present = set(split_level["model"].astype(str))
        target_medians = split_level.groupby("target")["delta_mae_mV_dec"].median()
        model_medians = split_level.groupby("model")["delta_mae_mV_dec"].median()
        target_status = (
            "pass"
            if set(config.required_targets).issubset(targets_present)
            and all(float(target_medians.get(target, np.nan)) > 0 for target in config.required_targets)
            else "fail_or_incomplete"
        )
        model_status = (
            "pass"
            if set(config.required_model_families).issubset(models_present)
            and all(float(model_medians.get(model, np.nan)) > 0 for model in config.required_model_families)
            else "fail_or_incomplete"
        )

        cell_rows: list[dict[str, Any]] = []
        for target in config.required_targets:
            expected = _expected_split_count(evaluation_dir, target, config.required_scope)
            required_splits = max(
                config.minimum_outer_splits_per_cell,
                int(expected) if expected is not None else config.minimum_outer_splits_per_cell,
            )
            for model in config.required_model_families:
                cell = split_level.loc[
                    split_level["target"].astype(str).eq(target)
                    & split_level["model"].astype(str).eq(model)
                ]
                fraction = float((cell["delta_mae_mV_dec"] > 0).mean()) if len(cell) else None
                cell_rows.append(
                    {
                        "target": target,
                        "model": model,
                        "n_splits": int(len(cell)),
                        "required_splits": required_splits,
                        "positive_fraction": fraction,
                        "pass": bool(
                            len(cell) >= required_splits
                            and fraction is not None
                            and np.isfinite(fraction)
                            and fraction >= config.positive_split_fraction_minimum
                        ),
                    }
                )
        split_status = "pass" if cell_rows and all(row["pass"] for row in cell_rows) else "fail_or_incomplete"
        observed_minimum = min(
            (
                row["positive_fraction"]
                for row in cell_rows
                if row["positive_fraction"] is not None
                and np.isfinite(row["positive_fraction"])
            ),
            default=np.nan,
        )
        minimum_splits = min((row["n_splits"] for row in cell_rows), default=0)

        block_masking = masking.loc[masking["block"].astype(str).eq(block)]
        ablation_median = float(pd.to_numeric(block_masking["delta_mae_mV_dec"], errors="coerce").median())
        method_status = "pass" if np.isfinite(ablation_median) and ablation_median > 0 else "fail_or_incomplete"

        block_effects = effect_primary.loc[effect_primary["block"].astype(str).eq(block)].copy()
        lab_status, lab_details = _stratum_stability(
            block_effects,
            records,
            column="lab_group_id",
            config=config,
            require_verified_lab=True,
        )
        catalyst_column = next(
            (column for column in ("catalyst_type", "catalyst_identity") if column in records),
            "catalyst_type",
        )
        catalyst_status, catalyst_details = _stratum_stability(
            block_effects, records, column=catalyst_column, config=config
        )
        electrolyte_status, electrolyte_details = _stratum_stability(
            block_effects, records, column="electrolyte_identity", config=config
        )
        overall = all(
            status == "pass"
            for status in (
                target_status,
                model_status,
                split_status,
                method_status,
                lab_status,
                catalyst_status,
                electrolyte_status,
            )
        )
        rows.append(
            {
                "panel": config.importance_panel,
                "block": block,
                "n_required_targets_present": len(set(config.required_targets).intersection(targets_present)),
                "n_required_targets": len(config.required_targets),
                "target_status": target_status,
                "target_details": json.dumps(
                    {str(key): float(value) for key, value in target_medians.items()},
                    sort_keys=True,
                ),
                "n_required_models_present": len(set(config.required_model_families).intersection(models_present)),
                "n_required_models": len(config.required_model_families),
                "model_status": model_status,
                "model_details": json.dumps(
                    {str(key): float(value) for key, value in model_medians.items()},
                    sort_keys=True,
                ),
                "n_target_model_cells": len(cell_rows),
                "minimum_splits_per_target_model_cell": minimum_splits,
                "positive_split_fraction_minimum": config.positive_split_fraction_minimum,
                "positive_split_fraction_observed_minimum": observed_minimum,
                "split_status": split_status,
                "split_details": json.dumps(cell_rows, sort_keys=True),
                "ablation_direction_median_delta_mae_mV_dec": ablation_median,
                "perturbation_method_status": method_status,
                "lab_status": lab_status,
                "catalyst_status": catalyst_status,
                "electrolyte_status": electrolyte_status,
                "lab_details": json.dumps(lab_details, sort_keys=True),
                "catalyst_details": json.dumps(catalyst_details, sort_keys=True),
                "electrolyte_details": json.dumps(electrolyte_details, sort_keys=True),
                "overall_stability_pass": overall,
                "interpretation_label": (
                    "stable_associational_hypothesis_only"
                    if overall
                    else "exploratory_or_negative_not_mechanistic"
                ),
            }
        )
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def _knn_one_se_gate(
    selection: pd.DataFrame,
    evaluation_dir: Path,
    config: InterpretationConfig,
) -> dict[str, Any]:
    if selection.empty:
        return {
            "status": "not_estimable_from_saved_artifacts",
            "reason": "inner one-SE model-family selection artifact is unavailable",
            "selected_knn_fraction": None,
        }
    primary_target = config.required_targets[0]
    rows = selection.loc[
        selection["target"].astype(str).eq(primary_target)
        & selection["scope"].astype(str).eq(config.required_scope)
        & selection["panel"].astype(str).eq(config.importance_panel)
    ].copy()
    expected = _expected_split_count(evaluation_dir, primary_target, config.required_scope)
    selected = rows.loc[_normalize_bool(rows.get("selected", pd.Series(False, index=rows.index)))].copy()
    per_split = selected.groupby(["outer_repeat", "outer_fold"])["selected_model_family"].agg(
        lambda values: tuple(sorted(set(map(str, values))))
    ) if not selected.empty else pd.Series(dtype=object)
    complete = bool(expected is not None and len(per_split) == expected and per_split.map(len).eq(1).all())
    selected_knn = per_split.map(lambda values: values == ("gower_knn",)) if len(per_split) else pd.Series(dtype=bool)
    ridge = rows.loc[rows["candidate_model_family"].astype(str).eq("bayesian_ridge")].copy()
    ridge_eligible = _normalize_bool(ridge.get("eligible_within_one_se", pd.Series(False, index=ridge.index)))
    ridge_not_eligible_everywhere = bool(len(ridge) == len(per_split) and (~ridge_eligible).all())
    wins = bool(complete and len(per_split) and selected_knn.all() and ridge_not_eligible_everywhere)
    return {
        "status": "pass" if wins else "gated",
        "reason": (
            "gower kNN was selected in every complete outer-training one-SE comparison and the simpler Bayesian ridge was never eligible"
            if wins
            else "kNN was not the unique one-SE winner in every registered primary outer-training fold"
        ),
        "expected_outer_splits": expected,
        "observed_outer_splits": int(len(per_split)),
        "selected_knn_fraction": float(selected_knn.mean()) if len(selected_knn) else None,
        "simpler_bayesian_ridge_ineligible_everywhere": ridge_not_eligible_everywhere,
        "selection_used_outer_test": False,
    }


def build_interpretation_layer(
    evaluation_dir: str | Path,
    records: pd.DataFrame | str | Path,
    *,
    output_dir: str | Path | None = None,
    smoke: bool = False,
    config: InterpretationConfig | None = None,
) -> dict[str, Any]:
    """Audit and gate interpretation artifacts across registered analyses."""

    settings = config or InterpretationConfig()
    evaluation_root = Path(evaluation_dir)
    destination = Path(output_dir) if output_dir is not None else evaluation_root / "interpretation"
    destination.mkdir(parents=True, exist_ok=True)
    record_frame = records.copy() if isinstance(records, pd.DataFrame) else pd.read_csv(records)
    record_frame["record_id"] = record_frame["record_id"].astype(str)
    importance = _read_nested(evaluation_root, "outer_test_block_importance.csv")
    effects = _read_nested(evaluation_root, "outer_test_block_effects.csv")
    selection = _read_nested(evaluation_root, "model_selection_one_se.csv")

    manifest_rows: list[dict[str, Any]] = []
    if smoke:
        stability = pd.DataFrame(columns=STABILITY_COLUMNS)
        importance_status = "not_evaluated_smoke_mode"
        importance_reason = "smoke mode validates structure only; outer-test interpretation is intentionally skipped"
        artifact_audit = {
            "status": "not_evaluated_smoke_mode",
            "n_issues": 0,
            "issues": [],
        }
    elif importance.empty or effects.empty:
        stability = pd.DataFrame(columns=STABILITY_COLUMNS)
        importance_status = "not_estimable_from_saved_artifacts"
        importance_reason = (
            "saved OOF predictions contain only unperturbed predictions; fitted outer-fold estimators or "
            "outer-test perturbed predictions are required for block importance"
        )
        artifact_audit = {
            "status": "not_estimable_from_saved_artifacts",
            "n_issues": 0,
            "issues": [],
        }
    else:
        artifact_audit = _audit_outer_test_artifacts(importance, effects, settings)
        if artifact_audit["status"] != "pass_complete_outer_test_posthoc_grid":
            stability = pd.DataFrame(columns=STABILITY_COLUMNS)
            importance_status = "fail_closed_invalid_outer_test_artifacts"
            issue_summary = "; ".join(map(str, artifact_audit["issues"][:5]))
            importance_reason = f"outer-test interpretation artifact audit failed closed: {issue_summary}"
        else:
            stability = _stability_table(importance, effects, record_frame, evaluation_root, settings)
            importance_status = "available_outer_test_posthoc"
            importance_reason = (
                "joint block permutations and masking were scored only on outer test folds after inner selection"
            )
    stability_path = destination / "interpretation_stability.csv"
    stability.to_csv(stability_path, index=False)
    manifest_rows.append(
        {
            "artifact": "outer_test_block_importance",
            "status": importance_status,
            "path": (
                str(evaluation_root / "*" / "outer_test_block_importance.csv")
                if importance_status == "available_outer_test_posthoc"
                else ""
            ),
            "reason": importance_reason,
        }
    )
    manifest_rows.append(
        {
            "artifact": "stability_gate",
            "status": (
                "pass_some_blocks"
                if not stability.empty and _normalize_bool(stability["overall_stability_pass"]).any()
                else (
                    "not_evaluated_smoke_mode"
                    if smoke
                    else (
                        "fail_closed_invalid_outer_test_artifacts"
                        if importance_status == "fail_closed_invalid_outer_test_artifacts"
                        else "no_blocks_pass"
                    )
                )
            ),
            "path": str(stability_path),
            "reason": (
                "A pass permits an associational mechanistic hypothesis, never a causal claim."
                if not stability.empty
                else importance_reason
            ),
        }
    )

    solvent_pass = False
    if not stability.empty:
        solvent = stability.loc[stability["block"].astype(str).eq("solvent_composition")]
        solvent_pass = bool(
            len(solvent) == 1 and _normalize_bool(solvent["overall_stability_pass"]).iloc[0]
        )
    if smoke:
        substitution_status = "not_evaluated_smoke_mode"
        substitution_reason = "smoke mode cannot release interpretation products"
    elif not solvent_pass:
        substitution_status = "gated"
        substitution_reason = (
            "solvent_composition did not pass target/split/model/lab/catalyst/electrolyte stability"
        )
    else:
        substitution_status = "not_estimable_from_saved_artifacts"
        substitution_reason = (
            "the stability gate passed, but a composition-conserving substitution requires retained outer-fold "
            "estimators (or a deterministic refit) plus a pre-registered donor/recipient contrast; OOF predictions "
            "alone cannot estimate it"
        )
    manifest_rows.append(
        {
            "artifact": "composition_conserving_solvent_substitution",
            "status": substitution_status,
            "path": "",
            "reason": substitution_reason,
        }
    )

    knn_gate = _knn_one_se_gate(selection, evaluation_root, settings) if not smoke else {
        "status": "not_evaluated_smoke_mode",
        "reason": "smoke mode cannot establish a registered one-SE winner",
        "selected_knn_fraction": None,
    }
    if knn_gate["status"] == "pass":
        neighbor_status = "not_estimable_from_saved_artifacts"
        neighbor_reason = (
            "kNN passed the one-SE gate, but saved OOF predictions do not contain fold-specific distances or "
            "neighbour identities; retain fitted Gower fold state or deterministically refit before case studies"
        )
    else:
        neighbor_status = str(knn_gate["status"])
        neighbor_reason = str(knn_gate["reason"])
    manifest_rows.append(
        {
            "artifact": "knn_neighbor_case_studies",
            "status": neighbor_status,
            "path": "",
            "reason": neighbor_reason,
        }
    )

    artifact_manifest = pd.DataFrame(manifest_rows)
    artifact_manifest_path = destination / "interpretation_artifact_manifest.csv"
    artifact_manifest.to_csv(artifact_manifest_path, index=False)
    status = {
        "status": "structural_only_smoke" if smoke else "complete_gate_audit",
        "smoke": smoke,
        "scientific_interpretation_allowed": bool(
            not smoke and not stability.empty and _normalize_bool(stability["overall_stability_pass"]).any()
        ),
        "importance_status": importance_status,
        "importance_reason": importance_reason,
        "importance_artifact_audit": artifact_audit,
        "n_stability_blocks": int(len(stability)),
        "n_blocks_passing_all_stability_dimensions": int(
            _normalize_bool(stability["overall_stability_pass"]).sum() if not stability.empty else 0
        ),
        "knn_one_se_gate": knn_gate,
        "composition_substitution_status": substitution_status,
        "neighbor_case_status": neighbor_status,
        "config": asdict(settings),
        "interpretation_boundary": (
            "Outer-test perturbation importance is conditional prediction sensitivity; it is not causal or mechanistic proof."
        ),
    }
    (destination / "interpretation_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return status
