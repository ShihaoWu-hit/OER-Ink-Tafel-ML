"""Auditable continuous-target construction for the Ink–Tafel study.

This module deliberately reads frozen artifacts instead of importing either the
legacy kNN monolith or the legacy current-conditioned package. The registered
ML response is the calibrated Bayesian local-line apparent Tafel slope in the
lowest-potential supported strictly increasing-current interval at
``eta >= 0.15 V``. Records without a supported interval remain registered with
a missing target; no selected-model or other fallback is permitted. Exact-2 mA
functionals are retained only as audit diagnostics and never enter ML training.
Incomplete named PPC review is surfaced rather than silently treated as complete.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .bayesian_evidence import (
    discover_bayesian_evidence,
    validate_bound_ppc_review,
    write_evidence_inventory,
)


LOG10 = float(np.log(10.0))
SINGLE_MODELS = {
    "linear",
    "paper_single_plateau",
    "single_background_plateau",
}
TWO_REGIME_MODELS = {
    "two_regime_plateau",
    "two_regime_background_plateau",
}
SUPPORTED_MODELS = SINGLE_MODELS | TWO_REGIME_MODELS

EXACT_FIXED_CURRENT_TARGET = "b_refit_at_2mA_mV_dec"
BAYESIAN_LOCAL_LINE_TARGET = "b_bayesian_local_line_mV_dec"
PRIMARY_TARGET = BAYESIAN_LOCAL_LINE_TARGET
# Compatibility aliases for legacy callers; neither denotes an active
# selected-model low branch or the rejected first-supported v1 target.
FIRST_SUPPORTED_TARGET = BAYESIAN_LOCAL_LINE_TARGET
LOW_BRANCH_TARGET = BAYESIAN_LOCAL_LINE_TARGET
SINGLE_REGIME_TARGET = "b_single_regime_mV_dec"


def marginal_slope_is_fixed_current_target(model_name: str) -> bool:
    """Return whether a marginal slope alone is the exact registered response.

    Only the strictly linear model has this property without a support-dependent
    joint calculation. Plateau/background and two-regime models require
    chain/draw-synchronous inversion of the complete model at the registered
    current before evaluating the local kinetic slope; a marginal summary is
    insufficient.
    """

    return str(model_name) == "linear"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 digest for an artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class TargetSourcePaths:
    """Frozen inputs used by the target adapter."""

    fit_summary: Path
    sample_candidates: Path
    model_comparison: Path
    bootstrap_draws: Path
    binned_points: Path
    current_manifest: Path
    bayesian_targets: Path
    bayes_dir: Path
    legacy_current_grid_summary: Path
    bayesian_manifest: Path | None = None
    deterministic_config: Path | None = None
    additional_bayes_dirs: tuple[Path, ...] = ()
    ppc_review_overlay: Path | None = None

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "TargetSourcePaths":
        root = Path(repo_root).resolve()
        deterministic = root / "Data/03_outputs/tafel_refitting/outputs/20260714_040837/csv"
        active_knn = root / "KNN_refit_58"
        archived_knn = root / "archive/legacy_20260716/workspace/KNN_refit_58"
        active_current = root / "Current_conditioned_Tafel"
        archived_current = root / "archive/legacy_20260716/workspace/Current_conditioned_Tafel"
        knn = _first_existing(active_knn, archived_knn)
        current = _first_existing(active_current, archived_current)
        return cls(
            fit_summary=deterministic / "curve_fit_summary.csv",
            sample_candidates=deterministic / "sample_refit_candidates.csv",
            model_comparison=deterministic / "model_comparison.csv",
            bootstrap_draws=deterministic / "bootstrap_draws.csv",
            binned_points=deterministic / "binned_curve_points.csv",
            current_manifest=current / "data/derived/sample_inclusion_manifest.csv",
            bayesian_targets=knn / "data/Data_ink_refit_58_bayesian.csv",
            bayes_dir=knn / "refit_runs/20260715_122421/bayes",
            legacy_current_grid_summary=(
                current / "data/derived/current_conditioned_target_summary.csv"
            ),
            bayesian_manifest=knn / "data/Data_ink_refit_58_bayesian_manifest.json",
            deterministic_config=deterministic.parent / "model_config.json",
            additional_bayes_dirs=(deterministic.parent / "bayes",),
        )

    @classmethod
    def from_explicit(
        cls,
        *,
        refit_run: Path,
        bayesian_table: Path,
        bayesian_manifest: Path | None,
        current_conditioned_dir: Path,
        bayesian_batch_dir: Path | None = None,
        ppc_review_overlay: Path | None = None,
    ) -> "TargetSourcePaths":
        """Resolve the integration-facing explicit artifact contract."""

        refit_run = Path(refit_run).resolve()
        refit_csv = refit_run / "csv" if (refit_run / "csv").is_dir() else refit_run
        current_root = Path(current_conditioned_dir).resolve()
        current_derived = (
            current_root / "data/derived"
            if (current_root / "data/derived").is_dir()
            else current_root
        )
        bayesian_table = Path(bayesian_table).resolve()
        manifest_path = Path(bayesian_manifest).resolve() if bayesian_manifest else None

        bayes_candidates: list[Path] = []
        if manifest_path and manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_run = payload.get("source_run")
            if source_run:
                bayes_candidates.append(Path(source_run) / "bayes")
                bayes_candidates.append(
                    bayesian_table.parent.parent
                    / "refit_runs"
                    / Path(source_run).name
                    / "bayes"
                )
        bayes_candidates.extend(
            sorted(bayesian_table.parent.parent.glob("refit_runs/*/bayes"), reverse=True)
        )
        if not bayes_candidates:
            bayes_candidates.append(bayesian_table.parent / "bayes")
        bayes_dir = _first_existing(*bayes_candidates)
        batch_root = Path(bayesian_batch_dir).resolve() if bayesian_batch_dir else None
        extra_dirs = [refit_run / "bayes"]
        if batch_root is not None:
            extra_dirs.append(batch_root / "deterministic_run_copy" / "bayes")
        return cls(
            fit_summary=refit_csv / "curve_fit_summary.csv",
            sample_candidates=refit_csv / "sample_refit_candidates.csv",
            model_comparison=refit_csv / "model_comparison.csv",
            bootstrap_draws=refit_csv / "bootstrap_draws.csv",
            binned_points=refit_csv / "binned_curve_points.csv",
            current_manifest=current_derived / "sample_inclusion_manifest.csv",
            bayesian_targets=bayesian_table,
            bayes_dir=bayes_dir,
            legacy_current_grid_summary=(
                current_derived / "current_conditioned_target_summary.csv"
            ),
            bayesian_manifest=manifest_path,
            deterministic_config=refit_run / "model_config.json",
            additional_bayes_dirs=tuple(extra_dirs),
            ppc_review_overlay=(
                Path(ppc_review_overlay).resolve()
                if ppc_review_overlay is not None
                else (batch_root / "ppc_review.csv" if batch_root is not None else None)
            ),
        )

    def validate(self) -> None:
        files = {
            "fit_summary": self.fit_summary,
            "sample_candidates": self.sample_candidates,
            "model_comparison": self.model_comparison,
            "bootstrap_draws": self.bootstrap_draws,
            "binned_points": self.binned_points,
            "current_manifest": self.current_manifest,
            "bayesian_targets": self.bayesian_targets,
            "legacy_current_grid_summary": self.legacy_current_grid_summary,
        }
        missing = [f"{name}: {path}" for name, path in files.items() if not path.is_file()]
        if not self.bayes_dir.is_dir():
            missing.append(f"bayes_dir: {self.bayes_dir}")
        if missing:
            raise FileNotFoundError("Missing target source artifacts:\n" + "\n".join(missing))

    def file_items(self) -> dict[str, Path]:
        items = {
            "fit_summary": self.fit_summary,
            "sample_candidates": self.sample_candidates,
            "model_comparison": self.model_comparison,
            "bootstrap_draws": self.bootstrap_draws,
            "binned_points": self.binned_points,
            "current_manifest": self.current_manifest,
            "bayesian_targets": self.bayesian_targets,
            "legacy_current_grid_summary": self.legacy_current_grid_summary,
        }
        if self.bayesian_manifest is not None and self.bayesian_manifest.is_file():
            items["bayesian_manifest"] = self.bayesian_manifest
        if self.deterministic_config is not None and self.deterministic_config.is_file():
            items["deterministic_config"] = self.deterministic_config
        if self.ppc_review_overlay is not None and self.ppc_review_overlay.is_file():
            items["ppc_review_overlay"] = self.ppc_review_overlay
        return items

    def all_bayes_dirs(self) -> tuple[Path, ...]:
        ordered = [self.bayes_dir, *self.additional_bayes_dirs]
        result: list[Path] = []
        seen: set[Path] = set()
        for raw in ordered:
            path = Path(raw).resolve()
            if path in seen or not path.is_dir():
                continue
            seen.add(path)
            result.append(path)
        return tuple(result)


@dataclass(frozen=True, kw_only=True)
class TargetBuildConfig:
    """Registered target-build constants and frozen parity expectations.

    Dataset-dependent counts are intentionally required.  Production callers
    must bind them to the resolved study configuration instead of inheriting a
    historical dataset snapshot from Python defaults.
    """

    expected_curve_count: int
    expected_record_count: int
    expected_primary_eligible: int
    expected_low_branch_eligible: int
    expected_single_regime_eligible: int
    fixed_current_density_mA_cm2: float = 2.0
    minimum_draw_support_fraction: float = 0.80
    inverse_grid_points: int = 2049

    def validate(self) -> None:
        if not np.isfinite(self.fixed_current_density_mA_cm2):
            raise ValueError("fixed current must be finite")
        if self.fixed_current_density_mA_cm2 <= 0:
            raise ValueError("fixed current must be positive")
        if not (0.0 < self.minimum_draw_support_fraction <= 1.0):
            raise ValueError("minimum draw support fraction must be in (0, 1]")
        if self.inverse_grid_points < 129:
            raise ValueError("inverse_grid_points must be at least 129")
        if self.expected_curve_count < 1 or self.expected_record_count < 1:
            raise ValueError("expected curve and record counts must be positive")
        if self.expected_record_count > self.expected_curve_count:
            raise ValueError("expected record count cannot exceed expected curve count")
        for name, value in (
            ("expected_primary_eligible", self.expected_primary_eligible),
            ("expected_low_branch_eligible", self.expected_low_branch_eligible),
            ("expected_single_regime_eligible", self.expected_single_regime_eligible),
        ):
            if value < 0 or value > self.expected_curve_count:
                raise ValueError(
                    f"{name} must be between zero and expected_curve_count"
                )


def _required_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return np.nan
    return converted if np.isfinite(converted) else np.nan


def kinetic_log10(
    model_name: str,
    x: np.ndarray,
    parameters: Mapping[str, float],
    transition_width_V: float,
) -> np.ndarray:
    """Latent kinetic log10 current for an audited selected model."""

    x = np.asarray(x, dtype=float)
    if model_name in SINGLE_MODELS:
        return parameters["a"] + np.exp(parameters["log_m"]) * x
    if model_name in TWO_REGIME_MODELS:
        m1 = np.exp(parameters["log_m1"])
        m2 = np.exp(parameters["log_m2"])
        c = parameters["c"]
        tau = float(transition_width_V)
        integrated_transition = tau * (
            np.logaddexp(0.0, (x - c) / tau) - np.logaddexp(0.0, -c / tau)
        )
        return parameters["a"] + m1 * x + (m2 - m1) * integrated_transition
    raise ValueError(f"Unsupported selected model: {model_name}")


def predict_log10(
    model_name: str,
    x: np.ndarray,
    parameters: Mapping[str, float],
    transition_width_V: float,
) -> np.ndarray:
    """Observed fitted log10 current, including plateau/background terms."""

    kinetic = kinetic_log10(model_name, x, parameters, transition_width_V)
    if model_name == "linear":
        return kinetic
    plateau = parameters["L"]
    signal = kinetic + plateau - np.logaddexp(
        kinetic * LOG10, plateau * LOG10
    ) / LOG10
    if model_name in {"single_background_plateau", "two_regime_background_plateau"}:
        return np.logaddexp(parameters["B"] * LOG10, signal * LOG10) / LOG10
    return signal


def kinetic_slope_mV_dec(
    model_name: str,
    x: np.ndarray,
    parameters: Mapping[str, float],
    transition_width_V: float,
) -> np.ndarray:
    """Return local kinetic dE/dlog10(j), in mV dec^-1."""

    x = np.asarray(x, dtype=float)
    if model_name in SINGLE_MODELS:
        return np.full_like(x, 1000.0 / np.exp(parameters["log_m"]), dtype=float)
    if model_name in TWO_REGIME_MODELS:
        m1 = np.exp(parameters["log_m1"])
        m2 = np.exp(parameters["log_m2"])
        z = np.clip((x - parameters["c"]) / transition_width_V, -700.0, 700.0)
        logistic = 1.0 / (1.0 + np.exp(-z))
        derivative = m1 + (m2 - m1) * logistic
        return 1000.0 / derivative
    raise ValueError(f"Unsupported selected model: {model_name}")


def apparent_local_slope_mV_dec(
    model_name: str,
    x: np.ndarray,
    parameters: Mapping[str, float],
    transition_width_V: float,
) -> np.ndarray:
    """Return dE/dlog10(j_model) for the complete selected model.

    Unlike the latent kinetic slope, this derivative includes the fitted
    plateau and, where present, background-current contribution.  It is kept
    only as a transport/background diagnostic; it is not the registered Tafel
    target.
    """

    x = np.asarray(x, dtype=float)
    kinetic_b = kinetic_slope_mV_dec(
        model_name, x, parameters, transition_width_V
    )
    if model_name == "linear":
        return kinetic_b
    kinetic_derivative = 1000.0 / kinetic_b
    kinetic = kinetic_log10(model_name, x, parameters, transition_width_V)
    plateau = parameters["L"]
    signal = kinetic + plateau - np.logaddexp(
        kinetic * LOG10, plateau * LOG10
    ) / LOG10
    if model_name in {
        "single_background_plateau",
        "two_regime_background_plateau",
    }:
        predicted = np.logaddexp(
            parameters["B"] * LOG10, signal * LOG10
        ) / LOG10
        plateau_fraction = np.clip(
            10.0 ** (signal - plateau), 0.0, 1.0 - 1e-12
        )
        signal_fraction = np.clip(10.0 ** (signal - predicted), 0.0, 1.0)
        derivative = kinetic_derivative * (1.0 - plateau_fraction) * signal_fraction
    else:
        predicted = signal
        plateau_fraction = np.clip(
            10.0 ** (predicted - plateau), 0.0, 1.0 - 1e-12
        )
        derivative = kinetic_derivative * (1.0 - plateau_fraction)
    return 1000.0 / np.maximum(derivative, 1e-12)


def _parameter_names(model_name: str) -> tuple[str, ...]:
    mapping = {
        "linear": ("a", "log_m"),
        "paper_single_plateau": ("a", "log_m", "L"),
        "single_background_plateau": ("B", "a", "log_m", "L"),
        "two_regime_plateau": ("a", "log_m1", "log_m2", "c", "L"),
        "two_regime_background_plateau": (
            "B",
            "a",
            "log_m1",
            "log_m2",
            "c",
            "L",
        ),
    }
    try:
        return mapping[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported selected model: {model_name}") from exc


def _parameters_from_row(model_name: str, row: Mapping[str, Any]) -> dict[str, float]:
    parameters = {name: _finite_float(row[name]) for name in _parameter_names(model_name)}
    if not np.isfinite(list(parameters.values())).all():
        raise ValueError(f"Non-finite selected-model parameters for {model_name}")
    return parameters


def _invert_current_within_potential_support(
    *,
    model_name: str,
    parameters: Mapping[str, float],
    transition_width_V: float,
    x_min: float,
    x_max: float,
    target_log10_current: float,
    inverse_grid_points: int,
) -> float:
    """Exactly invert a monotone selected model without extrapolation."""

    # Retain the argument in the public contract for backward compatibility;
    # the v2 implementation uses a bracketed root rather than grid interpolation.
    del inverse_grid_points
    bounds = predict_log10(
        model_name,
        np.asarray([x_min, x_max], dtype=float),
        parameters,
        transition_width_V,
    )
    if not np.isfinite(bounds).all() or bounds[1] < bounds[0]:
        return np.nan
    tolerance = 1e-10
    if target_log10_current < bounds[0] - tolerance:
        return np.nan
    if target_log10_current > bounds[1] + tolerance:
        return np.nan
    if np.isclose(target_log10_current, bounds[0], atol=tolerance, rtol=0.0):
        return float(x_min)
    if np.isclose(target_log10_current, bounds[1], atol=tolerance, rtol=0.0):
        return float(x_max)

    def residual(value: float) -> float:
        prediction = predict_log10(
            model_name,
            np.asarray([value], dtype=float),
            parameters,
            transition_width_V,
        )[0]
        return float(prediction - target_log10_current)

    try:
        return float(brentq(residual, x_min, x_max, xtol=1e-12, rtol=1e-12))
    except (ValueError, RuntimeError, FloatingPointError):
        return np.nan


def fixed_current_slope(
    *,
    model_name: str,
    parameters: Mapping[str, float],
    transition_width_V: float,
    potential_min_V: float,
    potential_max_V: float,
    observed_current_min_mA_cm2: float,
    observed_current_max_mA_cm2: float,
    target_current_mA_cm2: float,
    inverse_grid_points: int = 2049,
) -> dict[str, Any]:
    """Evaluate one model at an exact fixed current with a strict support gate.

    The complete model locates the fixed-current potential; ``slope_mV_dec`` is
    the local latent-kinetic slope at that coordinate. The complete-model
    apparent derivative is returned separately for diagnostics. The returned
    ``extrapolated`` field is always false. Unsupported targets are represented
    by ``eligible=False`` and a missing slope, never by an extrapolated value.
    """

    values = np.asarray(
        [
            transition_width_V,
            potential_min_V,
            potential_max_V,
            observed_current_min_mA_cm2,
            observed_current_max_mA_cm2,
            target_current_mA_cm2,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or potential_max_V <= potential_min_V:
        return {
            "eligible": False,
            "slope_mV_dec": np.nan,
            "kinetic_slope_mV_dec": np.nan,
            "apparent_slope_mV_dec": np.nan,
            "relative_potential_at_current_V": np.nan,
            "observed_support": False,
            "extrapolated": False,
            "reason": "invalid_or_missing_support_metadata",
        }
    observed_support = bool(
        observed_current_min_mA_cm2 <= target_current_mA_cm2
        <= observed_current_max_mA_cm2
    )
    if not observed_support:
        return {
            "eligible": False,
            "slope_mV_dec": np.nan,
            "kinetic_slope_mV_dec": np.nan,
            "apparent_slope_mV_dec": np.nan,
            "relative_potential_at_current_V": np.nan,
            "observed_support": False,
            "extrapolated": False,
            "reason": "fixed_current_outside_observed_support",
        }
    x_at_current = _invert_current_within_potential_support(
        model_name=model_name,
        parameters=parameters,
        transition_width_V=transition_width_V,
        x_min=potential_min_V,
        x_max=potential_max_V,
        target_log10_current=float(np.log10(target_current_mA_cm2)),
        inverse_grid_points=inverse_grid_points,
    )
    if not np.isfinite(x_at_current):
        return {
            "eligible": False,
            "slope_mV_dec": np.nan,
            "kinetic_slope_mV_dec": np.nan,
            "apparent_slope_mV_dec": np.nan,
            "relative_potential_at_current_V": np.nan,
            "observed_support": True,
            "extrapolated": False,
            "reason": "selected_model_does_not_support_current_within_potential_range",
        }
    coordinate = np.asarray([x_at_current])
    kinetic_slope = float(
        kinetic_slope_mV_dec(
            model_name, coordinate, parameters, transition_width_V
        )[0]
    )
    apparent_slope = float(
        apparent_local_slope_mV_dec(
            model_name, coordinate, parameters, transition_width_V
        )[0]
    )
    # Tafel slope follows the NC/Bayesian kinetic parameterization. The full
    # model (background/plateau included) locates the 2 mA potential; its
    # apparent derivative is retained as a transport/background diagnostic.
    slope = kinetic_slope
    eligible = bool(
        np.isfinite(slope)
        and slope > 0
        and np.isfinite(apparent_slope)
        and apparent_slope > 0
    )
    return {
        "eligible": eligible,
        "slope_mV_dec": slope if eligible else np.nan,
        "kinetic_slope_mV_dec": kinetic_slope if eligible else np.nan,
        "apparent_slope_mV_dec": apparent_slope if eligible else np.nan,
        "relative_potential_at_current_V": x_at_current,
        "observed_support": True,
        "extrapolated": False,
        "reason": "eligible" if eligible else "nonpositive_or_nonfinite_local_slope",
    }


def select_low_potential_branch(row: Mapping[str, Any]) -> dict[str, Any]:
    """Select the model-defined low-potential branch, never the numeric minimum."""

    point = _finite_float(row.get("slope_low_point_mV_dec"))
    low = _finite_float(row.get("slope_low_ci_low_mV_dec"))
    high = _finite_float(row.get("slope_low_ci_high_mV_dec"))
    high_branch = _finite_float(row.get("slope_high_point_mV_dec"))
    if not np.isfinite(point) or point <= 0:
        raise ValueError("Potential-dependent row lacks a positive low-potential slope")
    return {
        "target_value_mV_dec": point,
        "target_interval_low_mV_dec": low,
        "target_interval_high_mV_dec": high,
        "branch_semantics": "low_potential_low_current_branch",
        "branch_selection_rule": "model_coordinate_not_numeric_minimum",
        "numeric_minimum_branch_selected": bool(
            np.isfinite(high_branch) and point <= high_branch
        ),
    }


def _load_inputs(paths: TargetSourcePaths) -> dict[str, pd.DataFrame]:
    paths.validate()
    csv_inputs = {
        "fit_summary": paths.fit_summary,
        "sample_candidates": paths.sample_candidates,
        "model_comparison": paths.model_comparison,
        "bootstrap_draws": paths.bootstrap_draws,
        "binned_points": paths.binned_points,
        "current_manifest": paths.current_manifest,
        "bayesian_targets": paths.bayesian_targets,
        "legacy_current_grid_summary": paths.legacy_current_grid_summary,
    }
    frames = {name: pd.read_csv(path) for name, path in csv_inputs.items()}
    _required_columns(
        frames["fit_summary"],
        [
            "curve_id",
            "sample_record",
            "fit_eligible",
            "fit_status",
            "selected_model",
            "potential_dependent_supported",
            "slope_low_point_mV_dec",
            "slope_high_point_mV_dec",
            "ir_status",
        ],
        "fit summary",
    )
    _required_columns(
        frames["sample_candidates"],
        [
            "sample_record",
            "n_curves",
            "curve_ids",
            "refit_candidate_status",
        ],
        "sample refit candidates",
    )
    _required_columns(
        frames["current_manifest"],
        [
            "curve_id",
            "sample_record",
            "cohort",
            "selected_model",
            "current_min_source_numeric",
            "current_max_source_numeric",
            "included_primary",
            "exclusion_reason",
        ],
        "current support manifest",
    )
    _required_columns(
        frames["bayesian_targets"],
        [
            "curve_ids",
            "sample_record",
            "y_tafel_slope_mV_dec",
            "y_tafel_slope_interval_low_mV_dec",
            "y_tafel_slope_interval_high_mV_dec",
            "bayesian_ppc_approved",
        ],
        "Bayesian single-regime targets",
    )
    return frames


def _build_exact_fixed_current(
    frames: Mapping[str, pd.DataFrame],
    config: TargetBuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = frames["current_manifest"].copy()
    comparisons = frames["model_comparison"].copy()
    bootstrap = frames["bootstrap_draws"].copy()
    points = frames["binned_points"].copy()

    point_spans = (
        points.groupby("curve_id", as_index=False)["relative_potential_V"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "potential_min_V", "max": "potential_max_V"})
    )
    manifest = manifest.merge(point_spans, on="curve_id", how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []

    for support in manifest.to_dict("records"):
        curve_id = str(support["curve_id"])
        row: dict[str, Any] = {
            "curve_id": curve_id,
            "sample_record": int(support["sample_record"]),
            "cohort": support.get("cohort", "unknown"),
            "selected_model": support.get("selected_model", ""),
            "target_current_density_mA_cm2": config.fixed_current_density_mA_cm2,
            "current_min_observed_mA_cm2": _finite_float(
                support.get("current_min_source_numeric")
            ),
            "current_max_observed_mA_cm2": _finite_float(
                support.get("current_max_source_numeric")
            ),
            "potential_min_V": _finite_float(support.get("potential_min_V")),
            "potential_max_V": _finite_float(support.get("potential_max_V")),
            "target_value_mV_dec": np.nan,
            "target_interval_low_mV_dec": np.nan,
            "target_interval_high_mV_dec": np.nan,
            "target_sd_mV_dec": np.nan,
            "point_target_mV_dec": np.nan,
            "point_relative_potential_at_current_V": np.nan,
            "draw_median_relative_potential_at_current_V": np.nan,
            "n_successful_selected_draws": 0,
            "n_draws_supporting_current": 0,
            "draw_support_fraction": 0.0,
            "observed_support": False,
            "target_eligible": False,
            "extrapolated": False,
            "exclusion_reason": str(support.get("exclusion_reason", "not_in_scope")),
            "calculation_method": (
                "exact_complete_model_inverse_then_local_kinetic_slope"
            ),
            "interpolation_used": False,
        }
        if not _as_bool(support.get("included_primary")):
            rows.append(row)
            continue

        model_name = str(support["selected_model"])
        selected_point = comparisons.loc[
            comparisons["curve_id"].astype(str).eq(curve_id)
            & comparisons["model_name"].astype(str).eq(model_name)
            & comparisons["success"].map(_as_bool)
        ]
        if len(selected_point) != 1:
            raise ValueError(
                f"Expected one successful selected point fit for {curve_id}/{model_name}; "
                f"found {len(selected_point)}"
            )
        point_parameters = _parameters_from_row(model_name, selected_point.iloc[0])
        point_result = fixed_current_slope(
            model_name=model_name,
            parameters=point_parameters,
            transition_width_V=_finite_float(support.get("transition_width_V")),
            potential_min_V=row["potential_min_V"],
            potential_max_V=row["potential_max_V"],
            observed_current_min_mA_cm2=row["current_min_observed_mA_cm2"],
            observed_current_max_mA_cm2=row["current_max_observed_mA_cm2"],
            target_current_mA_cm2=config.fixed_current_density_mA_cm2,
            inverse_grid_points=config.inverse_grid_points,
        )
        row["point_target_mV_dec"] = point_result["slope_mV_dec"]
        row["point_relative_potential_at_current_V"] = point_result[
            "relative_potential_at_current_V"
        ]
        row["observed_support"] = point_result["observed_support"]
        if not point_result["eligible"]:
            row["exclusion_reason"] = point_result["reason"]
            rows.append(row)
            continue

        selected_draws = bootstrap.loc[
            bootstrap["curve_id"].astype(str).eq(curve_id)
            & bootstrap["model_name"].astype(str).eq(model_name)
            & bootstrap["success"].map(_as_bool)
        ]
        if selected_draws.empty:
            raise ValueError(f"No successful selected bootstrap draws for {curve_id}")
        slopes: list[float] = []
        x_values: list[float] = []
        for draw in selected_draws.to_dict("records"):
            try:
                parameters = _parameters_from_row(model_name, draw)
                result = fixed_current_slope(
                    model_name=model_name,
                    parameters=parameters,
                    transition_width_V=_finite_float(support.get("transition_width_V")),
                    potential_min_V=row["potential_min_V"],
                    potential_max_V=row["potential_max_V"],
                    observed_current_min_mA_cm2=row["current_min_observed_mA_cm2"],
                    observed_current_max_mA_cm2=row["current_max_observed_mA_cm2"],
                    target_current_mA_cm2=config.fixed_current_density_mA_cm2,
                    inverse_grid_points=config.inverse_grid_points,
                )
            except (KeyError, TypeError, ValueError, FloatingPointError):
                continue
            if result["eligible"]:
                slopes.append(float(result["slope_mV_dec"]))
                x_values.append(_finite_float(result["relative_potential_at_current_V"]))
                draw_rows.append(
                    {
                        "curve_id": curve_id,
                        "sample_record": int(support["sample_record"]),
                        "bootstrap_replicate": int(draw["bootstrap_replicate"]),
                        "target_current_density_mA_cm2": (
                            config.fixed_current_density_mA_cm2
                        ),
                        "relative_potential_at_current_V": result[
                            "relative_potential_at_current_V"
                        ],
                        "local_slope_mV_dec": result["slope_mV_dec"],
                        "kinetic_slope_mV_dec": result[
                            "kinetic_slope_mV_dec"
                        ],
                        "apparent_slope_mV_dec": result[
                            "apparent_slope_mV_dec"
                        ],
                        "uncertainty_source": "moving_block_bootstrap",
                    }
                )

        values = np.asarray(slopes, dtype=float)
        support_fraction = len(values) / len(selected_draws)
        eligible = bool(
            len(values)
            and support_fraction >= config.minimum_draw_support_fraction
            and point_result["eligible"]
        )
        row.update(
            {
                "target_value_mV_dec": (
                    float(np.quantile(values, 0.50)) if len(values) else np.nan
                ),
                "target_interval_low_mV_dec": (
                    float(np.quantile(values, 0.025)) if len(values) else np.nan
                ),
                "target_interval_high_mV_dec": (
                    float(np.quantile(values, 0.975)) if len(values) else np.nan
                ),
                "target_sd_mV_dec": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                ),
                "draw_median_relative_potential_at_current_V": (
                    float(np.nanmedian(x_values))
                    if np.isfinite(np.asarray(x_values, dtype=float)).any()
                    else np.nan
                ),
                "n_successful_selected_draws": int(len(selected_draws)),
                "n_draws_supporting_current": int(len(values)),
                "draw_support_fraction": float(support_fraction),
                "target_eligible": eligible,
                "exclusion_reason": (
                    "eligible"
                    if eligible
                    else "insufficient_bootstrap_support_at_fixed_current"
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(draw_rows)


def _bayesian_coverage_audit(
    fit_summary: pd.DataFrame,
    paths: TargetSourcePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evidence = discover_bayesian_evidence(
        paths.all_bayes_dirs(),
        current_deterministic_config=paths.deterministic_config,
    )

    # The archived review is retained as historical metadata only.  Formal
    # approval must come from the study-run overlay because that overlay binds
    # the reviewer decision to the current diagnostics/NetCDF/summary/PPC hashes.
    legacy_ppc_path = paths.bayes_dir / "ppc_review.csv"
    legacy_reviews: dict[str, dict[str, Any]] = {}
    if legacy_ppc_path.is_file():
        for row in pd.read_csv(legacy_ppc_path).to_dict("records"):
            legacy_reviews[str(row.get("curve_id", ""))] = row
    overlay_reviews: dict[str, dict[str, Any]] = {}
    if paths.ppc_review_overlay is not None and paths.ppc_review_overlay.is_file():
        overlay = pd.read_csv(paths.ppc_review_overlay, dtype=str).fillna("")
        if overlay["curve_id"].duplicated().any():
            raise ValueError("Manual PPC review overlay has duplicated curve_id rows")
        overlay_reviews = overlay.set_index("curve_id").to_dict("index")

    rows: list[dict[str, Any]] = []
    for fit in fit_summary.loc[fit_summary["fit_eligible"].map(_as_bool)].to_dict("records"):
        curve_id = str(fit["curve_id"])
        selected_model = str(fit.get("selected_model", ""))
        candidates = evidence.loc[
            evidence.get("curve_id", pd.Series(dtype=str)).astype(str).eq(curve_id)
            & evidence.get("model_name", pd.Series(dtype=str)).astype(str).eq(selected_model)
        ].copy()
        if not candidates.empty:
            candidates = candidates.sort_values(
                ["formal_sampling_candidate", "companion_artifacts_complete", "created_at_utc"],
                ascending=[False, False, False],
            )
            chosen = candidates.iloc[0].to_dict()
        else:
            chosen = {}
        formal_sampling = _as_bool(chosen.get("formal_sampling_candidate"))
        sampler_pass = _as_bool(chosen.get("sampler_diagnostics_pass"))
        nc_present = bool(chosen.get("netcdf_path"))
        legacy_review = legacy_reviews.get(curve_id, {})
        bound_review = overlay_reviews.get(curve_id, {})
        review_valid, review_validation = validate_bound_ppc_review(bound_review, chosen)
        ppc_declared = bool(
            _as_bool(bound_review.get("ppc_approved"))
            or _as_bool(legacy_review.get("ppc_approved"))
        )
        reviewer = str(bound_review.get("ppc_reviewer", "")).strip()
        review_date = str(bound_review.get("ppc_review_date", "")).strip()
        review_note = str(bound_review.get("ppc_review_note", "")).strip()
        named_ppc = bool(review_valid)
        formal = bool(formal_sampling and named_ppc)
        if not formal_sampling:
            action = "run_selected_model_nuts_then_named_ppc_review"
        elif not sampler_pass:
            action = "resolve_sampler_diagnostics_then_named_ppc_review"
        elif not _as_bool(bound_review.get("ppc_approved")):
            action = "perform_ppc_review"
        elif not named_ppc:
            action = f"repair_ppc_review_binding:{review_validation}"
        elif not nc_present:
            action = "restore_or_rerun_missing_netcdf"
        else:
            action = "complete"
        rows.append(
            {
                "curve_id": curve_id,
                "sample_record": int(fit["sample_record"]),
                "selected_model": selected_model,
                "potential_dependent_supported": _as_bool(
                    fit.get("potential_dependent_supported")
                ),
                "nuts_diagnostics_present": bool(len(candidates)),
                "formal_nuts_evidence_present": formal_sampling,
                "netcdf_present": nc_present,
                "max_r_hat": _finite_float(chosen.get("max_r_hat")),
                "min_ess_bulk": _finite_float(chosen.get("min_ess_bulk")),
                "divergences": _finite_float(chosen.get("divergences")),
                "sampler_diagnostics_pass": sampler_pass,
                "ppc_approved_declared": ppc_declared,
                "ppc_reviewer": reviewer,
                "ppc_review_date": review_date,
                "ppc_review_note": review_note,
                "ppc_review_binding_status": review_validation,
                "named_ppc_review_complete": named_ppc,
                "formal_nuts_ppc_approved": formal,
                "next_action": action,
                "evidence_id": chosen.get("evidence_id", ""),
                "diagnostics_path": chosen.get("diagnostics_path", ""),
                "diagnostics_sha256": chosen.get("diagnostics_sha256", ""),
                "netcdf_path": chosen.get("netcdf_path", ""),
                "netcdf_sha256": chosen.get("netcdf_sha256", ""),
                "posterior_summary_path": chosen.get("posterior_summary_path", ""),
                "posterior_summary_sha256": chosen.get("posterior_summary_sha256", ""),
                "ppc_plot_path": chosen.get("ppc_plot_path", ""),
                "ppc_plot_sha256": chosen.get("ppc_plot_sha256", ""),
                "posterior_slope_variable": chosen.get("posterior_slope_variable", ""),
                "posterior_slope_mean_mV_dec": _finite_float(
                    chosen.get("posterior_slope_mean_mV_dec")
                ),
                "posterior_slope_interval_low_mV_dec": _finite_float(
                    chosen.get("posterior_slope_interval_low_mV_dec")
                ),
                "posterior_slope_interval_high_mV_dec": _finite_float(
                    chosen.get("posterior_slope_interval_high_mV_dec")
                ),
                "functional_target_artifact_valid": _as_bool(
                    chosen.get("functional_target_artifact_valid")
                ),
                "functional_target_validation": chosen.get(
                    "functional_target_validation", ""
                ),
                "functional_target_total_draws": chosen.get(
                    "functional_target_total_draws", 0
                ),
                "functional_target_eligible_draws": chosen.get(
                    "functional_target_eligible_draws", 0
                ),
                "functional_target_support_fraction": _finite_float(
                    chosen.get("functional_target_support_fraction")
                ),
                "functional_target_supported": _as_bool(
                    chosen.get("functional_target_supported")
                ),
                "functional_target_median_mV_dec": _finite_float(
                    chosen.get("functional_target_median_mV_dec")
                ),
                "functional_target_mean_mV_dec": _finite_float(
                    chosen.get("functional_target_mean_mV_dec")
                ),
                "functional_target_hdi_low_mV_dec": _finite_float(
                    chosen.get("functional_target_hdi_low_mV_dec")
                ),
                "functional_target_hdi_high_mV_dec": _finite_float(
                    chosen.get("functional_target_hdi_high_mV_dec")
                ),
                "functional_draws_path": chosen.get("functional_draws_path", ""),
                "functional_draws_sha256": chosen.get("functional_draws_sha256", ""),
                "ppc_review_path": (
                    str(paths.ppc_review_overlay.resolve())
                    if paths.ppc_review_overlay is not None
                    and paths.ppc_review_overlay.exists()
                    else ""
                ),
            }
        )
    audit = pd.DataFrame(rows)
    queue = audit.loc[~audit["formal_nuts_ppc_approved"]].copy()
    queue.insert(
        0,
        "queue_priority",
        np.where(queue["formal_nuts_evidence_present"], 2, 1),
    )
    queue = queue.sort_values(
        ["queue_priority", "potential_dependent_supported", "sample_record"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return audit, queue, evidence


def _default_exclusion_reason(row: Mapping[str, Any]) -> str:
    if not _as_bool(row.get("fit_eligible")):
        return str(row.get("fit_exclusion_reason") or "curve_failed_curve_level_qc")
    if _as_bool(row.get("potential_dependent_supported")):
        return "potential_dependent_curve_requires_two_regime_target"
    reason = str(row.get("label_reason") or "").strip()
    if reason and reason.lower() != "nan":
        return reason
    status = str(row.get("refit_candidate_status") or "").strip()
    return status if status and status.lower() != "nan" else "not_supported_by_target_definition"


def _registry_base(
    frames: Mapping[str, pd.DataFrame], config: TargetBuildConfig
) -> pd.DataFrame:
    summary = frames["fit_summary"].copy()
    candidates = frames["sample_candidates"][
        ["sample_record", "n_curves", "curve_ids", "refit_candidate_status"]
    ].copy()
    if candidates["sample_record"].duplicated().any():
        raise ValueError("Sample refit candidates are not unique by sample_record")
    summary = summary.merge(candidates, on="sample_record", how="left", validate="many_to_one")
    observed_curves = int(summary["curve_id"].nunique())
    observed_records = int(summary["sample_record"].nunique())
    if (
        len(summary) != config.expected_curve_count
        or observed_curves != config.expected_curve_count
        or observed_records != config.expected_record_count
    ):
        raise ValueError(
            "Frozen curve registry parity failed: "
            f"observed {observed_curves} curves representing {observed_records} records; "
            f"expected {config.expected_curve_count} curves representing "
            f"{config.expected_record_count} records"
        )
    return summary


def _sample_candidate_gate(
    row: Mapping[str, Any], target_name: str
) -> tuple[bool, str]:
    """Return the sample-level candidate decision for one curve target.

    Bayesian sampling is intentionally curve-level, but modeling targets are
    record-level.  A record with unresolved variants therefore remains in the
    sampling queue while every variant is blocked from the target registry.
    """

    if not _as_bool(row.get("fit_eligible")):
        return False, "curve_failed_curve_level_qc"
    n_curves = _finite_float(row.get("n_curves"))
    listed = [
        item.strip()
        for item in str(row.get("curve_ids", "")).split(";")
        if item.strip() and item.strip().lower() != "nan"
    ]
    curve_id = str(row.get("curve_id", ""))
    if not np.isfinite(n_curves) or int(n_curves) != 1 or len(listed) != 1:
        return False, "multiple_curve_variants_require_resolution"
    if listed[0] != curve_id:
        return False, "sample_candidate_curve_id_mismatch"

    status = str(row.get("refit_candidate_status", "")).strip()
    is_potential_dependent = _as_bool(row.get("potential_dependent_supported"))
    selected_model = str(row.get("selected_model", "")).strip()
    if target_name == EXACT_FIXED_CURRENT_TARGET:
        return (
            (True, "diagnostic_selected_model_exact_current_evaluation")
            if selected_model in SUPPORTED_MODELS
            else (False, "unsupported_selected_model")
        )
    if target_name == LOW_BRANCH_TARGET:
        if selected_model in SINGLE_MODELS and not is_potential_dependent:
            return True, "eligible_selected_single_regime_posterior_slope"
        if selected_model in TWO_REGIME_MODELS and is_potential_dependent:
            return True, "eligible_selected_two_regime_low_potential_branch"
        return False, "selected_model_branch_semantics_inconsistent"
    if status == "eligible_single_curve":
        if is_potential_dependent:
            return False, "single_curve_candidate_status_conflicts_with_selected_model"
        return True, "eligible_single_curve"
    if status == "potential_dependent_no_single_scalar":
        if not is_potential_dependent:
            return False, "potential_dependent_candidate_status_conflicts_with_selected_model"
        if target_name == SINGLE_REGIME_TARGET:
            return False, "potential_dependent_excluded_single_regime_sensitivity"
        return True, "eligible_potential_dependent_curve"
    return False, status or "sample_not_an_eligible_refit_candidate"


def _empty_target_frame(base: pd.DataFrame, name: str, role: str) -> pd.DataFrame:
    columns = [
        "curve_id",
        "sample_record",
        "publication_id",
        "doi",
        "selected_model",
        "ir_status",
        "fit_status",
        "fit_eligible",
        "potential_dependent_supported",
        "n_curves",
        "curve_ids",
        "refit_candidate_status",
    ]
    frame = base[columns].copy()
    frame.insert(0, "target_name", name)
    frame.insert(1, "target_role", role)
    frame["cohort"] = "excluded"
    frame["target_value_mV_dec"] = np.nan
    frame["target_interval_low_mV_dec"] = np.nan
    frame["target_interval_high_mV_dec"] = np.nan
    frame["target_uncertainty_source"] = "not_available"
    frame["sample_candidate_gate_passed"] = False
    frame["sample_candidate_gate_reason"] = "not_evaluated"
    frame["target_structurally_eligible"] = False
    frame["target_eligible"] = False
    frame["exclusion_reason"] = [
        _default_exclusion_reason(row) for row in base.to_dict("records")
    ]
    frame["branch_semantics"] = "not_applicable"
    frame["branch_selection_rule"] = "not_applicable"
    frame["target_current_density_mA_cm2"] = np.nan
    frame["current_min_observed_mA_cm2"] = np.nan
    frame["current_max_observed_mA_cm2"] = np.nan
    frame["observed_support"] = False
    frame["extrapolated"] = False
    frame["calculation_method"] = "not_available"
    frame["interpolation_used"] = False
    frame["bayesian_status"] = "not_available"
    frame["ppc_status"] = "not_available"
    frame["posterior_target_supported"] = False
    frame["formal_target_approved"] = False
    frame["posterior_target_gap_reason"] = "target_not_eligible"
    frame["posterior_evidence_id"] = ""
    frame["posterior_summary_path"] = ""
    return frame


def _make_target_registry(
    frames: Mapping[str, pd.DataFrame],
    exact: pd.DataFrame,
    bayes_audit: pd.DataFrame,
    config: TargetBuildConfig,
) -> pd.DataFrame:
    base = _registry_base(frames, config)
    base_map = base.set_index(base["curve_id"].astype(str)).to_dict("index")
    audit_map = bayes_audit.set_index("curve_id").to_dict("index")

    primary = _empty_target_frame(base, EXACT_FIXED_CURRENT_TARGET, "diagnostic")
    exact_map = exact.set_index("curve_id").to_dict("index")
    for index, curve_id in primary["curve_id"].astype(str).items():
        candidate_passed, candidate_reason = _sample_candidate_gate(
            base_map[curve_id], EXACT_FIXED_CURRENT_TARGET
        )
        primary.loc[index, "sample_candidate_gate_passed"] = candidate_passed
        primary.loc[index, "sample_candidate_gate_reason"] = candidate_reason
        item = exact_map.get(curve_id)
        if item is None:
            continue
        primary.loc[index, "cohort"] = item.get("cohort", "unknown")
        for column in [
            "target_value_mV_dec",
            "target_interval_low_mV_dec",
            "target_interval_high_mV_dec",
            "target_eligible",
            "exclusion_reason",
            "target_current_density_mA_cm2",
            "current_min_observed_mA_cm2",
            "current_max_observed_mA_cm2",
            "observed_support",
            "extrapolated",
            "calculation_method",
            "interpolation_used",
        ]:
            primary.loc[index, column] = item.get(column)
        primary.loc[index, "target_uncertainty_source"] = "moving_block_bootstrap"
        primary.loc[index, "branch_semantics"] = (
            "local_kinetic_slope_at_complete_model_fixed_current_potential"
        )
        primary.loc[index, "branch_selection_rule"] = "exact_current_model_evaluation"
        primary.loc[index, "bayesian_status"] = "not_used_for_primary_common_uncertainty"
        primary.loc[index, "ppc_status"] = "bootstrap_model_qc; named_bayesian_ppc_separate"
        structurally_eligible = bool(
            candidate_passed and _as_bool(item.get("target_eligible"))
        )
        primary.loc[index, "target_structurally_eligible"] = structurally_eligible
        primary.loc[index, "target_eligible"] = structurally_eligible
        if not candidate_passed:
            primary.loc[index, "target_value_mV_dec"] = np.nan
            primary.loc[index, "target_interval_low_mV_dec"] = np.nan
            primary.loc[index, "target_interval_high_mV_dec"] = np.nan
            primary.loc[index, "exclusion_reason"] = candidate_reason
        audit = audit_map.get(str(curve_id), {})
        posterior_value = _finite_float(
            audit.get("functional_target_median_mV_dec")
        )
        posterior_low = _finite_float(
            audit.get("functional_target_hdi_low_mV_dec")
        )
        posterior_high = _finite_float(
            audit.get("functional_target_hdi_high_mV_dec")
        )
        if (
            _as_bool(primary.loc[index, "target_eligible"])
            and _as_bool(audit.get("formal_nuts_evidence_present"))
            and _as_bool(audit.get("functional_target_supported"))
            and np.isfinite([posterior_value, posterior_low, posterior_high]).all()
        ):
            primary.loc[index, "target_value_mV_dec"] = posterior_value
            primary.loc[index, "target_interval_low_mV_dec"] = posterior_low
            primary.loc[index, "target_interval_high_mV_dec"] = posterior_high
            primary.loc[index, "target_uncertainty_source"] = (
                "chain_draw_synchronous_selected_model_functional_hdi_95"
            )
            primary.loc[index, "calculation_method"] = (
                "joint_posterior_complete_model_inversion_then_local_kinetic_slope_at_exact_2mA"
            )
            primary.loc[index, "bayesian_status"] = (
                "diagnostic_posterior_target_supported_excluded_from_ml"
            )
            primary.loc[index, "posterior_target_supported"] = True
            primary.loc[index, "formal_target_approved"] = False
            primary.loc[index, "posterior_target_gap_reason"] = "complete"
            primary.loc[index, "posterior_evidence_id"] = audit.get("evidence_id", "")
            primary.loc[index, "posterior_summary_path"] = audit.get(
                "posterior_summary_path", ""
            )
        elif _as_bool(primary.loc[index, "target_eligible"]):
            primary.loc[index, "target_eligible"] = False
            primary.loc[index, "target_value_mV_dec"] = np.nan
            primary.loc[index, "target_interval_low_mV_dec"] = np.nan
            primary.loc[index, "target_interval_high_mV_dec"] = np.nan
            primary.loc[index, "posterior_target_gap_reason"] = (
                "current_selected_model_production_posterior_or_functional_target_missing"
            )

    bayesian = frames["bayesian_targets"].copy()
    if bayesian["curve_ids"].astype(str).duplicated().any():
        raise ValueError("Bayesian target source has duplicated curve_ids")
    bayesian_map = bayesian.set_index(bayesian["curve_ids"].astype(str)).to_dict("index")

    low_branch = _empty_target_frame(base, LOW_BRANCH_TARGET, "primary")
    single_only = _empty_target_frame(base, SINGLE_REGIME_TARGET, "sensitivity_a")
    for curve_id, base_item in base_map.items():
        row_indexes = low_branch.index[low_branch["curve_id"].astype(str).eq(curve_id)]
        if len(row_indexes) != 1:
            raise ValueError(f"Curve ID is not unique in registry: {curve_id}")
        index = row_indexes[0]
        low_gate, low_gate_reason = _sample_candidate_gate(
            base_item, LOW_BRANCH_TARGET
        )
        single_gate, single_gate_reason = _sample_candidate_gate(
            base_item, SINGLE_REGIME_TARGET
        )
        for target, gate, reason in (
            (low_branch, low_gate, low_gate_reason),
            (single_only, single_gate, single_gate_reason),
        ):
            target.loc[index, "sample_candidate_gate_passed"] = gate
            target.loc[index, "sample_candidate_gate_reason"] = reason
            target.loc[index, "target_structurally_eligible"] = gate
            if not gate:
                target.loc[index, "exclusion_reason"] = reason

        bayes_item = bayesian_map.get(curve_id)
        audit = audit_map.get(curve_id, {})
        posterior_value = _finite_float(audit.get("posterior_slope_median_mV_dec"))
        posterior_low = _finite_float(audit.get("posterior_slope_draw_hdi_low_mV_dec"))
        posterior_high = _finite_float(audit.get("posterior_slope_draw_hdi_high_mV_dec"))
        if not np.isfinite(posterior_value):
            posterior_value = _finite_float(audit.get("posterior_slope_mean_mV_dec"))
        if not np.isfinite(posterior_low):
            posterior_low = _finite_float(
                audit.get("posterior_slope_interval_low_mV_dec")
            )
        if not np.isfinite(posterior_high):
            posterior_high = _finite_float(
                audit.get("posterior_slope_interval_high_mV_dec")
            )
        posterior_supported = bool(
            _as_bool(audit.get("formal_nuts_evidence_present"))
            and np.isfinite([posterior_value, posterior_low, posterior_high]).all()
        )

        # A structurally eligible single curve is driven by the current,
        # workflow-bound selected-model posterior whenever it exists.  The
        # archived 58-row table is only a historical value fallback; its PPC
        # flag is unbound to the current artifacts and can never confer formal
        # approval.
        if low_gate and not _as_bool(base_item.get("potential_dependent_supported")):
            eligible_targets = (
                (low_branch, single_only) if single_gate else (low_branch,)
            )
            if posterior_supported:
                named = _as_bool(audit.get("named_ppc_review_complete"))
                formal = _as_bool(audit.get("formal_nuts_ppc_approved"))
                for target in eligible_targets:
                    target.loc[index, "target_value_mV_dec"] = posterior_value
                    target.loc[index, "target_interval_low_mV_dec"] = posterior_low
                    target.loc[index, "target_interval_high_mV_dec"] = posterior_high
                    target.loc[index, "cohort"] = "reliable_single_regime"
                    target.loc[index, "target_uncertainty_source"] = (
                        "selected_model_bayesian_posterior_hdi_95"
                    )
                    target.loc[index, "target_eligible"] = True
                    target.loc[index, "exclusion_reason"] = "eligible"
                    target.loc[index, "branch_semantics"] = (
                        "single_regime_posterior_slope"
                    )
                    target.loc[index, "branch_selection_rule"] = (
                        "selected_single_regime_model"
                    )
                    target.loc[index, "calculation_method"] = (
                        "current_selected_single_regime_posterior_median"
                    )
                    target.loc[index, "bayesian_status"] = (
                        "posterior_target_supported"
                    )
                    target.loc[index, "ppc_status"] = (
                        "named_approved" if named else "named_ppc_pending"
                    )
                    target.loc[index, "posterior_target_supported"] = True
                    target.loc[index, "formal_target_approved"] = formal
                    target.loc[index, "posterior_target_gap_reason"] = "complete"
                    target.loc[index, "posterior_evidence_id"] = audit.get(
                        "evidence_id", ""
                    )
                    target.loc[index, "posterior_summary_path"] = audit.get(
                        "posterior_summary_path", ""
                    )
                continue

            legacy_values = None
            if bayes_item is not None:
                candidate_values = {
                    "target_value_mV_dec": _finite_float(
                        bayes_item.get("y_tafel_slope_mV_dec")
                    ),
                    "target_interval_low_mV_dec": _finite_float(
                        bayes_item.get("y_tafel_slope_interval_low_mV_dec")
                    ),
                    "target_interval_high_mV_dec": _finite_float(
                        bayes_item.get("y_tafel_slope_interval_high_mV_dec")
                    ),
                }
                if np.isfinite(list(candidate_values.values())).all():
                    legacy_values = candidate_values
            if legacy_values is not None:
                for target in eligible_targets:
                    for column, value in legacy_values.items():
                        target.loc[index, column] = value
                    target.loc[index, "cohort"] = "reliable_single_regime"
                    target.loc[index, "target_uncertainty_source"] = (
                        "legacy_historical_bayesian_posterior_hdi_95"
                    )
                    target.loc[index, "target_eligible"] = True
                    target.loc[index, "exclusion_reason"] = "eligible_historical_fallback"
                    target.loc[index, "branch_semantics"] = (
                        "single_regime_posterior_slope"
                    )
                    target.loc[index, "branch_selection_rule"] = (
                        "selected_single_regime_model"
                    )
                    target.loc[index, "calculation_method"] = (
                        "legacy_historical_bayesian_posterior_mean"
                    )
                    target.loc[index, "bayesian_status"] = (
                        "legacy_historical_fallback_current_posterior_pending"
                    )
                    target.loc[index, "ppc_status"] = (
                        "historical_unbound_not_formal"
                    )
                    target.loc[index, "posterior_target_supported"] = False
                    target.loc[index, "formal_target_approved"] = False
                    target.loc[index, "posterior_target_gap_reason"] = (
                        "current_selected_model_production_posterior_missing"
                    )
                continue

            for target in eligible_targets:
                target.loc[index, "exclusion_reason"] = (
                    "current_selected_model_production_posterior_missing"
                )
                target.loc[index, "bayesian_status"] = (
                    "selected_model_nuts_pending"
                )
                target.loc[index, "ppc_status"] = "named_ppc_pending"
                target.loc[index, "posterior_target_gap_reason"] = (
                    "current_selected_model_production_posterior_missing"
                )
            continue

        if low_gate and _as_bool(base_item.get("potential_dependent_supported")):
            selected = select_low_potential_branch(base_item)
            for column in [
                "target_value_mV_dec",
                "target_interval_low_mV_dec",
                "target_interval_high_mV_dec",
                "branch_semantics",
                "branch_selection_rule",
            ]:
                low_branch.loc[index, column] = selected[column]
            low_branch.loc[index, "cohort"] = "potential_dependent"
            low_branch.loc[index, "target_uncertainty_source"] = "moving_block_bootstrap"
            low_branch.loc[index, "target_eligible"] = True
            low_branch.loc[index, "exclusion_reason"] = "eligible"
            low_branch.loc[index, "calculation_method"] = (
                "selected_two_regime_low_potential_branch"
            )
            low_branch.loc[index, "bayesian_status"] = "selected_model_nuts_pending"
            low_branch.loc[index, "ppc_status"] = "named_ppc_pending"
            low_branch.loc[index, "posterior_target_gap_reason"] = (
                "current_selected_model_production_posterior_missing"
            )
            if posterior_supported:
                low_branch.loc[index, "target_value_mV_dec"] = posterior_value
                low_branch.loc[index, "target_interval_low_mV_dec"] = posterior_low
                low_branch.loc[index, "target_interval_high_mV_dec"] = posterior_high
                low_branch.loc[index, "target_uncertainty_source"] = (
                    "selected_model_bayesian_posterior_hdi_95"
                )
                low_branch.loc[index, "calculation_method"] = (
                    "current_selected_two_regime_low_potential_posterior_median"
                )
                low_branch.loc[index, "bayesian_status"] = "posterior_target_supported"
                low_branch.loc[index, "posterior_target_supported"] = True
                low_branch.loc[index, "formal_target_approved"] = _as_bool(
                    audit.get("formal_nuts_ppc_approved")
                )
                low_branch.loc[index, "posterior_target_gap_reason"] = "complete"
                low_branch.loc[index, "posterior_evidence_id"] = audit.get(
                    "evidence_id", ""
                )
                low_branch.loc[index, "posterior_summary_path"] = audit.get(
                    "posterior_summary_path", ""
                )

    registry = pd.concat([primary, low_branch, single_only], ignore_index=True)
    registry["sample_candidate_gate_passed"] = registry[
        "sample_candidate_gate_passed"
    ].map(_as_bool)
    registry["target_structurally_eligible"] = registry[
        "target_structurally_eligible"
    ].map(_as_bool)
    registry["target_eligible"] = registry["target_eligible"].map(_as_bool)
    registry["observed_support"] = registry["observed_support"].map(_as_bool)
    registry["extrapolated"] = registry["extrapolated"].map(_as_bool)
    registry["interpolation_used"] = registry["interpolation_used"].map(_as_bool)
    registry["posterior_target_supported"] = registry["posterior_target_supported"].map(
        _as_bool
    )
    registry["formal_target_approved"] = registry["formal_target_approved"].map(
        _as_bool
    )
    return registry


def _targets_wide(registry: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in ["record_id", "sample_record", "cv_group_id", "electrolyte_scope"]
        if column in registry.columns
    ]
    wide = (
        registry[metadata_columns]
        .drop_duplicates()
        .sort_values("sample_record")
        .reset_index(drop=True)
    )
    if wide["sample_record"].duplicated().any():
        raise ValueError("Canonical record metadata is inconsistent within sample_record")
    for target_name in [
        EXACT_FIXED_CURRENT_TARGET,
        LOW_BRANCH_TARGET,
        SINGLE_REGIME_TARGET,
    ]:
        subset = registry.loc[
            registry["target_name"].eq(target_name) & registry["target_eligible"]
        ].copy()
        if subset["sample_record"].duplicated().any():
            duplicated = subset.loc[subset["sample_record"].duplicated(False), "sample_record"]
            raise ValueError(
                f"Target {target_name} has more than one eligible curve per record: "
                f"{sorted(duplicated.astype(int).unique())}"
            )
        mapped = subset.set_index("sample_record")
        wide[target_name] = wide["sample_record"].map(mapped["target_value_mV_dec"])
        wide[f"{target_name}__interval_low"] = wide["sample_record"].map(
            mapped["target_interval_low_mV_dec"]
        )
        wide[f"{target_name}__interval_high"] = wide["sample_record"].map(
            mapped["target_interval_high_mV_dec"]
        )
        wide[f"{target_name}__eligible"] = wide["sample_record"].isin(
            subset["sample_record"]
        )
    return wide


def _enrich_with_canonical_records(
    registry: pd.DataFrame,
    records_path: Path | None,
    config: TargetBuildConfig,
) -> pd.DataFrame:
    """Attach leakage-control and electrolyte-scope fields from curation."""

    if records_path is None:
        result = registry.copy()
        result.insert(
            0,
            "record_id",
            [f"R{int(value):03d}" for value in result["sample_record"]],
        )
        result.insert(
            2,
            "cv_group_id",
            [f"unresolved_record_{int(value):03d}" for value in result["sample_record"]],
        )
        result.insert(3, "electrolyte_scope", "unknown")
        return result

    path = Path(records_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical records file: {path}")
    records = pd.read_csv(path)
    _required_columns(
        records,
        ["record_id", "sample_record", "cv_group_id"],
        "canonical records",
    )
    if records["sample_record"].duplicated().any():
        raise ValueError("Canonical records are not unique by sample_record")
    if len(records) != config.expected_record_count:
        raise ValueError(
            "Canonical record parity failed: "
            f"expected {config.expected_record_count} rows; found {len(records)}"
        )
    if "electrolyte_scope" not in records.columns:
        if "electrolyte_identity" not in records.columns:
            records["electrolyte_scope"] = "unknown"
        else:
            identity = records["electrolyte_identity"].fillna("").astype(str).str.upper()
            records["electrolyte_scope"] = np.where(
                identity.isin(["KOH", "NAOH"]), "alkaline_primary", "all_electrolyte_only"
            )
    metadata = records[
        ["record_id", "sample_record", "cv_group_id", "electrolyte_scope"]
    ].copy()
    result = registry.merge(metadata, on="sample_record", how="left", validate="many_to_one")
    if result[["record_id", "cv_group_id", "electrolyte_scope"]].isna().any().any():
        missing = result.loc[result["record_id"].isna(), "sample_record"].unique().tolist()
        raise ValueError(f"Target curves could not be mapped to canonical records: {missing}")
    front = ["record_id", "sample_record", "cv_group_id", "electrolyte_scope"]
    return result[front + [column for column in result.columns if column not in front]]


def _legacy_grid_parity(
    exact: pd.DataFrame,
    legacy_grid: pd.DataFrame,
    fixed_current: float,
) -> dict[str, Any]:
    currents = np.sort(legacy_grid["current_density_mA_cm2"].dropna().unique())
    nearest = float(currents[np.argmin(np.abs(np.log10(currents) - np.log10(fixed_current)))])
    legacy = legacy_grid.loc[
        np.isclose(legacy_grid["current_density_mA_cm2"], nearest)
        & legacy_grid["target_eligible"].map(_as_bool),
        ["curve_id", "target_median_kinetic_slope_mV_dec"],
    ]
    current = exact.loc[
        exact["target_eligible"], ["curve_id", "target_value_mV_dec"]
    ]
    paired = current.merge(legacy, on="curve_id", how="inner", validate="one_to_one")
    differences = np.abs(
        paired["target_value_mV_dec"].to_numpy(dtype=float)
        - paired["target_median_kinetic_slope_mV_dec"].to_numpy(dtype=float)
    )
    return {
        "comparison": "exact_2mA_local_kinetic_vs_legacy_nearest_grid_kinetic",
        "legacy_grid_current_mA_cm2": nearest,
        "exact_current_mA_cm2": fixed_current,
        "n_common_eligible_curves": int(len(paired)),
        "median_absolute_difference_mV_dec": (
            float(np.median(differences)) if len(differences) else None
        ),
        "maximum_absolute_difference_mV_dec": (
            float(np.max(differences)) if len(differences) else None
        ),
        "interpretation": (
            "Coordinate-parity diagnostic only: the v2 target uses the complete "
            "model to locate exactly 2.0 mA cm^-2, then evaluates the local kinetic "
            "Tafel slope. The legacy grid used 1.995 mA cm^-2 and is not relabeled."
        ),
    }


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    # Fix the gzip header timestamp so identical source artifacts produce
    # byte-identical compressed draws across rebuilds.
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)


def _collect_posterior_functional_draws(
    bayes_audit: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Collect audit-only exact-2 mA functional draws."""

    columns = [
        "curve_id",
        "sample_record",
        "chain",
        "draw",
        "posterior_draw_index",
        "target_current_density_mA_cm2",
        "relative_potential_at_current_V",
        "local_slope_mV_dec",
        "kinetic_slope_mV_dec",
        "apparent_slope_mV_dec",
        "evidence_id",
        "uncertainty_source",
    ]
    exact_curves = set(
        registry.loc[
            registry["target_name"].eq(EXACT_FIXED_CURRENT_TARGET)
            & registry["target_eligible"].map(_as_bool),
            "curve_id",
        ].astype(str)
    )
    frames: list[pd.DataFrame] = []
    for item in bayes_audit.to_dict("records"):
        curve_id = str(item.get("curve_id", ""))
        if curve_id not in exact_curves or not _as_bool(
            item.get("formal_nuts_evidence_present")
        ):
            continue
        raw_path = str(item.get("functional_draws_path", "")).strip()
        if not raw_path:
            continue
        frame = pd.read_csv(raw_path)
        frame = frame.loc[frame["eligible"].map(_as_bool)].copy()
        if frame.empty:
            continue
        frame["uncertainty_source"] = (
            "selected_model_chain_draw_synchronous_functional_posterior"
        )
        frames.append(frame[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["sample_record", "posterior_draw_index"]).any():
        raise ValueError(
            "Posterior functional targets are not unique by record/draw; "
            "resolve multi-curve sample selection before uncertainty scoring"
        )
    return result.sort_values(
        ["posterior_draw_index", "sample_record", "curve_id"]
    ).reset_index(drop=True)


def _collect_posterior_slope_draws(
    bayes_audit: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Collect the registered low-potential Bayesian slope posterior draws."""

    columns = [
        "curve_id",
        "sample_record",
        "model_name",
        "chain",
        "draw",
        "posterior_draw_index",
        "target_name",
        "primary_slope_mV_dec",
        "slope_low_mV_dec",
        "slope_high_mV_dec",
        "change_potential_relative_V",
        "branch_semantics",
        "posterior_source_variable",
        "evidence_id",
    ]
    primary_curves = set(
        registry.loc[
            registry["target_name"].eq(PRIMARY_TARGET)
            & registry["target_eligible"].map(_as_bool),
            "curve_id",
        ].astype(str)
    )
    frames: list[pd.DataFrame] = []
    for item in bayes_audit.to_dict("records"):
        curve_id = str(item.get("curve_id", ""))
        if curve_id not in primary_curves or not _as_bool(
            item.get("formal_nuts_evidence_present")
        ):
            continue
        raw_path = str(item.get("slope_draws_path", "")).strip()
        if not raw_path:
            continue
        frame = pd.read_csv(raw_path)
        frame = frame.loc[frame["eligible"].map(_as_bool)].copy()
        if frame.empty:
            continue
        frames.append(frame[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["sample_record", "posterior_draw_index"]).any():
        raise ValueError(
            "Posterior slope targets are not unique by record/draw; "
            "resolve multi-curve sample selection before uncertainty scoring"
        )
    return result.sort_values(
        ["posterior_draw_index", "sample_record", "curve_id"]
    ).reset_index(drop=True)


def build_target_registry(
    *,
    output_dir: Path,
    records_path: Path | None = None,
    refit_run: Path | None = None,
    bayesian_table: Path | None = None,
    bayesian_manifest: Path | None = None,
    current_conditioned_dir: Path | None = None,
    bayesian_batch_dir: Path | None = None,
    ppc_review_overlay: Path | None = None,
    repo_root: Path | None = None,
    paths: TargetSourcePaths | None = None,
    config: TargetBuildConfig | None = None,
) -> dict[str, Path]:
    """Build all pre-registered continuous targets and their audit artifacts."""

    if config is None:
        raise ValueError(
            "TargetBuildConfig is required; dataset parity expectations must come "
            "from the resolved study configuration"
        )
    config.validate()
    if paths is None:
        explicit = [refit_run, bayesian_table, current_conditioned_dir]
        if all(value is not None for value in explicit):
            paths = TargetSourcePaths.from_explicit(
                refit_run=Path(refit_run),
                bayesian_table=Path(bayesian_table),
                bayesian_manifest=(
                    Path(bayesian_manifest) if bayesian_manifest is not None else None
                ),
                current_conditioned_dir=Path(current_conditioned_dir),
                bayesian_batch_dir=(
                    Path(bayesian_batch_dir) if bayesian_batch_dir is not None else None
                ),
                ppc_review_overlay=(
                    Path(ppc_review_overlay) if ppc_review_overlay is not None else None
                ),
            )
        elif repo_root is not None:
            paths = TargetSourcePaths.from_repo_root(repo_root)
        else:
            raise ValueError(
                "Provide either paths, repo_root, or the explicit refit_run/"
                "bayesian_table/current_conditioned_dir contract"
            )
    frames = _load_inputs(paths)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exact, fixed_draws = _build_exact_fixed_current(frames, config)
    bayes_audit, queue, bayes_evidence = _bayesian_coverage_audit(
        frames["fit_summary"], paths
    )
    registry = _make_target_registry(frames, exact, bayes_audit, config)
    registry = _enrich_with_canonical_records(registry, records_path, config)
    wide = _targets_wide(registry)

    counts = (
        registry.loc[registry["target_eligible"]]
        .groupby("target_name")
        .size()
        .to_dict()
    )
    structural_counts = (
        registry.loc[registry["target_structurally_eligible"]]
        .groupby("target_name")
        .size()
        .to_dict()
    )
    expected = {
        LOW_BRANCH_TARGET: config.expected_primary_eligible,
        SINGLE_REGIME_TARGET: config.expected_single_regime_eligible,
    }
    observed_ml_structural = {
        name: int(structural_counts.get(name, 0)) for name in expected
    }
    if observed_ml_structural != expected:
        raise ValueError(
            "Structural target eligibility parity failed: "
            f"observed={observed_ml_structural}, expected={expected}"
        )
    if registry.loc[registry["target_eligible"], "extrapolated"].any():
        raise ValueError("Eligible target registry contains an extrapolated value")
    exact_eligible = registry.loc[
        registry["target_name"].eq(EXACT_FIXED_CURRENT_TARGET)
        & registry["target_eligible"]
    ]
    if len(exact_eligible) and not exact_eligible["observed_support"].all():
        raise ValueError("Eligible fixed-current target lacks observed support")
    if len(exact_eligible) and not np.allclose(
        exact_eligible["target_current_density_mA_cm2"].to_numpy(dtype=float),
        config.fixed_current_density_mA_cm2,
    ):
        raise ValueError("Fixed-current registry contains a non-2.0 target coordinate")

    output_paths = {
        "target_registry": output_dir / "target_registry.csv",
        "targets_wide": output_dir / "targets_wide.csv",
        "fixed_current_draws": output_dir / "target_draws_at_2mA.csv.gz",
        "posterior_functional_draws": (
            output_dir / "posterior_functional_draws_at_2mA.csv.gz"
        ),
        "posterior_slope_draws": output_dir / "posterior_slope_draws.csv.gz",
        "bayesian_audit": output_dir / "bayesian_audit.csv",
        "bayesian_pending_queue": output_dir / "bayesian_pending_queue.csv",
        "bayesian_artifact_inventory": output_dir / "bayesian_artifact_inventory.csv",
        "target_qa": output_dir / "target_qa.json",
        "artifact_manifest": output_dir / "target_artifact_manifest.csv",
    }
    registry.to_csv(output_paths["target_registry"], index=False)
    wide.to_csv(output_paths["targets_wide"], index=False)
    _write_gzip_csv(fixed_draws, output_paths["fixed_current_draws"])
    posterior_functional_draws = _collect_posterior_functional_draws(
        bayes_audit, registry
    )
    _write_gzip_csv(
        posterior_functional_draws,
        output_paths["posterior_functional_draws"],
    )
    posterior_slope_draws = _collect_posterior_slope_draws(
        bayes_audit, registry
    )
    _write_gzip_csv(posterior_slope_draws, output_paths["posterior_slope_draws"])
    bayes_audit.to_csv(output_paths["bayesian_audit"], index=False)
    queue.to_csv(output_paths["bayesian_pending_queue"], index=False)
    write_evidence_inventory(
        output_paths["bayesian_artifact_inventory"], bayes_evidence
    )

    parity = _legacy_grid_parity(
        exact,
        frames["legacy_current_grid_summary"],
        config.fixed_current_density_mA_cm2,
    )
    eligible_registry = registry.loc[registry["target_eligible"]].copy()
    structural_registry = registry.loc[
        registry["target_structurally_eligible"]
        & registry["target_name"].isin(
            {PRIMARY_TARGET, SINGLE_REGIME_TARGET}
        )
    ].copy()
    posterior_supported_counts = (
        eligible_registry.loc[eligible_registry["posterior_target_supported"]]
        .groupby("target_name")
        .size()
        .to_dict()
    )
    formal_target_counts = (
        eligible_registry.loc[eligible_registry["formal_target_approved"]]
        .groupby("target_name")
        .size()
        .to_dict()
    )
    all_registered_targets_posterior_supported = bool(
        len(structural_registry)
        and structural_registry["posterior_target_supported"].all()
    )
    all_registered_targets_formally_approved = bool(
        len(structural_registry)
        and structural_registry["formal_target_approved"].all()
    )
    formal_evaluation_ready = bool(
        int(bayes_audit["formal_nuts_ppc_approved"].sum())
        == int(len(bayes_audit))
        and all_registered_targets_posterior_supported
        and all_registered_targets_formally_approved
    )
    blockers: list[str] = []
    if int(bayes_audit["formal_nuts_ppc_approved"].sum()) != int(len(bayes_audit)):
        blockers.append("production_selected_model_nuts_and_hash_bound_named_ppc_incomplete")
    if not all_registered_targets_posterior_supported:
        blockers.append(
            "posterior_target_coverage_incomplete; run current selected-model NUTS "
            "and its hash-bound chain/draw branch-slope extraction for every eligible ML target"
        )
    if not all_registered_targets_formally_approved:
        blockers.append("one_or_more_target_rows_lack_formal_bound_ppc_approval")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_with_formal_bayesian_review_pending",
        "config": asdict(config),
        "source_artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.file_items().items()
        },
        "target_eligible_counts": counts,
        "target_structurally_eligible_counts": structural_counts,
        "sample_candidate_gate_failures": int(
            (~registry["sample_candidate_gate_passed"]).sum()
        ),
        "n_curves_in_registry": int(frames["fit_summary"]["curve_id"].nunique()),
        "n_records_in_registry": int(frames["fit_summary"]["sample_record"].nunique()),
        "n_fit_eligible_curves": int(frames["fit_summary"]["fit_eligible"].map(_as_bool).sum()),
        "n_existing_nuts_diagnostics": int(bayes_audit["nuts_diagnostics_present"].sum()),
        "n_current_production_nuts_evidence": int(
            bayes_audit["formal_nuts_evidence_present"].sum()
        ),
        "n_sampler_diagnostics_pass": int(bayes_audit["sampler_diagnostics_pass"].sum()),
        "n_ppc_approved_declared": int(bayes_audit["ppc_approved_declared"].sum()),
        "n_named_ppc_review_complete": int(bayes_audit["named_ppc_review_complete"].sum()),
        "n_formal_nuts_ppc_approved": int(bayes_audit["formal_nuts_ppc_approved"].sum()),
        "n_bayesian_queue_rows": int(len(queue)),
        "posterior_supported_target_counts": posterior_supported_counts,
        "formal_approved_target_counts": formal_target_counts,
        "all_registered_targets_posterior_supported": (
            all_registered_targets_posterior_supported
        ),
        "all_registered_targets_formally_approved": (
            all_registered_targets_formally_approved
        ),
        "formal_evaluation_ready": formal_evaluation_ready,
        "formal_evaluation_blockers": blockers,
        "fixed_current_draw_rows": int(len(fixed_draws)),
        "posterior_functional_draw_rows": int(len(posterior_functional_draws)),
        "posterior_functional_draw_curves": int(
            posterior_functional_draws["curve_id"].nunique()
            if len(posterior_functional_draws)
            else 0
        ),
        "posterior_slope_draw_rows": int(len(posterior_slope_draws)),
        "posterior_slope_draw_curves": int(
            posterior_slope_draws["curve_id"].nunique()
            if len(posterior_slope_draws)
            else 0
        ),
        "n_bayesian_evidence_sets_discovered": int(len(bayes_evidence)),
        "n_bayesian_sidecar_artifacts_hashed": int(
            sum(
                bool(row.get(column))
                for row in bayes_evidence.to_dict("records")
                for column in (
                    "diagnostics_sha256",
                    "netcdf_sha256",
                    "posterior_summary_sha256",
                    "ppc_plot_sha256",
                    "slope_draws_sha256",
                    "functional_draws_sha256",
                )
            )
        ),
        "primary_target_policy": {
            "target_name": PRIMARY_TARGET,
            "single_regime_rule": "posterior slope_mV_dec median",
            "two_regime_rule": "posterior slope_low_mV_dec median",
            "two_regime_branch_selection": "model-coordinate low-potential branch",
            "numeric_minimum_selection_forbidden": True,
            "exact_2mA_role": "audit_only_excluded_from_ml",
        },
        "fixed_current_diagnostic_policy": {
            "exact_coordinate_mA_cm2": config.fixed_current_density_mA_cm2,
            "calculation": (
                "bracketed complete-selected-model inversion in log10 current, "
                "followed by local kinetic dE/dlog10(j_kinetic)"
            ),
            "complete_model_apparent_slope_role": (
                "transport/background diagnostic only; not the Tafel target"
            ),
            "interpolation_from_legacy_grid": False,
            "no_extrapolation": True,
            "uncertainty": "moving-block bootstrap",
        },
        "draw_synchronization": {
            "fixed_current_key": "curve_id + bootstrap_replicate",
            "sampling_rule": (
                "Sample one whole bootstrap replicate per curve; do not sample "
                "summary interval endpoints or current coordinates independently."
            ),
            "primary_branch_slope_draw_export": (
                "materialized chain/draw-synchronously from every selected-model NetCDF"
            ),
            "formal_fixed_current_functional_contract": (
                "Every selected-model NetCDF is evaluated chain/draw-synchronously at "
                "exactly 2.0 mA cm^-2. The complete fitted model is bracket-inverted "
                "within observed potential/current support, then the local latent-kinetic "
                "dE/dlog10(j_kinetic) is exported without extrapolation. The complete-model "
                "apparent derivative is retained only as a diagnostic, and the draw artifact "
                "is hash-bound to the NetCDF, diagnostics, and named PPC review."
            ),
        },
        "low_branch_policy": {
            "selection": "low-potential/low-current model branch",
            "numeric_minimum_selection_forbidden": True,
        },
        "parity_checks": [parity],
        "scope_caveats": [
            "The archived 58-row Bayesian table is a historical sensitivity-value fallback only. Its diagnostics and PPC flags are not bound to the current workflow, never confer formal approval, and are replaced whenever current workflow-bound posterior evidence exists.",
            "Potential-dependent sensitivity targets use selected-model moving-block bootstrap intervals until current two-regime NUTS plus named PPC review is complete.",
            "Bayesian sampling remains curve-level, while sample-level target eligibility requires a unique resolved curve candidate; unresolved variants such as sr042_v1/sr042_v2 remain in the Bayesian queue but cannot enter modeling targets.",
            "No iR correction is inferred or applied; these targets are not described as iR-free intrinsic slopes.",
            "Current-density units are based on the documented user-confirmed interpretation because the source workbook headers were not unit-explicit.",
        ],
    }
    output_paths["target_qa"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    artifact_rows: list[dict[str, Any]] = []
    for name, path in paths.file_items().items():
        artifact_rows.append(
            {
                "role": "source",
                "artifact_name": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    sidecar_paths: set[Path] = set()
    for item in bayes_evidence.to_dict("records"):
        for column in (
            "diagnostics_path",
            "netcdf_path",
            "posterior_summary_path",
            "ppc_plot_path",
            "slope_draws_path",
            "functional_draws_path",
        ):
            raw_path = str(item.get(column, "")).strip()
            if raw_path:
                sidecar_paths.add(Path(raw_path).resolve())
    for path in sorted(sidecar_paths):
        if path.is_file():
            artifact_rows.append(
                {
                    "role": "bayesian_sidecar",
                    "artifact_name": path.name,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    for name, path in output_paths.items():
        if name == "artifact_manifest" or not path.exists():
            continue
        artifact_rows.append(
            {
                "role": "output",
                "artifact_name": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(artifact_rows).to_csv(output_paths["artifact_manifest"], index=False)
    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ink–Tafel v2 continuous targets")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-current", type=float, default=2.0)
    parser.add_argument("--inverse-grid-points", type=int, default=2049)
    parser.add_argument("--expected-curves", type=int, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--expected-primary-eligible", type=int, required=True)
    parser.add_argument("--expected-low-branch-eligible", type=int, required=True)
    parser.add_argument("--expected-single-regime-eligible", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TargetBuildConfig(
        expected_curve_count=args.expected_curves,
        expected_record_count=args.expected_records,
        expected_primary_eligible=args.expected_primary_eligible,
        expected_low_branch_eligible=args.expected_low_branch_eligible,
        expected_single_regime_eligible=args.expected_single_regime_eligible,
        fixed_current_density_mA_cm2=args.fixed_current,
        inverse_grid_points=args.inverse_grid_points,
    )
    outputs = build_target_registry(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        config=config,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
