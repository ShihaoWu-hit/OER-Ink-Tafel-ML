from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import qmc


LOG10 = np.log(10.0)


@dataclass(frozen=True)
class ModelSpec:
    """Structural description of one member of the continuous Tafel family.

    ``model_name`` remains a stable artifact key.  Scientific reporting should
    use the structural fields instead of presenting every implementation
    variant as an unrelated model.
    """

    model_name: str
    kinetic_structure: str
    has_limiting_current: bool
    has_low_current_floor: bool
    role: str
    display_name: str
    # Stable salts prevent candidate-list changes from changing the optimizer
    # starts of retained technical variants.
    analysis_seed_offset: int
    cv_seed_offset: int

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.has_low_current_floor:
            names.append("B")
        names.append("a")
        if self.kinetic_structure == "constant":
            names.append("log_m")
        else:
            names.extend(("log_m1", "log_m2", "c"))
        if self.has_limiting_current:
            names.append("L")
        names.append("log_sigma")
        return tuple(names)


PRIMARY_MODEL_SPECS = {
    "linear": ModelSpec(
        model_name="linear",
        kinetic_structure="constant",
        has_limiting_current=False,
        has_low_current_floor=False,
        role="primary",
        display_name="constant slope; no plateau term",
        analysis_seed_offset=0,
        cv_seed_offset=11,
    ),
    "paper_single_plateau": ModelSpec(
        model_name="paper_single_plateau",
        kinetic_structure="constant",
        has_limiting_current=True,
        has_low_current_floor=False,
        role="primary",
        display_name="constant slope + limiting-current plateau term",
        analysis_seed_offset=1,
        cv_seed_offset=23,
    ),
    "two_regime_plateau": ModelSpec(
        model_name="two_regime_plateau",
        kinetic_structure="potential_dependent",
        has_limiting_current=True,
        has_low_current_floor=False,
        role="primary",
        display_name="potential-dependent slope + limiting-current plateau term",
        analysis_seed_offset=3,
        cv_seed_offset=37,
    ),
}

# These legacy artifact keys remain readable and testable, but are no longer
# primary candidates.  Without an independently measured blank, their
# constant additive term is called a low-current floor rather than a
# physically identified background current.
LEGACY_LOW_CURRENT_FLOOR_SENSITIVITY_SPECS = {
    "single_background_plateau": ModelSpec(
        model_name="single_background_plateau",
        kinetic_structure="constant",
        has_limiting_current=True,
        has_low_current_floor=True,
        role="low_current_floor_sensitivity",
        display_name="constant slope + plateau + low-current floor (sensitivity)",
        analysis_seed_offset=2,
        cv_seed_offset=31,
    ),
    "two_regime_background_plateau": ModelSpec(
        model_name="two_regime_background_plateau",
        kinetic_structure="potential_dependent",
        has_limiting_current=True,
        has_low_current_floor=True,
        role="low_current_floor_sensitivity",
        display_name=(
            "potential-dependent slope + plateau + low-current floor "
            "(sensitivity)"
        ),
        analysis_seed_offset=4,
        cv_seed_offset=43,
    ),
}

MODEL_SPECS = {
    **PRIMARY_MODEL_SPECS,
    **LEGACY_LOW_CURRENT_FLOOR_SENSITIVITY_SPECS,
}
if any(key != spec.model_name for key, spec in MODEL_SPECS.items()):
    raise RuntimeError("Model registry keys must match ModelSpec.model_name")
PRIMARY_MODEL_NAMES = tuple(PRIMARY_MODEL_SPECS)
LOW_CURRENT_FLOOR_SENSITIVITY_MODEL_NAMES = tuple(
    LEGACY_LOW_CURRENT_FLOOR_SENSITIVITY_SPECS
)
MODEL_PARAMETER_NAMES = {
    name: spec.parameter_names for name, spec in MODEL_SPECS.items()
}


def model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc


def model_structure_fields(model_name: str) -> dict[str, str]:
    spec = model_spec(model_name)
    return {
        "model_family": "continuous_tafel",
        "kinetic_structure": spec.kinetic_structure,
        "observation_parameterization": (
            "limiting_current_plateau"
            if spec.has_limiting_current
            else "none"
        ),
        "low_current_floor_policy": (
            "sensitivity_only"
            if spec.has_low_current_floor
            else "excluded_from_primary"
        ),
        "model_role": spec.role,
    }


@dataclass(frozen=True)
class FitResult:
    model_name: str
    parameters: dict[str, float]
    nll: float
    bic: float
    mae: float
    success: bool
    optimizer_message: str
    boundary_hits: tuple[str, ...]
    predictions: np.ndarray
    n_observations: int
    transition_width_V: float


def harmonic_plateau_log10(kinetic_log10: np.ndarray, plateau_log10: float) -> np.ndarray:
    kinetic = np.asarray(kinetic_log10, dtype=float)
    return kinetic + plateau_log10 - np.logaddexp(
        kinetic * LOG10, plateau_log10 * LOG10
    ) / LOG10


def kinetic_log10(
    model_name: str,
    x: np.ndarray,
    parameters: dict[str, float],
    transition_width_V: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    spec = model_spec(model_name)
    if spec.kinetic_structure == "constant":
        return parameters["a"] + np.exp(parameters["log_m"]) * x
    if spec.kinetic_structure == "potential_dependent":
        m1 = np.exp(parameters["log_m1"])
        m2 = np.exp(parameters["log_m2"])
        c = parameters["c"]
        tau = transition_width_V
        integrated_transition = tau * (
            np.logaddexp(0.0, (x - c) / tau)
            - np.logaddexp(0.0, -c / tau)
        )
        return parameters["a"] + m1 * x + (m2 - m1) * integrated_transition
    raise ValueError(f"Unknown kinetic structure: {spec.kinetic_structure}")


def predict_log10(
    model_name: str,
    x: np.ndarray,
    parameters: dict[str, float],
    transition_width_V: float,
) -> np.ndarray:
    spec = model_spec(model_name)
    kinetic = kinetic_log10(model_name, x, parameters, transition_width_V)
    if not spec.has_limiting_current:
        return kinetic
    signal = harmonic_plateau_log10(kinetic, parameters["L"])
    if spec.has_low_current_floor:
        return np.logaddexp(parameters["B"] * LOG10, signal * LOG10) / LOG10
    return signal


def kinetic_slope_dec_per_V(
    model_name: str,
    x: np.ndarray,
    parameters: dict[str, float],
    transition_width_V: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    spec = model_spec(model_name)
    if spec.kinetic_structure == "constant":
        return np.full_like(x, np.exp(parameters["log_m"]), dtype=float)
    if spec.kinetic_structure == "potential_dependent":
        m1 = np.exp(parameters["log_m1"])
        m2 = np.exp(parameters["log_m2"])
        return m1 + (m2 - m1) * expit(
            (x - parameters["c"]) / transition_width_V
        )
    raise ValueError(f"Unknown kinetic structure: {spec.kinetic_structure}")


def local_slopes_mV_dec(
    model_name: str,
    x: np.ndarray,
    parameters: dict[str, float],
    transition_width_V: float,
) -> tuple[np.ndarray, np.ndarray]:
    spec = model_spec(model_name)
    kinetic_slope = kinetic_slope_dec_per_V(
        model_name, x, parameters, transition_width_V
    )
    kinetic_b = 1000.0 / kinetic_slope
    if not spec.has_limiting_current:
        return kinetic_b, kinetic_b.copy()
    predicted = predict_log10(model_name, x, parameters, transition_width_V)
    if spec.has_low_current_floor:
        kinetic = kinetic_log10(model_name, x, parameters, transition_width_V)
        signal = harmonic_plateau_log10(kinetic, parameters["L"])
        plateau_fraction = np.clip(
            10.0 ** (signal - parameters["L"]), 0.0, 1.0 - 1e-12
        )
        signal_fraction = np.clip(10.0 ** (signal - predicted), 0.0, 1.0)
        apparent_slope = kinetic_slope * (1.0 - plateau_fraction) * signal_fraction
    else:
        plateau_fraction = np.clip(
            10.0 ** (predicted - parameters["L"]), 0.0, 1.0 - 1e-12
        )
        apparent_slope = kinetic_slope * (1.0 - plateau_fraction)
    apparent_b = 1000.0 / np.maximum(apparent_slope, 1e-12)
    return kinetic_b, apparent_b


def _student_t_nll(residual: np.ndarray, sigma: float, nu: float = 4.0) -> float:
    scaled = residual / sigma
    return float(
        np.sum(np.log(sigma) + 0.5 * (nu + 1.0) * np.log1p((scaled * scaled) / nu))
    )


def _initial_guess(model_name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    spec = model_spec(model_name)
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    a = float(beta[0])
    slope = float(np.clip(beta[1], 1.0, 100.0))
    sigma = float(np.clip(np.median(np.abs(y - design @ beta)) * 1.4826, 0.01, 0.3))
    values: list[float] = []
    if spec.has_low_current_floor:
        values.append(float(np.min(y)))
        values.append(float(np.min(y) - 1.0))
    else:
        values.append(a)

    if spec.kinetic_structure == "constant":
        values.append(float(np.log(slope)))
    else:
        split = max(3, len(x) // 3)
        early = np.linalg.lstsq(
            np.column_stack([np.ones(split), x[:split]]), y[:split], rcond=None
        )[0][1]
        late = np.linalg.lstsq(
            np.column_stack([np.ones(split), x[-split:]]), y[-split:], rcond=None
        )[0][1]
        values.extend(
            (
                float(np.log(np.clip(early, 1.0, 100.0))),
                float(np.log(np.clip(late, 1.0, 100.0))),
                float(np.median(x)),
            )
        )
    if spec.has_limiting_current:
        values.append(float(np.max(y) + 0.25))
    values.append(float(np.log(sigma)))
    return np.asarray(values, dtype=float)


def latent_plateau_bounds(y: np.ndarray) -> tuple[float, float]:
    """Keep a fitted plateau close enough to the data to be identifiable.

    A plateau one decade above the largest observation changes the modeled
    current by only about 0.04 log10 units at the top of the
    observed range.  Allowing the former six-decade upper tail therefore let a
    nominal plateau model collapse into an effectively no-plateau model.  In
    potential-dependent fits that created a disconnected posterior mode in
    which the change point and the second slope explained the upper curvature
    while ``L`` was completely uninformed.  The explicit no-plateau candidate
    handles curves for which a limiting current is not observed.

    The lower allowance is retained because Student-t observation noise can
    place individual observations above the latent limiting current.
    """

    values = np.asarray(y, dtype=float)
    return float(np.max(values) - 1.0), float(np.max(values) + 0.5)


def latent_low_current_floor_bounds(y: np.ndarray) -> tuple[float, float]:
    values = np.asarray(y, dtype=float)
    return float(np.min(values) - 6.0), float(np.min(values) + 1.0)


def latent_background_bounds(y: np.ndarray) -> tuple[float, float]:
    """Legacy name retained for readers of existing background-key artifacts."""

    return latent_low_current_floor_bounds(y)


def _bounds(model_name: str, x: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    spec = model_spec(model_name)
    a_bounds = (float(np.min(y) - 6.0), float(np.max(y) + 2.0))
    slope_bounds = (float(np.log(1.0)), float(np.log(100.0)))
    sigma_bounds = (float(np.log(0.005)), float(np.log(0.5)))
    # Observations include Student-t noise, so the latent plateau may sit below
    # the largest observed point. Pinning L to max(y) creates artificial lower-
    # boundary hits and biased slopes whenever the top point is a positive
    # residual. The upper bound keeps the plateau within the observed support's
    # identifiable neighborhood instead of admitting an effectively absent,
    # prior-volume-dominated plateau.
    plateau_bounds = latent_plateau_bounds(y)
    bounds: list[tuple[float, float]] = []
    if spec.has_low_current_floor:
        # This term is retained only for registered sensitivity analyses.  It
        # can sit above the smallest noisy observation; downstream sensitivity
        # diagnostics must not interpret that statistical fit as proof of a
        # physical background current.
        bounds.append(latent_low_current_floor_bounds(y))
        bounds.append((float(np.min(y) - 8.0), float(np.max(y) + 2.0)))
    else:
        bounds.append(a_bounds)

    if spec.kinetic_structure == "constant":
        bounds.append(slope_bounds)
    else:
        c_bounds = (float(np.quantile(x, 0.20)), float(np.quantile(x, 0.80)))
        if c_bounds[1] - c_bounds[0] < 1e-6:
            c_bounds = (float(np.min(x)), float(np.max(x) + 1e-6))
        bounds.extend((slope_bounds, slope_bounds, c_bounds))
    if spec.has_limiting_current:
        bounds.append(plateau_bounds)
    bounds.append(sigma_bounds)
    return bounds


def _vector_to_parameters(model_name: str, vector: Iterable[float]) -> dict[str, float]:
    return dict(zip(MODEL_PARAMETER_NAMES[model_name], map(float, vector), strict=True))


def fit_model(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    transition_width_V: float,
    n_starts: int,
    seed: int,
) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("x and y must contain at least three paired observations")
    bounds = _bounds(model_name, x, y)

    def objective(vector: np.ndarray) -> float:
        parameters = _vector_to_parameters(model_name, vector)
        predicted = predict_log10(model_name, x, parameters, transition_width_V)
        sigma = np.exp(parameters["log_sigma"])
        if not np.all(np.isfinite(predicted)) or not np.isfinite(sigma):
            return 1e30
        return _student_t_nll(y - predicted, sigma)

    initial = np.clip(
        _initial_guess(model_name, x, y),
        np.array([lo for lo, _ in bounds]),
        np.array([hi for _, hi in bounds]),
    )
    starts = [initial]
    if n_starts > 1:
        sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
        unit = sampler.random(n_starts - 1)
        starts.extend(qmc.scale(unit, [b[0] for b in bounds], [b[1] for b in bounds]))

    candidates = []
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        return FitResult(
            model_name=model_name,
            parameters={},
            nll=np.inf,
            bic=np.inf,
            mae=np.inf,
            success=False,
            optimizer_message="no finite optimization result",
            boundary_hits=(),
            predictions=np.full_like(y, np.nan),
            n_observations=len(y),
            transition_width_V=transition_width_V,
        )
    converged = [result for result in candidates if bool(result.success)]
    if not converged:
        best_failed = min(candidates, key=lambda result: result.fun)
        return FitResult(
            model_name=model_name,
            parameters={},
            nll=np.inf,
            bic=np.inf,
            mae=np.inf,
            success=False,
            optimizer_message=f"no converged result; best message: {best_failed.message}",
            boundary_hits=(),
            predictions=np.full_like(y, np.nan),
            n_observations=len(y),
            transition_width_V=transition_width_V,
        )
    best = min(converged, key=lambda result: result.fun)
    parameters = _vector_to_parameters(model_name, best.x)
    predictions = predict_log10(model_name, x, parameters, transition_width_V)
    nll = float(best.fun)
    k = len(best.x)
    bic = float(2.0 * nll + k * np.log(len(y)))
    mae = float(np.mean(np.abs(y - predictions)))
    boundary_hits: list[str] = []
    for name, value, (low, high) in zip(
        MODEL_PARAMETER_NAMES[model_name], best.x, bounds, strict=True
    ):
        tolerance = max(1e-6, 0.01 * (high - low))
        if value - low <= tolerance:
            boundary_hits.append(f"{name}_lower")
        if high - value <= tolerance:
            boundary_hits.append(f"{name}_upper")
    return FitResult(
        model_name=model_name,
        parameters=parameters,
        nll=nll,
        bic=bic,
        mae=mae,
        success=bool(best.success),
        optimizer_message=str(best.message),
        boundary_hits=tuple(boundary_hits),
        predictions=predictions,
        n_observations=len(y),
        transition_width_V=transition_width_V,
    )


def fit_paper_exact(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_starts: int,
    seed: int,
) -> dict[str, Any]:
    """Raw-point Gaussian/SSE point fit matching the paper equation.

    This deliberately uses every positive digitized point and is only a legacy
    comparison. It is excluded from formal model selection.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    model_name = "paper_single_plateau"
    bounds4 = _bounds(model_name, x, y)
    bounds = bounds4[:-1]
    initial = _initial_guess(model_name, x, y)[:-1]

    def objective(vector: np.ndarray) -> float:
        parameters = dict(zip(("a", "log_m", "L"), map(float, vector), strict=True))
        prediction = predict_log10(model_name, x, parameters, 0.01)
        return float(np.sum(np.square(y - prediction)))

    starts = [initial]
    if n_starts > 1:
        sampler = qmc.LatinHypercube(d=3, seed=seed)
        starts.extend(
            qmc.scale(
                sampler.random(n_starts - 1),
                [b[0] for b in bounds],
                [b[1] for b in bounds],
            )
        )
    candidates = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds) for start in starts
    ]
    finite = [
        result
        for result in candidates
        if np.isfinite(result.fun) and bool(result.success)
    ]
    if not finite:
        return {"success": False, "slope_mV_dec": np.nan, "sse": np.nan}
    best = min(finite, key=lambda result: result.fun)
    slope = 1000.0 / np.exp(best.x[1])
    return {
        "success": bool(best.success),
        "slope_mV_dec": float(slope),
        "sse": float(best.fun),
        "a": float(best.x[0]),
        "log_m": float(best.x[1]),
        "L": float(best.x[2]),
        "optimizer_message": str(best.message),
    }


def blocked_cross_validation(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    transition_width_V: float,
    n_starts: int,
    seed: int,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_folds = 5 if len(x) >= 20 else 3
    folds = np.array_split(np.arange(len(x)), n_folds)
    fold_mae: list[float] = []
    fold_nll: list[float] = []
    successful = 0
    model_seed = model_spec(model_name).cv_seed_offset
    for fold_index, test_indices in enumerate(folds):
        train_mask = np.ones(len(x), dtype=bool)
        train_mask[test_indices] = False
        try:
            fit = fit_model(
                model_name,
                x[train_mask],
                y[train_mask],
                transition_width_V=transition_width_V,
                n_starts=n_starts,
                seed=seed + model_seed + fold_index,
            )
        except Exception:
            continue
        if (
            not fit.success
            or not fit.parameters
            or not np.all(np.isfinite(list(fit.parameters.values())))
        ):
            continue
        prediction = predict_log10(
            model_name, x[test_indices], fit.parameters, transition_width_V
        )
        sigma = np.exp(fit.parameters["log_sigma"])
        fold_mae.append(float(np.mean(np.abs(y[test_indices] - prediction))))
        fold_nll.append(_student_t_nll(y[test_indices] - prediction, sigma) / len(test_indices))
        successful += 1
    return {
        "cv_mae": float(np.mean(fold_mae)) if fold_mae else np.inf,
        "cv_nll_per_point": float(np.mean(fold_nll)) if fold_nll else np.inf,
        "cv_folds_successful": successful,
        "cv_folds_requested": n_folds,
    }


def moving_block_bootstrap(
    fit: FitResult,
    x: np.ndarray,
    y: np.ndarray,
    *,
    repetitions: int,
    block_length: int,
    n_starts: int,
    seed: int,
) -> pd.DataFrame:
    if repetitions <= 0 or not fit.success or not fit.parameters:
        return pd.DataFrame()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    residual = y - fit.predictions
    residual = residual - np.mean(residual)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    n = len(y)
    block_length = int(np.clip(block_length, 1, n))
    for replicate in range(repetitions):
        sampled: list[float] = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            indices = (start + np.arange(block_length)) % n
            sampled.extend(residual[indices].tolist())
        y_star = fit.predictions + np.asarray(sampled[:n])
        try:
            refit = fit_model(
                fit.model_name,
                x,
                y_star,
                transition_width_V=fit.transition_width_V,
                n_starts=n_starts,
                seed=seed + replicate + 1,
            )
        except Exception as exc:
            rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "success": False,
                    "error": type(exc).__name__,
                }
            )
            continue
        row: dict[str, Any] = {
            "bootstrap_replicate": replicate,
            "success": bool(
                refit.success and refit.parameters and np.isfinite(refit.nll)
            ),
            "nll": refit.nll,
            "boundary_hits": ";".join(refit.boundary_hits),
            "optimizer_message": refit.optimizer_message,
        }
        row.update(refit.parameters)
        if refit.parameters:
            if model_spec(fit.model_name).kinetic_structure == "constant":
                row["slope_mV_dec"] = 1000.0 / np.exp(refit.parameters["log_m"])
            else:
                row["slope_low_mV_dec"] = 1000.0 / np.exp(
                    refit.parameters["log_m1"]
                )
                row["slope_high_mV_dec"] = 1000.0 / np.exp(
                    refit.parameters["log_m2"]
                )
                row["change_potential_relative_V"] = refit.parameters["c"]
        rows.append(row)
    return pd.DataFrame(rows)


def classify_tafel_regime(slope_mV_dec: float) -> str:
    if not np.isfinite(slope_mV_dec):
        return "unknown"
    if slope_mV_dec > 122.0:
        return "high_gt120"
    anchors = np.array([40.0, 60.0, 120.0])
    closest = int(np.argmin(np.abs(anchors - slope_mV_dec)))
    return "low_40_60" if closest in (0, 1) else "120_like"


def regime_probabilities(slopes: np.ndarray) -> dict[str, float]:
    values = np.asarray(slopes, dtype=float)
    values = values[np.isfinite(values)]
    classes = ("low_40_60", "120_like", "high_gt120")
    if not len(values):
        return {f"q_{name}": np.nan for name in classes}
    labels = np.array([classify_tafel_regime(value) for value in values])
    return {f"q_{name}": float(np.mean(labels == name)) for name in classes}
