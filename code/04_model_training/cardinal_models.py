"""Soft-label classifiers for posterior-aware cardinal Tafel regions.

The public estimators in this module consume one probability vector per
experimental record.  Posterior draws are therefore not expanded into
pseudo-independent observations.  Logistic regression uses a
probability-weighted class expansion that preserves one unit of total weight
per record; Extra Trees fits the probability vector directly as a multi-output
target so record-level leaf constraints remain valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cardinal_feature_engineering import CardinalFeatureBuilder
from .models import GowerKNeighborsRegressor


CARDINAL_CLASSES: tuple[str, ...] = (
    "low_30_40_like",
    "60_like",
    "120_like",
    "off_cardinal",
)
CLASSIC_CARDINAL_CLASSES: tuple[str, ...] = CARDINAL_CLASSES[:-1]
CARDINAL_GATE_CLASSES: tuple[str, ...] = ("cardinal", "off_cardinal")
CARDINAL_MODEL_NAMES: tuple[str, ...] = (
    "class_prior",
    "soft_ordinal_logistic",
    "soft_multinomial_logistic",
    "soft_gower_knn",
    "soft_extra_trees",
)


@dataclass(frozen=True)
class CardinalModelSpec:
    """A two-stage estimator and its strictly inner-CV parameter grid."""

    name: str
    estimator: BaseEstimator
    param_grid: Mapping[str, Sequence[Any]]
    complexity_rank: int


def _n_rows(X: Any) -> int:
    if not hasattr(X, "shape") or len(X.shape) != 2:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        return int(array.shape[0])
    return int(X.shape[0])


def _as_frame(X: Any, expected_columns: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy(deep=False)
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        columns = (
            list(expected_columns)
            if expected_columns is not None
            else [f"x{index}" for index in range(array.shape[1])]
        )
        if len(columns) != array.shape[1]:
            raise ValueError(
                f"X has {array.shape[1]} columns but {len(columns)} feature names were supplied"
            )
        frame = pd.DataFrame(array, columns=columns)
    if expected_columns is not None:
        missing = [column for column in expected_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Prediction table is missing fitted features: {missing}")
        frame = frame.loc[:, list(expected_columns)]
    return frame


def _slice_rows(X: Any, selector: np.ndarray) -> Any:
    if isinstance(X, pd.DataFrame):
        return X.iloc[np.flatnonzero(selector)]
    return np.asarray(X)[selector]


def _class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(map(str, class_names))
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_names must contain at least two unique names")
    return names


def _validate_soft_targets(
    y: Any,
    *,
    n_rows: int,
    class_names: Sequence[str],
) -> np.ndarray:
    names = _class_names(class_names)
    if isinstance(y, pd.DataFrame):
        plain_columns = list(names)
        posterior_columns = [f"q_{name}" for name in names]
        if all(column in y.columns for column in plain_columns):
            selected_columns = plain_columns
        elif all(column in y.columns for column in posterior_columns):
            selected_columns = posterior_columns
        else:
            missing = [name for name in names if name not in y.columns]
            raise ValueError(
                "Soft-target table is missing class columns; expected either "
                f"{plain_columns} or {posterior_columns}, missing plain classes={missing}"
            )
        values = y.loc[:, selected_columns].to_numpy(dtype=float)
    else:
        values = np.asarray(y, dtype=float)
    if values.ndim != 2 or values.shape != (n_rows, len(names)):
        raise ValueError(
            "Soft targets must have shape "
            f"({n_rows}, {len(names)}); received {tuple(values.shape)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Soft targets contain NaN or infinite values")
    if np.any(values < -1e-12):
        raise ValueError("Soft-target probabilities cannot be negative")
    values = np.maximum(values, 0.0)
    totals = values.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every soft-target row must have positive probability mass")
    return values / totals[:, None]


def _validate_sample_weight(sample_weight: Any, n_rows: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_rows, dtype=float)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.shape != (n_rows,):
        raise ValueError(
            f"sample_weight must have shape ({n_rows},); received {tuple(weights.shape)}"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and nonnegative")
    if not np.any(weights > 0):
        raise ValueError("At least one sample weight must be positive")
    return weights


def _normalize_probabilities(values: Any, n_classes: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != n_classes:
        raise ValueError(
            f"Expected a probability matrix with {n_classes} columns; "
            f"received {tuple(probabilities.shape)}"
        )
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.maximum(probabilities, 0.0)
    totals = probabilities.sum(axis=1)
    empty = totals <= 0
    if empty.any():
        probabilities[empty, :] = 1.0 / n_classes
        totals = probabilities.sum(axis=1)
    return probabilities / totals[:, None]


def _align_probabilities(
    probabilities: Any,
    fitted_classes: Sequence[Any],
    expected_classes: Sequence[str],
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    fitted = tuple(map(str, fitted_classes))
    expected = tuple(map(str, expected_classes))
    if raw.ndim != 2 or raw.shape[1] != len(fitted):
        raise ValueError("Fitted classifier returned an invalid probability matrix")
    unknown = sorted(set(fitted) - set(expected))
    if unknown:
        raise ValueError(f"Fitted classifier returned unknown classes: {unknown}")
    aligned = np.zeros((raw.shape[0], len(expected)), dtype=float)
    positions = {name: index for index, name in enumerate(expected)}
    for source, name in enumerate(fitted):
        aligned[:, positions[name]] = raw[:, source]
    return _normalize_probabilities(aligned, len(expected))


def _resolve_columns(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = tuple(map(str, numeric_columns))
    categorical = tuple(map(str, categorical_columns))
    overlap = sorted(set(numeric).intersection(categorical))
    unknown = sorted(set(numeric + categorical) - set(map(str, frame.columns)))
    untyped = sorted(set(map(str, frame.columns)) - set(numeric + categorical))
    if overlap or unknown or untyped:
        raise ValueError(
            "numeric_columns and categorical_columns must be disjoint and exhaustive; "
            f"overlap={overlap}, unknown={unknown}, untyped={untyped}"
        )
    if not numeric and not categorical:
        raise ValueError("A feature-dependent classifier requires at least one predictor")
    return numeric, categorical


def _tabular_preprocessor(
    numeric_columns: Sequence[str], categorical_columns: Sequence[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_columns),
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_columns),
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


class ClassPriorSoftClassifier(ClassifierMixin, BaseEstimator):
    """Training-fold weighted mean of the supplied soft class probabilities."""

    def __init__(
        self,
        *,
        class_names: Sequence[str] = CARDINAL_CLASSES,
        prior_smoothing: float = 1e-9,
    ) -> None:
        self.class_names = class_names
        self.prior_smoothing = prior_smoothing

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "ClassPriorSoftClassifier":
        n_rows = _n_rows(X)
        names = _class_names(self.class_names)
        targets = _validate_soft_targets(y, n_rows=n_rows, class_names=names)
        weights = _validate_sample_weight(sample_weight, n_rows)
        smoothing = float(self.prior_smoothing)
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("prior_smoothing must be finite and nonnegative")
        mass = np.sum(targets * weights[:, None], axis=0) + smoothing
        self.prior_ = _normalize_probabilities(mass[None, :], len(names))[0]
        self.classes_ = np.asarray(names, dtype=object)
        self.n_features_in_ = int(np.asarray(X).shape[1])
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "prior_"):
            raise RuntimeError("ClassPriorSoftClassifier must be fitted before predict_proba")
        return np.repeat(self.prior_[None, :], _n_rows(X), axis=0)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class SoftMultinomialLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Multinomial logistic regression fitted to posterior soft labels."""

    def __init__(
        self,
        *,
        numeric_columns: Sequence[str] = (),
        categorical_columns: Sequence[str] = (),
        class_names: Sequence[str] = CARDINAL_CLASSES,
        C: float = 1.0,
        max_iter: int = 2000,
        random_state: int | None = 42,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.class_names = class_names
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "SoftMultinomialLogisticClassifier":
        frame = _as_frame(X)
        names = _class_names(self.class_names)
        numeric, categorical = _resolve_columns(
            frame, self.numeric_columns, self.categorical_columns
        )
        targets = _validate_soft_targets(y, n_rows=len(frame), class_names=names)
        weights = _validate_sample_weight(sample_weight, len(frame))
        positive_records = weights > 0
        frame = frame.loc[positive_records].reset_index(drop=True)
        targets = targets[positive_records]
        weights = weights[positive_records]

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.classes_ = np.asarray(names, dtype=object)
        prior_mass = np.sum(targets * weights[:, None], axis=0)
        self.fallback_prior_ = _normalize_probabilities(
            (prior_mass + 1e-12)[None, :], len(names)
        )[0]

        self.preprocessor_ = _tabular_preprocessor(numeric, categorical)
        transformed = self.preprocessor_.fit_transform(frame)
        active_classes = int(np.sum(prior_mass > 1e-12))
        if active_classes < 2:
            self.model_ = None
            return self

        repeat_index = np.repeat(np.arange(len(frame)), len(names))
        expanded_X = transformed[repeat_index]
        expanded_y = np.tile(np.asarray(names, dtype=object), len(frame))
        expanded_weight = (targets * weights[:, None]).reshape(-1)
        positive = expanded_weight > 0
        self.model_ = LogisticRegression(
            C=float(self.C),
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=int(self.max_iter),
            random_state=self.random_state,
        )
        self.model_.fit(
            expanded_X[positive],
            expanded_y[positive],
            sample_weight=expanded_weight[positive],
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "preprocessor_"):
            raise RuntimeError(
                "SoftMultinomialLogisticClassifier must be fitted before predict_proba"
            )
        frame = _as_frame(X, expected_columns=list(self.feature_names_in_))
        if self.model_ is None:
            return np.repeat(self.fallback_prior_[None, :], len(frame), axis=0)
        transformed = self.preprocessor_.transform(frame)
        return _align_probabilities(
            self.model_.predict_proba(transformed),
            self.model_.classes_,
            self.classes_,
        )

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class SoftOrdinalLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Proportional-odds-style logistic model for ordered soft classes.

    Each record contributes one soft binary outcome at every cumulative
    threshold.  Threshold indicators provide separate intercepts while the
    feature coefficients are shared.  Expanded weights are divided by the
    number of thresholds, preserving one unit of total weight per record.
    """

    def __init__(
        self,
        *,
        numeric_columns: Sequence[str] = (),
        categorical_columns: Sequence[str] = (),
        class_names: Sequence[str] = CLASSIC_CARDINAL_CLASSES,
        C: float = 1.0,
        max_iter: int = 2000,
        random_state: int | None = 42,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.class_names = class_names
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _threshold_design(
        transformed: np.ndarray,
        n_thresholds: int,
    ) -> np.ndarray:
        repeated = np.repeat(transformed, n_thresholds, axis=0)
        if n_thresholds == 1:
            return repeated
        threshold_ids = np.tile(np.arange(n_thresholds), len(transformed))
        threshold_dummies = np.eye(n_thresholds, dtype=float)[
            threshold_ids, 1:
        ]
        return np.column_stack([repeated, threshold_dummies])

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "SoftOrdinalLogisticClassifier":
        frame = _as_frame(X)
        names = _class_names(self.class_names)
        numeric, categorical = _resolve_columns(
            frame, self.numeric_columns, self.categorical_columns
        )
        targets = _validate_soft_targets(y, n_rows=len(frame), class_names=names)
        weights = _validate_sample_weight(sample_weight, len(frame))
        positive_records = weights > 0
        frame = frame.loc[positive_records].reset_index(drop=True)
        targets = targets[positive_records]
        weights = weights[positive_records]
        if not np.isfinite(float(self.C)) or float(self.C) <= 0:
            raise ValueError("C must be finite and positive")

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.classes_ = np.asarray(names, dtype=object)
        prior_mass = np.sum(targets * weights[:, None], axis=0)
        self.fallback_prior_ = _normalize_probabilities(
            (prior_mass + 1e-12)[None, :], len(names)
        )[0]
        self.preprocessor_ = _tabular_preprocessor(numeric, categorical)
        transformed = np.asarray(
            self.preprocessor_.fit_transform(frame), dtype=float
        )
        self.n_thresholds_ = len(names) - 1
        cumulative = np.cumsum(targets, axis=1)[:, :-1]
        design = self._threshold_design(transformed, self.n_thresholds_)
        cumulative_flat = cumulative.reshape(-1)
        threshold_record_weights = (
            np.repeat(weights, self.n_thresholds_) / self.n_thresholds_
        )
        expanded_X = np.repeat(design, 2, axis=0)
        expanded_y = np.tile(np.asarray([0, 1], dtype=int), len(design))
        expanded_weight = (
            np.column_stack(
                [1.0 - cumulative_flat, cumulative_flat]
            ).reshape(-1)
            * np.repeat(threshold_record_weights, 2)
        )
        positive = expanded_weight > 0
        if len(np.unique(expanded_y[positive])) < 2:
            self.model_ = None
            return self
        self.model_ = LogisticRegression(
            C=float(self.C),
            solver="lbfgs",
            max_iter=int(self.max_iter),
            random_state=self.random_state,
        )
        self.model_.fit(
            expanded_X[positive],
            expanded_y[positive],
            sample_weight=expanded_weight[positive],
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "preprocessor_"):
            raise RuntimeError(
                "SoftOrdinalLogisticClassifier must be fitted before predict_proba"
            )
        frame = _as_frame(X, expected_columns=list(self.feature_names_in_))
        if self.model_ is None:
            return np.repeat(self.fallback_prior_[None, :], len(frame), axis=0)
        transformed = np.asarray(self.preprocessor_.transform(frame), dtype=float)
        design = self._threshold_design(transformed, self.n_thresholds_)
        raw = self.model_.predict_proba(design)
        positive_position = int(
            np.flatnonzero(np.asarray(self.model_.classes_) == 1)[0]
        )
        cumulative = raw[:, positive_position].reshape(
            len(frame), self.n_thresholds_
        )
        cumulative = np.maximum.accumulate(
            np.clip(cumulative, 0.0, 1.0), axis=1
        )
        probabilities = np.empty(
            (len(frame), len(self.classes_)), dtype=float
        )
        probabilities[:, 0] = cumulative[:, 0]
        if self.n_thresholds_ > 1:
            probabilities[:, 1:-1] = np.diff(cumulative, axis=1)
        probabilities[:, -1] = 1.0 - cumulative[:, -1]
        return _normalize_probabilities(probabilities, len(self.classes_))

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class SoftGowerKNeighborsClassifier(ClassifierMixin, BaseEstimator):
    """Missing-aware Gower neighbor smoother for record-level soft targets.

    The default ``neighbor_mode="knn"`` retains the historical inverse-distance
    kNN behavior, except that all exact matches and all neighbors tied at the
    kth-distance boundary are included.  Expanding the boundary ties makes the
    prediction invariant to the order of otherwise identical training rows.

    ``"exponential_kernel"`` and ``"rbf_kernel"`` are optional full-neighbor
    smoothers.  They apply ``exp(-d / bandwidth)`` and
    ``exp(-0.5 * (d / bandwidth) ** 2)``, respectively, to every training
    record.  Record weights are multiplied into either type of neighbor
    weight, while Gower block and feature weights continue to define ``d``.
    """

    def __init__(
        self,
        *,
        numeric_columns: Sequence[str] = (),
        categorical_columns: Sequence[str] = (),
        class_names: Sequence[str] = CARDINAL_CLASSES,
        blocks: Mapping[str, Sequence[str]] | None = None,
        block_weights: Mapping[str, float] | None = None,
        feature_weights: Mapping[str, float] | None = None,
        n_neighbors: int = 7,
        distance_power: float = 1.0,
        epsilon: float = 1e-12,
        neighbor_mode: str = "knn",
        bandwidth: float = 1.0,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.class_names = class_names
        self.blocks = blocks
        self.block_weights = block_weights
        self.feature_weights = feature_weights
        self.n_neighbors = n_neighbors
        self.distance_power = distance_power
        self.epsilon = epsilon
        self.neighbor_mode = neighbor_mode
        self.bandwidth = bandwidth

    @staticmethod
    def _resolve_neighbor_mode(value: str) -> str:
        aliases = {
            "knn": "knn",
            "exponential": "exponential_kernel",
            "exponential_kernel": "exponential_kernel",
            "rbf": "rbf_kernel",
            "rbf_kernel": "rbf_kernel",
        }
        normalized = str(value).strip().lower()
        if normalized not in aliases:
            raise ValueError(
                "neighbor_mode must be one of knn, exponential_kernel, or "
                "rbf_kernel"
            )
        return aliases[normalized]

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "SoftGowerKNeighborsClassifier":
        frame = _as_frame(X)
        names = _class_names(self.class_names)
        numeric, categorical = _resolve_columns(
            frame, self.numeric_columns, self.categorical_columns
        )
        targets = _validate_soft_targets(y, n_rows=len(frame), class_names=names)
        weights = _validate_sample_weight(sample_weight, len(frame))
        if int(self.n_neighbors) < 1:
            raise ValueError("n_neighbors must be positive")
        if (
            not np.isfinite(float(self.distance_power))
            or float(self.distance_power) <= 0
        ):
            raise ValueError("distance_power must be finite and positive")
        if not np.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not np.isfinite(float(self.bandwidth)) or float(self.bandwidth) <= 0:
            raise ValueError("bandwidth must be finite and positive")

        positive = weights > 0
        train = frame.loc[positive].reset_index(drop=True)
        self.soft_targets_ = targets[positive]
        self.record_weights_ = weights[positive]
        self.neighbor_mode_ = self._resolve_neighbor_mode(self.neighbor_mode)
        self.classes_ = np.asarray(names, dtype=object)
        self.feature_names_in_ = np.asarray(train.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        columns = tuple(map(str, train.columns))
        self.distance_model_ = GowerKNeighborsRegressor(
            n_neighbors=int(self.n_neighbors),
            distance_power=float(self.distance_power),
            numeric_features=numeric,
            categorical_features=categorical,
            blocks=self.blocks or {"all": columns},
            block_weights=self.block_weights,
            feature_weights=self.feature_weights,
            epsilon=float(self.epsilon),
        ).fit(train, np.zeros(len(train), dtype=float))
        return self

    def _weighted_target_average(
        self,
        selected: np.ndarray,
        log_neighbor_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Average selected targets using numerically stable positive weights."""

        log_weights = np.log(self.record_weights_[selected])
        if log_neighbor_weights is not None:
            log_weights = log_weights + np.asarray(log_neighbor_weights, dtype=float)
        log_weights = log_weights - float(np.max(log_weights))
        stable_weights = np.exp(log_weights)
        return np.average(
            self.soft_targets_[selected],
            axis=0,
            weights=stable_weights,
        )

    def _knn_probability(self, row_distances: np.ndarray, k: int) -> np.ndarray:
        """Return one kNN probability vector with order-invariant boundary ties."""

        exact = np.flatnonzero(row_distances <= float(self.epsilon))
        if exact.size:
            # All exact matches must contribute, even when there are more than k.
            return self._weighted_target_average(exact)

        if k >= len(row_distances):
            selected = np.arange(len(row_distances), dtype=int)
        else:
            kth_distance = float(np.partition(row_distances, k - 1)[k - 1])
            numerical_tolerance = max(
                float(self.epsilon),
                16.0
                * np.finfo(float).eps
                * max(1.0, abs(kth_distance)),
            )
            selected = np.flatnonzero(
                (row_distances < kth_distance)
                | np.isclose(
                    row_distances,
                    kth_distance,
                    rtol=16.0 * np.finfo(float).eps,
                    atol=numerical_tolerance,
                )
            )

        selected_distances = row_distances[selected]
        log_inverse_distance = -float(self.distance_power) * np.log(
            selected_distances + float(self.epsilon)
        )
        return self._weighted_target_average(selected, log_inverse_distance)

    def _kernel_probability(self, row_distances: np.ndarray) -> np.ndarray:
        """Return one all-neighbor kernel probability vector."""

        minimum_distance = float(np.min(row_distances))
        bandwidth = float(self.bandwidth)
        if self.neighbor_mode_ == "exponential_kernel":
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                log_kernel = -(row_distances - minimum_distance) / bandwidth
        else:
            # Subtracting the nearest record's squared-distance exponent is a
            # common factor and leaves normalized RBF weights unchanged.  This
            # form guarantees at least one finite log weight for tiny bandwidths.
            delta = row_distances - minimum_distance
            log_drop = np.zeros_like(row_distances, dtype=float)
            farther = delta > 0
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                log_drop[farther] = (
                    0.5
                    * (delta[farther] / bandwidth)
                    * ((row_distances[farther] + minimum_distance) / bandwidth)
                )
            log_kernel = -log_drop
        selected = np.arange(len(row_distances), dtype=int)
        return self._weighted_target_average(selected, log_kernel)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "distance_model_"):
            raise RuntimeError(
                "SoftGowerKNeighborsClassifier must be fitted before predict_proba"
            )
        frame = _as_frame(X, expected_columns=list(self.feature_names_in_))
        distances = self.distance_model_._pairwise_distances(frame)
        k = min(int(self.n_neighbors), len(self.soft_targets_))
        probabilities = np.empty((len(frame), len(self.classes_)), dtype=float)
        for row_index, row_distances in enumerate(distances):
            if self.neighbor_mode_ == "knn":
                probabilities[row_index] = self._knn_probability(row_distances, k)
            else:
                probabilities[row_index] = self._kernel_probability(row_distances)
        return _normalize_probabilities(probabilities, len(self.classes_))

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class SoftExtraTreesClassifier(ClassifierMixin, BaseEstimator):
    """Shallow multi-output Extra Trees fitted directly to soft targets.

    A classifier trained by expanding each record to one pseudo-row per class
    would make ``min_samples_leaf`` count pseudo-rows rather than experimental
    records.  The multi-output regressor keeps one row per record, so the leaf
    constraint retains its intended record-level meaning.  Its bounded
    probability-vector predictions are normalized at inference.
    """

    def __init__(
        self,
        *,
        numeric_columns: Sequence[str] = (),
        categorical_columns: Sequence[str] = (),
        class_names: Sequence[str] = CARDINAL_CLASSES,
        n_estimators: int = 200,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        max_features: float | str | None = 0.7,
        random_state: int | None = 42,
        n_jobs: int = 1,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.class_names = class_names
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "SoftExtraTreesClassifier":
        frame = _as_frame(X)
        names = _class_names(self.class_names)
        numeric, categorical = _resolve_columns(
            frame, self.numeric_columns, self.categorical_columns
        )
        targets = _validate_soft_targets(y, n_rows=len(frame), class_names=names)
        weights = _validate_sample_weight(sample_weight, len(frame))
        positive_records = weights > 0
        frame = frame.loc[positive_records].reset_index(drop=True)
        targets = targets[positive_records]
        weights = weights[positive_records]

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.classes_ = np.asarray(names, dtype=object)
        prior_mass = np.sum(targets * weights[:, None], axis=0)
        self.fallback_prior_ = _normalize_probabilities(
            (prior_mass + 1e-12)[None, :], len(names)
        )[0]
        self.preprocessor_ = _tabular_preprocessor(numeric, categorical)
        transformed = self.preprocessor_.fit_transform(frame)
        active_classes = int(np.sum(prior_mass > 1e-12))
        if active_classes < 2:
            self.model_ = None
            return self

        self.model_ = ExtraTreesRegressor(
            n_estimators=int(self.n_estimators),
            max_depth=int(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=int(self.n_jobs),
        )
        self.model_.fit(transformed, targets, sample_weight=weights)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "preprocessor_"):
            raise RuntimeError("SoftExtraTreesClassifier must be fitted before predict_proba")
        frame = _as_frame(X, expected_columns=list(self.feature_names_in_))
        if self.model_ is None:
            return np.repeat(self.fallback_prior_[None, :], len(frame), axis=0)
        transformed = self.preprocessor_.transform(frame)
        return _normalize_probabilities(
            self.model_.predict(transformed),
            len(self.classes_),
        )

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class TwoStageCardinalClassifier(ClassifierMixin, BaseEstimator):
    """Gate cardinal/off-cardinal, then classify conditional cardinal mass."""

    def __init__(
        self,
        *,
        gate_estimator: BaseEstimator | None = None,
        conditional_estimator: BaseEstimator | None = None,
        cardinal_epsilon: float = 1e-12,
    ) -> None:
        self.gate_estimator = gate_estimator
        self.conditional_estimator = conditional_estimator
        self.cardinal_epsilon = cardinal_epsilon

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "TwoStageCardinalClassifier":
        n_rows = _n_rows(X)
        targets = _validate_soft_targets(
            y, n_rows=n_rows, class_names=CARDINAL_CLASSES
        )
        weights = _validate_sample_weight(sample_weight, n_rows)
        n_classic = len(CLASSIC_CARDINAL_CLASSES)
        cardinal_mass = targets[:, :n_classic].sum(axis=1)
        gate_targets = np.column_stack(
            (cardinal_mass, targets[:, n_classic])
        )

        gate_template = self.gate_estimator or ClassPriorSoftClassifier(
            class_names=CARDINAL_GATE_CLASSES
        )
        conditional_template = self.conditional_estimator or ClassPriorSoftClassifier(
            class_names=CLASSIC_CARDINAL_CLASSES
        )
        self.gate_estimator_ = clone(gate_template)
        self.gate_estimator_.fit(X, gate_targets, sample_weight=weights)

        eligible = cardinal_mass > float(self.cardinal_epsilon)
        if eligible.any() and np.sum(weights[eligible] * cardinal_mass[eligible]) > 0:
            conditional_targets = (
                targets[eligible, :n_classic]
                / cardinal_mass[eligible, None]
            )
            conditional_weights = weights[eligible] * cardinal_mass[eligible]
            self.conditional_estimator_ = clone(conditional_template)
            self.conditional_estimator_.fit(
                _slice_rows(X, eligible),
                conditional_targets,
                sample_weight=conditional_weights,
            )
        else:
            # A fold containing no posterior cardinal mass cannot identify the
            # conditional stage.  Keep the interface fitted and return an
            # explicitly uninformative conditional distribution.
            uniform = np.full(
                (n_rows, n_classic), 1.0 / n_classic, dtype=float
            )
            self.conditional_estimator_ = ClassPriorSoftClassifier(
                class_names=CLASSIC_CARDINAL_CLASSES,
                prior_smoothing=0.0,
            ).fit(X, uniform, sample_weight=weights)

        self.classes_ = np.asarray(CARDINAL_CLASSES, dtype=object)
        self.n_features_in_ = int(np.asarray(X).shape[1])
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "gate_estimator_"):
            raise RuntimeError("TwoStageCardinalClassifier must be fitted before predict_proba")
        gate = _align_probabilities(
            self.gate_estimator_.predict_proba(X),
            self.gate_estimator_.classes_,
            CARDINAL_GATE_CLASSES,
        )
        conditional = _align_probabilities(
            self.conditional_estimator_.predict_proba(X),
            self.conditional_estimator_.classes_,
            CLASSIC_CARDINAL_CLASSES,
        )
        combined = np.column_stack(
            (conditional * gate[:, [0]], gate[:, 1])
        )
        return _normalize_probabilities(combined, len(CARDINAL_CLASSES))

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class EngineeredCardinalClassifier(ClassifierMixin, BaseEstimator):
    """Fit one feature recipe and one soft classifier inside the same fold.

    The wrapper is intentionally constructed only after an outer/inner split
    has been made.  This prevents composition zero replacement, property PCA,
    imputation, rare-level pooling, or constant removal from seeing held-out
    records.
    """

    def __init__(
        self,
        *,
        feature_recipe: str,
        base_model_family: str,
        base_model_params: Mapping[str, Any] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.feature_recipe = feature_recipe
        self.base_model_family = base_model_family
        self.base_model_params = base_model_params
        self.random_seed = random_seed

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> "EngineeredCardinalClassifier":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "EngineeredCardinalClassifier requires a pandas DataFrame"
            )
        self.feature_builder_ = CardinalFeatureBuilder(
            self.feature_recipe
        ).fit(X)
        engineered = self.feature_builder_.transform(X)
        numeric = tuple(self.feature_builder_.numeric_columns_out_)
        categorical = tuple(
            self.feature_builder_.categorical_columns_out_
        )
        blocks = self.feature_builder_.get_blocks_out()
        spec = make_cardinal_model_spec(
            self.base_model_family,
            numeric_columns=numeric,
            categorical_columns=categorical,
            blocks=blocks,
            random_seed=int(self.random_seed),
            **dict(self.base_model_params or {}),
        )
        self.estimator_ = clone(spec.estimator)
        if sample_weight is None:
            self.estimator_.fit(engineered, y)
        else:
            self.estimator_.fit(
                engineered, y, sample_weight=sample_weight
            )
        self.classes_ = np.asarray(self.estimator_.classes_, dtype=object)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = int(X.shape[1])
        self.engineered_feature_names_ = tuple(
            map(str, self.feature_builder_.get_feature_names_out())
        )
        self.engineered_numeric_columns_ = numeric
        self.engineered_categorical_columns_ = categorical
        self.engineered_blocks_ = blocks
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError(
                "EngineeredCardinalClassifier must be fitted before predict_proba"
            )
        engineered = self.feature_builder_.transform(X)
        return self.estimator_.predict_proba(engineered)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def _two_stage(
    estimator_type: type[BaseEstimator],
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    class_kwargs: Mapping[str, Any],
) -> TwoStageCardinalClassifier:
    common = dict(class_kwargs)
    if estimator_type is not ClassPriorSoftClassifier:
        common.update(
            numeric_columns=tuple(numeric_columns),
            categorical_columns=tuple(categorical_columns),
        )
    return TwoStageCardinalClassifier(
        gate_estimator=estimator_type(
            class_names=CARDINAL_GATE_CLASSES,
            **common,
        ),
        conditional_estimator=estimator_type(
            class_names=CLASSIC_CARDINAL_CLASSES,
            **common,
        ),
    )


def _ordinal_two_stage(
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    C: float,
    max_iter: int,
    random_seed: int,
) -> TwoStageCardinalClassifier:
    """Use a binary soft gate and an ordered low/60/120 conditional model."""

    common = {
        "numeric_columns": tuple(numeric_columns),
        "categorical_columns": tuple(categorical_columns),
        "C": float(C),
        "max_iter": int(max_iter),
        "random_state": int(random_seed),
    }
    return TwoStageCardinalClassifier(
        gate_estimator=SoftMultinomialLogisticClassifier(
            class_names=CARDINAL_GATE_CLASSES,
            **common,
        ),
        conditional_estimator=SoftOrdinalLogisticClassifier(
            class_names=CLASSIC_CARDINAL_CLASSES,
            **common,
        ),
    )


def make_cardinal_model_spec(
    model_name: str,
    *,
    numeric_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] = (),
    blocks: Mapping[str, Sequence[str]] | None = None,
    random_seed: int = 42,
    **params: Any,
) -> CardinalModelSpec:
    """Build one fixed two-stage candidate family.

    Model-specific ``params`` override conservative defaults.  Grid-valued
    options use a ``*_grid`` suffix and are returned only for inner grouped CV.
    Unknown options raise rather than being silently ignored.
    """

    name = str(model_name)
    numeric = tuple(numeric_columns)
    categorical = tuple(categorical_columns)
    if name == "class_prior":
        allowed = {"prior_smoothing"}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise TypeError(f"Unknown class_prior parameters: {unknown}")
        estimator = _two_stage(
            ClassPriorSoftClassifier,
            numeric_columns=numeric,
            categorical_columns=categorical,
            class_kwargs={"prior_smoothing": float(params.get("prior_smoothing", 1e-9))},
        )
        return CardinalModelSpec(name, estimator, {}, complexity_rank=0)

    if not numeric and not categorical:
        raise ValueError(f"Model {name!r} requires resolved predictors")

    if name == "soft_ordinal_logistic":
        allowed = {"C", "C_grid", "max_iter"}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise TypeError(f"Unknown soft_ordinal_logistic parameters: {unknown}")
        estimator = _ordinal_two_stage(
            numeric_columns=numeric,
            categorical_columns=categorical,
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 2000)),
            random_seed=int(random_seed),
        )
        grid = tuple(float(value) for value in params.get("C_grid", (0.3, 3.0)))
        return CardinalModelSpec(
            name,
            estimator,
            {
                "gate_estimator__C": grid,
                "conditional_estimator__C": grid,
            },
            complexity_rank=1,
        )

    if name == "soft_multinomial_logistic":
        allowed = {"C", "C_grid", "max_iter"}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise TypeError(f"Unknown soft_multinomial_logistic parameters: {unknown}")
        estimator = _two_stage(
            SoftMultinomialLogisticClassifier,
            numeric_columns=numeric,
            categorical_columns=categorical,
            class_kwargs={
                "C": float(params.get("C", 1.0)),
                "max_iter": int(params.get("max_iter", 2000)),
                "random_state": int(random_seed),
            },
        )
        grid = tuple(float(value) for value in params.get("C_grid", (0.3, 3.0)))
        return CardinalModelSpec(
            name,
            estimator,
            {
                "gate_estimator__C": grid,
                "conditional_estimator__C": grid,
            },
            complexity_rank=2,
        )

    if name == "soft_gower_knn":
        allowed = {
            "n_neighbors",
            "n_neighbors_grid",
            "distance_power",
            "distance_power_grid",
            "epsilon",
            "block_weights",
            "feature_weights",
            "neighbor_mode",
            "bandwidth",
        }
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise TypeError(f"Unknown soft_gower_knn parameters: {unknown}")
        estimator = _two_stage(
            SoftGowerKNeighborsClassifier,
            numeric_columns=numeric,
            categorical_columns=categorical,
            class_kwargs={
                "blocks": blocks or {"all": numeric + categorical},
                "block_weights": params.get("block_weights"),
                "feature_weights": params.get("feature_weights"),
                "n_neighbors": int(params.get("n_neighbors", 7)),
                "distance_power": float(params.get("distance_power", 1.0)),
                "epsilon": float(params.get("epsilon", 1e-12)),
                "neighbor_mode": str(params.get("neighbor_mode", "knn")),
                "bandwidth": float(params.get("bandwidth", 1.0)),
            },
        )
        neighbors = tuple(
            int(value) for value in params.get("n_neighbors_grid", (5, 9))
        )
        powers = tuple(
            float(value) for value in params.get("distance_power_grid", (1.0,))
        )
        return CardinalModelSpec(
            name,
            estimator,
            {
                "gate_estimator__n_neighbors": neighbors,
                "conditional_estimator__n_neighbors": neighbors,
                "gate_estimator__distance_power": powers,
                "conditional_estimator__distance_power": powers,
            },
            complexity_rank=3,
        )

    if name == "soft_extra_trees":
        allowed = {
            "n_estimators",
            "max_depth",
            "max_depth_grid",
            "min_samples_leaf",
            "min_samples_leaf_grid",
            "max_features",
            "n_jobs",
        }
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise TypeError(f"Unknown soft_extra_trees parameters: {unknown}")
        estimator = _two_stage(
            SoftExtraTreesClassifier,
            numeric_columns=numeric,
            categorical_columns=categorical,
            class_kwargs={
                "n_estimators": int(params.get("n_estimators", 200)),
                "max_depth": int(params.get("max_depth", 3)),
                "min_samples_leaf": int(params.get("min_samples_leaf", 5)),
                "max_features": params.get("max_features", 0.7),
                "random_state": int(random_seed),
                "n_jobs": int(params.get("n_jobs", 1)),
            },
        )
        depths = tuple(int(value) for value in params.get("max_depth_grid", (2, 4)))
        leaves = tuple(
            int(value) for value in params.get("min_samples_leaf_grid", (4,))
        )
        return CardinalModelSpec(
            name,
            estimator,
            {
                "gate_estimator__max_depth": depths,
                "conditional_estimator__max_depth": depths,
                "gate_estimator__min_samples_leaf": leaves,
                "conditional_estimator__min_samples_leaf": leaves,
            },
            complexity_rank=4,
        )

    raise KeyError(
        f"Unknown cardinal model {name!r}; choose from {', '.join(CARDINAL_MODEL_NAMES)}"
    )


def cardinal_model_catalog(
    *,
    numeric_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] = (),
    blocks: Mapping[str, Sequence[str]] | None = None,
    random_seed: int = 42,
    model_names: Sequence[str] = CARDINAL_MODEL_NAMES,
) -> tuple[CardinalModelSpec, ...]:
    """Return the ordered, finite candidate catalog used by nested CV."""

    return tuple(
        make_cardinal_model_spec(
            name,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            blocks=blocks,
            random_seed=random_seed,
        )
        for name in model_names
    )


__all__ = [
    "CARDINAL_CLASSES",
    "CLASSIC_CARDINAL_CLASSES",
    "CARDINAL_GATE_CLASSES",
    "CARDINAL_MODEL_NAMES",
    "CardinalModelSpec",
    "ClassPriorSoftClassifier",
    "SoftMultinomialLogisticClassifier",
    "SoftOrdinalLogisticClassifier",
    "SoftGowerKNeighborsClassifier",
    "SoftExtraTreesClassifier",
    "TwoStageCardinalClassifier",
    "EngineeredCardinalClassifier",
    "make_cardinal_model_spec",
    "cardinal_model_catalog",
]
