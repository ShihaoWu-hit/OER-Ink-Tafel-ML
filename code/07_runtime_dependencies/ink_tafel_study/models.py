"""Small-sample regression models used by the v2 study workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    """An estimator and its strictly inner-CV parameter grid."""

    name: str
    estimator: BaseEstimator
    param_grid: Mapping[str, Sequence[Any]]
    complexity_rank: int
    probabilistic: bool = False


class GowerKNeighborsRegressor(RegressorMixin, BaseEstimator):
    """Distance-weighted kNN regression with missing-aware Gower distance.

    Numeric ranges and all training values are learned in :meth:`fit`; there is
    no global preprocessing state.  Blocks must form a mutually exclusive,
    exhaustive partition of the supplied predictors.  Distances are averaged
    within a block and then across blocks, so a block with more columns does not
    automatically dominate another block.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        distance_power: float = 1.0,
        numeric_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        blocks: Mapping[str, Sequence[str]] | None = None,
        block_weights: Mapping[str, float] | None = None,
        feature_weights: Mapping[str, float] | None = None,
        epsilon: float = 1e-12,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.distance_power = distance_power
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.blocks = blocks
        self.block_weights = block_weights
        self.feature_weights = feature_weights
        self.epsilon = epsilon

    @staticmethod
    def _as_frame(X: Any, expected_columns: Sequence[str] | None = None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy(deep=False)
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("GowerKNeighborsRegressor requires a two-dimensional feature matrix")
            if expected_columns is None:
                columns = [f"x{idx}" for idx in range(array.shape[1])]
            else:
                columns = list(expected_columns)
            frame = pd.DataFrame(array, columns=columns)
        if expected_columns is not None:
            missing = [column for column in expected_columns if column not in frame]
            if missing:
                raise ValueError(f"Prediction table is missing fitted Gower features: {missing}")
            frame = frame.loc[:, list(expected_columns)]
        return frame

    @staticmethod
    def _validate_partition(
        columns: Sequence[str], blocks: Mapping[str, Sequence[str]]
    ) -> dict[str, tuple[str, ...]]:
        normalized = {str(name): tuple(map(str, members)) for name, members in blocks.items()}
        assigned = [column for members in normalized.values() for column in members]
        duplicated = sorted({column for column in assigned if assigned.count(column) > 1})
        if duplicated:
            raise ValueError(f"Gower distance blocks overlap; duplicated features: {duplicated}")
        expected = set(map(str, columns))
        assigned_set = set(assigned)
        missing = sorted(expected - assigned_set)
        unknown = sorted(assigned_set - expected)
        if missing or unknown:
            raise ValueError(
                "Gower blocks must be exhaustive and use only fitted columns; "
                f"missing={missing}, unknown={unknown}"
            )
        if any(not members for members in normalized.values()):
            raise ValueError("Gower distance blocks cannot be empty")
        return normalized

    def fit(self, X: Any, y: Any) -> "GowerKNeighborsRegressor":
        frame = self._as_frame(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(frame) != len(target):
            raise ValueError(f"X and y have inconsistent lengths: {len(frame)} != {len(target)}")
        if len(frame) == 0:
            raise ValueError("Cannot fit Gower kNN on an empty training set")
        if not np.isfinite(target).all():
            raise ValueError("Gower kNN target contains NaN or infinite values")
        if int(self.n_neighbors) < 1:
            raise ValueError("n_neighbors must be positive")
        if float(self.distance_power) <= 0:
            raise ValueError("distance_power must be positive")

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        columns = list(map(str, self.feature_names_in_))

        if self.numeric_features is None and self.categorical_features is None:
            numeric = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
            categorical = [column for column in columns if column not in numeric]
        else:
            numeric = list(map(str, self.numeric_features or ()))
            categorical = list(map(str, self.categorical_features or ()))
            overlap = sorted(set(numeric).intersection(categorical))
            unknown = sorted(set(numeric + categorical) - set(columns))
            untyped = sorted(set(columns) - set(numeric + categorical))
            if overlap or unknown or untyped:
                raise ValueError(
                    "Numeric and categorical features must be disjoint and exhaustive; "
                    f"overlap={overlap}, unknown={unknown}, untyped={untyped}"
                )

        if self.blocks is None:
            blocks: Mapping[str, Sequence[str]] = {"all": tuple(columns)}
        else:
            blocks = self.blocks
        self.blocks_ = self._validate_partition(columns, blocks)

        supplied_block_weights = dict(self.block_weights or {})
        unknown_weight_blocks = sorted(set(supplied_block_weights) - set(self.blocks_))
        if unknown_weight_blocks:
            raise ValueError(f"Weights supplied for unknown Gower blocks: {unknown_weight_blocks}")
        self.block_weights_ = {
            block: float(supplied_block_weights.get(block, 1.0)) for block in self.blocks_
        }
        if any((not np.isfinite(weight)) or weight <= 0 for weight in self.block_weights_.values()):
            raise ValueError("All Gower block weights must be finite and positive")

        supplied_feature_weights = dict(self.feature_weights or {})
        unknown_weight_features = sorted(set(supplied_feature_weights) - set(columns))
        if unknown_weight_features:
            raise ValueError(f"Weights supplied for unknown Gower features: {unknown_weight_features}")
        self.feature_weights_ = {
            column: float(supplied_feature_weights.get(column, 1.0)) for column in columns
        }
        if any((not np.isfinite(weight)) or weight <= 0 for weight in self.feature_weights_.values()):
            raise ValueError("All Gower feature weights must be finite and positive")

        self.numeric_features_ = tuple(numeric)
        self.categorical_features_ = tuple(categorical)
        self._numeric_set_ = set(numeric)
        self.X_train_ = frame.reset_index(drop=True).copy()
        for column in self.numeric_features_:
            self.X_train_[column] = pd.to_numeric(self.X_train_[column], errors="coerce")
        self.y_train_ = target.copy()
        self.numeric_ranges_ = {}
        for column in self.numeric_features_:
            values = self.X_train_[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            self.numeric_ranges_[column] = float(np.ptp(finite)) if finite.size else np.nan
        return self

    def _pairwise_distances(self, X: pd.DataFrame) -> np.ndarray:
        n_query = len(X)
        n_train = len(self.X_train_)
        overall_numerator = np.zeros((n_query, n_train), dtype=float)
        overall_denominator = np.zeros((n_query, n_train), dtype=float)

        for block, members in self.blocks_.items():
            block_numerator = np.zeros((n_query, n_train), dtype=float)
            block_denominator = np.zeros((n_query, n_train), dtype=float)
            for column in members:
                feature_weight = self.feature_weights_[column]
                if column in self._numeric_set_:
                    scale = self.numeric_ranges_[column]
                    # A constant or entirely missing training feature contains no
                    # out-of-sample distance information and is skipped.
                    if not np.isfinite(scale) or scale <= 0:
                        continue
                    query = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)[:, None]
                    train = self.X_train_[column].to_numpy(dtype=float)[None, :]
                    valid = np.isfinite(query) & np.isfinite(train)
                    contribution = np.minimum(np.abs(query - train) / scale, 1.0)
                else:
                    query = X[column].astype("object").to_numpy()[:, None]
                    train = self.X_train_[column].astype("object").to_numpy()[None, :]
                    valid = (~pd.isna(query)) & (~pd.isna(train))
                    contribution = (query != train).astype(float)
                block_numerator += np.where(valid, contribution * feature_weight, 0.0)
                block_denominator += valid.astype(float) * feature_weight

            valid_block = block_denominator > 0
            block_distance = np.divide(
                block_numerator,
                block_denominator,
                out=np.zeros_like(block_numerator),
                where=valid_block,
            )
            block_weight = self.block_weights_[block]
            overall_numerator += np.where(valid_block, block_distance * block_weight, 0.0)
            overall_denominator += valid_block.astype(float) * block_weight

        return np.divide(
            overall_numerator,
            overall_denominator,
            out=np.ones_like(overall_numerator),
            where=overall_denominator > 0,
        )

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "X_train_"):
            raise RuntimeError("GowerKNeighborsRegressor must be fitted before predict")
        frame = self._as_frame(X, expected_columns=list(self.feature_names_in_))
        distances = self._pairwise_distances(frame)
        k = min(int(self.n_neighbors), len(self.y_train_))
        predictions = np.empty(len(frame), dtype=float)
        for row_index, row_distances in enumerate(distances):
            nearest = np.argpartition(row_distances, k - 1)[:k]
            nearest_distances = row_distances[nearest]
            zero = nearest_distances <= float(self.epsilon)
            if zero.any():
                predictions[row_index] = float(np.mean(self.y_train_[nearest[zero]]))
                continue
            weights = 1.0 / np.power(nearest_distances + float(self.epsilon), float(self.distance_power))
            predictions[row_index] = float(np.average(self.y_train_[nearest], weights=weights))
        return predictions


def _tabular_preprocessor(
    numeric_columns: Sequence[str], categorical_columns: Sequence[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))
    if not transformers:
        raise ValueError("A non-dummy model requires at least one resolved predictor")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_model_spec(
    model_name: str,
    *,
    numeric_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] = (),
    blocks: Mapping[str, Sequence[str]] | None = None,
    random_seed: int = 42,
    gower_neighbors: Sequence[int] = (3, 5, 7, 9, 11, 15, 21),
    gower_distance_powers: Sequence[float] = (1.0, 2.0),
    extra_trees_n_estimators: int = 300,
    extra_trees_max_depths: Sequence[int] = (2, 3, 4),
    extra_trees_min_samples_leaf: Sequence[int] = (4, 6, 8),
) -> ModelSpec:
    """Construct one of the fixed model families.

    Parameter grids returned here are consumed only by inner grouped CV.
    """

    if model_name == "dummy_mean":
        return ModelSpec(model_name, DummyRegressor(strategy="mean"), {}, complexity_rank=0)
    if model_name == "dummy_median":
        return ModelSpec(model_name, DummyRegressor(strategy="median"), {}, complexity_rank=0)

    columns = tuple(numeric_columns) + tuple(categorical_columns)
    if not columns:
        raise ValueError(f"Model {model_name!r} requires resolved predictors")

    if model_name == "bayesian_ridge":
        estimator = Pipeline(
            [
                ("preprocess", _tabular_preprocessor(numeric_columns, categorical_columns)),
                ("regressor", BayesianRidge(compute_score=True)),
            ]
        )
        # BayesianRidge estimates regularization strengths from the training
        # fold.  There is no outer-fold tuning grid to leak.
        return ModelSpec(model_name, estimator, {}, complexity_rank=1, probabilistic=True)

    if model_name == "gower_knn":
        estimator = GowerKNeighborsRegressor(
            numeric_features=tuple(numeric_columns),
            categorical_features=tuple(categorical_columns),
            blocks=blocks or {"all": columns},
        )
        return ModelSpec(
            model_name,
            estimator,
            {
                "n_neighbors": tuple(int(value) for value in gower_neighbors),
                "distance_power": tuple(float(value) for value in gower_distance_powers),
            },
            complexity_rank=2,
        )

    if model_name == "extra_trees":
        estimator = Pipeline(
            [
                ("preprocess", _tabular_preprocessor(numeric_columns, categorical_columns)),
                (
                    "regressor",
                    ExtraTreesRegressor(
                        n_estimators=int(extra_trees_n_estimators),
                        max_depth=int(tuple(extra_trees_max_depths)[0]),
                        min_samples_leaf=int(tuple(extra_trees_min_samples_leaf)[0]),
                        max_features=0.7,
                        random_state=int(random_seed),
                        n_jobs=1,
                    ),
                ),
            ]
        )
        return ModelSpec(
            model_name,
            estimator,
            {
                "regressor__max_depth": tuple(int(value) for value in extra_trees_max_depths),
                "regressor__min_samples_leaf": tuple(
                    int(value) for value in extra_trees_min_samples_leaf
                ),
            },
            complexity_rank=3,
        )

    raise KeyError(
        f"Unknown model {model_name!r}; choose from dummy_mean, dummy_median, "
        "bayesian_ridge, gower_knn, extra_trees"
    )


def predict_log_with_uncertainty(
    fitted_estimator: BaseEstimator, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict log10 slope and, for BayesianRidge, its predictive SD."""

    if isinstance(fitted_estimator, Pipeline) and isinstance(
        fitted_estimator.named_steps.get("regressor"), BayesianRidge
    ):
        transformed = fitted_estimator.named_steps["preprocess"].transform(X)
        mean, std = fitted_estimator.named_steps["regressor"].predict(transformed, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    return np.asarray(fitted_estimator.predict(X), dtype=float), None


def gaussian_log_scores(
    observed_log10: np.ndarray,
    predicted_log10: np.ndarray,
    predictive_sd_log10: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gaussian NLPD and CRPS on the explicitly named log10 scale."""

    observed = np.asarray(observed_log10, dtype=float)
    predicted = np.asarray(predicted_log10, dtype=float)
    sd = np.maximum(np.asarray(predictive_sd_log10, dtype=float), 1e-12)
    z = (observed - predicted) / sd
    nlpd = 0.5 * np.log(2.0 * np.pi * sd**2) + 0.5 * z**2
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    crps = sd * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return nlpd, crps


def lognormal_original_scale_scores(
    observed_mV_dec: np.ndarray,
    predicted_log10: np.ndarray,
    predictive_sd_log10: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return NLPD and CRPS for the implied lognormal slope distribution.

    Unlike :func:`gaussian_log_scores`, CRPS is expressed directly in
    mV dec-1.  The Jacobian term is included in NLPD, making its declared
    density scale the original slope coordinate as well.
    """

    observed = np.maximum(np.asarray(observed_mV_dec, dtype=float), 1e-300)
    mu_natural = np.asarray(predicted_log10, dtype=float) * np.log(10.0)
    sd_natural = np.maximum(
        np.asarray(predictive_sd_log10, dtype=float) * np.log(10.0), 1e-12
    )
    log_observed = np.log(observed)
    z = (log_observed - mu_natural) / sd_natural
    nlpd = (
        0.5 * np.log(2.0 * np.pi * sd_natural**2)
        + 0.5 * z**2
        + log_observed
    )
    expected = np.exp(mu_natural + 0.5 * sd_natural**2)
    crps = observed * (2.0 * ndtr(z) - 1.0) - 2.0 * expected * (
        ndtr(z - sd_natural) + ndtr(sd_natural / np.sqrt(2.0)) - 1.0
    )
    return nlpd, np.maximum(crps, 0.0)
