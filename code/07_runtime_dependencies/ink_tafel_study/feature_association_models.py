"""Fixed small-sample models for the feature-association workflow.

The models in this module deliberately expose a narrow API.  They accept an
already engineered predictor table, learn every preprocessing quantity from
the supplied training rows, and fit a scikit-learn :class:`BayesianRidge`.
``gam_low_df`` is an empirical-Bayes additive spline model, not a full
Bayesian GAM or an MCMC fit.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .features import validate_no_target_leakage


MODEL_FAMILIES = ("bayesian_linear", "gam_low_df")

_MISSING_CATEGORY = "__MISSING__"
_RARE_CATEGORY = "__RARE__"
_RESERVED_CATEGORIES = frozenset({_MISSING_CATEGORY, _RARE_CATEGORY})


def _as_frame(
    X: Any,
    *,
    expected_columns: Sequence[str] | None = None,
    context: str,
) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{context} requires a pandas DataFrame")
    if X.empty:
        raise ValueError(f"{context} cannot use an empty predictor table")
    if not X.columns.is_unique:
        duplicated = X.columns[X.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"{context} predictor columns are duplicated: {duplicated}")
    if any(not isinstance(column, str) for column in X.columns):
        raise TypeError(f"{context} requires string predictor column names")
    validate_no_target_leakage(X.columns)
    if expected_columns is not None:
        expected = tuple(map(str, expected_columns))
        supplied = tuple(map(str, X.columns))
        missing = sorted(set(expected) - set(supplied))
        extra = sorted(set(supplied) - set(expected))
        if missing or extra:
            raise ValueError(
                f"{context} predictor schema differs from fit; "
                f"missing={missing}, extra={extra}"
            )
        return X.loc[:, list(expected)].copy(deep=False)
    return X.copy(deep=False)


class _RareCategoryPooler(TransformerMixin, BaseEstimator):
    """Fold-local missing-value and rare-level pooling for categorical data."""

    def __init__(self, *, min_frequency: int = 3) -> None:
        self.min_frequency = min_frequency

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        missing = series.isna()
        normalized = series.astype("string")
        normalized = normalized.where(~missing, _MISSING_CATEGORY)
        return normalized.astype("object")

    def fit(self, X: Any, y: Any = None) -> "_RareCategoryPooler":
        del y
        if int(self.min_frequency) < 1:
            raise ValueError("min_frequency must be positive")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("_RareCategoryPooler requires a pandas DataFrame")
        if not X.columns.is_unique:
            raise ValueError("Categorical predictor columns must be unique")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.frequent_levels_: dict[str, tuple[str, ...]] = {}
        self.categories_: dict[str, tuple[str, ...]] = {}
        for column in self.feature_names_in_:
            normalized = self._normalize(X[str(column)])
            # AssociationFeatureBuilder may already have pooled a training-fold
            # category to these two explicit sentinel levels.  Treat them as
            # structural missing/rare categories rather than as ordinary
            # values that can become "frequent" during the model's defensive
            # preprocessing pass.
            observed = normalized.loc[
                ~normalized.isin((_MISSING_CATEGORY, _RARE_CATEGORY))
            ]
            counts = observed.astype(str).value_counts(dropna=False)
            frequent = tuple(
                sorted(counts.loc[counts >= int(self.min_frequency)].index.astype(str))
            )
            self.frequent_levels_[str(column)] = frequent
            # Explicit reserved categories make missing and unseen levels safe
            # even when neither occurs in the training fold.  A real frequent
            # level is kept first so OneHot(drop='first') has an interpretable
            # training-supported reference whenever possible.
            categories = (*frequent, _MISSING_CATEGORY, _RARE_CATEGORY)
            if not frequent:
                categories = (_RARE_CATEGORY, _MISSING_CATEGORY)
            self.categories_[str(column)] = tuple(dict.fromkeys(categories))
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, ("feature_names_in_", "frequent_levels_"))
        if not isinstance(X, pd.DataFrame):
            raise TypeError("_RareCategoryPooler requires a pandas DataFrame")
        expected = list(map(str, self.feature_names_in_))
        missing = sorted(set(expected) - set(map(str, X.columns)))
        if missing:
            raise ValueError(f"Categorical transform is missing columns: {missing}")
        output = pd.DataFrame(index=X.index)
        for column in expected:
            normalized = self._normalize(X[column])
            frequent = set(self.frequent_levels_[column])
            is_missing = normalized.eq(_MISSING_CATEGORY)
            pooled = normalized.where(
                is_missing | normalized.astype(str).isin(frequent),
                _RARE_CATEGORY,
            )
            output[column] = pooled.astype("object")
        return output

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None:
            supplied = tuple(map(str, input_features))
            expected = tuple(map(str, self.feature_names_in_))
            if supplied != expected:
                raise ValueError(
                    f"input_features differ from fit: expected={expected}, supplied={supplied}"
                )
        return np.asarray(self.feature_names_in_, dtype=object)


class EmpiricalBayesAssociationModel(RegressorMixin, BaseEstimator):
    """Cloneable empirical-Bayes linear or low-df additive spline regressor.

    Predictor types are inferred from the engineered pandas dtypes.  Boolean
    columns are categorical; other numeric dtypes are numeric.  There are no
    interaction terms.  ``fit`` expects the workflow's response transform
    (normally log10 Tafel slope) to have already been applied to ``y``.
    """

    def __init__(
        self,
        model_family: str,
        *,
        gam_n_knots: int = 3,
        gam_degree: int = 1,
        rare_min_frequency: int = 3,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.model_family = model_family
        self.gam_n_knots = gam_n_knots
        self.gam_degree = gam_degree
        self.rare_min_frequency = rare_min_frequency
        self.max_iter = max_iter
        self.tol = tol

    def _validate_parameters(self) -> None:
        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(
                f"Unknown model_family {self.model_family!r}; choose from {MODEL_FAMILIES}"
            )
        if int(self.gam_n_knots) < 2:
            raise ValueError("gam_n_knots must be at least 2")
        if int(self.gam_degree) < 1:
            raise ValueError("gam_degree must be positive")
        if int(self.rare_min_frequency) < 1:
            raise ValueError("rare_min_frequency must be positive")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be positive")
        if not np.isfinite(float(self.tol)) or float(self.tol) <= 0:
            raise ValueError("tol must be finite and positive")

    @staticmethod
    def _numeric_pipeline(*, spline: bool, n_knots: int, degree: int) -> Pipeline:
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if spline:
            steps.append(
                (
                    "spline",
                    SplineTransformer(
                        n_knots=int(n_knots),
                        degree=int(degree),
                        knots="quantile",
                        extrapolation="constant",
                        include_bias=False,
                    ),
                )
            )
        steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    def _categorical_pipeline(
        self, frame: pd.DataFrame, columns: Sequence[str]
    ) -> Pipeline:
        pooler = _RareCategoryPooler(min_frequency=int(self.rare_min_frequency))
        pooler.fit(frame.loc[:, list(columns)])
        categories = [list(pooler.categories_[column]) for column in columns]
        return Pipeline(
            [
                (
                    "pooler",
                    _RareCategoryPooler(
                        min_frequency=int(self.rare_min_frequency)
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        categories=categories,
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

    def _make_preprocessor(self, frame: pd.DataFrame) -> ColumnTransformer:
        transformers: list[tuple[str, Any, list[str]]] = []
        spline_columns: list[str] = []
        fallback_columns: list[str] = []
        for position, column in enumerate(self.numeric_columns_):
            values = pd.to_numeric(frame[column], errors="coerce")
            if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"Numeric predictor {column!r} contains infinite values")
            finite = values.loc[np.isfinite(values)]
            if finite.empty:
                raise ValueError(
                    f"Numeric predictor {column!r} has no finite training values "
                    "for median imputation"
                )
            use_spline = bool(
                self.model_family == "gam_low_df" and finite.nunique() >= 4
            )
            if use_spline:
                spline_columns.append(column)
            elif self.model_family == "gam_low_df":
                fallback_columns.append(column)
            transformers.append(
                (
                    f"numeric_{position:03d}",
                    self._numeric_pipeline(
                        spline=use_spline,
                        n_knots=int(self.gam_n_knots),
                        degree=int(self.gam_degree),
                    ),
                    [column],
                )
            )
        if self.categorical_columns_:
            transformers.append(
                (
                    "categorical",
                    self._categorical_pipeline(frame, self.categorical_columns_),
                    list(self.categorical_columns_),
                )
            )
        if not transformers:
            raise ValueError("At least one numeric or categorical predictor is required")
        self.spline_columns_ = tuple(spline_columns)
        self.linear_fallback_columns_ = tuple(fallback_columns)
        return ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    @staticmethod
    def _response(y: Any, *, n_rows: int) -> np.ndarray:
        response = np.asarray(y, dtype=float)
        if response.ndim != 1:
            raise ValueError("y must be a one-dimensional continuous response")
        if len(response) != n_rows:
            raise ValueError(f"X and y have inconsistent lengths: {n_rows} != {len(response)}")
        if not np.isfinite(response).all():
            raise ValueError("y contains NaN or infinite values")
        return response

    @staticmethod
    def _sample_weight(sample_weight: Any, *, n_rows: int) -> np.ndarray | None:
        if sample_weight is None:
            return None
        weights = np.asarray(sample_weight, dtype=float)
        if weights.ndim != 1 or len(weights) != n_rows:
            raise ValueError("sample_weight must be one-dimensional and match X")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("sample_weight values must be finite and positive")
        return weights

    @staticmethod
    def _empirical_edf(
        design: np.ndarray,
        regressor: BayesianRidge,
        sample_weight: np.ndarray | None,
    ) -> float:
        if sample_weight is None:
            offset = np.mean(design, axis=0)
            centered = design - offset
        else:
            offset = np.average(design, axis=0, weights=sample_weight)
            centered = (design - offset) * np.sqrt(sample_weight)[:, None]
        smoother_trace = float(
            regressor.alpha_
            * np.trace(centered @ regressor.sigma_ @ centered.T)
        )
        maximum = 1.0 + float(np.linalg.matrix_rank(centered))
        return float(np.clip(1.0 + smoother_trace, 1.0, maximum))

    def fit(
        self,
        X: pd.DataFrame,
        y: Any,
        sample_weight: Any = None,
    ) -> "EmpiricalBayesAssociationModel":
        self._validate_parameters()
        frame = _as_frame(X, context="EmpiricalBayesAssociationModel.fit")
        response = self._response(y, n_rows=len(frame))
        weights = self._sample_weight(sample_weight, n_rows=len(frame))

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        numeric = [
            column
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column])
        ]
        self.numeric_columns_ = tuple(numeric)
        self.categorical_columns_ = tuple(
            column for column in frame.columns if column not in set(numeric)
        )
        self.preprocessor_ = self._make_preprocessor(frame)
        design = np.asarray(self.preprocessor_.fit_transform(frame), dtype=float)
        if design.ndim != 2 or design.shape[1] == 0:
            raise ValueError("Fold-local preprocessing produced no design columns")
        if not np.isfinite(design).all():
            raise ValueError("Fold-local preprocessing produced non-finite values")

        self.regressor_ = BayesianRidge(
            compute_score=True,
            max_iter=int(self.max_iter),
            tol=float(self.tol),
        )
        self.regressor_.fit(design, response, sample_weight=weights)
        self.design_feature_names_ = np.asarray(
            self.preprocessor_.get_feature_names_out(), dtype=object
        )
        self.design_column_count_ = int(design.shape[1])
        if len(self.design_feature_names_) != self.design_column_count_:
            raise AssertionError("Design names do not match transformed column count")
        self.empirical_edf_ = self._empirical_edf(
            design, self.regressor_, weights
        )

        category_levels: dict[str, list[str]] = {}
        if self.categorical_columns_:
            categorical = self.preprocessor_.named_transformers_["categorical"]
            pooler = categorical.named_steps["pooler"]
            category_levels = {
                column: list(pooler.categories_[column])
                for column in self.categorical_columns_
            }
        scientific_name = (
            "empirical-Bayes shrinkage linear model"
            if self.model_family == "bayesian_linear"
            else "empirical-Bayes shrinkage additive spline model"
        )
        self.model_definition_ = {
            "schema_version": "feature-association-model-v1",
            "model_family": self.model_family,
            "scientific_name": scientific_name,
            "final_estimator": "sklearn.linear_model.BayesianRidge",
            "full_bayesian_posterior": False,
            "interactions_included": False,
            "preprocessing_fit_scope": "training_fold_only",
            "numeric_imputation": "median",
            "numeric_columns": list(self.numeric_columns_),
            "categorical_columns": list(self.categorical_columns_),
            "categorical_missing_level": _MISSING_CATEGORY,
            "categorical_rare_level": _RARE_CATEGORY,
            "categorical_rare_min_frequency": int(self.rare_min_frequency),
            "categorical_levels_after_pooling": category_levels,
            "onehot_drop": "first",
            "onehot_handle_unknown": "ignore_after_mapping_to_rare",
            "gam_spline_columns": list(self.spline_columns_),
            "gam_linear_fallback_columns": list(self.linear_fallback_columns_),
            "gam_spline": {
                "n_knots": int(self.gam_n_knots),
                "degree": int(self.gam_degree),
                "knots": "quantile",
                "include_bias": False,
                "extrapolation": "constant",
                "minimum_unique_training_values": 4,
            },
            "design_feature_names": self.design_feature_names_.astype(str).tolist(),
            "design_column_count": self.design_column_count_,
            "empirical_edf": self.empirical_edf_,
            "bayesian_ridge": {
                "max_iter": int(self.max_iter),
                "tol": float(self.tol),
                "alpha_precision": float(self.regressor_.alpha_),
                "coefficient_precision": float(self.regressor_.lambda_),
            },
        }
        return self

    def _transform_for_prediction(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self,
            (
                "preprocessor_",
                "regressor_",
                "feature_names_in_",
                "design_feature_names_",
            ),
        )
        frame = _as_frame(
            X,
            expected_columns=self.feature_names_in_,
            context="EmpiricalBayesAssociationModel.predict",
        )
        design = np.asarray(self.preprocessor_.transform(frame), dtype=float)
        if design.shape[1] != self.design_column_count_:
            raise AssertionError("Prediction design width differs from fitted design")
        if not np.isfinite(design).all():
            raise ValueError("Prediction preprocessing produced non-finite values")
        return design

    def predict(
        self, X: pd.DataFrame, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        design = self._transform_for_prediction(X)
        if return_std:
            mean, std = self.regressor_.predict(design, return_std=True)
            return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        return np.asarray(self.regressor_.predict(design), dtype=float)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, ("feature_names_in_", "design_feature_names_"))
        if input_features is not None:
            supplied = tuple(map(str, input_features))
            expected = tuple(map(str, self.feature_names_in_))
            if supplied != expected:
                raise ValueError(
                    f"input_features differ from fit: expected={expected}, supplied={supplied}"
                )
        return self.design_feature_names_.copy()

    @property
    def model_definition(self) -> dict[str, Any]:
        check_is_fitted(self, "model_definition_")
        return deepcopy(self.model_definition_)
