"""Fold-local feature and model components for performance optimization.

This module is intentionally independent from the registered incremental
analysis in :mod:`ink_tafel_study.evaluate`.  A feature recipe, its learned
transformations, a target coordinate, a model family, and concrete
hyperparameters together form one cloneable candidate pipeline.

Only explicitly registered source columns are read.  Identifier, grouping,
target, posterior, curve-fit, and provenance columns are never inferred as
predictors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    QuantileRegressor,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from .features import is_target_like_column, validate_no_target_leakage
from .models import GowerKNeighborsRegressor


BaseRepresentation = Literal["conditions", "context"]
SolventRepresentation = Literal[
    "none",
    "fractions",
    "ilr",
    "properties_raw",
    "properties_pca2",
    "properties_lean",
    "fractions_plus_properties_pca2",
]
NumericTransform = Literal["raw", "log"]
TargetTransform = Literal["identity", "log10"]


CATALYST_CATEGORICAL = (
    "catalyst_active_material_type",
    "catalyst_support",
)
CONDITION_NUMERIC = (
    "catalyst_loading_mg_cm2_selected",
    "electrolyte_molarity_M",
    "rpm",
    "scan_rate_mV_s",
)
CONDITION_CATEGORICAL = (
    "electrolyte_identity",
    "ir_correction_status",
)
INK_NUMERIC = (
    "catalyst_concentration_mg_mL",
    "ionomer_to_solvent_ratio",
    "ionomer_concentration_wt_percent",
    "conductive_agent_dose_mg_per_mgcat",
)
INK_CATEGORICAL = ("conductive_agent_present",)
SOLVENT_FRACTIONS = (
    "solvent_fraction_acetone",
    "solvent_fraction_water",
    "solvent_fraction_ethanol",
    "solvent_fraction_isopropanol",
    "solvent_fraction_methanol",
)
PROPERTY_CORE = (
    "solvent_mixture_property_proxy_viscosity_mPa_s",
    "solvent_mixture_property_proxy_surface_tension_mN_m",
    "solvent_mixture_property_proxy_density_kg_m3",
)
PROPERTY_DIELECTRIC_STRICT = (
    "solvent_mixture_property_proxy_dielectric_constant"
)
PROPERTY_DIELECTRIC_BEST = (
    "solvent_mixture_property_proxy_dielectric_constant_best_available"
)
PROPERTY_DIELECTRIC_FLAG = (
    "solvent_mixture_property_proxy_dielectric_constant_estimated_flag"
)
PROCESS_NUMERIC = (
    "sonication_time_min",
    "drop_volume_uL",
    "electrode_area_cm2",
)
PROCESS_CATEGORICAL = ("dispersion_method",)

# ``Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv`` retains evidence-layer names, whereas the
# active curated records use the shorter aliases below.  Normalization is a
# deterministic rename/copy, never a learned transformation.
CANONICAL_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "catalyst_support": ("catalyst_support", "catalyst_support_type"),
    PROPERTY_DIELECTRIC_BEST: (
        PROPERTY_DIELECTRIC_BEST,
        "solvent_mixture_property_proxy_dielectric_constant__best_available",
    ),
    PROPERTY_DIELECTRIC_FLAG: (
        PROPERTY_DIELECTRIC_FLAG,
        "solvent_mixture_property_proxy_dielectric_constant__best_available_is_estimated",
    ),
}


def normalize_optimization_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the active optimization aliases materialized."""

    normalized = frame.copy()
    for destination, aliases in CANONICAL_COLUMN_ALIASES.items():
        if destination in normalized:
            continue
        source = next((column for column in aliases if column in normalized), None)
        if source is not None:
            normalized[destination] = normalized[source]
    return normalized


@dataclass(frozen=True)
class FeatureRecipe:
    """One finite, auditable source/engineering recipe."""

    name: str
    base: BaseRepresentation
    include_ink_amounts: bool
    solvent_representation: SolventRepresentation
    include_property_amount_interactions: bool = False
    include_dielectric_estimated_flag: bool = False
    include_process_features: bool = False
    strict_source_backed_dielectric: bool = False
    sensitivity_only: bool = False
    complete_case_required: bool = False
    rare_min_frequency: int = 3
    variance_threshold: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.base not in {"conditions", "context"}:
            raise ValueError(f"Unsupported base representation: {self.base!r}")
        if self.solvent_representation not in {
            "none",
            "fractions",
            "ilr",
            "properties_raw",
            "properties_pca2",
            "properties_lean",
            "fractions_plus_properties_pca2",
        }:
            raise ValueError(
                f"Unsupported solvent representation: {self.solvent_representation!r}"
            )
        if self.include_property_amount_interactions and (
            not self.include_ink_amounts
            or self.solvent_representation
            not in {"properties_pca2", "fractions_plus_properties_pca2"}
        ):
            raise ValueError(
                "Property/amount interactions require ink amounts and property PCA2"
            )
        if self.include_dielectric_estimated_flag and self.strict_source_backed_dielectric:
            raise ValueError(
                "A best-available dielectric estimate flag cannot accompany strict dielectric values"
            )
        if self.complete_case_required and not self.strict_source_backed_dielectric:
            raise ValueError(
                "Complete-case recipes are reserved for strict source-backed dielectric sensitivity"
            )
        if self.rare_min_frequency < 1:
            raise ValueError("rare_min_frequency must be positive")
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold cannot be negative")


def _build_feature_recipes() -> dict[str, FeatureRecipe]:
    recipes: dict[str, FeatureRecipe] = {}
    for base in ("conditions", "context"):
        for include_ink in (False, True):
            ink_label = "ink" if include_ink else "noink"
            for solvent in (
                "none",
                "fractions",
                "ilr",
                "properties_raw",
                "properties_pca2",
                "properties_lean",
            ):
                name = f"{base}_{ink_label}_{solvent}"
                recipes[name] = FeatureRecipe(
                    name=name,
                    base=base,
                    include_ink_amounts=include_ink,
                    solvent_representation=solvent,
                )

    for base in ("conditions", "context"):
        name = f"{base}_ink_properties_pca2_interactions"
        recipes[name] = FeatureRecipe(
            name=name,
            base=base,
            include_ink_amounts=True,
            solvent_representation="properties_pca2",
            include_property_amount_interactions=True,
        )

    sensitivity = (
        FeatureRecipe(
            name="context_ink_fractions_plus_properties_pca2",
            base="context",
            include_ink_amounts=True,
            solvent_representation="fractions_plus_properties_pca2",
            sensitivity_only=True,
        ),
        FeatureRecipe(
            name="context_ink_properties_pca2_with_estimate_flag",
            base="context",
            include_ink_amounts=True,
            solvent_representation="properties_pca2",
            include_dielectric_estimated_flag=True,
            sensitivity_only=True,
        ),
        FeatureRecipe(
            name="context_ink_properties_pca2_interactions_with_estimate_flag",
            base="context",
            include_ink_amounts=True,
            solvent_representation="properties_pca2",
            include_property_amount_interactions=True,
            include_dielectric_estimated_flag=True,
            sensitivity_only=True,
        ),
        FeatureRecipe(
            name="context_ink_fractions_with_process",
            base="context",
            include_ink_amounts=True,
            solvent_representation="fractions",
            include_process_features=True,
            sensitivity_only=True,
        ),
    )
    recipes.update({recipe.name: recipe for recipe in sensitivity})
    return recipes


FEATURE_RECIPES: Mapping[str, FeatureRecipe] = _build_feature_recipes()


def get_feature_recipe(recipe: str | FeatureRecipe) -> FeatureRecipe:
    if isinstance(recipe, FeatureRecipe):
        return recipe
    try:
        return FEATURE_RECIPES[str(recipe)]
    except KeyError as error:
        raise KeyError(
            f"Unknown optimization feature recipe {recipe!r}; choose from {sorted(FEATURE_RECIPES)}"
        ) from error


def property_columns(recipe: str | FeatureRecipe) -> tuple[str, ...]:
    selected = get_feature_recipe(recipe)
    dielectric = (
        PROPERTY_DIELECTRIC_STRICT
        if selected.strict_source_backed_dielectric
        else PROPERTY_DIELECTRIC_BEST
    )
    return PROPERTY_CORE + (dielectric,)


def required_source_columns(recipe: str | FeatureRecipe) -> tuple[str, ...]:
    """Return the exact predictor allowlist required by ``recipe``."""

    selected = get_feature_recipe(recipe)
    columns: list[str] = list(CONDITION_NUMERIC + CONDITION_CATEGORICAL)
    if selected.base == "context":
        columns.extend(CATALYST_CATEGORICAL)
    if selected.include_ink_amounts:
        columns.extend(INK_NUMERIC + INK_CATEGORICAL)
    if selected.solvent_representation in {
        "fractions",
        "ilr",
        "fractions_plus_properties_pca2",
    }:
        columns.extend(SOLVENT_FRACTIONS)
    if selected.solvent_representation in {
        "properties_raw",
        "properties_pca2",
        "properties_lean",
        "fractions_plus_properties_pca2",
    }:
        columns.extend(property_columns(selected))
    if selected.include_dielectric_estimated_flag:
        columns.append(PROPERTY_DIELECTRIC_FLAG)
    if selected.include_process_features:
        columns.extend(PROCESS_NUMERIC + PROCESS_CATEGORICAL)
    result = tuple(dict.fromkeys(columns))
    validate_no_target_leakage(result)
    forbidden = sorted(column for column in result if is_target_like_column(column))
    if forbidden:
        raise AssertionError(f"Optimization recipe contains forbidden predictors: {forbidden}")
    return result


def feature_recipe_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recipe in FEATURE_RECIPES.values():
        row = asdict(recipe)
        row["required_source_columns"] = "|".join(required_source_columns(recipe))
        row["n_required_source_columns"] = len(required_source_columns(recipe))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


_LOG10_COLUMNS = {
    "catalyst_loading_mg_cm2_selected",
    "electrolyte_molarity_M",
    "scan_rate_mV_s",
    "catalyst_concentration_mg_mL",
    "solvent_mixture_property_proxy_viscosity_mPa_s",
}
_LOG1P_COLUMNS = {
    "rpm",
    "ionomer_to_solvent_ratio",
    "ionomer_concentration_wt_percent",
    "conductive_agent_dose_mg_per_mgcat",
    "sonication_time_min",
    "drop_volume_uL",
    "electrode_area_cm2",
}


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _safe_log10(values: pd.Series, *, column: str) -> pd.Series:
    negative = values < 0
    if bool(negative.fillna(False).any()):
        raise ValueError(f"Feature {column!r} contains negative values outside the log10 domain")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    positive = values > 0
    result.loc[positive] = np.log10(values.loc[positive])
    return result


def _safe_log1p(values: pd.Series, *, column: str) -> pd.Series:
    invalid = values < 0
    if bool(invalid.fillna(False).any()):
        raise ValueError(f"Feature {column!r} contains negative values outside the log1p domain")
    return np.log1p(values)


class RecipeFeatureBuilder(TransformerMixin, BaseEstimator):
    """Learn one feature recipe using only rows supplied to :meth:`fit`."""

    def __init__(
        self,
        recipe: str | FeatureRecipe,
        *,
        numeric_transform: NumericTransform = "raw",
    ) -> None:
        self.recipe = recipe
        self.numeric_transform = numeric_transform

    def _validated_input(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RecipeFeatureBuilder requires a pandas DataFrame")
        normalized = normalize_optimization_columns(X)
        missing = sorted(set(required_source_columns(self.recipe_)) - set(normalized.columns))
        if missing:
            raise KeyError(
                f"Feature recipe {self.recipe_.name!r} is missing source columns: {missing}"
            )
        return normalized

    def _property_input(self, frame: pd.DataFrame) -> np.ndarray:
        columns = property_columns(self.recipe_)
        result = (
            frame.loc[:, list(columns)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float, copy=True)
        )
        if self.numeric_transform == "log":
            viscosity = result[:, 0]
            if np.any(np.isfinite(viscosity) & (viscosity < 0)):
                raise ValueError("Viscosity proxy contains negative values outside the log10 domain")
            result[:, 0] = np.where(viscosity > 0, np.log10(viscosity), np.nan)
        return result

    @staticmethod
    def _learn_medians(values: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        return np.where(np.isfinite(medians), medians, 0.0)

    @staticmethod
    def _fill_array(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(values), values, medians[None, :])

    def _fit_property_transform(self, frame: pd.DataFrame) -> None:
        values = self._property_input(frame)
        self.property_medians_ = self._learn_medians(values)
        filled = self._fill_array(values, self.property_medians_)
        self.property_scaler_ = StandardScaler().fit(filled)
        standardized = self.property_scaler_.transform(filled)

        from sklearn.decomposition import PCA

        n_components = min(2, standardized.shape[1], standardized.shape[0])
        if n_components < 2:
            raise ValueError("Property PCA2 requires at least two training records")
        self.property_pca_ = PCA(n_components=2, svd_solver="full").fit(standardized)
        signs = np.ones(2, dtype=float)
        dielectric_index = standardized.shape[1] - 1
        if self.property_pca_.components_[0, dielectric_index] < 0:
            signs[0] = -1.0
        if self.property_pca_.components_[1, 0] < 0:
            signs[1] = -1.0
        self.property_pca_signs_ = signs

    def _fit_lean_property_transform(self, frame: pd.DataFrame) -> None:
        columns = property_columns(self.recipe_)
        polarity_columns = columns[1:]
        values = frame.loc[:, list(polarity_columns)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        self.polarity_medians_ = self._learn_medians(values)
        filled = self._fill_array(values, self.polarity_medians_)
        self.polarity_scaler_ = StandardScaler().fit(filled)
        standardized = self.polarity_scaler_.transform(filled)

        from sklearn.decomposition import PCA

        self.polarity_pca_ = PCA(n_components=1, svd_solver="full").fit(standardized)
        dielectric_index = standardized.shape[1] - 1
        self.polarity_sign_ = (
            -1.0 if self.polarity_pca_.components_[0, dielectric_index] < 0 else 1.0
        )

    def _fit_ilr(self, frame: pd.DataFrame) -> None:
        values = frame.loc[:, list(SOLVENT_FRACTIONS)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        if np.any(np.isfinite(values) & (values < 0)):
            raise ValueError("Solvent fractions must be nonnegative")
        positive = values[np.isfinite(values) & (values > 0)]
        self.ilr_delta_ = float(min(1.0e-3, 0.5 * positive.min())) if positive.size else 1.0e-6
        self.ilr_basis_ = helmert(len(SOLVENT_FRACTIONS), full=False).T

    def _transform_ilr(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(SOLVENT_FRACTIONS)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        if np.any(np.isfinite(values) & (values < 0)):
            raise ValueError("Solvent fractions must be nonnegative")
        values = np.where(np.isfinite(values), values, 0.0)
        row_sums = values.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("Every solvent composition must have a positive row sum")
        closed = values / row_sums[:, None]
        replaced = np.where(closed > 0, closed, self.ilr_delta_)
        replaced = replaced / replaced.sum(axis=1, keepdims=True)
        return np.log(replaced) @ self.ilr_basis_

    def _add_numeric(
        self, output: pd.DataFrame, frame: pd.DataFrame, column: str
    ) -> None:
        values = _numeric_series(frame, column)
        if self.numeric_transform == "log" and column in _LOG10_COLUMNS:
            output[f"log10__{column}"] = _safe_log10(values, column=column)
        elif self.numeric_transform == "log" and column in _LOG1P_COLUMNS:
            output[f"log1p__{column}"] = _safe_log1p(values, column=column)
        else:
            output[column] = values

    def _engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        for column in CONDITION_NUMERIC:
            self._add_numeric(output, frame, column)
        for column in CONDITION_CATEGORICAL:
            output[column] = frame[column].astype("object")
        if self.recipe_.base == "context":
            for column in CATALYST_CATEGORICAL:
                output[column] = frame[column].astype("object")
        if self.recipe_.include_ink_amounts:
            for column in INK_NUMERIC:
                self._add_numeric(output, frame, column)
            for column in INK_CATEGORICAL:
                output[column] = frame[column].astype("object")

        representation = self.recipe_.solvent_representation
        if representation in {"fractions", "fractions_plus_properties_pca2"}:
            for column in SOLVENT_FRACTIONS:
                output[column] = _numeric_series(frame, column)
        elif representation == "ilr":
            values = self._transform_ilr(frame)
            for index in range(values.shape[1]):
                output[f"solvent_ilr_{index + 1}"] = values[:, index]

        if representation == "properties_raw":
            for column in property_columns(self.recipe_):
                self._add_numeric(output, frame, column)
        elif representation in {"properties_pca2", "fractions_plus_properties_pca2"}:
            raw = self._property_input(frame)
            filled = self._fill_array(raw, self.property_medians_)
            scores = self.property_pca_.transform(
                self.property_scaler_.transform(filled)
            ) * self.property_pca_signs_[None, :]
            output["property_pc1"] = scores[:, 0]
            output["property_pc2"] = scores[:, 1]
        elif representation == "properties_lean":
            viscosity = _numeric_series(frame, PROPERTY_CORE[0])
            output["log10_property_viscosity"] = _safe_log10(
                viscosity, column=PROPERTY_CORE[0]
            )
            polarity_columns = property_columns(self.recipe_)[1:]
            raw = frame.loc[:, list(polarity_columns)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            filled = self._fill_array(raw, self.polarity_medians_)
            polarity = self.polarity_pca_.transform(
                self.polarity_scaler_.transform(filled)
            )[:, 0]
            output["property_polarity_coordinate"] = polarity * self.polarity_sign_

        if self.recipe_.include_property_amount_interactions:
            catalyst_column = (
                "log10__catalyst_concentration_mg_mL"
                if self.numeric_transform == "log"
                else "catalyst_concentration_mg_mL"
            )
            ratio_column = (
                "log1p__ionomer_to_solvent_ratio"
                if self.numeric_transform == "log"
                else "ionomer_to_solvent_ratio"
            )
            # Interactions are explicitly defined on log amounts regardless of
            # the main raw/log recipe coordinate.
            catalyst_log = _safe_log10(
                _numeric_series(frame, "catalyst_concentration_mg_mL"),
                column="catalyst_concentration_mg_mL",
            )
            ratio_log = _safe_log1p(
                _numeric_series(frame, "ionomer_to_solvent_ratio"),
                column="ionomer_to_solvent_ratio",
            )
            amounts = np.column_stack([catalyst_log, ratio_log])
            amounts = self._fill_array(amounts, self.interaction_amount_medians_)
            output["property_pc1_x_log_catalyst_concentration"] = (
                output["property_pc1"].to_numpy(float) * amounts[:, 0]
            )
            output["property_pc2_x_log_catalyst_concentration"] = (
                output["property_pc2"].to_numpy(float) * amounts[:, 0]
            )
            output["property_pc1_x_log_ionomer_solvent_ratio"] = (
                output["property_pc1"].to_numpy(float) * amounts[:, 1]
            )
            output["property_pc2_x_log_ionomer_solvent_ratio"] = (
                output["property_pc2"].to_numpy(float) * amounts[:, 1]
            )
            if catalyst_column not in output or ratio_column not in output:
                raise AssertionError("Interaction source amounts are absent from engineered output")

        if self.recipe_.include_dielectric_estimated_flag:
            output[PROPERTY_DIELECTRIC_FLAG] = _numeric_series(
                frame, PROPERTY_DIELECTRIC_FLAG
            )
        if self.recipe_.include_process_features:
            for column in PROCESS_NUMERIC:
                self._add_numeric(output, frame, column)
            for column in PROCESS_CATEGORICAL:
                output[column] = frame[column].astype("object")
        validate_no_target_leakage(output.columns)
        return output

    def fit(self, X: Any, y: Any = None) -> "RecipeFeatureBuilder":
        self.recipe_ = get_feature_recipe(self.recipe)
        if self.numeric_transform not in {"raw", "log"}:
            raise ValueError("numeric_transform must be 'raw' or 'log'")
        frame = self._validated_input(X)
        representation = self.recipe_.solvent_representation
        if representation == "ilr":
            self._fit_ilr(frame)
        if representation in {"properties_pca2", "fractions_plus_properties_pca2"}:
            self._fit_property_transform(frame)
        if representation == "properties_lean":
            self._fit_lean_property_transform(frame)
        if self.recipe_.include_property_amount_interactions:
            amounts = np.column_stack(
                [
                    _safe_log10(
                        _numeric_series(frame, "catalyst_concentration_mg_mL"),
                        column="catalyst_concentration_mg_mL",
                    ),
                    _safe_log1p(
                        _numeric_series(frame, "ionomer_to_solvent_ratio"),
                        column="ionomer_to_solvent_ratio",
                    ),
                ]
            )
            self.interaction_amount_medians_ = self._learn_medians(amounts)

        engineered = self._engineer(frame)
        numeric_columns = list(engineered.select_dtypes(include=[np.number]).columns)
        categorical_columns = [
            column for column in engineered.columns if column not in numeric_columns
        ]
        self.numeric_medians_ = {}
        for column in numeric_columns:
            values = pd.to_numeric(engineered[column], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            self.numeric_medians_[column] = float(np.median(finite)) if finite.size else 0.0

        self.frequent_categories_ = {}
        for column in categorical_columns:
            values = engineered[column].astype("object")
            normalized = values.where(values.notna(), "__MISSING__").astype(str)
            counts = normalized.value_counts(dropna=False)
            frequent = set(counts.loc[counts >= self.recipe_.rare_min_frequency].index.astype(str))
            if not frequent and len(counts):
                frequent = {str(counts.index[0])}
            self.frequent_categories_[column] = tuple(sorted(frequent))

        completed = self._complete(engineered)
        kept: list[str] = []
        for column in completed.columns:
            if column in numeric_columns:
                variance = float(np.var(completed[column].to_numpy(float)))
                if np.isfinite(variance) and variance > self.recipe_.variance_threshold:
                    kept.append(column)
            elif completed[column].astype(str).nunique(dropna=False) > 1:
                kept.append(column)
        if not kept:
            raise ValueError(
                f"Feature recipe {self.recipe_.name!r} has no non-constant training features"
            )
        self.feature_names_out_ = np.asarray(kept, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def _complete(self, engineered: pd.DataFrame) -> pd.DataFrame:
        output = engineered.copy()
        for column, median in self.numeric_medians_.items():
            output[column] = pd.to_numeric(output[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).fillna(median).astype(float)
        for column, levels in self.frequent_categories_.items():
            values = output[column].astype("object")
            normalized = values.where(values.notna(), "__MISSING__").astype(str)
            frequent = set(levels)
            output[column] = normalized.where(normalized.isin(frequent), "__RARE__")
        return output

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("RecipeFeatureBuilder must be fitted before transform")
        frame = self._validated_input(X)
        engineered = self._engineer(frame)
        fitted_columns = set(self.numeric_medians_) | set(self.frequent_categories_)
        missing = sorted(fitted_columns - set(engineered))
        if missing:
            raise AssertionError(f"Engineered feature schema changed after fitting: {missing}")
        return self._complete(engineered).loc[:, list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("RecipeFeatureBuilder must be fitted before feature names are available")
        return self.feature_names_out_.copy()


class TargetTransformRegressor(RegressorMixin, BaseEstimator):
    """Train in identity/log10 coordinates and always predict original units."""

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        transform: TargetTransform = "identity",
        standardize: bool = True,
    ) -> None:
        self.estimator = estimator
        self.transform = transform
        self.standardize = standardize

    def fit(self, X: Any, y: Any) -> "TargetTransformRegressor":
        target = np.asarray(y, dtype=float).reshape(-1)
        if not np.isfinite(target).all():
            raise ValueError("Optimization target contains NaN or infinite values")
        if self.transform == "log10":
            if np.any(target <= 0):
                raise ValueError("log10 target transform requires strictly positive slopes")
            transformed = np.log10(target)
        elif self.transform == "identity":
            transformed = target.copy()
        else:
            raise ValueError("transform must be 'identity' or 'log10'")
        self.y_mean_ = float(np.mean(transformed)) if self.standardize else 0.0
        scale = float(np.std(transformed, ddof=0)) if self.standardize else 1.0
        self.y_scale_ = scale if np.isfinite(scale) and scale > 0 else 1.0
        standardized = (transformed - self.y_mean_) / self.y_scale_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, standardized)
        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("TargetTransformRegressor must be fitted before predict")
        prediction = np.asarray(self.estimator_.predict(X), dtype=float).reshape(-1)
        transformed = prediction * self.y_scale_ + self.y_mean_
        if self.transform == "log10":
            return np.power(10.0, np.clip(transformed, -12.0, 12.0))
        return transformed


@dataclass(frozen=True)
class OptimizationModelSpec:
    name: str
    estimator: BaseEstimator
    param_grid: Mapping[str, Sequence[Any]]
    complexity_rank: int
    optional_dependency: str | None = None


def make_candidate_regressor_specs(
    *,
    random_state: int = 42,
    include_catboost: bool = True,
) -> tuple[dict[str, OptimizationModelSpec], dict[str, str]]:
    """Return the frozen model family/search space and optional skips."""

    specs = {
        "dummy_mean": OptimizationModelSpec(
            "dummy_mean", DummyRegressor(strategy="mean"), {}, 0
        ),
        "dummy_median": OptimizationModelSpec(
            "dummy_median", DummyRegressor(strategy="median"), {}, 0
        ),
        "bayesian_ridge": OptimizationModelSpec(
            "bayesian_ridge", BayesianRidge(compute_score=True), {}, 1
        ),
        "ridge": OptimizationModelSpec(
            "ridge", Ridge(), {"alpha": (0.1, 1.0, 10.0, 100.0)}, 1
        ),
        "elastic_net": OptimizationModelSpec(
            "elastic_net",
            ElasticNet(max_iter=20_000, random_state=int(random_state)),
            {
                "alpha": (0.01, 0.1, 1.0, 10.0),
                "l1_ratio": (0.05, 0.25, 0.5, 0.75, 0.95),
            },
            2,
        ),
        "pls": OptimizationModelSpec(
            "pls",
            PLSRegression(scale=False, max_iter=1_000),
            {"n_components": (1, 2, 3, 4)},
            2,
        ),
        "huber": OptimizationModelSpec(
            "huber",
            HuberRegressor(max_iter=1_000),
            {"alpha": (1.0e-4, 1.0e-2, 1.0e-1), "epsilon": (1.2, 1.35, 1.5)},
            2,
        ),
        "quantile_median": OptimizationModelSpec(
            "quantile_median",
            QuantileRegressor(quantile=0.5, solver="highs"),
            {"alpha": (0.01, 0.1, 1.0)},
            2,
        ),
        "rbf_svr": OptimizationModelSpec(
            "rbf_svr",
            SVR(kernel="rbf"),
            {
                "C": (0.1, 1.0, 10.0, 100.0),
                "gamma": ("scale", 0.03, 0.1, 0.3, 1.0),
                "epsilon": (0.05, 0.1, 0.2),
            },
            3,
        ),
        "gower_knn": OptimizationModelSpec(
            "gower_knn",
            GowerKNeighborsRegressor(),
            {"n_neighbors": (3, 5, 7, 9, 15, 21), "distance_power": (1.0, 2.0)},
            3,
        ),
        "extra_trees": OptimizationModelSpec(
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=int(random_state),
                n_jobs=1,
            ),
            {
                "max_depth": (2, 3, 4, 6),
                "min_samples_leaf": (3, 5, 8),
                "max_features": (0.5, 0.8, 1.0),
            },
            3,
        ),
        "random_forest": OptimizationModelSpec(
            "random_forest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=int(random_state),
                n_jobs=1,
            ),
            {
                "max_depth": (2, 3, 4, 6),
                "min_samples_leaf": (3, 5, 8),
                "max_features": (0.5, 0.8, 1.0),
            },
            3,
        ),
    }
    skipped: dict[str, str] = {}
    if include_catboost:
        if importlib.util.find_spec("catboost") is None:
            skipped["catboost"] = "optional dependency unavailable; install exploratory-ml extra"
        else:
            from catboost import CatBoostRegressor

            specs["catboost"] = OptimizationModelSpec(
                "catboost",
                CatBoostRegressor(
                    loss_function="RMSE",
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1,
                    random_seed=int(random_state),
                ),
                {
                    "depth": (2, 3, 4),
                    "iterations": (200, 400),
                    "learning_rate": (0.03, 0.07),
                    "l2_leaf_reg": (10.0, 30.0),
                },
                4,
                optional_dependency="catboost",
            )
    return specs, skipped


def _dense_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant", fill_value="__MISSING__", keep_empty_features=True
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_optimization_pipeline(
    recipe: str | FeatureRecipe | None,
    model_spec: OptimizationModelSpec,
    *,
    target_transform: TargetTransform,
    numeric_transform: NumericTransform = "raw",
    model_params: Mapping[str, Any] | None = None,
) -> BaseEstimator:
    """Build one atomic candidate whose predictions are in mV dec-1."""

    regressor = clone(model_spec.estimator)
    if model_params:
        regressor.set_params(**dict(model_params))
    wrapped = TargetTransformRegressor(
        regressor,
        transform=target_transform,
        standardize=True,
    )
    if model_spec.name.startswith("dummy_"):
        return wrapped
    if recipe is None:
        raise ValueError(f"Model {model_spec.name!r} requires a feature recipe")
    builder = RecipeFeatureBuilder(recipe, numeric_transform=numeric_transform)
    if model_spec.name == "gower_knn":
        return Pipeline([("features", builder), ("target", wrapped)])
    return Pipeline(
        [
            ("features", builder),
            ("preprocess", _dense_preprocessor()),
            ("near_zero_variance", VarianceThreshold(0.0)),
            ("target", wrapped),
        ]
    )


def available_model_registry(
    *, random_state: int = 42, include_catboost: bool = True
) -> tuple[pd.DataFrame, dict[str, str]]:
    specs, skipped = make_candidate_regressor_specs(
        random_state=random_state, include_catboost=include_catboost
    )
    rows = []
    for spec in specs.values():
        rows.append(
            {
                "model": spec.name,
                "complexity_rank": spec.complexity_rank,
                "optional_dependency": spec.optional_dependency,
                "parameter_grid": repr(dict(spec.param_grid)),
            }
        )
    return pd.DataFrame(rows).sort_values(["complexity_rank", "model"]), skipped
