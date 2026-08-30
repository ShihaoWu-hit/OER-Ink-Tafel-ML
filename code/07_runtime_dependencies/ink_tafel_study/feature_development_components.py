"""Frozen, fold-local features for the 97-record RBF-SVR development stage.

The feature ladder in this module is intentionally separate from both the
registered incremental analysis and the broad performance-optimization
catalogue.  Every recipe is an explicit allowlist.  Raw solvent identities
and raw solvent-1/solvent-2 volumes are never predictors; aggregate liquid
volume is read only to construct intensive descriptors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from .features import is_target_like_column, validate_no_target_leakage
from .optimization_components import TargetTransformRegressor


Representation = Literal[
    "none",
    "composition_summary",
    "composition_ilr",
    "property_lean",
    "property_physics",
    "fractions_plus_properties",
]

CONTEXT_NUMERIC = (
    "catalyst_loading_mg_cm2_selected",
    "electrolyte_molarity_M",
    "rpm",
    "scan_rate_mV_s",
)
CONTEXT_CATEGORICAL = (
    "catalyst_type",
    "electrolyte_identity",
    "ir_correction_status",
)
RECORDED_NUMERIC = (
    "catalyst_concentration_mg_mL",
    "ionomer_to_solvent_ratio",
    "ionomer_concentration_wt_percent",
    "conductive_agent_dose_mg_per_mgcat",
)
RECORDED_CATEGORICAL = (
    "conductive_agent_present",
    "conductive_agent_class",
)
INTENSIVE_SOURCES = (
    "catalyst_mass_mg",
    "conductive_agent_mass_mg",
    "total_liquid_volume_uL",
    "drop_volume_uL",
    "ionomer_volume_uL_per_mgcat",
)
SOLVENT_FRACTIONS = (
    "solvent_fraction_acetone",
    "solvent_fraction_water",
    "solvent_fraction_ethanol",
    "solvent_fraction_isopropanol",
    "solvent_fraction_methanol",
)
PROPERTY_COLUMNS = (
    "solvent_mixture_property_proxy_viscosity_mPa_s",
    "solvent_mixture_property_proxy_surface_tension_mN_m",
    "solvent_mixture_property_proxy_density_kg_m3",
    "solvent_mixture_property_proxy_dielectric_constant_best_available",
)
PROCESS_NUMERIC = (
    "sonication_time_min",
    "drop_volume_uL",
    "drying_temperature_C",
    "drying_time_min",
)
PROCESS_CATEGORICAL = ("dispersion_method", "drying_process")
BATCH_MASS_COLUMNS = ("catalyst_mass_mg", "conductive_agent_mass_mg")

RAW_SOLVENT_PREDICTORS = frozenset(
    {
        "solvent_1_type",
        "solvent_2_type",
        "solvent_1_volume_uL",
        "solvent_2_volume_uL",
        "solvent_1_volume_uL_per_mgcat",
        "solvent_2_volume_uL_per_mgcat",
        "solvent_1_volume_uL_source_raw",
        "solvent_2_volume_uL_source_raw",
    }
)

_ALIASES: Mapping[str, tuple[str, ...]] = {
    "catalyst_type": ("catalyst_type", "catalyst_active_material_type"),
    PROPERTY_COLUMNS[-1]: (
        PROPERTY_COLUMNS[-1],
        "solvent_mixture_property_proxy_dielectric_constant__best_available",
    ),
}


@dataclass(frozen=True)
class FeatureDevelopmentRecipe:
    name: str
    complexity_rank: int
    include_recorded_amounts: bool = False
    include_intensive_amounts: bool = False
    representation: Representation = "none"
    include_property_interactions: bool = False
    include_process_features: bool = False
    include_batch_masses: bool = False
    sensitivity_only: bool = False
    rare_min_frequency: int = 3
    # Use exactly-zero removal here.  An absolute tolerance would incorrectly
    # discard kinematic viscosity simply because its SI values are ~1e-6.
    variance_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.include_intensive_amounts and not self.include_recorded_amounts:
            raise ValueError("Intensive recipes require recorded ink amounts")
        if self.include_property_interactions and self.representation != "property_physics":
            raise ValueError("Property interactions require property_physics")
        if self.rare_min_frequency < 1:
            raise ValueError("rare_min_frequency must be positive")

    @property
    def blocks(self) -> tuple[str, ...]:
        result = ["context_core"]
        if self.include_recorded_amounts:
            result.append("recorded_amounts")
        if self.include_intensive_amounts:
            result.append("si_intensive_amounts")
        if self.representation == "composition_summary":
            result.append("si_composition_summary")
        elif self.representation == "composition_ilr":
            result.append("composition_ilr")
        elif self.representation == "property_lean":
            result.append("property_lean")
        elif self.representation == "property_physics":
            result.append("property_physics")
        elif self.representation == "fractions_plus_properties":
            result.extend(("solvent_fractions_sensitivity", "property_physics"))
        if self.include_property_interactions:
            result.append("property_interactions")
        if self.include_process_features:
            result.append("process_features")
        if self.include_batch_masses:
            result.append("batch_masses")
        return tuple(result)


def _build_recipes() -> dict[str, FeatureDevelopmentRecipe]:
    primary = (
        FeatureDevelopmentRecipe("context_core", 1),
        FeatureDevelopmentRecipe("context_plus_recorded_amounts", 2, True),
        FeatureDevelopmentRecipe("si_intensive_amounts", 3, True, True),
        FeatureDevelopmentRecipe(
            "si_composition_summary", 4, True, True, "composition_summary"
        ),
        FeatureDevelopmentRecipe("composition_ilr", 5, True, True, "composition_ilr"),
        FeatureDevelopmentRecipe("property_lean", 6, True, True, "property_lean"),
        FeatureDevelopmentRecipe("property_physics", 7, True, True, "property_physics"),
        FeatureDevelopmentRecipe(
            "property_interactions", 8, True, True, "property_physics", True
        ),
    )
    sensitivity = (
        FeatureDevelopmentRecipe(
            "sensitivity_fractions_plus_properties",
            9,
            True,
            True,
            "fractions_plus_properties",
            sensitivity_only=True,
        ),
        FeatureDevelopmentRecipe(
            "sensitivity_process_features",
            9,
            True,
            True,
            include_process_features=True,
            sensitivity_only=True,
        ),
        FeatureDevelopmentRecipe(
            "sensitivity_batch_masses",
            9,
            True,
            True,
            include_batch_masses=True,
            sensitivity_only=True,
        ),
    )
    return {recipe.name: recipe for recipe in primary + sensitivity}


FEATURE_DEVELOPMENT_RECIPES: Mapping[str, FeatureDevelopmentRecipe] = _build_recipes()
PRIMARY_FEATURE_DEVELOPMENT_RECIPE_NAMES = tuple(
    name for name, recipe in FEATURE_DEVELOPMENT_RECIPES.items() if not recipe.sensitivity_only
)
SENSITIVITY_FEATURE_DEVELOPMENT_RECIPE_NAMES = tuple(
    name for name, recipe in FEATURE_DEVELOPMENT_RECIPES.items() if recipe.sensitivity_only
)


def get_feature_development_recipe(
    recipe: str | FeatureDevelopmentRecipe,
) -> FeatureDevelopmentRecipe:
    if isinstance(recipe, FeatureDevelopmentRecipe):
        return recipe
    try:
        return FEATURE_DEVELOPMENT_RECIPES[str(recipe)]
    except KeyError as error:
        raise KeyError(
            f"Unknown feature-development recipe {recipe!r}; choose from "
            f"{sorted(FEATURE_DEVELOPMENT_RECIPES)}"
        ) from error


def normalize_feature_development_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialize stable aliases and aggregate volume without reading raw solvents."""

    result = frame.copy()
    for destination, aliases in _ALIASES.items():
        if destination not in result:
            source = next((name for name in aliases if name in result), None)
            if source is not None:
                result[destination] = result[source]
    if (
        "total_liquid_volume_uL" not in result
        and "total_liquid_volume_uL_per_mgcat" in result
        and "catalyst_mass_mg" in result
    ):
        result["total_liquid_volume_uL"] = (
            pd.to_numeric(result["total_liquid_volume_uL_per_mgcat"], errors="coerce")
            * pd.to_numeric(result["catalyst_mass_mg"], errors="coerce")
        )
    return result


def required_feature_development_source_columns(
    recipe: str | FeatureDevelopmentRecipe,
) -> tuple[str, ...]:
    selected = get_feature_development_recipe(recipe)
    columns = list(CONTEXT_NUMERIC + CONTEXT_CATEGORICAL)
    if selected.include_recorded_amounts:
        columns.extend(RECORDED_NUMERIC + RECORDED_CATEGORICAL)
    if selected.include_intensive_amounts:
        columns.extend(INTENSIVE_SOURCES)
    if selected.representation in {
        "composition_summary",
        "composition_ilr",
        "fractions_plus_properties",
    }:
        columns.extend(SOLVENT_FRACTIONS)
    if selected.representation in {
        "property_lean",
        "property_physics",
        "fractions_plus_properties",
    }:
        columns.extend(PROPERTY_COLUMNS)
    if selected.include_process_features:
        columns.extend(PROCESS_NUMERIC + PROCESS_CATEGORICAL)
    if selected.include_batch_masses:
        columns.extend(BATCH_MASS_COLUMNS)
    output = tuple(dict.fromkeys(columns))
    validate_no_target_leakage(output)
    if RAW_SOLVENT_PREDICTORS.intersection(output):
        raise AssertionError("Raw solvent type/volume fields cannot be predictors")
    forbidden = [name for name in output if is_target_like_column(name)]
    if forbidden:
        raise AssertionError(f"Target-like predictors are forbidden: {forbidden}")
    return output


# Backward-compatible spelling used by the feature-development runner draft.
feature_development_required_source_columns = required_feature_development_source_columns


_BLOCK_COLUMNS = (
    "context_core",
    "recorded_amounts",
    "si_intensive_amounts",
    "si_composition_summary",
    "composition_ilr",
    "solvent_fractions_sensitivity",
    "property_lean",
    "property_physics",
    "property_interactions",
    "process_features",
    "batch_masses",
)

_NOMINAL_FEATURE_COUNTS = {
    "context_core": 7,
    "context_plus_recorded_amounts": 13,
    "si_intensive_amounts": 20,
    "si_composition_summary": 25,
    "composition_ilr": 24,
    "property_lean": 22,
    "property_physics": 23,
    "property_interactions": 28,
    "sensitivity_fractions_plus_properties": 28,
    "sensitivity_process_features": 26,
    "sensitivity_batch_masses": 22,
}


def feature_development_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recipe in FEATURE_DEVELOPMENT_RECIPES.values():
        row = asdict(recipe)
        row["feature_complexity"] = _NOMINAL_FEATURE_COUNTS[recipe.name]
        row["blocks"] = "|".join(recipe.blocks)
        row["required_source_columns"] = "|".join(
            required_feature_development_source_columns(recipe)
        )
        for block in _BLOCK_COLUMNS:
            row[f"block_{block}"] = block in recipe.blocks
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["sensitivity_only", "complexity_rank", "name"]
    ).reset_index(drop=True)


feature_development_recipe_registry = feature_development_registry


@dataclass(frozen=True)
class FeatureDefinition:
    feature_name: str
    block: str
    kind: Literal["numeric", "categorical"]
    unit: str
    formula: str
    source_columns: tuple[str, ...]
    fold_local_operation: str = "none"
    data_source: str = "Data/01_raw/formulation_source.xlsx Sheet1 -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"


def _definitions() -> tuple[FeatureDefinition, ...]:
    direct: list[FeatureDefinition] = []
    units = {
        "catalyst_loading_mg_cm2_selected": "mg cm^-2",
        "electrolyte_molarity_M": "mol L^-1",
        "rpm": "min^-1",
        "scan_rate_mV_s": "mV s^-1",
        "catalyst_concentration_mg_mL": "mg mL^-1",
        "ionomer_to_solvent_ratio": "dimensionless",
        "ionomer_concentration_wt_percent": "wt%",
        "conductive_agent_dose_mg_per_mgcat": "mg mg_cat^-1",
    }
    for name in CONTEXT_NUMERIC:
        direct.append(FeatureDefinition(name, "context_core", "numeric", units[name], name, (name,)))
    for name in CONTEXT_CATEGORICAL:
        data_source = (
            "Data/01_raw/formulation_source.xlsx Sheet2 reviewed iR evidence -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"
            if name == "ir_correction_status"
            else "Data/01_raw/formulation_source.xlsx Sheet1 -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"
        )
        direct.append(
            FeatureDefinition(
                name,
                "context_core",
                "categorical",
                "category",
                name,
                (name,),
                "fold-local rare-level pooling",
                data_source,
            )
        )
    for name in RECORDED_NUMERIC:
        direct.append(FeatureDefinition(name, "recorded_amounts", "numeric", units[name], name, (name,)))
    for name in RECORDED_CATEGORICAL:
        direct.append(FeatureDefinition(name, "recorded_amounts", "categorical", "category", name, (name,), "fold-local rare-level pooling"))
    derived = [
        FeatureDefinition("log10_solid_concentration_mg_mL", "si_intensive_amounts", "numeric", "dimensionless", "log10(((catalyst_mass_mg + conductive_agent_mass_mg) / (total_liquid_volume_uL / 1000)) / (1 mg mL^-1))", ("catalyst_mass_mg", "conductive_agent_mass_mg", "total_liquid_volume_uL")),
        FeatureDefinition("ink_dilution_uL_mgsolid", "si_intensive_amounts", "numeric", "uL mg_solid^-1", "total_liquid_volume_uL / (catalyst_mass_mg + conductive_agent_mass_mg)", ("total_liquid_volume_uL", "catalyst_mass_mg", "conductive_agent_mass_mg")),
        FeatureDefinition("deposited_catalyst_dose_mg", "si_intensive_amounts", "numeric", "mg", "catalyst_concentration_mg_mL * drop_volume_uL / 1000", ("catalyst_concentration_mg_mL", "drop_volume_uL")),
        FeatureDefinition("deposited_solid_dose_mg", "si_intensive_amounts", "numeric", "mg", "solid_concentration_mg_mL * drop_volume_uL / 1000", ("catalyst_mass_mg", "conductive_agent_mass_mg", "total_liquid_volume_uL", "drop_volume_uL")),
        FeatureDefinition("ionomer_solution_dose_uL_mgcat", "si_intensive_amounts", "numeric", "uL mg_cat^-1", "ionomer_volume_uL_per_mgcat", ("ionomer_volume_uL_per_mgcat",)),
        FeatureDefinition("log10_catalyst_loading_mg_cm2", "si_intensive_amounts", "numeric", "dimensionless", "log10(catalyst_loading_mg_cm2_selected / (1 mg cm^-2))", ("catalyst_loading_mg_cm2_selected",)),
        FeatureDefinition("log10_total_solid_loading_mg_cm2", "si_intensive_amounts", "numeric", "dimensionless", "log10(catalyst_loading_mg_cm2_selected * (1 + conductive_agent_dose_mg_per_mgcat) / (1 mg cm^-2))", ("catalyst_loading_mg_cm2_selected", "conductive_agent_dose_mg_per_mgcat")),
        FeatureDefinition("alcohol_fraction", "si_composition_summary", "numeric", "0-1", "F_ethanol + F_isopropanol + F_methanol", SOLVENT_FRACTIONS),
        FeatureDefinition("organic_fraction", "si_composition_summary", "numeric", "0-1", "F_acetone + F_ethanol + F_isopropanol + F_methanol", SOLVENT_FRACTIONS),
        FeatureDefinition("log10_water_to_alcohol_ratio", "si_composition_summary", "numeric", "dimensionless", "log10((F_water + delta_fold) / (F_alcohol + delta_fold))", SOLVENT_FRACTIONS, "delta learned on training fold"),
        FeatureDefinition("n_solvents_present", "si_composition_summary", "numeric", "count", "sum(1[F_i > 0])", SOLVENT_FRACTIONS),
        FeatureDefinition("solvent_entropy", "si_composition_summary", "numeric", "dimensionless", "-sum(F_i * ln(F_i)) for F_i > 0", SOLVENT_FRACTIONS),
    ]
    for index in range(4):
        derived.append(FeatureDefinition(f"solvent_ilr_{index + 1}", "composition_ilr", "numeric", "dimensionless", "ILR coordinate of closed five-solvent composition", SOLVENT_FRACTIONS, "zero replacement learned on training fold"))
    derived.extend(
        [
            FeatureDefinition("log10_solvent_viscosity_mPa_s", "property_lean", "numeric", "dimensionless", "log10(viscosity / (1 mPa s))", (PROPERTY_COLUMNS[0],), data_source="code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
            FeatureDefinition("solvent_polarity_coordinate", "property_lean", "numeric", "dimensionless", "PCA1 of standardized surface tension, density, and dielectric", PROPERTY_COLUMNS[1:], "median, scaling and PCA fitted on training fold", "code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
            FeatureDefinition("kinematic_viscosity_m2_s", "property_physics", "numeric", "m^2 s^-1", "(viscosity_mPa_s * 1e-3) / density_kg_m3", (PROPERTY_COLUMNS[0], PROPERTY_COLUMNS[2]), data_source="code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
            FeatureDefinition("capillary_flow_proxy_m_s", "property_physics", "numeric", "m s^-1", "(surface_tension_mN_m * 1e-3) / (viscosity_mPa_s * 1e-3)", (PROPERTY_COLUMNS[1], PROPERTY_COLUMNS[0]), data_source="code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
            FeatureDefinition("solvent_polarity_coordinate", "property_physics", "numeric", "dimensionless", "PCA1 of standardized surface tension, density, and dielectric", PROPERTY_COLUMNS[1:], "median, scaling and PCA fitted on training fold", "code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
            FeatureDefinition("log10_solvent_viscosity_mPa_s", "property_interactions", "numeric", "dimensionless", "log10(viscosity / (1 mPa s)); retained as the interaction main effect", (PROPERTY_COLUMNS[0],), data_source="code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
            FeatureDefinition("catalyst_type_[level]_x_log10_viscosity", "property_interactions", "numeric", "dimensionless", "one output per non-reference fold-local rare-pooled catalyst-type treatment contrast * log10(viscosity_mPa_s)", ("catalyst_type", PROPERTY_COLUMNS[0]), "rare pooling and contrast levels fitted on training fold", "Data/01_raw/formulation_source.xlsx Sheet1 + code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
            FeatureDefinition("catalyst_type_[level]_x_polarity_coordinate", "property_interactions", "numeric", "dimensionless", "one output per non-reference fold-local rare-pooled catalyst-type treatment contrast * polarity_coordinate", ("catalyst_type",) + PROPERTY_COLUMNS[1:], "rare pooling, contrast levels and polarity coordinate fitted on training fold", "Data/01_raw/formulation_source.xlsx Sheet1 + code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
            FeatureDefinition("log10_viscosity_x_log10_solid_concentration", "property_interactions", "numeric", "dimensionless", "log10(viscosity_mPa_s) * log10(solid_concentration_mg_mL)", (PROPERTY_COLUMNS[0], "catalyst_mass_mg", "conductive_agent_mass_mg", "total_liquid_volume_uL"), data_source="Data/01_raw/formulation_source.xlsx Sheet1 + code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
            FeatureDefinition("capillary_flow_x_log10_ionomer_solvent_ratio", "property_interactions", "numeric", "m s^-1", "capillary_flow_proxy_m_s * log10(replace_zero(ionomer_to_solvent_ratio, delta_fold))", (PROPERTY_COLUMNS[1], PROPERTY_COLUMNS[0], "ionomer_to_solvent_ratio"), "zero-replacement delta fitted on training fold", "Data/01_raw/formulation_source.xlsx Sheet1 + code/Ink_Tafel_Study/config/property_*.csv -> Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv"),
        ]
    )
    for name in SOLVENT_FRACTIONS:
        derived.append(FeatureDefinition(name, "solvent_fractions_sensitivity", "numeric", "0-1", name, (name,)))
    for name in PROCESS_NUMERIC:
        unit = {"sonication_time_min": "min", "drop_volume_uL": "uL", "drying_temperature_C": "degC", "drying_time_min": "min"}[name]
        derived.append(FeatureDefinition(name, "process_features", "numeric", unit, name, (name,)))
    for name in PROCESS_CATEGORICAL:
        derived.append(FeatureDefinition(name, "process_features", "categorical", "category", name, (name,), "fold-local rare-level pooling"))
    for name in BATCH_MASS_COLUMNS:
        derived.append(FeatureDefinition(name, "batch_masses", "numeric", "mg", name, (name,)))
    return tuple(direct + derived)


def feature_definition_registry() -> pd.DataFrame:
    return pd.DataFrame(asdict(item) for item in _definitions()).assign(
        source_columns=lambda value: value["source_columns"].map("|".join)
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _positive_log10(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array) & (array > 0), np.log10(array), np.nan)


class FeatureDevelopmentBuilder(TransformerMixin, BaseEstimator):
    """Cloneable recipe transformer with all learned state confined to ``fit``."""

    def __init__(self, recipe: str | FeatureDevelopmentRecipe) -> None:
        self.recipe = recipe

    def _validated(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureDevelopmentBuilder requires a pandas DataFrame")
        frame = normalize_feature_development_columns(X)
        missing = sorted(
            set(required_feature_development_source_columns(self.recipe_)) - set(frame.columns)
        )
        if missing:
            raise KeyError(f"Feature recipe {self.recipe_.name!r} is missing: {missing}")
        return frame

    def _closed_composition(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(SOLVENT_FRACTIONS)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        if np.any(np.isfinite(values) & (values < 0)):
            raise ValueError("Solvent fractions must be nonnegative")
        values = np.where(np.isfinite(values), values, 0.0)
        totals = values.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("Every solvent composition must have a positive row sum")
        return values / totals[:, None]

    def _fit_composition(self, frame: pd.DataFrame) -> None:
        values = self._closed_composition(frame)
        positive = values[values > 0]
        self.composition_delta_ = (
            float(min(1.0e-3, 0.5 * positive.min())) if positive.size else 1.0e-6
        )
        self.ilr_basis_ = helmert(len(SOLVENT_FRACTIONS), full=False).T

    @staticmethod
    def _medians(values: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        return np.where(np.isfinite(medians), medians, 0.0)

    def _fit_polarity(self, frame: pd.DataFrame) -> None:
        values = frame.loc[:, list(PROPERTY_COLUMNS[1:])].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        self.polarity_medians_ = self._medians(values)
        filled = np.where(np.isfinite(values), values, self.polarity_medians_[None, :])
        self.polarity_scaler_ = StandardScaler().fit(filled)
        standardized = self.polarity_scaler_.transform(filled)
        self.polarity_pca_ = PCA(n_components=1, svd_solver="full").fit(standardized)
        self.polarity_sign_ = (
            -1.0 if self.polarity_pca_.components_[0, -1] < 0 else 1.0
        )

    def _polarity(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(PROPERTY_COLUMNS[1:])].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        filled = np.where(np.isfinite(values), values, self.polarity_medians_[None, :])
        return (
            self.polarity_pca_.transform(self.polarity_scaler_.transform(filled))[:, 0]
            * self.polarity_sign_
        )

    def _fit_interactions(self, frame: pd.DataFrame) -> None:
        catalyst = frame["catalyst_type"].where(
            frame["catalyst_type"].notna(), "__MISSING__"
        ).astype(str)
        counts = catalyst.value_counts(dropna=False)
        frequent = set(
            counts[counts >= self.recipe_.rare_min_frequency].index.astype(str)
        )
        if not frequent and len(counts):
            frequent = {str(counts.index[0])}
        pooled = catalyst.where(catalyst.isin(frequent), "__RARE__")
        levels = tuple(sorted(pooled.unique().astype(str)))
        if not levels:
            levels = ("__MISSING__",)
        self.catalyst_interaction_levels_ = levels
        self.catalyst_interaction_reference_ = levels[0]
        ratio = _numeric(frame, "ionomer_to_solvent_ratio").to_numpy(float)
        if np.any(np.isfinite(ratio) & (ratio < 0)):
            raise ValueError("Ionomer/solvent ratio must be nonnegative")
        positive = ratio[np.isfinite(ratio) & (ratio > 0)]
        self.ionomer_ratio_median_ = (
            float(np.median(positive)) if positive.size else 1.0
        )
        self.ionomer_ratio_delta_ = (
            float(min(1.0e-3, 0.5 * positive.min()))
            if positive.size
            else 1.0e-3
        )

    @staticmethod
    def _level_slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "missing"

    def _pooled_catalyst(self, frame: pd.DataFrame) -> pd.Series:
        values = frame["catalyst_type"].where(
            frame["catalyst_type"].notna(), "__MISSING__"
        ).astype(str)
        fitted = set(self.catalyst_interaction_levels_)
        return values.where(values.isin(fitted), "__RARE__")

    def _intensive(self, frame: pd.DataFrame, output: pd.DataFrame) -> None:
        cat_mass = _numeric(frame, "catalyst_mass_mg").to_numpy(float)
        cond_mass = _numeric(frame, "conductive_agent_mass_mg").to_numpy(float)
        liquid = _numeric(frame, "total_liquid_volume_uL").to_numpy(float)
        drop = _numeric(frame, "drop_volume_uL").to_numpy(float)
        solid_mass = cat_mass + cond_mass
        valid_liquid = np.isfinite(liquid) & (liquid > 0)
        valid_solid = np.isfinite(solid_mass) & (solid_mass > 0)
        solid_concentration = np.full(len(frame), np.nan)
        solid_concentration[valid_liquid] = (
            1000.0 * solid_mass[valid_liquid] / liquid[valid_liquid]
        )
        dilution = np.full(len(frame), np.nan)
        dilution[valid_solid] = liquid[valid_solid] / solid_mass[valid_solid]
        output["log10_solid_concentration_mg_mL"] = _positive_log10(solid_concentration)
        output["ink_dilution_uL_mgsolid"] = dilution
        output["deposited_catalyst_dose_mg"] = (
            _numeric(frame, "catalyst_concentration_mg_mL").to_numpy(float) * drop / 1000.0
        )
        output["deposited_solid_dose_mg"] = solid_concentration * drop / 1000.0
        output["ionomer_solution_dose_uL_mgcat"] = _numeric(
            frame, "ionomer_volume_uL_per_mgcat"
        )
        loading = _numeric(frame, "catalyst_loading_mg_cm2_selected").to_numpy(float)
        ratio = _numeric(frame, "conductive_agent_dose_mg_per_mgcat").to_numpy(float)
        output["log10_catalyst_loading_mg_cm2"] = _positive_log10(loading)
        output["log10_total_solid_loading_mg_cm2"] = _positive_log10(
            loading * (1.0 + ratio)
        )

    def _engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        for name in CONTEXT_NUMERIC:
            output[name] = _numeric(frame, name)
        for name in CONTEXT_CATEGORICAL:
            output[name] = frame[name].astype("object")
        if self.recipe_.include_recorded_amounts:
            for name in RECORDED_NUMERIC:
                output[name] = _numeric(frame, name)
            for name in RECORDED_CATEGORICAL:
                output[name] = frame[name].astype("object")
        if self.recipe_.include_intensive_amounts:
            self._intensive(frame, output)

        representation = self.recipe_.representation
        if representation == "composition_summary":
            comp = self._closed_composition(frame)
            alcohol = comp[:, 2] + comp[:, 3] + comp[:, 4]
            output["alcohol_fraction"] = alcohol
            output["organic_fraction"] = comp[:, 0] + alcohol
            output["log10_water_to_alcohol_ratio"] = np.log10(
                (comp[:, 1] + self.composition_delta_)
                / (alcohol + self.composition_delta_)
            )
            output["n_solvents_present"] = (comp > 0).sum(axis=1).astype(float)
            log_comp = np.zeros_like(comp)
            np.log(comp, out=log_comp, where=comp > 0)
            output["solvent_entropy"] = -(comp * log_comp).sum(axis=1)
        elif representation == "composition_ilr":
            comp = self._closed_composition(frame)
            replaced = np.where(comp > 0, comp, self.composition_delta_)
            replaced /= replaced.sum(axis=1, keepdims=True)
            scores = np.log(replaced) @ self.ilr_basis_
            for index in range(4):
                output[f"solvent_ilr_{index + 1}"] = scores[:, index]
        elif representation == "fractions_plus_properties":
            comp = self._closed_composition(frame)
            for index, name in enumerate(SOLVENT_FRACTIONS):
                output[name] = comp[:, index]

        if representation == "property_lean":
            output["log10_solvent_viscosity_mPa_s"] = _positive_log10(
                _numeric(frame, PROPERTY_COLUMNS[0])
            )
            output["solvent_polarity_coordinate"] = self._polarity(frame)
        elif representation in {"property_physics", "fractions_plus_properties"}:
            viscosity = _numeric(frame, PROPERTY_COLUMNS[0]).to_numpy(float) * 1.0e-3
            surface = _numeric(frame, PROPERTY_COLUMNS[1]).to_numpy(float) * 1.0e-3
            density = _numeric(frame, PROPERTY_COLUMNS[2]).to_numpy(float)
            output["kinematic_viscosity_m2_s"] = np.where(
                (viscosity > 0) & (density > 0), viscosity / density, np.nan
            )
            output["capillary_flow_proxy_m_s"] = np.where(
                (surface >= 0) & (viscosity > 0), surface / viscosity, np.nan
            )
            output["solvent_polarity_coordinate"] = self._polarity(frame)

        if self.recipe_.include_property_interactions:
            log_viscosity = _positive_log10(_numeric(frame, PROPERTY_COLUMNS[0]))
            output["log10_solvent_viscosity_mPa_s"] = log_viscosity
            pooled_catalyst = self._pooled_catalyst(frame)
            for level in self.catalyst_interaction_levels_[1:]:
                indicator = pooled_catalyst.eq(level).to_numpy(float)
                slug = self._level_slug(level)
                output[f"catalyst_type_{slug}_x_log10_viscosity"] = (
                    indicator * log_viscosity
                )
                output[f"catalyst_type_{slug}_x_polarity_coordinate"] = (
                    indicator
                    * output["solvent_polarity_coordinate"].to_numpy(float)
                )
            output["log10_viscosity_x_log10_solid_concentration"] = (
                log_viscosity
                * output["log10_solid_concentration_mg_mL"].to_numpy(float)
            )
            ratio = _numeric(frame, "ionomer_to_solvent_ratio").to_numpy(float)
            if np.any(np.isfinite(ratio) & (ratio < 0)):
                raise ValueError("Ionomer/solvent ratio must be nonnegative")
            ratio = np.where(
                np.isfinite(ratio), ratio, self.ionomer_ratio_median_
            )
            replaced_ratio = np.where(
                ratio > 0, ratio, self.ionomer_ratio_delta_
            )
            output["capillary_flow_x_log10_ionomer_solvent_ratio"] = (
                output["capillary_flow_proxy_m_s"].to_numpy(float)
                * np.log10(replaced_ratio)
            )
        if self.recipe_.include_process_features:
            for name in PROCESS_NUMERIC:
                output[name] = _numeric(frame, name)
            for name in PROCESS_CATEGORICAL:
                output[name] = frame[name].astype("object")
        if self.recipe_.include_batch_masses:
            for name in BATCH_MASS_COLUMNS:
                output[name] = _numeric(frame, name)
        validate_no_target_leakage(output.columns)
        return output

    def fit(self, X: Any, y: Any = None) -> "FeatureDevelopmentBuilder":
        self.recipe_ = get_feature_development_recipe(self.recipe)
        frame = self._validated(X)
        if self.recipe_.representation in {
            "composition_summary",
            "composition_ilr",
            "fractions_plus_properties",
        }:
            self._fit_composition(frame)
        if self.recipe_.representation in {
            "property_lean",
            "property_physics",
            "fractions_plus_properties",
        }:
            self._fit_polarity(frame)
        if self.recipe_.include_property_interactions:
            self._fit_interactions(frame)
        engineered = self._engineer(frame)
        numeric = list(engineered.select_dtypes(include=[np.number]).columns)
        categorical = [name for name in engineered if name not in numeric]
        self.numeric_medians_ = {}
        for name in numeric:
            values = pd.to_numeric(engineered[name], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            self.numeric_medians_[name] = float(np.median(finite)) if finite.size else 0.0
        self.frequent_categories_ = {}
        for name in categorical:
            values = engineered[name].where(engineered[name].notna(), "__MISSING__").astype(str)
            counts = values.value_counts(dropna=False)
            levels = set(counts[counts >= self.recipe_.rare_min_frequency].index.astype(str))
            if not levels and len(counts):
                levels = {str(counts.index[0])}
            self.frequent_categories_[name] = tuple(sorted(levels))
        completed = self._complete(engineered)
        kept: list[str] = []
        for name in completed:
            if name in numeric:
                variance = float(np.var(completed[name].to_numpy(float)))
                if np.isfinite(variance) and variance > self.recipe_.variance_threshold:
                    kept.append(name)
            elif completed[name].nunique(dropna=False) > 1:
                kept.append(name)
        if not kept:
            raise ValueError(f"Feature recipe {self.recipe_.name!r} is constant")
        self.feature_names_out_ = np.asarray(kept, dtype=object)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _complete(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for name, median in self.numeric_medians_.items():
            output[name] = pd.to_numeric(output[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).fillna(median).astype(float)
        for name, levels in self.frequent_categories_.items():
            values = output[name].where(output[name].notna(), "__MISSING__").astype(str)
            output[name] = values.where(values.isin(set(levels)), "__RARE__")
        return output

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("FeatureDevelopmentBuilder must be fitted before transform")
        engineered = self._engineer(self._validated(X))
        return self._complete(engineered).loc[:, list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("FeatureDevelopmentBuilder must be fitted first")
        return self.feature_names_out_.copy()


def _dense_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_feature_development_pipeline(
    recipe: str | FeatureDevelopmentRecipe,
    *,
    model: Literal["rbf_svr", "bayesian_ridge", "dummy_mean", "dummy_median"] = "rbf_svr",
) -> BaseEstimator:
    """Build a frozen log-target model whose predictions are in mV dec-1."""

    if model == "rbf_svr":
        estimator: BaseEstimator = SVR(kernel="rbf", C=1.0, epsilon=0.05, gamma=0.1)
    elif model == "bayesian_ridge":
        estimator = BayesianRidge(compute_score=True)
    elif model == "dummy_mean":
        return TargetTransformRegressor(
            DummyRegressor(strategy="mean"), transform="log10", standardize=False
        )
    elif model == "dummy_median":
        return TargetTransformRegressor(
            DummyRegressor(strategy="median"), transform="log10", standardize=False
        )
    else:
        raise KeyError(f"Unknown fixed feature-development model: {model!r}")
    return Pipeline(
        [
            ("features", FeatureDevelopmentBuilder(recipe)),
            ("preprocess", _dense_preprocessor()),
            ("near_zero_variance", VarianceThreshold(0.0)),
            (
                "target",
                TargetTransformRegressor(estimator, transform="log10", standardize=False),
            ),
        ]
    )


make_fixed_feature_development_pipeline = make_feature_development_pipeline
