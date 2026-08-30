"""Fold-local feature recipes for posterior-soft cardinal classification.

The recipes in this module are deliberately compact.  They carry forward the
useful X-only transformations from the legacy KNN prototypes without carrying
forward their random-split validation, duplicated distance blocks, or
high-dimensional recipe fingerprints.

Every learned operation (zero replacement, PCA orientation, imputation,
rare-level pooling, and constant-feature removal) is fitted on the supplied
training fold.  Solvent representations are mutually exclusive within a
recipe: compact composition, ILR composition, and property proxies are never
stacked together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .features import is_target_like_column, validate_no_target_leakage


CompositionRepresentation = Literal[
    "none",
    "compact_composition",
    "legacy_safe_compact",
    "composition_ilr",
    "lean_property",
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
RECORDED_INK_NUMERIC = (
    "catalyst_concentration_mg_mL",
    "ionomer_to_solvent_ratio",
    "ionomer_concentration_wt_percent",
    "conductive_agent_dose_mg_per_mgcat",
)
RECORDED_INK_CATEGORICAL = ("conductive_agent_class",)
INTENSIVE_SOURCES = (
    "catalyst_mass_mg",
    "conductive_agent_mass_mg",
    "total_liquid_volume_uL",
    "drop_volume_uL",
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
PROCESS_COLUMNS = (
    "sonication_time_min",
    "drop_volume_uL",
    "electrode_area_cm2",
)

RAW_SOLVENT_COLUMNS = frozenset(
    {
        "solvent_1_type",
        "solvent_2_type",
        "solvent_1_volume_uL",
        "solvent_2_volume_uL",
        "solvent_1_volume_uL_per_mgcat",
        "solvent_2_volume_uL_per_mgcat",
    }
)


@dataclass(frozen=True)
class CardinalFeatureRecipe:
    """One finite, auditable feature representation."""

    name: str
    complexity_rank: int
    include_recorded_ink: bool = False
    include_intensive: bool = False
    representation: CompositionRepresentation = "none"
    include_process: bool = False
    include_catalyst_interactions: bool = False
    sensitivity_only: bool = False
    fixed_comparison_only: bool = False
    rare_min_frequency: int = 3

    def __post_init__(self) -> None:
        if self.include_intensive and not self.include_recorded_ink:
            raise ValueError("Intensive descriptors require recorded ink descriptors")
        if (
            self.representation != "none"
            and not self.include_recorded_ink
        ):
            raise ValueError("Solvent representations require recorded ink descriptors")
        if (
            self.include_catalyst_interactions
            and self.representation != "compact_composition"
        ):
            raise ValueError(
                "Catalyst interactions are limited to compact composition"
            )
        if self.rare_min_frequency < 1:
            raise ValueError("rare_min_frequency must be positive")
        if self.sensitivity_only and self.fixed_comparison_only:
            raise ValueError(
                "A recipe cannot be both sensitivity-only and fixed-comparison-only"
            )


def _recipes() -> dict[str, CardinalFeatureRecipe]:
    rows = (
        CardinalFeatureRecipe("context_log", 1),
        CardinalFeatureRecipe(
            "context_log_plus_ink_log",
            2,
            include_recorded_ink=True,
        ),
        CardinalFeatureRecipe(
            "legacy_safe_compact",
            3,
            include_recorded_ink=True,
            representation="legacy_safe_compact",
            fixed_comparison_only=True,
        ),
        CardinalFeatureRecipe(
            "intensive_formulation",
            3,
            include_recorded_ink=True,
            include_intensive=True,
        ),
        CardinalFeatureRecipe(
            "compact_composition",
            4,
            include_recorded_ink=True,
            include_intensive=True,
            representation="compact_composition",
        ),
        CardinalFeatureRecipe(
            "sensitivity_composition_ilr",
            5,
            include_recorded_ink=True,
            include_intensive=True,
            representation="composition_ilr",
            sensitivity_only=True,
        ),
        CardinalFeatureRecipe(
            "sensitivity_lean_property",
            5,
            include_recorded_ink=True,
            include_intensive=True,
            representation="lean_property",
            sensitivity_only=True,
        ),
        CardinalFeatureRecipe(
            "sensitivity_compact_process",
            6,
            include_recorded_ink=True,
            include_intensive=True,
            representation="compact_composition",
            include_process=True,
            sensitivity_only=True,
        ),
        CardinalFeatureRecipe(
            "sensitivity_compact_catalyst_interactions",
            7,
            include_recorded_ink=True,
            include_intensive=True,
            representation="compact_composition",
            include_catalyst_interactions=True,
            sensitivity_only=True,
        ),
    )
    return {row.name: row for row in rows}


CARDINAL_FEATURE_RECIPES: Mapping[str, CardinalFeatureRecipe] = _recipes()
PRIMARY_CARDINAL_FEATURE_RECIPES = tuple(
    name
    for name, recipe in CARDINAL_FEATURE_RECIPES.items()
    if not recipe.sensitivity_only and not recipe.fixed_comparison_only
)
SENSITIVITY_CARDINAL_FEATURE_RECIPES = tuple(
    name
    for name, recipe in CARDINAL_FEATURE_RECIPES.items()
    if recipe.sensitivity_only
)


def get_cardinal_feature_recipe(
    recipe: str | CardinalFeatureRecipe,
) -> CardinalFeatureRecipe:
    if isinstance(recipe, CardinalFeatureRecipe):
        return recipe
    try:
        return CARDINAL_FEATURE_RECIPES[str(recipe)]
    except KeyError as error:
        raise KeyError(
            f"Unknown cardinal feature recipe {recipe!r}; choose from "
            f"{sorted(CARDINAL_FEATURE_RECIPES)}"
        ) from error


def required_cardinal_feature_columns(
    recipes: str
    | CardinalFeatureRecipe
    | Sequence[str | CardinalFeatureRecipe],
) -> tuple[str, ...]:
    """Return the exact source allowlist needed by one or more recipes."""

    if isinstance(recipes, (str, CardinalFeatureRecipe)):
        selected = (recipes,)
    else:
        selected = tuple(recipes)
    columns: list[str] = []
    for value in selected:
        recipe = get_cardinal_feature_recipe(value)
        columns.extend(CONTEXT_NUMERIC + CONTEXT_CATEGORICAL)
        if recipe.include_recorded_ink:
            columns.extend(RECORDED_INK_NUMERIC + RECORDED_INK_CATEGORICAL)
        if recipe.include_intensive:
            columns.extend(INTENSIVE_SOURCES)
        if recipe.representation in {
            "compact_composition",
            "legacy_safe_compact",
            "composition_ilr",
        }:
            columns.extend(SOLVENT_FRACTIONS)
        if recipe.representation == "lean_property":
            columns.extend(PROPERTY_COLUMNS)
        if recipe.include_process:
            columns.extend(PROCESS_COLUMNS)
    output = tuple(dict.fromkeys(columns))
    validate_no_target_leakage(output)
    forbidden = sorted(
        set(output).intersection(RAW_SOLVENT_COLUMNS)
        | {name for name in output if is_target_like_column(name)}
    )
    if forbidden:
        raise AssertionError(f"Forbidden cardinal feature sources: {forbidden}")
    return output


def cardinal_feature_recipe_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recipe in CARDINAL_FEATURE_RECIPES.values():
        row = asdict(recipe)
        row["required_source_columns"] = "|".join(
            required_cardinal_feature_columns(recipe)
        )
        row["selection_role"] = (
            "fixed_comparison_only"
            if recipe.fixed_comparison_only
            else "sensitivity_only"
            if recipe.sensitivity_only
            else "primary_ladder"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["sensitivity_only", "complexity_rank", "name"], kind="stable"
    ).reset_index(drop=True)


def cardinal_engineered_feature_definitions() -> pd.DataFrame:
    """Human-readable formulas for every possible engineered output."""

    rows = [
        (
            "log10_catalyst_loading_mg_cm2",
            "context",
            "log10(catalyst_loading_mg_cm2_selected)",
            "mg cm^-2",
        ),
        (
            "log10_electrolyte_molarity_M",
            "context",
            "log10(electrolyte_molarity_M)",
            "mol L^-1",
        ),
        ("log1p_rpm", "context", "log1p(rpm)", "min^-1"),
        (
            "log10_scan_rate_mV_s",
            "context",
            "log10(scan_rate_mV_s)",
            "mV s^-1",
        ),
        (
            "log10_catalyst_concentration_mg_mL",
            "ink",
            "log10(catalyst_concentration_mg_mL)",
            "mg mL^-1",
        ),
        (
            "log1p_ionomer_to_solvent_ratio",
            "ink",
            "log1p(ionomer_to_solvent_ratio)",
            "dimensionless",
        ),
        (
            "log1p_ionomer_concentration_wt_percent",
            "ink",
            "log1p(ionomer_concentration_wt_percent)",
            "wt%",
        ),
        (
            "log1p_conductive_agent_dose_mg_per_mgcat",
            "ink",
            "log1p(conductive_agent_dose_mg_per_mgcat)",
            "mg mg_cat^-1",
        ),
        (
            "log10_solid_concentration_mg_mL",
            "ink",
            "log10(1000*(catalyst_mass_mg+conductive_agent_mass_mg)/"
            "total_liquid_volume_uL)",
            "mg mL^-1",
        ),
        (
            "log10_deposited_solid_dose_mg",
            "ink",
            "log10(solid_concentration_mg_mL*drop_volume_uL/1000)",
            "mg",
        ),
        (
            "log10_water_to_alcohol_balance",
            "composition",
            "log10((F_water+delta_train)/(F_EtOH+F_IPA+F_MeOH+delta_train))",
            "dimensionless",
        ),
        (
            "log10_water_to_organic_balance",
            "composition",
            "log10((F_water+delta_train)/"
            "(F_acetone+F_EtOH+F_IPA+F_MeOH+delta_train))",
            "dimensionless",
        ),
        (
            "log10_ethanol_to_isopropanol_balance",
            "composition",
            "log10((F_EtOH+delta_train)/(F_IPA+delta_train))",
            "dimensionless",
        ),
        (
            "solvent_entropy",
            "composition",
            "-sum(F_i*ln(F_i))",
            "dimensionless",
        ),
        (
            "solvent_ilr_[1-4]",
            "composition",
            "ILR(closed five-solvent composition; delta fitted on training fold)",
            "dimensionless",
        ),
        (
            "log10_solvent_viscosity_mPa_s",
            "property",
            "log10(solvent viscosity proxy)",
            "mPa s",
        ),
        (
            "solvent_polarity_coordinate",
            "property",
            "training-fold PCA1(surface tension,density,dielectric)",
            "dimensionless",
        ),
        (
            "log1p_[sonication_time|drop_volume|electrode_area]",
            "process",
            "log1p(source value)",
            "source unit",
        ),
        (
            "catalyst_[level]_x_[compact_coordinate]",
            "interaction",
            "training-fold pooled catalyst contrast * compact coordinate",
            "dimensionless",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["feature_name", "block", "formula", "source_unit"]
    ).assign(
        fit_scope="training_fold_only",
        target_derived=False,
    )


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def _positive_log10(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.full(array.shape, np.nan, dtype=float)
    valid = np.isfinite(array) & (array > 0)
    output[valid] = np.log10(array[valid])
    return output


def _nonnegative_log1p(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.full(array.shape, np.nan, dtype=float)
    valid = np.isfinite(array) & (array >= 0)
    output[valid] = np.log1p(array[valid])
    return output


class CardinalFeatureBuilder(TransformerMixin, BaseEstimator):
    """Cloneable fold-local transformer for the finite cardinal recipe ladder."""

    def __init__(self, recipe: str | CardinalFeatureRecipe) -> None:
        self.recipe = recipe

    def _validate_frame(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CardinalFeatureBuilder requires a pandas DataFrame")
        frame = X.copy()
        if "catalyst_type" not in frame and "catalyst_active_material_type" in frame:
            frame["catalyst_type"] = frame["catalyst_active_material_type"]
        required = set(required_cardinal_feature_columns(self.recipe_))
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(
                f"Cardinal feature recipe {self.recipe_.name!r} is missing: {missing}"
            )
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
        composition = self._closed_composition(frame)
        positive = composition[composition > 0]
        self.composition_delta_ = (
            float(min(1.0e-3, 0.5 * positive.min()))
            if positive.size
            else 1.0e-6
        )
        self.ilr_basis_ = helmert(len(SOLVENT_FRACTIONS), full=False).T

    def _fit_property(self, frame: pd.DataFrame) -> None:
        values = frame.loc[:, list(PROPERTY_COLUMNS[1:])].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        self.property_medians_ = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(
            np.isfinite(values), values, self.property_medians_[None, :]
        )
        self.property_scaler_ = StandardScaler().fit(filled)
        standardized = self.property_scaler_.transform(filled)
        self.property_pca_ = PCA(n_components=1, svd_solver="full").fit(
            standardized
        )
        # Give a stable direction: increasing dielectric must load positively.
        self.property_pca_sign_ = (
            -1.0 if self.property_pca_.components_[0, -1] < 0 else 1.0
        )

    def _property_coordinate(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(PROPERTY_COLUMNS[1:])].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        filled = np.where(
            np.isfinite(values), values, self.property_medians_[None, :]
        )
        return (
            self.property_pca_.transform(
                self.property_scaler_.transform(filled)
            )[:, 0]
            * self.property_pca_sign_
        )

    def _compact_composition(
        self, frame: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        composition = self._closed_composition(frame)
        water = composition[:, 1]
        ethanol = composition[:, 2]
        isopropanol = composition[:, 3]
        alcohol = ethanol + isopropanol + composition[:, 4]
        log_composition = np.zeros_like(composition)
        np.log(composition, out=log_composition, where=composition > 0)
        return {
            "log10_water_to_alcohol_balance": np.log10(
                (water + self.composition_delta_)
                / (alcohol + self.composition_delta_)
            ),
            "log10_ethanol_to_isopropanol_balance": np.log10(
                (ethanol + self.composition_delta_)
                / (isopropanol + self.composition_delta_)
            ),
            "solvent_entropy": -(composition * log_composition).sum(axis=1),
        }

    def _legacy_safe_compact_composition(
        self, frame: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Three nonredundant coordinates from the legacy KNN feature idea."""

        composition = self._closed_composition(frame)
        water = composition[:, 1]
        ethanol = composition[:, 2]
        isopropanol = composition[:, 3]
        organic = (
            composition[:, 0]
            + ethanol
            + isopropanol
            + composition[:, 4]
        )
        log_composition = np.zeros_like(composition)
        np.log(composition, out=log_composition, where=composition > 0)
        return {
            "log10_water_to_organic_balance": np.log10(
                (water + self.composition_delta_)
                / (organic + self.composition_delta_)
            ),
            "log10_ethanol_to_isopropanol_balance": np.log10(
                (ethanol + self.composition_delta_)
                / (isopropanol + self.composition_delta_)
            ),
            "solvent_entropy": -(composition * log_composition).sum(axis=1),
        }

    @staticmethod
    def _slug(value: str) -> str:
        return (
            str(value)
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

    def _engineer(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        output = pd.DataFrame(index=frame.index)
        blocks: dict[str, list[str]] = {
            "context": [],
            "ink": [],
            "composition": [],
            "property": [],
            "process": [],
            "interaction": [],
        }

        context_numeric = {
            "log10_catalyst_loading_mg_cm2": _positive_log10(
                _numeric(frame, "catalyst_loading_mg_cm2_selected")
            ),
            "log10_electrolyte_molarity_M": _positive_log10(
                _numeric(frame, "electrolyte_molarity_M")
            ),
            "log1p_rpm": _nonnegative_log1p(_numeric(frame, "rpm")),
            "log10_scan_rate_mV_s": _positive_log10(
                _numeric(frame, "scan_rate_mV_s")
            ),
        }
        for name, values in context_numeric.items():
            output[name] = values
            blocks["context"].append(name)
        for name in CONTEXT_CATEGORICAL:
            output[name] = frame[name].astype("object")
            blocks["context"].append(name)

        if self.recipe_.include_recorded_ink:
            ink_numeric = {
                "log10_catalyst_concentration_mg_mL": _positive_log10(
                    _numeric(frame, "catalyst_concentration_mg_mL")
                ),
                "log1p_ionomer_to_solvent_ratio": _nonnegative_log1p(
                    _numeric(frame, "ionomer_to_solvent_ratio")
                ),
                "log1p_ionomer_concentration_wt_percent": _nonnegative_log1p(
                    _numeric(frame, "ionomer_concentration_wt_percent")
                ),
                "log1p_conductive_agent_dose_mg_per_mgcat": _nonnegative_log1p(
                    _numeric(frame, "conductive_agent_dose_mg_per_mgcat")
                ),
            }
            for name, values in ink_numeric.items():
                output[name] = values
                blocks["ink"].append(name)
            output["conductive_agent_class"] = frame[
                "conductive_agent_class"
            ].astype("object")
            blocks["ink"].append("conductive_agent_class")

        if self.recipe_.include_intensive:
            catalyst_mass = _numeric(frame, "catalyst_mass_mg")
            conductive_mass = _numeric(frame, "conductive_agent_mass_mg")
            liquid_volume = _numeric(frame, "total_liquid_volume_uL")
            drop_volume = _numeric(frame, "drop_volume_uL")
            solid_mass = catalyst_mass + conductive_mass
            concentration = np.full(len(frame), np.nan, dtype=float)
            valid = (
                np.isfinite(solid_mass)
                & (solid_mass > 0)
                & np.isfinite(liquid_volume)
                & (liquid_volume > 0)
            )
            concentration[valid] = (
                1000.0 * solid_mass[valid] / liquid_volume[valid]
            )
            deposited = concentration * drop_volume / 1000.0
            for name, values in (
                (
                    "log10_solid_concentration_mg_mL",
                    _positive_log10(concentration),
                ),
                (
                    "log10_deposited_solid_dose_mg",
                    _positive_log10(deposited),
                ),
            ):
                output[name] = values
                blocks["ink"].append(name)

        compact: dict[str, np.ndarray] = {}
        if self.recipe_.representation == "compact_composition":
            compact = self._compact_composition(frame)
            for name, values in compact.items():
                output[name] = values
                blocks["composition"].append(name)
        elif self.recipe_.representation == "legacy_safe_compact":
            compact = self._legacy_safe_compact_composition(frame)
            for name, values in compact.items():
                output[name] = values
                blocks["composition"].append(name)
        elif self.recipe_.representation == "composition_ilr":
            composition = self._closed_composition(frame)
            replaced = np.where(
                composition > 0, composition, self.composition_delta_
            )
            replaced /= replaced.sum(axis=1, keepdims=True)
            scores = np.log(replaced) @ self.ilr_basis_
            for index in range(scores.shape[1]):
                name = f"solvent_ilr_{index + 1}"
                output[name] = scores[:, index]
                blocks["composition"].append(name)
        elif self.recipe_.representation == "lean_property":
            output["log10_solvent_viscosity_mPa_s"] = _positive_log10(
                _numeric(frame, PROPERTY_COLUMNS[0])
            )
            output["solvent_polarity_coordinate"] = self._property_coordinate(
                frame
            )
            blocks["property"].extend(
                [
                    "log10_solvent_viscosity_mPa_s",
                    "solvent_polarity_coordinate",
                ]
            )

        if self.recipe_.include_process:
            for source, name in (
                ("sonication_time_min", "log1p_sonication_time_min"),
                ("drop_volume_uL", "log1p_drop_volume_uL"),
                ("electrode_area_cm2", "log1p_electrode_area_cm2"),
            ):
                output[name] = _nonnegative_log1p(_numeric(frame, source))
                blocks["process"].append(name)

        if self.recipe_.include_catalyst_interactions:
            catalyst = frame["catalyst_type"].where(
                frame["catalyst_type"].notna(), "__MISSING__"
            ).astype(str)
            pooled = catalyst.where(
                catalyst.isin(set(self.catalyst_interaction_levels_)),
                "__RARE__",
            )
            for level in self.catalyst_interaction_levels_[1:]:
                indicator = pooled.eq(level).to_numpy(float)
                for coordinate, values in compact.items():
                    name = (
                        f"catalyst_{self._slug(level)}_x_{coordinate}"
                    )
                    output[name] = indicator * values
                    blocks["interaction"].append(name)

        validate_no_target_leakage(output.columns)
        if len(output.columns) != len(set(output.columns)):
            raise AssertionError("Engineered cardinal features must be unique")
        return output, blocks

    def _complete(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for name, median in self.numeric_medians_.items():
            output[name] = (
                pd.to_numeric(output[name], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(median)
                .astype(float)
            )
        for name, levels in self.frequent_categories_.items():
            values = output[name].where(
                output[name].notna(), "__MISSING__"
            ).astype(str)
            output[name] = values.where(
                values.isin(set(levels)), "__RARE__"
            )
        return output

    def fit(self, X: Any, y: Any = None) -> "CardinalFeatureBuilder":
        self.recipe_ = get_cardinal_feature_recipe(self.recipe)
        frame = self._validate_frame(X)
        if self.recipe_.representation in {
            "compact_composition",
            "legacy_safe_compact",
            "composition_ilr",
        }:
            self._fit_composition(frame)
        if self.recipe_.representation == "lean_property":
            self._fit_property(frame)
        if self.recipe_.include_catalyst_interactions:
            catalyst = frame["catalyst_type"].where(
                frame["catalyst_type"].notna(), "__MISSING__"
            ).astype(str)
            counts = catalyst.value_counts(dropna=False)
            frequent = set(
                counts[
                    counts >= self.recipe_.rare_min_frequency
                ].index.astype(str)
            )
            if not frequent and len(counts):
                frequent = {str(counts.index[0])}
            self.catalyst_interaction_levels_ = tuple(sorted(frequent))

        engineered, block_candidates = self._engineer(frame)
        numeric = list(engineered.select_dtypes(include=[np.number]).columns)
        categorical = [name for name in engineered if name not in numeric]
        self.numeric_medians_: dict[str, float] = {}
        for name in numeric:
            values = pd.to_numeric(
                engineered[name], errors="coerce"
            ).to_numpy(float)
            finite = values[np.isfinite(values)]
            self.numeric_medians_[name] = (
                float(np.median(finite)) if finite.size else 0.0
            )
        self.frequent_categories_: dict[str, tuple[str, ...]] = {}
        for name in categorical:
            values = engineered[name].where(
                engineered[name].notna(), "__MISSING__"
            ).astype(str)
            counts = values.value_counts(dropna=False)
            levels = tuple(
                sorted(
                    counts[
                        counts >= self.recipe_.rare_min_frequency
                    ].index.astype(str)
                )
            )
            if not levels and len(counts):
                levels = (str(counts.index[0]),)
            self.frequent_categories_[name] = levels

        completed = self._complete(engineered)
        kept: list[str] = []
        for name in completed:
            if name in numeric:
                values = completed[name].to_numpy(float)
                if np.isfinite(values).all() and float(np.var(values)) > 0:
                    kept.append(name)
            elif completed[name].nunique(dropna=False) > 1:
                kept.append(name)
        if not kept:
            raise ValueError(
                f"Cardinal feature recipe {self.recipe_.name!r} is constant"
            )
        kept_set = set(kept)
        blocks = {
            block: tuple(name for name in names if name in kept_set)
            for block, names in block_candidates.items()
        }
        blocks = {block: names for block, names in blocks.items() if names}
        flattened = [name for names in blocks.values() for name in names]
        if len(flattened) != len(set(flattened)) or set(flattened) != kept_set:
            raise AssertionError(
                "Engineered Gower blocks must be disjoint and exhaustive"
            )
        self.feature_names_out_ = np.asarray(kept, dtype=object)
        self.blocks_out_ = blocks
        self.numeric_columns_out_ = tuple(
            name for name in kept if name in numeric
        )
        self.categorical_columns_out_ = tuple(
            name for name in kept if name not in numeric
        )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("CardinalFeatureBuilder must be fitted first")
        engineered, _ = self._engineer(self._validate_frame(X))
        return self._complete(engineered).loc[
            :, list(self.feature_names_out_)
        ]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("CardinalFeatureBuilder must be fitted first")
        return self.feature_names_out_.copy()

    def get_blocks_out(self) -> dict[str, tuple[str, ...]]:
        if not hasattr(self, "blocks_out_"):
            raise RuntimeError("CardinalFeatureBuilder must be fitted first")
        return dict(self.blocks_out_)


__all__ = [
    "CARDINAL_FEATURE_RECIPES",
    "PRIMARY_CARDINAL_FEATURE_RECIPES",
    "SENSITIVITY_CARDINAL_FEATURE_RECIPES",
    "CardinalFeatureBuilder",
    "CardinalFeatureRecipe",
    "cardinal_engineered_feature_definitions",
    "cardinal_feature_recipe_registry",
    "get_cardinal_feature_recipe",
    "required_cardinal_feature_columns",
]
