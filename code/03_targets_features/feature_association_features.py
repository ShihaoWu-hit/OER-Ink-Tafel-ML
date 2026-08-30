"""Target-blind feature support audit and frozen association panels.

The association workflow deliberately starts from the predictor columns in the
hash-bound ``registered_panel_schema.csv``.  It never infers candidate
predictors by scanning the much wider records table.  This keeps identifiers,
provenance fields, curve coordinates, and target-derived columns outside the
feature search space by construction.

The panels below compare the same context adjustment set with exactly one
formulation coordinate at a time.  Every learned operation in
:class:`AssociationFeatureBuilder` is fitted on its supplied training fold:
composition zero replacement, property imputation/scaling/PCA, category
pooling, numeric imputation, and constant-feature removal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .features import is_target_like_column, validate_no_target_leakage


EXPECTED_REGISTERED_CANDIDATE_COUNT = 28

CATALYST_SOURCE = "catalyst_active_material_type"
CATALYST_SOURCE_ALIAS = "catalyst_type"
SCOPE_SOURCE = "electrolyte_scope"
GROUP_SOURCE = "cv_group_id"

CONTEXT_SOURCES = (
    CATALYST_SOURCE,
    "catalyst_loading_mg_cm2_selected",
    "electrolyte_molarity_M",
    "rpm",
    "scan_rate_mV_s",
    "ir_correction_status",
    SCOPE_SOURCE,
)
INK_AMOUNT_SOURCES = (
    "catalyst_concentration_mg_mL",
    "ionomer_to_solvent_ratio",
)
SOLVENT_FRACTION_SOURCES = (
    "solvent_fraction_acetone",
    "solvent_fraction_water",
    "solvent_fraction_ethanol",
    "solvent_fraction_isopropanol",
    "solvent_fraction_methanol",
)
PROPERTY_SOURCES = (
    "solvent_mixture_property_proxy_viscosity_mPa_s",
    "solvent_mixture_property_proxy_surface_tension_mN_m",
    "solvent_mixture_property_proxy_density_kg_m3",
    "solvent_mixture_property_proxy_dielectric_constant_best_available",
)
CONDUCTIVE_SOURCES = ("conductive_agent_present",)

PROPERTY_SOURCE_SET = frozenset(PROPERTY_SOURCES)
COMPOSITION_SOURCE_SET = frozenset(SOLVENT_FRACTION_SOURCES)

CATALYST_CONCENTRATION_FEATURE = "log10_catalyst_concentration_mg_mL"
IONOMER_RATIO_FEATURE = "log1p_ionomer_to_solvent_ratio"
WATER_ORGANIC_FEATURE = "log10_water_to_organic_balance"
ETHANOL_ISOPROPANOL_FEATURE = (
    "log10_ethanol_to_isopropanol_balance"
)
VISCOSITY_FEATURE = "log10_solvent_viscosity_mPa_s"
POLARITY_FEATURE = "solvent_polarity_coordinate"
CONDUCTIVE_FEATURE = "conductive_agent_present"

PRIMARY_FORMULATION_FEATURES = (
    CATALYST_CONCENTRATION_FEATURE,
    IONOMER_RATIO_FEATURE,
    WATER_ORGANIC_FEATURE,
    ETHANOL_ISOPROPANOL_FEATURE,
    VISCOSITY_FEATURE,
    POLARITY_FEATURE,
)

ENGINEERED_FEATURE_SOURCES: Mapping[str, tuple[str, ...]] = {
    CATALYST_CONCENTRATION_FEATURE: (
        "catalyst_concentration_mg_mL",
    ),
    IONOMER_RATIO_FEATURE: ("ionomer_to_solvent_ratio",),
    WATER_ORGANIC_FEATURE: SOLVENT_FRACTION_SOURCES,
    ETHANOL_ISOPROPANOL_FEATURE: SOLVENT_FRACTION_SOURCES,
    VISCOSITY_FEATURE: (PROPERTY_SOURCES[0],),
    POLARITY_FEATURE: PROPERTY_SOURCES[1:],
    CONDUCTIVE_FEATURE: CONDUCTIVE_SOURCES,
}

ENGINEERED_FEATURE_BLOCKS: Mapping[str, str] = {
    CATALYST_CONCENTRATION_FEATURE: "catalyst_concentration",
    IONOMER_RATIO_FEATURE: "ionomer_to_solvent_ratio",
    WATER_ORGANIC_FEATURE: "water_to_organic_balance",
    ETHANOL_ISOPROPANOL_FEATURE: "ethanol_to_isopropanol_balance",
    VISCOSITY_FEATURE: "solvent_viscosity",
    POLARITY_FEATURE: "solvent_polarity",
    CONDUCTIVE_FEATURE: "conductive_agent",
}

DIRECT_CONTEXT_SOURCES = frozenset(
    {
        CATALYST_SOURCE,
        "catalyst_loading_mg_cm2_selected",
        "electrolyte_molarity_M",
        "rpm",
        "scan_rate_mV_s",
        "ir_correction_status",
    }
)
DIRECT_PRIMARY_FEATURE_SOURCES = frozenset(
    {
        "catalyst_concentration_mg_mL",
        "ionomer_to_solvent_ratio",
        PROPERTY_SOURCES[0],
    }
)
POOLED_COMPOSITION_SOURCES = frozenset(SOLVENT_FRACTION_SOURCES)
POOLED_PROPERTY_SOURCES = frozenset(PROPERTY_SOURCES[1:])
HARD_EXCLUDED_SUPERVISED_SOURCES = frozenset(
    {
        "catalyst_support",
        "ionomer_concentration_wt_percent",
        "conductive_agent_dose_mg_per_mgcat",
        "electrolyte_identity",
        "dispersion_method",
        "sonication_time_min",
        "drop_volume_uL",
        "electrode_area_cm2",
        "solvent_mixture_property_proxy_dielectric_constant",
        (
            "solvent_mixture_property_proxy_dielectric_constant_"
            "estimated_flag"
        ),
    }
)


@dataclass(frozen=True)
class AssociationFeaturePanel:
    """One frozen context or context-plus-one association representation."""

    name: str
    parent: str | None
    added_block: str
    analysis_role: str
    effect_eligible: bool
    complexity_rank: int
    representation: str = "none"
    added_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.representation not in {"none", "composition", "property"}:
            raise ValueError(
                f"Unknown association representation {self.representation!r}"
            )
        unknown = sorted(
            set(self.added_features) - set(ENGINEERED_FEATURE_SOURCES)
        )
        if unknown:
            raise ValueError(
                f"Unknown engineered association features: {unknown}"
            )
        if self.name == "context_core" and self.added_features:
            raise ValueError(
                "context_core cannot contain a formulation increment"
            )
        if self.name != "context_core" and len(self.added_features) != 1:
            raise ValueError(
                "Every non-context association panel must add exactly one "
                "engineered feature"
            )
        has_composition = bool(
            set(self.added_features)
            & {WATER_ORGANIC_FEATURE, ETHANOL_ISOPROPANOL_FEATURE}
        )
        has_property = bool(
            set(self.added_features) & {VISCOSITY_FEATURE, POLARITY_FEATURE}
        )
        if has_composition and has_property:
            raise ValueError(
                "Composition and property coordinates are mutually exclusive"
            )
        expected_representation = (
            "composition"
            if has_composition
            else "property"
            if has_property
            else "none"
        )
        if self.representation != expected_representation:
            raise ValueError(
                f"Panel {self.name!r} representation must be "
                f"{expected_representation!r}"
            )

    @property
    def source_columns(self) -> tuple[str, ...]:
        columns = list(CONTEXT_SOURCES)
        for feature in self.added_features:
            columns.extend(ENGINEERED_FEATURE_SOURCES[feature])
        output = tuple(dict.fromkeys(columns))
        validate_no_target_leakage(output)
        return output


def _build_frozen_panels() -> dict[str, AssociationFeaturePanel]:
    rows = (
        AssociationFeaturePanel(
            name="context_core",
            parent=None,
            added_block="context",
            analysis_role="reference_context",
            effect_eligible=True,
            complexity_rank=1,
        ),
        AssociationFeaturePanel(
            name="context_plus_catalyst_concentration",
            parent="context_core",
            added_block="catalyst_concentration",
            analysis_role="primary_single_feature_increment",
            effect_eligible=True,
            complexity_rank=2,
            added_features=(CATALYST_CONCENTRATION_FEATURE,),
        ),
        AssociationFeaturePanel(
            name="context_plus_ionomer_ratio",
            parent="context_core",
            added_block="ionomer_to_solvent_ratio",
            analysis_role="primary_single_feature_increment",
            effect_eligible=True,
            complexity_rank=2,
            added_features=(IONOMER_RATIO_FEATURE,),
        ),
        AssociationFeaturePanel(
            name="context_plus_water_organic",
            parent="context_core",
            added_block="water_to_organic_balance",
            analysis_role="primary_single_feature_increment",
            effect_eligible=True,
            complexity_rank=2,
            representation="composition",
            added_features=(WATER_ORGANIC_FEATURE,),
        ),
        AssociationFeaturePanel(
            name="context_plus_ethanol_isopropanol",
            parent="context_core",
            added_block="ethanol_to_isopropanol_balance",
            analysis_role="primary_single_feature_increment",
            effect_eligible=True,
            complexity_rank=2,
            representation="composition",
            added_features=(ETHANOL_ISOPROPANOL_FEATURE,),
        ),
        AssociationFeaturePanel(
            name="context_plus_viscosity",
            parent="context_core",
            added_block="solvent_viscosity",
            analysis_role="alternative_property_single_feature",
            effect_eligible=True,
            complexity_rank=2,
            representation="property",
            added_features=(VISCOSITY_FEATURE,),
        ),
        AssociationFeaturePanel(
            name="context_plus_polarity",
            parent="context_core",
            added_block="solvent_polarity",
            analysis_role="alternative_property_single_feature",
            effect_eligible=True,
            complexity_rank=2,
            representation="property",
            added_features=(POLARITY_FEATURE,),
        ),
        AssociationFeaturePanel(
            name="context_plus_conductive_sensitivity",
            parent="context_core",
            added_block="conductive_agent",
            analysis_role="thin_support_sensitivity",
            effect_eligible=False,
            complexity_rank=2,
            added_features=(CONDUCTIVE_FEATURE,),
        ),
    )
    return {row.name: row for row in rows}


FROZEN_ASSOCIATION_PANELS: Mapping[str, AssociationFeaturePanel] = (
    _build_frozen_panels()
)


def get_association_panel(
    panel: str | AssociationFeaturePanel,
) -> AssociationFeaturePanel:
    if isinstance(panel, AssociationFeaturePanel):
        return panel
    try:
        return FROZEN_ASSOCIATION_PANELS[str(panel)]
    except KeyError as error:
        raise KeyError(
            f"Unknown association panel {panel!r}; choose from "
            f"{sorted(FROZEN_ASSOCIATION_PANELS)}"
        ) from error


def required_association_source_columns(
    panel: str | AssociationFeaturePanel,
) -> tuple[str, ...]:
    return get_association_panel(panel).source_columns


def association_panel_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for panel in FROZEN_ASSOCIATION_PANELS.values():
        row = asdict(panel)
        row["required_source_columns"] = "|".join(panel.source_columns)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["complexity_rank", "name"], kind="stable"
    ).reset_index(drop=True)


frozen_association_panel_registry = association_panel_registry


def association_engineered_feature_definitions() -> pd.DataFrame:
    """Return formulas and fold-local boundaries for engineered predictors."""

    rows = (
        (
            "catalyst_family",
            "context",
            "categorical",
            "Ir and IrO2 -> IrO2_plus_Ir; RuO2 -> RuO2",
            (CATALYST_SOURCE,),
            "fixed X-only mapping; original identity remains in source records",
        ),
        (
            "log10_catalyst_loading_mg_cm2",
            "context",
            "numeric",
            "log10(catalyst_loading_mg_cm2_selected)",
            ("catalyst_loading_mg_cm2_selected",),
            "training-fold median imputation",
        ),
        (
            "log10_electrolyte_molarity_M",
            "context",
            "numeric",
            "log10(electrolyte_molarity_M)",
            ("electrolyte_molarity_M",),
            "training-fold median imputation",
        ),
        (
            "log1p_rpm",
            "context",
            "numeric",
            "log1p(rpm)",
            ("rpm",),
            "training-fold median imputation",
        ),
        (
            "log10_scan_rate_mV_s",
            "context",
            "numeric",
            "log10(scan_rate_mV_s)",
            ("scan_rate_mV_s",),
            "training-fold median imputation",
        ),
        (
            "ir_correction_status",
            "context",
            "categorical",
            "reviewed three-state iR status",
            ("ir_correction_status",),
            "training-fold rare-level pooling",
        ),
        (
            "electrolyte_scope",
            "context",
            "categorical",
            "electrolyte_scope",
            (SCOPE_SOURCE,),
            "training-fold rare-level pooling and constant removal",
        ),
        (
            "log10_catalyst_concentration_mg_mL",
            "ink_amount",
            "numeric",
            "log10(catalyst_concentration_mg_mL)",
            ("catalyst_concentration_mg_mL",),
            "training-fold median imputation",
        ),
        (
            "log1p_ionomer_to_solvent_ratio",
            "ink_amount",
            "numeric",
            "log1p(ionomer_to_solvent_ratio)",
            ("ionomer_to_solvent_ratio",),
            "training-fold median imputation",
        ),
        (
            "log10_water_to_organic_balance",
            "composition",
            "numeric",
            "log10((F_water+delta_train)/(F_organic+delta_train))",
            SOLVENT_FRACTION_SOURCES,
            "closure and zero delta fitted on training fold",
        ),
        (
            "log10_ethanol_to_isopropanol_balance",
            "composition",
            "numeric",
            "log10((F_ethanol+delta_train)/(F_isopropanol+delta_train))",
            SOLVENT_FRACTION_SOURCES,
            "closure and zero delta fitted on training fold",
        ),
        (
            "log10_solvent_viscosity_mPa_s",
            "property",
            "numeric",
            "log10(solvent mixture viscosity proxy)",
            (PROPERTY_SOURCES[0],),
            "training-fold median imputation",
        ),
        (
            "solvent_polarity_coordinate",
            "property",
            "numeric",
            "PCA1(standardized surface tension,density,best-available dielectric)",
            PROPERTY_SOURCES[1:],
            "training-fold median, scaling and sign-anchored PCA",
        ),
        (
            "conductive_agent_present",
            "conductive_agent",
            "categorical",
            "reported conductive-agent presence",
            CONDUCTIVE_SOURCES,
            "training-fold rare-level pooling; sensitivity only",
        ),
    )
    return pd.DataFrame(
        [
            {
                "feature_name": name,
                "block": block,
                "kind": kind,
                "formula": formula,
                "source_columns": "|".join(sources),
                "fit_scope": fit_scope,
                "target_derived": False,
            }
            for name, block, kind, formula, sources, fit_scope in rows
        ]
    )


association_feature_definitions = association_engineered_feature_definitions


def _read_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def registered_candidate_registry(
    registered_panel_schema: pd.DataFrame | str | Path,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    """Return one row per registered, present physical predictor.

    Rows are deduplicated by ``selected_column`` in their persisted order.
    Synthetic/dummy rows, missing selections, and blank selections are
    excluded.  No records-table columns are inspected here.
    """

    schema = _read_frame(registered_panel_schema)
    required = {"status", "selected_column"}
    missing = sorted(required - set(schema.columns))
    if missing:
        raise KeyError(
            f"Registered panel schema is missing required columns: {missing}"
        )
    selected = schema.loc[
        schema["status"].astype("string").str.strip().str.lower().eq("present")
        & schema["selected_column"].notna()
    ].copy()
    selected["selected_column"] = (
        selected["selected_column"].astype("string").str.strip()
    )
    selected = selected.loc[selected["selected_column"].ne("")].copy()
    selected = selected.drop_duplicates("selected_column", keep="first")
    columns = tuple(selected["selected_column"].astype(str))
    validate_no_target_leakage(columns)
    leaking = sorted(name for name in columns if is_target_like_column(name))
    if leaking:
        raise ValueError(
            f"Target-like registered predictors are forbidden: {leaking}"
        )
    if expected_count is not None and len(selected) != int(expected_count):
        raise ValueError(
            f"Expected {int(expected_count)} registered candidate predictors, "
            f"found {len(selected)}"
        )
    keep = [
        name
        for name in (
            "logical_feature",
            "selected_column",
            "kind",
            "distance_block",
        )
        if name in selected.columns
    ]
    result = selected.loc[:, keep].reset_index(drop=True)
    if "kind" not in result:
        result["kind"] = "unknown"
    if "logical_feature" not in result:
        result["logical_feature"] = result["selected_column"]
    if "distance_block" not in result:
        result["distance_block"] = "unregistered"
    return result


def derive_candidate_predictor_allowlist(
    registered_panel_schema: pd.DataFrame | str | Path,
    *,
    expected_count: int | None = None,
) -> tuple[str, ...]:
    registry = registered_candidate_registry(
        registered_panel_schema, expected_count=expected_count
    )
    return tuple(registry["selected_column"].astype(str))


registered_candidate_predictors = derive_candidate_predictor_allowlist


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


class AssociationFeatureBuilder(TransformerMixin, BaseEstimator):
    """Fold-local transformer for a frozen low-dimensional panel."""

    def __init__(
        self,
        panel: str | AssociationFeaturePanel,
        *,
        rare_min_frequency: int = 3,
    ) -> None:
        self.panel = panel
        self.rare_min_frequency = rare_min_frequency

    def _validate_frame(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "AssociationFeatureBuilder requires a pandas DataFrame"
            )
        frame = X.copy()
        if CATALYST_SOURCE not in frame and CATALYST_SOURCE_ALIAS in frame:
            frame[CATALYST_SOURCE] = frame[CATALYST_SOURCE_ALIAS]
        missing = sorted(set(self.panel_.source_columns) - set(frame.columns))
        if missing:
            raise KeyError(
                f"Association panel {self.panel_.name!r} is missing: {missing}"
            )
        return frame

    @staticmethod
    def _catalyst_family(values: pd.Series) -> pd.Series:
        source = values.where(values.notna(), "__MISSING__").astype(str)
        return source.map(
            lambda value: (
                "IrO2_plus_Ir"
                if value in {"Ir", "IrO2"}
                else "RuO2"
                if value == "RuO2"
                else value
            )
        )

    def _closed_composition(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(SOLVENT_FRACTION_SOURCES)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        if np.any(np.isfinite(values) & (values < 0)):
            raise ValueError("Solvent fractions must be nonnegative")
        values = np.where(np.isfinite(values), values, 0.0)
        totals = values.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError(
                "Every solvent composition must have a positive row sum"
            )
        return values / totals[:, None]

    def _fit_composition(self, frame: pd.DataFrame) -> None:
        composition = self._closed_composition(frame)
        positive = composition[composition > 0]
        self.composition_delta_ = (
            float(min(1.0e-3, 0.5 * positive.min()))
            if positive.size
            else 1.0e-6
        )

    def _fit_property(self, frame: pd.DataFrame) -> None:
        values = frame.loc[:, list(PROPERTY_SOURCES[1:])].apply(
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
        dielectric_loading = float(self.property_pca_.components_[0, -1])
        self.property_pca_sign_ = -1.0 if dielectric_loading < 0 else 1.0

    def _property_coordinate(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(PROPERTY_SOURCES[1:])].apply(
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

    def _engineer(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        output = pd.DataFrame(index=frame.index)
        blocks: dict[str, list[str]] = {
            "context": [],
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
        output["catalyst_family"] = self._catalyst_family(
            frame[CATALYST_SOURCE]
        ).astype("object")
        blocks["context"].append("catalyst_family")
        for name, values in context_numeric.items():
            output[name] = values
            blocks["context"].append(name)
        output["ir_correction_status"] = frame[
            "ir_correction_status"
        ].astype("object")
        output["electrolyte_scope"] = frame[SCOPE_SOURCE].astype("object")
        blocks["context"].extend(
            ["ir_correction_status", "electrolyte_scope"]
        )

        added = set(self.panel_.added_features)
        if CATALYST_CONCENTRATION_FEATURE in added:
            output[CATALYST_CONCENTRATION_FEATURE] = _positive_log10(
                _numeric(frame, "catalyst_concentration_mg_mL")
            )
        if IONOMER_RATIO_FEATURE in added:
            output[IONOMER_RATIO_FEATURE] = _nonnegative_log1p(
                _numeric(frame, "ionomer_to_solvent_ratio")
            )

        if added & {WATER_ORGANIC_FEATURE, ETHANOL_ISOPROPANOL_FEATURE}:
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
            if WATER_ORGANIC_FEATURE in added:
                output[WATER_ORGANIC_FEATURE] = np.log10(
                    (water + self.composition_delta_)
                    / (organic + self.composition_delta_)
                )
            if ETHANOL_ISOPROPANOL_FEATURE in added:
                output[ETHANOL_ISOPROPANOL_FEATURE] = np.log10(
                    (ethanol + self.composition_delta_)
                    / (isopropanol + self.composition_delta_)
                )

        if VISCOSITY_FEATURE in added:
            output[VISCOSITY_FEATURE] = _positive_log10(
                _numeric(frame, PROPERTY_SOURCES[0])
            )
        if POLARITY_FEATURE in added:
            output[POLARITY_FEATURE] = self._property_coordinate(
                frame
            )

        if CONDUCTIVE_FEATURE in added:
            output[CONDUCTIVE_FEATURE] = (
                frame["conductive_agent_present"]
                .where(frame["conductive_agent_present"].notna(), "__MISSING__")
                .astype(str)
            )
        for feature in self.panel_.added_features:
            if feature in output:
                blocks.setdefault(
                    ENGINEERED_FEATURE_BLOCKS[feature], []
                ).append(feature)
        validate_no_target_leakage(output.columns)
        return output, blocks

    def _complete(self, engineered: pd.DataFrame) -> pd.DataFrame:
        output = engineered.copy()
        for name, median in self.numeric_medians_.items():
            output[name] = (
                pd.to_numeric(output[name], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(float(median))
                .astype(float)
            )
        for name, levels in self.frequent_categories_.items():
            values = (
                output[name]
                .where(output[name].notna(), "__MISSING__")
                .astype(str)
            )
            output[name] = values.where(
                values.isin(set(levels)), "__RARE__"
            )
        return output

    def fit(self, X: Any, y: Any = None) -> "AssociationFeatureBuilder":
        del y
        if int(self.rare_min_frequency) < 1:
            raise ValueError("rare_min_frequency must be positive")
        self.panel_ = get_association_panel(self.panel)
        frame = self._validate_frame(X)
        if set(self.panel_.added_features) & {
            WATER_ORGANIC_FEATURE,
            ETHANOL_ISOPROPANOL_FEATURE,
        }:
            self._fit_composition(frame)
        if POLARITY_FEATURE in self.panel_.added_features:
            self._fit_property(frame)
        engineered, block_candidates = self._engineer(frame)
        numeric = list(
            engineered.select_dtypes(include=[np.number]).columns
        )
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
            values = (
                engineered[name]
                .where(engineered[name].notna(), "__MISSING__")
                .astype(str)
            )
            counts = values.value_counts(dropna=False)
            levels = tuple(
                sorted(
                    counts[
                        counts >= int(self.rare_min_frequency)
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
                f"Association panel {self.panel_.name!r} is constant"
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
                "Association feature blocks must be disjoint and exhaustive"
            )
        self.feature_names_out_ = np.asarray(kept, dtype=object)
        self.numeric_columns_out_ = tuple(
            name for name in kept if name in numeric
        )
        self.categorical_columns_out_ = tuple(
            name for name in kept if name not in numeric
        )
        self.blocks_out_ = blocks
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError(
                "AssociationFeatureBuilder must be fitted before transform"
            )
        engineered, _ = self._engineer(self._validate_frame(X))
        return self._complete(engineered).loc[
            :, list(self.feature_names_out_)
        ]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError(
                "AssociationFeatureBuilder must be fitted first"
            )
        return self.feature_names_out_.copy()

    def get_blocks_out(self) -> dict[str, tuple[str, ...]]:
        if not hasattr(self, "blocks_out_"):
            raise RuntimeError(
                "AssociationFeatureBuilder must be fitted first"
            )
        return dict(self.blocks_out_)


def _scope_frames(
    records: pd.DataFrame,
    scopes: Sequence[str],
    *,
    scope_col: str,
) -> list[tuple[str, pd.DataFrame]]:
    output: list[tuple[str, pd.DataFrame]] = []
    for scope in scopes:
        name = str(scope)
        if name in {"all", "all97"}:
            frame = records.copy()
            label = "all97"
        else:
            if scope_col not in records:
                raise KeyError(f"Records table is missing {scope_col!r}")
            frame = records.loc[
                records[scope_col].astype(str).eq(name)
            ].copy()
            label = name
        if frame.empty:
            raise ValueError(f"Feature audit scope {name!r} is empty")
        output.append((label, frame.reset_index(drop=True)))
    return output


def _categorical_missing(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.isna() | text.eq("")


def _numeric_support_status(
    *,
    distinct: int,
    missing_fraction: float,
    nonzero_groups: int,
    zero_fraction: float,
) -> tuple[str, str]:
    if distinct <= 1:
        return "unsupported_constant", "zero or one finite value"
    if missing_fraction > 0.40:
        return "unsupported_high_missingness", "missing fraction exceeds 40%"
    if zero_fraction > 0.50 and nonzero_groups < 5:
        return (
            "thin_nonzero_group_support",
            "fewer than five connected groups carry a nonzero value",
        )
    if distinct < 4:
        return (
            "limited_numeric_levels",
            "fewer than four distinct finite values; GAM must use a linear term",
        )
    return "supported_descriptive", "passes X-only descriptive support checks"


def _categorical_support_status(
    *,
    distinct: int,
    missing_fraction: float,
    minimum_level_groups: int,
) -> tuple[str, str]:
    if distinct <= 1:
        return "unsupported_constant", "zero or one observed category"
    if missing_fraction > 0.40:
        return "unsupported_high_missingness", "missing fraction exceeds 40%"
    if minimum_level_groups < 5:
        return (
            "thin_category_group_support",
            "at least one category occurs in fewer than five connected groups",
        )
    return "supported_descriptive", "passes X-only descriptive support checks"


def _panel_memberships(source_column: str) -> tuple[str, ...]:
    return tuple(
        panel.name
        for panel in FROZEN_ASSOCIATION_PANELS.values()
        if source_column in panel.source_columns
    )


def _supervised_source_role(source_column: str) -> str:
    if source_column in DIRECT_CONTEXT_SOURCES:
        return "context_adjustment"
    if source_column in DIRECT_PRIMARY_FEATURE_SOURCES:
        return "direct_single_feature_input"
    if source_column in POOLED_COMPOSITION_SOURCES:
        return "pooled_composition_coordinate_input_only"
    if source_column in POOLED_PROPERTY_SOURCES:
        return "pooled_property_coordinate_input_only"
    if source_column in CONDUCTIVE_SOURCES:
        return "sensitivity_only"
    if source_column in HARD_EXCLUDED_SUPERVISED_SOURCES:
        return "excluded_from_supervised_models"
    return "descriptive_atlas_only"


def _effect_memberships(source_column: str) -> tuple[str, ...]:
    if _supervised_source_role(source_column) not in {
        "direct_single_feature_input",
    }:
        return ()
    return tuple(
        panel.name
        for panel in FROZEN_ASSOCIATION_PANELS.values()
        if panel.effect_eligible and source_column in panel.source_columns
    )


def _feature_support_tables(
    records: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    *,
    scopes: Sequence[str],
    scope_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    for scope, frame in _scope_frames(records, scopes, scope_col=scope_col):
        if group_col not in frame:
            raise KeyError(f"Records table is missing {group_col!r}")
        for candidate in candidate_registry.itertuples(index=False):
            column = str(candidate.selected_column)
            if column not in frame:
                raise KeyError(
                    f"Registered predictor {column!r} is missing from records"
                )
            kind = str(candidate.kind)
            raw = frame[column]
            if kind == "numeric":
                values = pd.to_numeric(raw, errors="coerce")
                finite = np.isfinite(values.to_numpy(float))
                missing_mask = ~finite
                observed = values.loc[finite].astype(float)
                distinct = int(observed.nunique(dropna=True))
                nonzero_mask = finite & values.ne(0).to_numpy()
                nonzero_groups = int(
                    frame.loc[nonzero_mask, group_col].astype(str).nunique()
                )
                zero_fraction = (
                    float(observed.eq(0).mean()) if len(observed) else 1.0
                )
                status, reason = _numeric_support_status(
                    distinct=distinct,
                    missing_fraction=float(missing_mask.mean()),
                    nonzero_groups=nonzero_groups,
                    zero_fraction=zero_fraction,
                )
                quantiles = (
                    observed.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                    if len(observed)
                    else pd.Series(index=[0.05, 0.25, 0.5, 0.75, 0.95])
                )
                dominant_value = (
                    str(observed.value_counts().index[0])
                    if len(observed)
                    else ""
                )
                dominant_fraction = (
                    float(observed.value_counts(normalize=True).iloc[0])
                    if len(observed)
                    else np.nan
                )
                n_nonzero = int(nonzero_mask.sum())
                minimum_level_groups = np.nan
            else:
                missing_mask = _categorical_missing(raw).to_numpy()
                observed = raw.loc[~missing_mask].astype(str)
                distinct = int(observed.nunique(dropna=True))
                group_counts: list[int] = []
                for level, level_frame in frame.loc[
                    ~missing_mask
                ].groupby(column, dropna=False):
                    n_groups = int(
                        level_frame[group_col].astype(str).nunique()
                    )
                    group_counts.append(n_groups)
                    level_rows.append(
                        {
                            "scope": scope,
                            "feature_name": column,
                            "logical_feature": str(
                                candidate.logical_feature
                            ),
                            "level": str(level),
                            "n_records": int(len(level_frame)),
                            "n_connected_groups": n_groups,
                            "record_fraction": float(
                                len(level_frame) / len(frame)
                            ),
                            "independent_level_support": (
                                "supported"
                                if n_groups >= 5
                                else "thin_fewer_than_five_groups"
                            ),
                        }
                    )
                minimum_level_groups = (
                    int(min(group_counts)) if group_counts else 0
                )
                status, reason = _categorical_support_status(
                    distinct=distinct,
                    missing_fraction=float(missing_mask.mean()),
                    minimum_level_groups=int(minimum_level_groups),
                )
                counts = observed.value_counts(normalize=True)
                dominant_value = str(counts.index[0]) if len(counts) else ""
                dominant_fraction = (
                    float(counts.iloc[0]) if len(counts) else np.nan
                )
                quantiles = pd.Series(
                    np.nan, index=[0.05, 0.25, 0.5, 0.75, 0.95]
                )
                n_nonzero = np.nan
                nonzero_groups = np.nan
                zero_fraction = np.nan
            nonmissing_mask = ~missing_mask
            variable_groups = 0
            for _, group in frame.loc[nonmissing_mask].groupby(
                group_col, dropna=False
            ):
                series = (
                    pd.to_numeric(group[column], errors="coerce")
                    if kind == "numeric"
                    else group[column].astype(str)
                )
                if series.dropna().nunique() > 1:
                    variable_groups += 1
            memberships = _panel_memberships(column)
            effect_memberships = _effect_memberships(column)
            supervised_role = _supervised_source_role(column)
            if supervised_role == "sensitivity_only":
                decision = "sensitivity_only"
            elif supervised_role.startswith("pooled_"):
                decision = supervised_role
            elif supervised_role == "excluded_from_supervised_models":
                decision = (
                    "descriptive_only_unsupported"
                    if status.startswith(("unsupported", "thin"))
                    else "descriptive_atlas_only"
                )
            elif memberships:
                decision = (
                    "frozen_but_support_limited"
                    if status.startswith("unsupported")
                    else "frozen_low_dimensional_panel"
                )
            else:
                decision = (
                    "descriptive_only_unsupported"
                    if status.startswith("unsupported")
                    or status.startswith("thin")
                    else "descriptive_atlas_only"
                )
            support_rows.append(
                {
                    "scope": scope,
                    "feature_name": column,
                    "source_column": column,
                    "logical_feature": str(candidate.logical_feature),
                    "kind": kind,
                    "block": str(candidate.distance_block),
                    "n_records": int(len(frame)),
                    "n_connected_groups": int(
                        frame[group_col].astype(str).nunique()
                    ),
                    "n_nonmissing": int(nonmissing_mask.sum()),
                    "missing_fraction": float(missing_mask.mean()),
                    "n_distinct": distinct,
                    "n_nonzero": n_nonzero,
                    "n_nonzero_groups": nonzero_groups,
                    "zero_fraction": zero_fraction,
                    "dominant_value": dominant_value,
                    "dominant_fraction": dominant_fraction,
                    "minimum_category_groups": minimum_level_groups,
                    "n_within_group_variable_groups": int(variable_groups),
                    "q05": float(quantiles.loc[0.05]),
                    "q25": float(quantiles.loc[0.25]),
                    "median": float(quantiles.loc[0.5]),
                    "q75": float(quantiles.loc[0.75]),
                    "q95": float(quantiles.loc[0.95]),
                    "support_status": status,
                    "support_reason": reason,
                    "atlas_status": (
                        "generated_descriptive"
                        if distinct > 1 and float(missing_mask.mean()) < 0.90
                        else "skipped_insufficient_support"
                    ),
                    "frozen_panel_memberships": "|".join(memberships),
                    "effect_eligible_panel_memberships": "|".join(
                        effect_memberships
                    ),
                    "supervised_role": supervised_role,
                    "x_only_feature_decision": decision,
                    "target_used_for_decision": False,
                }
            )
    return pd.DataFrame(support_rows), pd.DataFrame(level_rows)


def _recommended_representative(
    feature_a: str, feature_b: str
) -> tuple[str, str]:
    memberships_a = _panel_memberships(feature_a)
    memberships_b = _panel_memberships(feature_b)
    property_pair = (
        feature_a in PROPERTY_SOURCE_SET and feature_b in PROPERTY_SOURCE_SET
    )
    if property_pair:
        return (
            "solvent_polarity_coordinate",
            "correlated properties enter only through the fold-local polarity PCA",
        )
    if memberships_a and not memberships_b:
        return feature_a, "feature_a is retained in a frozen low-dimensional panel"
    if memberships_b and not memberships_a:
        return feature_b, "feature_b is retained in a frozen low-dimensional panel"
    return min(feature_a, feature_b), "deterministic lexical representative for audit"


def _numeric_redundancy(
    records: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    *,
    scopes: Sequence[str],
    scope_col: str,
    threshold: float,
) -> pd.DataFrame:
    numeric_columns = tuple(
        candidate_registry.loc[
            candidate_registry["kind"].astype(str).eq("numeric"),
            "selected_column",
        ].astype(str)
    )
    rows: list[dict[str, object]] = []
    for scope, frame in _scope_frames(records, scopes, scope_col=scope_col):
        numeric = frame.loc[:, list(numeric_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        for left_index, feature_a in enumerate(numeric_columns):
            for feature_b in numeric_columns[left_index + 1 :]:
                pair = numeric.loc[:, [feature_a, feature_b]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if (
                    len(pair) < 3
                    or pair[feature_a].nunique() <= 1
                    or pair[feature_b].nunique() <= 1
                ):
                    continue
                rho = float(
                    pair[feature_a].corr(pair[feature_b], method="spearman")
                )
                if not np.isfinite(rho) or abs(rho) < float(threshold):
                    continue
                x = pair[feature_a].to_numpy(float)
                y = pair[feature_b].to_numpy(float)
                design = np.column_stack([np.ones(len(x)), x])
                coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                fitted = design @ coefficients
                total = float(np.sum((y - np.mean(y)) ** 2))
                residual = float(np.sum((y - fitted) ** 2))
                affine_r2 = (
                    1.0 - residual / total if total > 0 else np.nan
                )
                representative, reason = _recommended_representative(
                    feature_a, feature_b
                )
                rows.append(
                    {
                        "scope": scope,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "n_pairwise_complete": int(len(pair)),
                        "spearman_rho": rho,
                        "abs_spearman_rho": abs(rho),
                        "exact_affine_r2": affine_r2,
                        "redundancy_status": (
                            "near_exact_affine"
                            if np.isfinite(affine_r2)
                            and affine_r2 >= 0.999999
                            else "high_rank_correlation"
                        ),
                        "recommended_representative": representative,
                        "reason": reason,
                        "target_used": False,
                    }
                )
    return pd.DataFrame(rows)


def _connected_components(
    pairs: pd.DataFrame, scope: str
) -> list[tuple[str, ...]]:
    selected = pairs.loc[pairs["scope"].astype(str).eq(scope)]
    adjacency: dict[str, set[str]] = {}
    for row in selected.itertuples(index=False):
        left, right = str(row.feature_a), str(row.feature_b)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components: list[tuple[str, ...]] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        stack = [start]
        members: set[str] = set()
        while stack:
            value = stack.pop()
            if value in members:
                continue
            members.add(value)
            stack.extend(adjacency.get(value, ()))
        remaining -= members
        if len(members) > 1:
            components.append(tuple(sorted(members)))
    return components


def _redundancy_clusters(
    candidate_columns: Sequence[str],
    pairs: pd.DataFrame,
    scopes: Sequence[str],
) -> pd.DataFrame:
    available = set(map(str, candidate_columns))
    rows: list[dict[str, object]] = []
    normalized_scopes = ["all97" if value in {"all", "all97"} else str(value) for value in scopes]
    for scope in normalized_scopes:
        for index, members in enumerate(
            _connected_components(pairs, scope), start=1
        ):
            rows.append(
                {
                    "scope": scope,
                    "cluster_id": f"empirical_{index:02d}",
                    "cluster_type": "empirical_high_rank_correlation",
                    "members": "|".join(members),
                    "relationship": "absolute Spearman rho at or above the frozen threshold",
                    "frozen_representation": "see pair-level recommended representative",
                    "target_used": False,
                }
            )
        composition = tuple(
            name for name in SOLVENT_FRACTION_SOURCES if name in available
        )
        if len(composition) > 1:
            rows.append(
                {
                    "scope": scope,
                    "cluster_id": "structural_solvent_closure",
                    "cluster_type": "structural_compositional",
                    "members": "|".join(composition),
                    "relationship": "closed composition; fractions are sum constrained",
                    "frozen_representation": (
                        "water_to_organic_balance|"
                        "ethanol_to_isopropanol_balance"
                    ),
                    "target_used": False,
                }
            )
        conductive = (
            "conductive_agent_present",
            "conductive_agent_dose_mg_per_mgcat",
        )
        if set(conductive).issubset(available):
            rows.append(
                {
                    "scope": scope,
                    "cluster_id": "structural_conductive_encoding",
                    "cluster_type": "structural_indicator_dose",
                    "members": "|".join(conductive),
                    "relationship": "presence and dose encode the same sparse addition",
                    "frozen_representation": "conductive_agent_present_sensitivity_only",
                    "target_used": False,
                }
            )
    return pd.DataFrame(rows)


def _feature_decisions(
    support: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    cluster_lookup: dict[tuple[str, str], list[str]] = {}
    for row in clusters.itertuples(index=False):
        for feature in str(row.members).split("|"):
            cluster_lookup.setdefault((str(row.scope), feature), []).append(
                str(row.cluster_id)
            )
    rows: list[dict[str, object]] = []
    for row in support.itertuples(index=False):
        memberships = str(row.frozen_panel_memberships)
        decision = str(row.x_only_feature_decision)
        supervised_role = str(row.supervised_role)
        if supervised_role.startswith("pooled_"):
            reason = (
                "raw source is not estimated independently; it contributes "
                f"only to a frozen pooled coordinate; support={row.support_status}"
            )
        elif supervised_role == "excluded_from_supervised_models":
            reason = (
                "excluded from every supervised panel by the target-blind "
                f"feature-reduction policy; support={row.support_status}"
            )
        elif supervised_role == "sensitivity_only":
            reason = (
                "retained only in the conductive-agent sensitivity panel; "
                f"support={row.support_status}"
            )
        elif memberships:
            reason = (
                "retained by the frozen context-plus-one panel definition; "
                f"support={row.support_status}"
            )
        else:
            reason = (
                "not retained in the low-dimensional supervised panels; "
                f"support={row.support_status}"
            )
        rows.append(
            {
                "scope": str(row.scope),
                "feature_name": str(row.feature_name),
                "support_status": str(row.support_status),
                "redundancy_clusters": "|".join(
                    cluster_lookup.get(
                        (str(row.scope), str(row.feature_name)), []
                    )
                ),
                "frozen_panel_memberships": memberships,
                "effect_eligible_panel_memberships": str(
                    row.effect_eligible_panel_memberships
                ),
                "supervised_role": supervised_role,
                "decision": decision,
                "reason": reason,
                "target_used_for_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _data_geometry(
    support: pd.DataFrame,
    *,
    panel_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, frame in support.groupby("scope", sort=False):
        first = frame.iloc[0]
        rows.append(
            {
                "scope": str(scope),
                "n_records": int(first["n_records"]),
                "n_connected_groups": int(first["n_connected_groups"]),
                "n_registered_candidate_predictors": int(len(frame)),
                "candidate_to_record_ratio": float(
                    len(frame) / int(first["n_records"])
                ),
                "candidate_to_group_ratio": float(
                    len(frame) / int(first["n_connected_groups"])
                ),
                "n_numeric_candidates": int(
                    frame["kind"].astype(str).eq("numeric").sum()
                ),
                "n_categorical_candidates": int(
                    frame["kind"].astype(str).eq("categorical").sum()
                ),
                "n_candidates_with_any_within_group_variation": int(
                    pd.to_numeric(
                        frame["n_within_group_variable_groups"],
                        errors="coerce",
                    ).gt(0).sum()
                ),
                "n_supported_descriptive": int(
                    frame["support_status"]
                    .astype(str)
                    .eq("supported_descriptive")
                    .sum()
                ),
                "n_thin_or_unsupported": int(
                    (
                        ~frame["support_status"]
                        .astype(str)
                        .eq("supported_descriptive")
                    ).sum()
                ),
                "median_missing_fraction": float(
                    pd.to_numeric(
                        frame["missing_fraction"], errors="coerce"
                    ).median()
                ),
                "n_frozen_panels": int(panel_count),
                "target_used": False,
            }
        )
    return pd.DataFrame(rows)


def audit_feature_support_and_redundancy(
    records: pd.DataFrame | str | Path,
    registered_panel_schema: pd.DataFrame | str | Path,
    *,
    scopes: Sequence[str] = ("all97",),
    scope_col: str = SCOPE_SOURCE,
    group_col: str = GROUP_SOURCE,
    abs_spearman_threshold: float = 0.90,
    require_exact_candidate_count: bool = True,
) -> dict[str, Any]:
    """Audit feature support and redundancy without reading any target.

    Only the allowlisted registered predictors plus the grouping/scope metadata
    are accessed from ``records``.  Extra columns, including any outcome or
    posterior fields supplied by a caller, are ignored.
    """

    if not 0.0 <= float(abs_spearman_threshold) <= 1.0:
        raise ValueError("abs_spearman_threshold must lie in [0, 1]")
    frame = _read_frame(records)
    registry = registered_candidate_registry(
        registered_panel_schema,
        expected_count=(
            EXPECTED_REGISTERED_CANDIDATE_COUNT
            if require_exact_candidate_count
            else None
        ),
    )
    candidates = tuple(registry["selected_column"].astype(str))
    required_metadata = {scope_col, group_col}
    missing_metadata = sorted(required_metadata - set(frame.columns))
    missing_candidates = sorted(set(candidates) - set(frame.columns))
    if missing_metadata or missing_candidates:
        raise KeyError(
            "Feature audit input is incomplete: "
            f"missing_metadata={missing_metadata}, "
            f"missing_candidates={missing_candidates}"
        )
    # This explicit subset is the safeguard against accidentally scanning the
    # remaining records-table fields.
    audited = frame.loc[
        :, list(dict.fromkeys((*candidates, scope_col, group_col)))
    ].copy()
    support, categorical_levels = _feature_support_tables(
        audited,
        registry,
        scopes=scopes,
        scope_col=scope_col,
        group_col=group_col,
    )
    redundancy_pairs = _numeric_redundancy(
        audited,
        registry,
        scopes=scopes,
        scope_col=scope_col,
        threshold=float(abs_spearman_threshold),
    )
    clusters = _redundancy_clusters(candidates, redundancy_pairs, scopes)
    decisions = _feature_decisions(support, clusters)
    geometry = _data_geometry(
        support, panel_count=len(FROZEN_ASSOCIATION_PANELS)
    )
    summary: dict[str, object] = {
        "status": "pass",
        "candidate_source": (
            "hash_bound_registered_panel_schema_present_selected_columns"
        ),
        "n_registered_candidate_predictors": int(len(candidates)),
        "candidate_predictors": list(candidates),
        "scopes": [
            "all97" if value in {"all", "all97"} else str(value)
            for value in scopes
        ],
        "abs_spearman_threshold": float(abs_spearman_threshold),
        "records_table_columns_scanned": list(
            dict.fromkeys((*candidates, scope_col, group_col))
        ),
        "unregistered_records_columns_scanned": False,
        "target_columns_read": False,
        "target_used_for_support": False,
        "target_used_for_redundancy": False,
        "frozen_panels": list(FROZEN_ASSOCIATION_PANELS),
        "composition_property_mutually_exclusive": True,
    }
    return {
        "candidate_registry": registry,
        "candidate_allowlist": candidates,
        "feature_support": support,
        "feature_support_audit": support,
        "categorical_levels": categorical_levels,
        "numeric_redundancy_pairs": redundancy_pairs,
        "redundancy_clusters": clusters,
        "feature_decisions": decisions,
        "data_geometry": geometry,
        "data_geometry_summary": geometry,
        "summary": summary,
    }


feature_support_and_redundancy_audit = audit_feature_support_and_redundancy


__all__ = [
    "AssociationFeatureBuilder",
    "AssociationFeaturePanel",
    "COMPOSITION_SOURCE_SET",
    "DIRECT_CONTEXT_SOURCES",
    "DIRECT_PRIMARY_FEATURE_SOURCES",
    "ENGINEERED_FEATURE_BLOCKS",
    "ENGINEERED_FEATURE_SOURCES",
    "EXPECTED_REGISTERED_CANDIDATE_COUNT",
    "FROZEN_ASSOCIATION_PANELS",
    "HARD_EXCLUDED_SUPERVISED_SOURCES",
    "POOLED_COMPOSITION_SOURCES",
    "POOLED_PROPERTY_SOURCES",
    "PRIMARY_FORMULATION_FEATURES",
    "PROPERTY_SOURCE_SET",
    "SOLVENT_FRACTION_SOURCES",
    "association_engineered_feature_definitions",
    "association_feature_definitions",
    "association_panel_registry",
    "audit_feature_support_and_redundancy",
    "derive_candidate_predictor_allowlist",
    "feature_support_and_redundancy_audit",
    "frozen_association_panel_registry",
    "get_association_panel",
    "registered_candidate_predictors",
    "registered_candidate_registry",
    "required_association_source_columns",
]
