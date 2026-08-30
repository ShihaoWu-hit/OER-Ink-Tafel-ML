"""Pre-registered feature panels for the Ink--Tafel study.

The canonical registry has changed names a few times during curation.  A
``PhysicalFeature`` therefore owns a short, ordered alias list, but panel
resolution deliberately chooses *one* alias only.  This is important: adding
both a raw and a derived expression of the same physical quantity would give
that quantity an unintended extra vote in a distance or additive model.

Target-like columns are never selected by inference.  Every predictor must be
listed here and is checked by :func:`validate_no_target_leakage` before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class PhysicalFeature:
    """One physical quantity and the canonical aliases accepted for it."""

    name: str
    aliases: tuple[str, ...]
    kind: str
    block: str

    def __post_init__(self) -> None:
        if self.kind not in {"numeric", "categorical"}:
            raise ValueError(f"Unsupported feature kind for {self.name!r}: {self.kind!r}")


@dataclass(frozen=True)
class ResolvedPanel:
    """A feature panel resolved against one concrete canonical table."""

    name: str
    columns: tuple[str, ...]
    logical_names: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    blocks: Mapping[str, tuple[str, ...]]
    missing_logical_features: tuple[str, ...]


FEATURES: dict[str, PhysicalFeature] = {
    # Catalyst identity/context.  Source is kept categorical: no ordinal code is
    # imposed on manufacturers or synthesis routes.
    "catalyst_identity": PhysicalFeature(
        "catalyst_identity",
        (
            "catalyst_active_material_type",
            "catalyst_type",
            "catalyst_identity",
            "catalyst",
        ),
        "categorical",
        "catalyst",
    ),
    "catalyst_source": PhysicalFeature(
        "catalyst_source", ("catalyst_source", "catalyst_manufacturer", "catalyst_origin"), "categorical", "catalyst"
    ),
    "catalyst_support": PhysicalFeature(
        "catalyst_support",
        ("catalyst_support", "catalyst_support_type", "support_type"),
        "categorical",
        "catalyst",
    ),
    # Measurement/electrode context.
    "electrode_type": PhysicalFeature(
        "electrode_type", ("electrode_type", "working_electrode"), "categorical", "electrode"
    ),
    "catalyst_loading": PhysicalFeature(
        "catalyst_loading",
        (
            "catalyst_loading_mg_cm2_selected",
            "catalyst_loading_mg_cm2",
            "catalyst_loading_mg_cm_2",
            "catalyst_loading",
        ),
        "numeric",
        "electrode",
    ),
    "electrolyte_identity": PhysicalFeature(
        "electrolyte_identity", ("electrolyte_identity", "electrolyte_type"), "categorical", "electrolyte"
    ),
    "electrolyte_molarity": PhysicalFeature(
        "electrolyte_molarity",
        ("electrolyte_molarity_M", "electrolyte_concentration_M", "electrolyte_molarity"),
        "numeric",
        "electrolyte",
    ),
    "rotation_rate": PhysicalFeature(
        "rotation_rate", ("rotation_rate_rpm", "rpm"), "numeric", "measurement"
    ),
    "scan_rate": PhysicalFeature(
        "scan_rate", ("scan_rate_mV_s", "scan_rate_mv_s", "scan_rate"), "numeric", "measurement"
    ),
    "ir_correction_status": PhysicalFeature(
        "ir_correction_status",
        ("ir_correction_status",),
        "categorical",
        "ir_correction",
    ),
    "electrode_area": PhysicalFeature(
        "electrode_area", ("electrode_area_cm2",), "numeric", "electrode"
    ),
    # Primary ink representation: exactly one expression for each quantity.
    "catalyst_concentration": PhysicalFeature(
        "catalyst_concentration",
        ("catalyst_concentration_mg_mL", "catalyst_concentration_mg_ml"),
        "numeric",
        "ink_amounts",
    ),
    "nafion_to_solvent_ratio": PhysicalFeature(
        "nafion_to_solvent_ratio",
        ("nafion_to_solvent_ratio", "ionomer_to_solvent_ratio"),
        "numeric",
        "ink_amounts",
    ),
    "ionomer_concentration": PhysicalFeature(
        "ionomer_concentration",
        ("ionomer_concentration_wt_percent",),
        "numeric",
        "ink_amounts",
    ),
    "conductive_agent_presence": PhysicalFeature(
        "conductive_agent_presence",
        ("conductive_agent_present", "conductive_agent_presence"),
        "categorical",
        "conductive_agent",
    ),
    "conductive_agent_dose": PhysicalFeature(
        "conductive_agent_dose",
        (
            "conductive_agent_dose_mg_mgcat",
            "conductive_agent_dose_mg_per_mgcat",
            "conductive_agent_mass_ratio_to_catalyst",
            "conductive_agent_to_catalyst_mass_ratio",
        ),
        "numeric",
        "conductive_agent",
    ),
    # User-directed 2026-07-22 amendment: retain all five reported solvent
    # fractions in composition panels, including methanol.  The five fractions
    # remain sum-constrained; this is an explicit full-composition
    # representation rather than a four-coordinate reference representation.
    "acetone_fraction": PhysicalFeature(
        "acetone_fraction", ("solvent_fraction_acetone", "F_acetone"), "numeric", "solvent_composition"
    ),
    "water_fraction": PhysicalFeature(
        "water_fraction",
        ("solvent_fraction_water", "solvent_fraction_deionized_water", "F_deionized_water"),
        "numeric",
        "solvent_composition",
    ),
    "ethanol_fraction": PhysicalFeature(
        "ethanol_fraction", ("solvent_fraction_ethanol", "F_ethanol"), "numeric", "solvent_composition"
    ),
    "isopropanol_fraction": PhysicalFeature(
        "isopropanol_fraction",
        ("solvent_fraction_isopropanol", "solvent_fraction_ipa", "F_isopropanol"),
        "numeric",
        "solvent_composition",
    ),
    "methanol_fraction": PhysicalFeature(
        "methanol_fraction",
        ("solvent_fraction_methanol", "F_methanol"),
        "numeric",
        "solvent_composition",
    ),
    # Process additions are a missingness sensitivity panel, not primary ink.
    "sonication_time": PhysicalFeature(
        "sonication_time", ("sonication_time_min", "ultrasonication_time_min"), "numeric", "ink_process"
    ),
    "dispersion_method": PhysicalFeature(
        "dispersion_method", ("dispersion_method",), "categorical", "ink_process"
    ),
    "drop_volume": PhysicalFeature(
        "drop_volume", ("drop_volume_uL", "dropwise_addition_uL"), "numeric", "ink_process"
    ),
    "drying_method": PhysicalFeature(
        "drying_method", ("drying_method", "drying_process", "drying_protocol"), "categorical", "ink_process"
    ),
    "drying_temperature": PhysicalFeature(
        "drying_temperature", ("drying_temperature_C", "drying_temp_C"), "numeric", "ink_process"
    ),
    "drying_time": PhysicalFeature(
        "drying_time", ("drying_time_min", "drying_duration_min"), "numeric", "ink_process"
    ),
    # These are mixture-property proxies, not measured properties of the ink.
    "mixture_viscosity_proxy": PhysicalFeature(
        "mixture_viscosity_proxy",
        ("solvent_mixture_property_proxy_viscosity_mPa_s",),
        "numeric",
        "solvent_property_proxy",
    ),
    "mixture_surface_tension_proxy": PhysicalFeature(
        "mixture_surface_tension_proxy",
        ("solvent_mixture_property_proxy_surface_tension_mN_m",),
        "numeric",
        "solvent_property_proxy",
    ),
    "mixture_density_proxy": PhysicalFeature(
        "mixture_density_proxy",
        ("solvent_mixture_property_proxy_density_kg_m3",),
        "numeric",
        "solvent_property_proxy",
    ),
    "mixture_dielectric_proxy": PhysicalFeature(
        "mixture_dielectric_proxy",
        ("solvent_mixture_property_proxy_dielectric_constant",),
        "numeric",
        "solvent_property_proxy",
    ),
    "mixture_dielectric_proxy_best_available": PhysicalFeature(
        "mixture_dielectric_proxy_best_available",
        (
            "solvent_mixture_property_proxy_dielectric_constant_best_available",
            "solvent_mixture_property_proxy_dielectric_constant__best_available",
        ),
        "numeric",
        "solvent_property_proxy",
    ),
    "mixture_dielectric_proxy_is_estimated": PhysicalFeature(
        "mixture_dielectric_proxy_is_estimated",
        (
            "solvent_mixture_property_proxy_dielectric_constant_estimated_flag",
            "solvent_mixture_property_proxy_dielectric_constant__best_available_is_estimated",
        ),
        "numeric",
        "solvent_property_proxy_estimation_flag",
    ),
}


CATALYST = ("catalyst_identity", "catalyst_support")
CONDITIONS = (
    "catalyst_loading",
    "electrolyte_identity",
    "electrolyte_molarity",
    "rotation_rate",
    "scan_rate",
    "ir_correction_status",
)
INK_AMOUNTS = (
    "catalyst_concentration",
    "nafion_to_solvent_ratio",
    "ionomer_concentration",
    "conductive_agent_presence",
    "conductive_agent_dose",
)
SOLVENT_COMPOSITION = (
    "acetone_fraction",
    "water_fraction",
    "ethanol_fraction",
    "isopropanol_fraction",
    "methanol_fraction",
)
# Registered sensitivity features.  Drying method/temperature/time are omitted
# from modeling because their primary-scope missing ratios are 72--92%.  The
# 24% missing sonication time is retained only in this non-primary panel, with
# imputation performed inside inner grouped CV.
INK_PROCESS = ("dispersion_method", "sonication_time", "drop_volume", "electrode_area")
PROPERTY_PROXY_CORE = (
    "mixture_viscosity_proxy",
    "mixture_surface_tension_proxy",
    "mixture_density_proxy",
)
PROPERTY_PROXY_SOURCE_BACKED = PROPERTY_PROXY_CORE + (
    "mixture_dielectric_proxy",
)
PROPERTY_PROXY_ESTIMATED = PROPERTY_PROXY_CORE + (
    "mixture_dielectric_proxy_best_available",
    "mixture_dielectric_proxy_is_estimated",
)
# Used only by the composition/property mutual-exclusion assertion.  The two
# dielectric representations never occur in the same registered panel.
PROPERTY_PROXY = tuple(dict.fromkeys(PROPERTY_PROXY_SOURCE_BACKED + PROPERTY_PROXY_ESTIMATED))


PANEL_FEATURES: dict[str, tuple[str, ...]] = {
    "dummy": (),
    "catalyst_only": CATALYST,
    "ink_only": INK_AMOUNTS + SOLVENT_COMPOSITION,
    "conditions_only": CONDITIONS,
    "context": CATALYST + CONDITIONS,
    # Non-solvent ink amounts are the strict incremental reference for every
    # property-proxy panel.  This prevents a property contrast from also
    # introducing catalyst concentration, Nafion ratio, or conductive-agent
    # terms relative to the plain context panel.
    "context_plus_ink_amounts": CATALYST + CONDITIONS + INK_AMOUNTS,
    "context_plus_ink": CATALYST + CONDITIONS + INK_AMOUNTS + SOLVENT_COMPOSITION,
    "context_plus_ink_process": CATALYST + CONDITIONS + INK_AMOUNTS + SOLVENT_COMPOSITION + INK_PROCESS,
    # Property proxies replace, rather than duplicate, solvent composition.
    # The core panel contains only the three fully source-backed properties.
    "context_plus_property_proxy_core": CATALYST + CONDITIONS + INK_AMOUNTS + PROPERTY_PROXY_CORE,
    # The strict four-property sensitivity is evaluated on complete cases.  Its
    # context-plus-ink-amounts reference is fitted on the identical subset in
    # ``evaluate.py``.
    "context_plus_property_proxy_4p_source_backed": (
        CATALYST + CONDITIONS + INK_AMOUNTS + PROPERTY_PROXY_SOURCE_BACKED
    ),
    # The estimated sensitivity uses the explicitly coalesced strict-first,
    # model-estimated-fallback dielectric column.  It never silently aliases the
    # strict dielectric value.
    "context_plus_property_proxy_4p_estimated": (
        CATALYST + CONDITIONS + INK_AMOUNTS + PROPERTY_PROXY_ESTIMATED
    ),
}


_FORBIDDEN_EXACT = {
    "target",
    "y",
    "response",
    "current_density_mA_cm2",
    "potential_V",
    "potential_v",
}
_FORBIDDEN_TOKENS = (
    "tafel",
    "slope",
    "overpotential",
    "posterior",
    "refit",
    "fit_derived",
    "target_",
)
_FORBIDDEN_PREFIXES = ("b_", "y_")


def is_target_like_column(column: str) -> bool:
    """Return whether a column is forbidden as a predictor.

    The check is intentionally conservative.  Curated measurement conditions
    such as ``scan_rate_mV_s`` remain allowed, while curve coordinates and any
    fitted or posterior quantity are excluded.
    """

    name = str(column).strip().lower()
    return (
        name in _FORBIDDEN_EXACT
        or name.startswith(_FORBIDDEN_PREFIXES)
        or any(token in name for token in _FORBIDDEN_TOKENS)
    )


def validate_no_target_leakage(columns: Iterable[str]) -> None:
    """Raise if any proposed predictor can encode the response or curve fit."""

    leaking = sorted({str(column) for column in columns if is_target_like_column(str(column))})
    if leaking:
        raise ValueError(f"Target/curve-derived columns are forbidden predictors: {leaking}")


def resolve_feature_panel(
    frame: pd.DataFrame,
    panel_name: str,
    *,
    require_all: bool = False,
) -> ResolvedPanel:
    """Resolve a pre-registered panel to columns present in ``frame``.

    Aliases are ordered and only the first present alias is used.  Missing
    logical features are reported so a run can preserve its exact schema.
    ``require_all=True`` is useful for frozen analyses; exploratory curation can
    retain the missing-feature manifest with ``require_all=False``.
    """

    if panel_name not in PANEL_FEATURES:
        raise KeyError(f"Unknown feature panel {panel_name!r}; choose from {sorted(PANEL_FEATURES)}")

    logical_names: list[str] = []
    columns: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []
    missing: list[str] = []
    block_columns: dict[str, list[str]] = {}

    available = set(map(str, frame.columns))
    for logical_name in PANEL_FEATURES[panel_name]:
        feature = FEATURES[logical_name]
        selected = next((alias for alias in feature.aliases if alias in available), None)
        if selected is None:
            missing.append(logical_name)
            continue
        if selected in columns:
            raise AssertionError(f"Column {selected!r} resolves more than one physical feature")
        logical_names.append(logical_name)
        columns.append(selected)
        (numeric if feature.kind == "numeric" else categorical).append(selected)
        block_columns.setdefault(feature.block, []).append(selected)

    if require_all and missing:
        raise ValueError(f"Panel {panel_name!r} is missing required logical features: {missing}")
    if panel_name != "dummy" and not columns:
        raise ValueError(f"Panel {panel_name!r} has no usable features in the supplied table")

    validate_no_target_leakage(columns)
    if len(columns) != len(set(columns)):
        raise AssertionError(f"Panel {panel_name!r} contains duplicate predictor columns")

    composition = set(SOLVENT_COMPOSITION).intersection(logical_names)
    properties = set(PROPERTY_PROXY).intersection(logical_names)
    if composition and properties:
        raise AssertionError(
            f"Panel {panel_name!r} mixes solvent composition and property proxies: "
            f"composition={sorted(composition)}, properties={sorted(properties)}"
        )

    return ResolvedPanel(
        name=panel_name,
        columns=tuple(columns),
        logical_names=tuple(logical_names),
        numeric_columns=tuple(numeric),
        categorical_columns=tuple(categorical),
        blocks={name: tuple(values) for name, values in block_columns.items()},
        missing_logical_features=tuple(missing),
    )


def panel_schema_frame(frame: pd.DataFrame, panel_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Return a long-form, auditable schema for the requested panels."""

    names = tuple(panel_names or PANEL_FEATURES)
    rows: list[dict[str, object]] = []
    for panel_name in names:
        resolved = resolve_feature_panel(frame, panel_name)
        selected_by_logical = dict(zip(resolved.logical_names, resolved.columns, strict=True))
        for logical_name in PANEL_FEATURES[panel_name]:
            feature = FEATURES[logical_name]
            rows.append(
                {
                    "panel": panel_name,
                    "logical_feature": logical_name,
                    "selected_column": selected_by_logical.get(logical_name),
                    "kind": feature.kind,
                    "distance_block": feature.block,
                    "status": "present" if logical_name in selected_by_logical else "missing",
                }
            )
        if panel_name == "dummy":
            rows.append(
                {
                    "panel": panel_name,
                    "logical_feature": "__constant__",
                    "selected_column": None,
                    "kind": "numeric",
                    "distance_block": "dummy",
                    "status": "synthetic",
                }
            )
    return pd.DataFrame(rows)
