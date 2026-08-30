"""Frozen contract loading for the active all-97 cardinal classifier.

The historical ``cardinal_classification`` section deliberately remains
unchanged.  This module authenticates the separate versioned contract used by
``posterior-cardinal-classify`` and the two target-blind reduced-feature
registries inherited from the completed feature-association run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping

import yaml

from .config import StudyConfig
from .feature_association_features import (
    FROZEN_ASSOCIATION_PANELS,
    association_engineered_feature_definitions,
    association_panel_registry,
)
from .provenance import sha256_file


CONTRACT_SECTION = "posterior_cardinal_classification"
EXPECTED_CONTRACT_ID = "posterior_cardinal_all97_v2"
SUPPORTED_CONTRACT_STATUSES = {
    "posterior_cardinal_all97_v2": "registered_fixed_all97_v2",
    "posterior_cardinal_all97_v3": "registered_fixed_all97_v3",
    "posterior_cardinal_all97_v4": "registered_fixed_all97_v4",
}
EXPECTED_STAGE = "posterior-cardinal-classify"
EXPECTED_RECORDS = 97
EXPECTED_GROUPS = 80
EXPECTED_WIDTHS = (0.15, 0.20, 0.25)
EXPECTED_OUTER_SEEDS = (
    42,
    142,
    242,
    342,
    442,
    542,
    642,
    742,
    842,
    942,
)
EXPECTED_FORMAL_PANELS = (
    "context_plus_catalyst_concentration",
    "context_plus_ionomer_ratio",
    "context_plus_water_organic",
    "context_plus_ethanol_isopropanol",
    "context_plus_viscosity",
    "context_plus_polarity",
)
EXPECTED_SENSITIVITY_PANELS = (
    "context_plus_conductive_sensitivity",
)


def _as_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return dict(value)


def _exact_sequence(
    value: Any,
    expected: tuple[Any, ...],
    *,
    field: str,
) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a sequence")
    observed = tuple(value)
    if observed != expected:
        raise ValueError(
            f"Frozen posterior-cardinal contract changed {field}: "
            f"expected={expected}, observed={observed}"
        )


def _validate_contract_contents(
    contract: Mapping[str, Any],
    *,
    expected_contract_id: str = EXPECTED_CONTRACT_ID,
) -> None:
    """Validate critical scientific invariants after byte authentication."""

    if contract.get("contract_id") != expected_contract_id:
        raise ValueError("Posterior-cardinal contract ID changed")
    if contract.get("status") != "frozen_before_first_fit":
        raise ValueError("Posterior-cardinal contract is not frozen")
    if contract.get("stage") != EXPECTED_STAGE:
        raise ValueError("Posterior-cardinal stage name changed")

    scope = _as_mapping(contract.get("scientific_scope"), field="scientific_scope")
    if scope.get("population") != "all97":
        raise ValueError("Active posterior-cardinal population must remain all97")
    if int(scope.get("expected_records", -1)) != EXPECTED_RECORDS:
        raise ValueError("Active posterior-cardinal record count must remain 97")
    if int(scope.get("expected_connected_groups", -1)) != EXPECTED_GROUPS:
        raise ValueError(
            "Active posterior-cardinal connected-group count must remain 80"
        )
    if scope.get("one_ml_row_per_record") is not True:
        raise ValueError("Every record must remain one ML row")
    if scope.get("posterior_draws_expanded_as_samples") is not False:
        raise ValueError("Posterior draws may not be expanded as ML samples")

    registry = _as_mapping(
        contract.get("reduced_feature_registry"),
        field="reduced_feature_registry",
    )
    if registry.get("context_panel") != "context_core":
        raise ValueError("Frozen context panel changed")
    _exact_sequence(
        registry.get("formal_increment_panels"),
        EXPECTED_FORMAL_PANELS,
        field="reduced_feature_registry.formal_increment_panels",
    )
    _exact_sequence(
        registry.get("sensitivity_only_panels"),
        EXPECTED_SENSITIVITY_PANELS,
        field="reduced_feature_registry.sensitivity_only_panels",
    )
    all_panels = (
        "context_core",
        *EXPECTED_FORMAL_PANELS,
        *EXPECTED_SENSITIVITY_PANELS,
    )
    missing_panels = sorted(set(all_panels) - set(FROZEN_ASSOCIATION_PANELS))
    if missing_panels:
        raise ValueError(
            f"Contract references unknown reduced panels: {missing_panels}"
        )

    label = _as_mapping(contract.get("label_scheme"), field="label_scheme")
    _exact_sequence(
        tuple(float(value) for value in label.get("anchors_mV_dec", ())),
        (30.0, 40.0, 60.0, 120.0),
        field="label_scheme.anchors_mV_dec",
    )
    _exact_sequence(
        tuple(
            float(value)
            for value in label.get("all_evaluated_relative_half_widths", ())
        ),
        EXPECTED_WIDTHS,
        field="label_scheme.all_evaluated_relative_half_widths",
    )
    if float(label.get("primary_relative_half_width", -1)) != 0.20:
        raise ValueError("Primary classic-band width must remain 0.20")
    if (
        _as_mapping(
            label.get("separate_30_vs_40_model"),
            field="label_scheme.separate_30_vs_40_model",
        ).get("enabled")
        is not False
    ):
        raise ValueError("The unsupported 30-vs-40 model must remain disabled")

    outer = _as_mapping(contract.get("outer_validation"), field="outer_validation")
    if int(outer.get("folds", -1)) != 5 or int(outer.get("repeats", -1)) != 10:
        raise ValueError("Outer validation must remain 10 repeats of 5 folds")
    _exact_sequence(
        tuple(int(value) for value in outer.get("repeat_seeds", ())),
        EXPECTED_OUTER_SEEDS,
        field="outer_validation.repeat_seeds",
    )
    inner = _as_mapping(contract.get("inner_validation"), field="inner_validation")
    if int(inner.get("folds", -1)) != 4:
        raise ValueError("Grouped inner validation must remain four-fold")
    preprocessing = _as_mapping(
        contract.get("preprocessing"), field="preprocessing"
    )
    if int(preprocessing.get("rare_min_frequency", -1)) != 3:
        raise ValueError("Rare-level pooling threshold must remain three")
    if (
        preprocessing.get("fitted_inside_every_inner_and_outer_training_fold")
        is not True
    ):
        raise ValueError("Preprocessing must remain fold-local")
    if (
        preprocessing.get("parallel_backend")
        != "deterministic_independent_outer_split_threads"
    ):
        raise ValueError("Frozen outer-split execution backend changed")

    models = _as_mapping(contract.get("models"), field="models")
    primary = _as_mapping(models.get("primary"), field="models.primary")
    if primary.get("inference") != "deterministic_gaussian_prior_MAP":
        raise ValueError("Primary estimator must remain Gaussian-prior MAP")
    if primary.get("full_bayesian_posterior_fit") is not False:
        raise ValueError("A MAP fit must not be described as full Bayesian")
    _exact_sequence(
        tuple(
            float(value)
            for value in primary.get(
                "shared_gate_and_conditional_feature_prior_scale_grid", ()
            )
        ),
        (0.25, 0.5, 1.0, 2.0),
        field="models.primary.feature_prior_scale_grid",
    )
    if _as_mapping(models.get("stack"), field="models.stack").get("enabled") is not False:
        raise ValueError("The optional stack must remain disabled in v1")
    if (
        _as_mapping(
            models.get("catalyst_partial_pooling"),
            field="models.catalyst_partial_pooling",
        ).get("enabled")
        is not False
    ):
        raise ValueError(
            "Catalyst partial pooling must remain disabled in the v1 primary run"
        )

    inference = _as_mapping(
        contract.get("connected_group_inference"),
        field="connected_group_inference",
    )
    if int(_as_mapping(inference.get("bootstrap"), field="bootstrap").get(
        "repetitions", -1
    )) != 2000:
        raise ValueError("Connected-group bootstrap must remain 2,000")
    p_values = _as_mapping(
        inference.get("one_sided_feature_p_values"),
        field="one_sided_feature_p_values",
    )
    if p_values.get("method") != "connected_group_Rademacher_sign_flip":
        raise ValueError("One-sided raw p-value construction changed")
    if int(p_values.get("repetitions", -1)) != 20_000:
        raise ValueError("Sign-flip repetitions must remain 20,000")

    curves = _as_mapping(
        contract.get("adjusted_probability_curve"),
        field="adjusted_probability_curve",
    )
    _exact_sequence(
        tuple(float(value) for value in curves.get("grid_percentiles", ())),
        tuple(round(0.05 * index, 2) for index in range(1, 20)),
        field="adjusted_probability_curve.grid_percentiles",
    )
    if float(curves.get("reference_percentile", -1)) != 0.50:
        raise ValueError("Adjusted-curve reference percentile changed")
    if float(curves.get("minimum_common_record_coverage_fraction", -1)) != 0.80:
        raise ValueError("Adjusted-curve common-record coverage changed")


def load_frozen_posterior_cardinal_contract(
    config: StudyConfig,
) -> tuple[dict[str, Any], Path, str]:
    """Load and authenticate a supported immutable all-97 contract."""

    settings = config.section(CONTRACT_SECTION)
    configured_id = str(settings.get("contract_id", "")).strip()
    if configured_id not in SUPPORTED_CONTRACT_STATUSES:
        raise ValueError(
            "Configured posterior-cardinal contract ID is unsupported"
        )
    expected_status = SUPPORTED_CONTRACT_STATUSES[configured_id]
    if settings.get("status") != expected_status:
        raise ValueError(
            "posterior_cardinal_classification.status does not match "
            f"{configured_id}: expected {expected_status}"
        )
    expected_hash = str(settings.get("contract_sha256", "")).strip().lower()
    if len(expected_hash) != 64:
        raise ValueError("Configured posterior-cardinal contract hash is invalid")
    contract_path = Path(str(settings.get("contract_path", ""))).expanduser()
    if not contract_path.is_absolute():
        contract_path = (config.project_root / contract_path).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Frozen posterior-cardinal contract is missing: {contract_path}"
        )
    actual_hash = sha256_file(contract_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Frozen posterior-cardinal contract hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    loaded = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    contract = _as_mapping(loaded, field="posterior-cardinal contract")
    _validate_contract_contents(
        contract,
        expected_contract_id=configured_id,
    )
    return contract, contract_path, actual_hash


def snapshot_reduced_feature_registries(
    config: StudyConfig,
    contract: Mapping[str, Any],
    destination: str | Path,
) -> dict[str, Any]:
    """Authenticate and copy the two inherited target-blind registries."""

    registry = _as_mapping(
        contract.get("reduced_feature_registry"),
        field="reduced_feature_registry",
    )
    source_root = Path(str(registry.get("source_run", ""))).expanduser()
    if not source_root.is_absolute():
        source_root = (config.repository_root / source_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Reduced-feature source run is missing: {source_root}"
        )
    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    code_payloads = {
        "engineered_feature_definitions": (
            association_engineered_feature_definitions()
            .to_csv(index=False)
            .encode("utf-8")
        ),
        "frozen_panel_registry": (
            association_panel_registry().to_csv(index=False).encode("utf-8")
        ),
    }
    copied: dict[str, Any] = {}
    for key, code_bytes in code_payloads.items():
        definition = _as_mapping(registry.get(key), field=key)
        relative_path = Path(str(definition.get("path", "")))
        source = (source_root / relative_path).resolve()
        if source_root not in source.parents or not source.is_file():
            raise FileNotFoundError(
                f"Frozen reduced-feature registry is missing: {source}"
            )
        expected_hash = str(definition.get("sha256", "")).strip().lower()
        actual_hash = sha256_file(source)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen reduced-feature registry hash mismatch for {key}: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        code_hash = hashlib.sha256(code_bytes).hexdigest()
        if code_hash != expected_hash:
            raise ValueError(
                f"Current feature-engineering code no longer serializes {key} "
                "byte-for-byte like the frozen registry"
            )
        target = output / relative_path.name
        shutil.copy2(source, target)
        copied_hash = sha256_file(target)
        if copied_hash != expected_hash:
            raise RuntimeError(
                f"Reduced-feature registry changed while copying: {source}"
            )
        copied[key] = {
            "source_path": str(source),
            "snapshot_path": str(target),
            "sha256": copied_hash,
            "size_bytes": target.stat().st_size,
            "code_serialization_sha256": code_hash,
            "code_serialization_matches": True,
        }
    return {
        "status": "pass",
        "source_run": str(source_root),
        "n_files": len(copied),
        "files": copied,
    }


def verify_reduced_feature_registry_snapshot(
    snapshot: str | Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash the copied registry snapshot after parsing or modeling."""

    root = Path(snapshot).expanduser().resolve()
    registry = _as_mapping(
        contract.get("reduced_feature_registry"),
        field="reduced_feature_registry",
    )
    rows: dict[str, Any] = {}
    for key in ("engineered_feature_definitions", "frozen_panel_registry"):
        definition = _as_mapping(registry.get(key), field=key)
        filename = Path(str(definition.get("path", ""))).name
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Reduced-feature registry snapshot is missing: {path}"
            )
        expected = str(definition.get("sha256", "")).strip().lower()
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Reduced-feature registry snapshot changed for {key}: "
                f"expected={expected}, actual={actual}"
            )
        rows[key] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return {"status": "pass", "n_files": len(rows), "files": rows}


__all__ = [
    "EXPECTED_CONTRACT_ID",
    "EXPECTED_FORMAL_PANELS",
    "EXPECTED_GROUPS",
    "EXPECTED_OUTER_SEEDS",
    "EXPECTED_RECORDS",
    "EXPECTED_SENSITIVITY_PANELS",
    "EXPECTED_STAGE",
    "EXPECTED_WIDTHS",
    "SUPPORTED_CONTRACT_STATUSES",
    "load_frozen_posterior_cardinal_contract",
    "snapshot_reduced_feature_registries",
    "verify_reduced_feature_registry_snapshot",
]
