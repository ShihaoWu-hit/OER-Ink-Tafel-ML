from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ink_tafel_study.config import StudyConfig, load_config
from ink_tafel_study.posterior_cardinal_contract import (
    EXPECTED_CONTRACT_ID,
    EXPECTED_FORMAL_PANELS,
    EXPECTED_OUTER_SEEDS,
    EXPECTED_WIDTHS,
    load_frozen_posterior_cardinal_contract,
    snapshot_reduced_feature_registries,
    verify_reduced_feature_registry_snapshot,
)

REVISED_CONFIG = Path(
    "code/Ink_Tafel_Study/config/"
    "study_ir_compensation_corrected_20260724.yaml"
)
LEGACY_V2_CONFIG = Path(
    "code/Ink_Tafel_Study/config/study_posterior_cardinal_v2_20260724.yaml"
)
V4_CONTRACT_SHA256 = (
    "dd93d316610201f4b03e676710b5315075b9728b77f3102da2df4d41c82dd8fe"
)


def _config_with(
    base: StudyConfig,
    *,
    raw: dict,
) -> StudyConfig:
    return StudyConfig(
        path=base.path,
        project_root=base.project_root,
        repository_root=base.repository_root,
        raw=raw,
    )


def test_active_contract_is_hash_bound_and_scientifically_locked() -> None:
    contract, path, digest = load_frozen_posterior_cardinal_contract(
        load_config()
    )

    assert path.name == "posterior_cardinal_all97_v4.yaml"
    assert digest == V4_CONTRACT_SHA256
    assert contract["contract_id"] == "posterior_cardinal_all97_v4"
    assert contract["scientific_scope"]["expected_records"] == 97
    assert contract["scientific_scope"]["expected_connected_groups"] == 80
    assert tuple(
        contract["outer_validation"]["repeat_seeds"]
    ) == EXPECTED_OUTER_SEEDS
    assert tuple(
        contract["label_scheme"]["all_evaluated_relative_half_widths"]
    ) == EXPECTED_WIDTHS
    assert tuple(
        contract["reduced_feature_registry"]["formal_increment_panels"]
    ) == EXPECTED_FORMAL_PANELS
    assert contract["models"]["primary"]["full_bayesian_posterior_fit"] is False
    assert contract["preprocessing"]["rare_min_frequency"] == 3
    assert (
        contract["metrics"]["descriptive_hard_metrics"][
            "may_select_or_release_models"
        ]
        is False
    )


def test_v4_ir_semantic_revision_contract_is_hash_bound() -> None:
    contract, path, digest = load_frozen_posterior_cardinal_contract(
        load_config(REVISED_CONFIG)
    )

    assert path.name == "posterior_cardinal_all97_v4.yaml"
    assert digest == V4_CONTRACT_SHA256
    assert contract["contract_id"] == "posterior_cardinal_all97_v4"
    assert contract["amendment"]["supersedes_contract_id"] == (
        "posterior_cardinal_all97_v3"
    )
    assert contract["scientific_scope"]["expected_records"] == 97
    assert contract["scientific_scope"]["expected_connected_groups"] == 80
    assert tuple(
        contract["outer_validation"]["repeat_seeds"]
    ) == EXPECTED_OUTER_SEEDS
    assert tuple(
        contract["label_scheme"]["all_evaluated_relative_half_widths"]
    ) == EXPECTED_WIDTHS
    assert tuple(
        contract["reduced_feature_registry"]["formal_increment_panels"]
    ) == EXPECTED_FORMAL_PANELS


def test_v2_v4_scientific_design_is_identical() -> None:
    v2, _, _ = load_frozen_posterior_cardinal_contract(
        load_config(LEGACY_V2_CONFIG)
    )
    v4, _, _ = load_frozen_posterior_cardinal_contract(
        load_config(REVISED_CONFIG)
    )
    invariant_sections = (
        "scientific_scope",
        "label_scheme",
        "outer_validation",
        "inner_validation",
        "preprocessing",
        "models",
        "metrics",
        "connected_group_inference",
        "release_gate",
        "adjusted_probability_curve",
        "smoke_mode",
    )
    for section in invariant_sections:
        assert v4[section] == v2[section]
    assert (
        v4["formal_input_snapshot"]["source_file_sha256"]
        != v2["formal_input_snapshot"]["source_file_sha256"]
    )
    assert v2["contract_id"] == EXPECTED_CONTRACT_ID


def test_interrupted_v3_contract_remains_replayable() -> None:
    base = load_config(REVISED_CONFIG)
    raw = deepcopy(base.raw)
    settings = raw["posterior_cardinal_classification"]
    settings["status"] = "registered_fixed_all97_v3"
    settings["contract_id"] = "posterior_cardinal_all97_v3"
    settings["contract_path"] = (
        "config/contracts/posterior_cardinal_all97_v3.yaml"
    )
    settings["contract_sha256"] = (
        "ab7701f3cc8fbae6d61ff889751a6cb7172b78fbb27755ca33921a552e26b312"
    )
    contract, path, digest = load_frozen_posterior_cardinal_contract(
        _config_with(base, raw=raw)
    )
    assert contract["contract_id"] == "posterior_cardinal_all97_v3"
    assert path.name == "posterior_cardinal_all97_v3.yaml"
    assert digest == settings["contract_sha256"]


def test_unsupported_contract_version_fails_closed() -> None:
    base = load_config(REVISED_CONFIG)
    raw = deepcopy(base.raw)
    raw["posterior_cardinal_classification"]["contract_id"] = (
        "posterior_cardinal_all97_v5"
    )
    unsupported = _config_with(base, raw=raw)

    with pytest.raises(ValueError, match="unsupported"):
        load_frozen_posterior_cardinal_contract(unsupported)


def test_contract_status_must_match_version() -> None:
    base = load_config(REVISED_CONFIG)
    raw = deepcopy(base.raw)
    raw["posterior_cardinal_classification"]["status"] = (
        "registered_fixed_all97_v2"
    )
    mismatched = _config_with(base, raw=raw)

    with pytest.raises(ValueError, match="status does not match"):
        load_frozen_posterior_cardinal_contract(mismatched)


def test_contract_byte_drift_fails_before_modeling(tmp_path: Path) -> None:
    base = load_config()
    contract, source, _ = load_frozen_posterior_cardinal_contract(base)
    changed = tmp_path / source.name
    changed.write_text(
        source.read_text(encoding="utf-8").replace(
            "folds: 5", "folds: 4", 1
        ),
        encoding="utf-8",
    )
    raw = deepcopy(base.raw)
    raw["posterior_cardinal_classification"]["contract_path"] = str(changed)
    drifted = _config_with(base, raw=raw)

    with pytest.raises(ValueError, match="contract hash mismatch"):
        load_frozen_posterior_cardinal_contract(drifted)

    assert contract["outer_validation"]["folds"] == 5


def test_reduced_feature_registries_match_code_and_rehash(
    tmp_path: Path,
) -> None:
    config = load_config()
    contract, _, _ = load_frozen_posterior_cardinal_contract(config)
    binding = snapshot_reduced_feature_registries(
        config,
        contract,
        tmp_path / "registries",
    )
    verification = verify_reduced_feature_registry_snapshot(
        tmp_path / "registries",
        contract,
    )

    assert binding["status"] == "pass"
    assert binding["n_files"] == 2
    assert verification["status"] == "pass"
    assert all(
        row["code_serialization_matches"]
        for row in binding["files"].values()
    )


def test_reduced_registry_drift_fails_closed(tmp_path: Path) -> None:
    config = load_config()
    contract, _, _ = load_frozen_posterior_cardinal_contract(config)
    root = tmp_path / "registries"
    snapshot_reduced_feature_registries(config, contract, root)
    panel_registry = root / "frozen_panel_registry.csv"
    panel_registry.write_text(
        panel_registry.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="snapshot changed"):
        verify_reduced_feature_registry_snapshot(root, contract)
