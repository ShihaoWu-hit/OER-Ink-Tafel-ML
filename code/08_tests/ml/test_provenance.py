import json
from pathlib import Path

import pytest

from ink_tafel_study.cli import _initialize_provenance, _input_manifest
from ink_tafel_study.config import StudyConfig
from ink_tafel_study.provenance import (
    artifact_manifest,
    directory_tree_summary,
    verify_artifact_manifest,
    write_json,
)


def test_input_manifest_commits_to_recursive_directory_contents(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "source.txt").write_text("v1\n", encoding="utf-8")
    config = StudyConfig(
        path=tmp_path / "study.yaml",
        project_root=tmp_path,
        repository_root=tmp_path,
        raw={"paths": {"source_directory": "input", "output_root": "outputs"}},
    )

    first = _input_manifest(config)["source_directory"]
    assert first["kind"] == "directory"
    assert first["file_count"] == 1
    assert len(first["tree_sha256"]) == 64

    (input_dir / "source.txt").write_text("v2\n", encoding="utf-8")
    second = _input_manifest(config)["source_directory"]
    assert second["tree_sha256"] != first["tree_sha256"]


def test_input_hash_registry_passes_and_fails_closed_on_change(tmp_path: Path) -> None:
    input_file = tmp_path / "source.csv"
    input_file.write_text("x\n1\n", encoding="utf-8")
    input_hash = directory_tree_summary(tmp_path)["tree_sha256"]
    # The registered artifact is a file, so freeze its content digest rather
    # than the directory digest used above to exercise both helper paths.
    from ink_tafel_study.provenance import sha256_file

    config = StudyConfig(
        path=tmp_path / "study.yaml",
        project_root=tmp_path,
        repository_root=tmp_path,
        raw={
            "paths": {"source": "source.csv", "output_root": "outputs"},
            "input_hashes": {
                "enforce": True,
                "tree_hash_algorithm": "sha256-relative-path-size-content-v1",
                "expected": {"source": sha256_file(input_file)},
            },
        },
    )
    manifest = _input_manifest(config)
    assert manifest["source"]["hash_matches_expected"] is True
    assert manifest["_hash_validation"]["status"] == "pass"
    assert input_hash

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _initialize_provenance(config, run_dir, smoke=True)

    input_file.write_text("x\n2\n", encoding="utf-8")
    failed = _input_manifest(config)
    assert failed["source"]["hash_matches_expected"] is False
    assert failed["_hash_validation"]["hash_mismatches"] == ["source"]
    with pytest.raises(ValueError, match="Configured input hash validation failed"):
        _initialize_provenance(config, run_dir, smoke=True)


def test_active_source_run_is_not_a_static_input_hash_registration(
    tmp_path: Path,
) -> None:
    from ink_tafel_study.provenance import sha256_file

    input_file = tmp_path / "source.csv"
    input_file.write_text("x\n1\n", encoding="utf-8")
    active_source = tmp_path / "active_source"
    active_source.mkdir()
    (active_source / "mutable_run_artifact.txt").write_text(
        "runner verifies this separately\n", encoding="utf-8"
    )
    config = StudyConfig(
        path=tmp_path / "study.yaml",
        project_root=tmp_path,
        repository_root=tmp_path,
        raw={
            "paths": {
                "source": "source.csv",
                "output_root": "outputs",
                "active_source_run": "active_source",
            },
            "input_hashes": {
                "enforce": True,
                "tree_hash_algorithm": (
                    "sha256-relative-path-size-content-v1"
                ),
                "expected": {"source": sha256_file(input_file)},
            },
        },
    )

    manifest = _input_manifest(config)

    assert "active_source_run" not in manifest
    assert manifest["_hash_validation"]["status"] == "pass"
    assert manifest["_hash_validation"]["missing_registrations"] == []


def test_posthoc_full_run_mode_forbids_scientific_conclusion_claims(
    tmp_path: Path,
) -> None:
    config = StudyConfig(
        path=tmp_path / "study.yaml",
        project_root=tmp_path,
        repository_root=tmp_path,
        raw={"paths": {"output_root": "outputs"}},
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _initialize_provenance(
        config,
        run_dir,
        smoke=False,
        scientific_conclusions_allowed=False,
        analysis_role="posthoc_exploratory_sensitivity",
    )

    mode = json.loads(
        (run_dir / "provenance" / "run_mode.json").read_text(
            encoding="utf-8"
        )
    )
    assert mode["smoke"] is False
    assert mode["scientific_conclusions_allowed"] is False
    assert mode["analysis_role"] == "posthoc_exploratory_sensitivity"


def test_directory_tree_summary_is_recursive_location_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "nested").mkdir(parents=True)
        (root / "a.txt").write_text("alpha\n", encoding="utf-8")
        (root / "nested" / "b.bin").write_bytes(b"\x00\x01")

    first = directory_tree_summary(left)
    assert first == directory_tree_summary(left)
    assert first == directory_tree_summary(right)
    assert first["file_count"] == 2
    assert first["size_bytes"] == 8

    (right / "nested" / "b.bin").write_bytes(b"\x00\x02")
    assert directory_tree_summary(right)["tree_sha256"] != first["tree_sha256"]
    (right / "nested" / "renamed.bin").write_bytes(b"\x00\x01")
    (right / "nested" / "b.bin").unlink()
    assert directory_tree_summary(right)["tree_sha256"] != first["tree_sha256"]


def test_artifact_manifest_recomputes_after_self_write_and_hashes_nested_names(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "result.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "data" / "artifact_manifest.json").write_text(
        '{"nested": true}\n', encoding="utf-8"
    )

    recorded = artifact_manifest(tmp_path)
    write_json(tmp_path / "artifact_manifest.json", recorded)
    recomputed = artifact_manifest(tmp_path)

    assert recomputed["files"] == recorded["files"]
    assert recomputed["tree_sha256"] == recorded["tree_sha256"]
    assert "artifact_manifest.json" not in {item["path"] for item in recorded["files"]}
    assert "data/artifact_manifest.json" in {item["path"] for item in recorded["files"]}
    assert verify_artifact_manifest(tmp_path)["status"] == "pass"


def test_artifact_manifest_verifier_detects_tampering_and_stale_inventory(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "result.txt"
    artifact.write_text("registered\n", encoding="utf-8")
    write_json(tmp_path / "artifact_manifest.json", artifact_manifest(tmp_path))

    artifact.write_text("modified\n", encoding="utf-8")
    modified = verify_artifact_manifest(tmp_path)
    assert modified["status"] == "fail"
    assert modified["modified_files"] == ["result.txt"]

    artifact.write_text("registered\n", encoding="utf-8")
    (tmp_path / "late.txt").write_text("unexpected\n", encoding="utf-8")
    unexpected = verify_artifact_manifest(tmp_path)
    assert unexpected["status"] == "fail"
    assert unexpected["unexpected_files"] == ["late.txt"]

    recorded = json.loads((tmp_path / "artifact_manifest.json").read_text(encoding="utf-8"))
    recorded["tree_sha256"] = "0" * 64
    internally_corrupt = verify_artifact_manifest(tmp_path, recorded)
    assert internally_corrupt["status"] == "fail"
    assert internally_corrupt["recorded_tree_self_consistent"] is False

    recorded = artifact_manifest(tmp_path)
    recorded["file_count"] += 1
    metadata_corrupt = verify_artifact_manifest(tmp_path, recorded)
    assert metadata_corrupt["status"] == "fail"
    assert metadata_corrupt["recorded_file_count_matches"] is False
