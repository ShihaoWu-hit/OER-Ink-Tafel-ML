from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_solvent_property_review.mjs"
NODE = shutil.which("node")


def _node(*arguments: str) -> subprocess.CompletedProcess[str]:
    if NODE is None:
        raise RuntimeError("managed Node runtime is unavailable")
    return subprocess.run(
        [NODE, str(BUILDER), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def test_builder_contract_is_stable_and_uses_exact_six_sheet_interface() -> None:
    if NODE is None:
        source = BUILDER.read_text(encoding="utf-8")
        ordered_tokens = [
            'name: "Record Properties"',
            'name: "Canonical Compositions"',
            'name: "Measurements"',
            'name: "Sources"',
            'name: "Legacy Comparison"',
            'name: "QA Summary"',
        ]
        positions = [source.index(token) for token in ordered_tokens]
        assert positions == sorted(positions)
        for input_name in (
            "record_solvent_properties.csv",
            "canonical_solvent_properties.csv",
            "authoritative_property_measurements_snapshot.csv",
            "property_sources_review.csv",
            "legacy_property_comparison.csv",
            "property_proxy_qa.json",
            "solvent_property_review.xlsx",
        ):
            assert input_name in source
        return
    result = _node("--print-contract")
    assert result.returncode == 0, result.stderr
    contract = json.loads(result.stdout)
    assert [item["name"] for item in contract["sheets"]] + [contract["qa"]["name"]] == [
        "Record Properties",
        "Canonical Compositions",
        "Measurements",
        "Sources",
        "Legacy Comparison",
        "QA Summary",
    ]
    assert [item["input"] for item in contract["sheets"]] == [
        "record_solvent_properties.csv",
        "canonical_solvent_properties.csv",
        "authoritative_property_measurements_snapshot.csv",
        "property_sources_review.csv",
        "legacy_property_comparison.csv",
    ]
    assert contract["qa"]["input"] == "property_proxy_qa.json"
    assert contract["output"] == "solvent_property_review.xlsx"


@pytest.mark.skipif(NODE is None, reason="managed Node executable is unavailable")
def test_builder_validates_inputs_without_spreadsheet_runtime(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "record_solvent_properties.csv",
        ["record_id", "solvent_composition_key"],
        [[index, "water=1.000000"] for index in range(1, 98)],
    )
    _write_csv(
        tmp_path / "canonical_solvent_properties.csv",
        ["solvent_composition_key", "property_name"],
        [[f"key-{index}", "density"] for index in range(132)],
    )
    _write_csv(
        tmp_path / "authoritative_property_measurements_snapshot.csv",
        ["property_name", "source_id"],
        [["density", f"source-{(index % 35) + 1}"] for index in range(199)],
    )
    _write_csv(
        tmp_path / "property_sources_review.csv",
        [
            "source_id",
            "source_url",
            "source_file_id",
            "resource_url",
            "sha256",
            "hash_status",
        ],
        [
            [
                f"source-{index}",
                f"https://example.test/source-{index}",
                f"file-{index}",
                f"https://example.test/file-{index}.csv",
                "a" * 64,
                "locally_hash_frozen",
            ]
            for index in range(1, 36)
        ],
    )
    _write_csv(
        tmp_path / "legacy_property_comparison.csv",
        ["property_name", "legacy_value"],
        [["density", 997.0] for _ in range(388)],
    )
    (tmp_path / "property_proxy_qa.json").write_text(
        json.dumps({"status": "review"}), encoding="utf-8"
    )

    result = _node("--validate-inputs", str(tmp_path))
    assert result.returncode == 0, result.stderr
    facts = json.loads(result.stdout)
    assert facts["ok"] is True
    assert [item["rowCount"] for item in facts["inputs"][:2]] == [97, 132]


@pytest.mark.skipif(NODE is None, reason="managed Node executable is unavailable")
def test_builder_fails_closed_when_artifact_runtime_is_unavailable(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "record_solvent_properties.csv",
        ["record_id", "solvent_composition_key"],
        [[index, "water=1.000000"] for index in range(1, 98)],
    )
    _write_csv(
        tmp_path / "canonical_solvent_properties.csv",
        ["solvent_composition_key", "property_name"],
        [[f"key-{index}", "density"] for index in range(132)],
    )
    _write_csv(
        tmp_path / "authoritative_property_measurements_snapshot.csv",
        ["property_name", "source_id"],
        [["density", f"source-{(index % 35) + 1}"] for index in range(199)],
    )
    _write_csv(
        tmp_path / "property_sources_review.csv",
        [
            "source_id",
            "source_url",
            "source_file_id",
            "resource_url",
            "sha256",
            "hash_status",
        ],
        [
            [
                f"source-{index}",
                f"https://example.test/source-{index}",
                f"file-{index}",
                f"https://example.test/file-{index}.csv",
                "a" * 64,
                "locally_hash_frozen",
            ]
            for index in range(1, 36)
        ],
    )
    _write_csv(
        tmp_path / "legacy_property_comparison.csv",
        ["property_name"],
        [["density"] for _ in range(388)],
    )
    (tmp_path / "property_proxy_qa.json").write_text("{}", encoding="utf-8")

    result = _node("--data-dir", str(tmp_path))
    if result.returncode == 0:
        assert (tmp_path / "solvent_property_review.xlsx").is_file()
    else:
        assert "@oai/artifact-tool is unavailable" in result.stderr
        assert "openpyxl/xlsxwriter" in result.stderr


def test_builder_does_not_reference_alternate_workbook_libraries() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    forbidden_imports = (
        'from "openpyxl"',
        "from 'openpyxl'",
        'from "xlsxwriter"',
        "from 'xlsxwriter'",
        'from "exceljs"',
        "from 'exceljs'",
    )
    assert not any(token in source for token in forbidden_imports)
    assert 'import("@oai/artifact-tool")' in source
    assert "Workbook formula error scan failed" in source
    assert "property_sources_review.csv" in source
    assert "autofitRows()" in source
