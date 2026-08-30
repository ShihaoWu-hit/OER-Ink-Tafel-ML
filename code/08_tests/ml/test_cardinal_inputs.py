from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil

import pytest

import ink_tafel_study.cardinal_inputs as cardinal_inputs
from ink_tafel_study.cardinal_inputs import snapshot_cardinal_inputs
from ink_tafel_study.config import load_config
from ink_tafel_study.provenance import sha256_file


def test_semantic_draw_binding_rejects_one_value_tamper(tmp_path: Path) -> None:
    config = load_config()
    original = config.resolve_path("active_source_run")
    source = tmp_path / "formal_source"
    shutil.copytree(original, source)

    draws_path = source / "targets" / "posterior_slope_draws.csv.gz"
    rewritten = tmp_path / "tampered_draws.csv.gz"
    with gzip.open(draws_path, "rb") as input_handle, gzip.open(
        rewritten, "wb"
    ) as output_handle:
        header = input_handle.readline()
        first_row = input_handle.readline()
        columns = header.decode("utf-8").rstrip("\r\n").split(",")
        values = first_row.decode("utf-8").rstrip("\r\n").split(",")
        slope_position = columns.index("primary_slope_mV_dec")
        values[slope_position] = str(float(values[slope_position]) + 1000.0)
        output_handle.write(header)
        output_handle.write((",".join(values) + "\n").encode("utf-8"))
        shutil.copyfileobj(input_handle, output_handle)
    shutil.move(rewritten, draws_path)

    expected_hashes = dict(
        config.section("cardinal_classification")["source_snapshot_sha256"]
    )
    expected_hashes["posterior_slope_draws.csv.gz"] = sha256_file(draws_path)
    output = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="semantically identical"):
        snapshot_cardinal_inputs(
            source,
            output,
            canonical_dataset=config.resolve_path("canonical_dataset"),
            expected_source_hashes=expected_hashes,
            require_formal_approval=True,
        )
    assert not output.exists()


def test_snapshot_is_rehashed_after_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    source = config.resolve_path("active_source_run")
    output = tmp_path / "snapshot"
    original_read_csv = cardinal_inputs.pd.read_csv
    tampered = False

    def read_then_tamper(path, *args, **kwargs):
        nonlocal tampered
        result = original_read_csv(path, *args, **kwargs)
        resolved = Path(path)
        if not tampered and resolved == output / "records.csv":
            with resolved.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            tampered = True
        return result

    monkeypatch.setattr(cardinal_inputs.pd, "read_csv", read_then_tamper)
    with pytest.raises(ValueError, match="changed after authentication"):
        snapshot_cardinal_inputs(
            source,
            output,
            canonical_dataset=config.resolve_path("canonical_dataset"),
            expected_source_hashes=config.section("cardinal_classification")[
                "source_snapshot_sha256"
            ],
            require_formal_approval=True,
        )
    assert tampered is True


def test_formal_readiness_requires_json_boolean_true(tmp_path: Path) -> None:
    config = load_config()
    original = config.resolve_path("active_source_run")
    source = tmp_path / "formal_source"
    shutil.copytree(original, source)
    target_qa_path = source / "targets" / "target_qa.json"
    target_qa = json.loads(target_qa_path.read_text(encoding="utf-8"))
    target_qa["formal_evaluation_ready"] = "false"
    target_qa_path.write_text(
        json.dumps(target_qa, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expected_hashes = dict(
        config.section("cardinal_classification")["source_snapshot_sha256"]
    )
    expected_hashes["target_qa.json"] = sha256_file(target_qa_path)

    with pytest.raises(ValueError, match="not formally evaluation-ready"):
        snapshot_cardinal_inputs(
            source,
            tmp_path / "snapshot",
            canonical_dataset=config.resolve_path("canonical_dataset"),
            expected_source_hashes=expected_hashes,
            require_formal_approval=True,
        )
