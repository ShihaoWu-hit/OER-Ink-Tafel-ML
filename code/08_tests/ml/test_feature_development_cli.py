from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ink_tafel_study.cli as cli


class _ConfigStub:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def resolve_path(self, name: str) -> Path:
        assert name == "output_root"
        return self.output_root


def _args(stage: str, run_dir: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(stage=stage, run_dir=run_dir)


def test_feature_develop_parser_accepts_parallel_worker_count() -> None:
    parsed = cli._parser().parse_args(
        ["feature-develop", "--run-dir", "new_run", "--n-jobs", "3"]
    )

    assert parsed.stage == "feature-develop"
    assert parsed.run_dir == Path("new_run")
    assert parsed.n_jobs == 3


def test_feature_develop_requires_an_explicit_run_directory(tmp_path: Path) -> None:
    config = _ConfigStub(tmp_path / "outputs")

    with pytest.raises(ValueError, match=r"feature-develop.*requires --run-dir"):
        cli._resolve_run(config, _args("feature-develop"))


@pytest.mark.parametrize("relative", [Path(), Path("child") / "nested"])
def test_feature_develop_refuses_current_97_and_descendants(
    tmp_path: Path, relative: Path
) -> None:
    output_root = tmp_path / "outputs"
    current_97 = output_root / "current_97"
    destination = current_97 / relative
    config = _ConfigStub(output_root)

    with pytest.raises(ValueError, match="registered formal run"):
        cli._resolve_run(config, _args("feature-develop", destination))

    assert not destination.exists()


def test_feature_develop_requires_new_or_empty_run_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    config = _ConfigStub(output_root)
    occupied = output_root / "occupied"
    occupied.mkdir(parents=True)
    (occupied / "old.txt").write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new or empty"):
        cli._resolve_run(config, _args("feature-develop", occupied))

    empty = output_root / "empty"
    empty.mkdir(parents=True)
    resolved = cli._resolve_run(config, _args("feature-develop", empty))

    assert resolved == empty.resolve()
    assert list(resolved.iterdir()) == [resolved / "provenance"]


def test_feature_develop_main_dispatches_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "outputs"
    destination = output_root / "feature_run"
    config = _ConfigStub(output_root)
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)

    def initialize(config_arg, run_dir: Path, *, smoke: bool) -> None:
        assert config_arg is config
        assert smoke is True
        (run_dir / "provenance").mkdir(parents=True, exist_ok=True)

    def run_feature_development(
        config_arg, run_dir: Path, *, smoke: bool, n_jobs: int
    ) -> dict[str, object]:
        observed.update(
            config=config_arg,
            run_dir=run_dir,
            smoke=smoke,
            n_jobs=n_jobs,
        )
        return {"primary_records": 97, "status": "complete_smoke"}

    monkeypatch.setattr(cli, "_initialize_provenance", initialize)
    monkeypatch.setattr(cli, "run_feature_development", run_feature_development)
    monkeypatch.setattr(cli, "artifact_manifest", lambda run_dir: {"files": []})
    monkeypatch.setattr(
        cli,
        "verify_artifact_manifest",
        lambda run_dir, manifest_path: {"status": "pass"},
    )

    assert (
        cli.main(
            [
                "feature-develop",
                "--run-dir",
                str(destination),
                "--smoke",
                "--n-jobs",
                "4",
            ]
        )
        == 0
    )

    assert observed == {
        "config": config,
        "run_dir": destination.resolve(),
        "smoke": True,
        "n_jobs": 4,
    }
    status = json.loads(
        (destination / "provenance" / "stage_feature_develop.json").read_text(
            encoding="utf-8"
        )
    )
    assert status == {
        "stage": "feature_develop",
        "status": "complete_smoke",
        "smoke": True,
        "n_jobs": 4,
        "result": {"primary_records": 97, "status": "complete_smoke"},
    }
    assert (destination / "artifact_manifest.json").is_file()


def test_feature_develop_failure_uses_normalized_stage_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "outputs"
    destination = output_root / "feature_run"
    config = _ConfigStub(output_root)

    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(
        cli,
        "_initialize_provenance",
        lambda config_arg, run_dir, *, smoke: None,
    )
    monkeypatch.setattr(
        cli,
        "run_feature_development",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    with pytest.raises(RuntimeError, match="failed"):
        cli.main(["feature-develop", "--run-dir", str(destination)])

    status = json.loads(
        (destination / "provenance" / "stage_feature_develop.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["stage"] == "feature_develop"
    assert status["status"] == "failed"
