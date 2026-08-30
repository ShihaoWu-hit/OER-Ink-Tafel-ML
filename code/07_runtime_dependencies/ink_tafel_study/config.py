from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class StudyConfig:
    """Loaded study configuration with reproducible path resolution."""

    path: Path
    project_root: Path
    repository_root: Path
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Configuration section {name!r} must be a mapping")
        return value

    def value(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def resolve_path(self, key: str, *, must_exist: bool = False) -> Path:
        paths = self.section("paths")
        if key not in paths:
            raise KeyError(f"Unknown configured path: {key}")
        path = Path(str(paths[key])).expanduser()
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Configured path {key!r} does not exist: {path}")
        return path


def load_config(path: str | Path | None = None) -> StudyConfig:
    project_root = Path(__file__).resolve().parents[2]
    config_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else project_root / "config" / "study.yaml"
    )
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Study configuration must be a mapping: {config_path}")
    required = {
        "study",
        "paths",
        "targets",
        "validation",
        "models",
        "feature_panels",
        "interpretation",
    }
    missing = sorted(required - set(loaded))
    if missing:
        raise ValueError(f"Study configuration is missing sections: {missing}")
    return StudyConfig(
        path=config_path,
        project_root=project_root,
        repository_root=project_root.parent,
        raw=loaded,
    )
