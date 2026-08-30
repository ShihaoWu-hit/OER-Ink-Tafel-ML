from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TREE_HASH_ALGORITHM = "sha256-relative-path-size-content-v1"
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def environment_manifest(repository_root: Path) -> dict[str, Any]:
    package_names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "openpyxl",
        "pyarrow",
        "PyYAML",
        "optuna",
        "pymc",
        "arviz",
        "catboost",
    ]
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": _git(repository_root, "rev-parse", "HEAD"),
        "git_branch": _git(repository_root, "branch", "--show-current"),
        "git_status_porcelain": _git(repository_root, "status", "--porcelain=v1"),
    }


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def inventory_files(paths: Iterable[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    root = relative_to.resolve()
    for input_path in paths:
        candidates = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
        for path in candidates:
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            rows.append(
                {
                    # Keep the lexical path in the tree.  Content reads may
                    # follow a file symlink, but a link to another location
                    # must not replace its registered relative name.
                    "path": Path(os.path.abspath(path)).relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sorted(rows, key=lambda item: item["path"])


def tree_sha256(files: Iterable[dict[str, Any]]) -> str:
    """Hash a file inventory independently of absolute location and timestamps.

    Each entry commits to the POSIX relative path, byte size, and file-content
    SHA256.  Canonical JSON lines make the digest deterministic while avoiding
    platform-specific path separators or dictionary ordering.
    """

    digest = hashlib.sha256()
    digest.update((TREE_HASH_ALGORITHM + "\n").encode("utf-8"))
    for item in sorted(files, key=lambda row: str(row["path"])):
        canonical = json.dumps(
            [str(item["path"]), int(item["size_bytes"]), str(item["sha256"])],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def directory_tree_summary(path: str | Path) -> dict[str, Any]:
    """Return a deterministic recursive content summary for a directory."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = inventory_files([root], relative_to=root)
    return {
        "kind": "directory",
        "tree_hash_algorithm": TREE_HASH_ALGORITHM,
        "tree_sha256": tree_sha256(files),
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
    }


def artifact_manifest(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    # Only the run-root manifest is self-excluded.  A nested file with the same
    # basename is an ordinary artifact and must remain covered by the run hash.
    manifest_path = root / ARTIFACT_MANIFEST_NAME
    paths = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {
        "manifest_schema_version": 1,
        "run_dir": str(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tree_hash_algorithm": TREE_HASH_ALGORITHM,
        "tree_sha256": tree_sha256(files),
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "self_excluded_path": ARTIFACT_MANIFEST_NAME,
        "files": files,
    }


def verify_artifact_manifest(
    run_dir: str | Path,
    manifest: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Recompute and compare every artifact covered by a saved manifest.

    Verification is deliberately side-effect free.  The root manifest is
    excluded from both recorded and recomputed inventories, so merely writing
    the manifest does not invalidate it.  Unexpected files fail verification;
    this prevents a stale manifest from certifying a subsequently changed run.
    """

    root = Path(run_dir).resolve()
    if manifest is None:
        manifest_path = root / ARTIFACT_MANIFEST_NAME
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif isinstance(manifest, dict):
        manifest_path = None
        recorded = manifest
    else:
        manifest_path = Path(manifest).resolve()
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))

    raw_recorded_rows = recorded.get("files", [])
    rows_well_formed = isinstance(raw_recorded_rows, list) and all(
        isinstance(item, dict) and {"path", "size_bytes", "sha256"}.issubset(item)
        for item in raw_recorded_rows
    )
    recorded_rows = raw_recorded_rows if rows_well_formed else []
    recorded_paths_in_order = [str(item.get("path")) for item in recorded_rows]
    path_counts = Counter(recorded_paths_in_order)
    duplicate_recorded_paths = sorted(path for path, count in path_counts.items() if count > 1)
    recorded_files = {str(item["path"]): item for item in recorded_rows if "path" in item}
    actual = artifact_manifest(root)
    actual_files = {str(item["path"]): item for item in actual["files"]}
    recorded_paths = set(recorded_files)
    actual_paths = set(actual_files)
    missing = sorted(recorded_paths - actual_paths)
    unexpected = sorted(actual_paths - recorded_paths)
    modified = sorted(
        path
        for path in recorded_paths & actual_paths
        if (
            str(recorded_files[path].get("sha256"))
            != str(actual_files[path].get("sha256"))
            or int(recorded_files[path].get("size_bytes", -1))
            != int(actual_files[path].get("size_bytes", -1))
        )
    )
    expected_tree = recorded.get("tree_sha256")
    recomputed_recorded_tree = tree_sha256(recorded_files.values())
    recorded_tree_self_consistent = expected_tree == recomputed_recorded_tree
    actual_tree_matches = expected_tree == actual["tree_sha256"]
    algorithm_matches = recorded.get("tree_hash_algorithm") == TREE_HASH_ALGORITHM
    schema_matches = recorded.get("manifest_schema_version") == 1
    recorded_file_count_matches = recorded.get("file_count") == len(recorded_rows)
    recorded_size_matches = recorded.get("size_bytes") == sum(
        int(item.get("size_bytes", -1)) for item in recorded_rows
    )
    passed = (
        not missing
        and not unexpected
        and not modified
        and not duplicate_recorded_paths
        and recorded_tree_self_consistent
        and actual_tree_matches
        and algorithm_matches
        and schema_matches
        and rows_well_formed
        and recorded_file_count_matches
        and recorded_size_matches
    )
    return {
        "status": "pass" if passed else "fail",
        "run_dir": str(root),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "tree_hash_algorithm": TREE_HASH_ALGORITHM,
        "recorded_tree_sha256": expected_tree,
        "actual_tree_sha256": actual["tree_sha256"],
        "recorded_tree_self_consistent": recorded_tree_self_consistent,
        "actual_tree_matches": actual_tree_matches,
        "algorithm_matches": algorithm_matches,
        "schema_matches": schema_matches,
        "rows_well_formed": rows_well_formed,
        "recorded_file_count_matches": recorded_file_count_matches,
        "recorded_size_matches": recorded_size_matches,
        "recorded_file_count": len(recorded_files),
        "actual_file_count": len(actual_files),
        "duplicate_recorded_paths": duplicate_recorded_paths,
        "missing_files": missing,
        "unexpected_files": unexpected,
        "modified_files": modified,
    }


def create_run_dir(output_root: Path, run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("data", "targets", "evaluation", "figures", "report", "provenance"):
        (run_dir / name).mkdir()
    return run_dir


def update_latest_pointer(output_root: Path, run_dir: Path) -> None:
    pointer = output_root / "LATEST"
    pointer.write_text(str(run_dir.resolve()) + os.linesep, encoding="utf-8")


def resolve_latest_run(output_root: Path) -> Path:
    pointer = output_root / "LATEST"
    if not pointer.is_file():
        raise FileNotFoundError(
            f"No latest run pointer at {pointer}; run the curate stage first or pass --run-dir"
        )
    run_dir = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"LATEST points to a missing run: {run_dir}")
    return run_dir
