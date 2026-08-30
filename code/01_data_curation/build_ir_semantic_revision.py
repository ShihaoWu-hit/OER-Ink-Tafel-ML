#!/usr/bin/env python3
"""Build a non-destructive canonical-97 iR semantic revision.

The source workbook and source canonical snapshot remain unchanged.  This
revision implements the user-confirmed data-dictionary rule that Sheet2
values beginning with ``Unreported`` mean that the curve was not iR compensated.
The user subsequently confirmed that the one
The ``Unverifiable (full text inaccessible)`` record (Record 28) was also not compensated.
Its raw wording and reporting-evidence state remain preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


REVISION_ID = "DATA0718-IR-SEM-001"
REVISION_DATE = "2026-07-24"
EXPECTED_COUNTS = {
    "corrected": 38,
    "not_corrected": 59,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_source_manifest(source_dir: Path) -> None:
    manifest_path = source_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for row in manifest["files"]:
        path = source_dir / row["path"]
        if not path.is_file():
            failures.append(f"missing:{path.name}")
        elif sha256_file(path) != row["sha256"]:
            failures.append(f"hash_mismatch:{path.name}")
    if failures:
        raise ValueError(f"Source canonical manifest failed: {failures}")


def _derive_compensation_state(master: pd.DataFrame) -> pd.Series:
    raw = master["ir_status_raw"].fillna("").astype(str).str.strip()
    state = pd.Series(index=master.index, dtype="object")
    state.loc[raw.str.startswith("Yes")] = "corrected"
    state.loc[raw.str.startswith("No")] = "not_corrected"
    state.loc[raw.str.startswith("Unreported")] = "not_corrected"
    state.loc[raw.str.contains("Unverifiable", regex=False)] = "not_corrected"
    if state.isna().any():
        unknown = master.loc[state.isna(), ["record_id", "ir_status_raw"]]
        raise ValueError(
            "Unrecognized iR raw states: "
            + unknown.to_dict(orient="records").__repr__()
        )
    counts = {str(key): int(value) for key, value in state.value_counts().items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Corrected iR counts differ from the registered revision: {counts}"
        )
    return state


def _assignment_basis(master: pd.DataFrame) -> pd.Series:
    evidence = master["ir_reporting_evidence_state"].astype(str)
    legacy = master["ir_correction_status_legacy"].astype(str)
    basis = pd.Series(index=master.index, dtype="object")
    basis.loc[legacy.eq("reported_corrected")] = "source_reported_corrected"
    basis.loc[legacy.eq("reported_not_corrected")] = (
        "source_reported_not_corrected"
    )
    basis.loc[evidence.eq("unreported")] = (
        "user_confirmed_unreported_means_not_corrected"
    )
    basis.loc[evidence.eq("unverifiable")] = (
        "user_confirmed_unverifiable_record_means_not_corrected"
    )
    if basis.isna().any():
        raise ValueError("Every record must have an iR assignment basis")
    return basis


def _write_artifact_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(output_dir / "artifact_manifest.json", {"files": rows})


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _validate_source_manifest(source_dir)
    output_dir.mkdir(parents=True)

    replaced = {
        "artifact_manifest.json",
        "README.md",
        "corrections.csv",
        "data_decisions.csv",
        "master.csv",
        "master.parquet",
        "ml_ready.csv",
        "qa.json",
        "target_registry.csv",
    }
    for source in sorted(source_dir.iterdir()):
        if source.is_file() and source.name not in replaced:
            shutil.copy2(source, output_dir / source.name)

    master = pd.read_csv(source_dir / "master.csv", low_memory=False)
    if len(master) != 97 or master["record_id"].nunique() != 97:
        raise ValueError("The source master must contain 97 unique records")
    master["ir_correction_status_legacy"] = master["ir_correction_status"]
    master["ir_correction_status"] = _derive_compensation_state(master)
    master["ir_compensation_assignment_basis"] = _assignment_basis(master)
    master["ir_compensation_state_user_confirmed"] = master[
        "ir_reporting_evidence_state"
    ].isin(["unreported", "unverifiable"])
    master["ir_compensation_state_revision_id"] = REVISION_ID
    master["ir_compensation_corrected_binary"] = master[
        "ir_correction_status"
    ].map({"corrected": 1.0, "not_corrected": 0.0})
    master.to_csv(output_dir / "master.csv", index=False)
    master.to_parquet(output_dir / "master.parquet", index=False)

    state_by_record = master.set_index("record_id")["ir_correction_status"]
    legacy_by_record = master.set_index("record_id")[
        "ir_correction_status_legacy"
    ]

    ml_ready = pd.read_csv(source_dir / "ml_ready.csv", low_memory=False)
    ml_ready["ir_correction_status_legacy"] = ml_ready[
        "ir_correction_status"
    ]
    ml_ready["ir_correction_status"] = ml_ready["record_id"].map(
        state_by_record
    )
    ml_ready.to_csv(output_dir / "ml_ready.csv", index=False)

    target_registry = pd.read_csv(
        source_dir / "target_registry.csv", low_memory=False
    )
    target_registry["ir_status_legacy"] = target_registry["ir_status"]
    target_registry["ir_status"] = target_registry["record_id"].map(
        state_by_record
    )
    target_registry.to_csv(output_dir / "target_registry.csv", index=False)

    corrections = pd.read_csv(source_dir / "corrections.csv", low_memory=False)
    affected = master.loc[
        master["ir_reporting_evidence_state"].isin(
            ["unreported", "unverifiable"]
        )
    ].copy()
    semantic_corrections = pd.DataFrame(
        {
            "correction_id": affected["record_id"].map(
                lambda value: f"{REVISION_ID}-R{int(value):03d}"
            ),
            "record_id": affected["record_id"].astype(int),
            "source_excel_row": affected["record_id"].astype(int) + 1,
            "field_name": "ir_correction_status",
            "source_value": affected["ir_status_raw"],
            "corrected_value": "not_corrected",
            "unit": "",
            "status": "approved",
            "decision_basis": affected["ir_compensation_assignment_basis"],
            "confirmed_by": "user",
            "confirmed_date": REVISION_DATE,
            "source_reference": affected["doi"],
            "notes": (
                "Preserve ir_status_raw and ir_reporting_evidence_state; "
                "use corrected compensation state for ML."
            ),
        }
    )
    corrections = pd.concat(
        [corrections, semantic_corrections[corrections.columns]],
        ignore_index=True,
    )
    corrections.to_csv(output_dir / "corrections.csv", index=False)

    decisions = pd.read_csv(source_dir / "data_decisions.csv")
    decision = pd.DataFrame(
        [
            {
                "data_domain": "iR compensation state",
                "controlling_source": (
                    "formulation_source.xlsx Sheet2 raw status plus user-confirmed "
                    f"data dictionary ({REVISION_ID})"
                ),
                "model_use": "canonical_context_with_separate_evidence_state",
                "decision_reason": (
                    "Raw Unreported* means not compensated; Record 28 remains "
                    "unverifiable because the full text is inaccessible."
                ),
            }
        ]
    )
    decisions = pd.concat([decisions, decision], ignore_index=True)
    decisions.to_csv(output_dir / "data_decisions.csv", index=False)

    qa = json.loads((source_dir / "qa.json").read_text(encoding="utf-8"))
    qa["ir_status_counts"] = EXPECTED_COUNTS
    qa["ir_semantic_revision"] = {
        "revision_id": REVISION_ID,
        "confirmed_date": REVISION_DATE,
        "affected_record_ids": affected["record_id"].astype(int).tolist(),
        "affected_count": int(len(affected)),
        "unverifiable_evidence_record_ids": master.loc[
            master["ir_reporting_evidence_state"].eq("unverifiable"),
            "record_id",
        ]
        .astype(int)
        .tolist(),
        "raw_status_preserved": True,
        "reporting_evidence_state_preserved": True,
        "target_values_changed": False,
        "posterior_draws_changed": False,
    }
    write_json(output_dir / "qa.json", qa)

    readme = (source_dir / "README.md").read_text(encoding="utf-8")
    revision_note = f"""# Canonical 97-record dataset — iR semantic revision

Revision `{REVISION_ID}` ({REVISION_DATE}) applies the user-confirmed data
dictionary rule that the 40 Sheet2 values beginning with `Unreported`, plus
Record 28 whose raw status is `Unverifiable (full text inaccessible)`, mean
`not_corrected`.  Raw wording and reporting-evidence fields are preserved
separately.  Tafel targets, posterior draws, curve identities, PPC decisions,
and connected groups are unchanged.

"""
    (output_dir / "README.md").write_text(
        revision_note + readme, encoding="utf-8"
    )

    source_target_columns = [
        column
        for column in master.columns
        if column.startswith("y_tafel_") or column.startswith("posterior_")
    ]
    original_master = pd.read_csv(source_dir / "master.csv", low_memory=False)
    if not original_master[source_target_columns].equals(
        master[source_target_columns]
    ):
        raise RuntimeError("The iR semantic revision changed target columns")
    if sha256_file(source_dir / "posterior_slope_draws.csv.gz") != sha256_file(
        output_dir / "posterior_slope_draws.csv.gz"
    ):
        raise RuntimeError("Posterior draw bytes changed during the iR revision")

    revision = {
        "revision_id": REVISION_ID,
        "date": REVISION_DATE,
        "source_canonical_dir": str(source_dir),
        "source_artifact_manifest_sha256": sha256_file(
            source_dir / "artifact_manifest.json"
        ),
        "status_counts": EXPECTED_COUNTS,
        "affected_record_ids": affected["record_id"].astype(int).tolist(),
        "unverifiable_evidence_record_ids": [28],
        "model_predictor_changed": True,
        "response_or_posterior_changed": False,
    }
    write_json(output_dir / "ir_semantic_revision.json", revision)
    _write_artifact_manifest(output_dir)
    _validate_source_manifest(output_dir)
    return revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
