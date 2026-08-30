"""Hash-bound input snapshot for cardinal-region classification runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .provenance import directory_tree_summary, sha256_file


TARGET_NAME = "b_bayesian_local_line_mV_dec"
EXPECTED_RECORDS = 97
EXPECTED_ALKALINE = 90
EXPECTED_DRAWS_PER_RECORD = 8_000

_INPUT_FILES = {
    "records.csv": ("data", "records.csv"),
    "registered_panel_schema.csv": ("data", "registered_panel_schema.csv"),
    "property_proxy_qa.json": ("data", "property_proxy_qa.json"),
    "integration_input_manifest.json": ("data", "integration_input_manifest.json"),
    "target_registry.csv": ("targets", "target_registry.csv"),
    "posterior_slope_draws.csv.gz": ("targets", "posterior_slope_draws.csv.gz"),
    "target_qa.json": ("targets", "target_qa.json"),
    "bayesian_audit.csv": ("targets", "bayesian_audit.csv"),
    "ppc_binding_audit.csv": ("targets", "ppc_binding_audit.csv"),
    "stage_refit.json": ("provenance", "stage_refit.json"),
    "source_input_manifest.json": ("provenance", "input_manifest.json"),
}


def assert_disjoint_cardinal_paths(
    source_run: str | Path,
    destination: str | Path,
) -> tuple[Path, Path]:
    """Resolve and reject lexical or inode-equivalent ancestor overlap.

    The inode checks are necessary on case-insensitive filesystems such as the
    default macOS APFS, where differently cased paths may resolve to the same
    directory while retaining the caller's spelling.
    """

    source = Path(source_run).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Cardinal destination and formal source run must be disjoint")

    def same_existing_path(left: Path, right: Path) -> bool:
        try:
            return left.exists() and right.exists() and os.path.samefile(left, right)
        except OSError:
            return False

    # Destination inside source: the destination itself may not exist yet, so
    # compare the source inode with every existing destination ancestor.
    for candidate in (output, *output.parents):
        if same_existing_path(source, candidate):
            raise ValueError(
                "Cardinal destination and formal source run must be disjoint"
            )

    # Source inside destination: an ancestor destination necessarily exists
    # when the source exists, but compare every source ancestor to handle path
    # aliases and symlinks robustly.
    if output.exists():
        for candidate in (source, *source.parents):
            if same_existing_path(output, candidate):
                raise ValueError(
                    "Cardinal destination and formal source run must be disjoint"
                )
    return source, output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _sha256_gzip_payload(path: Path) -> str:
    """Hash decompressed CSV bytes so gzip metadata cannot mask equivalence."""

    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_cardinal_input_snapshot(
    snapshot_dir: str | Path,
    expected_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Rehash a copied input snapshot and fail if any byte has changed."""

    root = Path(snapshot_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cardinal input snapshot does not exist: {root}")
    rows: dict[str, dict[str, object]] = {}
    for name, expected in expected_sha256.items():
        path = root / str(name)
        if not path.is_file():
            raise FileNotFoundError(f"Cardinal snapshot input is missing: {path}")
        actual = sha256_file(path)
        if actual != str(expected):
            raise ValueError(
                "Cardinal input snapshot changed after authentication: "
                f"{name}, expected={expected}, actual={actual}"
            )
        rows[str(name)] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return {
        "snapshot_dir": str(root),
        "status": "pass",
        "n_files": int(len(rows)),
        "files": rows,
    }


def _strict_bool_series(values: pd.Series, *, field: str) -> pd.Series:
    """Parse a persisted boolean field without treating arbitrary text as true."""

    if values.isna().any():
        raise ValueError(f"{field} contains missing boolean values")
    normalized = values.map(
        lambda value: (
            value
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower()
        )
    )
    parsed = normalized.map(
        {
            True: True,
            False: False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )
    if parsed.isna().any():
        invalid = sorted(set(values.loc[parsed.isna()].astype(str)))
        raise ValueError(f"{field} contains invalid boolean values: {invalid[:10]}")
    return parsed.astype(bool)


def snapshot_cardinal_inputs(
    source_run: str | Path,
    destination: str | Path,
    *,
    canonical_dataset: str | Path,
    expected_source_hashes: Mapping[str, str],
    require_formal_approval: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Copy the exact formal inputs and validate their cross-file binding.

    Returns ``records``, the approved target registry, posterior draws, and a
    JSON-serializable binding record.  Downstream code must use the copied
    snapshot rather than reopening ``source_run``.
    """

    source, output = assert_disjoint_cardinal_paths(source_run, destination)
    if not source.is_dir():
        raise FileNotFoundError(f"Formal source run does not exist: {source}")

    expected_hashes = {
        str(name): str(value).strip().lower()
        for name, value in expected_source_hashes.items()
    }
    expected_names = set(_INPUT_FILES)
    if set(expected_hashes) != expected_names:
        raise ValueError(
            "Frozen source hash registry must cover the exact classification inputs; "
            f"missing={sorted(expected_names - set(expected_hashes))}, "
            f"unexpected={sorted(set(expected_hashes) - expected_names)}"
        )
    invalid_hashes = sorted(
        name
        for name, value in expected_hashes.items()
        if re.fullmatch(r"[0-9a-f]{64}", value) is None
    )
    if invalid_hashes:
        raise ValueError(f"Frozen source hashes are invalid for: {invalid_hashes}")

    resolved_sources: dict[str, Path] = {}
    for output_name, parts in _INPUT_FILES.items():
        path = source.joinpath(*parts)
        if not path.is_file():
            raise FileNotFoundError(f"Required formal classification input is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hashes[output_name]:
            raise ValueError(
                "Formal source input differs from the frozen classification binding: "
                f"{output_name}, expected={expected_hashes[output_name]}, "
                f"actual={actual_hash}"
            )
        resolved_sources[output_name] = path

    stage_refit = _read_json(resolved_sources["stage_refit.json"])
    target_qa = _read_json(resolved_sources["target_qa.json"])
    if require_formal_approval:
        if stage_refit.get("status") != "formal_evaluation_ready":
            raise ValueError(
                "The source run refit stage is not formally evaluation-ready: "
                f"{stage_refit.get('status')!r}"
            )
        if target_qa.get("formal_evaluation_ready") is not True:
            raise ValueError("Target QA is not formally evaluation-ready")
        if target_qa.get("formal_evaluation_blockers") != []:
            raise ValueError("Target QA retains formal-evaluation blockers")
        hard_checks = target_qa.get("hard_checks")
        if (
            not isinstance(hard_checks, dict)
            or not hard_checks
            or any(value is not True for value in hard_checks.values())
        ):
            raise ValueError("Target QA contains a failed or malformed hard check")
        draw_hard_checks = target_qa.get(
            "posterior_slope_draw_aggregate", {}
        ).get("hard_checks")
        if (
            not isinstance(draw_hard_checks, dict)
            or not draw_hard_checks
            or any(value is not True for value in draw_hard_checks.values())
        ):
            raise ValueError(
                "Target QA contains a failed or malformed posterior-draw check"
            )
        if stage_refit.get("qa") != target_qa:
            raise ValueError(
                "Refit-stage QA and target_qa.json are not exactly identical"
            )
        if int(target_qa.get("n_formal_ppc_approved", -1)) != EXPECTED_RECORDS:
            raise ValueError(
                "Cardinal classification requires 97 named PPC approvals; "
                f"found {target_qa.get('n_formal_ppc_approved')!r}"
            )
        if int(target_qa.get("n_named_ppc_review_complete", -1)) != EXPECTED_RECORDS:
            raise ValueError("Named PPC review is incomplete")
    if target_qa.get("primary_target") != TARGET_NAME:
        raise ValueError(
            f"Formal source run uses the wrong active target: {target_qa.get('primary_target')!r}"
        )

    canonical_root = Path(canonical_dataset).expanduser().resolve()
    canonical_draws_path = canonical_root / "posterior_slope_draws.csv.gz"
    if not canonical_draws_path.is_file():
        raise FileNotFoundError(
            f"Frozen canonical posterior draws are missing: {canonical_draws_path}"
        )
    draw_qa = target_qa.get("posterior_slope_draw_aggregate", {})
    recorded_canonical_hash = str(draw_qa.get("source_aggregate_sha256", ""))
    canonical_byte_hash = sha256_file(canonical_draws_path)
    if canonical_byte_hash != recorded_canonical_hash:
        raise ValueError(
            "Canonical posterior-draw byte hash does not match the formal target QA: "
            f"expected={recorded_canonical_hash!r}, actual={canonical_byte_hash}"
        )
    canonical_semantic_hash = _sha256_gzip_payload(canonical_draws_path)
    source_semantic_hash = _sha256_gzip_payload(
        resolved_sources["posterior_slope_draws.csv.gz"]
    )
    if source_semantic_hash != canonical_semantic_hash:
        raise ValueError(
            "Formal source posterior draws are not semantically identical to the "
            "frozen canonical posterior draws"
        )

    source_tree_before = directory_tree_summary(source)
    output.mkdir(parents=True, exist_ok=True)
    copied_files: dict[str, dict[str, object]] = {}
    for output_name, source_path in resolved_sources.items():
        destination_path = output / output_name
        shutil.copy2(source_path, destination_path)
        source_hash = sha256_file(source_path)
        copied_hash = sha256_file(destination_path)
        if source_hash != expected_hashes[output_name]:
            raise RuntimeError(
                "Formal source input changed after frozen-hash authentication: "
                f"{source_path}"
            )
        if copied_hash != source_hash:
            raise RuntimeError(f"Input snapshot byte hash changed while copying {source_path}")
        copied_files[output_name] = {
            "source_path": str(source_path),
            "snapshot_path": str(destination_path),
            "sha256": copied_hash,
            "size_bytes": destination_path.stat().st_size,
        }

    records = pd.read_csv(output / "records.csv")
    targets = pd.read_csv(output / "target_registry.csv")
    draws = pd.read_csv(output / "posterior_slope_draws.csv.gz")
    bayesian_audit = pd.read_csv(output / "bayesian_audit.csv")
    ppc_audit = pd.read_csv(output / "ppc_binding_audit.csv")

    _validate_records(records)
    selected_targets = _validate_targets(targets)
    _validate_draws(draws, selected_targets)
    _validate_audit_binding(selected_targets, bayesian_audit, ppc_audit)
    _validate_cross_file_ids(records, selected_targets, draws)
    snapshot_verification = verify_cardinal_input_snapshot(
        output,
        expected_hashes,
    )

    source_tree_after = directory_tree_summary(source)
    if source_tree_after["tree_sha256"] != source_tree_before["tree_sha256"]:
        raise RuntimeError("Formal source run changed while classification inputs were copied")

    binding: dict[str, Any] = {
        "source_run": str(source),
        "source_run_tree_sha256_before": source_tree_before["tree_sha256"],
        "source_run_tree_sha256_after_snapshot": source_tree_after["tree_sha256"],
        "source_run_unchanged_during_snapshot": True,
        "snapshot_dir": str(output),
        "files": copied_files,
        "target_name": TARGET_NAME,
        "n_records": int(len(records)),
        "n_alkaline_records": int(
            records["electrolyte_scope"].astype(str).eq("alkaline").sum()
        ),
        "n_approved_targets": int(len(selected_targets)),
        "n_posterior_draws": int(len(draws)),
        "draws_per_record": EXPECTED_DRAWS_PER_RECORD,
        "n_connected_groups_all": int(records["cv_group_id"].astype(str).nunique()),
        "formal_evaluation_ready": (
            target_qa.get("formal_evaluation_ready") is True
        ),
        "n_formal_ppc_approved": int(
            target_qa.get("n_formal_ppc_approved", -1)
        ),
        "formal_run_draw_file_sha256": copied_files[
            "posterior_slope_draws.csv.gz"
        ]["sha256"],
        "frozen_source_file_hashes_verified": True,
        "frozen_source_file_sha256": expected_hashes,
        "canonical_draw_file": str(canonical_draws_path),
        "canonical_draw_file_sha256": canonical_byte_hash,
        "canonical_source_aggregate_sha256_recorded_by_target_qa": (
            recorded_canonical_hash
        ),
        "canonical_decompressed_csv_sha256": canonical_semantic_hash,
        "formal_run_decompressed_csv_sha256": source_semantic_hash,
        "formal_draws_semantically_match_canonical": True,
        "snapshot_verification_after_validation": snapshot_verification,
        "mechanistic_claim_allowed": False,
    }
    return records, selected_targets, draws, binding


def verify_source_run_unchanged(
    source_run: str | Path,
    *,
    expected_tree_sha256: str,
) -> dict[str, object]:
    """Verify that an independent run did not mutate its formal source."""

    source = Path(source_run).expanduser().resolve()
    summary = directory_tree_summary(source)
    actual = str(summary["tree_sha256"])
    if actual != str(expected_tree_sha256):
        raise RuntimeError(
            "Formal source run changed during cardinal classification: "
            f"expected={expected_tree_sha256}, actual={actual}"
        )
    return {
        "source_run": str(source),
        "expected_tree_sha256": str(expected_tree_sha256),
        "actual_tree_sha256": actual,
        "unchanged": True,
    }


def _validate_records(records: pd.DataFrame) -> None:
    required = {
        "record_id",
        "sample_record",
        "curve_id",
        "cv_group_id",
        "electrolyte_scope",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise KeyError(f"Formal records table is missing columns: {missing}")
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"Expected 97 formal records, found {len(records)}")
    for column in ("record_id", "sample_record", "curve_id"):
        if records[column].astype(str).duplicated().any():
            raise ValueError(f"Formal records column {column!r} must be unique")
    if records["cv_group_id"].isna().any():
        raise ValueError("Formal records contain missing cv_group_id")
    alkaline = int(records["electrolyte_scope"].astype(str).eq("alkaline").sum())
    if alkaline != EXPECTED_ALKALINE:
        raise ValueError(f"Expected 90 alkaline records, found {alkaline}")


def _validate_targets(targets: pd.DataFrame) -> pd.DataFrame:
    required = {
        "record_id",
        "sample_record",
        "curve_id",
        "cv_group_id",
        "electrolyte_scope",
        "target_name",
        "target_role",
        "target_eligible",
        "formal_target_approved",
        "posterior_evidence_id",
        "slope_draws_sha256",
    }
    missing = sorted(required - set(targets.columns))
    if missing:
        raise KeyError(f"Formal target registry is missing columns: {missing}")
    eligible = targets.loc[
        targets["target_name"].astype(str).eq(TARGET_NAME)
        & targets["target_role"].astype(str).eq("primary")
        & _strict_bool_series(
            targets["target_eligible"], field="target_registry.target_eligible"
        )
        & _strict_bool_series(
            targets["formal_target_approved"],
            field="target_registry.formal_target_approved",
        )
    ].copy()
    if len(eligible) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected 97 approved primary Bayesian local-line targets, found {len(eligible)}"
        )
    if eligible["record_id"].astype(str).duplicated().any():
        raise ValueError("Approved target registry has duplicate record_id")
    return eligible.reset_index(drop=True)


def _validate_draws(draws: pd.DataFrame, targets: pd.DataFrame) -> None:
    required = {
        "sample_record",
        "curve_id",
        "posterior_draw_index",
        "target_name",
        "primary_slope_mV_dec",
        "eligible",
        "evidence_id",
    }
    missing = sorted(required - set(draws.columns))
    if missing:
        raise KeyError(f"Posterior draw table is missing columns: {missing}")
    expected_rows = EXPECTED_RECORDS * EXPECTED_DRAWS_PER_RECORD
    if len(draws) != expected_rows:
        raise ValueError(f"Expected {expected_rows} posterior draws, found {len(draws)}")
    if not draws["target_name"].astype(str).eq(TARGET_NAME).all():
        raise ValueError("Posterior draw table contains a non-active target")
    if not _strict_bool_series(
        draws["eligible"], field="posterior_draws.eligible"
    ).all():
        raise ValueError("Posterior draw table contains ineligible draws")
    numeric = pd.to_numeric(draws["primary_slope_mV_dec"], errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Posterior slope draws contain NaN or infinite values")
    counts = draws.groupby("sample_record", dropna=False).size()
    if len(counts) != EXPECTED_RECORDS or not counts.eq(EXPECTED_DRAWS_PER_RECORD).all():
        raise ValueError("Every formal record must have exactly 8,000 posterior draws")
    if draws.duplicated(["curve_id", "posterior_draw_index"]).any():
        raise ValueError("Posterior draw table contains duplicate curve/draw indices")
    evidence_counts = draws.groupby("sample_record")["evidence_id"].nunique()
    if not evidence_counts.eq(1).all():
        raise ValueError("A record's posterior draws reference multiple evidence IDs")
    target_evidence = targets.set_index("sample_record")["posterior_evidence_id"].astype(str)
    draw_evidence = (
        draws.groupby("sample_record")["evidence_id"].first().astype(str)
    )
    if set(target_evidence.index) != set(draw_evidence.index):
        raise ValueError("Posterior draw and target sample sets differ")
    mismatch = [
        str(sample)
        for sample in target_evidence.index
        if target_evidence.loc[sample] != draw_evidence.loc[sample]
    ]
    if mismatch:
        raise ValueError(f"Posterior evidence ID mismatch for samples: {mismatch}")


def _validate_audit_binding(
    targets: pd.DataFrame,
    bayesian_audit: pd.DataFrame,
    ppc_audit: pd.DataFrame,
) -> None:
    if len(bayesian_audit) != EXPECTED_RECORDS or len(ppc_audit) != EXPECTED_RECORDS:
        raise ValueError("Bayesian/PPC audit tables must each contain 97 rows")
    target = targets.copy()
    target["_record_key"] = target["record_id"].astype(str)
    bayes = bayesian_audit.copy()
    bayes["_record_key"] = bayes["record_id"].astype(str)
    ppc = ppc_audit.copy()
    ppc["_record_key"] = ppc["record_id"].astype(str)
    joined = target[
        ["_record_key", "posterior_evidence_id", "slope_draws_sha256"]
    ].merge(
        bayes[
            [
                "_record_key",
                "evidence_id",
                "slope_draws_sha256",
                "formal_target_approved",
                "ppc_status",
            ]
        ],
        on="_record_key",
        suffixes=("_target", "_bayes"),
        validate="one_to_one",
    ).merge(
        ppc[
            [
                "_record_key",
                "evidence_id_review",
                "slope_draws_sha256_review",
                "formal_target_approved_bound",
                "ppc_status_bound",
            ]
        ],
        on="_record_key",
        validate="one_to_one",
    )
    if len(joined) != EXPECTED_RECORDS:
        raise ValueError("Target, Bayesian audit, and PPC audit record sets differ")
    if not (
        joined["posterior_evidence_id"].astype(str).eq(
            joined["evidence_id"].astype(str)
        )
        & joined["posterior_evidence_id"].astype(str).eq(
            joined["evidence_id_review"].astype(str)
        )
    ).all():
        raise ValueError("Evidence IDs disagree across target/Bayesian/PPC binding")
    if not (
        joined["slope_draws_sha256_target"].astype(str).eq(
            joined["slope_draws_sha256_bayes"].astype(str)
        )
        & joined["slope_draws_sha256_target"].astype(str).eq(
            joined["slope_draws_sha256_review"].astype(str)
        )
    ).all():
        raise ValueError("Per-curve slope-draw hashes disagree across binding tables")
    if not _strict_bool_series(
        joined["formal_target_approved"],
        field="bayesian_audit.formal_target_approved",
    ).all():
        raise ValueError("Bayesian audit contains an unapproved formal target")
    if not _strict_bool_series(
        joined["formal_target_approved_bound"],
        field="ppc_binding_audit.formal_target_approved_bound",
    ).all():
        raise ValueError("PPC audit contains an unbound approval")
    if not joined["ppc_status"].astype(str).eq("named_approved").all():
        raise ValueError("Bayesian audit contains a non-approved PPC status")
    if not joined["ppc_status_bound"].astype(str).eq("named_approved").all():
        raise ValueError("PPC binding audit contains a non-approved status")

    required_review_fields = {
        "ppc_approved",
        "ppc_approved_bool",
        "ppc_reviewer",
        "ppc_review_date",
    }
    missing_review_fields = sorted(required_review_fields - set(ppc.columns))
    if missing_review_fields:
        raise KeyError(
            f"PPC binding audit is missing named-review fields: {missing_review_fields}"
        )
    if not _strict_bool_series(
        ppc["ppc_approved"], field="ppc_binding_audit.ppc_approved"
    ).all():
        raise ValueError("PPC binding audit contains a human review not approved")
    if not _strict_bool_series(
        ppc["ppc_approved_bool"], field="ppc_binding_audit.ppc_approved_bool"
    ).all():
        raise ValueError("PPC binding audit contains a false parsed approval")
    reviewers = ppc["ppc_reviewer"].astype("string").str.strip()
    if reviewers.isna().any() or reviewers.eq("").any():
        raise ValueError("PPC binding audit requires a named reviewer for all records")
    review_dates = ppc["ppc_review_date"].astype("string").str.strip()
    iso_format = review_dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    parsed_dates = pd.to_datetime(review_dates, format="%Y-%m-%d", errors="coerce")
    if not iso_format.all() or parsed_dates.isna().any():
        raise ValueError("PPC binding audit requires valid ISO-8601 review dates")


def _validate_cross_file_ids(
    records: pd.DataFrame,
    targets: pd.DataFrame,
    draws: pd.DataFrame,
) -> None:
    left = records[
        ["record_id", "sample_record", "curve_id", "cv_group_id", "electrolyte_scope"]
    ].copy()
    right = targets[
        ["record_id", "sample_record", "curve_id", "cv_group_id", "electrolyte_scope"]
    ].copy()
    for frame in (left, right):
        for column in frame.columns:
            frame[column] = frame[column].astype(str)
    merged = left.merge(
        right,
        on="record_id",
        suffixes=("_records", "_targets"),
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_RECORDS:
        raise ValueError("Formal records and approved targets have different record sets")
    for column in ("sample_record", "curve_id", "cv_group_id", "electrolyte_scope"):
        if not merged[f"{column}_records"].eq(merged[f"{column}_targets"]).all():
            raise ValueError(f"Formal records and targets disagree on {column}")
    draw_pairs = set(
        zip(
            draws["sample_record"].astype(str),
            draws["curve_id"].astype(str),
            strict=False,
        )
    )
    record_pairs = set(
        zip(
            records["sample_record"].astype(str),
            records["curve_id"].astype(str),
            strict=False,
        )
    )
    if draw_pairs != record_pairs:
        raise ValueError("Posterior draws do not match formal sample/curve mappings")
