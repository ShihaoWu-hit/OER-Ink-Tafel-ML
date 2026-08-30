from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .bayesian_evidence import (
    discover_bayesian_evidence,
    sync_manual_ppc_review,
    write_evidence_inventory,
)
from .provenance import sha256_file


MAX_AUTOMATIC_FORMAL_ATTEMPTS = 2
_ATTEMPT_PATTERN = re.compile(r"_attempt(thisP<attempt>[0-9]+)(this:_|\.|$)")


def _stable_job_ordinal(queue: pd.DataFrame, curve_id: str) -> int:
    curve_ids = sorted(queue["curve_id"].astype(str).unique())
    if len(curve_ids) != len(queue):
        raise ValueError("Bayesian queue must contain one row per unique curve_id")
    try:
        return curve_ids.index(str(curve_id)) + 1
    except ValueError as exc:
        raise KeyError(f"Unknown curve_id in Bayesian queue: {curve_id}") from exc


def _sampling_seed(*, job_ordinal: int, attempt: int, smoke: bool) -> int:
    if job_ordinal < 1 or attempt < 1:
        raise ValueError("job_ordinal and attempt must be positive")
    namespace = 1_000_000_000 if smoke else 0
    return namespace + job_ordinal * 10_000 + attempt * 10 + 42


def _attempt_from_name(name: str) -> int | None:
    match = _ATTEMPT_PATTERN.search(name)
    return int(match.group("attempt")) if match else None


def _next_sampling_attempt(
    *,
    output: Path,
    bayes_dir: Path,
    curve_id: str,
    model: str,
) -> int:
    attempts: set[int] = set()
    prefix = f"{curve_id}_{model}_seed"
    if bayes_dir.is_dir():
        for path in bayes_dir.glob(f"{prefix}*_attempt*"):
            attempt = _attempt_from_name(path.name)
            if attempt is not None:
                attempts.add(attempt)
    for path in output.glob(f"{curve_id}_{model}_*_attempt*_seed*_nuts.log"):
        attempt = _attempt_from_name(path.name)
        if attempt is not None:
            attempts.add(attempt)
    history_path = output / "bayesian_batch_results.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        if {"curve_id", "selected_model", "sampling_attempt"}.issubset(history):
            used = pd.to_numeric(
                history.loc[
                    history["curve_id"].astype(str).eq(curve_id)
                    & history["selected_model"].astype(str).eq(model),
                    "sampling_attempt",
                ],
                errors="coerce",
            ).dropna()
            attempts.update(int(value) for value in used if int(value) >= 1)
    return max(attempts, default=0) + 1


def _completed_sampling_attempt_count(
    *,
    output: Path,
    bayes_dir: Path,
    curve_id: str,
    model: str,
) -> int:
    """Count completed samplings, excluding process/preflight failures.

    The artifact attempt ordinal remains monotone after any interrupted or
    process-failed launch, but an environment error before ``pm.sample`` must
    not consume one of the two preregistered scientific sampling attempts.
    """

    attempts: set[int] = set()
    prefix = f"{curve_id}_{model}_seed"
    if bayes_dir.is_dir():
        for path in bayes_dir.glob(f"{prefix}*_attempt*_diagnostics.json"):
            attempt = _attempt_from_name(path.name)
            if attempt is not None:
                attempts.add(attempt)
    history_path = output / "bayesian_batch_results.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        required = {"curve_id", "selected_model", "sampling_attempt", "status"}
        if required.issubset(history):
            completed = history.loc[
                history["curve_id"].astype(str).eq(curve_id)
                & history["selected_model"].astype(str).eq(model)
                & history["status"].astype(str).isin(
                    {"sampled_sampler_pass", "sampled_sampler_failed"}
                )
            ]
            attempts.update(
                int(value)
                for value in pd.to_numeric(
                    completed["sampling_attempt"], errors="coerce"
                ).dropna()
                if int(value) >= 1
            )
    return len(attempts)


def _ensure_functional_target_artifacts(
    *,
    isolated_run: Path,
    python: Path,
    env: dict[str, str],
) -> int:
    """Derive primary branch-slope and audit-only 2 mA sidecars."""

    bayes_dir = isolated_run / "bayes"
    if not bayes_dir.is_dir():
        return 0
    for diagnostics_path in sorted(bayes_dir.glob("*_diagnostics.json")):
        evidence_id = diagnostics_path.name.removesuffix("_diagnostics.json")
        functional_path = bayes_dir / f"{evidence_id}_functional_targets_2mA.csv.gz"
        slope_path = bayes_dir / f"{evidence_id}_posterior_slope_draws.csv.gz"
        if functional_path.is_file() and slope_path.is_file():
            continue
        command = [
            str(python),
            "-m",
            "ink_tafel_study.posterior_functional",
            "--run-dir",
            str(isolated_run),
            "--diagnostics",
            str(diagnostics_path),
            "--target-current",
            "2.0",
        ]
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Posterior functional-target extraction failed for "
                f"{diagnostics_path}: {result.stderr.strip()}"
            )
    return len(list(bayes_dir.glob("*_posterior_slope_draws.csv.gz")))


def _verify_workflow_compatibility(source: Path, project: Path) -> dict[str, Any]:
    """Fail closed when the sampling code differs from the deterministic run.

    The deterministic refit stores a hash for every workflow source file.  A
    selected-model NUTS result can only be treated as a formal confirmation
    candidate when those exact files are still present and unchanged.  This
    preflight avoids launching a batch that the refit CLI would reject one job
    at a time.
    """

    config_path = source / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_files = config.get("workflow_files_sha256")
    expected_tree = config.get("workflow_tree_sha256")
    if not isinstance(expected_files, dict) or not expected_files or not expected_tree:
        raise ValueError(
            "Deterministic refit run has no complete workflow-hash manifest; "
            "rerun the deterministic workflow before NUTS sampling"
        )

    mismatches: list[str] = []
    for relative, expected_hash in sorted(expected_files.items()):
        path = project / relative
        if not path.is_file():
            mismatches.append(f"{relative} (missing)")
        elif sha256_file(path) != expected_hash:
            mismatches.append(f"{relative} (changed)")
    if mismatches:
        preview = ", ".join(mismatches[:8])
        suffix = " ..." if len(mismatches) > 8 else ""
        raise ValueError(
            "Current Tafel_refitting workflow differs from the deterministic "
            f"run ({preview}{suffix}). Create a new deterministic run with the "
            "current workflow; do not bypass this gate for formal confirmation."
        )
    return {
        "workflow_hash_match": True,
        "deterministic_workflow_tree_sha256": str(expected_tree),
        "workflow_files_verified": len(expected_files),
    }


def _copy_deterministic_run(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    config_path = source / "model_config.json"
    status_path = source / "run_status.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = config.get("deterministic_artifacts_sha256", {})
    if not expected:
        raise ValueError("Deterministic refit run has no artifact hash manifest")
    for source_path in (config_path, status_path):
        destination_path = destination / source_path.name
        if destination_path.is_file():
            if sha256_file(destination_path) != sha256_file(source_path):
                raise ValueError(
                    "Existing isolated deterministic metadata conflicts with source: "
                    f"{destination_path}"
                )
        else:
            shutil.copy2(source_path, destination_path)
    for relative, expected_hash in expected.items():
        source_path = source / relative
        if sha256_file(source_path) != expected_hash:
            raise ValueError(f"Deterministic refit artifact failed source hash check: {source_path}")
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.is_file():
            if sha256_file(destination_path) != expected_hash:
                raise ValueError(
                    "Existing isolated deterministic artifact conflicts with source: "
                    f"{destination_path}"
                )
            continue
        shutil.copy2(source_path, destination_path)
        if sha256_file(destination_path) != expected_hash:
            raise RuntimeError(f"Copied refit artifact failed parity check: {destination_path}")


def run_missing_bayesian_jobs(
    *,
    deterministic_run: str | Path,
    queue_path: str | Path,
    output_dir: str | Path,
    refit_project: str | Path,
    limit: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run a resumable selected-model NUTS queue in an isolated run copy.

    Sampling output is never interpreted as PPC approval.  The generated review
    template requires a named human reviewer before downstream formal approval.
    """

    source = Path(deterministic_run).resolve()
    output = Path(output_dir).resolve()
    project = Path(refit_project).resolve()
    isolated_run = output / "deterministic_run_copy"
    output.mkdir(parents=True, exist_ok=True)
    workflow_verification = _verify_workflow_compatibility(source, project)
    _copy_deterministic_run(source, isolated_run)

    queue = pd.read_csv(queue_path)
    required = {"curve_id", "sample_record", "selected_model", "nuts_diagnostics_present", "next_action"}
    if not required.issubset(queue):
        raise KeyError(f"Bayesian queue is missing {sorted(required - set(queue))}")
    readiness_column = (
        "formal_nuts_evidence_present"
        if "formal_nuts_evidence_present" in queue.columns
        else "nuts_diagnostics_present"
    )
    pending = queue.loc[
        ~queue[readiness_column].fillna(False).astype(bool)
        & queue["next_action"].astype(str).str.contains("run_selected_model_nuts", regex=False)
    ].sort_values(["queue_priority", "sample_record", "curve_id"])
    if smoke:
        limit = 1 if limit is None else min(limit, 1)
    if limit is not None:
        pending = pending.head(max(0, int(limit)))

    python = project / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"Tafel refit environment is missing: {python}")
    env = os.environ.copy()
    study_src = Path(__file__).resolve().parents[2] / "src"
    python_paths = [str(study_src), str(project / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if not env.get("PYTENSOR_FLAGS"):
        pytensor_cache = Path(tempfile.gettempdir()) / "ink_tafel_pytensor_cache"
        pytensor_cache.mkdir(parents=True, exist_ok=True)
        env["PYTENSOR_FLAGS"] = f"base_compiledir={pytensor_cache}"
    _ensure_functional_target_artifacts(
        isolated_run=isolated_run,
        python=python,
        env=env,
    )
    draws, chains = (100, 2) if smoke else (2000, 4)
    rows: list[dict[str, Any]] = []
    history_path = output / "bayesian_batch_results.csv"
    history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()

    def record_result(row: dict[str, Any]) -> None:
        nonlocal history
        rows.append(row)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        history.to_csv(history_path, index=False)

    for _, item in pending.iterrows():
        curve_id = str(item["curve_id"])
        model = str(item["selected_model"])
        existing_evidence = discover_bayesian_evidence(
            [isolated_run / "bayes"],
            current_deterministic_config=isolated_run / "model_config.json",
        )
        existing_formal = existing_evidence.loc[
            existing_evidence.get("curve_id", pd.Series(dtype=str)).astype(str).eq(curve_id)
            & existing_evidence.get(
                "formal_sampling_candidate", pd.Series(dtype=bool)
            ).fillna(False).astype(bool)
        ]
        if not existing_formal.empty:
            record_result(
                {
                    "curve_id": curve_id,
                    "selected_model": model,
                    "status": "skipped_existing",
                    "diagnostics_path": str(existing_formal.iloc[0]["diagnostics_path"]),
                    "sampling_attempt": existing_formal.iloc[0].get(
                        "sampling_attempt", ""
                    ),
                    "sampling_seed": existing_formal.iloc[0].get(
                        "sampling_seed", ""
                    ),
                }
            )
            continue
        attempt = _next_sampling_attempt(
            output=output,
            bayes_dir=isolated_run / "bayes",
            curve_id=curve_id,
            model=model,
        )
        completed_sampling_attempts = _completed_sampling_attempt_count(
            output=output,
            bayes_dir=isolated_run / "bayes",
            curve_id=curve_id,
            model=model,
        )
        sampling_sequence = completed_sampling_attempts + 1
        if not smoke and sampling_sequence > MAX_AUTOMATIC_FORMAL_ATTEMPTS:
            record_result(
                {
                    "curve_id": curve_id,
                    "selected_model": model,
                    "status": "retry_limit_reached",
                    "sampling_attempt": attempt,
                    "sampling_sequence": sampling_sequence,
                    "returncode": "",
                    "sampler_diagnostics_pass": False,
                    "note": (
                        "Automatic retries are capped to avoid seed-shopping; "
                        "review model parameterization before another attempt."
                    ),
                }
            )
            continue
        job_ordinal = _stable_job_ordinal(queue, curve_id)
        seed = _sampling_seed(
            job_ordinal=job_ordinal,
            attempt=attempt,
            smoke=smoke,
        )
        tune = 100 if smoke else (1000 if sampling_sequence == 1 else 2000)
        target_accept = 0.95 if smoke or sampling_sequence == 1 else 0.99
        command = [
            str(python),
            "-m",
            "tafel_refit.cli",
            "bayes",
            "--run-dir",
            str(isolated_run),
            "--curve-id",
            curve_id,
            "--model",
            model,
            "--draws",
            str(draws),
            "--tune",
            str(tune),
            "--chains",
            str(chains),
            "--target-accept",
            str(target_accept),
            "--seed",
            str(seed),
            "--attempt",
            str(attempt),
        ]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        mode_name = "smoke" if smoke else "formal"
        log_path = output / (
            f"{curve_id}_{model}_{mode_name}_attempt{attempt:02d}_seed{seed}_nuts.log"
        )
        log_path.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        evidence_id = f"{curve_id}_{model}_seed{seed}_attempt{attempt:02d}"
        diagnostics_path = isolated_run / "bayes" / f"{evidence_id}_diagnostics.json"
        diagnostics = {}
        if diagnostics_path.is_file():
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        sampler_pass = bool(diagnostics.get("sampler_diagnostics_pass", False))
        if completed.returncode != 0:
            status = "process_failed"
        elif sampler_pass:
            status = "sampled_sampler_pass"
        else:
            status = "sampled_sampler_failed"
        record_result(
            {
                "curve_id": curve_id,
                "selected_model": model,
                "status": status,
                "returncode": completed.returncode,
                "job_ordinal": job_ordinal,
                "sampling_attempt": attempt,
                "sampling_sequence": sampling_sequence,
                "sampling_seed": seed,
                "posterior_predictive_seed": seed + 1,
                "draws": draws,
                "tune": tune,
                "chains": chains,
                "target_accept": target_accept,
                "sampler_diagnostics_pass": sampler_pass,
                "diagnostics_path": str(diagnostics_path)
                if diagnostics_path.is_file()
                else "",
                "log_path": str(log_path),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(
            columns=[
                "curve_id",
                "selected_model",
                "status",
                "returncode",
                "job_ordinal",
                "sampling_attempt",
                "sampling_sequence",
                "sampling_seed",
                "posterior_predictive_seed",
                "draws",
                "tune",
                "chains",
                "target_accept",
                "sampler_diagnostics_pass",
                "diagnostics_path",
                "log_path",
            ]
        )
        if not history_path.is_file():
            result.to_csv(history_path, index=False)
    functional_artifacts = _ensure_functional_target_artifacts(
        isolated_run=isolated_run,
        python=python,
        env=env,
    )
    evidence = discover_bayesian_evidence(
        [isolated_run / "bayes"],
        current_deterministic_config=isolated_run / "model_config.json",
    )
    write_evidence_inventory(output / "bayesian_artifact_inventory.csv", evidence)
    sample_records = {
        str(row["curve_id"]): int(row["sample_record"])
        for row in queue[["curve_id", "sample_record"]].to_dict("records")
    }
    review = sync_manual_ppc_review(
        output / "ppc_review.csv",
        evidence,
        sample_records=sample_records,
    )
    pending_review = review.loc[
        ~review["ppc_approved"].fillna("").astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "y"}
        )
    ].copy()
    pending_review.to_csv(output / "ppc_review_required.csv", index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "isolated_run": str(isolated_run),
        "selected_jobs": int(len(pending)),
        "sampled_jobs": int(
            result.get("status", pd.Series(dtype=str))
            .isin({"sampled_sampler_pass", "sampled_sampler_failed"})
            .sum()
        ),
        "sampler_pass_jobs": int(
            (result.get("status", pd.Series(dtype=str)) == "sampled_sampler_pass").sum()
        ),
        "sampler_failed_jobs": int(
            (result.get("status", pd.Series(dtype=str)) == "sampled_sampler_failed").sum()
        ),
        "process_failed_jobs": int(
            (result.get("status", pd.Series(dtype=str)) == "process_failed").sum()
        ),
        "failed_jobs": int(
            (result.get("status", pd.Series(dtype=str)) == "process_failed").sum()
        ),
        "retry_limit_jobs": int(
            (result.get("status", pd.Series(dtype=str)) == "retry_limit_reached").sum()
        ),
        "formal_sampling_evidence_sets": int(
            evidence.get("formal_sampling_candidate", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "functional_target_artifacts": int(functional_artifacts),
        "review_jobs": int(len(pending_review)),
        "smoke": smoke,
        "formal_ppc_approvals_added": 0,
        "next_step": (
            "Complete bayesian_batch/ppc_review.csv with a named reviewer and ISO-8601 "
            "review date, then rerun the refit stage to rebuild the target audit. "
            "Never edit ppc_review_required.csv; it is a derived pending-only view."
        ),
        **workflow_verification,
    }
    (output / "bayesian_batch_status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
