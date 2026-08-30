import json
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ink_tafel_study.provenance import sha256_file
from ink_tafel_study.bayesian_evidence import (
    discover_bayesian_evidence,
    sync_manual_ppc_review,
    validate_bound_ppc_review,
)
from ink_tafel_study.refit_queue import (
    _completed_sampling_attempt_count,
    _copy_deterministic_run,
    _next_sampling_attempt,
    _sampling_seed,
    _stable_job_ordinal,
    _verify_workflow_compatibility,
)


def test_pending_queue_distinguishes_sampling_from_named_ppc() -> None:
    queue = pd.DataFrame(
        {
            "nuts_diagnostics_present": [True, False],
            "named_ppc_review_complete": [False, False],
            "formal_nuts_ppc_approved": [False, False],
        }
    )
    assert queue["nuts_diagnostics_present"].sum() == 1
    assert queue["named_ppc_review_complete"].sum() == 0
    assert queue["formal_nuts_ppc_approved"].sum() == 0


def test_sampling_identity_is_unique_by_curve_and_attempt() -> None:
    queue = pd.DataFrame(
        {
            "curve_id": ["sr042_v2", "sr001", "sr042_v1"],
        }
    )
    ordinal_v1 = _stable_job_ordinal(queue, "sr042_v1")
    ordinal_v2 = _stable_job_ordinal(queue, "sr042_v2")
    seed_v1_attempt1 = _sampling_seed(
        job_ordinal=ordinal_v1, attempt=1, smoke=False
    )
    seed_v2_attempt1 = _sampling_seed(
        job_ordinal=ordinal_v2, attempt=1, smoke=False
    )
    seed_v1_attempt2 = _sampling_seed(
        job_ordinal=ordinal_v1, attempt=2, smoke=False
    )
    assert seed_v1_attempt1 != seed_v2_attempt1
    assert seed_v1_attempt2 not in {seed_v1_attempt1, seed_v1_attempt1 + 1}
    assert _sampling_seed(job_ordinal=ordinal_v1, attempt=1, smoke=True) > 1_000_000_000


def test_next_attempt_advances_after_partial_artifacts_logs_and_history(
    tmp_path: Path,
) -> None:
    output = tmp_path / "batch"
    bayes = output / "deterministic_run_copy" / "bayes"
    bayes.mkdir(parents=True)
    (bayes / "sr001_linear_seed10052_attempt01.nc").write_bytes(b"partial")
    assert (
        _next_sampling_attempt(
            output=output,
            bayes_dir=bayes,
            curve_id="sr001",
            model="linear",
        )
        == 2
    )
    (output / "sr001_linear_formal_attempt02_seed10062_nuts.log").write_text(
        "interrupted", encoding="utf-8"
    )
    assert (
        _next_sampling_attempt(
            output=output,
            bayes_dir=bayes,
            curve_id="sr001",
            model="linear",
        )
        == 3
    )
    pd.DataFrame(
        {
            "curve_id": ["sr001"],
            "selected_model": ["linear"],
            "sampling_attempt": [3],
        }
    ).to_csv(output / "bayesian_batch_results.csv", index=False)
    assert (
        _next_sampling_attempt(
            output=output,
            bayes_dir=bayes,
            curve_id="sr001",
            model="linear",
        )
        == 4
    )


def test_process_failure_does_not_consume_scientific_sampling_attempt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "batch"
    bayes = output / "deterministic_run_copy" / "bayes"
    bayes.mkdir(parents=True)
    pd.DataFrame(
        {
            "curve_id": ["sr001", "sr001", "sr001"],
            "selected_model": ["linear", "linear", "linear"],
            "sampling_attempt": [1, 2, 3],
            "status": [
                "process_failed",
                "sampled_sampler_failed",
                "sampled_sampler_pass",
            ],
        }
    ).to_csv(output / "bayesian_batch_results.csv", index=False)

    assert (
        _completed_sampling_attempt_count(
            output=output,
            bayes_dir=bayes,
            curve_id="sr001",
            model="linear",
        )
        == 2
    )


def test_isolated_deterministic_copy_resumes_and_fails_on_conflict(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "csv").mkdir(parents=True)
    artifact = source / "csv" / "input.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    (source / "run_status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (source / "model_config.json").write_text(
        json.dumps(
            {
                "deterministic_artifacts_sha256": {
                    "csv/input.csv": sha256_file(artifact)
                }
            }
        ),
        encoding="utf-8",
    )
    destination.mkdir()
    _copy_deterministic_run(source, destination)
    assert (destination / "csv" / "input.csv").read_bytes() == artifact.read_bytes()
    (destination / "csv" / "input.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with source"):
        _copy_deterministic_run(source, destination)


def test_bayesian_queue_fails_closed_on_workflow_hash_mismatch(tmp_path: Path) -> None:
    project = tmp_path / "Tafel_refitting"
    source = tmp_path / "deterministic_run"
    workflow_file = project / "src" / "tafel_refit" / "bayes.py"
    workflow_file.parent.mkdir(parents=True)
    source.mkdir()
    workflow_file.write_text("original\n", encoding="utf-8")
    (source / "model_config.json").write_text(
        json.dumps(
            {
                "workflow_tree_sha256": "registered-tree",
                "workflow_files_sha256": {
                    "src/tafel_refit/bayes.py": sha256_file(workflow_file)
                },
            }
        ),
        encoding="utf-8",
    )
    assert _verify_workflow_compatibility(source, project)["workflow_hash_match"]

    workflow_file.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the deterministic run"):
        _verify_workflow_compatibility(source, project)


def _write_complete_evidence(tmp_path: Path) -> tuple[Path, Path]:
    bayes = tmp_path / "bayes"
    bayes.mkdir()
    prefix = "sr001_linear_seed1042"
    expected_artifacts = {"csv/input.csv": "registered-input"}
    diagnostics = {
        "created_at_utc": "2026-07-16T00:00:00+00:00",
        "curve_id": "sr001",
        "model_name": "linear",
        "draws_per_chain": 2000,
        "tune_per_chain": 1000,
        "chains": 4,
        "max_r_hat": 1.001,
        "min_ess_bulk": 2500,
        "divergences": 0,
        "sampler_diagnostics_pass": True,
        "deterministic_artifacts_sha256": expected_artifacts,
        "deterministic_run_workflow_tree_sha256": "tree",
        "workflow_tree_sha256_at_bayes": "tree",
        "workflow_tree_sha256_at_bayes_end": "tree",
        "formal_confirmation_provenance_eligible": True,
        "model_matches_deterministic_selection": True,
        "allow_workflow_mismatch": False,
    }
    (bayes / f"{prefix}_diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )
    (bayes / f"{prefix}.nc").write_bytes(b"netcdf")
    (bayes / f"{prefix}_posterior_predictive.png").write_bytes(b"png")
    pd.DataFrame(
        {"mean": [100.0], "hdi_2.5%": [90.0], "hdi_97.5%": [110.0]},
        index=["slope_mV_dec"],
    ).to_csv(bayes / f"{prefix}_posterior_summary.csv")
    diagnostics_sha256 = sha256_file(bayes / f"{prefix}_diagnostics.json")
    netcdf_sha256 = sha256_file(bayes / f"{prefix}.nc")
    functional = pd.DataFrame(
        {
            "functional_schema_version": ["posterior-functional-target-v1"] * 8000,
            "curve_id": ["sr001"] * 8000,
            "model_name": ["linear"] * 8000,
            "chain": np.repeat(np.arange(4), 2000),
            "draw": np.tile(np.arange(2000), 4),
            "posterior_draw_index": np.arange(8000),
            "target_current_density_mA_cm2": [2.0] * 8000,
            "local_slope_mV_dec": [100.0] * 8000,
            "kinetic_slope_mV_dec": [100.0] * 8000,
            "apparent_slope_mV_dec": [101.0] * 8000,
            "target_slope_semantics": [
                "local_kinetic_slope_at_complete_model_2mA_potential"
            ] * 8000,
            "eligible": [True] * 8000,
            "observed_support": [True] * 8000,
            "extrapolated": [False] * 8000,
            "exclusion_reason": ["eligible"] * 8000,
            "evidence_id": [prefix] * 8000,
            "source_netcdf_sha256": [netcdf_sha256] * 8000,
            "source_diagnostics_sha256": [diagnostics_sha256] * 8000,
        }
    )
    functional_path = bayes / f"{prefix}_functional_targets_2mA.csv.gz"
    with functional_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                functional.to_csv(handle, index=False)
    slope_draws = pd.DataFrame(
        {
            "slope_schema_version": ["posterior-branch-slope-v1"] * 8000,
            "curve_id": ["sr001"] * 8000,
            "sample_record": [1] * 8000,
            "model_name": ["linear"] * 8000,
            "chain": np.repeat(np.arange(4), 2000),
            "draw": np.tile(np.arange(2000), 4),
            "posterior_draw_index": np.arange(8000),
            "target_name": ["b_low_potential_mV_dec"] * 8000,
            "primary_slope_mV_dec": [100.0] * 8000,
            "slope_low_mV_dec": [100.0] * 8000,
            "slope_high_mV_dec": [100.0] * 8000,
            "change_potential_relative_V": [np.nan] * 8000,
            "eligible": [True] * 8000,
            "branch_semantics": ["single_regime_bayesian_slope"] * 8000,
            "posterior_source_variable": ["slope_mV_dec"] * 8000,
            "evidence_id": [prefix] * 8000,
            "source_netcdf_sha256": [netcdf_sha256] * 8000,
            "source_diagnostics_sha256": [diagnostics_sha256] * 8000,
        }
    )
    slope_path = bayes / f"{prefix}_posterior_slope_draws.csv.gz"
    with slope_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                slope_draws.to_csv(handle, index=False)
    config = tmp_path / "model_config.json"
    config.write_text(
        json.dumps(
            {
                "deterministic_artifacts_sha256": expected_artifacts,
                "workflow_tree_sha256": "tree",
            }
        ),
        encoding="utf-8",
    )
    return bayes, config


def test_manual_ppc_overlay_is_persistent_and_hash_bound(tmp_path: Path) -> None:
    bayes, config = _write_complete_evidence(tmp_path)
    evidence = discover_bayesian_evidence(
        [bayes], current_deterministic_config=config
    )
    assert evidence["formal_sampling_candidate"].sum() == 1
    assert evidence.loc[0, "slope_draws_artifact_valid"]
    assert evidence.loc[0, "posterior_slope_median_mV_dec"] == 100.0
    assert evidence.loc[0, "functional_target_supported"]
    assert evidence.loc[0, "functional_target_median_mV_dec"] == 100.0
    assert evidence.loc[0, "functional_target_mean_mV_dec"] == 100.0
    review_path = tmp_path / "study_run" / "ppc_review.csv"
    review = sync_manual_ppc_review(
        review_path, evidence, sample_records={"sr001": 1}
    )
    assert review.loc[0, "ppc_approved"] == ""

    review.loc[0, "ppc_approved"] = "true"
    review.loc[0, "ppc_reviewer"] = "Named Reviewer"
    review.loc[0, "ppc_review_date"] = "2026-07-16"
    review.loc[0, "ppc_review_note"] = "PPC inspected"
    review.to_csv(review_path, index=False)
    preserved = sync_manual_ppc_review(
        review_path, evidence, sample_records={"sr001": 1}
    )
    valid, reason = validate_bound_ppc_review(
        preserved.iloc[0].to_dict(), evidence.iloc[0].to_dict()
    )
    assert valid is True
    assert "bound" in reason

    # Replacing any reviewed artifact invalidates the decision automatically.
    Path(evidence.loc[0, "ppc_plot_path"]).write_bytes(b"changed ppc")
    changed = discover_bayesian_evidence(
        [bayes], current_deterministic_config=config
    )
    reset = sync_manual_ppc_review(
        review_path, changed, sample_records={"sr001": 1}
    )
    assert reset.loc[0, "ppc_approved"] == ""
    assert reset.loc[0, "ppc_reviewer"] == ""


def test_smoke_sampling_can_never_be_formal_evidence(tmp_path: Path) -> None:
    bayes, config = _write_complete_evidence(tmp_path)
    diagnostics_path = next(bayes.glob("*_diagnostics.json"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics.update({"draws_per_chain": 100, "tune_per_chain": 100, "chains": 2})
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    evidence = discover_bayesian_evidence(
        [bayes], current_deterministic_config=config
    )
    assert evidence.loc[0, "sampler_diagnostics_pass"]
    assert not evidence.loc[0, "production_sampling_depth"]
    assert not evidence.loc[0, "formal_sampling_candidate"]
