from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .bibliography import fetch_crossref_bibliography_candidates
from .cardinal_development import run_cardinal_development
from .cardinal_inputs import assert_disjoint_cardinal_paths
from .cardinal_workflow import run_cardinal_classification
from .config import StudyConfig, load_config
from .curation import curate_canonical_registry
from .evaluate import EvaluationConfig, run_nested_evaluation
from .feature_development import run_feature_development
from .feature_association import run_feature_association
from .figures import generate_figures
from .interpretation import InterpretationConfig, build_interpretation_layer
from .integrated_workflow import build_integrated_curation, build_integrated_targets
from .performance_optimization import run_performance_optimization
from .posterior_cardinal_classification import (
    run_posterior_cardinal_classification,
)
from .posterior_cardinal_threeclass_optuna import (
    resolve_threeclass_v4_result_root,
    run_posterior_cardinal_threeclass_optuna,
)
from .posterior_cardinal_contiguous_threeclass import (
    resolve_contiguous_threeclass_classic_result_root,
    resolve_contiguous_threeclass_v4_result_root,
    run_posterior_cardinal_contiguous_threeclass,
)
from .provenance import (
    TREE_HASH_ALGORITHM,
    artifact_manifest,
    create_run_dir,
    directory_tree_summary,
    environment_manifest,
    resolve_latest_run,
    sha256_file,
    update_latest_pointer,
    verify_artifact_manifest,
    write_json,
)
from .properties import (
    build_legacy_property_source_audit,
)
from .report import build_report
from .solvent_properties import reconstruct_solvent_property_registries
from .targets import TargetBuildConfig, build_target_registry
from .temporal import run_historical_temporal_backtest
from .uncertainty import score_functional_target_draws


ANALYSES = (
    ("primary_bayesian_local_line_alkaline", "b_bayesian_local_line_mV_dec", "alkaline"),
    ("sensitivity_bayesian_local_line_all", "b_bayesian_local_line_mV_dec", "all"),
)

ESTIMATED_PROPERTY_PANEL = "context_plus_property_proxy_4p_estimated"
ESTIMATED_PROPERTY_PANEL_ELIGIBLE_STATUS = (
    "eligible_user_approved_with_estimate_indicator"
)
ESTIMATED_PROPERTY_PANEL_ELIGIBLE_STATUSES = frozenset(
    {
        ESTIMATED_PROPERTY_PANEL_ELIGIBLE_STATUS,
        # Kept for reproducibility of pre-Data0718 runs.  New runs use the
        # explicit user-approved status above.
        "eligible_sensitivity_only",
    }
)
SOURCE_BACKED_PROPERTY_PANEL = "context_plus_property_proxy_4p_source_backed"
SOURCE_BACKED_PROPERTY_PANEL_ELIGIBLE_STATUS = "eligible_complete_case_only"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ink-tafel-study",
        description="Rebuild the auditable ink formulation–Tafel study v2.",
    )
    parser.add_argument(
        "stage",
        choices=(
            "curate",
            "refit",
            "evaluate",
            "report",
            "optimize",
            "feature-develop",
            "cardinal-classify",
            "posterior-cardinal-classify",
            "posterior-cardinal-threeclass-optuna",
            "posterior-cardinal-contiguous-threeclass",
            "cardinal-develop",
            "feature-associate",
            "all",
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--source-run",
        type=Path,
        help=(
            "Read-only formal source run for cardinal-classify, "
            "posterior-cardinal-classify, "
            "posterior-cardinal-threeclass-optuna, "
            "posterior-cardinal-contiguous-threeclass, cardinal-develop, or "
            "feature-associate; defaults to paths.active_source_run when "
            "configured, otherwise Data/03_outputs/study/outputs/current_97"
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="Fast structural validation; not a scientific estimate")
    parser.add_argument(
        "--fetch-bibliography",
        action="store_true",
        help="Fetch DOI metadata candidates from Crossref; never auto-assign verified labs",
    )
    parser.add_argument("--crossref-mailto")
    parser.add_argument(
        "--run-bayes",
        action="store_true",
        help="Execute missing selected-model NUTS jobs from the queue; named PPC review remains manual",
    )
    parser.add_argument("--bayes-limit", type=int)
    parser.add_argument("--target", choices=tuple(dict.fromkeys(item[1] for item in ANALYSES)))
    parser.add_argument("--scope", choices=("alkaline", "all"))
    parser.add_argument(
        "--n-jobs",
        type=_positive_int,
        default=1,
        help="Parallel workers for independent ML stages",
    )
    parser.add_argument(
        "--optuna-trials",
        type=_positive_int,
        help=(
            "Smoke-only per-study Optuna trial override for "
            "posterior-cardinal-threeclass-optuna or "
            "posterior-cardinal-contiguous-threeclass; full runs use the "
            "hash-bound contract budget"
        ),
    )
    return parser


def _output_root(config: StudyConfig) -> Path:
    root = config.resolve_path("output_root")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _active_source_run(config: StudyConfig) -> Path:
    value_reader = getattr(config, "value", None)
    configured = (
        value_reader("paths.active_source_run")
        if callable(value_reader)
        else None
    )
    if configured is not None:
        return config.resolve_path("active_source_run").resolve()
    return (config.resolve_path("output_root") / "current_97").resolve()


def _cardinal_source_run(
    config: StudyConfig, args: argparse.Namespace
) -> Path:
    return (
        args.source_run.expanduser().resolve()
        if args.source_run is not None
        else _active_source_run(config)
    )


def _resolve_run(config: StudyConfig, args: argparse.Namespace) -> Path:
    independent_stages = {
        "optimize",
        "feature-develop",
        "cardinal-classify",
        "posterior-cardinal-classify",
        "posterior-cardinal-threeclass-optuna",
        "posterior-cardinal-contiguous-threeclass",
        "cardinal-develop",
        "feature-associate",
    }
    if args.run_dir is not None:
        run = args.run_dir.expanduser().resolve()
        if args.stage in independent_stages:
            if args.stage in {
                "cardinal-classify",
                "posterior-cardinal-classify",
                "posterior-cardinal-threeclass-optuna",
                "posterior-cardinal-contiguous-threeclass",
                "cardinal-develop",
                "feature-associate",
            }:
                source_run = _cardinal_source_run(config, args)
                if not source_run.is_dir():
                    raise FileNotFoundError(
                        f"Formal cardinal source run does not exist: {source_run}"
                    )
                source_run, run = assert_disjoint_cardinal_paths(
                    source_run, run
                )
                if args.stage == "posterior-cardinal-threeclass-optuna":
                    _, run = assert_disjoint_cardinal_paths(
                        resolve_threeclass_v4_result_root(config),
                        run,
                    )
                if (
                    args.stage
                    == "posterior-cardinal-contiguous-threeclass"
                ):
                    _, run = assert_disjoint_cardinal_paths(
                        resolve_contiguous_threeclass_v4_result_root(config),
                        run,
                    )
                    _, run = assert_disjoint_cardinal_paths(
                        resolve_contiguous_threeclass_classic_result_root(
                            config
                        ),
                        run,
                    )
                if not args.smoke:
                    workbook_root = (
                        config.repository_root / "Data" / "04_results" / "RESULTS" / "05_WORKBOOKS"
                    ).resolve()
                    if workbook_root not in run.parents:
                        raise ValueError(
                            "Full cardinal ML runs must be placed "
                            f"under {workbook_root}"
                        )
            formal_run = _active_source_run(config)
            if run == formal_run or formal_run in run.parents:
                stage_label = f"independent {args.stage}"
                raise ValueError(
                    f"The {stage_label} stage cannot use or write inside the "
                    f"registered formal run: {formal_run}"
                )
            if run.exists() and not run.is_dir():
                raise NotADirectoryError(run)
            if run.exists() and any(run.iterdir()):
                stage_label = {
                    "optimize": "Optimization",
                    "feature-develop": "Feature-development",
                    "cardinal-classify": "Cardinal-classification",
                    "posterior-cardinal-classify": (
                        "Posterior-cardinal-classification"
                    ),
                    "posterior-cardinal-threeclass-optuna": (
                        "Posterior-cardinal-three-class Optuna"
                    ),
                    "posterior-cardinal-contiguous-threeclass": (
                        "Posterior-cardinal contiguous three-class"
                    ),
                    "cardinal-develop": "Cardinal-development",
                    "feature-associate": "Feature-association",
                }[args.stage]
                raise FileExistsError(
                    f"{stage_label} run directories must be new or empty so stale smoke/full "
                    f"artifacts cannot be mixed: {run}"
                )
            run.mkdir(parents=True, exist_ok=True)
            (run / "provenance").mkdir(exist_ok=True)
        elif args.stage in {"curate", "all"}:
            run.mkdir(parents=True, exist_ok=True)
            for name in ("data", "targets", "evaluation", "figures", "report", "provenance"):
                (run / name).mkdir(exist_ok=True)
        elif not run.is_dir():
            raise FileNotFoundError(run)
        return run
    if args.stage in independent_stages:
        raise ValueError(
            f"The independent {args.stage} stage requires --run-dir so it cannot overwrite "
            "the active formal run"
        )
    output_root = _output_root(config)
    if args.stage in {"curate", "all"}:
        run = create_run_dir(output_root)
        update_latest_pointer(output_root, run)
        return run
    return resolve_latest_run(output_root)


def _input_manifest(config: StudyConfig) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, value in config.section("paths").items():
        if name in {"output_root", "active_source_run"}:
            continue
        try:
            path = config.resolve_path(name)
        except (KeyError, TypeError):
            continue
        if path.is_file():
            rows[name] = {
                "path": str(path),
                "kind": "file",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        elif path.is_dir():
            rows[name] = {"path": str(path), **directory_tree_summary(path)}
        else:
            rows[name] = {"path": str(path), "status": "missing"}

    hash_policy = config.section("input_hashes") if "input_hashes" in config.raw else {}
    expected = hash_policy.get("expected", {})
    if expected and not isinstance(expected, dict):
        raise TypeError("Configuration input_hashes.expected must be a mapping")
    expected = {str(name): str(value) for name, value in expected.items()}
    configured_inputs = {
        name
        for name in config.section("paths")
        if name not in {"output_root", "active_source_run"}
    }
    missing_registrations = sorted(configured_inputs - set(expected))
    unexpected_registrations = sorted(set(expected) - configured_inputs)
    mismatches: list[str] = []
    missing_inputs: list[str] = []
    for name, row in rows.items():
        if name not in expected:
            continue
        actual = row.get("sha256") if row.get("kind") == "file" else row.get("tree_sha256")
        row["expected_sha256"] = expected[name]
        row["hash_matches_expected"] = bool(actual == expected[name])
        if row.get("status") == "missing":
            missing_inputs.append(name)
        elif not row["hash_matches_expected"]:
            mismatches.append(name)

    enforce = bool(hash_policy.get("enforce", False))
    configured_tree_algorithm = hash_policy.get("tree_hash_algorithm")
    algorithm_matches = bool(configured_tree_algorithm == TREE_HASH_ALGORITHM)
    passed = bool(
        expected
        and algorithm_matches
        and not missing_registrations
        and not unexpected_registrations
        and not mismatches
        and not missing_inputs
    )
    rows["_hash_validation"] = {
        "status": (
            "pass"
            if passed
            else "fail"
            if enforce
            else "not_enforced"
        ),
        "enforce": enforce,
        "tree_hash_algorithm": configured_tree_algorithm,
        "tree_hash_algorithm_matches": algorithm_matches,
        "registered_inputs": sorted(expected),
        "missing_registrations": missing_registrations,
        "unexpected_registrations": unexpected_registrations,
        "missing_inputs": missing_inputs,
        "hash_mismatches": mismatches,
    }
    return rows


def _initialize_provenance(
    config: StudyConfig,
    run_dir: Path,
    *,
    smoke: bool,
    scientific_conclusions_allowed: bool | None = None,
    analysis_role: str | None = None,
) -> None:
    provenance = run_dir / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    resolved = dict(config.raw)
    resolved["resolved_paths"] = {
        name: str(config.resolve_path(name))
        for name in config.section("paths")
    }
    (provenance / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(provenance / "environment_manifest.json", environment_manifest(config.repository_root))
    inputs = _input_manifest(config)
    write_json(provenance / "input_manifest.json", inputs)
    validation = inputs["_hash_validation"]
    if validation["enforce"] and validation["status"] != "pass":
        raise ValueError(
            "Configured input hash validation failed: "
            + json.dumps(validation, ensure_ascii=False, sort_keys=True)
        )
    code_paths = sorted((config.project_root / "src" / "ink_tafel_study").glob("*.py"))
    code_paths.extend(
        path
        for path in (
            config.project_root / "run.sh",
            config.project_root / "pyproject.toml",
            config.path,
            config.project_root / "docs" / "study_protocol.md",
        )
        if path.is_file()
    )
    write_json(
        provenance / "code_manifest.json",
        {
            "files": [
                {
                    "path": str(path.resolve().relative_to(config.repository_root)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in code_paths
            ]
        },
    )
    write_json(
        provenance / "run_mode.json",
        {
            "smoke": smoke,
            "scientific_conclusions_allowed": (
                not smoke
                if scientific_conclusions_allowed is None
                else bool(scientific_conclusions_allowed)
            ),
            "analysis_role": analysis_role or "configured_stage_default",
            "registered_protocol": str(config.project_root / "docs" / "study_protocol.md"),
        },
    )


def _stage_status(run_dir: Path, stage: str, status: str, **details: Any) -> None:
    write_json(
        run_dir / "provenance" / f"stage_{stage}.json",
        {"stage": stage, "status": status, **details},
    )


def _evaluation_was_smoke(run_dir: Path) -> bool:
    status_path = run_dir / "provenance" / "stage_evaluate.json"
    if not status_path.is_file():
        return False
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    return payload.get("status") == "complete_smoke"


def _invalidate_completion_claims(run_dir: Path, requested_stage: str) -> None:
    """Remove stale downstream completion evidence before mutating a run."""

    (run_dir / "artifact_manifest.json").unlink(missing_ok=True)
    if requested_stage == "optimize":
        (run_dir / "provenance" / "stage_optimize.json").unlink(missing_ok=True)
        return
    if requested_stage == "feature-develop":
        (run_dir / "provenance" / "stage_feature_develop.json").unlink(
            missing_ok=True
        )
        return
    if requested_stage == "cardinal-classify":
        (
            run_dir / "provenance" / "stage_cardinal_classify.json"
        ).unlink(missing_ok=True)
        return
    if requested_stage == "posterior-cardinal-classify":
        (
            run_dir
            / "provenance"
            / "stage_posterior_cardinal_classify.json"
        ).unlink(missing_ok=True)
        return
    if requested_stage == "posterior-cardinal-threeclass-optuna":
        (
            run_dir
            / "provenance"
            / "stage_posterior_cardinal_threeclass_optuna.json"
        ).unlink(missing_ok=True)
        return
    if requested_stage == "posterior-cardinal-contiguous-threeclass":
        (
            run_dir
            / "provenance"
            / "stage_posterior_cardinal_contiguous_threeclass.json"
        ).unlink(missing_ok=True)
        return
    if requested_stage == "cardinal-develop":
        (
            run_dir / "provenance" / "stage_cardinal_develop.json"
        ).unlink(missing_ok=True)
        return
    if requested_stage == "feature-associate":
        (
            run_dir / "provenance" / "stage_feature_associate.json"
        ).unlink(missing_ok=True)
        return
    order = ("curate", "refit", "evaluate", "report")
    first = 0 if requested_stage == "all" else order.index(requested_stage)
    for stage in order[first:]:
        (run_dir / "provenance" / f"stage_{stage}.json").unlink(missing_ok=True)
    (run_dir / "provenance" / "stage_all.json").unlink(missing_ok=True)


def _require_formal_bayesian_approval(run_dir: Path, *, smoke: bool) -> None:
    """Block registered evaluation until every fit-eligible curve is approved."""

    if smoke:
        return
    qa_path = run_dir / "targets" / "target_qa.json"
    if not qa_path.is_file():
        raise FileNotFoundError(
            "Target QA is missing; registered evaluation requires a complete target audit"
        )
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    fit_eligible = int(
        qa.get("primary_eligible_all_scopes", qa.get("n_fit_eligible_curves", -1))
    )
    approved = int(
        qa.get("n_formal_ppc_approved", qa.get("n_formal_nuts_ppc_approved", -1))
    )
    if fit_eligible < 0 or approved < 0:
        raise RuntimeError(
            "Target QA lacks formal Bayesian approval counts; rebuild the target audit"
        )
    if approved != fit_eligible:
        raise RuntimeError(
            "Registered evaluation is blocked until calibrated Bayesian evidence and named PPC "
            f"review are complete for every fit-eligible curve ({approved}/{fit_eligible} approved). "
            "Use --smoke only for structural validation."
        )
    policy = qa.get("primary_target_policy", {})
    target_name = qa.get("primary_target", policy.get("target_name"))
    policy_is_current = bool(
        target_name == "b_bayesian_local_line_mV_dec"
        and (
            policy.get("common_estimand") == "robust Bayesian local-linear apparent slope"
            or policy.get("policy_id")
            == "bayesian_local_line_eta015_strict_then_pava_fallback_v2"
        )
    )
    if not policy_is_current:
        raise RuntimeError(
            "Registered evaluation is blocked because target QA predates the calibrated "
            "Bayesian local-line primary-target policy; rebuild canonical evidence and refit"
        )
    if not bool(qa.get("formal_evaluation_ready", False)):
        blockers = qa.get("formal_evaluation_blockers", ["unspecified_target_provenance_gap"])
        raise RuntimeError(
            "Registered evaluation remains blocked although curve-level review counts "
            "are complete, because one or more eligible targets are not supported by "
            f"an approved posterior derivation. Blockers: {blockers}"
        )


def _refresh_curation_manifest(data_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name == "curation_manifest.csv":
            continue
        if path.suffix == ".csv":
            count = len(pd.read_csv(path))
        elif path.suffix == ".parquet":
            count = len(pd.read_parquet(path))
        else:
            count = 1
        rows.append(
            {
                "artifact": path.stem,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": count,
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "curation_manifest.csv", index=False)


def _curate(config: StudyConfig, run_dir: Path, args: argparse.Namespace) -> None:
    if config.value("curation.source_mode") == "canonical_97":
        outputs = build_integrated_curation(
            dataset_dir=config.resolve_path("canonical_dataset", must_exist=True),
            output_dir=run_dir / "data",
            panel_names=tuple(config.value("feature_panels.enabled", [])),
        )
        if args.fetch_bibliography:
            fetch_crossref_bibliography_candidates(
                outputs["sources"],
                run_dir / "data",
                mailto=args.crossref_mailto,
            )
        qa = json.loads(outputs["curation_qa"].read_text(encoding="utf-8"))
        _refresh_curation_manifest(run_dir / "data")
        if qa.get("status") != "pass":
            raise RuntimeError(f"Data0718 curation acceptance tests failed: {qa}")
        _stage_status(run_dir, "curate", "complete", qa=qa)
        return

    outputs = curate_canonical_registry(
        raw_workbook=config.resolve_path("raw_workbook", must_exist=True),
        property_workbook=config.resolve_path("property_candidate_workbook", must_exist=True),
        refit_run=config.resolve_path("refit_run", must_exist=True),
        output_dir=run_dir / "data",
        legacy_metadata_table=config.resolve_path("legacy_metadata_table", must_exist=True),
    )
    records = pd.read_csv(outputs["records"])
    property_source_audit, property_source_qa = build_legacy_property_source_audit(
        outputs["property_registry"],
        config.resolve_path("property_value_source_map", must_exist=True),
        config.resolve_path("property_sources", must_exist=True),
        run_dir / "data",
    )
    enriched_records, _, _, property_qa = reconstruct_solvent_property_registries(
        records,
        config.resolve_path("authoritative_property_measurements", must_exist=True),
        config.resolve_path("property_component_constants", must_exist=True),
        config.resolve_path("property_sources", must_exist=True),
        config.resolve_path("property_source_files", must_exist=True),
        config.resolve_path("property_dielectric_validation", must_exist=True),
        run_dir / "data",
        legacy_audit=property_source_audit,
        target_temperature_K=float(
            config.value("property_reconstruction.target_temperature_K", 298.15)
        ),
        exact_temperature_tolerance_K=float(
            config.value("property_reconstruction.exact_temperature_tolerance_K", 0.05)
        ),
        expected_records=int(config.value("curation.expected_records", 97)),
        expected_compositions=int(
            config.value("property_reconstruction.expected_compositions", 33)
        ),
        expected_dielectric_strict_source_backed_values=int(
            config.value(
                "property_reconstruction.expected_dielectric_strict_source_backed_values",
                66,
            )
        ),
        expected_dielectric_A_or_B_values=int(
            config.value(
                "property_reconstruction.expected_dielectric_A_or_B_values", 40
            )
        ),
        expected_dielectric_C_values=int(
            config.value("property_reconstruction.expected_dielectric_C_values", 26)
        ),
        expected_dielectric_D_estimates=int(
            config.value(
                "property_reconstruction.expected_dielectric_D_estimates", 31
            )
        ),
        water_dielectric_constant=float(
            config.value("property_reconstruction.dielectric.water_static", 78.375)
        ),
        ipa_dielectric_constant=float(
            config.value("property_reconstruction.dielectric.isopropanol_static", 19.387)
        ),
        ipa_validation_mape_threshold_percent=float(
            config.value("property_reconstruction.dielectric.validation_mape_percent", 5.0)
        ),
        ipa_validation_max_error_threshold_percent=float(
            config.value("property_reconstruction.dielectric.validation_max_error_percent", 10.0)
        ),
    )
    enriched_records.to_csv(outputs["records"], index=False)

    workbook_status_path = run_dir / "data" / "solvent_property_review_build_status.json"
    workbook_builder = Path(__file__).resolve().parents[2] / "scripts" / "build_solvent_property_review.mjs"
    node = shutil.which("node")
    workbook_status: dict[str, Any]
    if node and workbook_builder.is_file():
        completed = subprocess.run(
            [
                node,
                str(workbook_builder),
                "--data-dir",
                str((run_dir / "data").resolve()),
                "--preview-dir",
                str((run_dir / "data" / "solvent_property_review_previews").resolve()),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        runtime_unavailable = "@oai/artifact-tool is unavailable" in completed.stderr
        workbook_status = {
            "status": (
                "generated_and_rendered"
                if completed.returncode == 0
                else "blocked_managed_artifact_runtime_unavailable"
                if runtime_unavailable
                else "failed_builder_validation"
            ),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "artifact_tool_only": True,
        }
    else:
        workbook_status = {
            "status": "blocked_managed_artifact_runtime_unavailable",
            "reason": "Node/@oai/artifact-tool supplied by load_workspace_dependencies is unavailable",
            "artifact_tool_only": True,
            "fallback_spreadsheet_library_used": False,
        }
    workbook_status_path.write_text(
        json.dumps(workbook_status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.fetch_bibliography:
        fetch_crossref_bibliography_candidates(
            outputs["sources"],
            run_dir / "data",
            mailto=args.crossref_mailto,
        )
    qa = json.loads(outputs["curation_qa"].read_text(encoding="utf-8"))
    qa["property_source_reconstruction"] = property_source_qa
    qa["authoritative_property_proxy"] = property_qa
    qa["solvent_property_review_workbook"] = workbook_status
    property_acceptance_passed = property_qa.get("status") == (
        "ready_with_registered_estimate_gate"
    )
    source_hash_acceptance_passed = bool(
        property_qa.get("full_source_file_hash_acceptance_passed", False)
    )
    workbook_acceptance_passed = (
        workbook_status.get("status") == "generated_and_rendered"
    )
    qa["solvent_property_data_acceptance_passed"] = property_acceptance_passed
    qa["solvent_property_source_hash_acceptance_passed"] = (
        source_hash_acceptance_passed
    )
    qa["solvent_property_workbook_acceptance_passed"] = workbook_acceptance_passed
    qa["solvent_property_full_acceptance_passed"] = (
        property_acceptance_passed
        and source_hash_acceptance_passed
        and workbook_acceptance_passed
    )
    qa["solvent_property_acceptance_blockers"] = [
        blocker
        for blocker, passed in (
            ("registered_data_coverage_or_evidence_gate", property_acceptance_passed),
            ("eligible_primary_source_file_hashes_pending", source_hash_acceptance_passed),
            ("artifact_tool_workbook_export_and_render", workbook_acceptance_passed),
        )
        if not passed
    ]
    qa["solvent_property_acceptance_passed"] = property_acceptance_passed
    if not property_acceptance_passed:
        qa["status"] = "fail"
        qa["failure_reason"] = (
            "registered solvent-property coverage acceptance failed: "
            f"{property_qa.get('status', 'unknown')}"
        )
    outputs["curation_qa"].write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _refresh_curation_manifest(run_dir / "data")
    if qa.get("status") != "pass":
        raise RuntimeError(f"Curation acceptance tests failed: {qa}")
    _stage_status(run_dir, "curate", "complete", qa=qa)


def _target_build_config(config: StudyConfig) -> TargetBuildConfig:
    """Bind every dataset-dependent target parity value to study.yaml."""

    curation = config.section("curation")
    targets = config.section("targets")
    required_curation = ("expected_records", "expected_curves")
    required_targets = (
        "fixed_current_mA_cm2",
        "minimum_draw_support_fraction",
        "inverse_grid_points",
        "expected_primary_eligible",
        "expected_low_branch_eligible",
        "expected_single_regime_eligible",
    )
    missing = [
        f"curation.{key}" for key in required_curation if key not in curation
    ]
    missing.extend(
        f"targets.{key}" for key in required_targets if key not in targets
    )
    if missing:
        raise KeyError(
            "Target build configuration is missing required registered fields: "
            + ", ".join(missing)
        )
    result = TargetBuildConfig(
        expected_curve_count=int(curation["expected_curves"]),
        expected_record_count=int(curation["expected_records"]),
        expected_primary_eligible=int(targets["expected_primary_eligible"]),
        expected_low_branch_eligible=int(targets["expected_low_branch_eligible"]),
        expected_single_regime_eligible=int(
            targets["expected_single_regime_eligible"]
        ),
        fixed_current_density_mA_cm2=float(targets["fixed_current_mA_cm2"]),
        minimum_draw_support_fraction=float(
            targets["minimum_draw_support_fraction"]
        ),
        inverse_grid_points=int(targets["inverse_grid_points"]),
    )
    result.validate()
    return result


def _refit(config: StudyConfig, run_dir: Path, args: argparse.Namespace) -> None:
    records = run_dir / "data" / "records.csv"
    if not records.is_file():
        raise FileNotFoundError("Canonical records are missing; run curate first")
    if config.value("curation.source_mode") == "canonical_97":
        if args.run_bayes:
            raise ValueError(
                "All 97 Bayesian local-line targets already have calibrated "
                "posterior draws; Records 23 and 74 use the registered "
                "quantization-aware fallback. "
                "Only named PPC review is required, not resampling."
            )
        outputs = build_integrated_targets(
            dataset_dir=config.resolve_path("canonical_dataset", must_exist=True),
            records_path=records,
            ppc_review_path=config.resolve_path(
                "ppc_review_registry", must_exist=True
            ),
            output_dir=run_dir / "targets",
            expected_primary_eligible=int(
                config.value("targets.expected_primary_eligible")
            ),
            expected_primary_alkaline_eligible=int(
                config.value("targets.expected_primary_alkaline_eligible")
            ),
            expected_low_branch_eligible=int(
                config.value("targets.expected_low_branch_eligible")
            ),
            expected_single_regime_eligible=int(
                config.value("targets.expected_single_regime_eligible")
            ),
        )
        qa = json.loads(outputs["target_qa"].read_text(encoding="utf-8"))
        _stage_status(
            run_dir,
            "refit",
            str(qa.get("status", "complete_with_formal_bayesian_review_pending")),
            qa=qa,
            bayesian_batch={
                "executed": False,
                "reason": "hash-bound existing NUTS evidence reused by curve_id",
            },
        )
        return

    batch_dir = run_dir / "targets" / "bayesian_batch"
    target_config = _target_build_config(config)

    def rebuild_target_audit() -> dict[str, Path]:
        return build_target_registry(
            records_path=records,
            refit_run=config.resolve_path("refit_run", must_exist=True),
            bayesian_table=config.resolve_path(
                "bayesian_single_regime_table", must_exist=True
            ),
            bayesian_manifest=config.resolve_path(
                "bayesian_single_regime_manifest", must_exist=True
            ),
            current_conditioned_dir=config.resolve_path(
                "current_conditioned_dir", must_exist=True
            ),
            bayesian_batch_dir=batch_dir,
            ppc_review_overlay=batch_dir / "ppc_review.csv",
            output_dir=run_dir / "targets",
            config=target_config,
        )

    # Rebuild before queue selection so prior batch artifacts and any manual
    # review edits in this study run are incorporated on every refit rerun.
    outputs = rebuild_target_audit()
    bayes_result: dict[str, Any] = {
        "executed": False,
        "reason": "Use --run-bayes to execute missing selected-model NUTS jobs; named PPC review cannot be automated.",
    }
    if args.run_bayes:
        from .refit_queue import run_missing_bayesian_jobs

        bayes_result = run_missing_bayesian_jobs(
            deterministic_run=config.resolve_path("refit_run", must_exist=True),
            queue_path=outputs["bayesian_pending_queue"],
            output_dir=batch_dir,
            refit_project=config.repository_root / "code" / "Tafel_refitting",
            limit=args.bayes_limit,
            smoke=args.smoke,
        )
        if int(bayes_result.get("failed_jobs", 0)):
            raise RuntimeError(
                "One or more selected-model NUTS jobs failed; inspect "
                f"{run_dir / 'targets' / 'bayesian_batch'}"
            )
        # Sampling creates new evidence inside the isolated copy.  Rebuild the
        # registry immediately so QA/queue counts and posterior-supported
        # target values never lag one invocation behind the batch.
        outputs = rebuild_target_audit()
    qa = json.loads(outputs["target_qa"].read_text(encoding="utf-8"))
    _stage_status(
        run_dir,
        "refit",
        "complete_with_formal_bayesian_review_pending",
        qa=qa,
        bayesian_batch=bayes_result,
    )


def _evaluation_config(config: StudyConfig, *, smoke: bool) -> EvaluationConfig:
    validation = config.section("validation")
    models = config.section("models")
    panel_config = config.section("feature_panels")
    interpretation = config.section("interpretation")
    panel_names = tuple(panel_config["enabled"])
    model_names = tuple(models["enabled"])
    if smoke:
        panel_names = (
            "dummy",
            "context",
            "context_plus_ink_amounts",
            "context_plus_ink",
            "context_plus_ink_process",
            "context_plus_property_proxy_core",
            "context_plus_property_proxy_4p_source_backed",
            "context_plus_property_proxy_4p_estimated",
        )
        model_names = ("dummy_mean", "dummy_median", "bayesian_ridge", "gower_knn")
    return EvaluationConfig(
        outer_folds=int(validation["outer_folds"]),
        outer_repeats=int(validation["outer_repeats"]),
        inner_folds=int(validation["inner_folds"]),
        random_seed=int(validation["seeds"][0]),
        outer_seeds=tuple(int(value) for value in validation["seeds"]),
        bootstrap_replicates=int(validation["group_bootstrap_repetitions"]),
        panel_names=panel_names,
        model_names=model_names,
        gower_neighbors=tuple(int(value) for value in models["gower_knn"]["k_grid"]),
        gower_distance_powers=(1.0, 2.0),
        interpretation_enabled=not smoke,
        importance_panel=str(interpretation["importance_panel"]),
        importance_permutation_repeats=int(interpretation["permutation_repeats"]),
        require_complete_panels=True,
        n_jobs=1,
    )


def _apply_property_panel_gate(
    settings: EvaluationConfig,
    property_proxy_qa: dict[str, Any],
) -> tuple[EvaluationConfig, dict[str, Any]]:
    """Apply property-panel eligibility as explicit fail-closed whitelists."""

    estimated_status = str(
        property_proxy_qa.get("estimated_panel_model_use_status", "not_registered")
    )
    source_backed_4p_status = str(
        property_proxy_qa.get(
            "source_backed_4p_panel_model_use_status", "not_registered"
        )
    )
    allowed = {
        "context_plus_property_proxy_core": True,
        SOURCE_BACKED_PROPERTY_PANEL: (
            source_backed_4p_status
            == SOURCE_BACKED_PROPERTY_PANEL_ELIGIBLE_STATUS
        ),
        ESTIMATED_PROPERTY_PANEL: (
            estimated_status in ESTIMATED_PROPERTY_PANEL_ELIGIBLE_STATUSES
        ),
    }
    gated_settings = replace(
        settings,
        panel_names=tuple(
            panel
            for panel in settings.panel_names
            if panel not in allowed or allowed[panel]
        ),
    )
    gate = {
        "policy": "fail_closed_explicit_status_whitelist",
        "reported_status": {
            "context_plus_property_proxy_core": "eligible",
            SOURCE_BACKED_PROPERTY_PANEL: source_backed_4p_status,
            ESTIMATED_PROPERTY_PANEL: estimated_status,
        },
        "required_status": {
            "context_plus_property_proxy_core": "eligible",
            SOURCE_BACKED_PROPERTY_PANEL: SOURCE_BACKED_PROPERTY_PANEL_ELIGIBLE_STATUS,
            ESTIMATED_PROPERTY_PANEL: sorted(
                ESTIMATED_PROPERTY_PANEL_ELIGIBLE_STATUSES
            ),
        },
        "decision": {
            panel: "included" if is_allowed else "excluded"
            for panel, is_allowed in allowed.items()
        },
    }
    return gated_settings, gate


def _interpretation_config(config: StudyConfig) -> InterpretationConfig:
    settings = config.section("interpretation")
    return InterpretationConfig(
        importance_panel=str(settings["importance_panel"]),
        permutation_repeats=int(settings["permutation_repeats"]),
        required_scope=str(settings["required_scope"]),
        required_targets=tuple(str(value) for value in settings["required_targets"]),
        required_model_families=tuple(
            str(value) for value in settings["required_model_families"]
        ),
        positive_split_fraction_minimum=float(
            settings["positive_split_fraction_minimum"]
        ),
        minimum_outer_splits_per_cell=int(settings["minimum_outer_splits_per_cell"]),
        minimum_records_per_stratum=int(settings["minimum_records_per_stratum"]),
        minimum_strata=int(settings["minimum_strata"]),
    )


def _selected_analyses(args: argparse.Namespace) -> tuple[tuple[str, str, str], ...]:
    analyses = ANALYSES
    if args.target:
        analyses = tuple(item for item in analyses if item[1] == args.target)
    if args.scope:
        analyses = tuple(item for item in analyses if item[2] == args.scope)
    if args.smoke and not args.target and not args.scope:
        analyses = (ANALYSES[0],)
    if not analyses:
        raise ValueError("No registered target/scope analysis matches the requested filters")
    return analyses


def _evaluate(config: StudyConfig, run_dir: Path, args: argparse.Namespace) -> None:
    records = run_dir / "data" / "records.csv"
    targets = run_dir / "targets" / "target_registry.csv"
    if not records.is_file() or not targets.is_file():
        raise FileNotFoundError("Canonical records/targets are missing; run curate and refit first")
    _require_formal_bayesian_approval(run_dir, smoke=args.smoke)
    settings = _evaluation_config(config, smoke=args.smoke)
    property_proxy_qa = json.loads(
        (run_dir / "data" / "property_proxy_qa.json").read_text(encoding="utf-8")
    )
    settings, property_gate = _apply_property_panel_gate(settings, property_proxy_qa)
    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "evaluation" / "property_panel_gate.json", property_gate)
    completed: list[dict[str, Any]] = []
    for analysis_id, target, scope in _selected_analyses(args):
        destination = run_dir / "evaluation" / analysis_id
        result = run_nested_evaluation(
            records,
            targets,
            destination,
            target_col=target,
            target_name=target,
            scope=scope,
            config=settings,
            smoke=args.smoke,
        )
        completed.append(
            {
                "analysis_id": analysis_id,
                "target": target,
                "scope": scope,
                "oof_rows": len(result.oof_predictions),
                "split_rows": len(result.split_manifest),
            }
        )

    primary_oof = run_dir / "evaluation" / "primary_low_alkaline" / "oof_aggregated.csv"
    posterior_draws = (
        run_dir / "targets" / "posterior_slope_draws.csv.gz"
    )
    if not args.smoke and primary_oof.is_file() and posterior_draws.is_file():
        score_functional_target_draws(
            primary_oof,
            posterior_draws,
            run_dir / "evaluation" / "target_uncertainty",
            target_name="b_bayesian_local_line_mV_dec",
            max_draws=int(config.value("validation.target_draw_repetitions", 100)),
            random_seed=int(config.value("validation.seeds", [42])[0]),
        )
    if not args.smoke and (not args.target or args.target == "b_bayesian_local_line_mV_dec") and (not args.scope or args.scope == "alkaline"):
        run_historical_temporal_backtest(
            records,
            targets,
            run_dir / "evaluation" / "historical_2024plus_backtest",
            target_name="b_bayesian_local_line_mV_dec",
            cutoff_year=int(config.value("scope.temporal_backtest_year_min", 2024)),
            inner_folds=int(config.value("validation.inner_folds", 4)),
            random_seed=int(config.value("validation.seeds", [42])[0]),
        )
    interpretation_status = build_interpretation_layer(
        run_dir / "evaluation",
        records,
        output_dir=run_dir / "evaluation" / "interpretation",
        smoke=args.smoke,
        config=_interpretation_config(config),
    )
    _stage_status(
        run_dir,
        "evaluate",
        "complete_smoke" if args.smoke else "complete",
        config=asdict(settings),
        analyses=completed,
        interpretation=interpretation_status,
    )


def _report(config: StudyConfig, run_dir: Path, args: argparse.Namespace) -> None:
    effective_smoke = bool(args.smoke or _evaluation_was_smoke(run_dir))
    formats = tuple(config.value("reporting.figure_formats", ["png", "pdf"]))
    dpi = int(config.value("reporting.figure_dpi", 300))
    records = run_dir / "data" / "records.csv"
    if records.is_file():
        build_interpretation_layer(
            run_dir / "evaluation",
            records,
            output_dir=run_dir / "evaluation" / "interpretation",
            smoke=effective_smoke,
            config=_interpretation_config(config),
        )
    figures = generate_figures(run_dir, formats=formats, dpi=dpi, smoke=effective_smoke)
    report, validation = build_report(run_dir, smoke=effective_smoke)
    _stage_status(
        run_dir,
        "report",
        "complete_smoke" if effective_smoke else "complete",
        smoke=effective_smoke,
        report=str(report),
        validation_report=str(validation),
        generated_figures=int((figures["status"] == "generated").sum()),
    )


def _optimize(config: StudyConfig, run_dir: Path, args: argparse.Namespace) -> None:
    settings = dict(config.section("performance_optimization"))
    settings["expected_canonical_tree_sha256"] = config.value(
        "input_hashes.expected.canonical_dataset"
    )
    result = run_performance_optimization(
        run_dir,
        project_root=config.repository_root,
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        settings=settings,
    )
    _stage_status(
        run_dir,
        "optimize",
        "complete_smoke" if args.smoke else "complete",
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        result=result,
    )


def _feature_develop(
    config: StudyConfig, run_dir: Path, args: argparse.Namespace
) -> None:
    result = run_feature_development(
        config,
        run_dir,
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
    )
    _stage_status(
        run_dir,
        "feature_develop",
        "complete_smoke" if args.smoke else "complete",
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        result=result,
    )


def _cardinal_classify(
    config: StudyConfig, run_dir: Path, args: argparse.Namespace
) -> None:
    source_run = _cardinal_source_run(config, args)
    result = run_cardinal_classification(
        config,
        run_dir,
        source_run=source_run,
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
    )
    _stage_status(
        run_dir,
        "cardinal_classify",
        "complete_smoke" if args.smoke else "complete",
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        source_run=str(source_run),
        result=result,
    )


def _cardinal_develop(
    config: StudyConfig, run_dir: Path, args: argparse.Namespace
) -> None:
    source_run = _cardinal_source_run(config, args)
    result = run_cardinal_development(
        config,
        run_dir,
        source_run=source_run,
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
    )
    _stage_status(
        run_dir,
        "cardinal_develop",
        "complete_smoke" if args.smoke else "complete",
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        source_run=str(source_run),
        result=result,
    )


def _posterior_cardinal_classify(
    config: StudyConfig, run_dir: Path, args: argparse.Namespace
) -> None:
    if args.scope is not None:
        raise ValueError(
            "posterior-cardinal-classify is fixed to all 97 records; "
            "do not pass --scope"
        )
    source_run = _cardinal_source_run(config, args)
    result = run_posterior_cardinal_classification(
        config,
        run_dir,
        source_run=source_run,
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
    )
    _stage_status(
        run_dir,
        "posterior_cardinal_classify",
        "complete_smoke" if args.smoke else "complete",
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        source_run=str(source_run),
        result=result,
    )


def _posterior_cardinal_threeclass_optuna(
    config: StudyConfig, run_dir: Path, args: argparse.Namespace
) -> None:
    if args.scope is not None:
        raise ValueError(
            "posterior-cardinal-threeclass-optuna is fixed to all 97 records; "
            "do not pass --scope"
        )
    source_run = _cardinal_source_run(config, args)
    result = run_posterior_cardinal_threeclass_optuna(
        config,
        run_dir,
        source_run=source_run,
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        optuna_trials=args.optuna_trials,
    )
    _stage_status(
        run_dir,
        "posterior_cardinal_threeclass_optuna",
        "complete_smoke" if args.smoke else "complete",
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        optuna_trials=args.optuna_trials,
        source_run=str(source_run),
        result=result,
    )


def _posterior_cardinal_contiguous_threeclass(
    config: StudyConfig, run_dir: Path, args: argparse.Namespace
) -> None:
    if args.scope is not None:
        raise ValueError(
            "posterior-cardinal-contiguous-threeclass is fixed to all 97 "
            "records; do not pass --scope"
        )
    source_run = _cardinal_source_run(config, args)
    result = run_posterior_cardinal_contiguous_threeclass(
        config,
        run_dir,
        source_run=source_run,
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        optuna_trials=args.optuna_trials,
    )
    _stage_status(
        run_dir,
        "posterior_cardinal_contiguous_threeclass",
        "complete_smoke" if args.smoke else "complete_exploratory",
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        optuna_trials=args.optuna_trials,
        source_run=str(source_run),
        result=result,
    )


def _feature_associate(
    config: StudyConfig, run_dir: Path, args: argparse.Namespace
) -> None:
    if args.scope is not None:
        raise ValueError(
            "feature-associate is fixed to all 97 records; do not pass "
            "--scope"
        )
    source_run = _cardinal_source_run(config, args)
    result = run_feature_association(
        config,
        run_dir,
        source_run=source_run,
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
    )
    _stage_status(
        run_dir,
        "feature_associate",
        "complete_smoke" if args.smoke else "complete",
        smoke=bool(args.smoke),
        n_jobs=int(args.n_jobs),
        source_run=str(source_run),
        result=result,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.stage
        in {
            "feature-associate",
            "posterior-cardinal-classify",
            "posterior-cardinal-threeclass-optuna",
            "posterior-cardinal-contiguous-threeclass",
        }
        and args.scope is not None
    ):
        raise ValueError(
            f"{args.stage} is fixed to all 97 records; do not pass --scope"
        )
    config = load_config(args.config)
    run_dir = _resolve_run(config, args)
    effective_smoke = bool(
        args.smoke or (args.stage == "report" and _evaluation_was_smoke(run_dir))
    )
    _invalidate_completion_claims(run_dir, args.stage)
    active_stage: str | None = None
    try:
        is_threeclass_exploration = args.stage in {
            "posterior-cardinal-threeclass-optuna",
            "posterior-cardinal-contiguous-threeclass",
        }
        if is_threeclass_exploration:
            _initialize_provenance(
                config,
                run_dir,
                smoke=effective_smoke,
                scientific_conclusions_allowed=False,
                analysis_role="posthoc_exploratory_sensitivity",
            )
        else:
            _initialize_provenance(
                config,
                run_dir,
                smoke=effective_smoke,
            )
        if args.stage in {"curate", "all"}:
            active_stage = "curate"
            _curate(config, run_dir, args)
        if args.stage in {"refit", "all"}:
            active_stage = "refit"
            _refit(config, run_dir, args)
        if args.stage in {"evaluate", "all"}:
            active_stage = "evaluate"
            _evaluate(config, run_dir, args)
        if args.stage in {"report", "all"}:
            active_stage = "report"
            _report(config, run_dir, args)
        if args.stage == "optimize":
            active_stage = "optimize"
            _optimize(config, run_dir, args)
        if args.stage == "feature-develop":
            active_stage = "feature_develop"
            _feature_develop(config, run_dir, args)
        if args.stage == "cardinal-classify":
            active_stage = "cardinal_classify"
            _cardinal_classify(config, run_dir, args)
        if args.stage == "posterior-cardinal-classify":
            active_stage = "posterior_cardinal_classify"
            _posterior_cardinal_classify(config, run_dir, args)
        if args.stage == "posterior-cardinal-threeclass-optuna":
            active_stage = "posterior_cardinal_threeclass_optuna"
            _posterior_cardinal_threeclass_optuna(config, run_dir, args)
        if args.stage == "posterior-cardinal-contiguous-threeclass":
            active_stage = "posterior_cardinal_contiguous_threeclass"
            _posterior_cardinal_contiguous_threeclass(
                config, run_dir, args
            )
        if args.stage == "cardinal-develop":
            active_stage = "cardinal_develop"
            _cardinal_develop(config, run_dir, args)
        if args.stage == "feature-associate":
            active_stage = "feature_associate"
            _feature_associate(config, run_dir, args)
        if args.stage == "all":
            _stage_status(
                run_dir,
                "all",
                "complete_smoke" if effective_smoke else "complete",
                smoke=effective_smoke,
                completed_stages=["curate", "refit", "evaluate", "report"],
            )
        if args.stage in {
            "report",
            "optimize",
            "feature-develop",
            "cardinal-classify",
            "posterior-cardinal-classify",
            "posterior-cardinal-threeclass-optuna",
            "posterior-cardinal-contiguous-threeclass",
            "cardinal-develop",
            "feature-associate",
            "all",
        }:
            manifest_path = run_dir / "artifact_manifest.json"
            write_json(manifest_path, artifact_manifest(run_dir))
            verification = verify_artifact_manifest(run_dir, manifest_path)
            if verification["status"] != "pass":
                raise RuntimeError(f"Artifact manifest verification failed: {verification}")
    except Exception as error:
        (run_dir / "artifact_manifest.json").unlink(missing_ok=True)
        failed_stage = active_stage or (
            "feature_develop"
            if args.stage == "feature-develop"
            else "cardinal_classify"
            if args.stage == "cardinal-classify"
            else "posterior_cardinal_classify"
            if args.stage == "posterior-cardinal-classify"
            else "posterior_cardinal_threeclass_optuna"
            if args.stage == "posterior-cardinal-threeclass-optuna"
            else "posterior_cardinal_contiguous_threeclass"
            if args.stage == "posterior-cardinal-contiguous-threeclass"
            else "cardinal_develop"
            if args.stage == "cardinal-develop"
            else "feature_associate"
            if args.stage == "feature-associate"
            else args.stage
        )
        _stage_status(run_dir, failed_stage, "failed", error=repr(error))
        if args.stage == "all":
            _stage_status(
                run_dir,
                "all",
                "failed",
                failed_stage=failed_stage,
                error=repr(error),
            )
        raise
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
