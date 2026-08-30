from __future__ import annotations

import importlib.metadata
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .analysis import CurveAnalysis, analyze_curve
from .data import InputBundle, load_inputs, sha256_file
from .plots import build_figure_manifest, plot_curve, plot_summary
from .preprocess import ProcessedCurve, preprocess_curve
from .reporting import aggregate_sample_candidates, write_summary_report


def _git_state(workspace: Path, workflow_project: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                str(workflow_project.relative_to(workspace) / "pyproject.toml"),
            ],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "commit": commit.stdout.strip(),
            "branch": branch.stdout.strip() or None,
            "dirty": bool(status.stdout.strip()),
            "workflow_files_tracked": tracked.returncode == 0,
        }
    except Exception:
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "workflow_files_tracked": False,
        }


def workflow_hashes(project_root: Path) -> tuple[dict[str, str], str]:
    paths = sorted((project_root / "src" / "tafel_refit").glob("*.py"))
    paths.extend(
        path
        for path in (
            project_root / "pyproject.toml",
            project_root / "uv.lock",
            project_root / "run.sh",
        )
        if path.is_file()
    )
    hashes = {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in sorted(paths)
    }
    digest = hashlib.sha256()
    for relative, file_hash in hashes.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return hashes, digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scipy", "matplotlib", "openpyxl", "pymc", "arviz"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _timestamped_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / stamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stamp}_{suffix:02d}"
        suffix += 1
    (candidate / "csv").mkdir(parents=True)
    (candidate / "figures" / "curves").mkdir(parents=True)
    (candidate / "bayes").mkdir(parents=True)
    return candidate


def validate_inputs(
    workbook_path: Path,
    mapping_csv_path: Path,
    canonical_csv_path: Path,
    config_path: Path,
    *,
    allow_unverified_hashes: bool = False,
) -> tuple[InputBundle, list[ProcessedCurve]]:
    bundle = load_inputs(
        workbook_path,
        mapping_csv_path,
        canonical_csv_path,
        config_path,
        allow_unverified_hashes=allow_unverified_hashes,
    )
    processed = []
    for _, manifest_row in bundle.manifest.iterrows():
        points = bundle.raw_points.loc[
            bundle.raw_points["curve_id"].eq(manifest_row["curve_id"])
        ]
        processed.append(preprocess_curve(manifest_row, points, bundle.config))
    return bundle, processed


def validation_summary(bundle: InputBundle, processed: list[ProcessedCurve]) -> dict[str, Any]:
    return {
        "status": "validated",
        "curves": len(bundle.manifest),
        "samples": int(bundle.manifest["sample_record"].nunique()),
        "finite_pairs": int(bundle.manifest["n_finite_pairs"].sum()),
        "nonfinite_pairs": int(bundle.manifest["n_nonfinite_pairs"].sum()),
        "nonpositive_points": int(
            (bundle.manifest["n_zero_current"] + bundle.manifest["n_negative_current"]).sum()
        ),
        "curves_fit_eligible": int(sum(curve.fit_eligible for curve in processed)),
        "curves_qc_excluded": int(sum(not curve.fit_eligible for curve in processed)),
        "workbook_sha256": bundle.workbook_sha256,
        "master_sha256": bundle.master_sha256,
    }


def _enriched_manifest(
    bundle: InputBundle, processed: Iterable[ProcessedCurve]
) -> pd.DataFrame:
    metrics = []
    for curve in processed:
        metrics.append(
            {
                "curve_id": curve.curve_id,
                **curve.metrics,
                "fit_eligible": curve.fit_eligible,
                "fit_exclusion_reason": curve.exclusion_reason,
                "qc_flags": ";".join(curve.qc_flags),
            }
        )
    return bundle.manifest.merge(
        pd.DataFrame(metrics), on="curve_id", how="left", validate="one_to_one"
    )


def run_pipeline(
    *,
    workbook_path: Path,
    mapping_csv_path: Path,
    canonical_csv_path: Path,
    config_path: Path,
    output_root: Path,
    mode: str,
    sample_records: list[int] | None,
    bootstrap_repetitions: int | None,
    optimizer_starts: int | None,
    make_plots: bool,
    allow_unverified_hashes: bool = False,
) -> Path:
    workbook_path = workbook_path.resolve()
    mapping_csv_path = mapping_csv_path.resolve()
    canonical_csv_path = canonical_csv_path.resolve()
    config_path = config_path.resolve()
    bundle, all_processed = validate_inputs(
        workbook_path,
        mapping_csv_path,
        canonical_csv_path,
        config_path,
        allow_unverified_hashes=allow_unverified_hashes,
    )
    config = dict(bundle.config)
    if mode == "smoke":
        default_records = [1, 3, 42]
        effective_records = sample_records or default_records
        repetitions = 5 if bootstrap_repetitions is None else bootstrap_repetitions
        starts = 4 if optimizer_starts is None else optimizer_starts
    elif mode == "quick":
        effective_records = sample_records
        repetitions = 30 if bootstrap_repetitions is None else bootstrap_repetitions
        starts = 8 if optimizer_starts is None else optimizer_starts
    elif mode == "full":
        effective_records = sample_records
        repetitions = (
            int(config["bootstrap_repetitions"])
            if bootstrap_repetitions is None
            else bootstrap_repetitions
        )
        starts = int(config["optimizer_starts"]) if optimizer_starts is None else optimizer_starts
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if effective_records is not None:
        requested = set(map(int, effective_records))
        unknown = sorted(requested.difference(bundle.manifest["sample_record"].astype(int)))
        if unknown:
            raise ValueError(f"Unknown sample_record values: {unknown}")
        processed = [curve for curve in all_processed if curve.sample_record in requested]
        manifest = bundle.manifest.loc[
            bundle.manifest["sample_record"].astype(int).isin(requested)
        ].copy()
    else:
        processed = all_processed
        manifest = bundle.manifest.copy()

    run_dir = _timestamped_run_dir(output_root.resolve())
    csv_dir = run_dir / "csv"
    figure_dir = run_dir / "figures"
    project_root = Path(__file__).resolve().parents[2]
    workspace = project_root.parent
    workflow_file_hashes, workflow_tree_hash = workflow_hashes(project_root)
    git_state = _git_state(workspace, project_root)
    initial_input_hashes = {
        "workbook": bundle.workbook_sha256,
        "mapping_csv": bundle.mapping_sha256,
        "canonical_csv": bundle.master_sha256,
        "config": sha256_file(config_path),
    }
    run_metadata = {
        "workflow_version": "0.4.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "sample_records": sorted(set(curve.sample_record for curve in processed)),
        "bootstrap_repetitions": int(repetitions),
        "optimizer_starts": int(starts),
        "random_seed": int(config["random_seed"]),
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "git": git_state,
        "workflow_tree_sha256": workflow_tree_hash,
        "workflow_files_sha256": workflow_file_hashes,
        "workflow_files_changed_during_run": None,
        "input_hash_verification_bypassed": bool(allow_unverified_hashes),
        "input_files_changed_during_run": None,
        "inputs": {
            "workbook": str(workbook_path),
            "workbook_sha256": initial_input_hashes["workbook"],
            "mapping_csv": str(mapping_csv_path),
            "mapping_sha256": initial_input_hashes["mapping_csv"],
            "canonical_csv": str(canonical_csv_path),
            "canonical_sha256": initial_input_hashes["canonical_csv"],
            "config": str(config_path),
            "config_sha256": initial_input_hashes["config"],
        },
        "effective_config": config,
    }
    (run_dir / "model_config.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    processed_by_id = {curve.curve_id: curve for curve in processed}
    enriched = _enriched_manifest(bundle, processed)
    enriched = enriched.loc[enriched["curve_id"].isin(processed_by_id)].copy()
    enriched.to_csv(csv_dir / "curve_manifest.csv", index=False)
    raw_frames = [curve.raw_points for curve in processed]
    binned_frames = [curve.binned_points for curve in processed if len(curve.binned_points)]
    pd.concat(raw_frames, ignore_index=True).to_csv(csv_dir / "curve_points_audited.csv", index=False)
    if binned_frames:
        pd.concat(binned_frames, ignore_index=True).to_csv(
            csv_dir / "binned_curve_points.csv", index=False
        )

    analyses: list[CurveAnalysis] = []
    errors: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    for index, curve in enumerate(processed, start=1):
        manifest_row = manifest.loc[manifest["curve_id"].eq(curve.curve_id)].iloc[0]
        print(f"[{index}/{len(processed)}] fitting {curve.curve_id}", flush=True)
        try:
            analysis = analyze_curve(
                curve,
                manifest_row,
                config,
                bootstrap_repetitions=int(repetitions),
                optimizer_starts=int(starts),
                seed=int(config["random_seed"]) + int(manifest_row["sheet_index"]) * 10_007,
            )
        except Exception as exc:
            errors.append(
                {
                    "curve_id": curve.curve_id,
                    "sample_record": curve.sample_record,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            analysis = CurveAnalysis(
                summary={
                    "curve_id": curve.curve_id,
                    "sample_record": curve.sample_record,
                    "curve_variant": int(manifest_row["curve_variant"]),
                    "sheet_index": int(manifest_row["sheet_index"]),
                    "sheet_name": manifest_row["sheet_name"],
                    "tafel_slope_source": manifest_row["tafel_slope_source"],
                    "existing_tafel_slope_mV_dec": manifest_row[
                        "y_tafel_slope_mV_dec"
                    ],
                    "fit_status": "error",
                    "fit_eligible": curve.fit_eligible,
                    "fit_exclusion_reason": curve.exclusion_reason,
                    "qc_flags": ";".join(curve.qc_flags),
                    "selected_model": "none",
                    "selected_model_display_name": "not fit",
                    "model_family": "continuous_tafel",
                    "kinetic_structure": "not_fit",
                    "observation_parameterization": "not_fit",
                    "low_current_floor_policy": "excluded_from_primary",
                    "model_role": "not_fit",
                    "potential_dependent_supported": False,
                    "label_eligible": False,
                    "label_reason": "runtime_error",
                },
                model_fits=pd.DataFrame(),
                bootstrap_draws=pd.DataFrame(),
                local_slopes=pd.DataFrame(),
                fitted_curves=pd.DataFrame(),
                fit_results={},
            )
        analyses.append(analysis)
        if make_plots:
            path = figure_dir / "curves" / f"{curve.curve_id}.png"
            plot_curve(curve, analysis, path)
            figure_paths.append(path)

    curve_summary = pd.DataFrame([analysis.summary for analysis in analyses])
    model_fits = pd.concat(
        [analysis.model_fits for analysis in analyses if len(analysis.model_fits)],
        ignore_index=True,
    ) if any(len(analysis.model_fits) for analysis in analyses) else pd.DataFrame()
    bootstrap = pd.concat(
        [analysis.bootstrap_draws for analysis in analyses if len(analysis.bootstrap_draws)],
        ignore_index=True,
    ) if any(len(analysis.bootstrap_draws) for analysis in analyses) else pd.DataFrame()
    local_slopes = pd.concat(
        [analysis.local_slopes for analysis in analyses if len(analysis.local_slopes)],
        ignore_index=True,
    ) if any(len(analysis.local_slopes) for analysis in analyses) else pd.DataFrame()
    fitted_curves = pd.concat(
        [analysis.fitted_curves for analysis in analyses if len(analysis.fitted_curves)],
        ignore_index=True,
    ) if any(len(analysis.fitted_curves) for analysis in analyses) else pd.DataFrame()

    curve_summary.to_csv(csv_dir / "curve_fit_summary.csv", index=False)
    model_fits.to_csv(csv_dir / "model_comparison.csv", index=False)
    bootstrap.to_csv(csv_dir / "bootstrap_draws.csv", index=False)
    local_slopes.to_csv(csv_dir / "local_slope_curves.csv", index=False)
    fitted_curves.to_csv(csv_dir / "fitted_curves.csv", index=False)
    error_columns = [
        "curve_id",
        "sample_record",
        "error_type",
        "error_message",
        "traceback",
    ]
    pd.DataFrame(errors, columns=error_columns).to_csv(
        csv_dir / "fit_errors.csv", index=False
    )

    selected_master = bundle.master.loc[
        bundle.master["sample_record"].astype(int).isin(curve_summary["sample_record"].astype(int))
    ].copy()
    sample_candidates, candidate_dataset = aggregate_sample_candidates(
        selected_master, curve_summary
    )
    sample_candidates.to_csv(csv_dir / "sample_refit_candidates.csv", index=False)
    candidate_dataset.to_csv(csv_dir / "data_ink_with_refit_candidates.csv", index=False)

    if make_plots:
        figure_paths.extend(plot_summary(curve_summary, figure_dir))
    figure_manifest = build_figure_manifest(figure_paths, run_dir)
    figure_manifest.to_csv(csv_dir / "figure_manifest.csv", index=False)

    run_metadata["deterministic_artifacts_sha256"] = {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(csv_dir.glob("*.csv"))
    }

    end_file_hashes, end_tree_hash = workflow_hashes(project_root)
    run_metadata["workflow_files_changed_during_run"] = bool(
        end_tree_hash != workflow_tree_hash
    )
    if end_tree_hash != workflow_tree_hash:
        run_metadata["workflow_tree_sha256_at_end"] = end_tree_hash
        run_metadata["workflow_files_sha256_at_end"] = end_file_hashes
    end_input_hashes = {
        "workbook": sha256_file(workbook_path),
        "mapping_csv": sha256_file(mapping_csv_path),
        "canonical_csv": sha256_file(canonical_csv_path),
        "config": sha256_file(config_path),
    }
    run_metadata["input_files_changed_during_run"] = bool(
        end_input_hashes != initial_input_hashes
    )
    run_metadata["input_sha256_at_end"] = end_input_hashes
    (run_dir / "model_config.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_summary_report(
        run_dir / "summary_report.md",
        run_metadata=run_metadata,
        manifest=enriched,
        curve_summary=curve_summary,
        sample_candidates=sample_candidates,
    )
    provenance_changed = bool(
        run_metadata["workflow_files_changed_during_run"]
        or run_metadata["input_files_changed_during_run"]
    )
    status = {
        "status": (
            "completed_with_provenance_change"
            if provenance_changed
            else ("complete" if not errors else "completed_with_errors")
        ),
        "curves_requested": len(processed),
        "curves_successful": int(curve_summary["fit_status"].eq("success").sum()),
        "curves_excluded": int(curve_summary["fit_status"].eq("excluded_by_qc").sum()),
        "curves_error": len(errors),
        "workflow_files_changed_during_run": run_metadata[
            "workflow_files_changed_during_run"
        ],
        "input_files_changed_during_run": run_metadata[
            "input_files_changed_during_run"
        ],
        "run_directory": str(run_dir),
    }
    (run_dir / "run_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Run directory: {run_dir}", flush=True)
    return run_dir
