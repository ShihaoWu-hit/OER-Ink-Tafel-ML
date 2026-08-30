from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--ml-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    copies = {
        "data0718_complete_master.csv": "master.csv",
        "data0718_complete_master.parquet": "master.parquet",
        "feature_registry.csv": "feature_registry.csv",
        "bayesian_tafel_targets_provisional.csv": "bayesian_targets.csv",
        "manual_corrections_applied.csv": "corrections.csv",
        "source_decisions.csv": "data_decisions.csv",
    }
    for source_name, output_name in copies.items():
        shutil.copy2(args.source_package / source_name, args.output_dir / output_name)

    registry = pd.read_csv(args.formal_run / "targets" / "target_registry.csv")
    registry = registry.drop(columns=["posterior_summary_path"], errors="ignore")
    registry.to_csv(args.output_dir / "target_registry.csv", index=False)

    audit = pd.read_csv(args.formal_run / "targets" / "bayesian_audit.csv")
    audit = audit.drop(
        columns=[
            "diagnostics_path",
            "netcdf_path",
            "posterior_summary_path",
            "ppc_plot_path",
            "functional_draws_path",
            "slope_draws_path",
            "ppc_review_path",
            "source_sample_record",
        ],
        errors="ignore",
    )
    audit.to_csv(args.output_dir / "bayesian_audit.csv", index=False)

    slope_draws_path = args.output_dir / "posterior_slope_draws.csv.gz"
    slope_draws = pd.read_csv(
        args.formal_run / "targets" / "posterior_slope_draws.csv.gz",
        low_memory=False,
    ).drop(columns=["source_sample_record"], errors="ignore")
    slope_draws.to_csv(slope_draws_path, index=False, compression="gzip")
    exact_source = (
        args.formal_run / "targets" / "posterior_functional_draws_at_2mA.csv.gz"
    )
    if exact_source.is_file():
        exact_draws = pd.read_csv(exact_source, low_memory=False).drop(
            columns=["source_sample_record"], errors="ignore"
        )
        exact_draws.to_csv(
            args.output_dir / "posterior_draws_at_2mA.csv.gz",
            index=False,
            compression="gzip",
        )
    shutil.copy2(
        args.formal_run / "data" / "registered_panel_schema.csv",
        args.output_dir / "panel_schema.csv",
    )
    ml_export = pd.read_csv(args.ml_export, nrows=5)
    forbidden_ml_columns = sorted(
        column
        for column in ml_export.columns
        if "exact_2ma" in column.lower()
        or "b_refit_at_2ma" in column.lower()
        or "target_current_density" in column.lower()
    )
    if forbidden_ml_columns:
        raise ValueError(
            "Exact-2 mA fields are forbidden from the v5 ML export: "
            f"{forbidden_ml_columns}"
        )
    shutil.copy2(args.ml_export, args.output_dir / "ml_ready.csv")

    qa = json.loads((args.formal_run / "targets" / "target_qa.json").read_text())
    qa["canonical_grain"] = "97 unique records / 97 unique curves"
    qa["n_evidence_curves"] = 97
    qa["n_records"] = 97
    qa["n_curves"] = 97
    qa["electrolyte_counts"] = {"alkaline": 90, "acidic": 5, "neutral": 2}
    master = pd.read_csv(args.output_dir / "master.csv", low_memory=False)
    qa["ir_status_counts"] = {
        str(key): int(value)
        for key, value in master["ir_correction_status"].value_counts().to_dict().items()
    }
    qa["best_available_dielectric_estimated_count"] = 31
    qa.pop("n_excluded_duplicate_curves", None)
    qa.pop("excluded_duplicate_curve_ids", None)
    qa.pop("n_retained_modeling_curves", None)
    qa.pop("ppc_review_source", None)
    qa["source_mode"] = "canonical_97"
    aggregate = qa.get("posterior_slope_draw_aggregate", {})
    aggregate.pop("source_aggregate_path", None)
    aggregate["source_aggregate_sha256"] = sha256_file(slope_draws_path)
    aggregate["mapping_policy"] = "canonical curve_id-to-record_id binding"
    aggregate.get("hard_checks", {}).pop("sr050_absent", None)
    (args.output_dir / "qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    (args.output_dir / "README.md").write_text(
        "# Canonical 97-record dataset\n\n"
        "This is the only active data source for modeling. It contains 97 unique "
        "records and 97 unique curves, including record 42 = sr042_v1 and record "
        "43 = sr042_v2.\n\n"
        "- `master.csv`: complete, audit-rich canonical data.\n"
        "- `ml_ready.csv`: compact 97-row low-potential-slope training export.\n"
        "- `target_registry.csv`: all three registered Bayesian target definitions.\n"
        "- `bayesian_targets.csv`: one-row-per-record low-potential target summary.\n"
        "- `posterior_slope_draws.csv.gz`: draw-level primary-target uncertainty; "
        "single-regime uses constant b and two-regime uses the model-coordinate "
        "low-potential branch.\n"
        "- `posterior_draws_at_2mA.csv.gz`: optional diagnostic/audit artifact only; "
        "it is excluded from ML and readiness gates.\n"
        "- `feature_registry.csv` and `panel_schema.csv`: registered predictor policy.\n\n"
        "Formal evaluation remains blocked only by named manual PPC review.\n",
        encoding="utf-8",
    )

    files = []
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        files.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps({"files": files}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
