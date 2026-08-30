from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

from .features import is_target_like_column

PRIMARY_TARGET = "b_bayesian_local_line_mV_dec"
PRIMARY_SOURCE = "calibrated_bayesian_local_line_eta015_contiguous_increase_window_mixture"
EXPECTED_EXCLUDED_IDS: list[int] = []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_slice(registry: pd.DataFrame, target_name: str, prefix: str) -> pd.DataFrame:
    columns = [
        "record_id",
        "curve_id",
        "target_value_mV_dec",
        "target_interval_low_mV_dec",
        "target_interval_high_mV_dec",
        "target_eligible",
        "posterior_target_supported",
        "exclusion_reason",
        "posterior_target_gap_reason",
        "target_current_density_mA_cm2",
    ]
    result = registry.loc[registry["target_name"].eq(target_name), columns].copy()
    if result["record_id"].duplicated().any():
        raise ValueError(f"Target registry has duplicate {target_name} rows")
    return result.rename(columns={column: f"{prefix}{column}" for column in columns if column != "record_id"})


def build_best_available_target_audit(
    records: pd.DataFrame, target_registry: pd.DataFrame
) -> pd.DataFrame:
    primary = _target_slice(target_registry, PRIMARY_TARGET, "primary_")
    audit = records[
        ["record_id", "curve_id", "electrolyte_scope", "row_data_quality_pass"]
    ].merge(primary, on="record_id", how="left", validate="one_to_one")
    if not audit["curve_id"].eq(audit["primary_curve_id"]).all():
        raise ValueError("Bayesian local-line target curve IDs do not match canonical records")

    primary_available = (
        audit["primary_target_eligible"].fillna(False).astype(bool)
        & audit["primary_posterior_target_supported"].fillna(False).astype(bool)
        & audit["primary_target_value_mV_dec"].notna()
    )
    audit["ml_training_eligible_bayesian_local_line"] = (
        audit["row_data_quality_pass"].fillna(False).astype(bool)
        & primary_available
    )
    audit["target_source"] = PRIMARY_SOURCE
    for output, source_column in (
        (
            "y_tafel_bayesian_local_line_mV_dec",
            "primary_target_value_mV_dec",
        ),
        (
            "y_tafel_bayesian_local_line_interval_low_mV_dec",
            "primary_target_interval_low_mV_dec",
        ),
        (
            "y_tafel_bayesian_local_line_interval_high_mV_dec",
            "primary_target_interval_high_mV_dec",
        ),
    ):
        audit[output] = audit[source_column].where(primary_available)

    return audit


def build_best_available_export(
    records: pd.DataFrame,
    target_registry: pd.DataFrame,
    primary_flat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    audit = build_best_available_target_audit(records, target_registry)
    eligible = audit.loc[audit["ml_training_eligible_bayesian_local_line"]].copy()

    old_identifier_and_target_columns = {
        "record_uid",
        "record_id",
        "sample_id",
        "curve_id",
        "publication_id",
        "doi",
        "recipe_hash",
        "electrolyte_scope",
        "row_data_quality_pass",
        "bayesian_primary_provisional_eligible",
        "bayesian_primary_formal_eligible",
        "y_tafel_bayesian_primary_mV_dec",
        "y_tafel_bayesian_primary_interval_low_mV_dec",
        "y_tafel_bayesian_primary_interval_high_mV_dec",
    }
    feature_columns = [
        column
        for column in primary_flat.columns
        if column not in old_identifier_and_target_columns
        and not is_target_like_column(column)
    ]
    missing_features = sorted(set(feature_columns) - set(records.columns))
    if missing_features:
        raise ValueError(f"Canonical records are missing feature columns: {missing_features}")

    identifiers = [
        "record_uid",
        "record_id",
        "sample_id",
        "curve_id",
        "publication_id",
        "doi",
        "recipe_hash",
        "electrolyte_scope",
        "row_data_quality_pass",
    ]
    target_columns = [
        "target_source",
        "y_tafel_bayesian_local_line_mV_dec",
        "y_tafel_bayesian_local_line_interval_low_mV_dec",
        "y_tafel_bayesian_local_line_interval_high_mV_dec",
    ]
    training = records[identifiers + feature_columns].merge(
        eligible[["record_id"] + target_columns],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    training = training[identifiers + target_columns + feature_columns].sort_values(
        "record_id"
    ).reset_index(drop=True)

    excluded_ids = audit.loc[
        ~audit["ml_training_eligible_bayesian_local_line"], "record_id"
    ].astype(int).tolist()
    scope_counts = {
        str(key): int(value)
        for key, value in training["electrolyte_scope"].value_counts().to_dict().items()
    }
    checks = {
        "canonical_records_are_97": len(records) == 97,
        "training_rows_are_97": len(training) == 97,
        "training_record_ids_unique": not training["record_id"].duplicated().any(),
        "training_curve_ids_unique": not training["curve_id"].duplicated().any(),
        "all_training_rows_pass_row_qa": bool(training["row_data_quality_pass"].all()),
        "all_training_targets_complete": bool(
            training[
                [
                    "y_tafel_bayesian_local_line_mV_dec",
                    "y_tafel_bayesian_local_line_interval_low_mV_dec",
                    "y_tafel_bayesian_local_line_interval_high_mV_dec",
                ]
            ].notna().all().all()
        ),
        "all_targets_use_bayesian_local_line_policy": training["target_source"].eq(
            PRIMARY_SOURCE
        ).all(),
        "exact_2mA_columns_absent": not any(
            "exact_2ma" in column.lower() or "b_refit_at_2ma" in column.lower()
            for column in training.columns
        ),
        "excluded_ids_match_policy": excluded_ids == EXPECTED_EXCLUDED_IDS,
        "scope_counts_match_policy": scope_counts
        == {"alkaline": 90, "acidic": 5, "neutral": 2},
    }
    if not all(checks.values()):
        raise ValueError(f"Best-available export QA failed: {checks}")

    qa: dict[str, object] = {
        "status": "ml_ready_for_sensitivity_training_ppc_pending",
        "n_rows": len(training),
        "n_features_inherited_from_primary_flat": len(feature_columns),
        "target_source_counts": {
            str(key): int(value)
            for key, value in training["target_source"].value_counts().to_dict().items()
        },
        "electrolyte_scope_counts": scope_counts,
        "excluded_record_ids": excluded_ids,
        "checks": checks,
        "formal_use_allowed": False,
        "formal_use_blocker": "named manual PPC review is incomplete",
    }
    return training, audit, qa


def write_export(
    *,
    source_dir: Path,
    output_dir: Path,
    flat_output: Path,
    records_path: Path | None = None,
    target_registry_path: Path | None = None,
    feature_template_path: Path | None = None,
) -> dict[str, object]:
    inputs = {
        "records": records_path or source_dir / "records.csv",
        "target_registry": target_registry_path or source_dir / "target_registry.csv",
        "primary_alkaline_flat": feature_template_path
        or source_dir / "primary_alkaline_flat.csv",
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required {name}: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if flat_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing flat export: {flat_output}")

    records = pd.read_csv(inputs["records"])
    target_registry = pd.read_csv(inputs["target_registry"])
    primary_flat = pd.read_csv(inputs["primary_alkaline_flat"])
    training, audit, qa = build_best_available_export(records, target_registry, primary_flat)

    output_dir.mkdir(parents=True)
    training_path = output_dir / "all_electrolyte_best_available_flat.csv"
    audit_path = output_dir / "best_available_target_audit.csv"
    qa_path = output_dir / "qa_summary.json"
    training.to_csv(training_path, index=False)
    training.to_csv(flat_output, index=False)
    audit.to_csv(audit_path, index=False)
    shutil.copy2(inputs["records"], output_dir / "records.csv")
    shutil.copy2(inputs["target_registry"], output_dir / "target_registry.csv")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = """# All-electrolyte Bayesian low-potential-slope ML export

This export uses one target definition for all 97 records:

- selected single-regime models contribute their posterior constant slope;
- selected two-regime models contribute the model-coordinate low-potential
  branch posterior slope;
- the stored value is the posterior median and its 95% draw interval.

The exact-2 mA functional target is not present in this ML export and is not a
fallback, eligibility gate, or current-conditioned training label. It remains
available only in the canonical diagnostic/audit package.

Use `registered_panel_schema.csv` and `feature_registry.csv` from the sibling
`Data_ink_Tafel_ML_ready_0718` package to select predictors. This export is for
the all-electrolyte/best-available sensitivity analysis. The registered primary
analysis is the alkaline low-potential-slope cohort. Named manual PPC review is still required
before formal performance claims.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "inputs": [
            {"name": name, "path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        ],
        "outputs": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        ],
        "flat_output": {
            "path": str(flat_output),
            "size_bytes": flat_output.stat().st_size,
            "sha256": _sha256(flat_output),
        },
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return qa


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the best-available ML export")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flat-output", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--target-registry", type=Path)
    parser.add_argument("--feature-template", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    qa = write_export(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        flat_output=args.flat_output,
        records_path=args.records,
        target_registry_path=args.target_registry,
        feature_template_path=args.feature_template,
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
