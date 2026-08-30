#!/usr/bin/env python3
"""Read-only target-count preflight before freezing registered parity values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ink_tafel_study.targets import (
    TargetBuildConfig,
    TargetSourcePaths,
    _build_exact_fixed_current,
    _load_inputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-run", type=Path, required=True)
    parser.add_argument("--current-support", type=Path, required=True)
    parser.add_argument("--bayesian-table", type=Path, required=True)
    parser.add_argument("--bayesian-manifest", type=Path, required=True)
    parser.add_argument("--fixed-current", type=float, default=2.0)
    parser.add_argument("--minimum-draw-support", type=float, default=0.8)
    parser.add_argument("--audit-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = TargetSourcePaths.from_explicit(
        refit_run=args.refit_run.resolve(),
        bayesian_table=args.bayesian_table.resolve(),
        bayesian_manifest=args.bayesian_manifest.resolve(),
        current_conditioned_dir=args.current_support.resolve(),
    )
    frames = _load_inputs(paths)
    candidates = frames["sample_candidates"]
    counts = candidates["refit_candidate_status"].value_counts()
    expected_low = int(
        counts.get("eligible_single_curve", 0)
        + counts.get("potential_dependent_no_single_scalar", 0)
    )
    expected_single = int(counts.get("eligible_single_curve", 0))
    config = TargetBuildConfig(
        expected_curve_count=int(len(frames["fit_summary"])),
        expected_record_count=int(frames["fit_summary"]["sample_record"].nunique()),
        expected_primary_eligible=0,
        expected_low_branch_eligible=expected_low,
        expected_single_regime_eligible=expected_single,
        fixed_current_density_mA_cm2=args.fixed_current,
        minimum_draw_support_fraction=args.minimum_draw_support,
        inverse_grid_points=2049,
    )
    exact, draws = _build_exact_fixed_current(frames, config)
    if args.audit_output is not None:
        output = args.audit_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        exact.to_csv(output, index=False)
    payload = {
        "expected_curve_count": config.expected_curve_count,
        "expected_record_count": config.expected_record_count,
        "expected_primary_eligible": int(exact["target_eligible"].sum()),
        "expected_low_branch_eligible": expected_low,
        "expected_single_regime_eligible": expected_single,
        "primary_exclusion_counts": {
            str(key): int(value)
            for key, value in exact["exclusion_reason"].value_counts().items()
        },
        "fixed_current_draw_rows": int(len(draws)),
        "audit_output": (
            str(args.audit_output.resolve()) if args.audit_output is not None else ""
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
