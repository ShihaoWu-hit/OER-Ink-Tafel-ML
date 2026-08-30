from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import PRIMARY_MODEL_NAMES
from .pipeline import run_pipeline, validate_inputs, validation_summary


def _sample_records(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample records must be comma-separated integers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tafel-refit",
        description="Audit and refit digitized OER Tafel curves without overwriting source data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, required=True)
    common.add_argument("--workbook", type=Path, required=True)
    common.add_argument("--mapping-csv", type=Path, required=True)
    common.add_argument("--canonical-csv", type=Path, required=True)
    common.add_argument(
        "--allow-unverified-hashes",
        action="store_true",
        help="Bypass the fail-closed SHA256 gate; use only after a new manual mapping audit.",
    )

    validate = subparsers.add_parser("validate", parents=[common])
    validate.set_defaults(handler=_handle_validate)

    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--mode", choices=("smoke", "quick", "full"), default="quick")
    run.add_argument("--sample-records", type=_sample_records)
    run.add_argument("--bootstrap-reps", type=int)
    run.add_argument("--optimizer-starts", type=int)
    run.add_argument("--no-plots", action="store_true")
    run.set_defaults(handler=_handle_run)

    bayes = subparsers.add_parser("bayes")
    bayes.add_argument("--run-dir", type=Path, required=True)
    bayes.add_argument("--curve-id", required=True)
    bayes.add_argument(
        "--model",
        choices=PRIMARY_MODEL_NAMES,
        help=(
            "Optional primary-model compatibility key; omit to use the "
            "deterministically selected structural variant."
        ),
    )
    bayes.add_argument("--draws", type=int, default=2000)
    bayes.add_argument("--tune", type=int, default=1000)
    bayes.add_argument("--chains", type=int, default=4)
    bayes.add_argument("--target-accept", type=float, default=0.95)
    bayes.add_argument("--seed", type=int, default=42)
    bayes.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="Positive sampling-attempt number used in non-overwriting artifact names.",
    )
    bayes.add_argument(
        "--paper-exact",
        action="store_true",
        help="Use raw positive points, iid Normal errors and fixed sigma=0.10 like the paper.",
    )
    bayes.add_argument(
        "--allow-workflow-mismatch",
        action="store_true",
        help=(
            "Allow a Bayesian sensitivity run when the current workflow hash differs "
            "from the deterministic run; the result is not a formal confirmation."
        ),
    )
    bayes.set_defaults(handler=_handle_bayes)
    return parser


def _handle_validate(args: argparse.Namespace) -> int:
    bundle, processed = validate_inputs(
        args.workbook,
        args.mapping_csv,
        args.canonical_csv,
        args.config,
        allow_unverified_hashes=args.allow_unverified_hashes,
    )
    print(json.dumps(validation_summary(bundle, processed), indent=2, ensure_ascii=False))
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    run_pipeline(
        workbook_path=args.workbook,
        mapping_csv_path=args.mapping_csv,
        canonical_csv_path=args.canonical_csv,
        config_path=args.config,
        output_root=args.output_root,
        mode=args.mode,
        sample_records=args.sample_records,
        bootstrap_repetitions=args.bootstrap_reps,
        optimizer_starts=args.optimizer_starts,
        make_plots=not args.no_plots,
        allow_unverified_hashes=args.allow_unverified_hashes,
    )
    return 0


def _handle_bayes(args: argparse.Namespace) -> int:
    from .bayes import run_bayesian_confirmation

    paths = run_bayesian_confirmation(
        run_dir=args.run_dir,
        curve_id=args.curve_id,
        model_name=args.model,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
        sampling_attempt=args.attempt,
        paper_exact=args.paper_exact,
        allow_workflow_mismatch=args.allow_workflow_mismatch,
    )
    for path in paths:
        print(path)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
