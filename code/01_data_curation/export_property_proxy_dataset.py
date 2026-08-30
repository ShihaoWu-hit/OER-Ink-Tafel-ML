#!/usr/bin/env python3
"""Build the 97-row exploratory property-proxy ML dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ink_tafel_study.exploratory_ml import export_property_proxy_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("Data/03_outputs/study/outputs/current_97/data/records.csv"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("Data/03_outputs/study/outputs/current_97/targets/targets_wide.csv"),
    )
    parser.add_argument(
        "--panel-schema",
        type=Path,
        default=Path("Data/03_outputs/study/outputs/current_97/data/registered_panel_schema.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Data/02_inputs/ml_dataset_context_plus_property_proxy_4p_estimated.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = export_property_proxy_dataset(
        records_path=args.records,
        targets_path=args.targets,
        panel_schema_path=args.panel_schema,
        output_path=args.output,
    )
    print(f"Wrote {len(frame)} rows x {len(frame.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
