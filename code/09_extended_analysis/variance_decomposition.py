#!/usr/bin/env python3
"""Design-based decomposition of reconstructed Tafel-slope posterior variance.

The existing posterior is an equal/nearly equal mixture over coordinate-perturbed
selection replicates.  This script applies the law of total variance in a nested
order that is identifiable from the archived run:

    total = regression + digitisation | window + window

Regression is the posterior variance within each fixed replicate. Digitisation
is the variance of replicate posterior means among replicates that retain the
same selected-bin window. Window is the variance of window-specific mean slopes.

The last two terms are design-based sensitivities. Since window switching is
itself triggered after coordinate perturbation, the attribution order is stated
explicitly and is not claimed to be a unique causal/physical partition.
"""

from __future__ import annotations

import argparse
from repository import REFIT
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_POSTERIOR = {
    "sample_record",
    "selection_replicate",
    "slope_mV_dec",
}
REQUIRED_WINDOWS = {
    "sample_record",
    "selection_replicate",
    "window_start_bin",
    "window_end_bin",
    "region_supported",
}


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    centre = weighted_mean(values, weights)
    return float(np.sum(weights * np.square(values - centre)) / np.sum(weights))


def make_window_id(frame: pd.DataFrame) -> pd.Series:
    start = pd.to_numeric(frame["window_start_bin"], errors="coerce")
    end = pd.to_numeric(frame["window_end_bin"], errors="coerce")
    if start.isna().any() or end.isna().any():
        bad = frame.loc[start.isna() | end.isna(), ["sample_record", "selection_replicate"]]
        raise ValueError(f"Posterior-bearing replicate has undefined window bins:\n{bad}")
    fallback = (
        frame.get("quantization_aware_fallback_used", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    prefix = np.where(fallback, "fallback", "strict")
    return pd.Series(
        [f"{p}:{int(a)}-{int(b)}" for p, a, b in zip(prefix, start, end, strict=True)],
        index=frame.index,
    )


def decompose_record(record_draws: pd.DataFrame, record_reps: pd.DataFrame) -> dict[str, object]:
    record = int(record_draws["sample_record"].iloc[0])
    slopes = pd.to_numeric(record_draws["slope_mV_dec"], errors="raise").to_numpy(float)
    if not np.isfinite(slopes).all():
        raise ValueError(f"Non-finite posterior slope for record {record}")

    x = record_reps.sort_values("selection_replicate").copy()
    weights = x["n_draws"].to_numpy(float)
    weights = weights / weights.sum()
    means = x["posterior_mean_mV_dec"].to_numpy(float)
    conditional_vars = x["posterior_variance_mV2_dec2"].to_numpy(float)

    regression_var = float(np.sum(weights * conditional_vars))
    between_replicate_var = weighted_var(means, weights)

    digitisation_var = 0.0
    window_rows: list[dict[str, object]] = []
    for window_id, group in x.groupby("window_id", sort=True):
        group_weights_raw = group["n_draws"].to_numpy(float)
        group_means = group["posterior_mean_mV_dec"].to_numpy(float)
        group_weight = float(group_weights_raw.sum() / x["n_draws"].sum())
        conditional_digitisation_var = weighted_var(group_means, group_weights_raw)
        digitisation_var += group_weight * conditional_digitisation_var
        window_rows.append(
            {
                "sample_record": record,
                "window_id": window_id,
                "n_selection_replicates": int(len(group)),
                "n_posterior_draws": int(group_weights_raw.sum()),
                "mixture_weight": group_weight,
                "window_posterior_mean_mV_dec": weighted_mean(group_means, group_weights_raw),
                "digitisation_variance_within_window_mV2_dec2": conditional_digitisation_var,
            }
        )

    windows = pd.DataFrame(window_rows)
    window_var = weighted_var(
        windows["window_posterior_mean_mV_dec"].to_numpy(float),
        windows["n_posterior_draws"].to_numpy(float),
    )

    total_var = float(np.var(slopes, ddof=0))
    decomposed_total = regression_var + digitisation_var + window_var
    closure_error = decomposed_total - total_var
    if not np.isclose(decomposed_total, total_var, rtol=1e-10, atol=1e-8):
        raise AssertionError(
            f"Variance decomposition failed for record {record}: "
            f"decomposed={decomposed_total}, total={total_var}"
        )

    components = {
        "regression": regression_var,
        "digitisation": digitisation_var,
        "window": window_var,
    }
    row: dict[str, object] = {
        "sample_record": record,
        "curve_id": str(record_draws["curve_id"].iloc[0]),
        "n_posterior_draws": int(len(slopes)),
        "n_selection_replicates_used": int(len(x)),
        "n_distinct_windows": int(x["window_id"].nunique()),
        "max_replicates_in_one_window": int(x["window_id"].value_counts().max()),
        "posterior_mean_mV_dec": float(np.mean(slopes)),
        "posterior_median_mV_dec": float(np.median(slopes)),
        "posterior_q025_mV_dec": float(np.quantile(slopes, 0.025)),
        "posterior_q975_mV_dec": float(np.quantile(slopes, 0.975)),
        "posterior_cri_width_mV_dec": float(
            np.quantile(slopes, 0.975) - np.quantile(slopes, 0.025)
        ),
        "total_variance_mV2_dec2": total_var,
        "total_sd_mV_dec": float(np.sqrt(total_var)),
        "between_replicate_variance_mV2_dec2": between_replicate_var,
        "variance_closure_error_mV2_dec2": closure_error,
        "window_component_interpretation": (
            "zero_within_registered_perturbation_design"
            if window_var < 1e-12
            else "observed_window_switching_sensitivity"
        ),
        "attribution_order": "regression_then_digitisation_given_window_then_window",
    }
    for name, variance in components.items():
        row[f"{name}_variance_mV2_dec2"] = variance
        row[f"{name}_sd_equivalent_mV_dec"] = float(np.sqrt(max(variance, 0.0)))
        row[f"{name}_variance_share_pct"] = float(100.0 * variance / total_var)
    return row, windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=REFIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    posterior_path = args.input_dir / "posterior_target_draws.csv.gz"
    windows_path = args.input_dir / "window_selection_replicates.csv.gz"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    posterior = pd.read_csv(posterior_path)
    windows = pd.read_csv(windows_path)
    missing_posterior = REQUIRED_POSTERIOR.difference(posterior.columns)
    missing_windows = REQUIRED_WINDOWS.difference(windows.columns)
    if missing_posterior or missing_windows:
        raise KeyError(
            f"Missing columns: posterior={sorted(missing_posterior)}, "
            f"windows={sorted(missing_windows)}"
        )

    posterior["sample_record"] = pd.to_numeric(posterior["sample_record"], errors="raise").astype(int)
    posterior["selection_replicate"] = pd.to_numeric(
        posterior["selection_replicate"], errors="raise"
    ).astype(int)
    windows["sample_record"] = pd.to_numeric(windows["sample_record"], errors="raise").astype(int)
    windows["selection_replicate"] = pd.to_numeric(
        windows["selection_replicate"], errors="raise"
    ).astype(int)

    replicate = (
        posterior.groupby(["sample_record", "selection_replicate"], sort=True)
        .agg(
            n_draws=("slope_mV_dec", "size"),
            posterior_mean_mV_dec=("slope_mV_dec", "mean"),
            posterior_median_mV_dec=("slope_mV_dec", "median"),
            posterior_variance_mV2_dec2=("slope_mV_dec", lambda s: float(np.var(s, ddof=0))),
            posterior_q025_mV_dec=("slope_mV_dec", lambda s: float(np.quantile(s, 0.025))),
            posterior_q975_mV_dec=("slope_mV_dec", lambda s: float(np.quantile(s, 0.975))),
        )
        .reset_index()
    )

    window_columns = [
        "sample_record",
        "selection_replicate",
        "window_start_bin",
        "window_end_bin",
        "window_start_eta_V",
        "window_end_eta_V",
        "window_n_bins",
        "potential_resolution_V",
        "log_current_resolution_dec",
        "region_supported",
        "quantization_aware_fallback_used",
    ]
    window_meta = windows[[c for c in window_columns if c in windows.columns]].copy()
    replicate = replicate.merge(
        window_meta,
        on=["sample_record", "selection_replicate"],
        how="left",
        validate="one_to_one",
    )
    if replicate["window_start_bin"].isna().any():
        bad = replicate.loc[
            replicate["window_start_bin"].isna(), ["sample_record", "selection_replicate"]
        ]
        raise ValueError(f"Missing supported window metadata:\n{bad}")
    replicate["window_id"] = make_window_id(replicate)

    record_rows: list[dict[str, object]] = []
    window_frames: list[pd.DataFrame] = []
    for record, record_draws in posterior.groupby("sample_record", sort=True):
        record_reps = replicate.loc[replicate["sample_record"].eq(record)].copy()
        row, record_windows = decompose_record(record_draws, record_reps)
        record_rows.append(row)
        window_frames.append(record_windows)

    records = pd.DataFrame(record_rows).sort_values("sample_record").reset_index(drop=True)
    window_summary = pd.concat(window_frames, ignore_index=True)
    long_components = records.melt(
        id_vars=[
            "sample_record",
            "curve_id",
            "posterior_median_mV_dec",
            "posterior_q025_mV_dec",
            "posterior_q975_mV_dec",
            "total_sd_mV_dec",
            "n_distinct_windows",
        ],
        value_vars=[
            "regression_variance_share_pct",
            "digitisation_variance_share_pct",
            "window_variance_share_pct",
        ],
        var_name="component",
        value_name="variance_share_pct",
    )
    long_components["component"] = long_components["component"].str.replace(
        "_variance_share_pct", "", regex=False
    )

    descriptive_rows: list[dict[str, object]] = []
    for component in ("regression", "digitisation", "window"):
        values = records[f"{component}_variance_share_pct"]
        descriptive_rows.append(
            {
                "component": component,
                "n_records": int(len(values)),
                "mean_share_pct": float(values.mean()),
                "sd_share_pct": float(values.std(ddof=1)),
                "q10_share_pct": float(values.quantile(0.10)),
                "q25_share_pct": float(values.quantile(0.25)),
                "median_share_pct": float(values.median()),
                "q75_share_pct": float(values.quantile(0.75)),
                "q90_share_pct": float(values.quantile(0.90)),
                "min_share_pct": float(values.min()),
                "max_share_pct": float(values.max()),
                "dominant_component_count": int(
                    (
                        records[
                            [
                                "regression_variance_share_pct",
                                "digitisation_variance_share_pct",
                                "window_variance_share_pct",
                            ]
                        ].idxmax(axis=1)
                        == f"{component}_variance_share_pct"
                    ).sum()
                ),
            }
        )
    descriptive = pd.DataFrame(descriptive_rows)

    records.to_csv(output_dir / "record_level_variance_decomposition.csv", index=False)
    replicate.to_csv(output_dir / "replicate_level_posterior_summary.csv", index=False)
    window_summary.to_csv(output_dir / "window_level_decomposition.csv", index=False)
    long_components.to_csv(output_dir / "component_contributions.csv", index=False)
    descriptive.to_csv(output_dir / "component_share_summary.csv", index=False)

    record_002 = records.loc[records["sample_record"].eq(2)].iloc[0]
    response_numbers = {
        "record_002": {
            key: (int(value) if isinstance(value, (np.integer,)) else float(value) if isinstance(value, (np.floating,)) else value)
            for key, value in record_002.to_dict().items()
        },
        "all_records": {
            "n_records": int(len(records)),
            "regression_share_median_pct": float(records["regression_variance_share_pct"].median()),
            "digitisation_share_median_pct": float(records["digitisation_variance_share_pct"].median()),
            "window_share_median_pct": float(records["window_variance_share_pct"].median()),
            "regression_dominant_count": int(
                (records[
                    [
                        "regression_variance_share_pct",
                        "digitisation_variance_share_pct",
                        "window_variance_share_pct",
                    ]
                ].idxmax(axis=1) == "regression_variance_share_pct").sum()
            ),
            "records_with_one_registered_window": int(records["n_distinct_windows"].eq(1).sum()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(response_numbers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(response_numbers, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
