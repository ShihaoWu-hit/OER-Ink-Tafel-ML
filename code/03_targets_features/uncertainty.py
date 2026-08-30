from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def score_functional_target_draws(
    oof_aggregated: pd.DataFrame | str | Path,
    target_draws: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    target_name: str = "b_bayesian_local_line_mV_dec",
    max_draws: int = 100,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate synchronized functional-target draws through OOF scoring.

    Models are not retrained for each draw.  This is therefore explicitly a
    score-level target-uncertainty sensitivity analysis, complementary to (not
    a replacement for) full refitting under posterior target draws.
    """

    oof = (
        oof_aggregated.copy()
        if isinstance(oof_aggregated, pd.DataFrame)
        else pd.read_csv(oof_aggregated)
    )
    draws = target_draws.copy() if isinstance(target_draws, pd.DataFrame) else pd.read_csv(target_draws)
    required_oof = {"record_id", "panel", "model", "predicted_mV_dec"}
    slope_column = next(
        (
            column
            for column in (
                "primary_slope_mV_dec",
                "local_slope_mV_dec",
                "kinetic_slope_mV_dec",
            )
            if column in draws.columns
        ),
        "",
    )
    draw_column = (
        "posterior_draw_index"
        if "posterior_draw_index" in draws.columns
        else "bootstrap_replicate"
    )
    required_draws = {"sample_record", draw_column, slope_column}
    if not required_oof.issubset(oof):
        raise KeyError(f"OOF table is missing {sorted(required_oof - set(oof))}")
    if not required_draws.issubset(draws):
        raise KeyError(f"Target draws are missing {sorted(required_draws - set(draws))}")
    if "target" in oof:
        oof = oof.loc[oof["target"].astype(str) == target_name].copy()
    if oof.empty:
        raise ValueError(f"No OOF rows available for target {target_name!r}")
    oof["record_id"] = pd.to_numeric(oof["record_id"], errors="raise").astype(int)
    draws["record_id"] = pd.to_numeric(draws["sample_record"], errors="raise").astype(int)
    draws[draw_column] = pd.to_numeric(draws[draw_column], errors="raise").astype(int)
    draws["local_slope_mV_dec"] = pd.to_numeric(draws[slope_column], errors="coerce")
    draws = draws.loc[np.isfinite(draws["local_slope_mV_dec"]) & (draws["local_slope_mV_dec"] > 0)].copy()
    if draws.duplicated(["record_id", draw_column]).any():
        raise ValueError(
            f"Functional target draws must be unique by record_id/{draw_column}"
        )

    available_draws = np.sort(draws[draw_column].unique())
    if len(available_draws) > max_draws:
        rng = np.random.default_rng(random_seed)
        selected_draws = np.sort(rng.choice(available_draws, size=max_draws, replace=False))
    else:
        selected_draws = available_draws
    draws = draws.loc[draws[draw_column].isin(selected_draws)]

    group_columns = [column for column in ("analysis_id", "target", "scope", "panel", "model") if column in oof]
    rows: list[dict[str, object]] = []
    for keys, predictions in oof.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(group_columns, key_values, strict=True))
        base = predictions[["record_id", "predicted_mV_dec"]].copy()
        if base["record_id"].duplicated().any():
            raise ValueError("OOF input must already be aggregated once per record/model/panel")
        for replicate, observed_draw in draws.groupby(draw_column, sort=True):
            paired = base.merge(
                observed_draw[["record_id", "local_slope_mV_dec"]],
                on="record_id",
                how="inner",
                validate="one_to_one",
            )
            if paired.empty:
                continue
            observed = paired["local_slope_mV_dec"].to_numpy(float)
            predicted = paired["predicted_mV_dec"].to_numpy(float)
            correlation = spearmanr(observed, predicted).statistic if len(paired) > 1 else np.nan
            rows.append(
                {
                    **metadata,
                    "target_draw": int(replicate),
                    "n_records": len(paired),
                    "mae_mV_dec": float(mean_absolute_error(observed, predicted)),
                    "rmse_mV_dec": float(np.sqrt(mean_squared_error(observed, predicted))),
                    "median_ae_mV_dec": float(median_absolute_error(observed, predicted)),
                    "spearman": float(correlation) if np.isfinite(correlation) else np.nan,
                    "r2": float(r2_score(observed, predicted)) if len(paired) > 1 else np.nan,
                    "uncertainty_propagation": "synchronized_functional_draw_score_only_no_model_retraining",
                }
            )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        summary = pd.DataFrame()
    else:
        summary_rows: list[dict[str, object]] = []
        for keys, frame in metrics.groupby(group_columns, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            row: dict[str, object] = dict(zip(group_columns, key_values, strict=True))
            row["n_target_draws"] = frame["target_draw"].nunique()
            row["n_records_min"] = int(frame["n_records"].min())
            for metric in ("mae_mV_dec", "rmse_mV_dec", "median_ae_mV_dec", "spearman", "r2"):
                values = pd.to_numeric(frame[metric], errors="coerce").dropna()
                row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
                row[f"{metric}_q025"] = float(values.quantile(0.025)) if len(values) else np.nan
                row[f"{metric}_q975"] = float(values.quantile(0.975)) if len(values) else np.nan
            summary_rows.append(row)
        summary = pd.DataFrame(summary_rows)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "target_draw_metrics.csv", index=False)
    summary.to_csv(destination / "target_draw_metric_summary.csv", index=False)
    qa = {
        "status": "pass" if not metrics.empty and metrics["target_draw"].nunique() == len(selected_draws) else "fail",
        "target": target_name,
        "requested_draws": max_draws,
        "used_draws": int(len(selected_draws)),
        "draw_ids": [int(value) for value in selected_draws],
        "synchronized_by": draw_column,
        "slope_semantics": (
            "selected_model_constant_slope_or_model_coordinate_low_potential_branch"
            if slope_column == "primary_slope_mV_dec"
            else "legacy_functional_target_draw_fallback"
        ),
        "method_boundary": "score-level target uncertainty; models are not retrained for every draw",
    }
    (destination / "target_draw_qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return metrics, summary
