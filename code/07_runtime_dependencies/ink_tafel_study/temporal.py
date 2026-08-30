from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from .evaluate import (
    DEFAULT_RELATION_COLUMNS,
    _models_for_panel,
    build_connected_group_ids,
    negative_original_mae,
    regression_metrics,
)
from .features import resolve_feature_panel
from .models import make_model_spec, predict_log_with_uncertainty


def run_historical_temporal_backtest(
    records: pd.DataFrame | str | Path,
    targets: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    target_name: str = "b_bayesian_local_line_mV_dec",
    cutoff_year: int = 2024,
    panel_names: Sequence[str] = ("dummy", "context", "context_plus_ink"),
    model_names: Sequence[str] = ("dummy_mean", "dummy_median", "bayesian_ridge", "gower_knn", "extra_trees"),
    inner_folds: int = 4,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a frozen historical 2024+ backtest without calling it untouched.

    If any record in a connected publication/lab/recipe/sample group is dated
    2024 or later, the entire group is assigned to the temporal test side.
    """

    record_frame = records.copy() if isinstance(records, pd.DataFrame) else pd.read_csv(records)
    target_frame = targets.copy() if isinstance(targets, pd.DataFrame) else pd.read_csv(targets)
    if {"target_name", "target_value_mV_dec"}.issubset(target_frame):
        target_frame = target_frame.loc[target_frame["target_name"].astype(str) == target_name, ["record_id", "target_value_mV_dec"]].rename(
            columns={"target_value_mV_dec": target_name}
        )
    elif target_name in target_frame:
        target_frame = target_frame[["record_id", target_name]]
    else:
        raise KeyError(f"Target registry does not contain {target_name}")
    target_frame[target_name] = pd.to_numeric(target_frame[target_name], errors="coerce")
    target_frame = target_frame.loc[np.isfinite(target_frame[target_name]) & (target_frame[target_name] > 0)]
    # Build connected components on the full canonical registry so an excluded
    # or target-missing bridge record cannot silently disconnect two groups.
    record_frame["cv_group_id"] = build_connected_group_ids(
        record_frame, relation_columns=DEFAULT_RELATION_COLUMNS
    )
    frame = record_frame.merge(target_frame, on="record_id", how="inner", validate="one_to_one")
    if "electrolyte_identity" in frame:
        frame = frame.loc[frame["electrolyte_identity"].astype(str).str.upper().isin({"KOH", "NAOH"})].copy()
    frame["publication_year"] = pd.to_numeric(frame["publication_year"], errors="coerce")
    frame = frame.loc[frame["publication_year"].notna()].reset_index(drop=True)
    temporal_groups = set(frame.loc[frame["publication_year"] >= cutoff_year, "cv_group_id"].astype(str))
    frame["temporal_role"] = np.where(frame["cv_group_id"].astype(str).isin(temporal_groups), "test_2024plus_connected", "train_pre2024")
    train = frame.loc[frame["temporal_role"] == "train_pre2024"].reset_index(drop=True)
    test = frame.loc[frame["temporal_role"] == "test_2024plus_connected"].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Historical temporal backtest requires nonempty pre-2024 train and 2024+ test groups")
    if set(train["cv_group_id"]).intersection(test["cv_group_id"]):
        raise AssertionError("Connected group crosses the temporal train/test boundary")
    y_train_log = np.log10(train[target_name].to_numpy(float))
    y_test = test[target_name].to_numpy(float)
    groups = train["cv_group_id"].astype(str).to_numpy()
    folds = min(inner_folds, len(np.unique(groups)))
    inner_cv = list(GroupKFold(n_splits=folds, shuffle=True, random_state=random_seed).split(train, groups=groups))
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for panel_name in panel_names:
        panel = resolve_feature_panel(frame, panel_name)
        if panel_name == "dummy":
            X_train = pd.DataFrame({"__constant__": np.ones(len(train))})
            X_test = pd.DataFrame({"__constant__": np.ones(len(test))})
        else:
            X_train = train.loc[:, list(panel.columns)]
            X_test = test.loc[:, list(panel.columns)]
        candidates: list[dict[str, object]] = []
        for model_name in _models_for_panel(panel_name, model_names):
            spec = make_model_spec(
                model_name,
                numeric_columns=("__constant__",) if panel_name == "dummy" else panel.numeric_columns,
                categorical_columns=() if panel_name == "dummy" else panel.categorical_columns,
                blocks={"dummy": ("__constant__",)} if panel_name == "dummy" else panel.blocks,
                random_seed=random_seed + sum(map(ord, model_name)),
            )
            search = GridSearchCV(
                spec.estimator,
                param_grid=dict(spec.param_grid) if spec.param_grid else [{}],
                scoring=negative_original_mae,
                cv=inner_cv,
                refit=True,
                error_score="raise",
                n_jobs=1,
            )
            search.fit(X_train, y_train_log, groups=groups)
            result = pd.DataFrame(search.cv_results_).iloc[int(search.best_index_)]
            score_columns = [column for column in result.index if column.startswith("split") and column.endswith("_test_score")]
            fold_mae = -pd.to_numeric(result[score_columns], errors="coerce").to_numpy(float)
            fold_mae = fold_mae[np.isfinite(fold_mae)]
            se = float(np.std(fold_mae, ddof=1) / np.sqrt(len(fold_mae))) if len(fold_mae) > 1 else 0.0
            candidates.append(
                {
                    "model": model_name,
                    "complexity_rank": spec.complexity_rank,
                    "inner_mae": -float(search.best_score_),
                    "inner_mae_se": se,
                    "search": search,
                }
            )
        best = min(candidates, key=lambda item: float(item["inner_mae"]))
        threshold = float(best["inner_mae"]) + float(best["inner_mae_se"])
        selected = min(
            (item for item in candidates if float(item["inner_mae"]) <= threshold),
            key=lambda item: (int(item["complexity_rank"]), float(item["inner_mae"]), str(item["model"])),
        )
        selected_search = selected["search"]
        predicted_log, predictive_sd = predict_log_with_uncertainty(selected_search.best_estimator_, X_test)
        predicted = np.power(10.0, predicted_log)
        for index, row in test.iterrows():
            prediction_rows.append(
                {
                    "target": target_name,
                    "scope": "alkaline",
                    "panel": panel_name,
                    "model": "nested_selected",
                    "selected_model_family": selected["model"],
                    "record_id": int(row["record_id"]),
                    "cv_group_id": row["cv_group_id"],
                    "publication_year": int(row["publication_year"]),
                    "observed_mV_dec": y_test[index],
                    "predicted_mV_dec": predicted[index],
                    "absolute_error_mV_dec": abs(y_test[index] - predicted[index]),
                    "predictive_sd_log10": predictive_sd[index] if predictive_sd is not None else np.nan,
                    "backtest_label": "historical_2024plus_not_untouched",
                }
            )
        metric_rows.append(
            {
                "target": target_name,
                "scope": "alkaline",
                "panel": panel_name,
                "model": "nested_selected",
                "selected_model_family": selected["model"],
                "n_train": len(train),
                "n_test": len(test),
                "n_train_groups": train["cv_group_id"].nunique(),
                "n_test_groups": test["cv_group_id"].nunique(),
                **regression_metrics(y_test, predicted),
                "backtest_label": "historical_2024plus_not_untouched",
            }
        )
        for item in candidates:
            selection_rows.append(
                {
                    "panel": panel_name,
                    "candidate_model_family": item["model"],
                    "inner_mae": item["inner_mae"],
                    "inner_mae_se": item["inner_mae_se"],
                    "one_se_threshold": threshold,
                    "selected": item["model"] == selected["model"],
                    "selection_data": "pre2024_training_inner_grouped_cv_only",
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination / "temporal_backtest_predictions.csv", index=False)
    metrics.to_csv(destination / "temporal_backtest_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(destination / "temporal_inner_selection.csv", index=False)
    frame[["record_id", "publication_year", "cv_group_id", "temporal_role"]].to_csv(
        destination / "temporal_split_manifest.csv", index=False
    )
    qa = {
        "status": "pass",
        "cutoff_year": cutoff_year,
        "label": "historical backtest; not an untouched or prospective test",
        "n_train": len(train),
        "n_test": len(test),
        "connected_group_overlap": 0,
        "groups_with_pre2024_records_promoted_to_test": int(
            test.loc[test["publication_year"] < cutoff_year, "cv_group_id"].nunique()
        ),
    }
    (destination / "temporal_backtest_qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return predictions, metrics
