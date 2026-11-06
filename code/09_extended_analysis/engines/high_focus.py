#!/usr/bin/env python3
"""High-focus reliance under repeated grouped CV and held-out-group refits."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


Q_COLUMNS = ["q_low_60_side", "q_120_side", "q_high_off"]
HIGH_THRESHOLD = 0.5
SOFT_FPR_CAP = 0.20
PENALTY_GRID = (0.003, 0.01, 0.03)
THRESHOLD_GRID = (
    0.03,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
    0.225,
    0.25,
    0.275,
    0.30,
    0.35,
    0.40,
    0.50,
    0.75,
    1.00,
)
N_PERMUTATIONS_REPEAT = 100
N_PERMUTATIONS_LOCO = 200
SEED = 20261016


def _load_high_gate_module():
    """Load the existing high-gate implementation without changing it."""

    source = Path(__file__).with_name("stage_reliance.py")
    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location("high_gate_feature_importance", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load high-gate implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HG = _load_high_gate_module()






def _binary_log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(target, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1.0e-9, 1.0 - 1.0e-9)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p))))


def _metrics(target: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(target, dtype=float)
    s = np.asarray(score, dtype=float)
    predicted = s >= float(threshold)
    metrics = HG._soft_high_metrics(y, predicted)
    metrics.update(
        {
            "soft_log_loss": _binary_log_loss(y, s),
            "mean_score": float(np.mean(s)),
            "mean_margin": float(np.mean(s - float(threshold))),
            "n": int(len(y)),
        }
    )
    return {str(k): float(v) for k, v in metrics.items()}


def _fit_high_branch(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    probability_penalty: float,
    policy_penalty: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_ids = train["record_id"].astype(str).tolist()
    y_train = labels.loc[train_ids, Q_COLUMNS].to_numpy(float)
    fitted = HG._fit_high_branch(
        train.reset_index(drop=True),
        train["cv_group_id"].astype(str).tolist(),
        test.reset_index(drop=True),
        y_train,
        probability_penalty=float(probability_penalty),
        policy_penalty=float(policy_penalty),
    )
    _, engineered_train, engineered_test, _ = HG._fold_local_engineered_features(
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        panel="context_plus_polarity",
        rare_min_frequency=HG.RARE_MIN_FREQUENCY,
    )
    return fitted, engineered_train, engineered_test


def _policy_score(fitted: dict[str, Any], engineered: pd.DataFrame) -> np.ndarray:
    if len(engineered) == 0:
        return np.empty(0, dtype=float)
    distances = fitted["distance_model"]._pairwise_distances(engineered)
    novelty = HG._k_smallest_mean(distances, HG.NOVELTY_NEIGHBORS)
    probability = fitted["policy_estimator"].predict_proba(
        pd.DataFrame({"gower_novelty": novelty})
    )
    return np.asarray(probability[:, 2], dtype=float)


def _perturbation_rows(
    fitted: dict[str, Any],
    engineered_train: pd.DataFrame,
    engineered_test: pd.DataFrame,
    target: np.ndarray,
    threshold: float,
    *,
    split_kind: str,
    split_id: str,
    seed: int,
    n_permutations: int,
) -> list[dict[str, Any]]:
    base_score = _policy_score(fitted, engineered_test)
    target = np.asarray(target, dtype=float)
    focus = target >= HIGH_THRESHOLD
    rows: list[dict[str, Any]] = []
    for feature_index, spec in enumerate(HG.FEATURES):
        row: dict[str, Any] = {
            "split_kind": split_kind,
            "split_id": str(split_id),
            "feature": spec.feature,
            "display_name": spec.display_name,
            "source_columns": ";".join(spec.source_columns),
            "high_gate_column": spec.high_column or "",
            "n_test": int(len(target)),
            "n_high_focus": int(focus.sum()),
            "threshold": float(threshold),
            "baseline_soft_log_loss_all": _binary_log_loss(target, base_score),
            "baseline_mean_margin_all": float(np.mean(base_score - threshold)),
            "baseline_soft_log_loss_high_focus": (
                _binary_log_loss(target[focus], base_score[focus]) if focus.any() else np.nan
            ),
            "baseline_mean_margin_high_focus": (
                float(np.mean(base_score[focus] - threshold)) if focus.any() else np.nan
            ),
            "n_permutations": int(n_permutations),
        }
        if spec.high_column is None or spec.high_column not in engineered_test.columns:
            row.update(
                {
                    "importance_soft_log_loss_all": 0.0,
                    "importance_soft_log_loss_high_focus": 0.0,
                    "importance_mean_margin_change_all": 0.0,
                    "importance_mean_margin_change_high_focus": 0.0,
                    "positive_importance_fraction_all": 1.0,
                    "positive_importance_fraction_high_focus": 1.0,
                    "structural_status": "not_used_by_high_gate",
                }
            )
            rows.append(row)
            continue
        train_values = engineered_train[spec.high_column].to_numpy(copy=True)
        if len(train_values) == 0:
            raise ValueError(f"No training values available for {spec.feature}")
        rng = np.random.default_rng(int(seed) + 1009 * feature_index)
        loss_all: list[float] = []
        loss_focus: list[float] = []
        margin_change_all: list[float] = []
        margin_change_focus: list[float] = []
        for _ in range(int(n_permutations)):
            perturbed = engineered_test.copy()
            perturbed[spec.high_column] = rng.choice(
                train_values, size=len(perturbed), replace=True
            )
            score = _policy_score(fitted, perturbed)
            loss_all.append(_binary_log_loss(target, score))
            margin_change_all.append(float(np.mean(score - base_score)))
            if focus.any():
                loss_focus.append(_binary_log_loss(target[focus], score[focus]))
                margin_change_focus.append(float(np.mean(score[focus] - base_score[focus])))
        base_all = float(row["baseline_soft_log_loss_all"])
        base_focus = row["baseline_soft_log_loss_high_focus"]
        row.update(
            {
                "importance_soft_log_loss_all": float(np.mean(loss_all) - base_all),
                "importance_soft_log_loss_high_focus": (
                    float(np.mean(loss_focus) - float(base_focus)) if focus.any() else np.nan
                ),
                "importance_mean_margin_change_all": float(np.mean(margin_change_all)),
                "importance_mean_margin_change_high_focus": (
                    float(np.mean(margin_change_focus)) if focus.any() else np.nan
                ),
                "positive_importance_fraction_all": float(
                    np.mean(np.asarray(loss_all) > base_all)
                ),
                "positive_importance_fraction_high_focus": (
                    float(np.mean(np.asarray(loss_focus) > float(base_focus)))
                    if focus.any()
                    else np.nan
                ),
                "structural_status": "active_high_gate_feature",
            }
        )
        rows.append(row)
    return rows


def _select_inner_parameters(
    train: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    outer_id: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    groups = train["cv_group_id"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    candidate_rows: list[dict[str, Any]] = []
    for penalty in PENALTY_GRID:
        fold_payloads: list[tuple[int, np.ndarray, np.ndarray]] = []
        for inner_fold, (train_pos, val_pos) in enumerate(
            splitter.split(train, groups=groups)
        ):
            inner_train = train.iloc[train_pos].reset_index(drop=True)
            inner_val = train.iloc[val_pos].reset_index(drop=True)
            fitted, _, _ = _fit_high_branch(
                inner_train,
                inner_val,
                labels,
                probability_penalty=penalty,
                policy_penalty=penalty,
            )
            scores = np.asarray(fitted["baseline_policy_score"], dtype=float)
            target = labels.loc[
                inner_val["record_id"].astype(str).tolist(), "q_high_off"
            ].to_numpy(float)
            fold_payloads.append((inner_fold, target, scores))
        for threshold in THRESHOLD_GRID:
            fold_metrics = [_metrics(target, scores, threshold) for _, target, scores in fold_payloads]
            candidate_rows.append(
                {
                    "outer_id": str(outer_id),
                    "penalty": float(penalty),
                    "threshold": float(threshold),
                    "soft_f0_5": float(np.mean([m["soft_f0_5"] for m in fold_metrics])),
                    "soft_high_recall": float(
                        np.mean([m["soft_high_recall"] for m in fold_metrics])
                    ),
                    "soft_high_precision": float(
                        np.mean([m["soft_high_precision"] for m in fold_metrics])
                    ),
                    "soft_high_false_positive_rate": float(
                        np.mean([m["soft_high_false_positive_rate"] for m in fold_metrics])
                    ),
                    "n_inner_folds": int(n_splits),
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    eligible = candidates.loc[
        candidates["soft_high_false_positive_rate"].le(SOFT_FPR_CAP + 1.0e-12)
    ].copy()
    if eligible.empty:
        eligible = candidates.copy()
    selected = eligible.sort_values(
        [
            "soft_f0_5",
            "soft_high_recall",
            "soft_high_false_positive_rate",
            "threshold",
            "penalty",
        ],
        ascending=[False, False, True, False, False],
    ).iloc[0]
    params = {
        "penalty": float(selected["penalty"]),
        "threshold": float(selected["threshold"]),
        "inner_soft_f0_5": float(selected["soft_f0_5"]),
        "inner_soft_high_recall": float(selected["soft_high_recall"]),
        "inner_soft_high_false_positive_rate": float(
            selected["soft_high_false_positive_rate"]
        ),
    }
    candidates["selected"] = (
        candidates["penalty"].eq(params["penalty"])
        & candidates["threshold"].eq(params["threshold"])
    )
    return params, candidates


def _run_loco(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    records = data["records"]
    labels = data["labels"]
    high_ids = labels.index[labels["q_high_off"].ge(HIGH_THRESHOLD)].astype(str)
    high_groups = sorted(
        records.loc[records["record_id"].isin(high_ids), "cv_group_id"].astype(str).unique()
    )
    loco_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    selection_rows: list[pd.DataFrame] = []
    for index, group in enumerate(high_groups):
        print(f"[LOCO {index + 1}/{len(high_groups)}] leaving group {group}", flush=True)
        test = records.loc[records["cv_group_id"].eq(group)].reset_index(drop=True)
        train = records.loc[~records["cv_group_id"].eq(group)].reset_index(drop=True)
        params, candidates = _select_inner_parameters(
            train, labels, outer_id=f"loco_{group}"
        )
        selection_rows.append(candidates)
        fitted, engineered_train, engineered_test = _fit_high_branch(
            train,
            test,
            labels,
            probability_penalty=params["penalty"],
            policy_penalty=params["penalty"],
        )
        target = labels.loc[test["record_id"].tolist(), "q_high_off"].to_numpy(float)
        score = np.asarray(fitted["baseline_policy_score"], dtype=float)
        metrics_all = _metrics(target, score, params["threshold"])
        focus = target >= HIGH_THRESHOLD
        metrics_focus = _metrics(target[focus], score[focus], params["threshold"]) if focus.any() else {}
        loco_rows.append(
            {
                "held_out_group": group,
                "n_test_records": int(len(test)),
                "n_high_slope_records": int(focus.sum()),
                "n_train_records": int(len(train)),
                "n_train_groups": int(train["cv_group_id"].nunique()),
                "selected_penalty": params["penalty"],
                "selected_threshold": params["threshold"],
                "inner_soft_f0_5": params["inner_soft_f0_5"],
                "inner_soft_high_recall": params["inner_soft_high_recall"],
                "inner_soft_high_false_positive_rate": params[
                    "inner_soft_high_false_positive_rate"
                ],
                "all_soft_log_loss": metrics_all["soft_log_loss"],
                "all_soft_f0_5": metrics_all["soft_f0_5"],
                "all_soft_high_recall": metrics_all["soft_high_recall"],
                "all_soft_high_precision": metrics_all["soft_high_precision"],
                "all_soft_high_false_positive_rate": metrics_all[
                    "soft_high_false_positive_rate"
                ],
                "all_mean_margin": metrics_all["mean_margin"],
                "high_focus_soft_log_loss": metrics_focus.get("soft_log_loss", np.nan),
                "high_focus_recall": metrics_focus.get("soft_high_recall", np.nan),
                "high_focus_mean_margin": metrics_focus.get("mean_margin", np.nan),
                "high_focus_predicted_count": metrics_focus.get("predicted_high_count", np.nan),
            }
        )
        rows = _perturbation_rows(
            fitted,
            engineered_train,
            engineered_test,
            target,
            params["threshold"],
            split_kind="loco",
            split_id=group,
            seed=SEED + index * 10000,
            n_permutations=N_PERMUTATIONS_LOCO,
        )
        for row in rows:
            row.update(
                {
                    "held_out_group": group,
                    "selected_penalty": params["penalty"],
                    "n_high_slope_records": int(focus.sum()),
                }
            )
        feature_rows.extend(rows)
    features = pd.DataFrame(feature_rows)
    features["rank_high_focus"] = np.nan
    for group, indices in features.groupby("held_out_group").groups.items():
        subset = features.loc[indices].copy()
        values = subset["importance_soft_log_loss_high_focus"].astype(float)
        valid = values.notna()
        ranks = values[valid].rank(method="min", ascending=False)
        features.loc[ranks.index, "rank_high_focus"] = ranks
    return {
        "group_results": pd.DataFrame(loco_rows),
        "feature_results": features,
        "inner_selection": pd.concat(selection_rows, ignore_index=True),
        "high_slope_registry": records.loc[
            records["record_id"].isin(high_ids),
            ["record_id", "cv_group_id", "sample_id", "curve_id"],
        ].merge(
            labels.reset_index(drop=True)[["record_id"] + Q_COLUMNS],
            on="record_id",
            how="left",
        ),
    }


def _run_repeated_cv(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    records = data["records"]
    labels = data["labels"]
    outer = data["outer"]
    policy_selection = data["policy_selection"]
    probability_selection = data["probability_selection"]
    policy_selection = policy_selection.set_index(["repeat", "fold"])
    probability_selection = probability_selection.set_index(["repeat", "fold"])
    prediction_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    split_groups = list(outer.groupby(["repeat", "fold"], sort=True))
    for split_index, ((repeat, fold), split) in enumerate(split_groups):
        test_ids = split["record_id"].astype(str).tolist()
        test = records.loc[records["record_id"].isin(test_ids)].copy()
        train = records.loc[~records["record_id"].isin(test_ids)].copy()
        if set(train["cv_group_id"]) & set(test["cv_group_id"]):
            raise ValueError("Connected group crosses repeated-CV split")
        prob_penalty = float(probability_selection.loc[(repeat, fold), "selected_penalty"])
        policy_penalty = float(
            policy_selection.loc[(repeat, fold), "selected_policy_penalty"]
        )
        threshold = float(policy_selection.loc[(repeat, fold), "selected_high_threshold"])
        fitted, engineered_train, engineered_test = _fit_high_branch(
            train.reset_index(drop=True),
            test.reset_index(drop=True),
            labels,
            probability_penalty=prob_penalty,
            policy_penalty=policy_penalty,
        )
        target = labels.loc[test["record_id"].tolist(), "q_high_off"].to_numpy(float)
        score = np.asarray(fitted["baseline_policy_score"], dtype=float)
        metrics_all = _metrics(target, score, threshold)
        focus = target >= HIGH_THRESHOLD
        metrics_focus = _metrics(target[focus], score[focus], threshold) if focus.any() else {}
        split_rows.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "n_test_records": int(len(test)),
                "n_high_slope_records": int(focus.sum()),
                "selected_probability_penalty": prob_penalty,
                "selected_policy_penalty": policy_penalty,
                "selected_threshold": threshold,
                "all_soft_log_loss": metrics_all["soft_log_loss"],
                "all_soft_f0_5": metrics_all["soft_f0_5"],
                "all_soft_high_recall": metrics_all["soft_high_recall"],
                "all_soft_high_false_positive_rate": metrics_all[
                    "soft_high_false_positive_rate"
                ],
                "high_focus_soft_log_loss": metrics_focus.get("soft_log_loss", np.nan),
                "high_focus_recall": metrics_focus.get("soft_high_recall", np.nan),
                "high_focus_mean_margin": metrics_focus.get("mean_margin", np.nan),
            }
        )
        for record_id, group_id, y_value, s_value, m_value in zip(
            test["record_id"].astype(str),
            test["cv_group_id"].astype(str),
            target,
            score,
            score - threshold,
        ):
            prediction_rows.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "record_id": record_id,
                    "cv_group_id": group_id,
                    "q_high_off": float(y_value),
                    "high_slope_record": bool(y_value >= HIGH_THRESHOLD),
                    "high_score": float(s_value),
                    "selected_threshold": threshold,
                    "raw_threshold_margin": float(m_value),
                    "predicted_high": bool(m_value >= 0.0),
                }
            )
        rows = _perturbation_rows(
            fitted,
            engineered_train,
            engineered_test,
            target,
            threshold,
            split_kind="repeated_cv",
            split_id=f"r{int(repeat)}_f{int(fold)}",
            seed=SEED + 100000 + split_index * 10000,
            n_permutations=N_PERMUTATIONS_REPEAT,
        )
        for row in rows:
            row.update(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "selected_penalty": policy_penalty,
                }
            )
        feature_rows.extend(rows)
        if (split_index + 1) % 5 == 0:
            print(f"[repeated CV] completed {split_index + 1}/{len(split_groups)} splits", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    split_metrics = pd.DataFrame(split_rows)
    features = pd.DataFrame(feature_rows)
    repeat_rows: list[dict[str, Any]] = []
    repeat_feature_rows: list[dict[str, Any]] = []
    for repeat, frame in predictions.groupby("repeat", sort=True):
        target = frame["q_high_off"].to_numpy(float)
        score = frame["high_score"].to_numpy(float)
        threshold = float(frame["selected_threshold"].mean())
        predicted = frame["predicted_high"].to_numpy(bool)
        metrics_all = HG._soft_high_metrics(target, predicted)
        metrics_all.update(
            {
                "soft_log_loss": _binary_log_loss(target, score),
                "mean_score": float(np.mean(score)),
                "mean_margin": float(frame["raw_threshold_margin"].mean()),
                "n": int(len(frame)),
            }
        )
        focus = frame["high_slope_record"].to_numpy(bool)
        metrics_focus = HG._soft_high_metrics(target[focus], predicted[focus])
        metrics_focus.update(
            {
                "soft_log_loss": _binary_log_loss(target[focus], score[focus]),
                "mean_score": float(np.mean(score[focus])),
                "mean_margin": float(frame.loc[focus, "raw_threshold_margin"].mean()),
                "n": int(focus.sum()),
            }
        )
        repeat_rows.append(
            {
                "repeat": int(repeat),
                "n_records": int(len(frame)),
                "n_high_slope_records": int(focus.sum()),
                "mean_selected_threshold": threshold,
                "mean_selected_policy_penalty": float(
                    split_metrics.loc[
                        split_metrics["repeat"].eq(repeat),
                        "selected_policy_penalty",
                    ].mean()
                ),
                "all_soft_log_loss": metrics_all["soft_log_loss"],
                "all_soft_f0_5": metrics_all["soft_f0_5"],
                "all_soft_high_recall": metrics_all["soft_high_recall"],
                "all_soft_high_precision": metrics_all["soft_high_precision"],
                "all_soft_high_false_positive_rate": metrics_all[
                    "soft_high_false_positive_rate"
                ],
                "high_focus_soft_log_loss": metrics_focus["soft_log_loss"],
                "high_focus_recall": metrics_focus["soft_high_recall"],
                "high_focus_mean_margin": metrics_focus["mean_margin"],
                "high_focus_predicted_count": metrics_focus["predicted_high_count"],
            }
        )
        fframe = features.loc[features["repeat"].eq(repeat)].copy()
        for feature, fsub in fframe.groupby("feature", sort=False):
            valid = fsub["importance_soft_log_loss_high_focus"].notna()
            if not valid.any():
                continue
            weights = fsub.loc[valid, "n_high_focus"].to_numpy(float)
            weights = weights / weights.sum()
            importance = float(
                np.sum(
                    fsub.loc[valid, "importance_soft_log_loss_high_focus"].to_numpy(float)
                    * weights
                )
            )
            margin_change = float(
                np.sum(
                    fsub.loc[valid, "importance_mean_margin_change_high_focus"].to_numpy(float)
                    * weights
                )
            )
            repeat_feature_rows.append(
                {
                    "repeat": int(repeat),
                    "feature": feature,
                    "display_name": fsub["display_name"].iloc[0],
                    "importance_high_focus": importance,
                    "margin_change_high_focus": margin_change,
                    "n_high_focus": int(fsub.loc[valid, "n_high_focus"].sum()),
                    "positive_importance_fraction_high_focus": float(
                        np.sum(
                            fsub.loc[valid, "positive_importance_fraction_high_focus"].to_numpy(float)
                            * weights
                        )
                    ),
                }
            )
    repeat_metrics = pd.DataFrame(repeat_rows)
    repeat_features = pd.DataFrame(repeat_feature_rows)
    repeat_features["rank"] = np.nan
    for repeat, indices in repeat_features.groupby("repeat").groups.items():
        vals = repeat_features.loc[indices, "importance_high_focus"]
        repeat_features.loc[indices, "rank"] = vals.rank(method="min", ascending=False)
    return {
        "predictions": predictions,
        "split_metrics": split_metrics,
        "repeat_metrics": repeat_metrics,
        "feature_split_results": features,
        "feature_repeat_results": repeat_features,
    }


def _rank_summary(repeat_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, frame in repeat_features.groupby("feature", sort=False):
        ranks = frame["rank"].dropna().to_numpy(float)
        importance = frame["importance_high_focus"].dropna().to_numpy(float)
        rows.append(
            {
                "feature": feature,
                "display_name": frame["display_name"].iloc[0],
                "n_repeats": int(len(ranks)),
                "top1_frequency": float(np.mean(ranks == 1)) if len(ranks) else np.nan,
                "top3_frequency": float(np.mean(ranks <= 3)) if len(ranks) else np.nan,
                "median_rank": float(np.median(ranks)) if len(ranks) else np.nan,
                "rank_iqr": float(np.percentile(ranks, 75) - np.percentile(ranks, 25))
                if len(ranks)
                else np.nan,
                "median_importance_high_focus": float(np.median(importance))
                if len(importance)
                else np.nan,
                "importance_iqr": float(
                    np.percentile(importance, 75) - np.percentile(importance, 25)
                )
                if len(importance)
                else np.nan,
                "positive_importance_frequency": float(np.mean(importance > 0))
                if len(importance)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["median_rank", "top3_frequency"], ascending=[True, False]
    )


def _loco_rank_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, frame in features.groupby("feature", sort=False):
        ranks = frame["rank_high_focus"].dropna().to_numpy(float)
        importance = frame["importance_soft_log_loss_high_focus"].dropna().to_numpy(float)
        rows.append(
            {
                "feature": feature,
                "display_name": frame["display_name"].iloc[0],
                "n_loco_groups": int(len(ranks)),
                "top1_frequency": float(np.mean(ranks == 1)) if len(ranks) else np.nan,
                "top3_frequency": float(np.mean(ranks <= 3)) if len(ranks) else np.nan,
                "median_rank": float(np.median(ranks)) if len(ranks) else np.nan,
                "rank_iqr": float(np.percentile(ranks, 75) - np.percentile(ranks, 25))
                if len(ranks)
                else np.nan,
                "median_importance_high_focus": float(np.median(importance))
                if len(importance)
                else np.nan,
                "importance_iqr": float(
                    np.percentile(importance, 75) - np.percentile(importance, 25)
                )
                if len(importance)
                else np.nan,
                "positive_importance_frequency": float(np.mean(importance > 0))
                if len(importance)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["median_rank", "top3_frequency"], ascending=[True, False]
    )
