"""Eight-input logistic baselines and paired connected-group comparisons."""
from __future__ import annotations

import hashlib
import argparse
from repository import ROOT, INPUT_FILES, input_path
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RESULTS = Path.cwd() / 'analysis_runs/baseline'

CLASSES = ["low_60_side", "120_side", "high_off"]
CLASS_DISPLAY = {"low_60_side": "Low", "120_side": "Intermediate", "high_off": "High"}
QCOLS = ["q_low_60_side", "q_120_side", "q_high_off"]
PCOLS = ["p_low_60_side", "p_120_side", "p_high_off"]
MODEL_DISPLAY = {
    "majority_class": "Majority class",
    "training_fold_prior": "Training-fold prior",
    "multinomial_lr": "Multinomial LR",
    "class_balanced_lr": "Class-balanced LR",
    "current_model_argmax": "Current model (argmax)",
    "current_final_policy": "Current final policy",
}
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261021

FEATURES = {
    "catalyst_family": ("catalyst_active_material_type", "categorical", "identity"),
    "log10_catalyst_loading": ("catalyst_loading_mg_cm2_selected", "numeric", "log10"),
    "log10_electrolyte_molarity": ("electrolyte_molarity_M", "numeric", "log10"),
    "log1p_rpm": ("rpm", "numeric", "log1p"),
    "log10_scan_rate": ("scan_rate_mV_s", "numeric", "log10"),
    "ir_correction_status": ("ir_correction_status", "categorical", "identity"),
    "electrolyte_scope": ("electrolyte_scope", "categorical", "identity"),
    "log1p_ionomer_ratio": ("ionomer_to_solvent_ratio", "numeric", "log1p"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    records = pd.read_csv(input_path("records.csv"))
    labels = pd.read_csv(input_path("soft_labels.csv"))
    outer = pd.read_csv(input_path("outer_split_manifest.csv"))
    inner = pd.read_csv(input_path("inner_split_manifest.csv"))
    parent = pd.read_csv(input_path("parent_oof_probabilities.csv"))
    policy = pd.read_csv(input_path("parent_policy_record_aggregated.csv"))
    for frame in (records, labels, outer, inner, parent, policy):
        frame["record_id"] = frame["record_id"].astype(int)
    if len(records) != 97 or labels["record_id"].nunique() != 97:
        raise AssertionError("Expected exactly 97 records")
    if labels["cv_group_id"].nunique() != 80:
        raise AssertionError("Expected exactly 80 connected groups")
    if len(outer) != 970 or outer.groupby("record_id").size().nunique() != 1:
        raise AssertionError("Outer manifest must contain 10 OOF appearances per record")
    return records, labels, outer, inner, parent, policy


def engineer(records: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"record_id": records["record_id"].astype(int)})
    for target, (source, kind, transform) in FEATURES.items():
        series = records[source] if source in records else pd.Series(np.nan, index=records.index)
        if kind == "categorical":
            out[target] = series.astype("string").fillna("missing").replace({"": "missing"})
        else:
            x = pd.to_numeric(series, errors="coerce")
            if transform == "log10":
                transformed = pd.Series(np.nan, index=x.index, dtype=float)
                valid = x > 0
                transformed.loc[valid] = np.log10(x.loc[valid])
                out[target] = transformed
            elif transform == "log1p":
                out[target] = np.where(x >= 0, np.log1p(x), np.nan)
            else:
                out[target] = x
    return out


def make_preprocessor() -> ColumnTransformer:
    numeric = [k for k, (_, kind, _) in FEATURES.items() if kind == "numeric"]
    categorical = [k for k, (_, kind, _) in FEATURES.items() if kind == "categorical"]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3))]), categorical),
        ],
        remainder="drop",
    )


def soft_log_loss(q: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-15, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return float(np.mean(-np.sum(np.asarray(q, float) * np.log(p), axis=1)))


def macro_soft_log_loss(q: np.ndarray, p: np.ndarray) -> float:
    """Class-balanced soft cross-entropy, with each posterior class mass equal-weighted."""
    q = np.asarray(q, float)
    p = np.clip(np.asarray(p, float), 1e-15, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    mass = np.maximum(q.sum(axis=0), 1e-15)
    return float(np.mean(np.sum(q * -np.log(p), axis=0) / mass))


def brier(q: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((np.asarray(p) - np.asarray(q)) ** 2, axis=1)))


def fit_soft_lr(X: pd.DataFrame, q: np.ndarray, C: float, balanced: bool):
    prep = make_preprocessor()
    Xt = prep.fit_transform(X)
    y_expanded = np.tile(np.arange(3), len(X))
    X_expanded = np.repeat(Xt.toarray() if hasattr(Xt, "toarray") else Xt, 3, axis=0)
    weights = q.reshape(-1).copy()
    if balanced:
        mass = np.maximum(q.sum(axis=0), 1e-12)
        class_factor = q.sum() / (3.0 * mass)
        weights *= np.tile(class_factor, len(X))
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=5000, random_state=0)
    model.fit(X_expanded, y_expanded, sample_weight=weights)
    return prep, model


def predict_soft_lr(fitted, X: pd.DataFrame) -> np.ndarray:
    prep, model = fitted
    Xt = prep.transform(X)
    p = model.predict_proba(Xt)
    ordered = np.zeros((len(X), 3), float)
    for j, cls in enumerate(model.classes_):
        ordered[:, int(cls)] = p[:, j]
    return ordered / ordered.sum(axis=1, keepdims=True)


def select_c(X_all, labels_idx, train_ids, inner_rows, balanced):
    scores = {c: [] for c in C_GRID}
    train_set = set(train_ids)
    for inner_fold in sorted(inner_rows["inner_fold"].unique()):
        val_ids = inner_rows.loc[inner_rows["inner_fold"].eq(inner_fold), "record_id"].astype(int).tolist()
        val_ids = [x for x in val_ids if x in train_set]
        fit_ids = sorted(train_set.difference(val_ids))
        if not fit_ids or not val_ids:
            continue
        X_fit = X_all.loc[fit_ids]
        X_val = X_all.loc[val_ids]
        q_fit = labels_idx.loc[fit_ids, QCOLS].to_numpy(float)
        q_val = labels_idx.loc[val_ids, QCOLS].to_numpy(float)
        for c in C_GRID:
            fitted = fit_soft_lr(X_fit, q_fit, c, balanced)
            pred = predict_soft_lr(fitted, X_val)
            selection_loss = macro_soft_log_loss(q_val, pred) if balanced else soft_log_loss(q_val, pred)
            scores[c].append(selection_loss)
    means = {c: float(np.mean(v)) for c, v in scores.items() if v}
    selected = min(means, key=lambda c: (means[c], c))
    return selected, means


def run_logistic_oof(records, labels, outer, inner):
    features = engineer(records).set_index("record_id").sort_index()
    labels_idx = labels.set_index("record_id").sort_index()
    all_ids = set(labels_idx.index.astype(int))
    predictions = []
    selections = []
    for repeat in sorted(outer["repeat"].unique()):
        for fold in sorted(outer["fold"].unique()):
            test_rows = outer.loc[outer["repeat"].eq(repeat) & outer["fold"].eq(fold)]
            test_ids = test_rows["record_id"].astype(int).tolist()
            train_ids = sorted(all_ids.difference(test_ids))
            inner_rows = inner.loc[inner["outer_repeat"].eq(repeat) & inner["outer_fold"].eq(fold)]
            for model_id, balanced in (("multinomial_lr", False), ("class_balanced_lr", True)):
                selected_c, cv_scores = select_c(features, labels_idx, train_ids, inner_rows, balanced)
                fitted = fit_soft_lr(features.loc[train_ids], labels_idx.loc[train_ids, QCOLS].to_numpy(float), selected_c, balanced)
                p = predict_soft_lr(fitted, features.loc[test_ids])
                for rid, probs in zip(test_ids, p):
                    row = {"repeat": repeat, "fold": fold, "record_id": rid,
                           "cv_group_id": labels_idx.loc[rid, "cv_group_id"], "model_id": model_id}
                    row.update(dict(zip(PCOLS, probs)))
                    predictions.append(row)
                selections.append({"repeat": repeat, "fold": fold, "model_id": model_id,
                                   "selected_C": selected_c,
                                   **{f"inner_loss_C_{c:g}": cv_scores.get(c, np.nan) for c in C_GRID}})
    return pd.DataFrame(predictions), pd.DataFrame(selections)


def aggregate_oof(oof: pd.DataFrame, labels_idx: pd.DataFrame) -> pd.DataFrame:
    agg = oof.groupby(["model_id", "record_id", "cv_group_id"], as_index=False)[PCOLS].mean()
    q = labels_idx.reset_index()[["record_id"] + QCOLS + ["threeclass_modal_label"]]
    return agg.merge(q, on="record_id", how="left", validate="many_to_one")


def metric_row(model_id, frame, decision_override=None, has_probabilities=True):
    q = frame[QCOLS].to_numpy(float)
    true = np.argmax(q, axis=1)
    p = frame[PCOLS].to_numpy(float) if has_probabilities else None
    pred = np.asarray(decision_override, int) if decision_override is not None else np.argmax(p, axis=1)
    recalls, precisions, f1s = [], [], []
    for k in range(3):
        tp = np.sum((true == k) & (pred == k)); fn = np.sum((true == k) & (pred != k)); fp = np.sum((true != k) & (pred == k))
        rec = tp / (tp + fn) if tp + fn else np.nan
        pre = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
        recalls.append(rec); precisions.append(pre); f1s.append(f1)
    row = {
        "model_id": model_id, "model": MODEL_DISPLAY[model_id], "n_records": len(frame),
        "accuracy": float(np.mean(pred == true)),
        "balanced_accuracy": float(np.mean(recalls)) if np.all(np.isfinite(recalls)) else np.nan,
        "macro_f1": float(np.mean(f1s)) if np.all(np.isfinite(recalls)) else np.nan,
    }
    for k, display in enumerate(("low", "intermediate", "high")):
        row[f"{display}_precision"] = precisions[k]; row[f"{display}_recall"] = recalls[k]; row[f"{display}_f1"] = f1s[k]
    if has_probabilities:
        row.update({"soft_log_loss": soft_log_loss(q, p), "multiclass_brier": brier(q, p),
                    "posterior_expected_hard_accuracy": float(np.mean(q[np.arange(len(q)), pred])),
                    "high_pr_auc": (float(average_precision_score((true == 2).astype(int), p[:, 2]))
                                     if np.any(true == 2) and np.any(true != 2) else np.nan)})
    else:
        row.update({"soft_log_loss": np.nan, "multiclass_brier": np.nan,
                    "posterior_expected_hard_accuracy": float(np.mean(q[np.arange(len(q)), pred])), "high_pr_auc": np.nan})
    return row


def confusion_rows(model_id, frame, pred):
    true = np.argmax(frame[QCOLS].to_numpy(float), axis=1)
    rows = []
    for i, tname in enumerate(("Low", "Intermediate", "High")):
        denom = max(1, int(np.sum(true == i)))
        for j, pname in enumerate(("Low", "Intermediate", "High")):
            n = int(np.sum((true == i) & (pred == j)))
            rows.append({"model_id": model_id, "model": MODEL_DISPLAY[model_id], "true_class": tname,
                         "predicted_class": pname, "n": n, "row_fraction": n / denom})
    return rows


def build_model_frames(log_oof, parent, policy, labels):
    labels_idx = labels.set_index("record_id").sort_index()
    frames = {}
    frames.update({m: f.reset_index(drop=True) for m, f in aggregate_oof(log_oof, labels_idx).groupby("model_id")})
    parent = parent.loc[parent["evaluation_model"].isin(["fold_local_contiguous_threeclass_soft_prior", "presto_high_gate_direct_gower_conditional"])].copy()
    parent["model_id"] = parent["evaluation_model"].map({"fold_local_contiguous_threeclass_soft_prior": "training_fold_prior",
                                                           "presto_high_gate_direct_gower_conditional": "current_model_argmax"})
    frames.update({m: f.reset_index(drop=True) for m, f in aggregate_oof(parent[["repeat","fold","record_id","cv_group_id","model_id"] + PCOLS], labels_idx).groupby("model_id")})
    base = labels[["record_id", "cv_group_id"] + QCOLS + ["threeclass_modal_label"]].copy().sort_values("record_id").reset_index(drop=True)
    majority = base.copy(); majority[PCOLS] = np.array([1.0, 0.0, 0.0]); frames["majority_class"] = majority
    pol = policy.loc[policy["decision_policy"].eq("presto_policy_only_soft_f05_mean_margin_v2")].copy().sort_values("record_id").reset_index(drop=True)
    policy_frame = base.merge(pol[["record_id", "predicted_class_index"] + PCOLS], on="record_id", validate="one_to_one")
    frames["current_final_policy"] = policy_frame
    return frames


def group_bootstrap(frames, decisions):
    base = frames["majority_class"]
    reference_ids = base["record_id"].to_numpy(int)
    for model_id, frame in frames.items():
        if not np.array_equal(frame["record_id"].to_numpy(int), reference_ids):
            raise AssertionError(f"Record order mismatch before paired bootstrap: {model_id}")
    groups = base["cv_group_id"].drop_duplicates().to_numpy()
    group_to_indices = {g: np.flatnonzero(base["cv_group_id"].to_numpy() == g) for g in groups}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for b in range(BOOTSTRAP_REPS):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([group_to_indices[g] for g in sampled])
        for model_id, frame in frames.items():
            sub = frame.iloc[idx].reset_index(drop=True)
            override = decisions[model_id][idx] if model_id in decisions else None
            row = metric_row(model_id, sub, override, model_id != "majority_class")
            rows.append({"bootstrap": b, **row})
    boot = pd.DataFrame(rows)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "low_recall", "intermediate_recall", "high_recall",
               "soft_log_loss", "multiclass_brier", "posterior_expected_hard_accuracy", "high_pr_auc"]
    summary = []
    for (model_id, model), z in boot.groupby(["model_id", "model"]):
        for metric in metrics:
            values = z[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            summary.append({"model_id": model_id, "model": model, "metric": metric,
                            "bootstrap_mean": float(np.mean(values)), "ci95_low": float(np.quantile(values, .025)),
                            "ci95_high": float(np.quantile(values, .975)), "n_bootstrap_valid": len(values)})
    return boot, pd.DataFrame(summary)


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULTS)
    args = parser.parse_args()
    RESULTS = args.output
    RESULTS.mkdir(parents=True, exist_ok=True)
    records, labels, outer, inner, parent, policy = load_inputs()
    counts = (labels["threeclass_modal_label"].map(CLASS_DISPLAY).value_counts()
              .reindex(["Low", "Intermediate", "High"]).rename_axis("class").reset_index(name="n_records"))
    counts["fraction"] = counts["n_records"] / counts["n_records"].sum()
    counts.to_csv(RESULTS / "class_distribution.csv", index=False)

    log_oof, selections = run_logistic_oof(records, labels, outer, inner)
    log_oof.to_csv(RESULTS / "logistic_oof_predictions.csv", index=False)
    selections.to_csv(RESULTS / "logistic_inner_selection.csv", index=False)

    frames = build_model_frames(log_oof, parent, policy, labels)
    metrics, confusion, decisions = [], [], {}
    for model_id, frame in frames.items():
        if model_id == "current_final_policy":
            pred = frame["predicted_class_index"].to_numpy(int)
        elif model_id == "majority_class":
            pred = np.zeros(len(frame), int)
        else:
            pred = np.argmax(frame[PCOLS].to_numpy(float), axis=1)
        decisions[model_id] = pred
        metrics.append(metric_row(model_id, frame, pred, model_id != "majority_class"))
        confusion.extend(confusion_rows(model_id, frame, pred))
    metrics = pd.DataFrame(metrics)
    order = list(MODEL_DISPLAY)
    metrics["_order"] = metrics["model_id"].map({m:i for i,m in enumerate(order)})
    metrics = metrics.sort_values("_order").drop(columns="_order")
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(RESULTS / "confusion_matrices.csv", index=False)

    boot, boot_summary = group_bootstrap(frames, decisions)
    boot.to_csv(RESULTS / "group_bootstrap_metrics.csv", index=False)
    boot_summary.to_csv(RESULTS / "group_bootstrap_summary.csv", index=False)

    agg_records = []
    for model_id, frame in frames.items():
        z = frame[["record_id", "cv_group_id"] + QCOLS + PCOLS].copy()
        z.insert(0, "model", MODEL_DISPLAY[model_id]); z.insert(0, "model_id", model_id)
        z["predicted_class_index"] = decisions[model_id]
        agg_records.append(z)
    pd.concat(agg_records, ignore_index=True).to_csv(RESULTS / "record_level_predictions.csv", index=False)

    hashes = {name: sha256(input_path(name)) for name in ['records.csv','soft_labels.csv','outer_split_manifest.csv','inner_split_manifest.csv','parent_oof_probabilities.csv','parent_policy_record_aggregated.csv']}
    provenance = {"n_logical_inputs": 8, "records": 97, "connected_groups": 80,
                  "outer_validation": "frozen 10 repeats x 5 connected-group folds",
                  "bootstrap": {"unit": "connected_group", "repetitions": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
                  "logistic_target": "three-component posterior soft labels; no posterior argmax used for training",
                  "balanced_training": "inverse training-fold soft class mass; sensitivity analysis only",
                  "C_grid": C_GRID, "input_sha256": hashes,
                  "python": sys.version, "platform": platform.platform()}
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))

if __name__ == '__main__':
    main()
