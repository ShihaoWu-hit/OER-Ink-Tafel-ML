"""Summarize paired connected-group bootstrap contrasts from existing bootstrap rows."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

R = Path.cwd() / 'analysis_runs/baseline'

LOWER = {"soft_log_loss", "multiclass_brier"}
PAIRS = [
    ("multinomial_lr", "training_fold_prior"),
    ("class_balanced_lr", "training_fold_prior"),
    ("current_model_argmax", "training_fold_prior"),
    ("current_final_policy", "majority_class"),
    ("current_final_policy", "training_fold_prior"),
    ("current_final_policy", "multinomial_lr"),
    ("current_final_policy", "class_balanced_lr"),
]
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "low_recall",
    "intermediate_recall",
    "high_recall",
    "soft_log_loss",
    "multiclass_brier",
    "posterior_expected_hard_accuracy",
    "high_pr_auc",
]


def main():
    global R
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=R)
    R = parser.parse_args().results
    boot = pd.read_csv(R / "group_bootstrap_metrics.csv")
    rows = []
    for challenger, reference in PAIRS:
        a = boot.loc[boot.model_id.eq(challenger)].set_index("bootstrap")
        b = boot.loc[boot.model_id.eq(reference)].set_index("bootstrap")
        for metric in METRICS:
            delta = (a[metric] - b[metric]).dropna().to_numpy(float)
            if not len(delta):
                continue
            rows.append(
                {
                    "challenger": challenger,
                    "reference": reference,
                    "metric": metric,
                    "delta_mean": np.mean(delta),
                    "ci95_low": np.quantile(delta, 0.025),
                    "ci95_high": np.quantile(delta, 0.975),
                    "favorable_probability": np.mean(
                        delta < 0 if metric in LOWER else delta > 0
                    ),
                    "n_bootstrap_valid": len(delta),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(R / "paired_group_bootstrap_contrasts.csv", index=False)

    probability_keys = (
        out.challenger.eq("current_model_argmax")
        & out.reference.eq("training_fold_prior")
        & out.metric.isin(["soft_log_loss", "multiclass_brier"])
    )
    majority_keys = (
        out.challenger.eq("current_final_policy")
        & out.reference.eq("majority_class")
        & out.metric.isin(["balanced_accuracy", "macro_f1"])
    )
    logistic_keys = (
        out.challenger.eq("current_final_policy")
        & out.reference.isin(["multinomial_lr", "class_balanced_lr"])
        & out.metric.isin(["balanced_accuracy", "macro_f1"])
    )
    selected = out.loc[probability_keys | majority_keys | logistic_keys]
    selected.to_csv(R / "key_paired_contrasts.csv", index=False)
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
