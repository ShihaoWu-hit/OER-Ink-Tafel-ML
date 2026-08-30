"""Auditable entropy-based abstention for cardinal-region predictions.

Abstention is a reporting policy applied to record-aggregated outer-OOF
probabilities.  It does not change model fitting, model selection, or the
full-coverage primary log-loss/Brier scores.  The current executable policy
uses a fixed normalized Shannon-entropy threshold registered before the run;
outer-test labels never choose the threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .cardinal_metrics import (
    CLASS_LABELS,
    PREDICTION_COLUMNS,
    SOFT_LABEL_COLUMNS,
    normalize_probabilities,
    two_stage_probability_metrics,
)
from .provenance import write_json


ABSTENTION_POLICY_ID = "fixed_normalized_predictive_entropy_v1"
FIXED_THRESHOLD_SOURCE = "pre_registered_fixed_no_outer_test_label_use"
DEFAULT_NORMALIZED_ENTROPY_THRESHOLD = 0.90
DEFAULT_COVERAGE_CURVE_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95, 1.00)

_SUMMARY_KEYS = (
    "analysis_id",
    "scope",
    "panel_role",
    "panel",
    "model_role",
    "model_family",
)


def _validate_threshold(value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError(
            "normalized_entropy_threshold must be finite and strictly "
            "between zero and one"
        )
    return threshold


def normalized_predictive_entropy(
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Shannon entropy in nats and normalized to the interval [0, 1]."""

    predicted = normalize_probabilities(probabilities)
    n_classes = int(predicted.shape[1])
    if n_classes < 2:
        raise ValueError("Predictive entropy requires at least two classes")
    entropy_nats = -np.sum(
        np.where(
            predicted > 0.0,
            predicted * np.log(np.clip(predicted, 1e-300, 1.0)),
            0.0,
        ),
        axis=1,
    )
    normalized = entropy_nats / np.log(float(n_classes))
    return entropy_nats, np.clip(normalized, 0.0, 1.0)


def entropy_abstention_decisions(
    oof_aggregated: pd.DataFrame,
    *,
    normalized_entropy_threshold: float = DEFAULT_NORMALIZED_ENTROPY_THRESHOLD,
    threshold_source: str = FIXED_THRESHOLD_SOURCE,
) -> pd.DataFrame:
    """Apply a fixed entropy policy to record-aggregated OOF probabilities.

    ``predicted_class_without_abstention`` is retained for audit only.
    ``reported_decision`` is the reader-facing output and is ``uncertain``
    whenever the fixed threshold is exceeded.
    """

    threshold = _validate_threshold(normalized_entropy_threshold)
    if str(threshold_source) != FIXED_THRESHOLD_SOURCE:
        raise ValueError(
            "This implementation accepts only the preregistered fixed "
            "threshold source; inner-OOF threshold selection requires a "
            "separate fold-indexed threshold table"
        )
    if oof_aggregated.empty:
        raise ValueError("Cannot apply abstention to an empty OOF table")
    required = {
        "record_id",
        *SOFT_LABEL_COLUMNS,
        *PREDICTION_COLUMNS,
    }
    missing = sorted(required - set(oof_aggregated.columns))
    if missing:
        raise KeyError(f"OOF table is missing abstention columns: {missing}")
    if "n_repeated_oof" not in oof_aggregated.columns:
        raise KeyError(
            "Abstention requires record-aggregated OOF input with "
            "n_repeated_oof"
        )

    result = oof_aggregated.copy()
    probabilities = normalize_probabilities(
        result.loc[:, PREDICTION_COLUMNS].to_numpy(float)
    )
    result.loc[:, PREDICTION_COLUMNS] = probabilities
    entropy_nats, normalized_entropy = normalized_predictive_entropy(
        probabilities
    )
    predicted_index = np.argmax(probabilities, axis=1)
    predicted_class = np.asarray(CLASS_LABELS, dtype=object)[predicted_index]
    accepted = normalized_entropy <= threshold

    result["predictive_entropy_nats"] = entropy_nats
    result["normalized_predictive_entropy"] = normalized_entropy
    result["effective_number_of_classes"] = np.exp(entropy_nats)
    result["maximum_predicted_probability"] = probabilities.max(axis=1)
    result["normalized_entropy_threshold"] = threshold
    result["threshold_source"] = str(threshold_source)
    result["threshold_selection_used_outer_test_data"] = False
    result["threshold_used_outer_test_labels"] = False
    result["accepted"] = accepted
    result["abstained"] = ~accepted
    result["predicted_class_without_abstention"] = predicted_class
    result["reported_decision"] = np.where(
        accepted, predicted_class, "uncertain"
    )
    return result


def abstention_summary(
    decisions: pd.DataFrame,
    *,
    group_columns: Sequence[str] = _SUMMARY_KEYS,
) -> pd.DataFrame:
    """Summarize coverage and selective probability metrics.

    Full-coverage scores are included as an audit copy.  They remain the
    primary performance estimates; accepted-only scores are conditional on
    coverage and must not replace them.
    """

    required = {
        "accepted",
        "abstained",
        "normalized_predictive_entropy",
        *SOFT_LABEL_COLUMNS,
        *PREDICTION_COLUMNS,
    }
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise KeyError(
            f"Abstention decisions are missing summary columns: {missing}"
        )
    keys = tuple(column for column in group_columns if column in decisions)
    if not keys:
        raise KeyError("No abstention summary grouping columns are available")

    rows: list[dict[str, object]] = []
    for values, frame in decisions.groupby(
        list(keys), dropna=False, sort=False
    ):
        if not isinstance(values, tuple):
            values = (values,)
        observed = frame.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
        predicted = frame.loc[:, PREDICTION_COLUMNS].to_numpy(float)
        accepted = frame["accepted"].astype(bool).to_numpy()
        full_metrics = two_stage_probability_metrics(observed, predicted)
        accepted_metrics = (
            two_stage_probability_metrics(
                observed[accepted], predicted[accepted]
            )
            if accepted.any()
            else {}
        )
        row: dict[str, object] = {
            **dict(zip(keys, values, strict=True)),
            "n_total": int(len(frame)),
            "n_accepted": int(accepted.sum()),
            "n_uncertain": int((~accepted).sum()),
            "coverage": float(accepted.mean()),
            "uncertain_rate": float((~accepted).mean()),
            "mean_normalized_predictive_entropy": float(
                frame["normalized_predictive_entropy"].mean()
            ),
            "mean_normalized_entropy_accepted": (
                float(
                    frame.loc[
                        frame["accepted"],
                        "normalized_predictive_entropy",
                    ].mean()
                )
                if accepted.any()
                else float("nan")
            ),
            "normalized_entropy_threshold": float(
                frame["normalized_entropy_threshold"].iloc[0]
            ),
            "threshold_source": str(frame["threshold_source"].iloc[0]),
            "full_coverage_primary_metrics_unchanged": True,
            "accepted_only_metrics_are_selective": True,
        }
        row.update(
            {
                f"full_coverage_{metric}": float(value)
                for metric, value in full_metrics.items()
            }
        )
        for metric in full_metrics:
            row[f"accepted_only_{metric}"] = float(
                accepted_metrics.get(metric, np.nan)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def abstention_coverage_curve(
    oof_aggregated: pd.DataFrame,
    *,
    normalized_entropy_thresholds: Sequence[float] = (
        DEFAULT_COVERAGE_CURVE_THRESHOLDS
    ),
    primary_threshold: float = DEFAULT_NORMALIZED_ENTROPY_THRESHOLD,
) -> pd.DataFrame:
    """Return a preregistered coverage/selective-risk sensitivity curve.

    The complete threshold grid is fixed before fitting and is descriptive.
    No point on this curve may be selected using outer-test performance.
    """

    primary = _validate_threshold(primary_threshold)
    thresholds = tuple(float(value) for value in normalized_entropy_thresholds)
    if not thresholds:
        raise ValueError("Coverage-curve threshold grid must be non-empty")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Coverage-curve threshold grid contains duplicates")
    if any(
        (not np.isfinite(value)) or value <= 0.0 or value > 1.0
        for value in thresholds
    ):
        raise ValueError(
            "Coverage-curve thresholds must be finite and in (0, 1]"
        )
    if not any(np.isclose(value, primary) for value in thresholds):
        raise ValueError(
            "Coverage-curve threshold grid must include the primary threshold"
        )

    base = entropy_abstention_decisions(
        oof_aggregated,
        normalized_entropy_threshold=primary,
        threshold_source=FIXED_THRESHOLD_SOURCE,
    )
    rows: list[pd.DataFrame] = []
    for threshold in thresholds:
        decisions = base.copy()
        accepted = (
            decisions["normalized_predictive_entropy"].to_numpy(float)
            <= threshold
        )
        decisions["normalized_entropy_threshold"] = threshold
        decisions["accepted"] = accepted
        decisions["abstained"] = ~accepted
        decisions["reported_decision"] = np.where(
            accepted,
            decisions["predicted_class_without_abstention"],
            "uncertain",
        )
        summary = abstention_summary(decisions)
        summary["is_primary_threshold"] = bool(
            np.isclose(threshold, primary)
        )
        summary["curve_role"] = (
            "preregistered_primary"
            if np.isclose(threshold, primary)
            else "preregistered_threshold_sensitivity"
        )
        summary["outer_test_metrics_used_to_select_threshold"] = False
        summary["outer_test_data_used_to_select_threshold"] = False
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def audit_entropy_abstention(
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    expected_rows: int,
) -> dict[str, object]:
    """Return machine-readable invariants for an abstention publication."""

    probabilities = normalize_probabilities(
        decisions.loc[:, PREDICTION_COLUMNS].to_numpy(float)
    )
    _, recalculated = normalized_predictive_entropy(probabilities)
    threshold = decisions["normalized_entropy_threshold"].to_numpy(float)
    expected_accepted = recalculated <= threshold
    accepted = decisions["accepted"].astype(bool).to_numpy()
    reported = decisions["reported_decision"].astype(str).to_numpy()
    expected_reported = np.where(
        expected_accepted,
        decisions["predicted_class_without_abstention"].astype(str),
        "uncertain",
    )
    checks = {
        "record_decision_row_count_matches_oof": (
            int(len(decisions)) == int(expected_rows)
        ),
        "all_probabilities_finite_and_normalized": bool(
            np.isfinite(probabilities).all()
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
        ),
        "entropy_recalculation_matches": bool(
            np.allclose(
                decisions["normalized_predictive_entropy"].to_numpy(float),
                recalculated,
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "accepted_decisions_match_fixed_threshold": bool(
            np.array_equal(accepted, expected_accepted)
        ),
        "uncertain_label_applied_to_all_abstentions": bool(
            np.array_equal(reported, expected_reported)
        ),
        "threshold_did_not_use_outer_test_labels": bool(
            (~decisions["threshold_used_outer_test_labels"].astype(bool)).all()
        ),
        "threshold_selection_did_not_use_outer_test_data": bool(
            (
                ~decisions[
                    "threshold_selection_used_outer_test_data"
                ].astype(bool)
            ).all()
        ),
        "threshold_source_is_preregistered_fixed": bool(
            decisions["threshold_source"].astype(str).eq(
                FIXED_THRESHOLD_SOURCE
            ).all()
        ),
        "coverage_is_bounded": bool(
            summary["coverage"].between(0.0, 1.0, inclusive="both").all()
        ),
        "full_coverage_primary_metrics_unchanged": bool(
            summary["full_coverage_primary_metrics_unchanged"]
            .astype(bool)
            .all()
        ),
    }
    return {
        "status": (
            "passed" if all(checks.values()) else "failed"
        ),
        "policy_id": ABSTENTION_POLICY_ID,
        "normalized_entropy_threshold": float(threshold[0]),
        "threshold_source": FIXED_THRESHOLD_SOURCE,
        "n_record_decision_rows": int(len(decisions)),
        "n_summary_rows": int(len(summary)),
        "checks": checks,
    }


def write_entropy_abstention_artifacts(
    oof_aggregated: pd.DataFrame,
    output_dir: str | Path,
    *,
    normalized_entropy_threshold: float = DEFAULT_NORMALIZED_ENTROPY_THRESHOLD,
    coverage_curve_thresholds: Sequence[float] = (
        DEFAULT_COVERAGE_CURVE_THRESHOLDS
    ),
    threshold_source: str = FIXED_THRESHOLD_SOURCE,
) -> dict[str, object]:
    """Write record decisions, selective metrics, definition, and QA."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    decisions = entropy_abstention_decisions(
        oof_aggregated,
        normalized_entropy_threshold=normalized_entropy_threshold,
        threshold_source=threshold_source,
    )
    summary = abstention_summary(decisions)
    coverage_curve = abstention_coverage_curve(
        oof_aggregated,
        normalized_entropy_thresholds=coverage_curve_thresholds,
        primary_threshold=normalized_entropy_threshold,
    )
    qa = audit_entropy_abstention(
        decisions, summary, expected_rows=len(oof_aggregated)
    )
    observed_curve_thresholds = tuple(
        sorted(
            coverage_curve["normalized_entropy_threshold"]
            .astype(float)
            .unique()
            .tolist()
        )
    )
    expected_curve_thresholds = tuple(
        sorted(float(value) for value in coverage_curve_thresholds)
    )
    qa["coverage_curve_thresholds_match_preregistered_grid"] = bool(
        observed_curve_thresholds == expected_curve_thresholds
    )
    qa["coverage_curve_never_selects_threshold_from_outer_test"] = bool(
        (
            ~coverage_curve[
                "outer_test_data_used_to_select_threshold"
            ].astype(bool)
        ).all()
    )
    decisions.to_csv(output / "abstention_record_decisions.csv", index=False)
    summary.to_csv(output / "abstention_summary.csv", index=False)
    coverage_curve.to_csv(output / "abstention_coverage_curve.csv", index=False)
    write_json(
        output / "abstention_definition.json",
        {
            "policy_id": ABSTENTION_POLICY_ID,
            "probability_source": (
                "record_aggregated_repeated_connected_group_outer_oof"
            ),
            "entropy": "-sum_k p_k * ln(p_k)",
            "normalization": "entropy / ln(number_of_classes)",
            "number_of_classes": len(CLASS_LABELS),
            "normalized_entropy_threshold": float(
                normalized_entropy_threshold
            ),
            "preregistered_coverage_curve_thresholds": [
                float(value) for value in coverage_curve_thresholds
            ],
            "acceptance_rule": (
                "normalized_predictive_entropy <= threshold"
            ),
            "abstention_output_label": "uncertain",
            "threshold_source": str(threshold_source),
            "outer_test_predictions_or_labels_used_to_choose_threshold": False,
            "outer_test_labels_used_to_choose_threshold": False,
            "full_coverage_logloss_and_brier_remain_primary": True,
            "accepted_only_metrics_are_coverage_conditional": True,
            "mechanistic_claim_allowed": False,
        },
    )
    write_json(output / "abstention_qa.json", qa)
    if not (
        qa["status"] == "passed"
        and qa["coverage_curve_thresholds_match_preregistered_grid"]
        and qa["coverage_curve_never_selects_threshold_from_outer_test"]
    ):
        raise AssertionError("Entropy-abstention QA failed")
    return {
        "decisions": decisions,
        "summary": summary,
        "coverage_curve": coverage_curve,
        "qa": qa,
        "output_dir": str(output),
    }


__all__ = [
    "ABSTENTION_POLICY_ID",
    "DEFAULT_COVERAGE_CURVE_THRESHOLDS",
    "DEFAULT_NORMALIZED_ENTROPY_THRESHOLD",
    "FIXED_THRESHOLD_SOURCE",
    "abstention_summary",
    "abstention_coverage_curve",
    "audit_entropy_abstention",
    "entropy_abstention_decisions",
    "normalized_predictive_entropy",
    "write_entropy_abstention_artifacts",
]
