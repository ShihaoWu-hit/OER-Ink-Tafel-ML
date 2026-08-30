"""Active all-97 reduced-panel posterior-aware cardinal classification.

This stage is intentionally separate from the historical
``cardinal-classify`` workflow.  It consumes the frozen v2 contract, snapshots
the formal all-97 evidence, derives posterior-soft labels at all three frozen
band widths, and evaluates only the registered reduced panels on shared
connected-group folds.

The primary estimator is a deterministic Gaussian-prior MAP approximation,
not a full Bayesian coefficient posterior.  Posterior slope draws define one
soft response vector per experimental record and are never expanded into ML
rows.  Hard-label accuracy is persisted only as a descriptive diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from .cardinal_cv import (
    make_inner_soft_group_splits,
    make_soft_stratified_group_manifest,
)
from .cardinal_inputs import (
    snapshot_cardinal_inputs,
    verify_cardinal_input_snapshot,
    verify_source_run_unchanged,
)
from .cardinal_labels import (
    DEFAULT_ANCHORS,
    derive_hierarchical_soft_labels,
    derive_posterior_soft_labels,
    hierarchical_support_summary,
    low_anchor_subdivision_table,
    support_summary,
    validate_hierarchical_soft_labels,
)
from .cardinal_metrics import (
    PREDICTION_COLUMNS,
    SOFT_LABEL_COLUMNS,
    calibration_table,
    expected_confusion_table,
    normalize_probabilities,
    soft_cross_entropy,
    two_stage_probability_metrics,
)
from .cardinal_reduced_models import (
    ReducedPanelGowerClassifier,
    ReducedPanelTwoStageMAPClassifier,
)
from .config import StudyConfig
from .feature_association_features import (
    FROZEN_ASSOCIATION_PANELS,
    required_association_source_columns,
)
from .posterior_cardinal_contract import (
    EXPECTED_FORMAL_PANELS,
    EXPECTED_SENSITIVITY_PANELS,
    EXPECTED_WIDTHS,
    load_frozen_posterior_cardinal_contract,
    snapshot_reduced_feature_registries,
    verify_reduced_feature_registry_snapshot,
)
from .posterior_cardinal_curves import (
    execute_released_adjusted_probability_curves,
)
from .posterior_cardinal_inference import (
    connected_group_signflip_pvalues,
    descriptive_probability_metrics,
    holm_one_sided_adjustment,
    paired_connected_group_bootstrap,
)
from .provenance import sha256_file, write_json


PRIMARY_WIDTH = 0.20
CONTEXT_PANEL = "context_core"
FORMAL_PANELS = EXPECTED_FORMAL_PANELS
SENSITIVITY_PANELS = EXPECTED_SENSITIVITY_PANELS
ALL_PANELS = (CONTEXT_PANEL, *FORMAL_PANELS, *SENSITIVITY_PANELS)
PRIMARY_MODEL = "two_stage_gaussian_prior_MAP"
SECONDARY_MODEL = "two_stage_soft_gower_knn"
PRIOR_MODEL = "fold_local_soft_prior"
PRIOR_PANEL = "prior"
IDENTITY_COLUMNS = (
    "record_id",
    "sample_record",
    "curve_id",
    "cv_group_id",
    "electrolyte_scope",
)
_CLI_BOOTSTRAP_PROVENANCE_FILES = frozenset(
    {
        "code_manifest.json",
        "environment_manifest.json",
        "input_manifest.json",
        "resolved_config.yaml",
        "run_mode.json",
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _width_id(width: float) -> str:
    return f"width_{int(round(100 * float(width))):02d}"


def _validate_run_frame(records: pd.DataFrame) -> None:
    missing = sorted(set(IDENTITY_COLUMNS) - set(records.columns))
    if missing:
        raise KeyError(f"Formal records are missing identity columns: {missing}")
    if len(records) != 97:
        raise ValueError(f"Active classification requires 97 records, found {len(records)}")
    if records["record_id"].astype(str).duplicated().any():
        raise ValueError("Active records must be unique by record_id")
    if records["cv_group_id"].astype(str).nunique() != 80:
        raise ValueError(
            "Active classification requires exactly 80 connected groups"
        )
    scope_counts = (
        records["electrolyte_scope"].astype(str).value_counts().to_dict()
    )
    expected = {"alkaline": 90, "acidic": 5, "neutral": 2}
    if scope_counts != expected:
        raise ValueError(
            f"Electrolyte-scope counts changed: expected={expected}, "
            f"observed={scope_counts}"
        )


def _assert_fresh_disjoint_run_directory(
    destination: Path,
    source: Path,
) -> None:
    """Defensively enforce independent, non-stale runner paths.

    The CLI writes a fixed five-file provenance bootstrap before invoking this
    runner, so that exact fresh state is allowed in addition to a new or empty
    directory. Any modeling artifact or unknown file fails closed.
    """

    if (
        destination == source
        or destination in source.parents
        or source in destination.parents
    ):
        raise ValueError(
            "Posterior-cardinal run and formal source paths must be disjoint"
        )
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    if not destination.exists():
        return
    root_entries = list(destination.iterdir())
    if not root_entries:
        return
    if {entry.name for entry in root_entries} != {"provenance"}:
        raise FileExistsError(
            "Posterior-cardinal run directory must be new or contain only "
            "the fresh CLI provenance bootstrap"
        )
    provenance = destination / "provenance"
    if not provenance.is_dir():
        raise FileExistsError("Fresh provenance bootstrap is not a directory")
    entries = list(provenance.iterdir())
    if any(not entry.is_file() for entry in entries):
        raise FileExistsError(
            "Fresh provenance bootstrap may contain files only"
        )
    names = {entry.name for entry in entries}
    if not names.issubset(_CLI_BOOTSTRAP_PROVENANCE_FILES):
        raise FileExistsError(
            "Posterior-cardinal provenance contains stale or unknown files"
        )


def derive_all_width_labels(
    records: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    widths: Sequence[float] = EXPECTED_WIDTHS,
) -> tuple[dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
    """Derive raw five-anchor and collapsed four-class labels for each width."""

    _validate_run_frame(records)
    mapping = records.loc[:, list(IDENTITY_COLUMNS)].copy()
    raw_by_width: dict[float, pd.DataFrame] = {}
    hierarchical_by_width: dict[float, pd.DataFrame] = {}
    for raw_width in widths:
        width = float(raw_width)
        raw = derive_posterior_soft_labels(
            draws,
            mapping,
            anchors=DEFAULT_ANCHORS,
            relative_half_width=width,
            expected_draws_per_record=8_000,
        )
        hierarchical = derive_hierarchical_soft_labels(
            raw,
            anchors=DEFAULT_ANCHORS,
        )
        validate_hierarchical_soft_labels(hierarchical)
        if len(hierarchical) != 97:
            raise AssertionError("Every width must retain all 97 records")
        raw_by_width[width] = raw
        hierarchical_by_width[width] = hierarchical
    if tuple(sorted(raw_by_width)) != tuple(sorted(EXPECTED_WIDTHS)):
        raise ValueError("Label-width set changed from the frozen contract")
    return raw_by_width, hierarchical_by_width


def _merge_primary_labels(
    records: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    selected = labels.loc[
        :, ["record_id", *SOFT_LABEL_COLUMNS]
    ].copy()
    selected["record_id"] = selected["record_id"].astype(str)
    frame = records.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    merged = frame.merge(
        selected,
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    if merged.loc[:, SOFT_LABEL_COLUMNS].isna().any().any():
        raise ValueError("Primary-width labels did not cover all formal records")
    return merged


def build_shared_outer_manifest(
    primary_frame: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    smoke: bool,
) -> pd.DataFrame:
    """Build the one soft-mass-balanced manifest reused everywhere."""

    outer = dict(contract["outer_validation"])
    folds = int(contract["smoke_mode"]["outer_folds"]) if smoke else int(
        outer["folds"]
    )
    repeats = int(contract["smoke_mode"]["outer_repeats"]) if smoke else int(
        outer["repeats"]
    )
    seeds = tuple(int(value) for value in outer["repeat_seeds"])[:repeats]
    manifest = make_soft_stratified_group_manifest(
        primary_frame,
        SOFT_LABEL_COLUMNS,
        group_col="cv_group_id",
        record_id_col="record_id",
        folds=folds,
        repeats=repeats,
        repeat_seeds=seeds,
    )
    manifest.insert(0, "manifest_id", "primary_width_20_shared_v2")
    return manifest


def build_shared_inner_manifest(
    primary_frame: pd.DataFrame,
    outer_manifest: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    smoke: bool,
) -> pd.DataFrame:
    """Create one grouped-inner assignment per outer training set."""

    inner_folds = (
        int(contract["smoke_mode"]["inner_folds"])
        if smoke
        else int(contract["inner_validation"]["folds"])
    )
    rows: list[dict[str, Any]] = []
    for (repeat, fold), outer_test in outer_manifest.groupby(
        ["repeat", "fold"], sort=True
    ):
        test_ids = set(outer_test["record_id"].astype(str))
        train = primary_frame.loc[
            ~primary_frame["record_id"].astype(str).isin(test_ids)
        ].reset_index(drop=True)
        split_seed = int(outer_test["split_seed"].iloc[0]) + 10_000 + int(fold)
        _, assignments = make_inner_soft_group_splits(
            train,
            SOFT_LABEL_COLUMNS,
            group_col="cv_group_id",
            folds=inner_folds,
            random_seed=split_seed,
        )
        for position, inner_fold in enumerate(assignments):
            rows.append(
                {
                    "manifest_id": "primary_width_20_shared_v2",
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "inner_fold": int(inner_fold),
                    "record_id": str(train.iloc[position]["record_id"]),
                    "cv_group_id": str(train.iloc[position]["cv_group_id"]),
                    "inner_split_seed": int(split_seed),
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["outer_repeat", "outer_fold", "inner_fold", "record_id"],
        kind="stable",
    ).reset_index(drop=True)
    expected_outer_splits = outer_manifest[["repeat", "fold"]].drop_duplicates()
    if len(
        manifest[["outer_repeat", "outer_fold"]].drop_duplicates()
    ) != len(expected_outer_splits):
        raise AssertionError("Inner manifest does not cover every outer split")
    if (
        manifest.groupby(
            ["outer_repeat", "outer_fold", "cv_group_id"]
        )["inner_fold"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError("A connected group crosses inner validation folds")
    return manifest


def _inner_splits(
    outer_train: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    selected = inner_manifest.loc[
        inner_manifest["outer_repeat"].eq(int(repeat))
        & inner_manifest["outer_fold"].eq(int(fold))
    ].copy()
    lookup = dict(
        zip(
            selected["record_id"].astype(str),
            selected["inner_fold"].astype(int),
            strict=True,
        )
    )
    assignments = outer_train["record_id"].astype(str).map(lookup)
    if assignments.isna().any():
        raise ValueError("Frozen inner manifest does not cover outer training rows")
    output: list[tuple[np.ndarray, np.ndarray, int]] = []
    for inner_fold in sorted(assignments.astype(int).unique()):
        validation = np.flatnonzero(assignments.to_numpy(int) == inner_fold)
        training = np.flatnonzero(assignments.to_numpy(int) != inner_fold)
        if not len(training) or not len(validation):
            raise AssertionError("Frozen inner split is empty")
        output.append((training, validation, int(inner_fold)))
    return output


@dataclass(frozen=True)
class _SelectedCandidate:
    candidate_id: str
    params: dict[str, Any]
    mean_loss: float
    standard_error: float


def _map_candidate_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray, int]],
    *,
    panel: str,
    scales: Sequence[float],
    rare_min_frequency: int,
    intercept_prior_scale: float,
    max_iter: int,
    tolerance: float,
    context: Mapping[str, Any],
) -> tuple[_SelectedCandidate, list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scale in scales:
        candidate_id = f"map_feature_prior_scale_{float(scale):g}"
        fold_losses: list[float] = []
        failure = ""
        for train_index, validation_index, inner_fold in splits:
            try:
                estimator = ReducedPanelTwoStageMAPClassifier(
                    panel=panel,
                    rare_min_frequency=int(rare_min_frequency),
                    feature_prior_scale=float(scale),
                    intercept_prior_scale=float(intercept_prior_scale),
                    max_iter=int(max_iter),
                    tolerance=float(tolerance),
                ).fit(X.iloc[train_index], y[train_index])
                if estimator.fit_status_ != "complete":
                    raise RuntimeError(
                        "MAP optimizer did not converge: "
                        f"{estimator.optimization_audit_}"
                    )
                predicted = estimator.predict_proba(X.iloc[validation_index])
                loss = soft_cross_entropy(y[validation_index], predicted)
                status = "ok"
                error = ""
                fold_losses.append(float(loss))
            except Exception as exc:  # fail closed and audit the candidate
                loss = np.nan
                status = "failed"
                error = repr(exc)
                failure = error
            detail_rows.append(
                {
                    **context,
                    "model_family": PRIMARY_MODEL,
                    "panel": panel,
                    "candidate_id": candidate_id,
                    "candidate_params": json.dumps(
                        {"feature_prior_scale": float(scale)},
                        sort_keys=True,
                    ),
                    "inner_fold": int(inner_fold),
                    "soft_log_loss": float(loss),
                    "status": status,
                    "error": error,
                }
            )
        complete = len(fold_losses) == len(splits)
        mean_loss = float(np.mean(fold_losses)) if complete else np.inf
        standard_error = (
            float(np.std(fold_losses, ddof=1) / np.sqrt(len(fold_losses)))
            if complete and len(fold_losses) > 1
            else np.inf
        )
        summaries.append(
            {
                "candidate_id": candidate_id,
                "feature_prior_scale": float(scale),
                "mean_loss": mean_loss,
                "standard_error": standard_error,
                "complete": complete,
                "failure": failure,
            }
        )
    valid = [row for row in summaries if row["complete"]]
    if not valid:
        raise RuntimeError(
            f"Every MAP nuisance candidate failed for panel {panel}"
        )
    best = min(valid, key=lambda row: (row["mean_loss"], row["candidate_id"]))
    threshold = float(best["mean_loss"] + best["standard_error"])
    eligible = [row for row in valid if row["mean_loss"] <= threshold]
    selected = min(
        eligible,
        key=lambda row: (row["feature_prior_scale"], row["candidate_id"]),
    )
    for row in detail_rows:
        row["selected_candidate_id"] = str(selected["candidate_id"])
        row["one_se_threshold"] = threshold
    return (
        _SelectedCandidate(
            candidate_id=str(selected["candidate_id"]),
            params={"feature_prior_scale": float(selected["feature_prior_scale"])},
            mean_loss=float(selected["mean_loss"]),
            standard_error=float(selected["standard_error"]),
        ),
        detail_rows,
    )


def _gower_candidate_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray, int]],
    *,
    panel: str,
    k_grid: Sequence[int],
    p_grid: Sequence[float],
    rare_min_frequency: int,
    context: Mapping[str, Any],
) -> tuple[_SelectedCandidate, list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for k in k_grid:
        for power in p_grid:
            candidate_id = f"gower_k_{int(k)}_p_{float(power):g}"
            fold_losses: list[float] = []
            failure = ""
            for train_index, validation_index, inner_fold in splits:
                try:
                    estimator = ReducedPanelGowerClassifier(
                        panel=panel,
                        rare_min_frequency=int(rare_min_frequency),
                        k=int(k),
                        p=float(power),
                    ).fit(X.iloc[train_index], y[train_index])
                    predicted = estimator.predict_proba(
                        X.iloc[validation_index]
                    )
                    loss = soft_cross_entropy(y[validation_index], predicted)
                    status = "ok"
                    error = ""
                    fold_losses.append(float(loss))
                except Exception as exc:
                    loss = np.nan
                    status = "failed"
                    error = repr(exc)
                    failure = error
                detail_rows.append(
                    {
                        **context,
                        "model_family": SECONDARY_MODEL,
                        "panel": panel,
                        "candidate_id": candidate_id,
                        "candidate_params": json.dumps(
                            {"k": int(k), "p": float(power)},
                            sort_keys=True,
                        ),
                        "inner_fold": int(inner_fold),
                        "soft_log_loss": float(loss),
                        "status": status,
                        "error": error,
                    }
                )
            complete = len(fold_losses) == len(splits)
            mean_loss = float(np.mean(fold_losses)) if complete else np.inf
            standard_error = (
                float(np.std(fold_losses, ddof=1) / np.sqrt(len(fold_losses)))
                if complete and len(fold_losses) > 1
                else np.inf
            )
            summaries.append(
                {
                    "candidate_id": candidate_id,
                    "k": int(k),
                    "p": float(power),
                    "mean_loss": mean_loss,
                    "standard_error": standard_error,
                    "complete": complete,
                    "failure": failure,
                }
            )
    valid = [row for row in summaries if row["complete"]]
    if not valid:
        raise RuntimeError(
            f"Every Gower nuisance candidate failed for panel {panel}"
        )
    best = min(valid, key=lambda row: (row["mean_loss"], row["candidate_id"]))
    threshold = float(best["mean_loss"] + best["standard_error"])
    eligible = [row for row in valid if row["mean_loss"] <= threshold]
    selected = min(
        eligible,
        key=lambda row: (-row["k"], row["p"], row["candidate_id"]),
    )
    for row in detail_rows:
        row["selected_candidate_id"] = str(selected["candidate_id"])
        row["one_se_threshold"] = threshold
    return (
        _SelectedCandidate(
            candidate_id=str(selected["candidate_id"]),
            params={"k": int(selected["k"]), "p": float(selected["p"])},
            mean_loss=float(selected["mean_loss"]),
            standard_error=float(selected["standard_error"]),
        ),
        detail_rows,
    )


def _prediction_rows(
    *,
    test: pd.DataFrame,
    y_test: np.ndarray,
    predicted: np.ndarray,
    width: float,
    repeat: int,
    fold: int,
    panel: str,
    model_family: str,
    candidate_id: str,
    candidate_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    probabilities = normalize_probabilities(predicted)
    rows: list[dict[str, Any]] = []
    for position, record in test.reset_index(drop=True).iterrows():
        row: dict[str, Any] = {
            "width": float(width),
            "width_id": _width_id(width),
            "repeat": int(repeat),
            "fold": int(fold),
            "record_id": str(record["record_id"]),
            "cv_group_id": str(record["cv_group_id"]),
            "panel": panel,
            "model_family": model_family,
            "candidate_id": candidate_id,
            "candidate_params": json.dumps(
                dict(candidate_params), sort_keys=True
            ),
        }
        row.update(
            {
                column: float(value)
                for column, value in zip(
                    SOFT_LABEL_COLUMNS,
                    y_test[position],
                    strict=True,
                )
            }
        )
        row.update(
            {
                column: float(value)
                for column, value in zip(
                    PREDICTION_COLUMNS,
                    probabilities[position],
                    strict=True,
                )
            }
        )
        rows.append(row)
    return rows


def _run_one_outer_split(
    records: pd.DataFrame,
    labels_by_width: Mapping[float, pd.DataFrame],
    outer_manifest: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    repeat: int,
    fold: int,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_manifest = outer_manifest.loc[
        outer_manifest["repeat"].eq(int(repeat))
        & outer_manifest["fold"].eq(int(fold))
    ]
    test_ids = set(selected_manifest["record_id"].astype(str))
    test_mask = records["record_id"].astype(str).isin(test_ids).to_numpy()
    train = records.loc[~test_mask].reset_index(drop=True)
    test = records.loc[test_mask].reset_index(drop=True)
    if len(test) != len(selected_manifest):
        raise AssertionError("Outer test manifest and records disagree")

    primary_lookup = (
        labels_by_width[PRIMARY_WIDTH]
        .set_index(labels_by_width[PRIMARY_WIDTH]["record_id"].astype(str))
    )
    primary_train = train.copy()
    primary_train.loc[:, SOFT_LABEL_COLUMNS] = primary_lookup.loc[
        train["record_id"].astype(str), SOFT_LABEL_COLUMNS
    ].to_numpy(float)
    splits = _inner_splits(
        primary_train,
        inner_manifest,
        repeat=repeat,
        fold=fold,
    )

    union_sources: list[str] = []
    for panel in ALL_PANELS:
        union_sources.extend(required_association_source_columns(panel))
    source_columns = tuple(dict.fromkeys(union_sources))
    X_train = train.loc[:, list(source_columns)]
    X_test = test.loc[:, list(source_columns)]

    primary_settings = dict(contract["models"]["primary"])
    secondary_settings = dict(contract["models"]["secondary"])
    rare_min_frequency = int(
        contract["preprocessing"]["rare_min_frequency"]
    )
    map_scales = tuple(
        float(value)
        for value in primary_settings[
            "shared_gate_and_conditional_feature_prior_scale_grid"
        ]
    )
    gower_k = tuple(int(value) for value in secondary_settings["k_grid"])
    gower_p = tuple(
        float(value) for value in secondary_settings["distance_power_grid"]
    )
    if smoke:
        map_scales = map_scales[:1]
        # The smoothest candidate follows the frozen deterministic tie-break.
        gower_k = (max(gower_k),)
        gower_p = (min(gower_p),)

    prediction_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for width in sorted(labels_by_width):
        labels = labels_by_width[width]
        label_lookup = labels.set_index(labels["record_id"].astype(str))
        y_train = label_lookup.loc[
            train["record_id"].astype(str), SOFT_LABEL_COLUMNS
        ].to_numpy(float)
        y_test = label_lookup.loc[
            test["record_id"].astype(str), SOFT_LABEL_COLUMNS
        ].to_numpy(float)
        prior = y_train.mean(axis=0)
        prior_predicted = np.repeat(prior[None, :], len(test), axis=0)
        prediction_rows.extend(
            _prediction_rows(
                test=test,
                y_test=y_test,
                predicted=prior_predicted,
                width=width,
                repeat=repeat,
                fold=fold,
                panel=PRIOR_PANEL,
                model_family=PRIOR_MODEL,
                candidate_id="arithmetic_training_soft_mean",
                candidate_params={"smoothing": 0.0},
            )
        )
        fit_rows.append(
            {
                "width": width,
                "repeat": repeat,
                "fold": fold,
                "panel": PRIOR_PANEL,
                "model_family": PRIOR_MODEL,
                "candidate_id": "arithmetic_training_soft_mean",
                "candidate_params": json.dumps({"smoothing": 0.0}),
                "inner_mean_soft_log_loss": np.nan,
                "inner_standard_error": np.nan,
                "fit_status": "complete",
                "optimizer_audit": "",
            }
        )

        for panel in ALL_PANELS:
            context = {
                "width": float(width),
                "width_id": _width_id(width),
                "outer_repeat": int(repeat),
                "outer_fold": int(fold),
            }
            selected_map, map_rows = _map_candidate_rows(
                X_train,
                y_train,
                splits,
                panel=panel,
                scales=map_scales,
                rare_min_frequency=rare_min_frequency,
                intercept_prior_scale=float(
                    primary_settings["intercept_prior_scale"]
                ),
                max_iter=int(primary_settings["max_iter"]),
                tolerance=float(primary_settings["tolerance"]),
                context=context,
            )
            candidate_rows.extend(map_rows)
            map_estimator = ReducedPanelTwoStageMAPClassifier(
                panel=panel,
                rare_min_frequency=rare_min_frequency,
                feature_prior_scale=float(
                    selected_map.params["feature_prior_scale"]
                ),
                intercept_prior_scale=float(
                    primary_settings["intercept_prior_scale"]
                ),
                max_iter=int(primary_settings["max_iter"]),
                tolerance=float(primary_settings["tolerance"]),
            ).fit(X_train, y_train)
            if map_estimator.fit_status_ != "complete":
                raise RuntimeError(
                    "Selected outer MAP fit did not converge: "
                    f"{map_estimator.optimization_audit_}"
                )
            map_predicted = map_estimator.predict_proba(X_test)
            prediction_rows.extend(
                _prediction_rows(
                    test=test,
                    y_test=y_test,
                    predicted=map_predicted,
                    width=width,
                    repeat=repeat,
                    fold=fold,
                    panel=panel,
                    model_family=PRIMARY_MODEL,
                    candidate_id=selected_map.candidate_id,
                    candidate_params=selected_map.params,
                )
            )
            fit_rows.append(
                {
                    "width": width,
                    "repeat": repeat,
                    "fold": fold,
                    "panel": panel,
                    "model_family": PRIMARY_MODEL,
                    "candidate_id": selected_map.candidate_id,
                    "candidate_params": json.dumps(
                        selected_map.params, sort_keys=True
                    ),
                    "inner_mean_soft_log_loss": selected_map.mean_loss,
                    "inner_standard_error": selected_map.standard_error,
                    "fit_status": map_estimator.fit_status_,
                    "optimizer_audit": json.dumps(
                        _jsonable(map_estimator.optimization_audit_),
                        sort_keys=True,
                    ),
                }
            )

            selected_gower, gower_rows = _gower_candidate_rows(
                X_train,
                y_train,
                splits,
                panel=panel,
                k_grid=gower_k,
                p_grid=gower_p,
                rare_min_frequency=rare_min_frequency,
                context=context,
            )
            candidate_rows.extend(gower_rows)
            gower_estimator = ReducedPanelGowerClassifier(
                panel=panel,
                rare_min_frequency=rare_min_frequency,
                k=int(selected_gower.params["k"]),
                p=float(selected_gower.params["p"]),
            ).fit(X_train, y_train)
            gower_predicted = gower_estimator.predict_proba(X_test)
            prediction_rows.extend(
                _prediction_rows(
                    test=test,
                    y_test=y_test,
                    predicted=gower_predicted,
                    width=width,
                    repeat=repeat,
                    fold=fold,
                    panel=panel,
                    model_family=SECONDARY_MODEL,
                    candidate_id=selected_gower.candidate_id,
                    candidate_params=selected_gower.params,
                )
            )
            fit_rows.append(
                {
                    "width": width,
                    "repeat": repeat,
                    "fold": fold,
                    "panel": panel,
                    "model_family": SECONDARY_MODEL,
                    "candidate_id": selected_gower.candidate_id,
                    "candidate_params": json.dumps(
                        selected_gower.params, sort_keys=True
                    ),
                    "inner_mean_soft_log_loss": selected_gower.mean_loss,
                    "inner_standard_error": selected_gower.standard_error,
                    "fit_status": "complete",
                    "optimizer_audit": "",
                }
            )
    return prediction_rows, candidate_rows, fit_rows


def run_shared_nested_oof(
    records: pd.DataFrame,
    labels_by_width: Mapping[float, pd.DataFrame],
    outer_manifest: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    smoke: bool,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all frozen panels and widths on the shared nested manifests."""

    tasks = [
        (int(row.repeat), int(row.fold))
        for row in outer_manifest[["repeat", "fold"]]
        .drop_duplicates()
        .itertuples(index=False)
    ]
    # Independent outer splits are deterministic. A thread backend avoids
    # platform semaphore limits that can prevent loky process pools from
    # starting in restricted execution environments.
    results = Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(_run_one_outer_split)(
            records,
            labels_by_width,
            outer_manifest,
            inner_manifest,
            contract,
            repeat=repeat,
            fold=fold,
            smoke=smoke,
        )
        for repeat, fold in tasks
    )
    predictions = pd.DataFrame(
        [row for result in results for row in result[0]]
    )
    candidates = pd.DataFrame(
        [row for result in results for row in result[1]]
    )
    fits = pd.DataFrame([row for result in results for row in result[2]])
    expected_repeats = (
        int(contract["smoke_mode"]["outer_repeats"])
        if smoke
        else int(contract["outer_validation"]["repeats"])
    )
    coverage = predictions.groupby(
        ["width", "panel", "model_family", "record_id"]
    ).size()
    if not coverage.eq(expected_repeats).all():
        raise AssertionError("Repeated OOF coverage is incomplete")
    return predictions, candidates, fits


def aggregate_oof_probabilities(oof: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "width",
        "width_id",
        "panel",
        "model_family",
        "record_id",
        "cv_group_id",
    ]
    aggregated = (
        oof.groupby(keys, as_index=False, dropna=False)
        .agg(
            **{column: (column, "first") for column in SOFT_LABEL_COLUMNS},
            **{column: (column, "mean") for column in PREDICTION_COLUMNS},
            n_repeated_oof=("repeat", "nunique"),
        )
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    aggregated.loc[:, PREDICTION_COLUMNS] = normalize_probabilities(
        aggregated.loc[:, PREDICTION_COLUMNS].to_numpy(float)
    )
    return aggregated


def _metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    descriptive = descriptive_probability_metrics(
        observed,
        predicted,
        entropy_threshold=0.90,
    )
    staged = two_stage_probability_metrics(observed, predicted)
    for key in ("soft_log_loss", "multiclass_brier"):
        if not np.isclose(
            float(descriptive[key]), float(staged[key]), atol=1e-12, rtol=0
        ):
            raise AssertionError(f"Metric implementations disagree for {key}")
    return {**descriptive, **staged}


def score_oof_predictions(
    oof: pd.DataFrame,
    aggregated: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    aggregate_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    for keys, frame in aggregated.groupby(
        ["width", "width_id", "panel", "model_family"], sort=True
    ):
        width, width_id, panel, model = keys
        observed = frame.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
        predicted = frame.loc[:, PREDICTION_COLUMNS].to_numpy(float)
        aggregate_rows.append(
            {
                "width": float(width),
                "width_id": str(width_id),
                "panel": str(panel),
                "model_family": str(model),
                "n_records": int(len(frame)),
                "n_connected_groups": int(frame["cv_group_id"].nunique()),
                **_metric_row(observed, predicted),
            }
        )
        calibration = calibration_table(
            observed, predicted, n_bins=8
        ).assign(
            width=float(width),
            width_id=str(width_id),
            panel=str(panel),
            model_family=str(model),
        )
        calibration_frames.append(calibration)
        confusion = expected_confusion_table(observed, predicted).assign(
            width=float(width),
            width_id=str(width_id),
            panel=str(panel),
            model_family=str(model),
        )
        confusion_frames.append(confusion)

    split_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for keys, frame in oof.groupby(
        ["width", "width_id", "panel", "model_family", "repeat", "fold"],
        sort=True,
    ):
        width, width_id, panel, model, repeat, fold = keys
        observed = frame.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
        predicted = frame.loc[:, PREDICTION_COLUMNS].to_numpy(float)
        split_rows.append(
            {
                "width": float(width),
                "width_id": str(width_id),
                "panel": str(panel),
                "model_family": str(model),
                "repeat": int(repeat),
                "fold": int(fold),
                "n_records": int(len(frame)),
                "n_connected_groups": int(frame["cv_group_id"].nunique()),
                **_metric_row(observed, predicted),
            }
        )
    for keys, frame in oof.groupby(
        ["width", "width_id", "panel", "model_family", "repeat"],
        sort=True,
    ):
        width, width_id, panel, model, repeat = keys
        observed = frame.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
        predicted = frame.loc[:, PREDICTION_COLUMNS].to_numpy(float)
        repeat_rows.append(
            {
                "width": float(width),
                "width_id": str(width_id),
                "panel": str(panel),
                "model_family": str(model),
                "repeat": int(repeat),
                "n_records": int(len(frame)),
                **_metric_row(observed, predicted),
            }
        )
    return {
        "aggregate_metrics": pd.DataFrame(aggregate_rows),
        "split_metrics": pd.DataFrame(split_rows),
        "repeat_metrics": pd.DataFrame(repeat_rows),
        "calibration": pd.concat(calibration_frames, ignore_index=True),
        "expected_confusion": pd.concat(confusion_frames, ignore_index=True),
    }


def _record_score_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    observed = aggregated.loc[:, SOFT_LABEL_COLUMNS].to_numpy(float)
    predicted = aggregated.loc[:, PREDICTION_COLUMNS].to_numpy(float)
    log_loss = -np.sum(
        observed * np.log(np.clip(predicted, 1e-15, 1.0)),
        axis=1,
    )
    brier = np.sum(np.square(predicted - observed), axis=1)
    return aggregated.loc[
        :,
        [
            "width",
            "width_id",
            "panel",
            "model_family",
            "record_id",
            "cv_group_id",
        ],
    ].assign(
        soft_log_loss=log_loss,
        multiclass_brier=brier,
    )


def _contrast_definitions() -> list[dict[str, str]]:
    rows = [
        {
            "contrast_id": "context_minus_prior",
            "role": "baseline_information",
            "augmented_panel": CONTEXT_PANEL,
            "augmented_model": PRIMARY_MODEL,
            "reference_panel": PRIOR_PANEL,
            "reference_model": PRIOR_MODEL,
        }
    ]
    for panel in (*FORMAL_PANELS, *SENSITIVITY_PANELS):
        token = panel.removeprefix("context_plus_")
        role = (
            "formal_augmented_vs_prior"
            if panel in FORMAL_PANELS
            else "sensitivity_augmented_vs_prior"
        )
        rows.append(
            {
                "contrast_id": f"{token}_minus_prior",
                "role": role,
                "augmented_panel": panel,
                "augmented_model": PRIMARY_MODEL,
                "reference_panel": PRIOR_PANEL,
                "reference_model": PRIOR_MODEL,
            }
        )
        rows.append(
            {
                "contrast_id": f"{token}_minus_context",
                "role": (
                    "formal_single_feature_increment"
                    if panel in FORMAL_PANELS
                    else "sensitivity_single_feature_increment"
                ),
                "augmented_panel": panel,
                "augmented_model": PRIMARY_MODEL,
                "reference_panel": CONTEXT_PANEL,
                "reference_model": PRIMARY_MODEL,
            }
        )
    return rows


def build_paired_inference(
    aggregated: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, pd.DataFrame]:
    record_scores = _record_score_table(aggregated)
    bootstrap_repetitions = (
        int(contract["smoke_mode"]["bootstrap_repetitions"])
        if smoke
        else int(
            contract["connected_group_inference"]["bootstrap"]["repetitions"]
        )
    )
    bootstrap_seed = int(
        contract["connected_group_inference"]["bootstrap"]["seed"]
    )
    draw_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    repeat_delta_rows: list[dict[str, Any]] = []
    delta_tables: dict[tuple[float, str], pd.DataFrame] = {}

    for width in sorted(aggregated["width"].unique()):
        for contrast in _contrast_definitions():
            augmented = record_scores.loc[
                record_scores["width"].eq(float(width))
                & record_scores["panel"].eq(contrast["augmented_panel"])
                & record_scores["model_family"].eq(
                    contrast["augmented_model"]
                )
            ].copy()
            reference = record_scores.loc[
                record_scores["width"].eq(float(width))
                & record_scores["panel"].eq(contrast["reference_panel"])
                & record_scores["model_family"].eq(
                    contrast["reference_model"]
                )
            ].copy()
            merged = augmented.merge(
                reference,
                on=["record_id", "cv_group_id", "width", "width_id"],
                suffixes=("_augmented", "_reference"),
                validate="one_to_one",
            )
            if len(merged) != 97:
                raise AssertionError(
                    f"Paired contrast lost records: {contrast['contrast_id']}"
                )
            delta = merged.loc[:, ["record_id", "cv_group_id"]].copy()
            delta["delta_soft_log_loss"] = (
                merged["soft_log_loss_augmented"]
                - merged["soft_log_loss_reference"]
            )
            delta["delta_multiclass_brier"] = (
                merged["multiclass_brier_augmented"]
                - merged["multiclass_brier_reference"]
            )
            delta_tables[(float(width), contrast["contrast_id"])] = delta
            draws, summary = paired_connected_group_bootstrap(
                delta,
                ["delta_soft_log_loss", "delta_multiclass_brier"],
                replicates=bootstrap_repetitions,
                seed=bootstrap_seed,
            )
            metadata = {
                "width": float(width),
                "width_id": _width_id(width),
                **contrast,
            }
            draw_frames.append(draws.assign(**metadata))
            summary_frames.append(summary.assign(**metadata))

            augmented_repeat = repeat_metrics.loc[
                repeat_metrics["width"].eq(float(width))
                & repeat_metrics["panel"].eq(contrast["augmented_panel"])
                & repeat_metrics["model_family"].eq(
                    contrast["augmented_model"]
                )
            ][["repeat", "soft_log_loss", "multiclass_brier"]]
            reference_repeat = repeat_metrics.loc[
                repeat_metrics["width"].eq(float(width))
                & repeat_metrics["panel"].eq(contrast["reference_panel"])
                & repeat_metrics["model_family"].eq(
                    contrast["reference_model"]
                )
            ][["repeat", "soft_log_loss", "multiclass_brier"]]
            paired_repeat = augmented_repeat.merge(
                reference_repeat,
                on="repeat",
                suffixes=("_augmented", "_reference"),
                validate="one_to_one",
            )
            for row in paired_repeat.itertuples(index=False):
                repeat_delta_rows.append(
                    {
                        **metadata,
                        "repeat": int(row.repeat),
                        "delta_soft_log_loss": float(
                            row.soft_log_loss_augmented
                            - row.soft_log_loss_reference
                        ),
                        "delta_multiclass_brier": float(
                            row.multiclass_brier_augmented
                            - row.multiclass_brier_reference
                        ),
                    }
                )

    signflip_repetitions = (
        int(contract["smoke_mode"]["sign_flip_repetitions"])
        if smoke
        else int(
            contract["connected_group_inference"][
                "one_sided_feature_p_values"
            ]["repetitions"]
        )
    )
    signflip_seed = int(
        contract["connected_group_inference"]["one_sided_feature_p_values"][
            "seed"
        ]
    )
    wide: pd.DataFrame | None = None
    feature_columns: list[str] = []
    panel_by_feature: dict[str, str] = {}
    for panel in FORMAL_PANELS:
        token = panel.removeprefix("context_plus_")
        contrast_id = f"{token}_minus_context"
        delta = delta_tables[(PRIMARY_WIDTH, contrast_id)].rename(
            columns={"delta_soft_log_loss": token}
        )
        selected = delta.loc[:, ["record_id", "cv_group_id", token]]
        wide = (
            selected
            if wide is None
            else wide.merge(
                selected,
                on=["record_id", "cv_group_id"],
                validate="one_to_one",
            )
        )
        feature_columns.append(token)
        panel_by_feature[token] = panel
    if wide is None:
        raise AssertionError("No formal feature deltas were constructed")
    signflip_draws, signflip_summary = connected_group_signflip_pvalues(
        wide,
        feature_columns,
        replicates=signflip_repetitions,
        seed=signflip_seed,
    )
    raw_p = dict(
        zip(
            signflip_summary["feature"].astype(str),
            signflip_summary["p_value_lower_tail"].astype(float),
            strict=True,
        )
    )
    adjusted = holm_one_sided_adjustment(raw_p)
    signflip_summary["holm_adjusted_p"] = signflip_summary["feature"].map(
        adjusted
    )
    signflip_summary["panel"] = signflip_summary["feature"].map(
        panel_by_feature
    )
    return {
        "record_scores": record_scores,
        "bootstrap_draws": pd.concat(draw_frames, ignore_index=True),
        "bootstrap_summary": pd.concat(summary_frames, ignore_index=True),
        "repeat_deltas": pd.DataFrame(repeat_delta_rows),
        "signflip_draws": signflip_draws,
        "signflip_summary": signflip_summary,
    }


def _summary_value(
    summary: pd.DataFrame,
    *,
    width: float,
    contrast_id: str,
    metric: str,
    column: str,
) -> float:
    selected = summary.loc[
        summary["width"].eq(float(width))
        & summary["contrast_id"].eq(contrast_id)
        & summary["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one inference row for {width}/{contrast_id}/{metric}, "
            f"found {len(selected)}"
        )
    return float(selected.iloc[0][column])


def build_release_decisions(
    inference: Mapping[str, pd.DataFrame],
    *,
    smoke: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bootstrap = inference["bootstrap_summary"]
    repeats = inference["repeat_deltas"]
    p_values = inference["signflip_summary"].set_index("panel")
    rows: list[dict[str, Any]] = []

    context_ci_high = _summary_value(
        bootstrap,
        width=PRIMARY_WIDTH,
        contrast_id="context_minus_prior",
        metric="delta_soft_log_loss",
        column="ci95_high",
    )
    context_baseline_pass = bool(context_ci_high < 0 and not smoke)

    for panel in FORMAL_PANELS:
        token = panel.removeprefix("context_plus_")
        prior_contrast = f"{token}_minus_prior"
        context_contrast = f"{token}_minus_context"
        prior_ci_high = _summary_value(
            bootstrap,
            width=PRIMARY_WIDTH,
            contrast_id=prior_contrast,
            metric="delta_soft_log_loss",
            column="ci95_high",
        )
        context_ci_high_feature = _summary_value(
            bootstrap,
            width=PRIMARY_WIDTH,
            contrast_id=context_contrast,
            metric="delta_soft_log_loss",
            column="ci95_high",
        )
        context_observed = _summary_value(
            bootstrap,
            width=PRIMARY_WIDTH,
            contrast_id=context_contrast,
            metric="delta_soft_log_loss",
            column="observed_mean",
        )
        brier_observed = _summary_value(
            bootstrap,
            width=PRIMARY_WIDTH,
            contrast_id=context_contrast,
            metric="delta_multiclass_brier",
            column="observed_mean",
        )
        selected_repeats = repeats.loc[
            repeats["width"].eq(PRIMARY_WIDTH)
            & repeats["contrast_id"].eq(context_contrast)
        ]
        repeat_fraction = float(
            selected_repeats["delta_soft_log_loss"].lt(0).mean()
        )
        raw_p = float(p_values.loc[panel, "p_value_lower_tail"])
        holm_p = float(p_values.loc[panel, "holm_adjusted_p"])
        width_deltas = {
            float(width): _summary_value(
                bootstrap,
                width=float(width),
                contrast_id=context_contrast,
                metric="delta_soft_log_loss",
                column="observed_mean",
            )
            for width in EXPECTED_WIDTHS
        }
        width_consistent = all(value < 0 for value in width_deltas.values())
        augmented_vs_prior_pass = prior_ci_high < 0
        augmented_vs_context_pass = context_ci_high_feature < 0
        holm_pass = holm_p <= 0.05
        repeat_pass = repeat_fraction >= 0.80
        brier_pass = brier_observed <= 0
        gate_pass = bool(
            augmented_vs_prior_pass
            and augmented_vs_context_pass
            and holm_pass
            and repeat_pass
            and brier_pass
            and not smoke
        )
        general_claim_pass = bool(gate_pass and width_consistent)
        rows.append(
            {
                "panel": panel,
                "feature": token,
                "role": "formal_single_feature_increment",
                "context_baseline_ci95_high": context_ci_high,
                "context_baseline_pass": context_baseline_pass,
                "augmented_vs_prior_ci95_high": prior_ci_high,
                "augmented_vs_prior_pass": augmented_vs_prior_pass,
                "augmented_vs_context_observed_delta_log_loss": context_observed,
                "augmented_vs_context_ci95_high": context_ci_high_feature,
                "augmented_vs_context_pass": augmented_vs_context_pass,
                "one_sided_raw_p": raw_p,
                "holm_adjusted_p": holm_p,
                "holm_pass": holm_pass,
                "repeat_improvement_fraction": repeat_fraction,
                "repeat_direction_pass": repeat_pass,
                "observed_delta_multiclass_brier": brier_observed,
                "brier_guardrail_pass": brier_pass,
                "width_15_delta_log_loss": width_deltas[0.15],
                "width_20_delta_log_loss": width_deltas[0.20],
                "width_25_delta_log_loss": width_deltas[0.25],
                "width_direction_consistent": width_consistent,
                "width_label": (
                    "not_applicable_failed_primary_gate"
                    if not gate_pass
                    else "width_consistent"
                    if width_consistent
                    else "width_sensitive"
                ),
                "primary_release_gate_pass": gate_pass,
                "general_cardinal_region_claim_pass": general_claim_pass,
                "adjusted_probability_curve_execution_authorized": (
                    general_claim_pass
                ),
                "curve_support_status": (
                    "pending_support"
                    if general_claim_pass
                    else "not_authorized"
                ),
                "adjusted_probability_curve_release_pass": False,
                "hard_accuracy_used_for_gate": False,
                "smoke": bool(smoke),
            }
        )

    for panel in SENSITIVITY_PANELS:
        token = panel.removeprefix("context_plus_")
        contrast_id = f"{token}_minus_context"
        rows.append(
            {
                "panel": panel,
                "feature": token,
                "role": "sensitivity_only_not_in_holm_family",
                "context_baseline_ci95_high": context_ci_high,
                "context_baseline_pass": context_baseline_pass,
                "augmented_vs_prior_ci95_high": _summary_value(
                    bootstrap,
                    width=PRIMARY_WIDTH,
                    contrast_id=f"{token}_minus_prior",
                    metric="delta_soft_log_loss",
                    column="ci95_high",
                ),
                "augmented_vs_prior_pass": False,
                "augmented_vs_context_observed_delta_log_loss": _summary_value(
                    bootstrap,
                    width=PRIMARY_WIDTH,
                    contrast_id=contrast_id,
                    metric="delta_soft_log_loss",
                    column="observed_mean",
                ),
                "augmented_vs_context_ci95_high": _summary_value(
                    bootstrap,
                    width=PRIMARY_WIDTH,
                    contrast_id=contrast_id,
                    metric="delta_soft_log_loss",
                    column="ci95_high",
                ),
                "augmented_vs_context_pass": False,
                "one_sided_raw_p": np.nan,
                "holm_adjusted_p": np.nan,
                "holm_pass": False,
                "repeat_improvement_fraction": float(
                    repeats.loc[
                        repeats["width"].eq(PRIMARY_WIDTH)
                        & repeats["contrast_id"].eq(contrast_id),
                        "delta_soft_log_loss",
                    ].lt(0).mean()
                ),
                "repeat_direction_pass": False,
                "observed_delta_multiclass_brier": _summary_value(
                    bootstrap,
                    width=PRIMARY_WIDTH,
                    contrast_id=contrast_id,
                    metric="delta_multiclass_brier",
                    column="observed_mean",
                ),
                "brier_guardrail_pass": False,
                "width_15_delta_log_loss": _summary_value(
                    bootstrap,
                    width=0.15,
                    contrast_id=contrast_id,
                    metric="delta_soft_log_loss",
                    column="observed_mean",
                ),
                "width_20_delta_log_loss": _summary_value(
                    bootstrap,
                    width=0.20,
                    contrast_id=contrast_id,
                    metric="delta_soft_log_loss",
                    column="observed_mean",
                ),
                "width_25_delta_log_loss": _summary_value(
                    bootstrap,
                    width=0.25,
                    contrast_id=contrast_id,
                    metric="delta_soft_log_loss",
                    column="observed_mean",
                ),
                "width_direction_consistent": False,
                "width_label": "sensitivity_only",
                "primary_release_gate_pass": False,
                "general_cardinal_region_claim_pass": False,
                "adjusted_probability_curve_execution_authorized": False,
                "curve_support_status": "not_authorized",
                "adjusted_probability_curve_release_pass": False,
                "hard_accuracy_used_for_gate": False,
                "smoke": bool(smoke),
            }
        )
    decisions = pd.DataFrame(rows)
    formal = decisions.loc[
        decisions["role"].eq("formal_single_feature_increment")
    ]
    any_augmented_beats_prior = bool(
        formal["augmented_vs_prior_pass"].astype(bool).any() and not smoke
    )
    any_release = bool(
        formal["primary_release_gate_pass"].astype(bool).any() and not smoke
    )
    any_curve_authorized = bool(
        formal[
            "adjusted_probability_curve_execution_authorized"
        ].astype(bool).any()
        and not smoke
    )
    stopping = {
        "context_baseline_pass": context_baseline_pass,
        "any_formal_augmented_panel_beats_prior": any_augmented_beats_prior,
        "any_formal_feature_passes_primary_release_gate": any_release,
        "any_adjusted_probability_curve_execution_authorized": (
            any_curve_authorized
        ),
        "any_adjusted_probability_curve_release": False,
        "baseline_information_stop": not any_augmented_beats_prior,
        "feature_claim_release_stopped": not any_release,
        "feature_interpretation_stopped": not any_release,
        "decision": (
            "negative_stop_no_augmented_panel_beats_prior"
            if not any_augmented_beats_prior
            else "no_formal_feature_passed_all_incremental_gates"
            if not any_release
            else "one_or_more_formal_features_passed"
        ),
        "smoke_structural_only": bool(smoke),
    }
    return decisions, stopping


def finalize_curve_release_decisions(
    decisions: pd.DataFrame,
    curve_audit: pd.DataFrame,
    curves: pd.DataFrame,
    stopping: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Separate execution authorization from actual support-qualified release."""

    finalized = decisions.copy()
    if (
        "adjusted_probability_curve_execution_authorized"
        not in finalized.columns
    ):
        raise KeyError("Curve execution authorization is missing")
    audit_by_panel = (
        curve_audit.loc[curve_audit["panel"].astype(str).ne("")]
        .set_index("panel")["status"]
        .astype(str)
        .to_dict()
        if not curve_audit.empty
        else {}
    )
    curve_panels = (
        set(curves["panel"].astype(str)) if not curves.empty else set()
    )
    support_status: list[str] = []
    released: list[bool] = []
    for row in finalized.itertuples(index=False):
        panel = str(row.panel)
        authorized = bool(
            row.adjusted_probability_curve_execution_authorized
        )
        if not authorized:
            status = "not_authorized"
            if panel in curve_panels:
                raise AssertionError(
                    f"Unauthorized panel emitted curve rows: {panel}"
                )
        else:
            status = audit_by_panel.get(panel, "")
            if status not in {"released", "passed_no_estimable_curve"}:
                raise AssertionError(
                    f"Authorized panel lacks a final support audit: {panel}"
                )
        actual_release = bool(authorized and status == "released")
        if actual_release:
            panel_curve = curves.loc[
                curves["panel"].astype(str).eq(panel)
            ]
            if (
                len(panel_curve) != 19 * 4
                or panel_curve["grid_percentile"].nunique() != 19
                or panel_curve["class_label"].nunique() != 4
            ):
                raise AssertionError(
                    f"Released curve is incomplete for panel {panel}"
                )
        elif panel in curve_panels:
            raise AssertionError(
                f"Withheld panel emitted curve rows: {panel}"
            )
        support_status.append(status)
        released.append(actual_release)
    finalized["curve_support_status"] = support_status
    finalized["adjusted_probability_curve_release_pass"] = released

    final_stopping = dict(stopping)
    final_stopping["any_adjusted_probability_curve_release"] = bool(
        any(released)
    )
    final_stopping["n_adjusted_probability_curves_released"] = int(
        sum(released)
    )
    final_stopping["n_adjusted_probability_curve_rows"] = int(len(curves))
    return finalized, final_stopping


def _write_report(
    destination: Path,
    *,
    aggregate_metrics: pd.DataFrame,
    label_support: pd.DataFrame,
    decisions: pd.DataFrame,
    stopping: Mapping[str, Any],
    smoke: bool,
) -> Path:
    primary = aggregate_metrics.loc[
        aggregate_metrics["width"].eq(PRIMARY_WIDTH)
        & aggregate_metrics["model_family"].isin(
            [PRIOR_MODEL, PRIMARY_MODEL, SECONDARY_MODEL]
        )
    ].copy()
    display_columns = [
        "panel",
        "model_family",
        "soft_log_loss",
        "target_entropy",
        "excess_cross_entropy_KL",
        "multiclass_brier",
        "modal_accuracy",
        "posterior_expected_hard_accuracy",
        "oracle_posterior_expected_hard_accuracy",
        "mean_normalized_predictive_entropy",
        "uncertain_count",
    ]
    support_all = label_support.loc[label_support["population"].eq("all")][
        [
            "class_label",
            "effective_posterior_mass",
            "modal_record_count",
        ]
    ]
    oracle = float(
        primary["oracle_posterior_expected_hard_accuracy"].iloc[0]
    )
    lines = [
        "# Active all-97 posterior-aware cardinal classification",
        "",
        (
            "**Status:** structural smoke only"
            if smoke
            else f"**Decision:** `{stopping['decision']}`"
        ),
        "",
        "## Registered question",
        "",
        (
            "The primary test is the paired change in full-coverage four-class "
            "soft log loss for each `context + one feature` panel relative to "
            "`context_core`, on identical connected-group outer folds."
        ),
        "",
        "## Data and labels",
        "",
        "- Population: 97 records in 80 connected groups.",
        "- Validation: 10 × 5 connected-group outer CV (smoke mode is structural only).",
        "- Response: one four-component posterior-soft vector per record.",
        "- Posterior draws were not expanded into ML samples.",
        "",
        support_all.to_markdown(index=False),
        "",
        "## Primary-width out-of-fold metrics",
        "",
        primary.loc[:, display_columns].to_markdown(index=False),
        "",
        "## Accuracy boundary",
        "",
        (
            f"The posterior-expected hard-accuracy oracle ceiling is "
            f"`{oracle:.6f}`. Therefore a target of 0.700 is not attainable "
            "under that uncertainty-respecting definition. Modal accuracy is "
            "reported only as a descriptive diagnostic and did not fit, tune, "
            "select, or release any model."
        ),
        "",
        "## Feature release gates",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        (
            "A failed release gate is a validated negative out-of-connected-"
            "group result under this frozen representation. It does not prove "
            "that a formulation descriptor has no physical effect. Apparent "
            "slope neighborhoods are not reaction mechanisms, unique rate-"
            "determining steps, or causal formulation effects."
        ),
        "",
    ]
    report = destination / "report" / "posterior_cardinal_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_posterior_cardinal_classification(
    config: StudyConfig,
    run_dir: str | Path,
    *,
    source_run: str | Path,
    smoke: bool = False,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Run the complete frozen all-97 posterior-cardinal workflow."""

    destination = Path(run_dir).expanduser().resolve()
    source = Path(source_run).expanduser().resolve()
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    _assert_fresh_disjoint_run_directory(destination, source)
    contract, contract_path, contract_hash = (
        load_frozen_posterior_cardinal_contract(config)
    )
    for name in (
        "inputs",
        "inputs/reduced_feature_registry",
        "registry",
        "folds",
        "analyses",
        "analyses/inference",
        "effects",
        "report",
        "provenance",
    ):
        (destination / name).mkdir(parents=True, exist_ok=True)

    contract_snapshot = destination / "registry" / contract_path.name
    shutil.copy2(contract_path, contract_snapshot)
    if sha256_file(contract_snapshot) != contract_hash:
        raise RuntimeError("Frozen contract changed while copying")
    reduced_binding = snapshot_reduced_feature_registries(
        config,
        contract,
        destination / "inputs" / "reduced_feature_registry",
    )
    source_hashes = contract["formal_input_snapshot"]["source_file_sha256"]
    records, targets, draws, input_binding = snapshot_cardinal_inputs(
        source,
        destination / "inputs",
        canonical_dataset=config.resolve_path("canonical_dataset"),
        expected_source_hashes=source_hashes,
        require_formal_approval=not smoke,
    )
    _validate_run_frame(records)
    parsed_snapshot_verification = verify_cardinal_input_snapshot(
        destination / "inputs", source_hashes
    )
    parsed_registry_verification = verify_reduced_feature_registry_snapshot(
        destination / "inputs" / "reduced_feature_registry",
        contract,
    )

    raw_by_width, labels_by_width = derive_all_width_labels(records, draws)
    for width in EXPECTED_WIDTHS:
        width_dir = destination / "registry" / _width_id(width)
        width_dir.mkdir(parents=True, exist_ok=True)
        raw_by_width[width].to_csv(
            width_dir / "raw_anchor_soft_labels.csv", index=False
        )
        labels_by_width[width].to_csv(
            width_dir / "hierarchical_soft_labels.csv", index=False
        )
        support_summary(raw_by_width[width]).to_csv(
            width_dir / "raw_anchor_support.csv", index=False
        )
        hierarchical_support_summary(labels_by_width[width]).to_csv(
            width_dir / "hierarchical_class_support.csv", index=False
        )
        low_anchor_subdivision_table(raw_by_width[width]).to_csv(
            width_dir / "low_anchor_subdivision.csv", index=False
        )

    primary_frame = _merge_primary_labels(
        records, labels_by_width[PRIMARY_WIDTH]
    )
    outer_manifest = build_shared_outer_manifest(
        primary_frame, contract, smoke=smoke
    )
    inner_manifest = build_shared_inner_manifest(
        primary_frame,
        outer_manifest,
        contract,
        smoke=smoke,
    )
    outer_manifest.to_csv(
        destination / "folds" / "outer_split_manifest.csv", index=False
    )
    inner_manifest.to_csv(
        destination / "folds" / "inner_split_manifest.csv", index=False
    )

    oof, candidate_scores, model_fits = run_shared_nested_oof(
        records,
        labels_by_width,
        outer_manifest,
        inner_manifest,
        contract,
        smoke=smoke,
        n_jobs=int(n_jobs),
    )
    aggregated = aggregate_oof_probabilities(oof)
    scores = score_oof_predictions(oof, aggregated)
    inference = build_paired_inference(
        aggregated,
        scores["repeat_metrics"],
        contract,
        smoke=smoke,
    )
    decisions, stopping = build_release_decisions(
        inference,
        smoke=smoke,
    )
    curve_result = execute_released_adjusted_probability_curves(
        records,
        labels_by_width[PRIMARY_WIDTH],
        outer_manifest,
        model_fits,
        decisions,
        contract,
        smoke=smoke,
    )
    decisions, stopping = finalize_curve_release_decisions(
        decisions,
        curve_result.audit,
        curve_result.curves,
        stopping,
    )
    curves = curve_result.curves

    oof.to_csv(destination / "analyses" / "oof_probabilities.csv", index=False)
    aggregated.to_csv(
        destination / "analyses" / "oof_aggregated.csv", index=False
    )
    candidate_scores.to_csv(
        destination / "analyses" / "inner_candidate_scores.csv", index=False
    )
    model_fits.to_csv(
        destination / "analyses" / "model_fit_registry.csv", index=False
    )
    for name, frame in scores.items():
        frame.to_csv(destination / "analyses" / f"{name}.csv", index=False)
    inference["record_scores"].to_csv(
        destination / "analyses" / "record_level_scores.csv", index=False
    )
    inference["bootstrap_draws"].to_csv(
        destination
        / "analyses"
        / "inference"
        / "paired_connected_group_bootstrap.csv",
        index=False,
    )
    inference["bootstrap_summary"].to_csv(
        destination
        / "analyses"
        / "inference"
        / "paired_connected_group_bootstrap_summary.csv",
        index=False,
    )
    inference["repeat_deltas"].to_csv(
        destination / "analyses" / "inference" / "repeat_deltas.csv",
        index=False,
    )
    inference["signflip_draws"].to_csv(
        destination
        / "analyses"
        / "inference"
        / "connected_group_signflip_null.csv",
        index=False,
    )
    inference["signflip_summary"].to_csv(
        destination
        / "analyses"
        / "inference"
        / "one_sided_holm_summary.csv",
        index=False,
    )
    decisions.to_csv(
        destination / "effects" / "feature_release_decisions.csv",
        index=False,
    )
    curve_result.fold_predictions.to_csv(
        destination
        / "effects"
        / "adjusted_probability_curve_fold_predictions.csv",
        index=False,
    )
    curve_result.audit.to_csv(
        destination / "effects" / "adjusted_probability_curve_audit.csv",
        index=False,
    )
    curves.to_csv(
        destination / "effects" / "adjusted_probability_curves.csv",
        index=False,
    )
    write_json(
        destination / "effects" / "stopping_decision.json",
        _jsonable(stopping),
    )

    modeled_snapshot_verification = verify_cardinal_input_snapshot(
        destination / "inputs", source_hashes
    )
    modeled_registry_verification = verify_reduced_feature_registry_snapshot(
        destination / "inputs" / "reduced_feature_registry",
        contract,
    )
    source_verification = verify_source_run_unchanged(
        source,
        expected_tree_sha256=input_binding[
            "source_run_tree_sha256_before"
        ],
    )
    write_json(
        destination / "provenance" / "posterior_cardinal_contract_binding.json",
        {
            "contract_id": contract["contract_id"],
            "source_path": str(contract_path),
            "snapshot_path": str(contract_snapshot),
            "sha256": contract_hash,
            "snapshot_sha256": sha256_file(contract_snapshot),
            "frozen_before_first_fit": True,
        },
    )
    write_json(
        destination
        / "provenance"
        / "posterior_cardinal_input_binding.json",
        input_binding,
    )
    write_json(
        destination / "provenance" / "reduced_feature_registry_binding.json",
        reduced_binding,
    )
    write_json(
        destination / "provenance" / "input_snapshot_verification_after_parse.json",
        parsed_snapshot_verification,
    )
    write_json(
        destination / "provenance" / "input_snapshot_verification_after_modeling.json",
        modeled_snapshot_verification,
    )
    write_json(
        destination / "provenance" / "reduced_registry_verification_after_parse.json",
        parsed_registry_verification,
    )
    write_json(
        destination
        / "provenance"
        / "reduced_registry_verification_after_modeling.json",
        modeled_registry_verification,
    )
    write_json(
        destination / "provenance" / "source_run_verification.json",
        source_verification,
    )

    label_support = hierarchical_support_summary(
        labels_by_width[PRIMARY_WIDTH]
    )
    primary_metrics = scores["aggregate_metrics"].loc[
        scores["aggregate_metrics"]["width"].eq(PRIMARY_WIDTH)
    ]
    oracle_values = primary_metrics[
        "oracle_posterior_expected_hard_accuracy"
    ].unique()
    if len(oracle_values) != 1:
        raise AssertionError("Oracle accuracy ceiling must be model-independent")
    failed_curve_panels = set(
        decisions.loc[
            ~decisions["primary_release_gate_pass"].astype(bool),
            "panel",
        ].astype(str)
    ).intersection(
        set(curves["panel"].astype(str)) if not curves.empty else set()
    )
    if failed_curve_panels:
        raise AssertionError(
            "Failed features emitted forbidden curve rows: "
            f"{sorted(failed_curve_panels)}"
        )
    qa = {
        "status": "complete_smoke" if smoke else "complete",
        "scientifically_releasable": not smoke,
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_hash,
        "n_records": int(len(records)),
        "n_connected_groups": int(records["cv_group_id"].nunique()),
        "n_posterior_draws": int(len(draws)),
        "draws_per_record": int(
            draws.groupby("sample_record").size().iloc[0]
        ),
        "n_widths_fully_refit": int(len(labels_by_width)),
        "n_panels": int(len(ALL_PANELS)),
        "n_formal_increment_panels": int(len(FORMAL_PANELS)),
        "outer_repeats": int(outer_manifest["repeat"].nunique()),
        "outer_folds": int(outer_manifest["fold"].nunique()),
        "inner_folds": int(inner_manifest["inner_fold"].nunique()),
        "one_ml_row_per_record": True,
        "posterior_draws_expanded_as_ml_samples": False,
        "class_rows_expanded_for_MAP": False,
        "fold_local_preprocessing": True,
        "outer_manifest_shared_across_widths_panels_models": True,
        "inner_manifest_shared_across_widths_panels_models": True,
        "hard_accuracy_used_for_training_or_selection": False,
        "hard_accuracy_used_for_release": False,
        "posterior_expected_hard_accuracy_oracle_ceiling": float(
            oracle_values[0]
        ),
        "requested_accuracy_0_70_reachable_under_posterior_expected_definition": bool(
            float(oracle_values[0]) >= 0.70
        ),
        "n_formal_features_passing_release": int(
            decisions.loc[
                decisions["role"].eq("formal_single_feature_increment"),
                "primary_release_gate_pass",
            ].astype(bool).sum()
        ),
        "n_adjusted_curve_executions_authorized": int(
            decisions[
                "adjusted_probability_curve_execution_authorized"
            ].astype(bool).sum()
        ),
        "n_adjusted_curves_released": int(
            decisions[
                "adjusted_probability_curve_release_pass"
            ].astype(bool).sum()
        ),
        "n_adjusted_curve_fold_prediction_rows": int(
            len(curve_result.fold_predictions)
        ),
        "n_adjusted_curve_rows": int(len(curves)),
        "failed_features_emitted_curves": False,
        "input_snapshot_verified_after_parse": (
            parsed_snapshot_verification["status"] == "pass"
        ),
        "input_snapshot_verified_after_modeling": (
            modeled_snapshot_verification["status"] == "pass"
        ),
        "reduced_registry_verified_after_modeling": (
            modeled_registry_verification["status"] == "pass"
        ),
        "formal_source_run_unchanged": bool(source_verification["unchanged"]),
        "mechanistic_claim_allowed": False,
        "causal_claim_allowed": False,
        "stopping_decision": stopping,
    }
    write_json(destination / "analyses" / "analysis_qa.json", _jsonable(qa))
    report = _write_report(
        destination,
        aggregate_metrics=scores["aggregate_metrics"],
        label_support=label_support,
        decisions=decisions,
        stopping=stopping,
        smoke=smoke,
    )
    validation = {
        "overall_assessment": (
            "structural_smoke_only"
            if smoke
            else "ready_to_share_with_registered_negative_result"
            if not stopping[
                "any_formal_feature_passes_primary_release_gate"
            ]
            else "ready_to_share_with_registered_feature_and_curve_support_results"
        ),
        "methodology": "frozen_all97_posterior_soft_connected_group_nested_cv",
        "qa": qa,
        "data_quality": {
            "records": 97,
            "connected_groups": 80,
            "posterior_draws": 776_000,
            "formal_input_hashes_verified": True,
            "reduced_feature_registry_hashes_verified": True,
        },
        "accuracy_boundary": {
            "requested": 0.70,
            "posterior_expected_oracle_ceiling": float(oracle_values[0]),
            "attainable_under_uncertainty_respecting_definition": bool(
                float(oracle_values[0]) >= 0.70
            ),
            "modal_accuracy_is_descriptive_only": True,
        },
        "report": str(report),
    }
    write_json(
        destination / "report" / "validation_report.json",
        _jsonable(validation),
    )
    return _jsonable(
        {
            "status": "complete_smoke" if smoke else "complete",
            "run_dir": str(destination),
            "source_run": str(source),
            "contract_id": contract["contract_id"],
            "contract_sha256": contract_hash,
            "qa": qa,
            "report": str(report),
        }
    )


__all__ = [
    "ALL_PANELS",
    "CONTEXT_PANEL",
    "FORMAL_PANELS",
    "PRIMARY_MODEL",
    "PRIMARY_WIDTH",
    "PRIOR_MODEL",
    "SECONDARY_MODEL",
    "SENSITIVITY_PANELS",
    "aggregate_oof_probabilities",
    "build_paired_inference",
    "build_release_decisions",
    "build_shared_inner_manifest",
    "build_shared_outer_manifest",
    "derive_all_width_labels",
    "finalize_curve_release_decisions",
    "run_posterior_cardinal_classification",
    "run_shared_nested_oof",
    "score_oof_predictions",
]
