#!/usr/bin/env python3
"""Stage-aware outer-test permutation reliance for PRESTO policy-v2.

The analysis deliberately separates two quantities that the frozen contract
keeps distinct:

1. reliance of the inherited PRESTO/direct-Gower hybrid probability engine,
   scored with posterior-soft three-class log loss; and
2. reliance of the policy-v2 high-alert score, scored with posterior-soft
   F0.5 under the frozen mean-margin aggregation rule.

Only outer-test engineered feature values are permuted.  Outer-training data,
fold-local preprocessing, fitted estimators, selected penalties, and selected
thresholds remain fixed.  The output is a post-hoc model-reliance audit, not a
feature-effect, causal, mechanistic, or independently confirmed analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT / "code" / "Ink_Tafel_Study"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_CACHE_ROOT = Path(tempfile.gettempdir()) / "ink_tafel_feature_importance_cache"
_MPL_CACHE = _CACHE_ROOT / "matplotlib"
_XDG_CACHE = _CACHE_ROOT / "xdg"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

from joblib import Parallel, delayed  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ink_tafel_study.cardinal_metrics import (  # noqa: E402
    multiclass_brier,
    normalize_probabilities,
    soft_cross_entropy,
)
from ink_tafel_study.cardinal_models import (  # noqa: E402
    SoftGowerKNeighborsClassifier,
)
from ink_tafel_study.models import GowerKNeighborsRegressor  # noqa: E402
from ink_tafel_study.plot_style import (  # noqa: E402
    PALETTE,
    apply_publication_style,
    save_figure,
)
from ink_tafel_study.posterior_cardinal_contiguous_threeclass import (  # noqa: E402
    CLASS_LABELS,
    P_COLUMNS,
    Q_COLUMNS,
)
from ink_tafel_study.posterior_cardinal_contiguous_threeclass_direct_gower import (  # noqa: E402
    DISTANCE_POWER as DIRECT_DISTANCE_POWER,
    EPSILON as DIRECT_EPSILON,
    NEIGHBOR_COUNTS,
)
from ink_tafel_study.posterior_cardinal_contiguous_threeclass_presto_gate import (  # noqa: E402
    L1_SMOOTHING,
    NOVELTY_NEIGHBORS,
    RARE_MIN_FREQUENCY,
    RIDGE,
    _k_smallest_mean,
)
from ink_tafel_study.presto_ordinal import (  # noqa: E402
    SoftFusedContinuationRatioClassifier,
)
from ink_tafel_study.provenance import (  # noqa: E402
    sha256_file,
    verify_artifact_manifest,
)
from ink_tafel_study.threeclass_sensitivity_models import (  # noqa: E402
    _fold_local_engineered_features,
    probability_matrix_audit,
)


VERIFIED_RUN = (
    REPOSITORY_ROOT
    / "RESULTS"
    / "05_WORKBOOKS"
    / "posterior_cardinal_contiguous_threeclass_presto_policy_v2_verified_20260726"
)
PARENT_PRESTO_RUN = (
    REPOSITORY_ROOT
    / "RESULTS"
    / "05_WORKBOOKS"
    / "posterior_cardinal_contiguous_threeclass_presto_gate_v1_20260726"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "Data" / "04_results" / "RESULTS" / "06_FEATURE_IMPORTANCE"

EXPECTED_VERIFIED_TREE_SHA256 = (
    "67005476961140e6d6b8a610b700c55863465f35953c426d5b9d698d4bb04f13"
)
EXPECTED_PRESTO_PARENT_TREE_SHA256 = (
    "257c7438545007a3eada169a8d2cdbcccb59a30c3b2f89803128001cf422c0db"
)
EXPECTED_POLICY_CONTRACT_SHA256 = (
    "85694b3ff89b10458e56cf159fa8d2da2fa771948845469c8951a14a7ee93dcd"
)
EXPECTED_PROBABILITY_MODEL = "presto_high_gate_direct_gower_conditional"
EXPECTED_POLICY = "presto_policy_only_soft_f05_mean_margin_v2"
EXPECTED_POLICY_SCORE_MODEL = "presto_inspired_policy_only_high_score_v2"
EXPECTED_RECORDS = 97
EXPECTED_GROUPS = 80
EXPECTED_OUTER_ROWS = 970
DEFAULT_PERMUTATIONS = 100
DEFAULT_BOOTSTRAPS = 2000
PERMUTATION_SEED = 20260726
BOOTSTRAP_SEED = 20260726
F_BETA = 0.5
BASELINE_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class FeatureSpec:
    feature: str
    display_name: str
    branch: str
    high_column: str | None
    conditional_column: str | None
    high_block: str | None
    conditional_block: str | None
    source_columns: tuple[str, ...]


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "catalyst_family",
        "Catalyst family",
        "both_branches",
        "catalyst_family",
        "catalyst_family",
        "context",
        "context",
        ("catalyst_active_material_type",),
    ),
    FeatureSpec(
        "catalyst_loading",
        "Catalyst loading",
        "both_branches",
        "log10_catalyst_loading_mg_cm2",
        "log10_catalyst_loading_mg_cm2",
        "context",
        "context",
        ("catalyst_loading_mg_cm2_selected",),
    ),
    FeatureSpec(
        "electrolyte_molarity",
        "Electrolyte molarity",
        "both_branches",
        "log10_electrolyte_molarity_M",
        "log10_electrolyte_molarity_M",
        "context",
        "context",
        ("electrolyte_molarity_M",),
    ),
    FeatureSpec(
        "rotation_rate",
        "Rotation rate",
        "both_branches",
        "log1p_rpm",
        "log1p_rpm",
        "context",
        "context",
        ("rpm",),
    ),
    FeatureSpec(
        "scan_rate",
        "Scan rate",
        "both_branches",
        "log10_scan_rate_mV_s",
        "log10_scan_rate_mV_s",
        "context",
        "context",
        ("scan_rate_mV_s",),
    ),
    FeatureSpec(
        "ir_compensation",
        "iR compensation",
        "both_branches",
        "ir_correction_status",
        "ir_correction_status",
        "context",
        "context",
        ("ir_correction_status",),
    ),
    FeatureSpec(
        "electrolyte_scope",
        "Electrolyte scope",
        "both_branches",
        "electrolyte_scope",
        "electrolyte_scope",
        "context",
        "context",
        ("electrolyte_scope",),
    ),
    FeatureSpec(
        "solvent_polarity_coordinate",
        "Solvent polarity coordinate",
        "high_gate_only",
        "solvent_polarity_coordinate",
        None,
        "solvent_polarity",
        None,
        (
            "solvent_mixture_property_proxy_surface_tension_mN_m",
            "solvent_mixture_property_proxy_density_kg_m3",
            "solvent_mixture_property_proxy_dielectric_constant_best_available",
        ),
    ),
    FeatureSpec(
        "ionomer_to_solvent_ratio",
        "Ionomer / solvent ratio",
        "conditional_only",
        None,
        "log1p_ionomer_to_solvent_ratio",
        None,
        "ionomer_to_solvent_ratio",
        ("ionomer_to_solvent_ratio",),
    ),
)


def _indexed(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    output = frame.copy()
    if "record_id" not in output.columns:
        raise KeyError(f"{name} lacks record_id")
    output["record_id"] = output["record_id"].astype(str)
    if output["record_id"].duplicated().any():
        raise ValueError(f"{name} has duplicate record_id values")
    return output.set_index("record_id", drop=False)


def _verify_authoritative_sources() -> dict[str, Any]:
    verification = verify_artifact_manifest(VERIFIED_RUN)
    if verification.get("status") != "pass":
        raise RuntimeError(
            "Authoritative PRESTO policy-v2 artifact verification failed: "
            f"{verification}"
        )
    if verification.get("actual_tree_sha256") != EXPECTED_VERIFIED_TREE_SHA256:
        raise RuntimeError("Authoritative PRESTO policy-v2 tree hash changed")

    parent_verification = verify_artifact_manifest(PARENT_PRESTO_RUN)
    if parent_verification.get("status") != "pass":
        raise RuntimeError(
            "PRESTO-v1 refit parent artifact verification failed: "
            f"{parent_verification}"
        )
    if (
        parent_verification.get("actual_tree_sha256")
        != EXPECTED_PRESTO_PARENT_TREE_SHA256
    ):
        raise RuntimeError("PRESTO-v1 refit parent tree hash changed")

    contract_path = (
        REPOSITORY_ROOT
        / "Ink_Tafel_Study"
        / "config"
        / "contracts"
        / "posterior_cardinal_contiguous_threeclass_presto_policy_v2.yaml"
    )
    contract_hash = sha256_file(contract_path)
    if contract_hash != EXPECTED_POLICY_CONTRACT_SHA256:
        raise RuntimeError("PRESTO policy-v2 contract hash changed")

    binding_path = VERIFIED_RUN / "provenance" / "parent_refit_code_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    recorded = binding["verified_after_modeling"]
    mismatches: list[str] = []
    for row in recorded["files"]:
        path = REPOSITORY_ROOT / str(row["path"])
        if not path.is_file() or sha256_file(path) != str(row["sha256"]):
            mismatches.append(str(row["path"]))
    if mismatches:
        raise RuntimeError(
            "Current refit code differs from the verified binding: "
            f"{mismatches}"
        )
    return {
        "artifact_verification": verification,
        "presto_parent_artifact_verification": parent_verification,
        "policy_contract_path": str(contract_path.relative_to(REPOSITORY_ROOT)),
        "policy_contract_sha256": contract_hash,
        "refit_code_tree_sha256": recorded["current_tree_sha256"],
        "refit_code_files_checked": len(recorded["files"]),
    }


def _load_inputs() -> dict[str, pd.DataFrame]:
    records = pd.read_csv(VERIFIED_RUN / "inputs" / "records.csv")
    labels = pd.read_csv(
        VERIFIED_RUN / "registry" / "contiguous_three_class_soft_labels.csv"
    )
    outer = pd.read_csv(VERIFIED_RUN / "folds" / "outer_split_manifest.csv")
    parent_selections = pd.read_csv(
        PARENT_PRESTO_RUN / "analyses" / "outer_selections.csv"
    )
    policy_selections = pd.read_csv(
        VERIFIED_RUN / "analyses" / "outer_selections.csv"
    )
    probability_oof = pd.read_csv(
        VERIFIED_RUN / "analyses" / "oof_probabilities.csv"
    )
    probability_oof = probability_oof.loc[
        probability_oof["evaluation_model"].astype(str).eq(
            EXPECTED_PROBABILITY_MODEL
        )
    ].copy()
    policy_oof = pd.read_csv(
        VERIFIED_RUN / "analyses" / "policy_oof_decisions.csv"
    )
    policy_oof = policy_oof.loc[
        policy_oof["decision_policy"].astype(str).eq(EXPECTED_POLICY)
    ].copy()

    for frame in (records, labels, outer, probability_oof, policy_oof):
        frame["record_id"] = frame["record_id"].astype(str)
    if (
        len(records) != EXPECTED_RECORDS
        or records["record_id"].nunique() != EXPECTED_RECORDS
        or records["cv_group_id"].astype(str).nunique() != EXPECTED_GROUPS
        or len(labels) != EXPECTED_RECORDS
        or len(outer) != EXPECTED_OUTER_ROWS
        or len(probability_oof) != EXPECTED_OUTER_ROWS
        or len(policy_oof) != EXPECTED_OUTER_ROWS
    ):
        raise ValueError("PRESTO feature-reliance input coverage changed")
    if policy_oof["policy_score_model"].astype(str).nunique() != 1 or not (
        policy_oof["policy_score_model"]
        .astype(str)
        .eq(EXPECTED_POLICY_SCORE_MODEL)
        .all()
    ):
        raise ValueError("Unexpected PRESTO policy score model")
    q = labels.loc[:, Q_COLUMNS].to_numpy(float)
    probability_matrix_audit(q)
    if outer.groupby(["repeat", "record_id"]).size().ne(1).any():
        raise ValueError("Outer manifest does not cover each record once per repeat")
    if outer.groupby(["repeat", "cv_group_id"])["fold"].nunique().gt(1).any():
        raise ValueError("A connected group crosses an outer fold")

    return {
        "records": records,
        "labels": labels,
        "outer": outer,
        "parent_selections": parent_selections,
        "policy_selections": policy_selections,
        "probability_oof": probability_oof,
        "policy_oof": policy_oof,
    }


def _gower_weight(
    blocks: Mapping[str, Sequence[str]],
    feature: str | None,
) -> float:
    if feature is None:
        return 0.0
    normalized = {
        str(block): tuple(map(str, members)) for block, members in blocks.items()
    }
    for members in normalized.values():
        if str(feature) in members:
            return float(1.0 / len(normalized) / len(members))
    return 0.0


def _repeat_permuted_frame(
    frame: pd.DataFrame,
    column: str,
    permutations: np.ndarray,
) -> pd.DataFrame:
    repeated = pd.concat(
        [frame] * int(permutations.shape[0]),
        ignore_index=True,
    )
    source = frame[str(column)].to_numpy(copy=True)
    repeated[str(column)] = np.concatenate(
        [source[index] for index in permutations],
        axis=0,
    )
    return repeated


def _fit_presto_estimator(
    novelty_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    penalty: float,
) -> SoftFusedContinuationRatioClassifier:
    estimator = SoftFusedContinuationRatioClassifier(
        numeric_columns=("gower_novelty",),
        categorical_columns=(),
        class_names=CLASS_LABELS,
        penalty=float(penalty),
        ridge=RIDGE,
        l1_smoothing=L1_SMOOTHING,
        max_iter=2000,
        tolerance=1.0e-9,
    ).fit(novelty_train, y_train)
    if estimator.fit_status_ != "complete":
        raise RuntimeError(
            "PRESTO refit did not converge: "
            f"{estimator.optimization_audit_}"
        )
    return estimator


def _fit_high_branch(
    X_train_raw: pd.DataFrame,
    train_groups: Sequence[str],
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    *,
    probability_penalty: float,
    policy_penalty: float,
) -> dict[str, Any]:
    builder, engineered_train, engineered_test, _ = (
        _fold_local_engineered_features(
            X_train_raw,
            X_test_raw,
            panel="context_plus_polarity",
            rare_min_frequency=RARE_MIN_FREQUENCY,
        )
    )
    blocks = builder.get_blocks_out()
    distance_model = GowerKNeighborsRegressor(
        n_neighbors=1,
        distance_power=1.0,
        numeric_features=builder.numeric_columns_out_,
        categorical_features=builder.categorical_columns_out_,
        blocks=blocks,
        epsilon=1.0e-12,
    ).fit(engineered_train, np.zeros(len(engineered_train), dtype=float))
    groups = np.asarray(list(map(str, train_groups)), dtype=object)
    train_distances = distance_model._pairwise_distances(engineered_train)
    train_distances[groups[:, None] == groups[None, :]] = np.inf
    train_score = _k_smallest_mean(train_distances, NOVELTY_NEIGHBORS)
    test_score = _k_smallest_mean(
        distance_model._pairwise_distances(engineered_test),
        NOVELTY_NEIGHBORS,
    )
    novelty_train = pd.DataFrame({"gower_novelty": train_score})
    novelty_test = pd.DataFrame({"gower_novelty": test_score})
    probability_estimator = _fit_presto_estimator(
        novelty_train,
        y_train,
        penalty=float(probability_penalty),
    )
    policy_estimator = _fit_presto_estimator(
        novelty_train,
        y_train,
        penalty=float(policy_penalty),
    )
    probability = probability_estimator.predict_proba(novelty_test)
    policy_score = policy_estimator.predict_proba(novelty_test)[:, 2]
    probability_matrix_audit(probability)
    return {
        "builder": builder,
        "engineered_test": engineered_test,
        "distance_model": distance_model,
        "probability_estimator": probability_estimator,
        "policy_estimator": policy_estimator,
        "baseline_probability": probability,
        "baseline_policy_score": policy_score,
        "blocks": blocks,
    }


def _direct_probabilities_from_distances(
    estimator: SoftGowerKNeighborsClassifier,
    distances: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(distances), len(CLASS_LABELS)), dtype=float)
    n_train = len(estimator.soft_targets_)
    for row_index, row_distances in enumerate(distances):
        components = [
            estimator._knn_probability(  # noqa: SLF001 - frozen implementation
                row_distances,
                min(int(count), n_train),
            )
            for count in NEIGHBOR_COUNTS
        ]
        output[row_index] = np.mean(np.stack(components, axis=0), axis=0)
    return normalize_probabilities(output)


def _fit_conditional_branch(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, Any]:
    builder, engineered_train, engineered_test, _ = (
        _fold_local_engineered_features(
            X_train_raw,
            X_test_raw,
            panel="context_plus_ionomer_ratio",
            rare_min_frequency=RARE_MIN_FREQUENCY,
        )
    )
    blocks = builder.get_blocks_out()
    estimator = SoftGowerKNeighborsClassifier(
        numeric_columns=builder.numeric_columns_out_,
        categorical_columns=builder.categorical_columns_out_,
        class_names=CLASS_LABELS,
        blocks=blocks,
        block_weights=None,
        feature_weights=None,
        n_neighbors=max(NEIGHBOR_COUNTS),
        distance_power=DIRECT_DISTANCE_POWER,
        epsilon=DIRECT_EPSILON,
        neighbor_mode="knn",
        bandwidth=1.0,
    ).fit(engineered_train, y_train)
    distances = estimator.distance_model_._pairwise_distances(engineered_test)
    probability = _direct_probabilities_from_distances(estimator, distances)
    probability_matrix_audit(probability)
    return {
        "builder": builder,
        "engineered_test": engineered_test,
        "estimator": estimator,
        "baseline_probability": probability,
        "blocks": blocks,
    }


def _hybrid_probabilities(
    high_source: np.ndarray,
    direct_source: np.ndarray,
) -> np.ndarray:
    high_probability = np.asarray(high_source, dtype=float)
    direct_probability = np.asarray(direct_source, dtype=float)
    if high_probability.shape != direct_probability.shape:
        raise ValueError("PRESTO and direct-Gower predictions do not align")
    non_high = direct_probability[:, :2].sum(axis=1)
    conditional = np.divide(
        direct_probability[:, :2],
        non_high[:, None],
        out=np.full((len(direct_probability), 2), 0.5, dtype=float),
        where=non_high[:, None] > 0,
    )
    high = high_probability[:, 2]
    return normalize_probabilities(
        np.column_stack(
            [
                (1.0 - high) * conditional[:, 0],
                (1.0 - high) * conditional[:, 1],
                high,
            ]
        )
    )


def _selection_row(
    frame: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
) -> pd.Series:
    selected = frame.loc[
        frame["repeat"].eq(int(repeat)) & frame["fold"].eq(int(fold))
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one selection for repeat={repeat}, fold={fold}")
    return selected.iloc[0]


def _saved_split(
    frame: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    test_ids: Sequence[str],
) -> pd.DataFrame:
    selected = frame.loc[
        frame["repeat"].eq(int(repeat)) & frame["fold"].eq(int(fold))
    ].copy()
    selected["record_id"] = selected["record_id"].astype(str)
    indexed = selected.set_index("record_id")
    ordered = list(map(str, test_ids))
    if set(indexed.index) != set(ordered):
        raise ValueError("Saved split rows do not match outer-test IDs")
    return indexed.loc[ordered].reset_index()


def _run_outer_split(
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    parent_selections: pd.DataFrame,
    policy_selections: pd.DataFrame,
    saved_probability: pd.DataFrame,
    saved_policy: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    test_ids: Sequence[str],
    n_permutations: int,
) -> dict[str, Any]:
    ordered_ids = records_by_id.index.astype(str).tolist()
    test = list(map(str, test_ids))
    test_set = set(test)
    train = [record_id for record_id in ordered_ids if record_id not in test_set]
    train_groups = records_by_id.loc[train, "cv_group_id"].astype(str).tolist()
    test_groups = set(records_by_id.loc[test, "cv_group_id"].astype(str))
    if set(train_groups) & test_groups:
        raise AssertionError("A connected group crosses an outer split")

    probability_selection = _selection_row(
        parent_selections,
        repeat=repeat,
        fold=fold,
    )
    policy_selection = _selection_row(
        policy_selections,
        repeat=repeat,
        fold=fold,
    )
    probability_penalty = float(probability_selection["selected_penalty"])
    policy_penalty = float(policy_selection["selected_policy_penalty"])
    threshold = float(policy_selection["selected_high_threshold"])

    X_train_raw = records_by_id.loc[train].reset_index(drop=True)
    X_test_raw = records_by_id.loc[test].reset_index(drop=True)
    y_train = labels_by_id.loc[train, Q_COLUMNS].to_numpy(float)
    high = _fit_high_branch(
        X_train_raw,
        train_groups,
        X_test_raw,
        y_train,
        probability_penalty=probability_penalty,
        policy_penalty=policy_penalty,
    )
    conditional = _fit_conditional_branch(
        X_train_raw,
        X_test_raw,
        y_train,
    )
    baseline_probability = _hybrid_probabilities(
        high["baseline_probability"],
        conditional["baseline_probability"],
    )
    baseline_policy_score = np.asarray(
        high["baseline_policy_score"], dtype=float
    )
    baseline_margin = baseline_policy_score - threshold

    saved_probability_split = _saved_split(
        saved_probability,
        repeat=repeat,
        fold=fold,
        test_ids=test,
    )
    saved_policy_split = _saved_split(
        saved_policy,
        repeat=repeat,
        fold=fold,
        test_ids=test,
    )
    probability_error = float(
        np.max(
            np.abs(
                baseline_probability
                - saved_probability_split.loc[:, P_COLUMNS].to_numpy(float)
            )
        )
    )
    score_error = float(
        np.max(
            np.abs(
                baseline_policy_score
                - saved_policy_split["high_score"].to_numpy(float)
            )
        )
    )
    threshold_error = float(
        np.max(
            np.abs(
                threshold
                - saved_policy_split["selected_high_threshold"].to_numpy(float)
            )
        )
    )
    if max(probability_error, score_error, threshold_error) > BASELINE_TOLERANCE:
        raise RuntimeError(
            "Unperturbed outer refit does not reproduce the verified OOF: "
            f"probability={probability_error}, score={score_error}, "
            f"threshold={threshold_error}"
        )

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [PERMUTATION_SEED, int(repeat), int(fold)]
        )
    )
    permutation_indices = np.vstack(
        [rng.permutation(len(test)) for _ in range(int(n_permutations))]
    )
    n_test = len(test)
    repeated_high_baseline = np.repeat(
        high["baseline_probability"][None, :, :],
        int(n_permutations),
        axis=0,
    )
    repeated_policy_score = np.repeat(
        baseline_policy_score[None, :],
        int(n_permutations),
        axis=0,
    )
    repeated_direct_baseline = np.repeat(
        conditional["baseline_probability"][None, :, :],
        int(n_permutations),
        axis=0,
    )

    permuted_probability: dict[str, np.ndarray] = {}
    permuted_margin: dict[str, np.ndarray] = {}
    weight_rows: list[dict[str, Any]] = []
    for spec in FEATURES:
        high_probability = repeated_high_baseline
        policy_score = repeated_policy_score
        direct_probability = repeated_direct_baseline

        if spec.high_column is not None and spec.high_column in high[
            "engineered_test"
        ].columns:
            high_frame = _repeat_permuted_frame(
                high["engineered_test"],
                spec.high_column,
                permutation_indices,
            )
            high_score = _k_smallest_mean(
                high["distance_model"]._pairwise_distances(high_frame),
                NOVELTY_NEIGHBORS,
            )
            novelty = pd.DataFrame({"gower_novelty": high_score})
            high_probability = high["probability_estimator"].predict_proba(
                novelty
            ).reshape(int(n_permutations), n_test, len(CLASS_LABELS))
            policy_score = high["policy_estimator"].predict_proba(
                novelty
            )[:, 2].reshape(int(n_permutations), n_test)

        if spec.conditional_column is not None and spec.conditional_column in (
            conditional["engineered_test"].columns
        ):
            conditional_frame = _repeat_permuted_frame(
                conditional["engineered_test"],
                spec.conditional_column,
                permutation_indices,
            )
            distances = conditional[
                "estimator"
            ].distance_model_._pairwise_distances(conditional_frame)
            direct_probability = _direct_probabilities_from_distances(
                conditional["estimator"],
                distances,
            ).reshape(int(n_permutations), n_test, len(CLASS_LABELS))

        flattened_hybrid = _hybrid_probabilities(
            high_probability.reshape(-1, len(CLASS_LABELS)),
            direct_probability.reshape(-1, len(CLASS_LABELS)),
        )
        permuted_probability[spec.feature] = flattened_hybrid.reshape(
            int(n_permutations), n_test, len(CLASS_LABELS)
        )
        permuted_margin[spec.feature] = policy_score - threshold
        weight_rows.append(
            {
                "feature": spec.feature,
                "repeat": int(repeat),
                "fold": int(fold),
                "high_gate_gower_weight": _gower_weight(
                    high["blocks"], spec.high_column
                ),
                "conditional_gower_weight": _gower_weight(
                    conditional["blocks"], spec.conditional_column
                ),
            }
        )

    return {
        "repeat": int(repeat),
        "fold": int(fold),
        "test_ids": test,
        "baseline_probability": baseline_probability,
        "baseline_margin": baseline_margin,
        "permuted_probability": permuted_probability,
        "permuted_margin": permuted_margin,
        "weight_rows": weight_rows,
        "probability_max_abs_error": probability_error,
        "policy_score_max_abs_error": score_error,
        "threshold_max_abs_error": threshold_error,
    }


def _soft_high_metrics(
    q_high: np.ndarray,
    predicted_high: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(q_high, dtype=float)
    predicted = np.asarray(predicted_high, dtype=bool)
    tp = float(np.sum(target * predicted))
    fn = float(np.sum(target * (~predicted)))
    fp = float(np.sum((1.0 - target) * predicted))
    tn = float(np.sum((1.0 - target) * (~predicted)))
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    beta_squared = F_BETA**2
    denominator = beta_squared * precision + recall
    f_beta = (
        (1.0 + beta_squared) * precision * recall / denominator
        if denominator > 0
        else 0.0
    )
    return {
        "soft_f0_5": float(f_beta),
        "soft_high_recall": float(recall),
        "soft_high_precision": float(precision),
        "soft_high_false_positive_rate": float(fpr),
        "predicted_high_count": int(np.sum(predicted)),
    }


def _policy_decisions(
    probability: np.ndarray,
    margin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probability, dtype=float)
    m = np.asarray(margin, dtype=float)
    high = m >= 0.0
    if p.ndim == 2:
        decision = np.argmax(p[:, :2], axis=1)
    elif p.ndim == 3:
        decision = np.argmax(p[:, :, :2], axis=2)
    else:
        raise ValueError("Policy probability array must have 2 or 3 dimensions")
    decision = np.asarray(decision, dtype=int)
    decision[high] = 2
    return decision, high


def _posterior_expected_accuracy(
    q: np.ndarray,
    decision: np.ndarray,
) -> float | np.ndarray:
    targets = np.asarray(q, dtype=float)
    choices = np.asarray(decision, dtype=int)
    if choices.ndim == 1:
        return float(np.mean(targets[np.arange(len(targets)), choices]))
    expanded = np.broadcast_to(
        targets[None, :, :],
        (choices.shape[0], targets.shape[0], targets.shape[1]),
    )
    selected = np.take_along_axis(
        expanded,
        choices[:, :, None],
        axis=2,
    )[:, :, 0]
    return np.mean(selected, axis=1)


def _vector_soft_high_metrics(
    q_high: np.ndarray,
    predicted_high: np.ndarray,
) -> dict[str, np.ndarray]:
    target = np.asarray(q_high, dtype=float)
    predicted = np.asarray(predicted_high, dtype=bool)
    tp = predicted @ target
    fn = (~predicted) @ target
    fp = predicted @ (1.0 - target)
    tn = (~predicted) @ (1.0 - target)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) > 0)
    beta_squared = F_BETA**2
    denominator = beta_squared * precision + recall
    f_beta = np.divide(
        (1.0 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return {
        "soft_f0_5": f_beta,
        "soft_high_recall": recall,
        "soft_high_precision": precision,
        "soft_high_false_positive_rate": fpr,
        "predicted_high_count": np.sum(predicted, axis=1).astype(float),
    }


def _aggregate_outer_results(
    results: Sequence[dict[str, Any]],
    records_by_id: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    *,
    n_permutations: int,
) -> dict[str, Any]:
    record_ids = records_by_id.index.astype(str).tolist()
    position = {record_id: index for index, record_id in enumerate(record_ids)}
    counts = np.zeros(len(record_ids), dtype=int)
    baseline_probability_sum = np.zeros((len(record_ids), len(CLASS_LABELS)))
    baseline_margin_sum = np.zeros(len(record_ids), dtype=float)
    permuted_probability_sum = {
        spec.feature: np.zeros(
            (int(n_permutations), len(record_ids), len(CLASS_LABELS)),
            dtype=float,
        )
        for spec in FEATURES
    }
    permuted_margin_sum = {
        spec.feature: np.zeros(
            (int(n_permutations), len(record_ids)), dtype=float
        )
        for spec in FEATURES
    }
    weight_rows: list[dict[str, Any]] = []
    for result in results:
        indices = np.asarray(
            [position[record_id] for record_id in result["test_ids"]],
            dtype=int,
        )
        counts[indices] += 1
        baseline_probability_sum[indices] += result["baseline_probability"]
        baseline_margin_sum[indices] += result["baseline_margin"]
        for spec in FEATURES:
            permuted_probability_sum[spec.feature][:, indices, :] += result[
                "permuted_probability"
            ][spec.feature]
            permuted_margin_sum[spec.feature][:, indices] += result[
                "permuted_margin"
            ][spec.feature]
        weight_rows.extend(result["weight_rows"])

    expected_repeats = 10
    if not np.all(counts == expected_repeats):
        raise AssertionError(
            f"Repeated OOF coverage changed: {np.unique(counts, return_counts=True)}"
        )
    baseline_probability = normalize_probabilities(
        baseline_probability_sum / counts[:, None]
    )
    baseline_margin = baseline_margin_sum / counts
    permuted_probability = {
        feature: values / counts[None, :, None]
        for feature, values in permuted_probability_sum.items()
    }
    for values in permuted_probability.values():
        probability_matrix_audit(values.reshape(-1, len(CLASS_LABELS)))
    permuted_margin = {
        feature: values / counts[None, :]
        for feature, values in permuted_margin_sum.items()
    }
    q = labels_by_id.loc[record_ids, Q_COLUMNS].to_numpy(float)
    groups = records_by_id.loc[record_ids, "cv_group_id"].astype(str).to_numpy()
    return {
        "record_ids": record_ids,
        "groups": groups,
        "q": q,
        "baseline_probability": baseline_probability,
        "baseline_margin": baseline_margin,
        "permuted_probability": permuted_probability,
        "permuted_margin": permuted_margin,
        "weights": pd.DataFrame(weight_rows),
        "maximum_probability_refit_error": max(
            float(result["probability_max_abs_error"]) for result in results
        ),
        "maximum_policy_score_refit_error": max(
            float(result["policy_score_max_abs_error"]) for result in results
        ),
        "maximum_threshold_error": max(
            float(result["threshold_max_abs_error"]) for result in results
        ),
    }


def _observed_and_permutation_metrics(
    aggregated: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    q = np.asarray(aggregated["q"], dtype=float)
    baseline_probability = np.asarray(
        aggregated["baseline_probability"], dtype=float
    )
    baseline_margin = np.asarray(aggregated["baseline_margin"], dtype=float)
    baseline_decision, baseline_high = _policy_decisions(
        baseline_probability,
        baseline_margin,
    )
    baseline_high_metrics = _soft_high_metrics(q[:, 2], baseline_high)
    baseline = {
        "soft_log_loss": soft_cross_entropy(q, baseline_probability),
        "multiclass_brier": multiclass_brier(q, baseline_probability),
        "posterior_expected_hard_accuracy": _posterior_expected_accuracy(
            q, baseline_decision
        ),
        **baseline_high_metrics,
    }

    rows: list[dict[str, Any]] = []
    derived: dict[str, Any] = {
        "baseline_decision": baseline_decision,
        "baseline_high": baseline_high,
        "permuted_decision": {},
        "permuted_high": {},
    }
    for spec in FEATURES:
        probability = np.asarray(
            aggregated["permuted_probability"][spec.feature], dtype=float
        )
        margin = np.asarray(
            aggregated["permuted_margin"][spec.feature], dtype=float
        )
        decision, high = _policy_decisions(probability, margin)
        derived["permuted_decision"][spec.feature] = decision
        derived["permuted_high"][spec.feature] = high
        expected_accuracy = np.asarray(
            _posterior_expected_accuracy(q, decision), dtype=float
        )
        high_metrics = _vector_soft_high_metrics(q[:, 2], high)
        probability_losses = np.asarray(
            [soft_cross_entropy(q, value) for value in probability],
            dtype=float,
        )
        probability_brier = np.asarray(
            [multiclass_brier(q, value) for value in probability],
            dtype=float,
        )
        alert_flips = np.sum(high != baseline_high[None, :], axis=1)
        for permutation in range(len(probability)):
            rows.append(
                {
                    "feature": spec.feature,
                    "display_name": spec.display_name,
                    "branch": spec.branch,
                    "permutation": int(permutation),
                    "soft_log_loss": float(probability_losses[permutation]),
                    "importance_soft_log_loss": float(
                        probability_losses[permutation]
                        - baseline["soft_log_loss"]
                    ),
                    "multiclass_brier": float(probability_brier[permutation]),
                    "importance_multiclass_brier": float(
                        probability_brier[permutation]
                        - baseline["multiclass_brier"]
                    ),
                    "soft_f0_5": float(
                        high_metrics["soft_f0_5"][permutation]
                    ),
                    "importance_soft_f0_5": float(
                        baseline["soft_f0_5"]
                        - high_metrics["soft_f0_5"][permutation]
                    ),
                    "posterior_expected_hard_accuracy": float(
                        expected_accuracy[permutation]
                    ),
                    "importance_posterior_expected_hard_accuracy": float(
                        baseline["posterior_expected_hard_accuracy"]
                        - expected_accuracy[permutation]
                    ),
                    "soft_high_recall": float(
                        high_metrics["soft_high_recall"][permutation]
                    ),
                    "soft_high_precision": float(
                        high_metrics["soft_high_precision"][permutation]
                    ),
                    "soft_high_false_positive_rate": float(
                        high_metrics[
                            "soft_high_false_positive_rate"
                        ][permutation]
                    ),
                    "predicted_high_count": int(
                        high_metrics["predicted_high_count"][permutation]
                    ),
                    "alert_flip_count": int(alert_flips[permutation]),
                }
            )
    return baseline, pd.DataFrame(rows), derived


def _bootstrap_indices(groups: np.ndarray, repetitions: int) -> list[np.ndarray]:
    group_values = np.asarray(groups, dtype=object)
    unique = np.asarray(sorted(set(map(str, group_values))), dtype=object)
    if len(unique) != EXPECTED_GROUPS:
        raise ValueError("Connected-group bootstrap requires 80 groups")
    lookup = {
        str(group): np.flatnonzero(group_values == str(group)) for group in unique
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: list[np.ndarray] = []
    for _ in range(int(repetitions)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        output.append(
            np.concatenate([lookup[str(group)] for group in sampled])
        )
    return output


def _bootstrap_reliance(
    aggregated: Mapping[str, Any],
    derived: Mapping[str, Any],
    *,
    repetitions: int,
) -> pd.DataFrame:
    q = np.asarray(aggregated["q"], dtype=float)
    baseline_probability = np.asarray(
        aggregated["baseline_probability"], dtype=float
    )
    baseline_high = np.asarray(derived["baseline_high"], dtype=bool)
    baseline_decision = np.asarray(derived["baseline_decision"], dtype=int)
    indices_list = _bootstrap_indices(
        np.asarray(aggregated["groups"], dtype=object), repetitions
    )
    rows: list[dict[str, Any]] = []
    for replicate, indices in enumerate(indices_list):
        q_selected = q[indices]
        baseline_probability_selected = baseline_probability[indices]
        baseline_loss = soft_cross_entropy(
            q_selected, baseline_probability_selected
        )
        baseline_brier = multiclass_brier(
            q_selected, baseline_probability_selected
        )
        baseline_high_metrics = _soft_high_metrics(
            q_selected[:, 2], baseline_high[indices]
        )
        baseline_accuracy = float(
            _posterior_expected_accuracy(
                q_selected, baseline_decision[indices]
            )
        )
        for spec in FEATURES:
            probability = np.asarray(
                aggregated["permuted_probability"][spec.feature], dtype=float
            )[:, indices, :]
            high = np.asarray(
                derived["permuted_high"][spec.feature], dtype=bool
            )[:, indices]
            decision = np.asarray(
                derived["permuted_decision"][spec.feature], dtype=int
            )[:, indices]
            probability_loss = np.mean(
                [soft_cross_entropy(q_selected, value) for value in probability]
            )
            probability_brier = np.mean(
                [multiclass_brier(q_selected, value) for value in probability]
            )
            high_metrics = _vector_soft_high_metrics(q_selected[:, 2], high)
            expected_accuracy = np.asarray(
                _posterior_expected_accuracy(q_selected, decision), dtype=float
            )
            rows.append(
                {
                    "feature": spec.feature,
                    "replicate": int(replicate),
                    "importance_soft_log_loss": float(
                        probability_loss - baseline_loss
                    ),
                    "importance_multiclass_brier": float(
                        probability_brier - baseline_brier
                    ),
                    "importance_soft_f0_5": float(
                        baseline_high_metrics["soft_f0_5"]
                        - np.mean(high_metrics["soft_f0_5"])
                    ),
                    "importance_posterior_expected_hard_accuracy": float(
                        baseline_accuracy - np.mean(expected_accuracy)
                    ),
                    "permuted_minus_baseline_soft_high_fpr": float(
                        np.mean(
                            high_metrics["soft_high_false_positive_rate"]
                        )
                        - baseline_high_metrics[
                            "soft_high_false_positive_rate"
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _interval_status(low: float, high: float) -> str:
    if low > 0:
        return "positive_conditional_interval"
    if high < 0:
        return "negative_conditional_interval"
    return "conditional_interval_crosses_zero"


def _feature_summary(
    baseline: Mapping[str, Any],
    permutation_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    n_permutations: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for order, spec in enumerate(FEATURES):
        observed = permutation_metrics.loc[
            permutation_metrics["feature"].eq(spec.feature)
        ].copy()
        draws = bootstrap.loc[bootstrap["feature"].eq(spec.feature)].copy()
        weight = weights.loc[weights["feature"].eq(spec.feature)].copy()
        probability_values = draws["importance_soft_log_loss"].to_numpy(float)
        policy_values = draws["importance_soft_f0_5"].to_numpy(float)
        accuracy_values = draws[
            "importance_posterior_expected_hard_accuracy"
        ].to_numpy(float)
        brier_values = draws["importance_multiclass_brier"].to_numpy(float)
        fpr_values = draws[
            "permuted_minus_baseline_soft_high_fpr"
        ].to_numpy(float)
        probability_low, probability_high = np.quantile(
            probability_values, [0.025, 0.975]
        )
        policy_low, policy_high = np.quantile(
            policy_values, [0.025, 0.975]
        )
        accuracy_low, accuracy_high = np.quantile(
            accuracy_values, [0.025, 0.975]
        )
        rows.append(
            {
                "display_order": int(order),
                "feature": spec.feature,
                "display_name": spec.display_name,
                "branch": spec.branch,
                "source_columns": "|".join(spec.source_columns),
                "high_gate_gower_block": spec.high_block or "not_used",
                "conditional_gower_block": spec.conditional_block or "not_used",
                "mean_high_gate_nominal_gower_weight": float(
                    weight["high_gate_gower_weight"].mean()
                ),
                "mean_conditional_nominal_gower_weight": float(
                    weight["conditional_gower_weight"].mean()
                ),
                "active_high_gate_outer_splits": int(
                    weight["high_gate_gower_weight"].gt(0).sum()
                ),
                "active_conditional_outer_splits": int(
                    weight["conditional_gower_weight"].gt(0).sum()
                ),
                "n_records": EXPECTED_RECORDS,
                "n_connected_groups": EXPECTED_GROUPS,
                "n_outer_splits": 50,
                "n_permutations": int(n_permutations),
                "n_bootstrap": int(n_bootstrap),
                "baseline_soft_log_loss": float(baseline["soft_log_loss"]),
                "permuted_soft_log_loss_mean": float(
                    observed["soft_log_loss"].mean()
                ),
                "importance_soft_log_loss": float(
                    observed["importance_soft_log_loss"].mean()
                ),
                "importance_soft_log_loss_ci95_low": float(probability_low),
                "importance_soft_log_loss_ci95_high": float(probability_high),
                "soft_log_loss_positive_permutation_fraction": float(
                    observed["importance_soft_log_loss"].gt(0).mean()
                ),
                "probability_reliance_status": _interval_status(
                    float(probability_low), float(probability_high)
                ),
                "baseline_multiclass_brier": float(
                    baseline["multiclass_brier"]
                ),
                "importance_multiclass_brier": float(
                    observed["importance_multiclass_brier"].mean()
                ),
                "importance_multiclass_brier_ci95_low": float(
                    np.quantile(brier_values, 0.025)
                ),
                "importance_multiclass_brier_ci95_high": float(
                    np.quantile(brier_values, 0.975)
                ),
                "baseline_soft_f0_5": float(baseline["soft_f0_5"]),
                "permuted_soft_f0_5_mean": float(
                    observed["soft_f0_5"].mean()
                ),
                "importance_soft_f0_5": float(
                    observed["importance_soft_f0_5"].mean()
                ),
                "importance_soft_f0_5_ci95_low": float(policy_low),
                "importance_soft_f0_5_ci95_high": float(policy_high),
                "soft_f0_5_positive_permutation_fraction": float(
                    observed["importance_soft_f0_5"].gt(0).mean()
                ),
                "policy_reliance_status": (
                    "structurally_not_used_by_high_gate"
                    if spec.high_column is None
                    else _interval_status(float(policy_low), float(policy_high))
                ),
                "baseline_posterior_expected_hard_accuracy": float(
                    baseline["posterior_expected_hard_accuracy"]
                ),
                "importance_posterior_expected_hard_accuracy": float(
                    observed[
                        "importance_posterior_expected_hard_accuracy"
                    ].mean()
                ),
                "importance_posterior_expected_hard_accuracy_ci95_low": float(
                    accuracy_low
                ),
                "importance_posterior_expected_hard_accuracy_ci95_high": float(
                    accuracy_high
                ),
                "baseline_soft_high_recall": float(
                    baseline["soft_high_recall"]
                ),
                "permuted_soft_high_recall_mean": float(
                    observed["soft_high_recall"].mean()
                ),
                "baseline_soft_high_precision": float(
                    baseline["soft_high_precision"]
                ),
                "permuted_soft_high_precision_mean": float(
                    observed["soft_high_precision"].mean()
                ),
                "baseline_soft_high_false_positive_rate": float(
                    baseline["soft_high_false_positive_rate"]
                ),
                "permuted_soft_high_false_positive_rate_mean": float(
                    observed["soft_high_false_positive_rate"].mean()
                ),
                "permuted_minus_baseline_soft_high_fpr_ci95_low": float(
                    np.quantile(fpr_values, 0.025)
                ),
                "permuted_minus_baseline_soft_high_fpr_ci95_high": float(
                    np.quantile(fpr_values, 0.975)
                ),
                "baseline_predicted_high_count": int(
                    baseline["predicted_high_count"]
                ),
                "permuted_predicted_high_count_mean": float(
                    observed["predicted_high_count"].mean()
                ),
                "mean_alert_flip_count": float(
                    observed["alert_flip_count"].mean()
                ),
                "interpretation_status": "posthoc_model_reliance_only",
                "feature_claim_allowed": False,
                "causal_claim_allowed": False,
                "mechanistic_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows).sort_values("display_order", kind="stable")


def _plot_summary(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    frame = summary.sort_values("display_order", kind="stable").copy()
    y = np.arange(len(frame))
    marker_by_branch = {
        "both_branches": "o",
        "high_gate_only": "^",
        "conditional_only": "s",
    }
    apply_publication_style(10)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 6.2),
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    panels = (
        (
            axes[0],
            "importance_soft_log_loss",
            "importance_soft_log_loss_ci95_low",
            "importance_soft_log_loss_ci95_high",
            PALETTE["blue_main"],
            "Probability engine",
            "Δ soft log loss\n(permuted − baseline)",
        ),
        (
            axes[1],
            "importance_soft_f0_5",
            "importance_soft_f0_5_ci95_low",
            "importance_soft_f0_5_ci95_high",
            PALETTE["teal"],
            "High-alert policy",
            "Δ soft F0.5\n(baseline − permuted)",
        ),
    )
    for axis, estimate_col, low_col, high_col, color, title, xlabel in panels:
        estimates = frame[estimate_col].to_numpy(float)
        lows = frame[low_col].to_numpy(float)
        highs = frame[high_col].to_numpy(float)
        axis.axvline(0.0, color=PALETTE["neutral_dark"], linewidth=1.0)
        axis.axhline(6.5, color=PALETTE["neutral"], linewidth=0.8)
        axis.errorbar(
            estimates,
            y,
            xerr=[estimates - lows, highs - estimates],
            fmt="none",
            ecolor=PALETTE["neutral_dark"],
            capsize=3,
            linewidth=1.2,
            zorder=1,
        )
        for position, (_, row) in enumerate(frame.iterrows()):
            axis.scatter(
                float(row[estimate_col]),
                position,
                color=color,
                marker=marker_by_branch[str(row["branch"])],
                s=58,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#E5E5E5", linewidth=0.6, zorder=0)
        bound = float(np.nanmax(np.abs(np.concatenate([lows, highs]))))
        bound = max(bound * 1.15, 1.0e-4)
        axis.set_xlim(-bound, bound)
    axes[0].set_yticks(y, frame["display_name"].tolist())
    axes[0].invert_yaxis()
    fig.suptitle(
        "PRESTO policy-v2 stage-aware permutation reliance",
        fontsize=13,
        weight="bold",
        y=0.98,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker_by_branch[branch],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=label,
        )
        for branch, label in (
            ("both_branches", "both branches"),
            ("high_gate_only", "high gate only"),
            ("conditional_only", "low/120 conditional only"),
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Positive values indicate reliance. Points are means over outer-test "
            "permutations; bars are 95% connected-group bootstrap intervals. "
            "Post-hoc model audit; includes frozen Gower block weighting."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color=PALETTE["neutral_dark"],
    )
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.24, right=0.98)
    return save_figure(
        fig,
        output_dir / "presto_policy_v2_stage_aware_permutation_reliance",
        formats=("png", "pdf", "svg"),
        dpi=300,
    )


def _write_outputs(
    output_dir: Path,
    *,
    source_qa: Mapping[str, Any],
    aggregated: Mapping[str, Any],
    baseline: Mapping[str, Any],
    permutation_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    summary: pd.DataFrame,
    n_permutations: int,
    n_bootstrap: int,
    figure_paths: Sequence[Path],
) -> dict[str, Any]:
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPOSITORY_ROOT))
        except ValueError:
            return str(path)

    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "presto_policy_v2_feature_reliance.csv"
    summary.to_csv(summary_path, index=False)
    permutation_metrics.to_csv(
        tables / "permutation_replicate_metrics.csv", index=False
    )
    bootstrap.to_csv(tables / "connected_group_bootstrap.csv", index=False)

    record_rows: list[dict[str, Any]] = []
    baseline_probability = np.asarray(
        aggregated["baseline_probability"], dtype=float
    )
    baseline_margin = np.asarray(aggregated["baseline_margin"], dtype=float)
    q = np.asarray(aggregated["q"], dtype=float)
    for index, record_id in enumerate(aggregated["record_ids"]):
        record_rows.append(
            {
                "record_id": str(record_id),
                "cv_group_id": str(aggregated["groups"][index]),
                **{
                    column: float(q[index, class_index])
                    for class_index, column in enumerate(Q_COLUMNS)
                },
                **{
                    column: float(baseline_probability[index, class_index])
                    for class_index, column in enumerate(P_COLUMNS)
                },
                "mean_policy_margin": float(baseline_margin[index]),
            }
        )
    pd.DataFrame(record_rows).to_csv(
        tables / "unpermuted_record_aggregated.csv", index=False
    )

    qa = {
        "status": "pass",
        "analysis_id": "presto_policy_v2_stage_aware_permutation_reliance_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": {
            "analysis_role": "posthoc_descriptive_model_reliance",
            "feature_claim_allowed": False,
            "causal_claim_allowed": False,
            "mechanistic_claim_allowed": False,
            "independent_confirmation": False,
        },
        "source": source_qa,
        "coverage": {
            "records": EXPECTED_RECORDS,
            "connected_groups": EXPECTED_GROUPS,
            "outer_repeats": 10,
            "outer_folds": 5,
            "outer_splits": 50,
            "logical_features": len(FEATURES),
            "permutations_per_feature": int(n_permutations),
            "bootstrap_repetitions": int(n_bootstrap),
        },
        "unpermuted_reproduction": {
            "maximum_probability_abs_error": float(
                aggregated["maximum_probability_refit_error"]
            ),
            "maximum_policy_score_abs_error": float(
                aggregated["maximum_policy_score_refit_error"]
            ),
            "maximum_threshold_abs_error": float(
                aggregated["maximum_threshold_error"]
            ),
            "baseline_soft_log_loss": float(baseline["soft_log_loss"]),
            "baseline_multiclass_brier": float(baseline["multiclass_brier"]),
            "baseline_soft_f0_5": float(baseline["soft_f0_5"]),
            "baseline_posterior_expected_hard_accuracy": float(
                baseline["posterior_expected_hard_accuracy"]
            ),
            "baseline_predicted_high_count": int(
                baseline["predicted_high_count"]
            ),
        },
        "outputs": {
            "summary": display_path(summary_path),
            "permutation_metrics": display_path(
                tables / "permutation_replicate_metrics.csv"
            ),
            "bootstrap": display_path(
                tables / "connected_group_bootstrap.csv"
            ),
            "figures": [display_path(path) for path in figure_paths],
        },
    }
    (output_dir / "qa.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return qa


def run_analysis(
    output_dir: Path,
    *,
    n_permutations: int,
    n_bootstrap: int,
    n_jobs: int,
) -> dict[str, Any]:
    if int(n_permutations) < 1 or int(n_bootstrap) < 1 or int(n_jobs) < 1:
        raise ValueError("permutations, bootstrap, and n_jobs must be positive")
    source_qa = _verify_authoritative_sources()
    inputs = _load_inputs()
    records_by_id = _indexed(inputs["records"], name="records")
    labels_by_id = _indexed(inputs["labels"], name="labels")
    split_groups = [
        (
            int(repeat),
            int(fold),
            frame["record_id"].astype(str).tolist(),
        )
        for (repeat, fold), frame in inputs["outer"].groupby(
            ["repeat", "fold"], sort=True
        )
    ]
    results = Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(_run_outer_split)(
            records_by_id,
            labels_by_id,
            inputs["parent_selections"],
            inputs["policy_selections"],
            inputs["probability_oof"],
            inputs["policy_oof"],
            repeat=repeat,
            fold=fold,
            test_ids=test_ids,
            n_permutations=int(n_permutations),
        )
        for repeat, fold, test_ids in split_groups
    )
    aggregated = _aggregate_outer_results(
        results,
        records_by_id,
        labels_by_id,
        n_permutations=int(n_permutations),
    )
    baseline, permutation_metrics, derived = (
        _observed_and_permutation_metrics(aggregated)
    )

    expected_probability_metrics = pd.read_csv(
        VERIFIED_RUN / "analyses" / "aggregate_metrics.csv"
    ).loc[
        lambda frame: frame["evaluation_model"].astype(str).eq(
            EXPECTED_PROBABILITY_MODEL
        )
    ].iloc[0]
    expected_policy_metrics = pd.read_csv(
        VERIFIED_RUN / "analyses" / "policy_aggregate_metrics.csv"
    ).loc[
        lambda frame: frame["decision_policy"].astype(str).eq(EXPECTED_POLICY)
    ].iloc[0]
    metric_errors = {
        "soft_log_loss": abs(
            float(baseline["soft_log_loss"])
            - float(
                expected_probability_metrics[
                    "full_coverage_three_class_soft_log_loss"
                ]
            )
        ),
        "multiclass_brier": abs(
            float(baseline["multiclass_brier"])
            - float(
                expected_probability_metrics[
                    "multiclass_brier_mean_sum_squared_error"
                ]
            )
        ),
        "posterior_expected_hard_accuracy": abs(
            float(baseline["posterior_expected_hard_accuracy"])
            - float(
                expected_policy_metrics[
                    "posterior_expected_hard_accuracy"
                ]
            )
        ),
        "predicted_high_count": abs(
            int(baseline["predicted_high_count"])
            - int(expected_policy_metrics["predicted_high_count"])
        ),
    }
    if max(float(value) for value in metric_errors.values()) > BASELINE_TOLERANCE:
        raise RuntimeError(
            "Unpermuted aggregate metrics do not reproduce verified PRESTO "
            f"policy-v2: {metric_errors}"
        )

    bootstrap = _bootstrap_reliance(
        aggregated,
        derived,
        repetitions=int(n_bootstrap),
    )
    summary = _feature_summary(
        baseline,
        permutation_metrics,
        bootstrap,
        aggregated["weights"],
        n_permutations=int(n_permutations),
        n_bootstrap=int(n_bootstrap),
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = _plot_summary(summary, figures_dir)
    qa = _write_outputs(
        output_dir,
        source_qa=source_qa,
        aggregated=aggregated,
        baseline=baseline,
        permutation_metrics=permutation_metrics,
        bootstrap=bootstrap,
        summary=summary,
        n_permutations=int(n_permutations),
        n_bootstrap=int(n_bootstrap),
        figure_paths=figure_paths,
    )
    return {"qa": qa, "summary": summary}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory (default: Data/04_results/RESULTS/06_FEATURE_IMPORTANCE)",
    )
    parser.add_argument(
        "--permutations", type=int, default=DEFAULT_PERMUTATIONS
    )
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_analysis(
        args.output_dir.resolve(),
        n_permutations=int(args.permutations),
        n_bootstrap=int(args.bootstrap),
        n_jobs=int(args.n_jobs),
    )
    summary = result["summary"]
    print(
        summary.loc[
            :,
            [
                "feature",
                "importance_soft_log_loss",
                "importance_soft_log_loss_ci95_low",
                "importance_soft_log_loss_ci95_high",
                "importance_soft_f0_5",
                "importance_soft_f0_5_ci95_low",
                "importance_soft_f0_5_ci95_high",
            ],
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
