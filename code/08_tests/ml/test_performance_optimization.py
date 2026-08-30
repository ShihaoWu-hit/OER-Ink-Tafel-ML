from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from ink_tafel_study.optimization_components import make_candidate_regressor_specs
from ink_tafel_study.performance_optimization import (
    CandidateDefinition,
    OptimizationConfig,
    TARGET_COLUMN,
    TARGET_NAME,
    _prepare_canonical_inputs,
    build_candidate_catalog,
    choose_inner_winner,
    _write_summary_report,
    recipes_for_policy,
    run_performance_optimization,
    select_outer_training_winner,
)


def _grouped_synthetic_frame(n_groups: int = 12, records_per_group: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = n_groups * records_per_group
    concentration = rng.uniform(0.1, 2.0, n)
    frame = pd.DataFrame(
        {
            "record_id": np.arange(1, n + 1).astype(str),
            "curve_id": [f"curve_{index:03d}" for index in range(n)],
            "publication_id": [
                f"publication_{index // records_per_group:03d}" for index in range(n)
            ],
            "lab_group_id": [f"lab_unknown_{index:03d}" for index in range(n)],
            "recipe_hash": [
                f"recipe_{index // records_per_group:03d}" for index in range(n)
            ],
            "sample_id": [f"sample_{index:03d}" for index in range(n)],
            "cv_group_id": [
                f"group_{index // records_per_group:03d}" for index in range(n)
            ],
            "catalyst_loading_mg_cm2_selected": concentration,
            "electrolyte_molarity_M": rng.choice([0.1, 0.5, 1.0], n),
            "rpm": rng.choice([0.0, 400.0, 1600.0], n),
            "scan_rate_mV_s": rng.choice([5.0, 10.0, 20.0], n),
            "electrolyte_identity": rng.choice(["KOH", "NaOH"], n),
            "ir_correction_status": rng.choice(
                ["reported_corrected", "reported_not_corrected"], n
            ),
        }
    )
    frame["synthetic_target"] = 30.0 + 150.0 * concentration
    return frame


def _two_candidate_catalog():
    specs, _ = make_candidate_regressor_specs(include_catboost=False)
    candidates = build_candidate_catalog(
        ("conditions_noink_none",),
        {name: specs[name] for name in ("dummy_mean", "dummy_median", "ridge")},
        budget=8,
        random_seed=42,
    )
    return specs, candidates


def test_outer_test_target_changes_cannot_change_inner_selection() -> None:
    frame = _grouped_synthetic_frame()
    specs, candidates = _two_candidate_catalog()
    outer_train = frame.iloc[:16].reset_index(drop=True)
    X_train = outer_train.loc[
        :,
        [
            "catalyst_loading_mg_cm2_selected",
            "electrolyte_molarity_M",
            "rpm",
            "scan_rate_mV_s",
            "electrolyte_identity",
            "ir_correction_status",
        ],
    ]
    y_all = frame["synthetic_target"].to_numpy(float)
    first = select_outer_training_winner(
        outer_train,
        X_train,
        y_all[:16],
        candidates,
        specs,
        inner_folds=3,
        inner_seed=77,
        tie_tolerance_mV_dec=0.0,
        n_jobs=1,
    )[0]
    poisoned = y_all.copy()
    poisoned[16:] = 1.0e12
    second = select_outer_training_winner(
        outer_train,
        X_train,
        poisoned[:16],
        candidates,
        specs,
        inner_folds=3,
        inner_seed=77,
        tie_tolerance_mV_dec=0.0,
        n_jobs=1,
    )[0]
    assert first.candidate_id == second.candidate_id


def test_joint_selector_recovers_linear_feature_model_signal() -> None:
    frame = _grouped_synthetic_frame()
    specs, candidates = _two_candidate_catalog()
    X = frame.loc[
        :,
        [
            "catalyst_loading_mg_cm2_selected",
            "electrolyte_molarity_M",
            "rpm",
            "scan_rate_mV_s",
            "electrolyte_identity",
            "ir_correction_status",
        ],
    ]
    winner, _, scores, _, assignments, _, _ = select_outer_training_winner(
        frame,
        X,
        frame["synthetic_target"].to_numpy(float),
        candidates,
        specs,
        inner_folds=4,
        inner_seed=42,
        tie_tolerance_mV_dec=0.0,
        n_jobs=1,
    )
    assert winner.model == "ridge"
    assert winner.recipe == "conditions_noink_none"
    assert set(assignments) == {0, 1, 2, 3}
    assert scores["candidate_id"].nunique() == len(candidates)
    assert scores["status"].eq("complete").all()


def test_near_tie_uses_complexity_only_inside_tolerance() -> None:
    complex_candidate = CandidateDefinition(
        "complex",
        "conditions_noink_none",
        "raw",
        "identity",
        "ridge",
        {"alpha": 1.0},
        3,
        10,
    )
    simple_candidate = CandidateDefinition(
        "simple", None, "raw", "identity", "dummy_mean", {}, 0, 0
    )
    lookup = {item.candidate_id: item for item in (complex_candidate, simple_candidate)}
    scores = pd.DataFrame(
        [
            {
                "candidate_id": "complex",
                "inner_mae_mV_dec": 10.0,
                "status": "complete",
                "complexity_rank": 3,
                "feature_complexity": 10,
            },
            {
                "candidate_id": "simple",
                "inner_mae_mV_dec": 10.05,
                "status": "complete",
                "complexity_rank": 0,
                "feature_complexity": 0,
            },
        ]
    )
    strict, _ = choose_inner_winner(
        scores,
        tie_tolerance_mV_dec=0.01,
        candidate_lookup=lookup,
    )
    near_tie, details = choose_inner_winner(
        scores,
        tie_tolerance_mV_dec=0.1,
        candidate_lookup=lookup,
    )
    assert strict.candidate_id == "complex"
    assert near_tie.candidate_id == "simple"
    assert details["simplicity_tie_break_applied"] is True


def test_frozen_catalog_covers_every_recipe_model_and_transform() -> None:
    specs, _ = make_candidate_regressor_specs(include_catboost=False)
    recipes = recipes_for_policy("primary")
    candidates = build_candidate_catalog(recipes, specs, budget=96, random_seed=42)
    assert len(candidates) == 96
    assert set(recipes).issubset(
        {candidate.recipe for candidate in candidates if candidate.recipe is not None}
    )
    assert set(specs) == {candidate.model for candidate in candidates}
    assert {candidate.numeric_transform for candidate in candidates} == {"raw", "log"}
    assert {candidate.target_transform for candidate in candidates} == {
        "identity",
        "log10",
    }
    for recipe in recipes:
        assert {
            candidate.numeric_transform
            for candidate in candidates
            if candidate.recipe == recipe
        } == {"raw", "log"}
    frame = pd.DataFrame([candidate.row() for candidate in candidates])
    recipe_counts = frame.loc[frame["recipe"].ne("dummy"), "recipe"].value_counts()
    model_counts = frame.loc[
        ~frame["model"].str.startswith("dummy_"), "model"
    ].value_counts()
    assert recipe_counts.max() - recipe_counts.min() <= 1
    assert model_counts.max() - model_counts.min() <= 1
    repeated = build_candidate_catalog(recipes, specs, budget=96, random_seed=42)
    assert [item.candidate_id for item in candidates] == [
        item.candidate_id for item in repeated
    ]


def test_full_configuration_requires_ten_outer_oof_repeats() -> None:
    config = OptimizationConfig()
    assert config.outer_repeats == 10
    assert len(config.outer_seeds) == 10
    assert config.outer_folds == 5
    assert config.inner_folds == 4


def test_strict_complete_case_changes_cohort_not_candidate_recipes() -> None:
    assert recipes_for_policy("strict_complete_case") == recipes_for_policy("primary")


def test_smoke_report_suppresses_scientific_classification_and_locking(
    tmp_path: Path,
) -> None:
    metrics = pd.DataFrame(
        [
            {
                "prediction_role": "joint_selector",
                "mae_mV_dec": 70.0,
                "r2": 0.1,
                "spearman": 0.2,
                "tail_mae_mV_dec": 90.0,
            },
            {
                "prediction_role": "dummy_selector",
                "mae_mV_dec": 80.0,
                "r2": -0.1,
                "spearman": 0.0,
                "tail_mae_mV_dec": 100.0,
            },
        ]
    )
    decision = {
        "classification": "Promising only",
        "delta_mae_joint_minus_dummy_mV_dec": -10.0,
        "delta_bootstrap_q025_mV_dec": -20.0,
        "delta_bootstrap_q975_mV_dec": 5.0,
    }
    final_spec = {
        "pipeline": {
            "candidate_id": "test",
            "recipe": "conditions_noink_none",
            "numeric_transform": "raw",
            "target_transform": "identity",
            "model": "ridge",
            "model_params_json": "{}",
        }
    }
    destination = tmp_path / "report.md"
    _write_summary_report(
        destination,
        decision,
        SimpleNamespace(metrics=metrics),
        final_spec,
        smoke=True,
    )
    report = destination.read_text(encoding="utf-8")
    assert "Status: **Structural smoke pass**" in report
    assert "Final locked full-data pipeline" not in report
    assert "Promising only" not in report
    assert "test artifact rather than a locked model" in report
    assert "not definitive lab-held-out validation" in report


def test_core_refuses_registered_formal_run_as_destination() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    formal_run = repository_root / "Data" / "03_outputs" / "study" / "outputs" / "current_97"
    with pytest.raises(ValueError, match="registered formal run"):
        run_performance_optimization(
            formal_run,
            project_root=repository_root,
            smoke=True,
        )


def test_core_refuses_stale_stage_outputs(tmp_path: Path) -> None:
    (tmp_path / "analyses").mkdir()
    with pytest.raises(FileExistsError, match="stale output mixing"):
        run_performance_optimization(
            tmp_path,
            project_root=Path(__file__).resolve().parents[2],
            smoke=True,
        )


def test_canonical_optimization_inputs_use_approved_target_registry() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    canonical_dir = repository_root / "Data" / "02_inputs" / "canonical_dataset_ir_corrected_v2"
    config = yaml.safe_load(
        (
            repository_root
            / "Ink_Tafel_Study"
            / "config"
            / "study_posterior_cardinal_v2_20260724.yaml"
        ).read_text(encoding="utf-8")
    )
    expected_tree = config["input_hashes"]["expected"]["canonical_dataset"]

    frame, binding = _prepare_canonical_inputs(
        canonical_dir,
        expected_tree_sha256=expected_tree,
        approved_target_registry=(
            repository_root
            / "Ink_Tafel_Study"
            / "outputs"
            / "current_97"
            / "targets"
            / "target_registry.csv"
        ),
        target_qa_path=(
            repository_root
            / "Ink_Tafel_Study"
            / "outputs"
            / "current_97"
            / "targets"
            / "target_qa.json"
        ),
    )

    registry = pd.read_csv(
        repository_root
        / "Ink_Tafel_Study"
        / "outputs"
        / "current_97"
        / "targets"
        / "target_registry.csv"
    )
    approved = registry.loc[
        registry["target_name"].astype(str).eq(TARGET_NAME)
        & registry["target_role"].astype(str).eq("primary")
        & registry["target_eligible"].astype(bool)
        & registry["formal_target_approved"].astype(bool)
    ]
    assert len(frame) == 97
    assert len(approved) == 97
    assert frame[TARGET_COLUMN].notna().all()
    assert (
        Path(binding["target_registry_csv"]).parent.name == "targets"
    )
    assert binding["target_qa_formal_evaluation_ready"] is True
