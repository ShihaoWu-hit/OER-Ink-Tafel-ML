from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ink_tafel_study.figures as figures_module
from ink_tafel_study.figures import generate_figures
from ink_tafel_study.interpretation import (
    InterpretationConfig,
    build_interpretation_layer,
    outer_test_block_perturbations,
)
from ink_tafel_study.models import make_model_spec


TARGETS = (
    "b_bayesian_local_line_mV_dec",
    "b_single_regime_mV_dec",
)
MODELS = ("bayesian_ridge", "gower_knn", "extra_trees")
IMPORTANCE_PANEL = "context_plus_property_proxy_core"


def test_interpretation_direct_api_defaults_to_property_proxy_panel() -> None:
    assert InterpretationConfig().importance_panel == IMPORTANCE_PANEL


def test_outer_test_block_perturbations_are_posthoc_and_partitioned() -> None:
    train = pd.DataFrame(
        {
            "amount": np.linspace(0.0, 1.0, 12),
            "composition": np.tile([0.2, 0.8], 6),
        }
    )
    target = np.log10(50.0 + 20.0 * train["amount"] + 10.0 * train["composition"])
    spec = make_model_spec(
        "bayesian_ridge",
        numeric_columns=("amount", "composition"),
        categorical_columns=(),
    )
    fitted = spec.estimator.fit(train, target)
    test = pd.DataFrame({"amount": [0.1, 0.4, 0.7, 0.9], "composition": [0.8, 0.2, 0.8, 0.2]})
    observed = 50.0 + 20.0 * test["amount"] + 10.0 * test["composition"]
    baseline = fitted.predict(test)

    effects, importance = outer_test_block_perturbations(
        fitted,
        test,
        observed,
        baseline,
        blocks={"ink_amounts": ("amount",), "solvent_composition": ("composition",)},
        record_ids=["r1", "r2", "r3", "r4"],
        cv_group_ids=["g1", "g2", "g3", "g4"],
        target=TARGETS[0],
        scope="alkaline",
        panel="context_plus_ink",
        model="bayesian_ridge",
        repeat=0,
        fold=0,
        selected_params="{}",
        permutation_repeats=2,
        random_seed=17,
    )

    assert len(importance) == 2 * (1 + 2)
    assert len(effects) == len(test) * len(importance)
    assert set(importance["perturbation_method"]) == {
        "block_masking_to_fitted_missingness_policy",
        "joint_block_permutation",
    }
    assert importance["outer_test_only"].all()
    assert set(importance["selection_role"]) == {"posthoc_metric_only_not_model_selection"}
    assert np.allclose(
        effects["delta_absolute_error_mV_dec"],
        effects["perturbed_absolute_error_mV_dec"] - effects["baseline_absolute_error_mV_dec"],
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        outer_test_block_perturbations(
            fitted,
            test,
            observed,
            baseline,
            blocks={"first": ("amount", "composition"), "second": ("composition",)},
            record_ids=["r1", "r2", "r3", "r4"],
            cv_group_ids=["g1", "g2", "g3", "g4"],
            target=TARGETS[0],
            scope="alkaline",
            panel="context_plus_ink",
            model="bayesian_ridge",
            repeat=0,
            fold=0,
            selected_params="{}",
        )


def _write_synthetic_interpretation_run(root: Path, *, verified_labs: bool = True) -> pd.DataFrame:
    records = pd.DataFrame(
        {
            "record_id": [f"r{index}" for index in range(8)],
            "lab_group_id": ["lab_a"] * 4 + ["lab_b"] * 4,
            "lab_group_confidence": ["verified" if verified_labs else "unknown"] * 8,
            "catalyst_type": ["RuO2"] * 4 + ["IrO2"] * 4,
            "electrolyte_identity": ["KOH", "KOH", "NaOH", "NaOH"] * 2,
        }
    )
    for target_index, target in enumerate(TARGETS):
        analysis = root / f"analysis_{target_index}"
        analysis.mkdir(parents=True)
        (analysis / "evaluation_config.json").write_text(
            json.dumps({"outer_folds": 2, "outer_repeats": 1}), encoding="utf-8"
        )
        (analysis / "evaluation_qa.json").write_text(
            json.dumps({"target": target, "scope": "alkaline"}), encoding="utf-8"
        )
        importance_rows: list[dict[str, object]] = []
        effect_rows: list[dict[str, object]] = []
        for model in MODELS:
            for fold in range(2):
                fold_records = records.loc[
                    np.arange(len(records)) % 2 == fold
                ]
                for method, perturbation_repeat, delta in (
                    ("joint_block_permutation", 0, 2.0),
                    ("block_masking_to_fitted_missingness_policy", -1, 1.0),
                ):
                    importance_rows.append(
                        {
                            "target": target,
                            "scope": "alkaline",
                            "panel": IMPORTANCE_PANEL,
                            "model": model,
                            "repeat": 0,
                            "fold": fold,
                            "block": "solvent_composition",
                            "perturbation_method": method,
                            "perturbation_repeat": perturbation_repeat,
                            "n_test": len(fold_records),
                            "baseline_mae_mV_dec": 5.0,
                            "perturbed_mae_mV_dec": 5.0 + delta,
                            "delta_mae_mV_dec": delta,
                            "outer_test_only": True,
                            "selection_role": "posthoc_metric_only_not_model_selection",
                        }
                    )
                    for record_index, record in fold_records.iterrows():
                        effect_rows.append(
                            {
                                "target": target,
                                "scope": "alkaline",
                                "panel": IMPORTANCE_PANEL,
                                "model": model,
                                "repeat": 0,
                                "fold": fold,
                                "record_id": record["record_id"],
                                "cv_group_id": f"g{record_index}",
                                "block": "solvent_composition",
                                "perturbation_method": method,
                                "perturbation_repeat": perturbation_repeat,
                                "baseline_absolute_error_mV_dec": 5.0,
                                "perturbed_absolute_error_mV_dec": 5.0 + delta,
                                "delta_absolute_error_mV_dec": delta,
                                "outer_test_only": True,
                                "selection_role": "posthoc_metric_only_not_model_selection",
                            }
                        )
        pd.DataFrame(importance_rows).to_csv(analysis / "outer_test_block_importance.csv", index=False)
        pd.DataFrame(effect_rows).to_csv(analysis / "outer_test_block_effects.csv", index=False)

        if target_index == 0:
            selection_rows: list[dict[str, object]] = []
            for fold in range(2):
                for candidate in MODELS:
                    selection_rows.append(
                        {
                            "target": target,
                            "scope": "alkaline",
                            "panel": IMPORTANCE_PANEL,
                            "outer_repeat": 0,
                            "outer_fold": fold,
                            "candidate_model_family": candidate,
                            "eligible_within_one_se": candidate != "bayesian_ridge",
                            "selected_model_family": "gower_knn",
                            "selected": candidate == "gower_knn",
                        }
                    )
            pd.DataFrame(selection_rows).to_csv(analysis / "model_selection_one_se.csv", index=False)
    return records


def test_stability_gate_requires_verified_labs_and_never_invents_counterfactuals(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    records = _write_synthetic_interpretation_run(evaluation, verified_labs=True)
    config = InterpretationConfig(
        permutation_repeats=1,
        minimum_outer_splits_per_cell=2,
        minimum_records_per_stratum=2,
        minimum_strata=2,
    )
    status = build_interpretation_layer(evaluation, records, config=config)
    stability = pd.read_csv(evaluation / "interpretation" / "interpretation_stability.csv")
    manifest = pd.read_csv(evaluation / "interpretation" / "interpretation_artifact_manifest.csv")

    assert status["n_blocks_passing_all_stability_dimensions"] == 1
    assert status["importance_artifact_audit"]["status"] == (
        "pass_complete_outer_test_posthoc_grid"
    )
    assert stability.loc[0, "panel"] == IMPORTANCE_PANEL
    assert stability.loc[0, "lab_status"] == "pass"
    assert bool(stability.loc[0, "overall_stability_pass"])
    # Passing the stability and one-SE gates does not make an unsaved
    # counterfactual or neighbour state magically identifiable.
    assert status["composition_substitution_status"] == "not_estimable_from_saved_artifacts"
    assert status["knn_one_se_gate"]["status"] == "pass"
    assert status["neighbor_case_status"] == "not_estimable_from_saved_artifacts"
    assert not (evaluation / "interpretation" / "solvent_substitution_effects.csv").exists()
    assert not (evaluation / "interpretation" / "knn_neighbor_cases.csv").exists()
    assert set(manifest["artifact"]) >= {
        "outer_test_block_importance",
        "composition_conserving_solvent_substitution",
        "knn_neighbor_case_studies",
    }

    unverified_root = tmp_path / "evaluation_unverified"
    unverified = _write_synthetic_interpretation_run(unverified_root, verified_labs=False)
    unverified_status = build_interpretation_layer(unverified_root, unverified, config=config)
    unverified_stability = pd.read_csv(
        unverified_root / "interpretation" / "interpretation_stability.csv"
    )
    assert unverified_status["n_blocks_passing_all_stability_dimensions"] == 0
    assert unverified_stability.loc[0, "lab_status"] == "not_estimable_unverified_lab_metadata"


def test_interpretation_normalizes_numeric_record_ids_loaded_from_csv(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    records = _write_synthetic_interpretation_run(evaluation, verified_labs=True)
    records["record_id"] = records["record_id"].str.removeprefix("r")
    for effects_path in evaluation.glob("analysis_*/outer_test_block_effects.csv"):
        effects = pd.read_csv(effects_path)
        effects["record_id"] = effects["record_id"].astype(str).str.removeprefix("r").astype(int)
        effects.to_csv(effects_path, index=False)

    status = build_interpretation_layer(
        evaluation,
        records,
        config=InterpretationConfig(
            permutation_repeats=1,
            minimum_outer_splits_per_cell=2,
            minimum_records_per_stratum=2,
            minimum_strata=2,
        ),
    )

    assert status["importance_artifact_audit"]["status"] == (
        "pass_complete_outer_test_posthoc_grid"
    )


def test_stability_uses_only_registered_importance_panel(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    records = _write_synthetic_interpretation_run(evaluation)
    for filename, delta_column in (
        ("outer_test_block_importance.csv", "delta_mae_mV_dec"),
        ("outer_test_block_effects.csv", "delta_absolute_error_mV_dec"),
    ):
        for path in evaluation.glob(f"*/{filename}"):
            registered = pd.read_csv(path)
            other_panel = registered.copy()
            other_panel["panel"] = "context_plus_ink"
            other_panel[delta_column] = -100.0
            if filename == "outer_test_block_importance.csv":
                other_panel["perturbed_mae_mV_dec"] = (
                    other_panel["baseline_mae_mV_dec"] - 100.0
                )
            else:
                other_panel["perturbed_absolute_error_mV_dec"] = (
                    other_panel["baseline_absolute_error_mV_dec"] - 100.0
                )
            pd.concat([registered, other_panel], ignore_index=True).to_csv(path, index=False)

    config = InterpretationConfig(
        permutation_repeats=1,
        minimum_outer_splits_per_cell=2,
        minimum_records_per_stratum=2,
        minimum_strata=2,
    )
    status = build_interpretation_layer(evaluation, records, config=config)
    stability = pd.read_csv(evaluation / "interpretation" / "interpretation_stability.csv")

    assert status["importance_status"] == "available_outer_test_posthoc"
    assert status["n_blocks_passing_all_stability_dimensions"] == 1
    assert stability["panel"].tolist() == [IMPORTANCE_PANEL]
    assert stability.loc[0, "target_status"] == "pass"


@pytest.mark.parametrize(
    ("filename", "column", "invalid_value", "issue_fragment"),
    (
        (
            "outer_test_block_importance.csv",
            "delta_mae_mV_dec",
            999.0,
            "mean record-level delta_absolute_error_mV_dec",
        ),
        (
            "outer_test_block_effects.csv",
            "delta_absolute_error_mV_dec",
            999.0,
            "perturbed - baseline absolute error",
        ),
        (
            "outer_test_block_importance.csv",
            "baseline_mae_mV_dec",
            999.0,
            "perturbed_mae_mV_dec - baseline_mae_mV_dec",
        ),
        (
            "outer_test_block_effects.csv",
            "perturbed_absolute_error_mV_dec",
            999.0,
            "mean saved record-level absolute error",
        ),
        (
            "outer_test_block_effects.csv",
            "delta_absolute_error_mV_dec",
            np.inf,
            "nonfinite delta_absolute_error_mV_dec",
        ),
    ),
)
def test_interpretation_fails_closed_on_numeric_reconciliation_errors(
    tmp_path: Path,
    filename: str,
    column: str,
    invalid_value: float,
    issue_fragment: str,
) -> None:
    evaluation = tmp_path / "evaluation"
    records = _write_synthetic_interpretation_run(evaluation)
    path = evaluation / "analysis_0" / filename
    frame = pd.read_csv(path)
    frame.loc[0, column] = invalid_value
    frame.to_csv(path, index=False)

    config = InterpretationConfig(
        permutation_repeats=1,
        minimum_outer_splits_per_cell=2,
        minimum_records_per_stratum=2,
        minimum_strata=2,
    )
    status = build_interpretation_layer(evaluation, records, config=config)
    stability = pd.read_csv(evaluation / "interpretation" / "interpretation_stability.csv")

    audit = status["importance_artifact_audit"]
    assert status["importance_status"] == "fail_closed_invalid_outer_test_artifacts"
    assert audit["status"] == "fail_closed_invalid_outer_test_artifacts"
    assert any(issue_fragment in issue for issue in audit["issues"])
    assert status["scientific_interpretation_allowed"] is False
    assert stability.empty


def test_numeric_reconciliation_audits_nonregistered_panels(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    records = _write_synthetic_interpretation_run(evaluation)
    for filename in (
        "outer_test_block_importance.csv",
        "outer_test_block_effects.csv",
    ):
        path = evaluation / "analysis_0" / filename
        registered = pd.read_csv(path)
        auxiliary = registered.copy()
        auxiliary["panel"] = "context_plus_ink"
        pd.concat([registered, auxiliary], ignore_index=True).to_csv(path, index=False)

    effects_path = evaluation / "analysis_0" / "outer_test_block_effects.csv"
    effects = pd.read_csv(effects_path)
    auxiliary_row = effects["panel"].eq("context_plus_ink")
    effects.loc[auxiliary_row.idxmax(), "delta_absolute_error_mV_dec"] += 0.5
    effects.to_csv(effects_path, index=False)

    config = InterpretationConfig(
        permutation_repeats=1,
        minimum_outer_splits_per_cell=2,
        minimum_records_per_stratum=2,
        minimum_strata=2,
    )
    status = build_interpretation_layer(evaluation, records, config=config)

    assert status["importance_status"] == "fail_closed_invalid_outer_test_artifacts"
    assert any(
        "mean record-level delta_absolute_error_mV_dec" in issue
        for issue in status["importance_artifact_audit"]["issues"]
    )


@pytest.mark.parametrize(
    ("filename", "column", "invalid_value"),
    (
        ("outer_test_block_importance.csv", "outer_test_only", False),
        (
            "outer_test_block_importance.csv",
            "selection_role",
            "used_for_model_selection",
        ),
        ("outer_test_block_effects.csv", "outer_test_only", False),
        (
            "outer_test_block_effects.csv",
            "selection_role",
            "used_for_model_selection",
        ),
    ),
)
def test_interpretation_fails_closed_on_invalid_posthoc_provenance(
    tmp_path: Path, filename: str, column: str, invalid_value: object
) -> None:
    evaluation = tmp_path / "evaluation"
    records = _write_synthetic_interpretation_run(evaluation)
    path = evaluation / "analysis_0" / filename
    frame = pd.read_csv(path)
    frame.loc[0, column] = invalid_value
    frame.to_csv(path, index=False)

    config = InterpretationConfig(
        permutation_repeats=1,
        minimum_outer_splits_per_cell=2,
        minimum_records_per_stratum=2,
        minimum_strata=2,
    )
    status = build_interpretation_layer(evaluation, records, config=config)
    stability = pd.read_csv(evaluation / "interpretation" / "interpretation_stability.csv")

    assert status["importance_status"] == "fail_closed_invalid_outer_test_artifacts"
    assert status["importance_artifact_audit"]["n_issues"] >= 1
    assert status["scientific_interpretation_allowed"] is False
    assert status["n_blocks_passing_all_stability_dimensions"] == 0
    assert stability.empty


@pytest.mark.parametrize(
    "corruption",
    ("missing_importance_repeat", "missing_effect_grid", "duplicate_importance_grid"),
)
def test_interpretation_fails_closed_on_incomplete_perturbation_grid(
    tmp_path: Path, corruption: str
) -> None:
    evaluation = tmp_path / "evaluation"
    records = _write_synthetic_interpretation_run(evaluation)
    if corruption in {"missing_importance_repeat", "duplicate_importance_grid"}:
        path = evaluation / "analysis_0" / "outer_test_block_importance.csv"
        frame = pd.read_csv(path)
        remove = (
            frame["model"].eq(MODELS[0])
            & frame["fold"].eq(0)
            & frame["perturbation_method"].eq("joint_block_permutation")
        )
    else:
        path = evaluation / "analysis_0" / "outer_test_block_effects.csv"
        frame = pd.read_csv(path)
        remove = (
            frame["model"].eq(MODELS[0])
            & frame["fold"].eq(0)
            & frame["perturbation_method"].eq(
                "block_masking_to_fitted_missingness_policy"
            )
        )
    if corruption == "duplicate_importance_grid":
        duplicated = frame.loc[remove].iloc[[0]]
        pd.concat([frame, duplicated], ignore_index=True).to_csv(path, index=False)
    else:
        frame.loc[~remove].to_csv(path, index=False)

    config = InterpretationConfig(
        permutation_repeats=1,
        minimum_outer_splits_per_cell=2,
        minimum_records_per_stratum=2,
        minimum_strata=2,
    )
    status = build_interpretation_layer(evaluation, records, config=config)
    stability = pd.read_csv(evaluation / "interpretation" / "interpretation_stability.csv")

    assert status["importance_status"] == "fail_closed_invalid_outer_test_artifacts"
    assert status["importance_artifact_audit"]["n_issues"] >= 1
    assert status["scientific_interpretation_allowed"] is False
    assert stability.empty


def test_saved_unperturbed_predictions_and_smoke_have_explicit_non_estimable_gates(tmp_path: Path) -> None:
    records = pd.DataFrame(
        {
            "record_id": ["r1", "r2"],
            "lab_group_id": ["unknown", "unknown"],
            "lab_group_confidence": ["unknown", "unknown"],
            "catalyst_type": ["RuO2", "IrO2"],
            "electrolyte_identity": ["KOH", "KOH"],
        }
    )
    evaluation = tmp_path / "evaluation"
    analysis = evaluation / "primary"
    analysis.mkdir(parents=True)
    pd.DataFrame(
        {
            "record_id": ["r1", "r2"],
            "observed_mV_dec": [60.0, 80.0],
            "predicted_mV_dec": [65.0, 75.0],
        }
    ).to_csv(analysis / "oof_predictions.csv", index=False)

    status = build_interpretation_layer(evaluation, records)
    assert status["importance_status"] == "not_estimable_from_saved_artifacts"

    smoke_status = build_interpretation_layer(
        evaluation, records, output_dir=evaluation / "interpretation_smoke", smoke=True
    )
    assert smoke_status["importance_status"] == "not_evaluated_smoke_mode"
    assert smoke_status["scientific_interpretation_allowed"] is False


def test_smoke_figure_manifest_omits_performance_and_interpretation_figures(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "targets").mkdir()
    (tmp_path / "evaluation" / "primary").mkdir(parents=True)
    pd.DataFrame(
        {
            "record_id": ["r1", "r2"],
            "electrolyte_identity": ["KOH", "KOH"],
            "publication_year": [2020, 2021],
        }
    ).to_csv(tmp_path / "data" / "records.csv", index=False)
    pd.DataFrame(
        {
            "record_id": ["r1", "r2"],
            "target_name": [TARGETS[0], TARGETS[0]],
            "target_value_mV_dec": [60.0, 80.0],
        }
    ).to_csv(tmp_path / "targets" / "target_registry.csv", index=False)
    pd.DataFrame(
        {
            "target": [TARGETS[0]],
            "scope": ["alkaline"],
            "model": ["nested_selected"],
            "delta_mae_estimate": [-1.0],
            "delta_mae_ci_low": [-2.0],
            "delta_mae_ci_high": [1.0],
        }
    ).to_csv(
        tmp_path / "evaluation" / "primary" / "paired_delta_group_bootstrap.csv",
        index=False,
    )
    manifest = generate_figures(tmp_path, formats=("png",), dpi=72, smoke=True)
    performance = manifest.loc[
        manifest["figure"].isin(
            {
                "paired_delta_mae",
                "oof_observed_vs_predicted",
                "oof_residual_diagnostics",
                "outer_test_block_ablation_permutation",
            }
        )
    ]
    assert (performance["status"] == "not_generated").all()
    assert performance["reason"].str.contains("smoke", case=False).all()


def _write_figure_gate_inputs(root: Path, *, solvent_pass: bool) -> None:
    analysis = root / "evaluation" / "primary"
    interpretation = root / "evaluation" / "interpretation"
    analysis.mkdir(parents=True)
    interpretation.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for panel, block in (
        ("context_plus_ink", "solvent_composition"),
        ("context_plus_ink", "ink_amounts"),
        ("context_plus_property_proxy_core", "solvent_composition"),
    ):
        for method, perturbation_repeat, delta in (
            ("joint_block_permutation", 0, 2.0),
            ("block_masking_to_fitted_missingness_policy", -1, 1.0),
        ):
            rows.append(
                {
                    "target": TARGETS[0],
                    "scope": "alkaline",
                    "panel": panel,
                    "model": "bayesian_ridge",
                    "repeat": 0,
                    "fold": 0,
                    "block": block,
                    "perturbation_method": method,
                    "perturbation_repeat": perturbation_repeat,
                    "delta_mae_mV_dec": delta,
                }
            )
    pd.DataFrame(rows).to_csv(analysis / "outer_test_block_importance.csv", index=False)
    pd.DataFrame(
        {
            "block": ["solvent_composition", "ink_amounts"],
            "overall_stability_pass": [solvent_pass, False],
        }
    ).to_csv(interpretation / "interpretation_stability.csv", index=False)
    (interpretation / "interpretation_status.json").write_text(
        json.dumps({"config": {"importance_panel": "context_plus_ink"}}),
        encoding="utf-8",
    )


def test_formal_block_figure_is_gated_when_no_stability_block_passes(tmp_path: Path) -> None:
    _write_figure_gate_inputs(tmp_path, solvent_pass=False)

    manifest = generate_figures(tmp_path, formats=("png",), dpi=72, smoke=False)
    row = manifest.loc[
        manifest["figure"].eq("outer_test_block_ablation_permutation")
    ].iloc[0]

    assert row["status"] == "gated"
    assert row["path"] == ""
    assert "conditional associational sensitivity" in row["reason"]
    assert "not a feature ranking or causal effect" in row["reason"]
    assert not (tmp_path / "figures" / "outer_test_block_ablation_permutation.png").exists()


def test_formal_block_figure_contains_only_released_panel_block_and_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_figure_gate_inputs(tmp_path, solvent_pass=True)
    captured: dict[str, object] = {}

    def capture_figure(fig, output_base: Path, formats, dpi: int) -> list[Path]:
        captured["title"] = fig._suptitle.get_text()
        captured["labels"] = [label.get_text() for label in fig.axes[0].get_xticklabels()]
        output = output_base.with_suffix(".png")
        output.touch()
        return [output]

    monkeypatch.setattr(figures_module, "save_figure", capture_figure)
    manifest = figures_module.generate_figures(tmp_path, formats=("png",), dpi=72, smoke=False)
    row = manifest.loc[
        manifest["figure"].eq("outer_test_block_ablation_permutation")
    ].iloc[0]

    assert row["status"] == "generated"
    assert "conditional associational sensitivity" in row["reason"]
    assert "not a feature ranking or causal effect" in row["reason"]
    assert captured["labels"] == ["context plus ink\nsolvent composition"]
    title = str(captured["title"])
    assert "conditional associational" in title
    assert "not a feature ranking or causal effect" in title
