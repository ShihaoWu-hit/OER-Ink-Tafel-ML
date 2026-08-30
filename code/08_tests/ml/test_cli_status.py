import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import ink_tafel_study.cli as cli
from ink_tafel_study.config import load_config


class _ConfigStub:
    def value(self, key: str, default=None):
        return default


def test_parser_accepts_independent_cardinal_classification_stage() -> None:
    args = cli._parser().parse_args(
        [
            "cardinal-classify",
            "--run-dir",
            "Data/04_results/RESULTS/05_WORKBOOKS/example",
            "--source-run",
            "Data/03_outputs/study/outputs/current_97",
            "--n-jobs",
            "4",
        ]
    )

    assert args.stage == "cardinal-classify"
    assert args.n_jobs == 4


def test_parser_accepts_independent_posterior_cardinal_classification_stage() -> None:
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-classify",
            "--run-dir",
            "Data/04_results/RESULTS/05_WORKBOOKS/posterior_cardinal_all97_v2",
            "--source-run",
            "Data/03_outputs/study/outputs/current_97",
            "--n-jobs",
            "4",
        ]
    )

    assert args.stage == "posterior-cardinal-classify"
    assert args.n_jobs == 4
    assert args.source_run == Path("Data/03_outputs/study/outputs/current_97")


def test_parser_accepts_independent_posterior_threeclass_optuna_stage() -> None:
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-threeclass-optuna",
            "--run-dir",
            "Data/04_results/RESULTS/05_WORKBOOKS/posterior_threeclass_optuna",
            "--source-run",
            "Data/03_outputs/study/outputs/current_97",
            "--n-jobs",
            "4",
            "--optuna-trials",
            "17",
        ]
    )

    assert args.stage == "posterior-cardinal-threeclass-optuna"
    assert args.n_jobs == 4
    assert args.optuna_trials == 17
    assert args.source_run == Path("Data/03_outputs/study/outputs/current_97")


def test_parser_rejects_nonpositive_optuna_trials() -> None:
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "posterior-cardinal-threeclass-optuna",
                "--run-dir",
                "Data/04_results/RESULTS/05_WORKBOOKS/posterior_threeclass_optuna",
                "--optuna-trials",
                "0",
            ]
        )


def test_default_cardinal_source_uses_configured_active_revision() -> None:
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-classify",
            "--run-dir",
            "Data/04_results/RESULTS/05_WORKBOOKS/example",
        ]
    )
    source = cli._cardinal_source_run(load_config(), args)

    assert source.name == (
        "current_97_ir_compensation_corrected_v2_20260724"
    )


def test_parser_accepts_independent_cardinal_development_stage() -> None:
    args = cli._parser().parse_args(
        [
            "cardinal-develop",
            "--run-dir",
            "Data/04_results/RESULTS/05_WORKBOOKS/example_development",
            "--source-run",
            "Data/03_outputs/study/outputs/current_97",
            "--n-jobs",
            "4",
        ]
    )

    assert args.stage == "cardinal-develop"
    assert args.n_jobs == 4


def test_parser_accepts_independent_feature_association_stage() -> None:
    args = cli._parser().parse_args(
        [
            "feature-associate",
            "--run-dir",
            "Data/04_results/RESULTS/05_WORKBOOKS/example_association",
            "--source-run",
            "Data/03_outputs/study/outputs/current_97",
            "--n-jobs",
            "4",
        ]
    )

    assert args.stage == "feature-associate"
    assert args.n_jobs == 4
    assert args.source_run == Path("Data/03_outputs/study/outputs/current_97")


def test_feature_association_rejects_legacy_scope_filter_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli._parser().parse_args(
        [
            "feature-associate",
            "--run-dir",
            str(tmp_path / "association"),
            "--scope",
            "alkaline",
            "--smoke",
        ]
    )
    called = False

    def _unexpected_source(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("source resolution should not be reached")

    monkeypatch.setattr(cli, "_cardinal_source_run", _unexpected_source)
    with pytest.raises(ValueError, match="fixed to all 97 records"):
        cli._feature_associate(_ConfigStub(), tmp_path, args)
    assert called is False


def test_main_rejects_feature_association_scope_before_creating_run_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "association"
    with pytest.raises(ValueError, match="fixed to all 97 records"):
        cli.main(
            [
                "feature-associate",
                "--run-dir",
                str(run_dir),
                "--scope",
                "all",
                "--smoke",
            ]
        )
    assert not run_dir.exists()


def test_posterior_cardinal_classification_rejects_scope_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-classify",
            "--run-dir",
            str(tmp_path / "classification"),
            "--scope",
            "all",
            "--smoke",
        ]
    )
    called = False

    def _unexpected_source(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("source resolution should not be reached")

    monkeypatch.setattr(cli, "_cardinal_source_run", _unexpected_source)
    with pytest.raises(ValueError, match="fixed to all 97 records"):
        cli._posterior_cardinal_classify(_ConfigStub(), tmp_path, args)
    assert called is False


def test_main_rejects_posterior_cardinal_scope_before_creating_run_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "classification"
    with pytest.raises(ValueError, match="fixed to all 97 records"):
        cli.main(
            [
                "posterior-cardinal-classify",
                "--run-dir",
                str(run_dir),
                "--scope",
                "alkaline",
                "--smoke",
            ]
        )
    assert not run_dir.exists()


def test_posterior_threeclass_optuna_rejects_scope_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-threeclass-optuna",
            "--run-dir",
            str(tmp_path / "classification"),
            "--scope",
            "all",
            "--smoke",
        ]
    )
    called = False

    def _unexpected_source(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("source resolution should not be reached")

    monkeypatch.setattr(cli, "_cardinal_source_run", _unexpected_source)
    with pytest.raises(ValueError, match="fixed to all 97 records"):
        cli._posterior_cardinal_threeclass_optuna(
            _ConfigStub(), tmp_path, args
        )
    assert called is False


def test_main_rejects_posterior_threeclass_scope_before_creating_run_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "classification"
    with pytest.raises(ValueError, match="fixed to all 97 records"):
        cli.main(
            [
                "posterior-cardinal-threeclass-optuna",
                "--run-dir",
                str(run_dir),
                "--scope",
                "alkaline",
                "--smoke",
            ]
        )
    assert not run_dir.exists()


def test_cardinal_custom_source_overlap_fails_before_creating_run_dir(
    tmp_path: Path,
) -> None:
    source = tmp_path / "custom_formal_source"
    source.mkdir()
    run_dir = source / "classification_child"
    args = cli._parser().parse_args(
        [
            "cardinal-classify",
            "--source-run",
            str(source),
            "--run-dir",
            str(run_dir),
        ]
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        cli._resolve_run(cli.load_config(), args)
    assert not run_dir.exists()


def test_full_cardinal_run_must_use_results_workbook_root(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "full_cardinal"
    args = cli._parser().parse_args(
        ["cardinal-classify", "--run-dir", str(run_dir)]
    )
    with pytest.raises(ValueError, match="Data/04_results/RESULTS/05_WORKBOOKS"):
        cli._resolve_run(cli.load_config(), args)
    assert not run_dir.exists()


def test_posterior_cardinal_custom_source_overlap_fails_before_mkdir(
    tmp_path: Path,
) -> None:
    source = tmp_path / "custom_formal_source"
    source.mkdir()
    run_dir = source / "posterior_classification_child"
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-classify",
            "--source-run",
            str(source),
            "--run-dir",
            str(run_dir),
            "--smoke",
        ]
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        cli._resolve_run(cli.load_config(), args)
    assert not run_dir.exists()


def test_full_posterior_cardinal_run_must_use_results_workbook_root(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "full_posterior_cardinal"
    args = cli._parser().parse_args(
        ["posterior-cardinal-classify", "--run-dir", str(run_dir)]
    )
    with pytest.raises(ValueError, match="Data/04_results/RESULTS/05_WORKBOOKS"):
        cli._resolve_run(cli.load_config(), args)
    assert not run_dir.exists()


def test_threeclass_output_cannot_be_created_inside_frozen_v4() -> None:
    config = load_config()
    v4_root = (
        config.repository_root
        / "RESULTS"
        / "05_WORKBOOKS"
        / "posterior_cardinal_all97_v4_ir_compensation_corrected_20260724"
    )
    run_dir = v4_root / "__forbidden_threeclass_cli_child__"
    assert not run_dir.exists()
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-threeclass-optuna",
            "--run-dir",
            str(run_dir),
            "--smoke",
        ]
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        cli._resolve_run(config, args)

    assert not run_dir.exists()


def test_posterior_cardinal_run_dir_must_be_new_or_empty(
    tmp_path: Path,
) -> None:
    source = tmp_path / "formal_source"
    source.mkdir()
    run_dir = tmp_path / "posterior_classification"
    run_dir.mkdir()
    (run_dir / "stale.txt").write_text("stale", encoding="utf-8")
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-classify",
            "--source-run",
            str(source),
            "--run-dir",
            str(run_dir),
            "--smoke",
        ]
    )

    with pytest.raises(FileExistsError, match="new or empty"):
        cli._resolve_run(cli.load_config(), args)


def test_posterior_threeclass_optuna_requires_explicit_run_dir() -> None:
    args = cli._parser().parse_args(
        ["posterior-cardinal-threeclass-optuna", "--smoke"]
    )

    with pytest.raises(ValueError, match="requires --run-dir"):
        cli._resolve_run(cli.load_config(), args)


def test_posterior_threeclass_source_overlap_fails_before_mkdir(
    tmp_path: Path,
) -> None:
    source = tmp_path / "custom_formal_source"
    source.mkdir()
    run_dir = source / "posterior_threeclass_child"
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-threeclass-optuna",
            "--source-run",
            str(source),
            "--run-dir",
            str(run_dir),
            "--smoke",
        ]
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        cli._resolve_run(cli.load_config(), args)
    assert not run_dir.exists()


def test_full_posterior_threeclass_run_must_use_results_workbook_root(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "full_posterior_threeclass"
    args = cli._parser().parse_args(
        ["posterior-cardinal-threeclass-optuna", "--run-dir", str(run_dir)]
    )

    with pytest.raises(ValueError, match="Data/04_results/RESULTS/05_WORKBOOKS"):
        cli._resolve_run(cli.load_config(), args)
    assert not run_dir.exists()


def test_posterior_threeclass_run_dir_must_be_new_or_empty(
    tmp_path: Path,
) -> None:
    source = tmp_path / "formal_source"
    source.mkdir()
    run_dir = tmp_path / "posterior_threeclass"
    run_dir.mkdir()
    (run_dir / "stale.txt").write_text("stale", encoding="utf-8")
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-threeclass-optuna",
            "--source-run",
            str(source),
            "--run-dir",
            str(run_dir),
            "--smoke",
        ]
    )

    with pytest.raises(FileExistsError, match="new or empty"):
        cli._resolve_run(cli.load_config(), args)


def test_full_feature_association_run_must_use_results_workbook_root(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "full_feature_association"
    args = cli._parser().parse_args(
        ["feature-associate", "--run-dir", str(run_dir)]
    )
    with pytest.raises(ValueError, match="Data/04_results/RESULTS/05_WORKBOOKS"):
        cli._resolve_run(cli.load_config(), args)
    assert not run_dir.exists()


def test_feature_association_source_and_output_must_be_disjoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "formal_source"
    source.mkdir()
    run_dir = source / "association_child"
    args = cli._parser().parse_args(
        [
            "feature-associate",
            "--source-run",
            str(source),
            "--run-dir",
            str(run_dir),
            "--smoke",
        ]
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        cli._resolve_run(cli.load_config(), args)
    assert not run_dir.exists()


def test_feature_association_start_invalidates_only_its_completion_claim(
    tmp_path: Path,
) -> None:
    feature_status = (
        tmp_path / "provenance" / "stage_feature_associate.json"
    )
    cardinal_status = (
        tmp_path / "provenance" / "stage_cardinal_classify.json"
    )
    _write_json(feature_status, {"status": "complete"})
    _write_json(cardinal_status, {"status": "complete"})

    cli._invalidate_completion_claims(tmp_path, "feature-associate")

    assert not feature_status.exists()
    assert cardinal_status.exists()


def test_posterior_cardinal_start_invalidates_only_its_completion_claim(
    tmp_path: Path,
) -> None:
    posterior_status = (
        tmp_path
        / "provenance"
        / "stage_posterior_cardinal_classify.json"
    )
    historical_status = (
        tmp_path / "provenance" / "stage_cardinal_classify.json"
    )
    _write_json(posterior_status, {"status": "complete"})
    _write_json(historical_status, {"status": "complete"})

    cli._invalidate_completion_claims(
        tmp_path, "posterior-cardinal-classify"
    )

    assert not posterior_status.exists()
    assert historical_status.exists()


def test_posterior_threeclass_start_invalidates_only_its_completion_claim(
    tmp_path: Path,
) -> None:
    threeclass_status = (
        tmp_path
        / "provenance"
        / "stage_posterior_cardinal_threeclass_optuna.json"
    )
    fourclass_status = (
        tmp_path
        / "provenance"
        / "stage_posterior_cardinal_classify.json"
    )
    _write_json(threeclass_status, {"status": "complete"})
    _write_json(fourclass_status, {"status": "complete"})

    cli._invalidate_completion_claims(
        tmp_path, "posterior-cardinal-threeclass-optuna"
    )

    assert not threeclass_status.exists()
    assert fourclass_status.exists()


def test_posterior_cardinal_helper_writes_its_own_stage_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "formal_source"
    source.mkdir()
    run_dir = tmp_path / "classification"
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-classify",
            "--source-run",
            str(source),
            "--run-dir",
            str(run_dir),
            "--smoke",
            "--n-jobs",
            "2",
        ]
    )
    observed: dict[str, object] = {}

    def _run(config, destination, **kwargs):
        observed.update(
            {
                "config": config,
                "destination": destination,
                **kwargs,
            }
        )
        return {"status": "complete_smoke"}

    monkeypatch.setattr(
        cli, "run_posterior_cardinal_classification", _run
    )
    cli._posterior_cardinal_classify(_ConfigStub(), run_dir, args)

    status = json.loads(
        (
            run_dir
            / "provenance"
            / "stage_posterior_cardinal_classify.json"
        ).read_text(encoding="utf-8")
    )
    assert status["stage"] == "posterior_cardinal_classify"
    assert status["status"] == "complete_smoke"
    assert status["smoke"] is True
    assert status["n_jobs"] == 2
    assert observed["source_run"] == source.resolve()
    assert observed["smoke"] is True
    assert observed["n_jobs"] == 2


def test_posterior_threeclass_helper_writes_its_own_stage_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "formal_source"
    source.mkdir()
    run_dir = tmp_path / "classification"
    args = cli._parser().parse_args(
        [
            "posterior-cardinal-threeclass-optuna",
            "--source-run",
            str(source),
            "--run-dir",
            str(run_dir),
            "--smoke",
            "--n-jobs",
            "2",
            "--optuna-trials",
            "7",
        ]
    )
    observed: dict[str, object] = {}

    def _run(config, destination, **kwargs):
        observed.update(
            {
                "config": config,
                "destination": destination,
                **kwargs,
            }
        )
        return {"status": "complete_smoke"}

    monkeypatch.setattr(
        cli, "run_posterior_cardinal_threeclass_optuna", _run
    )
    cli._posterior_cardinal_threeclass_optuna(
        _ConfigStub(), run_dir, args
    )

    status = json.loads(
        (
            run_dir
            / "provenance"
            / "stage_posterior_cardinal_threeclass_optuna.json"
        ).read_text(encoding="utf-8")
    )
    assert status["stage"] == "posterior_cardinal_threeclass_optuna"
    assert status["status"] == "complete_smoke"
    assert status["smoke"] is True
    assert status["n_jobs"] == 2
    assert status["optuna_trials"] == 7
    assert observed["source_run"] == source.resolve()
    assert observed["smoke"] is True
    assert observed["n_jobs"] == 2
    assert observed["optuna_trials"] == 7


def test_main_posterior_threeclass_writes_artifact_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "classification"
    run_dir.mkdir()

    monkeypatch.setattr(cli, "load_config", lambda path=None: _ConfigStub())
    monkeypatch.setattr(
        cli, "_resolve_run", lambda config, args: run_dir
    )

    def _initialize(
        config,
        destination,
        *,
        smoke,
        scientific_conclusions_allowed,
        analysis_role,
    ):
        assert scientific_conclusions_allowed is False
        assert analysis_role == "posthoc_exploratory_sensitivity"
        (destination / "provenance").mkdir(exist_ok=True)

    def _run(config, destination, args):
        cli._stage_status(
            destination,
            "posterior_cardinal_threeclass_optuna",
            "complete_smoke",
            smoke=True,
        )

    monkeypatch.setattr(cli, "_initialize_provenance", _initialize)
    monkeypatch.setattr(
        cli, "_posterior_cardinal_threeclass_optuna", _run
    )

    assert (
        cli.main(
            [
                "posterior-cardinal-threeclass-optuna",
                "--run-dir",
                str(run_dir),
                "--smoke",
            ]
        )
        == 0
    )

    manifest = json.loads(
        (run_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert {
        row["path"] for row in manifest["files"]
    } == {
        "provenance/stage_posterior_cardinal_threeclass_optuna.json"
    }


def test_cardinal_overlap_rejects_case_alias_before_mkdir(
    tmp_path: Path,
) -> None:
    source = tmp_path / "FormalSource"
    source.mkdir()
    case_alias = tmp_path / "formalsource"
    if not case_alias.exists():
        pytest.skip("The test filesystem is case-sensitive")
    run_dir = case_alias / "classification_child"
    args = cli._parser().parse_args(
        [
            "cardinal-classify",
            "--source-run",
            str(source),
            "--run-dir",
            str(run_dir),
            "--smoke",
        ]
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        cli._resolve_run(cli.load_config(), args)
    assert not (source / "classification_child").exists()


class _TargetConfigStub:
    def __init__(self, *, omit: str | None = None):
        self._sections = {
            "curation": {"expected_records": 97, "expected_curves": 98},
            "targets": {
                "fixed_current_mA_cm2": 2.0,
                "minimum_draw_support_fraction": 0.8,
                "inverse_grid_points": 2049,
                "expected_primary_eligible": 74,
                "expected_low_branch_eligible": 79,
                "expected_single_regime_eligible": 58,
            },
        }
        if omit is not None:
            section, key = omit.split(".", 1)
            self._sections[section].pop(key)

    def section(self, name: str):
        return self._sections[name]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_target_build_config_is_fully_wired_from_registered_yaml_fields() -> None:
    result = cli._target_build_config(_TargetConfigStub())

    assert result.expected_curve_count == 98
    assert result.expected_record_count == 97
    assert result.expected_primary_eligible == 74
    assert result.expected_low_branch_eligible == 79
    assert result.expected_single_regime_eligible == 58
    assert result.fixed_current_density_mA_cm2 == 2.0
    assert result.minimum_draw_support_fraction == 0.8
    assert result.inverse_grid_points == 2049


@pytest.mark.parametrize(
    "missing",
    [
        "curation.expected_curves",
        "curation.expected_records",
        "targets.expected_primary_eligible",
        "targets.expected_low_branch_eligible",
        "targets.expected_single_regime_eligible",
    ],
)
def test_target_build_config_fails_closed_when_registered_count_is_missing(
    missing: str,
) -> None:
    with pytest.raises(KeyError, match=missing):
        cli._target_build_config(_TargetConfigStub(omit=missing))


@pytest.mark.parametrize(
    "reported_status",
    ["excluded", "not_registered", "eligible", "future_unrecognized_status"],
)
def test_estimated_property_panel_gate_is_fail_closed(
    reported_status: str,
) -> None:
    settings = cli.EvaluationConfig(
        panel_names=(
            "context_plus_ink_amounts",
            "context_plus_property_proxy_core",
            "context_plus_property_proxy_4p_source_backed",
            "context_plus_property_proxy_4p_estimated",
        )
    )
    gated, audit = cli._apply_property_panel_gate(
        settings,
        {
            "estimated_panel_model_use_status": reported_status,
            "source_backed_4p_panel_model_use_status": "eligible_complete_case_only",
        },
    )

    assert "context_plus_property_proxy_4p_estimated" not in gated.panel_names
    assert (
        audit["decision"]["context_plus_property_proxy_4p_estimated"]
        == "excluded"
    )
    assert audit["policy"] == "fail_closed_explicit_status_whitelist"


def test_estimated_property_panel_gate_accepts_only_registered_whitelist_status() -> None:
    settings = cli.EvaluationConfig(
        panel_names=(
            "context_plus_ink_amounts",
            "context_plus_property_proxy_4p_estimated",
        )
    )
    gated, audit = cli._apply_property_panel_gate(
        settings,
        {"estimated_panel_model_use_status": "eligible_sensitivity_only"},
    )

    assert "context_plus_property_proxy_4p_estimated" in gated.panel_names
    assert (
        audit["decision"]["context_plus_property_proxy_4p_estimated"]
        == "included"
    )


def test_registered_evaluation_requires_formal_nuts_and_named_ppc(tmp_path: Path) -> None:
    qa_path = tmp_path / "targets" / "target_qa.json"
    _write_json(
        qa_path,
        {"n_fit_eligible_curves": 94, "n_formal_nuts_ppc_approved": 0},
    )

    with pytest.raises(RuntimeError, match=r"0/94 approved"):
        cli._require_formal_bayesian_approval(tmp_path, smoke=False)
    cli._require_formal_bayesian_approval(tmp_path, smoke=True)

    _write_json(
        qa_path,
        {
            "n_fit_eligible_curves": 94,
            "n_formal_nuts_ppc_approved": 94,
            "formal_evaluation_ready": True,
            "primary_target_policy": {
                    "target_name": "b_bayesian_local_line_mV_dec",
                    "common_estimand": "robust Bayesian local-linear apparent slope",
            },
        },
    )
    cli._require_formal_bayesian_approval(tmp_path, smoke=False)

    _write_json(
        qa_path,
        {
            "n_fit_eligible_curves": 94,
            "n_formal_nuts_ppc_approved": 94,
            "formal_evaluation_ready": True,
            "primary_target_policy": {
                "target_name": "b_refit_at_2mA_mV_dec",
                "exact_2mA_role": "primary",
            },
        },
    )
    with pytest.raises(RuntimeError, match="predates the calibrated"):
        cli._require_formal_bayesian_approval(tmp_path, smoke=False)

    _write_json(
        qa_path,
        {
            "n_fit_eligible_curves": 94,
            "n_formal_nuts_ppc_approved": 94,
            "formal_evaluation_ready": False,
            "primary_target_policy": {
                "target_name": "b_bayesian_local_line_mV_dec",
                "common_estimand": "robust Bayesian local-linear apparent slope",
            },
            "formal_evaluation_blockers": [
                "posterior_branch_slope_extraction_required"
            ],
        },
    )
    with pytest.raises(RuntimeError, match="approved posterior derivation"):
        cli._require_formal_bayesian_approval(tmp_path, smoke=False)


def test_report_preserves_smoke_semantics_from_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_json(
        tmp_path / "provenance" / "stage_evaluate.json",
        {"stage": "evaluate", "status": "complete_smoke"},
    )
    observed: dict[str, bool] = {}

    def fake_report(run_dir: Path, *, smoke: bool):
        observed["smoke"] = smoke
        report = run_dir / "report" / "study_report.md"
        validation = run_dir / "report" / "validation_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("smoke\n", encoding="utf-8")
        validation.write_text("smoke\n", encoding="utf-8")
        return report, validation

    monkeypatch.setattr(cli, "build_report", fake_report)
    monkeypatch.setattr(
        cli,
        "generate_figures",
        lambda *args, **kwargs: pd.DataFrame({"status": ["generated"]}),
    )
    cli._report(_ConfigStub(), tmp_path, SimpleNamespace(smoke=False))

    status = json.loads(
        (tmp_path / "provenance" / "stage_report.json").read_text(encoding="utf-8")
    )
    assert observed["smoke"] is True
    assert status["status"] == "complete_smoke"
    assert status["smoke"] is True


def test_all_records_actual_failing_substage_and_removes_stale_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "provenance").mkdir()
    (tmp_path / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")
    _write_json(
        tmp_path / "provenance" / "stage_report.json",
        {"stage": "report", "status": "complete"},
    )

    monkeypatch.setattr(cli, "load_config", lambda path=None: _ConfigStub())
    monkeypatch.setattr(cli, "_resolve_run", lambda config, args: tmp_path)
    monkeypatch.setattr(
        cli, "_initialize_provenance", lambda config, run_dir, smoke: None
    )
    monkeypatch.setattr(cli, "_curate", lambda config, run_dir, args: None)

    def fail_refit(config, run_dir, args):
        raise RuntimeError("Bayesian batch failed")

    monkeypatch.setattr(cli, "_refit", fail_refit)

    with pytest.raises(RuntimeError, match="Bayesian batch failed"):
        cli.main(["all", "--smoke"])

    refit = json.loads(
        (tmp_path / "provenance" / "stage_refit.json").read_text(encoding="utf-8")
    )
    chain = json.loads(
        (tmp_path / "provenance" / "stage_all.json").read_text(encoding="utf-8")
    )
    assert refit["status"] == "failed"
    assert chain["status"] == "failed"
    assert chain["failed_stage"] == "refit"
    assert not (tmp_path / "artifact_manifest.json").exists()
    assert not (tmp_path / "provenance" / "stage_report.json").exists()
