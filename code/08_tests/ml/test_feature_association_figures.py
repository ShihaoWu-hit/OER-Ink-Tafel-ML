from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import ink_tafel_study.feature_association_figures as association_figures
from ink_tafel_study.feature_association_figures import (
    generate_feature_association_figures,
)
from ink_tafel_study.plot_style import apply_publication_style, save_figure


def _write_atlas(root: Path, *, n_features: int = 1) -> None:
    atlas_dir = root / "atlas"
    atlas_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for feature_index in range(n_features):
        feature = f"numeric_feature_{feature_index:02d}"
        summaries.append(
            {
                "feature_name": feature,
                "kind": "numeric",
                "block": "ink_amounts",
                "n_records": 2,
                "n_groups": 2,
                "x_label": f"Feature {feature_index}",
                "x_scale": "linear",
            }
        )
        for record_index, (family, median) in enumerate(
            (("RuO2", 60.0), ("IrO2_plus_Ir", 120.0))
        ):
            rows.append(
                {
                    "record_id": f"r{record_index}",
                    "cv_group_id": f"g{record_index}",
                    "electrolyte_scope": "all97",
                    "catalyst_family": family,
                    "feature_name": feature,
                    "kind": "numeric",
                    "feature_value_numeric": float(record_index + feature_index),
                    "feature_value_category": "",
                    "target_median_mV_dec": median,
                    "target_q025_mV_dec": median * 0.8,
                    "target_q975_mV_dec": median * 1.2,
                }
            )
    pd.DataFrame(rows).to_csv(atlas_dir / "atlas_observations.csv", index=False)
    pd.DataFrame(summaries).to_csv(
        atlas_dir / "atlas_feature_summary.csv", index=False
    )


def _write_failed_release(root: Path) -> None:
    effects = root / "effects"
    effects.mkdir(parents=True)
    pd.DataFrame(
        {
            "block_id": ["catalyst_concentration"],
            "augmented_panel": ["context_plus_catalyst_concentration"],
            "primary_model": ["gam_low_df"],
            "released_model": ["gam_low_df"],
            "block_gate_passed": [False],
            "effect_curve_status": ["gated"],
            "reason": ["paired block comparison did not pass"],
        }
    ).to_csv(effects / "effect_release_registry.csv", index=False)
    pd.DataFrame(
        {
            "block_id": [
                "catalyst_concentration",
                "catalyst_concentration",
            ],
            "panel": [
                "context_plus_catalyst_concentration",
                "context_plus_catalyst_concentration",
            ],
            "model": ["gam_low_df", "gam_low_df"],
            "feature_name": ["ionomer_ratio", "ionomer_ratio"],
            "grid_value_numeric": [0.0, 1.0],
            "predicted_log10_centered": [0.0, 0.1],
            "ci95_low": [0.9, 1.0],
            "ci95_high": [1.1, 1.4],
        }
    ).to_csv(effects / "adjusted_effect_grid.csv", index=False)


def test_atlas_uses_log_target_axis_cardinal_guides_and_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_atlas(tmp_path)
    captured: dict[str, object] = {}

    def capture(
        fig: plt.Figure,
        output_base: Path,
        *,
        formats=("png", "pdf"),
        dpi=300,
    ) -> list[Path]:
        if output_base.parent.name == "atlas":
            axis = fig.axes[0]
            captured["yscale"] = axis.get_yscale()
            captured["horizontal_lines"] = {
                float(line.get_ydata()[0])
                for line in axis.lines
                if len(line.get_ydata())
                and all(value == line.get_ydata()[0] for value in line.get_ydata())
            }
            captured["figure_text"] = [text.get_text() for text in fig.texts]
        output = output_base.with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        plt.close(fig)
        return [output]

    monkeypatch.setattr(association_figures, "save_figure", capture)
    manifest = generate_feature_association_figures(
        tmp_path, formats=("png",), dpi=72
    )

    assert captured["yscale"] == "log"
    assert {30.0, 40.0, 60.0, 120.0}.issubset(
        captured["horizontal_lines"]
    )
    assert any(
        "not adjusted" in text and "not" in text
        for text in captured["figure_text"]
    )
    atlas_rows = manifest.loc[manifest["figure_family"].eq("atlas")]
    assert len(atlas_rows) == 1
    assert atlas_rows.iloc[0]["status"] == "generated"


def test_atlas_rejects_posterior_draw_pseudoreplication(tmp_path: Path) -> None:
    _write_atlas(tmp_path)
    path = tmp_path / "atlas" / "atlas_observations.csv"
    atlas = pd.read_csv(path)
    atlas = pd.concat([atlas, atlas.iloc[[0]]], ignore_index=True)
    atlas.to_csv(path, index=False)

    with pytest.raises(ValueError, match="one row per scope-record-feature"):
        generate_feature_association_figures(
            tmp_path, formats=("png",), dpi=72
        )


def test_registered_insufficient_support_feature_is_not_plotted(
    tmp_path: Path,
) -> None:
    _write_atlas(tmp_path)
    summary_path = tmp_path / "atlas" / "atlas_feature_summary.csv"
    summary = pd.read_csv(summary_path)
    summary["scope"] = "all97"
    summary["atlas_status"] = "skipped_insufficient_support"
    summary.to_csv(summary_path, index=False)

    manifest = generate_feature_association_figures(
        tmp_path, formats=("png",), dpi=72
    )

    atlas = manifest.loc[manifest["figure_family"].eq("atlas")]
    assert len(atlas) == 1
    assert atlas.iloc[0]["status"] == "gated"
    assert atlas.iloc[0]["gate_status"] == "insufficient_support"
    assert not list((tmp_path / "figures" / "atlas").glob("*.png"))


def test_numeric_atlas_uses_six_panels_per_page(tmp_path: Path) -> None:
    _write_atlas(tmp_path, n_features=7)

    manifest = generate_feature_association_figures(
        tmp_path, formats=("png",), dpi=40
    )

    atlas = manifest.loc[
        manifest["figure_family"].eq("atlas")
        & manifest["status"].eq("generated")
    ]
    assert set(atlas["figure_id"]) == {
        "all97__numeric__p01",
        "all97__numeric__p02",
    }


def test_support_and_redundancy_figures_never_mix_scopes(
    tmp_path: Path,
) -> None:
    audits = tmp_path / "audits"
    audits.mkdir()
    pd.DataFrame(
        {
            "scope": ["alkaline", "alkaline", "all97", "all97"],
            "feature_name": ["x", "z", "x", "z"],
            "n_records": [90, 80, 97, 87],
            "n_connected_groups": [76, 70, 80, 74],
        }
    ).to_csv(audits / "feature_support_audit.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["alkaline", "all97"],
            "feature_a": ["x", "x"],
            "feature_b": ["z", "z"],
            "spearman_rho": [0.91, 0.86],
        }
    ).to_csv(audits / "numeric_redundancy_pairs.csv", index=False)

    manifest = generate_feature_association_figures(
        tmp_path, formats=("png",), dpi=40
    )

    support = manifest.loc[
        manifest["figure_family"].eq("support")
        & manifest["status"].eq("generated")
    ]
    assert set(support["scope"]) == {"alkaline", "all97"}
    assert set(support["figure_id"]) == {
        "alkaline__feature_support_records_groups",
        "all97__feature_support_records_groups",
        "alkaline__numeric_redundancy_heatmap",
        "all97__numeric_redundancy_heatmap",
    }


def test_failed_effect_gate_never_creates_placeholder_curve(
    tmp_path: Path,
) -> None:
    _write_failed_release(tmp_path)

    manifest = generate_feature_association_figures(
        tmp_path, formats=("png",), dpi=72
    )

    effect = manifest.loc[manifest["figure_family"].eq("adjusted_effect")]
    assert len(effect) == 1
    assert effect.iloc[0]["status"] == "gated"
    assert effect.iloc[0]["gate_status"] == "failed"
    assert effect.iloc[0]["path"] == ""
    assert not list((tmp_path / "figures" / "effects").glob("*.png"))


def test_smoke_never_releases_a_passing_effect_block(tmp_path: Path) -> None:
    _write_failed_release(tmp_path)
    release_path = tmp_path / "effects" / "effect_release_registry.csv"
    release = pd.read_csv(release_path)
    release["block_gate_passed"] = True
    release["effect_curve_status"] = "released"
    release.to_csv(release_path, index=False)

    manifest = generate_feature_association_figures(
        tmp_path, formats=("png",), dpi=72, smoke=True
    )

    effect = manifest.loc[manifest["figure_family"].eq("adjusted_effect")]
    assert effect.iloc[0]["status"] == "gated"
    assert effect.iloc[0]["gate_status"] == "smoke_not_scientific"
    block = manifest.loc[manifest["figure_id"].eq("blockwise_primary_delta")]
    assert block.iloc[0]["status"] == "gated"
    assert not list((tmp_path / "figures" / "effects").glob("*.png"))


def test_passing_effect_gate_generates_only_the_registered_curve(
    tmp_path: Path,
) -> None:
    _write_failed_release(tmp_path)
    release_path = tmp_path / "effects" / "effect_release_registry.csv"
    release = pd.read_csv(release_path)
    release["block_gate_passed"] = True
    release["effect_curve_status"] = "released"
    release.to_csv(release_path, index=False)

    manifest = generate_feature_association_figures(
        tmp_path, formats=("png",), dpi=72
    )

    effect = manifest.loc[
        manifest["figure_family"].eq("adjusted_effect")
        & manifest["status"].eq("generated")
    ]
    assert len(effect) == 1
    assert effect.iloc[0]["feature"] == "ionomer_ratio"
    assert effect.iloc[0]["gate_status"] == "passed"
    assert (tmp_path / effect.iloc[0]["path"]).is_file()
    assert len(list((tmp_path / "figures" / "effects").glob("*.png"))) == 1


def test_vector_exports_are_byte_deterministic(tmp_path: Path) -> None:
    apply_publication_style(10)
    for stem in ("first", "second"):
        fig, axis = plt.subplots(figsize=(4.0, 3.0))
        axis.plot([0.0, 1.0], [1.0, 0.0])
        axis.set_title("deterministic")
        save_figure(
            fig,
            tmp_path / stem,
            formats=("pdf", "svg"),
            dpi=72,
        )

    for extension in ("pdf", "svg"):
        first = (tmp_path / f"first.{extension}").read_bytes()
        second = (tmp_path / f"second.{extension}").read_bytes()
        assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
