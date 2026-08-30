from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_human_facing_results_use_canonical_record_and_curve_ids() -> None:
    master = pd.read_csv(
        REPO_ROOT / "Data/02_inputs/canonical_dataset_ir_corrected_v2/master.csv",
        usecols=["record_id", "curve_id"],
    )
    mapping = dict(
        zip(master["curve_id"].astype(str), master["record_id"].astype(int))
    )
    assert len(mapping) == 97
    assert set(mapping.values()) == set(range(1, 98))
    assert "sr050" not in mapping

    result_tables = (
        REPO_ROOT / "Data/04_results/RESULTS/01_PPC_REVIEW/PPC_SELECTION.csv",
        REPO_ROOT / "Data/04_results/RESULTS/01_PPC_REVIEW/ppc_review.csv",
        REPO_ROOT / "Data/04_results/RESULTS/02_TAFEL_CURVE_QC/curve_manifest.csv",
        REPO_ROOT / "Data/04_results/RESULTS/02_TAFEL_CURVE_QC/curve_fit_summary.csv",
        REPO_ROOT / "Data/04_results/RESULTS/02_TAFEL_CURVE_QC/model_comparison.csv",
        REPO_ROOT / "Data/04_results/RESULTS/02_TAFEL_CURVE_QC/qc_review_queue.csv",
    )
    for path in result_tables:
        frame = pd.read_csv(path)
        expected_record = frame["curve_id"].astype(str).map(mapping)
        actual_record = pd.to_numeric(frame["sample_record"], errors="raise")
        assert expected_record.notna().all(), path
        assert actual_record.eq(expected_record).all(), path

    ppc_dir = REPO_ROOT / "Data/04_results/RESULTS/01_PPC_REVIEW"
    qc_dir = REPO_ROOT / "Data/04_results/RESULTS/02_TAFEL_CURVE_QC/images"
    expected_ppc = {
        f"record_{record_id:03d}__curve_{curve_id}__bayesian_local_line_regional_ppc.png"
        for curve_id, record_id in mapping.items()
    }
    expected_qc = {
        f"record_{record_id:03d}__curve_{curve_id}__qc.png"
        for curve_id, record_id in mapping.items()
    }
    assert {path.name for path in ppc_dir.glob("*.png")} == expected_ppc
    assert {path.name for path in qc_dir.glob("*.png")} == expected_qc
    assert not any(path.is_symlink() for path in ppc_dir.glob("*.png"))
    assert not any(path.is_symlink() for path in qc_dir.glob("*.png"))

    selection = pd.read_csv(result_tables[0])
    for row in selection.itertuples(index=False):
        filename = (
            f"record_{int(row.sample_record):03d}__curve_{row.curve_id}"
            "__bayesian_local_line_regional_ppc.png"
        )
        image_path = ppc_dir / filename
        assert row.ppc_image == filename
        assert image_path.is_file()
        review = pd.read_csv(result_tables[1]).set_index("curve_id")
        assert _sha256(image_path) == review.loc[row.curve_id, "ppc_sha256"]
