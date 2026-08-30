from pathlib import Path

import pandas as pd

from ink_tafel_study.cli import _refresh_curation_manifest


def test_parquet_manifest_preserves_row_count(tmp_path: Path) -> None:
    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(tmp_path / "curve_points.parquet", index=False)
    _refresh_curation_manifest(tmp_path)
    manifest = pd.read_csv(tmp_path / "curation_manifest.csv")
    assert manifest.loc[manifest.artifact.eq("curve_points"), "rows"].item() == 3

