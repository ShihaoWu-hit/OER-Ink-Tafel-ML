import pandas as pd

from ink_tafel_study.evaluate import build_connected_group_ids


def test_connected_recipe_promotes_pre_cutoff_record_to_temporal_test() -> None:
    frame = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "publication_id": ["p1", "p2", "p3"],
            "lab_group_id": ["unknown", "unknown", "unknown"],
            "recipe_hash": ["shared", "shared", "other"],
            "sample_id": ["s1", "s2", "s3"],
            "publication_year": [2023, 2024, 2023],
        }
    )
    frame["cv_group_id"] = build_connected_group_ids(frame)
    temporal_groups = set(frame.loc[frame.publication_year >= 2024, "cv_group_id"])
    assert frame.loc[frame.record_id.eq(1), "cv_group_id"].item() in temporal_groups

