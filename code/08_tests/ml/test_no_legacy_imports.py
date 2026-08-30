from pathlib import Path


def test_new_package_does_not_import_legacy_knn_monolith() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "ink_tafel_study"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    forbidden = ("knn_tafel_core", "knn_tafel_train", "knn_tafel_visualize")
    assert not any(name in source for name in forbidden)

