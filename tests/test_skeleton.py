import importlib

from cloth_segment.lib import paths


def test_modules_import():
    for mod in [
        "lib.data",
        "lib.models",
        "lib.runs",
        "lib.paths",
        "train",
        "evaluate",
        "attr_eval",
    ]:
        importlib.import_module(f"cloth_segment.{mod}")


def test_repo_root():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()


def test_raw_labels_are_contiguous():
    from cloth_segment.lib.data.sources import RAW_LABELS

    for name, labels in RAW_LABELS.items():
        assert list(labels) == list(range(len(labels))), name
    assert RAW_LABELS["lip"][10] != RAW_LABELS["cihp"][10]


def test_dfmm_outfit_id():
    from cloth_segment.lib.data.sources import dfmm_outfit_id

    assert dfmm_outfit_id("MEN-Denim-id_00000080-01_7_additional") == "id_00000080"
    assert dfmm_outfit_id("WOMEN-Graphic_Tees-id_00006246-01_7_additional") == "id_00006246"
