import importlib

from cloth_segment import paths


def test_modules_import():
    for mod in ["data", "models", "runs", "train", "evaluate", "paths"]:
        importlib.import_module(f"cloth_segment.{mod}")


def test_repo_root():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
