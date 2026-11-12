import datetime as dt
import json
import os
import stat

import pytest
import yaml

from cloth_segment.lib import runs
from cloth_segment.lib.runs import (
    AlreadyFinished,
    GpuBusy,
    config_hash,
    create_run,
    deep_merge,
    find_finished,
    gpu_lock,
    load_config,
    mark_finished,
    read_meta,
    run_name,
    slugify,
)

BASE = {
    "seed": 0,
    "data": {"version": "splits-v2-abc", "datasets": ["lip", "cihp", "dfmm"]},
    "input": {"size": 512},
    "augment": {"hflip": 0.5},
    "model": {"name": None},
    "train": {"batch_size": 8, "iters": 40000},
}
NOW = dt.datetime(2026, 11, 5, 9, 30)
GIT = ("abc1234", False)


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return p


@pytest.fixture
def base_path(tmp_path):
    return _write(tmp_path, "base.yaml", BASE)


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": 2}, "l": [1]}
    out = deep_merge(a, {"x": {"y": 9}, "l": [2]})
    assert out == {"x": {"y": 9, "z": 2}, "l": [2]}
    assert a == {"x": {"y": 1, "z": 2}, "l": [1]}


def test_load_config_merges_model_fields(tmp_path, base_path):
    exp = _write(tmp_path, "unet.yaml", {"model": {"name": "unet_r34"}, "train": {"lr": 1e-3}})
    cfg = load_config(exp, base_path)
    assert cfg["model"]["name"] == "unet_r34"
    assert cfg["train"] == {"batch_size": 8, "iters": 40000, "lr": 1e-3}


def test_load_config_base_itself(base_path):
    assert load_config(base_path, base_path) == BASE


@pytest.mark.parametrize(
    "over",
    [{"input": {"size": 384}}, {"train": {"batch_size": 16}}, {"data": {"datasets": ["lip"]}}],
)
def test_undeclared_shared_override_raises(tmp_path, base_path, over):
    exp = _write(tmp_path, "bad.yaml", over)
    with pytest.raises(ValueError, match="shared comparison fields"):
        load_config(exp, base_path)


def test_declared_shared_override_allowed(tmp_path, base_path):
    over = {"data": {"datasets": ["lip"]}, "run": {"overrides_shared": ["data"]}}
    cfg = load_config(_write(tmp_path, "e2.yaml", over), base_path)
    assert cfg["data"]["datasets"] == ["lip"]


def test_declared_but_unchanged_raises(tmp_path, base_path):
    over = {"run": {"overrides_shared": ["input"]}}
    with pytest.raises(ValueError, match="unchanged"):
        load_config(_write(tmp_path, "x.yaml", over), base_path)


def test_config_hash_ignores_order_and_run_section():
    a = {"b": 1, "a": {"y": 2, "x": 1}}
    b = {"a": {"x": 1, "y": 2}, "b": 1, "run": {"slug": "whatever"}}
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 12


def test_config_hash_changes_with_values():
    assert config_hash(BASE) != config_hash(deep_merge(BASE, {"train": {"iters": 40001}}))
    assert config_hash(BASE) != config_hash(deep_merge(BASE, {"data": {"version": "v3"}}))


def test_run_name_and_slug():
    assert run_name(NOW, "abc1234", False, "U-Net R34!") == "20261105-0930-abc1234-u-net-r34"
    assert run_name(NOW, "abc1234", True, "x") == "20261105-0930-abc1234-dirty-x"
    with pytest.raises(ValueError):
        slugify("!!!")


def test_create_run_writes_config_and_meta(tmp_path):
    d = create_run(BASE, "unet", tmp_path, now=NOW, git=GIT)
    assert d.name == "20261105-0930-abc1234-unet"
    assert yaml.safe_load((d / "config.yaml").read_text()) == BASE
    meta = read_meta(d)
    assert meta["status"] == "running"
    assert meta["config_hash"] == config_hash(BASE)
    assert meta["data_version"] == "splits-v2-abc"
    assert meta["commit"] == "abc1234" and meta["dirty"] is False


def test_create_run_never_reuses_folder(tmp_path):
    create_run(BASE, "unet", tmp_path, now=NOW, git=GIT)
    with pytest.raises(FileExistsError):
        create_run(BASE, "unet", tmp_path, now=NOW, git=GIT)


def test_cache_hit_only_for_finished_clean_runs(tmp_path):
    h = config_hash(BASE)
    d = create_run(BASE, "unet", tmp_path, now=NOW, git=GIT)
    assert find_finished(h, tmp_path) is None  # still running
    mark_finished(d, best_val_miou=None)
    assert find_finished(h, tmp_path) == d
    with pytest.raises(AlreadyFinished) as e:
        create_run(BASE, "unet-again", tmp_path, now=NOW + dt.timedelta(minutes=1), git=GIT)
    assert e.value.run_dir == d


def test_dirty_run_is_not_a_cache_hit(tmp_path):
    d = create_run(BASE, "unet", tmp_path, now=NOW, git=("abc1234", True))
    mark_finished(d)
    assert find_finished(config_hash(BASE), tmp_path) is None


def test_find_finished_skips_broken_meta(tmp_path):
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "meta.json").write_text("{not json")
    assert find_finished("x", tmp_path) is None
    assert find_finished("x", tmp_path / "missing") is None


def test_git_state_in_this_repo():
    commit, dirty = runs.git_state()
    assert len(commit) >= 7 and isinstance(dirty, bool)


def test_gpu_lock_exclusive_and_released(tmp_path):
    with gpu_lock("run-a", tmp_path) as path:
        holder = json.loads(path.read_text())
        assert holder["run"] == "run-a" and holder["pid"] == os.getpid()
        assert stat.S_IMODE(path.stat().st_mode) == 0o664
        with pytest.raises(GpuBusy, match="run-a"), gpu_lock("run-b", tmp_path):
            pass
    with gpu_lock("run-b", tmp_path):  # free again after release
        pass
    assert (tmp_path / ".gpu.lock").read_text() == ""


def test_gpu_lock_released_on_error(tmp_path):
    with pytest.raises(RuntimeError), gpu_lock("boom", tmp_path):
        raise RuntimeError
    with gpu_lock("next", tmp_path):
        pass
