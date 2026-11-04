"""Run management (D7): config loading, hashing, run folders, cache check, GPU lock.

- Config = `configs/base.yaml` deep-merged with one experiment YAML. The
  experiment may not change shared comparison fields (D5) unless it lists
  them under `run.overrides_shared` (e.g. E2 changes `data.datasets`).
- `config_hash` = sha256 of the resolved config as canonical JSON, without the
  `run` section (slug, notes: names, not experiment settings). It includes
  `data.version`, so the data version is part of the key.
- Run folder: `<YYYYMMDD-HHMM>-<shortcommit>[-dirty]-<slug>`, set at start,
  holding `config.yaml` (resolved) and `meta.json`.
- Cache: a finished, clean (not dirty) run with the same hash is never
  retrained. The runs folder itself is the registry.
- GPU lock: `flock` on `runs/.gpu.lock`, released by the kernel even if the
  process crashes. The file names the current holder.
"""

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import fcntl
import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path

import yaml

from cloth_segment import paths

SHARED_FIELDS: tuple[str, ...] = (
    "seed",
    "data",
    "input",
    "augment",
    "train.batch_size",
    "train.iters",
)
NOT_HASHED = ("run",)
LOCK_NAME = ".gpu.lock"


# ---------------------------------------------------------------- config


def deep_merge(base: dict, over: dict) -> dict:
    """Return `base` updated by `over`; nested dicts merge, everything else replaces."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _get(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


_MISSING = object()


def shared_overrides(base: dict, resolved: dict) -> list[str]:
    """Shared fields whose resolved value differs from base."""
    return [f for f in SHARED_FIELDS if _get(base, f) != _get(resolved, f)]


def load_config(path: Path, base_path: Path | None = None) -> dict:
    """Resolve an experiment YAML on top of base.yaml and enforce D5.

    Raises if a shared field changes without being declared in
    `run.overrides_shared`, or if a declared one does not actually change.
    """
    base_path = base_path or paths.CONFIGS / "base.yaml"
    base = yaml.safe_load(base_path.read_text()) or {}
    if Path(path).resolve() == base_path.resolve():
        return base
    exp = yaml.safe_load(Path(path).read_text()) or {}
    resolved = deep_merge(base, exp)
    changed = shared_overrides(base, resolved)
    declared = list((resolved.get("run") or {}).get("overrides_shared") or [])
    undeclared = [f for f in changed if f not in declared]
    if undeclared:
        raise ValueError(
            f"{path}: changes shared comparison fields {undeclared} (D5). "
            "List them in run.overrides_shared if this experiment really needs it."
        )
    unused = [f for f in declared if f not in changed]
    if unused:
        raise ValueError(f"{path}: run.overrides_shared lists unchanged fields {unused}")
    return resolved


def config_hash(cfg: dict) -> str:
    """12-hex sha256 of the resolved config without the `run` section; key order ignored."""
    hashed = {k: v for k, v in cfg.items() if k not in NOT_HASHED}
    blob = json.dumps(hashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- git / naming


def git_state(repo: Path = paths.REPO_ROOT) -> tuple[str, bool]:
    """(short commit, dirty). Untracked files do not count as dirty."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    commit = git("rev-parse", "--short=7", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return commit, dirty


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        raise ValueError(f"empty slug from {text!r}")
    return slug[:48]


def run_name(now: dt.datetime, commit: str, dirty: bool, slug: str) -> str:
    return f"{now:%Y%m%d-%H%M}-{commit}{'-dirty' if dirty else ''}-{slugify(slug)}"


# ---------------------------------------------------------------- run folders


def read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def find_finished(cfg_hash: str, runs_dir: Path = paths.RUNS) -> Path | None:
    """A finished, clean run with this config hash, or None (newest first)."""
    if not runs_dir.is_dir():
        return None
    for meta_path in sorted(runs_dir.glob("*/meta.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            meta.get("config_hash") == cfg_hash
            and meta.get("status") == "finished"
            and not meta.get("dirty", True)
        ):
            return meta_path.parent
    return None


class AlreadyFinished(RuntimeError):
    def __init__(self, run_dir: Path):
        super().__init__(f"a finished run with the same config hash exists: {run_dir}")
        self.run_dir = run_dir


def create_run(
    cfg: dict,
    slug: str,
    runs_dir: Path = paths.RUNS,
    *,
    now: dt.datetime | None = None,
    git: tuple[str, bool] | None = None,
) -> Path:
    """Make the run folder with `config.yaml` and `meta.json` (status running).

    Raises AlreadyFinished on a cache hit (D7). Never reuses an existing folder.
    """
    h = config_hash(cfg)
    if (hit := find_finished(h, runs_dir)) is not None:
        raise AlreadyFinished(hit)
    now = now or dt.datetime.now().astimezone()
    commit, dirty = git or git_state()
    run_dir = runs_dir / run_name(now, commit, dirty, slug)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {
        "name": run_dir.name,
        "status": "running",
        "config_hash": h,
        "data_version": (cfg.get("data") or {}).get("version"),
        "commit": commit,
        "dirty": dirty,
        "started_at": now.isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
    }
    _write_json(run_dir / "meta.json", meta)
    return run_dir


def update_meta(run_dir: Path, **fields) -> dict:
    meta = read_meta(run_dir) | fields
    _write_json(run_dir / "meta.json", meta)
    return meta


def mark_finished(run_dir: Path, **fields) -> dict:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    return update_meta(run_dir, status="finished", finished_at=now, **fields)


def mark_failed(run_dir: Path, error: str) -> dict:
    return update_meta(run_dir, status="failed", error=error)


# ---------------------------------------------------------------- GPU lock


class GpuBusy(RuntimeError):
    pass


@contextlib.contextmanager
def gpu_lock(run: str = "", lock_dir: Path = paths.RUNS, *, wait: bool = False) -> Iterator[Path]:
    """Hold the shared GPU lock for the duration of the block.

    Non-blocking by default: raises GpuBusy naming the current holder. The lock
    file is group-writable so every team member can take it.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / LOCK_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        with contextlib.suppress(PermissionError):  # file owned by another member
            os.fchmod(fd, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            holder = os.pread(fd, 4096, 0).decode(errors="replace").strip() or "unknown"
            raise GpuBusy(f"GPU lock held: {holder}") from None
        info = {
            "user": getpass.getuser(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "run": run,
            "since": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        os.ftruncate(fd, 0)
        os.pwrite(fd, json.dumps(info).encode(), 0)
        try:
            yield path
        finally:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
