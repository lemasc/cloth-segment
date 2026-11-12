"""Record which GPU a run used, so runs from different machines can be told apart.

- `gpu_fields` gives the four `meta.json` fields at run start. All are None off cuda.
- `backfill_run` adds the same fields to an older run, only where they are missing, and
  only on the machine that trained it. It is the one allowed way to write them afterwards.
- Software versions are not recorded here: `uv.lock` at the run's `commit` pins them.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import torch

from cloth_segment import runs

GPU_FIELDS: tuple[str, ...] = ("gpu_name", "gpu_total_mib", "gpu_capability", "driver_version")
_NO_GPU: dict = dict.fromkeys(GPU_FIELDS)


class BackfillRefused(RuntimeError):
    """The run cannot be backfilled from this machine (reason in the message)."""


def driver_version() -> str | None:
    """NVIDIA driver version from `nvidia-smi`, or None if it is missing or fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.strip().splitlines()  # one line per GPU, all the same driver
    return lines[0].strip() or None if lines else None


def gpu_fields(device: torch.device) -> dict:
    """The `meta.json` hardware fields for `device`; all None when it is not cuda."""
    if device.type != "cuda":
        return dict(_NO_GPU)
    index = torch.cuda.current_device() if device.index is None else device.index
    props = torch.cuda.get_device_properties(index)
    return {
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_total_mib": props.total_memory / 2**20,
        "gpu_capability": f"{props.major}.{props.minor}",
        "driver_version": driver_version(),
    }


def backfill_run(run_dir: Path, *, dry_run: bool = False, hostname: str | None = None) -> dict:
    """Add missing hardware fields to a run's `meta.json`; returns the fields added.

    Never overwrites a key that is already there (even null). Refuses a run from another
    host or one with no `device`, since this machine's GPU would be the wrong answer.
    A run that gets changes is marked `hw_backfilled: true`. `dry_run` writes nothing.
    """
    hostname = hostname or socket.gethostname()
    meta = runs.read_meta(run_dir)
    if meta.get("host") != hostname:
        raise BackfillRefused(f"run host {meta.get('host')!r} is not this machine {hostname!r}")
    if "device" not in meta:
        raise BackfillRefused("meta.json has no 'device'; cannot tell which GPU it used")
    device = torch.device(meta["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BackfillRefused("run used cuda but no GPU is available here")
    missing = [f for f in GPU_FIELDS if f not in meta]
    if not missing:
        return {}
    current = gpu_fields(device)
    changes = {f: current[f] for f in missing}
    if not dry_run:
        runs.update_meta(run_dir, **changes, hw_backfilled=True)
    return changes
