"""Measure peak GPU memory and speed per model and batch size (T5).

Use the result to pick one batch size for every model (D5). Runs a few
training steps on random data; no dataset, no download, no run folder.

This uses the GPU. On a shared machine, make sure nobody else is training
first (the GPU lock in runs.py is not built yet).

Usage:
    uv run python scripts/probe_memory.py
    uv run python scripts/probe_memory.py --models segformer_b2 --batch-sizes 8 12 16
    uv run python scripts/probe_memory.py --out probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from cloth_segment import paths
from cloth_segment.probe import ARCHITECTURES, format_table, probe_models


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=paths.CONFIGS / "base.yaml")
    p.add_argument("--models", nargs="+", default=list(ARCHITECTURES), choices=list(ARCHITECTURES))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 12, 16, 24])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="smoke test only")
    p.add_argument("--out", type=Path, help="also write results as JSON")
    args = p.parse_args()
    if min(args.batch_sizes) < 2:
        p.error("batch sizes must be >= 2 (DeepLabV3+ ASPP pooling cannot train on batch 1)")

    cfg = yaml.safe_load(args.config.read_text())
    size = int(cfg["input"]["size"])
    seed = int(cfg["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        free, total = torch.cuda.mem_get_info(device)
        total_mib, used_mib = total / 2**20, (total - free) / 2**20
        print(f"GPU: {torch.cuda.get_device_name(device)}, {total_mib:.0f} MiB total")
        if used_mib > 1024:
            print(f"WARNING: {used_mib:.0f} MiB already in use by other processes.")
        # Cap at the memory that is free now, so PyTorch raises OOM instead of the
        # WSL driver silently spilling into system RAM (very slow, not a real fit).
        max_mib = free / 2**20
        torch.cuda.set_per_process_memory_fraction(free / total, device)
        print(f"cap: {max_mib:.0f} MiB")
    elif args.allow_cpu:
        device, total_mib, max_mib = torch.device("cpu"), None, None
    else:
        raise SystemExit("No CUDA device. Use --allow-cpu for a smoke test.")

    amp = not args.no_amp
    print(f"size={size} amp={amp} iters={args.iters} warmup={args.warmup}\n")
    results = probe_models(
        args.models,
        args.batch_sizes,
        size=size,
        device=device,
        amp=amp,
        iters=args.iters,
        warmup=args.warmup,
        seed=seed,
        max_mib=max_mib,
        on_result=lambda r: print(format_table([r], total_mib).splitlines()[-1], flush=True),
    )
    print("\n" + format_table(results, total_mib))

    if args.out:
        meta = {
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "total_mib": total_mib,
            "torch": torch.__version__,
            "size": size,
            "amp": amp,
            "iters": args.iters,
            "warmup": args.warmup,
        }
        payload = {"meta": meta, "results": [r.as_dict() for r in results]}
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
