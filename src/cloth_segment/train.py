"""Training entry point: `python -m cloth_segment.train <config.yaml>`. Task T5.

One loop for every model (D5). Flow:

1. Resolve the config (`runs.load_config`, enforces D5). If a finished clean
   run with the same hash exists, stop (D7 cache).
2. On a GPU, take the shared GPU lock, then create the run folder.
3. Train for `train.iters` optimizer steps: AdamW, linear warmup then poly
   decay, cross-entropy with IGNORE skipped, autocast per `train.amp`.
4. Every `eval.every` steps and at the last step, score the internal val split
   with the eval-v1 protocol. Keep `ckpt/last.pt` (model, optimizer, step) and
   `ckpt/best.pt` (model only, highest val mIoU).
5. Log to `log.jsonl` (one JSON object per line); write the result to
   `meta.json` (`finished` or `failed`).

On CUDA, PyTorch is capped to the GPU memory free at start, so a batch that
does not fit raises OOM instead of spilling into system RAM (WSL, T5 probe).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
import traceback
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cloth_segment import evaluate, runs
from cloth_segment.data.dataset import SegDataset, collate, load_split, make_sampler
from cloth_segment.data.labels import IGNORE, SHARED_CLASSES
from cloth_segment.models import build_model, logits_at

LOSSES = ("ce",)


def lr_factor(step: int, iters: int, warmup: int, power: float) -> float:
    """LR multiplier at 0-based `step`: linear warmup to 1, then poly decay to 0 at `iters`."""
    if not 0 <= warmup < iters:
        raise ValueError(f"need 0 <= warmup_iters < iters, got {warmup}, {iters}")
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, 1.0 - (step - warmup) / (iters - warmup)) ** power


def eval_steps(iters: int, every: int) -> list[int]:
    """1-based steps after which val is scored: every `every`, and always the last."""
    if every <= 0:
        raise ValueError(f"eval.every must be positive, got {every}")
    steps = list(range(every, iters + 1, every))
    if not steps or steps[-1] != iters:
        steps.append(iters)
    return steps


def build_optimizer(model: nn.Module, optim_cfg: dict) -> torch.optim.Optimizer:
    if optim_cfg["name"] != "adamw":
        raise ValueError(f"optim.name must be adamw, got {optim_cfg['name']!r}")
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=optim_cfg["lr"],
        betas=tuple(optim_cfg["betas"]),
        weight_decay=optim_cfg["weight_decay"],
    )


def loss_fn(logits: torch.Tensor, target: torch.Tensor, loss_cfg: dict) -> torch.Tensor:
    """Segmentation loss on logits upsampled to the mask size; IGNORE pixels skipped."""
    if loss_cfg["name"] not in LOSSES:
        raise ValueError(f"loss.name must be one of {LOSSES}, got {loss_cfg['name']!r}")
    logits = logits_at(logits, target.shape[-2:])
    return F.cross_entropy(logits.float(), target, ignore_index=IGNORE)


def train_loader(dataset: SegDataset, cfg: dict) -> DataLoader:
    """Seeded loader: the generator fixes shuffling and each worker's augmentation RNG."""
    gen = torch.Generator().manual_seed(cfg["seed"])
    sampler = make_sampler(dataset, cfg["data"]["sampling"], generator=gen)
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        generator=gen,
        num_workers=cfg["run"]["num_workers"],
        collate_fn=collate,
        drop_last=True,  # DeepLabV3+ cannot train on a batch of 1 (T5 probe)
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg["run"]["num_workers"] > 0,
    )


def forever(loader: DataLoader) -> Iterator[dict]:
    """Loop over the loader epoch after epoch."""
    while True:
        yield from loader


class JsonLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, **fields) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")


def cap_gpu_memory(device: torch.device) -> float | None:
    """Cap this process to the GPU memory free now; returns the cap in MiB."""
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.set_per_process_memory_fraction(free / total, device)
    return free / 2**20


def fit(
    cfg: dict,
    run_dir: Path,
    device: torch.device,
    *,
    model: nn.Module,
    train_set: SegDataset,
    val_set: SegDataset,
) -> dict:
    """Train and validate; returns the fields for `meta.json`."""
    tr, oc = cfg["train"], cfg["optim"]
    iters = tr["iters"]
    torch.manual_seed(cfg["seed"])
    dtype = evaluate.amp_dtype(tr["amp"], device)
    model = model.to(device).train()
    opt = build_optimizer(model, oc)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, iters, oc["warmup_iters"], oc["poly_power"])
    )
    scaler = torch.amp.GradScaler(device.type, enabled=dtype is torch.float16)
    loader = train_loader(train_set, cfg)
    val_loader = evaluate.eval_loader(val_set, tr["batch_size"], cfg["run"]["num_workers"])
    checks = set(eval_steps(iters, cfg["eval"]["every"]))
    select = cfg["eval"]["select"]
    if select != "miou":
        raise ValueError(f"eval.select must be miou, got {select!r}")

    ckpt = run_dir / "ckpt"
    ckpt.mkdir(exist_ok=True)
    log = JsonLog(run_dir / "log.jsonl")
    log_every = cfg["run"]["log_every"]
    best = {"step": None, "miou": -math.inf}
    t0 = t_log = time.perf_counter()
    loss_sum, loss_n = 0.0, 0
    eval_seconds = 0.0
    batches = forever(loader)

    for step in range(1, iters + 1):
        batch = next(batches)
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with evaluate.autocast(device, dtype):
            logits = model(x)
        loss = loss_fn(logits, y, cfg["loss"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss is {loss.item()} at step {step}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        lr = opt.param_groups[0]["lr"]  # the LR this step used
        sched.step()
        loss_sum += loss.item()
        loss_n += 1

        if step % log_every == 0 or step == iters:
            now = time.perf_counter()
            log.write(
                step=step,
                loss=loss_sum / loss_n,
                lr=lr,
                sec_per_iter=(now - t_log) / loss_n,
            )
            print(f"step {step}/{iters} loss {loss_sum / loss_n:.4f}", flush=True)
            loss_sum, loss_n, t_log = 0.0, 0, now

        if step in checks:
            t_eval = time.perf_counter()
            cm = evaluate.predict_confusion(model, val_loader, device, dtype)
            model.train()
            s = evaluate.scores(cm, SHARED_CLASSES)
            log.write(step=step, split=cfg["eval"]["split"], **s)
            miou = "undefined" if s["miou"] is None else f"{s['miou']:.4f}"
            print(f"step {step} val mIoU {miou}", flush=True)
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                ckpt / "last.pt",
            )
            if s["miou"] is not None and s["miou"] > best["miou"]:
                best = {"step": step, "miou": s["miou"]}
                torch.save({"step": step, "model": model.state_dict()}, ckpt / "best.pt")
            t_log = time.perf_counter()  # do not count eval time as train time
            eval_seconds += t_log - t_eval

    if best["step"] is None:
        raise RuntimeError("no best.pt: val mIoU was undefined at every check (val has no labels?)")
    total = time.perf_counter() - t0
    return {
        "steps": iters,
        "best_step": best["step"],
        "best_val_miou": best["miou"],
        "train_seconds": round(total - eval_seconds, 1),  # without validation
        "eval_seconds": round(eval_seconds, 1),
    }


def run(config: Path, *, device: torch.device, wait: bool = False) -> Path:
    """Resolve, lock, create the run folder and train. Returns the run folder."""
    cfg = runs.load_config(config)
    cfg.setdefault("run", {})
    slug = cfg["run"].get("slug") or Path(config).stem
    if (hit := runs.find_finished(runs.config_hash(cfg))) is not None:
        raise runs.AlreadyFinished(hit)
    model = build_model(cfg["model"])  # fail before taking the lock or making a folder
    train_set = load_split(cfg, "train")
    val_set = load_split(cfg, cfg["eval"]["split"], train=False)

    lock = runs.gpu_lock(slug, wait=wait) if device.type == "cuda" else contextlib.nullcontext()
    with lock:
        run_dir = runs.create_run(cfg, slug)
        print(f"run: {run_dir}", flush=True)
        try:
            cap = cap_gpu_memory(device)
            runs.update_meta(run_dir, device=str(device), gpu_cap_mib=cap)
            result = fit(cfg, run_dir, device, model=model, train_set=train_set, val_set=val_set)
        except BaseException as e:  # also Ctrl-C: the folder must not stay "running"
            runs.mark_failed(run_dir, f"{type(e).__name__}: {e}")
            (run_dir / "error.txt").write_text(traceback.format_exc())
            raise
        runs.mark_finished(run_dir, **result)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", type=Path, help="experiment YAML (merged onto configs/base.yaml)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wait", action="store_true", help="wait for the GPU lock instead of failing")
    args = p.parse_args(argv)
    try:
        run(args.config, device=torch.device(args.device), wait=args.wait)
    except runs.AlreadyFinished as e:
        print(f"skip: {e}")
    except runs.GpuBusy as e:
        print(f"{e}; use --wait to queue", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
