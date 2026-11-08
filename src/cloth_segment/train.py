"""Training entry point: `python -m cloth_segment.train <config.yaml>`. Task T5.

One loop for every model (D5). Flow:

1. Resolve the config (`runs.load_config`, enforces D5). If a finished clean
   run with the same hash exists, stop (D7 cache).
2. On a GPU, take the shared GPU lock, then create the run folder.
   The model is built right after `torch.manual_seed(seed)`, so its randomly
   initialised layers (decoder, head) are the same on every run (M4).
3. Train for `train.iters` optimizer steps: AdamW, linear warmup then poly
   decay, cross-entropy with IGNORE skipped, autocast per `train.amp`.
4. Every `eval.every` steps and at the last step, score the internal val split
   per dataset (`evaluate`, eval-v2). Keep `ckpt/last.pt` (model, optimizer,
   step) and `ckpt/best.pt` (model only, highest `eval.select` value, by
   default the mean of the per-dataset val mIoUs).
5. Log to `log.jsonl` (one JSON object per line); write the result to
   `meta.json` (`finished` or `failed`). `meta.json` also records the
   parameter count (`params`, `trainable_params`), reported with results (D5).

Multi-task models (E1, `forward_multi`): the loss is
`L_seg + model.attr.loss_weight * L_attr` (`attributes.attr_loss`), and each
check also logs attribute macro-F1 on the DFMM val split. The checkpoint rule
(`eval.select`) stays segmentation-only, so `best.pt` is chosen as for the
other models. Segmentation-only models train exactly as before.

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

from cloth_segment import attr_eval, attributes, evaluate, hardware, runs
from cloth_segment.data.dataset import SegDataset, collate, load_split, make_sampler
from cloth_segment.data.labels import IGNORE, SHARED_CLASSES
from cloth_segment.models import build_model, count_parameters, logits_at

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
    # set_per_process_memory_fraction rejects a device without an index ("cuda")
    index = torch.cuda.current_device() if device.index is None else device.index
    free, total = torch.cuda.mem_get_info(index)
    torch.cuda.set_per_process_memory_fraction(free / total, index)
    return free / 2**20


def fit(
    cfg: dict,
    run_dir: Path,
    device: torch.device,
    *,
    model: nn.Module,
    train_set: SegDataset,
    val_set: SegDataset,
    attr_val_set: SegDataset | None = None,
) -> dict:
    """Train and validate; returns the fields for `meta.json`.

    `attr_val_set` (DFMM val) is needed when the model has attribute heads.
    """
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
    if select not in evaluate.SELECT_RULES:  # fail before training, not at the first check
        raise ValueError(f"eval.select must be one of {evaluate.SELECT_RULES}, got {select!r}")
    split = cfg["eval"]["split"]
    multi = hasattr(model, "forward_multi")
    if multi and attr_val_set is None:
        raise ValueError("a multi-task model needs attr_val_set")
    attr_loader = (
        evaluate.eval_loader(attr_val_set, tr["batch_size"], cfg["run"]["num_workers"])
        if multi
        else None
    )
    parts = {"seg_loss": 0.0, "attr_loss": 0.0, "attr_images": 0}

    ckpt = run_dir / "ckpt"
    ckpt.mkdir(exist_ok=True)
    log = JsonLog(run_dir / "log.jsonl")
    log_every = cfg["run"]["log_every"]
    best: dict = {"step": None, "value": -math.inf}
    t0 = t_log = time.perf_counter()
    loss_sum, loss_n = 0.0, 0
    eval_seconds = 0.0
    batches = forever(loader)

    for step in range(1, iters + 1):
        batch = next(batches)
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with evaluate.autocast(device, dtype):
            if multi:
                logits, attr_logits = model.forward_multi(x)
            else:
                logits = model(x)
        loss = loss_fn(logits, y, cfg["loss"])
        if multi:
            targets = attributes.head_targets(batch["attrs"], model.spec).to(device)
            a_loss, n_img = attributes.attr_loss(attr_logits, targets)
            parts["seg_loss"] += loss.item()
            parts["attr_loss"] += a_loss.item()
            parts["attr_images"] += n_img
            loss = loss + model.spec.loss_weight * a_loss
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
            extra = {}
            if multi:  # means over the logged steps; attr_loss counts empty batches as 0
                extra = {k: v / loss_n for k, v in parts.items()}
                parts = dict.fromkeys(parts, 0.0)
            log.write(
                step=step,
                loss=loss_sum / loss_n,
                lr=lr,
                sec_per_iter=(now - t_log) / loss_n,
                **extra,
            )
            print(f"step {step}/{iters} loss {loss_sum / loss_n:.4f}", flush=True)
            loss_sum, loss_n, t_log = 0.0, 0, now

        if step in checks:
            t_eval = time.perf_counter()
            cms = evaluate.predict_confusions(model, val_loader, device, dtype)
            model.train()
            val = evaluate.val_scores(cms, SHARED_CLASSES)
            for name, s in val["datasets"].items():
                log.write(step=step, split=split, dataset=name, **s)
            log.write(step=step, split=split, dataset="pooled", **val["pooled"])
            value = evaluate.selection_value(val, select)
            log.write(step=step, split=split, select=select, value=value)
            print(f"step {step} {split} {_summary(val, select, value)}", flush=True)
            attr = None
            if multi:
                ay, ap = attr_eval.predict_attrs(model, attr_loader, device, dtype)
                model.train()
                attr = attributes.all_metrics(ay, ap, model.spec, attr_eval.MIN_COUNT)
                f1 = {h: m["macro_f1"] for h, m in attr["heads"].items()}
                log.write(
                    step=step,
                    split=split,
                    task="attr",
                    dataset=attr_eval.DATASET,
                    mean_macro_f1=attr["mean_macro_f1"],
                    mean_accuracy=attr["mean_accuracy"],
                    macro_f1=f1,
                )
                mf1 = attr["mean_macro_f1"]
                print(f"step {step} {split} attr mean macro-F1 {_fmt(mf1)}", flush=True)
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
            if value is not None and value > best["value"]:
                best = {"step": step, "value": value, "val": val, "attr": attr}
                torch.save({"step": step, "model": model.state_dict()}, ckpt / "best.pt")
            t_log = time.perf_counter()  # do not count eval time as train time
            eval_seconds += t_log - t_eval

    if best["step"] is None:
        raise RuntimeError(
            f"no best.pt: val {select} was undefined at every check (a val dataset has no labels?)"
        )
    total = time.perf_counter() - t0
    extra = {}
    if multi:
        extra["best_val_attr_mean_macro_f1"] = best["attr"]["mean_macro_f1"]
    return {
        "steps": iters,
        "best_step": best["step"],
        "best_val_select": select,
        "best_val_value": best["value"],
        "best_val_miou_by_dataset": {n: s["miou"] for n, s in best["val"]["datasets"].items()},
        "best_val_miou": best["val"]["miou"],  # pooled (eval-v1 meaning), for comparison
        "train_seconds": round(total - eval_seconds, 1),  # without validation
        "eval_seconds": round(eval_seconds, 1),
        **extra,
    }


def seeded_model(cfg: dict) -> nn.Module:
    """Build `cfg["model"]` after seeding with `cfg["seed"]`: same seed, same initial weights."""
    torch.manual_seed(cfg["seed"])
    return build_model(cfg["model"])


def _fmt(x: float | None) -> str:
    return "undefined" if x is None else f"{x:.4f}"


def _summary(val: dict, select: str, value: float | None) -> str:
    parts = [f"{n} {_fmt(s['miou'])}" for n, s in val["datasets"].items()]
    return f"{select} {_fmt(value)} (mIoU {', '.join(parts)}; pooled {_fmt(val['miou'])})"


def run(config: Path, *, device: torch.device, wait: bool = False) -> Path:
    """Resolve, lock, create the run folder and train. Returns the run folder."""
    cfg = runs.load_config(config)
    cfg.setdefault("run", {})
    slug = cfg["run"].get("slug") or Path(config).stem
    if (hit := runs.find_finished(runs.config_hash(cfg))) is not None:
        raise runs.AlreadyFinished(hit)
    model = seeded_model(cfg)  # fail before taking the lock or making a folder
    train_set = load_split(cfg, "train")
    val_set = load_split(cfg, cfg["eval"]["split"], train=False)
    attr_val_set = None
    if hasattr(model, "forward_multi"):  # attribute labels exist on DFMM only
        dfmm = cfg | {"data": cfg["data"] | {"datasets": [attr_eval.DATASET]}}
        attr_val_set = load_split(dfmm, cfg["eval"]["split"], train=False)

    lock = runs.gpu_lock(slug, wait=wait) if device.type == "cuda" else contextlib.nullcontext()
    with lock:
        run_dir = runs.create_run(cfg, slug)
        print(f"run: {run_dir}", flush=True)
        try:
            cap = cap_gpu_memory(device)
            runs.update_meta(
                run_dir,
                device=str(device),
                gpu_cap_mib=cap,
                **hardware.gpu_fields(device),
                params=count_parameters(model),
                trainable_params=count_parameters(model, trainable_only=True),
            )
            result = fit(
                cfg,
                run_dir,
                device,
                model=model,
                train_set=train_set,
                val_set=val_set,
                attr_val_set=attr_val_set,
            )
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
