"""GPU memory and speed probe for choosing the shared batch size (T5, D5).

Runs a few real training steps (forward, loss, backward, optimizer step) per
model and batch size on random 512x512 inputs, and records peak GPU memory
and seconds per iteration. Random data is enough: the train transform always
crops to `size x size`, so memory does not depend on the images.

Models are built with random weights (no download): pretrained weights do not
change memory or speed. The builders here are provisional; T6-T8 own the real
ones in `cloth_segment.models`.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from cloth_segment.data.labels import IGNORE, NUM_CLASSES


class _LogitsOnly(nn.Module):
    """Wrap a Hugging Face model so it returns a plain logits tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).logits


def _unet_r34() -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet("resnet34", encoder_weights=None, classes=NUM_CLASSES)


def _deeplabv3p_r50() -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.DeepLabV3Plus("resnet50", encoder_weights=None, classes=NUM_CLASSES)


def _segformer_b2() -> nn.Module:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    # MiT-B2 sizes, as in the nvidia/mit-b2 config (written out to avoid a download).
    cfg = SegformerConfig(
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1],
        decoder_hidden_size=768,
        num_labels=NUM_CLASSES,
    )
    return _LogitsOnly(SegformerForSemanticSegmentation(cfg))


ARCHITECTURES: dict[str, Callable[[], nn.Module]] = {
    "unet_r34": _unet_r34,
    "deeplabv3p_r50": _deeplabv3p_r50,
    "segformer_b2": _segformer_b2,
}


@dataclass(frozen=True)
class ProbeResult:
    model: str
    batch_size: int
    ok: bool
    peak_mem_mib: float | None  # None on CPU
    sec_per_iter: float | None  # None if it ran out of memory
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def probe_step(
    name: str,
    model: nn.Module,
    batch_size: int,
    *,
    size: int,
    device: torch.device,
    amp: bool,
    iters: int,
    warmup: int,
    seed: int = 0,
) -> ProbeResult:
    """Time `iters` training steps after `warmup` steps; report peak memory.

    Logits smaller than the input (SegFormer predicts at 1/4) are upsampled to
    the input size before the loss, as the training loop will do. AdamW is used
    because it keeps more optimizer state than SGD, so the result is an upper
    bound for either choice.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch_size, 3, size, size, generator=gen).to(device)
    y = torch.randint(0, NUM_CLASSES, (batch_size, size, size), generator=gen).to(device)
    y[:, :8] = IGNORE  # some ignore pixels, like padding

    amp_dtype = torch.bfloat16
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(
                    logits, size=y.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = F.cross_entropy(logits.float(), y, ignore_index=IGNORE)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for _ in range(warmup):
            step()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        _sync(device)
        sec = (time.perf_counter() - t0) / iters
    except torch.OutOfMemoryError as e:
        return ProbeResult(name, batch_size, False, _peak(device), None, str(e).splitlines()[0])
    return ProbeResult(name, batch_size, True, _peak(device), sec)


def _peak(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


def _free(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def probe_models(
    names: list[str],
    batch_sizes: list[int],
    *,
    size: int,
    device: torch.device,
    amp: bool = True,
    iters: int = 20,
    warmup: int = 5,
    seed: int = 0,
    builders: dict[str, Callable[[], nn.Module]] | None = None,
    on_result: Callable[[ProbeResult], None] | None = None,
    max_mib: float | None = None,
) -> list[ProbeResult]:
    """Probe every (model, batch size). Stops growing the batch after the first failure.

    A run whose peak memory exceeds `max_mib` counts as a failure: on WSL the
    driver may spill into system RAM instead of raising OOM, which is very slow.
    """
    builders = builders or ARCHITECTURES
    results = []
    for name in names:
        for bs in sorted(batch_sizes):
            model = builders[name]()
            r = probe_step(
                name,
                model,
                bs,
                size=size,
                device=device,
                amp=amp,
                iters=iters,
                warmup=warmup,
                seed=seed,
            )
            if r.ok and max_mib and r.peak_mem_mib and r.peak_mem_mib > max_mib:
                r = replace(r, ok=False, error=f"peak {r.peak_mem_mib:.0f} MiB > {max_mib:.0f}")
            results.append(r)
            if on_result:
                on_result(r)
            del model
            _free(device)
            if not r.ok:
                break
    return results


def format_table(results: list[ProbeResult], total_mib: float | None = None) -> str:
    """Markdown table of results; adds % of total GPU memory when known."""
    head = "| model | batch | ok | peak MiB | % of GPU | s/iter | img/s |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    for r in results:
        mem = f"{r.peak_mem_mib:.0f}" if r.peak_mem_mib is not None else "-"
        pct = (
            f"{100 * r.peak_mem_mib / total_mib:.0f}"
            if r.peak_mem_mib is not None and total_mib
            else "-"
        )
        sec = f"{r.sec_per_iter:.3f}" if r.sec_per_iter else "-"
        ips = f"{r.batch_size / r.sec_per_iter:.1f}" if r.sec_per_iter else "-"
        ok = "yes" if r.ok else ("over" if r.sec_per_iter else "OOM")
        lines.append(f"| {r.model} | {r.batch_size} | {ok} | {mem} | {pct} | {sec} | {ips} |")
    return "\n".join(lines)
