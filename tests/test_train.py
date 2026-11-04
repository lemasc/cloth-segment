"""T5 training/eval pieces: LR schedule, eval steps, loss, model registry, val scoring.

No real data and no GPU; the training loop itself is not tested end to end
(docs/conventions.md).
"""

import pytest
import torch
import yaml
from test_dataset import _write_fixture
from torch import nn

from cloth_segment import evaluate, models, paths, runs
from cloth_segment.data.dataset import SegDataset, class_pixel_counts
from cloth_segment.data.labels import IGNORE, NUM_CLASSES
from cloth_segment.models import build_model, logits_at, register
from cloth_segment.train import build_optimizer, cap_gpu_memory, eval_steps, loss_fn, lr_factor

# --- LR schedule and eval steps ---------------------------------------------------


def test_lr_factor_warmup_then_poly():
    iters, warmup = 100, 10
    f = [lr_factor(s, iters, warmup, 1.0) for s in range(iters)]
    assert f[0] == pytest.approx(0.1)
    assert f[warmup - 1] == pytest.approx(1.0)
    assert f[warmup] == pytest.approx(1.0)
    assert all(a <= b for a, b in zip(f[:warmup], f[1:warmup]))  # rising
    assert all(a >= b for a, b in zip(f[warmup:], f[warmup + 1 :]))  # falling
    assert f[-1] == pytest.approx(1 / 90)
    assert lr_factor(iters, iters, warmup, 1.0) == 0.0


def test_lr_factor_power_and_no_warmup():
    assert lr_factor(0, 10, 0, 1.0) == 1.0
    assert lr_factor(5, 10, 0, 2.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        lr_factor(0, 10, 10, 1.0)


def test_eval_steps_include_last():
    assert eval_steps(40000, 4000) == list(range(4000, 40001, 4000))
    assert eval_steps(10, 4) == [4, 8, 10]
    assert eval_steps(3, 4) == [3]
    with pytest.raises(ValueError):
        eval_steps(10, 0)


# --- loss and optimizer -------------------------------------------------------------


def test_loss_upsamples_and_skips_ignore():
    target = torch.full((1, 8, 8), 2, dtype=torch.int64)
    target[0, :4] = IGNORE
    logits = torch.zeros(1, NUM_CLASSES, 2, 2)
    logits[:, 2] = 10.0  # confident and right
    assert loss_fn(logits, target, {"name": "ce"}) < 1e-3
    all_ignore = torch.full((1, 8, 8), IGNORE, dtype=torch.int64)
    assert torch.isnan(loss_fn(logits, all_ignore, {"name": "ce"}))  # caught by fit()
    with pytest.raises(ValueError):
        loss_fn(logits, target, {"name": "dice"})


def test_build_optimizer_from_base_config():
    cfg = yaml.safe_load((paths.CONFIGS / "base.yaml").read_text())
    opt = build_optimizer(nn.Linear(2, 2), cfg["optim"])
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == cfg["optim"]["lr"]
    with pytest.raises(ValueError):
        build_optimizer(nn.Linear(2, 2), cfg["optim"] | {"name": "sgd"})


# --- shared fields (D5) -------------------------------------------------------------


def _exp(tmp_path, data):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_model_config_may_set_lr_but_not_loss(tmp_path):
    base = paths.CONFIGS / "base.yaml"
    cfg = runs.load_config(_exp(tmp_path, {"optim": {"lr": 6e-5}}), base)
    assert cfg["optim"]["lr"] == 6e-5 and cfg["optim"]["name"] == "adamw"
    for change in ({"loss": {"name": "ce_dice"}}, {"train": {"amp": "none"}}):
        with pytest.raises(ValueError, match="shared comparison"):
            runs.load_config(_exp(tmp_path, change), base)
    e3 = {"loss": {"name": "ce_dice"}, "run": {"overrides_shared": ["loss"]}}
    assert runs.load_config(_exp(tmp_path, e3), base)["loss"]["name"] == "ce_dice"


def test_run_section_is_not_hashed():
    cfg = yaml.safe_load((paths.CONFIGS / "base.yaml").read_text())
    other = cfg | {"run": {"num_workers": 0, "log_every": 1}}
    assert runs.config_hash(cfg) == runs.config_hash(other)


# --- model registry -----------------------------------------------------------------


def test_build_model_errors():
    with pytest.raises(ValueError, match="no model.name"):
        build_model({"name": None})
    with pytest.raises(NotImplementedError, match="T7"):
        build_model({"name": "deeplabv3p_r50"})
    with pytest.raises(ValueError, match="unknown"):
        build_model({"name": "nope"})


def test_register_and_build(monkeypatch):
    monkeypatch.setattr(models, "REGISTRY", {})
    register("tiny")(lambda cfg: nn.Conv2d(3, NUM_CLASSES, 1))
    assert isinstance(build_model({"name": "tiny"}), nn.Conv2d)
    with pytest.raises(ValueError, match="twice"):
        register("tiny")(lambda cfg: nn.Identity())


def test_logits_at():
    x = torch.randn(2, NUM_CLASSES, 4, 4)
    assert logits_at(x, (4, 4)) is x
    assert logits_at(x, (16, 16)).shape == (2, NUM_CLASSES, 16, 16)


@pytest.mark.parametrize(("device", "index"), [("cuda", 0), ("cuda:1", 1)])
def test_cap_gpu_memory_passes_a_device_index(monkeypatch, device, index):
    # torch raises on a bare torch.device("cuda"), the default of `train.main`
    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (6 * 2**30, 12 * 2**30))
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda f, i: calls.append((f, i))
    )
    assert cap_gpu_memory(torch.device(device)) == 6 * 1024
    assert calls == [(0.5, index)]
    assert cap_gpu_memory(torch.device("cpu")) is None


def test_amp_dtype():
    cpu = torch.device("cpu")
    assert evaluate.amp_dtype("bf16", cpu) is None  # autocast only on CUDA
    with pytest.raises(ValueError):
        evaluate.amp_dtype("fp8", cpu)


# --- val scoring at original resolution (eval-v1) ----------------------------------

SPECS = [
    ("lip", "val", 120, 70, False),
    ("cihp", "val", 100, 150, False),
    ("dfmm", "val", 220, 150, False),
]


class _AlwaysClass(nn.Module):
    """Predicts `cls` everywhere, at 1/4 resolution like SegFormer."""

    def __init__(self, cls: int):
        super().__init__()
        self.cls = cls

    def forward(self, x):
        out = torch.zeros(x.shape[0], NUM_CLASSES, x.shape[2] // 4, x.shape[3] // 4)
        out[:, self.cls] = 1.0
        return out


def test_predict_confusion_counts_original_pixels(tmp_path):
    df, ir, mr = _write_fixture(tmp_path, SPECS)
    ds = SegDataset(df, size=64, train=False, image_root=ir, mask_root=mr)
    loader = evaluate.eval_loader(ds, batch_size=2, num_workers=0)
    cm = evaluate.predict_confusion(_AlwaysClass(4), loader, torch.device("cpu"))
    # Every non-ignore pixel of the original masks lands in column 4, once.
    assert cm[:, 4].tolist() == class_pixel_counts(df).tolist()
    assert cm.sum() == cm[:, 4].sum()
    s = evaluate.scores(cm)  # fixture masks use classes 0, 1, 4, 8
    assert s["iou"][evaluate.SHARED_CLASSES[4]] > 0
    assert s["iou"][evaluate.SHARED_CLASSES[1]] == 0


def test_eval_loader_rejects_train_dataset(tmp_path):
    df, ir, mr = _write_fixture(tmp_path, SPECS[:1])
    ds = SegDataset(df, size=64, train=True, image_root=ir, mask_root=mr)
    with pytest.raises(ValueError, match="train=False"):
        evaluate.WithOriginalMask(ds)


# --- fit(): timing and the no-best.pt failure ----------------------------------------


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, NUM_CLASSES, 1)

    def forward(self, x):
        return self.conv(x)


def _fit_cfg(iters: int) -> dict:
    return {
        "seed": 0,
        "data": {"sampling": "natural"},
        "train": {"batch_size": 2, "iters": iters, "amp": "none"},
        "optim": {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "warmup_iters": 1,
            "poly_power": 1.0,
        },
        "loss": {"name": "ce"},
        "eval": {"split": "val", "every": 2, "select": "miou"},
        "run": {"num_workers": 0, "log_every": 1},
    }


def _fit_sets(tmp_path):
    specs = [("lip", "train", 64, 64, True)] * 4 + [("lip", "val", 64, 64, False)] * 2
    df, ir, mr = _write_fixture(tmp_path, specs)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    kw = {"size": 64, "image_root": ir, "mask_root": mr}
    return SegDataset(train, train=True, **kw), SegDataset(val, train=False, **kw)


def test_fit_reports_train_and_eval_seconds_separately(tmp_path):
    from cloth_segment.train import fit

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_set, val_set = _fit_sets(tmp_path)
    out = fit(
        _fit_cfg(2),
        run_dir,
        torch.device("cpu"),
        model=_Tiny(),
        train_set=train_set,
        val_set=val_set,
    )
    assert out["best_step"] == 2 and out["best_val_miou"] is not None
    assert out["eval_seconds"] >= 0 and out["train_seconds"] >= 0
    assert (run_dir / "ckpt" / "best.pt").exists()


def test_fit_fails_clearly_when_val_miou_is_never_defined(tmp_path, monkeypatch):
    from cloth_segment import train

    monkeypatch.setattr(
        train.evaluate, "scores", lambda cm, names: {"miou": None, "eval_version": "x"}
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_set, val_set = _fit_sets(tmp_path)
    with pytest.raises(RuntimeError, match="no best.pt"):
        train.fit(
            _fit_cfg(2),
            run_dir,
            torch.device("cpu"),
            model=_Tiny(),
            train_set=train_set,
            val_set=val_set,
        )
