"""Training/eval pieces: LR schedule, eval steps, loss, model registry, val scoring.

No real data and no GPU; the training loop itself is not tested end to end.
"""

import json

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


# --- shared fields -------------------------------------------------------------


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


def test_build_model_errors(monkeypatch):
    # A fake planned name, so the test does not depend on which models are built yet.
    monkeypatch.setattr(models, "PLANNED", {"future_model": "T99"})
    with pytest.raises(ValueError, match="no model.name"):
        build_model({"name": None})
    with pytest.raises(NotImplementedError, match="T99"):
        build_model({"name": "future_model"})
    with pytest.raises(ValueError, match="unknown"):
        build_model({"name": "nope"})


def test_register_and_build(monkeypatch):
    monkeypatch.setattr(models, "REGISTRY", {})
    register("tiny")(lambda cfg: nn.Conv2d(3, NUM_CLASSES, 1))
    assert isinstance(build_model({"name": "tiny"}), nn.Conv2d)
    with pytest.raises(ValueError, match="twice"):
        register("tiny")(lambda cfg: nn.Identity())


class _RandomInit(nn.Module):
    """Randomly initialised layers, like a decoder and head on a pretrained encoder."""

    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Conv2d(3, NUM_CLASSES, 3)
        self.frozen = nn.Linear(2, 2)
        self.frozen.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


def test_seeded_model_is_reproducible(monkeypatch):
    from cloth_segment.train import seeded_model

    monkeypatch.setattr(models, "REGISTRY", {"rand": _RandomInit})
    cfg = {"seed": 0, "model": {"name": "rand"}}
    torch.rand(5)  # RNG state before the build must not matter
    a = seeded_model(cfg).state_dict()
    torch.rand(7)
    b = seeded_model(cfg).state_dict()
    c = seeded_model(cfg | {"seed": 1}).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a["conv.weight"], c["conv.weight"])


def test_run_seeds_the_model_and_records_params(tmp_path, monkeypatch):
    from cloth_segment import train

    monkeypatch.setattr(models, "REGISTRY", {"rand": _RandomInit})
    cfg = yaml.safe_load((paths.CONFIGS / "base.yaml").read_text())
    cfg["model"]["name"] = "rand"
    created = runs.create_run  # default runs_dir is the shared mount: redirect to tmp
    monkeypatch.setattr(train.runs, "load_config", lambda path: cfg)
    monkeypatch.setattr(train.runs, "find_finished", lambda h, runs_dir=None: None)
    monkeypatch.setattr(
        train.runs,
        "create_run",
        lambda c, slug: created(c, slug, tmp_path / "runs", git=("abc1234", False)),
    )
    monkeypatch.setattr(train, "load_split", lambda *a, **k: None)
    seen = {}

    def fake_fit(cfg, run_dir, device, *, model, train_set, val_set, attr_val_set=None):
        assert attr_val_set is None  # segmentation-only model
        seen["weight"] = model.conv.weight.detach().clone()
        return {"steps": 0}

    monkeypatch.setattr(train, "fit", fake_fit)
    run_dir = train.run(tmp_path / "x.yaml", device=torch.device("cpu"))
    meta = runs.read_meta(run_dir)
    model = _RandomInit(None)
    assert meta["status"] == "finished"
    assert meta["params"] == sum(p.numel() for p in model.parameters())
    assert meta["trainable_params"] == model.conv.weight.numel() + model.conv.bias.numel()
    torch.manual_seed(cfg["seed"])
    assert torch.equal(seen["weight"], _RandomInit(None).conv.weight)


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


# --- val scoring at original resolution, per dataset (eval-v2) ---------------------

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


def test_predict_confusions_keeps_datasets_apart(tmp_path):
    df, ir, mr = _write_fixture(tmp_path, SPECS)
    ds = SegDataset(df, size=64, train=False, image_root=ir, mask_root=mr)
    loader = evaluate.eval_loader(ds, batch_size=2, num_workers=0)  # batches mix datasets
    cms = evaluate.predict_confusions(_AlwaysClass(4), loader, torch.device("cpu"))
    assert list(cms) == ["lip", "cihp", "dfmm"]
    for name, cm in cms.items():
        assert cm[:, 4].tolist() == class_pixel_counts(df[df.dataset == name]).tolist()
    pooled = evaluate.predict_confusion(_AlwaysClass(4), loader, torch.device("cpu"))
    assert torch.equal(pooled, sum(cms.values()))


def _cm(diag: list[int], off: int = 0) -> torch.Tensor:
    """Confusion matrix with `diag` on the diagonal and `off` pixels of class 0 predicted as 1."""
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for i, v in enumerate(diag):
        cm[i, i] = v
    cm[0, 1] += off
    return cm


def test_val_scores_weighs_each_dataset_the_same():
    small = _cm([10, 10])  # perfect: mIoU 1
    big = _cm([1000, 1000], off=1000)  # IoU 0.5 and 0.5: mIoU 0.5
    val = evaluate.val_scores({"lip": small, "dfmm": big})
    assert val["datasets"]["lip"]["miou"] == 1.0
    assert val["datasets"]["dfmm"]["miou"] == 0.5
    assert val["dataset_mean_miou"] == 0.75
    # the pooled score follows the big dataset
    assert val["miou"] == val["pooled"]["miou"] < 0.6
    assert evaluate.selection_value(val, "dataset_mean_miou") == 0.75
    assert evaluate.selection_value(val, "miou") == val["miou"]


def test_val_scores_mean_undefined_if_a_dataset_has_no_labels():
    val = evaluate.val_scores({"lip": _cm([10, 10]), "cihp": _cm([])})
    assert val["datasets"]["cihp"]["miou"] is None
    assert val["dataset_mean_miou"] is None


def test_selection_value_rejects_unknown_rule():
    with pytest.raises(ValueError, match="eval.select"):
        evaluate.selection_value({"miou": 1.0}, "dice")


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
        "eval": {"split": "val", "every": 2, "select": "dataset_mean_miou"},
        "run": {"num_workers": 0, "log_every": 1},
    }


def _fit_sets(tmp_path):
    specs = [("lip", "train", 64, 64, True)] * 4 + [
        ("lip", "val", 64, 64, False),
        ("dfmm", "val", 96, 64, False),
    ]
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
    assert out["best_step"] == 2 and out["best_val_select"] == "dataset_mean_miou"
    by_ds = out["best_val_miou_by_dataset"]
    assert list(by_ds) == ["lip", "dfmm"]
    assert out["best_val_value"] == pytest.approx(sum(by_ds.values()) / 2)
    assert out["best_val_miou"] is not None  # pooled, kept for comparison with eval-v1
    assert out["eval_seconds"] >= 0 and out["train_seconds"] >= 0
    assert (run_dir / "ckpt" / "best.pt").exists()
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    val_rows = [r for r in rows if r.get("split") == "val"]
    assert [r.get("dataset") for r in val_rows if "dataset" in r] == ["lip", "dfmm", "pooled"]
    sel = [r for r in val_rows if "select" in r]
    assert sel == [
        {"step": 2, "split": "val", "select": "dataset_mean_miou", "value": out["best_val_value"]}
    ]


def test_fit_rejects_unknown_select_before_training(tmp_path):
    from cloth_segment.train import fit

    cfg = _fit_cfg(2)
    cfg["eval"]["select"] = "dice"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_set, val_set = _fit_sets(tmp_path)
    with pytest.raises(ValueError, match="eval.select"):
        fit(cfg, run_dir, torch.device("cpu"), model=_Tiny(), train_set=train_set, val_set=val_set)
    assert not (run_dir / "log.jsonl").exists()


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
