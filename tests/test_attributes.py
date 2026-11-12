"""Attribute heads: spec, targets, loss, metrics, model and multi-task fit. No data, no GPU."""

import json

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from cloth_segment.lib import attributes, paths, runs
from cloth_segment.lib.data.dataset import ATTR_NAMES, SegDataset
from cloth_segment.lib.data.labels import IGNORE, NUM_CLASSES
from cloth_segment.lib.models import build_model

HEADS = {"heads": ["sleeve", "glasses", "hat"], "ignore_values": {"sleeve": ["not_long"]}}


def test_parse_spec_maps_value_names_to_codes():
    spec = attributes.parse_spec(HEADS | {"loss_weight": 0.3})
    assert spec.heads == ("sleeve", "glasses", "hat")
    assert spec.ignore == {"sleeve": (4,)}  # sleeveless, short, medium, long, not_long
    assert spec.num_classes == (5, 4, 2)  # NA dropped
    assert spec.columns == [ATTR_NAMES.index(h) for h in spec.heads]
    assert spec.loss_weight == 0.3 and spec.hidden == 256 and spec.dropout == 0.2


@pytest.mark.parametrize(
    "cfg",
    [
        {"heads": []},
        {"heads": ["sleeves"]},
        {"heads": ["hat", "hat"]},
        {"heads": ["hat"], "ignore_values": {"sleeve": ["not_long"]}},
        {"heads": ["hat"], "ignore_values": {"hat": ["NA"]}},
        {"heads": ["hat"], "loss_weight": 0},
        {"heads": ["hat"], "dropout": 1.0},
        {"heads": ["hat"], "hidden": 0},
    ],
)
def test_parse_spec_fails_loudly(cfg):
    with pytest.raises(ValueError):
        attributes.parse_spec(cfg)


def test_head_targets_selects_columns_and_ignores_values():
    spec = attributes.parse_spec(HEADS)
    codes = torch.full((3, len(ATTR_NAMES)), IGNORE, dtype=torch.int64)
    codes[0, ATTR_NAMES.index("sleeve")] = 4  # not_long -> IGNORE
    codes[1, ATTR_NAMES.index("sleeve")] = 2
    codes[1, ATTR_NAMES.index("hat")] = 1
    y = attributes.head_targets(codes, spec)
    assert y.tolist() == [[IGNORE] * 3, [2, IGNORE, 1], [IGNORE] * 3]
    assert codes[0, ATTR_NAMES.index("sleeve")] == 4  # input not changed


def test_attr_loss_without_labels_is_a_zero_tied_to_the_graph():
    logits = [torch.randn(2, 5, requires_grad=True), torch.randn(2, 2, requires_grad=True)]
    loss, n = attributes.attr_loss(logits, torch.full((2, 2), IGNORE))
    assert loss.item() == 0.0 and n == 0
    loss.backward()
    assert all(lg.grad is not None and not lg.grad.any() for lg in logits)


def test_attr_loss_averages_only_heads_with_labels():
    logits = [torch.randn(3, 5), torch.randn(3, 2)]
    y = torch.tensor([[1, IGNORE], [IGNORE, IGNORE], [3, IGNORE]])
    loss, n = attributes.attr_loss(logits, y)
    want = nn.functional.cross_entropy(logits[0][[0, 2]], torch.tensor([1, 3]))
    assert loss.item() == pytest.approx(want.item()) and n == 2


def test_head_metrics_by_hand():
    y = np.array([0, 0, 0, 1, 1, 2, IGNORE])
    p = np.array([0, 0, 1, 1, 0, 2, 2])
    m = attributes.head_metrics(y, p, 4, ["a", "b", "c", "d"], min_count=2)
    assert m["n"] == 6 and m["accuracy"] == pytest.approx(4 / 6)
    f1 = {k: v["f1"] for k, v in m["classes"].items()}
    assert f1 == pytest.approx({"a": 2 * 2 / (3 + 3), "b": 2 * 1 / (2 + 2), "c": 1.0, "d": 0.0})
    assert m["macro_f1"] == pytest.approx((f1["a"] + f1["b"]) / 2)  # c has 1 item, d none
    assert m["macro_f1_present"] == pytest.approx((f1["a"] + f1["b"] + 1.0) / 3)
    assert not m["classes"]["c"]["scored"] and m["classes"]["c"]["support"] == 1


def test_head_metrics_empty_is_undefined():
    m = attributes.head_metrics(np.array([IGNORE]), np.array([0]), 2, ["a", "b"], 1)
    assert m["accuracy"] is None and m["macro_f1"] is None and m["n"] == 0


def test_majority_baseline_and_all_metrics():
    spec = attributes.parse_spec({"heads": ["hat", "wrist"]})
    train = np.array([[0, 1], [0, 1], [1, IGNORE], [IGNORE, 0]])
    pred = attributes.majority_predictions(train, 3, spec)
    assert pred.tolist() == [[0, 1]] * 3
    y = np.array([[0, 1], [1, 0], [0, 1]])
    res = attributes.all_metrics(y, pred, spec, min_count=1)
    assert res["heads"]["hat"]["accuracy"] == pytest.approx(2 / 3)
    assert res["mean_accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        attributes.majority_predictions(np.full((2, 2), IGNORE), 1, spec)


# --- model ------------------------------------------------------------------------------

MT = {"name": "segformer_b2_mt", "pretrained": False, "attr": HEADS}


@pytest.fixture(scope="module")
def mt():
    torch.manual_seed(0)
    return build_model(MT).eval()


@pytest.mark.parametrize("shape", [(2, 3, 64, 96), (1, 3, 100, 75)])
def test_mt_forward_is_segmentation_only(mt, shape):
    x = torch.randn(shape)
    with torch.no_grad():
        seg = mt(x)
        seg2, attr = mt.forward_multi(x)
    assert seg.shape == (shape[0], NUM_CLASSES, *shape[2:])
    assert torch.allclose(seg, seg2)
    assert [a.shape for a in attr] == [(shape[0], k) for k in (5, 4, 2)]


def test_mt_segmentation_part_starts_like_segformer_b2(mt):
    torch.manual_seed(0)
    plain = build_model({"name": "segformer_b2", "pretrained": False}).state_dict()
    sd = mt.state_dict()
    assert all(torch.equal(plain[k], sd[k]) for k in plain)
    assert {k.split(".")[0] for k in sd if k not in plain} == {"attr_head"}


def test_mt_needs_attr_section():
    with pytest.raises(ValueError, match="model.attr"):
        build_model({"name": "segformer_b2_mt", "pretrained": False})


def test_multitask_and_reference_configs_differ_only_in_the_model():
    """The multi-task config and its plain reference: seed 1 both, same but the model."""
    base = yaml.safe_load((paths.CONFIGS / "base.yaml").read_text())
    mt = runs.load_config(paths.CONFIGS / "segformer_b2_mt.yaml")
    plain = runs.load_config(paths.CONFIGS / "segformer_b2_s1.yaml")
    assert runs.shared_overrides(base, mt) == runs.shared_overrides(base, plain) == ["seed"]
    assert mt["seed"] == plain["seed"] == 1
    assert {k: v for k, v in mt.items() if k != "model"} == {
        k: v for k, v in plain.items() if k != "model"
    }
    assert {k: v for k, v in mt["model"].items() if k != "attr"} == plain["model"] | {
        "name": "segformer_b2_mt"
    }
    assert len(attributes.parse_spec(mt["model"]["attr"]).heads) == 12
    seed0 = runs.load_config(paths.CONFIGS / "segformer_b2.yaml")
    assert runs.config_hash(plain) != runs.config_hash(seed0)  # the run cache does not skip it


def test_existing_run_hashes_unchanged():
    """Adding the multi-task model must not change the cache key of existing configs."""
    assert runs.config_hash(runs.load_config(paths.CONFIGS / "segformer_b2.yaml")) == "8e9039649376"
    assert runs.config_hash(runs.load_config(paths.CONFIGS / "unet_r34.yaml")) == "d4f03d515ed0"


# --- multi-task fit -----------------------------------------------------------------------


class _TinyMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec = attributes.parse_spec(HEADS | {"loss_weight": 0.5})
        self.conv = nn.Conv2d(3, NUM_CLASSES, 1)
        self.heads = nn.ModuleList(nn.Linear(3, k) for k in self.spec.num_classes)

    def forward(self, x):
        return self.conv(x)

    def forward_multi(self, x):
        pooled = x.mean(dim=(2, 3))
        return self.conv(x), [h(pooled) for h in self.heads]


def _sets(tmp_path):
    from test_dataset import _write_fixture

    specs = [("lip", "train", 64, 64, False), ("dfmm", "train", 64, 64, False)] * 2 + [
        ("lip", "val", 64, 64, False),
        ("dfmm", "val", 64, 64, False),
    ]
    df, ir, mr = _write_fixture(tmp_path, specs)
    kw = {"size": 64, "image_root": ir, "mask_root": mr}
    val = df[df.split == "val"]
    return (
        SegDataset(df[df.split == "train"], train=True, **kw),
        SegDataset(val, train=False, **kw),
        SegDataset(val[val.dataset == "dfmm"], train=False, **kw),
    )


def test_fit_multitask_logs_both_tasks(tmp_path):
    from test_train import _fit_cfg

    from cloth_segment.train import fit

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_set, val_set, attr_set = _sets(tmp_path)
    out = fit(
        _fit_cfg(2),
        run_dir,
        torch.device("cpu"),
        model=_TinyMT(),
        train_set=train_set,
        val_set=val_set,
        attr_val_set=attr_set,
    )
    assert out["best_val_select"] == "dataset_mean_miou"  # checkpoint rule unchanged
    assert "best_val_attr_mean_macro_f1" in out
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    train_rows = [r for r in rows if "loss" in r]
    assert all({"seg_loss", "attr_loss", "attr_images"} <= set(r) for r in train_rows)
    attr_rows = [r for r in rows if r.get("task") == "attr"]
    assert len(attr_rows) == 1 and set(attr_rows[0]["macro_f1"]) == set(HEADS["heads"])


def test_fit_multitask_needs_attr_val_set(tmp_path):
    from test_train import _fit_cfg

    from cloth_segment.train import fit

    train_set, val_set, _ = _sets(tmp_path)
    with pytest.raises(ValueError, match="attr_val_set"):
        fit(
            _fit_cfg(2),
            tmp_path,
            torch.device("cpu"),
            model=_TinyMT(),
            train_set=train_set,
            val_set=val_set,
        )


def test_predict_attrs_shapes(tmp_path):
    from cloth_segment import attr_eval, evaluate

    _, _, attr_set = _sets(tmp_path)
    loader = evaluate.eval_loader(attr_set, 2, 0)
    y, p = attr_eval.predict_attrs(_TinyMT(), loader, torch.device("cpu"))
    assert y.shape == p.shape == (1, 3)
    assert y.tolist() == [[1, 1, 1]]  # fixture codes are 1
