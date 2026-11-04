"""T4 loader and transforms on tiny synthetic files; no real data, no GPU."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cloth_segment.data.dataset import (
    ATTR_NAMES,
    ATTR_NUM_CLASSES,
    DATASET_NAMES,
    SegDataset,
    attribute_targets,
    class_pixel_counts,
    collate,
    make_sampler,
    sample_weights,
    select_rows,
)
from cloth_segment.data.labels import IGNORE, NUM_CLASSES
from cloth_segment.data.transforms import (
    AugmentConfig,
    eval_transform,
    normalize,
    train_transform,
)

FULL_AUG = AugmentConfig(scale=(0.5, 2.0), hflip=0.5, brightness=0.2, contrast=0.2, saturation=0.2)


def _pair(h, w, seed=0):
    rng = np.random.default_rng(seed)
    image = torch.from_numpy(rng.integers(0, 256, (3, h, w), dtype=np.uint8))
    mask = torch.from_numpy(rng.choice([0, 3, 7, IGNORE], (h, w)).astype(np.uint8))
    return image, mask


# --- transforms ---------------------------------------------------------------


@pytest.mark.parametrize("hw", [(64, 40), (300, 200), (900, 600), (100, 580)])
def test_train_transform_shapes_and_values(hw):
    image, mask = _pair(*hw)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        x, y = train_transform(image, mask, 64, FULL_AUG, g)
        assert x.shape == (3, 64, 64) and x.dtype == torch.float32
        assert y.shape == (64, 64) and y.dtype == torch.int64
        assert set(y.unique().tolist()) <= {0, 3, 7, IGNORE}


def test_train_transform_is_seeded():
    image, mask = _pair(120, 80)
    a = train_transform(image, mask, 64, FULL_AUG, torch.Generator().manual_seed(3))
    b = train_transform(image, mask, 64, FULL_AUG, torch.Generator().manual_seed(3))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_padding_is_zero_image_and_ignore_mask():
    # Long side 40 -> 32 at scale 0.5 of size 64: the 64x64 output must be mostly padding.
    image, mask = _pair(40, 20)
    mask[:] = 3
    x, y = train_transform(image, mask, 64, AugmentConfig(scale=(0.5, 0.5)))
    valid = y != IGNORE
    assert valid.sum() == 32 * 16
    assert (x[:, ~valid] == 0).all()
    assert (y[valid] == 3).all()


def test_hflip_flips_image_and_mask_together():
    image, mask = _pair(64, 64)
    no = AugmentConfig()
    yes = AugmentConfig(hflip=1.0)
    x0, y0 = train_transform(image, mask, 64, no)
    x1, y1 = train_transform(image, mask, 64, yes)
    assert torch.equal(y1, y0.flip(-1))
    assert torch.allclose(x1, x0.flip(-1))


def test_identity_train_transform_equals_eval_on_square_input():
    image, mask = _pair(64, 64)
    x, y = train_transform(image, mask, 64, AugmentConfig())
    xe, ye, valid = eval_transform(image, mask, 64)
    assert valid == (64, 64)
    assert torch.equal(x, xe) and torch.equal(y, ye)
    assert torch.equal(y, mask.long())


def test_eval_transform_pads_bottom_right():
    image, mask = _pair(100, 50)
    x, y, (vh, vw) = eval_transform(image, mask, 64)
    assert (vh, vw) == (64, 32)
    assert (y[:, vw:] == IGNORE).all() and (x[:, :, vw:] == 0).all()
    assert set(y[:vh, :vw].unique().tolist()) <= {0, 3, 7, IGNORE}


def test_normalize_imagenet():
    img = torch.full((3, 2, 2), 255, dtype=torch.uint8)
    out = normalize(img)
    assert torch.allclose(out[0], torch.tensor((1 - 0.485) / 0.229))


def test_augment_config_rejects_bad_values():
    with pytest.raises(ValueError):
        AugmentConfig.from_dict({"scael": [1, 2]})
    with pytest.raises(ValueError):
        AugmentConfig.from_dict({"scale": [2, 1]})
    with pytest.raises(ValueError):
        AugmentConfig.from_dict({"hflip": 2})
    cfg = AugmentConfig.from_dict({"scale": [0.5, 2], "hflip": 0.5})
    assert cfg.scale == (0.5, 2.0)


def test_base_config_augment_parses():
    import yaml

    from cloth_segment.paths import CONFIGS

    cfg = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    AugmentConfig.from_dict(cfg["augment"])
    assert cfg["input"]["size"] % 32 == 0
    assert cfg["data"]["sampling"] in ("natural", "balanced")


# --- attributes ---------------------------------------------------------------


def test_attribute_targets_mask_na_and_missing():
    sleeve_na = ATTR_NUM_CLASSES["sleeve"]
    codes = np.zeros((3, len(ATTR_NAMES)), dtype=int)
    codes[0, 0] = 2
    codes[1, 0] = sleeve_na
    codes[2, :] = -1
    t = attribute_targets(codes)
    assert t[0, 0] == 2
    assert t[1, 0] == IGNORE
    assert (t[2] == IGNORE).all()
    assert (t[0, 1:] == 0).all()


def test_attribute_targets_reject_out_of_range():
    codes = np.zeros((1, len(ATTR_NAMES)), dtype=int)
    codes[0, 0] = ATTR_NUM_CLASSES["sleeve"] + 1
    with pytest.raises(ValueError):
        attribute_targets(codes)


# --- dataset on a synthetic manifest ------------------------------------------


def _write_fixture(tmp_path, specs):
    """specs: (dataset, split, h, w, gray) -> manifest-like DataFrame with files on disk."""
    img_root, mask_root = tmp_path / "datasets", tmp_path / "data"
    rows = []
    for k, (ds, split, h, w, gray) in enumerate(specs):
        rng = np.random.default_rng(k)
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        pil = Image.fromarray(img).convert("L") if gray else Image.fromarray(img)
        mask = rng.choice([0, 1, 4, 8, IGNORE], (h, w)).astype(np.uint8)
        ip, mp = f"{ds}/{k}.jpg", f"masks/{ds}/{k}.png"
        (img_root / ds).mkdir(parents=True, exist_ok=True)
        (mask_root / "masks" / ds).mkdir(parents=True, exist_ok=True)
        pil.save(img_root / ip)
        Image.fromarray(mask, mode="L").save(mask_root / mp)
        row = {
            "dataset": ds,
            "id": str(k),
            "split": split,
            "exclude_reason": "",
            "image": ip,
            "mask": mp,
            "h": h,
            "w": w,
        }
        row.update({f"c{c}": int((mask == c).sum()) for c in range(NUM_CLASSES)})
        row.update({a: (1 if ds == "dfmm" else -1) for a in ATTR_NAMES})
        rows.append(row)
    return pd.DataFrame(rows), img_root, mask_root


SPECS = [
    ("lip", "train", 120, 70, False),
    ("lip", "train", 90, 64, True),  # grayscale JPEG
    ("cihp", "train", 100, 150, False),
    ("dfmm", "train", 220, 150, False),
    ("dfmm", "val", 220, 150, False),
    ("cihp", "test", 80, 120, False),
]


def test_dataset_train_batch(tmp_path):
    df, ir, mr = _write_fixture(tmp_path, SPECS)
    rows = select_rows(df, "train", DATASET_NAMES)
    assert len(rows) == 4
    ds = SegDataset(rows, size=64, train=True, augment=FULL_AUG, image_root=ir, mask_root=mr)
    batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=collate)))
    assert batch["image"].shape == (4, 3, 64, 64) and batch["image"].dtype == torch.float32
    assert batch["mask"].shape == (4, 64, 64) and batch["mask"].dtype == torch.int64
    assert batch["attrs"].shape == (4, len(ATTR_NAMES))
    assert batch["dataset"].tolist() == [0, 0, 1, 2]
    assert (batch["attrs"][:3] == IGNORE).all() and (batch["attrs"][3] == 1).all()
    assert batch["orig_hw"][1] == (90, 64)
    assert set(batch["mask"].unique().tolist()) <= {0, 1, 4, 8, IGNORE}


def test_dataset_eval_sample(tmp_path):
    df, ir, mr = _write_fixture(tmp_path, SPECS)
    ds = SegDataset(select_rows(df, "test", ["cihp"]), 64, train=False, image_root=ir, mask_root=mr)
    s = ds[0]
    assert s["valid_hw"] == (43, 64) and s["orig_hw"] == (80, 120)
    assert (s["mask"][43:] == IGNORE).all()


def test_dataset_rejects_bad_files(tmp_path):
    df, ir, mr = _write_fixture(tmp_path, SPECS[:1])
    Image.fromarray(np.full((120, 70), 12, np.uint8), mode="L").save(mr / df["mask"][0])
    ds = SegDataset(df, 64, train=False, image_root=ir, mask_root=mr)
    with pytest.raises(ValueError, match="unexpected mask values"):
        ds[0]
    Image.fromarray(np.zeros((10, 10), np.uint8), mode="L").save(mr / df["mask"][0])
    with pytest.raises(ValueError, match="!="):
        ds[0]


def test_dataset_rejects_bad_size(tmp_path):
    df, ir, mr = _write_fixture(tmp_path, SPECS[:1])
    with pytest.raises(ValueError):
        SegDataset(df, 50, train=False, image_root=ir, mask_root=mr)


def test_select_rows_checks():
    df = pd.DataFrame(
        {"dataset": ["lip", "lip"], "split": ["train", "train"], "exclude_reason": ["", "x"]}
    )
    with pytest.raises(ValueError, match="excluded"):
        select_rows(df, "train", ["lip"])
    with pytest.raises(ValueError):
        select_rows(df, "excluded", ["lip"])
    with pytest.raises(ValueError):
        select_rows(df, "train", ["atr"])


# --- sampling and class counts ------------------------------------------------


def test_sample_weights():
    ds = ["lip"] * 6 + ["cihp"] * 3 + ["dfmm"] * 1
    nat = sample_weights(ds, "natural")
    assert np.allclose(nat, 0.1)
    bal = sample_weights(ds, "balanced")
    totals = {d: bal[np.array(ds) == d].sum() for d in set(ds)}
    assert np.allclose(list(totals.values()), 1 / 3)
    with pytest.raises(ValueError):
        sample_weights(ds, "uniform")


def test_make_sampler(tmp_path):
    df, ir, mr = _write_fixture(tmp_path, SPECS)
    ds = SegDataset(select_rows(df, "train", DATASET_NAMES), 64, True, image_root=ir, mask_root=mr)
    assert make_sampler(ds, "natural") is None
    s = make_sampler(ds, "balanced", torch.Generator().manual_seed(0))
    assert len(list(s)) == len(ds)


def test_class_pixel_counts(tmp_path):
    df, _, _ = _write_fixture(tmp_path, SPECS[:2])
    counts = class_pixel_counts(df)
    assert counts.shape == (NUM_CLASSES,)
    assert counts.sum() <= 120 * 70 + 90 * 64  # IGNORE pixels are not counted
    assert counts[2] == 0
