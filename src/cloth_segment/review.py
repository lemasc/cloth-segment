"""Visual review of model results (M10, analysis only).

Reads the output of `scripts/raw_class_breakdown.py` (per-image pixel counts),
picks images to look at, and draws them for `notebooks/review/`. Nothing here
changes labels, splits, the eval protocol or any run.

The review config (`configs/review/<task>.yaml`) names the runs, the questions
and, per question, which raw classes to sample and which verdicts a reviewer
can give. Reviewer input is stored as a small CSV (`read_verdicts` /
`write_verdicts`) and committed with the report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cloth_segment.data.labels import IGNORE, NUM_CLASSES, SHARED_CLASSES

VERDICT_COLUMNS = ["question", "key", "verdict", "note"]

# One colour per shared class (RGB), fixed so every panel and legend agree.
PALETTE = np.array(
    [
        [0, 0, 0],  # background
        [140, 86, 75],  # hair
        [255, 187, 120],  # face
        [247, 182, 210],  # skin
        [31, 119, 180],  # upper
        [44, 160, 44],  # outer
        [214, 39, 40],  # dress
        [148, 103, 189],  # skirt
        [255, 127, 14],  # pants
        [23, 190, 207],  # footwear
        [188, 189, 34],  # headwear
        [227, 119, 194],  # accessory
    ],
    dtype=np.uint8,
)
IGNORE_COLOUR = np.array([128, 128, 128], dtype=np.uint8)
assert len(PALETTE) == NUM_CLASSES

# One colour per raw id (raw masks have at most 24 classes; id 0 is background
# in every dataset). Not related to PALETTE: the per-image legend names them.
RAW_PALETTE = np.array(
    [
        [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
        [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200],
        [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128],
        [128, 128, 128], [255, 255, 255], [100, 100, 255], [255, 100, 100],
    ],
    dtype=np.uint8,
)  # fmt: skip


def load_review_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("out_dir", "runs", "rank_run", "seed", "questions"):
        if key not in cfg:
            raise ValueError(f"{path}: missing key {key!r}")
    if cfg["rank_run"] not in cfg["runs"]:
        raise ValueError(f"{path}: rank_run {cfg['rank_run']!r} is not in runs")
    for name, q in cfg["questions"].items():
        for key in ("title", "datasets", "raw", "min_share", "n_worst", "n_random", "verdicts"):
            if key not in q:
                raise ValueError(f"{path}: question {name!r} misses {key!r}")
    return cfg


# ------------------------------------------------------------ per-image data


def per_image_recall(raw: pd.DataFrame, shared: pd.DataFrame, raw_names: list[str]) -> pd.DataFrame:
    """One row per image that contains any of `raw_names`.

    `raw` and `shared` are one run's `per_image_raw.csv` and
    `per_image_shared.csv` for one test set. Pixels of the listed raw classes
    are summed per image. `share` = their pixels / image area; `recall` =
    pixels predicted as the shared class they map to / their pixels.
    """
    sel = raw[raw["raw_name"].isin(raw_names)]
    if sel["hit_px"].isna().any():
        raise ValueError(f"{raw_names}: a raw class maps to IGNORE and has no recall")
    g = sel.groupby(["index", "id"], as_index=False)[["raw_px", "hit_px"]].sum()
    hw = shared[["index", "h", "w"]].drop_duplicates("index")
    g = g.merge(hw, on="index", how="left", validate="one_to_one")
    if g["h"].isna().any():
        raise ValueError("an image in the raw table is missing from the shared table")
    g["share"] = g["raw_px"] / (g["h"] * g["w"])
    g["recall"] = g["hit_px"] / g["raw_px"]
    return g


def pick_samples(
    df: pd.DataFrame, n_worst: int, n_random: int, min_share: float, seed: int
) -> pd.DataFrame:
    """`n_worst` lowest-recall images plus `n_random` others, among images with
    `share >= min_share`.

    Worst first breaks ties by larger area, then by index. The random ones are
    drawn from the remaining candidates, so the two groups never overlap.
    A `pick` column says which group a row came from.
    """
    cand = df[df["share"] >= min_share]
    worst = cand.sort_values(["recall", "raw_px", "index"], ascending=[True, False, True])
    worst = worst.head(n_worst)
    rest = cand.drop(worst.index)
    rnd = rest.sample(n=min(n_random, len(rest)), random_state=seed).sort_values("index")
    return pd.concat([worst.assign(pick="worst"), rnd.assign(pick="random")], ignore_index=True)


def socks_context(raw: pd.DataFrame, leg_share: float = 0.1) -> pd.DataFrame:
    """One row per image with raw `socks`: socks pixels and hits, visible leg
    skin, and whether a skirt or dress is in the image.

    `raw` is one run's `per_image_raw.csv` (LIP or CIHP raw names).
    `tights_like`: a skirt or dress is present and visible leg skin
    (`left_leg` + `right_leg`) is below `leg_share` × the socks pixels, i.e.
    legs that would show under a skirt are covered by what is labelled socks.
    A per-image proxy: in images with several people the skirt, legs and
    socks may belong to different people.
    """
    p = raw.pivot_table(
        index="index", columns="raw_name", values="raw_px", aggfunc="sum", fill_value=0
    )
    for col in ("socks", "left_leg", "right_leg", "skirt", "dress"):
        if col not in p:
            p[col] = 0
    p = p[p["socks"] > 0]
    hit = raw[raw["raw_name"] == "socks"].set_index("index")["hit_px"]
    out = pd.DataFrame(
        {
            "socks_px": p["socks"],
            "hit_px": hit.reindex(p.index),
            "leg_px": p["left_leg"] + p["right_leg"],
            "skirt_or_dress": (p["skirt"] + p["dress"]) > 0,
        }
    )
    out["tights_like"] = out["skirt_or_dress"] & (out["leg_px"] < leg_share * out["socks_px"])
    return out.reset_index()


def pred_shares(raw: np.ndarray, pred: np.ndarray, raw_ids: list[int]) -> dict[str, float]:
    """Share of the pixels of `raw_ids` (raw mask) predicted as each shared class.

    All NaN when the image has none of those pixels.
    """
    sel = np.isin(raw, raw_ids)
    n = int(sel.sum())
    counts = np.bincount(pred[sel].ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return {c: (counts[i] / n if n else np.nan) for i, c in enumerate(SHARED_CLASSES)}


def verdict_shares(shares: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Mean `pred_shares` per question, verdict, pick group and run.

    `shares` has one row per image and run (`question`, `key`, `pick`, `run`
    and one column per shared class); `verdicts` is the verdict table. Images
    without a verdict are left out. `images` counts the images in each row.

    The pick groups stay apart: "worst" images were chosen by the rank run's
    recall, so that run's shares on them are low by construction. Only the
    "random" group shows what is typical.
    """
    v = verdicts.loc[verdicts["verdict"] != "", ["question", "key", "verdict"]]
    m = shares.merge(v, on=["question", "key"], how="inner")
    keys = ["question", "verdict", "pick", "run"]
    out = m.groupby(keys)[list(SHARED_CLASSES)].mean()
    out.insert(0, "images", m.groupby(keys)["key"].nunique())
    return out.reset_index()


def split_count(n: int, parts: int) -> list[int]:
    """`n` spread over `parts` as evenly as possible, earlier parts first."""
    return [n // parts + (i < n % parts) for i in range(parts)]


def size_table(df: pd.DataFrame, edges: list[float], labels: list[str]) -> pd.DataFrame:
    """Images, pixels and pixel recall per area group of a `per_image_recall` table."""
    if len(labels) != len(edges) - 1:
        raise ValueError("need one label per group")
    groups = pd.cut(df["share"], edges, labels=labels)
    t = df.groupby(groups, observed=False).agg(
        images=("index", "size"), raw_px=("raw_px", "sum"), hit_px=("hit_px", "sum")
    )
    t["recall"] = t["hit_px"] / t["raw_px"].where(t["raw_px"] > 0)
    return t


# ------------------------------------------------------------ drawing


def colour_mask(mask: np.ndarray) -> np.ndarray:
    """(H, W) shared ids -> (H, W, 3) uint8 colours; IGNORE is grey."""
    out = np.empty((*mask.shape, 3), dtype=np.uint8)
    ign = mask == IGNORE
    out[~ign] = PALETTE[mask[~ign]]
    out[ign] = IGNORE_COLOUR
    return out


def overlay(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.55, raw: bool = False
) -> np.ndarray:
    """Blend coloured `mask` over `image` (H, W, 3); background (0) stays the photo.

    `raw=True` colours raw ids with RAW_PALETTE instead of shared ids.
    """
    col = (RAW_PALETTE[mask] if raw else colour_mask(mask)).astype(np.float32)
    img = image.astype(np.float32)
    a = np.where((mask != 0)[..., None], alpha, 0.0)
    return (img * (1 - a) + col * a).round().astype(np.uint8)


def outline(region: np.ndarray, width: int = 2) -> np.ndarray:
    """Boolean edge pixels of a boolean region, `width` pixels thick (inside the region)."""
    edge = np.zeros_like(region)
    inner = region.copy()
    for _ in range(width):
        shrunk = inner.copy()
        shrunk[1:, :] &= inner[:-1, :]
        shrunk[:-1, :] &= inner[1:, :]
        shrunk[:, 1:] &= inner[:, :-1]
        shrunk[:, :-1] &= inner[:, 1:]
        edge |= inner & ~shrunk
        inner = shrunk
    return edge


def crop_box(region: np.ndarray, margin: float = 0.6, min_side: int = 96) -> tuple[int, ...]:
    """(y0, y1, x0, x1) around a boolean region, grown by `margin` × its size on
    each side and to at least `min_side`, clipped to the image. Whole image if empty."""
    h, w = region.shape
    if not region.any():
        return 0, h, 0, w
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    hh = max((y1 - y0) * (1 + 2 * margin), min_side) / 2
    hw = max((x1 - x0) * (1 + 2 * margin), min_side) / 2
    return (
        int(max(0, np.floor(cy - hh))),
        int(min(h, np.ceil(cy + hh))),
        int(max(0, np.floor(cx - hw))),
        int(min(w, np.ceil(cx + hw))),
    )


def raw_legend(raw: np.ndarray, dataset: str) -> list[dict]:
    """Raw classes present in one raw mask, largest first: id, name, the shared
    class it maps to (None = ignored) and its share of the image."""
    from cloth_segment.data.labels import MAPPING
    from cloth_segment.data.sources import RAW_LABELS

    ids, counts = np.unique(raw, return_counts=True)
    out = []
    for rid, n in sorted(zip(ids.tolist(), counts.tolist(), strict=True), key=lambda t: -t[1]):
        if rid not in RAW_LABELS[dataset]:
            raise ValueError(f"{dataset}: raw id {rid} is not a known raw class")
        sid = MAPPING[dataset].get(rid, IGNORE)
        out.append(
            {
                "raw_id": rid,
                "raw": RAW_LABELS[dataset][rid],
                "shared": SHARED_CLASSES[sid] if sid != IGNORE else None,
                "share": n / raw.size,
            }
        )
    return out


def raw_legend_html(items: list[dict]) -> str:
    """Colour chips for `raw_legend` items: raw name → shared class (share)."""

    def chip(it):
        rgb = tuple(int(v) for v in RAW_PALETTE[it["raw_id"]])
        target = it["shared"] or "ignored"
        return (
            f'<span style="display:inline-block;margin:0 12px 4px 0">'
            f'<span style="display:inline-block;width:12px;height:12px;background:rgb{rgb};'
            f'border:1px solid #888;vertical-align:middle;margin-right:4px"></span>'
            f"{it['raw']} → <b>{target}</b> ({it['share']:.1%})</span>"
        )

    return "<div>" + "".join(chip(it) for it in items if it["raw_id"] != 0) + "</div>"


def legend_html() -> str:
    chips = "".join(
        f'<span style="display:inline-block;margin:0 10px 4px 0">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f"background:rgb{tuple(int(v) for v in PALETTE[i])};border:1px solid #888;"
        f'vertical-align:middle;margin-right:4px"></span>{name}</span>'
        for i, name in enumerate(SHARED_CLASSES)
        if i
    )
    return f"<div>{chips}<span>(background = photo, grey = ignored)</span></div>"


# ------------------------------------------------------------ reviewer input


def read_verdicts(path: Path) -> pd.DataFrame:
    """The verdict CSV, or an empty table if it does not exist yet."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=VERDICT_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != VERDICT_COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)}, expected {VERDICT_COLUMNS}")
    return df


def write_verdicts(path: Path, df: pd.DataFrame) -> bool:
    """Write sorted rows that have a verdict or a note; return True if the file changed."""
    df = df[VERDICT_COLUMNS].astype(str)
    df = df[(df["verdict"] != "") | (df["note"] != "")]
    df = df.sort_values(["question", "key"]).reset_index(drop=True)
    text = df.to_csv(index=False, lineterminator="\n")
    path = Path(path)
    if path.exists() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def tally(verdicts: pd.DataFrame, samples: pd.DataFrame, codes: dict[str, str]) -> pd.DataFrame:
    """Verdict counts for one question, overall and per `pick` group and dataset."""
    v = verdicts.merge(samples[["key", "pick", "dataset"]], on="key", how="inner")
    v = v[v["verdict"] != ""]
    rows = []
    for code, text in codes.items():
        hit = v[v["verdict"] == code]
        rows.append(
            {
                "verdict": code,
                "meaning": text,
                "all": len(hit),
                "worst": int((hit["pick"] == "worst").sum()),
                "random": int((hit["pick"] == "random").sum()),
                **{
                    f"{d}": int((hit["dataset"] == d).sum())
                    for d in sorted(samples["dataset"].unique())
                },
            }
        )
    return pd.DataFrame(rows)
