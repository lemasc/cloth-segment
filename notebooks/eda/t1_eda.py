import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image

    from cloth_segment.data.sources import (
        RAW_LABELS, SPLITS, dataset_root, dfmm_outfit_id, list_pairs,
    )
    from cloth_segment.paths import DATA

    DATASETS = list(SPLITS)
    return (DATA, DATASETS, Image, RAW_LABELS, SPLITS, dataset_root,
            dfmm_outfit_id, list_pairs, mo, np, pd, plt)


@app.cell
def _(mo):
    mo.md("""
    # T1 — EDA of LIP, CIHP, DFMM

    Inputs come from `scripts/eda_scan.py` (one row per image/mask pair,
    every file decoded). Questions to settle here (see the kickoff handoff):

    - **Integrity:** corrupt files, image/mask size mismatches, EXIF rotation, unexpected mask values.
    - **Label conflicts:** LIP 10 *jumpsuits* vs CIHP 10 *torso-skin*; **Q6** socks; **Q7** DFMM belt; bags.
    - **Distribution:** raw and draft-shared class frequencies per dataset.
    - **DFMM:** outfit groups and attribute coverage (for T3/T4/E1).
    """)
    return


@app.cell
def _(DATA, DATASETS, mo, pd):
    _missing = [d for d in DATASETS if not (DATA / "eda" / f"scan_{d}.csv").exists()]
    mo.stop(bool(_missing), mo.md(f"Missing scans for {_missing}. Run `uv run python scripts/eda_scan.py`."))
    scans = {d: pd.read_csv(DATA / "eda" / f"scan_{d}.csv", keep_default_na=False,
                            dtype={"id": str, "exif_orient": str}) for d in DATASETS}
    return (scans,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Integrity
    """)
    return


@app.cell
def _(pd, scans):
    def _integrity(df):
        v = [c for c in df.columns if c.startswith("v")]
        ok = df[df.error == ""]
        return pd.Series({
            "pairs": len(df),
            "errors": (df.error != "").sum(),
            "size_mismatch": ((ok.img_w != ok.mask_w) | (ok.img_h != ok.mask_h)).sum(),
            "exif_rotated": (~ok.exif_orient.isin(["", "1"])).sum(),
            "img_modes": ", ".join(f"{k}:{n}" for k, n in ok.img_mode.value_counts().items()),
            "mask_modes": ", ".join(f"{k}:{n}" for k, n in ok.mask_mode.value_counts().items()),
            "unexpected_values": (ok.unexpected != "").sum(),
            "all_background": (ok[v[1:]].sum(axis=1) == 0).sum(),
        })

    integrity = pd.DataFrame({
        f"{d}/{s}": _integrity(g) for d, df in scans.items() for s, g in df.groupby("split")
    }).T
    integrity
    return (integrity,)


@app.cell
def _(mo, pd, scans):
    _bad = pd.concat([
        df[(df.error != "") | (df.unexpected != "") | (df.img_w != df.mask_w) | (df.img_h != df.mask_h)]
        for df in scans.values()
    ])[["dataset", "split", "id", "img_w", "img_h", "mask_w", "mask_h", "error", "unexpected"]]
    mo.vstack([mo.md(f"**Problem pairs: {len(_bad)}**"), mo.ui.table(_bad, page_size=10)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Image sizes
    """)
    return


@app.cell
def _(pd, plt, scans):
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for _d, _df in scans.items():
        _axes[0].hist(_df.img_h, bins=60, alpha=0.6, label=_d)
        _axes[1].hist(_df.img_w, bins=60, alpha=0.6, label=_d)
        _axes[2].hist(_df.img_h / _df.img_w, bins=60, alpha=0.6, label=_d)
    for _ax, _t in zip(_axes, ["height (px)", "width (px)", "aspect H/W"]):
        _ax.set_xlabel(_t)
        _ax.legend()
    _fig.tight_layout()
    sizes = pd.DataFrame({d: df[["img_h", "img_w"]].describe().stack() for d, df in scans.items()}).T.round(0)
    _fig
    return (sizes,)


@app.cell
def _(sizes):
    sizes
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Raw class distribution

    *pixel %* = share of all labelled pixels; *image %* = share of images
    that contain the class at all.
    """)
    return


@app.cell
def _(RAW_LABELS, pd, scans):
    def raw_stats(d, split=None):
        df = scans[d] if split is None else scans[d][scans[d].split == split]
        df = df[df.error == ""]
        cols = [f"v{k}" for k in RAW_LABELS[d]]
        px = df[cols].sum()
        return pd.DataFrame({
            "name": list(RAW_LABELS[d].values()),
            "pixel %": (100 * px / px.sum()).values,
            "image %": (100 * (df[cols] > 0).mean()).values,
            "images": (df[cols] > 0).sum().values,
        }, index=pd.Index(list(RAW_LABELS[d]), name="id"))

    raw = {d: raw_stats(d) for d in scans}
    return (raw,)


@app.cell
def _(mo, raw):
    mo.ui.tabs({d: mo.ui.table(t.round(3).reset_index(), page_size=25, selection=None) for d, t in raw.items()})
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Shared label set

    The final mapping from `cloth_segment.data.labels` (T2), applied to the
    raw pixel counts.
    """)
    return


@app.cell
def _():
    from cloth_segment.data.labels import IGNORE, MAPPING, SHARED_CLASSES

    SHARED = list(SHARED_CLASSES)
    DRAFT_MAP = MAPPING
    return DRAFT_MAP, IGNORE, SHARED


@app.cell
def _(DRAFT_MAP, IGNORE, RAW_LABELS, SHARED, pd, plt, scans):
    for _d in scans:
        assert set(DRAFT_MAP[_d]) == set(RAW_LABELS[_d]), f"{_d}: draft map does not cover every raw id"

    def shared_px(d):
        df = scans[d][scans[d].error == ""]
        out = pd.Series(0.0, index=SHARED + ["ignore"])
        pres = {}
        for raw_id, s in DRAFT_MAP[d].items():
            name = "ignore" if s == IGNORE else SHARED[s]
            out[name] += df[f"v{raw_id}"].sum()
            pres[name] = pres.get(name, 0) + (df[f"v{raw_id}"] > 0)
        img = pd.Series({k: 100 * (v > 0).mean() for k, v in pres.items()})
        return 100 * out / out.sum(), img.reindex(out.index).fillna(0)

    _px, _img = {}, {}
    for _d in scans:
        _px[_d], _img[_d] = shared_px(_d)
    shared_pixel = pd.DataFrame(_px)
    shared_image = pd.DataFrame(_img)

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
    shared_pixel.plot.bar(ax=_axes[0], logy=True, title="pixel % (log)")
    shared_image.plot.bar(ax=_axes[1], title="image % containing class")
    _fig.tight_layout()
    _fig
    return shared_image, shared_pixel


@app.cell
def _(mo, shared_image, shared_pixel):
    mo.hstack([
        mo.vstack([mo.md("**pixel %**"), shared_pixel.round(3)]),
        mo.vstack([mo.md("**image %**"), shared_image.round(1)]),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Sample viewer

    Pick a dataset and a raw class. Images containing that class are shown
    with the class highlighted (red); use it for Q6 (socks), Q7 (belt),
    LIP jumpsuits, CIHP torso-skin, DFMM leggings/rompers/bag.
    """)
    return


@app.cell
def _(DATASETS, mo):
    view_ds = mo.ui.dropdown(DATASETS, value="dfmm", label="dataset")
    view_ds
    return (view_ds,)


@app.cell
def _(RAW_LABELS, mo, view_ds):
    view_cls = mo.ui.dropdown({f"{k} {n}": k for k, n in RAW_LABELS[view_ds.value].items()},
                              value=f"10 {RAW_LABELS[view_ds.value][10]}", label="raw class")
    view_seed = mo.ui.number(start=0, stop=10_000, value=0, label="seed")
    view_min = mo.ui.slider(0, 5, value=0, step=0.1, label="min class pixel % of image")
    mo.hstack([view_cls, view_seed, view_min], justify="start")
    return view_cls, view_min, view_seed


@app.cell
def _(Image, SPLITS, list_pairs, np, plt, scans, view_cls, view_ds, view_min, view_seed):
    _d, _k = view_ds.value, view_cls.value
    _df = scans[_d][scans[_d].error == ""]
    _vcols = [c for c in _df.columns if c.startswith("v")]
    _share = 100 * _df[f"v{_k}"] / _df[_vcols].sum(axis=1)
    _hits = _df[(_df[f"v{_k}"] > 0) & (_share >= view_min.value)]
    _pick = _hits.sample(min(8, len(_hits)), random_state=view_seed.value) if len(_hits) else _hits
    _pairs = {(p.split, p.id): p for s in SPLITS[_d] for p in list_pairs(_d, s)}
    _cmap = plt.get_cmap("tab20", 24)

    _n = max(len(_pick), 1)
    _fig, _axes = plt.subplots(2, _n, figsize=(2.2 * _n, 6), squeeze=False)
    for _i, (_, _r) in enumerate(_pick.iterrows()):
        _p = _pairs[(_r.split, _r.id)]
        _img = np.asarray(Image.open(_p.image).convert("RGB")) / 255
        _m = np.asarray(Image.open(_p.mask))
        _axes[0, _i].imshow(np.where((_m == _k)[..., None], 0.5 * _img + 0.5 * np.array([1, 0, 0]), _img))
        _axes[0, _i].set_title(f"{_r.id[:18]}\n{_share[_r.name]:.1f}%", fontsize=7)
        _axes[1, _i].imshow(_m, cmap=_cmap, vmin=0, vmax=23, interpolation="nearest")
    for _ax in _axes.ravel():
        _ax.axis("off")
    _fig.suptitle(f"{_d}: raw {_k} ({len(_hits)} images match)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. DFMM outfits and attributes
    """)
    return


@app.cell
def _(dfmm_outfit_id, pd, plt, scans):
    _ids = scans["dfmm"].id.map(dfmm_outfit_id)
    outfit_sizes = _ids.value_counts()
    _fig, _ax = plt.subplots(figsize=(6, 3))
    outfit_sizes.value_counts().sort_index().plot.bar(ax=_ax)
    _ax.set_xlabel("masked images per outfit")
    _ax.set_ylabel("outfits")
    _fig.tight_layout()
    _cats = scans["dfmm"].id.str.split("-id_").str[0].groupby(_ids).nunique()
    dfmm_groups = pd.Series({
        "masked images": len(_ids),
        "outfits": _ids.nunique(),
        "outfits with >1 masked image": (outfit_sizes > 1).sum(),
        "ids listed under >1 category": (_cats > 1).sum(),
    })
    _fig
    return (dfmm_groups,)


@app.cell
def _(dfmm_groups):
    dfmm_groups
    return


@app.cell
def _(dataset_root, pd):
    _root = dataset_root("dfmm") / "labels"
    SHAPE = {
        "sleeve": ["sleeveless", "short", "medium", "long", "not_long", "NA"],
        "lower_length": ["three_point", "medium_short", "three_quarter", "long", "NA"],
        "socks": ["no", "socks", "leggings", "NA"],
        "hat": ["no", "yes", "NA"],
        "glasses": ["no", "eyeglasses", "sunglasses", "in_hand_or_clothes", "NA"],
        "neckwear": ["no", "yes", "NA"],
        "wrist": ["no", "yes", "NA"],
        "ring": ["no", "yes", "NA"],
        "waist_acc": ["no", "belt", "clothing", "hidden", "NA"],
        "neckline": ["v", "square", "round", "standing", "lapel", "suspenders", "NA"],
        "outer_cardigan": ["yes", "no", "NA"],
        "covers_navel": ["no", "yes", "NA"],
    }
    _fab = ["denim", "cotton", "leather", "furry", "knitted", "chiffon", "other", "NA"]
    _pat = ["floral", "graphic", "striped", "pure", "lattice", "other", "color_block", "NA"]
    TEXTURE = {f"{kind}_{part}": names for kind, names in (("fabric", _fab), ("pattern", _pat))
               for part in ("upper", "lower", "outer")}

    def _read(path, cols):
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["file", *cols])
        df["id"] = df.file.str.removesuffix(".jpg")
        return df.drop(columns="file").set_index("id")

    attrs = _read(_root / "shape/shape_anno_all.txt", list(SHAPE)).join(
        _read(_root / "texture/fabric_ann.txt", [f"fabric_{p}" for p in ("upper", "lower", "outer")]), how="outer"
    ).join(_read(_root / "texture/pattern_ann.txt", [f"pattern_{p}" for p in ("upper", "lower", "outer")]), how="outer")
    ATTR_NAMES = {**SHAPE, **TEXTURE}
    return ATTR_NAMES, attrs


@app.cell
def _(attrs, pd, scans):
    _masked = set(scans["dfmm"].id)
    _a = attrs.index
    attr_coverage = pd.Series({
        "images with any attribute": len(_a),
        "masked images": len(_masked),
        "masked with shape labels": attrs.loc[attrs.index.isin(_masked), "sleeve"].notna().sum(),
        "masked with fabric labels": attrs.loc[attrs.index.isin(_masked), "fabric_upper"].notna().sum(),
        "masked without any attribute": len(_masked - set(_a)),
    })
    attr_coverage
    return


@app.cell
def _(ATTR_NAMES, attrs, mo, pd, scans):
    _m = attrs[attrs.index.isin(set(scans["dfmm"].id))]
    _rows = []
    for _col, _names in ATTR_NAMES.items():
        _vc = _m[_col].dropna().astype(int).value_counts()
        for _v, _name in enumerate(_names):
            _rows.append({"attribute": _col, "value": _v, "name": _name, "count": int(_vc.get(_v, 0)),
                          "%": round(100 * _vc.get(_v, 0) / max(_vc.sum(), 1), 1)})
    attr_dist = pd.DataFrame(_rows)
    _na = attr_dist[attr_dist.name == "NA"].set_index("attribute")["%"].rename("NA %")
    mo.vstack([
        mo.md("Attribute distribution on **masked** DFMM images (the E1 pool). NA = part not visible."),
        _na.to_frame().T,
        mo.ui.table(attr_dist, page_size=20, selection=None),
    ])
    return (attr_dist,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Findings

    Record conclusions in the T1 handoff (`docs/handoffs/`), not here.
    """)
    return


if __name__ == "__main__":
    app.run()
