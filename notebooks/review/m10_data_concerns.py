import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import re
    from functools import lru_cache

    import marimo as mo
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw

    from cloth_segment.data.composition import merge_composition
    from cloth_segment.data.labels import SHARED_CLASSES
    from cloth_segment.data.sources import RAW_LABELS
    from cloth_segment.data.splits import load_manifest
    from cloth_segment.paths import CONFIGS, DATA, DATASETS, REPO_ROOT, RUNS
    from cloth_segment.rawclass import accessory_supply
    from cloth_segment.review import (
        crop_box,
        legend_html,
        load_review_config,
        outline,
        overlay,
        per_image_recall,
        pred_shares,
        raw_legend,
        raw_legend_html,
        read_verdicts,
        size_table,
        socks_context,
        tally,
        write_verdicts,
    )

    return (
        CONFIGS,
        DATA,
        DATASETS,
        Image,
        ImageDraw,
        RAW_LABELS,
        REPO_ROOT,
        RUNS,
        SHARED_CLASSES,
        accessory_supply,
        crop_box,
        json,
        legend_html,
        load_manifest,
        load_review_config,
        lru_cache,
        merge_composition,
        mo,
        np,
        outline,
        overlay,
        pd,
        per_image_recall,
        pred_shares,
        raw_legend,
        raw_legend_html,
        re,
        read_verdicts,
        size_table,
        socks_context,
        tally,
        write_verdicts,
    )


@app.cell
def _(CONFIGS, DATA, REPO_ROOT, load_review_config, pd, read_verdicts):
    cfg = load_review_config(CONFIGS / "review" / "m10.yaml")
    out_dir = DATA / cfg["out_dir"]
    breakdown_dir = DATA / cfg["breakdown_dir"]
    verdict_path = REPO_ROOT / cfg["verdicts"]
    samples = pd.read_csv(out_dir / "samples.csv", dtype={"id": str})
    saved = read_verdicts(verdict_path)
    saved_map = {(r.question, r.key): (r.verdict, r.note) for r in saved.itertuples()}
    runs = cfg["runs"]
    return breakdown_dir, cfg, out_dir, runs, samples, saved_map, verdict_path


@app.cell
def _(cfg, legend_html, mo, runs, samples, verdict_path):
    mo.md(f"""
    # M10 — Data concerns from model results

    All three comparison models are trained:
    {", ".join(f"**{a}** `{r}`" for a, r in runs.items())}.
    Some classes score low on some test sets. This notebook checks whether the
    reason is the data (labels, merged classes, object size) rather than the
    models. Answered questions show their numbers. Open questions show sample
    images: give each one a verdict, then tick the decisions at the end.

    **How to use.** Run with `uv run marimo edit notebooks/review/m10_data_concerns.py`
    on the team machine (images are on the shared mount). Every verdict, note
    and tick is saved at once to `{cfg["verdicts"]}`; commit that file. Nothing
    else is written.

    **Pictures.** Per image, two rows. Top: the photo and the ground truth;
    when the class is small, a zoomed view plus the full photo (yellow box =
    zoomed area). Bottom: each model's prediction of the same area. The white
    outline marks the raw class being asked about. Use the sliders above the
    tabs for panel size and mask opacity; the header says the native size and scale.
    {len(samples)} samples, picked by `scripts/review_predictions.py`
    (`configs/review/m10.yaml`): per question, the images with the lowest
    {cfg["rank_run"]} recall ("worst") plus random ones ("random"), so label
    errors are not over-counted. Verdicts file: `{verdict_path.name}`.

    {legend_html()}
    """)
    return


@app.cell
def _(RUNS, json, mo, pd, runs):
    _rows = []
    for _alias, _run in runs.items():
        for _ds in ("lip", "cihp", "dfmm"):
            _m = json.loads((RUNS / _run / "eval" / "eval-v2" / _ds / "metrics.json").read_text())
            for _c in ("dress", "skirt", "footwear", "outer", "accessory"):
                _rows.append({"class": _c, "test set": _ds, "run": _alias, "IoU": _m["iou"][_c]})
    _t = pd.DataFrame(_rows).pivot_table(
        index="class", columns=["test set", "run"], values="IoU", sort=False
    )
    _t.columns = [f"{d} {r}" for d, r in _t.columns]
    mo.vstack(
        [
            mo.md("""
    ## 1. Questions

    The model reports (T6, X3, X4) did not discuss these. Test IoU of the
    classes involved (`eval/eval-v2/<test set>/metrics.json`):
    """),
            mo.ui.table(_t.round(3).reset_index(), selection=None, page_size=10),
            mo.md("""
    | # | Question | Why it was raised | Status |
    |---|---|---|---|
    | Q1 | Is LIP `dress` weak because LIP jumpsuits are mapped to `dress`, while CIHP labels jumpsuits as upper + pants? | LIP `dress` far below CIHP and DFMM; LIP dress pixels go to `upper` and `pants` | answered (section 2); visual check in section 3 |
    | Q2 | Is CIHP `footwear` weak because feet are small, and do socks pull it down? | CIHP `footwear` lowest of the three; missed pixels go to background, skin, pants | answered for shoes; socks open (section 3) |
    | Q3 | Is `skirt` vs `pants` confusion partly shorts? LIP/CIHP have no shorts class. | `skirt` lowest garment class on LIP/CIHP; 17–28% of its pixels predicted `pants` | open (section 3) |
    | Q4 | Is `upper` ↔ `outer` confusion label ambiguity? | 10–18% of pixels go each way on LIP/CIHP in all models | open (section 3) |
    | Q5 | Are X3's lower DFMM scores on fine classes real, or run-to-run noise? | X3 below X2 on hair, skin, footwear, headwear on DFMM only | answered (section 2) |
    | Q6 | Can DFMM `accessory` IoU support any claim? | 0.0007 / 0.41 / 0.61 for X2 / X3 / X4 | answered (section 2) |
    """),
        ]
    )
    return


@app.cell
def _(breakdown_dir, json, mo, pd, runs):
    def raw_recall(pairs):
        rows = []
        for _ds, _raw in pairs:
            _row = {"test set": _ds, "raw class": _raw}
            for _alias, _run in runs.items():
                _j = json.loads((breakdown_dir / _run / _ds / "raw_confusion.json").read_text())
                _r = _j["raw_classes"][_raw]
                _row["pixels"] = _r["raw_px"]
                _row[f"recall {_alias}"] = round(_r["recall"], 3)
                _top = sorted(
                    ((c, v) for c, v in _r["where"].items() if c != _r["shared"]),
                    key=lambda t: -t[1],
                )[:2]
                _row[f"rest {_alias}"] = ", ".join(f"{c} {v / _r['raw_px']:.2f}" for c, v in _top)
            rows.append(_row)
        return pd.DataFrame(rows)

    _q1 = raw_recall(
        [
            ("lip", "dress"),
            ("lip", "jumpsuits"),
            ("cihp", "dress"),
            ("dfmm", "dress"),
            ("dfmm", "rompers"),
        ]
    )
    _q2 = raw_recall(
        [
            ("lip", "left_shoe"),
            ("lip", "socks"),
            ("cihp", "left_shoe"),
            ("cihp", "right_shoe"),
            ("cihp", "socks"),
            ("dfmm", "footwear"),
            ("dfmm", "socks"),
            ("dfmm", "leggings"),
        ]
    )
    mo.vstack(
        [
            mo.md("""
    ## 2. Answered

    Recall of a raw class = share of its pixels predicted as the shared class
    it maps to. "rest" = where most of the other pixels went. From
    `scripts/raw_class_breakdown.py` (`data/review/M10/breakdown/`).

    ### Q1: LIP jumpsuits are a real part of the weak LIP `dress`, but not all of it
    Jumpsuits are about a third of LIP test `dress` pixels and are missed far
    more than real dresses; the missed pixels go to upper + pants, the way CIHP
    labels jumpsuits. DFMM rompers are easy, so one-piece garments are not hard
    in general. Real LIP dresses are still below CIHP dresses.
    """),
            mo.ui.table(_q1, selection=None),
            mo.md("""
    ### Q2: CIHP shoes are missed mostly because they are small; socks are a separate problem
    """),
            mo.ui.table(_q2, selection=None),
        ]
    )
    return


@app.cell
def _(breakdown_dir, mo, pd, per_image_recall, runs, size_table):
    _edges = [0, 0.004, 0.008, 0.016, 0.032, 1]
    _labels = ["<0.4%", "0.4-0.8%", "0.8-1.6%", "1.6-3.2%", ">3.2%"]

    def _sizes(raw_names, datasets):
        rows = []
        for _ds in datasets:
            for _alias, _run in runs.items():
                _d = breakdown_dir / _run / _ds
                _df = per_image_recall(
                    pd.read_csv(_d / "per_image_raw.csv"),
                    pd.read_csv(_d / "per_image_shared.csv"),
                    raw_names,
                )
                _t = size_table(_df, _edges, _labels)
                rows.append(
                    {"test set": _ds, "run": _alias, "median area": f"{_df['share'].median():.2%}"}
                    | {
                        f"{k} (n={n})": round(r, 2)
                        for k, n, r in zip(_labels, _t["images"], _t["recall"])
                    }
                )
        return pd.DataFrame(rows)

    _shoes = pd.concat(
        [_sizes(["left_shoe", "right_shoe"], ["lip", "cihp"]), _sizes(["footwear"], ["dfmm"])]
    )
    _socks = _sizes(["socks"], ["lip", "cihp"])
    mo.vstack(
        [
            mo.md("""
    Recall by size. Area = the class's pixels in an image / image area; images
    grouped by that area. Shoes: recall rises with size everywhere, and CIHP
    shoes are smaller. Socks: on CIHP, recall does not rise with size, so size
    does not explain them (section 3).
    """),
            mo.md("**Shoes**"),
            mo.ui.table(_shoes, selection=None),
            mo.md("**Socks**"),
            mo.ui.table(_socks, selection=None),
        ]
    )
    return


@app.cell
def _(breakdown_dir, mo, pd, runs, socks_context):
    _rows = []
    for _ds in ("lip", "cihp"):
        for _alias, _run in runs.items():
            _c = socks_context(
                pd.read_csv(breakdown_dir / _run / _ds / "per_image_raw.csv", dtype={"id": str})
            )
            _t = _c["tights_like"]

            def _rec(m, c=_c):
                return round(c["hit_px"][m].sum() / c["socks_px"][m].sum(), 2)

            _rows.append(
                {
                    "test set": _ds,
                    "run": _alias,
                    "socks images": len(_c),
                    "tights-like images": int(_t.sum()),
                    "tights-like share of socks pixels": round(
                        _c["socks_px"][_t].sum() / _c["socks_px"].sum(), 2
                    ),
                    "recall tights-like": _rec(_t),
                    "recall rest": _rec(~_t),
                }
            )
    mo.vstack(
        [
            mo.md("""
    **Socks: are they tights?** "Tights-like" = a skirt or dress is in the
    image and visible leg skin is under 10% of the socks pixels, so the legs
    that a skirt would show are covered by what is labelled socks. A per-image
    proxy (`socks_context`): in CIHP images with several people, the skirt and
    the socks may belong to different people. The review tab checks it by eye.
    """),
            mo.ui.table(pd.DataFrame(_rows), selection=None),
        ]
    )
    return


@app.cell
def _(RUNS, accessory_supply, json, load_manifest, mo, pd, runs):
    _x1 = "20261105-0323-7b1798b-unet-r34"  # same config as X2: run-to-run noise
    _rows = []

    def _iou(run, version):
        return json.loads((RUNS / run / "eval" / version / "dfmm" / "metrics.json").read_text())[
            "iou"
        ]

    _a, _b, _c = _iou(_x1, "eval-v1"), _iou(runs["X2"], "eval-v2"), _iou(runs["X3"], "eval-v2")
    for _k in _a:
        _rows.append(
            {
                "class": _k,
                "X2 − X1 (noise)": round(_b[_k] - _a[_k], 4),
                "X3 − X2": round(_c[_k] - _b[_k], 4),
            }
        )
    _man, _ = load_manifest("splits-v2")
    _acc = accessory_supply(_man)
    mo.vstack(
        [
            mo.md(f"""
    ### Q5: the X3 drops on DFMM are larger than U-Net run-to-run noise
    X1 (`{_x1}`, `eval-v1`) and X2 have the same config and the same
    checkpoint step; test scoring is the same in `eval-v1` and `eval-v2` (M3).
    Caveat: the noise comes from one pair of U-Net runs.
    """),
            mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=12),
            mo.md("""
    ### Q6: no — DFMM `accessory` is too rare to compare models on
    """),
            mo.ui.table(_acc.round(3), selection=None),
        ]
    )
    return


@app.cell
def _(DATA, merge_composition, mo, pd):
    # Same tables as the T1 EDA notebook, section 4b (M5), on the train split:
    # what each shared class was built from when the models learned it.
    _classes = ["dress", "skirt", "pants", "upper", "outer", "footwear", "skin"]
    _m = pd.read_csv(
        DATA / "splits" / "splits-v2" / "manifest.csv",
        usecols=["dataset", "id", "split"],
        dtype={"id": str},
        keep_default_na=False,
    )
    _parts = []
    for _d in ("lip", "cihp", "dfmm"):
        _scan = pd.read_csv(
            DATA / "eda" / f"scan_{_d}.csv", dtype={"id": str}, keep_default_na=False
        )
        _keep = set(_m[(_m.dataset == _d) & (_m.split == "train")]["id"])
        _scan = _scan[(_scan.error == "") & _scan["id"].isin(_keep)]
        _parts.append(merge_composition(_scan, _d))
    _c = pd.concat(_parts)
    _c = _c[_c["shared"].isin(_classes) & (_c["pixels"] > 0)]
    _c["cell"] = _c.apply(lambda r: f"{r.pct_of_shared:.1f}% ({r.images:,} img)", axis=1)
    _t = _c.pivot_table(
        index=["shared", "raw"], columns="dataset", values="cell", aggfunc="first", fill_value="—"
    )
    _order = {c: i for i, c in enumerate(_classes)}
    _t = _t.reset_index().sort_values("shared", key=lambda s: s.map(_order), kind="stable")
    _t = _t[["shared", "raw", "lip", "cihp", "dfmm"]]
    mo.vstack(
        [
            mo.md("""
    ## Label merges (train split)

    Each shared class is a merge of raw (original dataset) classes. Per raw
    class and dataset: its share of that shared class's train pixels, and how
    many train images contain it. "—" = the dataset has no such raw class, so
    those pixels carry another label there (e.g. CIHP has no jumpsuits). Same
    numbers as section 4b of `notebooks/eda/t1_eda.py` (M5), on the train split.
    The raw-labels panel of each review image shows these raw classes.
    """),
            mo.ui.table(_t, selection=None, page_size=40),
        ]
    )
    return


@app.cell
def _(
    DATA,
    DATASETS,
    Image,
    ImageDraw,
    RAW_LABELS,
    crop_box,
    lru_cache,
    np,
    out_dir,
    outline,
    overlay,
    raw_legend,
    raw_legend_html,
    runs,
):
    ZOOM_MARGIN = 0.25  # crop grows by this share of the region's size on each side
    ZOOM_MAX_AREA = 0.6  # no zoom when the crop would cover more of the image than this
    GAP = 6

    def _fit(arr, h, resample=Image.LANCZOS):
        im = Image.fromarray(arr)
        return im.resize((max(1, round(im.width * h / im.height)), h), resample)

    def _label(im, text):
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 8 * len(text) + 8, 16), fill=(0, 0, 0))
        d.text((4, 2), text, fill=(255, 255, 255))
        return im

    def _row(panels, h):
        canvas = Image.new(
            "RGB", (sum(p.width for p in panels) + GAP * (len(panels) - 1), h), "white"
        )
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width + GAP
        return canvas

    @lru_cache(maxsize=512)
    def render(dataset, index, image, mask, raw_mask, raw_names, panel_h, alpha):
        """Two rows: [context photo if zoomed] photo, ground truth, raw labels /
        one panel per run.

        Panels are resized to `panel_h` (photos with Lanczos, masks blended
        at native size first); the outline is drawn after resizing so it stays
        2 px wide. Returns (image, info text, legend HTML of the raw classes
        in the shown area).
        """
        photo = np.asarray(Image.open(DATASETS / image).convert("RGB"))
        gt = np.asarray(Image.open(DATA / mask))
        raw = np.asarray(Image.open(DATASETS / raw_mask))
        h, w = gt.shape
        ids = [r for r, n in RAW_LABELS[dataset].items() if n in raw_names]
        region = np.isin(raw, ids)
        y0, y1, x0, x1 = crop_box(region, margin=ZOOM_MARGIN)
        zoomed = (y1 - y0) * (x1 - x0) <= ZOOM_MAX_AREA * h * w
        if not zoomed:
            y0, y1, x0, x1 = 0, h, 0, w
        crop_region = Image.fromarray(region[y0:y1, x0:x1].astype(np.uint8) * 255)
        cw = max(1, round((x1 - x0) * panel_h / (y1 - y0)))
        edge = outline(np.asarray(crop_region.resize((cw, panel_h), Image.NEAREST)) > 0, width=2)

        def _panel(m, title, raw_colours=False):
            arr = np.asarray(
                _fit(overlay(photo[y0:y1, x0:x1], m[y0:y1, x0:x1], alpha, raw_colours), panel_h)
            ).copy()
            arr[edge] = 255
            return _label(Image.fromarray(arr), title)

        top = []
        if zoomed:
            ctx = _fit(photo, panel_h)
            s = panel_h / h
            ImageDraw.Draw(ctx).rectangle(
                (x0 * s, y0 * s, x1 * s, y1 * s), outline=(255, 220, 0), width=3
            )
            top.append(_label(ctx, "full photo"))
        top += [_label(_fit(photo[y0:y1, x0:x1], panel_h), "zoom" if zoomed else "photo")]
        top.append(_panel(gt, "ground truth"))
        top.append(_panel(raw, f"raw labels ({dataset})", raw_colours=True))
        bottom = []
        for alias in runs:
            pred = np.asarray(Image.open(out_dir / "pred" / alias / dataset / f"{index}.png"))
            bottom.append(_panel(pred, alias))
        rows = [_row(top, panel_h), _row(bottom, panel_h)]
        canvas = Image.new("RGB", (max(r.width for r in rows), 2 * panel_h + GAP), "white")
        canvas.paste(rows[0], (0, 0))
        canvas.paste(rows[1], (0, panel_h + GAP))
        scale = panel_h / (y1 - y0)
        info = (
            f"native {w}×{h}"
            + (f", zoom {x1 - x0}×{y1 - y0}" if zoomed else "")
            + f", shown ×{scale:.2f}"
        )
        legend = raw_legend_html(raw_legend(raw[y0:y1, x0:x1], dataset))
        return canvas, info, legend

    return (render,)


@app.cell
def _(mo):
    panel_size = mo.ui.slider(
        260, 800, step=20, value=480, label="Panel height (px)", show_value=True
    )
    mask_alpha = mo.ui.slider(
        0.1, 0.9, step=0.05, value=0.45, label="Mask opacity", show_value=True
    )
    return mask_alpha, panel_size


@app.cell
def _(cfg, mo, saved_map):
    def _decision_ui(d, spec):
        boxes = {
            k: mo.ui.checkbox(label=text, value=saved_map.get((d, k), ("", ""))[0] == "x")
            for k, text in spec["options"].items()
        }
        note = saved_map.get((d, "note"), ("", ""))[1]
        return mo.ui.dictionary(
            boxes
            | {"note": mo.ui.text_area(value=note, placeholder="note (optional)", full_width=True)}
        )

    decision_ui = mo.ui.dictionary({d: _decision_ui(d, s) for d, s in cfg["decisions"].items()})
    return (decision_ui,)


@app.cell
def _(cfg, mask_alpha, mo, panel_size, re, read_verdicts, render, samples, verdict_path):
    # The verdict widgets are created here, in the cell that shows them: marimo
    # does not rerun the cell that defines a widget when its value changes, so
    # picking a verdict does not redraw the images. When a slider reruns this
    # cell, the widgets start from the verdicts file, which every change has
    # already been saved to.
    _saved = {
        (r.question, r.key): (r.verdict, r.note) for r in read_verdicts(verdict_path).itertuples()
    }

    def _verdict_ui(q, key, codes):
        v, note = _saved.get((q, key), ("", ""))
        options = {"(no verdict)": ""} | {f"{c}: {t}": c for c, t in codes.items()}
        label = next((lab for lab, c in options.items() if c == v), "(no verdict)")
        return mo.ui.dictionary(
            {
                "verdict": mo.ui.dropdown(options=options, value=label),
                "note": mo.ui.text(value=note, placeholder="note (optional)", full_width=True),
            }
        )

    review_ui = mo.ui.dictionary(
        {
            q: mo.ui.dictionary(
                {
                    r.key: _verdict_ui(q, r.key, spec["verdicts"])
                    for r in samples[samples["question"] == q].itertuples()
                }
            )
            for q, spec in cfg["questions"].items()
        }
    )

    def _image(q, r):
        img, info, legend = render(
            r.dataset,
            r.index,
            r.image,
            r.mask,
            r.raw_mask,
            tuple(cfg["questions"][q]["raw"]),
            panel_size.value,
            mask_alpha.value,
        )
        # Fit the window; a click opens the full-size image in a new tab
        # (marimo has no built-in zoom). In the editor the src is a ./@file/ URL.
        html = mo.image(
            img,
            style={
                "max-height": "90vh",
                "max-width": "100%",
                "height": "auto",
                "cursor": "zoom-in",
            },
        ).text
        src = re.search(r"src='([^']+)'", html).group(1)
        link = mo.Html(f'<a href="{src}" target="_blank" rel="noopener">{html}</a>')
        return mo.vstack(
            [
                mo.md(f"{info} · click the image to open it full size"),
                link,
                mo.Html(f"<small>Raw labels in view: {legend}</small>"),
            ]
        )

    def _sample_view(q, r):
        recall = " / ".join(
            f"{c[7:]} {getattr(r, c):.2f}" for c in samples.columns if c.startswith("recall_")
        )
        head = mo.md(
            f"**{r.key}** · id `{r.id}` · picked: {r.pick} · area {r.share:.2%} · recall {recall}"
        )
        ui = review_ui[q][r.key]
        # drawn only when scrolled into view
        img = mo.lazy(lambda: _image(q, r), show_loading_indicator=True)
        return mo.vstack([head, img, mo.hstack([ui["verdict"], ui["note"]], widths=[1, 2])])

    _tabs = {}
    for _q, _spec in cfg["questions"].items():
        _s = samples[samples["question"] == _q]
        _codes = "\n".join(f"- `{c}`: {t}" for c, t in _spec["verdicts"].items())
        _tabs[_spec["title"]] = mo.vstack(
            [
                mo.md(
                    f"Raw class: {', '.join(_spec['raw'])} · test sets: {', '.join(_spec['datasets'])}"
                    f" · {len(_s)} images\n\n{_codes}"
                )
            ]
            + [_sample_view(_q, r) for r in _s.itertuples()]
        )
    mo.vstack(
        [
            mo.md("## 3. Visual review"),
            mo.hstack([panel_size, mask_alpha], justify="start"),
            mo.ui.tabs(_tabs, lazy=True),
        ]
    )
    return (review_ui,)


@app.cell
def _(cfg, decision_ui, mo, pd, review_ui, samples, tally, verdict_path, write_verdicts):
    _rows = [
        {"question": q, "key": k, "verdict": v["verdict"], "note": v["note"]}
        for q, items in review_ui.value.items()
        for k, v in items.items()
    ]
    for _d, _v in decision_ui.value.items():
        _rows += [
            {"question": _d, "key": k, "verdict": "x" if on else "", "note": ""}
            for k, on in _v.items()
            if k != "note"
        ]
        _rows.append({"question": _d, "key": "note", "verdict": "", "note": _v["note"]})
    _df = pd.DataFrame(_rows, columns=["question", "key", "verdict", "note"])
    _any = ((_df["verdict"] != "") | (_df["note"] != "")).any()
    _changed = write_verdicts(verdict_path, _df) if _any or verdict_path.exists() else False

    _parts = [mo.md("## 4. Tally")]
    for _q, _spec in cfg["questions"].items():
        _done = (_df[_df["question"] == _q]["verdict"] != "").sum()
        _n = (samples["question"] == _q).sum()
        _parts += [
            mo.md(f"**{_spec['title']}**: {_done} of {_n} reviewed"),
            mo.ui.table(
                tally(
                    _df[_df["question"] == _q],
                    samples[samples["question"] == _q],
                    _spec["verdicts"],
                ),
                selection=None,
            ),
        ]
    _parts.append(mo.md(f"_{'Saved' if _changed else 'No change'}: `{verdict_path.name}`_"))
    verdict_table = _df
    mo.vstack(_parts)
    return (verdict_table,)


@app.cell
def _(DATASETS, Image, RAW_LABELS, cfg, np, out_dir, pd, pred_shares, samples):
    # Where each run puts the pixels of the question's raw class, per image.
    # Computed once; independent of the verdicts.
    _rows = []
    for _r in samples.itertuples():
        _names = cfg["questions"][_r.question]["raw"]
        _ids = [i for i, n in RAW_LABELS[_r.dataset].items() if n in _names]
        _raw = np.asarray(Image.open(DATASETS / _r.raw_mask))
        for _alias in cfg["runs"]:
            _pred = np.asarray(
                Image.open(out_dir / "pred" / _alias / _r.dataset / f"{_r.index}.png")
            )
            _rows.append(
                {"question": _r.question, "key": _r.key, "pick": _r.pick, "run": _alias}
                | pred_shares(_raw, _pred, _ids)
            )
    target_shares = pd.DataFrame(_rows)
    return (target_shares,)


@app.cell
def _(SHARED_CLASSES, cfg, mo, target_shares, verdict_table):
    _v = verdict_table[verdict_table["verdict"] != ""][["question", "key", "verdict"]]
    _m = target_shares.merge(_v, on=["question", "key"])
    _cls = list(SHARED_CLASSES)
    _parts = [
        mo.md(
            "### Where the target pixels go, per verdict\n\n"
            "Mean over images of the share of the question's raw-class pixels that "
            "each run predicts as each shared class. Classes below 0.04 everywhere are hidden."
        )
    ]
    for _q, _spec in cfg["questions"].items():
        _g = _m[_m["question"] == _q]
        if _g.empty:
            continue
        _t = _g.groupby(["verdict", "run"])[_cls].mean().round(2)
        _t = _t.loc[:, (_t > 0.04).any()].reset_index()
        _parts += [mo.md(f"**{_spec['title']}**"), mo.ui.table(_t, selection=None)]
    mo.vstack(_parts)
    return


@app.cell
def _(cfg, decision_ui, mo):
    mo.vstack(
        [
            mo.md("""
    ## 5. Decisions

    Tick what should happen for each finding (more than one may apply). A
    frozen-change touches the frozen label set (CONTRIBUTING section 4) and
    means retraining X2–X4.
    """)
        ]
        + [
            mo.vstack(
                [mo.md(f"**{spec['title']}**")]
                + [decision_ui[d][k] for k in [*spec["options"], "note"]]
            )
            for d, spec in cfg["decisions"].items()
        ]
    )
    return


if __name__ == "__main__":
    app.run()
