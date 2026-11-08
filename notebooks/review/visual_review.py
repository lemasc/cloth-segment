import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    from functools import lru_cache
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw

    from cloth_segment.data.labels import SHARED_CLASSES
    from cloth_segment.data.sources import RAW_LABELS
    from cloth_segment.paths import DATA, DATASETS, REPO_ROOT
    from cloth_segment.review import (
        crop_box,
        legend_html,
        load_review_config,
        outline,
        overlay,
        pred_shares,
        raw_legend,
        raw_legend_html,
        read_verdicts,
        shares_text,
        tally,
        verdict_shares,
        write_verdicts,
    )

    return (
        DATA,
        DATASETS,
        Image,
        ImageDraw,
        Path,
        RAW_LABELS,
        REPO_ROOT,
        SHARED_CLASSES,
        crop_box,
        legend_html,
        load_review_config,
        lru_cache,
        mo,
        np,
        outline,
        overlay,
        pd,
        pred_shares,
        raw_legend,
        raw_legend_html,
        re,
        read_verdicts,
        shares_text,
        tally,
        verdict_shares,
        write_verdicts,
    )


@app.cell
def _(DATA, Path, REPO_ROOT, load_review_config, mo, pd, read_verdicts):
    # The review config comes from the command line:
    #   uv run marimo edit notebooks/review/visual_review.py -- --config configs/review/t10.yaml
    _arg = mo.cli_args().get("config")
    if not _arg:
        raise SystemExit("pass --config configs/review/<task>.yaml after `--`")
    config_path = Path(str(_arg))
    cfg = load_review_config(REPO_ROOT / config_path)
    out_dir = DATA / cfg["out_dir"]
    verdict_path = REPO_ROOT / cfg["verdicts"]
    samples = pd.read_csv(out_dir / "samples.csv", dtype={"id": str})
    saved = read_verdicts(verdict_path)
    saved_map = {(r.question, r.key): (r.verdict, r.note) for r in saved.itertuples()}
    runs = cfg["runs"]
    return cfg, config_path, out_dir, runs, samples, saved_map, verdict_path


@app.cell
def _(cfg, config_path, legend_html, mo, runs, samples, verdict_path):
    mo.md(f"""
    # {cfg.get("title", "Visual review")}

    Runs: {", ".join(f"**{a}** `{r}`" for a, r in runs.items())}.
    Config: `{config_path}`.

    **How to use.** Run on the team machine (images are on the shared mount):
    `uv run marimo edit notebooks/review/visual_review.py -- --config {config_path}`.
    Give each image a verdict (and a note if useful), then tick the decisions at
    the end. Every change is saved at once to `{cfg["verdicts"]}`; commit that
    file. Nothing else is written.

    **Pictures.** Per image, two rows. Top: the photo, the ground truth and the
    raw (original dataset) labels; when the class is small, a zoomed view plus the
    full photo (yellow box = zoomed area). Bottom: each model's prediction of the
    same area. The white outline marks the raw class being asked about. Use the
    sliders above the tabs for panel size and mask opacity.
    {len(samples)} samples, picked by `scripts/review_predictions.py`: per
    question, the images with the lowest {cfg["rank_run"]} recall ("worst") plus
    random ones ("random"), so label errors are not over-counted. Verdicts file:
    `{verdict_path.name}`.

    {legend_html()}
    """)
    return


@app.cell
def _(
    DATA,
    DATASETS,
    Image,
    ImageDraw,
    RAW_LABELS,
    crop_box,
    legend_html,
    lru_cache,
    np,
    out_dir,
    outline,
    overlay,
    pred_shares,
    raw_legend,
    raw_legend_html,
    runs,
    shares_text,
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
        one panel per run. Returns (image, info text, legend HTML of the raw
        classes in the shown area, legend HTML of the shared classes in the
        shown area, per-run text of where the outlined pixels went)."""
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
        bottom, seen, where = [], set(np.unique(gt[y0:y1, x0:x1]).tolist()), []
        for alias in runs:
            pred = np.asarray(Image.open(out_dir / "pred" / alias / dataset / f"{index}.png"))
            bottom.append(_panel(pred, alias))
            seen |= set(np.unique(pred[y0:y1, x0:x1]).tolist())
            where.append(f"**{alias}**: {shares_text(pred_shares(raw, pred, ids))}")
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
        return canvas, info, legend, legend_html(sorted(seen)), " · ".join(where)

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

    decision_ui = mo.ui.dictionary(
        {d: _decision_ui(d, s) for d, s in cfg.get("decisions", {}).items()}
    )
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
        img, info, legend, colours, where = render(
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
                mo.Html(
                    f"<small>Colours in view (ground truth and predictions): {colours}</small>"
                ),
                mo.Html(f"<small>Raw labels in view: {legend}</small>"),
                mo.md(f"Outlined pixels predicted as: {where}"),
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
            mo.md("## 1. Visual review"),
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

    _parts = [mo.md("## 2. Tally")]
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
def _(SHARED_CLASSES, cfg, mo, target_shares, verdict_shares, verdict_table):
    _all = verdict_shares(target_shares, verdict_table)
    _cls = list(SHARED_CLASSES)
    _parts = [
        mo.md(
            "### Where the target pixels go, per verdict\n\n"
            "Mean over images of the share of the question's raw-class pixels that "
            "each run predicts as each shared class, with the random and worst picks "
            "kept apart. Classes below 0.04 everywhere are hidden.\n\n"
            f"The worst images were picked by {cfg['rank_run']}'s recall, so "
            f"{cfg['rank_run']}'s shares on them are low by construction. Only the "
            "random picks show what is typical; mind the `images` count."
        )
    ]
    for _q, _spec in cfg["questions"].items():
        _t = _all[_all["question"] == _q].drop(columns="question")
        if _t.empty:
            continue
        _t = _t.sort_values(["verdict", "pick", "run"]).round(2)
        _keep = [c for c in _cls if (_t[c] > 0.04).any()]
        _t = _t[["verdict", "pick", "run", "images", *_keep]].reset_index(drop=True)
        _parts += [mo.md(f"**{_spec['title']}**"), mo.ui.table(_t, selection=None)]
    mo.vstack(_parts)
    return


@app.cell
def _(cfg, decision_ui, mo):
    _decisions = cfg.get("decisions", {})
    mo.vstack(
        [
            mo.md("""
    ## 3. Decisions

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
            for d, spec in _decisions.items()
        ]
    ) if _decisions else mo.md("")
    return


if __name__ == "__main__":
    app.run()
