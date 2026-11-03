import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image

    from cloth_segment.data import splits as S
    from cloth_segment.data.dedup import score_candidates
    from cloth_segment.paths import DATA, DATASETS
    return DATA, DATASETS, Image, S, mo, np, pd, plt, score_candidates


@app.cell
def _(mo):
    mo.md("""
    # T3 — Duplicate threshold review

    Every image pair with dHash-64 distance ≤ `HASH_MAX_DIST` (plain or
    mirrored) is a **candidate**, scored by 32×32 thumbnail correlation. A
    candidate is a **duplicate** if its correlation reaches the threshold for
    its kind. Use the sliders to try other thresholds, the impact table to see
    what they would change, and the viewer to inspect pairs near the cut.

    Current rule (`cloth_segment.data.splits`): LIP/CIHP pairs ≥ `CORR_MIN`;
    any pair involving DFMM ≥ `CORR_MIN_DFMM_PAIR`. To change it, edit those
    constants, bump `SPLIT_VERSION` and rerun `scripts/build_splits.py`.
    """)
    return


@app.cell
def _(S):
    manifest, meta = S.load_manifest()
    return manifest, meta


@app.cell
def _(DATA, DATASETS, S, manifest, meta, mo, pd, score_candidates):
    # Cache keyed by data version and hash distance; the row indices a/b refer
    # to manifest rows, which is the same order build_splits.py hashes.
    _cache = DATA / "eda" / f"dedup_candidates_{meta['data_version']}_d{S.HASH_MAX_DIST}.csv"
    if _cache.exists():
        cand = pd.read_csv(_cache)
    else:
        with mo.status.spinner(f"Hashing {len(manifest)} images (~1 min, cached afterwards)…"):
            cand = score_candidates([DATASETS / p for p in manifest.image], S.HASH_MAX_DIST)
        cand.to_csv(_cache, index=False)
    return (cand,)


@app.cell
def _(cand, manifest, np):
    _ds = manifest.dataset.to_numpy()
    _src = manifest.source_split.to_numpy()
    _base_outfit = np.array([i.split("-id_")[-1].split("-")[0] if d == "dfmm" else ""
                             for d, i in zip(_ds, manifest.id)])
    a, b = cand.a.to_numpy(), cand.b.to_numpy()
    da, db = _ds[a], _ds[b]
    _dfmm_a, _dfmm_b = da == "dfmm", db == "dfmm"
    kind = np.select(
        [_dfmm_a & _dfmm_b & (_base_outfit[a] == _base_outfit[b]),
         _dfmm_a & _dfmm_b,
         _dfmm_a | _dfmm_b],
        ["dfmm, same outfit", "dfmm, different outfit", "dfmm × lip/cihp"],
        default="lip/cihp",
    )
    _test_a = (da != "dfmm") & (_src[a] == "val")
    _test_b = (db != "dfmm") & (_src[b] == "val")
    relation = np.select([_test_a & _test_b, _test_a ^ _test_b], ["within test", "crosses test"],
                         default="within train pool")
    pairs = cand.assign(kind=kind, relation=relation,
                        ds_a=da, ds_b=db,
                        name_a=manifest.dataset.to_numpy()[a] + "/" + manifest.source_split.to_numpy()[a] + " " + manifest.id.to_numpy()[a],
                        name_b=manifest.dataset.to_numpy()[b] + "/" + manifest.source_split.to_numpy()[b] + " " + manifest.id.to_numpy()[b])
    KINDS = ["lip/cihp", "dfmm, same outfit", "dfmm, different outfit", "dfmm × lip/cihp"]
    return KINDS, pairs


@app.cell
def _(S, mo):
    t_main = mo.ui.slider(0.50, 1.00, step=0.01, value=S.CORR_MIN, label="LIP/CIHP threshold", show_value=True)
    t_dfmm = mo.ui.slider(0.80, 1.00, step=0.005, value=S.CORR_MIN_DFMM_PAIR, label="DFMM-pair threshold", show_value=True)
    mo.hstack([t_main, t_dfmm], justify="start", gap=3)
    return t_dfmm, t_main


@app.cell
def _(S, pairs, t_dfmm, t_main):
    accepted = S.duplicate_mask(pairs.ds_a.to_numpy(), pairs.ds_b.to_numpy(), pairs["corr"].to_numpy(),
                                t_main.value, t_dfmm.value)
    current = S.duplicate_mask(pairs.ds_a.to_numpy(), pairs.ds_b.to_numpy(), pairs["corr"].to_numpy())
    return accepted, current


@app.cell
def _(mo):
    mo.md("""
    ## Correlation by pair kind

    Dashed: slider threshold. Counts on a log scale; only candidates (hash
    distance ≤ limit) are shown.
    """)
    return


@app.cell
def _(KINDS, np, pairs, plt, t_dfmm, t_main):
    _fig, _axes = plt.subplots(1, 4, figsize=(18, 3.2), sharex=True)
    _bins = np.linspace(-0.2, 1.0, 61)
    for _ax, _k in zip(_axes, KINDS):
        _p = pairs[pairs.kind == _k]
        for _rel, _c in [("within train pool", "C0"), ("crosses test", "C3"), ("within test", "C2")]:
            _v = _p[_p.relation == _rel]["corr"]
            if len(_v):
                _ax.hist(_v, bins=_bins, alpha=0.6, color=_c, label=f"{_rel} ({len(_v)})")
        _ax.axvline(t_main.value if _k == "lip/cihp" else t_dfmm.value, ls="--", c="k")
        _ax.set_yscale("log")
        _ax.set_title(f"{_k} ({len(_p)})", fontsize=10)
        _ax.legend(fontsize=7)
        _ax.set_xlabel("thumbnail correlation")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Impact of the slider thresholds

    Accepted pairs per kind, and the resulting split sizes compared with the
    built manifest (positive = more images in that split).
    """)
    return


@app.cell
def _(KINDS, S, accepted, current, manifest, mo, pairs, pd):
    _tab = pd.DataFrame({
        "candidates": pairs.groupby("kind").size(),
        "accepted (built)": pairs[current].groupby("kind").size(),
        "accepted (slider)": pairs[accepted].groupby("kind").size(),
        "crosses test (slider)": pairs[accepted & (pairs.relation == "crosses test")].groupby("kind").size(),
    }).reindex(KINDS).fillna(0).astype(int)

    try:
        _g, _split, _reason = S.apply_rules(manifest, pairs.a[accepted], pairs.b[accepted])
        _new = pd.crosstab(manifest.dataset, pd.Series(_split, name="split"))
        _old = pd.crosstab(manifest.dataset, manifest.split)
        _delta = _new.sub(_old, fill_value=0).astype(int)
        _impact = mo.hstack([
            mo.vstack([mo.md("**split sizes (slider)**"), _new]),
            mo.vstack([mo.md("**change vs built**"), _delta]),
        ], justify="start", gap=2)
    except ValueError as e:  # e.g. a DFMM group merged with LIP/CIHP train images
        _impact = mo.callout(mo.md(f"These thresholds break a split rule: `{e}`"), kind="danger")
    mo.vstack([_tab, _impact])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Pair viewer

    Top image = `a`, bottom = `b` (mirrored back when the match was found on
    the flipped image). ✓/✗ = duplicate under the slider thresholds.
    """)
    return


@app.cell
def _(KINDS, mo):
    v_kind = mo.ui.dropdown(KINDS, value="lip/cihp", label="kind")
    v_rel = mo.ui.dropdown(["any", "crosses test", "within train pool", "within test"], value="any", label="relation")
    v_range = mo.ui.range_slider(-0.2, 1.0, step=0.005, value=[0.65, 0.75], label="correlation", show_value=True)
    v_order = mo.ui.dropdown(["random", "ascending", "descending"], value="random", label="order")
    v_n = mo.ui.slider(6, 36, step=6, value=12, label="pairs", show_value=True)
    v_seed = mo.ui.number(start=0, stop=10_000, value=0, label="seed")
    mo.hstack([v_kind, v_rel, v_range, v_order, v_n, v_seed], justify="start", wrap=True)
    return v_kind, v_n, v_order, v_range, v_rel, v_seed


@app.cell
def _(accepted, pairs, v_kind, v_n, v_order, v_range, v_rel, v_seed):
    _lo, _hi = v_range.value
    _m = (pairs.kind == v_kind.value) & pairs["corr"].between(_lo, _hi)
    if v_rel.value != "any":
        _m &= pairs.relation == v_rel.value
    selected = pairs[_m].assign(dup=accepted[_m.to_numpy()])
    if v_order.value == "random":
        shown = selected.sample(min(v_n.value, len(selected)), random_state=v_seed.value)
    else:
        shown = selected.sort_values("corr", ascending=v_order.value == "ascending").head(v_n.value)
    return selected, shown


@app.cell
def _(DATASETS, Image, manifest, mo, selected, shown):
    import base64 as _b64
    import html as _html
    import io as _io

    def _thumb(row, flip):
        # Small JPEG data URIs keep the cell output well under marimo's size limit.
        with Image.open(DATASETS / manifest.image.iloc[row]) as im:
            im = im.convert("RGB")
            im.thumbnail((220, 220))
            if flip:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            buf = _io.BytesIO()
            im.save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + _b64.b64encode(buf.getvalue()).decode()

    def _card(r):
        colour = "#c0392b" if r.dup else "#555"
        mark = "✓ dup" if r.dup else "✗ kept"
        head = f"{mark} · r={r['corr']:.3f} · d={r.hash_dist}" + (" · mirrored" if r.flip else "")
        img = ('<img src="{}" style="max-width:220px;max-height:220px;display:block;margin:2px auto">')
        cap = '<div style="font-size:11px;word-break:break-all">{}</div>'
        return (f'<div style="border:2px solid {colour};border-radius:6px;padding:6px;width:236px">'
                f'<div style="font-size:12px;font-weight:600;color:{colour}">{head}</div>'
                + img.format(_thumb(r.a, False)) + cap.format(_html.escape(r.name_a))
                + img.format(_thumb(r.b, bool(r.flip))) + cap.format(_html.escape(r.name_b))
                + "</div>")

    if len(shown) == 0:
        _out = mo.md(f"No pairs in this range ({len(selected)} match the filters).")
    else:
        _grid = ('<div style="display:flex;flex-wrap:wrap;gap:10px">'
                 + "".join(_card(r) for _, r in shown.iterrows()) + "</div>")
        _out = mo.vstack([mo.md(f"{len(selected)} pairs match; showing {len(shown)}."), mo.Html(_grid)])
    _out
    return


if __name__ == "__main__":
    app.run()
