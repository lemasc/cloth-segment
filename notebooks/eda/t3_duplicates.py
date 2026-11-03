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
def _(S, mo):
    mo.md(f"""
    # T3 — Duplicate threshold review

    ## Why this notebook exists

    The same photo can appear twice in our data, for example once in a
    training split and once in a test split. If that happens, the model is
    tested on an image it has already seen, and the test score is too
    optimistic. `scripts/build_splits.py` finds such duplicates and keeps
    each set of duplicates inside one split. This notebook lets you check, by
    eye, whether its threshold for "duplicate" is right.

    ## How duplicates are found (two steps)

    **Step 1: hash search (fast, finds candidates).** Every image is shrunk
    to a 9×8 grayscale picture and turned into a 64-bit fingerprint
    (a *difference hash*, or dHash). Two images whose fingerprints differ in
    at most **{S.HASH_MAX_DIST} of 64 bits** become a **candidate pair**.
    Each image is also compared as a mirrored copy, which catches flipped
    duplicates. The hash survives resizing and JPEG re-compression, but it
    cannot detect crops.

    **Step 2: thumbnail correlation (slower, decides).** The hash is coarse,
    so many candidates are just similar-looking images. For each candidate,
    both images are shrunk to 32×32 grayscale and compared pixel by pixel
    with a correlation score:

    - **1.0** means identical apart from brightness and contrast;
    - around **0.7** means the same scene, but with changes such as a
      different crop, a watermark or a colour change;
    - **0** or below means unrelated.

    A candidate counts as a **duplicate** if its score reaches the threshold.

    ## The current thresholds

    | Pair involves | Threshold | Why |
    |---|---|---|
    | only LIP and/or CIHP | **{S.CORR_MIN}** (`CORR_MIN`) | In-the-wild photos: pairs at 0.7 and above were still the same photo when checked. |
    | any DFMM image | **{S.CORR_MIN_DFMM_PAIR}** (`CORR_MIN_DFMM_PAIR`) | DFMM is shot in a studio: same white background, same model, same pose. Different outfits on the same model still score up to ~0.98, and plain backgrounds can match LIP/CIHP photos at ~0.7. |

    Accepted duplicates are merged into one **group**. A group always goes
    to a single split, and if a group contains an official test image, its
    other images are **excluded** (`leak_test`). The test sets themselves
    never change.

    ## How to use this notebook

    1. **Histograms:** see how the scores of each pair kind are spread, and
       where the threshold falls.
    2. **Sliders:** try other thresholds. Nothing is written to disk.
    3. **Impact table:** see how many images would move between splits.
    4. **Pair viewer:** look at actual pairs just above and just below the
       threshold. Ask: *is this the same photo?*

    The thresholds are right when pairs just above the cut are the same
    photo (true duplicates) and pairs just below it are different photos.
    """)
    return


@app.cell
def _(S):
    manifest, meta = S.load_manifest()
    return manifest, meta


@app.cell
def _(DATA, DATASETS, S, manifest, meta, mo, pd, score_candidates):
    # Scored candidates are cached per data version and hash distance. The a/b
    # columns are row positions in the manifest, which build_splits.py hashes
    # in the same order.
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
    _outfit = np.array([i.split("-id_")[-1].split("-")[0] if d == "dfmm" else ""
                        for d, i in zip(_ds, manifest.id)])
    a, b = cand.a.to_numpy(), cand.b.to_numpy()
    da, db = _ds[a], _ds[b]
    _dfmm_a, _dfmm_b = da == "dfmm", db == "dfmm"
    kind = np.select(
        [_dfmm_a & _dfmm_b & (_outfit[a] == _outfit[b]),
         _dfmm_a & _dfmm_b,
         _dfmm_a | _dfmm_b],
        ["DFMM, same outfit", "DFMM, different outfits", "DFMM vs LIP/CIHP"],
        default="LIP/CIHP",
    )
    _test_a = (da != "dfmm") & (_src[a] == "val")
    _test_b = (db != "dfmm") & (_src[b] == "val")
    relation = np.select([_test_a & _test_b, _test_a ^ _test_b],
                         ["both in test", "train vs test"], default="both in train pool")

    def _name(i):
        return manifest.dataset.to_numpy()[i] + " / " + manifest.source_split.to_numpy()[i] + " / " + manifest.id.to_numpy()[i]

    pairs = cand.assign(kind=kind, relation=relation, ds_a=da, ds_b=db, name_a=_name(a), name_b=_name(b))
    KINDS = ["LIP/CIHP", "DFMM, same outfit", "DFMM, different outfits", "DFMM vs LIP/CIHP"]
    RELATIONS = ["train vs test", "both in train pool", "both in test"]
    return KINDS, RELATIONS, pairs


@app.cell
def _(mo):
    mo.md("""
    ## Pair kinds and relations

    Every candidate pair has a **kind** (which datasets it comes from) and a
    **relation** (which splits it comes from).

    | Kind | Meaning | What a duplicate does |
    |---|---|---|
    | **LIP/CIHP** | both images are from LIP or CIHP (either combination) | merges their groups |
    | **DFMM, same outfit** | two DFMM images with the same outfit ID | nothing: they are already in one group |
    | **DFMM, different outfits** | two DFMM images with different outfit IDs | merges the two outfits into one group |
    | **DFMM vs LIP/CIHP** | one studio image, one in-the-wild image | would merge across datasets (none found so far) |

    | Relation | Meaning | Why it matters |
    |---|---|---|
    | **train vs test** | one image is in the official LIP/CIHP test set (their official val), the other is not | **leakage**: if it's a duplicate, the non-test image is excluded |
    | **both in train pool** | neither image is in the official test set | only decides which images share a split |
    | **both in test** | both images are in the official test set | no effect: test sets are never changed |

    "Train pool" means everything except the official test sets: LIP and CIHP
    train, and all of DFMM, which we split ourselves.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Try other thresholds

    Move the sliders to see how the histograms, impact table and pair viewer
    would change. They start at the values used for the built splits.
    """)
    return


@app.cell
def _(S, mo):
    t_main = mo.ui.slider(0.50, 1.00, step=0.01, value=S.CORR_MIN, show_value=True,
                          label="Threshold for LIP/CIHP pairs")
    t_dfmm = mo.ui.slider(0.80, 1.00, step=0.005, value=S.CORR_MIN_DFMM_PAIR, show_value=True,
                          label="Threshold for pairs involving DFMM")
    mo.hstack([t_main, t_dfmm], justify="start", gap=3)
    return t_dfmm, t_main


@app.cell
def _(S, pairs, t_dfmm, t_main):
    accepted = S.duplicate_mask(pairs.ds_a.to_numpy(), pairs.ds_b.to_numpy(), pairs["corr"].to_numpy(),
                                t_main.value, t_dfmm.value)
    current = S.duplicate_mask(pairs.ds_a.to_numpy(), pairs.ds_b.to_numpy(), pairs["corr"].to_numpy())
    return accepted, current


@app.cell
def _(accepted, current, mo, pairs):
    _leaks = int((accepted & (pairs.relation == "train vs test").to_numpy()).sum())
    mo.callout(mo.md(
        f"With these thresholds, **{int(accepted.sum()):,}** of {len(pairs):,} candidate pairs are duplicates "
        f"(built splits: {int(current.sum()):,}). **{_leaks:,}** of them link a non-test image to a test image."
    ), kind="info")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Score distribution per pair kind

    One panel per pair kind. The x-axis is the thumbnail correlation; the
    y-axis counts pairs on a **log scale**, so small bars are visible next
    to large ones. Colours show the relation; red (**train vs test**) is the
    one that matters for leakage. The **dashed line** is the slider
    threshold: pairs to its right count as duplicates.

    What to look for: true duplicates usually form a cluster near 1.0 that is
    separate from the bulk of look-alikes. A good threshold sits in the gap
    between them.
    """)
    return


@app.cell
def _(KINDS, RELATIONS, np, pairs, plt, t_dfmm, t_main):
    _colours = {"train vs test": "C3", "both in train pool": "C0", "both in test": "C2"}
    _fig, _axes = plt.subplots(1, 4, figsize=(18, 3.4), sharex=True)
    _bins = np.linspace(-0.2, 1.0, 61)
    for _ax, _k in zip(_axes, KINDS):
        _p = pairs[pairs.kind == _k]
        for _rel in RELATIONS:
            _v = _p[_p.relation == _rel]["corr"]
            if len(_v):
                _ax.hist(_v, bins=_bins, alpha=0.6, color=_colours[_rel], label=f"{_rel} ({len(_v):,})")
        _ax.axvline(t_main.value if _k == "LIP/CIHP" else t_dfmm.value, ls="--", c="k", label="threshold")
        _ax.set_yscale("log")
        _ax.set_title(f"{_k}: {len(_p):,} candidate pairs", fontsize=10)
        _ax.legend(fontsize=7, loc="upper left")
        _ax.set_xlabel("thumbnail correlation")
    _axes[0].set_ylabel("pairs (log scale)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. What the slider thresholds would change

    **Pairs table**, one row per pair kind:

    - *candidate pairs*: all pairs found by the hash search;
    - *duplicates, built*: pairs accepted by the thresholds in `splits.py`
      (the splits on disk);
    - *duplicates, slider*: pairs accepted at the slider values;
    - *train-vs-test duplicates, slider*: the leakage pairs among them.

    **Split tables**: images per dataset and split if the splits were rebuilt
    at the slider values, and the difference from the built splits. For
    example, `-11` under `train` means 11 fewer training images, usually
    because they moved to `excluded` or `val`. All zeros means the slider
    values give the same splits as the ones on disk.
    """)
    return


@app.cell
def _(KINDS, S, accepted, current, manifest, mo, pairs, pd):
    _leak = (pairs.relation == "train vs test").to_numpy()
    _tab = pd.DataFrame({
        "candidate pairs": pairs.groupby("kind").size(),
        "duplicates, built": pairs[current].groupby("kind").size(),
        "duplicates, slider": pairs[accepted].groupby("kind").size(),
        "train-vs-test duplicates, slider": pairs[accepted & _leak].groupby("kind").size(),
    }).reindex(KINDS).fillna(0).astype(int).rename_axis("pair kind")

    try:
        _g, _split, _reason = S.apply_rules(manifest, pairs.a[accepted], pairs.b[accepted])
        _order = ["train", "val", "test", "excluded"]
        _new = pd.crosstab(manifest.dataset, pd.Series(_split, name="split")).reindex(columns=_order, fill_value=0)
        _old = pd.crosstab(manifest.dataset, manifest.split).reindex(columns=_order, fill_value=0)
        _delta = (_new - _old).astype(int)
        _impact = mo.hstack([
            mo.vstack([mo.md("**Images per split at the slider values**"), _new]),
            mo.vstack([mo.md("**Difference from the built splits**"), _delta]),
        ], justify="start", gap=2)
    except ValueError as e:
        _impact = mo.callout(mo.md(
            f"These thresholds would merge a DFMM outfit with LIP/CIHP training images, "
            f"and there is no rule for which split such a group should go to:\n\n`{e}`"
        ), kind="danger")
    mo.vstack([mo.md("**Pairs**"), _tab, _impact], gap=1)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Look at the pairs

    Choose a pair kind and a correlation range, and check whether the pairs
    are really the same photo.

    **Reading a card:**

    - **Top and bottom images** are the two images of the pair. A
      *mirrored* match was found on a flipped copy; the bottom image is
      flipped back so the two line up.
    - **Red border, "duplicate"**: at the slider values this pair counts as
      a duplicate. **Grey border, "not duplicate"**: it doesn't.
    - **correlation**: the step-2 score, deciding whether it's a duplicate.
      **hash distance**: the step-1 score (bits that differ, 0–64).
    - The captions give `dataset / source split / image id`. For LIP and
      CIHP, `val` is the official test set.

    **Good ranges to check:**

    | Pair kind | Range | Question |
    |---|---|---|
    | LIP/CIHP | 0.60–0.70 | Just below the cut: any real duplicates being missed? |
    | LIP/CIHP | 0.70–0.80 | Just above the cut: any different photos being caught? |
    | DFMM, different outfits | 0.95–0.98 | Same model in different clothes (correctly not duplicates)? |
    | DFMM, different outfits | 0.98–1.00 | Really the same photo under two outfit IDs? |
    | DFMM vs LIP/CIHP | 0.60–0.80 | Plain-background false matches? |

    Set **relation** to *train vs test* to see only the pairs that affect
    leakage. Change **seed** for a different random sample.
    """)
    return


@app.cell
def _(KINDS, RELATIONS, mo):
    v_kind = mo.ui.dropdown(KINDS, value="LIP/CIHP", label="Pair kind")
    v_rel = mo.ui.dropdown(["any", *RELATIONS], value="any", label="Relation")
    v_range = mo.ui.range_slider(-0.2, 1.0, step=0.005, value=[0.65, 0.75], show_value=True,
                                 label="Correlation range")
    v_order = mo.ui.dropdown({"random sample": "random", "lowest score first": "ascending",
                              "highest score first": "descending"}, value="random sample", label="Order")
    v_n = mo.ui.slider(6, 36, step=6, value=12, show_value=True, label="Pairs to show")
    v_seed = mo.ui.number(start=0, stop=10_000, value=0, label="Seed")
    mo.vstack([
        mo.hstack([v_kind, v_rel, v_order], justify="start", gap=2),
        mo.hstack([v_range, v_n, v_seed], justify="start", gap=2),
    ])
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
        colour = "#c0392b" if r.dup else "#777"
        verdict = "duplicate" if r.dup else "not duplicate"
        mirrored = '<div style="font-size:11px">mirrored match</div>' if r.flip else ""
        img = '<img src="{}" style="max-width:220px;max-height:220px;display:block;margin:4px auto">'
        cap = '<div style="font-size:11px;word-break:break-all;opacity:0.8">{}</div>'
        return (
            f'<div style="border:2px solid {colour};border-radius:6px;padding:6px;width:236px">'
            f'<div style="font-size:13px;font-weight:600;color:{colour}">{verdict}</div>'
            f'<div style="font-size:11px">correlation {r["corr"]:.3f} · hash distance {r.hash_dist}</div>'
            f"{mirrored}"
            + img.format(_thumb(r.a, False)) + cap.format(_html.escape(r.name_a))
            + img.format(_thumb(r.b, bool(r.flip))) + cap.format(_html.escape(r.name_b))
            + "</div>"
        )

    if len(shown) == 0:
        _out = mo.md(f"No pairs in this range ({len(selected):,} match the other filters).")
    else:
        _n_dup = int(selected.dup.sum())
        _grid = ('<div style="display:flex;flex-wrap:wrap;gap:10px">'
                 + "".join(_card(r) for _, r in shown.iterrows()) + "</div>")
        _out = mo.vstack([
            mo.md(f"**{len(selected):,}** pairs match these filters ({_n_dup:,} duplicates at the slider "
                  f"values, {len(selected) - _n_dup:,} not). Showing {len(shown)}."),
            mo.Html(_grid),
        ])
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Changing the thresholds

    The sliders only preview. To apply new thresholds:

    1. Edit `CORR_MIN` and/or `CORR_MIN_DFMM_PAIR` in
       `src/cloth_segment/data/splits.py`.
    2. Bump `SPLIT_VERSION` (e.g. `splits-v1` → `splits-v2`). The build
       refuses to overwrite an existing version, so runs trained on the old
       splits still point at the right data.
    3. Run `uv run python scripts/build_splits.py`, and record the new data
       version and your reasoning in the T3 handoff.
    """)
    return


if __name__ == "__main__":
    app.run()
