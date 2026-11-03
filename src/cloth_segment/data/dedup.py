"""Near-duplicate image detection for leakage checks (T3).

dHash (difference hash): grayscale, resize to (size+1)×size, compare
neighbouring pixels -> size² bits. Robust to rescaling and JPEG
re-compression, not to crops. Mirrored copies are found by also hashing the
flipped image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def dhash(img: Image.Image, size: int = 8) -> np.ndarray:
    """Return the dHash as a packed uint8 array of size²/8 bytes."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    return np.packbits(g[:, 1:] > g[:, :-1])


def hash_file(path: Path, size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """(hash, hash of the horizontally flipped image) for one image file."""
    with Image.open(path) as im:
        im.draft("L", (64, 64))  # fast JPEG decode at reduced scale
        im = im.convert("L")
        return dhash(im, size), dhash(im.transpose(Image.Transpose.FLIP_LEFT_RIGHT), size)


def to_u64(packed: np.ndarray) -> np.ndarray:
    """N×8 packed bytes -> N uint64 (for 64-bit hashes)."""
    return np.ascontiguousarray(packed).view(">u8").ravel().astype(np.uint64)


def near_pairs(a: np.ndarray, b: np.ndarray | None = None, max_dist: int = 4,
               chunk: int = 2048) -> np.ndarray:
    """Index pairs (i, j) with Hamming(a[i], b[j]) <= max_dist, as a K×3 array (i, j, dist).

    `a` and `b` are uint64 hash vectors. With b=None, compares a to itself and
    returns each pair once (i < j).
    """
    same = b is None
    b = a if same else b
    out = []
    for s in range(0, len(a), chunk):
        d = np.bitwise_count(a[s:s + chunk, None] ^ b[None, :])
        i, j = np.nonzero(d <= max_dist)
        i += s
        if same:
            keep = i < j
            i, j = i[keep], j[keep]
        out.append(np.stack([i, j, d[i - s, j]], axis=1))
    return np.concatenate(out) if out else np.empty((0, 3), dtype=np.int64)


def thumbnail(path: Path, size: int = 32) -> np.ndarray:
    """size×size grayscale thumbnail (aspect ignored), for verifying hash candidates."""
    with Image.open(path) as im:
        im.draft("L", (4 * size, 4 * size))
        return np.asarray(im.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def thumb_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between matching rows of two N×H×W thumbnail stacks."""
    def norm(x):
        x = x.reshape(len(x), -1).astype(np.float32)
        return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-3)
    return (norm(a) * norm(b)).mean(1)


def _hash_and_thumb(path: Path):
    h, hf = hash_file(path)
    return h, hf, thumbnail(path)


def score_candidates(paths: list[Path], max_dist: int, workers: int = 16):
    """All near-duplicate candidates among `paths`, scored by thumbnail correlation.

    Returns a DataFrame (a, b, hash_dist, flip, corr) with a < b as row
    indices into `paths`, one row per pair (the better of plain/mirrored).
    """
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_hash_and_thumb, paths, chunksize=128))
    h = to_u64(np.stack([r[0] for r in res]))
    hf = to_u64(np.stack([r[1] for r in res]))
    thumbs = np.stack([r[2] for r in res])

    plain = near_pairs(h, max_dist=max_dist)
    flip = near_pairs(h, hf, max_dist=max_dist)
    flip = flip[flip[:, 0] != flip[:, 1]]
    cand = pd.concat([
        pd.DataFrame({"a": plain[:, 0], "b": plain[:, 1], "hash_dist": plain[:, 2], "flip": False}),
        pd.DataFrame({"a": flip[:, :2].min(1), "b": flip[:, :2].max(1), "hash_dist": flip[:, 2], "flip": True}),
    ], ignore_index=True)
    tb = thumbs[cand.b.to_numpy()]
    tb = np.where(cand.flip.to_numpy()[:, None, None], tb[:, :, ::-1], tb)
    cand["corr"] = thumb_corr(thumbs[cand.a.to_numpy()], tb)
    return cand.sort_values("corr", ascending=False).drop_duplicates(["a", "b"]).reset_index(drop=True)
