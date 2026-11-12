"""DFMM attribute heads for multi-task training: spec, targets, loss and metrics.

The loader returns all 18 DFMM attributes per image, with NA, missing
and non-DFMM images set to IGNORE. `model.attr` in the experiment config picks
the heads and the extra values to ignore:

    model:
      attr:
        heads: [sleeve, lower_length, ...]          # names from DFMM_ATTRS
        ignore_values: {sleeve: [not_long]}         # value names, set to IGNORE
        loss_weight: 0.1                            # L = L_seg + w * L_attr
        hidden: 256
        dropout: 0.2

Each head keeps every value of its attribute as a logit (NA dropped), so the
class index is the raw DFMM code. Ignored values never get a gradient and are
left out of the metrics.

The settings live under `model`, not `loss` or `eval`: those are shared by
every model and stay the same as the segmentation-only runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from cloth_segment.lib.data.dataset import ATTR_NAMES, ATTR_NUM_CLASSES
from cloth_segment.lib.data.labels import IGNORE
from cloth_segment.lib.data.sources import DFMM_ATTRS


@dataclass(frozen=True)
class AttrSpec:
    heads: tuple[str, ...]
    ignore: dict[str, tuple[int, ...]] = field(default_factory=dict)  # head -> ignored codes
    loss_weight: float = 0.1
    hidden: int = 256
    dropout: float = 0.2

    @property
    def num_classes(self) -> tuple[int, ...]:
        return tuple(ATTR_NUM_CLASSES[h] for h in self.heads)

    @property
    def columns(self) -> list[int]:
        """Positions of the heads in the loader's `attrs` vector."""
        return [ATTR_NAMES.index(h) for h in self.heads]

    def value_names(self, head: str) -> tuple[str, ...]:
        return DFMM_ATTRS[head][: ATTR_NUM_CLASSES[head]]


def parse_spec(attr_cfg: Mapping) -> AttrSpec:
    """`model.attr` config -> AttrSpec; unknown heads or values fail loudly."""
    heads = tuple(attr_cfg.get("heads") or ())
    if not heads:
        raise ValueError("model.attr.heads is empty")
    unknown = [h for h in heads if h not in ATTR_NAMES]
    if unknown:
        raise ValueError(f"unknown attribute heads {unknown}; known: {list(ATTR_NAMES)}")
    if len(set(heads)) != len(heads):
        raise ValueError(f"duplicate attribute heads in {list(heads)}")
    ignore: dict[str, tuple[int, ...]] = {}
    for head, values in (attr_cfg.get("ignore_values") or {}).items():
        if head not in heads:
            raise ValueError(f"ignore_values names {head!r}, which is not a head")
        names = DFMM_ATTRS[head][: ATTR_NUM_CLASSES[head]]
        bad = [v for v in values if v not in names]
        if bad:
            raise ValueError(f"{head}: unknown values {bad}; known: {list(names)}")
        ignore[head] = tuple(names.index(v) for v in values)
    weight = float(attr_cfg.get("loss_weight", 0.1))
    dropout = float(attr_cfg.get("dropout", 0.2))
    hidden = int(attr_cfg.get("hidden", 256))
    if weight <= 0:
        raise ValueError(f"model.attr.loss_weight must be > 0 (got {weight})")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"model.attr.dropout must be in [0, 1) (got {dropout})")
    if hidden < 1:
        raise ValueError(f"model.attr.hidden must be >= 1 (got {hidden})")
    return AttrSpec(heads, ignore, weight, hidden, dropout)


def head_targets(attrs: torch.Tensor, spec: AttrSpec) -> torch.Tensor:
    """(B, 18) loader codes -> (B, len(heads)) targets, with `ignore_values` set to IGNORE."""
    y = attrs[:, spec.columns].clone()
    for k, head in enumerate(spec.heads):
        for code in spec.ignore.get(head, ()):
            y[:, k][y[:, k] == code] = IGNORE
    return y


def attr_loss(logits: Sequence[torch.Tensor], targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Mean over heads with a label in the batch of each head's CE; (loss, labelled images).

    A batch without any label (no DFMM image) gives a zero that is still tied
    to the graph, so the step runs the same way and AMP sees no missing grads.
    """
    terms = []
    for k, lg in enumerate(logits):
        y = targets[:, k]
        if (y != IGNORE).any():
            terms.append(F.cross_entropy(lg.float(), y, ignore_index=IGNORE))
    n_img = int((targets != IGNORE).any(dim=1).sum())
    if not terms:
        return sum(lg.sum() for lg in logits) * 0.0, n_img
    return torch.stack(terms).mean(), n_img


# ------------------------------------------------------------------ metrics


def head_metrics(
    y: np.ndarray, pred: np.ndarray, num_classes: int, names: Sequence[str], min_count: int
) -> dict:
    """Accuracy and macro-F1 of one head over labelled items (y != IGNORE).

    `macro_f1` averages classes with at least `min_count` true items; classes
    below it are reported but left out (too few to score). `macro_f1_present`
    averages every class with at least one true item.
    """
    keep = y != IGNORE
    y, pred = y[keep].astype(np.int64), pred[keep].astype(np.int64)
    support = np.bincount(y, minlength=num_classes)
    predicted = np.bincount(pred, minlength=num_classes)
    tp = np.bincount(y[y == pred], minlength=num_classes)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(num_classes), where=denom > 0)
    scored = support >= min_count
    present = support > 0
    return {
        "n": int(keep.sum()),
        "accuracy": float((y == pred).mean()) if len(y) else None,
        "macro_f1": float(f1[scored].mean()) if scored.any() else None,
        "macro_f1_present": float(f1[present].mean()) if present.any() else None,
        "classes": {
            names[c]: {"support": int(support[c]), "f1": float(f1[c]), "scored": bool(scored[c])}
            for c in range(num_classes)
        },
    }


def all_metrics(y: np.ndarray, pred: np.ndarray, spec: AttrSpec, min_count: int) -> dict:
    """`head_metrics` per head, plus the mean macro-F1 and accuracy over heads."""
    per = {
        h: head_metrics(y[:, k], pred[:, k], spec.num_classes[k], spec.value_names(h), min_count)
        for k, h in enumerate(spec.heads)
    }
    f1 = [m["macro_f1"] for m in per.values() if m["macro_f1"] is not None]
    acc = [m["accuracy"] for m in per.values() if m["accuracy"] is not None]
    return {
        "mean_macro_f1": float(np.mean(f1)) if f1 else None,
        "mean_accuracy": float(np.mean(acc)) if acc else None,
        "heads": per,
    }


def majority_predictions(y_train: np.ndarray, n: int, spec: AttrSpec) -> np.ndarray:
    """(n, heads) predictions of the most frequent labelled train value per head."""
    out = np.zeros((n, len(spec.heads)), dtype=np.int64)
    for k, c in enumerate(spec.num_classes):
        col = y_train[:, k]
        col = col[col != IGNORE]
        if len(col) == 0:
            raise ValueError(f"head {spec.heads[k]!r} has no labelled train items")
        out[:, k] = int(np.bincount(col, minlength=c).argmax())
    return out
