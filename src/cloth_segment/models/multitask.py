"""SegFormer-B2 with DFMM attribute heads on the shared encoder (multi-task).

`segformer_b2_mt` is the plain SegFormer-B2 (same builder, so with the same seed
the segmentation part starts from the same weights) plus a small
attribute head built after it:

    encoder stage 4 (512 ch, stride 32) -> global avg + max pool (1024)
      -> Linear(1024, hidden) -> GELU -> Dropout -> one Linear(hidden, K) per head

`forward(x)` returns segmentation logits only, like every other model, so
`evaluate.py` (frozen), `errors.py` and `rawclass.py` score it unchanged.
`forward_multi(x)` returns (segmentation logits, list of attribute logits) and
runs the encoder once for both. The state_dict keys of the segmentation part
are the same as `segformer_b2` (`net.*`); the head adds `attr_head.*`.
"""

from __future__ import annotations

import torch
from torch import nn

from cloth_segment.attributes import AttrSpec, parse_spec
from cloth_segment.models import register
from cloth_segment.models.segformer import build_segformer_b2
from cloth_segment.models.wrappers import PadToMultiple


class AttrHead(nn.Module):
    """Pooled deepest encoder feature -> shared hidden layer -> one classifier per head."""

    def __init__(self, in_channels: int, spec: AttrSpec) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(2 * in_channels, spec.hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
        )
        self.heads = nn.ModuleList(nn.Linear(spec.hidden, k) for k in spec.num_classes)

    def forward(self, feat: torch.Tensor) -> list[torch.Tensor]:
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        h = self.trunk(pooled)
        return [head(h) for head in self.heads]


class SegAttrModel(PadToMultiple):
    """`PadToMultiple` SegFormer plus `attr_head`; see the module docstring."""

    def __init__(self, seg: PadToMultiple, spec: AttrSpec) -> None:
        super().__init__(seg.net, seg.multiple)
        self.spec = spec
        self.attr_head = AttrHead(self.net.encoder.out_channels[-1], spec)

    def forward_multi(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hw = x.shape[-2:]
        xp = self.pad(x)
        features = self.net.encoder(xp)
        masks = self.net.segmentation_head(self.net.decoder(features))
        return self.restore(masks, xp.shape[-2:], hw), self.attr_head(features[-1])


@register("segformer_b2_mt")
def build_segformer_b2_mt(model_cfg: dict) -> nn.Module:
    """`segformer_b2` keys plus `attr` (see `cloth_segment.attributes`)."""
    if not model_cfg.get("attr"):
        raise ValueError("segformer_b2_mt needs a model.attr section")
    spec = parse_spec(model_cfg["attr"])
    seg = build_segformer_b2(model_cfg)  # first: same RNG use as segformer_b2
    return SegAttrModel(seg, spec)
