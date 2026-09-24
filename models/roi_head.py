"""ROI head (Phases 11-13): ROI Align -> Coordinate Attention -> cls/reg heads.

Multi-level ROI Align: proposals are assigned to P3/P4/P5 by scale (FPN mapping
with k0=4, canonical size 112), features are pooled per level with
torchvision.ops.roi_align, channels are unified with 1x1 convs, Coordinate
Attention is applied on the pooled 14x14 cubes, then classification and
bounding-box regression heads produce (C+1) logits and C*4 box deltas.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from models.coordinate_attention import CoordinateAttention


class ROIHead(nn.Module):
    def __init__(self, fpn_channels, num_classes: int, cfg: dict):
        super().__init__()
        out_h, out_w = cfg.get("output_size", [14, 14])
        self.roi_channels = 256
        self.out_size = (out_h, out_w)
        self.num_classes = num_classes

        # unify FPN level channels
        self.level_proj = nn.ModuleList([nn.Conv2d(c, self.roi_channels, 1)
                                         for c in fpn_channels])
        hidden = cfg.get("hidden_channels", 512)
        dropout = cfg.get("dropout", 0.2)

        self.spatial = nn.Sequential(
            nn.Conv2d(self.roi_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Coordinate Attention on the aligned ROI cubes (methodology stage)
        ca_reduction = cfg.get("ca_reduction", 16)
        self.coord_attn = CoordinateAttention(256, reduction=ca_reduction)

        self.fc = nn.Sequential(
            nn.Linear(256 * out_h * out_w, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden, num_classes + 1)   # + background
        self.box_head = nn.Linear(hidden, num_classes * 4)   # per-class deltas

        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.normal_(self.box_head.weight, std=0.001)
        nn.init.zeros_(self.cls_head.bias)
        nn.init.constant_(self.box_head.bias, 0.0)

        # FPN level assignment parameters (like detectron2): k0=4, canonical 112
        self.k0 = 4
        self.canonical = 112

    def _assign_levels(self, boxes: torch.Tensor):
        """FPN level assignment: level = k0 + log2(sqrt(area)/canonical)."""
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        levels = torch.floor(self.k0 + torch.log2(areas.sqrt().clamp(min=1e-6) / self.canonical))
        return levels.clamp(0, len(self.level_proj) - 1).long()

    def forward(self, fpn_feats, proposals):
        """fpn_feats: list [P3,P4,P5]; proposals: (N,4) xyxy (plus (N,) batch index if given).

        Returns cls_scores (N,C+1), bbox_deltas (N,C,4).
        """
        device = fpn_feats[0].device
        n = proposals.shape[0]
        if n == 0:
            z = torch.zeros(0, self.num_classes + 1, device=device)
            zb = torch.zeros(0, self.num_classes, 4, device=device)
            return z, zb

        levels = self._assign_levels(proposals)
        pooled = []
        for li, feat in enumerate(fpn_feats):
            f = self.level_proj[li](feat)
            idx = torch.nonzero(levels == li).flatten()
            if idx.numel() == 0:
                continue
            rois_k = torch.cat([torch.zeros(len(idx), 1, device=device),
                                proposals[idx]], dim=1)   # (K,5): batch idx + box
            pooled.append(roi_align(f, rois_k, self.out_size,
                                    spatial_scale=1.0, sampling_ratio=2,
                                    aligned=True))
        # order restoration: roi_align results are collected per level; reassemble
        feat_cubes = torch.empty(n, self.roi_channels, *self.out_size, device=device)
        cursor = 0
        for li, feat in enumerate(fpn_feats):
            idx = torch.nonzero(levels == li).flatten()
            if idx.numel() == 0:
                continue
            feat_cubes[idx] = pooled[cursor]
            cursor += 1

        x = self.spatial(feat_cubes)
        x = self.coord_attn(x)                        # Coordinate Attention
        x = x.flatten(1)
        x = self.fc(x)
        cls_scores = self.cls_head(x)
        box_deltas = self.box_head(x).view(n, self.num_classes, 4)
        return cls_scores, box_deltas


if __name__ == "__main__":
    head = ROIHead((128, 256, 512), num_classes=4,
                   cfg={"output_size": [14, 14], "hidden_channels": 512})
    p3, p4, p5 = (torch.randn(1, 128, 40, 40), torch.randn(1, 256, 20, 20),
                  torch.randn(1, 512, 10, 10))
    props = torch.tensor([[10., 10., 80., 90.], [20., 30., 200., 220.], [0., 0., 30., 30.]])
    cls, box = head([p3, p4, p5], props)
    print(cls.shape, box.shape)
