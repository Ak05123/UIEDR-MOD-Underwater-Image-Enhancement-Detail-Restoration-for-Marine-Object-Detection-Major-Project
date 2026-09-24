"""Coordinate Attention (Phase 12) - Hou et al., CVPR 2021.

Applied on ROI-aligned feature cubes (fixed size, BxCxHxW). Separately encodes
directional (H and W) context, producing coordinate-aware channel attention
while preserving positional information.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.conv_reduce = nn.Conv2d(channels, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        # directional pooling convs
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # coordinate information embedding: directional average pooling
        pool_h = x.mean(dim=3, keepdim=True)                    # (B,C,H,1)
        pool_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B,C,W,1)

        # shared 1x1 transform on concatenated directional descriptors
        proj = self.act(self.bn1(self.conv_reduce(
            torch.cat([pool_h, pool_w], dim=2)                  # (B,C,H+W,1)
        )))
        proj_h, proj_w = torch.split(proj, [h, w], dim=2)

        # directional attention maps (sigmoid activation)
        att_h = torch.sigmoid(self.conv_h(proj_h))              # (B,C,H,1)
        att_w = torch.sigmoid(self.conv_w(proj_w).permute(0, 1, 3, 2))  # (B,C,1,W)

        return x * att_h * att_w


if __name__ == "__main__":
    m = CoordinateAttention(256)
    x = torch.randn(4, 256, 14, 14)
    print(m(x).shape)
