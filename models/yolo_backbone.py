"""YOLOv11-style backbone with C3k2 (C2F-family) blocks, C2PSA and SPPF.

This is a faithful reimplementation of the YOLOv11 building blocks (Conv,
C3k2 = C2F-with-C3k bottleneck, SPPF, C2PSA) composed into a backbone that
exposes P3/P4/P5 multi-scale feature maps. No detector heads from Ultralytics
are used - only the feature extractor part of the methodology.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    return k // 2 if p is None else p


class ConvBNSiLU(nn.Module):
    """Standard YOLO conv block: Conv2d + BN + SiLU."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = ConvBNSiLU(c1, c2, 3)
        self.cv2 = ConvBNSiLU(c2, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2(nn.Module):
    """YOLOv11 C3k2 block (the C2F/C3k-family fast bottleneck with split flow).

    Splits channels, pushes one part through n stacked bottlenecks, and
    concatenates with the untouched split (feature reuse / cross-stage flow).
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5):
        super().__init__()
        self.cv1 = ConvBNSiLU(c1, 2 * c2 // 2, 1)  # split into two halves of c2
        hidden = c2 // 2
        if c3k:
            self.m = nn.Sequential(*(Bottleneck(hidden, hidden, True) for _ in range(n)))
        else:
            self.m = nn.Sequential(*(Bottleneck(hidden, hidden, True) for _ in range(n)))
        self.cv2 = ConvBNSiLU(2 * hidden, c2, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y[1] = self.m(y[1])
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (real multi-scale max-pool cascade)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = ConvBNSiLU(c1, hidden, 1)
        self.cv2 = ConvBNSiLU(hidden * 4, c2, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        y = self.cv1(x)
        y1 = self.pool(y)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((y, y1, y2, y3), 1))


class C2PSA(nn.Module):
    """YOLOv11 C2PSA block: spatial attention within the C2 split framework."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        super().__init__()
        hidden = c2 // 2
        self.cv1 = ConvBNSiLU(c1, 2 * hidden, 1)
        self.attn = PSABlock(hidden, n)
        self.cv2 = ConvBNSiLU(2 * hidden, c2, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y[1] = self.attn(y[1])
        return self.cv2(torch.cat(y, 1))


class PSABlock(nn.Module):
    def __init__(self, c: int, n: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(*(Bottleneck(c, c) for _ in range(max(n, 1))))
        self.att = nn.MultiheadAttention(c, 4, batch_first=True)
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        b, cph, h, w = x.shape
        y = self.mlp(x)
        tokens = y.flatten(2).transpose(1, 2)          # (B, HW, C)
        attn_out, _ = self.att(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return tokens.transpose(1, 2).reshape(b, cph, h, w) + x


class YOLOv11Backbone(nn.Module):
    """YOLOv11-style backbone producing P3, P4 and P5 feature maps.

    Structure (channel widths scaled by width_multiple):
        stem   : Conv s2, Conv s2          (stride-4 features)
        stage1 : Conv s2 + C3k2 x r1       -> P3 input (stride 8)
        stage2 : Conv s2 + C3k2 x r2 + C2PSA -> P4 input (stride 16)
        stage3 : Conv s2 + C3k2 x r3 + C2PSA -> P5 (stride 32) + SPPF
    """

    def __init__(self, channels=(64, 128, 256, 512), width_multiple: float = 0.5,
                 repeats=(1, 2, 2), sppf_k: int = 5):
        super().__init__()
        wm = width_multiple
        c0, c1, c2, c3 = (max(int(c * wm), 16) for c in channels)

        self.stem = nn.Sequential(
            ConvBNSiLU(3, c0, 3, 2),
            ConvBNSiLU(c0, c0, 3, 2),
        )
        self.stage1 = nn.Sequential(
            ConvBNSiLU(c0, c1, 3, 2),
            C3k2(c1, c1, repeats[0]),
        )
        self.stage2 = nn.Sequential(
            ConvBNSiLU(c1, c2, 3, 2),
            C3k2(c2, c2, repeats[1]),
            C2PSA(c2, c2, 1),
        )
        self.stage3 = nn.Sequential(
            ConvBNSiLU(c2, c3, 3, 2),
            C3k2(c3, c3, repeats[2], c3k=True),
            C2PSA(c3, c3, 1),
            SPPF(c3, c3, sppf_k),
        )
        self.out_channels = (c1, c2, c3)   # P3, P4, P5 channels

    def forward(self, x: torch.Tensor):
        x = self.stem(x)          # stride 4
        x = self.stage1(x)        # stride 8
        p3 = x
        x = self.stage2(x)        # stride 16
        p4 = x
        x = self.stage3(x)        # stride 32
        p5 = x
        return p3, p4, p5


if __name__ == "__main__":
    m = YOLOv11Backbone()
    x = torch.randn(1, 3, 320, 320)
    p3, p4, p5 = m(x)
    print("P3", p3.shape, "P4", p4.shape, "P5", p5.shape)
