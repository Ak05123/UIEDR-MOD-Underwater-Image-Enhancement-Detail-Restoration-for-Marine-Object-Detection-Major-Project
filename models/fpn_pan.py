"""FPN + PAN neck (Phases 7 & 9).

FPN: top-down pathway, semantic information flows from deep P5 down to P3.
PAN: bottom-up path-aggregation, refined spatial detail flows back up.
Outputs final P3_out / P4_out / P5_out used by the RPN and ROI Align.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.yolo_backbone import ConvBNSiLU


class FPNPANNeck(nn.Module):
    """Top-down FPN fusion followed by bottom-up PAN aggregation.

    FPN (top-down):  P5 --up+lat--> P4 --up+lat--> P3
    PAN (bottom-up): P3' --down+agg--> P4' --down+agg--> P5'
    """

    def __init__(self, backbone_channels=(128, 256, 512), out_channels=(128, 256, 512)):
        super().__init__()
        c3, c4, c5 = backbone_channels
        o3, o4, o5 = out_channels

        # --- lateral 1x1 projections (FPN)
        self.lat5 = nn.Conv2d(c5, o5, 1)
        self.lat4 = nn.Conv2d(c4, o4, 1)
        self.lat3 = nn.Conv2d(c3, o3, 1)
        # channel projections for top-down fusion (deep -> shallow widths)
        self.proj5to4 = nn.Conv2d(o5, o4, 1)
        self.proj4to3 = nn.Conv2d(o4, o3, 1)

        # --- top-down 3x3 smoothing convs (FPN)
        self.td5 = ConvBNSiLU(o5, o5, 3)
        self.td4 = ConvBNSiLU(o4, o4, 3)
        self.td3 = ConvBNSiLU(o3, o3, 3)

        # --- PAN bottom-up convs (3x3 s2 for downsampling)
        self.pan3 = ConvBNSiLU(o3, o3, 3)
        self.pan_down3 = ConvBNSiLU(o3, o4, 3, 2)
        self.pan4 = ConvBNSiLU(o4 + o4, o4, 3)
        self.pan_down4 = ConvBNSiLU(o4, o5, 3, 2)
        self.pan5 = ConvBNSiLU(o5 + o5, o5, 3)

        self.out_channels = out_channels

    def forward(self, p3, p4, p5):
        # ---------------- FPN (top-down fusion)
        l5 = self.lat5(p5)
        f5 = self.td5(l5)
        l4 = self.lat4(p4) + self.proj5to4(F_up(f5, p4.shape[-2:]))
        f4 = self.td4(l4)
        l3 = self.lat3(p3) + self.proj4to3(F_up(f4, p3.shape[-2:]))
        f3 = self.td3(l3)

        # ---------------- PAN (bottom-up aggregation)
        pan3 = self.pan3(f3)
        d43 = self.pan_down3(pan3) + f4
        pan4 = self.pan4(torch.cat([d43, f4], 1))
        d54 = self.pan_down4(pan4) + f5
        pan5 = self.pan5(torch.cat([d54, f5], 1))

        return pan3, pan4, pan5      # final P3_out, P4_out, P5_out


def F_up(x, size):
    return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    m = FPNPANNeck((128, 256, 512))
    p3, p4, p5 = torch.randn(1, 128, 40, 40), torch.randn(1, 256, 20, 20), torch.randn(1, 512, 10, 10)
    a, b, c = m(p3, p4, p5)
    print(a.shape, b.shape, c.shape)
