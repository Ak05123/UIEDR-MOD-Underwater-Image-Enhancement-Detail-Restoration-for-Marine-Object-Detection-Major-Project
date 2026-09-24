"""Region Proposal Network (Phase 10).

Multi-scale RPN over P3/P4/P5: per-level objectness + box regression heads,
anchor generation, delta decoding, clipping, filtering and NMS.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_iou

from models.yolo_backbone import ConvBNSiLU


def generate_anchors(feat_h, feat_w, stride, anchor_sizes, aspect_ratios, device):
    """Anchors in absolute image coordinates (x1,y1,x2,y2) for one level."""
    anchors = []
    cy = (torch.arange(feat_h, device=device) + 0.5) * stride
    cx = (torch.arange(feat_w, device=device) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")   # (H,W)
    for s in anchor_sizes:
        for r in aspect_ratios:
            h = s * (r ** 0.5)
            w = s / (r ** 0.5)
            a = torch.stack([
                grid_x - w / 2, grid_y - h / 2,
                grid_x + w / 2, grid_y + h / 2], dim=-1)     # (H,W,4) broadcasts
            anchors.append(a.reshape(-1, 4))
    return torch.cat(anchors, 0)  # (H*W*num_anchors, 4)


def deltas_to_boxes(deltas, anchors):
    """Decode (dx,dy,dw,dh) relative to anchors -> xyxy boxes."""
    a_w = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-3)
    a_h = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-3)
    a_cx = anchors[:, 0] + a_w / 2
    a_cy = anchors[:, 1] + a_h / 2
    cx = deltas[:, 0] * a_w + a_cx
    cy = deltas[:, 1] * a_h + a_cy
    dw = deltas[:, 2].clamp(max=4.134)   # exp(4.134) ~ 62
    dh = deltas[:, 3].clamp(max=4.134)
    w = a_w * torch.exp(dw)
    h = a_h * torch.exp(dh)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class RPN(nn.Module):
    """Region Proposal Network over multi-scale FPN features."""

    def __init__(self, in_channels_list, cfg: dict, image_size: int):
        super().__init__()
        self.cfg = cfg
        self.image_size = image_size
        anchor_sizes = cfg["anchor_sizes"]
        ratios = cfg["aspect_ratios"]
        self.strides = [image_size // (2 ** (4 + i)) for i in range(len(in_channels_list))]
        self.num_anchors_per_level = [len(s) * len(r) for s, r in zip(anchor_sizes, ratios)]

        # shared conv + per-level heads
        self.convs = nn.ModuleList([nn.Sequential(
            ConvBNSiLU(c, c, 3), ConvBNSiLU(c, c, 3)) for c in in_channels_list])
        self.obj_heads = nn.ModuleList([nn.Conv2d(c, n, 1) for c, n in
                                        zip(in_channels_list, self.num_anchors_per_level)])
        self.reg_heads = nn.ModuleList([nn.Conv2d(c, 4 * n, 1) for c, n in
                                        zip(in_channels_list, self.num_anchors_per_level)])
        for head in self.obj_heads + self.reg_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

        self.pre_nms_top_n = cfg.get("pre_nms_top_n", 2000)
        self.post_nms_top_n = cfg.get("post_nms_top_n", 300)
        self.obj_thr = cfg.get("objectness_threshold", 0.5)
        self.nms_thr = cfg.get("nms_threshold", 0.7)

    @property
    def total_num_anchors(self):
        return sum(self.num_anchors_per_level)

    def _all_anchors(self, feats, device):
        return [generate_anchors(f.shape[2], f.shape[3], s, sz, ar, device)
                for f, s, sz, ar in zip(feats, self.strides,
                                        self.cfg["anchor_sizes"], self.cfg["aspect_ratios"])]

    def forward(self, feats):
        """Returns (proposals (N,4), scores (N,), predictions dict for loss)."""
        device = feats[0].device
        anchors = self._all_anchors(feats, device)

        obj_flat, reg_flat = [], []
        for feat, conv, oh, rh in zip(feats, self.convs, self.obj_heads, self.reg_heads):
            x = conv(feat)
            obj_flat.append(oh(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1))
            reg_flat.append(rh(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4))

        # batch of 1 during inference; training uses per-image loops via loss fn
        pred_obj = torch.cat(obj_flat, 1)[0]           # (A,)
        pred_reg = torch.cat(reg_flat, 1)[0]           # (A,4)
        all_anchors = torch.cat(anchors, 0)            # (A,4)

        scores = pred_obj.sigmoid()
        # objectness threshold pre-filter, decode, clip, then NMS
        keep = scores > self.obj_thr
        kept_anchors = all_anchors[keep]
        boxes = deltas_to_boxes(pred_reg[keep], kept_anchors)
        boxes = torch.stack([
            boxes[:, 0].clamp(0, self.image_size), boxes[:, 1].clamp(0, self.image_size),
            boxes[:, 2].clamp(0, self.image_size), boxes[:, 3].clamp(0, self.image_size)], dim=-1)
        # drop degenerate boxes together with their scores
        valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes, kept_scores = boxes[valid], scores[keep][valid]

        if boxes.numel() == 0:
            return torch.zeros(0, 4, device=device), torch.zeros(0, device=device), \
                {"obj": pred_obj, "reg": pred_reg, "anchors": all_anchors}

        keep_n = nms(boxes, kept_scores, self.nms_thr)[: self.post_nms_top_n]
        return boxes[keep_n], kept_scores[keep_n], \
            {"obj": pred_obj, "reg": pred_reg, "anchors": all_anchors}


def _clip_boxes(boxes, size):
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, size)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, size)
    # drop degenerate
    valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
    return boxes[valid]
