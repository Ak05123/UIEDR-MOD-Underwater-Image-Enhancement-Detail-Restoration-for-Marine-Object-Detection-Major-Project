"""Loss functions (Phases 10 & 13): RPN + ROI losses combined."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from models.rpn import deltas_to_boxes, _clip_boxes


def encode_deltas(gt_boxes, anchors):
    """(gt, anchor) xyxy -> (dx,dy,dw,dh) targets."""
    a_w = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-3)
    a_h = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-3)
    a_cx = anchors[:, 0] + a_w / 2
    a_cy = anchors[:, 1] + a_h / 2
    g_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    g_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    g_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-3)
    g_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-3)
    return torch.stack([(g_cx - a_cx) / a_w, (g_cy - a_cy) / a_h,
                        torch.log(g_w / a_w), torch.log(g_h / a_h)], dim=-1)


def rpn_loss(rpn_out, gt_boxes_per_image, fg_iou=0.5, bg_iou=0.4,
             batch_size_per_image=128, pos_fraction=0.5, device="cpu"):
    """RPN objectness BCE + box smooth-L1 on positive anchors.

    rpn_out: dict with 'obj' (A,), 'reg' (A,4), 'anchors' (A,4)
    gt_boxes_per_image: list of (Gi,4) tensors
    """
    obj_logits = rpn_out["obj"]
    reg_pred = rpn_out["reg"]
    anchors = rpn_out["anchors"]
    losses_obj, losses_reg = [], []
    n = max(len(gt_boxes_per_image), 1)
    for gt in gt_boxes_per_image:
        if gt.numel() == 0:
            losses_obj.append(F.binary_cross_entropy_with_logits(
                obj_logits, torch.zeros_like(obj_logits)))
            losses_reg.append(reg_pred.sum() * 0)
            continue
        gt = gt.to(device)
        iou = box_iou(_clip_boxes(anchors.clone(), 10 ** 6), gt)   # (A,G)
        max_iou, gt_idx = iou.max(dim=1)
        pos = max_iou >= fg_iou
        neg = max_iou < bg_iou
        # subsample
        num_pos = min(int(pos.sum()), int(batch_size_per_image * pos_fraction))
        pos_idx = torch.nonzero(pos).flatten()
        neg_idx = torch.nonzero(neg).flatten()
        if pos_idx.numel() > num_pos:
            perm = torch.randperm(pos_idx.numel(), device=device)[:num_pos]
            pos_idx = pos_idx[perm]
        num_neg = min(int(neg.sum()), batch_size_per_image - num_pos)
        if neg_idx.numel() > num_neg:
            perm = torch.randperm(neg_idx.numel(), device=device)[:num_neg]
            neg_idx = neg_idx[perm]
        sampled = torch.cat([pos_idx, neg_idx])
        labels = torch.zeros(len(sampled), device=device)
        labels[: len(pos_idx)] = 1.0
        losses_obj.append(F.binary_cross_entropy_with_logits(obj_logits[sampled], labels))
        if pos_idx.numel() > 0:
            tgt = encode_deltas(gt[gt_idx[pos_idx]], anchors[pos_idx])
            losses_reg.append(F.smooth_l1_loss(reg_pred[pos_idx], tgt, beta=1 / 9))
        else:
            losses_reg.append(reg_pred.sum() * 0)
    return losses_obj[0] if len(losses_obj) == 1 else \
        sum(l for l in losses_obj) / n, sum(l for l in losses_reg) / max(len(losses_reg), 1)


def roi_loss(cls_scores, bbox_pred, labels, regression_targets,
             num_classes, fg_weight=1.0, bg_weight=1.0):
    """Faster R-CNN head loss: CE(cls) + smooth-L1(box) per class.

    cls_scores (N, C+1); bbox_pred (N, C, 4); labels (N,) 0=bg;
    regression_targets (N,4)
    """
    if cls_scores.numel() == 0:
        z = cls_scores.sum() * 0
        return z, z
    loss_cls = F.cross_entropy(cls_scores, labels)
    pos = labels > 0
    if pos.any():
        idx = torch.arange(len(labels), device=labels.device)[pos]
        pred = bbox_pred[idx, labels[pos] - 1]
        loss_box = F.smooth_l1_loss(pred, regression_targets[pos], beta=1 / 9)
    else:
        loss_box = bbox_pred.sum() * 0
    return loss_cls, loss_box
