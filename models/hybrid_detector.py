"""HybridYOLOFasterRCNN (Phase 14) - the complete model pipeline.

image -> enhancement (classical, applied in dataloader/preprocess)
      -> ResidualCNN (detail restoration)
      -> YOLOv11 backbone (C3k2/C2PSA/SPPF)
      -> FPN/PAN neck -> P3/P4/P5
      -> RPN proposals
      -> multi-level ROI Align
      -> Coordinate Attention
      -> classification + bbox regression
      -> NMS -> final detections

Training forward computes RPN + ROI losses; eval forward returns detections.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_iou

from models.residual_cnn import ResidualCNN
from models.yolo_backbone import YOLOv11Backbone
from models.fpn_pan import FPNPANNeck
from models.rpn import RPN, deltas_to_boxes, _clip_boxes
from models.roi_head import ROIHead
from models.losses import rpn_loss, roi_loss, encode_deltas


class HybridYOLOFasterRCNN(nn.Module):
    def __init__(self, num_classes: int, cfg: dict):
        super().__init__()
        self.num_classes = num_classes
        mcfg = cfg["model"]
        self.cfg = cfg
        img_size = cfg["dataset"]["image_size"]

        rc = mcfg.get("residual_cnn", {})
        self.residual_cnn = ResidualCNN(
            rc.get("in_channels", 3), rc.get("feature_channels", 32),
            rc.get("num_blocks", 3), rc.get("kernel_size", 3))
        self.residual_enabled = rc.get("enabled", True)

        bcfg = mcfg["backbone"]
        self.backbone = YOLOv11Backbone(
            tuple(bcfg["channels"]), bcfg.get("width_multiple", 0.5),
            tuple(mcfg.get("c2f", {}).get("repeats", [1, 2, 2])),
            mcfg.get("sppf", {}).get("kernel_size", 5))

        neck_out = mcfg["neck"]["out_channels"]
        self.neck = FPNPANNeck(self.backbone.out_channels, tuple(neck_out))
        self.rpn = RPN(tuple(neck_out), mcfg["rpn"], img_size)
        ca_cfg = dict(mcfg["roi"])
        ca_cfg["ca_reduction"] = mcfg.get("coordinate_attention", {}).get("reduction_ratio", 16)
        self.roi_head = ROIHead(tuple(neck_out), num_classes, ca_cfg)

        self.cls_thr = cfg["inference"]["confidence_threshold"]
        self.nms_thr = cfg["inference"]["nms_threshold"]
        self.max_det = cfg["inference"]["max_detections"]
        self.rpn_score_thr = mcfg["rpn"].get("objectness_threshold", 0.5)

    # ----------------------------------------------------------- backbone stages
    def extract_features(self, x: torch.Tensor):
        if self.residual_enabled:
            x = self.residual_cnn(x)
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return p3, p4, p5

    def shape_summary(self, x: torch.Tensor):
        """Debug utility: returns dict of tensor shapes through the full pipeline."""
        out = {"input": tuple(x.shape)}
        if self.residual_enabled:
            x = self.residual_cnn(x)
            out["residual_output"] = tuple(x.shape)
        p3, p4, p5 = self.backbone(x)
        out["backbone_P3"] = tuple(p3.shape)
        out["backbone_P4"] = tuple(p4.shape)
        out["backbone_P5"] = tuple(p5.shape)
        p3n, p4n, p5n = self.neck(p3, p4, p5)
        out["FPN_PAN_P3"] = tuple(p3n.shape)
        out["FPN_PAN_P4"] = tuple(p4n.shape)
        out["FPN_PAN_P5"] = tuple(p5n.shape)
        props, scores, rpn_out = self.rpn([p3n, p4n, p5n])
        out["RPN_proposals"] = tuple(props.shape)
        cls, box = self.roi_head([p3n, p4n, p5n], props)
        out["ROI_cls"] = tuple(cls.shape)
        out["ROI_box"] = tuple(box.shape)
        return out

    # ---------------------------------------------------------------- training
    def forward(self, images: torch.Tensor, gt_boxes=None, gt_labels=None):
        """
        Training: images (B,3,H,W); gt_boxes list of (Gi,4); gt_labels list of (Gi,)
        Returns dict of losses.

        Eval (gt_boxes None): returns list of dict(boxes, scores, labels) per image.
        """
        p3, p4, p5 = self.extract_features(images)
        feats = [p3, p4, p5]

        if gt_boxes is None:
            return self._inference(feats, images.shape[-1])

        device = images.device
        B = images.shape[0]
        loss_obj_list, loss_reg_list = [], []
        all_cls, all_box, all_labels, all_targets = [], [], [], []

        for i in range(B):
            gt = gt_boxes[i].to(device)
            labs = gt_labels[i].to(device)

            # ---- RPN proposals per image (batch dim 1 view of features)
            feats_i = [f[i: i + 1] for f in feats]
            props, scores, rpn_out = self.rpn(feats_i)
            lo_obj, lo_reg = rpn_loss(rpn_out, [gt], device=device)
            loss_obj_list.append(lo_obj)
            loss_reg_list.append(lo_reg)

            # if RPN produced no proposals (possible early in training) skip ROI
            if props.shape[0] == 0:
                continue
            if gt.numel() == 0:
                continue

            # ---- match proposals to GT for ROI training
            with torch.no_grad():
                iou = box_iou(props, gt)                      # (P,G)
                max_iou, gt_idx = iou.max(dim=1)
                pos = max_iou >= 0.5
                neg = (max_iou < 0.5)
                num_pos = min(int(pos.sum()), 128)
                pos_idx = torch.nonzero(pos).flatten()
                neg_idx = torch.nonzero(neg).flatten()
                if pos_idx.numel() > num_pos:
                    pos_idx = pos_idx[torch.randperm(pos_idx.numel())[:num_pos]]
                num_neg = min(int(neg.sum()), 256 - num_pos)
                if neg_idx.numel() > num_neg:
                    neg_idx = neg_idx[torch.randperm(neg_idx.numel())[:num_neg]]
                sampled = torch.cat([pos_idx, neg_idx])

                labels = torch.zeros(len(sampled), dtype=torch.long, device=device)
                if pos_idx.numel():
                    labels[: len(pos_idx)] = labs[gt_idx[pos_idx]] + 1   # shift: bg=0
                targets = torch.zeros(len(sampled), 4, device=device)
                if pos_idx.numel():
                    targets[: len(pos_idx)] = encode_deltas(gt[gt_idx[pos_idx]],
                                                            props[pos_idx])

            cls_scores, bbox_deltas = self.roi_head(feats, props[sampled])
            all_cls.append(cls_scores)
            all_box.append(bbox_deltas)
            all_labels.append(labels)
            all_targets.append(targets)

        losses = {}
        losses["rpn_objectness"] = sum(loss_obj_list) / max(len(loss_obj_list), 1)
        losses["rpn_box"] = sum(loss_reg_list) / max(len(loss_reg_list), 1)
        if all_cls:
            losses["roi_cls"], losses["roi_box"] = roi_loss(
                torch.cat(all_cls), torch.cat(all_box),
                torch.cat(all_labels), torch.cat(all_targets), self.num_classes)
        else:
            z = p3.sum() * 0
            losses["roi_cls"], losses["roi_box"] = z, z
        return losses

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def _inference(self, feats, img_size):
        props, scores, _ = self.rpn(feats)
        results = []
        if props.shape[0] == 0:
            return [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                     "labels": torch.zeros(0, dtype=torch.long)}]
        cls_scores, bbox_deltas = self.roi_head(feats, props)
        probs = F.softmax(cls_scores, dim=-1)
        score_bg, best_cls = probs[:, 0], probs[:, 1:].argmax(dim=1)
        best_score = probs[:, 1:].max(dim=1).values

        keep = (best_cls >= 0) & (best_score > self.cls_thr)
        boxes = props[keep]
        labels = best_cls[keep]
        confs = best_score[keep]
        if boxes.shape[0]:
            # refine boxes with predicted deltas
            pw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-3)
            ph = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-3)
            cx = boxes[:, 0] + pw / 2
            cy = boxes[:, 1] + ph / 2
            deltas = bbox_deltas[keep]
            dx = deltas[torch.arange(len(keep)), labels] if False else None
            # gather per-class deltas
            deltas_sel = bbox_deltas[keep][torch.arange(boxes.shape[0]), labels]
            rx = cx + deltas_sel[:, 0] * pw
            ry = cy + deltas_sel[:, 1] * ph
            rw = torch.exp(deltas_sel[:, 2].clamp(max=4.134)) * pw
            rh = torch.exp(deltas_sel[:, 3].clamp(max=4.134)) * ph
            refined = torch.stack([rx - rw / 2, ry - rh / 2, rx + rw / 2, ry + rh / 2], -1)
            refined = _clip_boxes(refined, img_size)
            # class-wise NMS
            final_b, final_s, final_l = [], [], []
            for c in labels.unique():
                idx = labels == c
                ki = nms(refined[idx], confs[idx], self.nms_thr)[: self.max_det]
                final_b.append(refined[idx][ki])
                final_s.append(confs[idx][ki])
                final_l.append(labels[idx][ki])
            results.append({
                "boxes": torch.cat(final_b).cpu(),
                "scores": torch.cat(final_s).cpu(),
                "labels": torch.cat(final_l).cpu(),
            })
        else:
            results.append({
                "boxes": torch.zeros(0, 4).cpu(),
                "scores": torch.zeros(0).cpu(),
                "labels": torch.zeros(0, dtype=torch.long).cpu(),
            })
        return results


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    model = HybridYOLOFasterRCNN(num_classes=4, cfg=cfg)
    model.eval()
    x = torch.randn(1, 3, 320, 320)
    print(model.shape_summary(x))
    det = model(x)
    print("inference ok:", det[0]["boxes"].shape)
    gt_b = [torch.tensor([[50., 60., 150., 180.], [200., 200., 300., 300.]])]
    gt_l = [torch.tensor([0, 1])]
    losses = model(x, gt_b, gt_l)
    print({k: v.item() for k, v in losses.items()})
