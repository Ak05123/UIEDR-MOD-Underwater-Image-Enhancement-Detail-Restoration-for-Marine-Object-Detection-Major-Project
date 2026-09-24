"""One-batch training smoke test on real DUO data.

Not a training run: loads 4 samples, one forward+backward+step, finite-loss
check, temporary checkpoint saved to checkpoints/smoke_test.pt.
"""
import torch

from training.trainer import load_config, build_loaders, build_model, make_optimizer, validate_dataset

cfg = load_config("config.yaml")
reports, report_path = validate_dataset(cfg)
print(f"[1] dataset validation -> {report_path}")
for name, r in reports.items():
    print(f"    {name}: images={r['num_images_declared']} anns={r['num_annotations']} "
          f"missing={len(r['missing_images'])} invalid_boxes={r['invalid_boxes']} "
          f"unknown_ids={r['unknown_class_ids']} classes={r['classes']}")

train_ds, _, train_dl, _ = build_loaders(cfg)
batch = next(iter(train_dl))
print(f"[2] batch image shape: {tuple(batch['images'].shape)}")
print(f"[3] targets: {[len(b) for b in batch['boxes']]} boxes; labels {[l.tolist() for l in batch['labels']]}")
assert batch["boxes"][0].shape[1] == 4

device = torch.device("cpu")
model = build_model(cfg, device)
model.train()
opt = make_optimizer(model, cfg["training"])

images = batch["images"].to(device)
gt = [b.to(device) for b in batch["boxes"]]
gl = [l.to(device) for l in batch["labels"]]

losses = model(images, gt_boxes=gt, gt_labels=gl)
total = sum(losses.values())
print("[4] losses:")
for k, v in losses.items():
    print(f"    {k}: {v.item():.6f}")
print(f"    total: {total.item():.6f}")
assert all(torch.isfinite(v) for v in losses.values()) and torch.isfinite(total)

opt.zero_grad(set_to_none=True)
total.backward()
print("[5] backward() OK")
opt.step()
print("[6] optimizer.step() OK")

path = model_save = "checkpoints/smoke_test.pt"
torch.save({
    "epoch": 0, "smoke_test": True,
    "model": model.state_dict(), "optimizer": opt.state_dict(),
    "config": cfg, "classes": cfg["classes"],
    "image_size": cfg["dataset"]["image_size"],
    "losses": {k: v.item() for k, v in losses.items()},
    "metrics": {"val_total": float("inf")},
}, path)
print(f"[7] smoke checkpoint saved: {path}")
print("[8] SMOKE TEST PASSED")
