#!/usr/bin/env python
"""Train + test one fold of the 3D RetinaNet nodule detector.

Training recipe = MONAI LUNA16 tutorial (16 GB config): random 192x192x80 patches
(4 per scan, half containing a nodule), flips/rot90/zoom/rotation/intensity
augmentation, ATSS matching, hard-negative sampling, SGD + AMP. Only the backbone
changes between runs (see model.py). Model selection uses the LUNA16 CPM on a
held-out 5% of the training scans; the test subset is only touched once, at the end.

  python dl/train.py --fold 0 --backbone ctfm --epochs 100 --out /workspace/runs
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from generate_transforms import (  # noqa: E402
    generate_detection_inference_transform,
    generate_detection_train_transform,
)
from model import build_detector  # noqa: E402

HU_RANGE = (-1024.0, 2048.0)  # CT-FM's input range, used for every backbone (same input for all)
PATCH = [192, 192, 80]
VAL_PATCH = [512, 512, 208]


def intensity():
    from monai.transforms import ScaleIntensityRanged
    global HU_RANGE

    return ScaleIntensityRanged(keys=["image"], a_min=HU_RANGE[0], a_max=HU_RANGE[1], b_min=0.0, b_max=1.0, clip=True)


def loaders(datalist, base, batch, workers, val_frac=0.05, seed=0, only_existing=False):
    from monai.data import DataLoader, Dataset, load_decathlon_datalist
    from monai.data.utils import no_collation

    train = load_decathlon_datalist(datalist, is_segmentation=True, data_list_key="training", base_dir=base)
    test = load_decathlon_datalist(datalist, is_segmentation=True, data_list_key="validation", base_dir=base)
    train = [d for d in train if len(d["box"])]  # as in MONAI's LUNA16 split: training scans with >= 1 nodule
    if only_existing:  # use the scans that have been resampled so far
        train = [d for d in train if Path(d["image"]).exists()]
    rng = random.Random(seed)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(train))))
    val = [train[i] for i in idx[:n_val]]
    train = [train[i] for i in idx[n_val:]]
    tt = generate_detection_train_transform("image", "box", "label", "cccwhd", intensity(), PATCH, batch,
                                            point_key="points", affine_lps_to_ras=True, amp=True)
    tl = DataLoader(Dataset(train, tt), batch_size=1, shuffle=True, num_workers=workers,
                    pin_memory=True, collate_fn=no_collation, persistent_workers=not only_existing)
    return tl, val, test


def predict(detector, items, base, workers=4, amp=True) -> pd.DataFrame:
    """Sliding-window inference on whole scans -> LUNA16 submission rows (world coords, LPS)."""
    from monai.data import DataLoader, Dataset
    from monai.data.utils import no_collation

    inf_t, post_t = generate_detection_inference_transform(
        "image", "pred_box", "pred_label", "pred_score", "cccwhd", intensity(), affine_lps_to_ras=True, amp=amp)
    dl = DataLoader(Dataset(items, inf_t), batch_size=1, num_workers=workers, collate_fn=no_collation)
    rows = []
    detector.eval()
    dev = next(detector.network.parameters()).device
    with torch.no_grad():
        for batch in dl:
            for d in batch:
                img = d["image"].to(dev)
                use_inferer = img[0].numel() >= np.prod(VAL_PATCH)
                with torch.autocast("cuda", enabled=amp):
                    out = detector([img], use_inferer=use_inferer)[0]
                d["pred_box"] = out[detector.target_box_key].float()
                d["pred_label"] = out[detector.target_label_key]
                d["pred_score"] = out[detector.pred_score_key].float()
                d = post_t(d)
                boxes = d["pred_box"].cpu().numpy().reshape(-1, 6)
                scores = d["pred_score"].cpu().numpy().reshape(-1)
                for b, s in zip(boxes, scores):
                    rows.append({"seriesuid": d["seriesuid"], "coordX": b[0], "coordY": b[1], "coordZ": b[2],
                                 "diameter_mm": float(np.mean(b[3:6])), "probability": float(s)})
    return pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "probability"])


def cpm(sub: pd.DataFrame, uids, ann, exc) -> float:
    from lungnod.evaluation import evaluate

    return evaluate(sub, ann, exc, list(uids))["cpm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--backbone", required=True, choices=["resnet50", "ctfm", "ctfm_wbs", "ctfm_rand"])
    ap.add_argument("--data", default="/workspace/prep")
    ap.add_argument("--luna", default="/workspace/luna16")
    ap.add_argument("--out", default="/workspace/runs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=4, help="patches per scan per step")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--mem-fraction", type=float, default=0.0, help="cap GPU memory (e.g. 0.5 = 12 GB on a 4090)")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: steps per epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=0, help="stop training after N minutes")
    ap.add_argument("--only-existing", action="store_true", help="train on resampled scans present now; re-check each epoch")
    ap.add_argument("--no-val", action="store_true", help="skip validation; test with the last checkpoint")
    ap.add_argument("--test-only", action="store_true", help="skip training; test the saved best.pt")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="keep the pretrained encoder fixed; train only FPN + detection head")
    ap.add_argument("--tag", default="", help="suffix for the run folder")
    ap.add_argument("--hu", type=float, nargs=2, default=None, help="intensity window (default: CT-FM's -1024 2048)")
    ap.add_argument("--weights", default=None, help="test-only: load these network weights instead of best.pt")
    a = ap.parse_args()

    from monai.utils import set_determinism

    global HU_RANGE
    if a.hu:
        HU_RANGE = tuple(a.hu)

    set_determinism(seed=a.seed)
    torch.backends.cudnn.benchmark = True
    if a.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    run = Path(a.out) / f"{a.backbone}{a.tag}_fold{a.fold}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=1))
    log = open(run / "log.txt", "a")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    ann = pd.read_csv(Path(a.luna) / "annotations.csv")
    exc = pd.read_csv(Path(a.luna) / "annotations_excluded.csv") if (Path(a.luna) / "annotations_excluded.csv").exists() else None
    dl_path = Path(a.data) / f"datalist_fold{a.fold}.json"
    tl, val_items, test_items = loaders(dl_path, a.data, a.batch, a.workers,
                                        val_frac=0.0 if a.no_val else 0.05, only_existing=a.only_existing)
    say(f"fold {a.fold} backbone {a.backbone}: {len(tl.dataset)} train, {len(val_items)} val, {len(test_items)} test scans")

    det, _ = build_detector(a.backbone)
    det.set_sliding_window_inferer(roi_size=VAL_PATCH, overlap=0.25, sw_batch_size=1, mode="gaussian", device="cuda")
    dev = torch.device("cuda")
    det = det.to(dev)
    body = det.network.feature_extractor.body
    if a.freeze_backbone:
        for p_ in body.parameters():
            p_.requires_grad_(False)
    params = [p_ for p_ in det.network.parameters() if p_.requires_grad]
    say(f"trainable parameters: {sum(p_.numel() for p_ in params) / 1e6:.1f}M"
        f" of {sum(p_.numel() for p_ in det.network.parameters()) / 1e6:.1f}M")
    opt = torch.optim.SGD(params, a.lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    steps_per_epoch = len(tl) if not a.max_steps else min(a.max_steps, len(tl))

    def lr_at(ep_float):  # linear warmup, then cosine to 0
        if ep_float < a.warmup:
            return a.lr * (ep_float + 1e-3) / a.warmup
        p = (ep_float - a.warmup) / max(1, a.epochs - a.warmup)
        return 0.5 * a.lr * (1 + math.cos(math.pi * min(p, 1.0)))

    scaler = torch.amp.GradScaler("cuda")
    best, start = -1.0, 0
    ck = run / "last.pt"
    if ck.exists():  # resume
        s = torch.load(ck, map_location="cpu", weights_only=False)
        det.network.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        start, best = s["epoch"] + 1, s["best"]
        say(f"resumed at epoch {start}, best val CPM {best:.3f}")

    t_start = time.time()
    for ep in range(start, a.epochs if not a.test_only else start):
        if a.time_budget and (time.time() - t_start) / 60 > a.time_budget:
            say(f"time budget reached after {ep} epochs")
            break
        if a.only_existing and ep > start:
            tl, _, _ = loaders(dl_path, a.data, a.batch, a.workers, val_frac=0.0 if a.no_val else 0.05,
                               only_existing=True)
            steps_per_epoch = len(tl) if not a.max_steps else min(a.max_steps, len(tl))
        det.train()
        if a.freeze_backbone:
            body.eval()  # also keep the encoder's normalisation statistics fixed
        t0, tot, n = time.time(), 0.0, 0
        for step, batch in enumerate(tl):
            if step >= steps_per_epoch:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(ep + step / steps_per_epoch)
            inputs = [d["image"].to(dev) for b in batch for d in b]
            targets = [dict(label=d["label"].to(dev), box=d["box"].to(dev)) for b in batch for d in b]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                out = det(inputs, targets)
                loss = out[det.cls_key] + out[det.box_reg_key]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); n += 1
        msg = f"epoch {ep + 1}/{a.epochs} loss {tot / max(n, 1):.4f} lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s"
        if not a.no_val and ((ep + 1) % a.val_every == 0 or ep + 1 == a.epochs):
            t1 = time.time()
            sub = predict(det, val_items, a.data)
            v = cpm(sub, [d["seriesuid"] for d in val_items], ann, exc)
            msg += f" | val CPM {v:.3f} ({time.time() - t1:.0f}s)"
            if v > best:
                best = v
                torch.save({"model": det.network.state_dict(), "epoch": ep, "val_cpm": v}, run / "best.pt")
                msg += " *"
        msg += f" | {len(tl)} train scans"
        say(msg)
        torch.save({"model": det.network.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best": best}, ck)
        if a.no_val:
            torch.save({"model": det.network.state_dict(), "epoch": ep, "val_cpm": float("nan")}, run / "best.pt")

    # final test on the held-out subset, with the best checkpoint
    if a.weights:
        w = torch.load(a.weights, map_location="cpu", weights_only=False)
        w = w.get("model", w) if isinstance(w, dict) else w.state_dict()
        missing, unexpected = det.network.load_state_dict(w, strict=False)
        say(f"loaded {a.weights}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        s = {"epoch": -1, "val_cpm": float("nan")}
    else:
        s = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        det.network.load_state_dict(s["model"])
    sub = predict(det, test_items, a.data)
    sub.to_csv(run / f"submission_fold{a.fold}.csv", index=False)
    t = cpm(sub, [d["seriesuid"] for d in test_items], ann, exc)
    say(f"TEST fold {a.fold} (subset {a.fold}): CPM {t:.3f}  (best val CPM {s['val_cpm']:.3f} at epoch {s['epoch'] + 1})")


if __name__ == "__main__":
    main()
