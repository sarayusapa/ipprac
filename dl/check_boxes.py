#!/usr/bin/env python
"""Sanity check: do the ground-truth boxes land on nodules after resampling + transforms?
Loads scans with the validation transform (boxes -> image voxel coords) and reports the
HU at each box centre (3x3x3 mean). Nodules are soft tissue: expect > -600 HU for most."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from generate_transforms import generate_detection_val_transform  # noqa: E402
from monai.data import load_decathlon_datalist  # noqa: E402
from monai.transforms import ScaleIntensityRanged  # noqa: E402

data = sys.argv[1] if len(sys.argv) > 1 else "/workspace/prep"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
items = [d for d in load_decathlon_datalist(f"{data}/datalist_fold9.json", True, "training", data) if len(d["box"]) and Path(d["image"]).exists()][:n]
ident = ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1, b_min=0, b_max=1, clip=False)  # keep HU
t = generate_detection_val_transform("image", "box", "label", "cccwhd", ident, affine_lps_to_ras=True, amp=False)
hu = []
for d in items:
    o = t(d)
    img = o["image"][0].numpy()
    for b in o["box"].numpy():
        c = np.round((b[:3] + b[3:]) / 2).astype(int)  # standard mode xyzxyz -> centre voxel
        sl = tuple(slice(max(ci - 1, 0), ci + 2) for ci in c)
        hu.append(float(img[sl].mean()))
hu = np.array(hu)
print(f"{len(hu)} nodules in {len(items)} scans: median centre HU {np.median(hu):.0f}, "
      f"fraction > -600 HU: {(hu > -600).mean():.2f}, fraction < -900 (air): {(hu < -900).mean():.2f}")
