#!/usr/bin/env python
"""Prepare LUNA16 for 3D detection (same recipe as MONAI's LUNA16 RetinaNet tutorial).

1. Resample every scan to 0.703125 x 0.703125 x 1.25 mm (RAS), store as int16 NIfTI.
2. Write one datalist JSON per fold: fold k = LUNA16 subset k is the test set,
   the other 9 subsets are training data (MONAI decathlon format).
   Boxes are [coordX, coordY, coordZ, d, d, d] in LUNA16 world coordinates (ITK/LPS);
   training uses affine_lps_to_ras=True, as in the tutorial.

  python dl/prepare.py --luna /workspace/luna16 --out /workspace/prep --workers 48
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SPACING = (0.703125, 0.703125, 1.25)


def resample_one(mhd: str, out: str) -> str:
    import numpy as np
    import torch
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, SaveImaged, Spacingd

    if Path(out).exists():
        return out
    t = Compose([
        LoadImaged(keys=["image"], image_only=False, meta_key_postfix="meta_dict",
                   reader="itkreader", affine_lps_to_ras=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=SPACING, padding_mode="border"),
    ])
    d = t({"image": mhd})
    img = d["image"]
    img.array = np.round(img.array).astype(np.int16)  # HU as int16 (halves disk/IO)
    tmp = Path(out).with_suffix("").with_suffix("")  # strip .nii.gz
    SaveImaged(keys="image", meta_keys="image_meta_dict", output_dir=str(tmp.parent),
               output_postfix="", resample=False, separate_folder=False,
               output_dtype=np.int16, print_log=False)(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luna", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--only-subsets", type=int, nargs="*", default=None, help="resample only these subsets")
    a = ap.parse_args()
    luna, out = Path(a.luna), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(luna / "annotations.csv")

    scans = {}  # uid -> (subset, mhd)
    for s in range(10):
        for p in sorted((luna / f"subset{s}").glob("*.mhd")):
            scans[p.stem] = (s, p)
    print(f"{len(scans)} scans")

    def rel(uid):
        return f"images/{uid}.nii.gz"

    for k in range(10):
        dl = {"training": [], "validation": []}
        for uid, (s, _) in scans.items():
            g = ann[ann.seriesuid == uid]
            item = {
                "image": rel(uid),
                "box": [[r.coordX, r.coordY, r.coordZ, r.diameter_mm, r.diameter_mm, r.diameter_mm]
                        for r in g.itertuples()],
                "label": [0] * len(g),
                "seriesuid": uid,
                "subset": s,
            }
            dl["validation" if s == k else "training"].append(item)
        (out / f"datalist_fold{k}.json").write_text(json.dumps(dl))
    print("wrote datalists")

    if a.skip_images:
        return
    todo = [(str(p), str(out / rel(uid))) for uid, (s, p) in scans.items()
            if a.only_subsets is None or s in a.only_subsets]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(resample_one, m, o) for m, o in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 50 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} resampled", flush=True)


if __name__ == "__main__":
    main()
