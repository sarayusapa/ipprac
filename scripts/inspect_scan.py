#!/usr/bin/env python
"""Run P0 on one scan and save figures for every step.

  python scripts/inspect_scan.py --mhd data/luna16/subset0/<uid>.mhd \
      --annotations data/luna16/annotations.csv --out outputs/inspect

Saves: slice overlays (lung mask, blobs, accepted tracks, ground truth) at each
annotated nodule (or the top tracks), Figure-2-style strips, radius profiles,
and the track table with explanations.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lungnod import P0Config, load_scan, run_p0
from lungnod.viz import nodule_strip, show_slice, track_profiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhd", required=True)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/inspect")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    cfg = P0Config.from_yaml(args.config)
    scan = load_scan(args.mhd)
    res = run_p0(scan, cfg, keep_intermediate=True)
    out = Path(args.out) / scan.seriesuid[-12:]
    out.mkdir(parents=True, exist_ok=True)

    t = res.table.sort_values("probability", ascending=False)
    t.to_csv(out / "tracks.csv", index=False)
    print(f"shape {scan.hu.shape}, spacing {scan.spacing}, timings {res.timings}")
    print(f"{len(t)} tracks, {int((t.probability >= 0.5).sum())} accepted")
    cols = ["track_id", "vox_z", "max_diameter_mm", "lifetime_mm", "aspect",
            "drift_ratio", "sphere_gain", "probability", "explanation"]
    print(t[cols].head(15).to_string(index=False))

    ann = pd.read_csv(args.annotations) if args.annotations else None
    focus = []
    if ann is not None:
        a = ann[ann.seriesuid == scan.seriesuid]
        if len(a):
            focus = [("gt", v) for v in scan.world_to_voxel(a[["coordX", "coordY", "coordZ"]].to_numpy())]
    focus += [("track", (r.vox_z, r.vox_y, r.vox_x)) for r in t.head(args.top).itertuples()]

    for i, (kind, zyx) in enumerate(focus):
        z = int(round(zyx[0]))
        fig, ax = plt.subplots(figsize=(7, 7))
        show_slice(scan, z, res, ann, ax=ax)
        fig.savefig(out / f"{i:02d}_{kind}_slice{z}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        nodule_strip(scan, zyx, path=out / f"{i:02d}_{kind}_strip.png")

    track_profiles(res, scan.spacing, t.track_id.head(args.top).tolist(), path=out / "profiles.png")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
