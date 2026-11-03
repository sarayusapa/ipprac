#!/usr/bin/env python
"""Run P0 on the synthetic phantom (no data needed) and save figures."""
from pathlib import Path

import pandas as pd

from lungnod import run_p0
from lungnod.synthetic import make_phantom
from lungnod.viz import show_slice, track_profiles

import matplotlib.pyplot as plt

out = Path("outputs/demo")
out.mkdir(parents=True, exist_ok=True)
scan, truth = make_phantom()
res = run_p0(scan, keep_intermediate=True)
t = res.table.sort_values("probability", ascending=False)

pd.set_option("display.width", 200)
print(truth.to_string(index=False))
print()
cols = ["track_id", "vox_z", "vox_y", "vox_x", "max_diameter_mm", "lifetime_mm", "aspect",
        "drift_ratio", "sphere_gain", "probability", "explanation"]
print(t[cols].head(12).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

nod = truth[truth.kind == "nodule"]
for i, z in enumerate(scan.world_to_voxel(nod[["coordX", "coordY", "coordZ"]].to_numpy())[:, 0]):
    fig, ax = plt.subplots(figsize=(6, 6))
    show_slice(scan, int(round(z)), res, nod, ax=ax)
    fig.savefig(out / f"nodule{i}_slice.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
track_profiles(res, scan.spacing, t.track_id.head(4).tolist(), path=out / "profiles.png")
print(f"\nfigures in {out}")
