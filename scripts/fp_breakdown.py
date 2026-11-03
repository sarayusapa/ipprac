#!/usr/bin/env python
"""FP breakdown (report Sec. 11.2): what do P0's highest-scoring false positives look like?

Takes the N highest-scoring tracks that hit no nodule and no excluded finding,
gives each a provisional label from its features, and saves for each FP an image
sheet (axial crop over 7 slices + coronal crop) so you can check the label by eye.

Provisional labels (rule of thumb from the features, to be checked by hand):
  wall    - centre within 3 mm of the lung mask edge (pleura / chest wall / ribs)
  vessel  - long (aspect > 1.5), shifting (drift ratio > 0.5), branching, or truncated
            by the region-growing window (attached to a longer structure)
  other   - none of the above (noise, scar, airway wall, ...)

Outputs in <run>/fp_breakdown/: fp_labels.csv (fill in the `label` column),
fp_XXX.png sheets, summary.txt.

  python scripts/fp_breakdown.py --run outputs/p0_s0_v2 --data-root data/luna16 \
      --annotations data/luna16/annotations.csv --excluded data/luna16/annotations_excluded.csv
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lungnod.io import find_scans, load_scan  # noqa: E402


def hits(t: pd.DataFrame, ref: pd.DataFrame, default_diam: float) -> np.ndarray:
    out = np.zeros(len(t), bool)
    for uid, g in ref.groupby("seriesuid"):
        m = (t.seriesuid == uid).to_numpy()
        if not m.any():
            continue
        c = t.loc[m, ["coordX", "coordY", "coordZ"]].to_numpy()
        for _, r in g.iterrows():
            d = r.diameter_mm if r.diameter_mm > 0 else default_diam
            out[np.nonzero(m)[0][np.linalg.norm(c - [r.coordX, r.coordY, r.coordZ], axis=1) <= d / 2]] = True
    return out


def provisional(r) -> str:
    if r.wall_dist_mm < 3.0:
        return "wall"
    if r.aspect > 1.5 or r.drift_ratio > 0.5 or r.branches > 0 or r.truncated_frac > 0.5:
        return "vessel"
    return "other"


def sheet(scan, r, path, crop_mm=40.0, n=7):
    z, y, x = int(round(r.vox_z)), int(round(r.vox_y)), int(round(r.vox_x))
    sy, sx = scan.spacing[1], scan.spacing[2]
    hy, hx = int(crop_mm / 2 / sy), int(crop_mm / 2 / sx)
    fig, axs = plt.subplots(1, n + 2, figsize=(2 * (n + 2), 2.5))
    for i in range(n):
        zi = z - n // 2 + i
        ax = axs[i]
        if 0 <= zi < scan.hu.shape[0]:
            ax.imshow(scan.hu[zi, max(y - hy, 0): y + hy, max(x - hx, 0): x + hx],
                      cmap="gray", vmin=-1000, vmax=400)
        ax.set_title(f"z={zi}" + (" *" if zi == z else ""), fontsize=8)
        ax.axis("off")
    # coronal and sagittal views (show tubes vs balls along z)
    zz = int(crop_mm / 2 / scan.spacing[0])
    z0, z1 = max(z - zz, 0), z + zz
    cor = scan.hu[z0:z1, y, max(x - hx, 0): x + hx]
    sag = scan.hu[z0:z1, max(y - hy, 0): y + hy, x]
    asp = scan.spacing[0] / sx
    axs[n].imshow(cor[::-1], cmap="gray", vmin=-1000, vmax=400, aspect=asp)
    axs[n].set_title("coronal", fontsize=8)
    axs[n + 1].imshow(sag[::-1], cmap="gray", vmin=-1000, vmax=400, aspect=asp)
    axs[n + 1].set_title("sagittal", fontsize=8)
    for a in axs[n:]:
        a.axis("off")
    fig.suptitle(
        f"#{r.rank}  p={r.probability:.3f}  auto={r.auto_label}  d={r.max_diameter_mm:.1f}mm  "
        f"aspect={r.aspect:.2f}  drift={r.drift_ratio:.2f}  sphere={r.sphere_gain:.2f}  "
        f"wall={r.wall_dist_mm:.1f}mm  {r.seriesuid[-6:]}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--excluded", default=None)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    run = Path(args.run)
    out = run / "fp_breakdown"
    out.mkdir(exist_ok=True)
    t = pd.read_csv(run / "tracks.csv")
    ann = pd.read_csv(args.annotations)
    fp = ~hits(t, ann, 0)
    if args.excluded:
        fp &= ~hits(t, pd.read_csv(args.excluded), 10.0)
    top = t[fp].sort_values("probability", ascending=False).head(args.n).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    top["auto_label"] = [provisional(r) for r in top.itertuples()]
    top["label"] = ""  # fill in by hand: vessel / rib / wall / airway / scar / noise / other

    paths = {p.stem: p for p in find_scans(args.data_root)}
    for uid, g in top.groupby("seriesuid"):
        scan = load_scan(paths[uid])
        for r in g.itertuples():
            sheet(scan, r, out / f"fp_{r.rank:03d}.png")

    cols = ["rank", "label", "auto_label", "probability", "seriesuid", "coordX", "coordY", "coordZ",
            "vox_z", "vox_y", "vox_x", "max_diameter_mm", "lifetime_mm", "aspect", "drift_ratio",
            "sphere_gain", "branches", "wall_dist_mm", "truncated_frac", "max_log_response", "explanation"]
    top[cols].to_csv(out / "fp_labels.csv", index=False)
    counts = top.auto_label.value_counts()
    summary = (f"Top {len(top)} false positives (by P0 score) in {run.name}\n"
               f"provisional labels:\n{counts.to_string()}\n\n"
               f"median features by provisional label:\n"
               f"{top.groupby('auto_label')[['max_diameter_mm', 'aspect', 'drift_ratio', 'sphere_gain', 'wall_dist_mm', 'n_slices']].median().round(2).to_string()}\n")
    (out / "summary.txt").write_text(summary)
    print(summary)
    print(f"sheets + fp_labels.csv in {out}")


if __name__ == "__main__":
    main()
