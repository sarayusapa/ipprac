"""Figures for inspecting each step (lung mask, candidates, tracks, FROC)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .io import CTScan  # noqa: E402
from .pipeline import P0Result  # noqa: E402

LUNG_WINDOW = (-1000, 400)


def show_slice(
    scan: CTScan,
    z: int,
    result: P0Result | None = None,
    annotations: pd.DataFrame | None = None,
    ax=None,
    min_prob: float = 0.5,
):
    """Slice in lung window + lung-mask contour + blobs (thin) + accepted tracks (thick)
    + ground-truth annotations (yellow)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    sy, sx = scan.spacing[1], scan.spacing[2]
    ax.imshow(scan.hu[z], cmap="gray", vmin=LUNG_WINDOW[0], vmax=LUNG_WINDOW[1])
    if result is not None and result.lung_mask is not None:
        ax.contour(result.lung_mask[z], levels=[0.5], colors="cyan", linewidths=0.8)
    if result is not None and result.blobs is not None:
        for b in result.blobs.get(z, []):
            ax.add_patch(plt.Circle((b.x, b.y), b.radius_mm / sx, fill=False, ec="orange", lw=0.6))
    if result is not None:
        t = result.table
        on = t[(t["z_first"] <= z) & (t["z_last"] >= z) & (t["probability"] >= min_prob)]
        for _, r in on.iterrows():
            ax.add_patch(
                plt.Circle((r.vox_x, r.vox_y), r.max_diameter_mm / sx, fill=False, ec="lime", lw=1.5)
            )
            ax.text(r.vox_x + 6, r.vox_y, f"{r.probability:.2f}", color="lime", fontsize=8)
    if annotations is not None:
        a = annotations[annotations["seriesuid"] == scan.seriesuid]
        if len(a):
            v = scan.world_to_voxel(a[["coordX", "coordY", "coordZ"]].to_numpy())
            for (az, ay, ax_), d in zip(v, a["diameter_mm"]):
                if abs(az - z) <= d / 2 / scan.spacing[0]:
                    ax.add_patch(plt.Circle((ax_, ay), d / sx, fill=False, ec="yellow", lw=1.2, ls="--"))
    ax.set_title(f"{scan.seriesuid[-12:]}  slice {z}/{scan.hu.shape[0]}")
    ax.axis("off")
    return ax


def nodule_strip(scan: CTScan, zyx, n: int = 7, crop_mm: float = 40.0, path=None):
    """Like report Figure 2: the same location over n consecutive slices."""
    z, y, x = (int(round(v)) for v in zyx)
    hy = int(crop_mm / 2 / scan.spacing[1])
    hx = int(crop_mm / 2 / scan.spacing[2])
    fig, axs = plt.subplots(1, n, figsize=(2 * n, 2.3))
    for i, ax in enumerate(np.atleast_1d(axs)):
        zi = z - n // 2 + i
        if 0 <= zi < scan.hu.shape[0]:
            crop = scan.hu[zi, max(y - hy, 0) : y + hy, max(x - hx, 0) : x + hx]
            ax.imshow(crop, cmap="gray", vmin=-1000, vmax=400)
        ax.set_title(f"slice {zi}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def track_profiles(result: P0Result, spacing, track_ids, path=None):
    """Like report Figure 4: blob radius along each track (sphere rises and falls,
    tube stays flat)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    tracks = {t.track_id: t for t in result.tracks}
    probs = dict(zip(result.table["track_id"], result.table["probability"]))
    for tid in track_ids:
        t = tracks[tid]
        z = np.array([b.z for b in t.blobs]) * spacing[0]
        z = z - z[np.argmax([b.radius_mm for b in t.blobs])]
        ax.plot(z, [b.radius_mm for b in t.blobs], "o-", label=f"track {tid}  p={probs[tid]:.2f}")
    ax.set_xlabel("slice position relative to widest slice (mm)")
    ax.set_ylabel("blob radius (mm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def plot_froc(results: dict[str, dict], path=None):
    """results: {label: output of evaluation.evaluate}."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # sample the curve with the same interpolation the CPM uses, so plot and number agree
    grid = np.logspace(np.log2(0.125), np.log2(8), 200, base=2)
    for label, r in results.items():
        ax.plot(grid, np.interp(grid, r["fps"], r["sens"]), label=f"{label}  CPM={r['cpm']:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.125, 8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("average FPs per scan")
    ax.set_ylabel("sensitivity")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig
