"""A small synthetic chest phantom for tests and demos (no real data needed).

Contains: body, two lungs, ribs (outside the lungs), a 8 mm and a 6 mm sphere
(nodules), a straight vertical tube and an oblique drifting tube (vessels),
all with partial-volume edges and Gaussian noise. Nodules and vessels have the
same HU, so only shape across slices can separate them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import CTScan

TISSUE_HU = 30.0
LUNG_HU = -850.0


def _soft(dist_mm: np.ndarray, radius_mm: float, edge_mm: float = 0.7) -> np.ndarray:
    return np.clip((radius_mm - dist_mm) / edge_mm + 0.5, 0.0, 1.0)


def make_phantom(
    shape=(64, 160, 160), spacing=(1.25, 0.7, 0.7), noise_hu=25.0, seed=0, seriesuid="phantom"
) -> tuple[CTScan, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    Z, H, W = shape
    dz, dy, dx = spacing
    zz, yy, xx = np.meshgrid(
        np.arange(Z) * dz, np.arange(H) * dy, np.arange(W) * dx, indexing="ij"
    )
    cy, cx = H * dy / 2, W * dx / 2
    vol = np.full(shape, -1000.0, dtype=np.float32)

    body = ((xx - cx) / 50) ** 2 + ((yy - cy) / 42) ** 2 <= 1
    vol[body] = 40.0

    # ribs: bright discs along the inside of the body wall, all slices
    for ang in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        rx, ry = cx + 44 * np.cos(ang), cy + 36 * np.sin(ang)
        rib = (xx - rx) ** 2 + (yy - ry) ** 2 <= 3.0**2
        vol[rib] = 700.0

    z_top, z_bot = 4 * dz, (Z - 4) * dz
    in_z = (zz >= z_top) & (zz <= z_bot)
    lungs_c = [(cx - 22, cy), (cx + 22, cy)]
    lung = np.zeros(shape, bool)
    for lx, ly in lungs_c:
        lung |= in_z & (((xx - lx) / 17) ** 2 + ((yy - ly) / 30) ** 2 <= 1)
    vol[lung] = LUNG_HU

    def add(weight):
        nonlocal vol
        w = weight * lung
        vol = vol * (1 - w) + TISSUE_HU * w

    zc = Z * dz / 2
    truth = []

    # nodule 1: R = 4 mm sphere, left lung
    n1 = (lungs_c[0][0], cy + 2, zc)
    add(_soft(np.sqrt((xx - n1[0]) ** 2 + (yy - n1[1]) ** 2 + (zz - n1[2]) ** 2), 4.0))
    truth.append(("nodule", *n1, 8.0))

    # nodule 2: R = 3 mm sphere, right lung
    n2 = (lungs_c[1][0] + 3, cy - 16, zc - 12)
    add(_soft(np.sqrt((xx - n2[0]) ** 2 + (yy - n2[1]) ** 2 + (zz - n2[2]) ** 2), 3.0))
    truth.append(("nodule", *n2, 6.0))

    # vessel 1: straight vertical tube r = 1.6 mm, left lung
    v1 = (lungs_c[0][0] + 2, cy - 18)
    add(_soft(np.sqrt((xx - v1[0]) ** 2 + (yy - v1[1]) ** 2), 1.6) * in_z)
    truth.append(("vessel_straight", v1[0], v1[1], zc, 3.2))

    # vessel 2: oblique tube r = 1.6 mm drifting ~0.6 mm per slice, right lung
    frac = (zz - z_top) / (z_bot - z_top)
    x_of_z = lungs_c[1][0] - 12 + 24 * frac
    y_of_z = cy + 4 + 22 * frac
    add(_soft(np.sqrt((xx - x_of_z) ** 2 + (yy - y_of_z) ** 2), 1.6) * in_z)
    truth.append(("vessel_oblique", lungs_c[1][0], cy + 16, zc, 3.2))

    vol += rng.normal(0, noise_hu, size=shape).astype(np.float32)

    origin = np.array([-cx, -cy, -100.0])  # arbitrary world origin (x, y, z)
    scan = CTScan(
        seriesuid=seriesuid,
        hu=vol.astype(np.float32),
        spacing=np.array(spacing, float),
        origin=origin,
        direction=np.eye(3),
    )
    df = pd.DataFrame(truth, columns=["kind", "x_mm", "y_mm", "z_mm", "diameter_mm"])
    df["coordX"] = df["x_mm"] + origin[0]
    df["coordY"] = df["y_mm"] + origin[1]
    df["coordZ"] = df["z_mm"] + origin[2]
    df["seriesuid"] = seriesuid
    return scan, df[["seriesuid", "kind", "coordX", "coordY", "coordZ", "diameter_mm"]]
