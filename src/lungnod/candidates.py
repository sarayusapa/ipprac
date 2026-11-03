"""Step 03 - Candidate detection (Section 8.3). Goal: high recall.

* multi-scale normalised LoG inside the lung mask:
      L = -sigma^2 (d2I/dx2 + d2I/dy2),  sigma = r / sqrt(2),  r in [1.5, 15] mm
* local maxima over (scale, y, x) above a threshold become candidate centres
* region growing from each centre (connected pixels > -600 HU) gives the outline
* per-blob 2D features: size, circularity, eccentricity, HU stats, wall distance
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import perimeter, regionprops

from .config import CandidateConfig
from .lung_mask import wall_distance


@dataclass
class Blob:
    z: int
    y: float  # centroid, pixels (global)
    x: float
    radius_mm: float  # equivalent radius sqrt(area / pi)
    area_mm2: float
    circularity: float  # 4 pi A / P^2
    eccentricity: float
    mean_hu: float
    std_hu: float
    wall_dist_mm: float
    log_response: float
    log_radius_mm: float
    truncated: bool  # region hit the growing window -> elongated / attached structure

    @property
    def diameter_mm(self) -> float:
        return 2.0 * self.radius_mm


def log_radii(cfg: CandidateConfig) -> np.ndarray:
    return np.geomspace(cfg.r_min_mm, cfg.r_max_mm, cfg.n_scales)


def detect_candidates(
    norm: np.ndarray,
    hu: np.ndarray,
    lung: np.ndarray,
    spacing: np.ndarray,
    cfg: CandidateConfig,
) -> dict[int, list[Blob]]:
    """Run 2D detection on every slice. Returns {z: [Blob, ...]}."""
    spacing_yx = (float(spacing[1]), float(spacing[2]))
    radii = log_radii(cfg)
    out: dict[int, list[Blob]] = {}
    for z in range(norm.shape[0]):
        blobs = detect_slice(norm[z], hu[z], lung[z], z, spacing_yx, cfg, radii)
        if blobs:
            out[z] = blobs
    return out


def log_stack(img: np.ndarray, spacing_yx, radii: np.ndarray) -> np.ndarray:
    """Scale-normalised, sign-flipped LoG stack (bright blobs -> positive)."""
    sy, sx = spacing_yx
    stack = np.empty((len(radii),) + img.shape, dtype=np.float32)
    for i, r in enumerate(radii):
        s = r / np.sqrt(2.0)
        sig = (s / sy, s / sx)
        stack[i] = -(sig[0] * sig[1]) * ndi.gaussian_laplace(img, sigma=sig, mode="nearest")
    return stack


def detect_slice(
    norm2d: np.ndarray,
    hu2d: np.ndarray,
    mask2d: np.ndarray,
    z: int,
    spacing_yx,
    cfg: CandidateConfig,
    radii: np.ndarray | None = None,
) -> list[Blob]:
    if not mask2d.any():
        return []
    sy, sx = spacing_yx
    radii = log_radii(cfg) if radii is None else radii

    # crop to the lung bounding box (+ filter support) for speed
    pad = int(np.ceil(3 * radii[-1] / np.sqrt(2) / min(sy, sx)))
    ys, xs = np.nonzero(mask2d)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask2d.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask2d.shape[1])
    m = mask2d[y0:y1, x0:x1]
    img = norm2d[y0:y1, x0:x1]
    hu = hu2d[y0:y1, x0:x1]
    if cfg.mask_outside:
        img = np.where(m, img, 0.0).astype(np.float32)  # outside lung -> air

    stack = log_stack(img, spacing_yx, radii)
    peaks = (stack == ndi.maximum_filter(stack, size=3, mode="nearest"))
    peaks &= stack > cfg.log_threshold
    peaks &= m[None]
    si, yi, xi = np.nonzero(peaks)
    if si.size == 0:
        return []
    resp = stack[si, yi, xi]
    order = np.argsort(-resp)

    dist = wall_distance(m, spacing_yx)
    solid = (hu > cfg.grow_hu) & m
    assigned = np.zeros_like(m)
    H, W = m.shape
    blobs: list[Blob] = []

    for k in order:
        if len(blobs) >= cfg.max_blobs_per_slice:
            break
        py, px, r_mm = yi[k], xi[k], float(radii[si[k]])
        if assigned[py, px]:
            continue

        hy = int(np.ceil((cfg.grow_margin * r_mm + 2.0) / sy))
        hx = int(np.ceil((cfg.grow_margin * r_mm + 2.0) / sx))
        wy0, wy1 = max(py - hy, 0), min(py + hy + 1, H)
        wx0, wx1 = max(px - hx, 0), min(px + hx + 1, W)
        cy, cx = py - wy0, px - wx0
        sub = solid[wy0:wy1, wx0:wx1]
        disc = _disc((wy1 - wy0, wx1 - wx0), (cy, cx), r_mm, spacing_yx)

        if sub[cy, cx]:
            lab, _ = ndi.label(sub)
            region = lab == lab[cy, cx]
            if cfg.clip_region_factor > 0:
                clip = _disc(disc.shape, (cy, cx), cfg.clip_region_factor * r_mm, spacing_yx)
                touches_clip = bool((region & ~clip).any())
                region &= clip
                lab, _ = ndi.label(region)
                region = lab == lab[cy, cx]
            else:
                touches_clip = False
        elif cfg.fallback_disc:
            region = disc & m[wy0:wy1, wx0:wx1]
            touches_clip = False
        else:
            assigned[wy0:wy1, wx0:wx1] |= _disc(disc.shape, (cy, cx), 0.5 * r_mm, spacing_yx)
            continue

        asg = assigned[wy0:wy1, wx0:wx1]
        n_px = int(region.sum())
        # same object already found from a stronger maximum -> skip
        if n_px == 0 or (region & asg).sum() > 0.5 * n_px:
            asg |= _disc(disc.shape, (cy, cx), 0.5 * r_mm, spacing_yx)
            continue
        asg |= region | _disc(disc.shape, (cy, cx), 0.5 * r_mm, spacing_yx)

        area = n_px * sy * sx
        radius = float(np.sqrt(area / np.pi))
        if 2 * radius < cfg.min_blob_diameter_mm:
            continue

        perim = perimeter(region) * np.sqrt(sy * sx)
        circ = float(min(4 * np.pi * area / perim**2, 1.0)) if perim > 0 else 1.0
        props = regionprops(region.astype(np.uint8))[0]
        if cfg.center_from_log:
            # sub-pixel peak (quadratic fit through the peak and its neighbours,
            # as in SIFT, Lowe 2004); avoids 1-pixel jitter reading as drift
            L = stack[si[k]]
            gy = py + _subpixel(L, py, px, axis=0)  # crop coordinates
            gx = px + _subpixel(L, py, px, axis=1)
        else:
            ry, rx = np.nonzero(region)
            gy, gx = ry.mean() + wy0, rx.mean() + wx0
        vals = hu[wy0:wy1, wx0:wx1][region]
        truncated = touches_clip or bool(
            region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any()
        )
        blobs.append(
            Blob(
                z=z,
                y=float(gy + y0),
                x=float(gx + x0),
                radius_mm=radius,
                area_mm2=float(area),
                circularity=circ,
                eccentricity=float(props.eccentricity),
                mean_hu=float(vals.mean()),
                std_hu=float(vals.std()),
                wall_dist_mm=float(dist[int(round(gy)), int(round(gx))]),
                log_response=float(resp[k]),
                log_radius_mm=r_mm,
                truncated=truncated,
            )
        )
    return blobs


def _subpixel(L: np.ndarray, y: int, x: int, axis: int) -> float:
    n = L.shape[axis]
    i = y if axis == 0 else x
    if i == 0 or i == n - 1:
        return 0.0
    if axis == 0:
        a, b, c = L[y - 1, x], L[y, x], L[y + 1, x]
    else:
        a, b, c = L[y, x - 1], L[y, x], L[y, x + 1]
    den = a - 2 * b + c
    if den >= 0:  # not a strict maximum along this axis
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def _disc(shape, center, r_mm: float, spacing_yx) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    dy = (yy - center[0]) * spacing_yx[0]
    dx = (xx - center[1]) * spacing_yx[1]
    return dy * dy + dx * dx <= r_mm * r_mm
