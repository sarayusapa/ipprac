"""Step 02 - Lung segmentation (Section 8.2). Ribs never become candidates.

Per slice:
  1. threshold at -400 HU (air + lung below, tissue + bone above)
  2. drop components touching the image border (air outside the body)
  3. drop the trachea (small, central) and keep the two largest components
  4. fill holes (vessels / nodules are holes in the air mask)
  5. rolling-ball closing (~10 mm) per lung, so wall-attached nodules are restored

Plus one 3D clean-up between 3 and 4: air components that are tiny in 3D
(bowel gas below the diaphragm) are removed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import clear_border

from .config import LungConfig


def segment_lungs(hu: np.ndarray, spacing: np.ndarray, cfg: LungConfig) -> np.ndarray:
    """Boolean lung mask (z, y, x)."""
    sy, sx = float(spacing[1]), float(spacing[2])
    mask = np.zeros(hu.shape, dtype=bool)
    for z in range(hu.shape[0]):
        mask[z] = _air_components(hu[z], (sy, sx), cfg)

    if cfg.min_3d_fraction > 0 and mask.any():
        mask = _drop_small_3d(mask, cfg.min_3d_fraction)

    for z in range(hu.shape[0]):
        if mask[z].any():
            mask[z] = _fill_and_close(mask[z], (sy, sx), cfg.closing_radius_mm)
    return mask


def _air_components(hu2d: np.ndarray, spacing_yx, cfg: LungConfig) -> np.ndarray:
    sy, sx = spacing_yx
    air = clear_border(hu2d < cfg.threshold_hu)
    lab, n = ndi.label(air)
    if n == 0:
        return air

    idx = np.arange(1, n + 1)
    areas = ndi.sum(air, lab, idx) * sy * sx
    centroids = ndi.center_of_mass(air, lab, idx)
    body_cx = _body_center_x(hu2d)
    width = hu2d.shape[1]

    keep = []
    for i, area, (_, cx) in zip(idx, areas, centroids):
        if area < cfg.min_component_area_mm2:
            continue
        is_trachea = (
            area <= cfg.trachea_max_area_mm2
            and abs(cx - body_cx) < cfg.trachea_center_band * width
        )
        if is_trachea:
            continue
        keep.append((area, i))
    keep.sort(reverse=True)
    kept_ids = [i for _, i in keep[: cfg.keep_n_largest]]
    return np.isin(lab, kept_ids)


def _body_center_x(hu2d: np.ndarray) -> float:
    body = hu2d > -500
    if not body.any():
        return hu2d.shape[1] / 2
    lab, n = ndi.label(body)
    sizes = ndi.sum(body, lab, np.arange(1, n + 1))
    biggest = lab == (np.argmax(sizes) + 1)
    return float(np.nonzero(biggest)[1].mean())


def _drop_small_3d(mask: np.ndarray, min_fraction: float) -> np.ndarray:
    lab, n = ndi.label(mask)  # 6-connectivity
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    keep_ids = np.nonzero(sizes >= min_fraction * sizes.max())[0] + 1
    return np.isin(lab, keep_ids)


def _fill_and_close(mask2d: np.ndarray, spacing_yx, radius_mm: float) -> np.ndarray:
    """Fill holes, then close each lung separately (so the two lungs are not bridged).

    Closing is done with Euclidean distance transforms (exact disk, in mm),
    which is much faster than a large structuring element.
    """
    sy, sx = spacing_yx
    pad_y = int(np.ceil(radius_mm / sy)) + 2
    pad_x = int(np.ceil(radius_mm / sx)) + 2
    out = np.zeros_like(mask2d)
    lab, n = ndi.label(mask2d)
    for sl, i in zip(ndi.find_objects(lab), range(1, n + 1)):
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        comp = lab[y0:y1, x0:x1] == i
        comp = ndi.binary_fill_holes(comp)
        if radius_mm > 0:
            comp = np.pad(comp, ((pad_y, pad_y), (pad_x, pad_x)))
            dilated = ndi.distance_transform_edt(~comp, sampling=(sy, sx)) <= radius_mm
            comp = ndi.distance_transform_edt(dilated, sampling=(sy, sx)) > radius_mm
            comp = ndi.binary_fill_holes(comp)
            # paste back, clipping the padded region at the image border
            Y0, X0 = y0 - pad_y, x0 - pad_x
            cy0, cx0 = max(0, -Y0), max(0, -X0)
            Y0c, X0c = max(Y0, 0), max(X0, 0)
            Y1c = min(Y0 + comp.shape[0], mask2d.shape[0])
            X1c = min(X0 + comp.shape[1], mask2d.shape[1])
            out[Y0c:Y1c, X0c:X1c] |= comp[cy0 : cy0 + (Y1c - Y0c), cx0 : cx0 + (X1c - X0c)]
        else:
            out[y0:y1, x0:x1] |= comp
    return out


def wall_distance(mask2d: np.ndarray, spacing_yx) -> np.ndarray:
    """Distance (mm) from each lung pixel to the lung wall."""
    return ndi.distance_transform_edt(mask2d, sampling=spacing_yx)
