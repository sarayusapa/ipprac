"""Step 01 - Preprocessing (Section 8.1).

* keep spacing from the header so everything downstream is in mm
* clip HU to [-1000, 400] and rescale to [0, 1]
* 3x3 median filter on every slice (removes low-dose speckle, keeps edges)
* no histogram equalisation: HU is a physical scale
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import PreprocessConfig


def preprocess(hu: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (filtered HU, normalised [0,1] image), both float32 (z, y, x)."""
    out = np.clip(hu, cfg.hu_min, cfg.hu_max).astype(np.float32)
    if cfg.median_size > 1:
        out = ndi.median_filter(out, size=(1, cfg.median_size, cfg.median_size))
    norm = (out - cfg.hu_min) / (cfg.hu_max - cfg.hu_min)
    return out, norm.astype(np.float32)
