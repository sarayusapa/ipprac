"""Reading LUNA16 MetaImage scans and CSVs; world <-> voxel coordinates (Section 7.3)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CTScan:
    """A CT volume in HU with its geometry.

    Arrays are indexed (z, y, x). ``spacing`` is (z, y, x) in mm.
    ``origin`` and ``direction`` follow ITK conventions (x, y, z world order).
    """

    seriesuid: str
    hu: np.ndarray
    spacing: np.ndarray
    origin: np.ndarray
    direction: np.ndarray

    @property
    def spacing_xyz(self) -> np.ndarray:
        return self.spacing[::-1]

    def voxel_to_world(self, zyx) -> np.ndarray:
        """(N, 3) voxel indices (z, y, x) -> (N, 3) world mm (x, y, z)."""
        zyx = np.atleast_2d(np.asarray(zyx, dtype=float))
        idx_xyz = zyx[:, ::-1]
        return self.origin + (idx_xyz * self.spacing_xyz) @ self.direction.T

    def world_to_voxel(self, xyz) -> np.ndarray:
        """(N, 3) world mm (x, y, z) -> (N, 3) voxel indices (z, y, x), float."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        idx_xyz = np.linalg.solve(self.direction, (xyz - self.origin).T).T / self.spacing_xyz
        return idx_xyz[:, ::-1]


def load_scan(path: str | Path) -> CTScan:
    import SimpleITK as sitk

    path = Path(path)
    img = sitk.ReadImage(str(path))
    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    return CTScan(
        seriesuid=path.stem,
        hu=hu,
        spacing=np.array(img.GetSpacing(), dtype=float)[::-1],
        origin=np.array(img.GetOrigin(), dtype=float),
        direction=np.array(img.GetDirection(), dtype=float).reshape(3, 3),
    )


def find_scans(data_root: str | Path, subsets: list[int] | None = None) -> list[Path]:
    """Return .mhd paths under data_root/subset{i}/ (all subsets if None)."""
    root = Path(data_root)
    if subsets is None:
        dirs = sorted(root.glob("subset*"))
    else:
        dirs = [root / f"subset{i}" for i in subsets]
    paths: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Missing subset directory: {d}")
        paths.extend(sorted(d.glob("*.mhd")))
    return paths


def load_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)
