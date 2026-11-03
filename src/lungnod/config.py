"""All hand-set parameters of P0 in one place.

Every value here comes from anatomy / geometry (report Sections 3.3, 8.x), not
from training. Override any of them with a YAML file (see configs/p0.yaml).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class PreprocessConfig:  # Section 8.1
    hu_min: float = -1000.0
    hu_max: float = 400.0
    median_size: int = 3  # 3x3 median per slice; 1 disables


@dataclass
class LungConfig:  # Section 8.2
    threshold_hu: float = -400.0
    min_component_area_mm2: float = 50.0  # ignore tiny air pockets on a slice
    keep_n_largest: int = 2  # the two lungs
    trachea_max_area_mm2: float = 600.0  # ~28 mm diameter airway
    trachea_center_band: float = 0.10  # |cx - body_cx| < band * image width
    min_3d_fraction: float = 0.05  # drop 3D air components (bowel gas) smaller than this x largest
    closing_radius_mm: float = 10.0  # rolling-ball closing keeps juxtapleural nodules


@dataclass
class CandidateConfig:  # Section 8.3
    r_min_mm: float = 1.5
    r_max_mm: float = 15.0
    n_scales: int = 10
    log_threshold: float = 0.10  # on sigma^2-normalised LoG of the [0,1] image
    mask_outside: bool = True  # set everything outside the lung to air before LoG
    grow_hu: float = -600.0  # region growing threshold
    grow_margin: float = 1.5  # region-growing window half-size = margin * r + 2 mm
    min_blob_diameter_mm: float = 1.5
    max_blobs_per_slice: int = 80
    fallback_disc: bool = False  # if centre < grow_hu, use the LoG disc (for ground-glass)
    # Region growing leaks into attached vessels / opacities (report Sec. 12). Guard against it:
    center_from_log: bool = True  # blob centre = LoG scale-space maximum, not region centroid
    clip_region_factor: float = 1.5  # keep region within this x LoG radius (0 disables)


@dataclass
class TrackingConfig:  # Section 8.4
    gate_min_mm: float = 3.0
    gate_radius_frac: float = 0.5  # gate = max(gate_min_mm, frac * blob radius)
    max_gap: int = 1  # one missing slice allowed
    gap_penalty_mm: float = 1.0  # prefer continuing tracks seen on the previous slice


@dataclass
class FeatureConfig:  # Section 8.5
    min_fit_slices: int = 3  # shorter tracks get a neutral sphere score
    sphere_radius_tolerance: float = 1.3  # R <= tol * sqrt(r_max^2 + (dz/2)^2)


@dataclass
class ClassifierConfig:  # Section 8.6
    aspect_thresh: float = 2.5
    aspect_scale: float = 0.3
    # drift_ratio = lateral shift along the track / radius. Above 1 the centre has moved
    # by more than the object's own radius: first and last cross-sections no longer
    # contain each other's centre -> not a sphere. Scale = 0.2 x threshold, the same
    # softness ratio the report uses (0.1 / 0.5).
    drift_thresh: float = 1.0
    drift_scale: float = 0.2
    sphere_center: float = 0.0  # sphere_gain = nrmse_tube - nrmse_sphere
    sphere_scale: float = 0.05
    neutral_sphere_score: float = 0.5
    diam_min_mm: float = 3.0
    diam_max_mm: float = 30.0
    min_slices: int = 2  # noise blobs: minimum lifetime of 2 slices
    branch_scale: float = 0.5  # s_branch = exp(-scale * branches)
    rules: list[str] = field(
        default_factory=lambda: ["aspect", "drift", "sphere", "size", "min_slices", "log_strength"]
    )  # add "branching" to use the branching feature
    accept_threshold: float = 0.5  # only for the accept/reject label; FROC uses the score


@dataclass
class P0Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    lung: LungConfig = field(default_factory=LungConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    @classmethod
    def from_dict(cls, d: dict | None) -> "P0Config":
        cfg = cls()
        _update(cfg, d or {})
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "P0Config":
        if path is None:
            return cls()
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _update(obj, d: dict) -> None:
    names = {f.name for f in fields(obj)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f"Unknown config key '{k}' for {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur):
            _update(cur, v or {})
        else:
            setattr(obj, k, v)
