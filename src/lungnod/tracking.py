"""Step 04 - Slice tracking (Section 8.4). The slice stack is treated as video.

Blobs on slice z are linked to open tracks with the Hungarian method on
in-plane centroid distance (mm). A pair is allowed only if
    distance <= max(3 mm, 0.5 * blob radius).
A gap of one slice is allowed. Each track is one 3D candidate.

Branching (Section 8.5): a blob that is within the gate of a track that was
already matched (split) or a track that ends next to a blob claimed by another
track (merge) increments the branch counter of the tracks involved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .candidates import Blob
from .config import TrackingConfig

_BIG = 1e9


@dataclass
class Track:
    track_id: int
    blobs: list[Blob] = field(default_factory=list)
    branches: int = 0

    @property
    def first_z(self) -> int:
        return self.blobs[0].z

    @property
    def last_z(self) -> int:
        return self.blobs[-1].z

    @property
    def last(self) -> Blob:
        return self.blobs[-1]

    @property
    def n_slices(self) -> int:
        """Slices spanned, gaps included."""
        return self.last_z - self.first_z + 1

    @property
    def widest(self) -> Blob:
        return max(self.blobs, key=lambda b: b.radius_mm)

    def blob_near_z(self, z: float) -> Blob:
        return min(self.blobs, key=lambda b: abs(b.z - z))


def _gate(a: Blob, b: Blob, cfg: TrackingConfig) -> float:
    return max(cfg.gate_min_mm, cfg.gate_radius_frac * max(a.radius_mm, b.radius_mm))


def _dist_mm(a: Blob, b: Blob, spacing_yx) -> float:
    return float(np.hypot((a.y - b.y) * spacing_yx[0], (a.x - b.x) * spacing_yx[1]))


def track_blobs(
    blobs_by_slice: dict[int, list[Blob]], spacing: np.ndarray, cfg: TrackingConfig
) -> list[Track]:
    spacing_yx = (float(spacing[1]), float(spacing[2]))
    tracks: list[Track] = []
    active: list[Track] = []
    if not blobs_by_slice:
        return tracks

    zs = sorted(blobs_by_slice)
    for z in range(zs[0], zs[-1] + 1):
        cur = blobs_by_slice.get(z, [])
        active = [t for t in active if z - t.last_z <= cfg.max_gap + 1]
        matched_track = [False] * len(active)
        matched_blob = [-1] * len(cur)  # index of the track each blob joined

        if active and cur:
            cost = np.full((len(active), len(cur)), _BIG)
            in_gate = np.zeros_like(cost, dtype=bool)
            for i, t in enumerate(active):
                for j, b in enumerate(cur):
                    d = _dist_mm(t.last, b, spacing_yx)
                    if d <= _gate(t.last, b, cfg):
                        in_gate[i, j] = True
                        cost[i, j] = d + cfg.gap_penalty_mm * (z - t.last_z - 1)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < _BIG:
                    matched_track[i] = True
                    matched_blob[j] = i

            # branching: splits (extra blob next to a matched track)
            for j in range(len(cur)):
                if matched_blob[j] < 0:
                    for i in np.nonzero(in_gate[:, j])[0]:
                        if matched_track[i] and active[i].last_z == z - 1:
                            active[i].branches += 1
            # merges (track ends next to a blob another track took)
            for i, t in enumerate(active):
                if not matched_track[i] and t.last_z == z - 1:
                    for j in np.nonzero(in_gate[i])[0]:
                        if matched_blob[j] >= 0:
                            t.branches += 1
                            active[matched_blob[j]].branches += 1
                            break

            for j, i in enumerate(matched_blob):
                if i >= 0:
                    active[i].blobs.append(cur[j])

        for j, b in enumerate(cur):
            if matched_blob[j] < 0:
                t = Track(track_id=len(tracks), blobs=[b])
                tracks.append(t)
                active.append(t)
    return tracks
