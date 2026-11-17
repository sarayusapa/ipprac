"""P0 end to end: steps 01-06 on one scan, plus scoring of a given candidate list
(LUNA16 false-positive-reduction track, Section 11.1)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .candidates import Blob, detect_candidates
from .classify import classify_table
from .config import P0Config
from .features import track_features
from .io import CTScan
from .lung_mask import segment_lungs
from .preprocess import preprocess
from .tracking import Track, track_blobs


@dataclass
class P0Result:
    seriesuid: str
    table: pd.DataFrame  # one row per track: features, rule scores, probability, explanation
    tracks: list[Track]
    timings: dict = field(default_factory=dict)
    lung_mask: np.ndarray | None = None
    blobs: dict[int, list[Blob]] | None = None

    def submission(self, min_prob: float = 0.0) -> pd.DataFrame:
        """LUNA16 submission rows: seriesuid, coordX, coordY, coordZ, probability."""
        t = self.table
        t = t[t["probability"] > min_prob]
        return t[["seriesuid", "coordX", "coordY", "coordZ", "probability"]].reset_index(drop=True)


def run_p0(scan: CTScan, cfg: P0Config | None = None, keep_intermediate: bool = False) -> P0Result:
    cfg = cfg or P0Config()
    tm: dict[str, float] = {}

    t0 = time.perf_counter()
    hu, norm = preprocess(scan.hu, cfg.preprocess)
    tm["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lung = segment_lungs(hu, scan.spacing, cfg.lung)
    tm["lung_mask"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    blobs = detect_candidates(norm, hu, lung, scan.spacing, cfg.candidates)
    tm["candidates"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tracks = track_blobs(blobs, scan.spacing, cfg.tracking)
    tm["tracking"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    table = tracks_to_table(scan, tracks, cfg)
    tm["features+classify"] = time.perf_counter() - t0

    return P0Result(
        seriesuid=scan.seriesuid,
        table=table,
        tracks=tracks,
        timings=tm,
        lung_mask=lung if keep_intermediate else None,
        blobs=blobs if keep_intermediate else None,
    )


def tracks_to_table(scan: CTScan, tracks: list[Track], cfg: P0Config) -> pd.DataFrame:
    rows = []
    for t in tracks:
        f = track_features(t, scan.spacing, cfg.features)
        w = t.widest  # track centre = centroid of its widest slice
        world = scan.voxel_to_world([w.z, w.y, w.x])[0]
        rows.append(
            {
                "seriesuid": scan.seriesuid,
                "track_id": t.track_id,
                "coordX": world[0],
                "coordY": world[1],
                "coordZ": world[2],
                "vox_z": w.z,
                "vox_y": w.y,
                "vox_x": w.x,
                "z_first": t.first_z,
                "z_last": t.last_z,
                **f,
            }
        )
    return classify_table(pd.DataFrame(rows), cfg.classifier)


def score_candidate_list(
    scan: CTScan, result: P0Result, candidates: pd.DataFrame, cfg: P0Config | None = None
) -> pd.DataFrame:
    """FP-reduction track: give each listed candidate the score of the P0 track
    passing through it (0 if no track does).

    A track passes through a candidate if it has a blob within one slice of the
    candidate and the in-plane distance is <= max(gate_min_mm, blob radius).
    """
    cfg = cfg or P0Config()
    sy, sx = scan.spacing[1], scan.spacing[2]
    prob_by_id = dict(zip(result.table["track_id"], result.table["probability"]))
    cands = candidates[candidates["seriesuid"] == scan.seriesuid]
    out = []
    if cands.empty:
        return pd.DataFrame(columns=["seriesuid", "coordX", "coordY", "coordZ", "probability"])
    vox = scan.world_to_voxel(cands[["coordX", "coordY", "coordZ"]].to_numpy())
    for (cz, cy, cx), (_, row) in zip(vox, cands.iterrows()):
        best = 0.0
        linked = []
        for t in result.tracks:
            if not (t.first_z - 1 <= cz <= t.last_z + 1):
                continue
            b = t.blob_near_z(cz)
            if abs(b.z - cz) > 1:
                continue
            d = np.hypot((b.y - cy) * sy, (b.x - cx) * sx)
            if d <= max(cfg.tracking.gate_min_mm, b.radius_mm):
                best = max(best, prob_by_id[t.track_id])
                linked.append(t.track_id)
        out.append(
            {
                "seriesuid": scan.seriesuid,
                "coordX": row["coordX"],
                "coordY": row["coordY"],
                "coordZ": row["coordZ"],
                "probability": best,
                # tracks passing through this candidate: lets the FPR track be re-scored
                # for new rules without re-running detection
                "track_ids": ";".join(map(str, linked)),
                **({"class": row["class"]} if "class" in row else {}),
            }
        )
    return pd.DataFrame(out)
