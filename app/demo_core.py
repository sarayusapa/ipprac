"""Data and computation helpers for the demo app (no Streamlit code here)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lungnod.candidates import detect_slice, hessian_stacks, log_radii
from lungnod.config import P0Config
from lungnod.features import sphere_profile, sphere_profile_fit
from lungnod.io import load_scan
from lungnod.lung_mask import segment_lungs
from lungnod.pipeline import run_p0
from lungnod.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "demo_cache"
DATA = ROOT / "data" / "luna16"
CFG_PATH = ROOT / "configs" / "p0.yaml"

RULES = ["aspect", "drift", "sphere", "size", "min_slices", "dot"]
RULE_TEXT = {
    "aspect": ("Aspect", "How long the object lasts across slices vs. how wide it is. "
               "A ball is about as tall as it is wide (aspect ≈ 1); a vessel running through "
               "the slices lasts much longer than its width."),
    "drift": ("Drift", "How far the centre moves sideways from the first to the last slice, "
              "in units of the radius. A ball stays put; a slanted vessel slides across."),
    "sphere": ("Sphere fit", "Does the radius grow and then shrink across slices like slices "
               "of a ball, or stay constant like a tube?"),
    "size": ("Size", "Diameter must be between 3 and 30 mm (the LUNA16 nodule range)."),
    "min_slices": ("Min slices", "Must appear on at least 2 slices."),
    "dot": ("Dot vs line", "On each slice, is the blob round (Hessian ratio q ≈ 1) or "
            "stretched like a vessel seen from the side (q ≈ 0)? Score = q²."),
}


def cfg() -> P0Config:
    return P0Config.from_yaml(CFG_PATH)


# ---------------------------------------------------------------- data access
def scan_index() -> pd.DataFrame:
    return pd.read_csv(CACHE / "scan_index.csv")


def load_tracks(uid: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / "tracks" / f"{uid}.parquet")


def annotations(uid: str):
    a = pd.read_csv(DATA / "annotations.csv")
    e = pd.read_csv(DATA / "annotations_excluded.csv")
    return a[a.seriesuid == uid].reset_index(drop=True), e[e.seriesuid == uid].reset_index(drop=True)


def label_marks(tracks: pd.DataFrame, ann: pd.DataFrame, exc: pd.DataFrame) -> pd.DataFrame:
    """LUNA16 rule per mark: TP if within a nodule's radius, 'ignored' if within an
    excluded finding (not counted either way), otherwise FP."""
    xyz = tracks[["coordX", "coordY", "coordZ"]].to_numpy(float)
    verdict = np.full(len(tracks), "FP", dtype=object)
    nod = np.full(len(tracks), -1)
    best_d = np.full(len(tracks), np.inf)
    for j, r in ann.iterrows():
        d = np.linalg.norm(xyz - [r.coordX, r.coordY, r.coordZ], axis=1)
        hit = (d <= r.diameter_mm / 2) & (d < best_d)
        verdict[hit], nod[hit], best_d[hit] = "TP", j, d[hit]
    for _, r in exc.iterrows():
        rad = (r.diameter_mm if r.diameter_mm > 0 else 10.0) / 2
        d = np.linalg.norm(xyz - [r.coordX, r.coordY, r.coordZ], axis=1)
        verdict[(d <= rad) & (verdict == "FP")] = "ignored"
    return pd.DataFrame({"verdict": verdict, "nodule_idx": nod}, index=tracks.index)


# ---------------------------------------------------------------- live P0 run
def live_path(uid: str) -> Path:
    return CACHE / "live" / f"{uid}.pkl"


def has_live(uid: str) -> bool:
    return live_path(uid).exists()


def run_live(uid: str, path: str | None = None) -> dict:
    """Run the full P0 pipeline on one scan and cache everything the app shows."""
    if path is None:
        path = next(DATA.glob(f"subset*/{uid}.mhd"))
    scan = load_scan(path)
    res = run_p0(scan, cfg(), keep_intermediate=True)
    out = {
        "table": res.table.sort_values("probability", ascending=False).reset_index(drop=True),
        "blobs": res.blobs,
        "tracks": {t.track_id: t.blobs for t in res.tracks},
        "lung_bits": np.packbits(res.lung_mask, axis=None),
        "lung_shape": res.lung_mask.shape,
        "timings": res.timings,
    }
    live_path(uid).parent.mkdir(parents=True, exist_ok=True)
    with open(live_path(uid), "wb") as f:
        pickle.dump(out, f)
    return out


def load_live(uid: str) -> dict:
    with open(live_path(uid), "rb") as f:
        d = pickle.load(f)
    n = int(np.prod(d["lung_shape"]))
    d["lung"] = np.unpackbits(d.pop("lung_bits"))[:n].reshape(d["lung_shape"]).astype(bool)
    return d


# ---------------------------------------------------------------- per-step views
def lung_mask(hu: np.ndarray, spacing) -> np.ndarray:
    filt, _ = preprocess(hu, cfg().preprocess)
    return segment_lungs(filt, spacing, cfg().lung)


def slice_steps(hu2d: np.ndarray, lung2d: np.ndarray, spacing, z: int) -> dict:
    """Everything step 1-3 computes on one slice: filtered image, LoG response,
    best scale, Hessian ratio map and the detected blobs."""
    c = cfg()
    filt, norm = preprocess(hu2d[None], c.preprocess)
    filt, norm = filt[0], norm[0]
    sy, sx = float(spacing[1]), float(spacing[2])
    radii = log_radii(c.candidates)
    img = np.where(lung2d, norm, 0.0).astype(np.float32) if c.candidates.mask_outside else norm
    log, hyy, hxx, hxy = hessian_stacks(img, (sy, sx), radii)
    best = log.argmax(0)
    i0 = np.indices(best.shape)
    L = log[best, i0[0], i0[1]]
    a, b, d = (h[best, i0[0], i0[1]] for h in (hyy, hxx, hxy))
    m = 0.5 * (a + b)
    disc = np.sqrt(0.25 * (a - b) ** 2 + d * d)
    strong, weak = m - disc, m + disc
    q = np.where(strong < 0, np.clip(weak / np.where(strong < 0, strong, -1), 0, 1), 0)
    blobs = detect_slice(norm, filt, lung2d, z, (sy, sx), c.candidates, radii)
    return {
        "filtered": filt,
        "log": np.where(lung2d, L, 0),
        "scale_mm": np.where(lung2d, radii[best], np.nan),
        "q": np.where(lung2d & (L > c.candidates.log_threshold), q, np.nan),
        "blobs": blobs,
        "threshold": c.candidates.log_threshold,
    }


def profile(blobs, spacing) -> dict:
    """Radius along a track plus the fitted sphere and tube (feature step)."""
    dz = float(spacing[0])
    z = np.array([b.z for b in blobs], float) * dz
    r = np.array([b.radius_mm for b in blobs], float)
    out = {"z": z, "r": r, "q": np.array([b.hess_ratio for b in blobs])}
    if len(blobs) >= cfg().features.min_fit_slices:
        f = sphere_profile_fit(z, r, dz, cfg().features.sphere_radius_tolerance)
        zz = np.linspace(z.min() - dz, z.max() + dz, 100)
        out.update(fit=f, zz=zz, sphere=sphere_profile(zz, f["sphere_R_mm"], f["sphere_z0_mm"]),
                   tube=np.full_like(zz, r.mean()))
    return out


# ---------------------------------------------------------------- in-memory run with progress
def _detect_chunk(args):
    norm, hu, lung, zs, spacing_yx, ccfg = args
    radii = log_radii(ccfg)
    out = {}
    for k, z in enumerate(zs):
        b = detect_slice(norm[k], hu[k], lung[k], z, spacing_yx, ccfg, radii)
        if b:
            out[z] = b
    return out


def run_pipeline(scan, progress=lambda frac, msg: None, workers: int = 0) -> dict:
    """Same steps and result as lungnod.pipeline.run_p0, kept in memory, with progress
    reports. Per-slice work runs in threads (numpy/scipy release the GIL)."""
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    from scipy import ndimage as ndi

    from lungnod.lung_mask import _air_components, _drop_small_3d, _fill_and_close
    from lungnod.pipeline import tracks_to_table
    from lungnod.tracking import track_blobs

    c = cfg()
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    ex = ThreadPoolExecutor(workers)
    nz = scan.hu.shape[0]
    parts = [list(range(i, nz, workers)) for i in range(workers)]
    tm = {}

    # 1. preprocessing: clip, 3x3 median per slice, scale to [0, 1]
    t0 = time.perf_counter()
    progress(0.02, "Preprocessing")
    pc = c.preprocess
    hu = np.clip(scan.hu, pc.hu_min, pc.hu_max).astype(np.float32)
    if pc.median_size > 1:
        def med(zs):
            for z in zs:
                hu[z] = ndi.median_filter(hu[z], size=pc.median_size)
        list(ex.map(med, parts))
    norm = ((hu - pc.hu_min) / (pc.hu_max - pc.hu_min)).astype(np.float32)
    tm["preprocess"] = time.perf_counter() - t0

    # 2. lung mask (same steps as lungnod.lung_mask.segment_lungs)
    t0 = time.perf_counter()
    progress(0.10, "Lung segmentation")
    lc = c.lung
    sp_yx = (float(scan.spacing[1]), float(scan.spacing[2]))
    lung = np.zeros(hu.shape, dtype=bool)

    def air(zs):
        for z in zs:
            lung[z] = _air_components(hu[z], sp_yx, lc)
    list(ex.map(air, parts))
    if lc.min_3d_fraction > 0 and lung.any():
        lung = _drop_small_3d(lung, lc.min_3d_fraction)

    def close(zs):
        for z in zs:
            if lung[z].any():
                lung[z] = _fill_and_close(lung[z], sp_yx, lc.closing_radius_mm)
    list(ex.map(close, parts))
    tm["lung mask"] = time.perf_counter() - t0

    # 3. candidate detection on every lung slice
    t0 = time.perf_counter()
    zs = [z for z in range(nz) if lung[z].any()]
    chunks = [sorted(zs[i::workers * 4]) for i in range(workers * 4)]
    chunks = [ch for ch in chunks if ch]
    blobs: dict = {}
    futs = [ex.submit(_detect_chunk, (norm[ch], hu[ch], lung[ch], ch, sp_yx, c.candidates)) for ch in chunks]
    for i, f in enumerate(futs):
        blobs.update(f.result())
        progress(0.20 + 0.65 * (i + 1) / len(futs), "Candidate detection")
    blobs = dict(sorted(blobs.items()))
    ex.shutdown()
    tm["candidates"] = time.perf_counter() - t0

    # 4. tracking, 5. features + scoring
    t0 = time.perf_counter()
    progress(0.88, "Tracking across slices")
    tracks = track_blobs(blobs, scan.spacing, c.tracking)
    tm["tracking"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    progress(0.94, "Shape features and scoring")
    table = tracks_to_table(scan, tracks, c)
    tm["features + scoring"] = time.perf_counter() - t0
    progress(1.0, "Done")
    return {
        "table": table.sort_values("probability", ascending=False).reset_index(drop=True),
        "tracks": {t.track_id: t.blobs for t in tracks},
        "lung": lung,
        "timings": tm,
    }
