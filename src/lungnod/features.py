"""Step 05 - Feature extraction (Section 8.5). Shape across slices only, never brightness.

Per track:
  lifetime_mm   = slices spanned x slice thickness       nodule ~ diameter, vessel >> diameter
  aspect        = lifetime / max diameter                 nodule ~ 1,        vessel > 2.5
  drift_ratio   = lateral shift over the track / radius   nodule ~ 0,        slanted vessel > 1
                  (line fit of centre vs z; independent of slice thickness)
  sphere fit    r(z) = sqrt(R^2 - (z - z0)^2) vs constant radius (tube)
  branches      splits / merges seen while tracking       nodule: no,        vessel: often
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .config import FeatureConfig
from .tracking import Track


def sphere_profile(z: np.ndarray, R: float, z0: float) -> np.ndarray:
    return np.sqrt(np.clip(R * R - (z - z0) ** 2, 0.0, None))


def sphere_profile_fit(
    z_mm: np.ndarray, r_mm: np.ndarray, dz: float, tol: float
) -> dict:
    """Least-squares fit of a sliced sphere vs a constant-radius tube.

    R is bounded to what a sphere seen on these slices can be: at least the
    widest measured radius, at most tol * sqrt(r_max^2 + (dz/2)^2) (the equator
    may fall between two slices). Without this bound a huge sphere would mimic
    a tube. Returns normalised RMSEs (RMSE / mean radius) and their difference.
    """
    z_mm = np.asarray(z_mm, float)
    r_mm = np.asarray(r_mm, float)
    r_max, mean_r = r_mm.max(), r_mm.mean()
    R_lo = r_max
    R_hi = tol * np.sqrt(r_max**2 + (dz / 2) ** 2)
    z_lo, z_hi = z_mm.min() - dz, z_mm.max() + dz
    x0 = [0.5 * (R_lo + R_hi), float(np.clip(z_mm[np.argmax(r_mm)], z_lo, z_hi))]

    res = least_squares(
        lambda p: sphere_profile(z_mm, p[0], p[1]) - r_mm,
        x0=x0,
        bounds=([R_lo, z_lo], [R_hi, z_hi]),
    )
    R, z0 = res.x
    rmse_sphere = float(np.sqrt(np.mean(res.fun**2)))
    rmse_tube = float(np.std(r_mm))
    nrmse_sphere = rmse_sphere / mean_r
    nrmse_tube = rmse_tube / mean_r
    return {
        "sphere_R_mm": float(R),
        "sphere_z0_mm": float(z0),
        "sphere_nrmse": nrmse_sphere,
        "tube_nrmse": nrmse_tube,
        "sphere_gain": nrmse_tube - nrmse_sphere,  # > 0: sphere explains the profile better
    }


def track_features(track: Track, spacing: np.ndarray, cfg: FeatureConfig) -> dict:
    dz, sy, sx = (float(s) for s in spacing)
    blobs = track.blobs
    z = np.array([b.z for b in blobs], float)
    r = np.array([b.radius_mm for b in blobs], float)
    ys = np.array([b.y for b in blobs], float)
    xs = np.array([b.x for b in blobs], float)

    lifetime = track.n_slices * dz
    max_diam = 2 * r.max()
    if len(blobs) > 1:
        step = np.hypot(np.diff(ys) * sy, np.diff(xs) * sx) / np.diff(z)
        drift = float(step.mean())
        # Spacing-independent drift: fit centre(z) with a straight line (least squares
        # averages out per-slice jitter), take the total lateral shift along the track
        # and divide by the blob radius. Sphere: ~0. Slanted tube: grows with length.
        zm = z * dz
        vy = np.polyfit(zm, ys * sy, 1)[0]
        vx = np.polyfit(zm, xs * sx, 1)[0]
        shift = float(np.hypot(vy, vx) * (zm.max() - zm.min()))
    else:
        drift, shift = 0.0, 0.0

    feat = {
        "n_slices": track.n_slices,
        "n_blobs": len(blobs),
        "lifetime_mm": lifetime,
        "max_diameter_mm": max_diam,
        "aspect": lifetime / max_diam,
        "drift_mm_per_slice": drift,  # report's original definition (kept for reference)
        "lateral_shift_mm": shift,
        "drift_ratio": shift / r.max(),  # shift / radius, used by the drift rule
        "branches": track.branches,
        "sphere_fitted": False,
        "sphere_R_mm": np.nan,
        "sphere_nrmse": np.nan,
        "tube_nrmse": np.nan,
        "sphere_gain": np.nan,
    }
    if len(blobs) >= cfg.min_fit_slices:
        feat.update(sphere_profile_fit(z * dz, r, dz, cfg.sphere_radius_tolerance))
        feat.pop("sphere_z0_mm")
        feat["sphere_fitted"] = True

    w = track.widest
    feat.update(
        {
            "mean_hu": w.mean_hu,
            "std_hu": w.std_hu,
            "circularity": w.circularity,
            "eccentricity": w.eccentricity,
            "wall_dist_mm": w.wall_dist_mm,
            "truncated_frac": float(np.mean([b.truncated for b in blobs])),
            "max_log_response": float(max(b.log_response for b in blobs)),
            # median over slices: a nodule touching a vessel on one slice keeps its score
            "hess_ratio": float(np.median([b.hess_ratio for b in blobs])),
            # alternative summaries, for experiments only
            "hess_ratio_mean": float(np.mean([b.hess_ratio for b in blobs])),
            "hess_ratio_min": float(np.min([b.hess_ratio for b in blobs])),
            "hess_ratio_widest": float(w.hess_ratio),
            "hess_ratio_strongest": float(max(blobs, key=lambda b: b.log_response).hess_ratio),
        }
    )
    return feat
