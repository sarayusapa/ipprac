"""Step 06 - Rule-based classification (Section 8.6). No training.

    s_aspect  = sigmoid((2.5 - aspect) / 0.3)
    s_drift   = sigmoid((1.0 - drift_ratio) / 0.2)       drift_ratio = lateral shift / radius
    s_sphere  = sigmoid((sphere_gain - 0) / 0.05)     (neutral 0.5 if < 3 slices)
    s_size    = 1 if 3 mm <= max diameter <= 30 mm else 0
    s_min_sl  = 1 if the track spans >= 2 slices else 0
    s_log     = min(peak LoG response / (2/e), 1)        detector confidence; 2/e is the
                                                       response of a full-contrast disc [11]
    s_branch  = exp(-0.5 * branches)                  (optional, off by default)
    s_dot     = sigmoid((2.5 - inplane_aspect) / 0.3)  in-plane elongation from the 2D Hessian
                (Li, Sone & Doi 2003): ratio q = |l_small|/|l_large| (median over slices),
                inplane_aspect = sqrt(3/q - 2), the axis ratio of an ellipse with that q
                (Gaussian blob model at the LoG scale sigma = r/sqrt2). Same threshold and
                softness as the 3D aspect rule; contrast-invariant, so it can replace s_log.

    score = product of the enabled rules, in [0, 1]  -> ranks candidates for FROC.

Functions are vectorised so the same code scores one track or rescored a whole
table for the ablation study.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import ClassifierConfig

ALL_RULES = ("aspect", "drift", "sphere", "size", "min_slices", "log_strength", "branching", "dot")
LOG_MAX_RESPONSE = 2 / np.e  # sigma^2-normalised LoG peak for a unit-contrast disc


def inplane_aspect(q) -> np.ndarray:
    """Axis ratio k of an ellipse whose Hessian ratio at the LoG scale is q.
    Model: ellipse ~ Gaussian with variances (r/2)^2, (k r/2)^2, smoothed at sigma^2 = r^2/2
    -> q = (1/4 + 1/2) / (k^2/4 + 1/2) = 3 / (k^2 + 2). Checked numerically on discs/ellipses."""
    q = np.clip(np.asarray(q, float), 1e-6, 1.0)
    return np.sqrt(np.maximum(3.0 / q - 2.0, 1.0))


def rule_scores(f: Mapping, cfg: ClassifierConfig) -> dict[str, np.ndarray]:
    aspect = np.asarray(f["aspect"], float)
    drift = np.asarray(f["drift_ratio"], float)
    log_resp = np.asarray(f["max_log_response"], float)
    gain = np.asarray(f["sphere_gain"], float)
    diam = np.asarray(f["max_diameter_mm"], float)
    n_sl = np.asarray(f["n_slices"], float)
    br = np.asarray(f["branches"], float)
    q = np.asarray(f["hess_ratio"], float) if "hess_ratio" in f else np.ones_like(aspect)

    s_sphere = expit((gain - cfg.sphere_center) / cfg.sphere_scale)
    s_sphere = np.where(np.isnan(gain), cfg.neutral_sphere_score, s_sphere)
    return {
        "aspect": expit((cfg.aspect_thresh - aspect) / cfg.aspect_scale),
        "drift": expit((cfg.drift_thresh - drift) / cfg.drift_scale),
        "sphere": s_sphere,
        "size": ((diam >= cfg.diam_min_mm) & (diam <= cfg.diam_max_mm)).astype(float),
        "min_slices": (n_sl >= cfg.min_slices).astype(float),
        "log_strength": np.clip(log_resp / LOG_MAX_RESPONSE, 0.0, 1.0),
        "branching": np.exp(-cfg.branch_scale * br),
        "dot": expit((cfg.aspect_thresh - inplane_aspect(q)) / cfg.aspect_scale),
    }


def combine(scores: Mapping[str, np.ndarray], rules) -> np.ndarray:
    unknown = set(rules) - set(ALL_RULES)
    if unknown:
        raise ValueError(f"Unknown rules: {unknown}")
    out = np.ones_like(next(iter(scores.values())), dtype=float)
    for r in rules:
        out = out * scores[r]
    return out


def explain(f: Mapping, s: Mapping, cfg: ClassifierConfig) -> str:
    """Human-readable reason for one track, e.g.
    'rejected: lifetime 34.0 mm, aspect 4.1 (> 2.5), drift 0.9 mm/slice (> 0.5), poor sphere fit'."""
    reasons = []
    rules = cfg.rules
    if "min_slices" in rules and s["min_slices"] < 0.5:
        reasons.append(f"seen on {int(f['n_slices'])} slice(s) (< {cfg.min_slices})")
    if "size" in rules and s["size"] < 0.5:
        reasons.append(
            f"diameter {f['max_diameter_mm']:.1f} mm outside [{cfg.diam_min_mm:g}, {cfg.diam_max_mm:g}]"
        )
    if "aspect" in rules and s["aspect"] < 0.5:
        reasons.append(
            f"lifetime {f['lifetime_mm']:.1f} mm, aspect {f['aspect']:.1f} (> {cfg.aspect_thresh:g})"
        )
    if "drift" in rules and s["drift"] < 0.5:
        reasons.append(
            f"centre shifts {f['drift_ratio']:.1f} x radius (> {cfg.drift_thresh:g})"
        )
    if "sphere" in rules and s["sphere"] < 0.5:
        reasons.append(f"poor sphere fit (gain {f['sphere_gain']:.2f})")
    if "log_strength" in rules and s["log_strength"] < 0.5:
        reasons.append(f"weak blob (LoG strength {s['log_strength']:.2f})")
    if "branching" in rules and s["branching"] < 0.5:
        reasons.append(f"branches {int(f['branches'])}")
    if "dot" in rules and s["dot"] < 0.5:
        reasons.append(f"line-like in plane (in-plane aspect {float(inplane_aspect(f['hess_ratio'])):.1f})")
    prob = float(np.prod([s[r] for r in rules]))
    if prob >= cfg.accept_threshold:
        return "accepted"
    return "rejected: " + (", ".join(reasons) if reasons else "low combined score")


def classify_table(df: pd.DataFrame, cfg: ClassifierConfig, rules=None) -> pd.DataFrame:
    """Add s_* columns, probability, decision and explanation to a feature table."""
    rules = list(cfg.rules if rules is None else rules)
    df = df.copy()
    if df.empty:
        for c in [f"s_{r}" for r in ALL_RULES] + ["probability", "decision", "explanation"]:
            df[c] = pd.Series(dtype=float if c.startswith(("s_", "prob")) else object)
        return df
    scores = rule_scores(df, cfg)
    for r, v in scores.items():
        df[f"s_{r}"] = v
    df["probability"] = combine(scores, rules)
    df["decision"] = np.where(df["probability"] >= cfg.accept_threshold, "nodule", "reject")
    sub_cfg = ClassifierConfig(**{**cfg.__dict__, "rules": rules})
    df["explanation"] = [
        explain(row, {r: row[f"s_{r}"] for r in ALL_RULES}, sub_cfg)
        for _, row in df.iterrows()
    ]
    return df
