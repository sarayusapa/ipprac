"""FROC / CPM following the LUNA16 protocol (Section 11.2).

Matching rules replicate the official noduleCADEvaluationLUNA16.py:
  * a candidate hits a nodule if it lies within the nodule radius (diameter / 2)
  * a nodule's score is the highest probability among the candidates that hit it
  * candidates that hit only an irrelevant finding (annotations_excluded.csv)
    are ignored (neither TP nor FP); excluded findings with diameter -1 get 10 mm
  * every other candidate is a false positive
  * CPM = mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FPs per scan

This re-implementation is for development. Report final numbers with the
official LUNA16 script.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FP_RATES = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])


@dataclass
class SeriesMatch:
    nodule_scores: np.ndarray  # one per included nodule; -inf if missed
    fp_scores: np.ndarray


def match_series(
    cands: pd.DataFrame, annots: pd.DataFrame, excluded: pd.DataFrame | None
) -> SeriesMatch:
    nod_xyz = annots[["coordX", "coordY", "coordZ"]].to_numpy(float).reshape(-1, 3)
    nod_r = annots["diameter_mm"].to_numpy(float) / 2
    if excluded is not None and len(excluded):
        exc_xyz = excluded[["coordX", "coordY", "coordZ"]].to_numpy(float)
        d = excluded["diameter_mm"].to_numpy(float)
        exc_r = np.where(d < 0, 10.0, d) / 2
    else:
        exc_xyz, exc_r = np.zeros((0, 3)), np.zeros(0)

    scores = np.full(len(nod_r), -np.inf)
    fps = []
    c_xyz = cands[["coordX", "coordY", "coordZ"]].to_numpy(float).reshape(-1, 3)
    probs = cands["probability"].to_numpy(float)
    for p, xyz in zip(probs, c_xyz):
        hit = np.linalg.norm(nod_xyz - xyz, axis=1) <= nod_r
        if hit.any():
            scores[hit] = np.maximum(scores[hit], p)
            continue
        if len(exc_r) and (np.linalg.norm(exc_xyz - xyz, axis=1) <= exc_r).any():
            continue
        fps.append(p)
    return SeriesMatch(scores, np.asarray(fps, float))


def match_all(
    submission: pd.DataFrame,
    annotations: pd.DataFrame,
    excluded: pd.DataFrame | None,
    seriesuids: list[str],
) -> dict[str, SeriesMatch]:
    sub_g = dict(tuple(submission.groupby("seriesuid")))
    ann_g = dict(tuple(annotations.groupby("seriesuid")))
    exc_g = dict(tuple(excluded.groupby("seriesuid"))) if excluded is not None else {}
    empty_c = submission.iloc[:0]
    empty_a = annotations.iloc[:0]
    return {
        uid: match_series(sub_g.get(uid, empty_c), ann_g.get(uid, empty_a), exc_g.get(uid))
        for uid in seriesuids
    }


def froc_from_matches(matches: list[SeriesMatch], fp_rates=FP_RATES) -> dict:
    nod = np.concatenate([m.nodule_scores for m in matches]) if matches else np.zeros(0)
    fp = np.concatenate([m.fp_scores for m in matches]) if matches else np.zeros(0)
    n_scans, n_nod = len(matches), len(nod)
    if n_nod == 0 or n_scans == 0:
        return {"cpm": float("nan"), "sens_at": dict.fromkeys(fp_rates, float("nan")),
                "fps": np.zeros(0), "sens": np.zeros(0), "n_nodules": n_nod, "n_scans": n_scans}

    thr = np.unique(np.concatenate([nod[np.isfinite(nod)], fp]))[::-1]
    fp_sorted = np.sort(fp)
    nod_sorted = np.sort(nod[np.isfinite(nod)])
    # counts with score >= t
    fps = (len(fp_sorted) - np.searchsorted(fp_sorted, thr, side="left")) / n_scans
    sens = (len(nod_sorted) - np.searchsorted(nod_sorted, thr, side="left")) / n_nod
    fps = np.concatenate([[0.0], fps])
    sens = np.concatenate([[0.0], sens])
    sens_at = {float(r): float(np.interp(r, fps, sens)) for r in fp_rates}
    return {
        "cpm": float(np.mean(list(sens_at.values()))),
        "sens_at": sens_at,
        "fps": fps,
        "sens": sens,
        "max_sensitivity": float(np.isfinite(nod).mean()),
        "n_nodules": int(n_nod),
        "n_scans": int(n_scans),
    }


def evaluate(
    submission: pd.DataFrame,
    annotations: pd.DataFrame,
    excluded: pd.DataFrame | None,
    seriesuids: list[str],
    n_bootstrap: int = 0,
    seed: int = 0,
) -> dict:
    """FROC + CPM over ``seriesuids``; optional bootstrap 95% CI over scans."""
    matches = match_all(submission, annotations, excluded, seriesuids)
    res = froc_from_matches(list(matches.values()))
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        ms = list(matches.values())
        cpms = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(ms), len(ms))
            cpms.append(froc_from_matches([ms[i] for i in idx])["cpm"])
        cpms = np.array(cpms)
        cpms = cpms[np.isfinite(cpms)]
        res["cpm_ci95"] = (float(np.percentile(cpms, 2.5)), float(np.percentile(cpms, 97.5)))
    return res


def candidate_recall(
    candidates: pd.DataFrame, annotations: pd.DataFrame, seriesuids: list[str]
) -> float:
    """Fraction of nodules hit by at least one candidate, whatever its score
    (upper bound of the detection track)."""
    c = candidates.assign(probability=1.0)
    m = match_all(c, annotations, None, seriesuids)
    nod = np.concatenate([x.nodule_scores for x in m.values()]) if m else np.zeros(0)
    return float(np.isfinite(nod).mean()) if len(nod) else float("nan")
