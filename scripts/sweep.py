#!/usr/bin/env python
"""Fast classifier-parameter experiments on an existing run (no re-detection).

Track-to-nodule matching does not depend on the scores, so it is computed once;
each parameter setting then only recomputes rule scores and the FROC. Results are
reported separately on a DEV split (subsets 0-4, used for decisions) and a TEST
split (subsets 5-9, only for checking), so tuning cannot leak into the test numbers.

  python scripts/sweep.py --run outputs/p0_all_v3 --data-root data/luna16
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from lungnod.classify import combine, rule_scores
from lungnod.config import P0Config
from lungnod.evaluation import FP_RATES
from lungnod.io import find_scans

DEV, TEST = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]


class Matcher:
    """Precomputed LUNA16 matching (same rules as evaluation.py / the official script)."""

    def __init__(self, tracks: pd.DataFrame, ann: pd.DataFrame, exc: pd.DataFrame, uids: list[str]):
        self.uids = list(uids)
        keep = tracks.seriesuid.isin(self.uids).to_numpy()
        self.rows = np.nonzero(keep)[0]
        t = tracks.iloc[self.rows]
        ann = ann[ann.seriesuid.isin(self.uids)].reset_index(drop=True)
        self.n_nod = len(ann)
        self.n_scans = len(self.uids)
        is_fp = np.ones(len(t), bool)
        pairs_t, pairs_n = [], []
        t_uid = t.seriesuid.to_numpy()
        xyz = t[["coordX", "coordY", "coordZ"]].to_numpy(float)
        by_uid = {u: np.nonzero(t_uid == u)[0] for u in self.uids}
        for j, r in ann.iterrows():
            idx = by_uid.get(r.seriesuid, np.zeros(0, int))
            d = np.linalg.norm(xyz[idx] - [r.coordX, r.coordY, r.coordZ], axis=1)
            h = idx[d <= r.diameter_mm / 2]
            pairs_t.append(h)
            pairs_n.append(np.full(len(h), j))
            is_fp[h] = False
        for _, r in exc[exc.seriesuid.isin(self.uids)].iterrows():
            idx = by_uid.get(r.seriesuid, np.zeros(0, int))
            rad = (r.diameter_mm if r.diameter_mm > 0 else 10.0) / 2
            d = np.linalg.norm(xyz[idx] - [r.coordX, r.coordY, r.coordZ], axis=1)
            h = idx[(d <= rad) & is_fp[idx]]
            is_fp[h] = False  # ignored: neither TP nor FP
        self.pt = np.concatenate(pairs_t) if pairs_t else np.zeros(0, int)
        self.pn = np.concatenate(pairs_n) if pairs_n else np.zeros(0, int)
        self.is_fp = is_fp

    def cpm(self, prob_all: np.ndarray) -> dict:
        p = prob_all[self.rows]
        nod = np.full(self.n_nod, -np.inf)
        if len(self.pt):
            np.maximum.at(nod, self.pn, np.where(p[self.pt] > 0, p[self.pt], -np.inf))
        fp = p[self.is_fp]
        fp = fp[fp > 0]
        thr = np.unique(np.concatenate([nod[np.isfinite(nod)], fp]))[::-1]
        fs, ns = np.sort(fp), np.sort(nod[np.isfinite(nod)])
        fps = np.concatenate([[0], (len(fs) - np.searchsorted(fs, thr, "left")) / self.n_scans])
        sens = np.concatenate([[0], (len(ns) - np.searchsorted(ns, thr, "left")) / self.n_nod])
        s = np.interp(FP_RATES, fps, sens)
        return {"cpm": float(s.mean()), "s1": float(s[3]), "s4": float(s[5])}


def score(tracks: pd.DataFrame, ccfg, rules) -> np.ndarray:
    return combine(rule_scores(tracks, ccfg), rules)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    tracks = pd.read_csv(run / "tracks.csv")
    ann = pd.read_csv(Path(args.data_root) / "annotations.csv")
    exc = pd.read_csv(Path(args.data_root) / "annotations_excluded.csv")
    dev = Matcher(tracks, ann, exc, [p.stem for p in find_scans(args.data_root, DEV)])
    test = Matcher(tracks, ann, exc, [p.stem for p in find_scans(args.data_root, TEST)])
    base = P0Config.from_yaml(run / "config.yaml").classifier
    base.rules = ["aspect", "drift", "sphere", "size", "min_slices", "dot"]

    rows = []

    def run_exp(group, label, ccfg=None, rules=None):
        ccfg = ccfg or base
        rules = rules or ccfg.rules
        p = score(tracks, ccfg, rules)
        d, t = dev.cpm(p), test.cpm(p)
        rows.append({"group": group, "setting": label, "dev_cpm": d["cpm"], "test_cpm": t["cpm"],
                     "dev_s@1": d["s1"], "dev_s@4": d["s4"]})
        print(f"{group:14s} {label:34s} dev {d['cpm']:.3f}  test {t['cpm']:.3f}", flush=True)

    def vary(group, field, values):
        for v in values:
            c = copy.deepcopy(base)
            setattr(c, field, v)
            run_exp(group, f"{field}={v}", c)

    run_exp("baseline", "P0 v3 default")
    # leave-one-rule-out
    for r in base.rules:
        run_exp("drop-one", f"without {r}", rules=[x for x in base.rules if x != r])
    run_exp("add-one", "+ log_strength", rules=base.rules + ["log_strength"])
    run_exp("add-one", "+ branching", rules=base.rules + ["branching"])
    # sensitivity of each hand-set value (one factor at a time)
    vary("dot", "dot_form", ["ratio_sq", "sigmoid"])
    vary("aspect", "aspect_thresh", [1.5, 2.0, 2.5, 3.0, 4.0])
    vary("aspect", "aspect_scale", [0.1, 0.3, 0.5, 1.0])
    vary("drift", "drift_thresh", [0.5, 1.0, 1.5, 2.0])
    vary("drift", "drift_scale", [0.1, 0.2, 0.5])
    vary("sphere", "sphere_scale", [0.02, 0.05, 0.1, 0.2])
    vary("sphere", "sphere_center", [-0.1, -0.05, 0.0, 0.05])
    vary("sphere", "neutral_sphere_score", [0.2, 0.5, 0.8, 1.0])
    vary("size", "diam_min_mm", [2.0, 3.0, 4.0])
    vary("size", "diam_max_mm", [20.0, 30.0, 40.0])
    vary("min_slices", "min_slices", [1, 2, 3])

    df = pd.DataFrame(rows)
    out = Path(args.out or run / "sweep.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
