#!/usr/bin/env python
"""Re-score an existing P0 run with the current rules (no re-detection).

Writes <out>/tracks.csv, submission.csv, config.yaml, seriesuids.txt and, if the run
saved fpr_links.csv (candidate -> tracks passing through it), submission_fpr.csv.

  python scripts/rescore.py --run outputs/p0_all_v6run --out outputs/p0_all_v7 [--config my.yaml]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lungnod.classify import combine, rule_scores
from lungnod.config import P0Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/p0.yaml")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = P0Config.from_yaml(a.config)

    t = pd.read_csv(run / "tracks.csv")
    short = t.n_blobs < cfg.features.min_fit_slices  # sphere fit only with enough slices
    t.loc[short, ["sphere_gain", "sphere_R_mm", "sphere_nrmse", "tube_nrmse"]] = np.nan
    t["sphere_fitted"] = ~short
    s = rule_scores(t, cfg.classifier)
    for r, v in s.items():
        t[f"s_{r}"] = v
    t["probability"] = combine(s, cfg.classifier.rules)
    t.to_csv(out / "tracks.csv", index=False)
    t[t.probability > 0][["seriesuid", "coordX", "coordY", "coordZ", "probability"]].to_csv(
        out / "submission.csv", index=False)
    cfg.to_yaml(out / "config.yaml")
    shutil.copy(run / "seriesuids.txt", out / "seriesuids.txt")

    links = run / "fpr_links.csv"
    if links.exists():
        lk = pd.read_csv(links, dtype={"track_ids": str})
        prob = {(u, i): p for u, i, p in zip(t.seriesuid, t.track_id, t.probability)}

        def best(u, ids):
            if not isinstance(ids, str) or not ids:
                return 0.0
            return max(prob.get((u, int(i)), 0.0) for i in ids.split(";"))

        lk["probability"] = [best(u, ids) for u, ids in zip(lk.seriesuid, lk.track_ids)]
        lk.drop(columns=["track_ids"]).to_csv(out / "submission_fpr.csv", index=False)
        print("re-scored FPR track from fpr_links.csv")
    else:
        print("no fpr_links.csv in the run: FPR track not re-scored (needs a run made with the current code)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
