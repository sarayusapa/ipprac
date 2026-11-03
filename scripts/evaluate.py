#!/usr/bin/env python
"""FROC / CPM of a submission against LUNA16 annotations (Section 11.2).

Example:
  python scripts/evaluate.py --run outputs/p0_s0 \
      --annotations data/luna16/annotations.csv \
      --excluded data/luna16/annotations_excluded.csv --bootstrap 1000
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from lungnod.evaluation import candidate_recall, evaluate
from lungnod.viz import plot_froc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output folder of run_p0.py")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--excluded", default=None)
    ap.add_argument("--bootstrap", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    uids = pd.read_csv(run / "seriesuids.txt", header=None)[0].tolist()
    ann = pd.read_csv(args.annotations)
    exc = pd.read_csv(args.excluded) if args.excluded else None

    report, curves = {}, {}
    tracks = pd.read_csv(run / "tracks.csv")
    report["candidate_recall_all_tracks"] = candidate_recall(tracks, ann, uids)

    for name in ["submission", "submission_fpr"]:
        f = run / f"{name}.csv"
        if not f.exists():
            continue
        r = evaluate(pd.read_csv(f), ann, exc, uids, n_bootstrap=args.bootstrap)
        curves[name] = r
        report[name] = {k: v for k, v in r.items() if k not in ("fps", "sens")}
        print(f"\n== {name} ==  ({r['n_scans']} scans, {r['n_nodules']} nodules)")
        print(f"CPM = {r['cpm']:.3f}" + (f"  95% CI {r['cpm_ci95'][0]:.3f}-{r['cpm_ci95'][1]:.3f}" if "cpm_ci95" in r else ""))
        for rate, s in r["sens_at"].items():
            print(f"  sens @ {rate:>5} FP/scan: {s:.3f}")
        print(f"  max sensitivity: {r['max_sensitivity']:.3f}")

    print(f"\ncandidate recall (any track, any score): {report['candidate_recall_all_tracks']:.3f}")
    with open(run / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    if curves:
        plot_froc(curves, run / "froc.png")
    print(f"wrote {run / 'metrics.json'}, {run / 'froc.png'}")


if __name__ == "__main__":
    main()
