#!/usr/bin/env python
"""P0 ablation (Section 11.2): which rule removes which FPs.

Rescores tracks.csv with growing rule sets - no need to rerun detection.

  python scripts/ablation.py --run outputs/p0_s0 --annotations data/luna16/annotations.csv \
      --excluded data/luna16/annotations_excluded.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from lungnod.classify import combine, rule_scores
from lungnod.config import P0Config
from lungnod.evaluation import evaluate
from lungnod.viz import plot_froc

_B = ["size", "min_slices"]
STEPS = [
    ("size only", _B),
    ("+ lifetime/aspect", _B + ["aspect"]),
    ("+ drift", _B + ["aspect", "drift"]),
    ("+ sphere fit", _B + ["aspect", "drift", "sphere"]),
    # 2D Hessian dot-vs-line (Li, Sone & Doi 2003): the last shape rule -> P0 default
    ("+ dot (P0)", _B + ["aspect", "drift", "sphere", "dot"]),
    # optional rules, each added to P0
    ("P0 + branching", _B + ["aspect", "drift", "sphere", "dot", "branching"]),
    ("P0 + LoG strength (uses contrast)", _B + ["aspect", "drift", "sphere", "dot", "log_strength"]),
    # earlier version (v2): LoG strength instead of the dot rule
    ("v2: sphere fit + LoG strength", _B + ["aspect", "drift", "sphere", "log_strength"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--excluded", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = P0Config.from_yaml(args.config or run / "config.yaml")
    uids = pd.read_csv(run / "seriesuids.txt", header=None)[0].tolist()
    ann = pd.read_csv(args.annotations)
    exc = pd.read_csv(args.excluded) if args.excluded else None
    tracks = pd.read_csv(run / "tracks.csv")
    has_dot = "hess_ratio" in tracks.columns

    rows, curves = [], {}
    for label, rules in STEPS:
        if "dot" in rules and not has_dot:
            continue  # run made before the Hessian feature existed
        prob = combine(rule_scores(tracks, cfg.classifier), rules)
        sub = tracks.assign(probability=prob)[prob > 0]
        r = evaluate(sub, ann, exc, uids)
        curves[label] = r
        rows.append({"rules": label, "cpm": r["cpm"], **{f"@{k}": v for k, v in r["sens_at"].items()}})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(run / "ablation.csv", index=False)
    plot_froc(curves, run / "ablation_froc.png")


if __name__ == "__main__":
    main()
