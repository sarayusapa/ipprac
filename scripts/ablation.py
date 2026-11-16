#!/usr/bin/env python
"""P0 ablation (Section 11.2): which rule removes which FPs.

Rescores tracks.csv with growing rule sets - no need to rerun detection.

  python scripts/ablation.py --run outputs/p0_s0 --annotations data/luna16/annotations.csv \
      --excluded data/luna16/annotations_excluded.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from lungnod.classify import classify_table
from lungnod.config import P0Config
from lungnod.evaluation import evaluate
from lungnod.viz import plot_froc

STEPS = [
    ("size only", ["size", "min_slices"]),
    ("+ lifetime/aspect", ["size", "min_slices", "aspect"]),
    ("+ drift", ["size", "min_slices", "aspect", "drift"]),
    ("+ sphere fit", ["size", "min_slices", "aspect", "drift", "sphere"]),
    ("+ LoG strength (P0)", ["size", "min_slices", "aspect", "drift", "sphere", "log_strength"]),
    ("+ branching", ["size", "min_slices", "aspect", "drift", "sphere", "log_strength", "branching"]),
    # 2D Hessian dot-vs-line (Li, Sone & Doi 2003), shape-only replacement for LoG strength
    ("sphere + dot", ["size", "min_slices", "aspect", "drift", "sphere", "dot"]),
    ("sphere + dot + LoG", ["size", "min_slices", "aspect", "drift", "sphere", "dot", "log_strength"]),
    ("sphere + dot + branching", ["size", "min_slices", "aspect", "drift", "sphere", "dot", "branching"]),
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
        t = classify_table(tracks, cfg.classifier, rules=rules)
        sub = t[t["probability"] > 0]
        r = evaluate(sub, ann, exc, uids)
        curves[label] = r
        rows.append({"rules": label, "cpm": r["cpm"], **{f"@{k}": v for k, v in r["sens_at"].items()}})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(run / "ablation.csv", index=False)
    plot_froc(curves, run / "ablation_froc.png")


if __name__ == "__main__":
    main()
