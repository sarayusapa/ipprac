#!/usr/bin/env python
"""Run P0 on LUNA16 subsets.

Writes to --out:
  tracks.csv            every track with all features, rule scores, explanation
  submission.csv        detection track (seriesuid, coordX, coordY, coordZ, probability)
  submission_fpr.csv    FP-reduction track, if --candidates is given
  timings.csv, config.yaml

Example:
  python scripts/run_p0.py --data-root data/luna16 --subsets 0 --out outputs/p0_s0 \
      --candidates data/luna16/candidates_V2.csv --workers 4
"""
import argparse
from pathlib import Path

import pandas as pd

from lungnod.batch import run_batch
from lungnod.config import P0Config
from lungnod.io import find_scans


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder containing subset0..subset9")
    ap.add_argument("--subsets", type=int, nargs="*", default=[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="YAML overriding defaults")
    ap.add_argument("--candidates", default=None, help="candidates_V2.csv for the FPR track")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--noise-hu", type=float, default=0.0,
                    help="added-noise experiment: Gaussian noise std in HU (Sec. 11.2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N scans (debug)")
    args = ap.parse_args()

    cfg = P0Config.from_yaml(args.config)
    paths = find_scans(args.data_root, args.subsets)[: args.limit]
    if not paths:
        raise SystemExit("No .mhd files found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out / "config.yaml")

    cands = pd.read_csv(args.candidates) if args.candidates else None
    tables, fpr, timings = run_batch(paths, cfg, cands, workers=args.workers,
                                     noise_hu=args.noise_hu, seed=args.seed)

    tables.to_csv(out / "tracks.csv", index=False)
    sub = tables[tables["probability"] > 0][["seriesuid", "coordX", "coordY", "coordZ", "probability"]]
    sub.to_csv(out / "submission.csv", index=False)
    if fpr is not None:
        fpr.drop(columns=["track_ids"], errors="ignore").to_csv(out / "submission_fpr.csv", index=False)
        if "track_ids" in fpr:
            fpr[["seriesuid", "coordX", "coordY", "coordZ", "track_ids"]].to_csv(out / "fpr_links.csv", index=False)
    timings.to_csv(out / "timings.csv", index=False)
    pd.Series([p.stem for p in paths]).to_csv(out / "seriesuids.txt", index=False, header=False)

    print(f"{len(paths)} scans, {len(tables)} tracks, {len(sub)} with score > 0")
    print(f"mean tracks/scan: {len(tables) / len(paths):.1f}")
    print(f"mean seconds/scan: {timings.drop(columns='seriesuid').sum(axis=1).mean():.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
