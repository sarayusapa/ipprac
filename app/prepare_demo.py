#!/usr/bin/env python
"""One-time preparation for the demo app (app/streamlit_app.py).

1. Splits a full P0 run's tracks.csv (3.8M rows) into one small parquet file per scan,
   so the app loads a scan's candidates instantly.
2. Builds a scan index: subset, path, number of nodules, and how well P0 ranks them in
   that scan (used to suggest showcase scans).
3. Optionally (--live UID ...) runs P0 end to end on a few scans and caches the full
   result (per-slice blobs, tracks, lung mask) so the app can show every step.

  python app/prepare_demo.py --run outputs/p0_all_v6 --data-root data/luna16
  python app/prepare_demo.py --live-showcase 6          # also cache 6 showcase scans
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from demo_core import CACHE, label_marks, run_live  # noqa: E402

COLS = [
    "seriesuid", "track_id", "coordX", "coordY", "coordZ", "vox_z", "vox_y", "vox_x",
    "z_first", "z_last", "n_slices", "n_blobs", "lifetime_mm", "max_diameter_mm", "aspect",
    "drift_ratio", "branches", "sphere_fitted", "sphere_R_mm", "sphere_gain", "wall_dist_mm",
    "hess_ratio_mean", "mean_hu", "max_log_response",
    "s_aspect", "s_drift", "s_sphere", "s_size", "s_min_slices", "s_dot", "probability",
]


def split_tracks(run: Path, data_root: Path):
    out = CACHE / "tracks"
    out.mkdir(parents=True, exist_ok=True)
    print("reading", run / "tracks.csv")
    t = pd.read_csv(run / "tracks.csv", usecols=COLS)
    ann = pd.read_csv(data_root / "annotations.csv")
    exc = pd.read_csv(data_root / "annotations_excluded.csv")
    paths = {p.stem: p for p in data_root.glob("subset*/*.mhd")}
    rows = []
    for uid, g in t.groupby("seriesuid", sort=False):
        g = g.drop(columns="seriesuid").sort_values("probability", ascending=False)
        g = g.reset_index(drop=True)
        g.to_parquet(out / f"{uid}.parquet", index=False)
        a = ann[ann.seriesuid == uid].reset_index(drop=True)
        lab = label_marks(g, a, exc[exc.seriesuid == uid])
        # each nodule's best score, and the false alarms ranked above the weakest nodule
        nod_best = []
        for j in range(len(a)):
            s = g.probability[lab.nodule_idx == j]
            nod_best.append(float(s.max()) if len(s) else 0.0)
        fps = g.probability[(lab.verdict == "FP") & (g.probability > 0)].to_numpy()
        worst = min(nod_best) if nod_best else np.nan
        p = paths.get(uid)
        rows.append(
            dict(
                seriesuid=uid,
                subset=p.parent.name if p else "",
                path=str(p) if p else "",
                n_nodules=len(a),
                diameters=", ".join(f"{d:.1f}" for d in a.diameter_mm),
                nodule_scores=", ".join(f"{s:.2f}" for s in nod_best),
                n_marks=int((g.probability > 0).sum()),
                fps_above_weakest=int((fps > worst).sum()) if len(a) and worst > 0 else -1,
                fps_above_best=int((fps > max(nod_best)).sum()) if len(a) and max(nod_best) > 0 else -1,
            )
        )
    idx = pd.DataFrame(rows)
    good = idx[(idx.n_nodules > 0) & (idx.fps_above_weakest >= 0)]
    idx["showcase_rank"] = np.nan
    order = good.sort_values(["fps_above_weakest", "n_nodules"], ascending=[True, False]).index
    idx.loc[order, "showcase_rank"] = np.arange(len(order))
    idx.to_csv(CACHE / "scan_index.csv", index=False)
    print(f"wrote {len(idx)} scans to {out}; {(idx.fps_above_weakest == 0).sum()} scans with every "
          f"nodule above every false alarm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/p0_all_v6")
    ap.add_argument("--data-root", default="data/luna16")
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument("--live", nargs="*", default=[], help="series UIDs to run P0 on and cache")
    ap.add_argument("--live-showcase", type=int, default=0, help="cache the N best showcase scans")
    a = ap.parse_args()
    run, data_root = ROOT / a.run, ROOT / a.data_root
    if not a.skip_split:
        split_tracks(run, data_root)
    uids = list(a.live)
    if a.live_showcase:
        idx = pd.read_csv(CACHE / "scan_index.csv")
        uids += idx.dropna(subset=["showcase_rank"]).sort_values("showcase_rank").seriesuid.head(
            a.live_showcase).tolist()
    for uid in uids:
        print("running P0 on", uid, flush=True)
        r = run_live(uid)
        print("  done:", {k: round(v, 1) for k, v in r["timings"].items()}, flush=True)


if __name__ == "__main__":
    main()
