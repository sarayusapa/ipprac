#!/usr/bin/env python
"""Second experiment set: rule *forms* and combinations (dev = subsets 0-4, test = 5-9).
Uses the precomputed matching from sweep.py."""
import argparse, copy
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit
from lungnod.classify import rule_scores, inplane_aspect
from lungnod.config import P0Config
from lungnod.io import find_scans
from sweep import Matcher, DEV, TEST

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--data-root", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args(); run = Path(a.run)
t = pd.read_csv(run / "tracks.csv")
ann = pd.read_csv(Path(a.data_root) / "annotations.csv"); exc = pd.read_csv(Path(a.data_root) / "annotations_excluded.csv")
dev = Matcher(t, ann, exc, [p.stem for p in find_scans(a.data_root, DEV)])
test = Matcher(t, ann, exc, [p.stem for p in find_scans(a.data_root, TEST)])
base = P0Config.from_yaml(run / "config.yaml").classifier
S = rule_scores(t, base)
q = np.clip(t.hess_ratio.to_numpy(float), 0, 1)
k = inplane_aspect(q)
gain = t.sphere_gain.to_numpy(float)
rows = []
def ev(group, label, p):
    d, te = dev.cpm(p), test.cpm(p)
    rows.append(dict(group=group, setting=label, dev_cpm=d["cpm"], test_cpm=te["cpm"], dev_s1=d["s1"], dev_s4=d["s4"]))
    print(f"{group:10s} {label:52s} dev {d['cpm']:.3f}  test {te['cpm']:.3f}", flush=True)
core = S["drift"] * S["sphere"] * S["size"] * S["min_slices"] * S["aspect"]
ev("base", "v3 default (dot sigmoid 2.5/0.3)", core * S["dot"])
for th, sc in [(1.5, 0.3), (1.75, 0.3), (1.5, 0.5), (1.75, 0.5), (2.0, 0.5)]:
    ev("dot-sig", f"dot sigmoid thresh {th} scale {sc}", core * expit((th - k) / sc))
ev("dot-form", "s_dot = q (no parameter)", core * q)
ev("dot-form", "s_dot = q^2", core * q**2)
ev("dot-form", "s_dot = sqrt(q)", core * np.sqrt(q))
ev("dot-form", "s_dot = 1/k (inverse in-plane aspect)", core * (1 / k))
ev("dot-form", "s_dot = 1/k^2", core * (1 / k**2))
# sphere form
sph_neutral = np.isnan(gain)
for c, sc in [(-0.1, 0.05), (-0.05, 0.02), (0.0, 0.02), (-0.1, 0.02)]:
    s_sph = np.where(sph_neutral, 0.5, expit((gain - c) / sc))
    ev("sphere", f"q-dot, sphere center {c} scale {sc}", core / S["sphere"] * s_sph * q)
ev("combo", "q-dot + log_strength", core * q * S["log_strength"])
ev("combo", "q-dot + branching", core * q * S["branching"])
ev("combo", "q-dot, no aspect rule", core / S["aspect"] * q)
ev("combo", "q-dot, no drift rule", core / S["drift"] * q)
ev("combo", "q-dot, no sphere rule", core / S["sphere"] * q)
pd.DataFrame(rows).to_csv(a.out or run / "sweep_forms.csv", index=False)
