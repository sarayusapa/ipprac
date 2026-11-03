"""End-to-end check on the synthetic phantom: nodules accepted, vessels and ribs rejected."""
import numpy as np
import pytest

from lungnod import run_p0
from lungnod.evaluation import evaluate
from lungnod.synthetic import make_phantom


@pytest.fixture(scope="module")
def run():
    scan, truth = make_phantom()
    return scan, truth, run_p0(scan, keep_intermediate=True)


def _nearest(table, xyz):
    d = np.linalg.norm(table[["coordX", "coordY", "coordZ"]].to_numpy() - xyz, axis=1)
    return table.iloc[int(np.argmin(d))], float(d.min())


def test_lung_mask_excludes_ribs(run):
    scan, _, res = run
    assert res.lung_mask.any()
    assert not (res.lung_mask & (scan.hu > 400)).any()


def test_nodules_found_and_accepted(run):
    _, truth, res = run
    for _, n in truth[truth.kind == "nodule"].iterrows():
        row, d = _nearest(res.table, n[["coordX", "coordY", "coordZ"]].to_numpy(float))
        assert d < n.diameter_mm / 2, f"nodule at {n.coordX, n.coordY, n.coordZ} not located"
        assert row.probability > 0.5, row.explanation


def test_vessels_rejected(run):
    _, truth, res = run
    t = res.table
    for _, v in truth[truth.kind.str.startswith("vessel")].iterrows():
        near = t[np.hypot(t.coordX - v.coordX, t.coordY - v.coordY) < 15]
        near = near[near.n_slices >= 5]  # the long vessel tracks
        assert len(near), f"{v.kind} not tracked"
        assert near.probability.max() < 0.1, near[["probability", "explanation"]]


def test_cpm_on_phantom(run):
    _, truth, res = run
    nod = truth[truth.kind == "nodule"]
    r = evaluate(res.submission(), nod, None, [res.seriesuid])
    assert r["max_sensitivity"] == 1.0
    assert r["cpm"] > 0.9
