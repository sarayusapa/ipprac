import numpy as np
import pandas as pd

from lungnod.candidates import Blob
from lungnod.classify import classify_table
from lungnod.config import P0Config, TrackingConfig
from lungnod.evaluation import evaluate
from lungnod.features import sphere_profile, sphere_profile_fit
from lungnod.io import CTScan
from lungnod.tracking import track_blobs


def test_world_to_voxel_matches_report_example():
    # report Table 3 / Section 7.3
    scan = CTScan("x", np.zeros((1, 1, 1)), np.array([2.5, 0.703125, 0.703125]),
                  np.array([-180.0, -157.5, -349.5]), np.eye(3))
    zyx = scan.world_to_voxel([68.06, 43.70, -266.02])[0]
    np.testing.assert_allclose(zyx, [33.39, 286.15, 352.80], atol=0.05)
    np.testing.assert_allclose(scan.voxel_to_world(zyx)[0], [68.06, 43.70, -266.02], atol=1e-6)


def test_sphere_fit_separates_sphere_from_tube():
    dz = 1.25
    z = np.arange(-3, 4) * dz
    sphere = sphere_profile(z, 4.0, 0.0)
    tube = np.full(12, 1.6) + np.random.default_rng(0).normal(0, 0.05, 12)
    fs = sphere_profile_fit(z, sphere, dz, 1.3)
    ft = sphere_profile_fit(np.arange(12) * dz, tube, dz, 1.3)
    assert fs["sphere_gain"] > 0.1
    assert ft["sphere_gain"] < -0.3


def _blob(z, y, x, r):
    return Blob(z, y, x, r, np.pi * r * r, 1, 0, 0, 0, 10, 1, r, False)


def test_tracking_links_with_gap_and_keeps_objects_apart():
    spacing = np.array([1.0, 1.0, 1.0])
    blobs = {
        0: [_blob(0, 10, 10, 2), _blob(0, 10, 30, 2)],
        1: [_blob(1, 10.5, 10, 2), _blob(1, 10, 31, 2)],
        # slice 2: first object missed
        2: [_blob(2, 10, 32, 2)],
        3: [_blob(3, 10, 10, 2), _blob(3, 10, 33, 2)],
    }
    tracks = track_blobs(blobs, spacing, TrackingConfig())
    assert len(tracks) == 2
    assert sorted(len(t.blobs) for t in tracks) == [3, 4]
    assert all(t.n_slices == 4 for t in tracks)


def test_tracking_counts_branching():
    spacing = np.array([1.0, 1.0, 1.0])
    blobs = {0: [_blob(0, 10, 10, 2)], 1: [_blob(1, 10, 9, 2), _blob(1, 10, 11.5, 2)]}
    tracks = track_blobs(blobs, spacing, TrackingConfig())
    assert tracks[0].branches == 1


def test_rules_reject_vessel_accept_nodule():
    cfg = P0Config().classifier
    df = pd.DataFrame([
        dict(aspect=1.1, drift_ratio=0.1, max_log_response=0.45, sphere_gain=0.15, max_diameter_mm=8,
             n_slices=7, branches=0, lifetime_mm=8.75),
        dict(aspect=6.0, drift_ratio=8.0, max_log_response=0.45, sphere_gain=-0.6, max_diameter_mm=3.2,
             n_slices=30, branches=2, lifetime_mm=37.5),
    ])
    out = classify_table(df, cfg)
    assert out.probability[0] > 0.5 and out.decision[0] == "nodule"
    assert out.probability[1] < 0.01
    assert "aspect" in out.explanation[1] and "shifts" in out.explanation[1]


def test_froc_cpm_toy():
    ann = pd.DataFrame({"seriesuid": ["a", "b"], "coordX": [0, 0], "coordY": [0, 0],
                        "coordZ": [0, 0], "diameter_mm": [10, 10]})
    exc = pd.DataFrame({"seriesuid": ["a"], "coordX": [50], "coordY": [0], "coordZ": [0],
                        "diameter_mm": [-1]})
    sub = pd.DataFrame({
        "seriesuid": ["a", "a", "a", "b"],
        "coordX": [1, 50, 100, 30], "coordY": [0] * 4, "coordZ": [0] * 4,
        "probability": [0.9, 0.95, 0.5, 0.8],
    })
    r = evaluate(sub, ann, exc, ["a", "b"])
    # nodule a found at 0.9; nodule b missed; FPs: 0.5 (a), 0.8 (b); 0.95 ignored (excluded)
    assert r["n_nodules"] == 2
    assert r["max_sensitivity"] == 0.5
    assert r["sens_at"][8.0] == 0.5
    assert r["sens_at"][0.125] <= 0.5


def test_hessian_dot_vs_line():
    """Disc -> ratio ~1 (in-plane aspect ~1, s_dot high); long bar -> ratio ~0 (s_dot ~0)."""
    from lungnod.candidates import hess_ratio, hessian_stacks
    from lungnod.classify import inplane_aspect

    n = 121
    yy, xx = np.mgrid[:n, :n] - n // 2
    disc = ((xx**2 + yy**2) <= 8**2).astype(np.float32)
    bar = ((np.abs(yy) <= 4) & (np.abs(xx) <= 50)).astype(np.float32)
    r = np.array([8.0])
    c = n // 2
    for img, lo, hi in [(disc, 0.95, 1.0), (bar, 0.0, 0.1)]:
        _, hyy, hxx, hxy = hessian_stacks(img, (1.0, 1.0), r if img is disc else np.array([4.0]))
        q = hess_ratio(hyy[0, c, c], hxx[0, c, c], hxy[0, c, c])
        assert lo <= q <= hi
    assert inplane_aspect(1.0) == 1.0
    assert abs(inplane_aspect(3 / (2.5**2 + 2)) - 2.5) < 1e-9
