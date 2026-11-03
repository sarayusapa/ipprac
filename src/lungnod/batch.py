"""Run P0 over many scans (optionally in parallel processes)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import zlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import P0Config
from .io import load_scan
from .pipeline import run_p0, score_candidate_list


def add_noise(scan, sigma_hu: float, seed: int) -> None:
    """Simulate a lower-dose scan: add zero-mean Gaussian noise (HU) in the image domain.

    CT quantum noise std scales as 1/sqrt(dose) and independent noise variances add, so
    adding sigma_hu = s0 * sqrt(1/f - 1) to a scan with noise s0 approximates dose x f
    (e.g. s0 = 25 HU: 25 HU ~ half dose, 43 HU ~ quarter dose). Image-domain Gaussian
    noise is a simplification of real low-dose noise (which is not white).
    A seed derived from the series makes runs reproducible.
    """
    rng = np.random.default_rng([seed, zlib.crc32(scan.seriesuid.encode())])
    scan.hu += rng.normal(0.0, sigma_hu, size=scan.hu.shape).astype(np.float32)


def process_scan(path: str, cfg_dict: dict, candidates: pd.DataFrame | None = None,
                 noise_hu: float = 0.0, seed: int = 0) -> dict:
    cfg = P0Config.from_dict(cfg_dict)
    scan = load_scan(path)
    if noise_hu > 0:
        add_noise(scan, noise_hu, seed)
    res = run_p0(scan, cfg)
    out = {"seriesuid": scan.seriesuid, "table": res.table, "timings": res.timings}
    if candidates is not None:
        out["fpr"] = score_candidate_list(scan, res, candidates, cfg)
    return out


def run_batch(
    paths: list[Path],
    cfg: P0Config,
    candidates: pd.DataFrame | None = None,
    workers: int = 1,
    noise_hu: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Returns (all track tables, FPR-track scores or None, per-scan timings)."""
    cfg_dict = cfg.to_dict()

    def cands_for(p: Path):
        if candidates is None:
            return None
        return candidates[candidates["seriesuid"] == p.stem]

    results = []
    if workers <= 1:
        for p in tqdm(paths, desc="P0"):
            results.append(process_scan(str(p), cfg_dict, cands_for(p), noise_hu, seed))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_scan, str(p), cfg_dict, cands_for(p), noise_hu, seed) for p in paths]
            for f in tqdm(as_completed(futs), total=len(futs), desc="P0"):
                results.append(f.result())

    tables = pd.concat([r["table"] for r in results], ignore_index=True)
    fpr = (
        pd.concat([r["fpr"] for r in results], ignore_index=True)
        if candidates is not None
        else None
    )
    timings = pd.DataFrame([{"seriesuid": r["seriesuid"], **r["timings"]} for r in results])
    return tables, fpr, timings
