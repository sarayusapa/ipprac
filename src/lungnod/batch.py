"""Run P0 over many scans (optionally in parallel processes)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import P0Config
from .io import load_scan
from .pipeline import run_p0, score_candidate_list


def process_scan(path: str, cfg_dict: dict, candidates: pd.DataFrame | None = None) -> dict:
    cfg = P0Config.from_dict(cfg_dict)
    scan = load_scan(path)
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
            results.append(process_scan(str(p), cfg_dict, cands_for(p)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_scan, str(p), cfg_dict, cands_for(p)) for p in paths]
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
