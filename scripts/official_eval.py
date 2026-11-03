#!/usr/bin/env python
"""Score a run with the OFFICIAL LUNA16 evaluation script (noduleCADEvaluationLUNA16.py).

The official script is Python 2. This wrapper copies it to <run>/official_eval/_script,
converts it to Python 3 with lib2to3 (no logic changes), and runs it on submission.csv
and submission_fpr.csv. Note: the official script keeps at most 100 CAD marks per scan.

  python scripts/official_eval.py --run outputs/p0_all
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data/luna16/evaluationScript/evaluationScript"


def prepare(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("exampleFiles", "*.pyc"))
    subprocess.run([sys.executable, "-m", "lib2to3", "-w", "-n", str(dst)],
                   check=True, capture_output=True)
    # Python 2 opened CSVs in binary mode; Python 3's csv module needs text mode (I/O only)
    csvt = dst / "tools" / "csvTools.py"
    t = csvt.read_text()
    t = t.replace('open(filename, "wb")', 'open(filename, "w", newline="")')
    t = t.replace('open(filename, "rb")', 'open(filename, "r", newline="")')
    csvt.write_text(t)
    main = dst / "noduleCADEvaluationLUNA16.py"
    s = main.read_text()
    s = s.replace("import matplotlib.pyplot as plt", "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt")
    # matplotlib renamed two plotting arguments (affects the figure only)
    s = s.replace("basex=2", "base=2").replace("plt.grid(b=True", "plt.grid(visible=True")
    main.write_text(s)
    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--script-dir", default=str(DEFAULT_SRC))
    args = ap.parse_args()

    run = Path(args.run).resolve()
    out = run / "official_eval"
    out.mkdir(exist_ok=True)
    script = prepare(Path(args.script_dir), out / "_script")
    ann = script.parent / "annotations"

    for name in ["submission", "submission_fpr"]:
        sub = run / f"{name}.csv"
        if not sub.exists():
            continue
        res_dir = out / name
        res_dir.mkdir(exist_ok=True)
        print(f"== {name} (official script) ==", flush=True)
        p = subprocess.run(
            [sys.executable, str(script), str(ann / "annotations.csv"),
             str(ann / "annotations_excluded.csv"), str(run / "seriesuids.txt"),
             str(sub), str(res_dir) + "/"],
            cwd=script.parent, capture_output=True, text=True,
        )
        (res_dir / "log.txt").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            print(p.stderr[-2000:])
            raise SystemExit(f"official script failed on {name}")
        import numpy as np
        rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        froc = np.loadtxt(res_dir / f"froc_{name}.txt", delimiter=",")
        sens = np.interp(rates, froc[:, 0], froc[:, 1])
        print(f"CPM = {sens.mean():.3f}   " + "  ".join(f"@{r}:{v:.3f}" for r, v in zip(rates, sens)))
        analysis = res_dir / "CADAnalysis.txt"
        if analysis.exists():
            for line in analysis.read_text().splitlines()[:12]:
                print("  " + line)


if __name__ == "__main__":
    main()
