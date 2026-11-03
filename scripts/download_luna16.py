#!/usr/bin/env python
"""Download LUNA16 from Zenodo into data/luna16/, verify MD5, extract, delete zips.

Resumable: rerun after an interruption; finished files are skipped.

  python scripts/download_luna16.py                  # everything
  python scripts/download_luna16.py --subsets 0      # CSVs + subset0 only
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

RECORDS = ["3723295", "4121926"]  # LUNA16 part 1/2 and part 2/2
EXTRAS = ["annotations.csv", "candidates_V2.zip", "evaluationScript.zip",
          "seg-lungs-LUNA16.zip", "sampleSubmission.csv"]


def list_files() -> dict:
    files = {}
    for rec in RECORDS:
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}") as r:
            for f in json.load(r)["files"]:
                files[f["key"]] = (f["links"]["self"], f["checksum"].split(":")[1], f["size"])
    return files


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, checksum: str, size: int, out: Path) -> Path:
    dst = out / "_downloads" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == size and md5(dst) == checksum:
        return dst
    print(f"downloading {name} ({size / 1e9:.2f} GB)", flush=True)
    # Zenodo drops long connections; resume (-C -) until the full size is on disk
    for attempt in range(1, 201):
        if dst.exists() and dst.stat().st_size >= size:
            break
        # No curl --retry: curl's internal retry restarts from byte 0 and truncates
        # the file. Each fresh curl call resumes from the size on disk instead.
        # --speed-limit/--speed-time abort stalled connections so the loop resumes.
        rc = subprocess.run(["curl", "-L", "--fail", "-C", "-", "-sS",
                             "--speed-limit", "1000", "--speed-time", "60",
                             "-o", str(dst), url]).returncode
        if rc == 0:
            break
        have = dst.stat().st_size if dst.exists() else 0
        print(f"  {name}: curl exit {rc} at {have / 1e9:.2f}/{size / 1e9:.2f} GB, "
              f"resuming (attempt {attempt})", flush=True)
        time.sleep(15)
    else:
        raise RuntimeError(f"giving up on {name}")
    got = md5(dst)
    if got != checksum:
        dst.unlink()
        raise RuntimeError(f"MD5 mismatch for {name}: {got} != {checksum}")
    return dst


def done_marker(out: Path, name: str) -> Path:
    return out / "_downloads" / f"{name}.done"


def install(name: str, path: Path, out: Path) -> None:
    if not name.endswith(".zip"):
        shutil.move(str(path), out / name)
        return
    stage = out / "_downloads" / name.removesuffix(".zip")
    with zipfile.ZipFile(path) as z:  # zipfile sanitises absolute / '..' paths
        z.extractall(stage)
    if name.startswith("subset"):
        sub = name.removesuffix(".zip")
        src = next((p for p in stage.rglob(sub) if p.is_dir()), stage)
        shutil.move(str(src), out / sub)
    elif name == "candidates_V2.zip":
        shutil.move(str(next(stage.rglob("candidates_V2.csv"))), out / "candidates_V2.csv")
    elif name == "evaluationScript.zip":
        ex = next(stage.rglob("annotations_excluded.csv"), None)
        if ex:
            shutil.copy(ex, out / "annotations_excluded.csv")
        shutil.move(str(stage), out / "evaluationScript")
    elif name == "seg-lungs-LUNA16.zip":
        shutil.move(str(stage), out / "seg-lungs-LUNA16")
    if stage.exists():
        shutil.rmtree(stage)
    path.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/luna16")
    ap.add_argument("--subsets", type=int, nargs="*", default=list(range(10)))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = list_files()
    wanted = EXTRAS + [f"subset{i}.zip" for i in args.subsets]
    for name in wanted:
        if done_marker(out, name).exists():
            print(f"skip {name} (done)")
            continue
        url, checksum, size = files[name]
        path = fetch(name, url, checksum, size, out)
        print(f"extracting {name}", flush=True)
        install(name, path, out)
        done_marker(out, name).touch()
        print(f"done {name}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
