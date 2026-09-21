"""Extract the val/test studies still sitting in downloaded-but-unzipped batches.

The complete-case evaluation splits are the binding constraint on this paper:
test carries only 9 findings with >=20 positives and val only 7, which is too
thin to support per-finding claims.  Ten batches were downloaded but never
unzipped, and roughly 1.6k of those studies belong to the official val/test
splits -- enough to lift the evaluation cohort by about half.

Only val/test studies are extracted.  Training has 12.4k complete-case studies
already, so spending disk and GPU on more train data buys much less than
spending it where the confidence intervals are.

Zips are kept (`--keep-zips` behaviour by default): scratch has ample headroom
and the corpus release's frozen state should not be disturbed by deleting its inputs.
"""
import argparse
import glob
import os
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PB = Path(os.environ.get("QS_FEATROOT",
          Path(__file__).resolve().parents[2] / "corpus_release"))
ZIP_ROOT = PB / "data" / "mrrate_zip" / "mri"
OUT_ROOT = PB / "data" / "mrrate" / "mri"


def targets():
    sp = pd.read_csv(PB / "data" / "mrrate" / "splits.csv")
    want = set(sp[sp.split.isin(["val", "test"])].study_uid.astype(str))
    have = set()
    for s in ("val", "test"):
        have |= {os.path.basename(f)[:-4]
                 for f in glob.glob(str(PB / "data" / "features" / s / "*.npz"))}
    out = []
    for d in sorted(ZIP_ROOT.glob("batch*")):
        for z in d.glob("*.zip"):
            u = z.stem
            if u in want and u not in have:
                out.append((str(z), str(OUT_ROOT / d.name)))
    return out


def extract_one(a):
    zip_path, out_dir = a
    study = Path(zip_path).stem
    dest = Path(out_dir) / study
    try:
        if dest.is_dir() and any(dest.rglob("*.nii.gz")):
            return study, "skipped"
        tmp = Path(out_dir) / (study + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # a zip may wrap its contents in a top-level dir named for the study
        inner = tmp / study
        src = inner if inner.is_dir() else tmp
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(src), str(dest))
        shutil.rmtree(tmp, ignore_errors=True)
        if not any(dest.rglob("*.nii.gz")):
            return study, "empty"
        return study, "ok"
    except Exception as e:                                   # noqa: BLE001
        return study, f"fail:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tg = targets()
    print(f"{len(tg)} val/test studies to extract", flush=True)
    if args.dry_run or not tg:
        return 0
    counts = {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(extract_one, a) for a in tg]
        for i, f in enumerate(as_completed(futs)):
            _, st = f.result()
            counts[st] = counts.get(st, 0) + 1
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tg)} {counts}", flush=True)
    print("done", counts, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
