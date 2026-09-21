"""Turn the GoAT archive into a cohort this paper's analyses can read.

Two jobs: unpack the archive, and derive one binary target per tumour subregion
from the shipped segmentation.

The targets are the point of the exercise. BraTS labels tumour subregions whose
modality dependence is fixed by *definition* rather than by empirical
regularity -- enhancing tissue is defined by contrast uptake, FLAIR
hyperintensity by its appearance on FLAIR -- which is what makes the
pre-registered prediction in PLAN.md (2026-08-04T19:12Z) a real test rather than
a story told afterwards.

BraTS label convention as shipped:
    1  NETC  non-enhancing tumour core
    2  SNFH  surrounding non-enhancing FLAIR hyperintensity
    3  ET    enhancing tissue
    4  RC    resection cavity
Composite regions used in the challenge are derived from these, and we add the
two standard ones (TC = NETC+ET, WT = all) so the lattice has targets of
differing difficulty.

A subregion counts as present when it exceeds `--min-voxels`, which is fixed in
advance rather than tuned: a handful of stray voxels is annotation noise, not a
finding.
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOAT_ZIP = (ROOT / "data_local" / "brats_goat" /
            "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth.zip")
OUT = ROOT / "data_local" / "brats_goat_x"

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
LABELS = {"NETC": [1], "SNFH": [2], "ET": [3], "RC": [4],
          "TC": [1, 3], "WT": [1, 2, 3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default=str(GOAT_ZIP))
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--min-voxels", type=int, default=50,
                    help="fixed in advance; a few stray voxels is noise")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        print(f"extracting {args.zip} -> {out}", flush=True)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(out)
        print("extracted", flush=True)

    cases = sorted({p.parent for p in out.rglob("*-seg.nii.gz")})
    print(f"{len(cases)} cases with a segmentation", flush=True)

    import nibabel as nib
    rows = []
    for i, cdir in enumerate(cases, 1):
        cid = cdir.name
        seg = cdir / f"{cid}-seg.nii.gz"
        mods = {m: (cdir / f"{cid}-{m}.nii.gz").exists() for m in MODALITIES}
        if not seg.exists() or not all(mods.values()):
            continue
        arr = np.asanyarray(nib.load(str(seg)).dataobj)
        rec = {"case": cid, "path": str(cdir)}
        for name, vals in LABELS.items():
            n = int(np.isin(arr, vals).sum())
            rec[f"n_{name}"] = n
            rec[name] = int(n >= args.min_voxels)
        rows.append(rec)
        if i % 200 == 0:
            print(f"  {i}/{len(cases)}", flush=True)

    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(out.parent / "goat_labels.csv", index=False)
    print(f"\nwrote {len(d)} cases -> {out.parent / 'goat_labels.csv'}")
    print("\nprevalence (binary target per subregion):")
    for name in LABELS:
        p = d[name].sum()
        print(f"  {name:5s} {p:5d} / {len(d)}  ({100 * p / len(d):5.1f}%)"
              f"   {'evaluable' if p >= 20 and p <= len(d) - 20 else 'NOT evaluable'}")
    with open(out.parent / "goat_meta.json", "w") as fh:
        json.dump({"n_cases": len(d), "modalities": MODALITIES,
                   "labels": {k: v for k, v in LABELS.items()},
                   "min_voxels": args.min_voxels}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
