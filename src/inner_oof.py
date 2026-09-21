"""Inner-CV prediction cache for strictly nested subset selection.

A reviewer inspecting the code found that `nested_policy` selected each outer
fold's subset using the single-layer OOF tensor: a selection row in fold
g != f was predicted by a head trained on every fold except g, which includes
the outer evaluation fold f.  Outer-fold labels therefore reached the models
underlying the selection criterion.  The scoring side was always proper; the
selection side was not.

The fix is the textbook one.  For each outer fold f, this script runs a full
inner cross-validation strictly within train(f), for every subset and finding,
and caches those inner-OOF predictions.  Selection for fold f may then read
only this cache; nothing fitted with fold f's labels can influence which
subset is chosen for fold f.

Writes data_local/inneroof_seed<seed><tag>.npz with one array per outer fold:
    inner_f{f} : float16 [n_subsets, len(train_f), n_findings]
    rows_f{f}  : int32 indices of train(f) in cohort order (the row axis)

Cost is (outer folds) x (inner folds) refits of the lattice, all embarrassingly
parallel over (subset, inner fold) cells.
"""
import argparse
import os
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, finding_filter,  # noqa: E402
                    patient_folds)
from subset_lattice import run_cell  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--task", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")),
                    help="shard index: task -> (seed=task//folds, "
                         "fold=task%%folds); one block per node makes the "
                         "whole cache a ten-minute array job")
    args = ap.parse_args()

    co = Cohort("train", complete_case=True)
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    subs = all_subsets()
    nF = len(co.findings)
    print(f"n={len(co)} subsets={len(subs)} findings(keep)={len(keep)} "
          f"outer={args.folds} inner={args.inner_folds}", flush=True)

    pairs = [(sd, f) for sd in range(args.seeds) for f in range(args.folds)]
    if args.task >= 0:
        pairs = [pairs[args.task]]
    for seed, fwant in pairs:
        out = f"{CACHE}/inneroof_s{seed}_f{fwant}{args.tag}.npz"
        if os.path.exists(out):
            print(f"seed {seed} fold {fwant}: exists, skipping", flush=True)
            continue
        folds = patient_folds(co.patients, k=args.folds, seed=seed)
        arrays = {}
        for f, (tr, _ev) in enumerate(folds):
            if f != fwant:
                continue
            tr = np.asarray(tr)
            inner = patient_folds(co.patients[tr], k=args.inner_folds,
                                  seed=1000 * seed + f)
            S = np.full((len(subs), len(tr), nF), np.nan, np.float32)
            cells = [(si, fi) for si in range(len(subs))
                     for fi in range(len(inner))]

            def cell(si, fi):
                itr, ite = inner[fi]          # positions within tr
                sc = run_cell(co.P, co.M, co.Y, keep, subs[si],
                              tr[itr], tr[ite], args.C)
                return si, ite, sc

            res = Parallel(n_jobs=args.jobs)(
                delayed(cell)(si, fi) for si, fi in cells)
            for si, ite, sc in res:
                S[si, ite, :] = sc
            arrays["inner"] = S.astype(np.float16)
            arrays["rows"] = np.asarray(tr, np.int32)
            print(f"  seed {seed} outer fold {f}: inner cache "
                  f"[{len(subs)}, {len(tr)}, {nF}] done", flush=True)
        np.savez_compressed(out, **arrays)
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
