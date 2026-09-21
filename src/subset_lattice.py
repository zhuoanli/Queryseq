"""Stage 1: the exhaustive input-subset utility lattice.

Four inputs give 2^4 - 1 = 15 non-empty subsets, so the lattice is enumerated in
full rather than sampled.  For every (subset, finding) we fit an identical
linear head and record out-of-fold AUROC/AUPRC.

Three decisions, all fixed before any number was looked at:

1. **Complete-case cohort.**  Sequence availability in MR-RATE is
   indication-driven, so scoring subset S on "studies that happen to carry S"
   compares different patient populations.  Restricting to studies that carry
   all four and masking makes withholding an input a real intervention.  The
   availability-mask-only control below measures how large that confound would
   have been.

2. **Out-of-fold on train, not on test.**  On the complete-case cohort the
   official test split holds only 9 findings with >=20 positives and val only 7
   -- far too thin for a 37-finding atlas.  Five-fold patient-level CV over the
   12.4k training studies makes every finding evaluable and leaves test
   untouched for the method comparison.

3. **A fixed readout, declared in advance**: L2 logistic, C=0.1,
   class_weight='balanced', standardised per fold.  Tuning C per subset would
   hand each subset its own degree of freedom and quietly favour whichever one
   the tuner had most room to help; an identical head is the more conservative
   comparison.  --c-sweep re-runs the lattice at other C to confirm the subset
   *ranking* is not an artifact of that choice.

Writes results/subset_lattice.csv and the out-of-fold prediction tensors that
the Shapley, synergy and paired-bootstrap analyses consume.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, finding_filter,  # noqa: E402
                    macro, patient_folds, safe_auprc, safe_auroc, subset_name)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def fit_fold(Xtr, ytr, Xte, C):
    """One binary linear probe.  Returns held-out scores."""
    if ytr.sum() < 3 or ytr.sum() == len(ytr):
        return np.full(len(Xte), np.nan)
    m = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    m.fit(Xtr, ytr)
    return m.decision_function(Xte)


def pool(P, M, mask):
    """Mean over the inputs that the subset selects and the study carries."""
    w = M * np.asarray(mask, np.float32)[None, :]
    d = w.sum(1, keepdims=True)
    return (P * w[:, :, None]).sum(1) / np.maximum(d, 1e-6)


def run_cell(P, M, Y, keep, mask, tr, te, C):
    """One (subset, fold) cell.

    Parallelising at this granularity rather than per subset is what keeps the
    node busy: there are only 15 subsets but 15*folds cells, and a 64-core node
    given 15 tasks leaves three quarters of itself idle.  Scaling is fitted on
    training rows only, so no held-out row influences its own normalisation.
    """
    X = pool(P, M, mask)
    sc = StandardScaler().fit(X[tr])
    A, B = sc.transform(X[tr]), sc.transform(X[te])
    out = np.full((len(te), Y.shape[1]), np.nan, np.float32)
    for j in keep:
        out[:, j] = fit_fold(A, Y[tr, j], B, C)
    return out


def oof_scores(X, Y, folds, keep, C):
    """Out-of-fold decision values, single-threaded (used by the mask control)."""
    S = np.full((len(X), Y.shape[1]), np.nan, np.float32)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]), sc.transform(X[te])
        for j in keep:
            S[te, j] = fit_fold(A, Y[tr, j], B, C)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--c-sweep", type=str, default="")
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    co = Cohort("train", complete_case=True, limit=args.limit)
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    subs = all_subsets()
    print(f"cohort n={len(co)}  findings kept={len(keep)}/{len(co.findings)}  "
          f"subsets={len(subs)}  jobs={args.jobs}", flush=True)

    Cs = [args.C] if not args.c_sweep else [float(x) for x in
                                            args.c_sweep.split(",")]
    rows = []
    for C in Cs:
        for seed in range(args.seeds):
            folds = patient_folds(co.patients, k=args.folds, seed=seed)
            cells = [(si, fi) for si in range(len(subs))
                     for fi in range(len(folds))]
            res = Parallel(n_jobs=args.jobs, verbose=5)(
                delayed(run_cell)(co.P, co.M, co.Y, keep, subs[si],
                                  folds[fi][0], folds[fi][1], C)
                for si, fi in cells)
            out = [np.full((len(co), co.Y.shape[1]), np.nan, np.float32)
                   for _ in subs]
            for (si, fi), r in zip(cells, res):
                out[si][folds[fi][1]] = r
            if C == args.C:
                # (subset, study, finding) out-of-fold scores.  Shapley,
                # synergy and every paired bootstrap read this rather than
                # refitting, which is what makes those analyses cheap.
                np.save(f"{CACHE}/oof_seed{seed}{args.tag}.npy",
                        np.stack(out).astype(np.float16))
            for m, S in zip(subs, out):
                for j in keep:
                    rows.append({
                        "C": C, "seed": seed,
                        "subset": subset_name(m),
                        "subset_bits": "".join(str(b) for b in m),
                        "k": int(sum(m)),
                        "finding": co.findings[j], "finding_idx": j,
                        "n_pos": int(co.Y[:, j].sum()), "n": len(co),
                        "auroc": safe_auroc(co.Y[:, j], S[:, j]),
                        "auprc": safe_auprc(co.Y[:, j], S[:, j]),
                    })
            print(f"C={C} seed={seed} done  {time.time() - t0:.0f}s",
                  flush=True)

    os.makedirs(RES, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/subset_lattice{args.tag}.csv", index=False)

    # ---- the missingness confound, measured rather than asserted ----------
    # Same protocol, but the only feature is which sequences the study carries.
    # Anything above chance here is patient selection leaking through
    # availability, and is the reason the lattice above is complete-case.
    full = Cohort("train", complete_case=False, limit=args.limit)
    fk = finding_filter(full.Y, full.findings, min_pos=args.min_pos)
    ffolds = patient_folds(full.patients, k=args.folds, seed=0)
    Sa = oof_scores(full.M.copy(), full.Y, ffolds, fk, args.C)
    arows = [{"finding": full.findings[j], "n_pos": int(full.Y[:, j].sum()),
              "n": len(full),
              "auroc": safe_auroc(full.Y[:, j], Sa[:, j]),
              "auprc": safe_auprc(full.Y[:, j], Sa[:, j])} for j in fk]
    pd.DataFrame(arows).to_csv(
        f"{RES}/availability_confound{args.tag}.csv", index=False)
    am = macro([r["auroc"] for r in arows])
    print(f"\navailability-mask-only macro AUROC = {am:.4f} "
          f"over {len(arows)} findings (chance = 0.5)", flush=True)

    d0 = df[df.C == args.C].groupby(["subset", "k"]).auroc.mean()
    print("\nmacro AUROC by subset:\n",
          d0.sort_values(ascending=False).to_string(), flush=True)
    print(f"\ntotal {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
