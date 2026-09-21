"""A permutation *distribution* for selection optimism, not a single draw.

selection_optimism.py permutes the lattice once per seed, which supports at most
"one permutation produced 0.0555".  A single draw of a noisy quantity is not
evidence about its typical size, and an extreme draw is exactly what a sceptical
reader would suspect.  This runs many permutations and reports the distribution.

Design choices that matter:

  * Permutation is at the *patient* level, so a patient's studies move together
    and the within-patient correlation that the bootstrap respects elsewhere is
    not silently destroyed here.
  * Labels are untouched, so every finding keeps its exact prevalence; only the
    correspondence between a study's scores and its labels is broken.
  * Fold structure is recomputed from the unpermuted patient vector, so the
    cross-fitting is identical to the real analysis.
  * The same permutation is applied to every subset, so subsets remain
    comparable to each other within a replicate -- otherwise the null would
    destroy the very selection problem it is meant to characterise.

Under this null no subset is better than any other, so the entire naive
best-of-K advantage is selection noise.  Its distribution is a search-only
reference, not a hard upper bound: it says how much apparent gain the search
manufactures when there is nothing to find, at this sample size and candidate
count.  Observed optimism can legitimately fall *below* it, because genuine
utility differences stabilise which subset wins across folds, so naive and
cross-fitted selection agree more often than they do under pure noise.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, finding_filter,  # noqa: E402
                    macro, patient_folds, safe_auroc)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def _auc(y, s):
    m = np.isfinite(s)
    return safe_auroc(y[m], s[m]) if m.sum() >= 10 else np.nan


def patient_permutation(patients, rng):
    """Block-preserving label permutation at patient level.

    Patients are permuted only within groups of equal study count, so every
    patient's rows receive the complete row-block of exactly one other
    patient (of the same size) and no block is ever sheared.  An earlier
    version concatenated variable-length blocks in two different orders and
    mapped position-wise, which mixed labels across patients whenever study
    counts differed -- a reviewer supplied the counterexample.  In this
    cohort 94.9% of patients have one study, so the two nulls are close in
    law, but only this one has the property the text claims.
    """
    uniq, first = np.unique(patients, return_index=True)
    idx_by = {g: np.where(patients == g)[0] for g in uniq}
    sizes = np.array([len(idx_by[g]) for g in uniq])
    perm = np.arange(len(patients))
    for sz in np.unique(sizes):
        grp = uniq[sizes == sz]
        order = rng.permutation(len(grp))
        for a, b in zip(grp, grp[order]):
            perm[idx_by[a]] = idx_by[b]
    return perm


def optimism(S, Y, keep, cand, folds, inner):
    """Naive best-of-K minus strictly nested cross-fitted, macro over findings.

    Selection for outer fold f reads the inner-CV tensor computed within
    train(f); the labels Y may be permuted by the caller, in which case the
    same permuted labels drive both arms.  The prediction tensors themselves
    always come from models fitted on the true labels, so a permutation
    changes only the score--label correspondence, never the fits.
    """
    naive, cf = [], []
    for j in keep:
        a = [_auc(Y[:, j], S[si, :, j]) for si in cand]
        if all(np.isnan(x) for x in a):
            continue
        naive.append(np.nanmax(a))
        pooled = np.full(len(Y), np.nan, np.float32)
        for f, (sel, ev) in enumerate(folds):
            rows, T = inner[f]
            b = [_auc(Y[rows, j], T[si, :, j]) for si in cand]
            if all(np.isnan(x) for x in b):
                continue
            pooled[ev] = S[cand[int(np.nanargmax(b))], ev, j]
        v = _auc(Y[:, j], pooled)
        if not np.isnan(v):
            cf.append(v)
    return macro(naive) - macro(cf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--ncands", type=str, default="2,4,8,15")
    ap.add_argument("--seed-file", type=int, default=0,
                    help="which oof_seed*.npy to permute")
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    co = Cohort("train", complete_case=True)
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    subs = all_subsets()
    S = np.load(f"{CACHE}/oof_seed{args.seed_file}{args.tag}.npy").astype(
        np.float32)
    folds = patient_folds(co.patients, k=args.folds, seed=0)
    from shapley_synergy import load_inner
    inner = load_inner(args.seed_file, args.tag, args.folds)
    ncands = [int(x) for x in args.ncands.split(",")]
    ncands = [k for k in ncands if k <= len(subs)]
    print(f"n={len(co)} findings={len(keep)} subsets={len(subs)} "
          f"perms={args.perms} K={ncands}", flush=True)

    obs = {}
    for K in ncands:
        r = np.random.default_rng(7)
        cand = (list(range(len(subs))) if K >= len(subs)
                else list(r.choice(len(subs), K, replace=False)))
        obs[K] = optimism(S, co.Y, keep, cand, folds, inner)
        print(f"  observed K={K:3d}: optimism {obs[K]:+.4f}", flush=True)

    def one(p, K):
        r = np.random.default_rng(10_000 + 97 * K + p)
        perm = patient_permutation(co.patients, r)
        cand = (list(range(len(subs))) if K >= len(subs)
                else list(np.random.default_rng(7).choice(
                    len(subs), K, replace=False)))
        # permute the LABELS rather than the score rows: distributionally
        # identical (whole patients move either way), and it lets the nested
        # selection arm read the inner tensors, whose row sets are fixed
        return K, p, optimism(S, co.Y[perm], keep, cand, folds, inner)

    jobs = [(p, K) for K in ncands for p in range(args.perms)]
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one)(p, K) for p, K in jobs)
    d = pd.DataFrame(out, columns=["n_candidates", "perm", "optimism"])
    d["observed"] = d.n_candidates.map(obs)
    d.to_csv(f"{RES}/null_optimism_draws{args.tag}.csv", index=False)

    rows = []
    for K in ncands:
        v = d[d.n_candidates == K].optimism.dropna().to_numpy()
        if not len(v):
            continue
        rows.append({
            "n_candidates": K, "n_perms": len(v), "observed": obs[K],
            "null_mean": float(v.mean()), "null_median": float(np.median(v)),
            "null_p05": float(np.percentile(v, 5)),
            "null_p95": float(np.percentile(v, 95)),
            "null_max": float(v.max()),
            # how often the null alone reaches what the real lattice showed
            "frac_null_ge_observed": float((v >= obs[K]).mean()),
        })
    s = pd.DataFrame(rows)
    s.to_csv(f"{RES}/null_optimism{args.tag}.csv", index=False)
    print("\n=== optimism under a patient-level label-preserving null ===")
    print(s.round(4).to_string(index=False))
    print("\nReport as: exhaustive subset search produced a median apparent "
          "improvement of\n[null_median] under the null, 95th percentile "
          "[null_p95] -- not 'one permutation gave X'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
