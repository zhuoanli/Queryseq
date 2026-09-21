"""How large an improvement can exhaustive subset search invent from noise?

We measured a 0.0103 AUROC gap between selecting-and-scoring on the same
predictions and cross-fitted selection -- about the same size as the real
cross-fitted effect.  That is worth more than a limitations paragraph, because
the same failure mode is available to any paper that searches a space of
subsets, prompts, checkpoints or routing policies and reports the winner.

This maps the optimism

    Optimism = U_naive(best subset) - U_crossfit(same selection rule)

as a function of the two things that control it: how many candidates were
searched, and how much evidence each candidate was judged on.  Candidate count
is varied by restricting the lattice to a random subfamily; evidence is varied
by subsampling patients.  A null lattice, in which labels are permuted so that
no subset is genuinely better than another, gives the search-only reference.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, finding_filter,  # noqa: E402
                    macro, patient_folds, safe_auroc)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def _auc(y, s):
    """AUROC over the entries where the score is defined.

    Lattice columns for findings below the prevalence floor are all-NaN, and a
    fold can fail to produce a score for a rare finding, so scoring has to skip
    undefined entries rather than propagate them.
    """
    m = np.isfinite(s)
    if m.sum() < 10:
        return np.nan
    return safe_auroc(y[m], s[m])


def evaluate(S, Y, keep, rows, cand, folds):
    """Naive (select==score) and cross-fitted macro for one configuration."""
    naive, cf = [], []
    for j in keep:
        a = [_auc(Y[rows, j], S[si, rows, j]) for si in cand]
        if all(np.isnan(x) for x in a):
            continue
        naive.append(np.nanmax(a))
        pooled = np.full(len(rows), np.nan, np.float32)
        for sel, ev in folds:
            b = [_auc(Y[rows[sel], j], S[si, rows[sel], j]) for si in cand]
            if all(np.isnan(x) for x in b):
                continue
            pooled[ev] = S[cand[int(np.nanargmax(b))], rows[ev], j]
        v = _auc(Y[rows, j], pooled)
        if not np.isnan(v):
            cf.append(v)
    return macro(naive), macro(cf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--ncands", type=str, default="2,4,8,15")
    ap.add_argument("--fracs", type=str, default="0.1,0.25,0.5,1.0")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    co = Cohort("train", complete_case=True)
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    subs = all_subsets()
    rng = np.random.default_rng(0)
    rows = []

    for seed in range(args.seeds):
        S = np.load(f"{CACHE}/oof_seed{seed}{args.tag}.npy").astype(np.float32)
        # null lattice: permuting patients breaks any real subset difference,
        # so whatever optimism survives here is pure selection noise
        Snull = S.copy()
        perm = rng.permutation(len(co))
        Snull = Snull[:, perm, :]
        for nc in [int(x) for x in args.ncands.split(",")]:
            for fr in [float(x) for x in args.fracs.split(",")]:
                for rep in range(args.reps):
                    r2 = np.random.default_rng(1000 * seed + 17 * nc + rep)
                    pats = np.unique(co.patients)
                    take = set(r2.choice(pats, max(2, int(len(pats) * fr)),
                                         replace=False))
                    idx = np.array([i for i, p in enumerate(co.patients)
                                    if p in take])
                    if len(idx) < 50:
                        continue
                    cand = list(r2.choice(len(subs), nc, replace=False)) \
                        if nc < len(subs) else list(range(len(subs)))
                    fl = patient_folds(co.patients[idx], k=args.folds,
                                       seed=rep)
                    # NB: not "null" -- pandas parses that string as NaN on read-back
                    for lab, T in (("real", S), ("permuted", Snull)):
                        nv, cf = evaluate(T, co.Y, keep, idx, cand, fl)
                        rows.append({
                            "seed": seed, "lattice": lab, "n_candidates": nc,
                            "frac_patients": fr, "n_studies": len(idx),
                            "rep": rep, "naive": nv, "crossfit": cf,
                            "optimism": nv - cf})
        print(f"  seed {seed} done ({len(rows)} rows)", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/selection_optimism{args.tag}.csv", index=False)
    print("\n=== optimism = naive(best-of-K) - cross-fitted, by K and n ===")
    for lab in ("real", "permuted"):
        print(f"\n-- {lab} lattice --")
        print(d[d.lattice == lab].pivot_table(
            index="n_candidates", columns="n_studies", values="optimism")
            .round(4).to_string())
    print("\noptimism by number of candidates (all sizes pooled):")
    print(d.groupby(["lattice", "n_candidates"]).optimism.agg(
        ["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
