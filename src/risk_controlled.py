"""Risk-controlled minimal-evidence selection.

The lattice says a smaller input set is often as good as reading everything.
Acting on that requires answering a harder question than "which subset scored
highest here", because with 15 candidates the maximum is biased upward and the
winner on one sample is frequently not the winner on the next.  The question
worth answering is:

    what is the smallest input set that provably does not lose more than delta,
    at confidence 1-alpha, after accounting for having tried 15 of them?

Formally, for finding f with all-input set M,

    S*_f = argmin_{S} |S|   subject to   LCB_{1-alpha'}[U_f(S) - U_f(M)] >= -delta

with LCB a one-sided lower confidence bound from a patient-level bootstrap of
the *paired* difference, and alpha' = alpha / (#candidates) a Bonferroni
correction for having searched the lattice.  Ties in |S| are broken by the point
estimate.  Selection is cross-fitted: the bound is computed on selection folds
and the chosen policy is scored on held-out folds, so the reported performance
is not the performance that drove the choice.

This is the paper's method contribution.  It is deliberately not a learned
router: the experiments in run_methods.py show that patient-conditioned and
query-conditioned routers do not reliably beat a well-estimated finding-specific
policy, so the useful object is a correctly estimated policy, not a bigger one.

Also answers the question a deployment actually faces: if a request asks about
several findings at once, the inputs needed are the *union* of their policies,
and that union may well be everything.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from joblib import Parallel, delayed

from common import (CACHE, NSEQ, Cohort, all_subsets,  # noqa: E402
                    finding_filter, macro, patient_folds, safe_auroc,
                    subset_name)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def bootstrap_indices(groups, rng, reps):
    """Patient-level resamples, drawn once and reused everywhere.

    Rebuilding these inside the (finding, subset) loop dominated the runtime:
    each draw concatenates several thousand per-patient index arrays, and the
    lattice asks for the same draw 15 x 33 times per fold.  Sharing them also
    makes the bounds for different subsets paired on identical resamples, which
    is what we want anyway.
    """
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        out.append(np.concatenate([idx_by[g] for g in pick]))
    return out


def paired_lcb(y, s_sub, s_full, boots, alpha):
    """One-sided lower bound on AUROC(subset) - AUROC(all inputs).

    Paired on the same patient resample, so the shared patient draw cancels and
    the bound is about the subset rather than about the split.
    """
    out = []
    for i in boots:
        yi = y[i]
        p = yi.sum()
        if 0 < p < len(i):
            out.append(safe_auroc(yi, s_sub[i]) - safe_auroc(yi, s_full[i]))
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 100 * alpha)), float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--delta", type=float, default=0.01,
                    help="tolerated AUROC loss vs reading all inputs")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    co = Cohort("train", complete_case=True)
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    subs = all_subsets()
    full_i = len(subs) - 1
    assert sum(subs[full_i]) == NSEQ, "last subset should be all inputs"
    # Bonferroni over the candidates actually searched
    alpha_c = args.alpha / len(subs)
    rng = np.random.default_rng(0)

    # Selection reads ONLY the inner-CV tensors of each fold's training
    # partition (see inner_oof.py): the earlier version certified candidates
    # on the single-layer OOF predictions of the selection folds, whose
    # underlying heads had seen the evaluation fold's labels -- the same
    # leakage class the nested gate recompute removed.  Bounds are computed
    # per finding in parallel; each cell is independent given the shared
    # per-fold bootstrap indices.
    from shapley_synergy import load_inner
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    rows, pol, summ = [], [], []
    for seed in range(args.seeds):
        S = np.load(f"{CACHE}/oof_seed{seed}{args.tag}.npy").astype(np.float32)
        folds = patient_folds(co.patients, k=args.folds, seed=seed)
        inner = load_inner(seed, args.tag, args.folds)
        sel_scores = np.full((len(co), co.Y.shape[1]), np.nan, np.float32)
        min_scores = np.full((len(co), co.Y.shape[1]), np.nan, np.float32)
        for fi, (sel, ev) in enumerate(folds):
            irows, T = inner[fi]
            assert np.array_equal(np.sort(irows), np.sort(np.asarray(sel))), \
                "inner cache rows must equal the fold's training partition"
            boots = bootstrap_indices(co.patients[irows], rng, args.boot)

            def certify(j):
                out, ok = [], []
                for si in range(len(subs)):
                    lo, mu = paired_lcb(co.Y[irows, j], T[si, :, j],
                                        T[full_i, :, j], boots, alpha_c)
                    if not np.isnan(lo) and lo >= -args.delta:
                        ok.append((sum(subs[si]), -mu, si))
                    if fi == 0 and seed == 0:
                        out.append({
                            "finding": co.findings[j],
                            "subset": subset_name(subs[si]),
                            "k": int(sum(subs[si])), "delta_mean": mu,
                            "delta_lcb": lo, "certified": bool(
                                not np.isnan(lo) and lo >= -args.delta)})
                best = min(ok)[2] if ok else full_i
                return j, best, bool(ok), out

            res = Parallel(n_jobs=n_jobs)(delayed(certify)(j) for j in keep)
            for j, best, was_ok, out in res:
                rows.extend(out)
                sel_scores[ev, j] = S[best, ev, j]
                min_scores[ev, j] = sum(subs[best])
                pol.append({"seed": seed, "fold": fi,
                            "finding": co.findings[j],
                            "subset": subset_name(subs[best]),
                            "k": int(sum(subs[best])),
                            "certified": was_ok})
        au = [safe_auroc(co.Y[:, j], sel_scores[:, j]) for j in keep]
        fu = [safe_auroc(co.Y[:, j], S[full_i, :, j]) for j in keep]
        summ.append({"seed": seed, "macro_policy": macro(au),
                     "macro_all_inputs": macro(fu),
                     "delta_macro": macro(au) - macro(fu),
                     "mean_inputs": float(np.nanmean(min_scores[:, keep])),
                     "n_findings": len(keep)})
        print(f"  seed {seed}: policy {macro(au):.4f} vs all-inputs "
              f"{macro(fu):.4f}  mean |S| = "
              f"{np.nanmean(min_scores[:, keep]):.2f}", flush=True)

    pd.DataFrame(rows).to_csv(f"{RES}/certified_subsets{args.tag}.csv",
                              index=False)
    P = pd.DataFrame(pol)
    P.to_csv(f"{RES}/risk_policy{args.tag}.csv", index=False)
    D = pd.DataFrame(summ)
    D.to_csv(f"{RES}/risk_controlled{args.tag}.csv", index=False)

    # ---- what a multi-finding request actually costs ----------------------
    # Selecting one input per finding saves nothing if answering a realistic
    # request means taking the union over many findings.
    mode = (P[P.seed == 0].groupby("finding").subset
            .agg(lambda x: x.value_counts().index[0]))
    name2mask = {subset_name(m): m for m in subs}
    urows = []
    r2 = np.random.default_rng(0)
    names = list(mode.index)
    for q in [1, 2, 5, 10, len(names)]:
        vals = []
        for _ in range(300):
            pick = r2.choice(names, size=min(q, len(names)), replace=False)
            u = np.zeros(NSEQ, int)
            for f in pick:
                u |= np.array(name2mask[mode[f]], int)
            vals.append(u.sum())
        urows.append({"n_findings_queried": q,
                      "mean_union_size": float(np.mean(vals)),
                      "frac_needing_all_four": float(np.mean(
                          [v == NSEQ for v in vals]))})
    U = pd.DataFrame(urows)
    U.to_csv(f"{RES}/policy_union{args.tag}.csv", index=False)

    print("\n=== risk-controlled minimal evidence "
          f"(delta={args.delta}, alpha={args.alpha}, Bonferroni over "
          f"{len(subs)} candidates) ===")
    print(D.round(4).to_string(index=False))
    print("\npolicy size distribution (seed 0):")
    print(P[P.seed == 0].k.value_counts().sort_index().to_string())
    print("\n=== union cost of a multi-finding request ===")
    print(U.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
