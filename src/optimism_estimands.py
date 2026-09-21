"""Two different optimism estimands, computed side by side and named apart.

An earlier draft reported 0.0102 in one place and 0.0149 in another, both called
"optimism".  They are both correct and they are not the same quantity, which is
exactly the kind of thing a reader is entitled to notice and we are obliged to
prevent.  This script computes both from the same predictions so the difference
is documented rather than latent.

  policy_oracle_gap
      Per-finding subset selection where the utility of each subset is first
      *averaged over seeds* and the argmax is taken on that average, against the
      cross-fitted policy averaged the same way.  This is what a careful
      practitioner running several seeds would experience.

  search_optimism
      Naive best-of-K on a single run against the cross-fitted policy from that
      same run.  This is what a practitioner running one experiment, taking the
      maximum and reporting it would experience -- the realistic winner's curse.

The gap between them is not noise: averaging several seeds before selecting
shrinks the maximum's upward bias, so seed averaging is itself a partial
mitigation of the effect this paper is about.  We report that reduction.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, finding_filter,  # noqa: E402
                    macro, patient_folds, safe_auroc)
from shapley_synergy import load_inner, nested_policy  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def _auc(y, s):
    m = np.isfinite(s)
    return safe_auroc(y[m], s[m]) if m.sum() >= 10 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    co = Cohort("train", complete_case=True)
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    subs = all_subsets()
    nS = len(subs)

    # ---- estimand 1: seed-averaged utilities, then select -----------------
    U = np.full((args.seeds, nS, co.Y.shape[1]), np.nan)
    for sd in range(args.seeds):
        S = np.load(f"{CACHE}/oof_seed{sd}{args.tag}.npy").astype(np.float32)
        for si in range(nS):
            for j in keep:
                U[sd, si, j] = _auc(co.Y[:, j], S[si, :, j])
    with np.errstate(invalid="ignore"):
        Um = np.nanmean(U, 0)
    oracle_avg = macro([Um[int(np.nanargmax(Um[:, j])), j] for j in keep])

    nested, search = [], []
    for sd in range(args.seeds):
        S = np.load(f"{CACHE}/oof_seed{sd}{args.tag}.npy").astype(np.float32)
        folds = patient_folds(co.patients, k=args.folds, seed=sd)
        pf, _ = nested_policy(S, co.Y, folds, keep, "per_finding",
                              inner=load_inner(sd, args.tag, args.folds))
        cf = macro([_auc(co.Y[:, j], pf[:, j]) for j in keep])
        nested.append(cf)
        naive = macro([np.nanmax([_auc(co.Y[:, j], S[si, :, j])
                                  for si in range(nS)]) for j in keep])
        search.append(naive - cf)
    nested_avg = float(np.mean(nested))

    rows = [
        {"estimand": "policy_oracle_gap",
         "description": "seed-averaged oracle minus seed-averaged cross-fitted",
         "naive_or_oracle": oracle_avg, "crossfitted": nested_avg,
         "optimism": oracle_avg - nested_avg, "n_seeds": args.seeds,
         "n_candidates": nS},
        {"estimand": "search_optimism",
         "description": "single-run naive best-of-K minus that run's cross-fitted",
         "naive_or_oracle": np.nan, "crossfitted": np.nan,
         "optimism": float(np.mean(search)), "n_seeds": args.seeds,
         "n_candidates": nS},
        {"estimand": "search_optimism_seed0",
         "description": "the single-run figure for seed 0, as quoted in the null table",
         "naive_or_oracle": np.nan, "crossfitted": np.nan,
         "optimism": float(search[0]), "n_seeds": 1, "n_candidates": nS},
        {"estimand": "seed_averaging_reduction",
         "description": "how much averaging seeds before selecting shrinks the bias",
         "naive_or_oracle": np.nan, "crossfitted": np.nan,
         "optimism": float(np.mean(search)) - (oracle_avg - nested_avg),
         "n_seeds": args.seeds, "n_candidates": nS},
    ]
    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/optimism_estimands{args.tag}.csv", index=False)
    print(d.round(4).to_string(index=False))
    print("\nBoth are correct.  They differ because averaging several seeds "
          "before taking the\nargmax reduces the maximum's upward bias.  Name "
          "them apart in the manuscript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
