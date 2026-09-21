"""Held-out replication of the selection gain on the eligible test findings.

The atlas gain is estimated inside the training cohort by nested
cross-validation, because only a minority of findings are evaluable on the
official test split. A reviewer reasonably asked for the complement: on the
findings that ARE test-evaluable, does train-side per-finding selection beat
the fixed all-input policy on genuinely untouched data?

Selection here uses the training cohort's standard OOF utilities (averaged
over seeds). With respect to the TEST cohort that selection is strictly
nested by construction: no test label ever influenced any training-fold fit.
For each eligible finding, the selected subset's head and the full-set head
are refitted on all of train and applied once to test; the paired difference
is bootstrapped at patient level on the test cohort.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, finding_filter,  # noqa: E402
                    safe_auroc, subset_name)
from subset_lattice import pool  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def refit_score(tr_co, te_co, mask, j, C):
    Xtr = pool(tr_co.P, tr_co.M, mask)
    Xte = pool(te_co.P, te_co.M, mask)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    m.fit(sc.transform(Xtr), tr_co.Y[:, j])
    return m.decision_function(sc.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    tr = Cohort("train", complete_case=True)
    te = Cohort("test", complete_case=True)
    keep_te = finding_filter(te.Y, te.findings, min_pos=args.min_pos)
    subs = all_subsets()
    print(f"train={len(tr)} test={len(te)} eligible test findings="
          f"{len(keep_te)}", flush=True)

    # seed-averaged train-side utility per (subset, finding): the deployment
    # selection a practitioner would make, nested w.r.t. test by construction
    U = np.zeros((len(subs), te.Y.shape[1]))
    for seed in range(args.seeds):
        S = np.load(f"{CACHE}/oof_seed{seed}{args.tag}.npy").astype(np.float32)
        for si in range(len(subs)):
            for j in keep_te:
                U[si, j] += safe_auroc(tr.Y[:, j], S[si, :, j]) / args.seeds

    full_mask = subs[-1]
    rows, sel_scores, full_scores = [], {}, {}
    for j in keep_te:
        si = int(np.nanargmax(U[:, j]))
        sel_scores[j] = refit_score(tr, te, subs[si], j, args.C)
        full_scores[j] = refit_score(tr, te, full_mask, j, args.C)
        a_sel = safe_auroc(te.Y[:, j], sel_scores[j])
        a_ful = safe_auroc(te.Y[:, j], full_scores[j])
        rows.append({"finding": te.findings[j],
                     "subset": subset_name(subs[si]),
                     "k": int(sum(subs[si])),
                     "n_pos_test": int(te.Y[:, j].sum()),
                     "auroc_selected": a_sel, "auroc_full": a_ful,
                     "gain": a_sel - a_ful})
        print(f"  {te.findings[j][:34]:34s} {subset_name(subs[si]):>34s} "
              f"sel={a_sel:.4f} full={a_ful:.4f} gain={a_sel - a_ful:+.4f}",
              flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(f"{RES}/heldout_gain_perfinding{args.tag}.csv", index=False)

    pats = np.unique(te.patients)
    by_pat = {p: np.where(te.patients == p)[0] for p in pats}

    def one_rep(r):
        rng = np.random.default_rng(30_000 + r)
        rr = np.concatenate(
            [by_pat[p] for p in rng.choice(pats, len(pats), replace=True)])
        y = te.Y[rr]
        diffs = [safe_auroc(y[:, j], sel_scores[j][rr])
                 - safe_auroc(y[:, j], full_scores[j][rr]) for j in keep_te]
        diffs = [d for d in diffs if np.isfinite(d)]
        return float(np.mean(diffs)) if diffs else np.nan

    boots = Parallel(n_jobs=args.jobs, verbose=1)(
        delayed(one_rep)(r) for r in range(args.reps))
    boots = np.array([b for b in boots if np.isfinite(b)])
    point = float(per.gain.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])

    pd.DataFrame([{
        "mean_gain": point, "ci_lo": float(lo), "ci_hi": float(hi),
        "n_findings": len(keep_te),
        "n_pos_gain": int((per.gain > 0).sum()),
        "n_reps": len(boots),
    }]).to_csv(f"{RES}/heldout_gain{args.tag}.csv", index=False)
    print(f"\nheld-out selection gain vs fixed full set: {point:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  "
          f"({(per.gain > 0).sum()}/{len(keep_te)} findings positive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
