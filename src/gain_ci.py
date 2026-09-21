"""Patient-level paired bootstrap for the headline cross-fitted gains.

Loads the pooled prediction arrays that shapley_synergy.py dumps after its
strictly nested run (nestedpf/nestedgl), plus the full-set OOF column, and
bootstraps two paired differences at patient level:

  * per-finding nested selection  vs  the cross-fitted global-selection
    procedure (the two procedures compared under identical nesting);
  * per-finding nested selection  vs  the actual all-input fixed policy.

One patient resample per replicate is shared by every arm and every seed.
The interval is conditional on the realised selections: policy choice is not
re-run inside replicates, so this quantifies evaluation uncertainty of the
selected policies, not selection variability under resampling -- the paper
states this distinction explicitly.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, finding_filter,  # noqa: E402
                    macro, safe_auroc)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    co = Cohort("train", complete_case=True)
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    Y = co.Y
    full_i = len(all_subsets()) - 1

    PF, GL, FX = [], [], []
    for seed in range(args.seeds):
        PF.append(np.load(f"{CACHE}/nestedpf_seed{seed}{args.tag}.npy")
                  .astype(np.float32))
        GL.append(np.load(f"{CACHE}/nestedgl_seed{seed}{args.tag}.npy")
                  .astype(np.float32))
        FX.append(np.load(f"{CACHE}/oof_seed{seed}{args.tag}.npy")
                  .astype(np.float32)[full_i])
    print(f"n={len(co)} findings={len(keep)} seeds={args.seeds}", flush=True)

    def macro_of(A, rows=None):
        r = slice(None) if rows is None else rows
        y = Y if rows is None else Y[rows]
        return macro([safe_auroc(y[:, j], A[r, j]) for j in keep])

    point_gl = float(np.mean([macro_of(PF[s]) - macro_of(GL[s])
                              for s in range(args.seeds)]))
    point_fx = float(np.mean([macro_of(PF[s]) - macro_of(FX[s])
                              for s in range(args.seeds)]))

    pats = np.unique(co.patients)
    by_pat = {p: np.where(co.patients == p)[0] for p in pats}

    def one_rep(r):
        rng = np.random.default_rng(10_000 + r)
        rows = np.concatenate(
            [by_pat[p] for p in rng.choice(pats, len(pats), replace=True)])
        d_gl = float(np.mean([macro_of(PF[s], rows) - macro_of(GL[s], rows)
                              for s in range(args.seeds)]))
        d_fx = float(np.mean([macro_of(PF[s], rows) - macro_of(FX[s], rows)
                              for s in range(args.seeds)]))
        return d_gl, d_fx

    boots = Parallel(n_jobs=args.jobs, verbose=1)(
        delayed(one_rep)(r) for r in range(args.reps))
    bg = np.array([b[0] for b in boots if np.isfinite(b[0])])
    bf = np.array([b[1] for b in boots if np.isfinite(b[1])])

    per = []
    for j in keep:
        g_gl = float(np.mean([safe_auroc(Y[:, j], PF[s][:, j])
                              - safe_auroc(Y[:, j], GL[s][:, j])
                              for s in range(args.seeds)]))
        g_fx = float(np.mean([safe_auroc(Y[:, j], PF[s][:, j])
                              - safe_auroc(Y[:, j], FX[s][:, j])
                              for s in range(args.seeds)]))
        per.append({"finding": co.findings[j], "gain_vs_global": g_gl,
                    "gain_vs_fixed": g_fx, "n_pos": int(Y[:, j].sum())})
    per = pd.DataFrame(per).sort_values("gain_vs_global", ascending=False)
    per.to_csv(f"{RES}/gain_perfinding{args.tag}.csv", index=False)

    pd.DataFrame([{
        "point": point_gl,
        "ci_lo": float(np.percentile(bg, 2.5)),
        "ci_hi": float(np.percentile(bg, 97.5)),
        "point_vs_fixed": point_fx,
        "ci_lo_fixed": float(np.percentile(bf, 2.5)),
        "ci_hi_fixed": float(np.percentile(bf, 97.5)),
        "n_reps": len(bg), "n_findings": len(keep),
        "n_pos_gain": int((per.gain_vs_global > 0).sum()),
        "n_pos_gain_fixed": int((per.gain_vs_fixed > 0).sum()),
        "median_gain": float(per.gain_vs_global.median()),
        "max_gain": float(per.gain_vs_global.max()),
        "min_gain": float(per.gain_vs_global.min()),
    }]).to_csv(f"{RES}/gain_ci{args.tag}.csv", index=False)

    print(f"vs global procedure: {point_gl:+.4f} "
          f"[{np.percentile(bg, 2.5):+.4f}, {np.percentile(bg, 97.5):+.4f}]")
    print(f"vs fixed all-input : {point_fx:+.4f} "
          f"[{np.percentile(bf, 2.5):+.4f}, {np.percentile(bf, 97.5):+.4f}]")
    print(f"findings improving vs global: {(per.gain_vs_global > 0).sum()}"
          f"/{len(per)}; vs fixed: {(per.gain_vs_fixed > 0).sum()}/{len(per)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
