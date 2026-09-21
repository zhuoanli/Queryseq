"""Patient-bootstrap rank stability for the prospectively specified controls.

"Exact Shapley" removes coalition-sampling error, not patient-sampling error.
A reviewer pointed at the load-bearing case: Cerebral hemorrhage's SWI-first
rank rests on a 0.0013 Shapley gap. This script prices every pre-specified
rank event under patient resampling:

  MR-RATE (SWI control): for each haemorrhage-family finding, P(SWI first)
  and the CI of the paired gap Shapley(SWI) - Shapley(runner-up); jointly,
  P(first in >= 3 of 4). For each blood-product-free control, P(SWI last),
  and jointly P(last in all 4).

  BraTS-GoAT: P(T1c first for ET), P(T2f first for SNFH), paired-gap CIs,
  under both the frozen amended threshold and the per-fold sensitivity
  labels.

Everything is recomputed per replicate from cached out-of-fold scores; no
model is refit.
"""
import argparse
import os
import sys
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, safe_auroc)  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")

HAEM = ["Cerebral hemorrhage", "Silent micro-hemorrhage of brain",
        "Cavernous hemangioma", "Subdural intracranial hemorrhage"]
CTRL = ["Cerebral atrophy", "Ventriculomegaly", "Empty sella syndrome",
        "Mega cisterna magna"]


def shapley_from_U(U, subs, n):
    """Exact Shapley per player from a utility vector over all subsets."""
    idx = {sum(b << i for i, b in enumerate(m)): k for k, m in enumerate(subs)}
    w = {s: factorial(s) * factorial(n - s - 1) / factorial(n)
         for s in range(n)}
    phi = np.zeros(n)
    for mi in range(n):
        tot = 0.0
        for r in range(n):
            for S in combinations([x for x in range(n) if x != mi], r):
                b = sum(1 << x for x in S)
                tot += w[r] * (U[idx[b | (1 << mi)]]
                               - (U[idx[b]] if b else 0.5))
        phi[mi] = tot
    return phi


def mrrate(args):
    co = Cohort("train", complete_case=True)
    subs = all_subsets()
    n = sum(subs[-1])
    from common import SEQUENCES, SEQ_SHORT
    seq_names = [SEQ_SHORT[s] for s in SEQUENCES]
    swi = seq_names.index("SWIax")
    names = list(co.findings)
    jH = [names.index(f) for f in HAEM]
    jC = [names.index(f) for f in CTRL]
    S = np.stack([np.load(f"{CACHE}/oof_seed{sd}{args.tag}.npy")
                  .astype(np.float32) for sd in range(args.seeds)])

    pats = np.unique(co.patients)
    by_pat = {p: np.where(co.patients == p)[0] for p in pats}

    def one_rep(r):
        rng = np.random.default_rng(50_000 + r)
        rows = np.concatenate(
            [by_pat[p] for p in rng.choice(pats, len(pats), replace=True)])
        y = co.Y[rows]
        out = {}
        for j in jH + jC:
            U = np.array([np.mean([safe_auroc(y[:, j], S[sd, si, rows, j])
                                   for sd in range(args.seeds)])
                          for si in range(len(subs))])
            phi = shapley_from_U(U, subs, n)
            order = np.argsort(-phi)
            gap = phi[swi] - np.max(np.delete(phi, swi))
            out[j] = (int(order[0] == swi), int(order[-1] == swi), float(gap))
        return out

    reps = Parallel(n_jobs=args.jobs, verbose=1)(
        delayed(one_rep)(r) for r in range(args.reps))

    rows_out = []
    firsts = {j: np.array([r[j][0] for r in reps]) for j in jH}
    lasts = {j: np.array([r[j][1] for r in reps]) for j in jC}
    gaps = {j: np.array([r[j][2] for r in reps]) for j in jH + jC}
    for f, j in zip(HAEM, jH):
        rows_out.append({"dataset": "mrrate", "finding": f, "event": "swi_first",
                         "prob": float(firsts[j].mean()),
                         "gap_lo": float(np.percentile(gaps[j], 2.5)),
                         "gap_hi": float(np.percentile(gaps[j], 97.5))})
    for f, j in zip(CTRL, jC):
        rows_out.append({"dataset": "mrrate", "finding": f, "event": "swi_last",
                         "prob": float(lasts[j].mean()),
                         "gap_lo": float(np.percentile(gaps[j], 2.5)),
                         "gap_hi": float(np.percentile(gaps[j], 97.5))})
    n_first = np.sum([firsts[j] for j in jH], axis=0)
    n_last = np.sum([lasts[j] for j in jC], axis=0)
    rows_out.append({"dataset": "mrrate", "finding": "(joint)",
                     "event": "first_ge3_of_4",
                     "prob": float((n_first >= 3).mean()),
                     "gap_lo": np.nan, "gap_hi": np.nan})
    rows_out.append({"dataset": "mrrate", "finding": "(joint)",
                     "event": "last_all_4",
                     "prob": float((n_last == 4).mean()),
                     "gap_lo": np.nan, "gap_hi": np.nan})
    return rows_out


def goat(args):
    mods = ["t1n", "t1c", "t2w", "t2f"]
    subs = [tuple((b >> j) & 1 for j in range(4)) for b in range(1, 16)]
    subs = sorted(subs, key=lambda m: (sum(m), m))
    targets = ["ET", "SNFH", "NETC"]
    out_rows = []
    for kind in ("burden", "burden_perfold"):
        Zs = []
        for sd in range(args.seeds):
            f = f"{RES}/goat_scores_{kind}_seed{sd}.npz"
            if not os.path.exists(f):
                print(f"  goat: missing {f}, skipping kind={kind}")
                break
            Zs.append(np.load(f))
        else:
            scores = [z["scores"].astype(np.float32) for z in Zs]
            # labels are per seed: under the per-(seed, fold) threshold
            # sensitivity each seed's fold split induces its own label array
            Ys = [z["y"].astype(int) for z in Zs]
            nc = Ys[0].shape[0]

            def one_rep(r):
                rng = np.random.default_rng(70_000 + r)
                rows = rng.choice(nc, nc, replace=True)   # 1 case = 1 patient
                res = {}
                for ti, t in enumerate(targets[:2]):      # ET, SNFH
                    U = np.array([np.mean([safe_auroc(y[rows, ti],
                                                      sc[si][rows, ti])
                                           for sc, y in zip(scores, Ys)])
                                  for si in range(len(subs))])
                    phi = shapley_from_U(U, subs, 4)
                    order = np.argsort(-phi)
                    want = mods.index("t1c") if t == "ET" else mods.index("t2f")
                    gap = phi[want] - np.max(np.delete(phi, want))
                    res[t] = (int(order[0] == want), float(gap))
                return res

            reps = Parallel(n_jobs=args.jobs, verbose=1)(
                delayed(one_rep)(r) for r in range(args.reps))
            for t, hyp in (("ET", "t1c_first"), ("SNFH", "t2f_first")):
                pr = np.array([r[t][0] for r in reps])
                gp = np.array([r[t][1] for r in reps])
                out_rows.append({"dataset": f"goat_{kind}", "finding": t,
                                 "event": hyp, "prob": float(pr.mean()),
                                 "gap_lo": float(np.percentile(gp, 2.5)),
                                 "gap_hi": float(np.percentile(gp, 97.5))})
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    rows = mrrate(args) + goat(args)
    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/rank_stability{args.tag}.csv", index=False)
    print(d.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
