"""Two robustness checks that the atlas claims have to survive.

**High-confidence sign flips.**  Saying that half of input pairs change the sign
of their interaction with context is only meaningful if those signs are
themselves resolved.  Many interactions are small, and a quantity wandering
around zero will "change sign" constantly without anything happening.  So each
interaction gets a patient-level bootstrap interval, and a cell counts as a
high-confidence flip only when the interval lies strictly on opposite sides of
zero in two different contexts.  The raw rate and the high-confidence rate are
both reported; the second is the one the claim should rest on.

**Eligibility sensitivity.**  Every aggregate depends on the prevalence floor
that decides which findings are evaluable.  The pre-registered floor is 20
positives; here the headline lattice quantities are recomputed at 10, 20 and 30
so a reader can see whether the conclusions are an artifact of that choice.
"""
import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, NSEQ, SEQUENCES, SEQ_SHORT, Cohort,  # noqa: E402
                    all_subsets, finding_filter, macro, safe_auroc,
                    subset_name)
from risk_controlled import bootstrap_indices  # noqa: E402


def _auc(y, s):
    """AUROC over defined entries; lattice columns below the floor are NaN."""
    m = np.isfinite(s)
    if m.sum() < 10:
        return np.nan
    return safe_auroc(y[m], s[m])

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def bits(mask):
    return sum(b << i for i, b in enumerate(mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--floors", type=str, default="20,30,50",
                    help="only >= the lattice floor; below it the "
                         "lattice has no predictions to average")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    co = Cohort("train", complete_case=True)
    subs = all_subsets()
    idx_of = {bits(m): i for i, m in enumerate(subs)}
    rng = np.random.default_rng(0)

    # ---- 1. bootstrap intervals on every interaction ---------------------
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    S = np.load(f"{CACHE}/oof_seed0{args.tag}.npy").astype(np.float32)
    boots = bootstrap_indices(co.patients, rng, args.boot)
    lo_q, hi_q = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)

    rows = []
    for j in keep:
        y = co.Y[:, j]
        # AUROC of every subset under every bootstrap replicate, once
        U = np.full((len(subs), len(boots)), np.nan)
        for si in range(len(subs)):
            for b, i in enumerate(boots):
                yi = y[i]
                if 0 < yi.sum() < len(i):
                    U[si, b] = _auc(yi, S[si, i, j])
        for i1, i2 in combinations(range(NSEQ), 2):
            rest = [x for x in range(NSEQ) if x not in (i1, i2)]
            for r in range(len(rest) + 1):
                for C in combinations(rest, r):
                    b0 = sum(1 << x for x in C)
                    if b0 == 0:
                        continue      # degenerate: see shapley_synergy.py
                    psi = (U[idx_of[b0 | (1 << i1) | (1 << i2)]]
                           - U[idx_of[b0 | (1 << i1)]]
                           - U[idx_of[b0 | (1 << i2)]]
                           + U[idx_of[b0]])
                    psi = psi[np.isfinite(psi)]
                    if len(psi) < 20:
                        continue
                    rows.append({
                        "finding": co.findings[j],
                        "pair": f"{SEQ_SHORT[SEQUENCES[i1]]}|"
                                f"{SEQ_SHORT[SEQUENCES[i2]]}",
                        "context": subset_name(
                            tuple(1 if x in C else 0 for x in range(NSEQ))),
                        "synergy": float(psi.mean()),
                        "ci_lo": float(np.percentile(psi, lo_q)),
                        "ci_hi": float(np.percentile(psi, hi_q)),
                    })
    d = pd.DataFrame(rows)
    d["sig_pos"] = d.ci_lo > 0
    d["sig_neg"] = d.ci_hi < 0
    d.to_csv(f"{RES}/synergy_ci{args.tag}.csv", index=False)

    cell = d.groupby(["finding", "pair"]).agg(
        raw_min=("synergy", "min"), raw_max=("synergy", "max"),
        any_pos=("sig_pos", "any"), any_neg=("sig_neg", "any"))
    raw_flip = float(((cell.raw_min < 0) & (cell.raw_max > 0)).mean())
    hc_flip = float((cell.any_pos & cell.any_neg).mean())
    resolved = float((d.sig_pos | d.sig_neg).mean())

    # ---- 2. eligibility sensitivity --------------------------------------
    srows = []
    for floor in [int(x) for x in args.floors.split(",")]:
        kp = finding_filter(co.Y, co.findings, min_pos=floor)
        Um = np.full((len(subs), len(co.findings)), np.nan)
        for seed in range(args.seeds):
            Ss = np.load(f"{CACHE}/oof_seed{seed}{args.tag}.npy").astype(
                np.float32)
            for si in range(len(subs)):
                for j in kp:
                    v = _auc(co.Y[:, j], Ss[si, :, j])
                    Um[si, j] = v if np.isnan(Um[si, j]) else (
                        Um[si, j] + v)
        Um /= args.seeds
        full_i = idx_of[(1 << NSEQ) - 1]
        best = {j: int(np.nanargmax(Um[:, j])) for j in kp}
        gb = int(np.nanargmax(np.nanmean(Um[:, kp], 1)))
        srows.append({
            "min_pos": floor, "n_findings": len(kp),
            "n_distinct_best": len(set(best.values())),
            "macro_full4": macro([Um[full_i, j] for j in kp]),
            "macro_best_fixed": macro([Um[gb, j] for j in kp]),
            "macro_oracle_perfinding": macro([Um[best[j], j] for j in kp]),
            "full4_rank": int(np.argsort(
                -np.nanmean(Um[:, kp], 1)).tolist().index(full_i)) + 1,
        })
    s = pd.DataFrame(srows)
    s.to_csv(f"{RES}/eligibility_sensitivity{args.tag}.csv", index=False)

    pd.DataFrame([{"metric": "synergy_raw_sign_flip", "value": raw_flip},
                  {"metric": "synergy_highconf_sign_flip", "value": hc_flip},
                  {"metric": "synergy_frac_resolved", "value": resolved},
                  {"metric": "synergy_n_cells", "value": len(cell)}]).to_csv(
        f"{RES}/synergy_confidence{args.tag}.csv", index=False)

    print(f"interactions with a CI excluding zero : {resolved:.3f}")
    print(f"raw sign-flip rate over contexts      : {raw_flip:.3f}")
    print(f"high-confidence sign-flip rate        : {hc_flip:.3f}"
          f"   (of {len(cell)} finding-pair cells)")
    print("\n=== eligibility sensitivity ===")
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
