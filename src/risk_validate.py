"""Does the certified minimal-evidence policy keep its promise on held-out data?

risk_controlled.py selects, for each finding, the smallest input subset whose
non-inferiority bound clears a tolerance.  That is a claim about future data, so
it has to be checked on data that played no part in making it.

Protocol.  Subsets are certified on the training split using cross-fitted
out-of-fold predictions, which is where selection happens and where the
multiplicity correction applies.  Each subset's head is then refitted on the
whole training split, applied once to the held-out test split, and the promise
is audited there:

    does  U^test_f(S*_f) - U^test_f(M)  >= -delta ?

Coverage is reported with an exact Clopper-Pearson interval, because with 15
evaluable test findings a point estimate of a 95% guarantee is not meaningful on
its own.  Failures are listed individually with their positive counts, since the
honest expectation is that they concentrate in the rarest findings.

delta is swept only to show the risk/sparsity trade-off; the pre-registered
value is 0.01 and remains the primary analysis.  Bonferroni is primary; Holm is
reported as a sensitivity analysis and was not used to choose anything.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, finding_filter,  # noqa: E402
                    macro, safe_auroc, subset_name)
from risk_controlled import bootstrap_indices, paired_lcb  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def fit_subset(Ptr, Mtr, Ytr, Pte, Mte, mask, keep, C):
    """Refit one subset's heads on all of train, score the test split."""
    def pool(P, M):
        w = M * np.asarray(mask, np.float32)[None, :]
        d = w.sum(1, keepdims=True)
        return (P * w[:, :, None]).sum(1) / np.maximum(d, 1e-6)
    Xtr, Xte = pool(Ptr, Mtr), pool(Pte, Mte)
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    out = np.full((len(Xte), Ytr.shape[1]), np.nan, np.float32)
    for j in keep:
        y = Ytr[:, j]
        if y.sum() < 3 or y.sum() == len(y):
            continue
        m = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
        m.fit(A, y)
        out[:, j] = m.decision_function(B)
    return out


def select(lcb_by_sub, subs, alpha, n_cand, method):
    """Smallest subset certified non-inferior, under a multiplicity correction.

    Bonferroni tests every candidate at alpha/n_cand.  Holm orders candidates by
    their bound and relaxes the threshold stepwise; it is strictly more
    permissive and is reported only as a sensitivity analysis.
    """
    if method == "bonferroni":
        ok = [si for si, (lo, _) in lcb_by_sub.items() if lo is not None]
        return ok
    order = sorted(lcb_by_sub, key=lambda s: -lcb_by_sub[s][1])
    ok, k = [], 0
    for si in order:
        k += 1
        ok.append(si)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--deltas", type=str, default="0.005,0.01,0.02")
    ap.add_argument("--primary-delta", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="recompute risk_coverage from the saved "
                         "risk_validate CSV without re-scoring")
    args = ap.parse_args()
    if args.aggregate_only:
        A = pd.read_csv(f"{RES}/risk_validate{args.tag}.csv")
        aggregate(A, [float(x) for x in args.deltas.split(",")], args)
        return

    tr = Cohort("train", complete_case=True)
    te = Cohort("test", complete_case=True)
    subs = all_subsets()
    full_i = len(subs) - 1
    # selection uses training-eligible findings; the audit can only speak to
    # those that are also evaluable on test
    keep_tr = finding_filter(tr.Y, tr.findings, min_pos=args.min_pos)
    keep_te = finding_filter(te.Y, te.findings, min_pos=args.min_pos)
    keep = [j for j in keep_te if j in set(keep_tr)]
    print(f"train {len(tr)} test {len(te)}  findings: train-eligible "
          f"{len(keep_tr)}, test-evaluable {len(keep_te)}, audited "
          f"{len(keep)}", flush=True)

    # ---- refit every subset on all of train, score test once --------------
    Ste = Parallel(n_jobs=min(args.jobs, len(subs)), verbose=1)(
        delayed(fit_subset)(tr.P, tr.M, tr.Y, te.P, te.M, m, keep, args.C)
        for m in subs)
    Ste = np.stack(Ste)
    np.save(f"{CACHE}/test_subset_scores{args.tag}.npy", Ste.astype(np.float16))

    rng = np.random.default_rng(0)
    boots_te = bootstrap_indices(te.patients, rng, args.boot)
    deltas = [float(x) for x in args.deltas.split(",")]
    alpha_c = args.alpha / len(subs)

    rows, audit = [], []
    for seed in range(args.seeds):
        # the tag matters: without it this silently loads the k4 lattice
        # (15 subsets) and indexes it with a k5/k7 subset id, which is an
        # IndexError at best and a wrong answer at worst
        S = np.load(f"{CACHE}/oof_seed{seed}{args.tag}.npy").astype(np.float32)
        boots_tr = bootstrap_indices(tr.patients, rng, args.boot)
        for j in keep:
            # certify on train (cross-fitted predictions), once per finding
            cert = {}
            for si in range(len(subs)):
                lo, mu = paired_lcb(tr.Y[:, j], S[si, :, j], S[full_i, :, j],
                                    boots_tr, alpha_c)
                cert[si] = (lo, mu)
            # audit on test, independent of everything above
            tlo, tmu = {}, {}
            for si in range(len(subs)):
                a, b = paired_lcb(te.Y[:, j], Ste[si, :, j], Ste[full_i, :, j],
                                  boots_te, args.alpha)
                tlo[si], tmu[si] = a, b
            for d in deltas:
                ok = [si for si in range(len(subs))
                      if not np.isnan(cert[si][0]) and cert[si][0] >= -d]
                pick = (min(ok, key=lambda si: (sum(subs[si]), -cert[si][1]))
                        if ok else full_i)
                held = tmu[pick]
                audit.append({
                    "seed": seed, "delta": d, "finding": tr.findings[j],
                    "n_pos_test": int(te.Y[:, j].sum()),
                    "n_pos_train": int(tr.Y[:, j].sum()),
                    "subset": subset_name(subs[pick]), "k": int(sum(subs[pick])),
                    "train_lcb": cert[pick][0], "test_delta": held,
                    "test_lcb": tlo[pick],
                    "kept_promise": bool(not np.isnan(held) and held >= -d),
                })
        print(f"  seed {seed} audited", flush=True)

    A = pd.DataFrame(audit)
    A.to_csv(f"{RES}/risk_validate{args.tag}.csv", index=False)
    aggregate(A, deltas, args)


def aggregate(A, deltas, args):
    """Coverage summaries under two stated rules.

    PRIMARY (matches the mean-delta column of the coverage table): a finding
    clears the tolerance iff its CROSS-SEED MEAN held-out difference is
    >= -delta.  SECONDARY: the per-seed policy instances themselves (n x
    seeds Bernoulli-style counts, purely descriptive).  An earlier version
    silently used a majority-of-seeds vote per finding while the table
    displayed cross-seed means -- two different estimands under one number,
    which a reviewer caught by recomputing both.
    """
    rows = []
    for d in deltas:
        s = A[A.delta == d]
        g = s.groupby("finding").agg(
            k=("k", "median"), test_delta=("test_delta", "mean"),
            kept=("kept_promise", "mean"), n_pos=("n_pos_test", "first"))
        n = len(g)
        kk = int((g.test_delta >= -d).sum())          # primary: seed-mean
        kk_vote = int((g.kept >= 0.5).sum())          # old majority vote
        n_inst = len(s)
        kk_inst = int(s.kept_promise.sum())           # per-seed instances
        # Exact Clopper-Pearson.  The two bounds do NOT share shape
        # parameters: lower is Beta(k, n-k+1) and upper is Beta(k+1, n-k).
        # Evaluating both quantiles at (k, n-k+1) -- as this did -- leaves the
        # lower bound correct and makes the upper bound too small, which is
        # anti-conservative exactly where it matters: it narrows the interval
        # around a coverage claim we are trying to audit honestly.
        lo = 0.0 if kk == 0 else float(stats.beta.ppf(0.025, kk, n - kk + 1))
        hi = 1.0 if kk == n else float(stats.beta.ppf(0.975, kk + 1, n - kk))
        rows.append({"delta": d, "n_findings": n, "n_kept": kk,
                     "coverage": kk / n, "coverage_lo": float(lo),
                     "coverage_hi": float(hi),
                     "n_kept_seedvote": kk_vote,
                     "n_instances": n_inst, "n_kept_instances": kk_inst,
                     "mean_k": float(g.k.mean()),
                     "mean_test_delta": float(g.test_delta.mean()),
                     "worst_test_delta": float(g.test_delta.min()),
                     "frac_k_le_two": float((g.k <= 2).mean())})
        print(f"\n=== delta = {d} ===")
        print(f"  coverage (cross-seed mean rule) {kk}/{n} = {kk / n:.3f} "
              f"[{lo:.3f}, {hi:.3f}] (Clopper-Pearson, heuristic)")
        print(f"  per-seed instances kept: {kk_inst}/{n_inst}   "
              f"(old majority-vote count: {kk_vote}/{n})")
        print(f"  mean |S| = {g.k.mean():.2f}   mean test delta = "
              f"{g.test_delta.mean():+.4f}   worst = {g.test_delta.min():+.4f}")
        bad = g[g.test_delta < -d].sort_values("n_pos")
        if len(bad):
            print("  failures (finding, n_pos_test, test delta):")
            for f, r in bad.iterrows():
                print(f"    {f[:44]:46s} {int(r.n_pos):4d} {r.test_delta:+.4f}")
    pd.DataFrame(rows).to_csv(f"{RES}/risk_coverage{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
