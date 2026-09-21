"""Paired patient-level bootstrap between QuerySeq and each baseline.

Comparing two methods by whether their separate confidence intervals overlap
is the wrong test and is much too conservative: both methods are scored on the
same patients, so the patient draw is shared and should cancel.  Each replicate
here resamples patients once, scores every method on that same resample, and
takes the difference, so the interval is about the methods rather than about who
landed in the test split.

Macro AUROC is recomputed inside each replicate rather than averaging
per-finding intervals, which keeps the correlation between findings intact.

The seeds are aggregated INSIDE each replicate: one patient resample per
replicate is shared by every method, seed and condition, the paired
seed-mean difference is computed within that resample, and the interval is
the percentile band of that seed-mean distribution.  An earlier version
computed a CI per seed and averaged the interval endpoints (and averaged
endpoints again across availability patterns); quantile endpoints do not
average into the quantiles of the mean, and a reviewer correctly rejected
those intervals.

Reports, for each comparison, the mean difference, a 95% interval, and the
fraction of replicates favouring QuerySeq.  The last of those is a bootstrap
sign probability, not a p-value, and is labelled that way.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (NSEQ, Cohort, all_subsets, boot_macro_delta,  # noqa: E402
                    finding_filter, subset_name)
from common.text import finding_text_emb  # noqa: E402
from query_diagnostics import load_ckpt  # noqa: E402
from run_methods import predict  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=str, default="queryseq")
    ap.add_argument("--against", type=str,
                    default="uniform,patient,finding,label_id")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out-tag", type=str, default=None,
                    help="suffix for output CSVs (default: same as --tag); "
                         "lets the nqh checkpoints (_k5nqh) write the "
                         "established _nqh_k5 result names")
    args = ap.parse_args()
    if args.out_tag is None:
        args.out_tag = args.tag

    device = "cuda" if torch.cuda.is_available() else "cpu"
    te = Cohort("test", complete_case=True)
    keep = finding_filter(te.Y, te.findings, min_pos=args.min_pos)
    Q = finding_text_emb("sentence", findings=te.findings)
    subs = all_subsets()
    full = subset_name(tuple([1] * NSEQ))
    others = args.against.split(",")
    rng = np.random.default_rng(0)

    # cache every method's scores under every condition, once
    S = {}
    for mode in [args.main] + others:
        for seed in range(args.seeds):
            m = load_ckpt(mode, seed, args.tag, len(te.findings),
                          Q.shape[1], device)
            if m is None:
                print(f"missing checkpoint {mode} s{seed}")
                continue
            for sm in subs:
                s, _ = predict(m, te, Q, device, avail=list(sm))
                S[(mode, seed, "avail", subset_name(sm))] = s
            for k in range(1, NSEQ + 1):
                s, _ = predict(m, te, Q, device, topk=k)
                S[(mode, seed, "budget", k)] = s

    from joblib import Parallel, delayed
    from common import safe_auroc

    conds = ([("avail", subset_name(sm)) for sm in subs]
             + [("budget", k) for k in range(1, NSEQ + 1)])
    avail_conds = [c for c in conds if c[0] == "avail"]
    pats = np.unique(te.patients)
    by_pat = {p_: np.where(te.patients == p_)[0] for p_ in pats}
    Y = te.Y

    def seedmean_delta(b, axis, cond, rows_):
        vals = []
        for seed in range(args.seeds):
            ka, kb = (args.main, seed, axis, cond), (b, seed, axis, cond)
            if ka not in S or kb not in S:
                continue
            da = [safe_auroc(Y[rows_, j], S[ka][rows_, j])
                  - safe_auroc(Y[rows_, j], S[kb][rows_, j]) for j in keep]
            da = [x for x in da if np.isfinite(x)]
            if da:
                vals.append(float(np.mean(da)))
        return float(np.mean(vals)) if vals else np.nan

    def one_rep(r):
        rg = np.random.default_rng(90_000 + r)
        rows_ = np.concatenate(
            [by_pat[p_] for p_ in rg.choice(pats, len(pats), replace=True)])
        out = {}
        for b in others:
            for axis, cond in conds:
                out[(b, axis, cond)] = seedmean_delta(b, axis, cond, rows_)
            out[(b, "avail_mean", "mean_over_15_subsets")] = float(np.nanmean(
                [out[(b, "avail", c)] for _, c in avail_conds]))
        return out

    full_rows = np.arange(len(te))
    reps = Parallel(n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
                    verbose=1)(delayed(one_rep)(r) for r in range(args.boot))

    rows = []
    all_conds = conds + [("avail_mean", "mean_over_15_subsets")]
    for b in others:
        for axis, cond in all_conds:
            if axis == "avail_mean":
                point = float(np.nanmean(
                    [seedmean_delta(b, "avail", c, full_rows)
                     for _, c in avail_conds]))
            else:
                point = seedmean_delta(b, axis, cond, full_rows)
            bs = np.array([rp[(b, axis, cond)] for rp in reps])
            bs = bs[np.isfinite(bs)]
            if not len(bs) or not np.isfinite(point):
                continue
            rows.append({"a": args.main, "b": b, "axis": axis,
                         "condition": cond, "n_seeds": args.seeds,
                         "delta": point,
                         "ci_lo": float(np.percentile(bs, 2.5)),
                         "ci_hi": float(np.percentile(bs, 97.5)),
                         "p_sign": float((bs > 0).mean()),
                         "n_reps": len(bs)})

    d = pd.DataFrame(rows)
    # headline rows: reading everything, and the tightest budget
    head = d[((d.axis == "avail") & (d.condition == full))
             | ((d.axis == "budget") & (d.condition == 2))].copy()
    head["condition"] = head.condition.astype(str)
    d[d.axis != "avail_mean"].to_csv(
        f"{RES}/method_deltas_all{args.out_tag}.csv", index=False)
    head.to_csv(f"{RES}/method_deltas{args.out_tag}.csv", index=False)

    # the average over availability patterns is the missing-input claim,
    # aggregated inside each replicate rather than across interval endpoints
    avg = d[d.axis == "avail_mean"].copy()
    avg.to_csv(f"{RES}/method_deltas_avgsubset{args.out_tag}.csv", index=False)

    print(f"\n=== {args.main} minus baseline, paired patient bootstrap "
          f"({args.boot} reps, {args.seeds} seeds) ===")
    if not d.empty:
        print("\nreading all four inputs:")
        print(d[(d.axis == "avail") & (d.condition == full)][
            ["b", "delta", "ci_lo", "ci_hi", "p_sign"]].round(4)
            .to_string(index=False))
        print("\nmean over all 15 availability patterns:")
        print(avg[["b", "delta", "ci_lo", "ci_hi", "p_sign"]].round(4)
              .to_string(index=False))
        print("\nby budget:")
        print(d[d.axis == "budget"].pivot_table(
            index="b", columns="condition", values="delta").round(4)
            .to_string())


if __name__ == "__main__":
    main()
