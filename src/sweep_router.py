"""Does query conditioning help once the gate can actually express a choice?

The first method run found QuerySeq indistinguishable from a patient-only
router.  Before treating that as evidence of no effect, one implementation
detail has to be ruled out: query embeddings are L2-normalised and the encoder
output is too, so the router logit is bounded in roughly $[-1,1]$; at
$\\tau{=}1$ the sigmoid gate is confined to about $[0.27, 0.73]$ and cannot
express "read this one, skip that one" at all.  A null result from a router that
structurally cannot route says nothing about routing.

This sweeps the gate temperature and the input-cost weight for the two modes
that the paper's central claim compares, selecting on validation only, and
reports test numbers for the selected setting.  A flat surface across tau and
beta is real evidence of no effect; a peak at small tau means the first run was
mis-specified.
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (NSEQ, Cohort, all_subsets, finding_filter,  # noqa: E402
                    macro, safe_auroc, subset_name, worst)
from common.text import finding_text_emb  # noqa: E402
from queryseq_model import budget_of  # noqa: E402
from run_methods import predict, train_model  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


class A:
    """Argument bag matching what train_model expects."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str, default="queryseq,patient")
    ap.add_argument("--taus", type=str, default="0.05,0.1,0.25,1.0")
    ap.add_argument("--betas", type=str, default="0,0.01,0.05")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--no-query-head", dest="query_head",
                    action="store_false")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Cohort("train", complete_case=True)
    va = Cohort("val", complete_case=True)
    te = Cohort("test", complete_case=True)
    Q = finding_text_emb("sentence", findings=tr.findings)
    keep = finding_filter(te.Y, te.findings, min_pos=args.min_pos)
    subs = all_subsets()
    full = subset_name(tuple([1] * NSEQ))
    print(f"device={device} test={len(te)} findings={len(keep)}", flush=True)

    rows, t0 = [], time.time()
    for mode, tau, beta in itertools.product(
            args.modes.split(","), [float(x) for x in args.taus.split(",")],
            [float(x) for x in args.betas.split(",")]):
        for seed in range(args.seeds):
            cfg = A(epochs=args.epochs, bs=256, lr=3e-4, wd=1e-4, dim=256,
                    tau=tau, alpha=1.0, beta=beta, min_train_pos=20,
                    val_min_pos=10, eval_every=10, verbose=False,
                    query_head=args.query_head)
            model, vbest = train_model(mode, tr, va, Q, cfg, seed, device)
            per_sub = {}
            for m in subs:
                s, _ = predict(model, te, Q, device, avail=list(m))
                per_sub[subset_name(m)] = macro(
                    [safe_auroc(te.Y[:, j], s[:, j]) for j in keep])
            r = {"mode": mode, "tau": tau, "beta": beta, "seed": seed,
                 "val": vbest, "macro_full": per_sub[full],
                 "macro_avg_subset": float(np.mean(list(per_sub.values()))),
                 "macro_worst_subset": float(np.min(list(per_sub.values())))}
            for k in (1, 2):
                s, g = predict(model, te, Q, device, topk=k)
                au = [safe_auroc(te.Y[:, j], s[:, j]) for j in keep]
                r[f"macro_k{k}"] = macro(au)
                r[f"worst_k{k}"] = worst(au)
            _, gf = predict(model, te, Q, device)
            # how much the gate actually varies -- a router whose gates are all
            # equal is a mean-pool with extra steps, whatever its name
            r["gate_mean"] = float(gf.mean())
            r["gate_spread_over_seq"] = float(gf.std(1).mean())
            r["gate_spread_over_findings"] = float(gf.mean(0).std(1).mean())
            r["avg_inputs_read"] = float((gf > 0.5).sum(1).mean())
            rows.append(r)
            print(f"  [{time.time() - t0:5.0f}s] {mode} tau={tau} beta={beta} "
                  f"s{seed} val={vbest:.4f} full={r['macro_full']:.4f} "
                  f"k1={r['macro_k1']:.4f} spread={r['gate_spread_over_seq']:.3f}",
                  flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/router_sweep{args.tag}.csv", index=False)
    g = d.groupby(["mode", "tau", "beta"])[
        ["val", "macro_full", "macro_k1", "worst_k1",
         "gate_spread_over_seq", "avg_inputs_read"]].mean()
    print("\n=== sweep (mean over seeds) ===")
    print(g.round(4).to_string())
    print("\n=== best setting per mode, selected on VALIDATION ===")
    for mode in d["mode"].unique():
        s = d[d["mode"] == mode].groupby(["tau", "beta"]).val.mean()
        bt, bb = s.idxmax()
        sel = d[(d["mode"] == mode) & (d.tau == bt) & (d.beta == bb)]
        print(f"  {mode:9s} tau={bt} beta={bb} -> full={sel.macro_full.mean():.4f} "
              f"avg={sel.macro_avg_subset.mean():.4f} "
              f"k1={sel.macro_k1.mean():.4f} "
              f"gatespread={sel.gate_spread_over_seq.mean():.3f}")


if __name__ == "__main__":
    main()
