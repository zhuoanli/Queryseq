"""Which pathway does the query actually act through?

Shuffling the query costs a lot of accuracy while leaving the routing almost
unchanged.  That is suggestive but not decisive, because a single shuffle
perturbs the gate and the readout at once.  The query enters the model twice --
once to compute the routing gate, once as input to the prediction head -- so the
two can be cut independently:

    gate query   readout query   what it isolates
    correct      correct         reference
    shuffled     correct         the gate pathway alone
    correct      shuffled        the readout pathway alone
    shuffled     shuffled        both

If accuracy survives the first perturbation and collapses under the second, the
query is functioning as an identifier for the prediction head rather than as an
evidence-selection signal, and "query-conditioned routing" is a description of
the architecture rather than of its behaviour.

Both pathways are cut at inference on the same trained model, so nothing here
depends on a retraining difference.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (Cohort, finding_filter, macro, safe_auprc,  # noqa: E402
                    safe_auroc)
from common.text import finding_text_emb  # noqa: E402
from query_diagnostics import load_ckpt, routing_matrix, agreement  # noqa: E402
from run_methods import to_t  # noqa: E402

from common import NSEQ  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


@torch.no_grad()
def predict_split(model, co, Q_gate, Q_head, device, bs=512):
    """Forward pass with different query embeddings on the two pathways."""
    model.eval()
    Qg, Qh = to_t(Q_gate, device), to_t(Q_head, device)
    S, G = [], []
    for i in range(0, len(co.P), bs):
        v = to_t(co.P[i:i + bs], device)
        a = to_t(co.M[i:i + bs], device)
        g, H = model.gates(v, Qg, a)
        S.append(model.fuse_predict(g, H, Qh).cpu().numpy())
        G.append(g.cpu().numpy())
    return np.concatenate(S), np.concatenate(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="queryseq")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--ckpt-tag", type=str, default=None,
                    help="checkpoint suffix; defaults to --tag. Kept separate "
                         "so an output name can follow the results/ tag "
                         "convention while loading differently-named ckpts")
    args = ap.parse_args()
    ckpt_tag = args.ckpt_tag if args.ckpt_tag is not None else args.tag

    device = "cuda" if torch.cuda.is_available() else "cpu"
    te = Cohort("test", complete_case=True)
    keep = finding_filter(te.Y, te.findings, min_pos=args.min_pos)
    Q = finding_text_emb("sentence", findings=te.findings)
    nF = len(te.findings)
    print(f"test={len(te)} findings={len(keep)} device={device}", flush=True)

    rows = []
    for seed in range(args.seeds):
        model = load_ckpt(args.mode, seed, ckpt_tag, nF, Q.shape[1], device)
        if model is None:
            print(f"no checkpoint for seed {seed}")
            continue
        # A checkpoint trained with --no-query-head has a per-finding readout
        # that never consumes the query, so the "readout shuffled" arm is a
        # no-op by construction and all four cells coincide.  That is a
        # property of the wiring, not a finding, and it looked like a finding
        # once already -- so refuse the run rather than write the table.
        if not getattr(model, "query_head", True):
            raise SystemExit(
                f"ckpt_{args.mode}_s{seed}{ckpt_tag}.pt was trained with "
                "query_head=False: its readout does not consume the query, so "
                "the 2x2 intervention is undefined on it. Re-run "
                "run_methods.py without --no-query-head.")
        rng = np.random.default_rng(2000 + seed)
        Qs = Q[rng.permutation(nF)]
        base_R = None
        for gate_q, head_q, name in (
                (Q, Q, "both correct"),
                (Qs, Q, "gate shuffled"),
                (Q, Qs, "readout shuffled"),
                (Qs, Qs, "both shuffled")):
            s, g = predict_split(model, te, gate_q, head_q, device)
            au = [safe_auroc(te.Y[:, j], s[:, j]) for j in keep]
            ap_ = [safe_auprc(te.Y[:, j], s[:, j]) for j in keep]
            R = routing_matrix(g)
            if name == "both correct":
                base_R = R
            rows.append({
                "seed": seed, "condition": name,
                "macro_auroc": macro(au), "macro_auprc": macro(ap_),
                "worst_auroc": float(np.nanmin(au)),
                "routing_cos_vs_base": float(np.mean(agreement(base_R, R))),
                # provenance travels with the numbers: verify_numbers.py
                # refuses a manuscript table whose query_head is not True,
                # because on such a checkpoint the readout arm is a no-op
                "query_head": bool(getattr(model, "query_head", True)),
                "ckpt_tag": ckpt_tag or "(none)",
            })
            print(f"  s{seed} {name:18s} AUROC {macro(au):.4f} "
                  f"AUPRC {macro(ap_):.4f} routing-cos "
                  f"{np.mean(agreement(base_R, R)):.4f}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/query_pathways{args.tag}.csv", index=False)
    if not d.empty:
        g = d.groupby("condition")[["macro_auroc", "macro_auprc",
                                    "routing_cos_vs_base"]].agg(["mean", "std"])
        print("\n=== 2x2 query intervention ===")
        print(g.round(4).to_string())
        base = d[d.condition == "both correct"].macro_auroc.mean()
        for c in ("gate shuffled", "readout shuffled", "both shuffled"):
            v = d[d.condition == c].macro_auroc.mean()
            print(f"  {c:18s} drop = {base - v:+.4f}")


if __name__ == "__main__":
    main()
