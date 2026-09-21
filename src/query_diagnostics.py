"""Stage 3: does the router follow the *meaning* of the query?

With a fixed label space of 37 findings, a router conditioned on text could be
doing nothing that a lookup table indexed by finding could not, and calling that
"language-conditioned" would be packaging rather than a contribution.  These
tests are designed to fail loudly if that is what is happening.

All of them swap the query at inference on an already-trained model, so what is
measured is the effect of the query itself rather than of a different training
run:

  paraphrase     five meaning-preserving phrasings.  Routing should barely move
                 and AUROC should hold.
  shuffled       the query->finding map is permuted.  If routing and accuracy
                 are unaffected, the query was being ignored and the text claim
                 is dead.
  counterfactual each finding is asked with a *specific* other finding's query,
                 which isolates "wrong meaning" from "scrambled input".

Held-out-finding transfer is the strong version and lives in run_methods.py
(--holdout-findings), because it needs the finding removed from training.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, SEQUENCES, SEQ_SHORT, Cohort,  # noqa: E402
                    finding_filter, macro, safe_auroc)
from common.text import PARAPHRASE_FAMILIES, finding_text_emb  # noqa: E402
from queryseq_model import QuerySeq  # noqa: E402
from run_methods import predict  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def load_ckpt(mode, seed, tag, n_find, dim_txt, device):
    p = f"{CACHE}/ckpt_{mode}_s{seed}{tag}.pt"
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location=device, weights_only=False)
    a = ck["args"]
    m = QuerySeq(dim_txt=dim_txt, d=a["dim"], n_seq=len(SEQUENCES),
                 n_find=n_find, mode=mode, tau=a["tau"],
                 query_head=a.get("query_head", True)).to(device)
    r = m.load_state_dict(ck["model"], strict=False)
    # The per-finding readout was added after these checkpoints were written and
    # is unused when query_head=True, so its absence is expected.  Anything else
    # missing means the checkpoint does not match this architecture, and a
    # silently half-loaded model would invalidate every number downstream --
    # so allow exactly those keys and fail on the rest.
    allowed = {"out_w.weight", "out_b.weight"}
    unexpected_missing = set(r.missing_keys) - allowed
    if unexpected_missing or r.unexpected_keys:
        raise RuntimeError(
            f"checkpoint {p} does not match the model: "
            f"missing={sorted(unexpected_missing)} "
            f"unexpected={sorted(r.unexpected_keys)}")
    m.eval()
    return m


def routing_matrix(g):
    """(F, S): mean gate each finding places on each input, over studies."""
    return g.mean(0).T


def agreement(A, B):
    """Cosine similarity between two routing matrices, per finding."""
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (a * b).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="queryseq")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-pos", type=int, default=10)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    te = Cohort("test", complete_case=True)
    keep = finding_filter(te.Y, te.findings, min_pos=args.min_pos)
    Qbase = finding_text_emb("sentence", findings=te.findings)
    nF = len(te.findings)

    rows, rrows, gates_by = [], [], {}
    for seed in range(args.seeds):
        model = load_ckpt(args.mode, seed, args.tag, nF, Qbase.shape[1], device)
        if model is None:
            print(f"no checkpoint for seed {seed}; run run_methods.py first")
            continue
        rng = np.random.default_rng(1000 + seed)

        variants = {f"para:{fam}": finding_text_emb(fam, findings=te.findings)
                    for fam in PARAPHRASE_FAMILIES}
        # meaning-breaking controls
        perm = rng.permutation(nF)
        variants["shuffled"] = Qbase[perm]
        # counterfactual: every finding asked with one specific wrong finding's
        # query (a fixed derangement), so the swap is systematic not random
        der = (np.arange(nF) + 1 + rng.integers(0, nF - 1)) % nF
        variants["counterfactual"] = Qbase[der]

        base_g = None
        for name, Q in variants.items():
            s, g = predict(model, te, Q, device)
            au = [safe_auroc(te.Y[:, j], s[:, j]) for j in keep]
            R = routing_matrix(g)
            if name == "para:sentence":
                base_g = R
            rows.append({"seed": seed, "variant": name,
                         "macro_auroc": macro(au),
                         "worst_auroc": float(np.nanmin(au)),
                         "mean_gate": float(g.mean()),
                         "routing_entropy": float(_entropy(R).mean())})
            gates_by[(seed, name)] = R
        for name, R in list(gates_by.items()):
            if name[0] != seed or base_g is None:
                continue
            ag = agreement(base_g, R)
            rrows.append({"seed": seed, "variant": name[1],
                          "routing_cos_vs_base": float(np.mean(ag)),
                          "routing_cos_min": float(np.min(ag))})

    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/query_diagnostics{args.tag}.csv", index=False)
    pd.DataFrame(rrows).to_csv(f"{RES}/query_routing_agreement{args.tag}.csv",
                               index=False)

    # cross-seed routing stability: is the atlas a property of the data or of
    # the initialisation?
    srows = []
    for a in range(args.seeds):
        for b in range(a + 1, args.seeds):
            ka, kb = (a, "para:sentence"), (b, "para:sentence")
            if ka in gates_by and kb in gates_by:
                srows.append({"seed_a": a, "seed_b": b,
                              "routing_cos": float(
                                  np.mean(agreement(gates_by[ka],
                                                    gates_by[kb])))})
    pd.DataFrame(srows).to_csv(f"{RES}/routing_seed_stability{args.tag}.csv",
                               index=False)

    if not df.empty:
        print("\n--- query variants (mean over seeds) ---")
        print(df.groupby("variant")[["macro_auroc", "worst_auroc",
                                     "mean_gate"]].mean().round(4).to_string())
        print("\n--- routing agreement vs canonical query ---")
        print(pd.DataFrame(rrows).groupby("variant")
              .routing_cos_vs_base.mean().round(4).to_string())
        if srows:
            print(f"\ncross-seed routing cosine: "
                  f"{np.mean([r['routing_cos'] for r in srows]):.4f}")


def _entropy(R):
    p = R / (R.sum(1, keepdims=True) + 1e-9)
    return -(p * np.log(p + 1e-9)).sum(1)


if __name__ == "__main__":
    main()
