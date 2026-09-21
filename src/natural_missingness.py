"""Evaluation on studies that are genuinely incomplete, not masked to be.

Everything else in this paper masks complete studies, because that is the only
way to make withholding an input an intervention rather than a change of
population.  But the deployment question is different: a model meets studies
whose sequences are missing for clinical reasons, and those studies are not a
random sample.  Roughly 1.3k of the newly encoded studies are naturally
incomplete, which is enough to ask whether the synthetic-missingness conclusions
survive contact with real missingness.

Nothing here is causal and it is not comparable to the masked results. The
availability pattern is a *consequence* of the clinical question, so a model can
score well partly by exploiting who tends to be missing what; the
availability-only control quantifies exactly that channel. The cohort is
described in full -- per-pattern counts, prevalence, vendor mix -- so the
selection effects are visible rather than averaged away.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (Cohort, finding_filter, macro, safe_auprc,  # noqa: E402
                    safe_auroc, subset_name)
from common.text import finding_text_emb  # noqa: E402
from query_diagnostics import load_ckpt  # noqa: E402
from run_methods import predict  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str,
                    default="uniform,patient,finding,label_id,queryseq")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--tag", type=str, default="_nqh")
    ap.add_argument("--ckpt-tag", type=str, default=None,
                    help="checkpoint suffix; defaults to --tag. Kept separate "
                         "so an output name can follow the results/ tag "
                         "convention while loading differently-named ckpts")
    args = ap.parse_args()
    ckpt_tag = args.ckpt_tag if args.ckpt_tag is not None else args.tag

    device = "cuda" if torch.cuda.is_available() else "cpu"
    te = Cohort("test", complete_case=False)
    inc = te.M.sum(1) < te.M.shape[1]
    Q = finding_text_emb("sentence", findings=te.findings)

    # ---- describe the cohort before scoring anything ----------------------
    pat = ["".join(str(int(x)) for x in r) for r in te.M]
    meta = te.meta()
    desc = []
    for p in sorted(set(pat)):
        s = np.array(pat) == p
        mask = tuple(int(c) for c in p)
        desc.append({
            "pattern": p, "subset": subset_name(mask), "n_inputs": sum(mask),
            "n_studies": int(s.sum()),
            "frac_of_test": float(s.mean()),
            "vendor_top": (meta.vendor[s].mode().iloc[0]
                           if meta.vendor[s].notna().any() else "NA"),
            "mean_prevalence": float(te.Y[s].mean()),
        })
    D = pd.DataFrame(desc).sort_values("n_studies", ascending=False)
    D.to_csv(f"{RES}/natural_cohort{args.tag}.csv", index=False)
    print(f"test split n={len(te)}  complete={int((~inc).sum())}  "
          f"naturally incomplete={int(inc.sum())}")
    print(D.to_string(index=False))

    # ---- score the trained models on real missingness ---------------------
    rows = []
    for group, sel in (("complete", ~inc), ("incomplete", inc),
                       ("all", np.ones(len(te), bool))):
        Y = te.Y[sel]
        keep = finding_filter(Y, te.findings, min_pos=args.min_pos)
        if not keep:
            continue
        sub = Cohort("test", complete_case=False)
        sub.P, sub.M, sub.Y = te.P[sel], te.M[sel], te.Y[sel]
        sub.uids, sub.patients = te.uids[sel], te.patients[sel]
        for mode in args.modes.split(","):
            for seed in range(args.seeds):
                m = load_ckpt(mode, seed, ckpt_tag, len(te.findings),
                              Q.shape[1], device)
                if m is None:
                    continue
                s, _ = predict(m, sub, Q, device)
                au = [safe_auroc(Y[:, j], s[:, j]) for j in keep]
                ap_ = [safe_auprc(Y[:, j], s[:, j]) for j in keep]
                rows.append({"group": group, "method": mode, "seed": seed,
                             "n": int(sel.sum()), "n_findings": len(keep),
                             "macro_auroc": macro(au),
                             "macro_auprc": macro(ap_),
                             "worst_auroc": float(np.nanmin(au))})
    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/natural_missingness{args.tag}.csv", index=False)
    if not d.empty:
        print("\n=== performance under real vs simulated missingness ===")
        print(d.pivot_table(index="method", columns="group",
                            values=["macro_auroc", "macro_auprc"])
              .round(4).to_string())


if __name__ == "__main__":
    main()
