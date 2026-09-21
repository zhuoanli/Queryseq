"""Stage 2: train every routing method and evaluate it under two protocols.

Two axes are kept strictly separate, because conflating them is the usual way
this kind of result gets overstated:

  AVAILABILITY -- which inputs exist for a study.  Varying this over all 15
      subsets with the model free to read everything available measures
      *missing-input robustness*: full, average-subset and worst-subset AUROC.

  BUDGET -- how many of the available inputs the model is allowed to read.
      Holding availability at all four and forcing a hard top-k selection
      measures *value-of-information*: the performance-budget Pareto front.

A method can look good on one and bad on the other, so both are reported for
every method rather than whichever flatters it.

Selection discipline: the fixed-subset policies (global-best and per-finding
best) are chosen on the *validation* split and then applied unchanged to test.
The oracle policy is chosen on test and is reported only as an upper bound,
labelled as such, so that QuerySeq's regret against it is meaningful.

Writes results/method_perfinding.csv, results/method_macro.csv, the routing
gates, and per-prediction rows for the consolidated table.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, Cohort, all_subsets, finding_filter,  # noqa: E402
                    macro, safe_auprc, safe_auroc, subset_name, worst)
from common.text import finding_text_emb  # noqa: E402
from queryseq_model import (QuerySeq, hard_topk, loss_fn,  # noqa: E402
                            sample_avail)
from common import NSEQ  # noqa: E402  -- 4, 5 or 7 per QS_SEQSET

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def to_t(x, device):
    return torch.from_numpy(np.asarray(x, np.float32)).to(device)


def train_model(mode, tr, va, Q, args, seed, device):
    """Train one routing mode; keep the epoch with the best validation macro."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n_find = tr.Y.shape[1]
    model = QuerySeq(dim_txt=Q.shape[1], d=args.dim, n_seq=NSEQ,
                     n_find=n_find, mode=mode, tau=args.tau,
                     query_head=getattr(args, "query_head", True)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)

    V = to_t(tr.P, device)
    Y = to_t(tr.Y, device)
    Qt = to_t(Q, device)
    # findings with too few training positives cannot be learned; masking them
    # out of the loss keeps them from injecting gradient noise into the router
    npos = tr.Y.sum(0)
    valid = to_t((npos >= args.min_train_pos).astype(np.float32), device)
    pw = torch.clamp(
        to_t((len(tr.Y) - npos) / np.maximum(npos, 1), device), 1.0, 50.0)

    n = len(tr.P)
    steps = max(1, n // args.bs)
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        for si in range(steps):
            b = perm[si * args.bs:(si + 1) * args.bs]
            vb, yb = V[b], Y[b]
            full = torch.ones(len(b), NSEQ, device=device)
            av = sample_avail(len(b), NSEQ, rng, device)
            fixed = av if mode == "random" else (
                torch.ones(NSEQ) if mode == "fixed" else None)
            with torch.no_grad():
                teacher, _ = model(vb, Qt, full)
            logit, g = model(vb, Qt, av, fixed_mask=fixed)
            loss, _ = loss_fn(logit, yb, teacher, g,
                              valid.expand_as(logit), args.alpha, args.beta, pw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = val_macro(model, va, Q, device, args)
            if m > best:
                best = m
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            if args.verbose:
                print(f"    {mode} s{seed} ep{ep + 1} val={m:.4f} "
                      f"best={best:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict(model, co, Q, device, avail=None, topk=None, bs=512):
    """Scores and gates for a cohort under a given availability/budget."""
    model.eval()
    Qt = to_t(Q, device)
    S, G = [], []
    for i in range(0, len(co.P), bs):
        v = to_t(co.P[i:i + bs], device)
        a = (torch.ones(len(v), NSEQ, device=device) if avail is None
             else to_t(np.tile(avail, (len(v), 1)), device))
        a = a * to_t(co.M[i:i + bs], device)
        fixed = a if model.mode == "random" else None
        g, H = model.gates(v, Qt, a, fixed_mask=fixed)
        if topk is not None:
            g = hard_topk(g, a, topk)
        S.append(model.fuse_predict(g, H, Qt).cpu().numpy())
        G.append(g.cpu().numpy())
    return np.concatenate(S), np.concatenate(G)


def val_macro(model, va, Q, device, args):
    """Validation score: mean macro AUROC over a few availability patterns.

    Averaging over patterns rather than scoring only the full input keeps model
    selection from silently preferring a model that is good with everything
    present and brittle without.
    """
    keep = finding_filter(va.Y, va.findings, min_pos=args.val_min_pos)
    if not keep:
        return 0.0
    out = []
    for av in _val_patterns(NSEQ):
        s, _ = predict(model, va, Q, device, avail=av)
        out.append(macro([safe_auroc(va.Y[:, j], s[:, j]) for j in keep]))
    return float(np.mean(out))


def _val_patterns(n):
    """Availability patterns used for model selection, for any input count.

    Everything present, each single input alone, and one leave-one-out -- enough
    to stop selection preferring a model that is strong with all inputs and
    brittle without, while staying cheap enough to run every few epochs.
    """
    pats = [[1] * n]
    pats.append([0 if i == 0 else 1 for i in range(n)])
    for i in range(min(n, 3)):
        pats.append([1 if j == i else 0 for j in range(n)])
    return pats


def score_rows(Y, S, keep, findings, **extra):
    rows = []
    for j in keep:
        rows.append(dict(finding=findings[j], n_pos=int(Y[:, j].sum()),
                         auroc=safe_auroc(Y[:, j], S[:, j]),
                         auprc=safe_auprc(Y[:, j], S[:, j]), **extra))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str,
                    default="uniform,patient,finding,label_id,queryseq")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--no-query-head", dest="query_head",
                    action="store_false",
                    help="per-finding readout instead of a query-conditioned "
                         "head, so routing is the only path for the query")
    ap.add_argument("--alpha", type=float, default=1.0, help="consistency")
    ap.add_argument("--beta", type=float, default=0.0, help="input cost")
    ap.add_argument("--min-pos", type=int, default=10, help="test evaluability")
    ap.add_argument("--val-min-pos", type=int, default=10)
    ap.add_argument("--min-train-pos", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--query-family", type=str, default="sentence")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tr = Cohort("train", complete_case=True)
    va = Cohort("val", complete_case=True)
    te = Cohort("test", complete_case=True)
    Q = finding_text_emb(args.query_family, findings=tr.findings)
    keep = finding_filter(te.Y, te.findings, min_pos=args.min_pos)
    subs = all_subsets()
    print(f"device={device} train={len(tr)} val={len(va)} test={len(te)} "
          f"findings={len(keep)} query={args.query_family}", flush=True)

    pf, mac, preds = [], [], []
    for seed in range(args.seeds):
        for mode in args.modes.split(","):
            model, vbest = train_model(mode, tr, va, Q, args, seed, device)
            print(f"  [{time.time() - t0:6.0f}s] {mode} s{seed} "
                  f"val={vbest:.4f}", flush=True)

            # ---- availability axis: all 15 subsets, read everything present
            per_sub = {}
            for m in subs:
                s, g = predict(model, te, Q, device, avail=list(m))
                nm = subset_name(m)
                r = score_rows(te.Y, s, keep, te.findings, method=mode,
                               seed=seed, axis="availability", subset=nm,
                               k=int(sum(m)), budget=-1)
                pf += r
                per_sub[nm] = macro([x["auroc"] for x in r])
                if seed == 0:
                    # built columnwise: the row-at-a-time version allocated
                    # ~1.7M Python tuples per run and dominated the runtime
                    nk, nt = len(keep), len(te)
                    preds.append(pd.DataFrame({
                        "patient_id": np.tile(te.patients, nk),
                        "study_uid": np.tile(te.uids, nk),
                        "seed": seed,
                        "finding": np.repeat([te.findings[j] for j in keep], nt),
                        "available_subset": nm, "selected_subset": nm,
                        "budget": -1,
                        "label": te.Y[:, keep].T.reshape(-1),
                        "prediction": s[:, keep].T.reshape(-1).astype(np.float32),
                        "method": mode,
                    }))
            mac.append(dict(method=mode, seed=seed, axis="availability",
                            macro_full=per_sub[subset_name(tuple([1] * NSEQ))],
                            macro_avg_subset=float(np.mean(list(per_sub.values()))),
                            macro_worst_subset=float(np.min(list(per_sub.values()))),
                            val_best=vbest))

            # ---- budget axis: everything available, read only k
            for k in range(1, NSEQ + 1):
                s, g = predict(model, te, Q, device, topk=k)
                r = score_rows(te.Y, s, keep, te.findings, method=mode,
                               seed=seed, axis="budget", subset="allinputs",
                               k=k, budget=k)
                pf += r
                mac.append(dict(method=mode, seed=seed, axis="budget",
                                budget=k,
                                macro_auroc=macro([x["auroc"] for x in r]),
                                macro_auprc=macro([x["auprc"] for x in r]),
                                worst_finding=worst([x["auroc"] for x in r])))
            # routing gates at full availability, for the atlas and the
            # counterfactual-query figures
            _, gfull = predict(model, te, Q, device)
            np.save(f"{CACHE}/gates_{mode}_s{seed}{args.tag}.npy",
                    gfull.astype(np.float16))
            # the query diagnostics swap Q at inference on this exact model, so
            # they measure the effect of the query rather than of retraining
            torch.save({"model": model.state_dict(), "mode": mode,
                        "seed": seed, "args": vars(args)},
                       f"{CACHE}/ckpt_{mode}_s{seed}{args.tag}.pt")

    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(pf).to_csv(f"{RES}/method_perfinding{args.tag}.csv",
                            index=False)
    pd.DataFrame(mac).to_csv(f"{RES}/method_macro{args.tag}.csv", index=False)
    if preds:
        meta = te.meta()[["vendor", "field_t", "station"]]
        p = pd.concat(preds, ignore_index=True)
        key = dict(zip(te.uids, zip(meta.vendor, meta.field_t, meta.station)))
        p[["vendor", "field_t", "station"]] = pd.DataFrame(
            [key.get(u, (None, None, None)) for u in p.study_uid],
            index=p.index)
        p.to_csv(f"{RES}/predictions{args.tag}.csv.gz", index=False,
                 compression="gzip")

    d = pd.DataFrame(mac)
    print("\n--- availability axis ---")
    print(d[d.axis == "availability"].groupby("method")[
        ["macro_full", "macro_avg_subset", "macro_worst_subset"]]
        .mean().round(4).to_string())
    print("\n--- budget axis (macro AUROC by k) ---")
    print(d[d.axis == "budget"].pivot_table(
        index="method", columns="budget", values="macro_auroc")
        .round(4).to_string())
    print(f"\ntotal {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
