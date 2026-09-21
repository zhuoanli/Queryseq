"""Stage 4: does routing survive a change of scanner?

The official split is random over patients, so it tests generalisation to new
patients but not to new equipment.  MR-RATE carries Manufacturer, field strength
and StationName for every study, which makes a genuine acquisition shift
available without simulating anything.

Two shifts are run:

  vendor   train on Siemens+Philips, test on GE
  field    train on 1.5T, test on 3.0T

These are **custom re-splits, not the official one**, and are labelled that way
wherever they appear.  Because the held-out group is excluded from training
entirely, every study in it is unseen regardless of which official split it came
from, so the external test set is much larger than the official test split and
per-finding intervals are correspondingly tighter.  Patients appearing in the
training group are dropped from the test group, so a patient scanned on both
sides of the shift cannot contaminate it.

The claim being tested is not merely that accuracy drops -- it will -- but
whether query-conditioned routing degrades more gracefully than fixed policies,
which is what a deployment argument would actually rest on.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (NSEQ, Cohort, all_subsets, finding_filter,  # noqa: E402
                    load_scanner_meta, macro, safe_auroc, subset_name, worst)
from common.text import finding_text_emb  # noqa: E402
from run_methods import predict, score_rows, train_model  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


class Sub:
    """A cohort restricted to a row subset, exposing the same interface."""

    def __init__(self, src, idx):
        self.P, self.M, self.Y = src.P[idx], src.M[idx], src.Y[idx]
        self.uids, self.patients = src.uids[idx], src.patients[idx]
        self.findings, self.split = src.findings, src.split

    def __len__(self):
        return len(self.uids)


def pooled_cohort():
    """All three official splits concatenated, complete-case only."""
    parts = [Cohort(s, complete_case=True) for s in ("train", "val", "test")]
    base = parts[0]
    base.P = np.concatenate([p.P for p in parts])
    base.M = np.concatenate([p.M for p in parts])
    base.Y = np.concatenate([p.Y for p in parts])
    base.uids = np.concatenate([p.uids for p in parts])
    base.patients = np.concatenate([p.patients for p in parts])
    return base


def build_shift(kind, min_heldout=100):
    """(train, val, test, held_out_label) for one acquisition shift.

    The held-out group is the largest *minority* group in the cohort actually
    being analysed, not a fixed vendor name.  Hardcoding "GE" was correct for
    the four-sequence cohort and silently degenerate for the five-sequence one:
    requiring susceptibility-weighted imaging leaves a single GE study, because
    in this corpus GE sites rarely acquire it.  That is the availability-site
    association this paper documents, showing up in our own experiment design,
    so the rule is stated rather than the vendor pinned.  Under-sized groups
    raise instead of returning a cohort whose every metric is NaN.
    """
    co = pooled_cohort()
    meta = load_scanner_meta().set_index("study_uid")
    m = meta.reindex(co.uids)
    if kind == "vendor":
        grp = m.vendor.fillna("Other").to_numpy().astype(str)
    else:
        grp = np.array([f"{v:.1f}" if np.isfinite(v) else "nan"
                        for v in pd.to_numeric(m.field_t, errors="coerce")])
    counts = pd.Series(grp).value_counts()
    counts = counts[~counts.index.isin(["Other", "nan", "NA"])]
    if len(counts) < 2:
        raise SystemExit(f"{kind}: fewer than two groups present -- "
                         f"no shift to hold out ({dict(counts)})")
    held = str(counts.index[1])              # largest group that is not the majority
    print(f"  {kind}: group sizes {dict(counts)} -> holding out '{held}' "
          f"(n={int(counts.iloc[1])})", flush=True)
    if counts.iloc[1] < min_heldout:
        raise SystemExit(
            f"{kind}: largest minority group '{held}' has only "
            f"{int(counts.iloc[1])} studies (< {min_heldout}). A held-out-"
            f"{kind} test is not defined on this cohort; report its absence "
            f"rather than a table of NaNs.")
    is_held = grp == held
    src_i = np.where(~is_held)[0]
    train_pats = set(co.patients[src_i])
    test_i = np.where(is_held & ~np.isin(co.patients, list(train_pats)))[0]
    rng = np.random.default_rng(0)
    pats = np.unique(co.patients[src_i])
    rng.shuffle(pats)
    vp = set(pats[:max(1, len(pats) // 12)])
    val_i = np.array([i for i in src_i if co.patients[i] in vp], dtype=int)
    train_i = np.array([i for i in src_i if co.patients[i] not in vp],
                       dtype=int)
    return Sub(co, train_i), Sub(co, val_i), Sub(co, test_i), held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=str, default="vendor,field")
    ap.add_argument("--modes", type=str,
                    default="uniform,patient,finding,label_id,queryseq")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--val-min-pos", type=int, default=10)
    ap.add_argument("--min-train-pos", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--query-family", type=str, default="sentence")
    # independent per-finding readouts, matching the main-text nqh
    # comparison; the earlier run silently inherited train_model's
    # query_head=True default while the table provenance said nqh
    ap.add_argument("--no-query-head", dest="query_head",
                    action="store_false", default=False)
    ap.add_argument("--query-head", dest="query_head", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subs = all_subsets()
    full_name = subset_name(tuple([1] * NSEQ))
    pf, mac = [], []
    for kind in args.shift.split(","):
        tr, va, te, held = build_shift(kind)
        Q = finding_text_emb(args.query_family, findings=tr.findings)
        keep = finding_filter(te.Y, te.findings, min_pos=args.min_pos)
        print(f"\n=== shift={kind} held-out={held} | train={len(tr)} "
              f"val={len(va)} test={len(te)} findings={len(keep)} ===",
              flush=True)
        Sfull = {}
        for seed in range(args.seeds):
            for mode in args.modes.split(","):
                model, vb = train_model(mode, tr, va, Q, args, seed, device)
                per_sub = {}
                for m in subs:
                    s, _ = predict(model, te, Q, device, avail=list(m))
                    if subset_name(m) == full_name:
                        Sfull[(mode, seed)] = np.asarray(s, np.float32)
                    nm = subset_name(m)
                    r = score_rows(te.Y, s, keep, te.findings, method=mode,
                                   seed=seed, shift=kind, held_out=held,
                                   axis="availability", subset=nm,
                                   k=int(sum(m)))
                    pf += r
                    per_sub[nm] = macro([x["auroc"] for x in r])
                vals = list(per_sub.values())
                mac.append(dict(shift=kind, held_out=held, method=mode,
                                seed=seed, axis="availability", n_test=len(te),
                                macro_full=per_sub[full_name],
                                macro_avg_subset=float(np.mean(vals)),
                                macro_worst_subset=float(np.min(vals))))
                for k in range(1, NSEQ + 1):
                    s, _ = predict(model, te, Q, device, topk=k)
                    r = score_rows(te.Y, s, keep, te.findings, method=mode,
                                   seed=seed, shift=kind, held_out=held,
                                   axis="budget", subset="allinputs", k=k)
                    pf += r
                    mac.append(dict(shift=kind, held_out=held, method=mode,
                                    seed=seed, axis="budget", budget=k,
                                    n_test=len(te),
                                    macro_auroc=macro([x["auroc"] for x in r]),
                                    macro_auprc=macro([x["auprc"] for x in r]),
                                    worst_finding=worst([x["auroc"]
                                                         for x in r])))
                print(f"  {kind} {mode} s{seed} val={vb:.4f} "
                      f"full={per_sub[full_name]:.4f}", flush=True)

        # paired patient bootstrap at full inputs, seeds aggregated inside
        # each replicate -- the main text's "well inside the paired interval"
        # sentence previously had no computed interval behind it
        base_modes = [m for m in args.modes.split(",") if m != "queryseq"]
        if ("queryseq", 0) in Sfull and base_modes:
            np.savez_compressed(
                f"{RES}/external_scores_{kind}{args.tag}.npz",
                y=te.Y.astype(np.int8), keep=np.array(keep),
                patients=te.patients,
                **{f"{mo}_s{sd}": Sfull[(mo, sd)]
                   for (mo, sd) in Sfull})
            pats_u = np.unique(te.patients)
            by_pat = {pp: np.where(te.patients == pp)[0] for pp in pats_u}

            def smd(bmode, rows_):
                vals = []
                for sd in range(args.seeds):
                    if ("queryseq", sd) not in Sfull or (bmode, sd) not in Sfull:
                        continue
                    da = [safe_auroc(te.Y[rows_, j],
                                     Sfull[("queryseq", sd)][rows_, j])
                          - safe_auroc(te.Y[rows_, j],
                                       Sfull[(bmode, sd)][rows_, j])
                          for j in keep]
                    da = [x for x in da if np.isfinite(x)]
                    if da:
                        vals.append(float(np.mean(da)))
                return float(np.mean(vals)) if vals else np.nan

            ext_rows = []
            full_rows_ = np.arange(len(te))
            for bmode in base_modes:
                bs = []
                for rrep in range(1000):
                    rg = np.random.default_rng(80_000 + rrep)
                    rows_ = np.concatenate(
                        [by_pat[pp] for pp in
                         rg.choice(pats_u, len(pats_u), replace=True)])
                    bs.append(smd(bmode, rows_))
                bs = np.array([x for x in bs if np.isfinite(x)])
                ext_rows.append({
                    "shift": kind, "held_out": held, "a": "queryseq",
                    "b": bmode, "delta": smd(bmode, full_rows_),
                    "ci_lo": float(np.percentile(bs, 2.5)),
                    "ci_hi": float(np.percentile(bs, 97.5)),
                    "p_sign": float((bs > 0).mean()), "n_reps": len(bs)})
            ed = pd.DataFrame(ext_rows)
            hdr = f"{RES}/external_deltas{args.tag}.csv"
            # replace this shift's rows rather than append, so reruns are
            # idempotent while single-shift invocations still accumulate
            if os.path.exists(hdr):
                old = pd.read_csv(hdr)
                ed = pd.concat([old[old["shift"] != kind], ed],
                               ignore_index=True)
            ed.to_csv(hdr, index=False)
            print(ed.round(4).to_string(index=False), flush=True)

    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(pf).to_csv(f"{RES}/external_perfinding{args.tag}.csv",
                            index=False)
    d = pd.DataFrame(mac)
    d.to_csv(f"{RES}/external_macro{args.tag}.csv", index=False)
    print("\n--- external, availability axis ---")
    print(d[d.axis == "availability"].groupby(["shift", "method"])[
        ["macro_full", "macro_avg_subset", "macro_worst_subset"]]
        .mean().round(4).to_string())
    print("\n--- external, budget axis (macro AUROC by k) ---")
    print(d[d.axis == "budget"].pivot_table(
        index=["shift", "method"], columns="budget", values="macro_auroc")
        .round(4).to_string())


if __name__ == "__main__":
    main()
