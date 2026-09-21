"""GoAT burden lattice and the pre-registered B1/B2/B3 test.

Implements the amendment recorded 2026-08-10T18:21:04Z. The original
presence/absence targets are degenerate on this cohort (ET 97.3% positive,
SNFH 97.9%), so the outcome is re-operationalised as high vs low lesion burden.
The directional hypotheses are unchanged:

    B1  t1c has the largest exact Shapley value for ET
    B2  t2f has the largest for SNFH
    B3  t1c is not largest for SNFH, and t2f is not largest for ET

Two rules from the amendment that the code enforces rather than assumes:

  * the median is computed on the **training partition only** and frozen. Taking
    it over all 1,351 cases would let the evaluation outcome distribution define
    the target, which is the same class of error as selecting and scoring on the
    same predictions.
  * ties go low: strictly `> median` is high burden. Zero-volume cases fall into
    low burden with no special handling.

Both the degenerate presence targets and the burden targets are evaluated and
written, so the failed original analysis is reported rather than replaced.
"""
import argparse
import json
import os
import sys
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import macro, patient_folds, safe_auroc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, DL = f"{ROOT}/results", f"{ROOT}/data_local"
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
SHORT = {"t1n": "T1n", "t1c": "T1c", "t2w": "T2w", "t2f": "T2f"}
TARGETS = ["ET", "SNFH", "NETC"]     # RC is empty, TC/WT are >99% -- excluded


def load():
    lab = pd.read_csv(f"{DL}/goat_labels.csv")
    P, keep = [], []
    for _, r in lab.iterrows():
        f = f"{DL}/goat_features/{r['case']}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        if not all(f"pooled_{m}" in z.files for m in MODALITIES):
            continue
        P.append(np.stack([z[f"pooled_{m}"] for m in MODALITIES]))
        keep.append(r)
    return np.asarray(P, np.float32), pd.DataFrame(keep).reset_index(drop=True)


def all_subsets(n):
    out = [tuple((b >> j) & 1 for j in range(n)) for b in range(1, 1 << n)]
    return sorted(out, key=lambda m: (sum(m), m))


def cell(P, Y, tr, te, mask, C=0.1):
    w = np.asarray(mask, np.float32)[None, :]
    X = (P * w[:, :, None]).sum(1) / max(sum(mask), 1)
    sc = StandardScaler().fit(X[tr])
    A, B = sc.transform(X[tr]), sc.transform(X[te])
    out = np.full((len(te), Y.shape[1]), np.nan, np.float32)
    for j in range(Y.shape[1]):
        y = Y[tr, j]
        if y.sum() < 3 or y.sum() == len(y):
            continue
        out[:, j] = LogisticRegression(
            max_iter=2000, C=C, class_weight="balanced").fit(A, y).decision_function(B)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()

    P, lab = load()
    n = len(lab)
    subs = all_subsets(len(MODALITIES))
    print(f"{n} cases, {len(MODALITIES)} modalities, {len(subs)} subsets",
          flush=True)

    # patient == case here; GoAT ships one study per subject
    groups = lab["case"].to_numpy()

    # ---- targets -------------------------------------------------------
    # presence: the original, degenerate, retained for reporting
    Ypres = lab[TARGETS].to_numpy().astype(int)
    # burden: median frozen on the training partition of fold 0, seed 0
    folds0 = patient_folds(groups, k=args.folds, seed=0)
    train_idx = folds0[0][0]
    thr = {t: float(np.median(lab[f"n_{t}"].to_numpy()[train_idx]))
           for t in TARGETS}
    Yburd = np.stack([(lab[f"n_{t}"].to_numpy() > thr[t]).astype(int)
                      for t in TARGETS], 1)
    print("burden thresholds (train partition median, frozen):", flush=True)
    for t in TARGETS:
        print(f"  {t:5s} median={thr[t]:9.0f}  high={Yburd[:, TARGETS.index(t)].sum():5d}"
              f" / {n}  presence={Ypres[:, TARGETS.index(t)].sum():5d}", flush=True)

    # sensitivity: per-(seed, fold) isolation.  An earlier version built one
    # label array from seed 0's folds and reused it for every seed, and a
    # training case could be labelled by another fold's median -- a median
    # whose computation saw the current fold's evaluation volumes (a reviewer
    # demonstrated this with a controlled perturbation).  Now, for each seed
    # and each fold, the threshold is the median of that fold's own training
    # partition, and the SAME threshold labels both that fold's training rows
    # and its evaluation rows; no evaluation volume can reach the threshold
    # that labels any training row of the model scored on it.
    vols = np.stack([lab[f"n_{t}"].to_numpy() for t in TARGETS], 1)

    rows, shap_rows = [], []
    for name in ("presence", "burden", "burden_perfold"):
        U = np.full((args.seeds, len(subs), len(TARGETS)), np.nan)
        for seed in range(args.seeds):
            folds = patient_folds(groups, k=args.folds, seed=seed)
            if name == "burden_perfold":
                Yfold, Y = [], np.zeros((n, len(TARGETS)), int)
                for tr_idx, ev_idx in folds:
                    thr_f = np.median(vols[tr_idx], 0)
                    yf = (vols > thr_f[None, :]).astype(int)
                    Yfold.append(yf)
                    Y[ev_idx] = yf[ev_idx]   # pooled eval labels, this seed
            else:
                Yfold = None
                Y = Ypres if name == "presence" else Yburd
            cells = [(si, fi) for si in range(len(subs))
                     for fi in range(len(folds))]
            res = Parallel(n_jobs=args.jobs)(
                delayed(cell)(P, Yfold[fi] if Yfold is not None else Y,
                              folds[fi][0], folds[fi][1], subs[si])
                for si, fi in cells)
            S = [np.full((n, len(TARGETS)), np.nan, np.float32)
                 for _ in subs]
            for (si, fi), r in zip(cells, res):
                S[si][folds[fi][1]] = r
            # per-case OOF dump so rank stability can be bootstrapped at
            # patient level -- a reviewer correctly noted the shipped
            # aggregates cannot support any uncertainty statement
            np.savez_compressed(
                f"{RES}/goat_scores_{name}_seed{seed}.npz",
                scores=np.stack(S).astype(np.float16),
                y=Y.astype(np.int8))
            for si in range(len(subs)):
                for j, t in enumerate(TARGETS):
                    U[seed, si, j] = safe_auroc(Y[:, j], S[si][:, j])
                    rows.append({"target_kind": name, "seed": seed,
                                 "target": t, "k": int(sum(subs[si])),
                                 "subset": "+".join(
                                     SHORT[m] for m, b in zip(MODALITIES, subs[si]) if b),
                                 "n_pos": int(Y[:, j].sum()),
                                 "auroc": U[seed, si, j]})
            print(f"  {name} seed {seed} done", flush=True)
        Um = np.nanmean(U, 0)

        # exact Shapley, same convention as the MRI corpus: U(empty) = chance
        idx = {sum(b << i for i, b in enumerate(m)): i for i, m in enumerate(subs)}
        N = len(MODALITIES)
        w = {s: factorial(s) * factorial(N - s - 1) / factorial(N)
             for s in range(N)}
        for j, t in enumerate(TARGETS):
            for mi, mod in enumerate(MODALITIES):
                tot = 0.0
                for r in range(N):
                    for Ssub in combinations([x for x in range(N) if x != mi], r):
                        b = sum(1 << x for x in Ssub)
                        with_m = Um[idx[b | (1 << mi)], j]
                        without = Um[idx[b], j] if b else 0.5
                        tot += w[r] * (with_m - without)
                shap_rows.append({"target_kind": name, "target": t,
                                  "modality": SHORT[mod], "shapley": tot})

    pd.DataFrame(rows).to_csv(f"{RES}/goat_lattice.csv", index=False)
    sh = pd.DataFrame(shap_rows)
    sh.to_csv(f"{RES}/goat_shapley.csv", index=False)
    with open(f"{RES}/goat_thresholds.json", "w") as fh:
        json.dump({"thresholds_train_median": thr, "n_cases": n,
                   "targets": TARGETS}, fh, indent=2)

    # ---- the pre-registered test ---------------------------------------
    print("\n" + "=" * 68)
    print("PRE-REGISTERED B1/B2/B3 (PLAN.md 2026-08-04, amended 2026-08-10)")
    print("=" * 68)
    for kind in ("burden", "presence"):
        piv = sh[sh.target_kind == kind].pivot_table(
            index="target", columns="modality", values="shapley")
        print(f"\n--- {kind}{'  (PRIMARY, per amendment)' if kind == 'burden' else '  (original, degenerate)'} ---")
        print(piv.round(4).to_string())
        top = {t: piv.loc[t].idxmax() for t in piv.index}
        b1 = top.get("ET") == "T1c"
        b2 = top.get("SNFH") == "T2f"
        b3 = top.get("SNFH") != "T1c" and top.get("ET") != "T2f"
        for t in piv.index:
            print(f"    {t:6s} top = {top[t]}")
        print(f"  B1  t1c largest for ET   -> {'HOLDS' if b1 else 'FAILS'}")
        print(f"  B2  t2f largest for SNFH -> {'HOLDS' if b2 else 'FAILS'}")
        print(f"  B3  discriminant         -> {'HOLDS' if b3 else 'FAILS'}")
        if kind == "burden":
            print(f"\n  VERDICT (primary): "
                  f"{'ALL THREE HOLD' if (b1 and b2 and b3) else 'NOT ALL HOLD -- report as a failed pre-registered replication'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
