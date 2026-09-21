"""Stage 1 analysis: what the lattice says about how inputs carry information.

Because there are only four inputs and the lattice is exhaustive, the
cooperative-game quantities below are computed *exactly* -- no Monte-Carlo
sampling of permutations, no approximation error to defend.

  marginal    D_{f,m}(S) = U_f(S + m) - U_f(S)
  Shapley     phi_{f,m}  = sum_S w(S) D_{f,m}(S),  w(S) = |S|!(n-|S|-1)!/n!
  synergy     psi_{f,ij}(S) = U_f(S+ij) - U_f(S+i) - U_f(S+j) + U_f(S)

Synergy is the quantity that carries the paper's empirical claim.  If inputs
contributed independently, psi would be zero everywhere; a spread of psi away
from zero is direct evidence that input value is *non-additive*, which is
strictly stronger than "different findings prefer different inputs" and much
harder to explain away as a per-finding accuracy difference.

Reads the out-of-fold score tensors written by subset_lattice.py, so all
intervals are patient-level bootstraps over the same cross-fitted predictions
rather than refits.
"""
import argparse
import math
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE, NSEQ, SEQUENCES, SEQ_SHORT, Cohort,  # noqa: E402
                    all_subsets, boot_auroc, finding_filter, macro,
                    patient_folds, safe_auroc, subset_name)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def bits(mask):
    return sum(b << i for i, b in enumerate(mask))


def load_inner(seed, tag, n_folds):
    """Per-outer-fold (rows, tensor) pairs from inner_oof.py's shard cache."""
    out = []
    for f in range(n_folds):
        z = np.load(f"{CACHE}/inneroof_s{seed}_f{f}{tag}.npz")
        out.append((z["rows"].astype(int), z["inner"].astype(np.float32)))
    return out


def nested_policy(S, Y, folds, keep, scope, cand=None, inner=None):
    """Cross-fitted evaluation of a subset-selection policy.

    Choosing each finding's best subset and scoring it on the same predictions
    is a winner's curse: with 15 candidates and findings carrying as few as 22
    positives, the maximum is biased upward by chance alone, and reporting that
    as the value of routing would overstate it substantially.

    Selection for outer fold f reads ONLY inner-CV predictions computed
    strictly within train(f) -- see inner_oof.py.  An earlier version selected
    on the other folds' single-layer OOF predictions; those were produced by
    heads whose training folds include f, so outer-fold labels reached the
    models underlying the selection criterion.  A reviewer caught this, and it
    is exactly the error nested cross-validation exists to prevent.  Scoring
    pools the standard outer OOF predictions, which were always proper.

    scope='per_finding' selects a subset for each finding; scope='global'
    selects one subset for all findings by macro AUROC.  Returns the pooled
    score matrix and the per-fold selections.
    """
    if inner is None:
        raise RuntimeError(
            "nested_policy now requires the inner-OOF cache (run "
            "inner_oof.py): selecting on single-layer OOF predictions lets "
            "outer-fold labels reach the selection criterion")
    cand = list(range(S.shape[0])) if cand is None else list(cand)
    out = np.full((S.shape[1], S.shape[2]), np.nan, np.float32)
    picks = []
    for f, (sel, ev) in enumerate(folds):
        rows, T = inner[f]          # T: [n_subsets, len(rows), n_findings]
        if scope == "global":
            m = [macro([safe_auroc(Y[rows, j], T[si, :, j]) for j in keep])
                 for si in cand]
            b = cand[int(np.nanargmax(m))]
            for j in keep:
                out[ev, j] = S[b, ev, j]
            picks.append({"fold": f, "finding": "*", "subset_idx": b})
        else:
            for j in keep:
                a = [safe_auroc(Y[rows, j], T[si, :, j]) for si in cand]
                b = cand[int(np.nanargmax(a))]
                out[ev, j] = S[b, ev, j]
                picks.append({"fold": f, "finding_idx": j, "subset_idx": b})
    return out, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5,
                    help="must match the value subset_lattice.py ran with")
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    co = Cohort("train", complete_case=True)
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    subs = all_subsets()
    idx_of = {bits(m): i for i, m in enumerate(subs)}
    rng = np.random.default_rng(0)

    # U[seed, subset, finding] -- the utility tensor the rest of this reads.
    U = np.full((args.seeds, len(subs), len(co.findings)), np.nan)
    for seed in range(args.seeds):
        p = f"{CACHE}/oof_seed{seed}{args.tag}.npy"
        if not os.path.exists(p):
            print(f"missing {p}; run subset_lattice.py first")
            return 1
        S = np.load(p).astype(np.float32)
        for si in range(len(subs)):
            for j in keep:
                U[seed, si, j] = safe_auroc(co.Y[:, j], S[si, :, j])
    Um = np.nanmean(U, 0)

    # ---- per-finding utility, with patient-level CIs on seed 0 -----------
    S0 = np.load(f"{CACHE}/oof_seed0{args.tag}.npy").astype(np.float32)
    rows = []
    for si, m in enumerate(subs):
        for j in keep:
            lo, hi = boot_auroc(co.Y[:, j], S0[si, :, j], co.patients, rng,
                                reps=args.boot)
            rows.append({
                "subset": subset_name(m), "k": int(sum(m)),
                "finding": co.findings[j], "n_pos": int(co.Y[:, j].sum()),
                "auroc_mean": float(np.nanmean(U[:, si, j])),
                "auroc_sd_seed": float(np.nanstd(U[:, si, j])),
                "ci_lo": lo, "ci_hi": hi,
            })
    pd.DataFrame(rows).to_csv(
        f"{RES}/finding_sequence_matrix{args.tag}.csv", index=False)

    # ---- exact Shapley value per (finding, input) ------------------------
    n = NSEQ
    w = {s: math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
         for s in range(n)}
    srows = []
    for j in keep:
        for mi in range(n):
            phi = []
            for seed in range(args.seeds):
                tot = 0.0
                for r in range(n):
                    for S in combinations([x for x in range(n) if x != mi], r):
                        b = sum(1 << x for x in S)
                        with_m = U[seed, idx_of[b | (1 << mi)], j]
                        # the empty set has no model, so U(empty) is chance
                        without = U[seed, idx_of[b], j] if b else 0.5
                        tot += w[r] * (with_m - without)
                phi.append(tot)
            srows.append({
                "finding": co.findings[j], "n_pos": int(co.Y[:, j].sum()),
                "sequence": SEQ_SHORT[SEQUENCES[mi]],
                "shapley": float(np.mean(phi)),
                "shapley_sd_seed": float(np.std(phi)),
            })
    sh = pd.DataFrame(srows)
    sh.to_csv(f"{RES}/shapley{args.tag}.csv", index=False)

    # ---- pairwise synergy ------------------------------------------------
    yrows = []
    for j in keep:
        for i1, i2 in combinations(range(n), 2):
            rest = [x for x in range(n) if x not in (i1, i2)]
            for r in range(len(rest) + 1):
                for S in combinations(rest, r):
                    b = sum(1 << x for x in S)
                    vals = []
                    for seed in range(args.seeds):
                        u_ij = U[seed, idx_of[b | (1 << i1) | (1 << i2)], j]
                        u_i = U[seed, idx_of[b | (1 << i1)], j]
                        u_j = U[seed, idx_of[b | (1 << i2)], j]
                        u_0 = U[seed, idx_of[b], j] if b else 0.5
                        vals.append(u_ij - u_i - u_j + u_0)
                    yrows.append({
                        "finding": co.findings[j],
                        "n_pos": int(co.Y[:, j].sum()),
                        "pair": f"{SEQ_SHORT[SEQUENCES[i1]]}|"
                                f"{SEQ_SHORT[SEQUENCES[i2]]}",
                        "context": subset_name(
                            tuple(1 if x in S else 0 for x in range(n))),
                        "synergy": float(np.mean(vals)),
                        "synergy_sd_seed": float(np.std(vals)),
                    })
    sy = pd.DataFrame(yrows)
    sy.to_csv(f"{RES}/synergy{args.tag}.csv", index=False)
    # The empty context is degenerate and must be excluded from any summary.
    # With U(empty) fixed at chance, psi = U(ij) - U(i) - U(j) + 0.5, so for a
    # finding whose single-input AUROCs are all ~0.94 the value is forced to
    # about -0.44 by arithmetic alone -- it measures the finding's accuracy,
    # not any interaction between inputs.  Summarising over it would have made
    # mean |psi| look like 0.070 instead of the real 0.013.
    syn = sy[sy.context != "none"]
    cell = syn.groupby(["finding", "pair"]).synergy.agg(["min", "max"])
    sign_flip = float(((cell["min"] < 0) & (cell["max"] > 0)).mean())

    # ---- the gate ---------------------------------------------------------
    full_i = idx_of[(1 << n) - 1]
    best_sub = {j: int(np.nanargmax(Um[:, j])) for j in keep}
    global_best = int(np.nanargmax(np.nanmean(Um[:, keep], 1)))
    n_distinct = len(set(best_sub.values()))
    full_macro = macro([Um[full_i, j] for j in keep])
    # oracle: selected and scored on the same predictions.  Upper bound only.
    oracle_macro = macro([Um[best_sub[j], j] for j in keep])

    # honest: subset chosen without seeing the rows it is scored on, averaged
    # over seeds.  This is the number the gate is judged on and the number the
    # paper reports as the value of finding-specific routing.
    hp, hg, hf, po = [], [], [], []
    for seed in range(args.seeds):
        Ss = np.load(f"{CACHE}/oof_seed{seed}{args.tag}.npy").astype(np.float32)
        folds = patient_folds(co.patients, k=args.folds, seed=seed)
        inner = load_inner(seed, args.tag, args.folds)
        pf_s, picks = nested_policy(Ss, co.Y, folds, keep, "per_finding",
                                    inner=inner)
        gl_s, _ = nested_policy(Ss, co.Y, folds, keep, "global", inner=inner)
        # the actual fixed policy: read every input, no selection anywhere --
        # plain OOF is unbiased for a policy that never chooses
        fx_s = Ss[full_i]
        hp.append(macro([safe_auroc(co.Y[:, j], pf_s[:, j]) for j in keep]))
        hg.append(macro([safe_auroc(co.Y[:, j], gl_s[:, j]) for j in keep]))
        hf.append(macro([safe_auroc(co.Y[:, j], fx_s[:, j]) for j in keep]))
        po.append(pd.DataFrame(picks).assign(seed=seed))
        # pooled prediction dumps so gain_ci bootstraps exactly these arrays
        np.save(f"{CACHE}/nestedpf_seed{seed}{args.tag}.npy",
                pf_s.astype(np.float16))
        np.save(f"{CACHE}/nestedgl_seed{seed}{args.tag}.npy",
                gl_s.astype(np.float16))
        if seed == 0:
            per_find_honest = {j: safe_auroc(co.Y[:, j], pf_s[:, j])
                               for j in keep}
            per_glob_honest = {j: safe_auroc(co.Y[:, j], gl_s[:, j])
                               for j in keep}
    pd.concat(po).to_csv(f"{RES}/policy_selections{args.tag}.csv", index=False)
    per_find_macro = float(np.mean(hp))
    global_macro = float(np.mean(hg))
    gain = per_find_macro - global_macro
    n_improved = sum(1 for j in keep
                     if per_find_honest[j] > per_glob_honest[j])

    # Budget frontier.  Both are reported: the oracle bound shows how much is
    # there in principle, the cross-fitted policy shows how much a selector
    # that never sees its evaluation rows can actually get.
    best_by_k, oracle_by_k = {}, {}
    for k in range(1, n + 1):
        cand = [si for si, m in enumerate(subs) if sum(m) == k]
        oracle_by_k[k] = macro([max(Um[si, j] for si in cand) for j in keep])
        vals = []
        for seed in range(args.seeds):
            Ss = np.load(f"{CACHE}/oof_seed{seed}{args.tag}.npy").astype(np.float32)
            folds = patient_folds(co.patients, k=args.folds, seed=seed)
            pk, _ = nested_policy(Ss, co.Y, folds, keep, "per_finding", cand,
                                  inner=load_inner(seed, args.tag, args.folds))
            vals.append(macro([safe_auroc(co.Y[:, j], pk[:, j]) for j in keep]))
        best_by_k[k] = float(np.mean(vals))

    loo = {}
    for mi in range(n):
        b = ((1 << n) - 1) & ~(1 << mi)
        loo[SEQ_SHORT[SEQUENCES[mi]]] = [Um[full_i, j] - Um[idx_of[b], j]
                                         for j in keep]
    loo_spread = {k: float(np.nanstd(v)) for k, v in loo.items()}

    g = {
        "n_findings": len(keep),
        "n_distinct_best_subsets": n_distinct,
        "global_best_subset": subset_name(subs[global_best]),
        "mean_gain_perfinding_over_global": gain,
        "macro_fixed_full_oof": float(np.mean(hf)),
        "gain_perfinding_over_fixed": per_find_macro - float(np.mean(hf)),
        "n_findings_improved": n_improved,
        "macro_full4": full_macro,
        "macro_global_fixed_nested": global_macro,
        "macro_perfinding_nested": per_find_macro,
        "macro_perfinding_oracle": oracle_macro,
        "selection_bias_oracle_minus_nested": oracle_macro - per_find_macro,
        "macro_best_k1": best_by_k[1], "macro_best_k2": best_by_k[2],
        "macro_best_k3": best_by_k[3], "macro_best_k4": best_by_k[4],
        "oracle_k1": oracle_by_k[1], "oracle_k2": oracle_by_k[2],
        "oracle_k3": oracle_by_k[3], "oracle_k4": oracle_by_k[4],
        "k2_minus_full": best_by_k[2] - full_macro,
        "loo_degradation_sd": loo_spread,
        "synergy_abs_mean": float(syn.synergy.abs().mean()),
        "synergy_frac_gt_005": float((syn.synergy.abs() > 0.005).mean()),
        "synergy_frac_gt_01": float((syn.synergy.abs() > 0.01).mean()),
        "synergy_frac_positive": float((syn.synergy > 0).mean()),
        "synergy_seed_sd": float(syn.synergy_sd_seed.mean()),
        # the cleanest statement of context-dependence: the same two inputs are
        # complementary given one context and redundant given another
        "synergy_sign_flips_with_context": sign_flip,
        "shapley_range_mean": float(
            sh.groupby("finding").shapley.apply(lambda x: x.max() - x.min())
            .mean()),
    }
    pd.DataFrame([{"metric": k, "value": v} for k, v in g.items()]).to_csv(
        f"{RES}/gate{args.tag}.csv", index=False)

    print("\n================ GATE ================")
    for k, v in g.items():
        print(f"{k:38s} {v}")
    print("\n--- criteria ---")
    checks = [
        ("A >=8 findings with differing best subset", n_distinct >= 8),
        ("B per-finding gain over global >= 0.01", gain >= 0.01),
        ("C some k<=2 policy within 0.01 of full", best_by_k[2] >= full_macro - 0.01),
        ("D leave-one-out effect heterogeneous",
         max(loo_spread.values()) > 0.01),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nVERDICT: {'PROCEED' if all(o for _, o in checks) else 'REVIEW'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
