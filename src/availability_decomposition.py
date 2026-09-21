"""Where does the availability-only signal come from?

A classifier given only the availability mask -- which sequences a study carries,
no pixels -- separates findings at 0.603 macro AUROC.  The first thing a reader
will say is that the mask is a proxy for site, vendor or protocol, and that the
"leak" is really institutional composition.

That objection does not overturn the conclusion: site-mediated label information
is still availability leakage, and still breaks any comparison between complete
and incomplete cohorts.  But it changes what the number *means*, so we decompose
it rather than leave the reader to raise it.

Five models, identical head and folds throughout:

  mask          availability bits alone
  site          vendor + field strength + station alone, no availability
  site+mask     both
  mask|vendor   availability alone, fitted and scored within the largest vendor,
                so institutional composition is held fixed
  mask-perm     availability permuted within strata.  This is the control that
                matters: it preserves site composition exactly and destroys only
                the mask-label association, so whatever it retains is
                attributable to site rather than to availability.

The informative contrast is mask versus mask-perm.  If mask stays well above its
site-matched permutation, availability carries label information beyond site.

Two stratifications are run, because they are not the same control and an
earlier version of this file conflated them.  Permuting within *vendor* holds
only the manufacturer fixed, which is a weak reference when the site block also
contains field strength and station.  Permuting within the full
vendor x field x station cell is the reference that actually matches the site
model, and it is the one the paper quotes.  Strata smaller than `--min-stratum`
are pooled into one residual cell, since a stratum of size 1 cannot be permuted
and would silently leave those rows unshuffled.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import (SEQSET, Cohort, finding_filter, macro,  # noqa: E402
                    patient_folds, safe_auroc)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def site_features(meta):
    """One-hot vendor, field strength and station."""
    v = pd.get_dummies(meta.vendor.fillna("NA"), prefix="v")
    f = pd.get_dummies(pd.to_numeric(meta.field_t, errors="coerce")
                       .round(2).fillna(-1).astype(str), prefix="f")
    s = pd.get_dummies(meta.station.fillna("NA"), prefix="s")
    return pd.concat([v, f, s], axis=1).to_numpy(np.float32)


def cv_macro(X, Y, keep, folds, C=0.1):
    """Out-of-fold macro AUROC for one feature block."""
    S = np.full((len(X), Y.shape[1]), np.nan, np.float32)
    for tr, te in folds:
        if X.shape[1] == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]), sc.transform(X[te])
        for j in keep:
            y = Y[tr, j]
            if y.sum() < 3 or y.sum() == len(y):
                continue
            m = LogisticRegression(max_iter=2000, C=C,
                                   class_weight="balanced").fit(A, y)
            S[te, j] = m.decision_function(B)
    per = {}
    for j in keep:
        s = S[:, j]
        ok = np.isfinite(s)
        per[j] = safe_auroc(Y[ok, j], s[ok]) if ok.sum() > 10 else np.nan
    return per


def permute_within(strata, M, rng):
    """Shuffle availability rows inside each stratum, keeping site mix fixed."""
    out = M.copy()
    for s in np.unique(strata):
        i = np.where(strata == s)[0]
        out[i] = M[rng.permutation(i)]
    return out


def build_strata(meta, kind, min_stratum):
    """Stratum label per study, with undersized cells pooled.

    Returns the labels and the number of rows that ended up in the pooled
    residual cell, which is reported rather than hidden: rows in a singleton
    stratum are unshufflable, so a null that leaves them in place is
    anti-conservative by exactly that fraction.
    """
    if kind == "vendor":
        s = meta.vendor.fillna("NA").astype(str).to_numpy()
    else:
        s = (meta.vendor.fillna("NA").astype(str) + "|"
             + pd.to_numeric(meta.field_t, errors="coerce").round(2)
                 .fillna(-1).astype(str) + "|"
             + meta.station.fillna("NA").astype(str)).to_numpy()
    cnt = pd.Series(s).value_counts()
    small = set(cnt[cnt < min_stratum].index)
    pooled = int(np.isin(s, list(small)).sum()) if small else 0
    s = np.where(np.isin(s, list(small)), "__pooled__", s) if small else s
    return s, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--min-stratum", type=int, default=20,
                    help="strata smaller than this are pooled; a singleton "
                         "stratum cannot be permuted and would bias the null")
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    # the whole split, not complete-case: the point is natural missingness
    co = Cohort("train", complete_case=False)
    keep = finding_filter(co.Y, co.findings, min_pos=args.min_pos)
    meta = co.meta()
    folds = patient_folds(co.patients, k=args.folds, seed=0)
    Xs = site_features(meta)
    Xm = co.M.astype(np.float32)
    vendor = meta.vendor.fillna("NA").to_numpy().astype(str)
    print(f"set={SEQSET} n={len(co)} findings={len(keep)} "
          f"mask_dim={Xm.shape[1]} site_dim={Xs.shape[1]}", flush=True)

    blocks = {
        "mask": Xm,
        "site": Xs,
        "site+mask": np.hstack([Xs, Xm]),
    }
    rows = []
    for name, X in blocks.items():
        per = cv_macro(X, co.Y, keep, folds)
        rows.append({"model": name, "n": len(co), "n_findings": len(keep),
                     "macro_auroc": macro(list(per.values()))})
        print(f"  {name:10s} macro AUROC = {rows[-1]['macro_auroc']:.4f}",
              flush=True)

    # availability alone, within the largest vendor only
    big = pd.Series(vendor).value_counts().index[0]
    sel = np.where(vendor == big)[0]
    sub_folds = patient_folds(co.patients[sel], k=args.folds, seed=0)
    kk = finding_filter(co.Y[sel], co.findings, min_pos=args.min_pos)
    per = cv_macro(Xm[sel], co.Y[sel], kk, sub_folds)
    rows.append({"model": f"mask|{big}", "n": len(sel), "n_findings": len(kk),
                 "macro_auroc": macro(list(per.values()))})
    print(f"  mask|{big:8s} macro AUROC = {rows[-1]['macro_auroc']:.4f} "
          f"(n={len(sel)})", flush=True)

    # the control: destroy the mask-label link, keep site composition exactly.
    # Folds are the unpermuted patient-level folds -- only the mask rows move,
    # so the null differs from the observed fit in exactly one respect.
    null_sets = {}
    for kind, off in (("vendor", 1000), ("site", 5000)):
        strata, pooled = build_strata(meta, kind, args.min_stratum)
        nstr = len(np.unique(strata))
        print(f"  strata[{kind}] cells={nstr} pooled_rows={pooled} "
              f"({pooled / len(co):.3%})", flush=True)

        def one_perm(s, strata=strata, off=off):
            r = np.random.default_rng(off + s)
            Xp = permute_within(strata, Xm, r)
            return macro(list(cv_macro(Xp, co.Y, keep, folds).values()))

        nulls = Parallel(n_jobs=min(args.jobs, args.perms), verbose=1)(
            delayed(one_perm)(s) for s in range(args.perms))
        nulls = np.array([x for x in nulls if np.isfinite(x)])
        null_sets[kind] = nulls
        rows.append({"model": f"mask-perm-within-{kind}", "n": len(co),
                     "n_findings": len(keep),
                     "macro_auroc": float(nulls.mean()),
                     "null_median": float(np.median(nulls)),
                     "null_p95": float(np.percentile(nulls, 95)),
                     "null_max": float(nulls.max()), "n_perms": len(nulls),
                     "n_strata": nstr, "pooled_rows": pooled})
    nulls = null_sets["site"]

    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/availability_decomposition{args.tag}.csv", index=False)
    obs = d[d.model == "mask"].macro_auroc.iloc[0]
    print("\n=== availability decomposition ===")
    print(d.round(4).to_string(index=False))
    print(f"\nmask observed                {obs:.4f}")
    for kind in ("vendor", "site"):
        z = null_sets[kind]
        print(f"mask perm within {kind:7s}    {z.mean():.4f} "
              f"[median {np.median(z):.4f}, p95 {np.percentile(z, 95):.4f}]"
              f"   excess {obs - z.mean():+.4f}")
    print("the paper quotes the vendor x field x station reference "
          "('site'), which is the stricter of the two")
    return 0


if __name__ == "__main__":
    sys.exit(main())
