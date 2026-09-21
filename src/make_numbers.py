"""Turn results/*.csv into LaTeX macros, so no number is ever typed by hand.

An earlier verifier caught hand-typed digits that had drifted from their source
more than once, which is the whole argument for this file existing.  Every
quantity the paper displays is defined here as a \\newcommand read from a CSV;
verify_numbers.py then re-runs this generator and fails if the output changed,
which is what catches a CSV moving underneath a number.

LaTeX control sequences cannot contain digits, so numerals in macro names are
spelled out: \\gateKTwoMinusFull, not \\gateK2MinusFull.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import robust_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results"
OUT = f"{ROOT}/paper/numbers.tex"

_DIGITS = str.maketrans({"0": "Zero", "1": "One", "2": "Two", "3": "Three",
                         "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
                         "8": "Eight", "9": "Nine"})


class Macros(dict):
    def put(self, name, value, fmt="{:.3f}"):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        self[name.translate(_DIGITS)] = fmt.format(value)

    def n(self, name, value):
        self.put(name, int(value), "{:,d}")

    def pct(self, name, value):
        self.put(name, 100 * value, "{:.1f}")


def exists(f):
    """Path to a result file, preferring the one for the active input set.

    Result files are written with a tag (`gate_k5.csv`) while the untagged name
    belongs to k4.  Reading untagged names unconditionally is how the abstract
    ended up quoting k4's risk-audit coverage (13/15) in a sentence about k5's
    failure (7/12) -- the numbers were each individually correct and the claim
    was wrong.  Prefer `<stem>_<SEQSET><ext>` when it exists, fall back
    otherwise, and record which was used so the paper can state it.
    """
    from common import SEQSET
    stem, ext = os.path.splitext(f)
    tagged = f"{RES}/{stem}_{SEQSET}{ext}"
    if SEQSET != "k4" and os.path.exists(tagged):
        _USED.add(os.path.basename(tagged))
        return tagged
    p = f"{RES}/{f}"
    if os.path.exists(p):
        _USED.add(f)
        return p
    return None


_USED = set()


def main():
    m = Macros()

    # ---- cohort ----------------------------------------------------------
    import json
    # FROZEN already follows QS_COHORT/QS_SEQSET.  Hardcoding the untagged
    # path made the abstract claim "12,412 complete five-sequence studies",
    # which is k4's four-sequence count -- the sentence was false while every
    # individual number in it was a real number from a real file.
    from common import FROZEN as fr
    if os.path.exists(fr):
        c = json.load(open(fr))["splits"]
        for s in ("train", "val", "test"):
            m.n(f"cohort{s.capitalize()}", c[s]["n"])
            m.n(f"cohort{s.capitalize()}Cc", c[s]["n_complete_case"])
        m.n("cohortTotalCc", sum(c[s]["n_complete_case"]
                                 for s in ("train", "val", "test")))

    # ---- the missingness confound ---------------------------------------
    if (f := exists("availability_confound.csv")):
        d = robust_csv(f)
        m.put("confoundMacro", d.auroc.mean())
        m.put("confoundMax", d.auroc.max())
        m.n("confoundNFindings", len(d))
        m.n("confoundNAboveSixty", int((d.auroc > 0.60).sum()))

    # ---- the lattice and the gate ---------------------------------------
    if (f := exists("subset_lattice.csv")):
        d = robust_csv(f)
        d = d[d.C == d.C.iloc[0]]
        m.n("latticeSubsets", d.subset.nunique())
        m.n("latticeFindings", d.finding.nunique())
        m.n("latticeSeeds", d.seed.nunique())
        piv = d.groupby(["finding", "subset"]).auroc.mean().unstack()
        macro = piv.mean(0).sort_values(ascending=False)
        m.put("latticeBestFixedMacro", macro.iloc[0])
        m[("latticeBestFixedName").translate(_DIGITS)] = macro.index[0].replace(
            "+", "{+}")
        # the all-inputs subset of the active set, not a pinned k4 string:
        # pinned, these three macros silently vanished under k5
        full = max(macro.index, key=lambda c: len(c.split("+")))
        if len(full.split("+")) == d.subset.map(
                lambda x: len(x.split("+"))).max():
            m.put("latticeFullMacro", macro[full])
            m.put("latticeBestMinusFull", macro.iloc[0] - macro[full], "{:.4f}")
            m.put("latticeSecondBestMacro", macro.iloc[1])
            m.put("latticeFullMinusSecond", macro[full] - macro.iloc[1], "{:.4f}")
            m.n("latticeFullRank", int(list(macro.index).index(full)) + 1)
        m.put("latticeWorstFixedMacro", macro.iloc[-1])
        m.put("latticeFixedSpread", macro.iloc[0] - macro.iloc[-1])

    if (f := exists("gate.csv")):
        g = robust_csv(f).set_index("metric").value
        def gv(k):
            try:
                return float(g[k])
            except (KeyError, ValueError, TypeError):
                return None
        for k, nm in [
                ("n_distinct_best_subsets", "gateDistinctSubsets"),
                ("n_findings", "gateNFindings"),
                ("n_findings_improved", "gateNImproved")]:
            v = gv(k)
            if v is not None:
                m.n(nm, v)
        # the number a naive analysis would have reported: per-finding oracle
        # against the same fixed baseline, both read off the same predictions
        # These three are compared with each other in the abstract, so three
        # decimals would render them as 0.020 / 0.010 / 0.010 and destroy the
        # very contrast they exist to make.
        o, g0 = gv("macro_perfinding_oracle"), gv("macro_global_fixed_nested")
        if o is not None and g0 is not None:
            m.put("gateNaiveGain", o - g0, "{:.4f}")
            cf = gv("mean_gain_perfinding_over_global")
            if cf:
                # how much larger the naive number is than the cross-fitted one,
                # as a percentage -- so prose can never round it to "twice"
                m.put("optNaiveExcessPct", 100 * ((o - g0) / cf - 1), "{:.0f}")
        m.put("gateGain", gv("mean_gain_perfinding_over_global"), "{:.4f}")
        m.put("gateGainVsFixed", gv("gain_perfinding_over_fixed"), "{:.4f}")
        m.put("gateFixedFullOof", gv("macro_fixed_full_oof"))
        m.put("gateSelectionBias", gv("selection_bias_oracle_minus_nested"),
              "{:.4f}")
        for k, nm in [
                ("macro_full4", "gateFullFour"),
                ("macro_global_fixed_nested", "gateGlobalFixed"),
                ("macro_perfinding_nested", "gatePerFinding"),
                ("macro_perfinding_oracle", "gateOracle"),
                ("macro_best_k1", "gateKOne"), ("macro_best_k2", "gateKTwo"),
                ("macro_best_k3", "gateKThree"),
                ("macro_best_k4", "gateKFour"),
                ("oracle_k1", "gateOracleKOne"),
                ("oracle_k2", "gateOracleKTwo"),
                ("k2_minus_full", "gateKTwoMinusFull"),
                ("synergy_abs_mean", "gateSynergyAbs"),
                ("shapley_range_mean", "gateShapleyRange")]:
            m.put(nm, gv(k))
        for k, nm in [("synergy_frac_gt_005", "gateSynergyFrac"),
                      ("synergy_frac_gt_01", "gateSynergyFracTen"),
                      ("synergy_sign_flips_with_context", "gateSynergyFlipPct"),
                      ("synergy_frac_positive", "gateSynergyPosPct")]:
            v = gv(k)
            if v is not None:
                m.pct(nm, v)
        m.put("gateSynergySeedSd", gv("synergy_seed_sd"), "{:.4f}")

    # derived, not typed: this has to follow QS_SEQSET or the paper would name
    # the wrong inputs the moment the input set changes
    from common import NSEQ, SEQSET, SEQUENCES, SEQ_SHORT
    m["seqShortNames".translate(_DIGITS)] = ", ".join(
        SEQ_SHORT[s] for s in SEQUENCES)
    m["seqSetName".translate(_DIGITS)] = SEQSET
    m.n("seqCount", NSEQ)
    m.n("seqSubsetCount", 2 ** NSEQ - 1)
    # the Bonferroni denominator is the number of candidates actually
    # tested, which is every non-empty subset including the full set --
    # testing the full set against itself is degenerate but keeping it in
    # the denominator is the conservative choice
    m.n("riskNCandidates", 2 ** NSEQ - 1)

    # ---- router temperature / cost sweep --------------------------------
    if (f := exists("router_sweep.csv")):
        d = robust_csv(f)
        m.n("sweepCells", d.groupby(["mode", "tau", "beta"]).ngroups)
        m.n("sweepTaus", d.tau.nunique())
        q = d[(d["mode"] == "queryseq") & (d.beta == 0)]
        if len(q):
            g = q.groupby("tau").macro_full.mean()
            m.put("sweepQsBestTau", float(g.max()))
            m.put("sweepQsWorstTau", float(g.min()))
        # tau is inert for mode=patient (no /tau in that gate), so those rows
        # are exact duplicates across tau -- dedupe before any aggregate
        d = d.drop_duplicates(subset=[c for c in d.columns if c != "tau"]
                              if "tau" in d.columns else None)
        m.n("sweepDistinctFits", len(d))
        # The fusion normalizes by the gate sum, so predictions are invariant
        # to uniform gate shrinkage: a raw-gate cost weight produces SHRINKAGE
        # (thresholded count -> 0 while accuracy is intact), not sparsity.
        # True collapse happens only when the gate sum falls to the epsilon in
        # the normalizer.  A previous macro averaged the two regimes into a
        # meaningless 0.713 "chance" figure; these keep them apart.
        low = d[d.avg_inputs_read < 0.01]
        if len(low):
            eps_dominated = low[low.gate_mean * 5 < 1e-6]
            shrunk = low[low.gate_mean * 5 >= 1e-6]
            if len(shrunk):
                m.put("sweepShrunkAuroc", float(shrunk.macro_full.mean()))
                m.n("sweepShrunkRuns", len(shrunk))
            if len(eps_dominated):
                m.put("sweepTrueCollapseAuroc",
                      float(eps_dominated.macro_full.mean()))
                m.n("sweepTrueCollapseRuns", len(eps_dominated))

    # ---- method comparison -----------------------------------------------
    # `n` prefix = the per-finding-readout variant, where the router is the only
    # path the query can act through; that is the paper's primary comparison.
    for suffix, pre in (("", "m"), ("_nqh", "n")):
        f = exists(f"method_macro{suffix}.csv")
        if not f:
            continue
        d = robust_csv(f)
        av = d[d.axis == "availability"]
        for meth in av.method.unique():
            s = av[av.method == meth]
            tag = "".join(w.capitalize() for w in meth.split("_"))
            m.put(f"{pre}{tag}Full", s.macro_full.mean())
            m.put(f"{pre}{tag}Avg", s.macro_avg_subset.mean())
            m.put(f"{pre}{tag}Worst", s.macro_worst_subset.mean())
            m.put(f"{pre}{tag}FullSd", s.macro_full.std(), "{:.4f}")
        bu = d[d.axis == "budget"]
        for meth in bu.method.unique():
            tag = "".join(w.capitalize() for w in meth.split("_"))
            for k in (1, 2, 3, 4):
                v = bu[(bu.method == meth) & (bu.budget == k)].macro_auroc
                if len(v):
                    m.put(f"{pre}{tag}K{k}".translate(_DIGITS), v.mean())
        if len(av):
            g = av.groupby("method").macro_full.mean()
            m.put(f"{pre}SpreadFull", float(g.max() - g.min()), "{:.4f}")
            m.put(f"{pre}MaxSeedSd",
                  float(av.groupby("method").macro_full.std().max()),
                  "{:.4f}")

    # ---- paired deltas ---------------------------------------------------
    if (f := exists("method_deltas.csv")):
        d = robust_csv(f)
        for _, r in d.iterrows():
            tag = ("d" + "".join(w.capitalize() for w in str(r.a).split("_"))
                   + "Vs" + "".join(w.capitalize()
                                    for w in str(r.b).split("_"))
                   + str(r.get("axis", "")).capitalize())
            m.put(tag + "Mean", r.get("delta"))
            m.put(tag + "Lo", r.get("ci_lo"))
            m.put(tag + "Hi", r.get("ci_hi"))
            if "p_sign" in r:
                m.put(tag + "Psign", r.p_sign)

    # ---- paired deltas, independent-readout (nqh) configuration ---------
    if (f := exists("method_deltas_nqh.csv")):
        d = robust_csv(f)
        for _, r in d.iterrows():
            tag = ("dNqh" + "".join(w.capitalize() for w in str(r.a).split("_"))
                   + "Vs" + "".join(w.capitalize()
                                    for w in str(r.b).split("_"))
                   + str(r.get("axis", "")).capitalize())
            m.put(tag + "Mean", r.get("delta"), "{:+.4f}")
            m.put(tag + "Lo", r.get("ci_lo"), "{:+.4f}")
            m.put(tag + "Hi", r.get("ci_hi"), "{:+.4f}")

    # ---- external paired deltas ------------------------------------------
    if (f := exists("external_deltas.csv")):
        d = robust_csv(f)
        for _, r in d.iterrows():
            tag = ("dExt" + str(r["shift"]).capitalize()
                   + "Vs" + "".join(w.capitalize()
                                    for w in str(r.b).split("_")))
            m.put(tag + "Mean", r.get("delta"), "{:+.4f}")
            m.put(tag + "Lo", r.get("ci_lo"), "{:+.4f}")
            m.put(tag + "Hi", r.get("ci_hi"), "{:+.4f}")

    # ---- query diagnostics ----------------------------------------------
    if (f := exists("query_diagnostics.csv")):
        d = robust_csv(f)
        a = d.groupby("variant").macro_auroc.mean()
        for v in a.index:
            tag = "q" + "".join(w.capitalize()
                                for w in v.replace(":", "_").split("_"))
            m.put(tag, a[v])
        para = a[[i for i in a.index if i.startswith("para:")]]
        if len(para):
            m.put("qParaMean", para.mean())
            m.put("qParaSpread", para.max() - para.min())
            if "para:sentence" in a.index and "para:definition" in a.index:
                m.put("qParaSentDef",
                      a["para:sentence"] - a["para:definition"], "{:.4f}")
        for c in ("shuffled", "counterfactual"):
            if c in a.index and len(para):
                # against the paraphrase mean (as labelled where used) ...
                m.put(f"qDrop{c.capitalize()}", para.mean() - a[c])
                # ... and against the canonical training-form query, the
                # contrast the main text describes
                if "para:sentence" in a.index:
                    m.put(f"qDrop{c.capitalize()}Canon",
                          a["para:sentence"] - a[c])

    if (f := exists("query_routing_agreement.csv")):
        d = robust_csv(f)
        a = d.groupby("variant").routing_cos_vs_base.mean()
        para = a[[i for i in a.index if i.startswith("para:")]]
        if len(para):
            m.put("qRoutingParaCos", para.mean())
        for c in ("shuffled", "counterfactual"):
            if c in a.index:
                m.put(f"qRoutingCos{c.capitalize()}", a[c])

    if (f := exists("routing_seed_stability.csv")):
        d = robust_csv(f)
        m.put("routingSeedCos", d.routing_cos.mean())

    # ---- the 2x2 query-pathway intervention ------------------------------
    if (f := exists("query_pathways.csv")):
        d = robust_csv(f).groupby("condition")[
            ["macro_auroc", "macro_auprc", "routing_cos_vs_base"]].mean()
        key = {"both correct": "Base", "gate shuffled": "Gate",
               "readout shuffled": "Head", "both shuffled": "Both"}
        for c, tag in key.items():
            if c in d.index:
                m.put(f"qPath{tag}Auroc", d.loc[c, "macro_auroc"])
                m.put(f"qPath{tag}Auprc", d.loc[c, "macro_auprc"])
        if "both correct" in d.index:
            b = d.loc["both correct", "macro_auroc"]
            for c, tag in key.items():
                if c in d.index and c != "both correct":
                    m.put(f"qPath{tag}Drop", b - d.loc[c, "macro_auroc"])
        if "gate shuffled" in d.index:
            m.put("qRoutingDistShuffled",
                  1 - d.loc["gate shuffled", "routing_cos_vs_base"])

    # ---- risk-control coverage on held-out data --------------------------
    if (f := exists("risk_coverage.csv")):
        d = robust_csv(f)
        for _, r in d.iterrows():
            t = {0.005: "Lo", 0.01: "Mid", 0.02: "Hi"}.get(round(r.delta, 4))
            if t is None:
                continue
            m.pct(f"riskCov{t}", r.coverage)
            m.n(f"riskKept{t}", r.n_kept)
            m.n(f"riskN{t}", r.n_findings)
            m.put(f"riskMeanK{t}", r.mean_k, "{:.2f}")
            m.put(f"riskMeanDelta{t}", r.mean_test_delta, "{:+.4f}")
            m.put(f"riskWorstDelta{t}", r.worst_test_delta, "{:+.4f}")
            m.pct(f"riskFracKTwo{t}", r.frac_k_le_two)
            m.pct(f"riskCovCiLo{t}", r.coverage_lo)
            m.pct(f"riskCovCiHi{t}", r.coverage_hi)
            if "n_kept_instances" in d.columns:
                m.n(f"riskKeptInst{t}", r.n_kept_instances)
                m.n(f"riskInstN{t}", r.n_instances)

    # ---- two optimism estimands, named apart -----------------------------
    # An earlier draft called both of these "optimism" and reported 0.0102 in
    # one place and 0.0149 in another.  They differ because averaging seeds
    # before the argmax shrinks the maximum's upward bias.  Distinct macros so
    # the manuscript cannot conflate them again.
    if (f := exists("optimism_estimands.csv")):
        d = robust_csv(f).set_index("estimand")
        key = {"policy_oracle_gap": "optPolicyOracleGap",
               "search_optimism": "optSearchOptimism",
               "search_optimism_seed0": "optSearchSeedZero",
               "seed_averaging_reduction": "optSeedAvgReduction"}
        for k, nm in key.items():
            if k in d.index:
                m.put(nm, float(d.loc[k, "optimism"]), "{:.4f}")
        if "policy_oracle_gap" in d.index:
            m.put("optPolicyOracle",
                  float(d.loc["policy_oracle_gap", "naive_or_oracle"]))
            m.put("optPolicyCrossfit",
                  float(d.loc["policy_oracle_gap", "crossfitted"]))

    # ---- matched-cohort SWI ablation --------------------------------------
    # The only comparison that isolates one input: same cohort, same findings,
    # same folds, SWI present or absent.  The larger k4-to-k5 difference mixes
    # this with cohort and label-space change and must not be quoted as SWI's.
    a, b = exists("gate_k5noswi.csv"), exists("gate_k5.csv")
    if a and b:
        ga = robust_csv(a).set_index("metric").value
        gb = robust_csv(b).set_index("metric").value
        try:
            no = float(ga["mean_gain_perfinding_over_global"])
            wi = float(gb["mean_gain_perfinding_over_global"])
            m.put("swiAblationNo", no, "{:.4f}")
            m.put("swiAblationWith", wi, "{:.4f}")
            m.put("swiAblationDelta", wi - no, "{:+.4f}")
        except (KeyError, ValueError):
            pass

    # ---- where the availability signal comes from ------------------------
    if (f := exists("availability_decomposition.csv")):
        d = robust_csv(f).set_index("model")
        def av(k, col="macro_auroc"):
            return float(d.loc[k, col]) if k in d.index else None
        m.put("avMask", av("mask"))
        m.put("avSite", av("site"))
        m.put("avSiteMask", av("site+mask"))
        for k in d.index:
            if str(k).startswith("mask|"):
                m.put("avMaskWithinSite", av(k))
                m["avLargestVendor"] = str(k).split("|", 1)[1]
                m.n("avWithinSiteN", d.loc[k, "n"])
        # Two permutation references, and the gap between them is itself worth
        # reporting: holding only the manufacturer fixed leaves a residual four
        # to six times larger than holding vendor x field x station fixed. The
        # paper quotes the stricter one and shows the other beside it, because
        # an earlier draft quoted the vendor-only figure while calling it a
        # site-matched reference.
        p = "mask-perm-within-site"
        if p in d.index:
            m.put("avMaskPermNull", av(p))
            m.put("avMaskPermP", av(p, "null_p95"))
            m.n("avPerms", d.loc[p, "n_perms"])
            m.n("avPermStrata", d.loc[p, "n_strata"])
            if av("mask") is not None:
                m.put("avMaskExcess", av("mask") - av(p))
        q = "mask-perm-within-vendor"
        if q in d.index:
            m.put("avMaskPermVendorNull", av(q))
            if av("mask") is not None:
                m.put("avMaskExcessVendor", av("mask") - av(q))
        if av("site") is not None and av("site+mask") is not None:
            m.put("avMaskOverSite", av("site+mask") - av("site"))

    # ---- second dataset: the pre-registered BraTS-GoAT replication -------
    # Deliberately untagged.  This corpus has four modalities and one input
    # set; it is not a variant of the MRI lattice and must not inherit its
    # suffix.  Both endpoints are emitted: the degenerate pre-registered
    # presence targets and the amended tumour-burden targets, so the failed
    # original analysis is reported rather than replaced.
    if os.path.exists(p := f"{RES}/goat_shapley.csv"):
        _USED.add("goat_shapley.csv")
        sh = robust_csv(p)
        for kind, pre in (("burden", "goatB"), ("presence", "goatP"),
                          ("burden_perfold", "goatPf")):
            q = sh[sh.target_kind == kind]
            if not len(q):
                continue
            piv = q.pivot_table(index="target", columns="modality",
                                values="shapley")
            for t in piv.index:
                row = piv.loc[t].sort_values(ascending=False)
                m[f"{pre}{t}Top"] = str(row.index[0])
                m.put(f"{pre}{t}TopVal", float(row.iloc[0]), "{:.4f}")
                m.put(f"{pre}{t}SecondVal", float(row.iloc[1]), "{:.4f}")
                m[f"{pre}{t}Second"] = str(row.index[1])
                for mod in piv.columns:
                    m.put(f"{pre}{t}{mod}", float(piv.loc[t, mod]), "{:.4f}")
            b1 = piv.loc["ET"].idxmax() == "T1c" if "ET" in piv.index else False
            b2 = (piv.loc["SNFH"].idxmax() == "T2f"
                  if "SNFH" in piv.index else False)
            b3 = ((piv.loc["SNFH"].idxmax() != "T1c")
                  and (piv.loc["ET"].idxmax() != "T2f"))
            for k, v in (("One", b1), ("Two", b2), ("Three", b3)):
                m[f"{pre}{k}"] = "holds" if v else "fails"
            m[f"{pre}Verdict"] = ("all three hold" if (b1 and b2 and b3)
                                  else "not all hold")
    if os.path.exists(p := f"{RES}/goat_lattice.csv"):
        _USED.add("goat_lattice.csv")
        la = robust_csv(p)
        m.n("goatSubsets", int(la.subset.nunique()))
        m.n("goatSeeds", int(la.seed.nunique()))
        b = la[la.target_kind == "burden"]
        for t in b.target.unique():
            m.n(f"goatNPos{t}", int(b[b.target == t].n_pos.iloc[0]))
        pr = la[la.target_kind == "presence"]
        for t in pr.target.unique():
            m.n(f"goatNPres{t}", int(pr[pr.target == t].n_pos.iloc[0]))
    if os.path.exists(p := f"{RES}/goat_thresholds.json"):
        import json
        j = json.load(open(p))
        m.n("goatCases", int(j["n_cases"]))
        for t, v in j["thresholds_train_median"].items():
            m.n(f"goatThr{t}", int(round(v)))

    # ---- the active input set, as prose ---------------------------------
    # Every "all four inputs" in the draft was a hand-typed constant that
    # survived the move to five sequences.  Prose now reads these.
    from common import NSEQ as _NSEQ
    m.n("nSeq", _NSEQ)
    m["nSeqWord"] = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                     7: "seven"}.get(_NSEQ, str(_NSEQ))

    # ---- headline gain: paired patient bootstrap -------------------------
    if (f := exists("gain_ci.csv")):
        d = robust_csv(f).iloc[0]
        m.put("gainCiLo", float(d.ci_lo), "{:+.4f}")
        m.put("gainCiHi", float(d.ci_hi), "{:+.4f}")
        m.n("gainReps", int(d.n_reps))
        m.n("gainPosFindings", int(d.n_pos_gain))
        m.n("gainNFindings", int(d.n_findings))
        m.put("gainMedian", float(d.median_gain), "{:+.4f}")
        m.put("gainMin", float(d.min_gain), "{:+.4f}")
        m.put("gainMax", float(d.max_gain), "{:+.4f}")
        if "point_vs_fixed" in d:
            m.put("gainFixedPoint", float(d.point_vs_fixed), "{:+.4f}")
            m.put("gainFixedCiLo", float(d.ci_lo_fixed), "{:+.4f}")
            m.put("gainFixedCiHi", float(d.ci_hi_fixed), "{:+.4f}")
            m.n("gainPosFindingsFixed", int(d.n_pos_gain_fixed))

    # ---- held-out replication of the selection gain ----------------------
    if (f := exists("heldout_gain.csv")):
        d = robust_csv(f).iloc[0]
        m.put("heldoutGain", float(d.mean_gain), "{:+.4f}")
        m.put("heldoutCiLo", float(d.ci_lo), "{:+.4f}")
        m.put("heldoutCiHi", float(d.ci_hi), "{:+.4f}")
        m.n("heldoutN", int(d.n_findings))
        m.n("heldoutPos", int(d.n_pos_gain))

    # ---- rank stability under patient bootstrap --------------------------
    if (f := exists("rank_stability.csv")):
        d = robust_csv(f)
        _USED.add(os.path.basename(f))
        key = {"Cerebral hemorrhage": "Ch", "Silent micro-hemorrhage of brain":
               "Mh", "Cavernous hemangioma": "Cav",
               "Subdural intracranial hemorrhage": "Sdh"}
        ckey = {"Cerebral atrophy": "Atr", "Ventriculomegaly": "Vent",
                "Empty sella syndrome": "Sella", "Mega cisterna magna": "Mcm"}
        for _, r in d.iterrows():
            if r.dataset == "mrrate" and r.finding in key:
                m.pct(f"rank{key[r.finding]}First", r.prob)
                m.put(f"rank{key[r.finding]}GapLo", r.gap_lo, "{:+.4f}")
                m.put(f"rank{key[r.finding]}GapHi", r.gap_hi, "{:+.4f}")
            if r.dataset == "mrrate" and r.finding in ckey:
                m.pct(f"rank{ckey[r.finding]}Last", r.prob)
            if r.dataset == "mrrate" and r.event == "first_ge3_of_4":
                m.pct("rankJointFirst", r.prob)
            if r.dataset == "mrrate" and r.event == "last_all_4":
                m.pct("rankJointLast", r.prob)
            if r.dataset == "goat_burden" and r.finding == "ET":
                m.pct("rankGoatEt", r.prob)
                m.put("rankGoatEtGapLo", r.gap_lo, "{:+.4f}")
            if r.dataset == "goat_burden" and r.finding == "SNFH":
                m.pct("rankGoatSnfh", r.prob)
                m.put("rankGoatSnfhGapLo", r.gap_lo, "{:+.4f}")
            if r.dataset == "goat_burden_perfold" and r.finding == "ET":
                m.pct("rankGoatEtPf", r.prob)
            if r.dataset == "goat_burden_perfold" and r.finding == "SNFH":
                m.pct("rankGoatSnfhPf", r.prob)
                m.put("rankGoatSnfhPfGapLo", r.gap_lo, "{:+.4f}")

    # ---- how sparse are the certified cells themselves -------------------
    if (f := exists("risk_policy.csv")):
        d = robust_csv(f)
        if "k" in d.columns and len(d):
            m.pct("riskCellsAllPct", float((d.k == 5).mean()))
            m.n("riskCellsAll", int((d.k == 5).sum()))
            m.n("riskCellsN", len(d))

    # ---- risk-audit failure analysis -------------------------------------
    # Contrasts between findings that kept and broke the certificate, from the
    # same per-finding audit file the coverage numbers come from.  Computed
    # here so the failure-analysis paragraph cannot drift from the CSV.
    if (f := exists("risk_validate.csv")):
        d = robust_csv(f)
        pd_ = d[d.delta == 0.01]
        g = pd_.groupby("finding").agg(
            td=("test_delta", "mean"), npte=("n_pos_test", "first"),
            nptr=("n_pos_train", "first"), lcb=("train_lcb", "mean"),
            kk=("k", "median"))
        # primary rule everywhere: a finding keeps iff its CROSS-SEED MEAN
        # held-out difference clears -delta (same estimand as the table)
        g["kept"] = g.td >= -0.01
        g["margin"] = g.lcb + 0.01
        g["shift"] = abs((g.npte / 862) / (g.nptr / 9393) - 1)
        ke, fa = g[g.kept], g[~g.kept]
        if len(ke) and len(fa):
            m.put("riskFailMarginKept", ke.margin.median(), "{:.4f}")
            m.put("riskFailMarginFail", fa.margin.median(), "{:.4f}")
            m.n("riskFailNposKept", ke.npte.median())
            m.n("riskFailNposFail", fa.npte.median())
            m.put("riskShiftKept", ke["shift"].median(), "{:.2f}")
            m.put("riskShiftFail", fa["shift"].median(), "{:.2f}")
            triv = int((ke.kk == 5).sum())
            m.n("riskKeptTrivial", triv)
            m.n("riskKeptReduced", int(len(ke) - triv))
            m.n("riskNReduced", int(len(g) - triv))
        m.n("riskBootDraws", 400)
        m.put("riskAlphaPrime", 0.05 / 31, "{:.4f}")

    # ---- deep lattice (seven sequences, 127 subsets) ---------------------
    # Read explicitly, not through the active-set resolver: this is a second,
    # smaller cohort (K and n change together), reported as its own series.
    if os.path.exists(p7 := f"{RES}/null_optimism_k7.csv"):
        _USED.add("null_optimism_k7.csv")
        d7 = robust_csv(p7)
        top = d7[d7.n_candidates == d7.n_candidates.max()].iloc[0]
        m.n("deepNCands", int(top.n_candidates))
        m.put("deepNullMedianTop", float(top.null_median), "{:.4f}")
        m.put("deepNullPNineFiveTop", float(top.null_p95), "{:.4f}")
        near = d7.iloc[(d7.n_candidates - 32).abs().argsort()].iloc[0]
        m.n("deepNCandsNear", int(near.n_candidates))
        m.put("deepNullMedianNear", float(near.null_median), "{:.4f}")
    if os.path.exists(p7 := f"{RES}/subset_lattice_k7.csv"):
        _USED.add("subset_lattice_k7.csv")
        d7 = robust_csv(p7)
        d7 = d7[d7.C == d7.C.iloc[0]]
        m.n("deepFindings", d7.finding.nunique())
        piv = d7.groupby(["finding", "subset"]).auroc.mean().unstack()
        mac = piv.mean(0).sort_values(ascending=False)
        full = max(mac.index, key=lambda c: len(c.split("+")))
        m.n("deepFullRank", int(list(mac.index).index(full)) + 1)
        m.n("deepSubsets", d7.subset.nunique())
    if os.path.exists(f7 := f"{ROOT}/data_local/cohort_frozen_k7.json"):
        import json as _json
        c7 = _json.load(open(f7))["splits"]
        m.n("deepTrainCc", c7["train"]["n_complete_case"])

    # ---- vendor composition, per input set -------------------------------
    # Deliberately read untagged: the point of this file is the *contrast*
    # between input sets, so it holds both and neither is the "active" one.
    if os.path.exists(p := f"{RES}/cohort_vendor_mix.csv"):
        _USED.add("cohort_vendor_mix.csv")
        d = robust_csv(p)
        for s in ("k4", "k5"):
            q = d[(d.seqset == s) & (d.split == "test")].sort_values(
                "n", ascending=False)
            if not len(q):
                continue
            key = "pcVendorK" + {"k4": "four", "k5": "five"}[s]
            m[key] = ", ".join(f"{100 * r.frac:.0f}\\% {r.vendor}"
                               for _, r in q.iterrows() if r.n >= 10)
            m.n(f"{key}N", int(q.n_cohort.iloc[0]))
            ge = q[q.vendor == "GE"]
            m.n(f"{key}Ge", int(ge.n.iloc[0]) if len(ge) else 0)

    # ---- selection optimism ----------------------------------------------
    # The permutation *distribution* in null_optimism.csv supersedes the single
    # permuted draw this file used to carry: one draw of a noisy quantity says
    # nothing about its typical size, and an extreme draw is exactly what a
    # sceptical reader would suspect.  optNull* below therefore come from 100
    # patient-level, label-preserving permutations per candidate count.
    NAME = {2: "Two", 4: "Four", 8: "Eight", 15: "Fifteen", 31: "ThirtyOne",
            127: "OneTwentySeven"}
    if (f := exists("null_optimism.csv")):
        d = robust_csv(f)
        for _, r in d.iterrows():
            t = NAME.get(int(r.n_candidates))
            if t is None:
                continue
            m.put(f"optNull{t}", r.null_median, "{:.4f}")
            m.put(f"optNull{t}P", r.null_p95, "{:.4f}")
            m.put(f"optNull{t}Max", r.null_max, "{:.4f}")
            m.put(f"optObs{t}", r.observed, "{:.4f}")
            m.n(f"optNullPerms", r.n_perms)
        m.put("optFracGeMin", d.frac_null_ge_observed.min(), "{:.2f}")
        m.put("optFracGeMax", d.frac_null_ge_observed.max(), "{:.2f}")

    if (f := exists("selection_optimism.csv")):
        d = robust_csv(f)
        # "null" round-trips as NaN through read_csv; accept either spelling
        d["lattice"] = d["lattice"].fillna("permuted").replace(
            {"null": "permuted"})
        g = d.groupby(["lattice", "n_candidates"]).optimism.mean()
        for k in (2, 4, 8, 15):
            t = NAME[k]
            if ("real", k) in g.index:
                m.put(f"optReal{t}", float(g[("real", k)]), "{:.4f}")
        big = d[d.n_studies == d.n_studies.max()]
        if len(big):
            m.n("optMaxStudies", d.n_studies.max())
            for lat, pre in (("real", "optFullReal"),
                             ("permuted", "optFullNull")):
                s = big[(big.lattice == lat) & (big.n_candidates == 15)]
                if len(s):
                    m.put(pre, float(s.optimism.mean()), "{:.4f}")

    # ---- do the interactions survive their own error bars? ---------------
    if (f := exists("synergy_confidence.csv")):
        d = robust_csv(f).set_index("metric").value
        m.pct("synRawFlipPct", float(d["synergy_raw_sign_flip"]))
        m.pct("synHighConfFlipPct", float(d["synergy_highconf_sign_flip"]))
        m.pct("synResolvedPct", float(d["synergy_frac_resolved"]))
        m.n("synCells", float(d["synergy_n_cells"]))

    # ---- eligibility sensitivity -----------------------------------------
    if (f := exists("eligibility_sensitivity.csv")):
        d = robust_csv(f)
        for _, r in d.iterrows():
            t = {20: "Twenty", 30: "Thirty", 50: "Fifty"}.get(int(r.min_pos))
            if t is None:
                continue
            m.n(f"elig{t}N", r.n_findings)
            m.n(f"elig{t}Distinct", r.n_distinct_best)
            m.n(f"elig{t}FullRank", r.full4_rank)

    # ---- risk-controlled policy and its held-out audit -------------------
    if (f := exists("risk_controlled.csv")):
        d = robust_csv(f)
        m.put("riskPolicyMacro", d.macro_policy.mean())
        m.put("riskAllInputsMacro", d.macro_all_inputs.mean())
        m.put("riskDeltaMacro", d.delta_macro.mean(), "{:+.4f}")
        m.put("riskMeanInputs", d.mean_inputs.mean(), "{:.2f}")
    if (f := exists("risk_validate.csv")):
        d = robust_csv(f)
        # The best-populated finding that still breaks its certificate: evidence
        # that failures are not confined to the rarest labels.
        for dl, t in ((0.01, "Mid"), (0.02, "Hi")):
            s = d[(np.isclose(d.delta, dl)) & (~d.kept_promise)]
            if len(s):
                w = s.loc[s.n_pos_test.idxmax()]
                m.n(f"riskFailMaxNpos{t}", w.n_pos_test)
                m[f"riskFailMaxName{t}".translate(_DIGITS)] = str(w.finding)
    if (f := exists("policy_union.csv")):
        d = robust_csv(f).set_index("n_findings_queried")
        for q, t in ((1, "One"), (2, "Two"), (5, "Five")):
            if q in d.index:
                m.put(f"unionQ{t}", d.loc[q, "mean_union_size"], "{:.2f}")
                m.pct(f"unionAllInputsQ{t}", d.loc[q, "frac_needing_all_four"])

    # ---- natural missingness ---------------------------------------------
    if (f := exists("natural_missingness_nqh.csv")):
        d = robust_csv(f)
        i = d[d.group == "incomplete"]
        c = d[d.group == "complete"]
        if len(i):
            m.n("natIncompleteN", i.n.iloc[0])
            m.put("natIncompleteAuroc", i.macro_auroc.mean())
        if len(c):
            m.put("natCompleteAuroc", c.macro_auroc.mean())
    if (f := exists("natural_cohort_nqh.csv")):
        from common import NSEQ
        d = robust_csv(f)
        s1 = d[d.n_inputs == 1]
        # "complete" means every sequence in the active set, not literally four:
        # hardcoding 4 here silently selected the four-input *patterns* under k5
        comp = d[d.n_inputs == NSEQ]
        full = comp.mean_prevalence.mean()
        if full and len(s1):
            # Study-weighted, because an availability pattern holding two studies
            # and one holding two hundred are not equally informative about what
            # a single-sequence study looks like.  The earlier unweighted mean
            # over patterns let the smallest patterns set the headline.
            w = ((s1.mean_prevalence * s1.n_studies).sum()
                 / max(s1.n_studies.sum(), 1))
            m.put("natPrevRatio", w / full, "{:.1f}")
            m.n("natPrevRatioN", int(s1.n_studies.sum()))
            top = s1.loc[s1.mean_prevalence.idxmax()]
            m.put("natPrevRatioMax", top.mean_prevalence / full, "{:.1f}")
            m.n("natPrevRatioMaxN", int(top.n_studies))
            m["natPrevRatioMaxSubset"] = str(top.subset)
        m.n("natPatterns", d.pattern.nunique())

    # ---- external validity ----------------------------------------------
    if (f := exists("external_macro.csv")):
        d = robust_csv(f)
        av = d[d.axis == "availability"]
        for sh in av["shift"].unique():
            s = av[av["shift"] == sh]
            # A shift whose held-out arm collapsed leaves n_test tiny and every
            # metric NaN.  Emitting the macros anyway once put "a 1-study
            # held-out-vendor test" into the abstract, where both numbers were
            # individually real and the sentence was false.  Refuse instead.
            n_test = int(s.n_test.iloc[0])
            if n_test < 100 or not np.isfinite(s.macro_full.to_numpy()).any():
                raise SystemExit(
                    f"external_macro: shift '{sh}' is degenerate on this input "
                    f"set (n_test={n_test}, all-NaN="
                    f"{not np.isfinite(s.macro_full.to_numpy()).any()}). "
                    f"Re-run external_shift.py for this set, or drop the shift "
                    f"from the manuscript -- do not emit its macros.")
            m.n(f"ext{sh.capitalize()}N", n_test)
            m[f"ext{sh.capitalize()}HeldOut"] = str(s.held_out.iloc[0])
            for meth in s.method.unique():
                tag = "".join(w.capitalize() for w in meth.split("_"))
                q = s[s.method == meth]
                m.put(f"ext{sh.capitalize()}{tag}Full", q.macro_full.mean())
                m.put(f"ext{sh.capitalize()}{tag}Worst",
                      q.macro_worst_subset.mean())

    os.makedirs(f"{ROOT}/paper", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("% Generated by src/make_numbers.py -- do not edit by hand.\n")
        for k in sorted(m):
            fh.write(f"\\newcommand{{\\{k}}}{{{m[k]}}}\n")
    print(f"wrote {len(m)} macros to {OUT}")
    from common import SEQSET
    tagged = sorted(x for x in _USED if f"_{SEQSET}." in x)
    plain = sorted(x for x in _USED if f"_{SEQSET}." not in x)
    print(f"  set={SEQSET}: {len(tagged)} tagged, {len(plain)} untagged")
    if plain:
        print("  untagged (k4 or set-independent): "
              + ", ".join(plain[:10]) + ("..." if len(plain) > 10 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
