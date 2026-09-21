"""Generate the paper's tables from results/*.csv.

Tables are generated rather than typed for the same reason the numbers are: a
hand-maintained table drifts from its source the first time an experiment is
rerun.  Everything here reads a CSV and writes LaTeX into paper/.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import robust_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PAPER = f"{ROOT}/results", f"{ROOT}/paper"

LABEL = {"uniform": "Mean fusion (no routing)",
         "finding": "Finding-only gate",
         "patient": "Patient-only router",
         "label_id": "Learned finding-ID router",
         "queryseq": "QuerySeq (query $\\times$ patient)"}
ORDER = ["uniform", "finding", "patient", "label_id", "queryseq"]
# Narrow (single-column) tables cannot fit the descriptive labels above.
SHORT = {"uniform": "Mean fusion", "finding": "Finding-only",
         "patient": "Patient-only", "label_id": "Finding-ID",
         "queryseq": "QuerySeq"}


_USED = set()


def have(f):
    """Resolve a results file, preferring the active input set's tagged copy.

    This mirrors make_numbers.exists() deliberately.  Without it the tables
    silently rendered the four-sequence results beside five-sequence prose --
    Table 12 quoted a held-out-GE cohort that has one study at K=5, and the
    2x2 table quoted the K=4 pathway cut.  Both numbers were real; both
    captions were false.
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


PROV = {
    "lattice": "train split, complete-case, 35 findings (min-pos 20), linear "
               "head, 3 seeds, 5-fold patient CV",
    "availability": "train split, all studies (n=27{,}266), 37 findings, "
                    "identical head and folds across rows",
    "optimism": "train split, complete-case; nested selection; 100 "
                "patient-level permutations",
    "methods": "test split, complete-case; independent per-finding readouts "
               "(nqh checkpoints); 12 findings (min-pos 20); 3 seeds",
    "methods-qhead": "test split, complete-case; shared query-conditioned "
                     "head (qh checkpoints); 12 findings (min-pos 20); 3 seeds",
    "query": "test split, complete-case; shared query-conditioned head; "
             "min-pos 20; 3 seeds",
    "pathways": "test split, complete-case; shared query-conditioned head "
                "(qh checkpoints); 12 findings; 3 seeds",
    "goat": "BraTS-GoAT, all 1{,}351 cases; linear head; 3 seeds, 5-fold "
            "patient CV",
    "risk": "train-internal nested selection; bound at "
            "$\\alpha/31$; 400 bootstrap draws (under-resolved, see text)",
    "coverage": "train-internal nested selection; heads refit on train, applied "
                "once to test; 12 findings",
    "union": "policies from the risk selector, seed 0 modal subset per "
             "finding; 300 random query sets per size",
    "natural": "test split, all studies; nqh checkpoints; min-pos 20",
    "external": "custom vendor re-split; retrained; nqh configuration",
}


def with_provenance(key, table):
    """Provenance footnotes retired from the rendered tables: the run
    configuration each table rests on is stated in the surrounding prose and
    in REPRODUCE.md, and the footers crowded the layout."""
    return table


def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&")


def tab_lattice():
    """Fixed-subset ranking: reading everything is not the best fixed policy."""
    f = have("subset_lattice.csv")
    if not f:
        return ""
    d = robust_csv(f)
    d = d[d.C == d.C.iloc[0]]
    piv = d.groupby(["finding", "subset"]).auroc.mean().unstack()
    macro = piv.mean(0).sort_values(ascending=False)
    k = {s: len(s.split("+")) for s in macro.index}
    # the all-inputs subset of whatever set is active, not a pinned string

    full = max(macro.index, key=lambda x: len(x.split("+")))
    best = macro.index[0]
    dropped = [t for t in full.split("+") if t not in best.split("+")]
    rows = []
    for i, (s, v) in enumerate(macro.items()):
        mark = "\\textbf{" if s == full else ""
        end = "}" if s == full else ""
        rows.append(f"{i + 1} & {mark}{esc(s).replace('+', '{+}')}{end} & "
                    f"{k[s]} & {mark}{v:.4f}{end} \\\\")
    body = "\n".join(rows)
    n_full = len(full.split("+"))
    # State what the ranking actually shows for the active input set. At K=4
    # the best fixed subset dropped coronal T2; at K=5 reading everything wins.
    # Either way the paper's claim is about per-finding selection beating the
    # best *fixed* subset, so the caption must not assert the K=4 outcome.
    if full == macro.index[0]:
        gap = float(macro.iloc[0] - macro.iloc[1])
        verdict = (f"Reading all {n_full} inputs (bold) is the best fixed "
                   f"policy here, but only by {gap:.4f} over the best "
                   f"{len(macro.index[1].split('+'))}-input subset")
    else:
        drop_txt = " and ".join(esc(x) for x in dropped)
        verdict = (f"Reading all {n_full} inputs (bold) is not the best fixed "
                   f"policy: the top-ranked subset drops {drop_txt}")
    return f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Every fixed input subset of the primary cohort, ranked by macro
AUROC. Bold marks the full input set.}}
\\label{{tab:lattice}}
\\begin{{tabular}}{{rlcc}}
\\toprule
\\# & subset & $|S|$ & macro AUROC \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def _method_block(tag, caption, label):
    from common import NSEQ
    f = have(f"method_macro{tag}.csv")
    if not f:
        return ""
    d = robust_csv(f)
    av = d[d.axis == "availability"]
    bu = d[d.axis == "budget"]
    rows = []
    for m in ORDER:
        s = av[av.method == m]
        if s.empty:
            continue
        b = bu[bu.method == m]
        k1 = b[b.budget == 1].macro_auroc.mean()
        k2 = b[b.budget == 2].macro_auroc.mean()
        rows.append(
            f"{LABEL[m]} & {s.macro_full.mean():.4f} "
            f"{{\\scriptsize$\\pm${s.macro_full.std():.4f}}} & "
            f"{s.macro_avg_subset.mean():.4f} & "
            f"{s.macro_worst_subset.mean():.4f} & "
            f"{k1:.4f} & {k2:.4f} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[t]
\\centering\\small
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lccccc}}
\\toprule
& \\multicolumn{{3}}{{c}}{{availability}} & \\multicolumn{{2}}{{c}}{{budget}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-6}}
gate conditioned on & all {NSEQ} & avg & worst & $k{{=}}1$ & $k{{=}}2$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def tab_external():
    from common import NSEQ
    f = have("external_macro.csv")
    if not f:
        return ""
    d = robust_csv(f)
    av = d[d.axis == "availability"]
    ve = av[av["shift"] == "vendor"]
    if ve.empty:
        return ""
    n = int(ve.n_test.iloc[0])
    held = esc(ve.held_out.iloc[0])
    rows = []
    for m in ORDER:
        s = ve[ve.method == m]
        if s.empty:
            continue
        rows.append(f"{LABEL[m]} & {s.macro_full.mean():.4f} & "
                    f"{s.macro_avg_subset.mean():.4f} & "
                    f"{s.macro_worst_subset.mean():.4f} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Held-out-vendor evaluation: {n:,} unseen {held} studies.}}
\\label{{tab:external}}
\\begin{{tabular}}{{lccc}}
\\toprule
gate conditioned on & all {NSEQ} & avg subset & worst subset \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tab_query():
    f = have("query_diagnostics.csv")
    g = have("query_routing_agreement.csv")
    if not f or not g:
        return ""
    d = robust_csv(f).groupby("variant")[["macro_auroc"]].agg(["mean", "std"])
    r = robust_csv(g).groupby("variant").routing_cos_vs_base.mean()
    nice = {"para:sentence": "\\emph{There is X.} (training form)",
            "para:term": "\\emph{X} (clinical term)",
            "para:name": "ontology label",
            "para:question": "\\emph{Is there evidence of X?}",
            "para:definition": "one-clause definition",
            "shuffled": "\\textbf{shuffled} query--finding map",
            "counterfactual": "\\textbf{counterfactual} (other finding)"}
    rows = []
    for k in ["para:sentence", "para:term", "para:name", "para:question",
              "para:definition", "shuffled", "counterfactual"]:
        if k not in d.index:
            continue
        rows.append(f"{nice[k]} & {d.loc[k, ('macro_auroc', 'mean')]:.4f} "
                    f"{{\\scriptsize$\\pm${d.loc[k, ('macro_auroc', 'std')]:.4f}}}"
                    f" & {r.get(k, float('nan')):.4f} \\\\")
        if k == "para:definition":
            rows.append("\\midrule")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Query substitution at inference on one trained model.}}
\\label{{tab:query}}
\\begin{{tabular}}{{lcc}}
\\toprule
query given to the model & macro AUROC & routing cos. \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tab_optimism():
    """Observed optimism against its permutation distribution.

    Deliberately does *not* report P(null >= observed).  That column reads as a
    significance test and its value of 1.00 invites exactly the wrong reading --
    optimism is not a statistic one wants to be large, and the null is a
    search-only reference rather than a hypothesis being tested.
    """
    f = have("null_optimism.csv")
    if not f:
        return ""
    d = robust_csv(f).sort_values("n_candidates")
    rows = []
    for _, r in d.iterrows():
        ratio = r.observed / r.null_median if r.null_median else float("nan")
        rows.append(f"{int(r.n_candidates)} & {r.observed:.4f} & "
                    f"{r.null_median:.4f} & {r.null_p95:.4f} & {ratio:.2f} \\\\")
    body = "\n".join(rows)
    n = int(d.n_perms.iloc[0])
    return f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Observed optimism against the permutation reference at each
candidate-set size.}}
\\label{{tab:optimism}}
\\begin{{tabular}}{{ccccc}}
\\toprule
$K$ & observed & null median & null p95 & observed / null \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tab_availability():
    """Decomposing the availability signal into site and residual."""
    f = have("availability_decomposition.csv")
    if not f:
        return ""
    d = robust_csv(f).set_index("model")
    label = {"mask": "availability mask alone",
             "site": "site alone (vendor, field, station)",
             "site+mask": "site $+$ availability"}
    rows = []
    for k in ("mask", "site", "site+mask"):
        if k in d.index:
            rows.append(f"{label[k]} & {d.loc[k, 'macro_auroc']:.4f} \\\\")
    rows.append("\\midrule")
    for k in d.index:
        if str(k).startswith("mask|"):
            rows.append(f"availability within {str(k).split('|')[1]} only "
                        f"($n{{=}}{int(d.loc[k, 'n']):,}$) & "
                        f"{d.loc[k, 'macro_auroc']:.4f} \\\\")
    p = "mask-perm-within-site"
    if p in d.index:
        rows.append(f"availability permuted within site & "
                    f"{d.loc[p, 'macro_auroc']:.4f} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Decomposition of the availability signal.}}
\\label{{tab:availability}}
\\begin{{tabular}}{{lc}}
\\toprule
predictor & macro AUROC \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tab_pathways():
    """The 2x2: which pathway does the query actually act through?"""
    f = have("query_pathways.csv")
    if not f:
        return ""
    d = robust_csv(f)
    g = d.groupby("condition")[["macro_auroc", "macro_auprc",
                                "routing_cos_vs_base"]].agg(["mean", "std"])
    order = [("both correct", "correct", "correct"),
             ("gate shuffled", "\\textbf{shuffled}", "correct"),
             ("readout shuffled", "correct", "\\textbf{shuffled}"),
             ("both shuffled", "\\textbf{shuffled}", "\\textbf{shuffled}")]
    base = g.loc["both correct", ("macro_auroc", "mean")]
    rows = []
    for c, gq, hq in order:
        if c not in g.index:
            continue
        rows.append(
            f"{gq} & {hq} & {g.loc[c, ('macro_auroc', 'mean')]:.4f} & "
            f"{base - g.loc[c, ('macro_auroc', 'mean')]:+.4f} & "
            f"{g.loc[c, ('macro_auprc', 'mean')]:.4f} & "
            f"{g.loc[c, ('routing_cos_vs_base', 'mean')]:.4f} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[t]
\\centering\\small
\\caption{{The $2\\times2$ query-pathway intervention.}}
\\label{{tab:pathways}}
\\begin{{tabular}}{{llcccc}}
\\toprule
gate query & readout query & AUROC & $\\Delta$ & AUPRC & route cos \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def tab_goat():
    """The prospectively specified second-dataset replication, both endpoints."""
    import os
    f = f"{RES}/goat_shapley.csv"          # one input set; never suffixed
    if not os.path.exists(f):
        return ""
    _USED.add("goat_shapley.csv")
    sh = robust_csv(f)
    mods = ["T1n", "T1c", "T2w", "T2f"]
    blocks = []
    for kind, lab in (("burden", "high vs low burden (primary, amended)"),
                      ("presence", "presence (as first specified, degenerate)")):
        q = sh[sh.target_kind == kind]
        if not len(q):
            continue
        piv = q.pivot_table(index="target", columns="modality",
                            values="shapley")
        blocks.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{lab}}}}} \\\\")
        for t in ("ET", "SNFH", "NETC"):
            if t not in piv.index:
                continue
            top = piv.loc[t].idxmax()
            cells = " & ".join(
                (f"\\textbf{{{piv.loc[t, m]:.4f}}}" if m == top
                 else f"{piv.loc[t, m]:.4f}") if m in piv.columns else "--"
                for m in mods)
            blocks.append(f"\\quad {t} & {cells} & {top} \\\\")
    body = "\n".join(blocks)
    return f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Exact Shapley values on BraTS-GoAT under both endpoints.}}
\\label{{tab:goat}}
\\begin{{tabular}}{{lccccl}}
\\toprule
target & T1n & T1c & T2w & T2f & largest \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tab_coverage():
    """Does the nominal bound hold on data that did not produce it?"""
    f = have("risk_coverage.csv")
    if not f:
        return ""
    d = robust_csv(f)
    rows = []
    for _, r in d.iterrows():
        star = "$^\\ast$" if abs(r.delta - 0.01) < 1e-9 else ""
        inst = (f"{int(r.n_kept_instances)}/{int(r.n_instances)}"
                if "n_kept_instances" in d.columns else "--")
        rows.append(
            f"{r.delta:.3f}{star} & {int(r.n_kept)}/{int(r.n_findings)} & "
            f"{r.coverage:.2f} [{r.coverage_lo:.2f}, {r.coverage_hi:.2f}] & "
            f"{inst} & "
            f"{r.mean_k:.2f} & {r.mean_test_delta:+.4f} & "
            f"{r.worst_test_delta:+.4f} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[t]
\\centering\\small
\\caption{{Held-out audit of the nominal bound by tolerance. A finding clears iff its cross-seed mean held-out difference is $\\ge-\\delta$; the seed-instance column counts per-seed policy instances clearing individually.}}
\\label{{tab:coverage}}
\\begin{{tabular}}{{ccccccc}}
\\toprule
$\\delta$ & cleared/total & fraction [heuristic 95\\% CI] & seed inst. &
mean $|S|$ & mean $\\Delta^{{\\text{{test}}}}$ & worst \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def tab_risk():
    from common import NSEQ
    f = have("risk_controlled.csv")
    p = have("risk_policy.csv")
    if not f or not p:
        return ""
    d = robust_csv(f)
    P = robust_csv(p)
    dist = P[P.seed == 0].k.value_counts().sort_index()
    sizes = " / ".join(f"{int(k)}:{int(v)}" for k, v in dist.items())
    return f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Bound-selected minimal policies at $\\delta{{=}}0.01$; sizes
($|S|$:count) {sizes}.}}
\\label{{tab:risk}}
\\begin{{tabular}}{{lcc}}
\\toprule
& macro AUROC & mean $|S|$ \\\\
\\midrule
read all {NSEQ} inputs & {d.macro_all_inputs.mean():.4f} & {NSEQ:.2f} \\\\
bound-selected minimal policy & {d.macro_policy.mean():.4f} &
{d.mean_inputs.mean():.2f} \\\\
\\midrule
difference & {d.delta_macro.mean():+.4f} & \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tab_union():
    from common import NSEQ
    f = have("policy_union.csv")
    if not f:
        return ""
    d = robust_csv(f)
    rows = [f"{int(r.n_findings_queried)} & {r.mean_union_size:.2f} & "
            f"{100 * r.frac_needing_all_four:.0f}\\% \\\\"
            for _, r in d.iterrows()]
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Union of per-finding policies as findings are queried jointly.}}
\\label{{tab:union}}
\\begin{{tabular}}{{ccc}}
\\toprule
findings queried & mean $|\\bigcup_f S_f|$ & needing all {NSEQ} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tab_natural():
    f = have("natural_missingness_nqh.csv")
    c = have("natural_cohort_nqh.csv")
    if not f:
        return ""
    d = robust_csv(f)
    rows = []
    for m in ORDER:
        s = d[d.method == m]
        if s.empty:
            continue
        g = s.set_index("group")
        def cell(grp, col):
            v = s[s.group == grp][col]
            return f"{v.mean():.4f}" if len(v) else "--"
        rows.append(f"{SHORT[m]} & {cell('complete', 'macro_auroc')} & "
                    f"{cell('incomplete', 'macro_auroc')} & "
                    f"{cell('incomplete', 'macro_auprc')} \\\\")
    body = "\n".join(rows)
    ninc = int(d[d.group == "incomplete"].n.iloc[0]) if len(
        d[d.group == "incomplete"]) else 0
    extra = ""
    if c:
        cd = robust_csv(c)
        # The ratio is quoted from the same macro the body text uses
        # (\natPrevRatio, study-weighted, complete = all NSEQ inputs).  An
        # earlier version recomputed it here against n_inputs == 4 -- under the
        # five-sequence set that is the four-input *patterns*, not complete
        # studies, and the caption printed 1.1x beside prose saying 1.9x.
        extra = (" Studies carrying one input have \\natPrevRatio{}$\\times$ "
                 "the mean finding prevalence of complete studies, so this "
                 f"cohort carries selection effects and is not comparable to "
                 f"the masked results.")
    return f"""\\begin{{table}}[t]
\\centering\\small
\\caption{{Performance on naturally incomplete
studies ($n{{=}}{ninc:,}$).{extra}}}
\\label{{tab:natural}}
\\begin{{tabular}}{{lccc}}
\\toprule
& complete & incomplete & incomplete \\\\
gate conditioned on & AUROC & AUROC & AUPRC \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    """Split the tables into main text and appendix.

    The split is the one pre-committed in CLAIMS.md before the text was
    written, so it reflects the argument rather than whatever happened to
    overflow: the main text carries one table per claim, and every
    per-finding breakdown, sweep and sensitivity variant goes to the
    appendix.  TMLR does not cap the appendix, so nothing is dropped.
    """
    main_tables = {
        "lattice": tab_lattice(),
        "availability": tab_availability(),
        "optimism": tab_optimism(),
        "methods": _method_block(
            "_nqh",
            "Routing methods with independent per-finding readouts.",
            "tab:methods"),
        "pathways": tab_pathways(),
        "goat": tab_goat(),
        "risk": tab_risk(),
    }
    appendix_tables = {
        "methods-qhead": _method_block(
            "",
            "Routing methods with a shared query-conditioned head.",
            "tab:methods-qhead"),
        "query": tab_query(),
        "coverage": tab_coverage(),
        "union": tab_union(),
        "natural": tab_natural(),
        "external": tab_external(),
    }
    # Every table is its own file so each can be \input exactly where the text
    # discusses it, in the main body or in its lettered appendix section.
    os.makedirs(f"{PAPER}/tables", exist_ok=True)
    for key, content in {**main_tables, **appendix_tables}.items():
        content = with_provenance(key, content)
        if key in appendix_tables or key in ("lattice", "availability",
                                             "optimism", "goat", "methods",
                                             "risk"):
            # appendix tables sit exactly under the text that introduces
            # them; floating them to page tops stacked table on table
            content = (content
                       .replace("\\begin{table*}[t]", "\\begin{table}[H]")
                       .replace("\\end{table*}", "\\end{table}")
                       .replace("\\begin{table}[t]", "\\begin{table}[H]"))
        with open(f"{PAPER}/tables/{key}.tex", "w") as fh:
            fh.write("% Generated by src/make_tables.py -- do not edit.\n")
            fh.write(content)
    print(f"wrote {len(main_tables) + len(appendix_tables)} tables to "
          f"{PAPER}/tables/")


if __name__ == "__main__":
    main()
