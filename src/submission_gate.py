"""The checks that must pass before the Submit button, run against the PDF.

`verify_numbers.py` gates the *sources*: every displayed number traces to a CSV
and regeneration is a fixed point.  This gates the *artefact*, because several
errors in this project's history were invisible in the sources and visible only
in the rendered document -- a caption pinned to the previous input set, a macro
that rendered a real number under a false sentence, a stale PDF that a clean
LaTeX log did not contradict.

Every check here corresponds to something that has actually gone wrong, or to a
venue requirement that cannot be fixed after submission.

Usage:  QS_SEQSET=k5 python src/submission_gate.py
"""
import os
import re
import shutil
import subprocess
import sys

import pypdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = f"{ROOT}/paper/main.pdf"
TEX = f"{ROOT}/paper"

# Values that belong to the four-sequence input set.  If one of these renders,
# a k4 result is sitting under five-sequence prose.
K4_ONLY = {
    "12,412": "k4 complete-case training cohort",
    "4,187": "k4 held-out GE cohort",
    "0.0096": "k4 cross-fitted gain",
    "0.1649": "k4 readout-shuffle drop",
    "13 of 15": "k4 risk-audit result",
    # values from the single-layer (leaky) selection estimator, superseded by
    # the strictly nested implementation -- rendering any of them would revive
    # the number the nested recompute retired
    "0.0277": "leaky-estimator headline gain",
    "0.0217": "leaky-estimator CI bound",
    # 0.0337 was the leaky-estimator CI bound; after the round-5 fixed
    # permutation rerun it is also the legitimate K=31 null median, so the
    # string ban is retired -- provenance is enforced by verify_numbers.
    "0.0397": "leaky-estimator naive gain",
    # 0.0071 was the leaky-estimator SWI matched increment; after the round-6
    # rerun it is also the legitimate k=2 QuerySeq-vs-uniform budget-matched
    # mean (\dNqhQueryseqVsUniformBudgetMean), so the string ban is retired --
    # provenance is enforced by verify_numbers.
}
# Anything that would break double-blind review.  Stored base64-encoded so
# the shipped source tree contains no literal occurrence of any identifying
# string -- a reviewer grepping the supplement for these must find nothing,
# including inside the checker that enforces it.
import base64 as _b64
IDENTIFYING = [_b64.b64decode(x).decode() for x in ['bGk0NTMz', 'cHVyZHVl', 'Z2F1dHNjaGk=', 'emh1b2FubGk=', 'cmNhYw==', 'L3NjcmF0Y2gv', 'L2hvbWUv', 'Z2l0aHViLmNvbQ==', 'YWNrbm93bGVkZw==']]
# Claims the paper must keep making, in the weaker form we settled on.
MUST_APPEAR = {
    # round 6: the primary rule is the cross-seed mean (9/12 at delta=0.01);
    # the retired majority-vote count (7/12) must stay disclosed as such.
    r"\b9\b[^.]{0,40}\b12\b": "the risk audit's 9-of-12 cross-seed-mean result",
    r"majority-vote": "the disclosure that an earlier draft used a "
                      "majority-vote count",
    r"class-degenerate": "the GoAT presence-endpoint failure",
    r"BraTS|GoAT": "the second-dataset replication",
    r"double-blind": "the anonymous TMLR title block",
}

# Phrases that describe the *previous* input set.  Checking rendered numbers is
# not enough: a caption reading "all four inputs" beside a five-sequence table
# is a false sentence built entirely from correct numbers, and that is the
# failure mode that survived every numeric audit.
#
# BraTS genuinely has four modalities and a 15-subset lattice, so these cannot
# be grepped globally -- the GoAT section and its table are excluded by
# splitting the text at that section heading.
STALE_SEMANTICS = [
    "all four inputs", "all four sequences", "these four inputs",
    "needing all four", "requiring all four", "of 4 inputs",
    "15 fixed policies", "15 candidates", "all 15 patterns",
    "dropping coronal T2", "coronal T2", "T2cor",
    "at least as good as every router",
    "blood products are read off T1,",
    # second audit round: stale rankings, stale counts, retired claims
    "all-four ranked", "against reading all four", "reading all four",
    "15 in the test cohort", "15 findings",
    "roughly twice", "about twice", "twice the real", "twice the cross-fitted",
    "mass away from zero means", "value is not additive",
    # reviewer round 3: retired framings
    "Without Selection Bias",          # the old title's overclaim
    "measures physics rather than",    # softened to construct validity
    "roughly twice",
    # round-4 reviewer: the stress test is not a coverage experiment
    "kept its promise", "kept the promise", "did not deliver",
    "insufficiently conservative", "exact Clopper",
    "over the best fixed policy",      # comparators must be named precisely
    # round-5 review: retired statistics and metrics
    "second winner",                   # the erroneous Bonferroni account
    "buys something only",             # the k=1-only claim its own table refutes
    "for a majority of finding",       # 224/525 = 42.7% is not a majority
    "prospectively validated, is the ranking",  # round-6: partial validation only
    "inherit exactly those optimistic bounds",  # direction not identified
    "conditioning on the clearing event",
    "0.713",                           # the mixed shrinkage/collapse average
    "0.0086",                          # leaky risk-selector policy delta
    "collapses the model to chance",
    "below the null median at every",
    "three directional",               # B3 is a corollary, count is two
]


def mrrate_text(pages):
    """Rendered text with the second-dataset section removed.

    The GoAT corpus really does have four modalities and 15 subsets, so its
    prose would trip every stale-semantics check if it were included.
    """
    t = " ".join(pages)
    lo = t.find("A prospectively specified construct validation")
    if lo < 0:
        return t
    hi = t.find("Audit of the nominal bound", lo)
    return t[:lo] + (t[hi:] if hi > lo else "")


def page_texts():
    return [" ".join(p.extract_text().split())
            for p in pypdf.PdfReader(PDF).pages]


def main():
    fail, warn = [], []
    # ---- 1. the source-level gate still passes --------------------------
    r = subprocess.run([sys.executable, f"{ROOT}/src/verify_numbers.py"],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        fail.append("verify_numbers.py does not pass:\n      "
                    + "\n      ".join(r.stdout.strip().splitlines()[-3:]))

    # ---- 2. build the artefact we are about to audit ---------------------
    # Rather than test whether main.pdf looks stale, produce it here.  A failed
    # pdflatex leaves the previous PDF in place beside an innocent-looking log,
    # so auditing that file reports the *previous* draft as clean -- which has
    # already happened in this project.  Building inside the gate makes "the
    # PDF matches the sources" true by construction rather than by timestamp.
    if not shutil.which("pdflatex"):
        print("FAILED: pdflatex not on PATH -- run `module load texlive`")
        return 1
    before = os.path.getmtime(PDF) if os.path.exists(PDF) else 0
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main"]):
        subprocess.run(cmd, cwd=TEX, capture_output=True, text=True)
    if not os.path.exists(PDF) or os.path.getmtime(PDF) <= before:
        errs = [l for l in open(f"{TEX}/main.log", errors="ignore")
                if l.startswith("! ")][:3]
        print("FAILED: pdflatex did not produce a new PDF\n      "
              + "      ".join(errs))
        return 1
    log = open(f"{TEX}/main.log", errors="ignore").read()
    for pat, why in ((r"^! ", "LaTeX error"),
                     (r"Undefined control sequence", "undefined macro"),
                     (r"Citation .* undefined", "undefined citation"),
                     (r"Reference .* undefined", "undefined reference")):
        n = len(re.findall(pat, log, re.M))
        if n:
            fail.append(f"{n} x {why} in the build log")

    pages = page_texts()
    text = " ".join(pages)
    low = text.lower()

    # ---- 3. no other input set's numbers rendered ------------------------
    for v, why in K4_ONLY.items():
        if v in text:
            fail.append(f"{v} ({why}) is rendered in a five-sequence paper")

    # ---- 4. double-blind ------------------------------------------------
    for w in IDENTIFYING:
        if w in low:
            fail.append(f"identifying string '{w}' appears in the PDF")
    meta = dict(pypdf.PdfReader(PDF).metadata or {})
    for k, v in meta.items():
        if any(w in str(v).lower() for w in IDENTIFYING):
            fail.append(f"PDF metadata {k} leaks identity: {v}")
    if "anonymous authors" not in low:
        fail.append("title block is not the anonymous one -- compiled with "
                    "[accepted] or [preprint]?")

    # ---- 5. the claims we committed to keep ------------------------------
    for pat, why in MUST_APPEAR.items():
        if not re.search(pat, text, re.I):
            fail.append(f"{why} is no longer stated in the paper")

    # ---- 6. every cross-reference resolved -------------------------------
    if "??" in text:
        fail.append("an unresolved '??' cross-reference is rendered")

    # ---- 6b. no prose describing the previous input set ------------------
    mr = mrrate_text(pages)
    for phrase in STALE_SEMANTICS:
        if phrase.lower() in mr.lower():
            fail.append(f"stale semantics outside the GoAT section: "
                        f"“{phrase}”")

    # ---- 6c. machine-checked invariants of the active analysis -----------
    # These compare the manuscript against the generators rather than against
    # a remembered value, so they keep working when the input set changes.
    sys.path.insert(0, f"{ROOT}/src")
    from common import NSEQ, result_path                      # noqa: E402
    import pandas as pd                                       # noqa: E402
    from scipy import stats                                   # noqa: E402

    n_cand = 2 ** NSEQ - 1
    lat = result_path("subset_lattice.csv")
    if lat is not None:
        d = pd.read_csv(lat)
        if d.subset.nunique() != n_cand:
            fail.append(f"lattice has {d.subset.nunique()} candidates but the "
                        f"active set implies {n_cand}")
        widest = max(d.subset, key=lambda x: len(x.split("+")))
        if len(widest.split("+")) != NSEQ:
            fail.append(f"widest lattice subset '{widest}' does not span the "
                        f"{NSEQ} active sequences")

    # the Bonferroni denominator the manuscript states must be the one the
    # risk generator actually divided by
    rc = open(f"{ROOT}/src/risk_controlled.py").read()
    if "args.alpha / len(subs)" not in rc:
        fail.append("risk_controlled.py no longer divides alpha by len(subs); "
                    "the manuscript's multiplicity claim needs rechecking")
    if f"\\riskNCandidates" not in open(f"{TEX}/main.tex").read():
        warn.append("main.tex states the multiplicity denominator literally "
                    "rather than through \\riskNCandidates")

    # exact Clopper-Pearson, recomputed here and compared to what is rendered
    cov = result_path("risk_coverage.csv")
    if cov is not None:
        c = pd.read_csv(cov)
        for _, r in c.iterrows():
            k, n = int(r.n_kept), int(r.n_findings)
            lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
            hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
            if (abs(r.coverage_lo - lo) > 5e-4
                    or abs(r.coverage_hi - hi) > 5e-4):
                fail.append(
                    f"delta={r.delta}: stored CI [{r.coverage_lo:.4f}, "
                    f"{r.coverage_hi:.4f}] is not the exact Clopper-Pearson "
                    f"interval for {k}/{n}, which is [{lo:.4f}, {hi:.4f}]")

    # ---- 6b2. "pre-registered" is reserved for the GoAT section ----------
    # Round-3 review: our provenance is a version-controlled planning record,
    # not third-party registration, so outside the section that defines the
    # distinction the paper says "prospectively specified".
    for w in ("pre-registered", "pre-registration", "preregistered"):
        if w in mr.lower():
            fail.append(f"'{w}' used outside the replication section -- say "
                        f"'prospectively specified'")
    if "certificate" in mr.lower() or "certified" in mr.lower():
        fail.append("certificate language outside the GoAT carve-out -- the "
                    "procedure is a nominal bound, not a certificate")

    # ---- 6c2. generator-level invariants from the second audit round -----
    # The natural-missingness caption must quote the SAME macro the body text
    # uses -- recomputing it in the table generator is how 1.1x got printed
    # beside prose saying 1.9x.
    app = open(f"{TEX}/tables/natural.tex").read()
    if "natural" in app.lower() and "\\natPrevRatio" not in app:
        fail.append("appendix natural-missingness caption does not quote "
                    "\\natPrevRatio -- it is recomputing the ratio locally")
    # The retired non-additivity panel title must stay retired.
    figsrc = open(f"{ROOT}/src/make_figures.py").read()
    if "Non-additivity" in figsrc:
        fail.append("make_figures still titles panel C 'Non-additivity' -- "
                    "the bootstrap analysis retired that claim")
    # The budget frontier must tick every budget in its data, so the k=NSEQ
    # endpoint cannot silently drop off the axis again.
    if "ax.set_xticks(sorted(d.budget.dropna().unique()))" not in figsrc:
        fail.append("fig_frontier no longer pins xticks to the data's budgets")
    if "set_xticks([1, 2, 3, 4])" in figsrc:
        fail.append("a hardcoded four-budget tick list is back in make_figures")
    if 'have("selection_optimism.csv")' in figsrc:
        fail.append("fig_optimism reads the four-sequence-era "
                    "selection_optimism.csv again")
    if 'have("method_macro.csv")' in figsrc:
        fail.append("fig_frontier reads the shared-head run while the tables "
                    "use nqh -- configurations must match")
    if "sys.exit(main())" not in figsrc:
        fail.append("make_figures no longer propagates figure failures as a "
                    "nonzero exit -- the clean-extract gate would go blind")
    if 'glob.glob(f"{ROOT}/data_local/gates_queryseq_s0.npy")' in figsrc:
        fail.append("fig_routing reads the untagged k4-era gates file again")
    if 'sy = sy[sy.context != "none"]' not in figsrc:
        fail.append("fig_atlas panel C lost the empty-context filter")

    # ---- 6d. figures were drawn from the active input set ----------------
    figs = f"{ROOT}/figures"
    newest_result = max(
        (os.path.getmtime(os.path.join(f"{ROOT}/results", x))
         for x in os.listdir(f"{ROOT}/results") if x.endswith(".csv")),
        default=0)
    for g in sorted(os.listdir(figs)):
        if g.endswith(".pdf") and os.path.getmtime(f"{figs}/{g}") < newest_result:
            warn.append(f"{g} is older than the newest results CSV -- rerun "
                        f"src/make_figures.py")

    # ---- 7. the quarantined artefact cannot be read ----------------------
    bad = f"{ROOT}/results/query_pathways_k5nqh.csv"
    if os.path.exists(bad):
        fail.append("results/query_pathways_k5nqh.csv is live again -- it "
                    "came from a query_head=False checkpoint")

    # ---- 8. length, against the TMLR reviewer-deadline threshold ---------
    try:
        refs = next(i for i, t in enumerate(pages)
                    if re.search(r"\bReferences\b", t))
    except StopIteration:
        fail.append("no References section found")
        refs = len(pages)
    if refs > 12:
        warn.append(f"main text is {refs} pages; TMLR gives reviewers a "
                    f"longer deadline above 12, which slows the decision")

    # ---- 9. every figure the text includes actually exists ---------------
    tex = open(f"{TEX}/main.tex").read()
    for g in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
        if not os.path.exists(os.path.normpath(os.path.join(TEX, g))):
            fail.append(f"figure missing on disk: {g}")

    print(f"main text     : {refs} pages   (total {len(pages)})")
    print(f"figures       : {len(re.findall(r'includegraphics', tex))} included")
    print(f"PDF metadata  : {sorted(meta)}")
    for w in warn:
        print(f"warning       : {w}")
    if fail:
        print("\nFAILED:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\nOK: submission gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
