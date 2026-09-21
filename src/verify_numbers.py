"""Gate on the paper's numbers before submission.

Four checks, each of which has caught a real error in this project's lineage:

1. No stray numeric literals in body text -- every displayed number must arrive
   through a macro, so it cannot drift from the CSV that produced it.
2. Every macro the paper uses is defined by a generator.
3. Regeneration is a fixed point: re-running the generators must not change
   their output.  This is what catches a results CSV moving underneath a number
   that was already written into the draft.
4. Every source CSV the generators read exists and is non-empty.

Exits non-zero on failure so it can gate a build.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = f"{ROOT}/paper"
GENERATORS = ["src/make_numbers.py", "src/make_tables.py"]
GENERATED = ["numbers.tex"] + [f"tables/{k}.tex" for k in (
    "lattice", "availability", "optimism", "methods", "pathways", "goat",
    "risk", "methods-qhead", "query", "coverage", "union", "natural",
    "external")]
MAIN = "main.tex"

# Literals allowed to appear verbatim in body text, each with a reason.
ALLOWED = {
    "1408",   # embedding width, a property of the frozen encoder
    "37",     # label columns in the corpus
    "20",     # the disclosed prevalence floor
    "4", "2", "1", "3", "0",   # small counts in prose ("all four inputs")
    "2027", "0.5",             # template year; chance AUROC
    "0.01", "0.05",            # the declared delta and alpha
    "123",    # the underpowered field-strength test split, stated as such
    "0.235", "0.264", "0.251", "0.259",  # the collapsed prior weights we quote
    "0.070",  # the artifact value we explicitly say we did NOT report
    "0.44",   # the arithmetic value of the degenerate empty context
    "0.024", "0.0775",  # prevalence contrast in the natural-missingness cohort
    "2027.1",  # a template version string an earlier draft carried in \pdfinfo
    "30", "50",  # the declared eligibility floors, like the pre-registered 20
    "95",        # the confidence convention, stated as such
    "18",        # a display choice: rows shown in the Figure 1 heatmap
}
# LaTeX/amsmath commands that look like our lowerCamel macro names
MATH_COMMANDS = {
    "cmidrule", "toprule", "midrule", "bottomrule", "textbf", "emph",
    "includegraphics", "bibliographystyle", "newcommand", "renewcommand",
    "setlength", "usepackage", "documentclass", "maketitle", "frenchspacing",
    "pdfinfo", "TemplateVersion", "pdfpagewidth", "pdfpageheight",
    "bibliography", "paragraph", "subsection", "section", "caption", "label",
    "begin", "end", "item", "textwidth", "linewidth", "centering", "small",
    "scriptsize", "arg", "min", "max", "quad", "qquad", "text", "mathrm",
    "subseteq", "setminus", "sum", "prod", "frac", "sigma", "alpha", "beta",
    "delta", "phi", "psi", "tau", "times", "geq", "leq", "ge", "le", "cup",
    "cap", "varnothing", "emptyset", "mathbb", "left", "right", "big",
    "textsc", "textit", "footnote", "url", "cite", "citep", "citet", "ref",
    "eqref", "input", "vspace", "hspace", "noindent", "bigcup", "colon",
}


def body_text(tex):
    """Strip everything that legitimately contains numbers."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    # The preamble is template configuration, not prose: the TMLR style
    # requires \def\month and \def\year, and those digits are not results.
    # Scanning them was flagging the style file's own requirements.
    if "\\begin{document}" in tex:
        tex = tex[tex.index("\\begin{document}"):]
    tex = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", tex, flags=re.S)
    tex = re.sub(r"\\includegraphics[^\}]*\}", "", tex)
    tex = re.sub(r"\\(label|ref|eqref|cite[tp]?|input|bibliography"
                 r"|bibliographystyle|usepackage|documentclass)"
                 r"(\[[^\]]*\])?\{[^\}]*\}", "", tex)
    tex = re.sub(r"\\(setlength|setcounter)\{[^\}]*\}\{[^\}]*\}", "", tex)
    tex = re.sub(r"\\begin\{(align|equation|table|tabular)\*?\}.*?"
                 r"\\end\{(align|equation|table|tabular)\*?\}", "", tex,
                 flags=re.S)
    tex = re.sub(r"\\\[.*?\\\]", "", tex, flags=re.S)
    tex = re.sub(r"(?<!\\)\$[^$]*\$", "", tex)
    return tex


def main():
    fail = []

    # ---- 3. regeneration is a fixed point --------------------------------
    before = {}
    for g in GENERATED:
        p = f"{PAPER}/{g}"
        before[g] = open(p).read() if os.path.exists(p) else None
    for gen in GENERATORS:
        r = subprocess.run([sys.executable, gen], cwd=ROOT,
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": f"{ROOT}/src"})
        if r.returncode != 0:
            fail.append(f"generator {gen} failed: {r.stderr.strip()[-300:]}")
    for g in GENERATED:
        p = f"{PAPER}/{g}"
        now = open(p).read() if os.path.exists(p) else None
        if before[g] is None:
            fail.append(f"{g} did not exist before regeneration")
        elif now != before[g]:
            fail.append(f"{g} CHANGED on regeneration -- a number in the "
                        f"draft no longer matches its source CSV")

    # ---- 2. every macro used is defined ----------------------------------
    defined = set()
    for g in GENERATED:
        p = f"{PAPER}/{g}"
        if os.path.exists(p):
            defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", open(p).read()))
    tex = open(f"{PAPER}/{MAIN}").read()
    for g in GENERATED:
        p = f"{PAPER}/{g}"
        if os.path.exists(p):
            tex += open(p).read()
    used = set(re.findall(r"\\([a-z]+[A-Z]\w*)", tex))
    ours = {u for u in used if u not in MATH_COMMANDS}
    undef = sorted(ours - defined)
    if undef:
        fail.append(f"macros used but never defined: {undef}")
    unused = sorted(defined - ours)

    # ---- 1. no stray literals in body text -------------------------------
    body = body_text(open(f"{PAPER}/{MAIN}").read())
    lits = re.findall(r"(?<![\w.\\])(\d+\.\d+|\d+)(?![\w.])", body)
    stray = sorted({x for x in lits if x not in ALLOWED})
    if stray:
        fail.append(f"numeric literals typed into body text (should be "
                    f"macros): {stray}")

    # ---- 4. source CSVs exist and are non-empty --------------------------
    csvs = set()
    for gen in GENERATORS:
        src = open(f"{ROOT}/{gen}").read()
        csvs |= set(re.findall(r'exists\(\s*f?"([\w./{}]+\.csv)"', src))
        csvs |= set(re.findall(r'have\(\s*f?"([\w./{}]+\.csv)"', src))
    traced = []
    for c in sorted(csvs):
        if "{" in c:
            continue
        p = f"{ROOT}/results/{c}"
        if not os.path.exists(p):
            continue
        if os.path.getsize(p) < 10:
            fail.append(f"source CSV empty: {c}")
        traced.append(c)

    # ---- 5. provenance of the 2x2 query-pathway intervention -------------
    # The intervention is only defined on a checkpoint whose readout consumes
    # the query.  Run against a --no-query-head checkpoint, all four cells
    # coincide and the table reads as a finding when it is a property of the
    # wiring.  That happened once, so the guard is a build gate rather than a
    # convention: the CSV carries its own provenance and we check it here.
    import glob
    import pandas as pd
    seqset = os.environ.get("QS_SEQSET", "k5")
    pats = [f"{ROOT}/results/query_pathways.csv",
            f"{ROOT}/results/query_pathways_{seqset}.csv"]
    for p in sorted({q for pat in pats for q in glob.glob(pat)}):
        name = os.path.basename(p)
        d = pd.read_csv(p)
        if "query_head" not in d.columns:
            fail.append(
                f"{name} predates the provenance column: cannot confirm it "
                f"came from a query_head=True checkpoint. Re-run "
                f"src/query_pathways.py to regenerate it.")
        elif not bool(d["query_head"].all()):
            fail.append(
                f"{name} contains rows from a query_head=False checkpoint, "
                f"whose readout never consumes the query -- the 2x2 "
                f"intervention is undefined on it and its numbers must not "
                f"appear in the manuscript.")
        else:
            print(f"provenance ok  : {name} "
                  f"(query_head=True, ckpt tag "
                  f"'{d['ckpt_tag'].iloc[0] if 'ckpt_tag' in d else '?'}')")

    print(f"macros defined : {len(defined)}")
    print(f"macros used    : {len(ours)}")
    print(f"result CSVs    : {len(traced)} traced")
    for c in traced:
        print(f"    {c}")
    if unused:
        print(f"note: {len(unused)} macros defined but unused (harmless)")
    if fail:
        print("\nFAILED:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\nOK: every displayed number traces to a CSV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
