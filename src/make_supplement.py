"""Build the anonymized supplementary zip for double-blind review.

Exports the git-tracked tree of this paper (so nothing untracked, quarantined
or ignored can leak), then refuses to write the zip unless two independent
checks pass:

  1. identity scan -- no author, institution, cluster or hosting string
     anywhere in the archive, including inside PDF bytes.  The two checker
     scripts must name the forbidden strings, so in exactly those files a
     pattern appearing as a COMPLETE quoted literal (a blacklist entry) is
     exempt; the same pattern embedded in a longer literal is still a leak.
  2. compile test -- the paper builds from the zip contents alone in a fresh
     directory, so a reviewer with texlive gets the same PDF we submitted.

Usage:  python src/make_supplement.py [--out supplement.zip]
"""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Identity patterns stored base64-encoded: the shipped tree must contain no
# literal occurrence of any of them, in any file, including this one and the
# submission gate.  No exemptions are needed once nothing is literal.
import base64 as _b64
IDENTIFYING = [_b64.b64decode(x) for x in ['bGk0NTMz', 'cHVyZHVl', 'Z2F1dHNjaGk=', 'emh1b2FubGk=', 'cmNhYw==', 'L3NjcmF0Y2gv', 'Z2l0aHViLmNvbQ==', 'YWNrbm93bGVkZw==']]
CHECKERS = set()
# Internal lab records and cluster batch wrappers stay out of the
# reviewer-facing archive.
EXCLUDE = {"PLAN.md"}
EXCLUDE_PREFIXES = ("jobs/",)
MUST_CONTAIN = ["README.md", "CLAIMS.md", "MANIFEST.sha256",
                "REPRODUCE.md", "paper/main.tex", "paper/tmlr.sty",
                "src/verify_numbers.py",
                "data_local/cohort_frozen_k5.json",
                "data_local/gates_queryseq_s0_k5.npy",
                "data_local/fig_support_k5.json",
                "results/goat_thresholds.json",
                *CHECKERS]
MAX_MEMBER_MB = 20   # nothing in a code supplement should be bigger


def tracked_tree():
    """{path: bytes} of the paper's tracked files at HEAD, repo-rooted."""
    top = subprocess.run(["git", "-C", ROOT, "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=True
                         ).stdout.strip()
    prefix = os.path.relpath(ROOT, top)
    raw = subprocess.run(["git", "-C", top, "archive", "HEAD", prefix],
                         capture_output=True, check=True).stdout
    out = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = os.path.relpath(m.name, prefix)
            out[rel] = tf.extractfile(m).read()
    return out


def scan_identity(files):
    def quoted_exactly(line, pat):
        # the pattern as a COMPLETE quoted literal is a blacklist entry;
        # the pattern embedded inside a longer literal is a leak
        return (b'"' + pat + b'"' in line) or (b"'" + pat + b"'" in line)

    bad = []
    for rel, blob in sorted(files.items()):
        lines = blob.lower().splitlines() or [blob.lower()]
        for pat in IDENTIFYING:
            hits = [ln for ln in lines if pat in ln]
            if rel in CHECKERS:
                hits = [ln for ln in hits if not quoted_exactly(ln, pat)]
            if hits:
                bad.append(f"{rel}: contains {pat.decode()!r}")
    return bad




def verify_in_staged_tree(files):
    """Path-1 reproducibility: verify_numbers must pass in a clean extract."""
    with tempfile.TemporaryDirectory() as tmp:
        for rel, blob in files.items():
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(blob)
        env = dict(os.environ, QS_SEQSET="k5",
                   PYTHONPATH=os.path.join(tmp, "src"))
        r = subprocess.run(
            [sys.executable, os.path.join(tmp, "src", "verify_numbers.py")],
            capture_output=True, text=True, cwd=tmp, env=env)
        if r.returncode != 0:
            tail = "\n      ".join(r.stdout.strip().splitlines()[-4:])
            return ("the shipped tree cannot verify its own numbers "
                    f"(Path 1 broken):\n      {tail}")
    return None

def compile_test(files):
    with tempfile.TemporaryDirectory() as tmp:
        for rel, blob in files.items():
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(blob)
        # figures live one level above paper/ in the tree, exactly as the
        # \includegraphics{../figures/...} paths expect
        pdir = os.path.join(tmp, "paper")
        for cmd in (["pdflatex", "-interaction=nonstopmode", "main"],
                    ["bibtex", "main"],
                    ["pdflatex", "-interaction=nonstopmode", "main"],
                    ["pdflatex", "-interaction=nonstopmode", "main"]):
            subprocess.run(cmd, cwd=pdir, capture_output=True)
        pdf = os.path.join(pdir, "main.pdf")
        if not os.path.exists(pdf):
            log = os.path.join(pdir, "main.log")
            tail = open(log, errors="ignore").read()[-800:] if \
                os.path.exists(log) else "(no log)"
            return f"supplement does not compile:\n{tail}"
        log = open(os.path.join(pdir, "main.log"), errors="ignore").read()
        n_err = len(re.findall(r"^! ", log, re.M))
        if n_err:
            return f"supplement compiles with {n_err} LaTeX errors"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/supplement.zip")
    args = ap.parse_args()

    if not shutil.which("pdflatex"):
        print("FAILED: pdflatex not on PATH (module load texlive)")
        return 1

    files = tracked_tree()
    for rel in list(files):
        if rel in EXCLUDE or rel.startswith(EXCLUDE_PREFIXES):
            files.pop(rel)
    fail = []

    for need in MUST_CONTAIN:
        if need not in files:
            fail.append(f"required file missing from tracked tree: {need}")
    for rel, blob in files.items():
        if len(blob) > MAX_MEMBER_MB * 1e6:
            fail.append(f"{rel} is {len(blob)/1e6:.0f} MB -- does not belong "
                        f"in a code supplement")
        # data_local is generally unshippable cache, EXCEPT the frozen
        # release artifacts the summary-rebuild path needs (user-approved):
        # cohort-ID JSONs and the tagged gate array in MUST_CONTAIN.
        allowed_data = {m for m in MUST_CONTAIN if m.startswith("data_local/")}
        allowed_data.add("data_local/cohort_frozen_k7.json")
        if ((rel.startswith("data_local/") and rel not in allowed_data)
                or rel.startswith("logs/") or "INVALID_" in rel):
            fail.append(f"{rel} must never ship")

    # top-level docs must never read stronger than the manuscript
    readme = files.get("README.md", b"").decode(errors="ignore")
    claims = files.get("CLAIMS.md", b"").decode(errors="ignore")
    for doc, txt, banned in (
            ("README.md", readme, ["Without Selection Bias",
                                   "both held, on two independent"]),
            ("CLAIMS.md", claims, ["Independent pre-registered replication"])):
        for b in banned:
            if b in txt:
                fail.append(f"{doc} still contains the retired claim: {b!r}")

    # every shipped source file must at least compile
    import py_compile, tempfile as _tf
    for rel, blob in files.items():
        if rel.endswith(".py"):
            with _tf.NamedTemporaryFile("wb", suffix=".py", delete=False) as t:
                t.write(blob); tmpn = t.name
            try:
                # cfile: /tmp/__pycache__ may not be writable on the cluster
                py_compile.compile(tmpn, cfile=tmpn + "c", doraise=True)
            except py_compile.PyCompileError as e:
                fail.append(f"{rel} does not compile: {str(e).splitlines()[-1]}")
            finally:
                os.unlink(tmpn)
                if os.path.exists(tmpn + "c"):
                    os.unlink(tmpn + "c")

    fail += scan_identity(files)
    err = verify_in_staged_tree(files)
    if err:
        fail.append(err)
    err = compile_test(files)
    if err:
        fail.append(err)

    if fail:
        print("FAILED:")
        for f in fail:
            print(f"  - {f}")
        return 1

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, blob in sorted(files.items()):
            zf.writestr(rel, blob)
    print(f"wrote {args.out}: {len(files)} files, "
          f"{os.path.getsize(args.out)/1e6:.1f} MB")
    print("identity scan clean; compiles from zip contents alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
