"""SHA256 manifest for the pre-registrations and every result they rest on.

Git records that a file had these contents at commit time. It does not prove a
file existed earlier than the commit, so this manifest is not a substitute for
the independently recorded planning timestamps -- it is a tamper check going
forward. If a result CSV is silently regenerated with different
numbers, the hash moves and the diff is visible.

Run before every freeze; commit the output alongside the results.
"""
import hashlib
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "MANIFEST.sha256")
INCLUDE = (("paper", (".tex", ".bib")),
           ("results", (".csv",)),
           ("src", (".py",)),
           ("jobs", (".sbatch",)))


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for spec in INCLUDE:
        if len(spec) == 1:
            p = os.path.join(ROOT, spec[0])
            if os.path.exists(p):
                rows.append((sha256(p), os.path.getsize(p), spec[0]))
            continue
        sub, exts = spec
        base = os.path.join(ROOT, sub)
        for dirpath, _, names in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for n in sorted(names):
                if not n.endswith(exts):
                    continue
                p = os.path.join(dirpath, n)
                rel = os.path.relpath(p, ROOT)
                rows.append((sha256(p), os.path.getsize(p), rel))
    rows.sort(key=lambda r: r[2])
    with open(OUT, "w") as fh:
        fh.write(f"# generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        fh.write(f"# {len(rows)} files\n")
        for h, sz, rel in rows:
            fh.write(f"{h}  {sz:>12d}  {rel}\n")
    print(f"wrote {len(rows)} hashes to {OUT}")
    for h, _, rel in rows:
        if rel == "PLAN.md":
            print(f"  PLAN.md (both pre-registrations): {h[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
