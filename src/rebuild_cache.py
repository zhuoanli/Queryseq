"""Rebuild the val/test feature caches and re-freeze the cohort.

Run this after new studies are encoded.  It asserts the *training* cohort is
unchanged: the subset lattice is fitted on train, so if train moved, those
results would silently no longer correspond to the data they claim to.

Run it on a compute node, not the login node -- the login node's Lustre mount
intermittently drops (Errno 108, "transport endpoint shutdown") when reading
files a GPU job wrote minutes earlier, and no amount of retrying inside the
process fixes a broken mount.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (FROZEN, SEQSET, Cohort, build_cache,  # noqa: E402
                    finding_filter, freeze_cohort)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    # A new input set needs all three splits rebuilt; adding studies to an
    # existing set only needs the evaluation splits.
    ap.add_argument("--splits", type=str, default="val,test")
    args = ap.parse_args()

    print(f"input set = {SEQSET}, frozen file = {os.path.basename(FROZEN)}",
          flush=True)
    old = json.load(open(FROZEN))["splits"]["train"]["uids"] \
        if os.path.exists(FROZEN) else None
    for s in args.splits.split(","):
        build_cache(s)
    fr = freeze_cohort()
    if old is not None:
        assert fr["splits"]["train"]["uids"] == old, \
            "TRAIN COHORT CHANGED -- the subset lattice would be invalid"
        print("train cohort unchanged: lattice results remain valid",
              flush=True)
    for s in ("train", "val", "test"):
        co = Cohort(s, complete_case=True)
        line = f"  {s:5s} complete-case n={len(co):6d}"
        for mp in (10, 20, 50):
            line += f"  >={mp}pos:{len(finding_filter(co.Y, co.findings, mp)):3d}"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
