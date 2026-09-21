"""Download BraTS from Synapse, resumably, and verify it is the right data.

The previous download turned out to be BraTS-Path (whole-slide histopathology
TIFFs) plus orphan BraTS-MET segmentation masks with no imaging attached -- it
contained zero MRI modality files.  That cost a day, so this script *verifies
what it got* rather than assuming, and says plainly whether the result can
support the subset-utility experiment.

What that experiment needs, and what the check enforces:
  * >= 3 co-registered modalities per case, so a subset lattice exists at all
  * the four BraTS modality suffixes: t1n, t1c, t2w, t2f
  * a segmentation per case, from which per-target labels are derived

Authentication.  Never paste a token into a shell command or a chat -- it lands
in history and in logs.  Put it in ~/.synapseConfig, which synapseclient reads
automatically:

    [authentication]
    authtoken = <token from synapse.org -> Account Settings -> Personal Access
                 Tokens, with "Download" scope>

then `chmod 600 ~/.synapseConfig`.  A SYNAPSE_AUTH_TOKEN environment variable
also works and is preferred in a batch script.
"""
import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "data_local" / "brats"
MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def verify(root):
    """Report whether a downloaded tree can support the experiment."""
    root = Path(root)
    nii = list(root.rglob("*.nii.gz"))
    other = Counter(p.suffix for p in root.rglob("*") if p.is_file())
    per_case = {}
    for p in nii:
        m = re.match(r"(BraTS-\w+-\d+-\d+)-(\w+)\.nii\.gz$", p.name)
        if not m:
            continue
        per_case.setdefault(m.group(1), set()).add(m.group(2))
    mod_counts = Counter()
    for mods in per_case.values():
        for m in mods:
            mod_counts[m] += 1
    complete = sum(1 for m in per_case.values()
                   if all(x in m for x in MODALITIES))

    print(f"\n{'=' * 68}\nVERIFICATION: {root}\n{'=' * 68}")
    print(f"  .nii.gz files      : {len(nii):,}")
    print(f"  cases parsed       : {len(per_case):,}")
    print(f"  per-modality counts: {dict(mod_counts)}")
    print(f"  cases with all 4   : {complete:,}")
    if other:
        top = ", ".join(f"{k or '<none>'}:{v}" for k, v in
                        other.most_common(6))
        print(f"  other file types   : {top}")

    ok = complete >= 100
    if ok:
        print(f"\n  USABLE: {complete} complete four-modality cases -> "
              f"{2**4 - 1} non-empty subsets, same lattice structure as the\n"
              f"  MRI corpus, so the utility/optimism analyses port directly.")
    else:
        print("\n  NOT USABLE for the subset-utility experiment.")
        if not mod_counts:
            print("  No modality files at all. If this is mostly .tif, you have "
                  "BraTS-Path\n  (histopathology) -- single-modality, no subset "
                  "lattice is possible.")
        else:
            print("  Too few complete cases. The experiment needs the *training* "
                  "release\n  where every case ships t1n, t1c, t2w and t2f.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn-id", type=str, default=None,
                    help="Synapse entity/entities, comma separated")
    ap.add_argument("--group", type=str, default=None,
                    help="named group from the cart manifest: ped | goat_extra "
                         "| met_extra | synthesis | path")
    ap.add_argument("--from-cart", action="store_true",
                    help="download everything in your Synapse download cart")
    ap.add_argument("--list-cart", action="store_true",
                    help="show what is in the cart WITHOUT downloading it")
    ap.add_argument("--dest", type=str, default=str(DEST))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        return 0 if verify(dest) else 1

    import synapseclient
    syn = synapseclient.Synapse(cache_root_dir=str(dest / ".cache"))
    tok = os.environ.get("SYNAPSE_AUTH_TOKEN")
    syn.login(authToken=tok) if tok else syn.login()
    print(f"logged in as {syn.getUserProfile()['userName']}", flush=True)

    if args.list_cart:
        # Look before downloading.  get_download_list_manifest() returns a path
        # to a CSV, not an iterable of names -- reading it as an iterable was an
        # earlier bug here that produced a confident and wrong "no MRI present"
        # verdict on a cart that did contain MRI.
        import csv
        p = syn.get_download_list_manifest()
        with open(p) as fh:
            rows = list(csv.DictReader(fh))
        tot = sum(float(r["dataFileSizeBytes"] or 0) for r in rows) / 1e9
        print(f"\ncart holds {len(rows)} files, {tot:.1f} GB")
        usable, skip = [], 0
        for r in rows:
            n, gb = r["name"], float(r["dataFileSizeBytes"] or 0) / 1e9
            low = n.lower()
            if (low.endswith(".tif") or low.startswith("shard-")
                    or "-seg.nii" in low or "path" in low):
                skip += 1
                continue
            if low.endswith((".zip", ".tar")) and gb > 1:
                usable.append((gb, r["ID"], n))
        usable.sort(reverse=True)
        print(f"  {skip} files are pathology / shards / orphan masks -- skip")
        print(f"\n  multi-GB archives that may hold four-modality MRI:")
        for gb, sid, n in usable:
            print(f"    {gb:7.2f} GB  {sid:14s} {n[:66]}")
        print("\n  Download by --syn-id rather than the whole cart: most of the "
              "cart is\n  histopathology this experiment cannot use.")
        return 0

    if args.from_cart:
        # The cart is what the web UI's "Download Programmatically" dialog
        # refers to.  Downloading it is resumable: already-present files are
        # skipped, so a killed job can simply be resubmitted.
        syn.get_download_list(downloadLocation=str(dest))
    elif args.group:
        # Grouped download straight off the cart manifest.  Listing 42 shard IDs
        # by hand invites typos; filtering the manifest is checkable.
        import csv
        import synapseclient as _sc
        p = syn.get_download_list_manifest()
        with open(p) as fh:
            rows = list(csv.DictReader(fh))
        FILTERS = {
            "ped":        lambda n: "BraTS26_PED" in n,
            "goat_extra": lambda n: "GoAT" in n and "With-GroundTruth" not in n,
            "met_extra":  lambda n: "MET-Challenge" in n and "TrainingData" not in n,
            "synthesis":  lambda n: "Local-Synthesis" in n,
            "path":       lambda n: (n.endswith(".tif") or n.startswith("shard-")
                                     or "Path" in n or "val-shard" in n
                                     or "foreground_MASK" in n),
        }
        f = FILTERS.get(args.group)
        if f is None:
            print(f"unknown group; pick from {list(FILTERS)}")
            return 2
        sel = [r for r in rows if f(r["name"])]
        gb = sum(float(r["dataFileSizeBytes"] or 0) for r in sel) / 1e9
        print(f"group '{args.group}': {len(sel)} files, {gb:.1f} GB", flush=True)
        for i, r in enumerate(sel, 1):
            out = dest / r["name"]
            if out.exists() and out.stat().st_size == int(
                    float(r["dataFileSizeBytes"] or 0)):
                print(f"  [{i}/{len(sel)}] skip (present) {r['name']}", flush=True)
                continue
            print(f"  [{i}/{len(sel)}] {r['name']} "
                  f"({float(r['dataFileSizeBytes'] or 0)/1e9:.2f} GB)", flush=True)
            try:
                syn.get(r["ID"], downloadLocation=str(dest),
                        ifcollision="keep.local")
            except Exception as exc:                          # noqa: BLE001
                print(f"      FAILED {type(exc).__name__}: {exc}", flush=True)
        return 0
    elif args.syn_id:
        import synapseutils
        for sid in args.syn_id.split(","):
            sid = sid.strip()
            if not sid:
                continue
            print(f"--- syncing {sid} ---", flush=True)
            synapseutils.syncFromSynapse(syn, sid, path=str(dest))
    else:
        print("give --syn-id, --group or --from-cart")
        return 2

    return 0 if verify(dest) else 1


if __name__ == "__main__":
    sys.exit(main())
