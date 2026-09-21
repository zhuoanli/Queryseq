"""Encode the sequences Round 1 left out, without touching the corpus release.

Round 1 cached four sequences and omitted SWI, which is the sequence that
carries blood-product signal and is present for 53.8% of studies.  Widening the
input set needs `swi-raw-axi`, `flair-raw-sag`, `t1w-raw-sag` and
`flair-raw-cor` encoded with the *same* frozen encoder and the *same*
preprocessing, or the new inputs are not comparable to the old ones.

The corpus release is frozen, so rather than edit its extractor this script
reproduces its `load_volume` and `encode` verbatim (see
`src/extract_features.py` in that release) and writes to a tree this repo owns:

    data_local/features_extra/{split}/{study_uid}.npz

`common.data.build_cache` reads a sequence from the release's cache if it is there
and from this tree otherwise, so the two together cover all eight sequences and
neither is mutated.

Two deliberate differences from the release's extractor:
  * only the pooled vector is stored, not the 4x4x4 spatial grid.  Nothing in
    this paper reads grids, and they are 98% of the bytes.
  * skipping is per *sequence*, not per study, so a study already encoded for
    one new sequence still gets the others.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PB = Path(os.environ.get("QS_FEATROOT",
          Path(__file__).resolve().parents[2] / "corpus_release"))
MRI = PB / "data" / "mrrate" / "mri"
PB_FEAT = PB / "data" / "features"
OUT = ROOT / "data_local" / "features_extra"
SPLITS = PB / "data" / "mrrate" / "splits.csv"

MODEL_ID = "facebook/vjepa2-vitg-fpc64-384"
FRAMES, RES, GRID = 64, 384, 4

# Sequences the corpus release already cached; anything in ALL_SEQUENCES but not here has
# to be encoded now.
ALREADY = {"t1w-raw-axi", "t2w-raw-axi", "flair-raw-axi", "t2w-raw-cor"}


def build_targets(only=None):
    """Sequences still to encode.

    `only` restricts to a named input set, which matters when GPU time is
    scarce: k5 needs just swi-raw-axi and flair-raw-sag, while t1w-raw-sag and
    flair-raw-cor exist solely for the k7 deep lattice.  Encoding only what the
    primary analysis needs roughly halves the work.
    """
    # Load common/data.py directly by path rather than importing the package.
    # `import common.data` executes common/__init__, which pulls in sklearn via
    # metrics; the encoder needs none of that, and the smallgpu environment does
    # not have it.  The encoder should not depend on the analysis stack.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_qs_data", str(ROOT / "src" / "common" / "data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    want = mod.SEQUENCE_SETS[only] if only else mod.ALL_SEQUENCES
    return [s for s in want if s not in ALREADY]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--split", default="all",
                    choices=["all", "train", "val", "test"])
    ap.add_argument("--only", type=str, default=None,
                    help="restrict to one input set, e.g. k5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if os.environ.get("HF_HOME"):
        os.environ.setdefault(
            "HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    import torch.nn.functional as F
    import nibabel as nib

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def load_volume(path):
        """NIfTI -> (FRAMES, 3, RES, RES) float32. Verbatim from the corpus release."""
        vol = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)
        if vol.ndim != 3 or min(vol.shape) < 2:
            return None
        lo, hi = np.percentile(vol, [0.5, 99.5])
        if hi <= lo:
            return None
        vol = np.clip((vol - lo) / (hi - lo), 0, 1)
        t = torch.from_numpy(vol).permute(2, 0, 1)[:, None]
        t = F.interpolate(t, size=(RES, RES), mode="bilinear",
                          align_corners=False)
        t = t.permute(1, 0, 2, 3)[None]
        t = F.interpolate(t, size=(FRAMES, RES, RES), mode="trilinear",
                          align_corners=False)
        t = t[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
        return (t - MEAN) / STD

    @torch.no_grad()
    def encode(model, clip, device):
        """(FRAMES,3,RES,RES) -> pooled (1408,) fp16. Verbatim from the corpus release."""
        x = clip[None].to(device, torch.bfloat16)
        out = (model.get_vision_features(x)
               if hasattr(model, "get_vision_features")
               else model(pixel_values_videos=x).last_hidden_state)
        tok = out[0] if out.ndim == 3 else out.flatten(0, -2)
        return tok.float().mean(0).half().cpu().numpy()

    targets = build_targets(args.only)
    have_batches = set(os.listdir(MRI))
    sp = pd.read_csv(SPLITS)
    sp = sp[sp.batch_id.isin(have_batches)]
    if args.split != "all":
        sp = sp[sp.split == args.split]
    order = {"test": 0, "val": 1, "train": 2}
    sp = sp.assign(_o=sp.split.map(order)).sort_values(["_o", "study_uid"])
    rows = list(sp.itertuples())[args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(rows)} studies, "
          f"targets={targets}", flush=True)

    if args.dry_run:
        need = 0
        for r in rows:
            sdir = MRI / r.batch_id / r.study_uid / "img"
            if not sdir.is_dir():
                continue
            need += sum(1 for s in targets
                        if sorted(sdir.glob(f"*_{s}.nii.gz")))
        print(f"volumes that would be encoded in this shard: {need}", flush=True)
        return 0

    from transformers import AutoModel
    device = "cuda"
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    print(f"loaded {MODEL_ID}", flush=True)

    t0, vols, studies, skipped = time.time(), 0, 0, 0
    for i, r in enumerate(rows, 1):
        dest = OUT / r.split / f"{r.study_uid}.npz"
        sdir = MRI / r.batch_id / r.study_uid / "img"
        if not sdir.is_dir():
            continue
        existing = {}
        if dest.exists():
            try:
                with np.load(dest) as z:
                    existing = {k: z[k] for k in z.files}
            except Exception:                                # noqa: BLE001
                existing = {}
        pooled = {k: v for k, v in existing.items()
                  if k.startswith("pooled_")}
        added = False
        for seq in targets:
            if f"pooled_{seq}" in pooled:
                continue
            cands = sorted(sdir.glob(f"*_{seq}.nii.gz"))
            if not cands:
                continue
            try:
                clip = load_volume(cands[0])
                if clip is None:
                    continue
                pooled[f"pooled_{seq}"] = encode(model, clip, device)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {r.study_uid}/{seq}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            vols += 1
            added = True
        if not added:
            skipped += 1
            continue
        names = sorted(k[len("pooled_"):] for k in pooled)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, sequences=np.array(names), **pooled)
        studies += 1
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(rows)} studies={studies} vols={vols} "
                  f"skip={skipped} {el/60:.1f} min "
                  f"{vols/max(el,1)*3600:.0f} vols/h", flush=True)

    print(f"shard {args.shard} finished: {studies} studies, {vols} volumes, "
          f"{skipped} nothing-to-do, {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
