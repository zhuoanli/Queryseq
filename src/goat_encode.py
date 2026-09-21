"""Encode GoAT with the same frozen encoder the MRI corpus used.

Cross-task replication only means something if the representation is held
fixed: the same V-JEPA2 checkpoint, the same percentile intensity scaling, the
same resampling to 64x384x384, the same mean-pooled 1408-d output. Anything
else and a difference between datasets could be a difference in preprocessing.

Writes one .npz per case with `pooled_<modality>` keys, mirroring the layout of
the MRI feature cache so the downstream lattice code needs no special case.

Runs on smallgpu under train_gpu_env: the `life` account has exhausted its
billing quota, and smallgpu bills at zero.  It therefore imports nothing from
`common` -- that package pulls in sklearn, which smallgpu's environment lacks.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data_local" / "brats_goat_x"
OUT = ROOT / "data_local" / "goat_features"
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
MODEL_ID = "facebook/vjepa2-vitg-fpc64-384"
FRAMES, RES = 64, 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if os.environ.get("HF_HOME"):
        os.environ.setdefault(
            "HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    cases = sorted({p.parent for p in SRC.rglob("*-seg.nii.gz")})
    cases = cases[args.shard::args.nshards]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.nshards}: {len(cases)} cases", flush=True)
    if args.dry_run:
        return 0

    import torch
    import torch.nn.functional as F
    import nibabel as nib
    from transformers import AutoModel

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def load_volume(path):
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

    device = "cuda"
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    print(f"loaded {MODEL_ID}", flush=True)

    @torch.no_grad()
    def encode(clip):
        x = clip[None].to(device, torch.bfloat16)
        out = (model.get_vision_features(x)
               if hasattr(model, "get_vision_features")
               else model(pixel_values_videos=x).last_hidden_state)
        tok = out[0] if out.ndim == 3 else out.flatten(0, -2)
        return tok.float().mean(0).half().cpu().numpy()

    t0, done, vols = time.time(), 0, 0
    for i, cdir in enumerate(cases, 1):
        cid = cdir.name
        dest = OUT / f"{cid}.npz"
        if dest.exists():
            continue
        pooled = {}
        for m in MODALITIES:
            p = cdir / f"{cid}-{m}.nii.gz"
            if not p.exists():
                continue
            try:
                clip = load_volume(p)
                if clip is None:
                    continue
                pooled[f"pooled_{m}"] = encode(clip)
                vols += 1
            except Exception as exc:                          # noqa: BLE001
                print(f"  {cid}/{m}: {type(exc).__name__}: {exc}", flush=True)
        if not pooled:
            continue
        np.savez_compressed(
            dest, sequences=np.array(sorted(k[7:] for k in pooled)), **pooled)
        done += 1
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(cases)} cases={done} vols={vols} "
                  f"{el/60:.1f} min {vols/max(el,1)*3600:.0f} vols/h",
                  flush=True)
    print(f"shard {args.shard} finished: {done} cases, {vols} volumes, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
