"""Evaluate the pre-registered SWI prediction (PLAN.md, 2026-07-29T22:01:53Z).

This file was written **before** the SWI features existed and before any k5
lattice was run, so the decision rule cannot have been shaped by the answer. The
finding lists and the success criteria below are copied from the pre-registration
and must not be edited.

  P1  Among the haemorrhage-family findings that clear the 20-positive floor,
      SWI-axial has the largest exact Shapley value of the five inputs for a
      strict majority.
  P2  SWI does *not* rank first for the four control findings, which have no
      blood-product basis.  P1 alone could be satisfied by SWI simply being a
      strong input everywhere; P2 is what makes it a claim about contrast physics.

Prints the outcome either way and writes results/swi_prediction.csv.  It is a
report, not a gate: a failure is a finding about the atlas, not a bug.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import robust_csv  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")

# Fixed by the pre-registration. Do not add, remove or reorder.
HAEMORRHAGE = [
    "Cerebral hemorrhage",
    "Silent micro-hemorrhage of brain",
    "Cavernous hemangioma",
    "Subdural intracranial hemorrhage",
]
CONTROLS = [
    "Cerebral atrophy",
    "Ventriculomegaly",
    "Empty sella syndrome",
    "Mega cisterna magna",
]
SWI = "SWIax"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="_k5")
    args = ap.parse_args()

    f = f"{RES}/shapley{args.tag}.csv"
    if not os.path.exists(f):
        print(f"missing {f}; run shapley_synergy.py with QS_SEQSET=k5 first")
        return 1
    d = robust_csv(f)
    if SWI not in set(d.sequence):
        print(f"{SWI} absent from {f} -- wrong input set?")
        return 1

    piv = d.pivot_table(index="finding", columns="sequence", values="shapley")
    npos = d.groupby("finding").n_pos.first()
    rows = []
    for group, names in (("haemorrhage", HAEMORRHAGE), ("control", CONTROLS)):
        for nm in names:
            if nm not in piv.index:
                rows.append({"group": group, "finding": nm, "evaluable": False,
                             "n_pos": int(npos.get(nm, 0)), "swi_rank": None,
                             "swi_shapley": None, "top_sequence": None})
                continue
            r = piv.loc[nm].sort_values(ascending=False)
            rows.append({
                "group": group, "finding": nm, "evaluable": True,
                "n_pos": int(npos[nm]),
                "swi_rank": int(list(r.index).index(SWI)) + 1,
                "swi_shapley": float(r[SWI]),
                "top_sequence": str(r.index[0]),
                "top_shapley": float(r.iloc[0]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(f"{RES}/swi_prediction{args.tag}.csv", index=False)

    h = out[(out.group == "haemorrhage") & out.evaluable]
    c = out[(out.group == "control") & out.evaluable]
    h_first = int((h.swi_rank == 1).sum())
    c_first = int((c.swi_rank == 1).sum())
    p1 = len(h) > 0 and h_first > len(h) / 2
    p2 = len(c) > 0 and c_first == 0

    print("=" * 66)
    print("PRE-REGISTERED SWI PREDICTION (PLAN.md 2026-07-29T22:01:53Z)")
    print("=" * 66)
    for group in ("haemorrhage", "control"):
        print(f"\n{group}:")
        for _, r in out[out.group == group].iterrows():
            if not r.evaluable:
                print(f"  {r.finding[:44]:46s} n={r.n_pos:4d}  not evaluable")
            else:
                mark = "<-- SWI first" if r.swi_rank == 1 else ""
                print(f"  {r.finding[:44]:46s} n={r.n_pos:4d}  "
                      f"SWI rank {r.swi_rank}/5  top={r.top_sequence:9s} {mark}")
    print(f"\nP1  SWI first for a strict majority of evaluable haemorrhage "
          f"findings: {h_first}/{len(h)}  -> {'HOLDS' if p1 else 'FAILS'}")
    print(f"P2  SWI never first among control findings: "
          f"{c_first}/{len(c)} first    -> {'HOLDS' if p2 else 'FAILS'}")
    verdict = ("BOTH HOLD -- the atlas recovers contrast physics it was never "
               "told" if (p1 and p2) else
               "NOT BOTH -- weaken the atlas claims accordingly; do not "
               "re-specify to pass")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
