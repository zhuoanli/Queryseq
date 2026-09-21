"""Vendor composition of the complete-case cohort, per input set.

Requiring more sequences does not thin the cohort evenly across manufacturers.
In this corpus GE sites rarely acquire susceptibility-weighted imaging, so the
complete-case test split is majority GE-capable at K=4 and has essentially no GE
at K=5.  That is worth a number in the paper for two reasons: it is a direct
instance of the availability-site association the paper measures, and it is the
reason the held-out-vendor experiment holds out a different manufacturer at K=5
than at K=4.

Writes one row per (input set, split, vendor) so the manuscript macro and the
sentence it supports both trace to a CSV rather than to a shell session.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results"
SETS = ["k4", "k5"]
SPLITS = ["train", "test"]


def mix(seqset, split):
    """Vendor counts for the complete-case cohort of one input set."""
    os.environ["QS_SEQSET"] = seqset
    os.environ["QS_COHORT"] = seqset
    for mod in [m for m in list(sys.modules)
                if m == "common" or m.startswith("common.")]:
        del sys.modules[mod]
    from common import Cohort, SEQUENCES
    co = Cohort(split, complete_case=True)
    v = co.meta().vendor.fillna("NA").value_counts()
    return [{"seqset": seqset, "n_sequences": len(SEQUENCES), "split": split,
             "vendor": str(k), "n": int(n), "frac": float(n) / len(co),
             "n_cohort": len(co)} for k, n in v.items()]


def main():
    rows = []
    for s in SETS:
        for sp in SPLITS:
            try:
                rows += mix(s, sp)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {s}/{sp}: {type(exc).__name__}: {exc}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/cohort_vendor_mix.csv", index=False)
    print(d.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
