"""Cohort assembly for the sequence-routing experiments.

Every experiment in this paper reads its studies through `load_cohort`, which is
backed by a frozen uid list on disk.  The corpus is still growing (batches are
being extracted in the background), and a result that silently changes because
four hundred new studies appeared is not a result.  `freeze_cohort` writes the
uid list once; everything downstream is pinned to it.

The central design decision lives here.  Sequence availability in MR-RATE is
*indication-driven*: FLAIR is ordered when demyelination is suspected, coronal
T2w when a particular structure is in question.  So evaluating an input subset S
on "the studies that happen to carry S" compares different patient populations,
and any sequence utility measured that way is partly patient selection.  The
lattice therefore runs on the complete-case cohort, where withholding a sequence
is a genuine counterfactual intervention rather than a change of population.
`load_cohort(complete_case=False)` exists for the separate, clearly-labelled
natural-missingness analysis and must never be mixed into the main tables.
"""
import glob
import io
import json
import os
import time

import numpy as np
import pandas as pd

# Repo-relative so a clone runs anywhere; QS_FEATROOT points at the corpus
# checkout (features + labels from the MR-RATE extraction) when it is not a
# sibling directory.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PB = os.environ.get("QS_FEATROOT", os.path.join(os.path.dirname(ROOT), "corpus_release"))

# Which inputs the analysis reads.  Selected by joint availability measured over
# all 32,023 extracted studies rather than by assumption -- see PLAN.md.  These
# are separately acquired series, but availability is indication-driven and we
# never measured scan minutes, so cost claims are about input/read budget, never
# acquisition time.
#
#   k4  Round 1's set.  Kept so those results stay reproducible.  A poor choice:
#       it includes t2w-raw-cor and omits SWI entirely, and SWI is the sequence
#       that carries blood-product signal.
#   k5  The primary set.  Highest joint availability of any 5 (n=10,851, 33.9%),
#       includes SWI and a sagittal plane.  31 non-empty subsets.
#   k7  The deep lattice (n=6,286, 19.6%), used only for the selection-optimism
#       curve, where the whole point is how optimism scales with the number of
#       candidates searched: 127 subsets instead of 15.
#
# 8 inputs is not viable -- only 2.8% of studies carry all eight.  Note swi
# exists axially only; swi-raw-cor and swi-raw-sag are 0%.
SEQUENCE_SETS = {
    "k4": ["t1w-raw-axi", "t2w-raw-axi", "flair-raw-axi", "t2w-raw-cor"],
    "k5": ["t1w-raw-axi", "t2w-raw-axi", "flair-raw-axi", "flair-raw-sag",
           "swi-raw-axi"],
    "k7": ["t1w-raw-axi", "t2w-raw-axi", "flair-raw-axi", "flair-raw-sag",
           "swi-raw-axi", "t1w-raw-sag", "flair-raw-cor"],
    # k5 with SWI removed.  Its only purpose is the matched-cohort ablation:
    # k4 and k5 differ in cohort (12,412 vs 9,393 complete cases), findings
    # (33 vs 35) and candidate space (15 vs 31) all at once, so the jump in
    # per-finding gain from 0.0096 to 0.0277 cannot be attributed to SWI from
    # that comparison alone.  Running this set reuses the *k5* cohort, folds and
    # findings, changing only whether SWI is available, which isolates it.
    "k5noswi": ["t1w-raw-axi", "t2w-raw-axi", "flair-raw-axi",
                "flair-raw-sag"],
}
SEQ_SHORT_ALL = {
    "t1w-raw-axi": "T1ax", "t2w-raw-axi": "T2ax", "flair-raw-axi": "FLAIRax",
    "t2w-raw-cor": "T2cor", "flair-raw-sag": "FLAIRsag", "swi-raw-axi": "SWIax",
    "t1w-raw-sag": "T1sag", "flair-raw-cor": "FLAIRcor", "t1w-raw-cor": "T1cor",
    "t2w-raw-sag": "T2sag",
}

# Every consumer keys off this, so one variable switches the whole pipeline.
# Caches and frozen cohorts are namespaced by it, so the sets never collide.
SEQSET = os.environ.get("QS_SEQSET", "k5")
assert SEQSET in SEQUENCE_SETS, f"QS_SEQSET must be one of {list(SEQUENCE_SETS)}"
SEQUENCES = SEQUENCE_SETS[SEQSET]
SEQ_SHORT = {s: SEQ_SHORT_ALL[s] for s in SEQUENCES}
DIM = 1408
NSEQ = len(SEQUENCES)

# Union of every sequence any set needs -- what has to be encoded on disk.
ALL_SEQUENCES = sorted({s for v in SEQUENCE_SETS.values() for s in v})

CACHE = f"{ROOT}/data_local"

# Which frozen cohort to read.  Normally the one belonging to the active input
# set, but the SWI ablation must reuse *k5's* studies, folds and findings while
# dropping one input -- otherwise k5noswi would compute its own complete-case
# cohort, which does not require SWI and is therefore larger, and the comparison
# would confound "removing SWI" with "changing the patient population".
COHORT = os.environ.get("QS_COHORT", SEQSET)
FROZEN = f"{CACHE}/cohort_frozen_{COHORT}.json"


def _cpath(split, kind):
    """Cache path, namespaced by input set so k4/k5/k7 never overwrite each other."""
    return f"{CACHE}/{COHORT}_{split}_{kind}"


def robust_csv(path, tries=6):
    """Read a CSV through transient Lustre EIO.

    This cluster intermittently fails to open a freshly written file; the corpus pipeline hit
    it often enough to warrant a retry wrapper (capability_matrix.py:47).
    """
    last = None
    for _ in range(tries):
        try:
            with open(path, "rb") as fh:
                return pd.read_csv(io.BytesIO(fh.read()))
        except OSError as e:
            last = e
            time.sleep(2)
    raise last


def robust_npz(path, tries=6):
    """np.load through transient Lustre EIO.

    Same failure as robust_csv, and it bites hardest on files written minutes
    earlier by a just-finished job -- exactly the case when a freshly encoded
    batch is first read.
    """
    last = None
    for _ in range(tries):
        try:
            with open(path, "rb") as fh:
                return np.load(io.BytesIO(fh.read()), allow_pickle=True)
        except OSError as e:
            last = e
            time.sleep(2)
    raise last


def subset_name(mask):
    """Canonical name for an input subset, e.g. (1,0,1,0) -> 'T1ax+FLAIRax'."""
    on = [SEQ_SHORT[s] for s, m in zip(SEQUENCES, mask) if m]
    return "+".join(on) if on else "none"


def all_subsets():
    """Every non-empty subset of the active input set, as 0/1 tuples.

    2^n - 1 of them: 15 at k4, 31 at k5, 127 at k7.  Even at k7 the lattice is
    enumerated in full, which is why the Shapley values are exact rather than
    sampled.
    """
    out = []
    for b in range(1, 1 << NSEQ):
        out.append(tuple((b >> j) & 1 for j in range(NSEQ)))
    return sorted(out, key=lambda m: (sum(m), m))


# --------------------------------------------------------------------------
# feature cache
# --------------------------------------------------------------------------

EXTRA_FEAT = f"{ROOT}/data_local/features_extra"


def build_cache(split, verbose=True):
    """Collapse a split's per-study .npz files into three flat arrays.

    The upstream corpus code re-globs and re-parses ~30k npz files in every analysis script, which
    costs a couple of minutes each time.  We pay it once per split.

    Sequences live in two trees: the four the corpus release cached, and the four
    this paper added (src/extract_extra.py).  Both are read, the release's first, so neither has
    to be mutated and every input set is served from the same encoder.
    """
    os.makedirs(CACHE, exist_ok=True)
    fs = sorted(glob.glob(f"{PB}/data/features/{split}/*.npz"))
    n = len(fs)
    P = np.zeros((n, NSEQ, DIM), np.float16)
    M = np.zeros((n, NSEQ), np.uint8)
    uids, bad = [], 0
    for i, f in enumerate(fs):
        try:
            z = robust_npz(f)
        except OSError:
            # a study we cannot read is dropped rather than silently zero-filled,
            # which would look like "all four sequences missing"
            bad += 1
            uids.append(None)
            continue
        uid = os.path.basename(f)[:-4]
        have = {str(s) for s in z["sequences"]}
        for j, s in enumerate(SEQUENCES):
            if s in have:
                P[i, j] = z[f"pooled_{s}"]
                M[i, j] = 1
        # anything the active set needs that the corpus release did not cache
        want = [s for j, s in enumerate(SEQUENCES) if M[i, j] == 0]
        if want:
            ex = f"{EXTRA_FEAT}/{split}/{uid}.npz"
            if os.path.exists(ex):
                try:
                    ze = robust_npz(ex)
                    for j, s in enumerate(SEQUENCES):
                        k = f"pooled_{s}"
                        if M[i, j] == 0 and k in ze.files:
                            P[i, j] = ze[k]
                            M[i, j] = 1
                except OSError:
                    pass          # missing extras are absent inputs, not errors
        uids.append(uid)
        if verbose and (i + 1) % 5000 == 0:
            print(f"  {split}: {i + 1}/{n}", flush=True)
    if bad:
        ok = np.array([u is not None for u in uids])
        P, M = P[ok], M[ok]
        uids = [u for u in uids if u is not None]
        print(f"  {split}: dropped {bad} unreadable feature files", flush=True)
    np.save(_cpath(split, "pooled.npy"), P)
    np.save(_cpath(split, "mask.npy"), M)
    with open(_cpath(split, "uids.json"), "w") as fh:
        json.dump(uids, fh)
    if verbose:
        print(f"  {split}: cached {n} studies, "
              f"{int((M.sum(1) == NSEQ).sum())} complete-case", flush=True)
    return P, M, uids


def _read_cache(split):
    p = _cpath(split, "pooled.npy")
    if not os.path.exists(p):
        return build_cache(split)
    P = np.load(p)
    M = np.load(_cpath(split, "mask.npy"))
    uids = json.load(open(_cpath(split, "uids.json")))
    return P, M, uids


# --------------------------------------------------------------------------
# labels, splits, scanner metadata
# --------------------------------------------------------------------------

def load_labels():
    return robust_csv(
        f"{PB}/data/mrrate/pathology_labels/mrrate_labels.csv"
    ).set_index("study_uid")


def load_splits():
    return robust_csv(f"{PB}/data/mrrate/splits.csv")


def _vendor(x):
    x = str(x).upper()
    if "SIEMENS" in x:
        return "Siemens"
    if "PHILIPS" in x:
        return "Philips"
    if x.startswith("GE"):
        return "GE"
    return "Other"


def load_scanner_meta():
    """One row per study: vendor, field strength, station, model, date.

    Enables the held-out-vendor and field-strength shift experiments.  Cached
    because it means parsing 28 wide DICOM-tag CSVs (705k series rows).
    """
    cp = f"{CACHE}/scanner_meta.csv"
    if os.path.exists(cp):
        return robust_csv(cp)
    cols = ["study_uid", "Manufacturer", "Manufacturer'sModelName",
            "FieldStrength_T", "StationName", "anon_study_date"]
    ds = []
    for f in sorted(glob.glob(f"{PB}/data/mrrate/metadata/batch*_metadata.csv")):
        ds.append(pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False))
    d = pd.concat(ds, ignore_index=True).drop_duplicates("study_uid")
    d["vendor"] = d.Manufacturer.map(_vendor)
    d["field_t"] = pd.to_numeric(d.FieldStrength_T, errors="coerce").round(2)
    d = d.rename(columns={"StationName": "station",
                          "Manufacturer'sModelName": "model",
                          "anon_study_date": "study_date"})
    d = d[["study_uid", "vendor", "field_t", "station", "model", "study_date"]]
    os.makedirs(CACHE, exist_ok=True)
    d.to_csv(cp, index=False)
    return d


# --------------------------------------------------------------------------
# the frozen cohort
# --------------------------------------------------------------------------

def freeze_cohort(verbose=True):
    """Pin the study set every experiment runs on, and record why each was kept.

    A study is usable when it has cached features, a label row, and an official
    split assignment.  We store the complete-case flag alongside so downstream
    code never has to re-derive it.
    """
    lab = load_labels()
    sp = load_splits()
    smap = dict(zip(sp.study_uid.astype(str), sp.split))
    pmap = dict(zip(sp.study_uid.astype(str), sp.patient_uid.astype(str)))
    out = {"sequences": SEQUENCES, "splits": {}}
    for split in ("train", "val", "test"):
        P, M, uids = _read_cache(split)
        keep, cc = [], []
        lab_idx = set(lab.index.astype(str))
        for i, u in enumerate(uids):
            if u not in lab_idx or smap.get(u) != split:
                continue
            keep.append(u)
            cc.append(bool(M[i].sum() == NSEQ))
        out["splits"][split] = {
            "uids": keep,
            "complete_case": cc,
            "patient": [pmap.get(u, u) for u in keep],
            "n": len(keep),
            "n_complete_case": int(sum(cc)),
        }
        if verbose:
            print(f"{split}: usable {len(keep)}  complete-case {sum(cc)}",
                  flush=True)
    os.makedirs(CACHE, exist_ok=True)
    with open(FROZEN, "w") as fh:
        json.dump(out, fh)
    return out


class Cohort:
    """Feature matrix, availability mask, labels and grouping for one split.

    Attributes
    ----------
    P : (n, 4, 1408) float32   per-sequence pooled embeddings
    M : (n, 4) float32         availability mask (all ones when complete_case)
    Y : (n, F) int             finding labels
    findings : list[str]       label column names, aligned to Y
    uids, patients : (n,)      study and patient identifiers
    """

    def __init__(self, split, complete_case=True, limit=0):
        if not os.path.exists(FROZEN):
            freeze_cohort()
        blob = json.load(open(FROZEN))
        fr = blob["splits"][split]
        P, M, uids = _read_cache(split)
        # The cache belongs to COHORT, whose sequence list can be longer than
        # the active one -- that is exactly the ablation case, where k5noswi
        # borrows k5's cohort but reads only four of its five inputs.  Select
        # columns by sequence *name*, never by position, so a reordering of
        # SEQUENCE_SETS cannot silently pair a mask bit with the wrong input.
        cached_seqs = blob.get("sequences", SEQUENCES)
        if list(cached_seqs) != list(SEQUENCES):
            missing = [s for s in SEQUENCES if s not in cached_seqs]
            if missing:
                raise ValueError(
                    f"cohort '{COHORT}' has {cached_seqs} and cannot serve "
                    f"input set '{SEQSET}', which needs {missing}")
            take = [list(cached_seqs).index(s) for s in SEQUENCES]
            P, M = P[:, take], M[:, take]
        pos = {u: i for i, u in enumerate(uids)}
        sel_u = [u for u, c in zip(fr["uids"], fr["complete_case"])
                 if c or not complete_case]
        sel_p = [p for p, c in zip(fr["patient"], fr["complete_case"])
                 if c or not complete_case]
        if limit:
            sel_u, sel_p = sel_u[:limit], sel_p[:limit]
        idx = np.array([pos[u] for u in sel_u])
        self.split = split
        self.complete_case = complete_case
        self.uids = np.array(sel_u)
        self.patients = np.array(sel_p)
        self.P = P[idx].astype(np.float32)
        self.M = M[idx].astype(np.float32)
        lab = load_labels()
        ya = lab.reindex(self.uids)
        self.findings = list(lab.columns)
        self.Y = ya.to_numpy().astype(np.int64)

    def __len__(self):
        return len(self.uids)

    def meta(self):
        """Join scanner metadata onto this cohort, in row order."""
        m = load_scanner_meta().set_index("study_uid")
        return m.reindex(self.uids).reset_index()

    def pool(self, mask, renorm=True):
        """Mean-pool the embeddings of the sequences selected by `mask`.

        `mask` is a 0/1 tuple over SEQUENCES.  Sequences that the subset selects
        but the study does not carry are simply absent from the average, so the
        result is the mean over *present and selected* inputs.  Rows with no
        remaining input return zeros and are reported, never silently kept.
        """
        w = self.M * np.asarray(mask, np.float32)[None, :]
        d = w.sum(1, keepdims=True)
        z = (self.P * w[:, :, None]).sum(1)
        if renorm:
            z = z / np.maximum(d, 1e-6)
        return z, (d[:, 0] > 0)


def finding_filter(Y, findings, min_pos=20, max_frac=1.0):
    """Indices of findings with enough positives to be evaluable.

    The threshold is a disclosed property of the label space, not a tuned
    parameter; per-finding n is reported alongside every result so a reader can
    discount the rare ones themselves.
    """
    keep = []
    n = len(Y)
    for j, f in enumerate(findings):
        p = int(Y[:, j].sum())
        if p >= min_pos and p < n * max_frac:
            keep.append(j)
    return keep


# --- results-file resolution, shared by every generator -------------------
# Three separate generators (numbers, tables, figures) each carried their own
# copy of "open results/<name>.csv", and each in turn silently rendered the
# four-sequence results beside five-sequence prose.  The failures were not
# wrong numbers; they were correct numbers under a false caption, which is
# harder to see.  One resolver, used everywhere, ends that class of error.
RESULTS_USED = set()


def result_path(name, res_dir=None):
    """Path to a results CSV, preferring the active input set's tagged copy.

    `k4` is the historical untagged set, so it never takes a suffix.  Returns
    None when neither form exists, so callers keep their "skip this block"
    behaviour.  Every resolved name is recorded in RESULTS_USED so a generator
    can report which input set each of its numbers actually came from.
    """
    res_dir = res_dir or f"{ROOT}/results"
    stem, ext = os.path.splitext(name)
    tagged = f"{res_dir}/{stem}_{SEQSET}{ext}"
    if SEQSET != "k4" and os.path.exists(tagged):
        RESULTS_USED.add(os.path.basename(tagged))
        return tagged
    plain = f"{res_dir}/{name}"
    if os.path.exists(plain):
        RESULTS_USED.add(name)
        return plain
    return None


def report_results_used(label=""):
    """Print which resolved files were tagged and which fell back."""
    tagged = sorted(x for x in RESULTS_USED if f"_{SEQSET}." in x)
    plain = sorted(x for x in RESULTS_USED if f"_{SEQSET}." not in x)
    print(f"  {label}set={SEQSET}: {len(tagged)} tagged, {len(plain)} untagged")
    if plain:
        print(f"  untagged (k4 or set-independent): {', '.join(plain)}")
