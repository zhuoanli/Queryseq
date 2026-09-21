# QuerySeq — Finding-Specific MRI Evidence, Measured Without Selection Optimism

Code and result tables for *Not Every Finding Needs Every Sequence:
Measuring Finding-Specific MRI Evidence with Nested Selection*.

Which MRI sequence carries the evidence for a finding depends on the finding:
blood products are conspicuous on susceptibility-weighted imaging,
demyelinating plaques on FLAIR. This repository measures that structure with
an exhaustive subset lattice and quantifies three ways the measurement itself
can mislead: evaluating on observed availability, searching subsets, and
trusting a query-conditioned router.

## What is here

| Path | Contents |
|---|---|
| `src/` | The full analysis pipeline, one stage per script (see below). |
| `src/common/` | Shared cohort construction, metrics, patient-level statistics, query text. |
| `results/` | Every CSV the paper's numbers derive from; regenerated deterministically by the pipeline. |
| `data_local/` | Frozen cohort pins only (`cohort_frozen_k5.json`, `cohort_frozen_k7.json`); thresholds live in `results/goat_thresholds.json`. No image data. |

## Pipeline

1. **Encode** — `src/extract_extra.py`: mean-pooled frozen V-JEPA2
   embeddings per sequence; skips work already done, so it resumes cleanly.
2. **Cache** — `src/rebuild_cache.py`: assembles the pinned complete-case
   cohort (`data_local/cohort_frozen_k5.json`) and asserts it has not moved.
3. **Lattice** — `src/subset_lattice.py`, `src/shapley_synergy.py`: all
   $2^5-1$ input subsets for every finding; exact Shapley values and
   interactions; patient-bootstrap rank stability (`src/rank_ci.py`).
4. **Selection under control** — `src/selection_optimism.py`,
   `src/risk_controlled.py`, `src/risk_validate.py`: strictly nested
   selection, a label-permutation reference for search optimism, and a
   held-out audit of a nominal non-inferiority selector, reported as a
   diagnosis.
5. **Routing** — `src/queryseq_model.py`, `src/run_methods.py`,
   `src/query_pathways.py`: one architecture, five gate parameterisations,
   and the 2×2 query-pathway intervention.
6. **Robustness** — `src/external_shift.py`, `src/natural_missingness.py`,
   `src/availability_decomposition.py`: held-out-vendor and field-strength
   re-splits, naturally incomplete studies, and the availability confound.
7. **Replication** — `src/goat_lattice.py`: the pre-specified BraTS-GoAT
   analysis (1,351 cases, 4 modalities, 15 subsets).

Analyses run with three seeds and patient-level cross-validation throughout;
every displayed number traces to a CSV in `results/` via
`src/verify_numbers.py`.

## Data access

No image data is redistributed here.

- **MR-RATE** is gated on Hugging Face (CC BY-NC-SA 4.0): accept the terms,
  set `HF_TOKEN`, and run the extraction; then point `QS_FEATROOT` at the
  directory holding the extracted features.
- **BraTS-GoAT** is obtained through the challenge's Synapse registration and
  is subject to its data-use terms.

## Environment

Python ≥ 3.10 with `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`,
`matplotlib`, `transformers`, and `nibabel`. Encoding needs one GPU; all
statistics and tables rebuild on CPU from `results/`.
