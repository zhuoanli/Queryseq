"""Uncertainty and significance.

Studies are not independent -- a patient can contribute several -- so every
interval here resamples *patients*, not rows.  Ported from the corpus release's
prompt_families.boot_auroc and pivot_stats.exact_perm_p, which were the two
functions that survived review there.
"""
import itertools

import numpy as np
from sklearn.metrics import roc_auc_score

from .metrics import safe_auroc


def _index_by_group(groups):
    uniq = np.unique(groups)
    return uniq, {g: np.where(groups == g)[0] for g in uniq}


def boot_auroc(y, score, groups, rng, reps=1000, alpha=0.05):
    """Patient-level bootstrap CI for a single AUROC."""
    uniq, idx_by = _index_by_group(groups)
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        i = np.concatenate([idx_by[g] for g in pick])
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], score[i]))
    if not out:
        return np.nan, np.nan
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def paired_boot_delta(y, s_a, s_b, groups, rng, reps=1000, alpha=0.05):
    """CI and sign probability for AUROC(a) - AUROC(b) on the same patients.

    Paired at the patient level: each replicate resamples patients once and
    scores both methods on that same resample, so the shared patient draw
    cancels and the interval is about the method difference rather than about
    who happened to be in the split.
    """
    uniq, idx_by = _index_by_group(groups)
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        i = np.concatenate([idx_by[g] for g in pick])
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
    if not out:
        return np.nan, np.nan, np.nan, np.nan
    out = np.asarray(out)
    return (float(out.mean()),
            float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))),
            float((out > 0).mean()))


def boot_macro_delta(Y, Sa, Sb, keep, groups, rng, reps=1000, alpha=0.05):
    """Paired patient-level CI for the difference in *macro* AUROC.

    The macro average is recomputed inside each replicate, so the interval
    accounts for the correlation between findings rather than treating the
    per-finding AUROCs as independent draws.
    """
    uniq, idx_by = _index_by_group(groups)
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        i = np.concatenate([idx_by[g] for g in pick])
        da, db = [], []
        for j in keep:
            yj = Y[i, j]
            if 0 < yj.sum() < len(i):
                da.append(roc_auc_score(yj, Sa[i, j]))
                db.append(roc_auc_score(yj, Sb[i, j]))
        if da:
            out.append(np.mean(da) - np.mean(db))
    if not out:
        return np.nan, np.nan, np.nan, np.nan
    out = np.asarray(out)
    return (float(out.mean()),
            float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))),
            float((out > 0).mean()))


def exact_perm_p(scores, labels, n_max=200000, seed=0):
    """Permutation p-value; exact when the arrangement count is small.

    Used where positives are few enough that a bootstrap interval would be
    dishonest.  Returns (observed_auroc, p, n_arrangements, was_exact).
    """
    labels = np.asarray(labels)
    k = int(labels.sum())
    n = len(labels)
    obs = safe_auroc(labels, scores)
    if np.isnan(obs):
        return np.nan, np.nan, 0, False
    total = 1
    for i in range(k):
        total = total * (n - i) // (i + 1)
    if total <= n_max:
        cnt = 0
        for c in itertools.combinations(range(n), k):
            y = np.zeros(n, int)
            y[list(c)] = 1
            if roc_auc_score(y, scores) >= obs:
                cnt += 1
        return obs, cnt / total, int(total), True
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_max):
        y = np.zeros(n, int)
        y[rng.choice(n, k, replace=False)] = 1
        if roc_auc_score(y, scores) >= obs:
            cnt += 1
    return obs, cnt / n_max, n_max, False


def patient_folds(patients, k=5, seed=0):
    """K folds that never split a patient across folds.

    The atlas is built from out-of-fold predictions on the training split
    because the official evaluation splits are too thin to carry a 37-finding
    analysis; this is what keeps that legitimate.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(patients)
    rng.shuffle(uniq)
    assign = {g: i % k for i, g in enumerate(uniq)}
    fold = np.array([assign[p] for p in patients])
    return [(np.where(fold != i)[0], np.where(fold == i)[0]) for i in range(k)]
