"""Metrics, defined once.

The upstream corpus code carries three copies of `ece` with different bin defaults and five
mutually incompatible `load_split` signatures; the copies had already drifted by
the time anyone noticed.  Everything numeric this paper reports comes from here.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auroc(y, s):
    """AUROC, or nan when the label vector is degenerate."""
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(roc_auc_score(y, s))


def safe_auprc(y, s):
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, s))


def ece(prob, y, bins=15):
    """Expected calibration error, equal-width bins."""
    prob = np.asarray(prob, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (prob > edges[i]) & (prob <= edges[i + 1])
        if not m.any():
            continue
        e += m.sum() / n * abs(y[m].mean() - prob[m].mean())
    return float(e)


def brier(prob, y):
    return float(np.mean((np.asarray(prob, float) - np.asarray(y, float)) ** 2))


def macro(vals):
    """Mean over findings, ignoring findings that were not evaluable."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else np.nan


def per_finding(Y, S, keep):
    """AUROC and AUPRC for each kept finding.

    Y : (n, F) labels, S : (n, F) scores, keep : list of column indices.
    """
    out = []
    for j in keep:
        out.append({
            "finding_idx": j,
            "n_pos": int(Y[:, j].sum()),
            "auroc": safe_auroc(Y[:, j], S[:, j]),
            "auprc": safe_auprc(Y[:, j], S[:, j]),
        })
    return out


def worst(vals):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return float(v.min()) if len(v) else np.nan
