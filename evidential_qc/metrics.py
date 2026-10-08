"""Evaluation metrics, computed from a per-image predictions table. Torch-free.

Every number reported in the paper should come from these functions applied to saved
predictions, so the confusion matrix, accuracy and per-geometry values cannot drift apart again.

Definitions (positive class = good / acceptable print):
  TP: good predicted good            TN: bad predicted bad
  FP: bad predicted good (defect accepted -- the risky error for QC)
  FN: good predicted bad (over-rejection)
  sensitivity = TP/(TP+FN)   specificity = TN/(TN+FP)
  accuracy = (TP+TN)/N       balanced accuracy = (sens+spec)/2
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def counts(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return dict(TP=int(((y == 1) & (yhat == 1)).sum()), TN=int(((y == 0) & (yhat == 0)).sum()),
                FP=int(((y == 0) & (yhat == 1)).sum()), FN=int(((y == 1) & (yhat == 0)).sum()))


def ece(y, p, n_bins=10):
    """Expected calibration error for P(good), equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def summarize(y, p, thr=0.5, u=None, score=None):
    """score: optional ranking score for AUC (defaults to p). Used for MS-SSIM, whose p is a hard 0/1 decision."""
    y, p = np.asarray(y), np.asarray(p, float)
    s = p if score is None else np.asarray(score, float)
    c = counts(y, (p >= thr).astype(int))
    n = len(y)
    sens = c["TP"] / max(1, c["TP"] + c["FN"])
    spec = c["TN"] / max(1, c["TN"] + c["FP"])
    prec = c["TP"] / max(1, c["TP"] + c["FP"])
    out = dict(N=n, n_good=int(y.sum()), n_bad=int(n - y.sum()), **c,
               accuracy=(c["TP"] + c["TN"]) / n, balanced_accuracy=(sens + spec) / 2,
               sensitivity=sens, specificity=spec, precision=prec,
               f1=2 * prec * sens / max(1e-12, prec + sens),
               auc=roc_auc_score(y, s) if 0 < y.sum() < n else np.nan,
               brier=float(np.mean((p - y) ** 2)), ece=ece(y, p))
    if u is not None:
        u = np.asarray(u, float)
        correct = (p >= thr).astype(int) == y
        out["mean_u_correct"] = float(u[correct].mean()) if correct.any() else np.nan
        out["mean_u_wrong"] = float(u[~correct].mean()) if (~correct).any() else np.nan
        # Does uncertainty rank errors above correct predictions? (0.5 = no signal)
        out["auc_u_detects_errors"] = roc_auc_score(~correct, u) if 0 < correct.sum() < n else np.nan
    return out


def summarize_by(df, by="geometry", thr=0.5):
    rows = [dict(group="ALL", **summarize(df.label, df.p, thr, df.get("u")))]
    for g, d in df.groupby(by):
        rows.append(dict(group=g, **summarize(d.label, d.p, thr, d.get("u"))))
    return pd.DataFrame(rows)
