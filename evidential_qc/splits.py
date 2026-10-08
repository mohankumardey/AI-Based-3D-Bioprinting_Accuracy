"""Data splits. Torch-free.

paper_split     : single 80/20 split stratified on geometry x label (the original protocol).
grouped_cv      : K folds, whole fabrication-session blocks held out together
                  (StratifiedGroupKFold on geometry x label).
logo            : leave-one-geometry-out (generalization to an unseen geometry).
inner_val       : a validation subset carved from a training fold, used only for checkpoint
                  selection and the LR scheduler, so the test fold is never touched during training.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def strat_key(df):
    return df.geometry.astype(str) + "_" + df.label.astype(str)


def paper_split(df, seed=42, val_frac=0.2):
    idx = np.arange(len(df))
    tr, va = train_test_split(idx, test_size=val_frac, stratify=strat_key(df), random_state=seed)
    return np.sort(tr), np.sort(va)


def grouped_cv(df, k=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(np.sort(tr), np.sort(te)) for tr, te in sgkf.split(np.zeros(len(df)), strat_key(df), df.session)]


def logo(df, geometries):
    out = []
    for g in geometries:
        te = np.where(df.geometry.values == g)[0]
        tr = np.where(df.geometry.values != g)[0]
        out.append((g, tr, te))
    return out


def inner_val(df, train_idx, seed=42, frac=0.15):
    """Hold out whole sessions from the training fold for model selection."""
    sub = df.iloc[train_idx]
    k = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(np.zeros(len(sub)), strat_key(sub), sub.session))
    return train_idx[tr], train_idx[va]
