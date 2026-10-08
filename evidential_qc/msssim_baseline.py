"""MS-SSIM baseline on the same splits as the learned models.

Each print is compared with the reference image of its geometry using multi-scale SSIM
(Wang, Simoncelli & Bovik, 2003). A per-geometry threshold is fitted on the training images by
maximising Youden's J (TPR - FPR); test images scoring >= threshold are called acceptable.

    python msssim_baseline.py --repo /path/to/AI-Based-3D-Bioprinting_Accuracy --out results
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from splits import grouped_cv, paper_split

REF = {"line": "IDSR_reference.png", "square_grid": "IDSQR_reference.png",
       "circle": "IDCR_reference.png", "circle_grid": "IDCBR_reference.png"}
W = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])  # Wang et al. 2003 scale weights


def _ssim_cs(x, y, c1=(0.01 * 255) ** 2, c2=(0.03 * 255) ** 2):
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    cs = (2 * sxy + c2) / (sxx + syy + c2)
    ssim = ((2 * mx * my + c1) / (mx * mx + my * my + c1)) * cs
    return ssim.mean(), cs.mean()


def ms_ssim(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    vals = []
    for s in range(5):
        ssim, cs = _ssim_cs(x, y)
        vals.append(max(cs if s < 4 else ssim, 1e-6))   # relu-style guard against negative values
        x, y = cv2.pyrDown(x), cv2.pyrDown(y)
    return float(np.prod(np.array(vals) ** W))


def load_gray(path, size=256):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)


def fit_predict(df, tr, te):
    out = []
    for g, d_te in df.iloc[te].groupby("geometry"):
        d_tr = df.iloc[tr][df.iloc[tr].geometry == g]
        fpr, tpr, thr = roc_curve(d_tr.label, d_tr.score)
        t = thr[np.argmax(tpr - fpr)]
        d = d_te[["path", "label", "geometry", "session", "score"]].copy()
        d["threshold"] = t
        d["p"] = (d.score >= t).astype(float)   # hard decision; AUC is computed from the score instead
        # Score relative to its geometry's threshold, so pooled AUC ranks images consistently with the decisions
        d["margin"] = d.score - t
        out.append(d)
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="..", help="dataset repository root (default: ..)")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.manifest)
    refs = {g: load_gray(os.path.join(a.repo, "data", f)) for g, f in REF.items()}
    df["score"] = [ms_ssim(load_gray(os.path.join(a.repo, p)), refs[g]) for p, g in zip(df.path, df.geometry)]
    df.to_csv(os.path.join(a.out, "msssim_scores.csv"), index=False)

    tr, va = paper_split(df, seed=42)
    fit_predict(df, tr, va).to_csv(os.path.join(a.out, "msssim_paper_split_preds.csv"), index=False)

    use = df[df.usable].reset_index(drop=True)
    cv = []
    for k, (tr, te) in enumerate(grouped_cv(use, k=5, seed=42)):
        cv.append(fit_predict(use, tr, te).assign(fold=k))
    pd.concat(cv).to_csv(os.path.join(a.out, "msssim_cv_preds.csv"), index=False)
    print("MS-SSIM scores written to", a.out)


if __name__ == "__main__":
    main()
