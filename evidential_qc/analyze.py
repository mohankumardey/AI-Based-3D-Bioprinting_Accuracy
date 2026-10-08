"""Turn saved predictions into tables and figures. Torch-free.

    python analyze.py --results results --out report
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from metrics import summarize, summarize_by

KEYS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc", "ece", "brier"]
ORDER = ["msssim", "resnet_bce", "resnet_geom_bce", "resnet_beta", "full", "full_onehot",
         "full_embed8", "full_kl", "full_frozen"]


def rank(d):
    """Ranking score for AUC: the MS-SSIM margin when present (its p is a hard decision), else p."""
    return d["margin"] if "margin" in d else d["p"]


def load_group(results, group):
    runs = {}
    for f in sorted(glob.glob(os.path.join(results, group, "**", "preds.csv"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(f), os.path.join(results, group)).split(os.sep)
        runs.setdefault(rel[0], {})["/".join(rel[1:]) or rel[0]] = pd.read_csv(f)
    return runs


def mean_sd_table(per_run):
    """per_run: {variant: {fold: preds}} -> mean ± SD over folds, plus pooled values."""
    rows = []
    for v in sorted(per_run, key=lambda x: ORDER.index(x) if x in ORDER else 99):
        folds = per_run[v]
        fs = pd.DataFrame([summarize(d.label, d.p, score=rank(d)) for d in folds.values()])
        allf = pd.concat(folds.values())
        pooled = summarize(allf.label, allf.p, score=rank(allf))
        r = dict(variant=v, n_runs=len(folds))
        for k in KEYS:
            r[k] = f"{fs[k].mean():.3f} ± {fs[k].std(ddof=1) if len(fs) > 1 else 0:.3f}"
        r["pooled_acc"] = round(pooled["accuracy"], 4)
        r["pooled_TP_TN_FP_FN"] = f"{pooled['TP']}/{pooled['TN']}/{pooled['FP']}/{pooled['FN']}"
        rows.append(r)
    return pd.DataFrame(rows)


def paired_compare(a, b, n_boot=2000, seed=0):
    """Compare two variants on the same pooled test images. McNemar exact test on correctness,
    bootstrap 95% CI for the accuracy and AUC difference (b - a)."""
    m = a.merge(b, on=["path", "label"], suffixes=("_a", "_b"))
    ca, cb = (m.p_a >= .5) == m.label, (m.p_b >= .5) == m.label
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pval = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    rng = np.random.default_rng(seed)
    y, pa, pb = m.label.values, m.p_a.values, m.p_b.values
    ra = (m.margin_a if "margin_a" in m else m.margin) if ("margin" in a) else pa
    rb = (m.margin_b if "margin_b" in m else m.margin) if ("margin" in b) else pb
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    dacc, dauc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(m), len(m))
        if 0 < y[i].sum() < len(i):
            dacc.append(((pb[i] >= .5) == y[i]).mean() - ((pa[i] >= .5) == y[i]).mean())
            dauc.append(roc_auc_score(y[i], rb[i]) - roc_auc_score(y[i], ra[i]))
    q = lambda z: f"{np.mean(z):+.3f} [{np.percentile(z, 2.5):+.3f}, {np.percentile(z, 97.5):+.3f}]"
    return dict(n=len(m), only_a_right=n01, only_b_right=n10, mcnemar_p=round(pval, 4),
                d_acc_95ci=q(dacc), d_auc_95ci=q(dauc))


def plot_history(hdir, title, fn):
    h = pd.read_csv(os.path.join(hdir, "history.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.sel_loss, label="validation")
    ax[1].plot(h.epoch, h.train_auc, label="train"); ax[1].plot(h.epoch, h.sel_auc, label="validation")
    be = int(h.sel_loss.idxmin()) + 1
    for a_, yl in zip(ax, ["NLL loss", "ROC-AUC"]):
        a_.axvline(be, ls="--", c="gray", lw=1, label=f"selected epoch {be}")
        a_.set_xlabel("epoch"); a_.set_ylabel(yl); a_.legend(frameon=False)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)


def reliability(df, fn, title):
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(df.p, bins) - 1, 0, 9)
    acc = [df.label[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    conf = [df.p[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", c="gray"); ax.plot(conf, acc, "o-")
    ax.set_xlabel("predicted P(good)"); ax.set_ylabel("observed fraction good"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = ["# Experiment report\n", "All numbers are computed from saved per-image predictions (metrics.py).\n"]

    # ---- 1. original single-split protocol
    rep = load_group(a.results, "repro")
    if rep:
        md.append("## 1. Original protocol (single 80/20 split, all 864 files, checkpoint chosen on the scored split)\n")
        for name, d in rep.items():
            d = list(d.values())[0]
            t = summarize_by(d)
            t.to_csv(os.path.join(a.out, f"repro_{name}.csv"), index=False)
            md.append(f"### {name}\n\n" + t.round(4).to_markdown(index=False) + "\n")
            meta = json.load(open(os.path.join(a.results, "repro", name, "meta.json")))
            md.append(f"best epoch {meta['best_epoch']}, trainable params {meta['trainable_params']:,}\n")
            plot_history(os.path.join(a.results, "repro", name), f"Original protocol – {name}",
                         os.path.join(a.out, f"curves_repro_{name}.png"))
        s = pd.DataFrame([dict(run=n, **summarize(d.label, d.p)) for n, dd in rep.items() for d in dd.values()])
        md.append("### Across seeds (each a different 80/20 split)\n\n"
                  + s[["run"] + KEYS].round(4).to_markdown(index=False) + "\n\n"
                  + s[KEYS].agg(["mean", "std"]).round(4).to_markdown() + "\n")
    msp = os.path.join(a.results, "msssim_paper_split_preds.csv")
    if os.path.exists(msp):
        m = pd.read_csv(msp)
        md.append("### MS-SSIM baseline on the seed-42 paper split (AUC from score margin; ECE/Brier are of hard decisions)\n\n"
                  + pd.DataFrame([dict(group="ALL", **summarize(m.label, m.p, score=rank(m)))]
                                 + [dict(group=g, **summarize(d.label, d.p, score=rank(d))) for g, d in m.groupby("geometry")])
                  [["group", "N"] + KEYS].round(4).to_markdown(index=False) + "\n")

    # ---- 2. ablations under grouped CV
    cv = load_group(a.results, "cv")
    ms = os.path.join(a.results, "msssim_cv_preds.csv")
    if os.path.exists(ms):
        m = pd.read_csv(ms); cv["msssim"] = {f"fold{k}": d for k, d in m.groupby("fold")}
    if cv:
        md.append("## 2. Ablations, 5-fold session-grouped CV (mean ± SD over folds)\n")
        t = mean_sd_table(cv); t.to_csv(os.path.join(a.out, "ablation_cv.csv"), index=False)
        md.append(t.to_markdown(index=False) + "\n")
        pooled = {v: pd.concat(f.values()) for v, f in cv.items()}
        if "full" in pooled:
            g = summarize_by(pooled["full"]); g.to_csv(os.path.join(a.out, "full_cv_by_geometry.csv"), index=False)
            md.append("### Full model, pooled CV predictions by geometry\n\n" + g.round(4).to_markdown(index=False) + "\n")
            reliability(pooled["full"], os.path.join(a.out, "reliability_full_cv.png"), "Full model (CV)")
            comps = []
            for base in ["msssim", "resnet_bce", "resnet_geom_bce", "resnet_beta", "full_onehot", "full_embed8", "full_kl"]:
                if base in pooled:
                    comps.append(dict(comparison=f"{base} -> full", **paired_compare(pooled[base], pooled["full"])))
            if "resnet_bce" in pooled and "resnet_geom_bce" in pooled:
                comps.append(dict(comparison="resnet_bce -> resnet_geom_bce",
                                  **paired_compare(pooled["resnet_bce"], pooled["resnet_geom_bce"])))
            c = pd.DataFrame(comps); c.to_csv(os.path.join(a.out, "paired_comparisons.csv"), index=False)
            md.append("### Paired comparisons on identical test images (positive = second model better)\n\n"
                      + c.to_markdown(index=False) + "\n")
            unc = []
            for v in ["resnet_bce", "resnet_beta", "full", "full_kl"]:
                if v not in pooled:
                    continue
                d = pooled[v]
                # Baseline: does plain closeness of p to 0.5 flag errors as well as u does?
                conf = summarize(d.label, d.p, u=-(d.p - .5).abs())["auc_u_detects_errors"]
                if "u" not in d:
                    unc.append(dict(variant=v, auc_conf_detects_errors=conf))
                    continue
                s = summarize(d.label, d.p, u=d.u)
                unc.append(dict(variant=v, mean_u_correct=s["mean_u_correct"], mean_u_wrong=s["mean_u_wrong"],
                                auc_u_detects_errors=s["auc_u_detects_errors"], auc_conf_detects_errors=conf,
                                corr_u_vs_abs_p_minus_half=float(np.corrcoef(d.u, (d.p - .5).abs())[0, 1]),
                                median_alpha_plus_beta=float((d.alpha + d.beta).median())))
            if unc:
                md.append("### Does u = 2/(α+β) flag errors? (AUC 0.5 = no better than chance; auc_conf = same test using only −|p−0.5|)\n\n"
                          + pd.DataFrame(unc).round(4).to_markdown(index=False) + "\n")
        for v in ["full", "resnet_bce"]:
            d = os.path.join(a.results, "cv", v, "fold0")
            if os.path.exists(os.path.join(d, "history.csv")):
                plot_history(d, f"{v}, CV fold 0 (validation = inner session-held-out split)",
                             os.path.join(a.out, f"curves_cv_{v}_fold0.png"))

    # ---- 3. leave-one-geometry-out
    lg = load_group(a.results, "logo")
    if lg:
        rows = [dict(variant=v, held_out=g, **{k: round(summarize(d.label, d.p)[k], 3) for k in KEYS[:6]})
                for v, gs in lg.items() for g, d in gs.items()]
        t = pd.DataFrame(rows); t.to_csv(os.path.join(a.out, "logo.csv"), index=False)
        md.append("## 3. Leave-one-geometry-out\n\n" + t.to_markdown(index=False) + "\n")

    open(os.path.join(a.out, "REPORT.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
