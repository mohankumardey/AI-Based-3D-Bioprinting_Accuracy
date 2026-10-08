"""Result figures from saved predictions/histories. Torch-free.

    python make_figures.py --results results --out figures
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import counts, summarize

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
GEOMS = [("square_grid", "Square grid"), ("line", "Line"), ("circle", "Circular"), ("circle_grid", "Circular grid")]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def pooled(results, variant):
    return pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{results}/cv/{variant}/fold*/preds.csv")))


def curves(ax_loss, ax_auc, hdir, title):
    h = pd.read_csv(os.path.join(hdir, "history.csv"))
    be = int(h.sel_loss.idxmin()) + 1
    for ax, tr, va, yl in [(ax_loss, "train_loss", "sel_loss", "NLL loss"), (ax_auc, "train_auc", "sel_auc", "ROC-AUC")]:
        ax.plot(h.epoch, h[tr], color=BLUE, lw=1.5, label="Training")
        ax.plot(h.epoch, h[va], color=ORANGE, lw=1.5, label="Validation")
        ax.axvline(be, color=GRAY, ls="--", lw=1, label=f"Selected epoch ({be})")
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax_loss.set_title(title, loc="left", fontsize=9, color=INK)


def confusion(ax, y, p, title):
    c = counts(y, (np.asarray(p) >= .5).astype(int))
    m = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    frac = m / m.sum(1, keepdims=True)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}\n({100 * frac[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if frac[i, j] > .55 else INK, fontsize=9)
    ax.set_xticks([0, 1], ["Defective", "Acceptable"]); ax.set_yticks([0, 1], ["Defective", "Acceptable"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    full, ms = pooled(a.results, "full"), pd.read_csv(f"{a.results}/msssim_cv_preds.csv")

    # Figure 10: loss and AUC curves, original protocol and one CV fold
    fig, ax = plt.subplots(2, 2, figsize=(8, 5.6))
    curves(ax[0, 0], ax[0, 1], f"{a.results}/repro/full_seed42", "(a) Original protocol, 80/20 split (seed 42)")
    curves(ax[1, 0], ax[1, 1], f"{a.results}/cv/full/fold0", "(b) Session-grouped CV, fold 1 (validation = held-out sessions)")
    ax[0, 1].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig10_training_curves.png"); plt.close(fig)

    # Figure 11: confusion matrices, full model vs MS-SSIM (pooled CV)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
    confusion(ax[0], full.label, full.p, f"(a) Full model (N = {len(full)})")
    confusion(ax[1], ms.label, ms.p, f"(b) MS-SSIM baseline (N = {len(ms)})")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig11_confusion.png"); plt.close(fig)

    # Figure 12: per-geometry confusion matrices, full model
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    for k, (g, name) in enumerate(GEOMS):
        d = full[full.geometry == g]
        s = summarize(d.label, d.p)
        confusion(ax[k], d.label, d.p, f"({'abcd'[k]}) {name}  (bal. acc. {s['balanced_accuracy']:.2f})")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig12_geometry_confusion.png"); plt.close(fig)

    # Figure 13: per-geometry accuracy, full model vs MS-SSIM
    rows = []
    for g, name in GEOMS:
        rows.append((name, summarize(full[full.geometry == g].label, full[full.geometry == g].p)["accuracy"],
                     summarize(ms[ms.geometry == g].label, ms[ms.geometry == g].p)["accuracy"]))
    x, w = np.arange(len(rows)), 0.36
    fig, ax = plt.subplots(figsize=(6, 3.4))
    b1 = ax.bar(x - w / 2 - 0.01, [r[1] for r in rows], w, color=BLUE, label="Full model")
    b2 = ax.bar(x + w / 2 + 0.01, [r[2] for r in rows], w, color=ORANGE, label="MS-SSIM baseline")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{100 * b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy (pooled CV)")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True); ax.legend(loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(f"{a.out}/fig13_geometry_vs_msssim.png"); plt.close(fig)

    # Figure 14: class-conditioned probability distribution and reliability diagram
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    bins = np.linspace(0, 1, 21)
    hb, _ = np.histogram(full.p[full.label == 0], bins)
    hg, _ = np.histogram(full.p[full.label == 1], bins)
    c, bw = (bins[:-1] + bins[1:]) / 2, (bins[1] - bins[0]) * 0.42
    ax[0].bar(c - bw / 2 - 0.002, hb, bw, color=ORANGE, label="Defective (true)")
    ax[0].bar(c + bw / 2 + 0.002, hg, bw, color=BLUE, label="Acceptable (true)")
    ax[0].axvline(.5, color=GRAY, ls="--", lw=1)
    ax[0].set_xlabel("Predicted P(acceptable)"); ax[0].set_ylabel("Prints"); ax[0].legend(loc="upper center")
    ax[0].set_title("(a) Predicted probability by true class", loc="left", fontsize=9, color=INK)
    idx = np.clip(np.digitize(full.p, np.linspace(0, 1, 11)) - 1, 0, 9)
    conf = [full.p[idx == b].mean() for b in range(10) if (idx == b).any()]
    obs = [full.label[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax[1].plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
    ax[1].plot(conf, obs, "o-", color=BLUE, lw=1.5, ms=5)
    s = summarize(full.label, full.p)
    ax[1].set_xlabel("Predicted P(acceptable)"); ax[1].set_ylabel("Observed fraction acceptable")
    ax[1].set_title(f"(b) Reliability (ECE = {s['ece']:.3f})", loc="left", fontsize=9, color=INK)
    ax[1].set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig14_probabilities.png"); plt.close(fig)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
