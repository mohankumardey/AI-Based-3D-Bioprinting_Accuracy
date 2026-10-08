"""Figure 15: qualitative examples from out-of-fold CV predictions of the full model.

For each geometry: the reference image, the correctly classified acceptable print with the highest
P(acceptable), and the correctly classified defective print with the lowest P(acceptable).
Add --errors to append a column with the most confident misclassification per geometry.

    python make_fig15.py [--errors]
"""
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

GREEN, RED, INK, MUTED = "#1f7a3a", "#c62828", "#0b0b0b", "#52514e"
ROWS = [("circle_grid", "Circular grid", "IDCBR_reference.png"), ("circle", "Circular pattern", "IDCR_reference.png"),
        ("line", "Line pattern", "IDSR_reference.png"), ("square_grid", "Square grid", "IDSQR_reference.png")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="..")
    ap.add_argument("--results", default="results")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--out", default="figures/fig15_qualitative")
    a = ap.parse_args()
    d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{a.results}/cv/full/fold*/preds.csv")))
    d["correct"] = (d.p >= .5) == d.label
    ncol = 4 if a.errors else 3
    fig, axes = plt.subplots(4, ncol, figsize=(2.55 * ncol, 11.2))
    picks = []
    for i, (g, name, ref) in enumerate(ROWS):
        dg = d[d.geometry == g]
        good = dg[(dg.label == 1) & dg.correct].sort_values("p", ascending=False).iloc[0]
        bad = dg[(dg.label == 0) & dg.correct].sort_values("p").iloc[0]
        cells = [("ref", f"{a.repo}/data/{ref}", f"Reference:\n{name}", INK),
                 ("good", f"{a.repo}/{good.path}", f"Acceptable print\np = {good.p:.3f}, u = {good.u:.2f}", GREEN),
                 ("bad", f"{a.repo}/{bad.path}", f"Defective print\np = {bad.p:.3f}, u = {bad.u:.2f}", RED)]
        rows = [good, bad]
        if a.errors:
            err = dg[~dg.correct].assign(conf=lambda x: (x.p - .5).abs()).sort_values("conf", ascending=False).iloc[0]
            truth = "acceptable" if err.label == 1 else "defective"
            cells.append(("err", f"{a.repo}/{err.path}", f"Misclassified (true: {truth})\np = {err.p:.3f}, u = {err.u:.2f}", MUTED))
            rows.append(err)
        for j, (_, path, title, color) in enumerate(cells):
            ax = axes[i, j]
            ax.imshow(Image.open(path).convert("L").resize((256, 256), Image.BOX), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(title, fontsize=9.5, color=color, fontweight="bold", pad=5, linespacing=1.3)
            if j > 0:
                r = rows[j - 1]
                ax.text(0.5, -0.04, f"{r.prefix}_{int(r.print_id)}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.5, color=MUTED)
        picks += [dict(geometry=g, role=k, path=r.path, label=r.label, p=r.p, u=r.u)
                  for k, r in zip(["acceptable", "defective", "error"][:len(rows)], rows)]
    fig.tight_layout(h_pad=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    pd.DataFrame(picks).to_csv(f"{a.out}_picks.csv", index=False)
    print(pd.DataFrame(picks).to_string(index=False))


if __name__ == "__main__":
    main()
