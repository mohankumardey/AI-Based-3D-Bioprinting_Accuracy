"""Figure 5: model architecture (vector). python make_fig5.py -> figures/fig5_architecture.{png,pdf,svg}"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import beta as beta_dist

INK, MUTED, LINE = "#1d1d1b", "#5a5955", "#9a9993"
BLUE, BLUE_BG = "#2a78d6", "#e8f1fc"
ORANGE, ORANGE_BG = "#d9622b", "#fdeee6"
GREEN, GREEN_BG = "#1f7a52", "#e6f4ec"
PANEL = "#f6f6f4"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 9})

fig = plt.figure(figsize=(13.5, 5.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 135); ax.set_ylim(0, 56); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.2, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def text(x, y, s, size=9, color=INK, weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def arrow(x0, y0, x1, y1, color=LINE, cs="arc3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.3,
                                 color=color, connectionstyle=cs, shrinkA=0, shrinkB=0))


# column panels and titles
cols = [(1, 20, "Inputs"), (22.5, 25, "Feature extraction"), (50, 11, "Fusion"),
        (63.5, 22, "MLP"), (88, 23, "Evidential head"), (113.5, 20.5, "Outputs")]
for x, w, t in cols:
    box(x, 11.5, w, 41, PANEL, "none", r=1.5)
    text(x + w / 2, 50, t, size=10.5, weight="bold")

# ---- inputs
ya, yb = 36, 20  # image branch / geometry branch centre lines
for k in range(3):
    ax.add_patch(Rectangle((4.5 + 1.1 * k, ya - 3.5 - 1.1 * k), 8, 8, fc="white", ec=BLUE, lw=1.1))
yy, xx = np.mgrid[0:1:40j, 0:1:40j]
grid = ((np.abs(np.sin(xx * np.pi * 4)) < .25) | (np.abs(np.sin(yy * np.pi * 4)) < .25)).astype(float)
ax.imshow(grid, extent=(6.7, 14.7, ya - 5.7, ya + 2.3), cmap="Blues", vmin=-.3, vmax=1.6, zorder=3)
ax.add_patch(Rectangle((6.7, ya - 5.7), 8, 8, fc="none", ec=BLUE, lw=1.1, zorder=4))
text(11, ya + 8.3, "(A) Printed scaffold\nimage $I_p$", size=9, weight="bold", linespacing=1.3)
text(11, ya - 7.6, "[B, 3, 224, 224]", size=8, color=MUTED)
text(11, yb + 5.0, "(B) Geometry ID", size=9, weight="bold")
for k, lab in enumerate(["Line", "Sq. grid", "Circle", "Circ. grid"]):
    box(3 + 4 * k, yb - 1.6, 3.6, 3.2, ORANGE_BG, ORANGE, lw=1, r=.5)
    text(4.8 + 4 * k, yb, str(k + 1), size=8.5, color=ORANGE, weight="bold")
text(11, yb - 4.2, "$s \\in \\{1, 2, 3, 4\\}$   [B]", size=8.5)
text(11, yb - 6.8, "line · square grid · circle · circle grid", size=7, color=MUTED)

# ---- feature extraction
box(24, ya - 5.5, 22, 11, BLUE_BG, BLUE)
text(35, ya + 2.6, "ResNet-18 backbone", size=9.5, weight="bold", color=BLUE)
text(35, ya - 0.3, "ImageNet-pretrained, fine-tuned", size=8)
text(35, ya - 3.0, "FC removed · global avg. pooling", size=8, color=MUTED)
text(35, ya - 7.6, "$v = \\mathcal{B}(I_p) \\in \\mathbb{R}^{512}$   [B, 512]", size=8.5)
box(24, yb - 5.5, 22, 11, ORANGE_BG, ORANGE)
text(35, yb + 2.6, "Geometry embedding", size=9.5, weight="bold", color=ORANGE)
text(35, yb - 0.3, "learned lookup table (4 × 32)", size=8)
text(35, yb - 3.0, "one vector per geometry", size=8, color=MUTED)
text(35, yb - 7.4, "$e_s \\in \\mathbb{R}^{32}$   [B, 32]", size=8.5)
arrow(16.5, ya, 24, ya, BLUE); arrow(19.5, yb, 24, yb, ORANGE)

# ---- fusion
box(51.5, yb - 4, 8, ya - yb + 8, "white", INK, lw=1.2)
text(55.5, (ya + yb) / 2 + 3, "Concat", size=9.5, weight="bold")
text(55.5, (ya + yb) / 2 - 0.5, "$z = [v;\\, e_s]$", size=9)
text(55.5, (ya + yb) / 2 - 4, "$\\mathbb{R}^{544}$", size=9)
text(55.5, yb - 6.5, "[B, 544]", size=8, color=MUTED)
arrow(46, ya, 51.5, ya, BLUE); arrow(46, yb, 51.5, yb, ORANGE)

# ---- MLP
ym = (ya + yb) / 2
for k, (lab, sub) in enumerate([("Linear 544 → 256", "ReLU · Dropout 0.3"), ("Linear 256 → 256", "ReLU · Dropout 0.3")]):
    x = 65 + 10 * k
    box(x, ym - 11, 8.5, 22, GREEN_BG, GREEN)
    ax.text(x + 3.1, ym, lab, rotation=90, fontsize=9, fontweight="bold", color=GREEN, ha="center", va="center")
    ax.text(x + 5.9, ym, sub, rotation=90, fontsize=8, color=INK, ha="center", va="center")
arrow(59.5, ym, 65, ym, INK); arrow(73.5, ym, 75, ym, INK)
text(74.5, ym - 15, "$h \\in \\mathbb{R}^{256}$   [B, 256]", size=8.5)

# ---- evidential head
arrow(83.5, ym, 90, ym, INK)
box(90, ym - 2.6, 19, 5.2, GREEN_BG, GREEN)
text(99.5, ym, "Linear 256 → 2   $(z_1, z_2)$", size=8.8, weight="bold", color=GREEN)
arrow(99.5, ym - 2.6, 99.5, ym - 4.6, INK)
box(90, 13.2, 19, 10.2, "white", INK)
text(99.5, 21.0, "softplus + 1", size=8.8, weight="bold")
text(99.5, 18.3, "$\\alpha = \\mathrm{softplus}(z_1) + 1$", size=8.3)
text(99.5, 15.9, "$\\beta = \\mathrm{softplus}(z_2) + 1$", size=8.3)
text(99.5, 12.3, "$\\alpha, \\beta > 1$, one pair per image", size=7.3, color=MUTED)
# Beta density inset
ins = fig.add_axes([91 / 135, 34 / 56, 17 / 135, 9 / 56])
th = np.linspace(0.001, 0.999, 300)
for (a_, b_), c in [((9, 2.5), BLUE), ((2.2, 2.0), LINE)]:
    ins.plot(th, beta_dist.pdf(th, a_, b_), color=c, lw=1.4)
ins.set_xticks([0, 1]); ins.set_yticks([]); ins.tick_params(labelsize=7, colors=MUTED, length=2)
for s_ in ["top", "right", "left"]:
    ins.spines[s_].set_visible(False)
ins.spines["bottom"].set_color(LINE); ins.patch.set_alpha(0)
ins.set_xlabel("$\\theta$ = P(acceptable)", fontsize=7, color=MUTED, labelpad=0)
text(99.5, 45.3, "Beta($\\alpha$, $\\beta$) over $\\theta$", size=8)
text(106.6, 37.2, "confident", size=7, color=BLUE, ha="left")
text(93.5, 39.6, "uncertain", size=7, color=MUTED)

# ---- outputs
arrow(109, 20.5, 115, ya - 1, INK, "arc3,rad=-0.2"); arrow(109, 17, 115, 22, INK, "arc3,rad=0.1")
box(115, ya - 6, 17, 11, BLUE_BG, BLUE)
text(123.5, ya + 1.5, "$p = \\dfrac{\\alpha}{\\alpha + \\beta}$", size=10)
text(123.5, ya - 4.0, "P(acceptable print)", size=8, weight="bold", color=BLUE)
box(115, 16.5, 17, 11, "white", INK)
text(123.5, 24.0, "$u = \\dfrac{2}{\\alpha + \\beta}$", size=10)
text(123.5, 18.5, "predictive uncertainty", size=8, weight="bold")
text(123.5, 13.6, "acceptable if $p \\geq 0.5$", size=8, color=MUTED)

# ---- loss footer
box(22.5, 1.2, 89, 8.3, "white", LINE, lw=1)
text(25, 6.7, "Training loss (Eq. 1)", size=9, weight="bold", ha="left")
text(25, 3.4, "$\\mathcal{L}_{NLL} = -\\frac{1}{N}\\sum_i \\left[ y_i \\log \\frac{\\alpha_i}{\\alpha_i+\\beta_i} + (1-y_i)\\log \\frac{\\beta_i}{\\alpha_i+\\beta_i} \\right]$",
     size=9.5, ha="left")
text(110, 6.7, "Beta–Bernoulli NLL; Adam, lr 1e-3, batch 32, 50 epochs", size=8, color=MUTED, ha="right")
text(110, 3.4, "KL regulariser (Eq. 2) evaluated only as an ablation", size=8, color=MUTED, ha="right")

os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"figures/fig5_architecture.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("written figures/fig5_architecture.{png,pdf,svg}")
