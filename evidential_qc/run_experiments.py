"""Run all experiments. Resumable: finished runs are skipped.

    python run_experiments.py --smoke          # ~1 min sanity check
    python run_experiments.py                  # everything (~3.5 h on an M2 Pro)
    python run_experiments.py --only repro     # one group

Groups
  repro : the original single-split protocol (all 864 files, single composite-stratified 80/20 split,
          checkpoint chosen on the same 20% it is scored on, no resolution harmonisation), 3 seeds.
  cv    : ablations under 5-fold session-grouped CV; checkpoint chosen on an inner session-held-out
          validation split, test fold never seen during training. Conflicting duplicates removed.
  logo  : leave-one-geometry-out (is the model usable on a geometry it has never seen?).
"""
import argparse
import os

import pandas as pd

from splits import grouped_cv, inner_val, logo, paper_split
from train import train_one

GEOMS = ["line", "square_grid", "circle", "circle_grid"]

# name -> config. "full" is the proposed model.
VARIANTS = {
    "resnet_bce":        dict(geom="none",    head="bce"),    # conventional ResNet-18 classifier
    "resnet_geom_bce":   dict(geom="embed32", head="bce"),    # + geometry embedding
    "resnet_beta":       dict(geom="none",    head="beta"),   # + evidential head, no geometry
    "full":              dict(geom="embed32", head="beta"),   # proposed model
    "full_onehot":       dict(geom="onehot",  head="beta"),   # one-hot geometry code
    "full_embed8":       dict(geom="embed8",  head="beta"),   # 8-d embedding
    "full_kl":           dict(geom="embed32", head="beta", kl_weight=1.0),  # Sensoy-style KL term
    "full_frozen":       dict(geom="embed32", head="beta", freeze=True),    # ResNet-18 frozen
}
LOGO_VARIANTS = ["resnet_beta", "full"]


def plan(df_all, df_use, seeds):
    runs = []
    for s in seeds:
        tr, va = paper_split(df_all, seed=s)
        runs.append(("repro", f"repro/full_seed{s}", df_all, tr, va, va,
                     dict(VARIANTS["full"], harmonize=False), s))
    for v, cfg in VARIANTS.items():
        for k, (tr, te) in enumerate(grouped_cv(df_use, k=5, seed=42)):
            itr, iva = inner_val(df_use, tr, seed=42)
            runs.append(("cv", f"cv/{v}/fold{k}", df_use, itr, iva, te, dict(cfg, harmonize=True), 42))
    for v in LOGO_VARIANTS:
        for g, tr, te in logo(df_use, GEOMS):
            itr, iva = inner_val(df_use, tr, seed=42)
            runs.append(("logo", f"logo/{v}/{g}", df_use, itr, iva, te, dict(VARIANTS[v], harmonize=True), 42))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="..", help="dataset repository root (default: ..)")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", choices=["repro", "cv", "logo"])
    ap.add_argument("--variants", nargs="*", help="restrict cv/logo to these variant names")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 3 runs, written to results_smoke/")
    a = ap.parse_args()

    df_all = pd.read_csv(a.manifest)
    df_use = df_all[df_all.usable].reset_index(drop=True)
    runs = plan(df_all, df_use, a.seeds)
    if a.only:
        runs = [r for r in runs if r[0] in a.only]
    if a.variants:
        runs = [r for r in runs if r[0] == "repro" or r[1].split("/")[1] in a.variants]
    out, epochs = a.out, a.epochs
    if a.smoke:
        pick = ["repro/full_seed42", "cv/resnet_bce/fold0", "cv/full_kl/fold0"]
        runs, out, epochs = [r for r in runs if r[1] in pick], "results_smoke", 2

    todo = [r for r in runs if not os.path.exists(os.path.join(out, r[1], "preds.csv"))]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to go")
    for i, (grp, name, df, tr, sel, ev, cfg, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {name}  train={len(tr)} select={len(sel)} eval={len(ev)}  cfg={cfg}", flush=True)
        train_one(df, a.repo, tr, sel, ev, dict(cfg, epochs=epochs), os.path.join(out, name), seed=seed)
    print("\nAll done. Next: python analyze.py --results", out)


if __name__ == "__main__":
    main()
