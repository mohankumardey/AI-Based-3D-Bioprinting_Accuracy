"""Build a clean, auditable manifest of the 864 scaffold images.

Usage:
    python build_manifest.py --repo /path/to/AI-Based-3D-Bioprinting_Accuracy

Writes manifest.csv (one row per image file) and prints an audit summary.
Torch-free, so it runs anywhere.
"""
import argparse
import hashlib
import os
import re

import pandas as pd
from PIL import Image

# filename prefix -> (geometry, tip, sheet name in 3D_Bioprinting Features.xlsx)
# Verified visually against the four reference images in data/.
PREFIX_MAP = {
    "IDSR":  ("line",        "regular", "Line Pattern_Regular"),
    "IDST":  ("line",        "tapered", "Line Pattern_Tappered"),
    "IDSQR": ("square_grid", "regular", "Square Grid_Regular"),
    "ID":    ("square_grid", "tapered", "Square Grid_Tappered"),
    "IDCR":  ("circle",      "regular", "Circle_Regular"),
    "IDCT":  ("circle",      "tapered", "Circle_Tappered"),
    "IDCBR": ("circle_grid", "regular", "Circle Grid_Regular"),
    "IDCB":  ("circle_grid", "tapered", "Circle Grid_Tappered"),
}
GEOMETRIES = ["line", "square_grid", "circle", "circle_grid"]
FNAME = re.compile(r"^(ID[A-Z]*)_\s*\(?(\d+)\)?\.png$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="..", help="dataset repository root (default: ..)")
    ap.add_argument("--out", default="manifest.csv")
    a = ap.parse_args()

    xls = pd.ExcelFile(os.path.join(a.repo, "3D_Bioprinting Features.xlsx"))
    params = {}
    for sheet in xls.sheet_names:
        d = xls.parse(sheet).iloc[:, :5]
        d.columns = ["id", "gauge", "temp_c", "pressure_psi", "speed_mm_s"]
        params[sheet] = d.set_index("id")

    rows = []
    for folder, label in [("Good_png", 1), ("Bad_png", 0)]:
        for f in sorted(os.listdir(os.path.join(a.repo, "data", folder))):
            m = FNAME.match(f)
            if not m:
                raise ValueError(f"Unrecognised filename: {folder}/{f}")
            prefix, pid = m.group(1), int(m.group(2))
            geom, tip, sheet = PREFIX_MAP[prefix]
            path = os.path.join("data", folder, f)
            full = os.path.join(a.repo, path)
            p = params[sheet].loc[pid]
            rows.append(dict(
                path=path, label=label, geometry=geom, geometry_id=GEOMETRIES.index(geom),
                tip=tip, prefix=prefix, print_id=pid,
                gauge=int(p.gauge), temp_c=int(p.temp_c),
                pressure_psi=int(p.pressure_psi), speed_mm_s=int(p.speed_mm_s),
                # Proxy for fabrication session: same sheet, needle gauge and temperature (18 prints each)
                session=f"{prefix}_G{int(p.gauge)}_T{int(p.temp_c)}",
                width=Image.open(full).size[0],
                md5=hashlib.md5(open(full, "rb").read()).hexdigest(),
            ))
    df = pd.DataFrame(rows)

    # Label conflicts: byte-identical files present in both Good and Bad folders
    lab = df.groupby("md5")["label"].nunique()
    df["label_conflict"] = df.md5.map(lab) > 1
    df["usable"] = ~df.label_conflict
    df.to_csv(a.out, index=False)

    print(f"{len(df)} files, {df.md5.nunique()} distinct images, {df.session.nunique()} session blocks")
    print("\nGood/bad by geometry:")
    t = pd.crosstab(df.geometry, df.label.map({1: "good", 0: "bad"}), margins=True)
    t["% good"] = (100 * t["good"] / t["All"]).round(1)
    print(t)
    print("\nGood/bad by geometry x tip:")
    print(pd.crosstab([df.geometry, df.tip], df.label.map({1: "good", 0: "bad"})))
    print("\nLabel-conflict files (identical image in Good and Bad):")
    print(df[df.label_conflict][["path", "prefix", "print_id"]].to_string(index=False))
    for prefix, (_, _, sheet) in PREFIX_MAP.items():
        missing = sorted(set(params[sheet].index) - set(df[df.prefix == prefix].print_id))
        if missing:
            print(f"Missing print IDs for {prefix}: {missing}")
    print("\nImage width by prefix:")
    print(df.groupby("prefix").width.value_counts().unstack(fill_value=0))
    print(f"\nUsable (conflicts removed): {df.usable.sum()}")


if __name__ == "__main__":
    main()
