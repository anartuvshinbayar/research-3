"""Match SuperCam spectra to the team's mineral labels and save data/processed/mars.npz.

primary is the group of the first reported feature; groups lists every reported group.
Targets (rocks) are split about 30/70 into dev and test.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from spectra import GRID, preprocess

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data/supercam_derived/supercam_raman.csv"
CAL = ROOT / "data/supercam_cal"
OUT = ROOT / "data/processed/mars.npz"

KEYWORDS = [
    ("olivine", "olivine"),
    ("carbonate", "carbonate"),
    ("anhydrite", "sulfate"),
    ("bassanite", "sulfate"),
    ("gypsum", "sulfate"),
    ("sulfate", "sulfate"),
    ("pyroxene", "pyroxene"),
    ("serpentine", "serpentine"),
    ("plagioclase", "feldspar"),
    ("feldspar", "feldspar"),
    ("quartz", "silica"),
    ("perchlorate", "perchlorate"),
    ("apatite", "phosphate"),
]


def to_group(text):
    t = str(text).lower()
    for key, group in KEYWORDS:
        if key in t:
            return group
    return None


def norm_conf(text):
    t = str(text).lower()
    if "high" in t and ("medium" in t or "med" in t):
        return "medium"
    for level in ("high", "medium", "low"):
        if level in t:
            return level
    return "unknown"


def split_for(target):
    """Deterministic ~30/70 dev/test split at the target (rock) level."""
    h = int(hashlib.md5(target.encode()).hexdigest(), 16)
    return "dev" if h % 10 < 3 else "test"


def load_spectrum(path):
    with fits.open(path) as f:
        x = f["WAVELENGTH"].data["Wavelength"].astype(float)
        y = f["STATISTICS"].data["Mean"].astype(float)
    return x, y


def main():
    table = pd.read_csv(TABLE)
    manifest = pd.read_csv(CAL / "manifest.csv")
    fits_for = dict(zip(manifest.source_product, manifest.fits_file))

    rows, X = [], []
    for _, r in table.iterrows():
        product = r["Source Product"].strip()
        name = fits_for.get(product, "")
        if not str(name).endswith(".fits"):
            continue
        groups = []
        for i in range(1, 5):
            g = to_group(r.get(f"Feature{i}_PotentialMinerals"))
            if g and g not in groups:
                groups.append(g)
        if not groups:
            continue
        x, y = load_spectrum(CAL / name)
        spec = preprocess(x, y)
        if spec is None:
            continue
        target = str(r["Target"]).strip()
        X.append(spec)
        rows.append(dict(
            product=product,
            target=target,
            sol=int(product.split("_")[1]),
            primary=groups[0],
            groups="|".join(groups),
            conf=norm_conf(r["Feature1_Confidence"]),
            split=split_for(target),
        ))

    meta = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=np.stack(X), **{c: meta[c].to_numpy() for c in meta.columns})
    print(f"saved {len(meta)} Mars spectra -> {OUT}")
    print(pd.crosstab(meta.primary, meta.split, margins=True))
    print(meta.conf.value_counts())


if __name__ == "__main__":
    main()
