"""Load RRUFF laboratory Raman spectra (unoriented, processed) and tag mineral groups.

Output:
  data/processed/rruff_raw.npz  native-resolution (x, y) arrays, for augmentation
  data/processed/rruff.npz      preprocessed spectra on the common grid
Both share the metadata columns: name, rruff_id, laser, chem, group.
"""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from spectra import preprocess

ROOT = Path(__file__).resolve().parent.parent
ZIPS = [ROOT / "data/rruff_zip/excellent_unoriented.zip", ROOT / "data/rruff_zip/fair_unoriented.zip"]
OUT = ROOT / "data/processed"

NAME_GROUPS = {
    "olivine": ["forsterite", "fayalite", "tephroite", "monticellite", "kirschsteinite",
                "liebenbergite", "glaucochroite", "laihunite"],
    "pyroxene": ["enstatite", "ferrosilite", "clinoenstatite", "clinoferrosilite", "pigeonite",
                 "diopside", "hedenbergite", "augite", "aegirine", "aegirine-augite", "jadeite",
                 "omphacite", "spodumene", "johannsenite", "kosmochlor", "esseneite",
                 "petedunnite", "kanoite", "donpeacorite", "natalyite", "jervisite", "davisite",
                 "grossmanite", "kushiroite", "burnettite"],
    "feldspar": ["albite", "anorthite", "labradorite", "andesine", "oligoclase", "bytownite",
                 "orthoclase", "microcline", "sanidine", "anorthoclase", "celsian", "hyalophane"],
    "serpentine": ["antigorite", "lizardite", "chrysotile", "clinochrysotile", "orthochrysotile",
                   "greenalite", "cronstedtite", "nepouite", "pecoraite", "caryopilite",
                   "berthierine", "amesite"],
    "silica": ["quartz", "cristobalite", "tridymite", "coesite", "stishovite", "moganite"],
}
NAME_TO_GROUP = {n: g for g, names in NAME_GROUPS.items() for n in names}


# Elements that form their own oxyanion groups (arsenate, phosphate, vanadate, borate, ...).
# A carbonate or sulfate may contain none of these besides its own C or S; otherwise mixed-anion
# minerals such as beudantite (arsenate-sulfate) would be filed as sulfates.
ANION_FORMERS = {"C", "S", "As", "P", "V", "Cr", "Mo", "W", "Se", "Te", "B", "Si", "N", "Sb", "Ge", "U", "I"}


def elements(chem):
    chem = re.sub(r"\^[^^]*\^", "", chem).replace("NH_4", "")  # drop charges; ammonium is a cation
    return set(re.findall(r"[A-Z][a-z]?", chem))


def group_for(name, chem):
    n = name.lower().strip()
    if n in NAME_TO_GROUP:
        return NAME_TO_GROUP[n]
    c = chem.replace(" ", "")
    others = lambda own: (elements(c) & ANION_FORMERS) - own
    if "ClO_4" in c and not others(set()):
        return "perchlorate"
    if "CO_3" in c and not others({"C"}):
        return "carbonate"
    if "SO_4" in c and not others({"S"}):
        return "sulfate"
    if "PO_4" in c and n.endswith("apatite"):
        return "phosphate"
    return "other"


def parse(text):
    meta, xs, ys = {}, [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("##"):
            if "=" in line:
                k, v = line[2:].split("=", 1)
                meta.setdefault(k.strip().upper(), v.strip())
            continue
        parts = re.split(r"[,\s]+", line)
        if len(parts) >= 2:
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                pass
    return meta, np.array(xs), np.array(ys)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, raw_x, raw_y, X = [], [], [], []
    for zpath in ZIPS:
        if not zpath.exists():
            print("skip missing", zpath.name)
            continue
        quality = zpath.stem.split("_")[0]
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                fname = Path(info.filename).name
                if not fname.endswith(".txt") or "Processed" not in fname:
                    continue
                parts = fname.split("__")
                laser = parts[3] if len(parts) > 3 else "?"
                meta, x, y = parse(z.read(info).decode("latin-1"))
                if len(x) < 50:
                    continue
                name = meta.get("NAMES", parts[0]).split(",")[0].strip()
                chem = meta.get("IDEAL CHEMISTRY", "")
                spec = preprocess(x, y)
                if spec is None:
                    continue
                rows.append(dict(name=name, rruff_id=meta.get("RRUFFID", parts[1]),
                                 laser=laser, chem=chem, group=group_for(name, chem),
                                 quality=quality, fname=fname))
                raw_x.append(x.astype(np.float32))
                raw_y.append(y.astype(np.float32))
                X.append(spec)

    meta = pd.DataFrame(rows)
    meta.to_csv(OUT / "rruff_meta.csv", index=False)
    cols = {c: meta[c].to_numpy() for c in meta.columns}
    np.savez_compressed(OUT / "rruff.npz", X=np.stack(X), **cols)
    np.savez_compressed(OUT / "rruff_raw.npz",
                        x=np.array(raw_x, dtype=object), y=np.array(raw_y, dtype=object), **cols)
    print(f"{len(meta)} spectra, {meta.name.nunique()} species, {meta.rruff_id.nunique()} samples")
    print(meta.laser.value_counts().to_string())
    print(meta.drop_duplicates("name").group.value_counts().to_string())


if __name__ == "__main__":
    main()
