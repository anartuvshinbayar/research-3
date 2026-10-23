"""Side analyses quoted in the paper. Writes results/extra.json and results/dev_library.csv."""
import json

import numpy as np
import pandas as pd

from common import (BUMP_MAX, MARS_GROUPS, RES, broadened_library, lab_split, load, mars_scores, mars_split,
                    window)
from library import CosineLibrary
from spectra import GRID

AMORPHOUS = ("glass", "opal", "amorphous", "obsidian", "tektite", "moldavite", "allophane", "hisingerite")


def is_amorphous(name):
    return any(a in name.lower() for a in AMORPHOUS)


def main():
    meta, X, raw_x, raw_y, mars, Xmars = load()
    tr, te = lab_split(meta)
    names = meta.name.to_numpy()
    sg = dict(zip(meta.name, meta.group))
    allow = {s for s in names if sg[s] in MARS_GROUPS}
    ev, dev = mars_split(mars, "test"), mars_split(mars, "dev")
    Xev, Xdev = Xmars[ev.index.to_numpy()], Xmars[dev.index.to_numpy()]
    out = {}

    # What does open-library search match Mars spectra to?
    lib = CosineLibrary().fit(X[tr], names[tr])
    S = lib.scores(Xev)
    top = lib.y[S.argmax(1)]
    out["open_top1_frac_amorphous"] = float(np.mean([is_amorphous(t) for t in top]))
    out["open_top1_frac_other_group"] = float(np.mean([sg[t] == "other" for t in top]))
    out["open_top1_species"] = pd.Series(top).value_counts().head(10).to_dict()
    out["median_best_cosine_mars"] = float(np.median(S.max(1)))
    out["median_best_cosine_lab"] = float(np.median(lib.scores(X[te]).max(1)))
    low = GRID < BUMP_MAX
    out["median_energy_below_560_mars"] = float(np.median((Xev[:, low] ** 2).sum(1)))
    out["median_energy_below_560_lab"] = float(np.median((X[te][:, low] ** 2).sum(1)))
    libw = CosineLibrary().fit(window(X[tr], BUMP_MAX), names[tr])
    topw = libw.y[libw.scores(window(Xev, BUMP_MAX)).argmax(1)]
    out["open_window_top1_frac_amorphous"] = float(np.mean([is_amorphous(t) for t in topw]))

    # Dev-split scores of the library variants (CNN dev scores are in dev_selection.csv).
    Xb = broadened_library(raw_x, raw_y, tr)
    variants = {"cosine": (X[tr], None), "cosine-window": (window(X[tr], BUMP_MAX), BUMP_MAX),
                "cosine-broadened": (Xb, None), "cosine-broadened-window": (window(Xb, BUMP_MAX), BUMP_MAX)}
    rows = []
    for method, (Xlib, lo) in variants.items():
        sp = CosineLibrary().fit(Xlib, names[tr]).predict(window(Xdev, lo), allow)
        r = mars_scores(np.array([sg[s] for s in sp]), dev, n_boot=1)
        rows.append(dict(method=method, acc=r["acc"], bal_acc=r["bal_acc"]))
    pd.DataFrame(rows).to_csv(RES / "dev_library.csv", index=False)
    out["dev_library"] = {r["method"]: round(r["bal_acc"], 3) for r in rows}

    # Accuracy by team confidence, and the most common confusions.
    p = pd.read_csv(RES / "mars_predictions.csv")
    for col in ["cosine-window|mars", "cnn-clean|mars", "cosine|mars"]:
        hit = np.array([a in g.split("|") for a, g in zip(p[col], p.groups)])
        out[f"acc_by_conf::{col}"] = p.assign(hit=hit).groupby("conf").hit.mean().round(3).to_dict()
        wrong = p[~hit]
        out[f"confusions::{col}"] = {g: wrong[wrong.primary == g][col].value_counts().to_dict()
                                     for g in sorted(wrong.primary.unique())}
    out["n_by_conf"] = p.conf.value_counts().to_dict()

    with open(RES / "extra.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
