"""Side analyses quoted in the text:
  - what open-library search thinks Mars rocks are (glass/amorphous matches)
  - dev-split scores of the library-search variants (CNN dev scores are in dev_selection.csv)
  - accuracy by the team's confidence level
  - what the best methods confuse each group with
Writes results/extra.json and results/dev_library.csv."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import (ARTIFACT_MAX, EVAL_EXCLUDE, MARS_GROUPS, broadened_library, lab_split, load,
                             mars_scores, window)
from spectra import GRID

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

AMORPHOUS = ("glass", "opal", "amorphous", "obsidian", "tektite", "moldavite", "allophane", "hisingerite")


class CosineLibrary:
    """Same as models.CosineLibrary, duplicated so this script does not import torch."""

    def fit(self, X, y):
        self.X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        self.y = np.asarray(y)
        return self

    def scores(self, Q, allowed=None):
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        S = Q @ self.X.T
        if allowed is not None:
            S[:, ~np.isin(self.y, list(allowed))] = -np.inf
        return S


def main():
    meta, X, raw_x, raw_y, mars, Xmars = load()
    tr, te = lab_split(meta)
    names = meta.name.to_numpy()
    sg = dict(zip(meta.name, meta.group))
    allow = {s for s in names if sg[s] in MARS_GROUPS}
    ev = mars[(mars.split == "test") & ~mars.primary.isin(EVAL_EXCLUDE)]
    dev = mars[(mars.split == "dev") & ~mars.primary.isin(EVAL_EXCLUDE)]
    Xev, Xdev = Xmars[ev.index.to_numpy()], Xmars[dev.index.to_numpy()]
    out = {}

    # Open-library search: what are the top matches?
    lib = CosineLibrary().fit(X[tr], names[tr])
    S = lib.scores(Xev)
    top = lib.y[S.argmax(1)]
    is_amorph = np.array([any(a in t.lower() for a in AMORPHOUS) for t in top])
    out["open_top1_frac_amorphous"] = float(is_amorph.mean())
    out["open_top1_frac_other_group"] = float(np.mean([sg[t] == "other" for t in top]))
    out["open_top1_species"] = pd.Series(top).value_counts().head(10).to_dict()
    out["median_best_cosine_mars"] = float(np.median(S.max(1)))
    out["median_best_cosine_lab"] = float(np.median(lib.scores(X[te]).max(1)))
    low = GRID < ARTIFACT_MAX
    out["median_energy_below_560_mars"] = float(np.median((Xev[:, low] ** 2).sum(1)))
    out["median_energy_below_560_lab"] = float(np.median((X[te][:, low] ** 2).sum(1)))
    # The same with the bump region skipped.
    libw = CosineLibrary().fit(window(X[tr], ARTIFACT_MAX), names[tr])
    topw = libw.y[libw.scores(window(Xev, ARTIFACT_MAX)).argmax(1)]
    out["open_window_top1_frac_amorphous"] = float(np.mean([any(a in t.lower() for a in AMORPHOUS) for t in topw]))

    # Dev-split scores for library variants (Mars-candidate setting).
    Xb = broadened_library(raw_x, raw_y, tr)
    rows = []
    for method, (Xlib, lo) in {"cosine": (X[tr], None), "cosine-window": (window(X[tr], ARTIFACT_MAX), ARTIFACT_MAX),
                               "cosine-broadened": (Xb, None),
                               "cosine-broadened-window": (window(Xb, ARTIFACT_MAX), ARTIFACT_MAX)}.items():
        L = CosineLibrary().fit(Xlib, names[tr])
        sp = L.y[L.scores(window(Xdev, lo), allow).argmax(1)]
        r = mars_scores(np.array([sg[s] for s in sp]), dev, n_boot=1)
        rows.append(dict(method=method, acc=r["acc"], bal_acc=r["bal_acc"]))
    pd.DataFrame(rows).to_csv(RES / "dev_library.csv", index=False)
    out["dev_library"] = {r["method"]: round(r["bal_acc"], 3) for r in rows}

    # Accuracy by team confidence, and confusions, for the two headline methods.
    p = pd.read_csv(RES / "mars_predictions.csv")
    for col in ["cosine-window|mars", "cnn-clean|mars", "cosine|mars"]:
        hit = np.array([a in g.split("|") for a, g in zip(p[col], p.groups)])
        out[f"acc_by_conf::{col}"] = p.assign(hit=hit).groupby("conf").hit.mean().round(3).to_dict()
        wrong = p[~hit]
        out[f"confusions::{col}"] = {g: wrong[wrong.primary == g][col].value_counts().to_dict()
                                     for g in sorted(wrong.primary.unique())}
    out["n_by_conf"] = p.conf.value_counts().to_dict()
    json.dump(out, open(RES / "extra.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
