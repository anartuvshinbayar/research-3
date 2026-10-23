"""Paths, data loading, and evaluation helpers shared by the experiment scripts."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from augment import marsify_set
from spectra import GRID, SUPERCAM_FWHM, gaussian_broaden, preprocess

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
CACHE = ROOT / "data/cache"
RES = ROOT / "results"

MARS_GROUPS = ["olivine", "pyroxene", "feldspar", "silica", "carbonate", "sulfate", "serpentine",
               "perchlorate", "phosphate"]
EVAL_EXCLUDE = {"serpentine"}  # identified on Mars only from the OH band, outside our window
BUMP_MAX = 560.0  # cm^-1, upper edge of the SuperCam fiber bump
COPIES = 8
SEEDS = [0, 1, 2]


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def load():
    meta = pd.read_csv(PROC / "rruff_meta.csv")
    lab = np.load(PROC / "rruff.npz", allow_pickle=True)
    raw = np.load(PROC / "rruff_raw.npz", allow_pickle=True)
    m = np.load(PROC / "mars.npz", allow_pickle=True)
    mars = pd.DataFrame({k: m[k] for k in ["product", "target", "sol", "primary", "groups", "conf", "split"]})
    return meta, lab["X"], list(raw["x"]), list(raw["y"]), mars, m["X"]


def lab_split(meta, seed=0):
    """Hold out one physical sample per species that has at least two."""
    rng = np.random.default_rng(seed)
    test_ids = set()
    for _, g in meta.groupby("name"):
        ids = g.rruff_id.unique()
        if len(ids) >= 2:
            test_ids.add(rng.choice(ids))
    is_test = meta.rruff_id.isin(test_ids).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]


def mars_split(mars, split):
    return mars[(mars.split == split) & ~mars.primary.isin(EVAL_EXCLUDE)]


def window(X, lo):
    """Zero everything below `lo` cm^-1 and renormalize."""
    if lo is None:
        return X
    Y = X.copy()
    Y[:, GRID < lo] = 0
    return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)


def broadened_library(raw_x, raw_y, idx, fwhm=SUPERCAM_FWHM):
    out = []
    for i in idx:
        x, y = gaussian_broaden(np.asarray(raw_x[i], float), np.asarray(raw_y[i], float), fwhm)
        s = preprocess(x, y)
        out.append(s if s is not None else np.zeros(len(GRID), np.float32))
    return np.stack(out)


def cached_marsify(tag, raw_x, raw_y, groups, idx, copies, seed, **kw):
    path = CACHE / f"aug_{tag}_s{seed}.npz"
    if path.exists():
        d = np.load(path)
        return d["X"].astype(np.float32), d["src"]
    t = time.time()
    X, src = marsify_set(raw_x, raw_y, groups, idx, copies, seed=seed, **kw)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X.astype(np.float16), src=src)
    log(f"augmented {tag} seed {seed}: {len(X)} spectra in {time.time() - t:.0f}s")
    return X, src


def species_to_group(P, species, species_group, allowed=None, rule="max"):
    """Map species probabilities to a mineral group.

    "max" takes the group of the most probable allowed species. "sum" adds probabilities
    within each group; it favors groups with many species and is kept only for comparison.
    """
    sp_groups = np.array([species_group[s] for s in species])
    if rule == "max":
        Q = P.copy()
        if allowed is not None:
            Q[:, ~np.isin(sp_groups, list(allowed))] = -1
        return sp_groups[Q.argmax(1)]
    groups = sorted(set(sp_groups))
    G = np.zeros((len(P), len(groups)))
    for k, g in enumerate(sp_groups):
        G[:, groups.index(g)] += P[:, k]
    if allowed is not None:
        G[:, [g not in allowed for g in groups]] = -1
    return np.array(groups)[G.argmax(1)]


def mars_scores(pred, df, n_boot=2000, seed=0):
    """Accuracy and balanced accuracy with a cluster bootstrap over targets."""
    hit = np.array([p in g.split("|") for p, g in zip(pred, df.groups)])
    classes = sorted(df.primary.unique())
    targets = df.target.unique()
    ti = {t: i for i, t in enumerate(targets)}
    ci = {c: i for i, c in enumerate(classes)}
    n = np.zeros((len(targets), len(classes)))
    h = np.zeros_like(n)
    for t, c, k in zip(df.target, df.primary, hit):
        n[ti[t], ci[c]] += 1
        h[ti[t], ci[c]] += k

    def metrics(nn, hh):
        tot = nn.sum(0)
        return hh.sum() / nn.sum(), (hh.sum(0)[tot > 0] / tot[tot > 0]).mean()

    acc, bal = metrics(n, h)
    rng = np.random.default_rng(seed)
    boots = [metrics(n[p], h[p]) for p in (rng.integers(0, len(targets), len(targets)) for _ in range(n_boot))]
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    out = dict(acc=acc, acc_lo=lo[0], acc_hi=hi[0], bal_acc=bal, bal_lo=lo[1], bal_hi=hi[1])
    for c in classes:
        out[f"recall_{c}"] = hit[(df.primary == c).to_numpy()].mean()
    return out
