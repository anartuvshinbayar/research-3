"""Main experiment: lab-trained Raman classifiers evaluated on lab and on real SuperCam spectra.

Writes everything to results/:
  lab_results.csv     accuracy on held-out lab samples, clean and progressively "marsified"
  mars_results.csv    accuracy on the SuperCam test split, with 95% bootstrap CIs (by target)
  mars_predictions.csv per-spectrum predictions for every method
  artifact_template.npy  instrument background template learned from unlabeled dev spectra
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from augment import marsify_set

from spectra import GRID, MARS_PARTS, gaussian_broaden, preprocess, SUPERCAM_FWHM

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
RES = ROOT / "results"
CACHE = ROOT / "data/cache"

MARS_GROUPS = ["olivine", "pyroxene", "feldspar", "silica", "carbonate", "sulfate", "serpentine",
               "perchlorate", "phosphate"]
# Serpentine was identified on Mars only from its OH band at ~3690 cm^-1, outside the
# fingerprint window (and outside most RRUFF spectra), so it is excluded from scoring.
EVAL_EXCLUDE = {"serpentine"}
ARTIFACT_MAX = 560.0  # cm^-1; upper edge of the broad instrumental feature
COPIES = 8
SEEDS = [0, 1, 2]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- data
def load():
    meta = pd.read_csv(PROC / "rruff_meta.csv")
    R = np.load(PROC / "rruff.npz", allow_pickle=True)
    raw = np.load(PROC / "rruff_raw.npz", allow_pickle=True)
    M = np.load(PROC / "mars.npz", allow_pickle=True)
    mars = pd.DataFrame({k: M[k] for k in ["product", "target", "sol", "primary", "groups", "conf", "split"]})
    return meta, R["X"], list(raw["x"]), list(raw["y"]), mars, M["X"]


def lab_split(meta, seed=0):
    """Hold out one whole sample (RRUFF ID) for every species that has >= 2 samples."""
    rng = np.random.default_rng(seed)
    test_ids = set()
    for name, g in meta.groupby("name"):
        ids = g.rruff_id.unique()
        if len(ids) >= 2:
            test_ids.add(rng.choice(ids))
    is_test = meta.rruff_id.isin(test_ids).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]


def artifact_template(Xmars, mars):
    """Median of unlabeled dev-split Mars spectra below ARTIFACT_MAX, tapered to zero."""
    T = np.median(Xmars[(mars.split == "dev").to_numpy()], axis=0)
    taper = np.clip((ARTIFACT_MAX + 20 - GRID) / 40.0, 0, 1)
    T = T * taper
    return (T / np.linalg.norm(T)).astype(np.float32)


def add_artifact(X, T, rng, max_amp=2.0):
    a = rng.uniform(0, max_amp, size=(len(X), 1)).astype(np.float32)
    Y = X + a * T[None, :]
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def window(X, lo):
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


# ---------------------------------------------------------------- scoring
def group_from_species_proba(P, species, species_group, allowed_groups=None, rule="max"):
    """Mineral group from species probabilities.

    rule="max": group of the most probable (allowed) species -- the same decision rule as
    library search. rule="sum": add up probabilities within each group; under domain shift this
    favors groups with many species (e.g. 141 sulfates vs 7 olivines), so it is reported only
    for comparison.
    """
    sp_groups = np.array([species_group[s] for s in species])
    if rule == "max":
        Q = P.copy()
        if allowed_groups is not None:
            Q[:, ~np.isin(sp_groups, list(allowed_groups))] = -1
        return sp_groups[Q.argmax(1)]
    groups = sorted(set(sp_groups))
    G = np.zeros((len(P), len(groups)))
    for k, g in enumerate(sp_groups):
        G[:, groups.index(g)] += P[:, k]
    if allowed_groups is not None:
        G[:, [g not in allowed_groups for g in groups]] = -1
    return np.array(groups)[G.argmax(1)]


def mars_scores(pred, mars_eval, n_boot=2000, seed=0):
    hit = np.array([p in g.split("|") for p, g in zip(pred, mars_eval.groups)])
    df = mars_eval.assign(hit=hit)
    classes = sorted(df.primary.unique())
    targets = df.target.unique()
    # Per-target counts: n[t, c] spectra and h[t, c] hits for class c.
    n = np.zeros((len(targets), len(classes)))
    h = np.zeros_like(n)
    ti = {t: i for i, t in enumerate(targets)}
    ci = {c: i for i, c in enumerate(classes)}
    for t, c, k in zip(df.target, df.primary, df.hit):
        n[ti[t], ci[c]] += 1
        h[ti[t], ci[c]] += k

    def metrics(nn, hh):
        acc = hh.sum() / nn.sum()
        tot = nn.sum(0)
        rec = hh.sum(0)[tot > 0] / tot[tot > 0]
        return acc, rec.mean()

    acc, bal = metrics(n, h)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(targets), len(targets))
        boots.append(metrics(n[pick], h[pick]))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    out = dict(acc=acc, acc_lo=lo[0], acc_hi=hi[0], bal_acc=bal, bal_lo=lo[1], bal_hi=hi[1])
    for c in classes:
        out[f"recall_{c}"] = df[df.primary == c].hit.mean()
    return out


# ---------------------------------------------------------------- main
def main():
    from models import CosineLibrary, cnn_proba, train_cnn  # torch stays out of worker processes
    RES.mkdir(exist_ok=True)
    meta, X, raw_x, raw_y, mars, Xmars = load()
    groups = meta.group.to_numpy()
    tr, te = lab_split(meta)
    log(f"lab train {len(tr)} spectra / test {len(te)} spectra ({meta.iloc[te].name.nunique()} species)")

    species = sorted(meta.name.unique())
    sp_index = {s: i for i, s in enumerate(species)}
    species_group = dict(zip(meta.name, meta.group))
    y_sp = meta.name.map(sp_index).to_numpy()

    T = artifact_template(Xmars, mars)
    np.save(RES / "artifact_template.npy", T)

    ev = mars[(mars.split == "test") & ~mars.primary.isin(EVAL_EXCLUDE)].copy()
    Xev = Xmars[ev.index.to_numpy()]
    log(f"Mars test spectra scored: {len(ev)} from {ev.target.nunique()} targets")
    allowed = set(MARS_GROUPS)

    # Lab test sets at increasing degradation strength (fixed seed, no mixing partner bias).
    lab_sets = {0.0: X[te]}
    for s in [0.5, 1.0, 1.5, 2.0]:
        Xs, _ = cached_marsify(f"labtest_{s}", raw_x, raw_y, groups, te, 1, seed=99, strength=s)
        lab_sets[s] = Xs

    lab_rows, mars_rows, preds = [], [], {}

    def record(method, setting, lab_pred_fn, mars_group_pred):
        for s, Xs in lab_sets.items():
            sp_pred, grp_pred = lab_pred_fn(Xs, s)
            lab_rows.append(dict(method=method, setting=setting, strength=s,
                                 species_acc=np.mean(sp_pred == meta.name.to_numpy()[te]),
                                 group_acc=np.mean(grp_pred == groups[te])))
        r = mars_scores(mars_group_pred, ev)
        mars_rows.append(dict(method=method, setting=setting, **r))
        preds[f"{method}|{setting}"] = mars_group_pred
        log(f"{method:28s} {setting:6s} Mars acc {r['acc']:.2f} bal {r['bal_acc']:.2f}")

    # --- Cosine library search variants
    names = meta.name.to_numpy()
    lib_variants = {
        "cosine": (X[tr], None),
        "cosine-window": (window(X[tr], ARTIFACT_MAX), ARTIFACT_MAX),
    }
    Xb = broadened_library(raw_x, raw_y, tr)
    lib_variants["cosine-broadened"] = (Xb, None)
    lib_variants["cosine-broadened-window"] = (window(Xb, ARTIFACT_MAX), ARTIFACT_MAX)
    for method, (Xlib, lo) in lib_variants.items():
        lib_sp = CosineLibrary().fit(Xlib, names[tr])
        for setting, allow in [("open", None), ("mars", allowed)]:
            allow_sp = None if allow is None else {s for s in species if species_group[s] in allow}

            def lab_fn(Xs, s, lib_sp=lib_sp, lo=lo, allow_sp=allow_sp):
                sp = lib_sp.predict(window(Xs, lo), allow_sp)
                return sp, np.array([species_group[p] for p in sp])

            sp_m = lib_sp.predict(window(Xev, lo), allow_sp)
            record(method, setting, lab_fn, np.array([species_group[p] for p in sp_m]))

    # --- CNNs
    def train_variant(tag, Xtr, ytr, seed):
        path = CACHE / f"cnn_{tag}_s{seed}.pt"
        import torch
        from models import SpectrumCNN, device
        if path.exists():
            m = SpectrumCNN(Xtr.shape[1], len(species))
            m.load_state_dict(torch.load(path, map_location="cpu"))
            return m.to(device()).eval()
        t = time.time()
        m = train_cnn(Xtr, ytr, len(species), epochs=20, seed=seed, log=lambda *a: None)
        torch.save(m.state_dict(), path)
        log(f"trained cnn {tag} seed {seed} in {time.time() - t:.0f}s")
        return m

    def train_config(method, cfg, seeds):
        models = []
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            if cfg.get("aug"):
                parts = cfg.get("parts")
                tag = "train" if parts is None else "train_" + "-".join(parts)
                Xa, src = cached_marsify(tag, raw_x, raw_y, groups, tr, COPIES, seed=seed, parts=parts)
                Xtr = np.concatenate([X[tr], Xa])
                ytr = np.concatenate([y_sp[tr], y_sp[src]])
            else:
                Xtr = np.repeat(X[tr], 4, axis=0)
                ytr = np.repeat(y_sp[tr], 4)
            if cfg.get("artifact"):
                half = rng.random(len(Xtr)) < 0.5
                Xtr = Xtr.copy()
                Xtr[half] = add_artifact(Xtr[half], T, rng)
            Xtr = window(Xtr, cfg.get("window"))
            models.append(train_variant(method, Xtr.astype(np.float32), ytr, seed))
        return models

    def proba(models, Xq, cfg):
        Xq = window(Xq, cfg.get("window"))
        return np.mean([cnn_proba(m, Xq) for m in models], axis=0)

    def evaluate_cnn(method, cfg, models, seed_list):
        def lab_fn(Xs, s, allow=None):
            P = proba(models, Xs, cfg)
            return np.array(species)[P.argmax(1)], group_from_species_proba(P, species, species_group, allow)

        P = proba(models, Xev, cfg)
        for setting, allow in [("open", None), ("mars", allowed)]:
            g = group_from_species_proba(P, species, species_group, allow)
            record(method, setting, lambda Xs, s, a=allow: lab_fn(Xs, s, allow=a), g)
            g_sum = group_from_species_proba(P, species, species_group, allow, rule="sum")
            sum_rows.append(dict(method=method, setting=setting, **mars_scores(g_sum, ev)))
            for seed, m in zip(seed_list, models):
                gs = group_from_species_proba(proba([m], Xev, cfg), species, species_group, allow)
                r = mars_scores(gs, ev, n_boot=1)
                seed_results.append(dict(method=method, setting=setting, seed=seed,
                                         acc=r["acc"], bal_acc=r["bal_acc"]))

    W = ARTIFACT_MAX
    cnn_configs = {
        "cnn-clean": dict(aug=False),
        "cnn-clean-window": dict(aug=False, window=W),
        "cnn-marsified": dict(aug=True),
        "cnn-marsified-window": dict(aug=True, window=W),
        "cnn-marsified+artifact": dict(aug=True, artifact=True),
    }
    # Model selection uses ONLY the development split (never the test split).
    dev = mars[(mars.split == "dev") & ~mars.primary.isin(EVAL_EXCLUDE)]
    Xdev = Xmars[dev.index.to_numpy()]
    seed_results, sum_rows, dev_rows, trained = [], [], [], {}
    for method, cfg in cnn_configs.items():
        trained[method] = train_config(method, cfg, SEEDS)
        evaluate_cnn(method, cfg, trained[method], SEEDS)
        g_dev = group_from_species_proba(proba(trained[method], Xdev, cfg), species, species_group, allowed)
        dev_rows.append(dict(method=method, **mars_scores(g_dev, dev, n_boot=1)))
    dev_df = pd.DataFrame(dev_rows)
    best = dev_df.sort_values("bal_acc", ascending=False).method.iloc[0]
    log(f"selected on dev split: {best} (dev bal acc {dev_df.bal_acc.max():.2f})")

    # Ablations of the dev-selected configuration (2 seeds each, compared with a 2-seed full model).
    ab_seeds = SEEDS[:2]
    base = cnn_configs[best]
    ablations = {"ablate-full": (base, best)}
    if base.get("aug"):
        for p in MARS_PARTS:
            ablations[f"ablate-no-{p}"] = (dict(base, parts=[q for q in MARS_PARTS if q != p]), None)
        ablations["ablate-no-simulation"] = (dict(base, aug=False), None)
    if base.get("artifact"):
        ablations["ablate-no-template"] = (dict(base, artifact=False), None)
    for method, (cfg, reuse) in ablations.items():
        models = trained[reuse][:2] if reuse else train_config(method, cfg, ab_seeds)
        evaluate_cnn(method, cfg, models, ab_seeds)

    # --- Trivial baseline
    maj = np.array(["olivine"] * len(ev))
    mars_rows.append(dict(method="always-olivine", setting="-", **mars_scores(maj, ev)))

    pd.DataFrame(lab_rows).to_csv(RES / "lab_results.csv", index=False)
    pd.DataFrame(mars_rows).to_csv(RES / "mars_results.csv", index=False)
    pd.DataFrame(seed_results).to_csv(RES / "mars_seed_results.csv", index=False)
    pd.DataFrame(sum_rows).to_csv(RES / "mars_results_sumrule.csv", index=False)
    dev_df.assign(selected=dev_df.method == best).to_csv(RES / "dev_selection.csv", index=False)
    out = ev[["product", "target", "sol", "primary", "groups", "conf"]].copy()
    for k, v in preds.items():
        out[k] = v
    out.to_csv(RES / "mars_predictions.csv", index=False)
    json.dump(dict(n_lab_train=int(len(tr)), n_lab_test=int(len(te)),
                   n_species=len(species), n_lab_test_species=int(meta.iloc[te].name.nunique()),
                   n_mars_eval=int(len(ev)), n_mars_targets=int(ev.target.nunique()),
                   selected_model=best, n_dev_eval=int(len(dev)),
                   mars_eval_counts=ev.primary.value_counts().to_dict()),
              open(RES / "summary.json", "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
