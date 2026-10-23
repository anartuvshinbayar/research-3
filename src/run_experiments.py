"""Train all models and evaluate them on held-out lab spectra and on the SuperCam test split."""
import json
import time

import numpy as np
import pandas as pd

from common import (BUMP_MAX, CACHE, COPIES, MARS_GROUPS, RES, SEEDS, broadened_library, cached_marsify,
                    lab_split, load, log, mars_scores, mars_split, species_to_group, window)
from library import CosineLibrary
from spectra import GRID, MARS_PARTS

CNN_CONFIGS = {
    "cnn-clean": dict(aug=False),
    "cnn-clean-window": dict(aug=False, window=BUMP_MAX),
    "cnn-marsified": dict(aug=True),
    "cnn-marsified-window": dict(aug=True, window=BUMP_MAX),
    "cnn-marsified+artifact": dict(aug=True, artifact=True),
}


def bump_template(Xmars, mars):
    """Median of the unlabeled dev-split spectra below BUMP_MAX, tapered to zero."""
    T = np.median(Xmars[(mars.split == "dev").to_numpy()], axis=0)
    T = T * np.clip((BUMP_MAX + 20 - GRID) / 40.0, 0, 1)
    return (T / np.linalg.norm(T)).astype(np.float32)


def add_bump(X, T, rng, max_amp=2.0):
    a = rng.uniform(0, max_amp, size=(len(X), 1)).astype(np.float32)
    Y = X + a * T[None, :]
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def main():
    import torch
    from models import SpectrumCNN, cnn_proba, device, train_cnn

    RES.mkdir(exist_ok=True)
    meta, X, raw_x, raw_y, mars, Xmars = load()
    groups = meta.group.to_numpy()
    names = meta.name.to_numpy()
    tr, te = lab_split(meta)
    log(f"lab train {len(tr)} spectra / test {len(te)} spectra ({meta.iloc[te].name.nunique()} species)")

    species = sorted(meta.name.unique())
    species_group = dict(zip(meta.name, meta.group))
    y_sp = meta.name.map({s: i for i, s in enumerate(species)}).to_numpy()
    allowed = set(MARS_GROUPS)

    T = bump_template(Xmars, mars)
    np.save(RES / "artifact_template.npy", T)

    ev = mars_split(mars, "test").copy()
    Xev = Xmars[ev.index.to_numpy()]
    dev = mars_split(mars, "dev")
    Xdev = Xmars[dev.index.to_numpy()]
    log(f"Mars test spectra scored: {len(ev)} from {ev.target.nunique()} targets")

    lab_sets = {0.0: X[te]}
    for s in [0.5, 1.0, 1.5, 2.0]:
        lab_sets[s], _ = cached_marsify(f"labtest_{s}", raw_x, raw_y, groups, te, 1, seed=99, strength=s)

    lab_rows, mars_rows, sum_rows, seed_rows, dev_rows, preds = [], [], [], [], [], {}

    def record(method, setting, lab_fn, mars_pred):
        for s, Xs in lab_sets.items():
            sp, grp = lab_fn(Xs)
            lab_rows.append(dict(method=method, setting=setting, strength=s,
                                 species_acc=np.mean(sp == names[te]), group_acc=np.mean(grp == groups[te])))
        r = mars_scores(mars_pred, ev)
        mars_rows.append(dict(method=method, setting=setting, **r))
        preds[f"{method}|{setting}"] = mars_pred
        log(f"{method:28s} {setting:6s} Mars acc {r['acc']:.2f} bal {r['bal_acc']:.2f}")

    # Library search
    Xb = broadened_library(raw_x, raw_y, tr)
    libraries = {
        "cosine": (X[tr], None),
        "cosine-window": (window(X[tr], BUMP_MAX), BUMP_MAX),
        "cosine-broadened": (Xb, None),
        "cosine-broadened-window": (window(Xb, BUMP_MAX), BUMP_MAX),
    }
    for method, (Xlib, lo) in libraries.items():
        lib = CosineLibrary().fit(Xlib, names[tr])
        for setting, allow in [("open", None), ("mars", allowed)]:
            allow_sp = None if allow is None else {s for s in species if species_group[s] in allow}

            def lab_fn(Xs, lib=lib, lo=lo, allow_sp=allow_sp):
                sp = lib.predict(window(Xs, lo), allow_sp)
                return sp, np.array([species_group[p] for p in sp])

            sp = lib.predict(window(Xev, lo), allow_sp)
            record(method, setting, lab_fn, np.array([species_group[p] for p in sp]))

    # CNNs
    def train_one(tag, Xtr, ytr, seed):
        path = CACHE / f"cnn_{tag}_s{seed}.pt"
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
                Xtr, ytr = np.concatenate([X[tr], Xa]), np.concatenate([y_sp[tr], y_sp[src]])
            else:
                Xtr, ytr = np.repeat(X[tr], 4, axis=0), np.repeat(y_sp[tr], 4)
            if cfg.get("artifact"):
                half = rng.random(len(Xtr)) < 0.5
                Xtr = Xtr.copy()
                Xtr[half] = add_bump(Xtr[half], T, rng)
            Xtr = window(Xtr, cfg.get("window"))
            models.append(train_one(method, Xtr.astype(np.float32), ytr, seed))
        return models

    def proba(models, Xq, cfg):
        Xq = window(Xq, cfg.get("window"))
        return np.mean([cnn_proba(m, Xq) for m in models], axis=0)

    def evaluate(method, cfg, models, seeds):
        P = proba(models, Xev, cfg)
        for setting, allow in [("open", None), ("mars", allowed)]:
            def lab_fn(Xs, allow=allow):
                Pl = proba(models, Xs, cfg)
                return np.array(species)[Pl.argmax(1)], species_to_group(Pl, species, species_group, allow)

            record(method, setting, lab_fn, species_to_group(P, species, species_group, allow))
            g_sum = species_to_group(P, species, species_group, allow, rule="sum")
            sum_rows.append(dict(method=method, setting=setting, **mars_scores(g_sum, ev)))
            for seed, m in zip(seeds, models):
                r = mars_scores(species_to_group(proba([m], Xev, cfg), species, species_group, allow), ev, n_boot=1)
                seed_rows.append(dict(method=method, setting=setting, seed=seed, acc=r["acc"], bal_acc=r["bal_acc"]))

    trained = {}
    for method, cfg in CNN_CONFIGS.items():
        trained[method] = train_config(method, cfg, SEEDS)
        evaluate(method, cfg, trained[method], SEEDS)
        g_dev = species_to_group(proba(trained[method], Xdev, cfg), species, species_group, allowed)
        dev_rows.append(dict(method=method, **mars_scores(g_dev, dev, n_boot=1)))

    # Model selection uses the dev split only.
    dev_df = pd.DataFrame(dev_rows)
    best = dev_df.sort_values("bal_acc", ascending=False).method.iloc[0]
    log(f"selected on dev split: {best} (dev bal acc {dev_df.bal_acc.max():.2f})")

    base = CNN_CONFIGS[best]
    ablations = {"ablate-full": (base, best)}
    if base.get("aug"):
        for p in MARS_PARTS:
            ablations[f"ablate-no-{p}"] = (dict(base, parts=[q for q in MARS_PARTS if q != p]), None)
        ablations["ablate-no-simulation"] = (dict(base, aug=False), None)
    if base.get("artifact"):
        ablations["ablate-no-template"] = (dict(base, artifact=False), None)
    for method, (cfg, reuse) in ablations.items():
        models = trained[reuse][:2] if reuse else train_config(method, cfg, SEEDS[:2])
        evaluate(method, cfg, models, SEEDS[:2])

    mars_rows.append(dict(method="always-olivine", setting="-", **mars_scores(np.array(["olivine"] * len(ev)), ev)))

    pd.DataFrame(lab_rows).to_csv(RES / "lab_results.csv", index=False)
    pd.DataFrame(mars_rows).to_csv(RES / "mars_results.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(RES / "mars_seed_results.csv", index=False)
    pd.DataFrame(sum_rows).to_csv(RES / "mars_results_sumrule.csv", index=False)
    dev_df.assign(selected=dev_df.method == best).to_csv(RES / "dev_selection.csv", index=False)
    out = ev[["product", "target", "sol", "primary", "groups", "conf"]].copy()
    for k, v in preds.items():
        out[k] = v
    out.to_csv(RES / "mars_predictions.csv", index=False)
    summary = dict(n_lab_train=int(len(tr)), n_lab_test=int(len(te)), n_species=len(species),
                   n_lab_test_species=int(meta.iloc[te].name.nunique()), n_mars_eval=int(len(ev)),
                   n_mars_targets=int(ev.target.nunique()), selected_model=best, n_dev_eval=int(len(dev)),
                   mars_eval_counts=ev.primary.value_counts().to_dict())
    with open(RES / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
