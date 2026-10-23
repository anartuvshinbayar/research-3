"""Follow-up: which simulated-Mars effect helps or hurts? Train the CNN with ONE effect at a time
(2 seeds each) and score it on the dev and test splits. Writes results/single_effects.csv."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import (CACHE, COPIES, EVAL_EXCLUDE, MARS_GROUPS, RES, cached_marsify,
                             group_from_species_proba, lab_split, load, log, mars_scores)
from spectra import MARS_PARTS


def main():
    import torch
    from models import SpectrumCNN, cnn_proba, device, train_cnn

    meta, X, raw_x, raw_y, mars, Xmars = load()
    groups = meta.group.to_numpy()
    tr, _ = lab_split(meta)
    species = sorted(meta.name.unique())
    sp_index = {s: i for i, s in enumerate(species)}
    species_group = dict(zip(meta.name, meta.group))
    y_sp = meta.name.map(sp_index).to_numpy()
    allowed = set(MARS_GROUPS)
    splits = {s: mars[(mars.split == s) & ~mars.primary.isin(EVAL_EXCLUDE)] for s in ["dev", "test"]}

    def models_for(tag, parts, seeds):
        out = []
        for seed in seeds:
            path = CACHE / f"cnn_{tag}_s{seed}.pt"
            if path.exists():
                m = SpectrumCNN(X.shape[1], len(species))
                m.load_state_dict(torch.load(path, map_location="cpu"))
                out.append(m.to(device()).eval())
                continue
            if parts is None:
                Xtr, ytr = np.repeat(X[tr], 4, axis=0), np.repeat(y_sp[tr], 4)
            else:
                Xa, src = cached_marsify("train_" + "-".join(parts), raw_x, raw_y, groups, tr, COPIES,
                                         seed=seed, parts=parts)
                Xtr, ytr = np.concatenate([X[tr], Xa]), np.concatenate([y_sp[tr], y_sp[src]])
            t = time.time()
            m = train_cnn(Xtr.astype(np.float32), ytr, len(species), epochs=20, seed=seed, log=lambda *a: None)
            torch.save(m.state_dict(), path)
            log(f"trained {tag} seed {seed} in {time.time() - t:.0f}s")
            out.append(m)
        return out

    variants = {"none (lab only)": ("cnn-clean", None)}
    for p in MARS_PARTS:
        variants[f"only {p}"] = (f"single-{p}", [p])
    variants["all five"] = ("cnn-marsified", list(MARS_PARTS))
    rows = []
    for label, (tag, parts) in variants.items():
        models = models_for(tag, parts, [0, 1])
        for split, df in splits.items():
            Xq = Xmars[df.index.to_numpy()]
            P = np.mean([cnn_proba(m, Xq) for m in models], axis=0)
            g = group_from_species_proba(P, species, species_group, allowed)
            r = mars_scores(g, df, n_boot=2000 if split == "test" else 1)
            rows.append(dict(variant=label, split=split, **r))
            log(f"{label:18s} {split:4s} acc {r['acc']:.2f} bal {r['bal_acc']:.2f}")
    pd.DataFrame(rows).to_csv(RES / "single_effects.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
