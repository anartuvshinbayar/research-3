"""Paired cluster-bootstrap tests of balanced-accuracy differences between methods."""
import numpy as np
import pandas as pd

from common import RES
PAIRS = [("cosine-window|mars", "cosine|mars"), ("cosine-window|mars", "cnn-clean|mars"),
         ("cnn-clean|mars", "cnn-marsified|mars"), ("cnn-marsified+artifact|mars", "cnn-marsified|mars"),
         ("cosine|mars", "cosine|open"), ("cosine-window|mars", "cosine-broadened-window|mars")]


def bal(hit, prim, classes):
    return np.mean([hit[prim == c].mean() for c in classes if (prim == c).any()])


def main(n_boot=5000, seed=0):
    p = pd.read_csv(RES / "mars_predictions.csv")
    classes = sorted(p.primary.unique())
    targets = p.target.unique()
    idx = {t: np.where(p.target.to_numpy() == t)[0] for t in targets}
    prim = p.primary.to_numpy()
    hits = {c: np.array([a in g.split("|") for a, g in zip(p[c], p.groups)]) for pair in PAIRS for c in pair}
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in PAIRS:
        d0 = bal(hits[a], prim, classes) - bal(hits[b], prim, classes)
        ds = []
        for _ in range(n_boot):
            sel = np.concatenate([idx[t] for t in rng.choice(targets, len(targets))])
            ds.append(bal(hits[a][sel], prim[sel], classes) - bal(hits[b][sel], prim[sel], classes))
        ds = np.array(ds)
        lo, hi = np.percentile(ds, [2.5, 97.5])
        p_two = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
        rows.append(dict(a=a, b=b, diff=d0, lo=lo, hi=hi, p=min(p_two, 1.0)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "paired_tests.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
