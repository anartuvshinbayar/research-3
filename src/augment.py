"""Parallel generation of simulated-Mars training spectra."""
from multiprocessing import Pool

import numpy as np

from spectra import marsify, preprocess

ROCK_FORMING = {"olivine", "pyroxene", "feldspar", "silica", "carbonate", "sulfate", "serpentine"}

_state = {}


def _init(raw_x, raw_y, partners, opts):
    _state.update(raw_x=raw_x, raw_y=raw_y, partners=partners, opts=opts)


def _job(args):
    i, seed = args
    rng = np.random.default_rng(seed)
    s, opts = _state, _state["opts"]
    other = None
    if opts["mix"] and len(s["partners"]):
        j = int(rng.choice(s["partners"]))
        other = (s["raw_x"][j], s["raw_y"][j])
    x, y = marsify(s["raw_x"][i], s["raw_y"][i], rng, other=other, strength=opts["strength"], parts=opts["parts"])
    return preprocess(x, y)


def marsify_set(raw_x, raw_y, groups, indices, copies, seed=0, strength=1.0, mix=True, parts=None, processes=6):
    """Return (X, src): `copies` degraded versions of each index, and the source index of each row."""
    partners = np.where(np.isin(groups, list(ROCK_FORMING)))[0]
    jobs = [(int(i), seed * 10_000_019 + int(i) * 1000 + c) for i in indices for c in range(copies)]
    opts = dict(strength=strength, mix=mix, parts=parts)
    with Pool(processes, initializer=_init, initargs=(raw_x, raw_y, partners, opts)) as pool:
        out = pool.map(_job, jobs, chunksize=256)
    keep = [k for k, o in enumerate(out) if o is not None]
    return np.stack([out[k] for k in keep]), np.array([jobs[k][0] for k in keep])
