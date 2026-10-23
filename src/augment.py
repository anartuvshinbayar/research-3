"""Generate Mars-style ("marsified") copies of lab spectra, in parallel."""
from multiprocessing import Pool

import numpy as np

from spectra import marsify, preprocess

ROCK_FORMING = {"olivine", "pyroxene", "feldspar", "silica", "carbonate", "sulfate", "serpentine"}

_STATE = {}


def _init(raw_x, raw_y, partners, options):
    _STATE.update(raw_x=raw_x, raw_y=raw_y, partners=partners, options=options)


def _job(args):
    i, seed = args
    rng = np.random.default_rng(seed)
    s = _STATE
    opts = s["options"]
    j = int(rng.choice(s["partners"])) if opts.get("mix", True) and len(s["partners"]) else None
    other = (s["raw_x"][j], s["raw_y"][j]) if j is not None else None
    x, y = marsify(s["raw_x"][i], s["raw_y"][i], rng, other=other,
                   strength=opts.get("strength", 1.0), parts=opts.get("parts"))
    return preprocess(x, y)


def marsify_set(raw_x, raw_y, groups, indices, copies, seed=0, strength=1.0, mix=True, parts=None,
                processes=6):
    """Return (X_aug, source_index) with `copies` marsified versions of each index."""
    partners = np.where(np.isin(groups, list(ROCK_FORMING)))[0]
    jobs = [(int(i), seed * 10_000_019 + int(i) * 1000 + c) for i in indices for c in range(copies)]
    opts = dict(strength=strength, mix=mix, parts=parts)
    with Pool(processes, initializer=_init, initargs=(raw_x, raw_y, partners, opts)) as pool:
        out = pool.map(_job, jobs, chunksize=256)
    keep = [k for k, o in enumerate(out) if o is not None]
    X = np.stack([out[k] for k in keep])
    src = np.array([jobs[k][0] for k in keep])
    return X, src
