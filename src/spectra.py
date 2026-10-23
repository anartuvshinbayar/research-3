"""Spectrum preprocessing and simulated-Mars degradation."""
import numpy as np
from scipy.linalg import solveh_banded

GRID = np.arange(150.0, 1300.0 + 1e-9, 2.0)  # cm^-1
SUPERCAM_FWHM = 12.0  # cm^-1, Wiens et al. (2021)
MARS_PARTS = ("res", "shift", "mix", "bg", "noise")


def _second_diff_bands(n, lam):
    main = np.full(n, 6.0)
    main[[0, -1]] = 1.0
    main[[1, -2]] = 5.0
    first = np.full(n - 1, -4.0)
    first[[0, -1]] = -2.0
    ab = np.zeros((3, n))
    ab[0], ab[1, :-1], ab[2, :-2] = main, first, np.ones(n - 2)
    return lam * ab


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """Asymmetric least squares baseline (Eilers & Boelens 2005)."""
    bands = _second_diff_bands(len(y), lam)
    w = np.ones(len(y))
    for _ in range(niter):
        ab = bands.copy()
        ab[0] += w
        z = solveh_banded(ab, w * y, lower=True)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def to_grid(x, y, grid=GRID):
    order = np.argsort(x)
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    out = np.interp(grid, x, y, left=np.nan, right=np.nan)
    valid = ~np.isnan(out)
    if valid.sum() < 10:
        return None, valid
    out[~valid] = np.interp(grid[~valid], grid[valid], out[valid])
    return out, valid


def preprocess(x, y, grid=GRID):
    """Resample, subtract baseline, normalize to unit length."""
    g, valid = to_grid(x, y, grid)
    if g is None:
        return None
    g = g - als_baseline(g)
    g[~valid] = 0.0
    g = np.clip(g, -0.05 * np.abs(g).max(), None)
    norm = np.linalg.norm(g)
    if not np.isfinite(norm) or norm == 0:
        return None
    return (g / norm).astype(np.float32)


def gaussian_broaden(x, y, fwhm):
    x, idx = np.unique(np.asarray(x, float), return_index=True)
    y = np.asarray(y, float)[idx]
    step = max(np.median(np.diff(x)), 0.25)
    xe = np.linspace(x[0], x[-1], int(round((x[-1] - x[0]) / step)) + 1)
    ye = np.interp(xe, x, y)
    sigma = fwhm / 2.3548 / step
    half = int(np.ceil(4 * sigma))
    if half < 1:
        return xe, ye
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    return xe, np.convolve(ye, k / k.sum(), mode="same")


def marsify(x, y, rng, other=None, strength=1.0, parts=None):
    """Degrade a lab spectrum to look like a SuperCam spectrum.

    parts: subset of MARS_PARTS to apply. other: optional (x, y) spectrum to mix in.
    """
    parts = set(MARS_PARTS if parts is None else parts) if strength > 0 else set()
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y = y - np.percentile(y, 1)
    y = y / (y.max() + 1e-12)

    fwhm = rng.uniform(9.0, 15.0) if "res" in parts else 0.0
    if fwhm > 0:
        x, y = gaussian_broaden(x, y, fwhm)
        y = y / (y.max() + 1e-12)

    if "shift" in parts:
        x = x + rng.normal(0, 3.0 * strength)

    if "mix" in parts and other is not None and rng.random() < min(0.5 * strength, 0.8):
        ox, oy = other
        oy = np.interp(x, ox, oy - np.percentile(oy, 1), left=0, right=0)
        if fwhm > 0:
            ox2, oy = gaussian_broaden(x, oy, fwhm)
            oy = np.interp(x, ox2, oy)
        oy = oy / (oy.max() + 1e-12)
        y = y + rng.uniform(0.1, 0.6) * oy

    if "bg" in parts:
        t = (x - x.min()) / (x.max() - x.min() + 1e-12)
        bg = (rng.uniform(0, 3) * np.exp(-t / rng.uniform(0.05, 0.4))
              + rng.uniform(0, 1.5) * np.polyval(rng.normal(0, 1, 3), t) ** 2
              + rng.uniform(0, 2) * np.exp(-0.5 * ((t - rng.uniform(0, 1)) / rng.uniform(0.2, 0.6)) ** 2))
        y = y + strength * bg

    if "noise" in parts:
        snr = rng.uniform(3, 60) / max(strength, 1e-3)
        y = y + rng.normal(0, 1.0 / snr, size=y.shape) * np.sqrt(np.clip(y, 0, None) + 0.3)
    return x, y
