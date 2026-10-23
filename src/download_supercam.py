"""Download the SuperCam Raman label table and the calibrated spectra it refers to."""
import csv
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = "https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_supercam"
BASE = f"{BUNDLE}/data_calibrated_spectra"
TABLE = ROOT / "data/supercam_derived/supercam_raman.csv"
OUT = ROOT / "data/supercam_cal"

_listing_cache = {}


def sol_listing(sol):
    if sol not in _listing_cache:
        url = f"{BASE}/sol_{sol:05d}/"
        html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
        _listing_cache[sol] = re.findall(r'HREF="[^"]*/([^"/]+\.fits)"', html, flags=re.I)
    return _listing_cache[sol]


def resolve(product):
    """Map a Source Product ID to the newest FITS file name in its sol folder."""
    sol = int(product.split("_")[1])
    stem = product.lower()
    matches = [f for f in sol_listing(sol) if f.lower().startswith(stem)]
    if not matches:
        return sol, None
    return sol, sorted(matches)[-1]


def fetch(product):
    try:
        sol, name = resolve(product)
    except Exception as e:  # network hiccup on listing
        return product, f"listing error: {e}"
    if name is None:
        return product, "not found"
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 0:
        return product, name
    urllib.request.urlretrieve(f"{BASE}/sol_{sol:05d}/{name}", dest)
    return product, name


def main():
    if not TABLE.exists():
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BUNDLE}/data_derived_spectra/supercam_raman.csv", TABLE)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(TABLE, newline="") as fh:
        products = [row["Source Product"].strip() for row in csv.DictReader(fh)]
    # one listing request per sol, serially
    for sol in sorted({int(p.split("_")[1]) for p in products}):
        sol_listing(sol)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, products))
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_product", "fits_file"])
        w.writerows(results)
    missing = [p for p, r in results if not r.endswith(".fits")]
    print(f"{len(results) - len(missing)} downloaded/present, {len(missing)} missing")
    for p in missing[:20]:
        print("  missing:", p, dict(results)[p])


if __name__ == "__main__":
    main()
