"""Dataset loaders, selectable via --dataset <spec>:

  uniform              2-D uniform random (default)
  skewed               Gaussian-cluster mixture + uniform background
  csv:<path>           first two columns of a CSV
  npy:<path>           a 2-D .npy array
  download:cities      SimpleMaps worldcities (~44k points, public domain)
  download:geonames    GeoNames AllCountries (~13M points, CC-BY 4.0)

All loaders rescale to [0, max_value]^d.
"""
from __future__ import annotations

import io
import os
import zipfile
from urllib.request import urlopen

import numpy as np


CITIES_URL = ("https://simplemaps.com/static/data/world-cities/basic/"
              "simplemaps_worldcities_basicv1.77.zip")
GEONAMES_URL = "https://download.geonames.org/export/dump/allCountries.zip"


def load_dataset(spec: str, N: int, min_value: float, max_value: float,
                 data_dim: int = 2, seed: int = 42, cache_dir: str | None = None):
    """Load (or generate) N points in the box [min_value, max_value]^data_dim."""
    rng = np.random.default_rng(seed)
    spec = (spec or "uniform").strip()

    if spec == "uniform":
        return rng.uniform(min_value, max_value, size=(N, data_dim)).astype(np.float64)

    if spec == "skewed":
        return _skewed_synthetic(N, min_value, max_value, data_dim, rng)

    if spec.startswith("csv:"):
        return _from_csv(spec[4:], N, min_value, max_value, rng)

    if spec.startswith("npy:"):
        return _from_npy(spec[4:], N, min_value, max_value, rng)

    if spec == "download:cities":
        return _download_cities(N, min_value, max_value, rng, cache_dir)

    if spec == "download:geonames":
        return _download_geonames(N, min_value, max_value, rng, cache_dir)

    raise ValueError(f"unknown dataset spec: {spec!r}")


# ---------------------------------------------------------------------------
# Skewed synthetic distribution: mixture of Gaussian clusters + uniform tail
# ---------------------------------------------------------------------------

def _skewed_synthetic(N, lo, hi, d, rng):
    """80% of points in 20 Gaussian clusters of varying density, 20% uniform."""
    span = hi - lo
    n_clusters = 20
    centres = rng.uniform(lo + span * 0.05, hi - span * 0.05, size=(n_clusters, d))
    sigmas = rng.uniform(span * 0.005, span * 0.05, size=n_clusters)
    weights = rng.dirichlet(np.ones(n_clusters) * 0.3)

    n_cluster_pts = int(0.8 * N)
    n_uniform_pts = N - n_cluster_pts

    counts = rng.multinomial(n_cluster_pts, weights)
    pieces = []
    for c, n_c, s_c in zip(centres, counts, sigmas):
        if n_c > 0:
            pieces.append(rng.normal(c, s_c, size=(n_c, d)))
    if n_uniform_pts > 0:
        pieces.append(rng.uniform(lo, hi, size=(n_uniform_pts, d)))
    data = np.concatenate(pieces, axis=0)
    rng.shuffle(data, axis=0)
    return np.clip(data, lo, hi).astype(np.float64)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def _from_csv(path, N, lo, hi, rng):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    arr = np.loadtxt(path, delimiter=',', usecols=(0, 1), skiprows=1)
    return _subsample_and_rescale(arr, N, lo, hi, rng)


def _from_npy(path, N, lo, hi, rng):
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPY not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"expected 2-D array with >=2 cols, got {arr.shape}")
    return _subsample_and_rescale(arr[:, :2], N, lo, hi, rng)


def _download_cities(N, lo, hi, rng, cache_dir):
    cache_dir = cache_dir or os.path.expanduser("~/.cache/gpu_lisa")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "worldcities.npy")
    if not os.path.exists(cache_path):
        print(f"  Downloading worldcities from {CITIES_URL}")
        data = urlopen(CITIES_URL, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as f:
                arr = np.loadtxt(f, delimiter=',', usecols=(2, 3), skiprows=1,
                                 dtype=str)
        coords = np.array([[float(s.strip('"')) for s in row] for row in arr],
                          dtype=np.float64)
        np.save(cache_path, coords)
    coords = np.load(cache_path)
    return _subsample_and_rescale(coords, N, lo, hi, rng)


def _download_geonames(N, lo, hi, rng, cache_dir):
    """GeoNames AllCountries — ~12M real geographic features, lat/lon columns 4 and 5."""
    cache_dir = cache_dir or os.path.expanduser("~/.cache/gpu_lisa")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "geonames.npy")
    if not os.path.exists(cache_path):
        print(f"  Downloading GeoNames AllCountries from {GEONAMES_URL}")
        print("  (~340 MB zipped; first run only, cached afterwards)")
        data = urlopen(GEONAMES_URL, timeout=300).read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            with zf.open("allCountries.txt") as f:
                # pandas is ~10× faster than np.loadtxt on 12M rows.
                try:
                    import pandas as pd
                    df = pd.read_csv(f, sep='\t', header=None,
                                     usecols=[4, 5], dtype=np.float64,
                                     na_values=[''], on_bad_lines='skip',
                                     low_memory=False)
                    coords = df.to_numpy()
                except ImportError:
                    print("  pandas not installed; falling back to np.loadtxt (slow)")
                    coords = np.loadtxt(f, delimiter='\t', usecols=(4, 5),
                                        dtype=np.float64)
        coords = coords[~np.isnan(coords).any(axis=1)]
        print(f"  Parsed {coords.shape[0]:,} GeoNames features")
        np.save(cache_path, coords)
    coords = np.load(cache_path)
    return _subsample_and_rescale(coords, N, lo, hi, rng)


def _subsample_and_rescale(arr, N, lo, hi, rng):
    if arr.shape[0] < N:
        # Bootstrap: sample with replacement, jitter slightly so we don't get
        # exact duplicates which break the index assumption of unique points.
        idx = rng.integers(0, arr.shape[0], size=N)
        arr = arr[idx]
        jitter_scale = (np.ptp(arr, axis=0) + 1e-9) * 1e-5
        arr = arr + rng.normal(0.0, jitter_scale, size=arr.shape)
    else:
        idx = rng.choice(arr.shape[0], size=N, replace=False)
        arr = arr[idx]
    return _rescale(arr, lo, hi)


def _rescale(arr, lo, hi):
    arr_min = arr.min(axis=0)
    arr_max = arr.max(axis=0)
    span = np.maximum(arr_max - arr_min, 1e-12)
    return (lo + (arr - arr_min) / span * (hi - lo)).astype(np.float64)
