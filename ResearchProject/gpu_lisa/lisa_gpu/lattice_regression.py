"""GPU lattice-regression inference (training stays on CPU).

For each query: per-dim searchsorted → 2^d corner anchor indices →
inverse-distance weights → gather B[corners] and weighted sum.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class GPULatticeRegression:
    """Wraps a trained CPU LatticeRegression; runs fit() on the GPU."""

    def __init__(self, cpu_lr):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available.")
        self.cpu = cpu_lr
        self.data_dim = cpu_lr.data_dim
        self.n_nodes_each_dim = cpu_lr.n_nodes_each_dim
        self._loaded = False

    def load_to_gpu(self):
        self.B_gpu = cp.asarray(self.cpu.B)
        self.node_coords_gpu = cp.asarray(self.cpu.node_coordinates)
        self.n_corners = 1 << self.data_dim
        self._loaded = True

    def fit_gpu(self, X_cpu):
        """X_cpu: (n, d). Returns (predictions_cpu, timing)."""
        if not self._loaded:
            self.load_to_gpu()
        timing = {}

        t0 = time.perf_counter()
        X = cp.asarray(X_cpu)
        cp.cuda.Stream.null.synchronize()
        timing['transfer_to_gpu'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = self._fit_internal(X)
        cp.cuda.Stream.null.synchronize()
        timing['compute'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = out.get()
        cp.cuda.Stream.null.synchronize()
        timing['transfer_to_cpu'] = time.perf_counter() - t0
        return result, timing

    def _fit_internal(self, X):
        """X is already on GPU. Returns predictions (still on GPU)."""
        d = self.data_dim
        nn = self.n_nodes_each_dim
        n = X.shape[0]
        coords = self.node_coords_gpu

        cell_idxes = cp.empty((d, n), dtype=cp.int64)
        for k in range(d):
            ix = cp.searchsorted(coords, X[:, k], side='right')
            ix = cp.clip(ix, 1, nn - 1)
            cell_idxes[k] = ix - 1

        n_corners = self.n_corners
        corner_idxes = cp.zeros((n, n_corners), dtype=cp.int64)
        dists_sq = cp.zeros((n, n_corners), dtype=cp.float64)
        stride = 1
        for k in range(d):
            for c in range(n_corners):
                lo_or_hi = (c >> k) & 1
                idx_k = cell_idxes[k] + lo_or_hi
                corner_idxes[:, c] += idx_k * stride
                node_coord_k = coords[idx_k]
                dists_sq[:, c] += (X[:, k] - node_coord_k) ** 2
            stride *= nn
        dists = cp.sqrt(dists_sq)
        inv = 1.0 / (dists + 1e-8)
        weights = inv / inv.sum(axis=1, keepdims=True)

        gathered = self.B_gpu[corner_idxes]
        return (gathered * weights[:, :, None]).sum(axis=1)
