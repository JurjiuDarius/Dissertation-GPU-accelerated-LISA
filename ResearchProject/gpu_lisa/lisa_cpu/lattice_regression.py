"""Lattice regression: smooth interpolation on a regular d-dim grid.

Used by LISA to predict a kNN search radius from a query point.
Training is a one-time sparse linear solve (CPU); inference is a dense
gather of the 2^d corner anchors plus inverse-distance weighting.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
from scipy.sparse import csc_matrix, save_npz, load_npz
from scipy.sparse.linalg import spsolve


class LatticeRegression:
    """Smooth function approximator f(x) on a regular d-dim lattice."""

    def __init__(self, data_dim: int = 2, n_nodes_each_dim: int = 11,
                 min_value: float = 0.0, max_value: float = 10000.0,
                 alpha: float = 1.0):
        self.data_dim = data_dim
        self.n_nodes_each_dim = n_nodes_each_dim
        self.min_value = min_value
        self.max_value = max_value
        self.alpha = alpha
        self.m = n_nodes_each_dim ** data_dim
        self.B = None

        offsets = np.linspace(min_value, max_value, n_nodes_each_dim)
        self.node_coordinates = offsets.astype(np.float64)

        coords = np.meshgrid(*([offsets] * data_dim), indexing='ij')
        self.A = np.stack([c.reshape(-1) for c in coords], axis=0)

    # ------------------------------------------------------------------
    # Lattice structure
    # ------------------------------------------------------------------

    def _build_laplacian(self):
        n = self.n_nodes_each_dim
        d = self.data_dim
        m = self.m
        strides = [n ** k for k in range(d)]
        rows, cols = [], []
        for node in range(m):
            coords = []
            tmp = node
            for k in range(d):
                coords.append(tmp % n)
                tmp //= n
            for k in range(d):
                if coords[k] + 1 < n:
                    nbr = node + strides[k]
                    rows.append(node); cols.append(nbr)
                    rows.append(nbr);  cols.append(node)
        rows = np.array(rows, dtype=np.int64)
        cols = np.array(cols, dtype=np.int64)
        vals = np.ones(rows.shape[0], dtype=np.float64)
        E = csc_matrix((vals, (rows, cols)), shape=(m, m))
        deg = np.asarray(E.sum(axis=1)).reshape(-1)
        idx = np.arange(m)
        D = csc_matrix((deg, (idx, idx)), shape=(m, m))
        total = float(deg.sum())
        self.L = (D - E) / total * 2.0

    # ------------------------------------------------------------------
    # Weight matrix for input points
    # ------------------------------------------------------------------

    def _corner_indices_and_weights(self, X):
        """X: (n_points, d). Returns (corner_idxes (n, 2^d), weights (n, 2^d))."""
        n = X.shape[0]
        d = self.data_dim
        nn = self.n_nodes_each_dim
        coords = self.node_coordinates

        cell_idxes = np.empty((d, n), dtype=np.int64)
        for k in range(d):
            ix = np.searchsorted(coords, X[:, k], side='right')
            ix = np.clip(ix, 1, nn - 1)
            cell_idxes[k] = ix - 1

        n_corners = 1 << d
        corner_idxes = np.zeros((n, n_corners), dtype=np.int64)
        dists = np.zeros((n, n_corners), dtype=np.float64)
        stride = 1
        for k in range(d):
            for c in range(n_corners):
                lo_or_hi = (c >> k) & 1
                idx_k = cell_idxes[k] + lo_or_hi
                corner_idxes[:, c] += idx_k * stride
                node_coord_k = coords[idx_k]
                dists[:, c] += (X[:, k] - node_coord_k) ** 2
            stride *= nn
        dists = np.sqrt(dists)

        inv = 1.0 / (dists + 1e-8)
        weights = inv / inv.sum(axis=1, keepdims=True)
        return corner_idxes, weights

    def _build_W(self, X):
        corner_idxes, weights = self._corner_indices_and_weights(X)
        n = X.shape[0]
        rows = corner_idxes.reshape(-1)
        cols = np.repeat(np.arange(n, dtype=np.int64), corner_idxes.shape[1])
        vals = weights.reshape(-1)
        return csc_matrix((vals, (rows, cols)), shape=(self.m, n))

    # ------------------------------------------------------------------
    # Train / fit
    # ------------------------------------------------------------------

    def train(self, X_train, Y_train):
        """X_train: (n, d), Y_train: (n,) or (n, k). Solves for B."""
        if Y_train.ndim == 1:
            Y_train = Y_train[:, None]
        self._build_laplacian()
        self.training_W = self._build_W(X_train)
        m = self.m
        left = self.training_W.dot(Y_train) / m
        regular = self.training_W.dot(self.training_W.T) / m + self.alpha * self.L
        self.B = spsolve(regular.T, left).T
        if self.B.ndim == 1:
            self.B = self.B[:, None]

    def fit(self, X):
        """Predict y for each point in X. X: (n, d) -> (n, k_outputs)."""
        corner_idxes, weights = self._corner_indices_and_weights(X)
        gathered = self.B[corner_idxes]
        return (gathered * weights[:, :, None]).sum(axis=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        np.save(os.path.join(model_dir, 'B.npy'), self.B)
        np.save(os.path.join(model_dir, 'node_coords.npy'), self.node_coordinates)
        save_npz(os.path.join(model_dir, 'L.npz'), self.L.tocsc())
        meta = np.array([self.data_dim, self.n_nodes_each_dim, self.min_value,
                         self.max_value, self.alpha], dtype=np.float64)
        np.save(os.path.join(model_dir, 'meta.npy'), meta)

    def load(self, model_dir):
        meta = np.load(os.path.join(model_dir, 'meta.npy'))
        self.data_dim = int(meta[0])
        self.n_nodes_each_dim = int(meta[1])
        self.min_value = float(meta[2])
        self.max_value = float(meta[3])
        self.alpha = float(meta[4])
        self.m = self.n_nodes_each_dim ** self.data_dim
        self.B = np.load(os.path.join(model_dir, 'B.npy'))
        self.node_coordinates = np.load(os.path.join(model_dir, 'node_coords.npy'))
        self.L = load_npz(os.path.join(model_dir, 'L.npz'))


def train_radius_model_from_data(data, k, n_train_points=2000,
                                 n_nodes_each_dim=11, min_value=0.0,
                                 max_value=10000.0, alpha=1.0,
                                 rng=None):
    """Train a radius predictor: given a query point x, estimate the kNN radius.

    Training data is generated by sampling n_train_points from `data`, computing
    each one's true kNN radius (distance to the k-th neighbour) via brute force.
    For very large datasets this brute-force sampling is the main cost — by
    design it scales with n_train_points * len(data), not |data|^2.
    """
    rng = rng or np.random.default_rng(0)
    n_train_points = min(n_train_points, data.shape[0])
    pick = rng.choice(data.shape[0], size=n_train_points, replace=False)
    X = data[pick]

    # Brute-force kNN radius for each training sample.
    t_sample = time.perf_counter()
    radii = np.zeros(n_train_points, dtype=np.float64)
    chunk = 200
    for s in range(0, n_train_points, chunk):
        e = min(s + chunk, n_train_points)
        diff = data[None, :, :] - X[s:e, None, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        part = np.partition(dist, k, axis=1)[:, :k + 1]
        radii[s:e] = part.max(axis=1)
    radius_sample_time = time.perf_counter() - t_sample

    # Sparse linear solve for the lattice B matrix.
    lr = LatticeRegression(data_dim=data.shape[1],
                           n_nodes_each_dim=n_nodes_each_dim,
                           min_value=min_value, max_value=max_value,
                           alpha=alpha)
    t_solve = time.perf_counter()
    lr.train(X, radii)
    lattice_solve_time = time.perf_counter() - t_solve

    # Return both phases plus their sum (kept for backward compat with
    # call sites that just wanted "total lattice regression train time").
    return lr, {
        'total': radius_sample_time + lattice_solve_time,
        'radius_sample_time': radius_sample_time,
        'lattice_solve_time': lattice_solve_time,
    }
