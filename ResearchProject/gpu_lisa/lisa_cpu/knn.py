"""CPU kNN query using lattice-regression-predicted radius + range_query.

Algorithm (matches the spirit of src/solution/LISA.knn_query):
  1. For each query point, predict an initial search radius via the trained
     lattice regression.
  2. Build a query box [x - r, x + r] for each query and run range_query.
  3. If a query's box yielded < k candidates, double its radius and re-query;
     repeat until all queries have >= k candidates (or radius saturates).
  4. For each query, compute distances to its candidates and pick the k nearest.

Returns the k nearest data points (and their distances) per query.
"""
from __future__ import annotations

import numpy as np


def knn_query_cpu(idx, lat_reg, query_points, k=10, max_iters=8, scale=2.0):
    if not hasattr(idx, 'all_points'):
        idx.build_flat_layout()

    n_q = query_points.shape[0]
    d = idx.data_dim

    radii = lat_reg.fit(query_points).reshape(-1).astype(np.float64)
    radii = np.maximum(radii, 1e-6)

    span = idx.max_value_each_dim - idx.min_value_each_dim
    cap = span * np.sqrt(d)

    results_dist = np.full((n_q, k), np.inf, dtype=np.float64)
    done = np.zeros(n_q, dtype=bool)

    for it in range(max_iters):
        active = ~done
        if not active.any():
            break
        idxs = np.flatnonzero(active)
        boxes = np.empty((idxs.shape[0], 2 * d), dtype=np.float64)
        boxes[:, :d] = query_points[idxs] - radii[idxs, None]
        boxes[:, d:] = query_points[idxs] + radii[idxs, None]
        np.clip(boxes[:, :d], idx.min_value_each_dim,
                idx.max_value_each_dim, out=boxes[:, :d])
        np.clip(boxes[:, d:], idx.min_value_each_dim,
                idx.max_value_each_dim, out=boxes[:, d:])

        _, points_list = idx.range_query(boxes, return_points=True)

        for li, qi in enumerate(idxs):
            cand = points_list[li]
            if cand.shape[0] < k:
                if radii[qi] >= cap:
                    done[qi] = True
                else:
                    radii[qi] = min(radii[qi] * scale, cap)
                continue
            diff = cand - query_points[qi]
            dists = np.sqrt((diff * diff).sum(axis=1))
            order = np.argsort(dists)[:k]
            results_dist[qi] = dists[order]
            done[qi] = True

    return results_dist, radii
