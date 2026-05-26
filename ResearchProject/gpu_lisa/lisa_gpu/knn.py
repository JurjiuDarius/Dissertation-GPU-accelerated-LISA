"""GPU kNN: predict radius (lattice regression), scan candidates inside the
box, double the radius if too few matches, then take top-k distances.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def knn_query_gpu(gpu_idx, gpu_lr, query_points_cpu, k=10, max_iters=8, scale=2.0):
    """Returns (top_k_distances [n_q, k] CPU array, radii_final CPU, timing)."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available.")
    if not gpu_idx._loaded_points:
        gpu_idx.load_points_to_gpu()
    if not gpu_lr._loaded:
        gpu_lr.load_to_gpu()

    timing = {'transfer_to_gpu': 0.0, 'compute': 0.0, 'transfer_to_cpu': 0.0,
              'iters_used': 0}

    t0 = time.perf_counter()
    X = cp.asarray(query_points_cpu)
    cp.cuda.Stream.null.synchronize()
    timing['transfer_to_gpu'] += time.perf_counter() - t0

    t0 = time.perf_counter()
    d = gpu_idx.data_dim
    n_q = X.shape[0]

    radii = gpu_lr._fit_internal(X).reshape(-1).astype(cp.float64)
    radii = cp.maximum(radii, 1e-6)
    span = float(gpu_idx.cpu_index.max_value_each_dim
                 - gpu_idx.cpu_index.min_value_each_dim)
    cap = span * float(np.sqrt(d))
    minv = float(gpu_idx.cpu_index.min_value_each_dim)
    maxv = float(gpu_idx.cpu_index.max_value_each_dim)

    best_dists = cp.full((n_q, k), cp.inf, dtype=cp.float64)
    done = cp.zeros(n_q, dtype=bool)

    for it in range(max_iters):
        active = ~done
        if not bool(active.any().item()):
            break
        timing['iters_used'] = it + 1

        active_idx = cp.where(active)[0]
        Xa = X[active_idx]
        ra = radii[active_idx]
        n_a = int(Xa.shape[0])

        low = cp.maximum(Xa - ra[:, None], minv)
        high = cp.minimum(Xa + ra[:, None], maxv)

        low_maps = _monotone_mappings_gpu(gpu_idx, low)
        high_maps = _monotone_mappings_gpu(gpu_idx, high)
        m_lo = cp.minimum(low_maps, high_maps)
        m_hi = cp.maximum(low_maps, high_maps)
        s_lo = cp.clip(gpu_idx._predict_shard_ids_internal(m_lo),
                       0, gpu_idx.n_shards - 1)
        s_hi = cp.clip(gpu_idx._predict_shard_ids_internal(m_hi),
                       0, gpu_idx.n_shards - 1)

        p_start = gpu_idx.shard_point_starts_gpu[s_lo]
        p_end = gpu_idx.shard_point_ends_gpu[s_hi]
        cand_counts = cp.maximum(p_end - p_start, 0)
        total = int(cand_counts.sum().item())
        if total == 0:
            radii = cp.where(active, cp.minimum(radii * scale, cap), radii)
            done = done | (active & (radii >= cap))
            continue

        from .lisa_index import repeat_by_counts
        q_ids_local = repeat_by_counts(cand_counts)
        cum = cp.concatenate([cp.zeros(1, dtype=cp.int64),
                              cp.cumsum(cand_counts)])
        local_off = cp.arange(total, dtype=cp.int64) - cum[q_ids_local]
        point_ids = p_start[q_ids_local] + local_off

        cand_pts = gpu_idx.all_points_gpu[point_ids]
        low_q = low[q_ids_local]
        high_q = high[q_ids_local]
        in_box = ((cand_pts >= low_q) & (cand_pts <= high_q)).all(axis=1)

        diff = cand_pts - Xa[q_ids_local]
        dist_sq = (diff * diff).sum(axis=1)
        dist_sq = cp.where(in_box, dist_sq, cp.inf)

        match_counts = cp.zeros(n_a, dtype=cp.int64)
        _scatter_add(match_counts, q_ids_local, in_box.astype(cp.int64))
        enough = match_counts >= k

        if bool(enough.any().item()):
            # For queries with enough matches: extract their candidates and
            # top-k. Pad shorter ones to a common width with inf so we can do
            # one batched topk.
            full_counts = cp.where(enough, match_counts, cp.zeros_like(match_counts))
            max_n = int(full_counts.max().item())
            if max_n > 0:
                # Per-active-query padded distance matrix.
                pad = cp.full((n_a, max_n), cp.inf, dtype=cp.float64)
                # Build (q_id, slot_idx) for the in-box candidates only.
                in_box_idx = cp.where(in_box)[0]
                if int(in_box_idx.shape[0]) > 0:
                    q_of_inbox = q_ids_local[in_box_idx]
                    # Local slot: position of this candidate among in-box
                    # candidates of its query, computed via running count.
                    # Simple way: argsort by (q_of_inbox, position) to assign
                    # slots 0..count-1 in order.
                    order = cp.argsort(q_of_inbox, kind='stable')
                    sorted_qs = q_of_inbox[order]
                    sorted_dists = dist_sq[in_box_idx][order]
                    # Slot = position - cumulative count up to each query.
                    pos = cp.arange(sorted_qs.shape[0], dtype=cp.int64)
                    counts_by_q = cp.bincount(sorted_qs, minlength=n_a)
                    cum_counts = cp.concatenate(
                        [cp.zeros(1, dtype=cp.int64), cp.cumsum(counts_by_q)]
                    )
                    slots = pos - cum_counts[sorted_qs]
                    pad[sorted_qs, slots] = sorted_dists

                topk = cp.sort(pad, axis=1)[:, :k]
                sub_done = enough & ~done[active_idx]
                sub_done_full = cp.zeros(n_q, dtype=bool)
                sub_done_full[active_idx] = sub_done
                replace = sub_done_full
                # Write best dists.
                topk_full = cp.full((n_q, k), cp.inf, dtype=cp.float64)
                topk_full[active_idx] = topk
                best_dists = cp.where(replace[:, None], topk_full, best_dists)
                done = done | sub_done_full

        # Queries that didn't reach k: grow radius (or give up at cap).
        not_enough = active & ~done
        radii = cp.where(not_enough, cp.minimum(radii * scale, cap), radii)
        done = done | (not_enough & (radii >= cap))

    # Convert squared distances to actual distances.
    best_dists = cp.sqrt(best_dists)
    cp.cuda.Stream.null.synchronize()
    timing['compute'] += time.perf_counter() - t0

    t0 = time.perf_counter()
    out = best_dists.get()
    radii_out = radii.get()
    cp.cuda.Stream.null.synchronize()
    timing['transfer_to_cpu'] += time.perf_counter() - t0

    return out, radii_out, timing


def _monotone_mappings_gpu(gpu_idx, X):
    """GPU monotone_mappings (2-D only)."""
    idx = gpu_idx.cpu_index
    if idx.data_dim != 2:
        raise NotImplementedError("GPU monotone_mappings only implemented for d=2")
    boundaries = cp.asarray(idx.all_split_points_without_head_and_tail[0][0])
    idxes = cp.searchsorted(boundaries, X[:, 0], side='right')

    borders = cp.asarray(idx.borders)
    cell_measures = cp.asarray(idx.cell_measures)
    left = X[:, :-1]
    last_dim = X[:, -1]
    measures = cp.prod(left - borders[idxes], axis=1) / cell_measures[idxes]
    return (measures * idx.eta
            + last_dim / idx.max_value_each_dim * (idx.n_parts_each_dim - 1)
            + idxes * idx.n_parts_each_dim)


def _scatter_add(target, indices, value):
    """Same contract as cpx_add_at but also handles per-index `value` arrays."""
    if hasattr(value, 'shape') and value.shape == indices.shape:
        cnt = cp.bincount(indices, weights=value.astype(cp.float64),
                          minlength=target.shape[0])
        target += cnt.astype(target.dtype)
    else:
        cnt = cp.bincount(indices, minlength=target.shape[0])
        target += cnt.astype(target.dtype) * value
