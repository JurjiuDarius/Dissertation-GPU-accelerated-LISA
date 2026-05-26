"""GPU batched insert / delete over GPULISAIndex's flat point layout.

Insert: stable-sort (existing + new) by shard id, then bincount/cumsum to
rebuild the per-shard offsets. Delete: mask + compact.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def insert_batch_gpu(gpu_idx, new_points_cpu):
    """Insert new points into the GPU-resident layout. Returns (n_inserted, timing)."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available.")
    if not gpu_idx._loaded_points:
        gpu_idx.load_points_to_gpu()

    timing = {}
    t0 = time.perf_counter()
    new_pts = cp.asarray(new_points_cpu)
    cp.cuda.Stream.null.synchronize()
    timing['transfer_to_gpu'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from .knn import _monotone_mappings_gpu
    new_mappings = _monotone_mappings_gpu(gpu_idx, new_pts)
    new_shard_ids = gpu_idx._predict_shard_ids_internal(new_mappings)
    new_shard_ids = cp.clip(new_shard_ids, 0, gpu_idx.n_shards - 1)

    existing_shard_ids = _existing_shard_ids_gpu(gpu_idx)
    all_pts = cp.concatenate([gpu_idx.all_points_gpu, new_pts], axis=0)
    all_shards = cp.concatenate([existing_shard_ids, new_shard_ids])

    order = cp.argsort(all_shards, kind='stable')
    gpu_idx.all_points_gpu = all_pts[order]
    sorted_shards = all_shards[order]

    counts = cp.bincount(sorted_shards, minlength=gpu_idx.n_shards).astype(cp.int64)
    offsets = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(counts)])
    gpu_idx.shard_point_starts_gpu = offsets[:-1]
    gpu_idx.shard_point_ends_gpu = offsets[1:]
    cp.cuda.Stream.null.synchronize()
    timing['compute'] = time.perf_counter() - t0

    return int(new_pts.shape[0]), timing


def delete_batch_gpu(gpu_idx, points_to_delete_cpu, tol=1e-9):
    """Delete matching rows from the GPU-resident layout."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available.")
    if not gpu_idx._loaded_points:
        gpu_idx.load_points_to_gpu()
    if (gpu_idx.all_points_gpu.shape[0] == 0
            or points_to_delete_cpu.shape[0] == 0):
        return 0, {'transfer_to_gpu': 0.0, 'compute': 0.0}

    timing = {}
    t0 = time.perf_counter()
    targets = cp.asarray(points_to_delete_cpu)
    cp.cuda.Stream.null.synchronize()
    timing['transfer_to_gpu'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    d = gpu_idx.data_dim
    # Lexicographic key: pack each row into a tuple-equivalent scalar via a
    # hash-style multiplicative combination. For exact equality at fp64 we can
    # use the bit pattern but multi-d compare is cleanest via per-dim sort.
    # Cheap approach: sort by every dim in reverse priority to get lexicographic
    # order on both, then use searchsorted on a single combined key.
    keys = _lex_keys(gpu_idx.all_points_gpu)
    tgt_keys = _lex_keys(targets)
    keep = cp.ones(gpu_idx.all_points_gpu.shape[0], dtype=bool)

    # For each target, find existing rows whose lex key is within tol.
    order_existing = cp.argsort(keys, kind='stable')
    sorted_keys = keys[order_existing]
    pos = cp.searchsorted(sorted_keys, tgt_keys, side='left')
    pos = cp.clip(pos, 0, sorted_keys.shape[0] - 1)
    # Verify exact match by comparing full rows.
    cand_idx = order_existing[pos]
    match = (cp.abs(gpu_idx.all_points_gpu[cand_idx] - targets) < tol).all(axis=1)
    keep_idx = cand_idx[match]
    keep[keep_idx] = False
    deleted = int((~keep).sum().item())

    if deleted > 0:
        existing_shard_ids = _existing_shard_ids_gpu(gpu_idx)[keep]
        gpu_idx.all_points_gpu = gpu_idx.all_points_gpu[keep]
        counts = cp.bincount(existing_shard_ids, minlength=gpu_idx.n_shards).astype(cp.int64)
        offsets = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(counts)])
        gpu_idx.shard_point_starts_gpu = offsets[:-1]
        gpu_idx.shard_point_ends_gpu = offsets[1:]
    cp.cuda.Stream.null.synchronize()
    timing['compute'] = time.perf_counter() - t0
    return deleted, timing


def _existing_shard_ids_gpu(gpu_idx):
    from .lisa_index import repeat_by_counts
    lens = (gpu_idx.shard_point_ends_gpu - gpu_idx.shard_point_starts_gpu).astype(cp.int64)
    return repeat_by_counts(lens)


def _lex_keys(arr):
    """One scalar key per row via scaled+summed coordinates, for fast
    sort/search-based matching. The delete path checks exact row equality
    after the sorted lookup, so a key collision can only ever miss a delete,
    never produce a wrong one.
    """
    d = arr.shape[1]
    multipliers = cp.asarray(np.array([1.0, 1.0e7, 1.0e14, 1.0e21][:d],
                                      dtype=np.float64))
    return (arr * multipliers).sum(axis=1)
