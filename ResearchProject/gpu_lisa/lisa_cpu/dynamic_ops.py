"""CPU batched insert / delete over LISAIndex's flat point layout.

Insert: stable-sort (existing + new) by shard id, then bincount/cumsum to
rebuild offsets. Delete: locate matching rows, mask, compact.
"""
from __future__ import annotations

import numpy as np


def insert_batch_cpu(idx, new_points):
    """Insert `new_points` into the flat layout. Returns inserted count."""
    if not hasattr(idx, 'all_points'):
        idx.build_flat_layout()

    new_mappings = idx.monotone_mappings(new_points)
    new_shard_ids = idx.predict_shard_ids(new_mappings)

    n_shards = idx.shard_point_starts.shape[0]
    existing_shard_ids = _existing_shard_ids(idx)

    all_pts = np.concatenate([idx.all_points, new_points], axis=0)
    all_shards = np.concatenate([existing_shard_ids,
                                 np.clip(new_shard_ids, 0, n_shards - 1)])
    order = np.argsort(all_shards, kind='stable')
    idx.all_points = all_pts[order]
    sorted_shards = all_shards[order]

    counts = np.bincount(sorted_shards, minlength=n_shards).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    idx.shard_point_starts = offsets[:-1]
    idx.shard_point_ends = offsets[1:]
    return new_points.shape[0]


def delete_batch_cpu(idx, points_to_delete, tol=1e-9):
    """Delete rows in all_points matching any row in `points_to_delete`
    (within `tol`, first match wins, mirroring the original CPU LISA's
    delete_record_from_page semantics)."""
    if not hasattr(idx, 'all_points'):
        idx.build_flat_layout()
    if idx.all_points.shape[0] == 0 or points_to_delete.shape[0] == 0:
        return 0

    keep = np.ones(idx.all_points.shape[0], dtype=bool)
    # Sort targets and existing points lexicographically for fast matching.
    sort_target = np.lexsort(points_to_delete.T[::-1])
    targets_sorted = points_to_delete[sort_target]
    sort_existing = np.lexsort(idx.all_points.T[::-1])
    existing_sorted = idx.all_points[sort_existing]

    j = 0
    deleted = 0
    n_e = existing_sorted.shape[0]
    for t in targets_sorted:
        while j < n_e:
            row = existing_sorted[j]
            cmp = _lex_compare(row, t, tol)
            if cmp < 0:
                j += 1
            elif cmp == 0:
                keep[sort_existing[j]] = False
                j += 1
                deleted += 1
                break
            else:
                break
    if deleted == 0:
        return 0

    n_shards = idx.shard_point_starts.shape[0]
    existing_shard_ids = _existing_shard_ids(idx)[keep]
    idx.all_points = idx.all_points[keep]
    counts = np.bincount(existing_shard_ids, minlength=n_shards).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    idx.shard_point_starts = offsets[:-1]
    idx.shard_point_ends = offsets[1:]
    return deleted


def _existing_shard_ids(idx):
    n_shards = idx.shard_point_starts.shape[0]
    out = np.repeat(np.arange(n_shards, dtype=np.int64),
                    (idx.shard_point_ends - idx.shard_point_starts).astype(np.int64))
    if out.shape[0] != idx.all_points.shape[0]:
        # Empty shards or anomalies — fall back to bisect by start.
        out = np.searchsorted(idx.shard_point_starts,
                              np.arange(idx.all_points.shape[0]), side='right') - 1
        out = np.clip(out, 0, n_shards - 1)
    return out


def _lex_compare(a, b, tol):
    for i in range(a.shape[0]):
        if a[i] < b[i] - tol:
            return -1
        if a[i] > b[i] + tol:
            return 1
    return 0
