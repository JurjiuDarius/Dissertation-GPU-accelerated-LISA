"""
GPU-accelerated LISA layout utilities using CuPy.

Key accelerations vs the CPU version:
  1. partition_gpu   — uses cp.argsort (GPU sort) instead of np.argsort
  2. generate_grid_cells_gpu — vectorises the per-cell mapping computation
     over all cells at once, then does a SINGLE large cp.argsort instead of
     one sort per cell.  For 2D data with T=240 cells this replaces 240 small
     CPU sorts with one GPU sort.

Data-transfer boundaries are explicit so the benchmark can time them
separately from GPU computation.
"""
import math
import time

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_split_points_and_idxes_gpu(x_data_gpu, N, max_value):
    """
    GPU version of get_split_points_and_idxes.
    x_data_gpu is a sorted 1-D CuPy array.
    Returns (split_points_gpu, split_idxes_gpu).
    """
    n = int(x_data_gpu.shape[0])
    n_every_part = n // N
    n_remainder = n % N

    x_split_points = []
    for i in range(n_remainder):
        if i == N - 1:
            x_split_points.append(max_value)
            continue
        idx = (i + 1) * (n_every_part + 1) - 1
        x_split_points.append(float(x_data_gpu[idx]))   # tiny scalar transfer

    for i in range(n_remainder, N):
        if i == N - 1:
            x_split_points.append(max_value)
            continue
        idx = (i + 1) * n_every_part + n_remainder - 1
        x_split_points.append(float(x_data_gpu[idx]))

    split_points_gpu = cp.array(x_split_points, dtype=x_data_gpu.dtype)
    split_idxes_gpu = cp.searchsorted(x_data_gpu, split_points_gpu, side='left')
    # Force the last cell to extend to n (see CPU version for explanation).
    if split_idxes_gpu.shape[0] > 0:
        split_idxes_gpu[-1] = n
    return split_points_gpu, split_idxes_gpu


def _partition_gpu(data_gpu, dim, start, end, n_parts,
                   split_points_list, split_idxes_list, max_value_each_dim):
    """
    Recursive GPU partition: sorts data_gpu[start:end] by column `dim` in-place.
    split_points_list and split_idxes_list are filled with **CPU** numpy arrays
    (the split points are small and needed by create_borders on CPU).
    """
    part = data_gpu[start:end]
    sorted_idxes = cp.argsort(part[:, dim], kind='stable')
    data_gpu[start:end] = part[sorted_idxes]

    one_dim = data_gpu[start:end, dim]
    sp_gpu, si_gpu = _get_split_points_and_idxes_gpu(one_dim, n_parts, max_value_each_dim)

    # Transfer small arrays to CPU — needed for create_borders and loop control
    split_points_list[dim].append(sp_gpu.get().astype(np.float64))
    si_cpu = si_gpu.get().astype(np.int64) + start
    split_idxes_list[dim].append(si_cpu)

    next_start = start
    for i in range(si_cpu.shape[0]):
        next_end = int(si_cpu[i])
        if dim < data_gpu.shape[1] - 2:
            _partition_gpu(data_gpu, dim + 1, next_start, next_end,
                           n_parts, split_points_list, split_idxes_list,
                           max_value_each_dim)
        next_start = next_end


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_grid_cells_gpu(data_cpu, n_parts_each_dim, n_models,
                             min_value_each_dim, max_value_each_dim, eta):
    """
    GPU-accelerated counterpart of lisa_cpu.layout_utils.generate_grid_cells.

    Returns
    -------
    sorted_data   : ndarray (N, D)   — back on CPU
    mappings      : ndarray (N,)     — back on CPU
    params        : ndarray          — packed LISA index parameters (CPU)
    cell_measures : ndarray (n_cells,) — CPU
    timing        : dict with keys:
                      'transfer_to_gpu', 'partition', 'mapping_and_sort',
                      'transfer_to_cpu'
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available; cannot run GPU version.")

    # Import CPU helper only for create_borders (params computation stays on CPU)
    from lisa_cpu.layout_utils import create_borders, get_split_points_and_idxes

    size = data_cpu.shape[0]
    n_dim = data_cpu.shape[1]
    timing = {}

    # ── Transfer to GPU ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    data_gpu = cp.array(data_cpu)
    cp.cuda.Stream.null.synchronize()
    timing['transfer_to_gpu'] = time.perf_counter() - t0

    # ── Partition (GPU argsort per recursive call) ────────────────────────────
    split_upper_bounds_list = [[] for _ in range(n_dim - 1)]
    split_idxes_list = [[] for _ in range(n_dim - 1)]

    t0 = time.perf_counter()
    _partition_gpu(data_gpu, 0, 0, size, n_parts_each_dim,
                   split_upper_bounds_list, split_idxes_list, max_value_each_dim)
    cp.cuda.Stream.null.synchronize()
    timing['partition'] = time.perf_counter() - t0

    # CPU work: borders and last-dim split points (small arrays, fast)
    from lisa_cpu.layout_utils import create_borders
    borders_cpu, cell_measures_cpu = create_borders(split_upper_bounds_list)

    last_dim_data_cpu = data_gpu[:, -1].get()
    last_dim_sorted = np.sort(last_dim_data_cpu)
    last_dim_split_upper_bounds, _ = get_split_points_and_idxes(
        last_dim_sorted, n_parts_each_dim, max_value=max_value_each_dim
    )

    # ── Vectorised mapping + single global sort ───────────────────────────────
    #
    # After partition, data is ordered: all points in cell 0, then cell 1, …
    # Build a cell-ID array from the partition split indices, then compute all
    # mappings at once and do ONE global GPU argsort.
    #
    # This replaces `n_parts_each_dim` small per-cell sorts with a single large
    # GPU sort — this is the key GPU speedup in the build phase.

    t0 = time.perf_counter()

    # Build cell_ids array on GPU from the last-dim split indices
    # split_idxes_list[-1][0] contains the absolute end-indices of each cell
    si = split_idxes_list[-1][0]   # shape (n_cells,), CPU numpy int64 array
    counts = np.empty(si.shape[0], dtype=np.int64)
    counts[0] = si[0]
    counts[1:] = si[1:] - si[:-1]
    cell_ids_cpu = np.repeat(np.arange(si.shape[0], dtype=np.int64), counts)
    cell_ids_gpu = cp.array(cell_ids_cpu)

    borders_gpu = cp.array(borders_cpu)            # (n_cells, n_dim-1)
    cell_measures_gpu = cp.array(cell_measures_cpu) # (n_cells,)

    # Vectorised measure computation: shape (N, n_dim-1) → (N,)
    data_left = data_gpu[:, :-1]                   # (N, n_dim-1)
    borders_per_pt = borders_gpu[cell_ids_gpu]     # (N, n_dim-1)
    measures = (
        cp.prod(data_left - borders_per_pt, axis=1)
        / cell_measures_gpu[cell_ids_gpu]
    )

    # Mapping formula (same as CPU version)
    mappings_gpu = (
        measures * eta
        + data_gpu[:, -1] / max_value_each_dim * (n_parts_each_dim - 1)
        + cell_ids_gpu * n_parts_each_dim
    )

    # Single global GPU sort
    sort_idxes = cp.argsort(mappings_gpu, kind='stable')
    data_gpu = data_gpu[sort_idxes]
    mappings_gpu = mappings_gpu[sort_idxes]

    cp.cuda.Stream.null.synchronize()
    timing['mapping_and_sort'] = time.perf_counter() - t0

    # ── Transfer back to CPU ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    sorted_data = data_gpu.get()
    mappings_cpu = mappings_gpu.get()
    cp.cuda.Stream.null.synchronize()
    timing['transfer_to_cpu'] = time.perf_counter() - t0

    # ── Build model split mappings (CPU, same as original) ───────────────────
    max_mapping = math.pow(n_parts_each_dim, n_dim)
    split_mappings = np.zeros(n_models, dtype=sorted_data.dtype)
    split_mappings[-1] = max_mapping + 1
    offset = mappings_cpu.shape[0] // n_models
    for i in range(1, n_models):
        idx = i * offset
        split_mappings[i - 1] = (mappings_cpu[idx] + mappings_cpu[idx - 1]) / 2.0

    si_models = np.searchsorted(mappings_cpu, split_mappings, side='right')
    model_split_mappings = np.empty(si_models.shape[0])
    model_split_mappings[:-1] = mappings_cpu[si_models[:-1]]
    model_split_mappings[-1] = max_mapping + 1

    params = []
    for one_dim_list in split_upper_bounds_list:
        for arr in one_dim_list:
            params.extend(arr.tolist())
    params.extend(last_dim_split_upper_bounds.tolist())
    params.extend(model_split_mappings.tolist())
    params.append(eta)
    params.append(float(min_value_each_dim))
    params.append(float(max_value_each_dim))
    params.append(float(n_models))
    params.append(n_parts_each_dim + 0.5)
    params.append(n_dim + 0.5)

    return (sorted_data, mappings_cpu,
            np.array(params, dtype=np.float64), cell_measures_cpu, timing)
