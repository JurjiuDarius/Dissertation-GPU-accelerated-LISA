"""LISA partition + monotone mapping (CPU). See lisa_gpu/layout_utils.py
for the CuPy counterpart."""
import math
import time

import numpy as np


def get_split_points_and_idxes(x_data, N, max_value=None):
    """
    Split sorted 1-D array x_data into N load-balanced parts.
    Returns (split_upper_bounds, split_end_indices).
    """
    x_split_points = []
    if max_value is None:
        max_value = x_data[-1] + 0.01
    n_every_part = x_data.shape[0] // N          # Python 3: explicit integer division
    n_remainder = x_data.shape[0] % N

    for i in range(n_remainder):
        if i == N - 1:
            x_split_points.append(max_value)
            continue
        idx = (i + 1) * (n_every_part + 1)
        x_split_points.append(x_data[idx - 1])

    for i in range(n_remainder, N):
        if i == N - 1:
            x_split_points.append(max_value)
            continue
        idx = (i + 1) * n_every_part + n_remainder
        x_split_points.append(x_data[idx - 1])

    x_split_idxes = np.searchsorted(x_data, x_split_points, side='left')
    # Force the last cell to extend to N. searchsorted(side='left') against
    # the sentinel `max_value` returns the index of the first matching element,
    # not N, when the data actually contains that value (real datasets often do).
    if x_split_idxes.shape[0] > 0:
        x_split_idxes[-1] = x_data.shape[0]
    return np.array(x_split_points, dtype=x_data.dtype), x_split_idxes


def partition(data, dim, start, end, n_parts, split_points_list, split_idxes_list, max_value_each_dim):
    """
    Recursively sort data in-place by each dimension (depth-first grid partitioning).
    Fills split_points_list and split_idxes_list as a side effect.
    """
    part_data = data[start:end]
    one_dim_data = part_data[:, dim]
    sorted_idxes = np.argsort(one_dim_data, kind='stable')
    data[start:end] = part_data[sorted_idxes]
    one_dim_data = data[start:end, dim]

    split_points, split_idxes = get_split_points_and_idxes(
        one_dim_data, n_parts, max_value=max_value_each_dim
    )
    split_points_list[dim].append(split_points)
    split_idxes += start
    split_idxes_list[dim].append(split_idxes)

    next_dim_start = start
    for i in range(split_idxes.shape[0]):
        next_dim_end = int(split_idxes[i])
        if dim < data.shape[1] - 2:
            partition(
                data, dim + 1, next_dim_start, next_dim_end,
                n_parts, split_points_list, split_idxes_list, max_value_each_dim
            )
        next_dim_start = next_dim_end


def create_borders(split_points_list):
    """
    Build per-cell lower-left corner coordinates and cell area measures from split points.
    Returns (borders [n_cells, n_dims-1], cell_measures [n_cells]).
    """
    n_parts = split_points_list[0][0].shape[0]
    n_cells = len(split_points_list[-1]) * n_parts

    borders = np.zeros(
        shape=[n_cells, len(split_points_list)],
        dtype=split_points_list[0][0].dtype
    )
    all_cell_measures = np.ones(shape=[n_cells], dtype=borders.dtype)

    for dim, one_dim_split_points_list in enumerate(split_points_list):
        # Python 3: use // for integer division
        n_repeat = (n_cells // n_parts) // len(one_dim_split_points_list)

        start = 0
        for split_points in one_dim_split_points_list:
            front_split_points = np.zeros_like(split_points)
            front_split_points[1:] = split_points[:-1]
            lens = split_points - front_split_points

            tiled_fronts = np.tile(front_split_points, n_repeat)
            tiled_lens = np.tile(lens, n_repeat)

            tmp = np.reshape(
                np.array([front_split_points] * n_repeat).transpose(), [-1]
            )
            borders[start:start + tmp.shape[0], dim] = tmp

            tmp_lens = np.reshape(
                np.array([lens] * n_repeat).transpose(), [-1]
            )
            all_cell_measures[start:start + tmp_lens.shape[0]] *= tmp_lens
            start += tmp.shape[0]

    return borders, all_cell_measures


def generate_grid_cells(data, n_parts_each_dim, n_models,
                        min_value_each_dim, max_value_each_dim, eta):
    """
    Full build-phase pipeline: partition data into grid cells, compute monotone 1D
    mappings, sort each cell by mapping value, and compute model split parameters.

    Returns
    -------
    sorted_data : ndarray (N, D)
    mappings    : ndarray (N,)  — monotone 1D mapping for each point
    params      : ndarray       — packed LISA index parameters
    cell_measures : ndarray (n_cells,)
    timing      : dict with keys 'partition', 'mapping_and_sort'  (seconds)
    """
    size = data.shape[0]
    n_dim = data.shape[1]

    split_upper_bounds_list = [[] for _ in range(n_dim - 1)]
    split_idxes_list = [[] for _ in range(n_dim - 1)]

    # ── Stage 1: partition (sort by dims 0 .. n_dim-2) ────────────────────────
    t0 = time.perf_counter()
    partition(
        data, 0, 0, size, n_parts_each_dim,
        split_upper_bounds_list, split_idxes_list, max_value_each_dim
    )
    t_partition = time.perf_counter() - t0

    borders, all_cell_measures = create_borders(split_upper_bounds_list)

    # Compute last-dim split points (used only for params, not for sorting here)
    last_dim_data_sorted = np.sort(data[:, -1])
    last_dim_split_upper_bounds, _ = get_split_points_and_idxes(
        last_dim_data_sorted, n_parts_each_dim, max_value=max_value_each_dim
    )

    mappings = np.zeros(shape=[size], dtype=data.dtype)

    # ── Stage 2: compute mappings and sort within each cell ───────────────────
    t0 = time.perf_counter()
    second_last_split_idxes = split_idxes_list[-1]
    cell_id = 0
    start = 0
    for split_idxes in second_last_split_idxes:
        for i in range(split_idxes.shape[0]):
            end = int(split_idxes[i])
            if end > start:
                part_data = data[start:end]
                part_borders = borders[cell_id]
                part_cell_measure = all_cell_measures[cell_id]

                part_measures = (
                    np.prod(part_data[:, :-1] - part_borders, axis=1)
                    / part_cell_measure
                )
                part_mappings = (
                    part_measures * eta
                    + part_data[:, -1] / max_value_each_dim * (n_parts_each_dim - 1)
                    + cell_id * n_parts_each_dim
                )
                sort_idxes = np.argsort(part_mappings, kind='stable')
                data[start:end] = part_data[sort_idxes]
                mappings[start:end] = part_mappings[sort_idxes]

                start = end
            cell_id += 1
    t_mapping_sort = time.perf_counter() - t0

    timing = {
        'partition': t_partition,
        'mapping_and_sort': t_mapping_sort,
    }

    # ── Build model split mappings ────────────────────────────────────────────
    max_mapping = math.pow(n_parts_each_dim, n_dim)
    split_mappings = np.zeros(shape=[n_models], dtype=data.dtype)
    split_mappings[-1] = max_mapping + 1

    offset = mappings.shape[0] // n_models
    for i in range(1, n_models):
        idx = i * offset
        split_mappings[i - 1] = (mappings[idx] + mappings[idx - 1]) / 2.0

    split_idxes_models = np.searchsorted(mappings, split_mappings, side='right')
    model_split_mappings = np.empty(split_idxes_models.shape[0])
    model_split_mappings[:-1] = mappings[split_idxes_models[:-1]]
    model_split_mappings[-1] = max_mapping + 1

    # ── Pack params array ─────────────────────────────────────────────────────
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

    return data, mappings, np.array(params, dtype=np.float64), all_cell_measures, timing
