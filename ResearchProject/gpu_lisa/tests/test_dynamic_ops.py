"""Local correctness check for CPU batched insert/delete."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lisa_cpu.layout_utils import generate_grid_cells
from lisa_cpu.lisa_index import LISAIndex
from lisa_cpu.dynamic_ops import insert_batch_cpu, delete_batch_cpu
from lisa_gpu.piecewise_model import train_models_gpu


def main():
    rng = np.random.default_rng(42)
    N = 3000
    data = rng.uniform(0, 1000.0, size=(N, 2))

    sorted_data, mappings, params, _, _ = generate_grid_cells(
        data, 8, 8, 0.0, 1000.0, 0.01
    )
    idx = LISAIndex(params=params, data_dim=2, page_size=50, sigma=8)
    _, col_split = idx.monotone_mappings_and_col_split_idxes(sorted_data)
    A, B, _ = train_models_gpu(mappings, col_split, sigma=8, max_iters=50, xp=np)
    idx.set_piecewise_models(A, B)
    idx.generate_pages(sorted_data, mappings, col_split)
    idx.build_flat_layout()

    n0 = idx.all_points.shape[0]
    print(f"initial points: {n0}")

    new_points = rng.uniform(0, 1000.0, size=(200, 2))
    n_ins = insert_batch_cpu(idx, new_points)
    print(f"inserted {n_ins} points, now: {idx.all_points.shape[0]}")
    assert idx.all_points.shape[0] == n0 + n_ins

    # Range query should now find the inserted points.
    test_box = np.array([[0, 0, 1000, 1000]], dtype=np.float64)
    counts, _ = idx.range_query(test_box)
    print(f"range_query whole-domain count: {counts[0]} (expect {n0 + n_ins})")
    assert counts[0] == n0 + n_ins, (
        f"some points missing: {counts[0]} vs {n0 + n_ins}")

    # Delete half of the new points.
    to_delete = new_points[:100]
    n_del = delete_batch_cpu(idx, to_delete)
    print(f"deleted {n_del} points, now: {idx.all_points.shape[0]}")
    assert idx.all_points.shape[0] == n0 + n_ins - n_del

    print("OK")


if __name__ == "__main__":
    main()
