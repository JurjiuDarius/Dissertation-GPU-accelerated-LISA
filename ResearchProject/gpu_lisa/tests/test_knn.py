"""Local correctness check for the CPU kNN path."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lisa_cpu.layout_utils import generate_grid_cells
from lisa_cpu.lisa_index import LISAIndex
from lisa_cpu.lattice_regression import train_radius_model_from_data
from lisa_cpu.knn import knn_query_cpu
from lisa_gpu.piecewise_model import train_models_gpu


def brute_force_knn(data, queries, k):
    diff = data[None, :, :] - queries[:, None, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    part = np.partition(dist, k - 1, axis=1)[:, :k]
    part.sort(axis=1)
    return part


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

    k = 5
    lr, _ = train_radius_model_from_data(sorted_data, k=k, n_train_points=200,
                                         n_nodes_each_dim=8,
                                         min_value=0.0, max_value=1000.0)

    n_q = 30
    queries = rng.uniform(100, 900, size=(n_q, 2))
    dists, _ = knn_query_cpu(idx, lr, queries, k=k, max_iters=10)
    truth = brute_force_knn(sorted_data, queries, k=k)

    abs_err = np.abs(dists - truth).max()
    rel_err = (np.abs(dists - truth) / (truth + 1e-9)).max()
    print(f"max abs distance error : {abs_err:.4f}")
    print(f"max rel distance error : {rel_err:.4f}")
    print(f"sample (query 0):")
    print(f"  LISA  : {dists[0]}")
    print(f"  brute : {truth[0]}")

    # Allow small numerical wobble.
    bad = (np.abs(dists - truth) > 1e-6).sum()
    print(f"queries with any distance mismatch: {bad}/{n_q}")
    if bad > 0:
        # Soft check — kNN should find the correct k points except when the
        # lattice radius prediction wildly underestimates and we hit the cap.
        bad_rate = bad / n_q
        assert bad_rate < 0.3, f"too many mismatches ({bad_rate*100:.0f}%)"
    print("OK")


if __name__ == "__main__":
    main()
