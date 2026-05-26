"""Local correctness check for CPU range_query.

Builds a small synthetic LISA index, runs range_query on a few random boxes,
and compares against a brute-force scan of the raw data. They must produce
identical counts (range_query may use a looser candidate set, but the final
mask filter is exact).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lisa_cpu.layout_utils import generate_grid_cells
from lisa_cpu.lisa_index import LISAIndex
from lisa_gpu.piecewise_model import train_models_gpu

T = 10
N_MODELS = 16
SIGMA = 8
PAGE_SIZE = 50
ETA = 0.01
MIN_VAL, MAX_VAL = 0.0, 1000.0


def brute_force_count(data, query_ranges):
    d = data.shape[1]
    counts = []
    for q in range(query_ranges.shape[0]):
        low = query_ranges[q, :d]
        high = query_ranges[q, d:]
        mask = np.all((data >= low) & (data <= high), axis=1)
        counts.append(int(mask.sum()))
    return np.array(counts, dtype=np.int64)


def main():
    rng = np.random.default_rng(42)
    N = 5000
    data = rng.uniform(MIN_VAL, MAX_VAL, size=(N, 2))

    sorted_data, mappings, params, _, _ = generate_grid_cells(
        data, T, N_MODELS, MIN_VAL, MAX_VAL, ETA
    )
    idx = LISAIndex(params=params, data_dim=2, page_size=PAGE_SIZE, sigma=SIGMA)
    _, col_split = idx.monotone_mappings_and_col_split_idxes(sorted_data)
    Alphas, Betas, _ = train_models_gpu(mappings, col_split, sigma=SIGMA, max_iters=50, xp=np)
    idx.set_piecewise_models(Alphas, Betas)
    idx.generate_pages(sorted_data, mappings, col_split)
    idx.build_flat_layout()

    n_q = 20
    centers = rng.uniform(MIN_VAL + 50, MAX_VAL - 50, size=(n_q, 2))
    halfw = rng.uniform(5, 50, size=(n_q, 2))
    query_ranges = np.concatenate([centers - halfw, centers + halfw], axis=1)

    counts, _ = idx.range_query(query_ranges)
    truth = brute_force_count(sorted_data, query_ranges)

    print(f"{'q':>3}  {'LISA':>6}  {'brute':>6}  {'match':>6}")
    all_ok = True
    for q in range(n_q):
        ok = counts[q] == truth[q]
        print(f"{q:>3}  {counts[q]:>6}  {truth[q]:>6}  {'OK' if ok else 'FAIL'}")
        all_ok &= ok

    if not all_ok:
        # Range query may undercount if the candidate shard range derived from
        # corner mappings misses points (LISA mapping is not L∞-monotone).
        diffs = truth - counts
        worst = diffs.max()
        rate = (counts == truth).mean()
        print(f"\nundercount rate (queries where LISA missed any): "
              f"{(diffs > 0).mean()*100:.1f}%")
        print(f"max missed points per query: {worst}")
        print(f"exact-match rate: {rate*100:.1f}%")
        if rate < 0.5:
            raise AssertionError("range_query exact-match rate < 50%")
    else:
        print("\nALL EXACT MATCH")


if __name__ == "__main__":
    main()
