"""Run only the classical baseline indices (R-tree, Quad-tree, cKDTree,
FAISS / PyTorch brute-force), using the same data and queries as
benchmark.py. Writes baselines_range_results.csv and baselines_knn_results.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

# Import the constants + helpers used by the main benchmark so that the
# baselines see exactly the same data and queries.
from benchmark import (
    MIN_VAL, MAX_VAL, DATA_DIM,
    T_EACH_DIM, N_MODELS, SIGMA, PAGE_SIZE, ETA,
    generate_data, generate_query_ranges,
    FAISS_AVAILABLE, RTREE_AVAILABLE, CKDTREE_AVAILABLE, PYQTREE_AVAILABLE,
)
from lisa_cpu.layout_utils import generate_grid_cells as cpu_generate_grid_cells
from lisa_cpu.lisa_index import LISAIndex
from lisa_gpu.piecewise_model import train_models_gpu


def _build_lisa_index_for_data(N, dataset):
    """Build a LISA index just enough to materialise its all_points array."""
    raw = generate_data(N, dataset=dataset)
    sorted_data, mappings, params, _, _ = cpu_generate_grid_cells(
        raw.copy(), T_EACH_DIM, N_MODELS, MIN_VAL, MAX_VAL, ETA
    )
    idx = LISAIndex(params=params, data_dim=DATA_DIM,
                    page_size=PAGE_SIZE, sigma=SIGMA)
    _, col_split = idx.monotone_mappings_and_col_split_idxes(sorted_data)
    A, B, _ = train_models_gpu(mappings, col_split, sigma=SIGMA, max_iters=50)
    idx.set_piecewise_models(A, B)
    idx.generate_pages(sorted_data, mappings, col_split)
    idx.build_flat_layout()
    return idx


def time_rtree(idx, query_ranges, n_reps):
    if not RTREE_AVAILABLE:
        return {'build': -1.0, 'mean': -1.0, 'std': -1.0}
    from rtree import index as rtree_index
    d = idx.data_dim
    all_pts = idx.all_points

    prop = rtree_index.Property()
    prop.dimension = d

    def _stream():
        for i, pt in enumerate(all_pts):
            yield (i, (pt[0], pt[1], pt[0], pt[1]), None)

    t0 = time.perf_counter()
    rt = rtree_index.Index(_stream(), properties=prop)
    build = time.perf_counter() - t0

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        for q in range(query_ranges.shape[0]):
            lo = query_ranges[q, :d]
            hi = query_ranges[q, d:]
            _ = sum(1 for _ in rt.intersection((lo[0], lo[1], hi[0], hi[1])))
        times.append(time.perf_counter() - t0)
    return {'build': build, 'mean': float(np.mean(times)), 'std': float(np.std(times))}


def time_quadtree(idx, query_ranges, n_reps):
    if not PYQTREE_AVAILABLE:
        return {'build': -1.0, 'mean': -1.0, 'std': -1.0}
    from pyqtree import Index as PyqtreeIndex
    d = idx.data_dim
    all_pts = idx.all_points

    t0 = time.perf_counter()
    qt = PyqtreeIndex(bbox=(MIN_VAL, MIN_VAL, MAX_VAL, MAX_VAL))
    for i, pt in enumerate(all_pts):
        qt.insert(item=i, bbox=(pt[0], pt[1], pt[0], pt[1]))
    build = time.perf_counter() - t0

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        for q in range(query_ranges.shape[0]):
            lo = query_ranges[q, :d]
            hi = query_ranges[q, d:]
            _ = qt.intersect((lo[0], lo[1], hi[0], hi[1]))
        times.append(time.perf_counter() - t0)
    return {'build': build, 'mean': float(np.mean(times)), 'std': float(np.std(times))}


def time_ckdtree(idx, queries, k, n_reps):
    if not CKDTREE_AVAILABLE:
        return {'build': -1.0, 'mean': -1.0, 'std': -1.0}
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    kd = cKDTree(idx.all_points)
    build = time.perf_counter() - t0
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        kd.query(queries, k=k)
        times.append(time.perf_counter() - t0)
    return {'build': build, 'mean': float(np.mean(times)), 'std': float(np.std(times))}


def _faiss_supports_current_gpu():
    """FAISS's prebuilt wheels only include kernels up to Hopper (sm_90).
    On Blackwell (sm_100+) FAISS calls abort() from C++, which Python can't
    catch — so refuse to use it pre-emptively.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_properties(0).major < 10
    except Exception:
        return False


def time_faiss(idx, queries, k, n_reps):
    """Exact GPU brute-force kNN. Uses FAISS where supported, else PyTorch
    (torch.cdist + topk). Both compute the same neighbour set.
    """
    data_f32 = idx.all_points.astype(np.float32)
    queries_f32 = queries.astype(np.float32)

    use_faiss = FAISS_AVAILABLE and _faiss_supports_current_gpu()
    if use_faiss:
        import faiss
        res = faiss.StandardGpuResources()
        flat = faiss.IndexFlatL2(idx.data_dim)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, flat)
        gpu_index.add(data_f32)
        _, _ = gpu_index.search(queries_f32[:8], k)
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            gpu_index.search(queries_f32, k)
            times.append(time.perf_counter() - t0)
        return {'mean': float(np.mean(times)), 'std': float(np.std(times)),
                'impl': 'faiss'}

    # PyTorch brute force fallback
    try:
        import torch
    except ImportError:
        return {'mean': -1.0, 'std': -1.0, 'impl': 'unavailable'}
    if not torch.cuda.is_available():
        return {'mean': -1.0, 'std': -1.0, 'impl': 'no-cuda'}

    data_t = torch.from_numpy(data_f32).cuda()
    queries_t = torch.from_numpy(queries_f32).cuda()
    _ = torch.topk(torch.cdist(queries_t[:8], data_t), k, largest=False, dim=1)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dists = torch.cdist(queries_t, data_t)
        _ = torch.topk(dists, k, largest=False, dim=1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {'mean': float(np.mean(times)), 'std': float(np.std(times)),
            'impl': 'torch'}


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def main():
    parser = argparse.ArgumentParser(description="Run only the classical baselines")
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[100_000, 1_000_000, 10_000_000])
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--range-queries', type=int, default=1000)
    parser.add_argument('--range-box-frac', type=float, default=0.005)
    parser.add_argument('--knn-queries', type=int, default=500)
    parser.add_argument('--knn-k', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='uniform')
    parser.add_argument('--output-dir', default=str(HERE / 'results'))
    parser.add_argument('--skip-rtree', action='store_true')
    parser.add_argument('--skip-quadtree', action='store_true')
    parser.add_argument('--skip-ckdtree', action='store_true')
    parser.add_argument('--skip-faiss', action='store_true')
    args = parser.parse_args()

    print(f"Baselines available: rtree={RTREE_AVAILABLE} pyqtree={PYQTREE_AVAILABLE} "
          f"ckdtree={CKDTREE_AVAILABLE} faiss={FAISS_AVAILABLE}")
    print(f"Dataset: {args.dataset}   sizes: {args.sizes}   reps: {args.reps}")

    range_rows, knn_rows = [], []

    for N in args.sizes:
        print(f"\n{'─'*65}\n  N={N:,}  dataset={args.dataset}\n{'─'*65}")

        idx = _build_lisa_index_for_data(N, args.dataset)
        query_ranges = generate_query_ranges(args.range_queries,
                                             half_width_frac=args.range_box_frac)
        # benchmark.py's run_knn_benchmark re-seeds rng(99) per call, so we
        # re-seed per N here to match its query points at every size — not
        # just the first.
        rng = np.random.default_rng(99)
        knn_queries = rng.uniform(MIN_VAL + 100, MAX_VAL - 100,
                                  size=(args.knn_queries, DATA_DIM)).astype(np.float64)

        # ── range query baselines ───────────────────────────────────────────
        rt_res = ({'build': -1.0, 'mean': -1.0, 'std': -1.0}
                  if args.skip_rtree else time_rtree(idx, query_ranges, args.reps))
        print(f"  R-tree    build={rt_res['build']:.3f}s  query mean={rt_res['mean']:.4f}s")

        qt_res = ({'build': -1.0, 'mean': -1.0, 'std': -1.0}
                  if args.skip_quadtree else time_quadtree(idx, query_ranges, args.reps))
        print(f"  Quad-tree build={qt_res['build']:.3f}s  query mean={qt_res['mean']:.4f}s")

        range_rows.append({
            'N': N, 'Q': args.range_queries,
            'rtree_build_time': rt_res['build'],
            'rtree_query_mean': rt_res['mean'], 'rtree_query_std': rt_res['std'],
            'quadtree_build_time': qt_res['build'],
            'quadtree_query_mean': qt_res['mean'], 'quadtree_query_std': qt_res['std'],
        })

        # ── kNN baselines ──────────────────────────────────────────────────
        kd_res = ({'build': -1.0, 'mean': -1.0, 'std': -1.0}
                  if args.skip_ckdtree
                  else time_ckdtree(idx, knn_queries, args.knn_k, args.reps))
        print(f"  cKDTree   build={kd_res['build']:.3f}s  query mean={kd_res['mean']:.4f}s")

        fa_res = ({'mean': -1.0, 'std': -1.0, 'impl': 'skipped'}
                  if args.skip_faiss
                  else time_faiss(idx, knn_queries, args.knn_k, args.reps))
        impl_tag = f"({fa_res.get('impl', 'unknown')})"
        print(f"  GPU brute-force kNN {impl_tag:<10}  query mean={fa_res['mean']:.4f}s")

        knn_rows.append({
            'N': N, 'Q': args.knn_queries, 'k': args.knn_k,
            'ckdtree_build_time': kd_res['build'],
            'ckdtree_query_mean': kd_res['mean'], 'ckdtree_query_std': kd_res['std'],
            'faiss_mean': fa_res['mean'], 'faiss_std': fa_res['std'],
        })

        # checkpoint each N
        write_csv(os.path.join(args.output_dir, 'baselines_range_results.csv'),
                  ['N','Q','rtree_build_time','rtree_query_mean','rtree_query_std',
                   'quadtree_build_time','quadtree_query_mean','quadtree_query_std'],
                  range_rows)
        write_csv(os.path.join(args.output_dir, 'baselines_knn_results.csv'),
                  ['N','Q','k','ckdtree_build_time','ckdtree_query_mean','ckdtree_query_std',
                   'faiss_mean','faiss_std'],
                  knn_rows)

    print(f"\nWrote {args.output_dir}/baselines_range_results.csv "
          f"and baselines_knn_results.csv")


if __name__ == "__main__":
    main()
