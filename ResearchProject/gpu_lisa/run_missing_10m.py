"""Run the four query-path benchmarks at N=10M with GPU-trained weights.

Writes `<output-dir>/missing_benchmarks_10m.csv` in long format
(columns: dataset, stage, field, value) plus a sibling system_info.txt.

Each timed GPU stage is preceded by a warm-up call so first-use kernel
compilation does not land in a timed repetition.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import benchmark
from benchmark import (
    run_build_benchmark, run_query_benchmark, run_range_query_benchmark,
    run_knn_benchmark, run_dynamic_ops_benchmark, CUPY_AVAILABLE,
)
# These are only exposed at module scope when CuPy loads; guard the import
# so this script imports cleanly on machines without CUDA.
if CUPY_AVAILABLE:
    from benchmark import GPULISAIndex, gpu_warmup
else:
    GPULISAIndex = None
    def gpu_warmup():  # no-op fallback
        return


DATASETS = ['uniform', 'skewed', 'download:geonames']


def _warmup_gpu_stages(qr):
    """Run one untimed call of each GPU op so kernel compilation does not
    appear in a timed repetition."""
    if not CUPY_AVAILABLE:
        return
    gpu_warmup()
    idx = qr['_idx']
    idx.build_flat_layout()
    gpu_idx = GPULISAIndex(idx)
    gpu_idx.load_to_gpu()
    gpu_idx.load_points_to_gpu()
    tiny_q = np.zeros(16, dtype=np.float64) + float(idx.model_split_mappings[0])
    gpu_idx.predict_shard_ids_gpu(tiny_q)
    from benchmark import generate_query_ranges
    tiny_ranges = generate_query_ranges(8, half_width_frac=0.001, seed=12345)
    d = idx.data_dim
    low_maps = idx.monotone_mappings(tiny_ranges[:, :d])
    high_maps = idx.monotone_mappings(tiny_ranges[:, d:])
    gpu_idx.range_query_gpu(tiny_ranges, low_maps, high_maps)


def _system_info_text():
    info = benchmark.get_system_info()
    lines = []
    lines.append(f"timestamp        : {datetime.now().isoformat()}")
    lines.append(f"os               : {info.get('os')}")
    lines.append(f"python           : {info.get('python')}")
    lines.append(f"numpy            : {info.get('numpy')}")
    lines.append(f"cupy             : {info.get('cupy')}")
    lines.append(f"cuda             : {info.get('cuda')}")
    lines.append(f"gpu_driver       : {info.get('driver')}")
    lines.append(f"cpu              : {info.get('cpu')}")
    lines.append(f"gpu              : {info.get('gpu_name')}")
    if info.get('gpu_vram_gb', -1) > 0:
        lines.append(f"gpu_vram_gb      : {info['gpu_vram_gb']}")
        lines.append(f"gpu_free_gb_init : {info['gpu_free_gb']}")
    return "\n".join(lines) + "\n"


def _row_iter(dataset, stage, payload):
    """Yield (dataset, stage, field, value) rows from a result dict."""
    for k, v in payload.items():
        if k.startswith('_'):
            continue
        yield {'dataset': dataset, 'stage': stage, 'field': k, 'value': v}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help='datasets to re-measure at N=10M')
    parser.add_argument('--output-dir', default=str(HERE / 'results_10m_fixed'))
    parser.add_argument('--reuse-weights', action='store_true',
                        help='If set, weights are cached/reloaded from --output-dir')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'missing_benchmarks_10m.csv')

    sysinfo_text = _system_info_text()
    with open(os.path.join(args.output_dir, 'system_info.txt'), 'w') as f:
        f.write(sysinfo_text)
    print(sysinfo_text)

    N = 10_000_000
    n_reps = 3

    rows = []
    for dataset in args.datasets:
        print(f"\n{'='*65}")
        print(f"  DATASET: {dataset}   N={N:,}")
        print(f"{'='*65}")

        br = run_build_benchmark(
            N, n_reps, skip_training=False, dataset=dataset,
            reuse_weights_dir=(args.output_dir if args.reuse_weights else None),
        )

        qr = run_query_benchmark(br, n_reps)
        assert qr['weights_source'] == 'gpu-trained', (
            f"expected gpu-trained weights but got {qr['weights_source']!r}")
        _warmup_gpu_stages(qr)
        rr = run_range_query_benchmark(br, qr, n_reps,
                                       n_queries=1000,
                                       half_width_frac=0.005)
        _warmup_gpu_stages(qr)
        kr = run_knn_benchmark(br, qr, n_reps,
                               n_queries=500, k=10, n_train_points=300)
        _warmup_gpu_stages(qr)
        dr_rows = run_dynamic_ops_benchmark(
            br, qr, batch_sizes=(1000, 10000, 100000))

        for r in _row_iter(dataset, 'query', qr):
            rows.append(r)
        for r in _row_iter(dataset, 'range_query', rr):
            rows.append(r)
        for r in _row_iter(dataset, 'knn', kr):
            rows.append(r)
        for dr in dr_rows:
            for r in _row_iter(dataset, f'dynamic_ops_batch{dr["batch_size"]}', dr):
                rows.append(r)

        _write_consolidated_csv(csv_path, rows, sysinfo_text)
        print(f"\n  wrote {len(rows)} rows to {csv_path}")


def _write_consolidated_csv(path, rows, sysinfo_text):
    """Write the long-format consolidated CSV. Prepends a `# system:` block."""
    with open(path, 'w', newline='') as f:
        f.write("# system_info:\n")
        for line in sysinfo_text.splitlines():
            f.write(f"#   {line}\n")
        f.write("#\n")
        w = csv.DictWriter(f, fieldnames=['dataset', 'stage', 'field', 'value'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
