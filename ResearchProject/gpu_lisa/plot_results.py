"""Generate PNG plots from the CSVs that benchmark.py writes."""
from __future__ import annotations

import argparse
import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def _read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _coerce(v):
    if v == '' or v is None:
        return float('nan')
    try:
        f = float(v)
        # Sentinel -1 means "not measured" (GPU unavailable); plot as NaN.
        return float('nan') if f == -1.0 else f
    except ValueError:
        return v


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    try:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    except ValueError:
        # No positive data on a log axis (e.g. GPU columns all NaN). Fall
        # back to linear and try again.
        for ax in fig.axes:
            if ax.get_xscale() == 'log':
                ax.set_xscale('linear')
            if ax.get_yscale() == 'log':
                ax.set_yscale('linear')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


def plot_build_stages(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    Ns = [r['N'] for r in rows]
    for stage, color in [('partition', 'tab:blue'),
                         ('map_sort', 'tab:orange'),
                         ('train', 'tab:green')]:
        cpu = [r.get(f'cpu_{stage}_mean', float('nan')) for r in rows]
        gpu = [r.get(f'gpu_{stage}_mean', float('nan')) for r in rows]
        ax.plot(Ns, cpu, 'o-', color=color, label=f'CPU {stage}')
        ax.plot(Ns, gpu, 's--', color=color, alpha=0.6, label=f'GPU {stage}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('time (s)')
    ax.set_title('LISA build stages: CPU vs GPU')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    _save(fig, out_dir, 'build_stages')


def plot_query_speedup(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    Ns = [r['N'] for r in rows]
    sp_total = [r.get('speedup_total', float('nan')) for r in rows]
    sp_compute = [r.get('speedup_compute', float('nan')) for r in rows]
    ax.plot(Ns, sp_total, 'o-', label='total (incl. transfer)')
    ax.plot(Ns, sp_compute, 's--', label='compute only')
    ax.axhline(1.0, color='grey', linestyle=':')
    ax.set_xscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('GPU speedup ×')
    ax.set_title('predict_shard_ids speedup')
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, out_dir, 'query_speedup')


def plot_range_query(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    Ns = [r['N'] for r in rows]
    cpu = [r['cpu_mean'] for r in rows]
    gpu = [r['gpu_total_mean'] for r in rows]
    ax.plot(Ns, cpu, 'o-', label='CPU')
    ax.plot(Ns, gpu, 's--', label='GPU total')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('time (s) for Q range queries')
    ax.set_title('Range query latency vs N')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    _save(fig, out_dir, 'range_query')


def plot_knn(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    Ns = [r['N'] for r in rows]
    cpu = [r['cpu_mean'] for r in rows]
    gpu = [r['gpu_total_mean'] for r in rows]
    ax.plot(Ns, cpu, 'o-', label='CPU')
    ax.plot(Ns, gpu, 's--', label='GPU total')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('time (s)')
    ax.set_title('kNN latency vs N')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    _save(fig, out_dir, 'knn')


def plot_dynamic_ops(rows, out_dir):
    # Group by N — one subplot per N showing throughput vs batch size.
    by_N = {}
    for r in rows:
        by_N.setdefault(r['N'], []).append(r)
    Ns = sorted(by_N)
    fig, axes = plt.subplots(1, len(Ns), figsize=(5 * len(Ns), 4.5),
                             squeeze=False)
    for ax, N in zip(axes[0], Ns):
        rs = sorted(by_N[N], key=lambda r: r['batch_size'])
        Bs = [r['batch_size'] for r in rs]
        cpu_ins = [r['batch_size'] / r['cpu_insert_s'] for r in rs]
        gpu_ins = [(r['batch_size'] / r['gpu_insert_s'])
                   if r['gpu_insert_s'] > 0 else float('nan') for r in rs]
        ax.plot(Bs, cpu_ins, 'o-', label='CPU insert')
        ax.plot(Bs, gpu_ins, 's--', label='GPU insert')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('batch size'); ax.set_ylabel('pts/sec')
        ax.set_title(f'N = {int(N):,}')
        ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    fig.suptitle('Batched insert throughput vs batch size')
    _save(fig, out_dir, 'dynamic_ops')


def plot_mixed_precision(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    Ns = [r['N'] for r in rows]
    sp = [r['speedup_fp16_vs_fp64'] for r in rows]
    mm = [r['fp16_shard_mismatch_rate'] * 100 for r in rows]
    axes[0].plot(Ns, sp, 'o-', color='tab:purple')
    axes[0].axhline(1.0, color='grey', linestyle=':')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('N'); axes[0].set_ylabel('fp16 speedup ×')
    axes[0].set_title('fp16 vs fp64 throughput')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(Ns, mm, 'o-', color='tab:red')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('N'); axes[1].set_ylabel('shard ID mismatch (%)')
    axes[1].set_title('fp16 accuracy cost')
    axes[1].grid(True, alpha=0.3)
    _save(fig, out_dir, 'mixed_precision')


def plot_mlp(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    Ns = [r['N'] for r in rows]
    pwl_l = [r['piecewise_mean_loss'] for r in rows]
    mlp_l = [r['mlp_mean_loss'] for r in rows]
    axes[0].plot(Ns, pwl_l, 'o-', label='piecewise-linear')
    axes[0].plot(Ns, mlp_l, 's--', label='monotonic MLP')
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('N'); axes[0].set_ylabel('mean training loss per model')
    axes[0].set_title('Local model accuracy')
    axes[0].legend(); axes[0].grid(True, which='both', alpha=0.3)

    pwl_t = [r['piecewise_train_time'] for r in rows]
    mlp_t = [r['mlp_train_time'] for r in rows]
    axes[1].plot(Ns, pwl_t, 'o-', label='piecewise-linear')
    axes[1].plot(Ns, mlp_t, 's--', label='monotonic MLP')
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('N'); axes[1].set_ylabel('train time (s)')
    axes[1].set_title('Local model training time')
    axes[1].legend(); axes[1].grid(True, which='both', alpha=0.3)
    _save(fig, out_dir, 'mlp_vs_piecewise')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    if not HAVE_MPL:
        print("matplotlib not installed — `pip install matplotlib` first")
        sys.exit(1)
    out_dir = args.out_dir or os.path.join(args.results_dir, 'plots')
    print(f"reading from {args.results_dir}, writing to {out_dir}")

    def go(name, csv_name, fn):
        rows = _read_csv(os.path.join(args.results_dir, csv_name))
        if not rows:
            print(f"  skipping {name}: {csv_name} not found")
            return
        # Rename build CSV columns to a shorter scheme used by plot_build_stages.
        if name == 'build':
            for r in rows:
                r['cpu_partition_mean'] = r.get('cpu_partition_mean')
                r['cpu_map_sort_mean'] = r.get('cpu_map_sort_mean')
                r['cpu_train_mean']    = r.get('cpu_train_mean')
                r['gpu_partition_mean'] = r.get('gpu_part_mean')
                r['gpu_map_sort_mean']  = r.get('gpu_ms_mean')
                r['gpu_train_mean']     = r.get('gpu_train_mean')
        fn(rows, out_dir)

    go('build', 'build_results.csv', plot_build_stages)
    go('query', 'query_results.csv', plot_query_speedup)
    go('range', 'range_query_results.csv', plot_range_query)
    go('knn',   'knn_results.csv',         plot_knn)
    go('dyn',   'dynamic_ops_results.csv', plot_dynamic_ops)
    go('mp',    'mixed_precision_results.csv', plot_mixed_precision)
    go('mlp',   'mlp_results.csv',          plot_mlp)


if __name__ == "__main__":
    main()
