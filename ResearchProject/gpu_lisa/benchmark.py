"""Benchmark CPU (NumPy) vs GPU (CuPy) LISA across build, query, range query,
kNN, dynamic ops, mixed precision, and MLP local-model stages.

    python benchmark.py
    python benchmark.py --sizes 100000 1000000 10000000
    python benchmark.py --dataset skewed --output-dir results_skewed/

Writes one CSV per stage plus paper_report.txt into --output-dir.
"""
import argparse
import csv
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from lisa_cpu.layout_utils import generate_grid_cells as cpu_generate_grid_cells
from lisa_cpu.piecewise_model import PiecewiseModel
from lisa_cpu.lisa_index import LISAIndex
from lisa_cpu.lattice_regression import train_radius_model_from_data
from lisa_cpu.knn import knn_query_cpu
from lisa_cpu.dynamic_ops import insert_batch_cpu, delete_batch_cpu

try:
    import cupy as cp
    cp.array([1.0])          # force CUDA initialisation — catches driver errors
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

from lisa_gpu.piecewise_model import train_models_gpu  # has numpy fallback

if CUPY_AVAILABLE:
    from lisa_gpu.layout_utils import generate_grid_cells_gpu
    from lisa_gpu.lisa_index import GPULISAIndex, gpu_warmup
    from lisa_gpu.lattice_regression import GPULatticeRegression
    from lisa_gpu.knn import knn_query_gpu
    from lisa_gpu.dynamic_ops import insert_batch_gpu, delete_batch_gpu

try:
    from lisa_gpu.mlp_model import train_mlp_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import faiss  # GPU brute-force kNN baseline (Meta)
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from rtree import index as rtree_index  # classic spatial-DB R-tree (libspatialindex)
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False

try:
    from scipy.spatial import cKDTree   # standard CPU KD-tree
    CKDTREE_AVAILABLE = True
except ImportError:
    CKDTREE_AVAILABLE = False

try:
    from pyqtree import Index as PyqtreeIndex
    PYQTREE_AVAILABLE = True
except ImportError:
    PYQTREE_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Index parameters
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SIZES = [100_000, 1_000_000, 10_000_000, 50_000_000]
N_QUERIES     = 10_000
N_REPS        = 3

T_EACH_DIM = 50
N_MODELS   = 64
SIGMA      = 50
PAGE_SIZE  = 100
ETA        = 0.01
MIN_VAL    = 0.0
MAX_VAL    = 10_000.0
DATA_DIM   = 2

TRAIN_MODELS_MAX_SIZE = 1_000_000   # skip training above this


# ──────────────────────────────────────────────────────────────────────────────
# System info helpers
# ──────────────────────────────────────────────────────────────────────────────

def _shell(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def get_system_info():
    info = {}
    info['date']    = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info['python']  = platform.python_version()
    info['numpy']   = np.__version__
    info['os']      = platform.platform()

    # CPU
    cpu = _shell("lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2")
    if not cpu or cpu == "unknown":
        cpu = _shell("sysctl -n machdep.cpu.brand_string 2>/dev/null")
    info['cpu'] = cpu.strip() or platform.processor() or "unknown"

    # CUDA
    info['cuda'] = _shell("nvcc --version 2>/dev/null | grep 'release' | "
                          "sed 's/.*release //' | sed 's/,.*//'")
    try:
        info['driver'] = _shell("nvidia-smi --query-gpu=driver_version "
                                "--format=csv,noheader").split('\n')[0]
    except Exception:
        info['driver'] = "unknown"

    if CUPY_AVAILABLE:
        info['cupy'] = cp.__version__
        try:
            dev = cp.cuda.Device()
            # nvidia-smi is more reliable than CuPy's runtime bindings
            name = _shell("nvidia-smi --query-gpu=name --format=csv,noheader"
                          ).split('\n')[0].strip()
            if not name or name == "unknown":
                props = cp.cuda.runtime.getDeviceProperties(dev.id)
                name  = props['name']
                if isinstance(name, bytes):
                    name = name.decode()
            info['gpu_name'] = name or "unknown"
            free, total = dev.mem_info
            info['gpu_vram_gb'] = round(total / 1024**3, 1)
            info['gpu_free_gb'] = round(free  / 1024**3, 1)
        except Exception as e:
            info['gpu_name']    = f"unknown ({e})"
            info['gpu_vram_gb'] = -1
            info['gpu_free_gb'] = -1
    else:
        info['cupy']        = "not installed"
        info['gpu_name']    = "N/A"
        info['gpu_vram_gb'] = -1
        info['gpu_free_gb'] = -1

    return info


def memory_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        return -1


def gpu_memory_mb():
    """CuPy memory pool size in MB (pool total, includes free blocks)."""
    if not CUPY_AVAILABLE:
        return -1
    return cp.get_default_memory_pool().total_bytes() / 1024**2


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def generate_data(N, seed=42, dataset='uniform'):
    """Generate / load a dataset of N points in [MIN_VAL, MAX_VAL]^DATA_DIM."""
    from datasets import load_dataset
    return load_dataset(dataset, N, MIN_VAL, MAX_VAL, data_dim=DATA_DIM, seed=seed)


def generate_query_mappings(index, N_q, seed=0):
    max_map = float(index.model_split_mappings[-1]) * 0.99
    return np.random.default_rng(seed).uniform(0, max_map, size=N_q).astype(np.float64)


def generate_query_ranges(N_q, half_width_frac=0.005, seed=1):
    """Random axis-aligned boxes with width ≈ half_width_frac * (MAX-MIN)."""
    rng = np.random.default_rng(seed)
    span = MAX_VAL - MIN_VAL
    half = half_width_frac * span
    centers = rng.uniform(MIN_VAL + half, MAX_VAL - half, size=(N_q, DATA_DIM))
    halfw = rng.uniform(half * 0.5, half * 1.5, size=(N_q, DATA_DIM))
    return np.concatenate([centers - halfw, centers + halfw], axis=1).astype(np.float64)


def train_models(mappings, col_split_idxes, sigma):
    n_cols = col_split_idxes.shape[0]
    Alphas = np.zeros([n_cols, sigma], dtype=np.float64)
    Betas  = np.zeros([n_cols, sigma], dtype=np.float64)
    t0 = time.perf_counter()
    start = 0
    for i in range(n_cols):
        end = int(col_split_idxes[i])
        pm  = PiecewiseModel(i, mappings[start:end], sigma)
        pm.train()
        Alphas[i] = pm.alphas if pm.alphas is not None else pm.init_alphas
        Betas[i]  = pm.betas  if pm.betas  is not None else pm.init_betas
        start = end
    return Alphas, Betas, time.perf_counter() - t0


def _eval_training_loss(mappings, col_split_idxes, Alphas, Betas):
    """Sum of squared residuals across all models — used as CPU/GPU sanity check.

    Both CPU PiecewiseModel and the GPU BatchedPiecewiseTrainer return Betas
    in *shifted* coordinates (the per-column min has been subtracted). We
    shift the segment mappings the same way and then evaluate f(x) directly
    against arange(seg_len) — no extra min subtraction on Betas.
    """
    total = 0.0
    start = 0
    for i in range(col_split_idxes.shape[0]):
        end = int(col_split_idxes[i])
        seg = mappings[start:end]
        if seg.size == 0:
            start = end
            continue
        seg_rel = seg - seg.min()
        A = np.maximum(seg_rel[:, None] - Betas[i][None, :], 0.0)
        pred = (A * Alphas[i][None, :]).sum(axis=1).clip(0, end - start)
        positions = np.arange(end - start, dtype=np.float64)
        total += float(((pred - positions) ** 2).sum())
        start = end
    return total


def make_synthetic_models(n_models, sigma, model_split_mappings, seed=7):
    rng = np.random.default_rng(seed)
    Betas  = np.sort(rng.uniform(0, float(model_split_mappings[-1]),
                                  size=(n_models, sigma)), axis=1)
    Alphas = np.abs(rng.uniform(0, 0.01, size=(n_models, sigma)))
    return Alphas, Betas


def _weights_cache_path(output_dir, N, dataset, sigma):
    """Returns the on-disk path for the GPU-trained weight cache."""
    safe_dataset = dataset.replace(':', '-').replace('/', '_')
    return os.path.join(output_dir,
                        f'weights_N{N}_{safe_dataset}_sigma{sigma}.npz')


def _load_cached_weights(output_dir, N, dataset, sigma):
    """Load (Alphas, Betas) from disk if a matching cache exists."""
    path = _weights_cache_path(output_dir, N, dataset, sigma)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        if int(data['N']) != N or str(data['dataset']) != dataset or int(data['sigma']) != sigma:
            return None
        return data['Alphas'], data['Betas']
    except Exception:
        return None


def _save_cached_weights(output_dir, N, dataset, sigma, Alphas, Betas):
    os.makedirs(output_dir, exist_ok=True)
    path = _weights_cache_path(output_dir, N, dataset, sigma)
    np.savez(path, N=N, dataset=dataset, sigma=sigma,
             Alphas=Alphas, Betas=Betas)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ──────────────────────────────────────────────────────────────────────────────

def fmt(t, unit='auto'):
    if t < 0:
        return "     N/A"
    if unit == 'auto':
        if t < 1e-3: return f"{t*1e6:7.1f} µs"
        if t < 1:    return f"{t*1e3:7.1f} ms"
        return             f"{t:7.3f}  s"
    return f"{t:.6f}"


def speedup(cpu_t, gpu_t):
    if gpu_t <= 0 or cpu_t <= 0:
        return float('nan')
    return cpu_t / gpu_t


def speedup_str(cpu_t, gpu_t):
    s = speedup(cpu_t, gpu_t)
    return "   N/A" if np.isnan(s) else f"{s:5.1f}×"


# ──────────────────────────────────────────────────────────────────────────────
# Build benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_build_benchmark(N, n_reps, skip_training, dataset='uniform',
                        reuse_weights_dir=None):
    print(f"\n{'─'*65}")
    print(f"  BUILD  N={N:,}  dataset={dataset}  (reps={n_reps})")
    print(f"{'─'*65}")

    raw = generate_data(N, dataset=dataset)

    # ── CPU ──────────────────────────────────────────────────────────────────
    cpu_parts, cpu_maps_sorts = [], []
    last_result = None
    for _ in range(n_reps):
        data = raw.copy()
        out = cpu_generate_grid_cells(data, T_EACH_DIM, N_MODELS, MIN_VAL, MAX_VAL, ETA)
        cpu_parts.append(out[4]['partition'])
        cpu_maps_sorts.append(out[4]['mapping_and_sort'])
        last_result = out

    sorted_cpu, maps_cpu, params_cpu = last_result[0], last_result[1], last_result[2]

    cpu_p_mean, cpu_p_std   = np.mean(cpu_parts),     np.std(cpu_parts)
    cpu_ms_mean, cpu_ms_std = np.mean(cpu_maps_sorts), np.std(cpu_maps_sorts)

    print(f"  CPU partition      : {fmt(cpu_p_mean)}  ± {fmt(cpu_p_std)}")
    print(f"  CPU mapping+sort   : {fmt(cpu_ms_mean)}  ± {fmt(cpu_ms_std)}")

    # ── CPU model training ────────────────────────────────────────────────────
    cpu_train_mean = cpu_train_std = -1.0
    Alphas_ref = Betas_ref = None

    if not skip_training and N <= TRAIN_MODELS_MAX_SIZE:
        train_times = []
        for _ in range(max(1, n_reps // 2)):   # fewer reps — training is slow
            data = raw.copy()
            _, m, p, _, _ = cpu_generate_grid_cells(
                data, T_EACH_DIM, N_MODELS, MIN_VAL, MAX_VAL, ETA)
            idx_tmp = LISAIndex(params=p, data_dim=DATA_DIM,
                                page_size=PAGE_SIZE, sigma=SIGMA)
            _, col_split = idx_tmp.monotone_mappings_and_col_split_idxes(data)
            A, B, t = train_models(m, col_split, SIGMA)
            train_times.append(t)
        Alphas_ref, Betas_ref = A, B
        cpu_train_mean = np.mean(train_times)
        cpu_train_std  = np.std(train_times)
        print(f"  CPU model training : {fmt(cpu_train_mean)}  ± {fmt(cpu_train_std)}"
              f"  ({N_MODELS} models, σ={SIGMA})")
    else:
        reason = "--skip-training" if skip_training else f"N > {TRAIN_MODELS_MAX_SIZE:,}"
        print(f"  CPU model training : skipped ({reason})")

    # ── GPU ──────────────────────────────────────────────────────────────────
    gpu_h2d_mean = gpu_p_mean = gpu_ms_mean = gpu_d2h_mean = -1.0
    gpu_h2d_std  = gpu_p_std  = gpu_ms_std  = gpu_d2h_std  = -1.0
    maps_gpu = None

    if CUPY_AVAILABLE:
        gpu_warmup()
        stages = {k: [] for k in ('transfer_to_gpu','partition',
                                   'mapping_and_sort','transfer_to_cpu')}
        last_gpu = None
        for _ in range(n_reps):
            data = raw.copy()
            out  = generate_grid_cells_gpu(data, T_EACH_DIM, N_MODELS,
                                           MIN_VAL, MAX_VAL, ETA)
            for k in stages:
                stages[k].append(out[4][k])
            last_gpu = out

        maps_gpu = last_gpu[1]

        gpu_h2d_mean,  gpu_h2d_std  = np.mean(stages['transfer_to_gpu']),  np.std(stages['transfer_to_gpu'])
        gpu_p_mean,    gpu_p_std    = np.mean(stages['partition']),         np.std(stages['partition'])
        gpu_ms_mean,   gpu_ms_std   = np.mean(stages['mapping_and_sort']),  np.std(stages['mapping_and_sort'])
        gpu_d2h_mean,  gpu_d2h_std  = np.mean(stages['transfer_to_cpu']),   np.std(stages['transfer_to_cpu'])

        print(f"  GPU H→D transfer   : {fmt(gpu_h2d_mean)}  ± {fmt(gpu_h2d_std)}")
        print(f"  GPU partition      : {fmt(gpu_p_mean)}  ± {fmt(gpu_p_std)}"
              f"  (speedup: {speedup_str(cpu_p_mean, gpu_p_mean)})")
        print(f"  GPU mapping+sort   : {fmt(gpu_ms_mean)}  ± {fmt(gpu_ms_std)}"
              f"  (speedup: {speedup_str(cpu_ms_mean, gpu_ms_mean)})")
        print(f"  GPU D→H transfer   : {fmt(gpu_d2h_mean)}  ± {fmt(gpu_d2h_std)}")

        # Correctness check
        ms_cpu_ref = np.sort(maps_cpu)
        ms_gpu_ref = np.sort(maps_gpu)
        max_diff   = float(np.max(np.abs(ms_cpu_ref - ms_gpu_ref)))
        ok = "✓ correct" if max_diff < 1e-6 else f"✗ max_diff={max_diff:.2e}"
        print(f"  Correctness        : {ok}")
        del ms_cpu_ref, ms_gpu_ref, maps_gpu, last_gpu
    else:
        print("  GPU                : not available (CuPy not installed)")

    # ── GPU model training ───────────────────────────────────────────────────
    gpu_train_mean = gpu_train_std = -1.0
    gpu_mem_high_water_mb = -1.0
    Alphas_gpu = Betas_gpu = None
    if CUPY_AVAILABLE and not skip_training:
        # Cache hit? Skip the expensive training and use the saved weights.
        cached = (_load_cached_weights(reuse_weights_dir, N, dataset, SIGMA)
                  if reuse_weights_dir else None)
        if cached is not None:
            Alphas_gpu, Betas_gpu = cached
            gpu_train_mean = 0.0   # timing is meaningless when we didn't train
            gpu_train_std  = 0.0
            print(f"  GPU model training : reused cache "
                  f"({_weights_cache_path(reuse_weights_dir, N, dataset, SIGMA)})")
        else:
            cp.get_default_memory_pool().free_all_blocks()
            gpu_warmup()
            train_times = []
            for _ in range(max(1, n_reps // 2)):
                data = raw.copy()
                _, m, p, _, _ = cpu_generate_grid_cells(
                    data, T_EACH_DIM, N_MODELS, MIN_VAL, MAX_VAL, ETA)
                idx_tmp = LISAIndex(params=p, data_dim=DATA_DIM,
                                    page_size=PAGE_SIZE, sigma=SIGMA)
                _, col_split = idx_tmp.monotone_mappings_and_col_split_idxes(data)
                A, B, t = train_models_gpu(m, col_split, sigma=SIGMA, max_iters=200)
                train_times.append(t)
            Alphas_gpu, Betas_gpu = A, B
            # Sample pool size while the training tensors are still resident.
            gpu_mem_high_water_mb = gpu_memory_mb()
            gpu_train_mean = float(np.mean(train_times))
            gpu_train_std  = float(np.std(train_times))
            if reuse_weights_dir:
                path = _save_cached_weights(reuse_weights_dir, N, dataset, SIGMA,
                                            Alphas_gpu, Betas_gpu)
                print(f"  GPU weights cached  : {path}")
        if cpu_train_mean > 0:
            print(f"  GPU model training : {fmt(gpu_train_mean)}  ± {fmt(gpu_train_std)}"
                  f"  (speedup: {speedup_str(cpu_train_mean, gpu_train_mean)})")
        else:
            print(f"  GPU model training : {fmt(gpu_train_mean)}  ± {fmt(gpu_train_std)}")

        # Correctness vs CPU-trained models (compare training losses, not exact
        # parameters — batched skips the monotone fallback so weights differ).
        if Alphas_ref is not None:
            cpu_loss = _eval_training_loss(maps_cpu, col_split, Alphas_ref, Betas_ref)
            gpu_loss = _eval_training_loss(maps_cpu, col_split, Alphas_gpu, Betas_gpu)
            ratio = gpu_loss / max(cpu_loss, 1e-12)
            tag = "✓" if ratio < 2.0 else "⚠"
            print(f"  Training loss      : CPU={cpu_loss:.2f}  GPU={gpu_loss:.2f}"
                  f"  (GPU/CPU = {ratio:.2f}× {tag})")

    return {
        'N': N,
        # CPU
        'cpu_partition_mean': cpu_p_mean,    'cpu_partition_std': cpu_p_std,
        'cpu_map_sort_mean':  cpu_ms_mean,   'cpu_map_sort_std':  cpu_ms_std,
        'cpu_train_mean':     cpu_train_mean,'cpu_train_std':     cpu_train_std,
        # GPU
        'gpu_h2d_mean':    gpu_h2d_mean,   'gpu_h2d_std':    gpu_h2d_std,
        'gpu_part_mean':   gpu_p_mean,     'gpu_part_std':   gpu_p_std,
        'gpu_ms_mean':     gpu_ms_mean,    'gpu_ms_std':     gpu_ms_std,
        'gpu_d2h_mean':    gpu_d2h_mean,   'gpu_d2h_std':    gpu_d2h_std,
        'gpu_train_mean':  gpu_train_mean, 'gpu_train_std':  gpu_train_std,
        # Speedups
        'speedup_partition': speedup(cpu_p_mean,  gpu_p_mean),
        'speedup_map_sort':  speedup(cpu_ms_mean, gpu_ms_mean),
        'speedup_training':  speedup(cpu_train_mean, gpu_train_mean),
        # Memory (peak)
        'cpu_mem_mb': memory_mb(),
        'gpu_mem_mb': gpu_mem_high_water_mb,
        # Internal artefacts passed to downstream stages
        '_sorted_data': sorted_cpu,
        '_mappings':    maps_cpu,
        '_params':      params_cpu,
        '_Alphas':      Alphas_ref,
        '_Betas':       Betas_ref,
        '_Alphas_gpu':  Alphas_gpu,
        '_Betas_gpu':   Betas_gpu,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Query benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_query_benchmark(build_result, n_reps):
    N = build_result['N']
    print(f"\n{'─'*65}")
    print(f"  QUERY  N={N:,}  Q={N_QUERIES:,}  (reps={n_reps})")
    print(f"{'─'*65}")

    params      = build_result['_params']
    sorted_data = build_result['_sorted_data']
    mappings    = build_result['_mappings']

    idx = LISAIndex(params=params, data_dim=DATA_DIM,
                    page_size=PAGE_SIZE, sigma=SIGMA)
    _, col_split = idx.monotone_mappings_and_col_split_idxes(sorted_data)

    # Prefer CPU-trained weights when available, else fall back to GPU-trained,
    # else random (only happens with --skip-training at every size).
    if build_result['_Alphas'] is not None:
        Alphas, Betas = build_result['_Alphas'], build_result['_Betas']
        weights_source = 'cpu-trained'
    elif build_result.get('_Alphas_gpu') is not None:
        Alphas, Betas = build_result['_Alphas_gpu'], build_result['_Betas_gpu']
        weights_source = 'gpu-trained'
    else:
        Alphas, Betas = make_synthetic_models(N_MODELS, SIGMA, idx.model_split_mappings)
        weights_source = 'synthetic'
    print(f"  (model weights: {weights_source})")

    idx.set_piecewise_models(Alphas, Betas)
    idx.generate_pages(sorted_data, mappings, col_split)

    query_maps = generate_query_mappings(idx, N_QUERIES)

    # ── CPU ──────────────────────────────────────────────────────────────────
    cpu_times = []
    cpu_result = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        cpu_result = idx.predict_shard_ids(query_maps)
        cpu_times.append(time.perf_counter() - t0)

    cpu_mean, cpu_std = np.mean(cpu_times), np.std(cpu_times)
    cpu_tput = N_QUERIES / cpu_mean
    print(f"  CPU predict_shard_ids: {fmt(cpu_mean)}  ± {fmt(cpu_std)}"
          f"  ({cpu_tput:,.0f} lookups/s)")

    # ── GPU ──────────────────────────────────────────────────────────────────
    gpu_total_mean = gpu_compute_mean = -1.0
    gpu_total_std  = gpu_compute_std  = -1.0
    h2d_mean = d2h_mean               = -1.0

    if CUPY_AVAILABLE:
        gpu_warmup()
        gpu_idx = GPULISAIndex(idx)
        gpu_idx.load_to_gpu()

        total_times, compute_times, h2d_times, d2h_times = [], [], [], []
        gpu_result = None
        for _ in range(n_reps):
            t0 = time.perf_counter()
            gpu_result, tg = gpu_idx.predict_shard_ids_gpu(query_maps)
            total_times.append(time.perf_counter() - t0)
            compute_times.append(tg['compute'])
            h2d_times.append(tg['transfer_to_gpu'])
            d2h_times.append(tg['transfer_to_cpu'])

        gpu_total_mean,   gpu_total_std   = np.mean(total_times),   np.std(total_times)
        gpu_compute_mean, gpu_compute_std = np.mean(compute_times), np.std(compute_times)
        h2d_mean = np.mean(h2d_times)
        d2h_mean = np.mean(d2h_times)
        gpu_tput = N_QUERIES / gpu_total_mean

        print(f"  GPU H→D transfer     : {fmt(h2d_mean)}")
        print(f"  GPU compute          : {fmt(gpu_compute_mean)}  ± {fmt(gpu_compute_std)}"
              f"  (speedup over CPU total: {speedup_str(cpu_mean, gpu_compute_mean)})")
        print(f"  GPU D→H transfer     : {fmt(d2h_mean)}")
        print(f"  GPU total            : {fmt(gpu_total_mean)}  ± {fmt(gpu_total_std)}"
              f"  (speedup: {speedup_str(cpu_mean, gpu_total_mean)})")
        print(f"  GPU throughput       : {gpu_tput:,.0f} lookups/s")

        match = np.array_equal(cpu_result, gpu_result)
        print(f"  Correctness          : {'✓ exact match' if match else '✗ MISMATCH'}")
    else:
        print("  GPU: not available")

    return {
        'N': N, 'Q': N_QUERIES,
        'cpu_mean': cpu_mean,         'cpu_std': cpu_std,
        'gpu_total_mean': gpu_total_mean, 'gpu_total_std': gpu_total_std,
        'gpu_compute_mean': gpu_compute_mean, 'gpu_compute_std': gpu_compute_std,
        'gpu_h2d_mean': h2d_mean,     'gpu_d2h_mean': d2h_mean,
        'speedup_total':   speedup(cpu_mean, gpu_total_mean),
        'speedup_compute': speedup(cpu_mean, gpu_compute_mean),
        'weights_source': weights_source,
        '_idx': idx,  # for range_query stage
        '_weights_source': weights_source,  # for downstream stages
    }


# ──────────────────────────────────────────────────────────────────────────────
# Range query benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_range_query_benchmark(build_result, query_result, n_reps,
                              n_queries=1000, half_width_frac=0.005):
    N = build_result['N']
    print(f"\n{'─'*65}")
    print(f"  RANGE QUERY  N={N:,}  Q={n_queries:,}  (reps={n_reps}, "
          f"box ≈ {half_width_frac*100:.1f}% of span)")
    print(f"{'─'*65}")

    idx = query_result['_idx']
    idx.build_flat_layout()
    query_ranges = generate_query_ranges(n_queries, half_width_frac=half_width_frac)

    cpu_times = []
    cpu_counts = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        cpu_counts, _ = idx.range_query(query_ranges)
        cpu_times.append(time.perf_counter() - t0)
    cpu_mean = float(np.mean(cpu_times))
    cpu_std  = float(np.std(cpu_times))
    cpu_tput = n_queries / cpu_mean
    avg_match = float(cpu_counts.mean()) if cpu_counts is not None else 0.0
    print(f"  CPU range_query      : {fmt(cpu_mean)}  ± {fmt(cpu_std)}"
          f"  ({cpu_tput:,.0f} queries/s, ~{avg_match:.1f} pts/query)")

    gpu_total_mean = gpu_compute_mean = -1.0
    gpu_total_std  = gpu_compute_std  = -1.0
    h2d_mean = d2h_mean = -1.0
    if CUPY_AVAILABLE:
        gpu_warmup()
        gpu_idx = GPULISAIndex(idx)
        gpu_idx.load_to_gpu()
        gpu_idx.load_points_to_gpu()

        d = idx.data_dim
        low_maps  = idx.monotone_mappings(query_ranges[:, :d])
        high_maps = idx.monotone_mappings(query_ranges[:, d:])

        total_times, compute_times, h2d_times, d2h_times = [], [], [], []
        gpu_counts = None
        for _ in range(n_reps):
            t0 = time.perf_counter()
            gpu_counts, tg = gpu_idx.range_query_gpu(query_ranges, low_maps, high_maps)
            total_times.append(time.perf_counter() - t0)
            compute_times.append(tg['compute'])
            h2d_times.append(tg['transfer_to_gpu'])
            d2h_times.append(tg['transfer_to_cpu'])

        gpu_total_mean,   gpu_total_std   = np.mean(total_times),   np.std(total_times)
        gpu_compute_mean, gpu_compute_std = np.mean(compute_times), np.std(compute_times)
        h2d_mean = np.mean(h2d_times)
        d2h_mean = np.mean(d2h_times)
        gpu_tput = n_queries / gpu_total_mean
        print(f"  GPU H→D transfer     : {fmt(h2d_mean)}")
        print(f"  GPU compute          : {fmt(gpu_compute_mean)}  ± {fmt(gpu_compute_std)}"
              f"  (speedup over CPU total: {speedup_str(cpu_mean, gpu_compute_mean)})")
        print(f"  GPU D→H transfer     : {fmt(d2h_mean)}")
        print(f"  GPU total            : {fmt(gpu_total_mean)}  ± {fmt(gpu_total_std)}"
              f"  (speedup: {speedup_str(cpu_mean, gpu_total_mean)})")
        print(f"  GPU throughput       : {gpu_tput:,.0f} queries/s")

        match = np.array_equal(cpu_counts, gpu_counts)
        if not match:
            diff = np.abs(cpu_counts - gpu_counts)
            print(f"  Correctness          : ✗ MISMATCH max diff={diff.max()} "
                  f"({(diff > 0).sum()}/{n_queries} queries differ)")
        else:
            print(f"  Correctness          : ✓ exact match on counts")
    else:
        print("  GPU: not available")

    # ── R-tree CPU baseline (libspatialindex) ───────────────────────────────
    # Classical spatial-DB range-query baseline. Bulk-loaded via STR for fast
    # construction; query is one box at a time (the package only exposes a
    # Python-level intersection() call), so this is genuinely the CPU
    # reference number, not a vectorised optimisation.
    rtree_build_time = -1.0
    rtree_query_mean = -1.0
    rtree_query_std = -1.0
    if RTREE_AVAILABLE:
        d = idx.data_dim
        all_pts = idx.all_points
        prop = rtree_index.Property()
        prop.dimension = d

        def _stream():
            for i, pt in enumerate(all_pts):
                yield (i, (pt[0], pt[1], pt[0], pt[1]), None)

        t0 = time.perf_counter()
        rt = rtree_index.Index(_stream(), properties=prop)
        rtree_build_time = time.perf_counter() - t0

        rtree_times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for q in range(n_queries):
                lo = query_ranges[q, :d]
                hi = query_ranges[q, d:]
                _ = sum(1 for _ in rt.intersection((lo[0], lo[1], hi[0], hi[1])))
            rtree_times.append(time.perf_counter() - t0)
        rtree_query_mean = float(np.mean(rtree_times))
        rtree_query_std  = float(np.std(rtree_times))
        print(f"  R-tree build         : {fmt(rtree_build_time)} (one-time, STR bulk-load)")
        print(f"  R-tree query         : {fmt(rtree_query_mean)}  ± {fmt(rtree_query_std)}"
              f"  (LISA-CPU vs R-tree: {speedup_str(rtree_query_mean, cpu_mean)})")
    else:
        print("  R-tree             : not installed (pip install rtree)")

    # ── Quad-tree CPU baseline (pure-Python Pyqtree) ────────────────────────
    quadtree_build_time = -1.0
    quadtree_query_mean = -1.0
    quadtree_query_std  = -1.0
    if PYQTREE_AVAILABLE:
        d = idx.data_dim
        all_pts = idx.all_points

        t0 = time.perf_counter()
        qt = PyqtreeIndex(bbox=(MIN_VAL, MIN_VAL, MAX_VAL, MAX_VAL))
        for i, pt in enumerate(all_pts):
            qt.insert(item=i, bbox=(pt[0], pt[1], pt[0], pt[1]))
        quadtree_build_time = time.perf_counter() - t0

        quadtree_times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for q in range(n_queries):
                lo = query_ranges[q, :d]
                hi = query_ranges[q, d:]
                _ = qt.intersect((lo[0], lo[1], hi[0], hi[1]))
            quadtree_times.append(time.perf_counter() - t0)
        quadtree_query_mean = float(np.mean(quadtree_times))
        quadtree_query_std  = float(np.std(quadtree_times))
        print(f"  Quad-tree build      : {fmt(quadtree_build_time)} (one-time, pure-Python)")
        print(f"  Quad-tree query      : {fmt(quadtree_query_mean)}  ± {fmt(quadtree_query_std)}"
              f"  (LISA-CPU vs Quad-tree: {speedup_str(quadtree_query_mean, cpu_mean)})")
    else:
        print("  Quad-tree          : not installed (pip install pyqtree)")

    return {
        'N': N, 'Q': n_queries,
        'cpu_mean': cpu_mean, 'cpu_std': cpu_std,
        'gpu_total_mean': gpu_total_mean, 'gpu_total_std': gpu_total_std,
        'gpu_compute_mean': gpu_compute_mean, 'gpu_compute_std': gpu_compute_std,
        'gpu_h2d_mean': h2d_mean, 'gpu_d2h_mean': d2h_mean,
        'speedup_total':   speedup(cpu_mean, gpu_total_mean),
        'speedup_compute': speedup(cpu_mean, gpu_compute_mean),
        'avg_match_per_query': avg_match,
        'box_half_width_frac': half_width_frac,
        'rtree_build_time': rtree_build_time,
        'rtree_query_mean': rtree_query_mean,
        'rtree_query_std':  rtree_query_std,
        'speedup_lisa_gpu_vs_rtree': speedup(rtree_query_mean, gpu_total_mean),
        'quadtree_build_time': quadtree_build_time,
        'quadtree_query_mean': quadtree_query_mean,
        'quadtree_query_std':  quadtree_query_std,
        'speedup_lisa_gpu_vs_quadtree': speedup(quadtree_query_mean, gpu_total_mean),
        'weights_source': query_result.get('_weights_source', 'unknown'),
    }


# ──────────────────────────────────────────────────────────────────────────────
# kNN benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_knn_benchmark(build_result, query_result, n_reps,
                      n_queries=500, k=10, n_train_points=300):
    N = build_result['N']
    print(f"\n{'─'*65}")
    print(f"  kNN QUERY  N={N:,}  Q={n_queries:,}  k={k}  (reps={n_reps})")
    print(f"{'─'*65}")

    idx = query_result['_idx']
    idx.build_flat_layout()
    rng = np.random.default_rng(99)
    queries = rng.uniform(MIN_VAL + 100, MAX_VAL - 100,
                          size=(n_queries, DATA_DIM)).astype(np.float64)

    lr, lr_times = train_radius_model_from_data(
        idx.all_points, k=k, n_train_points=n_train_points,
        n_nodes_each_dim=11, min_value=MIN_VAL, max_value=MAX_VAL, alpha=1.0,
        rng=rng,
    )
    lr_train_time = lr_times['total']
    radius_sample_time = lr_times['radius_sample_time']
    lattice_solve_time = lr_times['lattice_solve_time']
    print(f"  Lattice regression train :"
          f"  radius-sample {fmt(radius_sample_time)}"
          f" + lattice-solve {fmt(lattice_solve_time)}"
          f"  (= total {fmt(lr_train_time)})")

    cpu_times = []
    cpu_dists = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        cpu_dists, _ = knn_query_cpu(idx, lr, queries, k=k)
        cpu_times.append(time.perf_counter() - t0)
    cpu_mean = float(np.mean(cpu_times))
    cpu_std  = float(np.std(cpu_times))
    print(f"  CPU kNN query     : {fmt(cpu_mean)}  ± {fmt(cpu_std)}"
          f"  ({n_queries/cpu_mean:,.0f} q/s)")

    gpu_total_mean = gpu_compute_mean = -1.0
    gpu_total_std  = gpu_compute_std  = -1.0
    if CUPY_AVAILABLE:
        gpu_warmup()
        gpu_idx = GPULISAIndex(idx)
        gpu_idx.load_to_gpu()
        gpu_idx.load_points_to_gpu()
        gpu_lr = GPULatticeRegression(lr)
        gpu_lr.load_to_gpu()

        total_times, compute_times = [], []
        gpu_dists = None
        for _ in range(n_reps):
            t0 = time.perf_counter()
            gpu_dists, _, tg = knn_query_gpu(gpu_idx, gpu_lr, queries, k=k)
            total_times.append(time.perf_counter() - t0)
            compute_times.append(tg['compute'])
        gpu_total_mean = float(np.mean(total_times))
        gpu_total_std  = float(np.std(total_times))
        gpu_compute_mean = float(np.mean(compute_times))
        gpu_compute_std  = float(np.std(compute_times))
        print(f"  GPU kNN query     : {fmt(gpu_total_mean)}  ± {fmt(gpu_total_std)}"
              f"  (speedup: {speedup_str(cpu_mean, gpu_total_mean)})")
        print(f"  GPU kNN compute   : {fmt(gpu_compute_mean)}  ± {fmt(gpu_compute_std)}"
              f"  (speedup over CPU: {speedup_str(cpu_mean, gpu_compute_mean)})")

        # Correctness: distances should match to ~1e-3 relative.
        finite = np.isfinite(cpu_dists) & np.isfinite(gpu_dists)
        if finite.any():
            rel = np.abs(cpu_dists[finite] - gpu_dists[finite]) / (cpu_dists[finite] + 1e-9)
            print(f"  Correctness       : max rel dist diff = {rel.max():.2e}, "
                  f"agreement = {(rel < 1e-3).mean()*100:.1f}%")
    else:
        print("  GPU: not available")

    # ── GPU brute-force kNN baseline (FAISS or PyTorch fallback) ────────────
    # FAISS's pre-built wheels lack Blackwell kernels, so on those GPUs we use
    # torch.cdist + topk for the same exact-L2 brute-force comparison.
    faiss_mean = faiss_std = -1.0
    faiss_recall_at_k = -1.0
    faiss_impl = 'unavailable'
    if CUPY_AVAILABLE:
        from baselines import _faiss_supports_current_gpu
        use_faiss = FAISS_AVAILABLE and _faiss_supports_current_gpu()

        d = idx.data_dim
        data_f32 = idx.all_points.astype(np.float32)
        queries_f32 = queries.astype(np.float32)
        topk_dists_brute = None  # exact L2 distances from the brute-force path

        if use_faiss:
            import faiss
            res = faiss.StandardGpuResources()
            flat = faiss.IndexFlatL2(d)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, flat)
            gpu_index.add(data_f32)
            _, _ = gpu_index.search(queries_f32[:8], k)  # warm-up

            faiss_times = []
            D_faiss = None
            for _ in range(n_reps):
                t0 = time.perf_counter()
                D_faiss, _ = gpu_index.search(queries_f32, k)
                faiss_times.append(time.perf_counter() - t0)
            faiss_mean = float(np.mean(faiss_times))
            faiss_std  = float(np.std(faiss_times))
            faiss_impl = 'faiss'
            # FAISS IndexFlatL2 returns squared L2; convert for recall compare.
            topk_dists_brute = np.sqrt(D_faiss).astype(np.float64)
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    data_t = torch.from_numpy(data_f32).cuda()
                    queries_t = torch.from_numpy(queries_f32).cuda()
                    # Chunk queries to keep the (chunk, N) distance tensor < ~4 GB
                    # at fp32 — relevant at N=10M where (Q, N) is too big at once.
                    bytes_per_query_row = idx.all_points.shape[0] * 4
                    q_chunk = max(1, min(queries_t.shape[0],
                                         int(4_000_000_000 / max(bytes_per_query_row, 1))))
                    # Warm-up
                    _ = torch.topk(torch.cdist(queries_t[:min(8, queries_t.shape[0])],
                                               data_t),
                                   k, largest=False, dim=1)
                    torch.cuda.synchronize()

                    torch_times = []
                    topk_torch_all = None
                    for _ in range(n_reps):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        topk_pieces = []
                        for qs in range(0, queries_t.shape[0], q_chunk):
                            qe = min(qs + q_chunk, queries_t.shape[0])
                            dists = torch.cdist(queries_t[qs:qe], data_t)
                            top = torch.topk(dists, k, largest=False, dim=1).values
                            topk_pieces.append(top)
                        topk_torch_all = torch.cat(topk_pieces, dim=0)
                        torch.cuda.synchronize()
                        torch_times.append(time.perf_counter() - t0)
                    faiss_mean = float(np.mean(torch_times))
                    faiss_std  = float(np.std(torch_times))
                    faiss_impl = 'torch'
                    topk_dists_brute = topk_torch_all.cpu().numpy().astype(np.float64)
            except ImportError:
                pass

        if faiss_impl != 'unavailable':
            print(f"  GPU brute-force kNN ({faiss_impl}): {fmt(faiss_mean)}  ± {fmt(faiss_std)}"
                  f"  (LISA-GPU vs brute force: {speedup_str(faiss_mean, gpu_total_mean)})")

            # Fraction of LISA's per-query, per-rank distances that match the exact result.
            if gpu_dists is not None and topk_dists_brute is not None:
                agree = np.isclose(gpu_dists, topk_dists_brute,
                                   rtol=1e-3, atol=1e-3)
                faiss_recall_at_k = float(agree.mean())
                print(f"  LISA recall vs exact  : {faiss_recall_at_k*100:.2f}% "
                      f"of (query, rank) pairs match")
        else:
            print("  GPU brute-force kNN: PyTorch not available either")
    elif not FAISS_AVAILABLE:
        print("  FAISS              : not installed (pip install faiss-gpu-cu12)")

    # ── scipy cKDTree CPU baseline ──────────────────────────────────────────
    # Classical CPU spatial-tree benchmark. Build time is reported separately
    # because it's one-time; query latency is what compares to LISA's CPU kNN.
    ckdtree_build_time = -1.0
    ckdtree_query_mean = -1.0
    ckdtree_query_std = -1.0
    if CKDTREE_AVAILABLE:
        t0 = time.perf_counter()
        kd = cKDTree(idx.all_points)
        ckdtree_build_time = time.perf_counter() - t0
        ckdtree_times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            kd.query(queries, k=k)
            ckdtree_times.append(time.perf_counter() - t0)
        ckdtree_query_mean = float(np.mean(ckdtree_times))
        ckdtree_query_std  = float(np.std(ckdtree_times))
        print(f"  cKDTree build         : {fmt(ckdtree_build_time)} (one-time)")
        print(f"  cKDTree query         : {fmt(ckdtree_query_mean)}  ± {fmt(ckdtree_query_std)}"
              f"  (LISA-CPU vs cKDTree: {speedup_str(ckdtree_query_mean, cpu_mean)})")
    else:
        print("  cKDTree            : scipy not available")

    return {
        'N': N, 'Q': n_queries, 'k': k,
        'lr_train_time': lr_train_time,
        'radius_sample_time': radius_sample_time,
        'lattice_solve_time': lattice_solve_time,
        'cpu_mean': cpu_mean, 'cpu_std': cpu_std,
        'gpu_total_mean': gpu_total_mean, 'gpu_total_std': gpu_total_std,
        'gpu_compute_mean': gpu_compute_mean, 'gpu_compute_std': gpu_compute_std,
        'speedup_total':   speedup(cpu_mean, gpu_total_mean),
        'speedup_compute': speedup(cpu_mean, gpu_compute_mean),
        'faiss_mean': faiss_mean, 'faiss_std': faiss_std,
        'speedup_lisa_vs_faiss': speedup(faiss_mean, gpu_total_mean),
        'lisa_recall_vs_faiss': faiss_recall_at_k,   # legacy name, kept for back-compat
        'lisa_recall_at_k': faiss_recall_at_k,       # canonical name (works for either impl)
        'brute_force_impl': faiss_impl,
        'ckdtree_build_time': ckdtree_build_time,
        'ckdtree_query_mean': ckdtree_query_mean,
        'ckdtree_query_std':  ckdtree_query_std,
        'speedup_lisa_gpu_vs_ckdtree': speedup(ckdtree_query_mean, gpu_total_mean),
        'weights_source': query_result.get('_weights_source', 'unknown'),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MLP local-model benchmark (exploratory)
# ──────────────────────────────────────────────────────────────────────────────

def run_mlp_benchmark(build_result, hidden=16, max_iters=500, lr=1e-2,
                      dataset='uniform'):
    N = build_result['N']
    print(f"\n{'─'*65}")
    print(f"  MLP LOCAL MODEL  N={N:,}  hidden={hidden}  iters={max_iters}")
    print(f"{'─'*65}")

    if not TORCH_AVAILABLE:
        print("  PyTorch not available — skipping MLP benchmark")
        return {'N': N, 'hidden': hidden, 'max_iters': max_iters,
                'piecewise_train_time': -1.0, 'piecewise_mean_loss': -1.0,
                'mlp_train_time': -1.0, 'mlp_mean_loss': -1.0,
                'mlp_vs_piecewise_loss_ratio': float('nan')}

    # Prefer the mappings + column split the build stage already computed —
    # we want the MLP to be trained on identical input to the piecewise model.
    if build_result.get('_mappings') is not None:
        m = build_result['_mappings']
        params = build_result['_params']
        idx = LISAIndex(params=params, data_dim=DATA_DIM,
                        page_size=PAGE_SIZE, sigma=SIGMA)
        _, col_split = idx.monotone_mappings_and_col_split_idxes(
            build_result['_sorted_data']
        )
    else:
        raw = generate_data(N, dataset=dataset)
        sorted_data, m, p, _, _ = cpu_generate_grid_cells(
            raw.copy(), T_EACH_DIM, N_MODELS, MIN_VAL, MAX_VAL, ETA
        )
        idx = LISAIndex(params=p, data_dim=DATA_DIM, page_size=PAGE_SIZE, sigma=SIGMA)
        _, col_split = idx.monotone_mappings_and_col_split_idxes(sorted_data)

    # Piecewise-linear baseline: prefer the models the build stage already
    # trained (free); otherwise train fresh on whichever backend is available
    # (CuPy if installed, NumPy fallback only as last resort).
    if build_result.get('_Alphas_gpu') is not None:
        A_pwl = build_result['_Alphas_gpu']
        B_pwl = build_result['_Betas_gpu']
        t_pwl = build_result.get('gpu_train_mean', -1.0)
    elif build_result.get('_Alphas') is not None:
        A_pwl = build_result['_Alphas']
        B_pwl = build_result['_Betas']
        t_pwl = build_result.get('cpu_train_mean', -1.0)
    else:
        A_pwl, B_pwl, t_pwl = train_models_gpu(m, col_split, sigma=SIGMA,
                                               max_iters=200)
    pwl_loss = _eval_training_loss(m, col_split, A_pwl, B_pwl) / max(N_MODELS, 1)

    _meta, t_mlp, mlp_losses = train_mlp_models(
        m, col_split, hidden=hidden, max_iters=max_iters, lr=lr
    )
    mlp_loss = float(mlp_losses.mean())

    print(f"  Piecewise (σ={SIGMA}) : train time {fmt(t_pwl)}  mean loss {pwl_loss:,.2f}")
    print(f"  MonotonicMLP (H={hidden}, iters={max_iters}) :"
          f" train time {fmt(t_mlp)}  mean loss {mlp_loss:,.2f}")
    rel = mlp_loss / max(pwl_loss, 1e-9)
    print(f"  MLP loss / piecewise loss : {rel:.2f}x  "
          f"({'better' if rel < 1 else 'worse'})")

    return {
        'N': N, 'hidden': hidden, 'max_iters': max_iters,
        'piecewise_train_time': t_pwl,
        'piecewise_mean_loss': pwl_loss,
        'mlp_train_time': t_mlp,
        'mlp_mean_loss': mlp_loss,
        'mlp_vs_piecewise_loss_ratio': rel,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mixed-precision benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_mixed_precision_benchmark(build_result, query_result, n_reps):
    N = build_result['N']
    print(f"\n{'─'*65}")
    print(f"  MIXED PRECISION  N={N:,}  Q={N_QUERIES:,}  (reps={n_reps})")
    print(f"{'─'*65}")

    if not CUPY_AVAILABLE:
        print("  GPU: not available — skipping mixed precision")
        return {'N': N, 'Q': N_QUERIES,
                'gpu_fp64_mean': -1.0, 'gpu_fp16_mean': -1.0,
                'speedup_fp16_vs_fp64': float('nan'),
                'fp16_shard_mismatch_rate': float('nan'),
                'fp16_max_abs_shard_diff': float('nan')}

    idx = query_result['_idx']
    query_maps = generate_query_mappings(idx, N_QUERIES, seed=11)

    gpu_warmup()
    gpu_idx = GPULISAIndex(idx)
    gpu_idx.load_to_gpu()

    # fp64 reference (single rep to get a baseline)
    fp64_ref, _ = gpu_idx.predict_shard_ids_gpu(query_maps)

    fp64_times, fp16_times = [], []
    fp16_result = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _, _ = gpu_idx.predict_shard_ids_gpu(query_maps)
        fp64_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fp16_result, _ = gpu_idx.predict_shard_ids_gpu_fp16(query_maps)
        fp16_times.append(time.perf_counter() - t0)

    fp64_mean = float(np.mean(fp64_times))
    fp16_mean = float(np.mean(fp16_times))
    mismatch = float(np.mean(fp16_result != fp64_ref))
    abs_diff = float(np.max(np.abs(fp16_result.astype(np.int64) - fp64_ref.astype(np.int64))))

    print(f"  GPU fp64 predict     : {fmt(fp64_mean)}")
    print(f"  GPU fp16 predict     : {fmt(fp16_mean)}"
          f"  (speedup vs fp64: {speedup_str(fp64_mean, fp16_mean)})")
    print(f"  Shard ID mismatch    : {mismatch*100:.2f}%  (max |diff| = {abs_diff})")

    return {
        'N': N, 'Q': N_QUERIES,
        'gpu_fp64_mean': fp64_mean, 'gpu_fp16_mean': fp16_mean,
        'speedup_fp16_vs_fp64': speedup(fp64_mean, fp16_mean),
        'fp16_shard_mismatch_rate': mismatch,
        'fp16_max_abs_shard_diff': abs_diff,
        'weights_source': query_result.get('_weights_source', 'unknown'),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic ops benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_dynamic_ops_benchmark(build_result, query_result, batch_sizes=(1000, 10000, 100000)):
    N = build_result['N']
    print(f"\n{'─'*65}")
    print(f"  DYNAMIC OPS  N={N:,}  batch_sizes={list(batch_sizes)}")
    print(f"{'─'*65}")

    idx = query_result['_idx']
    idx.build_flat_layout()
    rng = np.random.default_rng(7)

    results = []
    for B in batch_sizes:
        new_pts = rng.uniform(MIN_VAL, MAX_VAL, size=(B, DATA_DIM)).astype(np.float64)

        # CPU insert
        idx_cpu = _clone_layout(idx)
        t0 = time.perf_counter()
        insert_batch_cpu(idx_cpu, new_pts)
        cpu_ins = time.perf_counter() - t0

        # CPU delete
        t0 = time.perf_counter()
        delete_batch_cpu(idx_cpu, new_pts)
        cpu_del = time.perf_counter() - t0

        gpu_ins = gpu_del = -1.0
        if CUPY_AVAILABLE:
            gpu_warmup()
            gpu_idx = GPULISAIndex(idx)
            gpu_idx.load_to_gpu()
            gpu_idx.load_points_to_gpu()

            t0 = time.perf_counter()
            insert_batch_gpu(gpu_idx, new_pts)
            gpu_ins = time.perf_counter() - t0
            t0 = time.perf_counter()
            delete_batch_gpu(gpu_idx, new_pts)
            gpu_del = time.perf_counter() - t0

        cpu_ins_tput = B / cpu_ins
        cpu_del_tput = B / cpu_del
        line = (f"  B={B:>8,}  insert: CPU {fmt(cpu_ins)} ({cpu_ins_tput:,.0f} pts/s)")
        if gpu_ins > 0:
            line += f"  GPU {fmt(gpu_ins)} ({B/gpu_ins:,.0f} pts/s) {speedup_str(cpu_ins, gpu_ins)}"
        print(line)
        line = (f"  B={B:>8,}  delete: CPU {fmt(cpu_del)} ({cpu_del_tput:,.0f} pts/s)")
        if gpu_del > 0:
            line += f"  GPU {fmt(gpu_del)} ({B/gpu_del:,.0f} pts/s) {speedup_str(cpu_del, gpu_del)}"
        print(line)

        results.append({
            'N': N, 'batch_size': B,
            'cpu_insert_s': cpu_ins, 'cpu_delete_s': cpu_del,
            'gpu_insert_s': gpu_ins, 'gpu_delete_s': gpu_del,
            'speedup_insert': speedup(cpu_ins, gpu_ins),
            'speedup_delete': speedup(cpu_del, gpu_del),
            'weights_source': query_result.get('_weights_source', 'unknown'),
        })
    return results


def _clone_layout(idx):
    """Shallow clone of an idx with its flat layout copied (so insert is non-destructive)."""
    import copy
    clone = copy.copy(idx)
    clone.all_points = idx.all_points.copy()
    clone.shard_point_starts = idx.shard_point_starts.copy()
    clone.shard_point_ends = idx.shard_point_ends.copy()
    return clone


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(build_rows, query_rows):
    print(f"\n{'═'*80}")
    print("  RESULTS SUMMARY")
    print(f"{'═'*80}")

    hdr = f"  {'N':>12}  {'CPU part':>10}  {'CPU m+s':>10}  {'GPU part':>10}  {'GPU m+s':>10}  {'Spd part':>9}  {'Spd m+s':>9}"
    sep = "  " + "─"*12 + "  " + ("─"*10 + "  ")*5 + "─"*9
    print(f"\n  BUILD\n{hdr}\n{sep}")
    for r in build_rows:
        print(f"  {r['N']:>12,}  "
              f"{fmt(r['cpu_partition_mean']):>10}  "
              f"{fmt(r['cpu_map_sort_mean']):>10}  "
              f"{fmt(r['gpu_part_mean']):>10}  "
              f"{fmt(r['gpu_ms_mean']):>10}  "
              f"{speedup_str(r['cpu_partition_mean'], r['gpu_part_mean']):>9}  "
              f"{speedup_str(r['cpu_map_sort_mean'],  r['gpu_ms_mean']):>9}")

    hdr2 = f"  {'N':>12}  {'Q':>8}  {'CPU total':>12}  {'GPU total':>12}  {'GPU compute':>12}  {'Spd total':>10}"
    sep2 = "  " + "─"*12 + "  " + "─"*8 + "  " + ("─"*12 + "  ")*2 + "─"*12 + "  " + "─"*10
    print(f"\n  QUERY (predict_shard_ids)\n{hdr2}\n{sep2}")
    for r in query_rows:
        print(f"  {r['N']:>12,}  {r['Q']:>8,}  "
              f"{fmt(r['cpu_mean']):>12}  "
              f"{fmt(r['gpu_total_mean']):>12}  "
              f"{fmt(r['gpu_compute_mean']):>12}  "
              f"{speedup_str(r['cpu_mean'], r['gpu_total_mean']):>10}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Paper report
# ──────────────────────────────────────────────────────────────────────────────

def generate_paper_report(sysinfo, build_rows, query_rows, output_dir,
                          range_rows=None, knn_rows=None, dyn_rows=None,
                          mp_rows=None, mlp_rows=None):
    """
    Write a structured text report that can be pasted directly into Claude
    to generate a research paper section on these benchmark results.
    """
    lines = []
    w = lines.append

    w("=" * 72)
    w("GPU-ACCELERATED LISA: BENCHMARK REPORT")
    w(f"Generated: {sysinfo['date']}")
    w("=" * 72)
    w("")
    w("─" * 72)
    w("SYSTEM INFORMATION")
    w("─" * 72)
    w(f"Date        : {sysinfo['date']}")
    w(f"OS          : {sysinfo['os']}")
    w(f"Python      : {sysinfo['python']}")
    w(f"NumPy       : {sysinfo['numpy']}")
    w(f"CuPy        : {sysinfo.get('cupy', 'not installed')}")
    w(f"CUDA        : {sysinfo['cuda']}")
    w(f"GPU driver  : {sysinfo['driver']}")
    w(f"CPU         : {sysinfo['cpu']}")
    w(f"GPU         : {sysinfo['gpu_name']}")
    if sysinfo['gpu_vram_gb'] > 0:
        w(f"GPU VRAM    : {sysinfo['gpu_vram_gb']} GB total, "
          f"{sysinfo['gpu_free_gb']} GB free")
    w("")
    w("─" * 72)
    w("BENCHMARK CONFIGURATION")
    w("─" * 72)
    w(f"Data distribution    : 2D uniform random, coordinates in [0, {MAX_VAL:.0f}]²")
    w(f"Dataset sizes tested : {[r['N'] for r in build_rows]}")
    w(f"Range queries (Q)    : {N_QUERIES:,}")
    w(f"Repetitions          : {N_REPS} (mean ± std reported)")
    w("")
    w("LISA Index Parameters:")
    w(f"  T_each_dim (grid partitions per dimension) : {T_EACH_DIM}")
    w(f"    Note: paper default is 240; using {T_EACH_DIM} for benchmark speed")
    w(f"  n_piecewise_models                         : {N_MODELS}")
    w(f"    Note: paper default is 1024; using {N_MODELS} for benchmark speed")
    w(f"  sigma (ReLU basis functions per model)     : {SIGMA}")
    w(f"    Note: paper default is 100; using {SIGMA} for benchmark speed")
    w(f"  page_size                                  : {PAGE_SIZE}")
    w(f"  eta (cell measure weight)                  : {ETA}")
    w("")
    w("─" * 72)
    w("GPU-ACCELERATED STAGES")
    w("─" * 72)
    w("Build:        partition (cp.argsort), mapping+sort (one global sort),")
    w("              piecewise-linear training (batched Newton over all columns).")
    w("Query:        predict_shard_ids (batched ReLU + dot product),")
    w("              range_query end-to-end (corner mapping, candidate scan,")
    w("              broadcast filter, scatter-add).")
    w("kNN:          lattice-regression inference + GPU range_query with")
    w("              radius doubling; lattice training stays on CPU.")
    w("Dynamic ops:  batched insert/delete via stable sort + bincount over")
    w("              the flat point layout.")
    w("")
    w("─" * 72)
    w("BUILD BENCHMARK RESULTS (all times in seconds)")
    w("─" * 72)
    w("")
    w(f"{'N':>12}  {'Stage':<20}  {'CPU mean':>12}  {'CPU std':>10}  "
      f"{'GPU mean':>12}  {'GPU std':>10}  {'Speedup':>8}")
    w(f"{'─'*12}  {'─'*20}  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")
    for r in build_rows:
        def fv(v): return f"{v:.6f}" if v >= 0 else "N/A"
        def sp(s): return f"{s:.2f}x" if not np.isnan(s) else "N/A"
        N = r['N']
        w(f"{N:>12,}  {'partition':<20}  "
          f"{fv(r['cpu_partition_mean']):>12}  {fv(r['cpu_partition_std']):>10}  "
          f"{fv(r['gpu_part_mean']):>12}  {fv(r['gpu_part_std']):>10}  "
          f"{sp(r['speedup_partition']):>8}")
        w(f"{N:>12,}  {'mapping+sort':<20}  "
          f"{fv(r['cpu_map_sort_mean']):>12}  {fv(r['cpu_map_sort_std']):>10}  "
          f"{fv(r['gpu_ms_mean']):>12}  {fv(r['gpu_ms_std']):>10}  "
          f"{sp(r['speedup_map_sort']):>8}")
        if r['cpu_train_mean'] >= 0 or r.get('gpu_train_mean', -1) >= 0:
            gpu_mean = r.get('gpu_train_mean', -1)
            gpu_std  = r.get('gpu_train_std', -1)
            spd = r.get('speedup_training', float('nan'))
            w(f"{N:>12,}  {'model training':<20}  "
              f"{fv(r['cpu_train_mean']):>12}  {fv(r['cpu_train_std']):>10}  "
              f"{fv(gpu_mean):>12}  {fv(gpu_std):>10}  {sp(spd):>8}")
        # Data transfer breakdown
        if r['gpu_h2d_mean'] >= 0:
            w(f"{N:>12,}  {'GPU H→D transfer':<20}  "
              f"{'N/A':>12}  {'N/A':>10}  "
              f"{fv(r['gpu_h2d_mean']):>12}  {fv(r['gpu_h2d_std']):>10}  {'N/A':>8}")
            w(f"{N:>12,}  {'GPU D→H transfer':<20}  "
              f"{'N/A':>12}  {'N/A':>10}  "
              f"{fv(r['gpu_d2h_mean']):>12}  {fv(r['gpu_d2h_std']):>10}  {'N/A':>8}")
    w("")
    w("─" * 72)
    w(f"QUERY BENCHMARK RESULTS — predict_shard_ids (Q={N_QUERIES:,})")
    w("─" * 72)
    w("")
    w(f"{'N':>12}  {'Measurement':<25}  {'Mean (s)':>12}  {'Std (s)':>10}  {'Speedup':>8}")
    w(f"{'─'*12}  {'─'*25}  {'─'*12}  {'─'*10}  {'─'*8}")
    for r in query_rows:
        def fv(v): return f"{v:.6f}" if v >= 0 else "N/A"
        def sp(s): return f"{s:.2f}x" if not np.isnan(s) else "N/A"
        N = r['N']
        w(f"{N:>12,}  {'CPU total':<25}  {fv(r['cpu_mean']):>12}  {fv(r['cpu_std']):>10}  {'baseline':>8}")
        if r['gpu_total_mean'] >= 0:
            w(f"{N:>12,}  {'GPU H→D transfer':<25}  {fv(r['gpu_h2d_mean']):>12}  {'N/A':>10}  {'N/A':>8}")
            w(f"{N:>12,}  {'GPU compute only':<25}  {fv(r['gpu_compute_mean']):>12}  {fv(r['gpu_compute_std']):>10}  {sp(r['speedup_compute']):>8}")
            w(f"{N:>12,}  {'GPU D→H transfer':<25}  {fv(r['gpu_d2h_mean']):>12}  {'N/A':>10}  {'N/A':>8}")
            w(f"{N:>12,}  {'GPU total (inc. transfer)':<25}  {fv(r['gpu_total_mean']):>12}  {fv(r['gpu_total_std']):>10}  {sp(r['speedup_total']):>8}")
            cpu_tput = N_QUERIES / r['cpu_mean']
            gpu_tput = N_QUERIES / r['gpu_total_mean']
            w(f"{N:>12,}  {'CPU throughput (Q/s)':<25}  {cpu_tput:>12,.0f}  {'N/A':>10}  {'N/A':>8}")
            w(f"{N:>12,}  {'GPU throughput (Q/s)':<25}  {gpu_tput:>12,.0f}  {'N/A':>10}  {'N/A':>8}")
    w("")
    w("─" * 72)
    w("ANALYSIS NOTES FOR PAPER")
    w("─" * 72)
    w("")
    w("The following observations are drawn from the raw numbers above.")
    w("Use these as a starting point — verify against the actual values.")
    w("")

    # Auto-generate some observations
    gpu_rows = [r for r in build_rows if r['gpu_ms_mean'] >= 0]
    if gpu_rows:
        best_ms = max(gpu_rows, key=lambda r: r['speedup_map_sort']
                      if not np.isnan(r['speedup_map_sort']) else 0)
        w(f"1. Mapping+sort speedup peaks at N={best_ms['N']:,} with "
          f"{best_ms['speedup_map_sort']:.1f}x speedup. This stage benefits most "
          f"from GPU because the single global argsort over N points exhibits "
          f"high parallelism.")
        w("")
        best_p = max(gpu_rows, key=lambda r: r['speedup_partition']
                     if not np.isnan(r['speedup_partition']) else 0)
        w(f"2. Partition speedup peaks at N={best_p['N']:,} with "
          f"{best_p['speedup_partition']:.1f}x. For 2D data, partition is a single "
          f"sort of N elements — GPU advantage grows with N.")
        w("")
        # Transfer overhead
        r = gpu_rows[-1]
        if r['gpu_ms_mean'] > 0 and r['gpu_h2d_mean'] > 0:
            transfer_pct = (r['gpu_h2d_mean'] + r['gpu_d2h_mean']) / (
                r['gpu_h2d_mean'] + r['gpu_part_mean'] +
                r['gpu_ms_mean'] + r['gpu_d2h_mean']
            ) * 100
            w(f"3. At N={r['N']:,}, data transfer (H→D + D→H) accounts for "
              f"~{transfer_pct:.0f}% of total GPU build time. For small N, "
              f"transfer overhead can negate compute savings.")
        w("")

    q_gpu = [r for r in query_rows if r['gpu_total_mean'] >= 0]
    if q_gpu:
        r = q_gpu[-1]
        w(f"4. Query speedup (GPU compute only) at N={r['N']:,}: "
          f"{r['speedup_compute']:.1f}x. Total speedup including transfer: "
          f"{r['speedup_total']:.1f}x. The gap shows that for small Q={N_QUERIES:,}, "
          f"transfer dominates; larger batches would show higher total speedup.")
        w("")
        # Cite throughput from the largest-N row, not the first. The first
        # row is N=100K where JIT compilation and transfer overhead can
        # dominate, so the largest-N row is the more representative number.
        r0 = q_gpu[-1]
        w(f"5. At N={r0['N']:,}, CPU query throughput: "
          f"{N_QUERIES/r0['cpu_mean']:,.0f} lookups/s. "
          f"GPU throughput (total): {N_QUERIES/r0['gpu_total_mean']:,.0f} lookups/s.")
        w("")

    if any(r.get('gpu_train_mean', -1) > 0 for r in build_rows):
        best_train = max(
            (r for r in build_rows
             if r.get('speedup_training', float('nan')) == r.get('speedup_training', float('nan'))
             and r.get('speedup_training', 0) > 0),
            key=lambda r: r.get('speedup_training', 0),
            default=None)
        if best_train is not None:
            w(f"6. Build-phase model training is GPU-accelerated via the batched "
              f"piecewise-linear trainer. Peak measured training speedup: "
              f"{best_train['speedup_training']:.1f}× at N={best_train['N']:,}. "
              f"Above N=1M, CPU training is skipped (TRAIN_MODELS_MAX_SIZE), so "
              f"only the GPU absolute time is reported.")
            w("")
    if range_rows:
        w("─" * 72)
        w("RANGE QUERY RESULTS — LISA vs classical baselines (R-tree, Quad-tree)")
        w("─" * 72)
        w("")
        w(f"{'N':>12}  {'Q':>5}  {'CPU (s)':>10}  {'LISA-GPU (s)':>13}  "
          f"{'GPU spd':>8}  {'R-tree (s)':>11}  {'vs R-tree':>10}  "
          f"{'Quad (s)':>10}  {'vs Quad':>8}  {'avg/q':>6}")
        w(f"{'─'*12}  {'─'*5}  {'─'*10}  {'─'*13}  {'─'*8}  {'─'*11}  "
          f"{'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}")
        for r in range_rows:
            def fv(v): return f"{v:.6f}" if v >= 0 else "N/A"
            def sps(s): return "N/A" if (s != s) else f"{s:.2f}x"
            w(f"{r['N']:>12,}  {r['Q']:>5,}  {fv(r['cpu_mean']):>10}  "
              f"{fv(r['gpu_total_mean']):>13}  "
              f"{sps(r.get('speedup_total', float('nan'))):>8}  "
              f"{fv(r.get('rtree_query_mean', -1)):>11}  "
              f"{sps(r.get('speedup_lisa_gpu_vs_rtree', float('nan'))):>10}  "
              f"{fv(r.get('quadtree_query_mean', -1)):>10}  "
              f"{sps(r.get('speedup_lisa_gpu_vs_quadtree', float('nan'))):>8}  "
              f"{r.get('avg_match_per_query', 0):>6.2f}")
        w("")
        w("Box size: half-width as fraction of the [0, MAX]^d span. R-tree and "
          "Quad-tree are the classical comparison structures used throughout "
          "the learned-index literature. R-tree = libspatialindex via the `rtree` "
          "package (STR bulk-loaded). Quad-tree = pure-Python Pyqtree. Speedup "
          "columns are baseline_time / LISA-GPU_time; >1 means LISA-GPU is faster.")
        w("")
    if knn_rows:
        w("─" * 72)
        w("kNN QUERY RESULTS — LISA vs FAISS (GPU brute force) vs cKDTree (CPU)")
        w("─" * 72)
        w("")
        w(f"{'N':>12}  {'k':>4}  {'CPU (s)':>11}  {'LISA-GPU (s)':>13}  "
          f"{'FAISS (s)':>11}  {'cKDTree (s)':>12}  {'vs FAISS':>9}  "
          f"{'vs cKDTree':>11}  {'recall':>7}")
        w(f"{'─'*12}  {'─'*4}  {'─'*11}  {'─'*13}  {'─'*11}  {'─'*12}  "
          f"{'─'*9}  {'─'*11}  {'─'*7}")
        for r in knn_rows:
            def fv(v, w_=11): return f"{v:.6f}" if v >= 0 else "N/A"
            def sps(s): return "N/A" if (s != s) else f"{s:.2f}x"
            recall = r.get('lisa_recall_vs_faiss', -1)
            recall_s = "N/A" if recall < 0 else f"{recall*100:.1f}%"
            w(f"{r['N']:>12,}  {r['k']:>4}  "
              f"{fv(r['cpu_mean']):>11}  {fv(r['gpu_total_mean']):>13}  "
              f"{fv(r.get('faiss_mean', -1)):>11}  "
              f"{fv(r.get('ckdtree_query_mean', -1)):>12}  "
              f"{sps(r.get('speedup_lisa_vs_faiss', float('nan'))):>9}  "
              f"{sps(r.get('speedup_lisa_gpu_vs_ckdtree', float('nan'))):>11}  "
              f"{recall_s:>7}")
        w("")
        w("Baselines: FAISS = Meta's GPU brute-force IndexFlatL2 (exact k-NN). "
          "cKDTree = scipy.spatial CPU KD-tree (bulk-loaded). "
          "Speedup columns are baseline_time / LISA-GPU time (>1 means LISA is faster). "
          "'recall' = fraction of LISA's (query, rank) distances that match FAISS exact distances.")
        w("")
    if mlp_rows:
        w("─" * 72)
        w("MONOTONIC MLP LOCAL MODEL — exploratory")
        w("─" * 72)
        w("")
        w(f"{'N':>12}  {'hidden':>6}  {'iters':>6}  {'pwl train (s)':>14}  "
          f"{'pwl loss':>12}  {'mlp train (s)':>14}  {'mlp loss':>12}  "
          f"{'mlp/pwl':>8}")
        w(f"{'─'*12}  {'─'*6}  {'─'*6}  {'─'*14}  {'─'*12}  {'─'*14}  "
          f"{'─'*12}  {'─'*8}")
        for r in mlp_rows:
            def fv(v): return f"{v:.6f}" if v >= 0 else "N/A"
            rel = r.get('mlp_vs_piecewise_loss_ratio', float('nan'))
            rel_s = "N/A" if (rel != rel) else f"{rel:.2f}x"
            w(f"{r['N']:>12,}  {r['hidden']:>6}  {r['max_iters']:>6}  "
              f"{fv(r.get('piecewise_train_time', -1)):>14}  "
              f"{r.get('piecewise_mean_loss', 0):>12,.2f}  "
              f"{fv(r.get('mlp_train_time', -1)):>14}  "
              f"{r.get('mlp_mean_loss', 0):>12,.2f}  {rel_s:>8}")
        w("")
        w("Architecture: 1 → H → H → 1 with weights = exp(raw) (always positive) "
          "and ReLU activations — proves monotone by construction. Training: "
          "batched Adam across all n_models columns simultaneously. Loss = sum "
          "of squared position residuals per column. The MLP is NOT yet plugged "
          "into predict_shard_ids; this is an exploratory comparison to decide "
          "whether full integration is worth doing.")
        w("")
    if mp_rows:
        w("─" * 72)
        w("MIXED-PRECISION RESULTS — fp16 predict_shard_ids vs fp64 reference")
        w("─" * 72)
        w("")
        w(f"{'N':>12}  {'Q':>8}  {'fp64 (s)':>12}  {'fp16 (s)':>12}  "
          f"{'Speedup':>8}  {'mismatch %':>11}  {'max |Δ shard|':>14}")
        w(f"{'─'*12}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*11}  {'─'*14}")
        for r in mp_rows:
            def fv(v): return f"{v:.6f}" if v >= 0 else "N/A"
            sp = r.get('speedup_fp16_vs_fp64', float('nan'))
            sp_s = "N/A" if (sp != sp) else f"{sp:.2f}x"
            mm = r.get('fp16_shard_mismatch_rate', float('nan'))
            mm_s = "N/A" if (mm != mm) else f"{mm*100:.2f}"
            md = r.get('fp16_max_abs_shard_diff', float('nan'))
            md_s = "N/A" if (md != md) else f"{int(md)}"
            w(f"{r['N']:>12,}  {r['Q']:>8,}  {fv(r['gpu_fp64_mean']):>12}  "
              f"{fv(r['gpu_fp16_mean']):>12}  {sp_s:>8}  {mm_s:>11}  {md_s:>14}")
        w("")
        w("Mixed precision casts Alphas/Betas/mappings to fp16 for the ReLU + dot "
          "product (the bulk of FLOPs in predict_shard_ids), then accumulates and "
          "clips back in fp32/int64. Most modern NVIDIA GPUs include fp16 tensor "
          "cores that accelerate the matmul; the resulting accuracy/throughput "
          "trade-off depends on the specific hardware.")
        w("")
    if dyn_rows:
        w("─" * 72)
        w("DYNAMIC OPS RESULTS — batched insert / delete throughput")
        w("─" * 72)
        w("")
        w(f"{'N':>12}  {'batch':>8}  {'CPU ins (s)':>13}  {'GPU ins (s)':>13}  "
          f"{'Ins spd':>8}  {'CPU del (s)':>13}  {'GPU del (s)':>13}  {'Del spd':>8}")
        w(f"{'─'*12}  {'─'*8}  {'─'*13}  {'─'*13}  {'─'*8}  "
          f"{'─'*13}  {'─'*13}  {'─'*8}")
        for r in dyn_rows:
            def fv(v): return f"{v:.6f}" if v >= 0 else "N/A"
            def sp(s): return f"{s:.2f}x" if (s == s and s > 0) else "N/A"
            w(f"{r['N']:>12,}  {r['batch_size']:>8,}  "
              f"{fv(r['cpu_insert_s']):>13}  {fv(r['gpu_insert_s']):>13}  "
              f"{sp(r['speedup_insert']):>8}  "
              f"{fv(r['cpu_delete_s']):>13}  {fv(r['gpu_delete_s']):>13}  "
              f"{sp(r['speedup_delete']):>8}")
        w("")
        w("Insert/delete framing: per-point inserts can't beat CPU on GPU; "
          "batched throughput is the right comparison. Stable-sort over (existing "
          "+ new) by shard_id, then bincount/cumsum to rebuild offsets. Both "
          "backends share the same algorithm; GPU just runs sort and scatter in "
          "parallel.")
        w("")
    w("─" * 72)
    w("RAW CSV DATA")
    w("─" * 72)
    w("")
    w("Build results:")
    build_fields = ['N','cpu_partition_mean','cpu_partition_std',
                    'cpu_map_sort_mean','cpu_map_sort_std',
                    'cpu_train_mean','cpu_train_std',
                    'gpu_h2d_mean','gpu_part_mean','gpu_part_std',
                    'gpu_ms_mean','gpu_ms_std','gpu_d2h_mean',
                    'speedup_partition','speedup_map_sort',
                    'cpu_mem_mb','gpu_mem_mb']
    w(",".join(build_fields))
    for r in build_rows:
        w(",".join(str(r.get(k, '')) for k in build_fields))
    w("")
    w("Query results:")
    query_fields = ['N','Q','cpu_mean','cpu_std',
                    'gpu_total_mean','gpu_total_std',
                    'gpu_compute_mean','gpu_compute_std',
                    'gpu_h2d_mean','gpu_d2h_mean',
                    'speedup_total','speedup_compute']
    w(",".join(query_fields))
    for r in query_rows:
        w(",".join(str(r.get(k, '')) for k in query_fields))
    w("")
    w("=" * 72)
    w("END OF REPORT")
    w("=" * 72)

    report = "\n".join(lines)

    # Print to stdout so it's visible in Colab output
    print(f"\n{'═'*72}")
    print("PAPER REPORT (also saved to results/paper_report.txt)")
    print(f"{'═'*72}")
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'paper_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {path}")
    return report


# ──────────────────────────────────────────────────────────────────────────────
# CSV output
# ──────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = {
    'build_results': [
        'N','cpu_partition_mean','cpu_partition_std',
        'cpu_map_sort_mean','cpu_map_sort_std',
        'cpu_train_mean','cpu_train_std',
        'gpu_h2d_mean','gpu_part_mean','gpu_ms_mean','gpu_d2h_mean',
        'gpu_train_mean','gpu_train_std',
        'speedup_partition','speedup_map_sort','speedup_training',
        'cpu_mem_mb','gpu_mem_mb'],
    'query_results': [
        'N','Q','cpu_mean','cpu_std','gpu_total_mean','gpu_total_std',
        'gpu_compute_mean','speedup_total','speedup_compute',
        'weights_source'],
    'range_query_results': [
        'N','Q','cpu_mean','cpu_std','gpu_total_mean','gpu_total_std',
        'gpu_compute_mean','gpu_compute_std','gpu_h2d_mean','gpu_d2h_mean',
        'speedup_total','speedup_compute','avg_match_per_query',
        'box_half_width_frac',
        'rtree_build_time','rtree_query_mean','rtree_query_std',
        'speedup_lisa_gpu_vs_rtree',
        'quadtree_build_time','quadtree_query_mean','quadtree_query_std',
        'speedup_lisa_gpu_vs_quadtree',
        'weights_source'],
    'knn_results': [
        'N','Q','k','lr_train_time','radius_sample_time','lattice_solve_time',
        'cpu_mean','cpu_std','gpu_total_mean',
        'gpu_total_std','gpu_compute_mean','gpu_compute_std',
        'speedup_total','speedup_compute',
        'faiss_mean','faiss_std','speedup_lisa_vs_faiss',
        'lisa_recall_vs_faiss','lisa_recall_at_k','brute_force_impl',
        'ckdtree_build_time','ckdtree_query_mean','ckdtree_query_std',
        'speedup_lisa_gpu_vs_ckdtree',
        'weights_source'],
    'dynamic_ops_results': [
        'N','batch_size','cpu_insert_s','gpu_insert_s','speedup_insert',
        'cpu_delete_s','gpu_delete_s','speedup_delete',
        'weights_source'],
    'mixed_precision_results': [
        'N','Q','gpu_fp64_mean','gpu_fp16_mean','speedup_fp16_vs_fp64',
        'fp16_shard_mismatch_rate','fp16_max_abs_shard_diff',
        'weights_source'],
    'mlp_results': [
        'N','hidden','max_iters','piecewise_train_time','piecewise_mean_loss',
        'mlp_train_time','mlp_mean_loss','mlp_vs_piecewise_loss_ratio'],
}


def _write_csv(name, rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fields = CSV_FIELDS[name]
    with open(os.path.join(output_dir, f'{name}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def save_csv(build_rows, query_rows, output_dir):
    _write_csv('build_results', build_rows, output_dir)
    _write_csv('query_results', query_rows, output_dir)

def save_range_csv(rows, output_dir):     _write_csv('range_query_results',     rows, output_dir)
def save_knn_csv(rows, output_dir):       _write_csv('knn_results',             rows, output_dir)
def save_dynamic_csv(rows, output_dir):   _write_csv('dynamic_ops_results',     rows, output_dir)
def save_mixed_precision_csv(rows, o):    _write_csv('mixed_precision_results', rows, o)
def save_mlp_csv(rows, output_dir):       _write_csv('mlp_results',             rows, output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LISA CPU vs GPU benchmark")
    parser.add_argument('--sizes', nargs='+', type=int, default=DEFAULT_SIZES)
    parser.add_argument('--reps', type=int, default=N_REPS)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--reuse-weights', action='store_true',
                        help='Cache GPU-trained Alphas/Betas to '
                             '<output-dir>/weights_N<size>_<dataset>_sigma<sigma>.npz '
                             'and reload them on subsequent runs that match N, dataset, sigma')
    parser.add_argument('--range-queries', type=int, default=1000,
                        help='number of range queries per dataset size')
    parser.add_argument('--range-box-frac', type=float, default=0.005,
                        help='range query box half-width as fraction of span')
    parser.add_argument('--skip-knn', action='store_true',
                        help='skip kNN benchmark')
    parser.add_argument('--knn-queries', type=int, default=500,
                        help='number of kNN queries per dataset size')
    parser.add_argument('--knn-k', type=int, default=10,
                        help='k for kNN queries')
    parser.add_argument('--knn-train-points', type=int, default=300,
                        help='training sample size for the radius lattice model')
    parser.add_argument('--skip-dynamic', action='store_true',
                        help='skip batched insert/delete benchmark')
    parser.add_argument('--dynamic-batches', type=int, nargs='+',
                        default=[1000, 10000, 100000],
                        help='batch sizes for the dynamic ops benchmark')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='dataset spec: uniform | skewed | csv:path | '
                             'npy:path | download:cities')
    parser.add_argument('--skip-mixed-precision', action='store_true',
                        help='skip fp16 mixed-precision benchmark')
    parser.add_argument('--skip-mlp', action='store_true',
                        help='skip MLP local-model benchmark')
    parser.add_argument('--mlp-hidden', type=int, default=32,
                        help='hidden width of the monotonic MLP per column')
    parser.add_argument('--mlp-iters', type=int, default=2000,
                        help='training iterations for the monotonic MLP')
    parser.add_argument('--mlp-lr', type=float, default=5e-3,
                        help='Adam learning rate (cosine-decayed to lr/100)')
    parser.add_argument('--output-dir', default=str(HERE / 'results'))
    args = parser.parse_args()

    sysinfo = get_system_info()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          GPU-Accelerated LISA Benchmark                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  GPU available  : {CUPY_AVAILABLE}")
    print(f"  FAISS available: {FAISS_AVAILABLE}")
    print(f"  GPU            : {sysinfo['gpu_name']}")
    if sysinfo['gpu_vram_gb'] > 0:
        print(f"  GPU VRAM       : {sysinfo['gpu_vram_gb']} GB")
    print(f"  CUDA           : {sysinfo['cuda']}")
    print(f"  Dataset sizes  : {args.sizes}")
    print(f"  Repetitions    : {args.reps}")
    print(f"  N_QUERIES      : {N_QUERIES:,}")
    print(f"  Params         : T={T_EACH_DIM}, n_models={N_MODELS}, σ={SIGMA}, page={PAGE_SIZE}")

    build_rows, query_rows, range_rows, knn_rows, dyn_rows, mp_rows, mlp_rows = [], [], [], [], [], [], []

    def _flush():
        """Write every CSV and the paper report from the accumulated rows."""
        save_csv(build_rows, query_rows, args.output_dir)
        save_range_csv(range_rows, args.output_dir)
        if knn_rows:
            save_knn_csv(knn_rows, args.output_dir)
        if dyn_rows:
            save_dynamic_csv(dyn_rows, args.output_dir)
        if mp_rows:
            save_mixed_precision_csv(mp_rows, args.output_dir)
        if mlp_rows:
            save_mlp_csv(mlp_rows, args.output_dir)
        generate_paper_report(sysinfo, build_rows, query_rows, args.output_dir,
                              range_rows=range_rows, knn_rows=knn_rows,
                              dyn_rows=dyn_rows, mp_rows=mp_rows,
                              mlp_rows=mlp_rows)

    def _reclaim_gpu():
        """Return idle CuPy memory blocks to CUDA."""
        if CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

    for N in args.sizes:
        try:
            br = run_build_benchmark(N, args.reps, args.skip_training,
                                     dataset=args.dataset,
                                     reuse_weights_dir=(args.output_dir
                                                        if args.reuse_weights
                                                        else None))
            build_rows.append({k: v for k, v in br.items() if not k.startswith('_')})
            _reclaim_gpu()
            qr = run_query_benchmark(br, args.reps)
            _reclaim_gpu()
            rr = run_range_query_benchmark(br, qr, args.reps,
                                           n_queries=args.range_queries,
                                           half_width_frac=args.range_box_frac)
            _reclaim_gpu()
            if not args.skip_knn:
                kr = run_knn_benchmark(br, qr, args.reps,
                                       n_queries=args.knn_queries, k=args.knn_k,
                                       n_train_points=args.knn_train_points)
                knn_rows.append(kr)
                _reclaim_gpu()
            if not args.skip_dynamic:
                dr = run_dynamic_ops_benchmark(br, qr, batch_sizes=args.dynamic_batches)
                dyn_rows.extend(dr)
                _reclaim_gpu()
            if not args.skip_mixed_precision:
                mp = run_mixed_precision_benchmark(br, qr, args.reps)
                mp_rows.append(mp)
                _reclaim_gpu()
            if not args.skip_mlp:
                mlp = run_mlp_benchmark(br, hidden=args.mlp_hidden,
                                        max_iters=args.mlp_iters, lr=args.mlp_lr,
                                        dataset=args.dataset)
                mlp_rows.append(mlp)
                _reclaim_gpu()
            query_rows.append({k: v for k, v in qr.items() if not k.startswith('_')})
            range_rows.append(rr)
        except Exception as e:
            import traceback
            print(f"\n!!! Stage failed at N={N:,}: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("Saving results gathered so far and continuing with next size.")
        _flush()
        _reclaim_gpu()

    print_summary_table(build_rows, query_rows)


if __name__ == '__main__':
    main()
