# GPU-Accelerated LISA

> A learned spatial index rebuilt for the GPU — MSc dissertation, High Performance Computing.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![CuPy](https://img.shields.io/badge/CuPy-GPU%20arrays-1f6feb)
![NumPy](https://img.shields.io/badge/NumPy-baseline-013243?logo=numpy&logoColor=white)

LISA is a *learned* index for spatial data ([SIGMOD 2020](https://doi.org/10.1145/3318464.3389703)): instead of an R-tree, it sorts points by a clever 1-D mapping and trains a set of small per-shard models that predict *where* a mapped value sits in the sorted array. A lookup becomes a model evaluation plus a short scan — no tree traversal.

This project takes the reference CPU implementation and **rebuilds the entire pipeline on the GPU** with CuPy/CUDA — sorting and index construction, model training, point/range/kNN queries, and bulk insert/delete — then benchmarks it end-to-end against classical spatial indexes (R-tree, quad-tree, kd-tree) and a GPU brute-force baseline, across datasets from 100K to 50M points and multiple GPUs (NVIDIA A100, L4, T4).

## Highlights

- **Full GPU pipeline** — build/sort, piecewise-model training, range queries, kNN, and dynamic insert/delete all run on the GPU, with a matching NumPy CPU implementation for correctness and speed parity.
- **Range queries up to ~100× faster** than the reference CPU LISA, and **11–17× faster than an R-tree**.
- **Index training 13–27× faster** — a 10M-point index builds in under three minutes.
- **Bulk inserts at ~17M points/second.**
- **Runs anywhere** — the benchmark falls back to CPU automatically when CuPy isn't installed.

> kNN is the honest weak spot: it remains slower than a tuned kd-tree and is documented as future work in the dissertation.

## Repository layout

| Path | What it is |
|------|-----------|
| `ResearchProject/gpu_lisa/lisa_gpu/` | GPU implementation (CuPy): index, piecewise & monotonic-MLP models, kNN, dynamic ops |
| `ResearchProject/gpu_lisa/lisa_cpu/` | NumPy CPU implementation used as the correctness/speed baseline |
| `ResearchProject/gpu_lisa/baselines.py` | R-tree / quad-tree / kd-tree / GPU brute-force baselines |
| `ResearchProject/gpu_lisa/benchmark.py` | End-to-end benchmark harness (one CSV per stage + summary report) |
| `ResearchProject/gpu_lisa/tests/` | Unit tests: range query, kNN, dynamic ops, batched training, MLP |
| `src/` | The original LISA reference implementation, ported to Python 3 |
| `Dissertation.pdf` | The full written dissertation |

## Running the benchmark

```bash
cd ResearchProject/gpu_lisa
pip install -r requirements.txt
pip install cupy-cuda12x          # match your CUDA version; omit to run CPU-only

python benchmark.py                                   # default sizes
python benchmark.py --sizes 100000 1000000 10000000   # pick dataset sizes
python benchmark.py --dataset skewed --output-dir results_skewed/
```

Each stage (build, query, range query, kNN, dynamic ops, mixed precision, MLP) writes a CSV plus a summary report to the output directory. Pass `--skip-mlp` to avoid the PyTorch dependency. The accelerated path needs an NVIDIA GPU + CUDA; CPU-only runs work without CuPy.

## Tech stack

Python · CuPy · CUDA · NumPy · SciPy · PyTorch *(optional, monotonic-MLP stage)* · Matplotlib

## Attribution

Forked from [`pfl-cs/LISA`](https://github.com/pfl-cs/LISA) (Li et al., *LISA: A Learned Index Structure for Spatial Data*, SIGMOD 2020). The GPU acceleration, the classical baselines, the benchmark suite (`ResearchProject/gpu_lisa/`), and the dissertation are my own work.

**Darius Jurjiu** — MSc, High Performance Computing.
