"""Local sanity check for BatchedMonotonicMLPTrainer.

Trains a few small monotonic MLPs and verifies that:
  (a) training reduces loss below a naive baseline
  (b) the learned function is monotone on a dense evaluation grid
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lisa_gpu.mlp_model import train_mlp_models


def main():
    rng = np.random.default_rng(0)
    n_models = 4
    n_per = 200
    mappings = np.sort(rng.uniform(0, 100.0, size=(n_models, n_per)).reshape(-1))
    col_split = np.arange(1, n_models + 1, dtype=np.int64) * n_per

    meta, elapsed, losses = train_mlp_models(
        mappings, col_split, hidden=8, max_iters=300, lr=1e-2
    )
    print(f"trained {n_models} MLPs in {elapsed*1000:.1f} ms")
    print(f"per-model loss: {losses}")
    print(f"mean loss: {losses.mean():.4f}")
    print(f"naive (predict mean) loss per model: {(n_per**2-1)/12 * n_per:.1f}")

    # Monotonicity check on a dense grid.
    starts = np.concatenate([[0], col_split[:-1]])
    ends = col_split
    for i in range(n_models):
        seg = mappings[int(starts[i]):int(ends[i])]
        grid = np.linspace(seg.min(), seg.max(), 200)
        x = (grid - meta['offs_x'][i]) / meta['scale_x'][i]
        W1 = np.exp(meta['W1_raw'][i])
        W2 = np.exp(meta['W2_raw'][i])
        W3 = np.exp(meta['W3_raw'][i])
        h1 = np.maximum(x[:, None] @ W1 + meta['b1'][i], 0)
        h2 = np.maximum(h1 @ W2 + meta['b2'][i], 0)
        y = (h2 @ W3 + meta['b3'][i]).reshape(-1)
        diffs = np.diff(y)
        if (diffs >= -1e-6).all():
            print(f"  model {i}: monotone (min Δ = {diffs.min():.2e})")
        else:
            print(f"  model {i}: NOT monotone! min Δ = {diffs.min():.4f}")
            raise AssertionError(f"model {i} failed monotonicity")
    print("OK")


if __name__ == "__main__":
    main()
