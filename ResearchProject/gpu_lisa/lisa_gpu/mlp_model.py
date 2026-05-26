"""Batched monotonic MLP local model (exploratory; not used by predict_shard_ids).

Trains all `n_models` tiny per-column MLPs in parallel with PyTorch.
Monotonicity comes from `exp(raw)` weights + ReLU activations.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BatchedMonotonicMLPTrainer:
    """Trains n_models independent monotonic MLPs in parallel on a GPU/CPU.

    Architecture per model: 1 → hidden → hidden → 1, with all weights = exp(raw)
    and ReLU activations between layers. Bias terms are unconstrained.
    """

    def __init__(self, hidden: int = 32, max_iters: int = 2000,
                 lr: float = 5e-3, device: str = "cuda"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed.")
        self.hidden = hidden
        self.max_iters = max_iters
        self.lr = lr
        self.device = device if torch.cuda.is_available() else "cpu"

    def fit(self, mappings, col_split_idxes):
        """Same call signature as BatchedPiecewiseTrainer.fit."""
        n_models = len(col_split_idxes)
        starts = np.concatenate([[0], np.asarray(col_split_idxes[:-1])])
        ends = np.asarray(col_split_idxes)
        seg_lens = (ends - starts).astype(np.int64)
        max_len = int(seg_lens.max())
        if max_len < 2 or n_models == 0:
            return (np.zeros((n_models, self.hidden), dtype=np.float64),) * 2

        # Pack data: (n_models, max_len) of (mapping_normalised, position_normalised).
        X = np.zeros((n_models, max_len), dtype=np.float32)
        Y = np.zeros((n_models, max_len), dtype=np.float32)
        mask = np.zeros((n_models, max_len), dtype=np.float32)
        scale_x = np.ones(n_models, dtype=np.float32)
        offs_x = np.zeros(n_models, dtype=np.float32)
        scale_y = np.ones(n_models, dtype=np.float32)
        for i in range(n_models):
            s, e = int(starts[i]), int(ends[i])
            seg = mappings[s:e].astype(np.float32)
            if seg.size < 2:
                continue
            x_min, x_max = float(seg.min()), float(seg.max())
            offs_x[i] = x_min
            scale_x[i] = max(x_max - x_min, 1e-6)
            scale_y[i] = max(e - s - 1, 1)
            X[i, :seg.size] = (seg - offs_x[i]) / scale_x[i]
            Y[i, :seg.size] = np.arange(seg.size, dtype=np.float32) / scale_y[i]
            mask[i, :seg.size] = 1.0

        device = self.device
        X_t = torch.from_numpy(X).to(device)
        Y_t = torch.from_numpy(Y).to(device)
        M_t = torch.from_numpy(mask).to(device)

        H = self.hidden
        # Init effective weights ≈ 1/fan_in so output scale doesn't blow up
        # across the three positive-weight layers.
        import math
        torch.manual_seed(0)
        m1 = math.log(1.0 / 1)
        mH = math.log(1.0 / H)
        W1_raw = torch.randn(n_models, 1, H, device=device) * 0.3 + m1
        b1 = torch.zeros(n_models, 1, H, device=device)
        W2_raw = torch.randn(n_models, H, H, device=device) * 0.3 + mH
        b2 = torch.zeros(n_models, 1, H, device=device)
        W3_raw = torch.randn(n_models, H, 1, device=device) * 0.3 + mH
        b3 = torch.zeros(n_models, 1, 1, device=device)
        for p in (W1_raw, b1, W2_raw, b2, W3_raw, b3):
            p.requires_grad_(True)

        opt = torch.optim.Adam([W1_raw, b1, W2_raw, b2, W3_raw, b3], lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_iters, eta_min=self.lr / 100
        )

        Xb = X_t.unsqueeze(-1)  # (n_models, max_len, 1)
        Yb = Y_t.unsqueeze(-1)
        Mb = M_t.unsqueeze(-1)
        denom = M_t.sum(dim=1, keepdim=True).clamp(min=1.0)

        for _ in range(self.max_iters):
            W1 = torch.exp(W1_raw)
            W2 = torch.exp(W2_raw)
            W3 = torch.exp(W3_raw)
            h1 = torch.relu(torch.matmul(Xb, W1) + b1)
            h2 = torch.relu(torch.matmul(h1, W2) + b2)
            pred = torch.matmul(h2, W3) + b3
            err = (pred - Yb) * Mb
            loss_per_model = (err * err).sum(dim=1) / denom
            loss = loss_per_model.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        # Return raw weights + meta so we can rebuild the model later.
        meta = {
            'W1_raw': W1_raw.detach().cpu().numpy(),
            'b1':     b1.detach().cpu().numpy(),
            'W2_raw': W2_raw.detach().cpu().numpy(),
            'b2':     b2.detach().cpu().numpy(),
            'W3_raw': W3_raw.detach().cpu().numpy(),
            'b3':     b3.detach().cpu().numpy(),
            'offs_x':  offs_x, 'scale_x': scale_x, 'scale_y': scale_y,
        }
        return meta

    def evaluate_loss(self, meta, mappings, col_split_idxes):
        """Recompute per-model training loss on the host (for reporting)."""
        n_models = len(col_split_idxes)
        starts = np.concatenate([[0], np.asarray(col_split_idxes[:-1])])
        ends = np.asarray(col_split_idxes)
        losses = np.zeros(n_models, dtype=np.float64)
        for i in range(n_models):
            s, e = int(starts[i]), int(ends[i])
            seg = mappings[s:e].astype(np.float32)
            if seg.size < 2:
                continue
            x = (seg - meta['offs_x'][i]) / meta['scale_x'][i]
            x = x[:, None]
            W1 = np.exp(meta['W1_raw'][i])
            W2 = np.exp(meta['W2_raw'][i])
            W3 = np.exp(meta['W3_raw'][i])
            h1 = np.maximum(x @ W1 + meta['b1'][i], 0)
            h2 = np.maximum(h1 @ W2 + meta['b2'][i], 0)
            pred = (h2 @ W3 + meta['b3'][i]).reshape(-1) * meta['scale_y'][i]
            target = np.arange(seg.size, dtype=np.float32)
            err = pred - target
            losses[i] = float((err * err).sum())
        return losses


def train_mlp_models(mappings, col_split_idxes, hidden=32, max_iters=2000,
                     lr=5e-3):
    """Top-level convenience. Returns (meta, elapsed_s, loss_per_model)."""
    trainer = BatchedMonotonicMLPTrainer(hidden=hidden, max_iters=max_iters, lr=lr)
    t0 = time.perf_counter()
    meta = trainer.fit(mappings, col_split_idxes)
    if trainer.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    losses = trainer.evaluate_loss(meta, mappings, col_split_idxes)
    return meta, elapsed, losses
