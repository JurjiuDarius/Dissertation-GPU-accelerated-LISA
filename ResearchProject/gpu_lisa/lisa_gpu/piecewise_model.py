"""Batched GPU training of all piecewise-linear local models simultaneously.

Each column's mapping slice is padded to a common length so all `n_models`
problems fit in one `(n_models, max_len, sigma)` tensor; one
`cp.linalg.solve` then runs the Newton step across every column. Differs
from the per-model CPU version in three ways: ridge regularisation in
place of per-model condition checks, a fixed iteration budget with an
"active" mask, and no monotone re-projection fallback.

Uses CuPy if available; falls back to NumPy so tests run without CUDA.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import cupy as cp
    _DEFAULT_XP = cp
except Exception:
    cp = None
    _DEFAULT_XP = np


LR_GRID = (0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0)
RIDGE = 1e-10
PAD_MAPPING = -1e18


def _is_cupy(xp):
    return cp is not None and xp is cp


def _to_bool(x):
    return bool(x.item()) if hasattr(x, "item") else bool(x)


def _to_int(x):
    return int(x.item()) if hasattr(x, "item") else int(x)


class BatchedPiecewiseTrainer:
    """Trains all n_models piecewise-linear ReLU models in one batched pass."""

    def __init__(self, sigma: int = 50, max_iters: int = 200, xp=None):
        self.sigma = sigma
        self.max_iters = max_iters
        self.xp = xp if xp is not None else _DEFAULT_XP

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, mappings: np.ndarray, col_split_idxes: np.ndarray):
        """Train all models. Returns (Alphas, Betas) as host (numpy) arrays."""
        xp = self.xp
        sigma = self.sigma

        mappings_d = xp.asarray(mappings, dtype=xp.float64)
        col_split = xp.asarray(col_split_idxes, dtype=xp.int64)
        n_models = int(col_split.shape[0])

        starts = xp.concatenate([xp.zeros(1, dtype=xp.int64), col_split[:-1]])
        ends = col_split
        seg_lens = ends - starts
        max_len = _to_int(seg_lens.max())

        if max_len < 2 or n_models == 0:
            return (np.zeros((n_models, sigma), dtype=np.float64),
                    np.zeros((n_models, sigma), dtype=np.float64))

        j_range = xp.arange(max_len, dtype=xp.int64)
        valid = j_range[None, :] < seg_lens[:, None]
        # Clamp to a safe in-bounds index for padded slots.
        global_idx = xp.minimum(starts[:, None] + j_range[None, :], ends[-1] - 1)
        raw_2d = mappings_d[global_idx]

        big = xp.asarray(np.finfo(np.float64).max, dtype=xp.float64)
        masked_for_min = xp.where(valid, raw_2d, big)
        min_values = masked_for_min.min(axis=1)
        undersized = seg_lens < sigma + 1
        min_values = xp.where(undersized, xp.zeros_like(min_values), min_values)

        mappings_2d = xp.where(valid, raw_2d - min_values[:, None],
                               xp.full_like(raw_2d, PAD_MAPPING))
        positions_2d = xp.where(valid,
                                xp.broadcast_to(j_range[None, :].astype(xp.float64),
                                                (n_models, max_len)),
                                xp.zeros((1, max_len), dtype=xp.float64))

        k_range = xp.arange(sigma, dtype=xp.int64)
        n_each_cell = xp.where(undersized,
                               xp.ones_like(seg_lens),
                               seg_lens // sigma)
        split_idxes = xp.clip(k_range[None, :] * n_each_cell[:, None], 0, max_len - 1)
        betas = xp.take_along_axis(mappings_2d, split_idxes, axis=1).astype(xp.float64)

        alphas = xp.zeros((n_models, sigma), dtype=xp.float64)
        best_alphas = alphas.copy()
        best_betas = betas.copy()
        best_loss = xp.full((n_models,), np.inf, dtype=xp.float64)
        active = ~undersized

        # Seed best from initial-beta solve.
        a0, A0 = self._batched_cal_alphas(mappings_2d, positions_2d, betas)
        l0 = self._loss_from_A(A0, a0, positions_2d, seg_lens)
        v0 = self._batched_valid(a0, betas)
        improved = v0 & (l0 < best_loss) & active
        best_alphas = xp.where(improved[:, None], a0, best_alphas)
        best_betas = xp.where(improved[:, None], betas, best_betas)
        best_loss = xp.where(improved, l0, best_loss)
        alphas = xp.where(active[:, None], a0, alphas)

        for _ in range(self.max_iters):
            if not _to_bool(active.any()):
                break

            betas, alphas, step_loss, step_valid, made_progress = self._newton_step(
                mappings_2d, positions_2d, betas, alphas, seg_lens, active
            )

            improved = step_valid & (step_loss < best_loss) & active
            best_alphas = xp.where(improved[:, None], alphas, best_alphas)
            best_betas = xp.where(improved[:, None], betas, best_betas)
            best_loss = xp.where(improved, step_loss, best_loss)

            active = active & made_progress

        # Return betas in *shifted* (per-column-min subtracted) coordinates —
        # this matches what the CPU PiecewiseModel stores and what
        # LISAIndex.predict_shard_ids expects (it subtracts col_min from the
        # query mapping and compares against betas directly).
        Alphas_h = self._to_host(best_alphas)
        Betas_h = self._to_host(best_betas)
        return Alphas_h, Betas_h

    # ------------------------------------------------------------------
    # Newton step
    # ------------------------------------------------------------------

    def _newton_step(self, mappings_2d, positions_2d, betas, alphas,
                     seg_lens, active):
        xp = self.xp
        sigma = self.sigma
        n_models = mappings_2d.shape[0]

        A = self._relu(mappings_2d[:, :, None] - betas[:, None, :])

        init_loss = self._loss_from_A(A, alphas, positions_2d, seg_lens)

        # CPU code uses np.sign(A). The values are {0, 1} so fp32 is plenty;
        # this halves the largest tensor in the Newton step (~GB on big N).
        sign_A = (A > 0).astype(xp.float32)
        G_T = -sign_A   # (M, n, σ) — keep the non-transposed view, einsum can
                         # transpose along axis labels for free.

        pred = (A * alphas[:, None, :]).sum(axis=2)
        pred_clipped = xp.minimum(xp.maximum(pred, 0.0),
                                  seg_lens[:, None].astype(xp.float64))
        r = pred_clipped - positions_2d

        # 'mns,mn->ms' = G^T r in the original (M,σ,n)×(M,n) formulation.
        Gr = xp.einsum("mns,mn->ms", G_T, r.astype(xp.float32)).astype(xp.float64)
        inv_n = xp.where(seg_lens > 0,
                         1.0 / xp.maximum(seg_lens.astype(xp.float64), 1.0),
                         xp.zeros_like(seg_lens, dtype=xp.float64))
        g = 2.0 * alphas * Gr * inv_n[:, None]

        # GGT(M,σ,σ) = (G)(Gᵀ) where G = -sign_A.T. With G_T = -sign_A in
        # (M, n, σ) layout, we want G_T.T @ G_T  → einsum 'mns,mnt->mst'.
        GGT = xp.einsum("mns,mnt->mst", G_T, G_T).astype(xp.float64)
        del sign_A, G_T   # release the (M, n, σ) fp32 buffer ASAP
        Y = 2.0 * alphas[:, :, None] * GGT * alphas[:, None, :] * inv_n[:, None, None]
        Y = Y + RIDGE * self._eye(n_models, sigma)

        try:
            s_newton = -self._batched_solve(Y, g)
        except Exception:
            s_newton = -g

        new_betas, new_alphas, new_loss, new_valid, ok_newton = self._line_search(
            mappings_2d, positions_2d, betas, s_newton, init_loss, seg_lens
        )

        need_fallback = active & ~ok_newton
        if _to_bool(need_fallback.any()):
            fb_betas, fb_alphas, fb_loss, _fb_valid, ok_fb = self._line_search(
                mappings_2d, positions_2d, betas, -g, init_loss, seg_lens
            )
            sel = need_fallback & ok_fb
            new_betas = xp.where(sel[:, None], fb_betas, new_betas)
            new_alphas = xp.where(sel[:, None], fb_alphas, new_alphas)
            new_loss = xp.where(sel, fb_loss, new_loss)
            new_valid = new_valid | sel
            ok_newton = ok_newton | sel

        new_betas = xp.where(active[:, None], new_betas, betas)
        new_alphas = xp.where(active[:, None], new_alphas, alphas)
        return new_betas, new_alphas, new_loss, new_valid, ok_newton

    # ------------------------------------------------------------------
    # Line search
    # ------------------------------------------------------------------

    def _line_search(self, mappings_2d, positions_2d, betas, s, init_loss, seg_lens):
        xp = self.xp
        n_lrs = len(LR_GRID)
        n_models, sigma = betas.shape
        losses_all = xp.full((n_lrs, n_models), np.inf, dtype=xp.float64)
        valid_all = xp.zeros((n_lrs, n_models), dtype=bool)
        betas_all = xp.empty((n_lrs, n_models, sigma), dtype=xp.float64)
        alphas_all = xp.empty((n_lrs, n_models, sigma), dtype=xp.float64)

        for k, lr in enumerate(LR_GRID):
            cand = xp.sort(betas + lr * s, axis=1)
            a, A = self._batched_cal_alphas(mappings_2d, positions_2d, cand)
            loss = self._loss_from_A(A, a, positions_2d, seg_lens)
            valid = self._batched_valid(a, cand)
            losses_all[k] = xp.where(valid, loss, np.inf)
            valid_all[k] = valid
            betas_all[k] = cand
            alphas_all[k] = a
            del A   # release the (M, n, σ) buffer before the next iter allocates

        best_k = xp.argmin(losses_all, axis=0)
        rng = xp.arange(n_models)
        best_betas = betas_all[best_k, rng]
        best_alphas = alphas_all[best_k, rng]
        best_loss = losses_all[best_k, rng]
        best_valid = valid_all[best_k, rng]
        improved = best_valid & (best_loss < init_loss)
        return best_betas, best_alphas, best_loss, best_valid, improved

    # ------------------------------------------------------------------
    # Core kernels
    # ------------------------------------------------------------------

    def _batched_cal_alphas(self, mappings_2d, positions_2d, betas):
        xp = self.xp
        n_models, sigma = betas.shape
        A = self._relu(mappings_2d[:, :, None] - betas[:, None, :])
        ATA = xp.einsum("mij,mik->mjk", A, A) + RIDGE * self._eye(n_models, sigma)
        ATp = xp.einsum("mij,mi->mj", A, positions_2d)
        alphas = self._batched_solve(ATA, ATp)
        return alphas, A

    def _loss_from_A(self, A, alphas, positions_2d, seg_lens):
        xp = self.xp
        pred = (A * alphas[:, None, :]).sum(axis=2)
        pred_clipped = xp.minimum(xp.maximum(pred, 0.0),
                                  seg_lens[:, None].astype(xp.float64))
        r = pred_clipped - positions_2d
        return (r * r).sum(axis=1)

    def _batched_valid(self, alphas, betas):
        xp = self.xp
        beta_diffs = betas[:, 1:] - betas[:, :-1]
        beta_ok = (beta_diffs > 0).all(axis=1)
        alpha_ok = (xp.cumsum(alphas, axis=1) >= 0).all(axis=1)
        return beta_ok & alpha_ok

    def _batched_solve(self, A, b):
        xp = self.xp
        if b.ndim == 2:
            return xp.linalg.solve(A, b[:, :, None])[:, :, 0]
        return xp.linalg.solve(A, b)

    @staticmethod
    def _relu(x):
        return x.clip(min=0)

    def _eye(self, n_models, sigma):
        xp = self.xp
        return xp.broadcast_to(xp.eye(sigma, dtype=xp.float64),
                               (n_models, sigma, sigma))

    def _to_host(self, arr):
        if _is_cupy(self.xp):
            return cp.asnumpy(arr)
        return np.asarray(arr)


def train_models_gpu(mappings, col_split_idxes, sigma=50, max_iters=200, xp=None):
    """Train all piecewise models on GPU. Returns (Alphas, Betas, elapsed_s)."""
    if xp is None:
        xp = _DEFAULT_XP

    if _is_cupy(xp):
        cp.cuda.Device().synchronize()
    t0 = time.perf_counter()

    trainer = BatchedPiecewiseTrainer(sigma=sigma, max_iters=max_iters, xp=xp)
    Alphas, Betas = trainer.fit(mappings, col_split_idxes)

    if _is_cupy(xp):
        cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - t0
    return Alphas, Betas, elapsed
