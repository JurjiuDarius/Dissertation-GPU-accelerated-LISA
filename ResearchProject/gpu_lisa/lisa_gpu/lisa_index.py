"""GPULISAIndex: GPU implementations of predict_shard_ids, range_query, and
an fp16 mixed-precision variant of predict_shard_ids.
"""
import time

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class GPULISAIndex:
    """
    Thin GPU wrapper around a pre-built LISAIndex.
    Call load_to_gpu() once, then predict_shard_ids_gpu() for each batch.
    """

    def __init__(self, cpu_index):
        """
        Parameters
        ----------
        cpu_index : lisa_cpu.lisa_index.LISAIndex
            A fully built CPU index (Alphas, Betas, col_min_mappings, etc. set).
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available.")
        self.cpu_index = cpu_index
        self._loaded = False

    def load_to_gpu(self):
        """Transfer all model parameters to GPU memory (called once)."""
        idx = self.cpu_index
        self.Alphas_gpu = cp.array(idx.Alphas)                             # (n_models, sigma)
        self.Betas_gpu = cp.array(idx.Betas)                               # (n_models, sigma)
        self.col_min_gpu = cp.array(idx.col_min_mappings)                  # (n_models,)
        self.col_split_shard_ids_gpu = cp.array(
            idx.col_split_shard_ids.astype(np.int64))                      # (n_models+1,)
        self.shard_numbers_each_col_gpu = cp.array(
            idx.shard_numbers_each_col.astype(np.int64))                   # (n_models,)
        self.model_split_mappings_without_tail_gpu = cp.array(
            idx.model_split_mappings_without_tail)                         # (n_models-1,)
        self.page_size = idx.page_size
        self.data_dim = idx.data_dim
        self._loaded = True
        self._loaded_points = False

    def load_points_to_gpu(self):
        """Transfer flattened pages and per-shard point ranges to GPU."""
        if not self._loaded:
            self.load_to_gpu()
        idx = self.cpu_index
        if not hasattr(idx, 'all_points'):
            idx.build_flat_layout()
        self.all_points_gpu = cp.asarray(idx.all_points)
        self.shard_point_starts_gpu = cp.asarray(idx.shard_point_starts.astype(np.int64))
        self.shard_point_ends_gpu   = cp.asarray(idx.shard_point_ends.astype(np.int64))
        self.n_shards = int(idx.shard_point_starts.shape[0])
        self._loaded_points = True

    # ------------------------------------------------------------------
    # GPU batch prediction  ← primary benchmark target
    # ------------------------------------------------------------------

    def predict_shard_ids_gpu(self, mappings_cpu):
        """
        GPU-accelerated predict_shard_ids.

        Parameters
        ----------
        mappings_cpu : ndarray (Q,) — mapping values on CPU

        Returns
        -------
        shard_ids_cpu : ndarray (Q,) int64 — predicted shard IDs back on CPU
        timing        : dict with 'transfer_to_gpu', 'compute', 'transfer_to_cpu'
        """
        if not self._loaded:
            self.load_to_gpu()

        timing = {}

        # Transfer query mappings to GPU
        t0 = time.perf_counter()
        mappings_gpu = cp.array(mappings_cpu)
        cp.cuda.Stream.null.synchronize()
        timing['transfer_to_gpu'] = time.perf_counter() - t0

        # Core GPU computation
        t0 = time.perf_counter()

        col_idxes = cp.searchsorted(
            self.model_split_mappings_without_tail_gpu, mappings_gpu, side='right'
        )                                                          # (Q,) int64

        trans_mappings = mappings_gpu - self.col_min_gpu[col_idxes]        # (Q,)
        shard_offsets = self.col_split_shard_ids_gpu[col_idxes]            # (Q,)
        max_pred = self.shard_numbers_each_col_gpu[col_idxes] - 1          # (Q,)

        all_alphas = self.Alphas_gpu[col_idxes]                            # (Q, sigma)
        all_betas = self.Betas_gpu[col_idxes]                              # (Q, sigma)

        # ReLU basis:  A[i,j] = max(0, trans_mapping[i] - beta[i,j])
        relu_input = trans_mappings[:, None] - all_betas                   # (Q, sigma)
        all_A = cp.maximum(relu_input, 0.0)                                # (Q, sigma)

        pred = (
            (all_A * all_alphas).sum(axis=1) / self.page_size
        ).astype(cp.int64)
        pred = cp.clip(pred, 0, max_pred) + shard_offsets

        cp.cuda.Stream.null.synchronize()
        timing['compute'] = time.perf_counter() - t0

        # Transfer result back
        t0 = time.perf_counter()
        result = pred.get()
        cp.cuda.Stream.null.synchronize()
        timing['transfer_to_cpu'] = time.perf_counter() - t0

        return result, timing


    def predict_shard_ids_gpu_fp16(self, mappings_cpu):
        """fp16 variant of predict_shard_ids_gpu: Alphas / Betas / mappings
        are cast to fp16 for the ReLU + dot product, accumulated in fp32,
        then clipped and offset back to int64. Returns (shard_ids_cpu, timing).
        """
        if not self._loaded:
            self.load_to_gpu()
        timing = {}

        t0 = time.perf_counter()
        mappings_gpu = cp.asarray(mappings_cpu, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        timing['transfer_to_gpu'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # The searchsorted on the column-boundary array stays fp32 / int (we
        # don't lose anything material there).
        msmwt_f32 = self.model_split_mappings_without_tail_gpu.astype(cp.float32)
        col_idxes = cp.searchsorted(msmwt_f32, mappings_gpu, side='right')
        col_min_f32 = self.col_min_gpu.astype(cp.float32)
        trans_mappings = (mappings_gpu - col_min_f32[col_idxes]).astype(cp.float16)

        shard_offsets = self.col_split_shard_ids_gpu[col_idxes]
        max_pred = self.shard_numbers_each_col_gpu[col_idxes] - 1

        alphas_f16 = self.Alphas_gpu.astype(cp.float16)[col_idxes]
        betas_f16 = self.Betas_gpu.astype(cp.float16)[col_idxes]

        relu_input = trans_mappings[:, None] - betas_f16
        all_A = cp.maximum(relu_input, cp.float16(0.0))
        # Accumulate in fp32 to limit rounding (still cheaper than fp64).
        pred_f32 = (all_A.astype(cp.float32) * alphas_f16.astype(cp.float32)
                   ).sum(axis=1)
        pred = (pred_f32 / self.page_size).astype(cp.int64)
        pred = cp.clip(pred, 0, max_pred) + shard_offsets
        cp.cuda.Stream.null.synchronize()
        timing['compute'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = pred.get()
        cp.cuda.Stream.null.synchronize()
        timing['transfer_to_cpu'] = time.perf_counter() - t0
        return result, timing

    def _predict_shard_ids_internal(self, mappings_gpu):
        """Same as predict_shard_ids_gpu but takes GPU-side mappings."""
        col_idxes = cp.searchsorted(
            self.model_split_mappings_without_tail_gpu, mappings_gpu, side='right'
        )
        trans_mappings = mappings_gpu - self.col_min_gpu[col_idxes]
        shard_offsets = self.col_split_shard_ids_gpu[col_idxes]
        max_pred = self.shard_numbers_each_col_gpu[col_idxes] - 1
        all_alphas = self.Alphas_gpu[col_idxes]
        all_betas = self.Betas_gpu[col_idxes]
        relu_input = trans_mappings[:, None] - all_betas
        all_A = cp.maximum(relu_input, 0.0)
        pred = ((all_A * all_alphas).sum(axis=1) / self.page_size).astype(cp.int64)
        return cp.clip(pred, 0, max_pred) + shard_offsets

    def range_query_gpu(self, query_ranges_cpu, low_maps_cpu, high_maps_cpu):
        """GPU range query.

        Computes corner mappings on CPU (cheap — small Q), then does the
        full candidate scan + filter on GPU.

        Returns
        -------
        counts_cpu : (n_q,) int64 — per-query match count
        timing     : dict with 'transfer_to_gpu', 'compute', 'transfer_to_cpu'
        """
        if not self._loaded_points:
            self.load_points_to_gpu()

        d = self.data_dim
        n_q = query_ranges_cpu.shape[0]
        timing = {}

        t0 = time.perf_counter()
        qr_gpu = cp.asarray(query_ranges_cpu)
        low_g  = qr_gpu[:, :d]
        high_g = qr_gpu[:, d:]
        low_maps_g  = cp.asarray(low_maps_cpu)
        high_maps_g = cp.asarray(high_maps_cpu)
        cp.cuda.Stream.null.synchronize()
        timing['transfer_to_gpu'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m_lo = cp.minimum(low_maps_g, high_maps_g)
        m_hi = cp.maximum(low_maps_g, high_maps_g)
        s_lo = cp.clip(self._predict_shard_ids_internal(m_lo), 0, self.n_shards - 1)
        s_hi = cp.clip(self._predict_shard_ids_internal(m_hi), 0, self.n_shards - 1)

        p_start = self.shard_point_starts_gpu[s_lo]
        p_end   = self.shard_point_ends_gpu[s_hi]
        cand_counts = cp.maximum(p_end - p_start, 0)
        total = int(cand_counts.sum().item())

        out_counts = cp.zeros(n_q, dtype=cp.int64)
        if total > 0:
            # Chunk queries so each gather tensor stays under ~1 GB.
            budget = max(1, 1_000_000_000 // (d * 8))
            cum_counts_cpu = np.concatenate(
                [[0], np.cumsum(cand_counts.get().astype(np.int64))]
            )
            chunk_start = 0
            while chunk_start < n_q:
                chunk_end = chunk_start + 1
                while (chunk_end < n_q
                       and cum_counts_cpu[chunk_end + 1] - cum_counts_cpu[chunk_start] <= budget):
                    chunk_end += 1

                chunk_counts = cand_counts[chunk_start:chunk_end]
                chunk_total = int(cum_counts_cpu[chunk_end] - cum_counts_cpu[chunk_start])

                if chunk_total > 0:
                    local_q_ids = repeat_by_counts(chunk_counts)
                    cum_chunk = cp.concatenate(
                        [cp.zeros(1, dtype=cp.int64), cp.cumsum(chunk_counts)]
                    )
                    local_off = cp.arange(chunk_total, dtype=cp.int64) - cum_chunk[local_q_ids]

                    chunk_p_start = p_start[chunk_start:chunk_end]
                    point_ids = chunk_p_start[local_q_ids] + local_off
                    abs_q_ids = local_q_ids + chunk_start

                    cand_pts = self.all_points_gpu[point_ids]
                    low_q    = low_g[abs_q_ids]
                    high_q   = high_g[abs_q_ids]
                    mask = ((cand_pts >= low_q) & (cand_pts <= high_q)).all(axis=1)
                    cpx_add_at(out_counts, abs_q_ids[mask], 1)

                    del local_q_ids, local_off, point_ids, cand_pts, low_q, high_q, mask, abs_q_ids

                chunk_start = chunk_end
        cp.cuda.Stream.null.synchronize()
        timing['compute'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        counts_cpu = out_counts.get()
        cp.cuda.Stream.null.synchronize()
        timing['transfer_to_cpu'] = time.perf_counter() - t0

        return counts_cpu, timing


def repeat_by_counts(counts):
    """GPU equivalent of np.repeat(arange(len(counts)), counts).
    Uses cumsum + searchsorted because cp.repeat does not accept a device
    array as its `repeats` argument."""
    if counts.shape[0] == 0:
        return cp.zeros(0, dtype=counts.dtype)
    cum = cp.cumsum(counts)
    total = int(cum[-1].item())
    if total == 0:
        return cp.zeros(0, dtype=counts.dtype)
    return cp.searchsorted(cum, cp.arange(total, dtype=counts.dtype), side='right')


def cpx_add_at(target, indices, value):
    """Add a scalar `value` at each position in `indices` of `target`,
    using bincount (works for any integer dtype)."""
    cnt = cp.bincount(indices, minlength=target.shape[0])
    target += cnt.astype(target.dtype) * value


def gpu_warmup():
    """Run a dummy GPU op to initialise the CUDA context."""
    if not CUPY_AVAILABLE:
        return
    dummy = cp.ones((512, 512), dtype=cp.float64)
    _ = cp.sort(dummy.ravel())
    cp.cuda.Stream.null.synchronize()
    del dummy
