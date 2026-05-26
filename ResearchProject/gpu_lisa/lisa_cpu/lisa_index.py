"""LISAIndex: the CPU index that the GPU implementation wraps."""
import os
import pickle

import numpy as np

from .layout_utils import create_borders
from .piecewise_model import PiecewiseModel


class LISAIndex:
    """Minimal CPU LISA index used by the benchmark."""

    def __init__(self, params=None, data_dim=2, page_size=100, sigma=100):
        self.data_dim = data_dim
        self.page_size = page_size
        self.sigma = sigma
        self.params = params

        self.Alphas = None
        self.Betas = None
        self.col_split_shard_ids = None
        self.col_min_mappings = None
        self.shard_numbers_each_col = None
        self.model_split_mappings_without_tail = None
        self.pages = []
        self.shard_infos = []

        if params is not None:
            self._params_dump()

    # ------------------------------------------------------------------
    # Parameter unpacking
    # ------------------------------------------------------------------

    def _params_dump(self):
        p = self.params
        # Stored as n_dim + 0.5, n_parts + 0.5, n_models — use int() to truncate
        self.data_dim = int(p[-1])
        self.n_parts_each_dim = int(p[-2])
        self.n_piecewise_models = int(p[-3])
        self.max_value_each_dim = p[-4]
        self.min_value_each_dim = p[-5]
        self.eta = p[-6]

        self.all_split_upper_bounds = []
        self.all_front_split_points = []
        self.all_split_points_without_head_and_tail = []
        self.all_cell_lens = []

        start = 0
        n_repeat = 1
        for _ in range(self.data_dim - 1):
            one_dim_sub = []
            for _ in range(n_repeat):
                end = start + self.n_parts_each_dim
                sub = p[start:end]
                one_dim_sub.append(sub)
                start = end

            arr = np.array(one_dim_sub, dtype=np.float64)
            front = np.zeros_like(arr)
            front[:, 1:] = arr[:, :-1]
            self.all_split_upper_bounds.append(arr)
            self.all_front_split_points.append(front)
            self.all_split_points_without_head_and_tail.append(arr[:, :-1])
            self.all_cell_lens.append(arr - front)
            n_repeat *= self.n_parts_each_dim

        self.borders, self.cell_measures = create_borders(self.all_split_upper_bounds)

        end = start + self.n_parts_each_dim
        self.last_dim_split_upper_bounds = np.array(p[start:end], dtype=np.float64)
        ldf = np.zeros_like(self.last_dim_split_upper_bounds)
        ldf[1:] = self.last_dim_split_upper_bounds[:-1]
        self.last_dim_front_split_points = ldf
        self.last_dim_cell_lens = self.last_dim_split_upper_bounds - ldf
        start = end

        end = start + self.n_piecewise_models
        self.model_split_mappings = p[start:end]
        self.model_split_mappings_without_tail = self.model_split_mappings[:-1]
        assert end == len(p) - 6

    # ------------------------------------------------------------------
    # Mapping computation
    # ------------------------------------------------------------------

    def monotone_mappings(self, data):
        """Map N-D data points to 1-D monotone values."""
        idxes = np.searchsorted(
            self.all_split_points_without_head_and_tail[0][0], data[:, 0], side='right'
        )
        for i in range(1, data.shape[1] - 1):
            for j in range(idxes.shape[0]):
                idxes[j] = (
                    idxes[j] * self.n_parts_each_dim
                    + np.searchsorted(
                        self.all_split_points_without_head_and_tail[i][idxes[j]],
                        data[j, i], side='right'
                    )
                )
        last_dim = data[:, -1]
        left_data = data[:, :-1]
        measures = np.prod(left_data - self.borders[idxes], axis=1) / self.cell_measures[idxes]
        mappings = (
            measures * self.eta
            + last_dim / self.max_value_each_dim * (self.n_parts_each_dim - 1)
            + idxes * self.n_parts_each_dim
        )
        return mappings

    def monotone_mappings_and_col_split_idxes(self, sorted_points):
        mappings = self.monotone_mappings(sorted_points)
        col_idxes = np.searchsorted(self.model_split_mappings_without_tail, mappings, side='right')
        N = int(col_idxes[-1])
        col_split_idxes = [0] * (N + 1)
        for idx in col_idxes:
            col_split_idxes[idx] += 1
        count = col_split_idxes[0]
        for i in range(1, N + 1):
            count += col_split_idxes[i]
            col_split_idxes[i] = count
        return mappings.astype(sorted_points.dtype), np.array(col_split_idxes)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_piecewise_models(self, piecewise_models_dir, n_models):
        self.Alphas = np.zeros([n_models, self.sigma], dtype=np.float64)
        self.Betas = np.zeros([n_models, self.sigma], dtype=np.float64)
        for i in range(n_models):
            model_dir = os.path.join(piecewise_models_dir, str(i))
            a_path = os.path.join(model_dir, 'alphas.npy')
            b_path = os.path.join(model_dir, 'betas.npy')
            if os.path.exists(a_path) and os.path.exists(b_path):
                self.Alphas[i] = np.load(a_path)
                self.Betas[i] = np.load(b_path)
            else:
                return False
        return True

    def set_piecewise_models(self, Alphas, Betas):
        """Directly inject pre-trained model weights (used by benchmark)."""
        self.Alphas = Alphas
        self.Betas = Betas

    # ------------------------------------------------------------------
    # Batch shard prediction  ← primary GPU target
    # ------------------------------------------------------------------

    def predict_shard_ids(self, mappings):
        """
        For each mapping value, predict the shard ID it falls into.

        This is the core learned lookup: for Q input mappings returns Q shard IDs.
        Shape: mappings (Q,) → shard_ids (Q,)
        """
        col_idxes = np.searchsorted(
            self.model_split_mappings_without_tail, mappings, side='right'
        )
        trans_mappings = mappings - self.col_min_mappings[col_idxes]
        shard_id_offsets = self.col_split_shard_ids[col_idxes]
        max_pred_idxes = self.shard_numbers_each_col[col_idxes] - 1

        all_alphas = self.Alphas[col_idxes]   # (Q, sigma)
        all_betas = self.Betas[col_idxes]     # (Q, sigma)

        # ReLU basis evaluation: A[i,j] = max(0, trans_mapping[i] - beta[i,j])
        all_A = PiecewiseModel.relu(trans_mappings[:, None] - all_betas)  # (Q, sigma)

        pred_shard_ids = (
            (all_A * all_alphas).sum(axis=1) / self.page_size
        ).astype(np.int64).clip(0, max_pred_idxes)

        pred_shard_ids += shard_id_offsets
        return pred_shard_ids

    # ------------------------------------------------------------------
    # Page / shard generation (called after model training)
    # ------------------------------------------------------------------

    def generate_pages(self, sorted_data, one_dim_mappings, col_split_idxes):
        n_cols = col_split_idxes.shape[0]
        col_split_shard_ids = [0]
        self.shard_infos = []
        self.pages = []
        col_min_mappings = []
        n_shards = 0
        page_no = 0
        shard_id = 0
        start = 0

        for i in range(n_cols):
            end = int(col_split_idxes[i])
            one_dim_input = one_dim_mappings[start:end]
            min_mapping = one_dim_input.min()
            col_min_mappings.append(min_mapping)

            pred_idxes = self._cal_pred_idxes(one_dim_input - min_mapping, i)
            entries_count = self._shards_layout(pred_idxes, self.page_size)

            n_shards += len(entries_count)
            col_split_shard_ids.append(n_shards)

            entry_start = start
            for e_count in entries_count:
                shard_info = [[], []]
                entry_end = entry_start + e_count
                shard_id += 1
                if e_count > 0:
                    pages_data = sorted_data[entry_start:entry_end]
                    one_dim_pages = one_dim_mappings[entry_start:entry_end]
                    if e_count <= self.page_size:
                        self.pages.append(pages_data)
                        shard_info[0].append(page_no)
                        page_no += 1
                    else:
                        n_pages = (e_count + self.page_size - 1) // self.page_size
                        b_k = 0
                        for k in range(n_pages):
                            e_k = min(b_k + self.page_size, e_count)
                            self.pages.append(pages_data[b_k:e_k])
                            shard_info[0].append(page_no)
                            page_no += 1
                            if k > 0:
                                shard_info[1].append(one_dim_pages[b_k])
                            b_k = e_k
                self.shard_infos.append(shard_info)
                entry_start = entry_end
            start = end

        self.col_split_shard_ids = np.array(col_split_shard_ids, dtype=np.int64)
        self._cal_shard_numbers_each_col()
        self.col_min_mappings = np.array(col_min_mappings, dtype=np.float64)

    def _cal_pred_idxes(self, mappings, col_id):
        alphas = self.Alphas[col_id]
        betas = self.Betas[col_id]
        A = PiecewiseModel.relu(
            np.tile(mappings.reshape(-1, 1), [1, self.sigma]) - betas
        )
        return A @ alphas

    @staticmethod
    def _shards_layout(pred_idxes, N):
        # Edge case: fewer points than a page fits — return a single shard.
        if pred_idxes.shape[0] <= N:
            return [int(pred_idxes.shape[0])] if pred_idxes.shape[0] > 0 else []
        pred_idxes = (pred_idxes / N).astype(np.int64)
        max_shard_id = int(pred_idxes.max())
        pred_idxes = np.clip(pred_idxes, 0, max_shard_id)
        entries_count = [0] * (max_shard_id + 1)
        for idx in pred_idxes:
            entries_count[int(idx)] += 1
        n_last = entries_count[max_shard_id]
        if n_last < N:
            max_shard_id -= 1
            while max_shard_id >= 0:
                n_last += entries_count[max_shard_id]
                if n_last > N:
                    break
                max_shard_id -= 1
            if max_shard_id < 0:
                # Total < N — merge everything into shard 0.
                return [int(pred_idxes.shape[0])]
            entries_count[max_shard_id] = n_last
            entries_count = entries_count[:max_shard_id + 1]
        return entries_count

    def _cal_shard_numbers_each_col(self):
        self.shard_numbers_each_col = (
            self.col_split_shard_ids[1:] - self.col_split_shard_ids[:-1]
        ).astype(np.int64)

    # ------------------------------------------------------------------
    # Range query
    # ------------------------------------------------------------------

    def build_flat_layout(self):
        """Flatten pages into a single (N, d) array and per-shard point ranges.

        Pages are appended to self.pages in iteration order over shards, so
        each shard owns a contiguous slice of the flattened array.
        """
        if not self.pages:
            self.all_points = np.zeros((0, self.data_dim), dtype=np.float64)
            self.shard_point_starts = np.zeros(0, dtype=np.int64)
            self.shard_point_ends = np.zeros(0, dtype=np.int64)
            return

        page_sizes = np.array([p.shape[0] for p in self.pages], dtype=np.int64)
        page_offsets = np.concatenate([[0], np.cumsum(page_sizes)])
        self.all_points = np.concatenate(self.pages, axis=0)

        n_shards = len(self.shard_infos)
        shard_first_page = np.full(n_shards + 1, len(self.pages), dtype=np.int64)
        for s in range(n_shards - 1, -1, -1):
            pages = self.shard_infos[s][0]
            if pages:
                shard_first_page[s] = pages[0]
            else:
                shard_first_page[s] = shard_first_page[s + 1]
        self.shard_point_starts = page_offsets[shard_first_page[:-1]]
        self.shard_point_ends   = page_offsets[shard_first_page[1:]]

    def range_query(self, query_ranges, return_points=False):
        """Range query: for each box, count points inside (and optionally return them).

        query_ranges: (n_q, 2*d) — first d cols are low corner, next d are high.
        Returns: counts (n_q,) and optionally a list of (n_i, d) result arrays.
        """
        if not hasattr(self, 'all_points'):
            self.build_flat_layout()

        d = self.data_dim
        n_q = query_ranges.shape[0]
        low = query_ranges[:, :d]
        high = query_ranges[:, d:]

        low_maps = self.monotone_mappings(low)
        high_maps = self.monotone_mappings(high)
        m_lo = np.minimum(low_maps, high_maps)
        m_hi = np.maximum(low_maps, high_maps)
        s_lo = self.predict_shard_ids(m_lo)
        s_hi = self.predict_shard_ids(m_hi)

        n_shards = self.shard_point_starts.shape[0]
        s_lo = np.clip(s_lo, 0, n_shards - 1)
        s_hi = np.clip(s_hi, 0, n_shards - 1)

        counts = np.zeros(n_q, dtype=np.int64)
        results = [] if return_points else None
        for q in range(n_q):
            ps = self.shard_point_starts[s_lo[q]]
            pe = self.shard_point_ends[s_hi[q]]
            if pe <= ps:
                if return_points:
                    results.append(np.zeros((0, d), dtype=self.all_points.dtype))
                continue
            cand = self.all_points[ps:pe]
            mask = np.all((cand >= low[q]) & (cand <= high[q]), axis=1)
            counts[q] = int(mask.sum())
            if return_points:
                results.append(cand[mask])
        return counts, results

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        np.save(os.path.join(model_dir, 'page_data.npy'),
                np.concatenate(self.pages, axis=0))
        np.save(os.path.join(model_dir, 'm_counts.npy'),
                np.array([p.shape[0] for p in self.pages], dtype=np.int64))
        meta = [self.page_size, self.sigma] + self.col_split_shard_ids.tolist()
        np.save(os.path.join(model_dir, 'meta_infos.npy'),
                np.array(meta, dtype=np.int64))
        np.save(os.path.join(model_dir, 'col_params.npy'), self.params)
        np.save(os.path.join(model_dir, 'col_min_mappings.npy'), self.col_min_mappings)
        np.save(os.path.join(model_dir, 'shard_params.npy'),
                np.concatenate([self.Alphas, self.Betas], axis=0))
        with open(os.path.join(model_dir, 'local_models.pkl'), 'wb') as f:
            pickle.dump(self.shard_infos, f)

    def load(self, model_dir):
        m_counts = np.load(os.path.join(model_dir, 'm_counts.npy'))
        page_data = np.load(os.path.join(model_dir, 'page_data.npy'))
        self.pages = []
        start = 0
        for c in m_counts:
            self.pages.append(page_data[start:start + c])
            start += c

        self.params = np.load(os.path.join(model_dir, 'col_params.npy'))
        self._params_dump()
        self.col_min_mappings = np.load(os.path.join(model_dir, 'col_min_mappings.npy'))

        meta = np.load(os.path.join(model_dir, 'meta_infos.npy'))
        self.page_size = int(meta[0])
        self.sigma = int(meta[1])
        self.col_split_shard_ids = meta[2:]
        self._cal_shard_numbers_each_col()

        shard_params = np.load(os.path.join(model_dir, 'shard_params.npy'))
        n_cols = shard_params.shape[0] // 2      # Python 3: integer division
        self.Alphas = shard_params[:n_cols]
        self.Betas = shard_params[n_cols:]

        with open(os.path.join(model_dir, 'local_models.pkl'), 'rb') as f:
            self.shard_infos = pickle.load(f)
        return True
