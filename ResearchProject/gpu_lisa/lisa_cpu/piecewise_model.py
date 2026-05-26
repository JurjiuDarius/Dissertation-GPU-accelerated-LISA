"""Piecewise-linear ReLU local model that maps mapping values to positions
within one LISA column. The CPU trainer runs columns sequentially; the
batched GPU version (lisa_gpu/piecewise_model.py) trains all columns at once.
"""
import os
import sys

import numpy as np


class PiecewiseModel:
    def __init__(self, model_id, sorted_mappings, sigma=100):
        self.id = model_id
        self.sigma = sigma

        self.min_value = sorted_mappings.min()
        self.sorted_mappings = sorted_mappings - self.min_value
        self.positions = np.arange(self.sorted_mappings.shape[0], dtype=np.int64)

        self.alphas = None
        self.betas = np.zeros(self.sigma, dtype=np.float64)

        self.init_alphas = np.zeros(self.sigma, dtype=np.float64)
        self.init_betas = np.zeros(self.sigma, dtype=np.float64)
        self.sorted_mappings_reshape = self.sorted_mappings.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    @staticmethod
    def relu(A):
        return np.maximum(A, 0)

    def cal_alphas(self, betas, mappings=None, positions=None):
        if mappings is None or positions is None:
            mappings = self.sorted_mappings_reshape
            positions = self.positions
        A = self.relu(np.tile(mappings, [1, self.sigma]) - betas)
        symm = A.T @ A
        if np.linalg.cond(symm) < 1.0 / sys.float_info.epsilon:
            alphas = np.linalg.inv(symm) @ (A.T @ positions)
            return alphas, A
        return None, None

    def cal_loss(self, A, alphas):
        r = (A @ alphas).clip(0, self.sorted_mappings.shape[0]) - self.positions
        return float(np.sum(r * r))

    def lr_search(self, s, init_betas, init_loss):
        lrs = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8]
        best_lr, best_loss, best_betas, best_alphas = -1, init_loss, None, None

        for lr in lrs:
            betas = np.sort(init_betas + lr * s)
            alphas, A = self.cal_alphas(betas)
            if alphas is None:
                continue
            if np.cumsum(alphas).min() <= 0:
                continue
            loss = self.cal_loss(A, alphas)
            if loss < best_loss:
                best_lr, best_loss, best_betas, best_alphas = lr, loss, betas, alphas

        return best_lr, best_loss, best_betas, best_alphas

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def cal_init_alphas(self, betas):
        idxes = np.searchsorted(self.sorted_mappings, betas, side='right')
        pred_positions = (idxes - 0.5).clip(min=0)
        alphas = np.zeros(self.sigma, dtype=np.float64)

        for i in range(1, pred_positions.shape[0]):
            v = sum(alphas[j] * (betas[i] - betas[j]) for j in range(i - 1))
            diff = betas[i] - betas[i - 1]
            alphas[i - 1] = 0.0 if diff <= 0 else (pred_positions[i] - v) / diff

        max_mapping = self.sorted_mappings[-1]
        v = sum(alphas[j] * (betas[j] - betas[j - 1]) for j in range(1, self.sigma))
        alphas[-1] = (self.sorted_mappings.shape[0] - 1) / (max_mapping - betas[-1]) if (max_mapping - betas[-1]) != 0 else 0.0
        if np.cumsum(alphas)[-1] < 0:
            alphas[-1] = -np.cumsum(alphas)[-2]
        return alphas

    def cal_alphas_with_monotone_constrain(self, betas, old_alphas):
        pred_positions = self.relu(
            np.tile(betas.reshape(-1, 1), [1, self.sigma]) - betas
        ) @ old_alphas
        pred_positions = np.sort(pred_positions.clip(0, self.sorted_mappings.shape[0]))

        alphas = np.zeros(self.sigma, dtype=np.float64)
        for i in range(1, pred_positions.shape[0]):
            v = sum(alphas[j] * (betas[i] - betas[j]) for j in range(i - 1))
            diff = betas[i] - betas[i - 1]
            alphas[i - 1] = 0.0 if diff <= 0 else (pred_positions[i] - v) / diff

        max_mapping = self.sorted_mappings[-1]
        v = sum(alphas[j] * (betas[j] - betas[j - 1]) for j in range(1, self.sigma))
        alphas[-1] = (self.sorted_mappings.shape[0] - 1) / (max_mapping - betas[-1]) if (max_mapping - betas[-1]) != 0 else 0.0
        cs = np.cumsum(alphas)
        if cs[-1] < 0:
            alphas[-1] = -cs[-2]
        return alphas

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        n_each_cell = self.sorted_mappings.shape[0] // self.sigma
        split_idxes = np.arange(self.sigma, dtype=np.int64) * n_each_cell
        self.betas = self.sorted_mappings[split_idxes].reshape(-1)
        self.init_betas = self.betas.copy()
        self.init_alphas = self.cal_init_alphas(self.init_betas)

        for k in range(200):
            betas = self.betas
            alphas, A = self.cal_alphas(betas)
            if A is None:
                break

            init_loss = self.cal_loss(A, alphas)
            if not self.check_if_alphas_and_betas_valid(alphas, betas):
                alphas = self.cal_alphas_with_monotone_constrain(betas, alphas)
                init_loss = self.cal_loss(A, alphas)
                if not self.check_if_alphas_and_betas_valid(alphas, betas):
                    self.alphas = None
                    self.betas = None

            G = -np.sign(A).T
            r = (A @ alphas).clip(0, self.positions.shape[0]) - self.positions
            K = np.diag(alphas)
            g = 2 * K @ (G @ r) / self.sorted_mappings.shape[0]
            Y = 2 * (K @ (G @ G.T) @ K) / self.sorted_mappings.shape[0]

            if np.linalg.cond(Y) < 1.0 / sys.float_info.epsilon:
                s = -np.linalg.inv(Y) @ g
                second_grad = True
            else:
                s = -g
                second_grad = False

            lr, loss, tmp_betas, tmp_alphas = self.lr_search(s, betas, init_loss)
            if lr > 0:
                self.betas = tmp_betas
                self.alphas = tmp_alphas
            else:
                if not second_grad:
                    break
                lr, loss, tmp_betas, tmp_alphas = self.lr_search(-g, betas, init_loss)
                if lr > 0:
                    self.betas = tmp_betas
                    self.alphas = tmp_alphas
                else:
                    break

    # ------------------------------------------------------------------
    # Validation / persistence
    # ------------------------------------------------------------------

    @staticmethod
    def check_if_alphas_and_betas_valid(alphas, betas):
        if (betas[1:] - betas[:-1]).min() <= 0:
            return False
        return np.cumsum(alphas).min() >= 0

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        a = self.alphas if self.alphas is not None else self.init_alphas
        b = self.betas if self.betas is not None else self.init_betas
        np.save(os.path.join(model_dir, 'alphas.npy'), a)
        np.save(os.path.join(model_dir, 'betas.npy'), b)

    def load(self, model_dir):
        self.alphas = np.load(os.path.join(model_dir, 'alphas.npy'))
        self.betas = np.load(os.path.join(model_dir, 'betas.npy'))
