from __future__ import annotations

import numpy as np

from ..config import InferenceModel


class Model(InferenceModel):
    def __init__(self, pred_len: int, sp: int) -> None:
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if sp <= 0:
            raise ValueError("sp must be positive.")
        self.pred_len = pred_len
        self.sp = sp

    def fit_predict(
        self,
        x_endo: np.ndarray,
        x_exo_con: np.ndarray,
        x_exo_dis: np.ndarray,
        y_exo_con: np.ndarray,
        y_exo_dis: np.ndarray,
    ) -> np.ndarray:
        del x_exo_con, x_exo_dis, y_exo_con, y_exo_dis
        n_samples, input_len, x_dim = x_endo.shape
        if self.sp > input_len:
            raise ValueError(f"seasonal period sp={self.sp} exceeds input length {input_len}")

        pred = np.empty((n_samples, self.pred_len, x_dim), dtype=x_endo.dtype)
        for offset in range(self.pred_len):
            seasonal_offset = (input_len - self.sp + offset) % self.sp
            indices = np.arange(seasonal_offset, input_len, self.sp)
            pred[:, offset, :] = np.mean(x_endo[:, indices, :], axis=1, dtype=np.float64).astype(
                x_endo.dtype,
                copy=False,
            )
        return pred
