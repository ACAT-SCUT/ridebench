from ..config import InferenceModel
import numpy as np


class Model(InferenceModel):
    def __init__(self, pred_len: int, sp: int) -> None:
        self.pred_len = pred_len
        self.sp = sp

    def _fit_predict(self, x_endo: np.ndarray, exo: np.ndarray) -> np.ndarray:
        input_len = x_endo.shape[1]
        if self.sp <= 0:
            raise ValueError(f"seasonal period must be positive, got sp={self.sp}")
        if self.sp > input_len:
            raise ValueError(f"seasonal period sp={self.sp} exceeds input length {input_len}")

        # Benchmark SeasonalNaive is a direct seasonal copy from the input window.
        indices = input_len - self.sp + (np.arange(self.pred_len) % self.sp)
        return x_endo[:, indices, :]
