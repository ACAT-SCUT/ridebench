from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import InferenceModel


class Model(InferenceModel):
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        batch_size: int,
        moirai_model_path: str,
        moirai_device: str,
        moirai_context_length: int,
        moirai_local_files_only: bool,
    ) -> None:
        if input_len <= 0:
            raise ValueError("input_len must be positive.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")

        self.input_len = input_len
        self.pred_len = pred_len
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size
        self.moirai_model_path = moirai_model_path
        self.moirai_device = moirai_device
        requested_context_length = moirai_context_length if moirai_context_length > 0 else input_len
        self.moirai_context_length = max(1, requested_context_length)
        self.moirai_local_files_only = moirai_local_files_only
        self._model = None
        self._median_index = None
        self._x_dim = None

    def _resolve_device(self) -> str:
        if self.moirai_device != "auto":
            return self.moirai_device

        try:
            import torch
        except Exception:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_module(self):
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except Exception as exc:
            raise RuntimeError(
                "Moirai2 requires the uni2ts package and its dependencies. Install them with `uv sync`."
            ) from exc

        model_ref = self.moirai_model_path
        if not model_ref:
            raise RuntimeError(
                "Moirai2 requires --moirai_model_path or BENCH_MOIRAI2_MODEL to point "
                "to a local checkpoint directory or a valid model id."
            )

        local_path = Path(model_ref).expanduser()
        is_local_like = any(sep in model_ref for sep in ("/", "\\")) or model_ref.startswith(".")
        if is_local_like and not local_path.exists():
            raise RuntimeError(
                f"Moirai2 expected a local checkpoint at {local_path}, but it does not exist. "
                "Download it first or pass a valid Hugging Face model id."
            )

        module = Moirai2Module.from_pretrained(
            model_ref,
            local_files_only=self.moirai_local_files_only,
        )
        model = Moirai2Forecast(
            module=module,
            prediction_length=self.pred_len,
            context_length=self.moirai_context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self._resolve_device())
        model.eval()
        self._median_index = min(
            range(len(module.quantile_levels)),
            key=lambda idx: abs(float(module.quantile_levels[idx]) - 0.5),
        )
        return model

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_module()
        return self._model

    def fit_predict(
        self,
        x_endo: np.ndarray,
        x_exo_con: np.ndarray,
        x_exo_dis: np.ndarray,
        y_exo_con: np.ndarray,
        y_exo_dis: np.ndarray,
    ) -> np.ndarray:
        n_samples, _, x_dim = x_endo.shape
        self._x_dim = x_dim
        past_target = np.stack(
            [
                np.asarray(
                    x_endo[sample_idx, -self.moirai_context_length :, var_idx],
                    dtype=np.float32,
                )
                for sample_idx in range(n_samples)
                for var_idx in range(x_dim)
            ],
            axis=0,
        )[:, :, np.newaxis]
        inputs = {
            "past_target": torch.as_tensor(
                np.nan_to_num(past_target, nan=0.0),
                device=self.model.device,
                dtype=torch.float32,
            ),
            "past_observed_target": torch.as_tensor(
                np.isfinite(past_target),
                device=self.model.device,
                dtype=torch.bool,
            ),
            "past_is_pad": torch.zeros(
                (past_target.shape[0], self.moirai_context_length),
                device=self.model.device,
                dtype=torch.bool,
            ),
        }

        outputs = []
        with torch.no_grad():
            for start in range(0, past_target.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_inputs = {
                    "past_target": inputs["past_target"][start:end],
                    "past_observed_target": inputs["past_observed_target"][start:end],
                    "past_is_pad": inputs["past_is_pad"][start:end],
                }
                forecast = self.model(**batch_inputs).detach().cpu().numpy()
                outputs.append(forecast)
        pred = np.asarray(np.concatenate(outputs, axis=0)[:, self._median_index, :], dtype=x_endo.dtype)
        pred = pred.reshape(n_samples, x_dim, self.pred_len).transpose(0, 2, 1)
        return pred
