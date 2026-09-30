from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import InferenceModel


class Model(InferenceModel):
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        timesfm_model_path: str,
        timesfm_device: str,
        timesfm_per_core_batch_size: int,
        timesfm_max_context: int,
        timesfm_max_horizon: int,
        timesfm_normalize_inputs: bool,
        timesfm_use_continuous_quantile_head: bool,
        timesfm_force_flip_invariance: bool,
        timesfm_infer_is_positive: bool,
        timesfm_fix_quantile_crossing: bool,
        timesfm_torch_compile: bool,
        timesfm_local_files_only: bool,
    ) -> None:
        if input_len <= 0:
            raise ValueError("input_len must be positive.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if timesfm_per_core_batch_size <= 0:
            raise ValueError("timesfm_per_core_batch_size must be positive.")

        self.input_len = input_len
        self.pred_len = pred_len
        self.timesfm_model_path = timesfm_model_path
        self.timesfm_device = timesfm_device
        self.timesfm_per_core_batch_size = timesfm_per_core_batch_size
        self.timesfm_max_context = max(timesfm_max_context, input_len) if timesfm_max_context > 0 else input_len
        self.timesfm_max_horizon = max(timesfm_max_horizon, pred_len) if timesfm_max_horizon > 0 else pred_len
        self.timesfm_normalize_inputs = timesfm_normalize_inputs
        self.timesfm_use_continuous_quantile_head = timesfm_use_continuous_quantile_head
        self.timesfm_force_flip_invariance = timesfm_force_flip_invariance
        self.timesfm_infer_is_positive = timesfm_infer_is_positive
        self.timesfm_fix_quantile_crossing = timesfm_fix_quantile_crossing
        self.timesfm_torch_compile = timesfm_torch_compile
        self.timesfm_local_files_only = timesfm_local_files_only
        self._pipeline = None

    def _resolve_device(self) -> str:
        if self.timesfm_device != "auto":
            return self.timesfm_device

        try:
            import torch
        except Exception:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _set_pipeline_device(self, pipeline) -> None:
        import torch

        device = self._resolve_device()
        if device == "cpu":
            pipeline.model.device = torch.device("cpu")
            pipeline.model.device_count = 1
            pipeline.model.to(pipeline.model.device)
            return

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("TimesFM2p5 requested CUDA but CUDA is not available.")
            pipeline.model.device = torch.device("cuda:0")
            pipeline.model.device_count = torch.cuda.device_count()
            pipeline.model.to(pipeline.model.device)

    def _build_forecast_config(self, timesfm):
        return timesfm.ForecastConfig(
            max_context=self.timesfm_max_context,
            max_horizon=self.timesfm_max_horizon,
            normalize_inputs=self.timesfm_normalize_inputs,
            per_core_batch_size=self.timesfm_per_core_batch_size,
            use_continuous_quantile_head=self.timesfm_use_continuous_quantile_head,
            force_flip_invariance=self.timesfm_force_flip_invariance,
            infer_is_positive=self.timesfm_infer_is_positive,
            fix_quantile_crossing=self.timesfm_fix_quantile_crossing,
            return_backcast=False,
        )

    def _build_pipeline(self):
        # Benchmark runtime policy: keep TimesFM's XReg helper on CPU so local
        # runs don't depend on an operational CUDA/JAX stack.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        try:
            import timesfm
        except Exception as exc:
            raise RuntimeError(
                "TimesFM2p5 requires the local timesfm package and its dependencies. "
                "Install them with `uv sync`."
            ) from exc

        model_ref = self.timesfm_model_path
        if not model_ref:
            raise RuntimeError(
                "TimesFM2p5 requires --timesfm_model_path or BENCH_TIMESFM2P5_MODEL "
                "to point to a local checkpoint or a valid model id."
            )

        path = Path(model_ref).expanduser()
        is_local_like = any(sep in model_ref for sep in ("/", "\\")) or model_ref.startswith(".")
        if is_local_like and not path.exists():
            raise RuntimeError(
                f"TimesFM2p5 expected a local checkpoint at {path}, but it does not exist. "
                "Download it first or pass a valid Hugging Face model id."
            )

        if path.is_dir() or path.is_file():
            pipeline = timesfm.TimesFM_2p5_200M_torch()
            self._set_pipeline_device(pipeline)
            weight_path = path / "model.safetensors" if path.is_dir() else path
            if not weight_path.exists():
                raise RuntimeError(
                    f"TimesFM2p5 expected local weights at {weight_path}, but the file does not exist."
                )
            pipeline.model.load_checkpoint(str(weight_path), torch_compile=self.timesfm_torch_compile)
        else:
            pipeline = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                model_ref,
                local_files_only=self.timesfm_local_files_only,
            )
            self._set_pipeline_device(pipeline)

        pipeline.compile(self._build_forecast_config(timesfm))
        return pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline

    def _iter_tasks(self, x_endo: np.ndarray):
        n_samples, _, x_dim = x_endo.shape
        for sample_idx in range(n_samples):
            for var_idx in range(x_dim):
                yield sample_idx, var_idx

    def fit_predict(
        self,
        x_endo: np.ndarray,
        x_exo_con: np.ndarray,
        x_exo_dis: np.ndarray,
        y_exo_con: np.ndarray,
        y_exo_dis: np.ndarray,
    ) -> np.ndarray:
        n_samples, _, x_dim = x_endo.shape
        inputs: list[np.ndarray] = []
        del x_exo_con, x_exo_dis, y_exo_con, y_exo_dis

        for sample_idx, var_idx in self._iter_tasks(x_endo):
            inputs.append(np.asarray(x_endo[sample_idx, :, var_idx], dtype=np.float32))

        point_forecast, _ = self.pipeline.forecast(self.pred_len, inputs)

        pred = point_forecast.reshape(n_samples, x_dim, self.pred_len).transpose(0, 2, 1)
        return pred.astype(x_endo.dtype, copy=False)
