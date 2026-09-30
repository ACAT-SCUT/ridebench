from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import InferenceModel


class Model(InferenceModel):
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        timers1_model_path: str,
        timers1_device: str,
        timers1_torch_dtype: str,
        timers1_batch_size: int,
        timers1_context_length: int,
        timers1_use_revin: bool,
        timers1_use_cache: bool,
        timers1_output_quantile: float,
        timers1_local_files_only: bool,
    ) -> None:
        if input_len <= 0:
            raise ValueError("input_len must be positive.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if timers1_batch_size <= 0:
            raise ValueError("timers1_batch_size must be positive.")
        if timers1_context_length < 0:
            raise ValueError("timers1_context_length must be non-negative.")

        self.input_len = input_len
        self.pred_len = pred_len
        self.timers1_model_path = timers1_model_path
        self.timers1_device = timers1_device
        self.timers1_torch_dtype = timers1_torch_dtype
        self.timers1_batch_size = timers1_batch_size
        self.timers1_context_length = timers1_context_length
        self.timers1_use_revin = timers1_use_revin
        self.timers1_use_cache = timers1_use_cache
        self.timers1_output_quantile = float(timers1_output_quantile)
        self.timers1_local_files_only = timers1_local_files_only
        self._model = None
        self._quantile_index = None
        self._max_context_length = None

    def _load_torch(self):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "Timer-S1 requires torch, transformers, accelerate, and huggingface_hub. "
                "Install them with `uv sync`."
            ) from exc
        return torch

    def _resolve_torch_dtype(self):
        torch = self._load_torch()
        if self.timers1_torch_dtype == "auto":
            return "auto"
        return getattr(torch, self.timers1_torch_dtype)

    def _resolve_load_kwargs(self) -> dict[str, object]:
        torch = self._load_torch()
        kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "local_files_only": self.timers1_local_files_only,
            "torch_dtype": self._resolve_torch_dtype(),
        }
        if self.timers1_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Timer-S1 requested CUDA but CUDA is not available.")
            kwargs["device_map"] = "auto"
        elif self.timers1_device == "auto":
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"
        return kwargs

    def _validate_local_checkpoint(self, local_path: Path) -> None:
        required_files = [
            local_path / "config.json",
            local_path / "modeling_TimerS1.py",
        ]
        missing = [str(path) for path in required_files if not path.exists()]
        has_weight = any(local_path.glob("*.safetensors")) or (local_path / "model.safetensors.index.json").exists()
        if not has_weight:
            missing.append(str(local_path / "*.safetensors"))
        if missing:
            raise RuntimeError(
                "Timer-S1 local checkpoint is incomplete. Missing: "
                f"{', '.join(missing)}. Upload the full Hugging Face snapshot directory."
            )

    def _build_model(self):
        try:
            from transformers import AutoModelForCausalLM
        except Exception as exc:
            raise RuntimeError(
                "Timer-S1 requires transformers~=4.57 and accelerate. Install them with `uv sync`."
            ) from exc

        model_ref = self.timers1_model_path
        if not model_ref:
            raise RuntimeError(
                "Timer-S1 requires --timers1_model_path or BENCH_TIMERS1_MODEL to point "
                "to a local checkpoint directory or a valid Hugging Face model id."
            )

        local_path = Path(model_ref).expanduser()
        is_local_like = any(sep in model_ref for sep in ("/", "\\")) or model_ref.startswith(".")
        if is_local_like:
            if not local_path.exists():
                raise RuntimeError(
                    f"Timer-S1 expected a local checkpoint at {local_path}, but it does not exist. "
                    "Download it first or pass a valid Hugging Face model id."
                )
            if not local_path.is_dir():
                raise RuntimeError(
                    f"Timer-S1 expects a checkpoint directory, but got file: {local_path}."
                )
            self._validate_local_checkpoint(local_path)

        model = AutoModelForCausalLM.from_pretrained(model_ref, **self._resolve_load_kwargs())
        if self.timers1_device == "cpu":
            model = model.to("cpu")
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = bool(self.timers1_use_cache)
        self._max_context_length = self._detect_max_context_length(model)
        self._quantile_index = self._resolve_quantile_index(model)
        return model

    @staticmethod
    def _detect_max_context_length(model) -> int:
        config = model.config
        for attr in ("context_length", "max_position_embeddings", "n_positions"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return 11520

    def _resolve_quantile_index(self, model) -> int:
        quantiles = (
            getattr(model.config, "quantiles", None)
            or getattr(model.config, "quantile_levels", None)
            or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        quantiles = [float(q) for q in quantiles]
        return min(range(len(quantiles)), key=lambda idx: abs(quantiles[idx] - self.timers1_output_quantile))

    @property
    def model(self):
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _model_device(self):
        try:
            return self.model.device
        except Exception:
            pass

        for parameter in self.model.parameters():
            return parameter.device
        raise RuntimeError("Timer-S1 model has no parameters to infer a runtime device from.")

    def _model_dtype(self):
        for parameter in self.model.parameters():
            return parameter.dtype
        try:
            return self.model.dtype
        except Exception as exc:
            raise RuntimeError("Timer-S1 model has no parameters to infer a runtime dtype from.") from exc

    def _effective_context_length(self) -> int:
        if self.timers1_context_length > 0:
            return min(self.timers1_context_length, self.input_len, self._max_context_length or self.input_len)
        return min(self.input_len, self._max_context_length or self.input_len)

    def fit_predict(
        self,
        x_endo: np.ndarray,
        x_exo_con: np.ndarray,
        x_exo_dis: np.ndarray,
        y_exo_con: np.ndarray,
        y_exo_dis: np.ndarray,
    ) -> np.ndarray:
        del x_exo_con, x_exo_dis, y_exo_con, y_exo_dis

        torch = self._load_torch()
        n_samples, _, x_dim = x_endo.shape
        context_length = self._effective_context_length()

        # Benchmark compatibility adaptation: Timer-S1 is used as a
        # single-target foundation model, so we flatten each endogenous channel
        # into an independent zero-shot task and reshape it back afterwards.
        contexts = (
            np.asarray(x_endo[:, -context_length:, :], dtype=np.float32)
            .transpose(0, 2, 1)
            .reshape(n_samples * x_dim, context_length)
        )

        outputs = []
        device = self._model_device()
        dtype = self._model_dtype()
        with torch.no_grad():
            for start in range(0, contexts.shape[0], self.timers1_batch_size):
                batch_contexts = torch.as_tensor(
                    contexts[start : start + self.timers1_batch_size],
                    device=device,
                    dtype=dtype,
                )
                forecast = self.model.generate(
                    batch_contexts,
                    max_new_tokens=self.pred_len,
                    revin=self.timers1_use_revin,
                )
                if not torch.is_tensor(forecast):
                    forecast = torch.as_tensor(forecast)
                forecast = forecast.detach().to("cpu")
                if forecast.dim() == 3:
                    forecast = forecast[:, self._quantile_index, :]
                elif forecast.dim() != 2:
                    raise RuntimeError(
                        f"Timer-S1 returned an unexpected forecast shape: {tuple(forecast.shape)}"
                    )
                outputs.append(forecast[:, : self.pred_len].numpy())

        pred = np.concatenate(outputs, axis=0).reshape(n_samples, x_dim, self.pred_len).transpose(0, 2, 1)
        return pred.astype(x_endo.dtype, copy=False)
