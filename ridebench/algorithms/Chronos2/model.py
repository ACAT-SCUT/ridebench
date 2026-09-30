from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import InferenceModel


class Model(InferenceModel):
    def __init__(
        self,
        pred_len: int,
        use_exog: bool,
        chronos_model_path: str,
        chronos_device_map: str,
        chronos_torch_dtype: str,
        chronos_batch_size: int,
        chronos_context_length: int,
        chronos_cross_learning: bool,
        chronos_local_files_only: bool,
        exo_con_names: list[str],
        exo_dis_names: list[str],
    ) -> None:
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if chronos_batch_size <= 0:
            raise ValueError("chronos_batch_size must be positive.")

        self.pred_len = pred_len
        self.use_exog = use_exog
        self.chronos_model_path = chronos_model_path
        self.chronos_device_map = chronos_device_map
        self.chronos_torch_dtype = chronos_torch_dtype
        self.chronos_batch_size = chronos_batch_size
        self.chronos_context_length = chronos_context_length
        self.chronos_cross_learning = chronos_cross_learning
        self.chronos_local_files_only = chronos_local_files_only
        self.exo_con_names = exo_con_names
        self.exo_dis_names = exo_dis_names
        self._pipeline = None

    def _resolve_device_map(self) -> str:
        if self.chronos_device_map != "auto":
            return self.chronos_device_map

        try:
            import torch
        except Exception:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _pipeline_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "device_map": self._resolve_device_map(),
            "local_files_only": self.chronos_local_files_only,
        }
        if self.chronos_torch_dtype != "auto":
            kwargs["torch_dtype"] = self.chronos_torch_dtype
        return kwargs

    def _build_pipeline(self):
        try:
            from chronos import BaseChronosPipeline
        except Exception as exc:
            raise RuntimeError(
                "Chronos2 requires chronos-forecasting and its dependencies. "
                "Install them with `uv sync`."
            ) from exc

        model_ref = self.chronos_model_path
        if not model_ref:
            raise RuntimeError(
                "Chronos2 requires --chronos_model_path or BENCH_CHRONOS2_MODEL to point "
                "to a local checkpoint directory or a valid model id."
            )

        local_path = Path(model_ref).expanduser()
        is_local_like = any(sep in model_ref for sep in ("/", "\\")) or model_ref.startswith(".")
        if is_local_like and not local_path.exists():
            raise RuntimeError(
                f"Chronos2 expected a local checkpoint at {local_path}, but it does not exist. "
                "Download it first or pass a valid Hugging Face model id."
            )
        return BaseChronosPipeline.from_pretrained(model_ref, **self._pipeline_kwargs())

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline

    @staticmethod
    def _as_numeric_feature(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _as_categorical_feature(values: np.ndarray) -> np.ndarray:
        return np.asarray(values).astype(str)

    def _build_task(
        self,
        x_endo: np.ndarray,
        x_exo_con: np.ndarray,
        x_exo_dis: np.ndarray,
        y_exo_con: np.ndarray,
        y_exo_dis: np.ndarray,
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        task: dict[str, np.ndarray | dict[str, np.ndarray]] = {
            "target": np.asarray(x_endo.T, dtype=np.float32),
        }
        if not self.use_exog:
            return task

        past_covariates: dict[str, np.ndarray] = {}
        future_covariates: dict[str, np.ndarray] = {}

        for idx, name in enumerate(self.exo_con_names[: x_exo_con.shape[-1]]):
            past_covariates[name] = self._as_numeric_feature(x_exo_con[:, idx])
            future_covariates[name] = self._as_numeric_feature(y_exo_con[:, idx])

        for idx, name in enumerate(self.exo_dis_names[: x_exo_dis.shape[-1]]):
            feature_name = f"{name}_cat"
            past_covariates[feature_name] = self._as_categorical_feature(x_exo_dis[:, idx])
            future_covariates[feature_name] = self._as_categorical_feature(y_exo_dis[:, idx])

        if past_covariates:
            task["past_covariates"] = past_covariates
        if future_covariates:
            task["future_covariates"] = future_covariates
        return task

    def fit_predict(
        self,
        x_endo: np.ndarray,
        x_exo_con: np.ndarray,
        x_exo_dis: np.ndarray,
        y_exo_con: np.ndarray,
        y_exo_dis: np.ndarray,
    ) -> np.ndarray:
        tasks = [
            self._build_task(
                x_endo=x_endo[i],
                x_exo_con=x_exo_con[i],
                x_exo_dis=x_exo_dis[i],
                y_exo_con=y_exo_con[i],
                y_exo_dis=y_exo_dis[i],
            )
            for i in range(x_endo.shape[0])
        ]

        _, mean = self.pipeline.predict_quantiles(
            tasks,
            prediction_length=self.pred_len,
            quantile_levels=[0.5],
            batch_size=self.chronos_batch_size,
            context_length=self.chronos_context_length or None,
            cross_learning=self.chronos_cross_learning,
            limit_prediction_length=False,
        )
        pred = np.stack([forecast.numpy(force=True).T for forecast in mean], axis=0)
        return pred.astype(x_endo.dtype, copy=False)
