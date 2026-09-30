import math

import torch
import torch.nn as nn

from ..covariate_adapter import build_covariate_adapter


class Patch_Embedding(nn.Module):
    def __init__(self, seq_len, patch_num, patch_len, d_model, d_ff, variate_num):
        super(Patch_Embedding, self).__init__()
        self.pad_num = patch_num * patch_len - seq_len
        self.patch_len = patch_len
        self.linear = nn.Sequential(
            nn.LayerNorm([variate_num, patch_num, patch_len]),
            nn.Linear(patch_len, d_ff),
            nn.LayerNorm([variate_num, patch_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm([variate_num, patch_num, d_model]),
            nn.ReLU(),
        )

    def forward(self, x):
        x = nn.functional.pad(x, (0, self.pad_num))
        x = x.unfold(2, self.patch_len, self.patch_len)
        x = self.linear(x)
        return x


class De_Patch_Embedding(nn.Module):
    def __init__(self, pred_len, patch_num, d_model, d_ff, variate_num):
        super(De_Patch_Embedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(patch_num * d_model, d_ff),
            nn.LayerNorm([variate_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, pred_len),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.ms = configs.features == "MS"
        self.EPS = 1e-5
        self.pred_len = configs.pred_len
        self.target_dim = configs.target_dim
        self.covariate_adapter = build_covariate_adapter(configs.covariate_spec)
        self.use_future_exog = bool(
            configs.use_future_exog and configs.covariate_spec.model_exog_dim > 0
        )
        patch_len = configs.patch_len
        patch_num = math.ceil(configs.seq_len / patch_len)
        variate_num = 1 if self.ms else configs.dec_in
        self.alpha = nn.Parameter(torch.ones([1]) * configs.alpha)
        self.beta = nn.Parameter(torch.ones([1]) * configs.beta)
        self.correlation_embedding = nn.Conv1d(configs.dec_in, variate_num, 3, padding="same")
        self.value_embedding = Patch_Embedding(
            configs.seq_len,
            patch_num,
            patch_len,
            configs.d_model,
            configs.d_ff,
            variate_num,
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, variate_num, patch_num, configs.d_model))
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.head = De_Patch_Embedding(configs.pred_len, patch_num, configs.d_model, configs.d_ff, variate_num)

    def forecast(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        mean = torch.mean(x_obj, dim=-1, keepdim=True)
        std = torch.std(x_obj, dim=-1, keepdim=True)
        x_enc = (x_enc - torch.mean(x_enc, dim=-1, keepdim=True)) / (
            torch.std(x_enc, dim=-1, keepdim=True) + self.EPS
        )
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        x_obj = self.alpha * x_obj + (1 - self.alpha) * self.correlation_embedding(x_enc)
        x_obj = self.beta * self.value_embedding(x_obj) + (1 - self.beta) * self.pos_embedding
        y_out = self.head(x_obj)
        y_out = y_out * std + mean
        y_out = y_out.permute(0, 2, 1)
        return y_out

    def forward(self, x_enc, x_exo_con=None, x_exo_dis=None, y_exo_con=None, y_exo_dis=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            # Benchmark adaptation: CrossLinear still consumes one multivariate
            # tensor only. We therefore append adapted exogenous channels on
            # the feature axis, and when future exogenous inputs are enabled we
            # extend the consumed time axis with zero-padded future endogenous
            # placeholders plus future-known exogenous channels.
            exog_history, exog_future = self.covariate_adapter(
                x_exo_con,
                x_exo_dis,
                y_exo_con,
                y_exo_dis,
                dtype=x_enc.dtype,
            )
            if self.use_future_exog:
                future_endo_padding = x_enc.new_zeros((x_enc.shape[0], self.pred_len, x_enc.shape[-1]))
                x_enc = torch.cat([x_enc, future_endo_padding], dim=1)
                if exog_history is not None:
                    exog_history = torch.cat([exog_history, exog_future], dim=1)
            if exog_history is not None:
                x_enc = torch.cat([x_enc, exog_history], dim=-1)
            return self.forecast(x_enc)[..., : self.target_dim]
        raise NotImplementedError(f"Task {self.task_name} is not supported.")
