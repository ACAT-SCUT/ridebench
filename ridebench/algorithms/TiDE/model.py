import torch
import torch.nn as nn
import torch.nn.functional as F
from ..covariate_adapter import build_covariate_adapter


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = nn.Identity() if output_dim == 1 else LayerNorm(output_dim, bias=bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


class TiDECore(nn.Module):
    def __init__(self, configs, bias=True):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.d_model
        self.res_hidden = configs.d_model
        self.encoder_num = configs.e_layers
        self.decoder_num = configs.d_layers
        self.freq = configs.freq
        self.feature_encode_dim = configs.feature_encode_dim
        self.decode_dim = configs.c_out
        self.temporal_decoder_hidden = configs.d_ff
        self.use_future_exog = configs.use_future_exog
        dropout = configs.dropout

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        if self.use_future_exog:
            self.feature_dim = configs.covariate_dim
        else:
            self.feature_dim = freq_map[self.freq]

        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        self.feature_encoder = ResBlock(
            self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, bias
        )
        self.encoders = nn.Sequential(
            ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias),
            *(
                [
                    ResBlock(
                        self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias
                    )
                ]
                * (self.encoder_num - 1)
            )
        )
        self.decoders = nn.Sequential(
            *(
                [
                    ResBlock(
                        self.hidden_dim,
                        self.res_hidden,
                        self.hidden_dim,
                        dropout,
                        bias,
                    )
                ]
                * (self.decoder_num - 1)
            ),
            ResBlock(
                self.hidden_dim,
                self.res_hidden,
                self.decode_dim * self.pred_len,
                dropout,
                bias,
            ),
        )
        self.temporal_decoder = ResBlock(
            self.decode_dim + self.feature_encode_dim,
            self.temporal_decoder_hidden,
            1,
            dropout,
            bias,
        )
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)

    def forecast(self, x_enc, batch_y_mark):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        feature = self.feature_encoder(batch_y_mark)
        hidden = self.encoders(
            torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1)
        )
        decoded = self.decoders(hidden).reshape(
            hidden.shape[0], self.pred_len, self.decode_dim
        )
        dec_out = (
            self.temporal_decoder(torch.cat([feature[:, self.seq_len :], decoded], dim=-1))
            .squeeze(-1)
            + self.residual_proj(x_enc)
        )

        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, batch_y_mark=None, mask=None):
        if self.task_name not in ["long_term_forecast", "short_term_forecast"]:
            raise NotImplementedError("This reproduction supports forecasting only.")

        if not self.use_future_exog:
            if x_mark_enc is None or batch_y_mark is None:
                raise ValueError("x_mark_enc and batch_y_mark are required when use_future_exog=False.")
            all_marks = torch.cat([x_mark_enc, batch_y_mark[:, -self.pred_len :, :]], dim=1)
            dec_out = torch.stack(
                [self.forecast(x_enc[:, :, feature], all_marks) for feature in range(x_enc.shape[-1])],
                dim=-1,
            )
            return dec_out

        if x_dec is None:
            raise ValueError("x_dec (future exogenous variables) is required when use_future_exog=True.")

        series_dim = x_enc.shape[-1] - x_dec.shape[-1]
        all_marks = torch.cat([x_enc[:, :, series_dim:], x_dec], dim=1).to(dtype=torch.float32)
        dec_out = torch.stack(
            [self.forecast(x_enc[:, :, feature], all_marks) for feature in range(series_dim)],
            dim=-1,
        )
        return dec_out


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.core = TiDECore(config)
        self.covariate_spec = config.covariate_spec
        self.covariate_adapter = build_covariate_adapter(self.covariate_spec)

    def forward(
        self,
        input,
        x_exo_con=None,
        x_exo_dis=None,
        y_exo_con=None,
        y_exo_dis=None,
    ):
        # Benchmark adaptation: TiDE upstream expects dense float covariates,
        # so benchmark discrete IDs are expanded before entering the original
        # covariate encoder path.
        exog_history, exog_future = self.covariate_adapter(
            x_exo_con,
            x_exo_dis,
            y_exo_con,
            y_exo_dis,
            dtype=input.dtype,
        )
        x_enc = torch.cat([input, exog_history], dim=-1)

        if self.core.use_future_exog:
            return self.core(x_enc=x_enc, x_dec=exog_future)

        all_marks = torch.cat([exog_history, exog_future], dim=1)
        return self.core(
            x_enc=input,
            x_mark_enc=exog_history,
            batch_y_mark=all_marks,
        )
