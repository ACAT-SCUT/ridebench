import torch
from torch import nn

from ..covariate_adapter import build_covariate_adapter


class GatingBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.weight(x)


class ForecastMulti(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model,
        channel,
        t_ff,
        c_ff,
        t_dropout,
        c_dropout,
        embed_dropout,
    ):
        super().__init__()
        self.d_model = d_model
        self.channel = channel
        self.projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(embed_dropout),
        )
        self.global_token = nn.Parameter(torch.ones(1, channel, d_model))
        self.temporal_gating = GatingBlock(2 * d_model, t_ff, t_dropout)
        self.channel_gating = GatingBlock(2 * channel, c_ff, c_dropout)

    def forward(self, x):
        # x: [batch, channel, seq_len]
        batch_size = x.shape[0]
        embed = self.projection(x)

        global_token = self.global_token.expand(batch_size, -1, -1)
        temporal_input = torch.cat([embed, global_token], dim=-1)
        temporal_output = self.temporal_gating(temporal_input)

        local_embed = temporal_output[:, :, : self.d_model]
        global_embed = temporal_output[:, :, self.d_model :]

        channel_input = torch.cat([embed, global_embed], dim=1)
        channel_output = self.channel_gating(channel_input.permute(0, 2, 1))
        cross_embed = channel_output[:, :, self.channel :]

        return torch.cat([local_embed, cross_embed.permute(0, 2, 1)], dim=-1)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.t_ff = configs.t_ff
        self.c_ff = configs.c_ff
        self.norm = configs.usenorm
        self.embed_dropout = configs.embed_dropout
        self.head_dropout = configs.head_dropout
        self.t_dropout = configs.t_dropout
        self.c_dropout = configs.c_dropout
        self.feature = configs.features
        self.target_dim = configs.target_dim
        self.covariate_adapter = build_covariate_adapter(configs.covariate_spec)
        self.use_future_exog = bool(
            configs.use_future_exog and configs.covariate_spec.model_exog_dim > 0
        )
        self.input_steps = self.seq_len + self.pred_len if self.use_future_exog else self.seq_len
        self.channel = configs.enc_in

        if self.feature != "M":
            raise NotImplementedError("XLinear benchmark integration currently supports features='M' only.")

        self.backbone = ForecastMulti(
            self.input_steps,
            self.d_model,
            self.channel,
            self.t_ff,
            self.c_ff,
            self.t_dropout,
            self.c_dropout,
            self.embed_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(2 * self.d_model, self.pred_len),
        )

    def forecast_multi(self, x_enc):
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means = None
            stdev = None

        x_enc = x_enc.permute(0, 2, 1)
        encoded = self.backbone(x_enc)
        dec_out = self.head(encoded).permute(0, 2, 1)

        if self.norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out

    def forward(
        self,
        x_enc,
        x_exo_con=None,
        x_exo_dis=None,
        y_exo_con=None,
        y_exo_dis=None,
    ):
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

        return self.forecast_multi(x_enc)[..., : self.target_dim]
