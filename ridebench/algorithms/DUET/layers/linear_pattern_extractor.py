import torch
import torch.nn as nn

from .Autoformer_EncDec import SeriesDecomp


class LinearExtractor(nn.Module):
    def __init__(self, configs, individual=False):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.d_model
        self.decomposition = SeriesDecomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.enc_in = 1 if configs.CI else configs.enc_in

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(self.channels):
                seasonal = nn.Linear(self.seq_len, self.pred_len)
                trend = nn.Linear(self.seq_len, self.pred_len)
                seasonal.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                trend.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.linear_seasonal.append(seasonal)
                self.linear_trend.append(trend)
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            self.linear_seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.linear_trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

    def encoder(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, self.enc_in), device=x_enc.device)
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]
