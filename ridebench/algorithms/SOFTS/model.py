import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Embed import DataEmbedding_inverted
from .layers.Transformer_EncDec import Encoder, EncoderLayer


class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, inputs, *args, **kwargs):
        batch_size, channels, _ = inputs.shape

        combined_mean = F.gelu(self.gen1(inputs))
        combined_mean = self.gen2(combined_mean)

        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1).reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        combined_mean = torch.cat([inputs, combined_mean], dim=-1)
        combined_mean = F.gelu(self.gen3(combined_mean))
        output = self.gen4(combined_mean)
        return output, None


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = bool(configs.use_norm)

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, num_series = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :num_series]

        if self.use_norm:
            scale = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            bias = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out * scale + bias

        return dec_out

    def forward(self, x_enc, *args, **kwargs):
        return self.forecast(x_enc)[:, -self.pred_len :, :]
