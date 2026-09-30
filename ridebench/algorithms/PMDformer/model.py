import torch
import torch.nn as nn
from einops import rearrange

from .layers.SelfAttention_Family import SelfAttention, TrendRestorationAttention
from .layers.Transformer_Enc import Encoder_PVA, Encoder_TRA, EncoderLayer_PVA, EncoderLayer_TRA


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.e_layers = configs.e_layers
        self.use_norm = configs.use_norm
        self.patch_size = configs.patch_size
        self.stride = configs.patch_size

        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.Embedding_layer = nn.Linear(self.patch_size, configs.d_model)
        self.Predicting_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        self.Encoder_TRA = Encoder_TRA(
            [
                EncoderLayer_TRA(
                    TrendRestorationAttention(d_model=configs.d_model, n_heads=configs.n_heads, dropout=configs.dropout),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.Encoder_PVA = Encoder_PVA(
            [
                EncoderLayer_PVA(
                    SelfAttention(d_model=configs.d_model, n_heads=configs.n_heads, dropout=configs.dropout),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                )
                for _ in range(configs.v_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.patch_pos = nn.Parameter(torch.zeros(self.patch_num, configs.d_model))

    def forward(self, x_endo, x_exo_con=None, x_exo_dis=None, y_exo_con=None, y_exo_dis=None, mask=None):
        x = x_endo
        B, _, N = x.shape

        x = rearrange(x, "b l n -> b n l")
        if self.use_norm:
            means = x.mean(-1, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, "b n p s -> (b n) p s")

        patch_means = x.mean(-1, keepdim=True)
        x = x - patch_means

        x = self.Embedding_layer(x)
        x = x + self.patch_pos.unsqueeze(0)

        x_main = x[:, :-1, :]
        x_last = x[:, -1:, :]
        x_last = rearrange(x_last, "(b n) p d -> (b p) n d", b=B, n=N)
        x_last, _ = self.Encoder_PVA(x_last)
        x_last = rearrange(x_last, "(b p) n d -> (b n) p d", b=B, n=N)
        x = torch.cat([x_main, x_last], dim=1)

        x, _ = self.Encoder_TRA(x, patch_means)
        x = x + patch_means
        x = rearrange(x, "(b n) p d -> b n (p d)", b=B, n=N)
        x = self.Predicting_layer(x)

        if self.use_norm:
            x = x * stdev + means

        return rearrange(x, "b n t -> b t n")
