import torch
import torch.nn as nn
from einops import repeat

from .layers.cross_decoder import Decoder
from .layers.cross_embed import DSWEmbedding
from .layers.cross_encoder import Encoder


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.data_dim = configs.data_dim
        self.in_len = configs.in_len
        self.out_len = configs.out_len
        self.seg_len = configs.seg_len
        self.merge_win = configs.win_size
        self.baseline = configs.baseline

        self.pad_in_len = ((self.in_len + self.seg_len - 1) // self.seg_len) * self.seg_len
        self.pad_out_len = ((self.out_len + self.seg_len - 1) // self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        self.enc_value_embedding = DSWEmbedding(self.seg_len, configs.d_model)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, self.pad_in_len // self.seg_len, configs.d_model)
        )
        self.pre_norm = nn.LayerNorm(configs.d_model)

        self.encoder = Encoder(
            configs.e_layers,
            configs.win_size,
            configs.d_model,
            configs.n_heads,
            configs.d_ff,
            block_depth=1,
            dropout=configs.dropout,
            in_seg_num=self.pad_in_len // self.seg_len,
            factor=configs.factor,
        )

        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, self.pad_out_len // self.seg_len, configs.d_model)
        )
        self.decoder = Decoder(
            self.seg_len,
            configs.e_layers + 1,
            configs.d_model,
            configs.n_heads,
            configs.d_ff,
            configs.dropout,
            out_seg_num=self.pad_out_len // self.seg_len,
            factor=configs.factor,
        )

    def forward(self, x_seq, *args, **kwargs):
        base = x_seq.mean(dim=1, keepdim=True) if self.baseline else 0
        batch_size = x_seq.shape[0]
        if self.in_len_add != 0:
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq = self.pre_norm(x_seq + self.enc_pos_embedding)

        enc_out = self.encoder(x_seq)
        dec_in = repeat(self.dec_pos_embedding, "b ts_d l d -> (repeat b) ts_d l d", repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        return base + predict_y[:, : self.out_len, :]
