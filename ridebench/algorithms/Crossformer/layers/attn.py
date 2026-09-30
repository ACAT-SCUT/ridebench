from math import sqrt

import torch
import torch.nn as nn
from einops import rearrange, repeat


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        _, _, _, dim = queries.shape
        scale = self.scale or 1.0 / sqrt(dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attn, values).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        heads = self.n_heads

        queries = self.query_projection(queries).view(batch_size, query_len, heads, -1)
        keys = self.key_projection(keys).view(batch_size, key_len, heads, -1)
        values = self.value_projection(values).view(batch_size, key_len, heads, -1)
        out = self.inner_attention(queries, keys, values).view(batch_size, query_len, -1)
        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.mlp1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.mlp2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        batch_size = x.shape[0]
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.mlp1(dim_in))
        dim_in = self.norm2(dim_in)

        dim_send = rearrange(dim_in, "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model", b=batch_size)
        batch_router = repeat(self.router, "seg_num factor d_model -> (repeat seg_num) factor d_model", repeat=batch_size)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.mlp2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        return rearrange(dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model", b=batch_size)
