import torch.nn as nn
from einops import rearrange

from .attn import AttentionLayer, TwoStageAttentionLayer


class DecoderLayer(nn.Module):
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num=10, factor=10):
        super().__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, d_ff, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch_size = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, "b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model")
        cross = rearrange(cross, "b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model")
        x = x + self.dropout(self.cross_attention(x, cross, cross))
        y = x = self.norm1(x)
        dec_output = self.norm2(x + self.mlp1(y))
        dec_output = rearrange(dec_output, "(b ts_d) seg_num d_model -> b ts_d seg_num d_model", b=batch_size)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, "b out_d seg_num seg_len -> b (out_d seg_num) seg_len")
        return dec_output, layer_predict


class Decoder(nn.Module):
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout, router=False, out_seg_num=10, factor=10):
        super().__init__()
        self.router = router
        self.decode_layers = nn.ModuleList(
            [DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, out_seg_num, factor) for _ in range(d_layers)]
        )

    def forward(self, x, cross):
        final_predict = None
        ts_dim = x.shape[1]
        for i, layer in enumerate(self.decode_layers):
            x, layer_predict = layer(x, cross[i])
            final_predict = layer_predict if final_predict is None else final_predict + layer_predict
        return rearrange(final_predict, "b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d", out_d=ts_dim)
