import torch.nn as nn
from einops import rearrange


class DSWEmbedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super().__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch_size, _, ts_dim = x.shape
        x_segment = rearrange(x, "b (seg_num seg_len) d -> (b d seg_num) seg_len", seg_len=self.seg_len)
        x_embed = self.linear(x_segment)
        return rearrange(x_embed, "(b d seg_num) d_model -> b d seg_num d_model", b=batch_size, d=ts_dim)
