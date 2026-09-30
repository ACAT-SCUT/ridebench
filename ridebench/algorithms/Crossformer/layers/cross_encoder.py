from math import ceil

import torch
import torch.nn as nn

from .attn import TwoStageAttentionLayer


class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        _, _, seg_num, _ = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = [x[:, :, i::self.win_size, :] for i in range(self.win_size)]
        x = torch.cat(seg_to_merge, dim=-1)
        return self.linear_trans(self.norm(x))


class ScaleBlock(nn.Module):
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, factor=10):
        super().__init__()
        self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm) if win_size > 1 else None
        self.encode_layers = nn.ModuleList(
            [TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout) for _ in range(depth)]
        )

    def forward(self, x):
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num=10, factor=10):
        super().__init__()
        self.encode_blocks = nn.ModuleList()
        self.encode_blocks.append(
            ScaleBlock(1, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num, factor)
        )
        for i in range(1, e_blocks):
            self.encode_blocks.append(
                ScaleBlock(
                    win_size,
                    d_model,
                    n_heads,
                    d_ff,
                    block_depth,
                    dropout,
                    ceil(in_seg_num / win_size ** i),
                    factor,
                )
            )

    def forward(self, x):
        encode_x = [x]
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)
        return encode_x
