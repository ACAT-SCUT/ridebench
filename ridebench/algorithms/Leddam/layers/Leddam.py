import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn_scores = torch.matmul(q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        return torch.matmul(attn_weights, v)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads)
        self.w_k = nn.Linear(d_model, d_k * n_heads)
        self.w_v = nn.Linear(d_model, d_v * n_heads)
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, q: torch.Tensor, k: torch.Tensor | None = None, v: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = q.size(0)
        if k is None:
            k = q
        if v is None:
            v = q
        q_proj = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_proj = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        output = self.sdp_attn(q_proj, k_proj, v_proj)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        return self.to_out(output)


class AutoAttention(nn.Module):
    def __init__(self, period: int, d_model: int, proj_dropout: float = 0.0):
        super().__init__()
        self.period = period
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.scale = nn.Parameter(torch.tensor(d_model ** -0.5), requires_grad=False)
        self.out_projector = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(proj_dropout))

    def auto_attention(self, x: torch.Tensor) -> torch.Tensor:
        query = self.w_q(x[:, :, 0, :].unsqueeze(-2))
        keys = self.w_k(x)
        values = self.w_v(x)
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        attn_scores = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_scores, values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        time_steps = x.size(-1)
        rotated_sequences = [x]
        index = time_steps // self.period - 1 if time_steps % self.period == 0 else time_steps // self.period
        for i in range(index):
            end = (i + 1) * self.period
            rotated_sequences.append(torch.cat([x[:, :, end:], x[:, :, :end]], dim=-1))
        output = torch.stack(rotated_sequences, dim=-1).permute(0, 1, 3, 2)
        output = self.auto_attention(output).squeeze(-2)
        return self.out_projector(output).permute(0, 2, 1)


class LearnableDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode="replicate",
            bias=True,
        )
        kernel_size_half = kernel_size // 2
        sigma = 1.0
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        channels = torch.split(x, 1, dim=1)
        output = torch.cat([self.conv(channel) for channel in channels], dim=1)
        return output.permute(0, 2, 1)


class ChannelAttentionBlock(nn.Module):
    def __init__(self, enc_in: int, d_model: int, dropout: float):
        super().__init__()
        self.channel_attn_norm = nn.BatchNorm1d(enc_in)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.channel_attn = MultiheadAttention(d_model=d_model, n_heads=1, proj_dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        x = residual.permute(0, 2, 1)
        x = self.channel_attn_norm(self.channel_attn(x) + x)
        x = self.ffn_norm(self.ffn(x) + x)
        return x.permute(0, 2, 1)


class AutoAttentionBlock(nn.Module):
    def __init__(self, enc_in: int, d_model: int, dropout: float):
        super().__init__()
        self.auto_attn_norm = nn.BatchNorm1d(enc_in)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.auto_attn = AutoAttention(period=64, d_model=d_model, proj_dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        x = self.auto_attn_norm((self.auto_attn(residual) + residual).permute(0, 2, 1))
        x = self.ffn_norm(self.ffn(x) + x)
        return x.permute(0, 2, 1)


def sincos_pos_encoding(q_len: int, d_model: int, normalize: bool = True) -> torch.Tensor:
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def coord2d_pos_encoding(q_len: int, d_model: int, exponential: bool = False, normalize: bool = True, eps: float = 1e-3) -> torch.Tensor:
    exponent = 0.5 if exponential else 1.0
    for _ in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** exponent) * (
            torch.linspace(0, 1, d_model).reshape(1, -1) ** exponent
        ) - 1
        if abs(cpe.mean()) <= eps:
            break
        exponent += 0.001 if cpe.mean() > eps else -0.001
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def coord1d_pos_encoding(q_len: int, exponential: bool = False, normalize: bool = True) -> torch.Tensor:
    exponent = 0.5 if exponential else 1.0
    cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** exponent) - 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe: str | None, learn_pe: bool, q_len: int, d_model: int) -> nn.Parameter:
    if pe is None or pe == "no":
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        w_pos = torch.empty((q_len, 1))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "zeros":
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe in {"normal", "gauss"}:
        w_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        w_pos = coord1d_pos_encoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        w_pos = coord1d_pos_encoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        w_pos = coord2d_pos_encoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == "exp2d":
        w_pos = coord2d_pos_encoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == "sincos":
        w_pos = sincos_pos_encoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid positional encoding type.")
    return nn.Parameter(w_pos, requires_grad=learn_pe)


class DataEmbedding(nn.Module):
    def __init__(self, pe_type: str, seq_len: int, d_model: int, c_in: int, dropout: float):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_embedding = positional_encoding(pe=pe_type, learn_pe=True, q_len=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding)


class Leddam(nn.Module):
    def __init__(self, enc_in: int, seq_len: int, d_model: int, dropout: float, pe_type: str, kernel_size: int, n_layers: int = 3):
        super().__init__()
        self.learnable_decomposition = LearnableDecomposition(kernel_size=kernel_size)
        self.channel_attn_blocks = nn.ModuleList(
            [ChannelAttentionBlock(enc_in, d_model, dropout) for _ in range(n_layers)]
        )
        self.auto_attn_blocks = nn.ModuleList(
            [AutoAttentionBlock(enc_in, d_model, dropout) for _ in range(n_layers)]
        )
        self.position_embedder = DataEmbedding(pe_type=pe_type, seq_len=seq_len, d_model=d_model, c_in=enc_in, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.position_embedder(x.permute(0, 2, 1)).permute(0, 2, 1)
        main = self.learnable_decomposition(x)
        residual = x - main
        res_auto = residual
        res_channel = residual
        for block in self.auto_attn_blocks:
            res_auto = block(res_auto)
        for block in self.channel_attn_blocks:
            res_channel = block(res_channel)
        return res_auto + res_channel, main
