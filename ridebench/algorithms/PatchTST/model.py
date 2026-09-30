import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    PatchTST (simple, channel-independent):
      input : x_endo [B, seq_len, C]
      output: y_hat  [B, pred_len, C]
    Exogenous inputs are ignored.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.patch_len = int(configs.patch_len)
        self.stride = int(configs.stride)

        self.d_model = int(configs.d_model)
        self.n_heads = int(configs.n_heads)
        self.e_layers = int(configs.e_layers)
        self.d_ff = int(configs.d_ff)

        self.dropout = float(configs.dropout)
        self.head_dropout = float(configs.head_dropout)
        self.revin = int(getattr(configs, "revin", 0))
        self.eps = 1e-5

        assert self.patch_len > 0
        assert self.stride > 0

        # ---- compute number of patches and pad length (to keep fixed head size) ----
        L = self.seq_len
        if L < self.patch_len:
            self.n_patches = 1
            self.pad_len = self.patch_len - L
        else:
            self.n_patches = 1 + math.ceil((L - self.patch_len) / self.stride)
            total_len = (self.n_patches - 1) * self.stride + self.patch_len
            self.pad_len = max(0, total_len - L)

        # ---- patch embedding ----
        self.patch_proj = nn.Linear(self.patch_len, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.in_drop = nn.Dropout(self.dropout)

        # ---- transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=F.gelu,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)
        self.norm = nn.LayerNorm(self.d_model)

        # ---- forecasting head: flatten all patch tokens ----
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.n_patches * self.d_model, self.pred_len),
        )

    def _norm(self, x_bc: torch.Tensor):
        # x_bc: [BC, L]
        mean = x_bc.mean(dim=1, keepdim=True)
        std = x_bc.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x_bc - mean) / std, mean, std

    def forward(self, x, *args, **kwargs):
        # x: [B, L, C]
        B, L, C = x.shape

        # channel-independent: treat each channel as a separate sample
        x_bc = x.permute(0, 2, 1).contiguous().view(B * C, L)  # [BC, L]

        if self.pad_len > 0:
            x_bc = F.pad(x_bc, (0, self.pad_len))  # pad to fixed length

        if self.revin:
            x_bc, mean, std = self._norm(x_bc)

        patches = x_bc.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [BC, n_patches, patch_len]

        # safety: align n_patches if edge cases happen
        if patches.size(1) != self.n_patches:
            if patches.size(1) > self.n_patches:
                patches = patches[:, :self.n_patches, :]
            else:
                pad_n = self.n_patches - patches.size(1)
                patches = torch.cat([patches, patches.new_zeros(patches.size(0), pad_n, patches.size(2))], dim=1)

        tok = self.patch_proj(patches) + self.pos_embed  # [BC, n_patches, d_model]
        tok = self.in_drop(tok)

        tok = self.encoder(tok)
        tok = self.norm(tok)

        y_bc = self.head(tok)  # [BC, pred_len]

        if self.revin:
            y_bc = y_bc * std + mean

        y = y_bc.view(B, C, self.pred_len).permute(0, 2, 1).contiguous()  # [B, pred, C]
        return y
