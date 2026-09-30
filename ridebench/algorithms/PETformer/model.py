import math

import torch
import torch.nn as nn

from .layers.RevIN import RevIN


class Model(nn.Module):
    """
    PETformer:
    - shared placeholder tokens for future horizon
    - long sub-sequence tokenization
    - optional channel interaction modules
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.ff_factor = configs.ff_factor
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout_rate = configs.dropout
        self.win_len = configs.win_len
        self.win_stride = configs.win_stride
        self.channel_attn = configs.channel_attn
        self.attn_type = configs.attn_type

        if self.seq_len < self.win_len:
            raise ValueError("seq_len must be >= win_len for PETformer.")
        if self.win_stride <= 0:
            raise ValueError("win_stride must be positive.")

        self.input_token_size = int((self.seq_len - self.win_len) / self.win_stride + 1)
        self.output_token_size = int(math.ceil(self.pred_len / self.win_len))
        self.all_token_size = self.input_token_size + self.output_token_size

        self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=True)
        self.mapping = nn.Linear(self.win_len, self.d_model)
        self.placeholder = nn.Parameter(torch.randn(self.d_model))
        self.position_embedding = nn.Embedding(self.all_token_size, self.d_model)
        self.transformer_encoder_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                batch_first=True,
                dim_feedforward=self.d_model * self.ff_factor,
                dropout=self.dropout_rate,
            )
            for _ in range(self.e_layers)
        )
        self.batch_norm = nn.ModuleList(nn.BatchNorm1d(self.all_token_size) for _ in range(self.e_layers))
        self.predict = nn.Linear(self.d_model, self.win_len)

        if self.attn_type == 0:
            mask = torch.full((self.all_token_size, self.all_token_size), False)
        elif self.attn_type == 1:
            mask = torch.full((self.all_token_size, self.all_token_size), True)
            mask[:, : self.input_token_size] = False
            mask[torch.eye(self.all_token_size, dtype=torch.bool)] = False
        elif self.attn_type == 2:
            mask = torch.full((self.all_token_size, self.all_token_size), True)
            mask[self.input_token_size :, :] = False
            mask[torch.eye(self.all_token_size, dtype=torch.bool)] = False
        elif self.attn_type == 3:
            mask = torch.full((self.all_token_size, self.all_token_size), True)
            mask[self.input_token_size :, : self.input_token_size] = False
            mask[torch.eye(self.all_token_size, dtype=torch.bool)] = False
        else:
            mask = torch.full((self.all_token_size, self.all_token_size), False)
        self.register_buffer("mask", mask, persistent=False)

        if self.channel_attn != 0:
            self.attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=1, batch_first=True, dropout=0.5)
            self.channel_batch_norm = nn.BatchNorm1d(self.enc_in)
        if self.channel_attn in (2, 3):
            self.channel_pe = nn.Embedding(self.enc_in, self.d_model)

    def forward(self, x_enc, x_exo_con=None, x_exo_dis=None, y_exo_con=None, y_exo_dis=None):
        x = self.revin_layer(x_enc, "norm").permute(0, 2, 1)

        input_tokens = self.mapping(x.unfold(dimension=-1, size=self.win_len, step=self.win_stride))
        all_tokens = torch.cat(
            (
                input_tokens,
                self.placeholder.repeat(x_enc.size(0), self.enc_in, self.output_token_size, 1),
            ),
            dim=2,
        )

        batch_index = torch.arange(self.all_token_size, device=x.device).expand(
            x_enc.size(0) * self.enc_in, self.all_token_size
        )
        all_tokens = all_tokens.view(-1, self.all_token_size, self.d_model) + self.position_embedding(batch_index)

        for i in range(self.e_layers):
            all_tokens = self.batch_norm[i](self.transformer_encoder_layers[i](all_tokens, src_mask=self.mask)) + all_tokens

        output_tokens = all_tokens.view(-1, self.enc_in, self.all_token_size, self.d_model)[:, :, self.input_token_size :, :]

        if self.channel_attn == 1:
            output_tokens = output_tokens.permute(0, 2, 1, 3).reshape(-1, self.enc_in, self.d_model)
            output_tokens = self.channel_batch_norm(self.attn(output_tokens, output_tokens, output_tokens)[0]) + output_tokens
            output_tokens = output_tokens.view(-1, self.output_token_size, self.enc_in, self.d_model).permute(0, 2, 1, 3)
        elif self.channel_attn == 2:
            channel_index = torch.arange(self.enc_in, device=x.device).expand(x_enc.size(0) * self.output_token_size, self.enc_in)
            query = self.channel_pe(channel_index)
            output_tokens = output_tokens.permute(0, 2, 1, 3).reshape(-1, self.enc_in, self.d_model)
            output_tokens = self.channel_batch_norm(self.attn(query, query, output_tokens)[0]) + output_tokens
            output_tokens = output_tokens.view(-1, self.output_token_size, self.enc_in, self.d_model).permute(0, 2, 1, 3)
        elif self.channel_attn == 3:
            channel_index = torch.arange(self.enc_in, device=x.device).expand(x_enc.size(0) * self.output_token_size, self.enc_in)
            channel_identity = self.channel_pe(channel_index)
            output_tokens = output_tokens.permute(0, 2, 1, 3).reshape(-1, self.enc_in, self.d_model)
            output_tokens = output_tokens + channel_identity
            output_tokens = output_tokens.view(-1, self.output_token_size, self.enc_in, self.d_model).permute(0, 2, 1, 3)

        y = self.predict(output_tokens).view(-1, self.enc_in, self.output_token_size * self.win_len).permute(0, 2, 1)
        y = y[:, : self.pred_len, :]
        y = self.revin_layer(y, "denorm")
        return y

