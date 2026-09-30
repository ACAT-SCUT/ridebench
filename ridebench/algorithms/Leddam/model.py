import torch
import torch.nn as nn
from .layers.Leddam import Leddam
from .layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.use_revin = configs.revin
        self.revin = RevIN(output_dim=configs.pred_len) if self.use_revin else None
        self.leddam = Leddam(
            enc_in=configs.enc_in,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            dropout=configs.dropout,
            pe_type=configs.pe_type,
            kernel_size=configs.kernel_size,
            n_layers=configs.n_layers,
        )
        self.main_head = nn.Linear(configs.d_model, configs.pred_len)
        self.residual_head = nn.Linear(configs.d_model, configs.pred_len)
        self.main_head.weight = nn.Parameter((1 / configs.d_model) * torch.ones(configs.pred_len, configs.d_model))
        self.residual_head.weight = nn.Parameter((1 / configs.d_model) * torch.ones(configs.pred_len, configs.d_model))

    def forward(self, x_endo, *args, **kwargs):
        if self.use_revin:
            x_endo = self.revin(x_endo)
        residual, main = self.leddam(x_endo)
        main_out = self.main_head(main.permute(0, 2, 1)).permute(0, 2, 1)
        residual_out = self.residual_head(residual.permute(0, 2, 1)).permute(0, 2, 1)
        pred = main_out + residual_out
        if self.use_revin:
            pred = self.revin.inverse_normalize(pred)
        return pred
