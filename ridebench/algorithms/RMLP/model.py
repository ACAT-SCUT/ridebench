import torch.nn as nn

from .layers.Invertible import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
        )
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.channel) if configs.rev else None

    def forward(self, x_endo, x_exo_con=None, x_exo_dis=None, y_exo_con=None, y_exo_dis=None):
        del x_exo_con, x_exo_dis, y_exo_con, y_exo_dis
        x = self.rev(x_endo, "norm") if self.rev else x_endo
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        pred = self.projection(x.transpose(1, 2)).transpose(1, 2)
        return self.rev(pred, "denorm") if self.rev else pred
