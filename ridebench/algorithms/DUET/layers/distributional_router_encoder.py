import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distribution_fit = nn.Sequential(
            nn.Linear(config.seq_len, config.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_experts, bias=False),
        )

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        return self.distribution_fit(mean)
