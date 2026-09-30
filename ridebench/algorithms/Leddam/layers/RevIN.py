import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.means = None
        self.stdev = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.means = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        return (x - self.means) / self.stdev

    def inverse_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.means is None or self.stdev is None:
            raise RuntimeError("RevIN statistics are unavailable before normalization.")
        scale = self.stdev[:, :1, :].expand(-1, self.output_dim, -1)
        shift = self.means[:, :1, :].expand(-1, self.output_dim, -1)
        return x * scale + shift
