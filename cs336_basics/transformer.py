import torch
import torch.nn as nn
import numpy as np
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        mean = 0
        std = np.sqrt(2 / (in_features + out_features))
        weight = nn.Parameter(
            torch.empty(in_features, out_features), device=device, dtype=dtype
        )

        self.weight = nn.trunc_normal(
            weight,
            mean=mean, std=std,
            a=-3*std, b=3*std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum('dout din, din -> dout', self.weight, x)