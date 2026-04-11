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

        self.weight = nn.__init_subclass__(
            weight,
            mean=mean, std=std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight