import torch
import torch.nn as nn
import numpy as np
from einops import einsum, reduce

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        mean = 0
        std = np.sqrt(2 / (in_features + out_features))
        weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        self.weight = nn.init.trunc_normal_(
            weight,
            mean=mean, std=std,
            a=-3*std, b=3*std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, 'dout din, ... din -> ... dout')

class Embedding(torch.nn.Module):
    def __init__(self,num_embeddings, embedding_dim,device=None,dtype=None):
        super().__init__()
        weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.weight = nn.init.trunc_normal_(
            weight,
            mean=0, std=1,
            a=-3, b=3
        )

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        return self.weight[token_ids]

class rmsnorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS_x = torch.sqrt(reduce(torch.pow(x, 2), "... d_model -> ... 1", "mean") + self.eps)
        result = (x*self.weight)/RMS_x

        return result.to(in_dtype)
    