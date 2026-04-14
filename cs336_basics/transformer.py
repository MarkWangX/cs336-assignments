import torch
import torch.nn as nn
import numpy as np
from einops import einsum, reduce, repeat

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
    
class positionwise_feedforward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        w1_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        w2_weight = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        w3_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

        mean = 0
        std = np.sqrt(2 / (d_ff + d_model))
        self.w1_weight = nn.init.trunc_normal_(w1_weight, mean, std, a=-3*std, b=3*std)
        self.w2_weight = nn.init.trunc_normal_(w2_weight, mean, std, a=-3*std, b=3*std)
        self.w3_weight = nn.init.trunc_normal_(w3_weight, mean, std, a=-3*std, b=3*std)

    def SiLU(self, x: torch.Tensor)-> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def GLU(self, x: torch.Tensor)-> torch.Tensor:
        return self.SiLU(einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")) * einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return einsum(self.w2_weight, self.GLU(x), "d_model d_ff, ... d_ff -> ... d_model")
    
class rope(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        cos = torch.empty(max_seq_len, device=device)
        sin = torch.empty(max_seq_len, device=device)
        
        k_seq = (torch.arange(d_k, device=device)) // 2
        i_seq = torch.arange(max_seq_len, device=device)
        theta_k = pow(theta, -2*k_seq/d_k)
        theta_ik = einsum(theta_k, i_seq, "k, i -> i k")
        cos = torch.cos(theta_ik)
        sin = torch.sin(theta_ik)
        
        self.register_buffer("cos_cached", cos,  persistent=False)
        self.register_buffer("sin_cached", sin,  persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        x = [q1, q2, q3, q4]
        x_rotate_half = [-q2, q1, -q4, q3]
        '''
        x_rotate_half = torch.empty_like(x)
        x_rotate_half[..., 0::2] = -x[..., 1::2]
        x_rotate_half[..., 1::2] = x[..., 0::2]

        cos, sin = self.cos_cached[token_positions], self.sin_cached[token_positions]
    
        return cos * x + sin * x_rotate_half