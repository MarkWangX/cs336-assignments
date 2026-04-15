import torch
import torch.nn as nn
import numpy as np
from einops import einsum, reduce, rearrange, repeat

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
    def __init__(self, num_embeddings, embedding_dim, device=None,dtype=None):
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
    
def softmax(x: torch.Tensor, i: int)-> torch.Tensor:
    x_max, _ = torch.max(x, dim=i, keepdim=True)
    exp_x = (x - x_max).exp()
    return exp_x / exp_x.sum(dim=i, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    #############################################################
    #   Q: Float[Tensor, " batch_size ... queries d_k"],
    #   K: Float[Tensor, " ... keys d_k"],
    #   V: Float[Tensor, " ... values d_v"],
    #   mask: Bool[Tensor, " ... queries keys"]
    #############################################################
    d_k = Q.size(-1)

    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)
    scores_masked = scores.masked_fill(mask == False, float('-inf'))
    scores_masked_softmax = softmax(scores_masked, -1)
    # keys == values
    attention = einsum(scores_masked_softmax, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention

class multihead_self_attention(torch.nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        max_seq_len: int | None = None,
        theta: float | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.Q_weight = Linear(d_model, num_heads * self.d_k)
        self.K_weight = Linear(d_model, num_heads * self.d_k)
        self.V_weight = Linear(d_model, num_heads * self.d_k)

        self.O_weight = Linear(num_heads * self.d_k, d_model)

        if max_seq_len is not None and theta is not None:
            self.rope = rope(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        seq_len = x.size(-2)

        if token_positions is None:
            batch, sequence_length = x.size(0), x.size(1)
            token_positions = torch.arange(sequence_length, dtype=torch.long, device=x.device)
            token_positions = repeat(token_positions, "sequence_length -> batch sequence_length", batch=batch)

        Q = self.Q_weight.forward(x)
        K = self.K_weight.forward(x)
        V = self.V_weight.forward(x)

        Q_heads = rearrange(Q, "... seq_len (num_heads d_q) -> ... num_heads seq_len d_q", num_heads=self.num_heads)
        K_heads = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V_heads = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        if self.rope is not None:
            Q_heads = self.rope.forward(Q_heads, token_positions)
            K_heads = self.rope.forward(K_heads, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        MHA = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        MHA_concat = rearrange(MHA, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)", num_heads=self.num_heads)
        MHSA = self.O_weight.forward(MHA_concat)
        return MHSA

class transformer_block(torch.nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        max_seq_len: int, 
        theta: int, 
        d_ff: int
    ):
        super().__init__()
        self.rmsnorm1 = rmsnorm(d_model)
        self.rmsnorm2 = rmsnorm(d_model)
        self.attn = multihead_self_attention(d_model, num_heads, max_seq_len, theta)
        self.ffn = positionwise_feedforward(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        x =  x + self.attn.forward(self.rmsnorm1.forward(x), token_positions)
        x = x + self.ffn.forward(self.rmsnorm2.forward(x))
        return x

class transformer_lm(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int, 
        num_heads: int, 
        theta: int, 
        d_ff: int
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            transformer_block(d_model, num_heads, context_length, theta, d_ff)
            for _ in range(num_layers)
        ])

        self.rmsnorm = rmsnorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.rmsnorm(x)
        
        x = self.linear(x)
        
        return x