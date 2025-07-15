from jaxtyping import Float, Int
import numpy.typing as npt
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):

    def __init__(self, in_features:int, out_features:int, device: torch.device | None = None, dtype: torch.dtype | None =None):
        super().__init__()
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2/(in_features + out_features)**(0.5))
        torch.nn.init.trunc_normal_(W, mean=0, std=std, a=-3*std, b=3*std)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):

    def __init__(self, vocab_size:int, d_model:int, device: torch.device | None = None, dtype: torch.dtype | None =None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        W = torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(W, mean=0, std=1, a=-3, b=3)
        self.W = nn.Parameter(W)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None =None):
        super().__init__()
        self.eps = eps
        self.G = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared = x**2
        x_squared_mean = x_squared.mean(-1, keepdim=True)
        rms = (x_squared_mean + self.eps)**(0.5)
        x_normalized = x / rms
        result = einsum(x_normalized, self.G, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)

class SWIGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None =None):
        super().__init__()
        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)
        self.W3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.W1(x)
        silux = w1x * torch.sigmoid(w1x)
        w3x = self.W3(x)
        elew1w3 = silux * w3x
        return self.W2(elew1w3)
        

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.r = torch.zeros(max_seq_len, d_k, d_k, device=device)
        
        for i in range(max_seq_len):
            for k in range(d_k//2):
                freq = 1.0 / (theta ** (2*k / d_k))
                angle = i * freq
                
                cos_val = torch.cos(torch.tensor(angle, device=device))
                sin_val = torch.sin(torch.tensor(angle, device=device))
                
                self.r[i, 2*k, 2*k] = cos_val
                self.r[i, 2*k, 2*k+1] = -sin_val
                self.r[i, 2*k+1, 2*k] = sin_val
                self.r[i, 2*k+1, 2*k+1] = cos_val
                
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        ri_token_pos = self.r[token_positions]
        return einsum(x, ri_token_pos, "... seq d_k_in, ... seq d_k_out d_k_in -> ... seq d_k_out")
    

def softmax(x: torch.Tensor, dim: int):
    max_xi = torch.amax(x, dim=dim, keepdim=True)
    x_shifted = x - max_xi
    x_exp = torch.exp(x_shifted)
    sum_x_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    result = x_exp / sum_x_exp
    return result

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    wei = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (Q.shape[-1] ** 0.5)
    if mask is not None:
        wei = wei.masked_fill(mask == 0, float('-inf'))
    wei = softmax(wei, dim=-1)
    return einsum(wei, V, "... queries keys, ... keys d_v -> ... queries d_v")

class MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, device=None, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.d_model = d_model
        self.Q = Linear(d_model, self.dk * self.num_heads, device=device)
        self.K = Linear(d_model, self.dk * self.num_heads, device=device)
        self.V = Linear(d_model, self.dk * self.num_heads, device=device)
        self.Wo = Linear(self.dk * num_heads, d_model, device=device)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q,k,v = self.Q(x), self.K(x), self.V(x)
        q = rearrange(q, "... seq (num_heads dk) -> ... num_heads seq dk", num_heads=self.num_heads)
        k = rearrange(k, "... seq (num_heads dk) -> ... num_heads seq dk", num_heads=self.num_heads)
        v = rearrange(v, "... seq (num_heads dv) -> ... num_heads seq dv", num_heads=self.num_heads)

        if self.rope != None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq = k.shape[-2]
        mask = torch.tril(torch.ones(seq, seq))
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, "... num_heads seq dv -> ... seq (num_heads dv)")
        return self.Wo(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device=None, rope=None):
        super().__init__()
        self.mha = MultiheadAttention(d_model, num_heads, device=device, rope=rope)
        self.ffn = SWIGLU(d_model, d_ff, device=device)
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.mha(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x