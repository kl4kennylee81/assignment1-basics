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
        self.W3 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.W1(x)
        silux = w1x * torch.sigmoid(w1x)
        w3x = self.W3(x)
        elew1w3 = silux * w3x
        return self.W2(elew1w3)
        


        