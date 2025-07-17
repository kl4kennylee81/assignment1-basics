from jaxtyping import Float, Int
import numpy.typing as npt
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange
from collections.abc import Callable, Iterable
from typing import Optional
import math

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    inputs_shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values

    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_shifted), dim=-1, keepdim=True))

    logits = inputs_shifted - log_sum_exp

    nlls = -logits[torch.arange(len(targets)), targets]
    
    return nlls.mean()


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        defaults = {"lr" : lr}
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                p.data -= lr / math.sqrt(t + 1) * p.grad.data
                state["t"] = t + 1
        return loss

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters. 
        loss = (weights**2).mean() # Compute a scalar loss value. 
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients. 
        opt.step() # Run optimizer step.







