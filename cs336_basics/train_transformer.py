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
    
class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9,0.999), eps = 1e-8):
        defaults = {"lr" : lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1,b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                g = p.grad.data
                m = b1 * m + (1-b1) * g
                v = b2*v + (1-b2) * g**2
                lr_t = lr * math.sqrt((1-b2**t)) / (1 - b1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

def lr_cosine_schedule(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    cosine_cycle_iters: int,
):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * it / warmup_steps
    # 2) if it > cosine_cycle_iters, return min learning rate
    if it >= cosine_cycle_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (cosine_cycle_iters - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grad_params = [p for p in parameters if p.grad is not None]
    l2norm = torch.sqrt(sum([torch.sum(p.grad ** 2) for p in grad_params]))
    if l2norm < max_l2_norm:
        return
    for p in grad_params:
        p.grad *= (max_l2_norm/ (l2norm + eps))



