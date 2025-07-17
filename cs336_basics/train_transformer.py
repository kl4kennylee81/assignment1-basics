from jaxtyping import Float, Int
import numpy.typing as npt
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    inputs_shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values

    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_shifted), dim=-1, keepdim=True))

    logits = inputs_shifted - log_sum_exp

    nlls = -logits[torch.arange(len(targets)), targets]
    
    return nlls.mean()





