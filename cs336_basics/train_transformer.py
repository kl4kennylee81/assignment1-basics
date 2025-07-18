# Standard library imports
import argparse
import math
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, BinaryIO, IO, Optional

# Third-party imports
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

# Local imports
from cs336_basics.transformer import Transformer


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


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for i in range(batch_size):
        idx = np.random.randint(0, len(dataset) - context_length)
        xs.append(dataset[idx: idx + context_length])
        ys.append(dataset[idx+1: idx + context_length + 1])

    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)
    return (xs,ys)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)



def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]

class TrainModelArgs:
    # model args
    vocab_size: int = 10000
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: Optional[int] = 10000

    # adamw args
    weight_decay: float = 0.01
    betas: tuple[float, float] = (.9, .999)

    max_learning_rate: float = 3e-4
    min_learning_rate: float =   1e-5
    warmup_iters: int = 500
    cosine_cycle_iters: int = 4000

    #training loop args
    training_set: str | os.PathLike | BinaryIO | IO[bytes]
    validation_set: str | os.PathLike | BinaryIO | IO[bytes]

    tokenizer_state: str | os.PathLike | BinaryIO | IO[bytes]

    validation_step_interval: int = 100
    checkpoint_step_interval: int = 1000
    steps: int = 5000
    batch_size: int = 32

    gradient_clipping: Optional[float] = 1.0

    device: torch.device = torch.device(
    'mps:0' if torch.backends.mps.is_available() else 
    'cuda' if torch.cuda.is_available() else 
    'cpu')

def parse_args():
    """Parse command line arguments using defaults from TrainModelArgs."""
    # Create a default instance to get the defaults
    defaults = TrainModelArgs()
    
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--vocab-size', type=int, default=defaults.vocab_size,
                            help=f'Vocabulary size (default: {defaults.vocab_size})')
    model_group.add_argument('--context-length', type=int, default=defaults.context_length,
                            help=f'Context length (default: {defaults.context_length})')
    model_group.add_argument('--num-layers', type=int, default=defaults.num_layers,
                            help=f'Number of transformer layers (default: {defaults.num_layers})')
    model_group.add_argument('--d-model', type=int, default=defaults.d_model,
                            help=f'Model dimension (default: {defaults.d_model})')
    model_group.add_argument('--num-heads', type=int, default=defaults.num_heads,
                            help=f'Number of attention heads (default: {defaults.num_heads})')
    model_group.add_argument('--d-ff', type=int, default=defaults.d_ff,
                            help=f'Feed-forward dimension (default: {defaults.d_ff})')
    model_group.add_argument('--rope-theta', type=int, default=defaults.rope_theta,
                            help=f'RoPE theta parameter (default: {defaults.rope_theta})')
    
    # Optimizer arguments
    optim_group = parser.add_argument_group('Optimizer Configuration')
    optim_group.add_argument('--weight-decay', type=float, default=defaults.weight_decay,
                            help=f'Weight decay (default: {defaults.weight_decay})')
    optim_group.add_argument('--beta1', type=float, default=defaults.betas[0],
                            help=f'Adam beta1 (default: {defaults.betas[0]})')
    optim_group.add_argument('--beta2', type=float, default=defaults.betas[1],
                            help=f'Adam beta2 (default: {defaults.betas[1]})')
    
    # Learning rate schedule arguments
    lr_group = parser.add_argument_group('Learning Rate Schedule')
    lr_group.add_argument('--max-learning-rate', type=float, default=defaults.max_learning_rate,
                         help=f'Maximum learning rate (default: {defaults.max_learning_rate})')
    lr_group.add_argument('--min-learning-rate', type=float, default=defaults.min_learning_rate,
                         help=f'Minimum learning rate (default: {defaults.min_learning_rate})')
    lr_group.add_argument('--warmup-iters', type=int, default=defaults.warmup_iters,
                         help=f'Number of warmup iterations (default: {defaults.warmup_iters})')
    lr_group.add_argument('--cosine-cycle-iters', type=int, default=defaults.cosine_cycle_iters,
                         help=f'Cosine cycle iterations (default: {defaults.cosine_cycle_iters})')
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--training-set', type=str, default=defaults.training_set,
                           help=f'Path to training data (default: {defaults.training_set})')
    data_group.add_argument('--validation-set', type=str, default=defaults.validation_set,
                           help=f'Path to validation data (default: {defaults.validation_set})')
    data_group.add_argument('--tokenizer-state', type=str, default=defaults.tokenizer_state,
                           help=f'Path to tokenizer state (default: {defaults.tokenizer_state})')
    
    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--validation-step-interval', type=int, default=defaults.validation_step_interval,
                            help=f'Validation frequency (default: {defaults.validation_step_interval})')
    train_group.add_argument('--checkpoint-step-interval', type=int, default=defaults.checkpoint_step_interval,
                            help=f'Checkpoint frequency (default: {defaults.checkpoint_step_interval})')
    train_group.add_argument('--steps', type=int, default=defaults.steps,
                            help=f'Total training steps (default: {defaults.steps})')
    train_group.add_argument('--batch-size', type=int, default=defaults.batch_size,
                            help=f'Batch size (default: {defaults.batch_size})')
    train_group.add_argument('--gradient-clipping', type=float, default=defaults.gradient_clipping,
                            help=f'Gradient clipping threshold (default: {defaults.gradient_clipping})')
    
    # Device arguments
    device_group = parser.add_argument_group('Device Configuration')
    device_group.add_argument('--device', type=str, default=None,
                             help='Device to use (cpu, cuda, mps). If not specified, auto-detect.')
    
    return parser.parse_args()

def get_device(device_str=None):
    """Get the appropriate device."""
    if device_str is not None:
        return torch.device(device_str)
    
    # Auto-detect device
    if torch.backends.mps.is_available():
        return torch.device('mps:0')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def args_to_dataclass(args):
    """Convert parsed arguments to TrainModelArgs dataclass."""
    return TrainModelArgs(
        # Model args
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        
        # Optimizer args
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        
        # Learning rate schedule
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        
        # Data paths
        training_set=args.training_set,
        validation_set=args.validation_set,
        tokenizer_state=args.tokenizer_state,
        
        # Training config
        validation_step_interval=args.validation_step_interval,
        checkpoint_step_interval=args.checkpoint_step_interval,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_clipping=args.gradient_clipping,
        
        # Device
        device=get_device(args.device)
    )

class TrainModel:
    def __init__(self, args:TrainModelArgs):
        self.args = args
        self.cur_step = 0
        self.model = Transformer(
            vocab_size = args.vocab_size,
            context_length=args.context_length,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=args.device
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.max_learning_rate,
            weight_decay=args.weight_decay,
            betas=args.betas
        )

        self.training_set = np.load(self.args.training_set, mmap_mode='r')
        self.validation_set = np.load(self.args.validation_set, mmap_mode='r')
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            x, label = get_batch(self.validation_set, self.args.batch_size, self.args.context_length, device=self.args.device)
            output = self.model(x)
            loss = cross_entropy(output, label)
            perplexity = loss.exp()
            return loss, perplexity
        
    @classmethod
    def find_latest_checkpoint(cls, checkpoint_dir="./output"):
        if not os.path.exists(checkpoint_dir):
            return None
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint-") and filename.endswith(".pth"):
                try:
                    step_str = filename.replace("checkpoint-", "").replace(".pth", "")
                    step = int(step_str)
                    checkpoint_files.append((step, os.path.join(checkpoint_dir, filename)))
                except ValueError:
                    continue
        if not checkpoint_files:
            return None
        
        _, latest_path = max(checkpoint_files, key=lambda x: x[0])
        return latest_path


    def train(self):
        latest_path = self.find_latest_checkpoint()
        self.cur_step = load_checkpoint(latest_path, self.model, self.optimizer) 
        valid_loss, valid_perplexity = self.evaluate()
        pbar = tqdm(range(self.cur_step, self.args.steps))
        for step in pbar:
            self.cur_step = step
            self.model.train()
            self.optimizer.zero_grad()
            lr = lr_cosine_schedule(
                step,
                self.args.max_learning_rate,
                self.args.min_learning_rate,
                self.args.warmup_iters,
                self.args.cosine_cycle_iters)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            x, targets = get_batch(self.training_set, self.args.batch_size, self.args.context_length, device=self.args.device)
            logits = self.model(x)
            loss = cross_entropy(logits, targets)
            loss.backward()
            gradient_clipping(self.model.parameters(), self.args.gradient_clipping)
            self.optimizer.step()

            if step % self.args.checkpoint_step_interval == 0:
                save_checkpoint(self.model, self.optimizer, step, f"./output/checkpoint-{step}.pth")
            
            if step % self.args.validation_step_interval == 0:
                valid_loss, valid_perplexity = self.evaluate()
            
            pbar.set_postfix({
                "train_loss": f"{loss.cpu().item():.2f}",
                "valid_loss": f"{valid_loss.cpu().item():.2f}",
                "valid_perplexity": f"{valid_perplexity.cpu().item():.2f}",
            })

        save_checkpoint(self.model, self.optimizer, self.cur_step, f"./output/checkpoint-{self.cur_step}.pth")

def main():
    # Parse command line arguments
    parsed_args = parse_args()
    
    # Convert to dataclass
    args = args_to_dataclass(parsed_args)
    print(args)
    
    # Initialize trainer
    print("Initializing model and trainer...")
    trainer = TrainModel(args)
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    print("✅ Training completed successfully!")

# Example usage:
# python train.py --steps 10000 --batch-size 64 --max-learning-rate 1e-3
# python train.py --training-set /path/to/train.npy --validation-set /path/to/val.npy
# python train.py --device cuda --num-layers 6 --d-model 768 
if __name__ == "__main__":
    main()