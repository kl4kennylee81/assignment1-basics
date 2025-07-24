import wandb
import torch
import numpy as np
import os
from dataclasses import dataclass, asdict
from typing import Literal, cast, Optional, BinaryIO, IO
import time
from tqdm.notebook import tqdm

# Default training configuration at top of file
DefaultTrainModelArgs = {
    # Model args
    "vocab_size": 10000,
    "context_length": 256,
    "num_layers": 4,
    "d_model": 512,
    "num_heads": 16,
    "d_ff": 1344,
    "rope_theta": 10000,

    # Optimizer args
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),

    # Learning rate schedule
    "max_learning_rate": 1e-3,
    "min_learning_rate": 1e-5,
    "warmup_iters": 2000,
    "cosine_cycle_iters": 40960,

    # Data paths - keep as is
    "training_set": "TinyStoriesV2-GPT4-train.npy",
    "validation_set": "TinyStoriesV2-GPT4-valid.npy",
    "tokenizer_vocab": "tinystories_vocab.json",
    "tokenizer_merges": "tinystories_merges.txt",

    # Training config
    "validation_step_interval": 500,
    "checkpoint_step_interval": 10000,
    "steps": 40960,  # 327M tokens target
    "batch_size": 32,
    "gradient_clipping": 1.0,

    # gdrive
    "save_gdrive": False,
    "load_model_gdrive": "",

    # Device
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # wandb
    "wandb_active": False,
    "wandb_run": ""
}

@dataclass
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
    betas: tuple[float, float] = (0.9, 0.999)

    # Learning rate schedule
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 40960

    # training loop args
    training_set: str | os.PathLike | BinaryIO | IO[bytes] = "TinyStoriesV2-GPT4-train.npy"
    validation_set: str | os.PathLike | BinaryIO | IO[bytes] = "TinyStoriesV2-GPT4-valid.npy"
    tokenizer_vocab: str | os.PathLike | BinaryIO | IO[bytes] = "tinystories_vocab.json"
    tokenizer_merges: str | os.PathLike | BinaryIO | IO[bytes] = "tinystories_merges.txt"

    validation_step_interval: int = 500
    checkpoint_step_interval: int = 10000
    steps: int = 40960
    batch_size: int = 32
    gradient_clipping: Optional[float] = 1.0

    # gdrive
    save_gdrive: bool = False
    load_model_gdrive: str = ""

    # wandb logging
    wandb_active: bool = False
    wandb_run: Optional[str] = ""

    # device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class TrainModel:
    def __init__(self, args: TrainModelArgs):
        self.args = args
        self.cur_step = 0
        self.model = Transformer(
            vocab_size=args.vocab_size,
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

        self.tokenizer = Tokenizer.from_files(args.tokenizer_vocab, args.tokenizer_merges, ["<|endoftext|>"])

        self.training_set = np.load(self.args.training_set, mmap_mode='r')
        self.validation_set = np.load(self.args.validation_set, mmap_mode='r')

        if args.wandb_active and wandb.run:
            wandb.watch(self.model, log=cast(Literal["gradients", "parameters", "all"], "gradients"), log_freq=10)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_size = self.training_set.size + self.validation_set.size
            eval_size = total_size // 1000
            num_batches = eval_size // (self.args.batch_size * self.args.context_length)

            num_batches = max(1, num_batches)

            for _ in range(num_batches):
                x, label = get_batch(self.validation_set, self.args.batch_size, self.args.context_length, device=self.args.device)
                with torch.autocast(device_type=self.args.device, dtype=torch.bfloat16):
                  output = self.model(x)
                loss = cross_entropy(output, label)
                total_loss += loss.item()

            avg_loss = torch.tensor(total_loss / num_batches)
            perplexity = avg_loss.exp()
            return avg_loss, perplexity

    def train(self):
        if self.args.load_model_gdrive != "":
          self.cur_step = load_checkpoint(self.args.load_model_gdrive, self.model, self.optimizer)

        valid_loss, valid_perplexity = self.evaluate()
        if self.args.wandb_active and wandb.run:
            wandb.log({"valid_loss": valid_loss, "valid_perplexity": valid_perplexity}, step=self.cur_step)

        pbar = tqdm(range(self.cur_step, self.args.steps))
        start_time = time.time()
        tokens_processed = 0

        for step in pbar:
            step_start_time = time.time()

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
            with torch.autocast(device_type=self.args.device, dtype=torch.bfloat16):
              logits = self.model(x)
            loss = cross_entropy(logits, targets)
            loss.backward()
            l2norm = gradient_clipping(self.model.parameters(), self.args.gradient_clipping)
            self.optimizer.step()

            # Calculate metrics
            batch_tokens = x.shape[0] * x.shape[1]
            tokens_processed += batch_tokens
            elapsed_time = time.time() - start_time
            tokens_per_second = tokens_processed / elapsed_time if elapsed_time > 0 else 0
            dt = time.time() - step_start_time

            if self.args.save_gdrive and step % self.args.checkpoint_step_interval == 0 and step > 0 :
                os.makedirs(f'{drive_cs336_dir}/output', exist_ok=True)
                save_checkpoint(self.model, self.optimizer, step, f'{drive_cs336_dir}/output/checkpoint-{step}.pth')

            if (step % self.args.validation_step_interval == 0 and step > 0) or (step == self.args.steps-1):
                valid_loss, valid_perplexity = self.evaluate()

            pbar.set_postfix({
                "loss": f"{loss.item():.2f}",
                "valid_loss": f"{valid_loss.item():.2f}",
                "valid_perplexity": f"{valid_perplexity.item():.2f}",
            })

            if self.args.wandb_active and wandb.run:
                wandb.log({
                    "train_loss": loss.item(),
                    "train_perplexity": loss.exp().item(),
                    "valid_loss": valid_loss.item(),
                    "valid_perplexity": valid_perplexity.item(),
                    "grad_norm": l2norm,
                    "lr": lr,
                    "tokens_per_second": tokens_per_second,
                    "step_time_seconds": dt,
                    "gpu_memory_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                    "tokens_processed": tokens_processed,
                }, step=step)

        # Save final checkpoint
        if self.args.save_gdrive:
          save_checkpoint(self.model, self.optimizer, step, f'{drive_cs336_dir}/output/checkpoint-{step}.pth')

        if self.args.wandb_active and wandb.run:
            local_checkpoint_path = f'{self.args.wandb_run}-checkpoint-{step}.pth'
            save_checkpoint(self.model, self.optimizer, step, local_checkpoint_path)

            artifact = wandb.Artifact(f"{self.args.wandb_run}-checkpoint_{step}", type="model")
            artifact.add_file(local_checkpoint_path)
            wandb.log_artifact(artifact)

            # Clean up local file after uploading to wandb
            os.remove(local_checkpoint_path)
