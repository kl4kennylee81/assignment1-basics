import wandb
from dataclasses import asdict

# Complete sweep configuration
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'valid_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {'values': [32,64,128,256]},
    }
}

def train_sweep():
    # Initialize wandb run
    run = wandb.init(
      group="Tinystories-lr-sweep",
      force=True
    )
    config = wandb.config

    # Create training arguments - only override what's different from defaults
    train_args = {
        **DefaultTrainModelArgs,

        # Only the sweep parameters that differ from DefaultTrainModelArgs
        "steps": 160000 // config.batch_size, # 40,960,000 tokens processed
        "batch_size": config.batch_size,
        "validation_step_interval": 4096 // config.batch_size,
        "cosine_cycle_iters": 160000 // config.batch_size,
        "warmup_iters": 8000 // config.batch_size,
        "max_learning_rate": 1e-3 * (config.batch_size//16),
        "min_learning_rate": 1e-5 * (config.batch_size//16),

        # wandb settings - always override for sweep
        "wandb_active": True,                     # Enable for sweep
        "wandb_run" : f"batch_size_{config.batch_size}"
    }

    # Initialize and run training
    trainer = TrainModel(TrainModelArgs(**train_args))
    config.update(asdict(trainer.args))
    wandb.run.name = trainer.args.wandb_run
    trainer.train()
    wandb.finish()

# Create the sweep
sweep_id = wandb.sweep(
    sweep=sweep_config,
    project="cs336-llm-assignment1",
    entity="kl4kennylee81-kenneth-personal"
)

print(f"Sweep created successfully!")
print(f"Sweep ID: {sweep_id}")
print(f"Project: cs336-llm-assignment1")
print(f"wandb agent {sweep_id}")

# Run the sweep agent
# wandb.agent(sweep_id, train_sweep, count=8)