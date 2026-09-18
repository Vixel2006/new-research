"""Training loop with optax and orbax checkpointing"""

import optax
import orbax.checkpoint as ocp
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Optional
from pathlib import Path
import time
from tqdm import tqdm
import wandb


@dataclass
class TrainConfig:
    """Training configuration"""
    # Model
    embed_dim: int = 192
    img_size: int = 64
    action_dim: int = 2
    history_size: int = 3
    num_preds: int = 1

    # Optimizer
    lr: float = 5e-5
    weight_decay: float = 1e-3
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Data
    batch_size: int = 128
    num_workers: int = 0

    # Loss
    sigreg_weight: float = 0.1

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    max_checkpoints: int = 3


def create_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    """Create optimizer with warmup and cosine decay"""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=config.lr * 0.01,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=config.weight_decay),
    )


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: dict,
) -> dict:
    """Single training step"""

    def loss_fn(model: nnx.Module):
        losses = model.compute_loss(batch["pixels"], batch["actions"])
        return losses["loss"], losses

    (loss, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    return losses


@nnx.jit
def eval_step(model: nnx.Module, batch: dict) -> dict:
    """Evaluation step (no gradients)"""
    return model.compute_loss(batch["pixels"], batch["actions"])


class Trainer:
    """LeWorldModel Trainer"""

    def __init__(
        self,
        model: nnx.Module,
        config: TrainConfig,
        train_iter: Iterator,
        val_iter: Optional[Iterator] = None,
        seed: int = 0,
    ):
        self.model = model
        self.config = config
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.step = 0

        # Optimizer
        self.optimizer = nnx.Optimizer(model, create_optimizer(config))

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpointer = ocp.PyTreeCheckpointer()
        self.checkpoint_manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            self.checkpointer,
            options=ocp.CheckpointManagerOptions(max_to_keep=config.max_checkpoints),
        )

        # Metrics
        self.train_losses = []
        self.val_losses = []

    def train_step(self, batch: dict) -> dict:
        """Run one training step"""
        losses = train_step(self.model, self.optimizer, batch)
        self.step += 1
        return losses

    def evaluate(self, num_batches: int = 10) -> dict:
        """Run evaluation"""
        if self.val_iter is None:
            return {}

        val_losses = []
        for _ in range(num_batches):
            try:
                batch = next(self.val_iter)
            except StopIteration:
                break
            losses = eval_step(self.model, batch)
            val_losses.append(losses)

        if not val_losses:
            return {}

        # Average losses
        avg_losses = {}
        for k in val_losses[0].keys():
            avg_losses[k] = float(jnp.mean(jnp.array([l[k] for l in val_losses])))
        return avg_losses

    def save_checkpoint(self):
        """Save model checkpoint"""
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "step": self.step,
            "config": self.config,
        }
        self.checkpoint_manager.save(self.step, args=ocp.args.StandardSave(state))
        print(f"Saved checkpoint at step {self.step}")

    def load_checkpoint(self, step: Optional[int] = None):
        """Load model checkpoint"""
        if step is None:
            step = self.checkpoint_manager.latest_step()
        if step is None:
            print("No checkpoint found")
            return

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "step": 0,
            "config": self.config,
        }
        restored = self.checkpoint_manager.restore(step, args=ocp.args.StandardRestore(state))
        self.model = restored["model"]
        self.optimizer = restored["optimizer"]
        self.step = restored["step"]
        print(f"Loaded checkpoint at step {self.step}")

    def train(self, num_steps: Optional[int] = None):
        """Main training loop"""
        num_steps = num_steps or self.config.max_steps

        pbar = tqdm(total=num_steps, initial=self.step, desc="Training")

        while self.step < num_steps:
            try:
                batch = next(self.train_iter)
            except StopIteration:
                print("Dataset exhausted, restarting...")
                continue

            # Training step
            losses = self.train_step(batch)
            self.train_losses.append(losses)

            # Logging
            if self.step % self.config.log_every == 0:
                log_dict = {f"train/{k}": float(v) for k, v in losses.items()}
                log_dict["step"] = self.step
                if wandb.run is not None:
                    wandb.log(log_dict, step=self.step)
                pbar.set_postfix({k: f"{float(v):.4f}" for k, v in losses.items()})

            # Evaluation
            if self.val_iter is not None and self.step % self.config.eval_every == 0:
                val_losses = self.evaluate()
                if val_losses:
                    self.val_losses.append(val_losses)
                    if wandb.run is not None:
                        wandb.log({f"val/{k}": v for k, v in val_losses.items()}, step=self.step)
                    print(f"Step {self.step} | Val: {val_losses}")

            # Checkpoint
            if self.step % self.config.save_every == 0:
                self.save_checkpoint()

            pbar.update(1)

        pbar.close()
        self.save_checkpoint()  # Final save


def create_trainer(
    model: nnx.Module,
    train_data: Iterator,
    val_data: Optional[Iterator] = None,
    **kwargs,
) -> Trainer:
    """Factory function to create trainer with config"""
    config = TrainConfig(**kwargs)
    return Trainer(model, config, train_data, val_data)