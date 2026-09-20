from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx
from tqdm import tqdm

from ..models import LeWorldModel


@dataclass
class TrainConfig:
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
    warmup_steps = min(config.warmup_steps, max(0, config.max_steps - 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=warmup_steps,
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
    sigreg_fn,
) -> dict:
    def loss_fn(model: LeWorldModel):
        losses = model.compute_loss(batch["pixels"], batch["actions"], sigreg_fn)
        return losses["loss"], losses

    (loss, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return losses


@nnx.jit
def eval_step(model: LeWorldModel, batch: dict, sigreg_fn) -> dict:
    """Evaluation step (no gradients)"""
    return model.compute_loss(batch["pixels"], batch["actions"], sigreg_fn)


class Trainer:
    def __init__(
        self,
        model: LeWorldModel,
        config: TrainConfig,
        train_iter: Iterator,
        val_iter: Iterator | None = None,
        seed: int = 0,
    ):
        self.model = model
        self.config = config
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.step = 0

        # Optimizer
        self.optimizer = nnx.Optimizer(model, create_optimizer(config), wrt=nnx.Param)

        # SIGReg
        self.sigreg_fn = model.config if hasattr(model, "config") else None

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpointer = ocp.StandardCheckpointer()
        self.checkpoint_manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            self.checkpointer,
            options=ocp.CheckpointManagerOptions(max_to_keep=config.max_checkpoints),
        )

        # Metrics
        self.train_losses = []
        self.val_losses = []

    @classmethod
    def create(
        cls,
        model: LeWorldModel,
        train_data: Iterator,
        val_data: Iterator | None = None,
        **kwargs,
    ) -> "Trainer":
        """Factory method to create Trainer with default config."""
        config = TrainConfig(**kwargs)
        return cls(model, config, train_data, val_data)

    def train_step(self, batch: dict, sigreg_fn) -> dict:
        """Run one training step"""
        losses = train_step(self.model, self.optimizer, batch, sigreg_fn)
        self.step += 1
        return losses

    def evaluate(self, sigreg_fn, num_batches: int = 10) -> dict:
        """Run evaluation"""
        if self.val_iter is None:
            return {}

        val_losses = []
        for _ in range(num_batches):
            try:
                batch = next(self.val_iter)
            except StopIteration:
                break
            losses = eval_step(self.model, batch, sigreg_fn)
            val_losses.append(losses)

        if not val_losses:
            return {}

        # Average losses
        avg_losses = {}
        for k in val_losses[0].keys():
            avg_losses[k] = float(jnp.mean(jnp.array([l[k] for l in val_losses])))
        return avg_losses

    def save_checkpoint(self):
        try:
            # orbax cannot serialize nnx.Modules or PRNG keys natively, so we
            # convert the (Param-only) model state and optimizer state to plain
            # pytrees of arrays first.
            state = {
                "model": nnx.to_pure_dict(nnx.state(self.model, nnx.Param)),
                "optimizer": nnx.to_pure_dict(nnx.state(self.optimizer)),
                "step": self.step,
            }
            self.checkpoint_manager.save(self.step, args=ocp.args.StandardSave(state))
            print(f"Saved checkpoint at step {self.step}")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"Checkpoint save failed (non-fatal): {type(e).__name__}: {msg}")

    def load_checkpoint(self, step: int | None = None):
        if step is None:
            step = self.checkpoint_manager.latest_step()
        if step is None:
            print("No checkpoint found")
            return

        try:
            # Restore into structure-matching templates (orbax requires the
            # provided target tree to match what was saved).
            template = {
                "model": nnx.to_pure_dict(nnx.state(self.model, nnx.Param)),
                "optimizer": nnx.to_pure_dict(nnx.state(self.optimizer)),
                "step": 0,
            }
            restored = self.checkpoint_manager.restore(
                step, args=ocp.args.StandardRestore(template)
            )
            nnx.update(self.model, nnx.State(restored["model"]))
            nnx.update(self.optimizer, nnx.State(restored["optimizer"]))
            self.step = int(np.asarray(restored["step"]))
            print(f"Loaded checkpoint at step {self.step}")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"Checkpoint load failed: {type(e).__name__}: {msg}")

    def train(self, num_steps: int | None = None, sigreg_fn=None):
        num_steps = num_steps or self.config.max_steps

        if sigreg_fn is None:
            # Import here to avoid circular import
            from ..loss.sigreg import SIGReg

            sigreg_fn = SIGReg(
                knots=self.config.sigreg_knots
                if hasattr(self.config, "sigreg_knots")
                else 17,
                num_proj=self.config.sigreg_num_proj
                if hasattr(self.config, "sigreg_num_proj")
                else 1024,
                embed_dim=self.model.config.embed_dim,
            )

        pbar = tqdm(total=num_steps, initial=self.step, desc="Training")

        while self.step < num_steps:
            try:
                batch = next(self.train_iter)
            except StopIteration:
                print("Dataset exhausted, restarting...")
                continue

            # Training step
            losses = self.train_step(batch, sigreg_fn)
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
                val_losses = self.evaluate(sigreg_fn)
                if val_losses:
                    self.val_losses.append(val_losses)
                    if wandb.run is not None:
                        wandb.log(
                            {f"val/{k}": v for k, v in val_losses.items()},
                            step=self.step,
                        )
                    print(f"Step {self.step} | Val: {val_losses}")

            # Checkpoint
            if self.step % self.config.save_every == 0:
                self.save_checkpoint()

            pbar.update(1)

        pbar.close()
        self.save_checkpoint()  # Final save


def create_trainer(
    model: LeWorldModel,
    train_data: Iterator,
    val_data: Iterator | None = None,
    **kwargs,
) -> Trainer:
    config = TrainConfig(**kwargs)
    return Trainer(model, config, train_data, val_data)
