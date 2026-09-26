import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

try:
    from .jepa import JEPA, ModelConfig
    from .utils import checkpoint
except ImportError:  # running as a plain script: python src/train.py
    from jepa import JEPA, ModelConfig
    from utils import checkpoint


@dataclass
class TrainConfig:
    """Training hyperparameters for the JEPA trainer."""

    peak_lr: float = 3e-4
    warmup_steps: int = 1_000
    decay_steps: int = 100_000
    end_lr: float = 0.0
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    def schedule(self) -> optax.Schedule:
        """Linear warmup up to ``peak_lr``, then cosine decay to ``end_lr``."""
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.end_lr,
        )

    def optimizer(self, schedule: optax.Schedule) -> optax.GradientTransformation:
        """AdamW on ``schedule``, with global-norm gradient clipping."""
        tx = optax.adamw(learning_rate=schedule, weight_decay=self.weight_decay)
        if self.grad_clip:
            tx = optax.chain(optax.clip_by_global_norm(self.grad_clip), tx)
        return tx


def _log_line(step: int, metrics: dict) -> str:
    """One console line for a logged step, whatever terms ``loss`` returned."""
    terms = "  ".join(
        f"{k} {v:.4g}" for k, v in metrics.items() if k not in ("step", "lr")
    )
    return f"[{step:7d}]  {terms}"


class Trainer:
    """JEPA trainer: jitted AdamW step, metrics logging, best-on-val checkpointing.

    The *only* checkpoint the trainer writes is the one that beats the best
    validation loss so far -- weights on disk go through
    :mod:`src.utils.checkpoint` (``best.json`` records the winning tag).
    """

    def __init__(self, model: JEPA, config: TrainConfig):
        self.model = model
        self.config = config
        self.schedule = config.schedule()

        # NNX optimizer wraps the model (Params target by default), so its
        # state rides along inside the jitted step with no extra plumbing.
        self.optimizer = nnx.Optimizer(
            model, config.optimizer(self.schedule), wrt=nnx.Param
        )

        self.history: list[dict] = []  # one dict per logged step, `step` included
        self.step = 0
        self.best_val = float("inf")  # lowest val_loss seen so far
        self.best_step = 0
        self.best_tag: str | None = None

        # The module graph never changes, so define it once here; only the
        # State crosses the jit boundary, and a step never retraces.
        self._graphdef, _ = nnx.split((self.model, self.optimizer))
        self._train_step = self._build_train_step()

    def _build_train_step(self):
        def _loss(model, pixels, actions):
            losses = model.loss(pixels, actions)
            return losses["loss"], losses  # differentiated + aux (metrics)

        @jax.jit
        def step(state, pixels, actions):
            model, optimizer = nnx.merge(self._graphdef, state)
            (_, losses), grads = nnx.value_and_grad(_loss, has_aux=True)(
                model, pixels, actions
            )

            # NaN/inf guard: one bad mini-batch (e.g. a SIGReg overflow on a
            # near-degenerate batch of frames) must not poison the weights.
            # Zero such gradients; the optimizer then leaves params unchanged
            # (moments just decay) and training continues on the next batch.
            finite = jnp.isfinite(losses["loss"])
            grads = jax.tree.map(
                lambda g: jnp.where(finite, g, jnp.zeros_like(g)), grads
            )

            optimizer.update(model, grads)
            return nnx.split((model, optimizer))[1], losses

        return step

    def train(
        self,
        train_iter,
        *,
        num_iters: int | None = None,
        log_interval: int = 100,
        eval_interval: int = 100,
        val_batches=(),
        ckpt_dir: str | Path = "checkpoints",
        log_fn=None,
    ) -> list[dict]:
        """Run ``num_iters`` steps, logging every ``log_interval`` and validating
        every ``eval_interval``; each new best ``val_loss`` is checkpointed.
        """
        num_iters = num_iters or self.config.decay_steps
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        val_batches = list(val_batches)
        if not val_batches or eval_interval < 1:
            print("warning: no validation pool, so no checkpoint will be written")

        batches = iter(train_iter)
        final = self.step + num_iters

        for t in range(self.step, final):
            batch = next(batches)
            state, losses = self._train_step(
                nnx.state((self.model, self.optimizer)),
                batch["pixels"],
                batch["actions"],
            )
            nnx.update((self.model, self.optimizer), state)
            self.step = step = t + 1

            metrics = {"step": step, "lr": float(self.schedule(t)), **losses}
            improved = False
            validate = bool(val_batches) and (
                step % eval_interval == 0 or step == final
            )
            if validate:
                metrics.update(self._validate(val_batches))
                improved = self._save_if_best(ckpt_dir, metrics["val_loss"])

            if step % log_interval == 0 or validate:
                self.history.append(metrics)
                print(_log_line(step, metrics) + ("  <- best" if improved else ""))
                if log_fn is not None:
                    log_fn(step, metrics)

        return self.history

    def _validate(self, val_batches: list) -> dict:
        """Mean of every loss term over the fixed pool (not jitted)."""
        totals: dict[str, float] = {}
        for batch in val_batches:
            losses = self.model.loss(batch["pixels"], batch["actions"])
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v)
        return {f"val_{k}": v / len(val_batches) for k, v in totals.items()}

    def _save_if_best(self, ckpt_dir: Path, val_loss) -> bool:
        """Write ``best_<step>`` when this ``val_loss`` beats the best so far.

        Tags carry the step, so reruns never collide, and a non-finite value
        never counts as an improvement.
        """
        val_loss = float(val_loss)
        if not math.isfinite(val_loss) or val_loss >= self.best_val:
            return False
        self.best_val, self.best_step = val_loss, self.step
        self.best_tag = f"best_{self.step}"
        checkpoint.save(self.model, ckpt_dir, tag=self.best_tag)
        checkpoint.record_best(ckpt_dir, self.best_tag, self.best_step, self.best_val)
        return True


if __name__ == "__main__":

    def fake_batches(rngs, batch=4, frames=4):
        """Endless random (pixels, actions) batches, as ``PushT.batches`` would."""
        while True:
            key = rngs()
            yield {
                "pixels": jax.random.uniform(
                    key, (batch, frames, 64, 64, 3), minval=0.0, maxval=1.0
                ),
                "actions": jax.random.normal(
                    jax.random.fold_in(key, 1), (batch, frames, 2)
                ),
            }

    model_cfg = ModelConfig(
        embed_dim=32, enc_layers=1, enc_heads=2, pred_depth=1, pred_heads=2,
        pred_mlp_dim=64,  # tiny, so this smoke test stays fast on CPU
    )
    model = JEPA(model_cfg, nnx.Rngs(0))
    trainer = Trainer(
        model=model,
        config=TrainConfig(peak_lr=3e-4, warmup_steps=10, decay_steps=30),
    )

    batches = fake_batches(nnx.Rngs(1))
    pool = [next(batches) for _ in range(2)]
    trainer.train(
        batches,
        num_iters=6,
        log_interval=3,
        eval_interval=3,
        val_batches=pool,
        ckpt_dir="/tmp/opencode/train_ckpt",
    )
    print(f"best: {trainer.best_tag} val_loss {trainer.best_val:.4f}")

    fresh = next(batches)
    final = model.loss(fresh["pixels"], fresh["actions"])["loss"]
    print(f"final loss:  {final:.4f}")

    init = jax.random.uniform(jax.random.key(9), (2, 3, 64, 64, 3))
    acts = jax.random.normal(jax.random.key(8), (2, 2, 2))
    preds, _goal = model.rollout(init, acts, init[0, -1])
    print(f"rollout:     {preds.shape}")

    checkpoint.save(model, "/tmp/opencode/train_ckpt", tag="latest")
    reloaded = checkpoint.load(
        JEPA(model_cfg, nnx.Rngs(0)),
        ckpt_dir="/tmp/opencode/train_ckpt",
        tag="latest",
    )
    assert jnp.allclose(
        reloaded.loss(fresh["pixels"], fresh["actions"])["loss"], final
    )
