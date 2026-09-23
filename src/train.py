from dataclasses import dataclass, field
from pathlib import Path

import jax
import optax
from flax import nnx
from orbax import checkpoint as orbax

try:
    from .jepa import JEPA
    from .modules import SIGReg
except ImportError:
    from jepa import JEPA
    from modules import SIGReg


def create_lr_schedule(
    peak_lr: float,
    warmup_steps: int,
    decay_steps: int,
    end_lr: float = 0.0,
) -> optax.Schedule:
    """Linear warmup up to ``peak_lr``, then cosine decay to ``end_lr``."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_lr,
    )


def create_optimizer(
    peak_lr: float = 3e-4,
    warmup_steps: int = 1_000,
    decay_steps: int = 100_000,
    end_lr: float = 0.0,
    weight_decay: float = 0.05,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    grad_clip: float | None = 1.0,
) -> optax.GradientTransformation:
    """AdamW with warmup + cosine LR schedule, plus global-norm grad clipping."""
    schedule = create_lr_schedule(peak_lr, warmup_steps, decay_steps, end_lr)

    tx = optax.adamw(
        learning_rate=schedule,
        weight_decay=weight_decay,
        b1=b1,
        b2=b2,
        eps=eps,
    )
    if grad_clip is not None and grad_clip > 0:
        tx = optax.chain(optax.clip_by_global_norm(grad_clip), tx)
    return tx


@dataclass
class MetricHistory:
    """Ring-buffer of per-step training metrics.

    Stores every logged step; use :meth:`mean` / :meth:`last` / :meth:`sma`
    to summarise the run. Small enough to keep the whole run in memory.
    """

    steps: list[int] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)

    def record(self, step: int, metrics: dict) -> None:
        self.steps.append(step)
        self.metrics.append(dict(metrics))

    def last(self) -> dict:
        """Most recently recorded metrics (or an empty dict)."""
        return self.metrics[-1] if self.metrics else {}

    def mean(self, key: str, since: int = 0) -> float:
        """Mean of ``key`` over all recorded steps >= ``since``."""
        vals = [
            m[key] for m, s in zip(self.metrics, self.steps) if s >= since and key in m
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    def sma(self, key: str, window: int = 100) -> float:
        """Simple moving average of ``key`` over the last ``window`` steps."""
        vals = [m[key] for m in self.metrics[-window:] if key in m]
        return sum(vals) / len(vals) if vals else float("nan")

    def lr(self) -> float:
        """Current learning rate (the last recorded one)."""
        m = self.last()
        return m.get("lr", float("nan"))


@dataclass
class TrainConfig:
    """Training hyperparameters for the JEPA trainer."""

    peak_lr: float = 3e-4
    warmup_steps: int = 1_000
    decay_steps: int = 100_000
    end_lr: float = 0.0
    weight_decay: float = 0.05
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    grad_clip: float | None = 1.0


class Trainer:
    """JEPA trainer: jitted AdamW step, metric tracking, orbax checkpointing."""

    def __init__(
        self,
        model: JEPA,
        sigreg_fn: SIGReg,
        config: "TrainConfig",
        optimizer: optax.GradientTransformation | None = None,
    ):
        self.model = model
        self.sigreg_fn = sigreg_fn
        self.config = config
        self.tx = optimizer or create_optimizer(
            peak_lr=config.peak_lr,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
            end_lr=config.end_lr,
            weight_decay=config.weight_decay,
            b1=config.b1,
            b2=config.b2,
            eps=config.eps,
            grad_clip=config.grad_clip,
        )

        # NNX optimizer wraps the model (Params target by default), so its
        # state rides along inside the jitted step with no extra plumbing.
        self.optimizer = nnx.Optimizer(model, self.tx, wrt=nnx.Param)

        self.history = MetricHistory()
        self._step = 0
        self._train_step = self._build_train_step()

    def _build_train_step(self):
        """Split out graph + state once, then jit a pure function state -> state.

        Returns a callable ``step(state, pixels, actions) -> (state, metrics)``
        where ``state`` is the merged NNX State of (model, optimizer).
        """
        graphdef, state = nnx.split((self.model, self.optimizer))

        def _loss(model, pixels, actions):
            losses = model.loss(pixels, actions, self.sigreg_fn)
            return losses["loss"], losses  # scalar (loss) + aux (metrics dict)

        @jax.jit
        def step(state, pixels, actions):
            model, optimizer = nnx.merge(graphdef, state)

            (loss, losses), grads = nnx.value_and_grad(_loss, has_aux=True)(
                model, pixels, actions
            )

            optimizer.update(model, grads)

            metrics = {
                "loss": loss,
                "mse": losses["mse"],
                "sigreg": losses["sigreg"],
            }

            state = nnx.split((model, optimizer))[1]
            return state, metrics

        return step

    def train(
        self,
        train_iter,
        *,
        num_iters: int | None = None,
        log_interval: int = 100,
        ckpt_interval: int = 1_000,
        ckpt_dir: str | Path = "checkpoints",
        log_fn=None,
        verbose: bool = True,
    ) -> MetricHistory:
        num_iters = num_iters or self.config.decay_steps
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        schedule = create_lr_schedule(
            self.config.peak_lr,
            self.config.warmup_steps,
            self.config.decay_steps,
            self.config.end_lr,
        )

        train_iter = iter(train_iter)
        start = self._step

        for t in range(start, start + num_iters):
            batch = next(train_iter)
            pixels = batch["pixels"] if isinstance(batch, dict) else batch[0]
            actions = batch["actions"] if isinstance(batch, dict) else batch[1]

            # Split inside the loop so each jit call starts from the latest
            # merged state (models/optimizers are small; only arrays are traced).
            graphdef, state = nnx.split((self.model, self.optimizer))
            state, metrics = self._train_step(state, pixels, actions)
            nnx.update((self.model, self.optimizer), state)

            # Restore self._step from the recorded step (avoids drift).
            self._step = t + 1
            lr = float(schedule(t))
            metrics = {"step": self._step, "lr": lr, **metrics}

            if self._step % log_interval == 0 or self._step == start + num_iters:
                self.history.record(self._step, dict(metrics))
                if verbose:
                    print(
                        f"[{self._step:7d}] "
                        f"loss {float(metrics['loss']):.4f} "
                        f"mse {float(metrics['mse']):.4f} "
                        f"sigreg {float(metrics['sigreg']):.4f} "
                        f"lr {lr:.2e}"
                    )
                if log_fn is not None:
                    log_fn(self._step, metrics)

            if ckpt_interval and self._step % ckpt_interval == 0:
                self.save_checkpoint(ckpt_dir, tag=str(self._step))

        return self.history

    def save_checkpoint(self, ckpt_dir: str | Path, tag: str = "latest") -> Path:
        path = Path(ckpt_dir) / tag
        ckptr = orbax.PyTreeCheckpointer()
        ckptr.save(
            path,
            nnx.state((self.model, self.optimizer), self.sigreg_fn),
            force=True,
        )
        return path

    @classmethod
    def load_checkpoint(cls, model: JEPA, sigreg_fn: SIGReg | None = None):
        raise NotImplementedError(
            "load_checkpoint is not implemented yet; pass a pre-built model "
            "and SIGReg into Trainer to resume."
        )


if __name__ == "__main__":
    import jax.numpy as jnp

    class FakeBatcher:
        """Endless iterator of random (pixels, actions) batches.

        Each batch is a dict with "pixels" (B, T, H, W, C) and "actions"
        (B, T, action_dim), matching what Trainer.train() unpacks.
        """

        def __init__(self, rngs: nnx.Rngs, dtype=jnp.float32):
            self.rngs = rngs
            self.dtype = dtype

        def __iter__(self):
            return self

        def __next__(self):
            key = self.rngs()
            pixels = jax.random.uniform(
                key, (4, 4, 64, 64, 3), minval=0.0, maxval=1.0, dtype=self.dtype
            )
            key2 = jax.random.fold_in(key, 1)
            actions = jax.random.normal(key2, (4, 4, 2), dtype=self.dtype)
            return {"pixels": pixels, "actions": actions}

    rngs = nnx.Rngs(0)
    model = JEPA(rngs, img_size=64, action_dim=2, history_size=3)
    sigreg = SIGReg(embed_dim=model.embed_dim)

    trainer = Trainer(
        model=model,
        sigreg_fn=sigreg,
        config=TrainConfig(
            peak_lr=3e-4,
            warmup_steps=10,
            decay_steps=30,
            end_lr=0.0,
        ),
    )

    batcher = FakeBatcher(nnx.Rngs(1))
    history = trainer.train(
        batcher,
        num_iters=6,
        log_interval=1,
        verbose=True,
    )

    print(
        f"\nfinal loss:  {trainer.model.loss(batcher.__next__()['pixels'], batcher.__next__()['actions'])['loss']:.4f}"
    )

    init = jax.random.uniform(jax.random.key(9), (2, 3, 64, 64, 3))
    acts = jax.random.normal(jax.random.key(8), (2, 2, 2))
    preds = trainer.model.rollout(init, acts)
    print(f"rollout:     {preds.shape}")
    print("\nsmoke test OK ✓")
