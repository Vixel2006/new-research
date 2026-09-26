from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class CEMPlanner:
    """Cross-entropy planner over the single :meth:`JEPA.rollout` interface.

    The whole CEM loop is jitted: ``num_iterations`` rounds of
    sample -> rollout -> elite update compile into a single graph, so replanning
    at every environment step pays dispatch, not tracing.

    The model is *traced*, not closed over. ``nnx.BatchNorm`` updates its running
    statistics in place on every call, and flax rejects such a mutation across
    trace levels, so the state is passed in as an argument and merged inside the
    trace: the rollout works on a throwaway copy and the update is discarded.
    That is harmless because ``use_running_average=False`` means the
    normalization always uses the current batch's statistics anyway -- which is
    also why a single-frame encoding is ill-defined, and why predictions and the
    goal embedding have to come out of one rollout call.
    """

    model: nnx.Module = field(repr=False)
    horizon: int = 8
    action_dim: int = 2
    num_samples: int = 128
    topk: int = 10
    num_iterations: int = 5
    action_min: float = -1.0
    action_max: float = 1.0

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.action_dim < 1:
            raise ValueError("horizon and action_dim must be positive")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not 1 <= self.topk <= self.num_samples:
            raise ValueError("topk must be between 1 and num_samples")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if self.action_min >= self.action_max:
            raise ValueError("action_min must be smaller than action_max")

        # Frozen dataclass, so stash the compiled entry point and the state it
        # is called with. The graphdef closes over the call (static structure);
        # only the leaves travel as an argument, exactly like the train step.
        graphdef, params = nnx.split(self.model)

        def search(
            params, history, goal_pixels, key, init_mean, init_std, hist_actions
        ):
            model = nnx.merge(graphdef, params)
            lower = jnp.asarray(self.action_min)
            upper = jnp.asarray(self.action_max)
            midpoint = 0.5 * (lower + upper)
            half_range = 0.5 * (upper - lower)

            mean, std = self._initial_distribution(
                midpoint, half_range, init_mean, init_std
            )
            trajectory_batch = jnp.repeat(history[None], self.num_samples, axis=0)
            batched_history_actions = self._batch_history_actions(
                history.shape[0], hist_actions
            )

            for _ in range(self.num_iterations):
                key, sample_key = jax.random.split(key)
                actions = self._sample(sample_key, mean, std, lower, upper)
                costs = self._evaluate(
                    model.rollout,
                    trajectory_batch,
                    actions,
                    goal_pixels,
                    batched_history_actions,
                )
                mean, std = self._update_distribution(actions, costs, half_range)

            return mean

        object.__setattr__(self, "_params", params)
        object.__setattr__(self, "_search", jax.jit(search))

    def plan(
        self,
        history: jax.Array,
        goal_pixels: jax.Array,
        *,
        seed: int = 0,
        init_mean=None,
        init_std=None,
        history_actions=None,
    ) -> jax.Array:
        """Best action sequence found, (horizon, action_dim); run its first step.

        ``init_mean``/``init_std`` seed the CEM distribution (both ``None`` for
        the default midpoint start).
        """
        return self._search(
            self._params,
            jnp.asarray(history),
            jnp.asarray(goal_pixels),
            jax.random.key(seed),
            init_mean,
            init_std,
            history_actions,
        )

    def _initial_distribution(self, midpoint, half_range, init_mean, init_std):
        shape = (self.horizon, self.action_dim)
        mean = (
            jnp.asarray(init_mean, dtype=jnp.float32)
            if init_mean is not None
            else jnp.full(shape, midpoint)
        )
        std = (
            jnp.asarray(init_std, dtype=jnp.float32)
            if init_std is not None
            else jnp.full(shape, half_range)
        )
        return mean, std

    def _batch_history_actions(self, history_size, history_actions):
        if history_actions is None:
            actions = jnp.zeros((history_size - 1, self.action_dim), dtype=jnp.float32)
        else:
            actions = jnp.asarray(history_actions, dtype=jnp.float32)
        if actions.ndim == 2:
            actions = actions[None]
        return jnp.repeat(actions, self.num_samples, axis=0)

    def _sample(self, key, mean, std, lower, upper):
        noise = jax.random.normal(
            key, (self.num_samples, self.horizon, self.action_dim)
        )
        return jnp.clip(mean + std * noise, lower, upper)

    def _evaluate(
        self, rollout, trajectory_batch, actions, goal_pixels, history_actions
    ):
        # Both outputs come from one rollout call, so predictions and the goal
        # embedding share the same BatchNorm batch.
        predictions, goal_embedding = rollout(
            trajectory_batch, actions, goal_pixels, history_actions
        )
        return jnp.mean((predictions[:, -1] - goal_embedding) ** 2, axis=-1)

    def _update_distribution(self, actions, costs, half_range):
        topk = actions[jnp.argsort(costs)[: self.topk]]
        mean = jnp.mean(topk, axis=0)
        std = jnp.std(topk, axis=0) + 1e-3 * half_range
        return mean, std
