from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..models import LeWorldModel


@dataclass
class CEMConfig:
    horizon: int = 5
    num_samples: int = 300
    num_elites: int = 30
    num_iterations: int = 10
    action_dim: int = 2
    action_min: float = -1.0
    action_max: float = 1.0
    init_var: float = 1.0
    var_scale: float = 1.0
    elite_frac: float = 0.1


class CEMPlanner:
    def __init__(
        self,
        model: LeWorldModel,
        config: CEMConfig,
        seed: int = 0,
    ):
        self.model = model
        self.config = config
        self.key = jax.random.key(seed)

    def plan(
        self,
        init_pixels: jax.Array,
        goal_pixels: jax.Array,
    ) -> jax.Array:
        """
        Plan action sequence to reach goal.
        Args:
            init_pixels: (H, H, W, C) or (H, W, C) - history frames
            goal_pixels: (H, W, C) - goal observation
        Returns:
            (horizon, action_dim) - best action sequence
        """
        cfg = self.config

        # Ensure batch dimension
        if init_pixels.ndim == 4:
            init_pixels = init_pixels[None]  # (1, H, H, W, C)
        if goal_pixels.ndim == 3:
            goal_pixels = goal_pixels[None]  # (1, H, W, C)

        # Encode goal
        goal_emb = self.model.encode(goal_pixels)  # (1, D)
        goal_emb = goal_emb[0]  # (D,)

        # Encode initial history
        init_emb = self.model.encode(init_pixels)  # (1, H, D)

        # Initialize action distribution at the CENTER of the action range.
        # A zero-mean init (e.g. old 'mean = zeros, var = 1') clips every
        # initial sample into one corner for non-symmetric bounds like
        # PushT's [0, 512], so the plan never explores. For a symmetric
        # [-1, 1] space this reduces to the classic mean=0, var=1.
        center = 0.5 * (cfg.action_min + cfg.action_max)
        spread = 0.5 * (cfg.action_max - cfg.action_min)
        std0 = spread * cfg.init_var
        mean = jnp.full((cfg.horizon, cfg.action_dim), center)
        var = jnp.full((cfg.horizon, cfg.action_dim), std0**2)

        for i in range(cfg.num_iterations):
            self.key, subkey = jax.random.split(self.key)

            # Sample action sequences
            actions = mean + jnp.sqrt(var) * jax.random.normal(
                subkey, (cfg.num_samples, cfg.horizon, cfg.action_dim)
            )
            actions = jnp.clip(actions, cfg.action_min, cfg.action_max)

            # Evaluate all samples in parallel
            costs = self._evaluate_actions(init_emb, goal_emb, actions)

            # Select elites
            elite_idx = jnp.argsort(costs)[: cfg.num_elites]
            elite_actions = actions[elite_idx]

            # Update distribution
            mean = jnp.mean(elite_actions, axis=0)
            var = jnp.var(elite_actions, axis=0) * cfg.var_scale

        return mean

    def _evaluate_actions(
        self,
        init_emb: jax.Array,
        goal_emb: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """
        Evaluate batch of action sequences.
        Args:
            init_emb: (1, H, D)
            goal_emb: (D,)
            actions: (N, horizon, action_dim)
        Returns:
            (N,) - costs
        """
        N, _, _ = actions.shape

        # Repeat init_emb for all samples
        init_emb = jnp.repeat(init_emb, N, axis=0)  # (N, H, D)

        # Rollout in latent space (starts from pre-encoded embeddings)
        pred_emb = self.model.rollout_from_embeddings(init_emb, actions)  # (N, T, D)

        # Final state cost
        final_emb = pred_emb[:, -1]  # (N, D)
        costs = jnp.sum((final_emb - goal_emb) ** 2, axis=-1)

        return costs

    def plan_mpc(
        self,
        env,
        init_pixels: jax.Array,
        goal_pixels: jax.Array,
        exec_horizon: int = 1,
    ) -> tuple:
        """
        Model Predictive Control: plan, execute first K actions, replan.
        Returns:
            (actions_executed, observations, rewards)
        """
        all_actions = []
        all_obs = [init_pixels[-1] if init_pixels.ndim == 4 else init_pixels]
        all_rewards = []

        curr_pixels = init_pixels

        while True:
            # Plan
            action_seq = self.plan(curr_pixels, goal_pixels)  # (horizon, action_dim)

            # Execute first exec_horizon actions
            for i in range(min(exec_horizon, self.config.horizon)):
                action = action_seq[i]
                obs, reward, terminated, truncated, _ = env.step(np.array(action))
                all_actions.append(action)
                all_obs.append(obs)
                all_rewards.append(reward)

                if terminated or truncated:
                    return (
                        np.array(all_actions),
                        np.array(all_obs),
                        np.array(all_rewards),
                    )

                # Update history
                if curr_pixels.ndim == 4:
                    # Shift history window
                    curr_pixels = jnp.concatenate(
                        [curr_pixels[:, 1:], obs[None, None]], axis=1
                    )
                else:
                    curr_pixels = obs

        return np.array(all_actions), np.array(all_obs), np.array(all_rewards)


def create_planner(model: LeWorldModel, **kwargs) -> CEMPlanner:
    """Factory function to create CEM planner"""
    config = CEMConfig(**kwargs)
    return CEMPlanner(model, config)
