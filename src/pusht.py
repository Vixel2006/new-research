from __future__ import annotations

from collections.abc import Iterator

import gym_pusht  # noqa: F401  (importing registers the "gym_pusht" namespace)
import gymnasium as gym
import jax.numpy as jnp
import numpy as np

IMG_SIZE = 64  # frame side length, must match the ViT's img_size
ACTION_MAX = 512.0  # agent puck target range, in screen coordinates
PUSH_THRESHOLD = 3.0  # block displacement (512-space px) that counts as a push
HEURISTIC_MIX = 0.5  # share of `mixed` episodes that push toward the goal


class PushT:
    def __init__(
        self,
        seed: int = 0,
        *,
        history_size: int = 3,
        batch_size: int = 16,
        policy: str = "mixed",
        max_episode_steps: int = 150,
        buffer_capacity: int = 512,
    ):
        self.env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels",
            observation_width=IMG_SIZE,
            observation_height=IMG_SIZE,
            render_mode="rgb_array",
        )
        self.env.reset(seed=seed)
        self.history_size = history_size
        self.batch_size = batch_size
        self.policy = policy
        self.max_episode_steps = max_episode_steps
        self.buffer_capacity = buffer_capacity
        self._rng = np.random.default_rng(seed + 1)
        self._sampler = np.random.default_rng(seed + 2)
        self._heuristic_episode = False
        # Replay buffer of (H+1 frames, H+1 actions, block displacement).
        self._samples: list[tuple[np.ndarray, np.ndarray, float]] = []

    def reset(self, seed: int | None = None, state=None) -> np.ndarray:
        """Reset the env and return the first model frame (H, W, 3)."""
        options = {"reset_to_state": state} if state is not None else None
        obs, _ = self.env.reset(seed=seed, options=options)
        return self._to_frame(obs)

    def step(self, action) -> tuple[np.ndarray, bool, dict]:
        """Apply an action; return the next model frame, done, and info."""
        obs, _reward, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        return self._to_frame(obs), terminated or truncated, info

    def goal_frame(self) -> np.ndarray:
        """The fixed goal pose [agent_x, agent_y, block_x, block_y, theta] as a
        model frame: the T-block sitting in the green goal zone.

        The goal reaches the model as pixels, so it is encoded in the same
        BatchNorm batch as the rollout context.
        """
        return self.reset(state=np.array([256.0, 400.0, 256.0, 256.0, np.pi / 4]))

    @staticmethod
    def _to_frame(obs) -> np.ndarray:
        """uint8 (H, W, 3) observation -> float32 [0, 1] model input."""
        return np.asarray(obs, dtype=np.float32) / 255.0

    def batches(self) -> Iterator[dict]:
        """Endless ``{pixels, actions}`` batches for ``Trainer.train``."""
        while True:
            yield self.next_batch()

    def next_batch(self) -> dict:
        """One batch::

            {"pixels":  (B, H + 1, 64, 64, 3) float32 in [0, 1],
             "actions": (B, H + 1, 2)         float32 in [0, 512]}

        Each sample is a contiguous H+1-frame window with the H+1 actions
        aligned to it, i.e. one teacher-forced next-latent target per window.
        """
        while len(self._samples) < self.batch_size:
            self._collect_episode()

        # Sample WITH REPLACEMENT rather than draining the buffer: a buffer
        # hovering near empty yields one ultra-correlated episode per batch,
        # which blows SIGReg up to NaN. The eviction below streams fresh
        # episodes back in, so the pool spans many episodes either way.
        disps = np.array([s[2] for s in self._samples], dtype=np.float32)
        pushing = np.flatnonzero(disps > PUSH_THRESHOLD)
        static = np.flatnonzero(disps <= PUSH_THRESHOLD)
        n_push = int(0.75 * self.batch_size)

        # Bias 3/4 of the batch toward pushing windows: agent motion dominates
        # the buffer and would teach "only the puck moves", which is fatal for
        # planning (CEM's cost then has no signal about block/goal dynamics).
        # The rest trains agent motion + static structure.
        parts = []
        if len(pushing) >= n_push:
            parts.append(self._sampler.choice(pushing, size=n_push, replace=True))
            n_static = self.batch_size - n_push
            if len(static) >= n_static:
                parts.append(self._sampler.choice(static, size=n_static, replace=True))
        n_have = sum(len(p) for p in parts)
        if n_have < self.batch_size:  # a class is still short: top up uniformly
            parts.append(
                self._sampler.integers(
                    0, len(self._samples), size=self.batch_size - n_have
                )
            )
        idx = np.concatenate(parts)

        windows = [self._samples[i] for i in idx]
        return {
            "pixels": jnp.asarray(np.stack([w[0] for w in windows]), dtype=jnp.float32),
            "actions": jnp.asarray(
                np.stack([w[1] for w in windows]), dtype=jnp.float32
            ),
        }

    def warmup(self, episodes: int = 8) -> None:
        """Force-collect episodes before the buffer is sampled.

        Call this before snapshotting a validation pool: a buffer that still
        holds only an episode or two of near-static frames is degenerate and
        makes val loss meaningless (the model can hit ~0 by predicting "the
        frame barely changes").
        """
        for _ in range(episodes):
            self._collect_episode()

    def _collect_episode(self) -> None:
        """Roll one episode of the data policy, appending a window per step."""
        self._heuristic_episode = self.policy == "heuristic" or (
            self.policy == "mixed" and self._rng.uniform() < HEURISTIC_MIX
        )
        frames = [self.reset()]
        actions: list[np.ndarray] = []
        block_poses: list[np.ndarray] = []  # (2,) 512-space, per step
        info = None

        for _ in range(self.max_episode_steps):
            if len(frames) > self.history_size + 2:
                frames.pop(0)  # keep per-episode memory bounded
                actions.pop(0)
                block_poses.pop(0)

            action = self._sample_action(info)
            frame, done, info = self.step(action)
            frames.append(frame)
            actions.append(action)
            block_poses.append(np.asarray(info["block_pose"][:2], dtype=np.float32))

            if len(actions) >= self.history_size + 1:
                self._add_window(frames, actions, block_poses)
            if done:
                break

        self._evict()

    def _add_window(self, frames, actions, block_poses) -> None:
        """Append the trailing H+1-frame window and its aligned actions.

        A training sample needs H+1 frames AND the H+1 actions ALIGNED with
        them: action[i] is the action taken AT frame[i] (it moves frame[i] ->
        frame[i+1]). ``frames`` always has one more element than ``actions``
        (the initial obs plus one frame per step), so the naive paired slice is
        off by one -- it would hand the model the action that *produced* frame[i]
        while dropping the action that produces the target frame. Duplicate the
        causal action into the unused last slot; the loss only reads the first H.
        """
        H = self.history_size
        self._samples.append(
            (
                np.stack(frames[-H - 1 :]),
                np.concatenate([np.stack(actions[-H:]), actions[-1:]]),
                # How far the T-block moved across the window: ~0 is a static
                # scene or agent-only motion, > ~2 is a real push. This is what
                # `next_batch` biases sampling toward.
                float(np.linalg.norm(block_poses[-1] - block_poses[-H - 1])),
            )
        )

    def _evict(self) -> None:
        """Trim the replay buffer, preferring to drop NON-pushing windows.

        A plain FIFO trim over ~150-step mixed episodes ends up ~90% "agent
        wanders / block static" windows (the pushing phases get evicted), which
        starves the predictor of action->outcome signal and makes it plan-blind
        (CEM's cost has no action dependence). Pushes age out only once no old
        non-push window remains.
        """
        over = len(self._samples) - self.buffer_capacity
        if over <= 0:
            return
        stale = [i for i, s in enumerate(self._samples) if s[2] <= PUSH_THRESHOLD]
        drop = set(stale[:over])
        if len(drop) < over:  # ran out of non-push: fall back to oldest
            survivors = [i for i in range(len(self._samples)) if i not in drop]
            drop |= set(survivors[: over - len(drop)])
        self._samples = [s for i, s in enumerate(self._samples) if i not in drop]

    def _sample_action(self, info) -> np.ndarray:
        """One exploratory action: uniform anywhere, or a push from behind."""
        if self._heuristic_episode and info is not None:
            return self._push_toward_goal(info)
        return self._rng.uniform(0.0, ACTION_MAX, size=2).astype(np.float32)

    def _push_toward_goal(self, info) -> np.ndarray:
        """Stand behind the block (its far side from the goal) and push it
        toward the goal. These near-goal transitions are what the model needs
        to learn the goal configuration. Noisy so the data isn't constant."""
        block = np.asarray(info["block_pose"][:2])
        goal = np.asarray(info["goal_pose"][:2])
        away = block - goal
        dist = np.linalg.norm(away)
        away = np.array([1.0, 0.0]) if dist < 1e-6 else away / dist
        push = block + 45.0 * away + self._rng.normal(0.0, 15.0, size=2)
        return np.clip(push, 0.0, ACTION_MAX)

    def run_mpc(self, planner, *, n_steps: int = 200, seed: int = 0) -> dict:
        """Execute one episode of receding-horizon CEM planning.

        The planner owns the compiled rollout and the search settings, so only
        the planner is needed here. Returns the episode's frames, per-step
        coverage, whether the goal was reached, and the step count.
        """
        H = self.history_size
        goal_frame = self.goal_frame()
        history = [self.reset(seed=seed)] * H  # pad by repeating the 1st frame
        executed: list[np.ndarray] = []  # for the predictor's action context
        frames, coverage, done, info = [], [], False, None
        mean = std = None  # CEM start point, seeded from the first observation

        for step in range(n_steps):
            mean, std = self._push_seed(info, planner)
            action = planner.plan(
                np.stack(history),
                goal_frame,
                seed=seed + step,
                init_mean=mean,
                init_std=std,
                history_actions=self._history_actions(executed, planner),
            )[0]

            frame, done, info = self.step(action)
            frames.append(frame)
            coverage.append(float(info["coverage"]))
            history = (history + [frame])[-H:]
            executed.append(np.asarray(action, dtype=np.float32))
            if done:
                break

        return {
            "frames": np.array(frames),
            "coverage": np.array(coverage),
            "success": done,
            "steps_taken": len(executed),
        }

    def random_baseline(self, *, n_steps: int = 200, seed: int = 0) -> np.ndarray:
        """Per-step coverage of one uniform-random episode (context for MPC)."""
        self.reset(seed=seed)
        rng = np.random.default_rng(seed + 999)
        coverage = []
        for _ in range(n_steps):
            _frame, done, info = self.step(rng.uniform(0.0, ACTION_MAX, size=2))
            coverage.append(float(info["coverage"]))
            if done:
                break
        return np.array(coverage)

    def _push_seed(self, info, planner) -> tuple[np.ndarray | None, np.ndarray | None]:
        """CEM's initial distribution: the block's push point, behind it and
        away from the goal. Uniform samples over the whole 512x512 box almost
        never touch the block, so the search needs a good starting mean."""
        mean = std = None
        if info is not None:
            block = np.asarray(info["block_pose"][:2], dtype=np.float32)
            away = block - np.asarray(info["goal_pose"][:2], dtype=np.float32)
            dist = np.linalg.norm(away)
            if dist > 1e-6:
                push = block + 45.0 * away / dist
                mean = np.tile(
                    np.clip(push, planner.action_min, planner.action_max),
                    (planner.horizon, 1),
                ).astype(np.float32)
                std = np.full(
                    (planner.horizon, planner.action_dim), 40.0, dtype=np.float32
                )
        return mean, std

    def _history_actions(self, executed, planner) -> np.ndarray:
        """The last H-1 executed actions, zero-padded at the episode start."""
        n = self.history_size - 1
        out = np.zeros((max(n, 0), planner.action_dim), dtype=np.float32)
        if n > 0 and executed:
            recent = np.asarray(executed[-n:], dtype=np.float32)
            out[-len(recent) :] = recent
        return out

    def close(self) -> None:
        self.env.close()


if __name__ == "__main__":
    from flax import nnx

    from jepa import JEPA, ModelConfig
    from planner import CEMPlanner

    env = PushT(seed=0, batch_size=4)
    print(f"goal frame: {env.goal_frame().shape}")

    env.warmup(episodes=1)
    batch = env.next_batch()
    print(
        f"batch:      pixels {batch['pixels'].shape}  actions {batch['actions'].shape}"
    )

    model = JEPA(ModelConfig(img_size=IMG_SIZE), nnx.Rngs(0))
    planner = CEMPlanner(
        model,
        horizon=2,
        num_samples=8,
        topk=2,
        num_iterations=2,
        action_min=0.0,
        action_max=ACTION_MAX,
    )
    res = env.run_mpc(planner, n_steps=2, seed=3)
    print(f"mpc:        steps {res['steps_taken']}  cov {res['coverage']}")
    print(f"baseline:   cov {env.random_baseline(n_steps=2, seed=3)}")

    assert batch["pixels"].shape == (4, 4, 64, 64, 3)
    assert batch["actions"].shape == (4, 4, 2)
    assert len(res["frames"]) == len(res["coverage"]) == res["steps_taken"]
    env.close()
