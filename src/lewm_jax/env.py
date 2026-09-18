"""Gym environment wrapper and data collection for LeWorldModel"""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
import jax.numpy as jnp
import jax


@dataclass
class EnvConfig:
    """Environment configuration"""
    env_id: str = "PushT-v1"  # or custom
    img_size: int = 64
    frameskip: int = 5
    max_episode_steps: int = 200
    render_mode: str = "rgb_array"


class GymEnvWrapper:
    """Wrapper for gym environments to collect trajectories"""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.env = gym.make(config.env_id, render_mode=config.render_mode)
        self.img_size = config.img_size

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        return self._process_obs(obs), info

    def step(self, action: np.ndarray):
        # Repeat action for frameskip
        total_reward = 0.0
        for _ in range(self.config.frameskip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return self._process_obs(obs), total_reward, terminated, truncated, info

    def _process_obs(self, obs):
        """Process observation to (H, W, C) uint8"""
        if isinstance(obs, dict):
            # Handle dict observations (e.g., with pixels key)
            if "pixels" in obs:
                img = obs["pixels"]
            elif "image" in obs:
                img = obs["image"]
            else:
                # Try first array value
                for v in obs.values():
                    if isinstance(v, np.ndarray) and v.ndim >= 2:
                        img = v
                        break
        else:
            img = obs

        # Ensure 3 channels
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] > 3:
            img = img[..., :3]

        # Resize if needed
        if img.shape[:2] != (self.img_size, self.img_size):
            from PIL import Image
            img = np.array(Image.fromarray(img).resize((self.img_size, self.img_size)))

        return img.astype(np.uint8)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space


def collect_trajectories(
    env: GymEnvWrapper,
    num_episodes: int,
    policy: Optional[Callable] = None,
    max_steps: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """
    Collect trajectories from environment.
    Args:
        env: GymEnvWrapper
        num_episodes: Number of episodes to collect
        policy: Function (obs) -> action, if None uses random actions
        max_steps: Max steps per episode
    Returns:
        dict with keys: pixels, actions, rewards, dones
    """
    if policy is None:
        policy = lambda obs: env.action_space.sample()

    max_steps = max_steps or env.config.max_episode_steps

    all_pixels = []
    all_actions = []
    all_rewards = []
    all_dones = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_pixels = [obs]
        ep_actions = []
        ep_rewards = []
        ep_dones = [False]

        for step in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_pixels.append(obs)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)

            if done:
                break

        all_pixels.append(np.stack(ep_pixels))  # (T+1, H, W, C)
        all_actions.append(np.stack(ep_actions))  # (T, action_dim)
        all_rewards.append(np.array(ep_rewards))
        all_dones.append(np.array(ep_dones))

    # Pad to same length
    max_len = max(len(p) for p in all_pixels)
    action_dim = all_actions[0].shape[-1]

    def pad_seq(seq, max_len, pad_value=0):
        if len(seq) >= max_len:
            return seq[:max_len]
        pad_shape = (max_len - len(seq),) + seq.shape[1:]
        return np.concatenate([seq, np.full(pad_shape, pad_value, dtype=seq.dtype)])

    pixels = np.stack([pad_seq(p, max_len) for p in all_pixels])  # (N, T, H, W, C)
    actions = np.stack([pad_seq(a, max_len - 1, 0) for a in all_actions])  # (N, T-1, action_dim)
    rewards = np.stack([pad_seq(r, max_len - 1, 0) for r in all_rewards])
    dones = np.stack([pad_seq(d, max_len, True) for d in all_dones])

    return {
        "pixels": pixels,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def make_dataset(
    trajectories: dict,
    history_size: int = 3,
    num_preds: int = 1,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 0,
):
    """
    Create batched dataset from trajectories for LeWM training.
    Each sample: (pixels: (H, W, C), actions: (action_dim)) x (history_size + num_preds)
    """
    pixels = trajectories["pixels"]  # (N, T, H, W, C)
    actions = trajectories["actions"]  # (N, T-1, action_dim)

    N, T, H, W, C = pixels.shape
    seq_len = history_size + num_preds

    # Create sliding windows
    windows = []
    for i in range(N):
        for t in range(T - seq_len + 1):
            pix_window = pixels[i, t:t+seq_len]  # (seq_len, H, W, C)
            act_window = actions[i, t:t+seq_len-1]  # (seq_len-1, action_dim)
            # Pad actions to seq_len (last action repeated)
            act_padded = np.concatenate([act_window, act_window[-1:]], axis=0)
            windows.append((pix_window, act_padded))

    pixels_batch = np.stack([w[0] for w in windows])  # (num_windows, seq_len, H, W, C)
    actions_batch = np.stack([w[1] for w in windows])  # (num_windows, seq_len, action_dim)

    # Normalize pixels to [0, 1]
    pixels_batch = pixels_batch.astype(np.float32) / 255.0

    # Create batches
    num_windows = len(windows)
    indices = np.arange(num_windows)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    for i in range(0, num_windows, batch_size):
        idx = indices[i:i+batch_size]
        yield {
            "pixels": jnp.array(pixels_batch[idx]),
            "actions": jnp.array(actions_batch[idx]),
        }


class Simple2DEnv:
    """Simple 2D point mass pushing task for quick testing"""

    def __init__(self, img_size: int = 64):
        self.img_size = img_size
        self.agent_pos = np.array([0.5, 0.5])
        self.block_pos = np.array([0.3, 0.3])
        self.target_pos = np.array([0.7, 0.7])
        self.action_space = gym.spaces.Box(-0.1, 0.1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        self.step_count = 0
        self.max_steps = 100

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.agent_pos = np.array([0.5, 0.5])
        self.block_pos = np.array([0.3, 0.3])
        self.target_pos = np.array([0.7, 0.7])
        self.step_count = 0
        return self._render(), {}

    def step(self, action):
        self.step_count += 1

        # Agent moves
        self.agent_pos = np.clip(self.agent_pos + action, 0, 1)

        # Push block if close
        dist = np.linalg.norm(self.agent_pos - self.block_pos)
        if dist < 0.08:
            self.block_pos = np.clip(self.block_pos + action * 0.5, 0, 1)

        # Reward: negative distance to target
        reward = -np.linalg.norm(self.block_pos - self.target_pos)

        terminated = np.linalg.norm(self.block_pos - self.target_pos) < 0.05
        truncated = self.step_count >= self.max_steps

        return self._render(), reward, terminated, truncated, {}

    def _render(self):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        # Target
        tx, ty = (self.target_pos * (self.img_size - 1)).astype(int)
        img[max(0,ty-3):ty+3, max(0,tx-3):tx+3] = [0, 255, 0]
        # Block
        bx, by = (self.block_pos * (self.img_size - 1)).astype(int)
        img[max(0,by-4):by+4, max(0,bx-4):bx+4] = [255, 100, 0]
        # Agent
        ax, ay = (self.agent_pos * (self.img_size - 1)).astype(int)
        img[max(0,ay-2):ay+2, max(0,ax-2):ax+2] = [0, 100, 255]
        return img

    def render(self):
        return self._render()

    def close(self):
        pass