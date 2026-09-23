import jax
import jax.numpy as jnp
import numpy as np

from jepa import JEPA


def encode_goal(model: JEPA, frame: jax.Array) -> jax.Array:
    """(H, W, C) frame -> (D,) goal embedding."""
    return model.encode(jnp.asarray(frame)[None])[0, -1]


def plan_cem(
    model: JEPA,
    history: jax.Array,
    goal_pixels=None,
    goal_emb=None,
    horizon: int = 8,
    action_dim: int = 2,
    num_samples: int = 128,
    num_elites: int = 10,
    num_iterations: int = 5,
    action_min: float = -1.0,
    action_max: float = 1.0,
    seed: int = 0,
):
    """Best (horizon, action_dim) action sequence to reach the goal."""
    goal_emb = goal_emb if goal_emb is not None else encode_goal(model, goal_pixels)
    ctx = jnp.asarray(history)[None]  # (1, H, H, W, C)

    lo, hi = jnp.asarray(action_min), jnp.asarray(action_max)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    # Initialise at the CENTRE of the action range, std covering half the
    # range: a zero-mean init clips every sample into one corner for
    # asymmetric spaces like PushT's [0, 512].
    mean = jnp.full((horizon, action_dim), center)
    std = jnp.full((horizon, action_dim), half)
    key = jax.random.key(seed)

    for _ in range(num_iterations):
        key, sub = jax.random.split(key)
        actions = jnp.clip(
            mean + std * jax.random.normal(sub, (num_samples, horizon, action_dim)),
            lo,
            hi,
        )
        # Distance of each rollout's final latent to the goal embedding.
        preds = model.rollout(
            jnp.repeat(ctx, num_samples, axis=0), actions
        )  # (N, T, D)
        costs = jnp.mean((preds[:, -1] - goal_emb) ** 2, axis=-1)  # (N,)
        elites = actions[jnp.argsort(costs)[:num_elites]]
        mean = jnp.mean(elites, axis=0)
        # Refit std to the elites, floored so the search doesn't collapse
        # onto a constant action before it converges.
        std = jnp.std(elites, axis=0) + 1e-3 * half
    return mean


def plan_mpc(
    model: JEPA,
    env,
    init_pixels,
    goal_pixels=None,
    goal_emb=None,
    n_steps: int = 100,
    **cem,
):
    """Receding-horizon loop: plan, execute ONE action, observe, replan."""
    goal_emb = goal_emb if goal_emb is not None else encode_goal(model, goal_pixels)
    h = model.history_size

    frames = np.asarray(init_pixels)
    if frames.ndim == 3:
        frames = frames[None]
    history = [jnp.asarray(f) for f in frames]
    while len(history) < h:
        history.append(history[-1])
    history = history[-h:]

    actions, obs, rewards = [], [], []
    for _ in range(n_steps):
        action = plan_cem(model, jnp.stack(history), goal_emb=goal_emb, **cem)[0]
        frame, reward, terminated, truncated, _ = env.step(np.asarray(action))
        actions.append(np.asarray(action))
        obs.append(frame)
        rewards.append(reward)
        history.append(jnp.asarray(frame))
        history = history[-h:]
        if terminated or truncated:
            break
    return np.array(actions), np.array(obs), np.array(rewards)


def _dot(pos: np.ndarray) -> np.ndarray:
    """Render a 64x64 float frame with a dot at ``pos``."""
    img = np.zeros((64, 64, 3), dtype=np.float32)
    y, x = (64 * np.asarray(pos)).astype(int)
    img[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = 0.5
    return img


class _DotChaser:
    """Synthetic env: actions nudge a dot; observations are 64x64 frames."""

    def __init__(self):
        self.pos = np.array([0.3, 0.3])

    def step(self, action):
        self.pos = np.clip(self.pos + 0.1 * np.asarray(action), 0.0, 1.0)
        done = bool(np.all(self.pos > 0.95))
        return _dot(self.pos), 0.0, done, False, {}


if __name__ == "__main__":
    from flax import nnx

    model = JEPA(nnx.Rngs(0), img_size=64, action_dim=2, history_size=3)
    cem = dict(
        horizon=3, action_dim=2, num_samples=16, num_elites=4, num_iterations=3, seed=7
    )

    k1, k2 = jax.random.split(jax.random.key(1))
    history = jax.random.uniform(k1, (3, 64, 64, 3))
    goal = jax.random.uniform(k2, (64, 64, 3))
    seq = plan_cem(model, history, goal, **cem)
    print(f"CEM plan:  {seq.shape}  (expect (3, 2))")

    actions, obs, rewards = plan_mpc(
        model, _DotChaser(), _dot([0.2, 0.2]), _dot([0.8, 0.8]), n_steps=5, **cem
    )
    print(f"MPC:       {actions.shape} {obs.shape} {rewards.shape}")

    assert seq.shape == (3, 2)
    assert actions.ndim == 2 and actions.shape[1] == 2
    assert len(obs) == len(actions) == len(rewards)
    print("\nsmoke test OK ✓")
