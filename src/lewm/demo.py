"""LeWorldModel JAX - Demo & Training Script

Usage:
    # Quick demo with simple 2D environment
    python -m lewm.demo --mode demo

    # Train on gym environment
    python -m lewm.demo --mode train --env gym_pusht/PushT-v0 --episodes 500

    # Train and plan
    python -m lewm.demo --mode train_plan --env gym_pusht/PushT-v0
"""

import argparse
import os

import jax
import numpy as np
import wandb
from flax import nnx

from lewm import (
    CEMPlanner,
    EnvConfig,
    GymEnvWrapper,
    LeWorldModel,
    SIGReg,
    Simple2DEnv,
    Trainer,
    collect_trajectories,
    make_dataset,
    create_planner,
)


def demo_simple():
    print("=" * 60)
    print("LeWorldModel JAX - Simple 2D Demo")
    print("=" * 60)

    # Create model
    print("\n1. Creating model...")
    model = LeWorldModel.create(
        embed_dim=128,  # Smaller for demo
        img_size=64,
        action_dim=2,
        history_size=3,
        seed=42,
    )
    print(f"   Model params: {sum(p.size for p in jax.tree.leaves(jax.tree.map(lambda x: x, model))) if hasattr(model, 'params') else 'N/A'}")

    # Create environment
    print("\n2. Creating environment...")
    env = Simple2DEnv(img_size=64)

    # Collect random trajectories
    print("\n3. Collecting random trajectories...")
    trajectories = collect_trajectories(env, num_episodes=50, max_steps=50)
    print(f"   Collected {trajectories['pixels'].shape[0]} episodes")
    print(f"   Pixels shape: {trajectories['pixels'].shape}")
    print(f"   Actions shape: {trajectories['actions'].shape}")

    # Create dataset
    print("\n4. Creating dataset...")
    train_iter = make_dataset(
        trajectories,
        history_size=3,
        num_preds=1,
        batch_size=32,
        shuffle=True,
    )

    # Quick training
    print("\n5. Training (100 steps)...")
    sigreg_fn = SIGReg(
        knots=model.config.sigreg_knots,
        num_proj=model.config.sigreg_num_proj,
        embed_dim=model.config.embed_dim,
    )
    trainer = Trainer.create(
        model,
        train_iter,
        max_steps=100,
        warmup_steps=10,
        lr=1e-4,
        batch_size=32,
        log_every=20,
        sigreg_weight=model.config.sigreg_weight,
    )
    trainer.train(num_steps=100, sigreg_fn=sigreg_fn)

    # Test planning
    print("\n6. Testing planning...")
    planner = create_planner(model, horizon=5, num_samples=50, num_iterations=5)

    # Get initial and goal observations
    init_obs, _ = env.reset(seed=123)
    goal_obs, _ = env.reset(seed=456)

    # Plan
    actions = planner.plan(init_obs[None], goal_obs[None])
    print(f"   Planned actions shape: {actions.shape}")
    print(f"   Actions: {actions}")

    # Execute in env
    print("\n7. Executing plan...")
    env.reset(seed=123)
    for i, action in enumerate(actions):
        obs, reward, term, trunc, _ = env.step(np.array(action))
        print(f"   Step {i}: action={action}, reward={reward:.3f}")
        if term or trunc:
            break

    print("\n✓ Demo complete!")
    env.close()


def train_on_gym(env_id: str, episodes: int = 500, steps: int = 20000):
    print(f"\nTraining on {env_id}...")

    # Setup
    env_config = EnvConfig(env_id=env_id, img_size=64, frameskip=5)
    env = GymEnvWrapper(env_config)

    # Collect expert-ish trajectories (random for now)
    print("Collecting trajectories...")
    trajectories = collect_trajectories(env, num_episodes=episodes, max_steps=100)
    print(f"Collected: {trajectories['pixels'].shape}")

    # Create dataset
    train_iter = make_dataset(
        trajectories,
        history_size=3,
        num_preds=1,
        batch_size=128,
        shuffle=True,
    )

    # Create model
    action_dim = env.action_space.shape[0]
    model = LeWorldModel.create(
        embed_dim=192,
        img_size=64,
        action_dim=action_dim,
        history_size=3,
        seed=42,
    )
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Model params: {n_params:,}")

    # Trainer
    sigreg_fn = SIGReg(
        knots=model.config.sigreg_knots,
        num_proj=model.config.sigreg_num_proj,
        embed_dim=model.config.embed_dim,
    )
    trainer = Trainer.create(
        model,
        train_iter,
        max_steps=steps,
        lr=5e-5,
        batch_size=128,
        log_every=100,
        eval_every=1000,
        save_every=5000,
        checkpoint_dir=f"./checkpoints/{env_id.replace('/', '__')}",
        sigreg_weight=model.config.sigreg_weight,
    )

    # Initialize wandb (fall back to offline, and never crash if unavailable)
    wandb_run = None
    try:
        if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
            os.environ["WANDB_MODE"] = "offline"
        wandb_run = wandb.init(
            project="lewm-jax",
            name=f"{env_id}-train",
            config={
                "env": env_id,
                "episodes": episodes,
                "steps": steps,
                "embed_dim": 192,
            },
        )
    except Exception as e:
        print(f"  (wandb init failed: {type(e).__name__}: {e}; continuing without logging)")

    # Train
    try:
        trainer.train(sigreg_fn=sigreg_fn)
    finally:
        if wandb_run is not None:
            wandb.finish()
        env.close()
    return model


def train_and_plan(env_id: str, episodes: int = 200, steps: int = 10000):
    model = train_on_gym(env_id, episodes=episodes, steps=steps)

    # Test planning
    env_config = EnvConfig(env_id=env_id, img_size=64)
    env = GymEnvWrapper(env_config)

    # Use the environment's actual action space for CEM sampling
    act_dim = int(env.action_space.shape[0])
    act_low = float(env.action_space.low.min())
    act_high = float(env.action_space.high.max())
    planner = create_planner(
        model,
        horizon=10,
        num_samples=300,
        num_iterations=10,
        action_dim=act_dim,
        action_min=act_low,
        action_max=act_high,
    )

    # Build a short frame history so the model has context to plan from
    hist_len = model.config.history_size
    env.reset(seed=1)
    goal_obs, _ = env.reset(seed=2)

    init_obs, _ = env.reset(seed=1)
    history = [init_obs]
    for _ in range(hist_len - 1):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        history.append(obs)
        if term or trunc:
            break
    init_hist = np.stack(history)  # (T, H, W, C) - history frames

    print("\nPlanning to goal...")
    actions = planner.plan(init_hist, goal_obs)

    print("Executing plan...")
    env.reset(seed=1)
    for _ in range(hist_len - 1):
        env.step(env.action_space.sample())
    for i, action in enumerate(actions):
        obs, reward, term, trunc, _ = env.step(np.array(action))
        print(f"  Step {i}: reward={reward:.3f}")
        if term or trunc:
            print(f"  {'Success!' if term else 'Truncated'}")
            break

    env.close()


def main():
    parser = argparse.ArgumentParser(description="LeWorldModel JAX Demo")
    parser.add_argument(
        "--mode",
        choices=["demo", "train", "train_plan"],
        default="demo",
        help="Run mode",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="gym_pusht/PushT-v0",
        help="Gym environment ID (e.g. gym_pusht/PushT-v0)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="Training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    jax.config.update("jax_enable_x64", False)

    if args.mode == "demo":
        demo_simple()
    elif args.mode == "train":
        train_on_gym(args.env, args.episodes, args.steps)
    elif args.mode == "train_plan":
        train_and_plan(args.env, args.episodes, args.steps)


if __name__ == "__main__":
    main()