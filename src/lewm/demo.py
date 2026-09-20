"""LeWorldModel JAX - Demo & Training Script

Usage:
    # Quick demo with simple 2D environment
    python -m lewm.demo --mode demo

    # Train on gym environment
    python -m lewm.demo --mode train --env gym_pusht/PushT-v0 --episodes 500

    # Train and plan (one shot)
    python -m lewm.demo --mode train_plan --env gym_pusht/PushT-v0

    # Watch the trained model act in the gym env (loads checkpoint,
    # re-plans with CEM every few steps, saves an overlaid GIF):
    python -m lewm.demo --mode run --env gym_pusht/PushT-v0 --gif runs/pusht.gif
"""

import argparse
import os
from collections import deque
from pathlib import Path

import jax
import numpy as np
import wandb
from flax import nnx
from PIL import Image, ImageDraw

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


def _to_screen(pts, scale=680.0 / 512.0):
    """Map world coords (0..512, no y-flip) to the 680x680 render."""
    return [(float(x) * scale, float(y) * scale) for x, y in pts]


def _goal_polygon(raw_env, scale=680.0 / 512.0):
    """Goal-zone polygon (tee shape at goal pose) in screen coords."""
    goal_body = raw_env.get_goal_pose_body(raw_env.goal_pose)
    pts = []
    for shape in raw_env.block.shapes:
        for v in shape.get_vertices():
            w = goal_body.local_to_world(v)
            pts.append((float(w.x) * scale, float(w.y) * scale))
    return pts


def _draw_plan_frame(
    frame,
    goal_poly,
    plan_actions,
    executed_actions,
    goal_inset=None,
    step=0,
    reward=0.0,
    coverage=0.0,
    success=False,
):
    """Overlay the current CEM plan and executed path onto a rendered frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Goal zone (dashed green outline)
    if goal_poly is not None and len(goal_poly) > 2:
        for i in range(len(goal_poly)):
            ax, ay = goal_poly[i]
            bx, by = goal_poly[(i + 1) % len(goal_poly)]
            steps = max(2, int(np.hypot(bx - ax, by - ay) / 12))
            for j in range(0, steps, 2):
                t0, t1 = j / steps, min(1.0, (j + 1) / steps)
                draw.line(
                    [ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1],
                    fill=(0, 180, 0),
                    width=2,
                )

    # Planned path (orange) - where the CEM plan wants the pusher to go
    if plan_actions is not None and len(plan_actions):
        pts = _to_screen(plan_actions)
        if len(pts) > 1:
            draw.line(pts, fill=(255, 140, 0), width=3, joint="curve")
        for i, p in enumerate(pts):
            r = 5 if i == 0 else 3
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 180, 40))

    # Executed path (thin blue) - where the pusher has actually been commanded
    if executed_actions is not None and len(executed_actions) > 1:
        pts = _to_screen(executed_actions)
        draw.line(pts, fill=(40, 90, 220), width=2)
    if executed_actions is not None and len(executed_actions) > 0:
        px, py = _to_screen([executed_actions[-1]])[0]
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(40, 90, 220), width=2)

    # Text
    status = "SUCCESS" if success else "running"
    txt = f"step {step}  reward {reward:.3f}  coverage {coverage:.2f}  [{status}]"
    draw.text((8, 8), txt, fill=(20, 20, 20))

    # Goal inset (what the model is aiming at)
    if goal_inset is not None:
        g = Image.fromarray(goal_inset).resize((136, 136))
        img.paste(g, (680 - 136 - 8, 8))
        draw.rectangle([680 - 136 - 8, 8, 680 - 8, 8 + 136], outline=(255, 255, 255), width=2)

    return img


def run_inference(
    env_id: str = "gym_pusht/PushT-v0",
    checkpoint_dir: str | None = None,
    ckpt_step: int | None = None,
    horizon: int = 10,
    num_samples: int = 200,
    num_iterations: int = 5,
    replan_every: int = 5,
    max_steps: int = 60,
    gif_path: str = "runs/pusht_rollout.gif",
    seed: int = 1,
    goal_seed: int = 2,
    overlay: bool = True,
):
    """Load a trained model and watch it act in the gym env.

    The model has no policy - it acts by Model-Predictive Control with
    CEM in latent space: every `replan_every` steps we encode the current
    frame history, simulate candidate action sequences through the learned
    world model, and execute the first action of the best sequence.
    The whole episode is recorded to an overlaid animated GIF.
    """
    if checkpoint_dir is None:
        checkpoint_dir = f"./checkpoints/{env_id.replace('/', '__')}"

    print("=" * 60)
    print(f"Running trained model in {env_id}")
    print("=" * 60)

    # Environment (frameskip must match training so the model perceives
    # the same observation cadence it was trained on)
    env_config = EnvConfig(env_id=env_id, img_size=64, frameskip=5)
    env = GymEnvWrapper(env_config)

    # Model with the same config used by train_on_gym
    action_dim = int(env.action_space.shape[0])
    model = LeWorldModel.create(
        embed_dim=192,
        img_size=64,
        action_dim=action_dim,
        history_size=3,
        seed=42,
    )

    # Recreate a trainer pointing at the checkpoint dir and load weights
    trainer = Trainer.create(
        model,
        iter(()),
        max_steps=1,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.load_checkpoint(ckpt_step)
    if trainer.step == 0:
        raise RuntimeError(
            f"No checkpoint found in {checkpoint_dir}. Train first with:\n"
            f"  python -m lewm.demo --mode train --env {env_id}"
        )

    # Planner over the env's real action space
    planner = create_planner(
        model,
        horizon=horizon,
        num_samples=num_samples,
        num_iterations=num_iterations,
        action_dim=action_dim,
        action_min=float(env.action_space.low.min()),
        action_max=float(env.action_space.high.max()),
    )

    hist_len = model.config.history_size

    # Goal frame: place the block on the goal zone so the target state is
    # unambiguous. The model's view (64x64) is used for planning; a 680x680
    # thumbnail is shown in the GIF corner as "what the model is aiming at".
    raw = env.env
    goal_state = np.array([256.0, 256.0, 256.0, 256.0, np.pi / 4])
    try:
        obs, _ = raw.reset(options={"reset_to_state": goal_state})
    except Exception:
        obs, _ = raw.reset(seed=goal_seed)  # env without reset_to_state
    goal_obs = env._process_obs(obs)
    goal_render = np.asarray(raw.render())

    # Warm-up: build the initial frame history deterministically (these exact
    # random actions are replayed below so the run starts identically).
    hist_actions = []
    env.reset(seed=seed)
    for _ in range(hist_len - 1):
        act = env.action_space.sample()
        hist_actions.append(act)
        env.step(act)

    # Start the run from the same seeded state and replay the warm-up actions.
    env.reset(seed=seed)
    obs, _ = env.reset(seed=seed)
    history = deque([obs], maxlen=hist_len)
    for act in hist_actions:
        obs, _, _, _, _ = env.step(act)
        history.append(obs)

    plan_actions = None
    executed = []
    frames = []
    total_reward = 0.0
    success = False
    covered = 0.0

    print(f"\nRunning {max_steps} steps, re-planning every {replan_every}...")
    for t in range(max_steps):
        if t % replan_every == 0:
            hist_stack = np.stack(list(history))  # (H, 64, 64, 3)
            plan_actions = np.asarray(planner.plan(hist_stack, goal_obs))
            print(f"  [t={t}] planned {len(plan_actions)} actions (iter {t // replan_every + 1})")

        act = plan_actions[min(t % replan_every, len(plan_actions) - 1)]
        obs, reward, term, trunc, info = env.step(np.asarray(act, dtype=np.float32))
        history.append(obs)
        executed.append(np.asarray(act, dtype=np.float32))
        total_reward += reward
        snap = np.asarray(raw.render())
        frames.append(
            {
                "frame": snap,
                "plan": plan_actions,
                "executed": np.stack(executed) if executed else None,
                "step": t,
                "reward": reward,
                "coverage": float(info.get("coverage", 0.0)),
                "success": bool(info.get("is_success", term)),
            }
        )
        covered = float(info.get("coverage", covered))
        if term or trunc:
            success = bool(info.get("is_success", term))
            print(f"  Terminated at t={t}: {'SUCCESS' if success else 'truncated'}")
            break

    env.close()

    print(f"\nTotal reward: {total_reward:.3f}  final coverage: {covered:.3f}  success: {success}")

    # Build the GIF
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    goal_poly = _goal_polygon(raw.unwrapped) if overlay else None
    pil_frames = []
    for rec in frames:
        if overlay:
            pil = _draw_plan_frame(
                rec["frame"],
                goal_poly,
                rec["plan"] if (rec["plan"] is not None) else None,
                rec["executed"],
                goal_inset=goal_render,
                step=rec["step"],
                reward=rec["reward"],
                coverage=rec["coverage"],
                success=rec["success"],
            )
        else:
            pil = Image.fromarray(rec["frame"])
        pil_frames.append(pil)

    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=250,
        loop=0,
    )
    print(f"Saved rollout GIF: {gif_path}")
    return gif_path


def main():
    parser = argparse.ArgumentParser(description="LeWorldModel JAX Demo")
    parser.add_argument(
        "--mode",
        choices=["demo", "train", "train_plan", "run"],
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

    # run mode options
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: latest)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="CEM planning horizon (run mode)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="CEM candidate sequences per iteration (run mode)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="CEM refinement iterations (run mode)",
    )
    parser.add_argument(
        "--replan-every",
        type=int,
        default=5,
        help="Re-plan every N steps (run mode)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=60,
        help="Max steps of the rollout (run mode)",
    )
    parser.add_argument(
        "--gif",
        type=str,
        default="runs/pusht_rollout.gif",
        help="Output path for the rollout GIF (run mode)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Record the plain env frames without plan overlay (run mode)",
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
    elif args.mode == "run":
        run_inference(
            env_id=args.env,
            ckpt_step=args.checkpoint_step,
            horizon=args.horizon,
            num_samples=args.num_samples,
            num_iterations=args.num_iterations,
            replan_every=args.replan_every,
            max_steps=args.rollout_steps,
            gif_path=args.gif,
            overlay=not args.no_overlay,
        )


if __name__ == "__main__":
    main()