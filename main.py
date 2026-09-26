from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from flax import nnx

from src.jepa import JEPA, ModelConfig
from src.planner import CEMPlanner
from src.pusht import PushT
from src.train import TrainConfig, Trainer
from src.utils import checkpoint, vis

POLICIES = ("random", "heuristic", "mixed")
MODE_ALIASES = {"eval": "solve", "quick": "train_solve"}

DEFAULT_MODEL = ModelConfig(
    embed_dim=128,
    enc_layers=2,
    enc_heads=8,
    pred_depth=3,
    pred_heads=8,
    pred_mlp_dim=512,
    sigreg_weight=0.01,
)

QUICK_MODEL = {
    "embed_dim": 64,
    "enc_layers": 1,
    "enc_heads": 4,
    "pred_depth": 1,
    "pred_heads": 4,
    "pred_mlp_dim": 128,
}
QUICK = {
    "batch_size": 4,
    "num_iters": 6,
    "warmup_steps": 2,
    "decay_steps": 6,
    "log_interval": 1,
    "eval_interval": 1,
    "num_val_batches": 1,
    "mpc_horizon": 2,
    "mpc_num_samples": 8,
    "mpc_num_elites": 2,
    "mpc_num_iterations": 2,
    "mpc_max_steps": 2,
    "num_eval_episodes": 1,
    "video": False,
}


@dataclass
class ExperimentConfig:
    """Everything that defines a training / solve run."""

    model: ModelConfig = field(default_factory=lambda: replace(DEFAULT_MODEL))

    batch_size: int = 16
    num_iters: int = 2_000
    peak_lr: float = 3e-4
    warmup_steps: int = 100
    decay_steps: int = 2_000
    log_interval: int = 100
    eval_interval: int = 100  # validate + best-checkpoint every N steps
    num_val_batches: int = 16  # fixed validation pool snapshotted pre-training

    collect_policy: str = "mixed"  # random | heuristic | mixed
    max_episode_steps: int = 150

    mpc_horizon: int = 4
    mpc_num_samples: int = 96
    mpc_num_elites: int = 12
    mpc_num_iterations: int = 4
    mpc_max_steps: int = 200

    seed: int = 0
    run_dir: str = "runs/run0"
    num_eval_episodes: int = 5
    ckpt_tag: str = "latest"  # latest | best | <step>  (what `solve` loads)
    video: bool = True

    def __post_init__(self) -> None:
        # run_config.json round-trips through JSON, so the nested model config
        # comes back as a plain dict.
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)


def _mean(xs) -> float:
    return sum(xs) / len(xs) if len(xs) else 0.0


def _peak(xs) -> float:
    return float(max(xs)) if len(xs) else 0.0


def _log_metrics(run_dir: Path, step: int, metrics: dict) -> None:
    """Append one JSON line per logged step to ``<run_dir>/metrics.jsonl``."""
    row = {k: float(v) for k, v in metrics.items() if k != "step"} | {"step": int(step)}
    (run_dir / "metrics.jsonl").open("a").write(json.dumps(row) + "\n")


def run_train(cfg: ExperimentConfig) -> None:
    """Fit the world model to PushT data collected online."""
    run_dir, ckpt_dir = Path(cfg.run_dir), Path(cfg.run_dir) / "checkpoints"
    print("building model + collecting PushT observations (online) ...")
    model = JEPA(cfg.model, nnx.Rngs(cfg.seed))
    trainer = Trainer(
        model=model,
        config=TrainConfig(
            peak_lr=cfg.peak_lr,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.decay_steps,
        ),
    )

    pusht = PushT(
        seed=cfg.seed,
        history_size=cfg.model.history_size,
        batch_size=cfg.batch_size,
        policy=cfg.collect_policy,
        max_episode_steps=cfg.max_episode_steps,
    )

    # Warm the replay buffer so the snapshotted validation pool is diverse.
    # Validating on a buffer still near-empty (one early episode of near-static
    # frames) made val loss look ~16x better than reality and hid the fact
    # that the model was just predicting "nothing changes".
    pusht.warmup(episodes=12)

    # Fixed validation pool: batches snapshotted once, before training starts.
    pool = (
        [pusht.next_batch() for _ in range(cfg.num_val_batches)]
        if cfg.eval_interval > 0 and cfg.num_val_batches > 0
        else []
    )

    trainer.train(
        pusht.batches(),
        num_iters=cfg.num_iters,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_batches=pool,
        ckpt_dir=ckpt_dir,
        log_fn=lambda step, m: _log_metrics(run_dir, step, m),
    )
    checkpoint.save(model, ckpt_dir, tag="latest")
    pusht.close()

    last = trainer.history[-1] if trainer.history else {}
    print(
        "\ntraining done: final "
        + "  ".join(
            f"{k} {last.get(k, float('nan')):.4f}" for k in ("loss", "mse", "sigreg")
        )
    )
    if trainer.best_tag is not None:
        print(
            f"best checkpoint: {trainer.best_tag}  "
            f"val_loss {trainer.best_val:.4f} @ step {trainer.best_step}"
        )
    vis.plot_training_curves(trainer.history, run_dir / "training_curves.png")
    if pool:
        vis.plot_predictions(model, pool, run_dir / "predictions.png")


def load_model(cfg: ExperimentConfig) -> JEPA:
    """Rebuild the architecture from the snapshotted config, then load weights.

    The sizes come from ``cfg.model`` (the JSON snapshot), so this reproduces the
    exact architecture the checkpoint was trained with and ``checkpoint.load``
    can verify the shapes line up.
    """
    ckpt_dir = Path(cfg.run_dir) / "checkpoints"
    tag = cfg.ckpt_tag
    if tag == "best":
        tag = checkpoint.resolve_best_tag(ckpt_dir) or "latest"
    print(f"loading model from {ckpt_dir / tag} ...")
    model = JEPA(cfg.model, nnx.Rngs(cfg.seed))
    checkpoint.load(model, ckpt_dir=ckpt_dir, tag=tag)
    return model


def run_solve(cfg: ExperimentConfig) -> None:
    """Plan to the fixed PushT goal with CEM and report coverage + success."""
    run_dir = Path(cfg.run_dir)
    model = load_model(cfg)
    pusht = PushT(seed=cfg.seed, history_size=cfg.model.history_size)

    # Fixed (deterministic) goal: the T-block sitting in the green zone. The
    # goal enters planning as PIXELS so it is encoded in the same proj-BN
    # batch as the rollout context (LeJEPA-style BN, no running average).
    goal = pusht.goal_frame()
    print(f"goal frame: {goal.shape} {goal.dtype}")

    planner = CEMPlanner(
        model=model,
        horizon=cfg.mpc_horizon,
        num_samples=cfg.mpc_num_samples,
        topk=cfg.mpc_num_elites,
        num_iterations=cfg.mpc_num_iterations,
        action_min=0.0,
        action_max=512.0,
    )
    print(
        f"planning {cfg.num_eval_episodes} episodes "
        f"(MPC, horizon={planner.horizon}) ..."
    )

    episodes, best = [], None  # best = (max_coverage, frames, coverage)
    for i in range(cfg.num_eval_episodes):
        seed = cfg.seed + 100 + i
        res = pusht.run_mpc(planner, n_steps=cfg.mpc_max_steps, seed=seed)
        cov = _peak(res["coverage"])
        episodes.append(
            {
                "episode": i,
                "seed": seed,
                "steps": int(res["steps_taken"]),
                "max_coverage": cov,
                "success": bool(res["success"]),
            }
        )
        print(
            f"  episode {i}: steps {res['steps_taken']:3d}  "
            f"max coverage {cov * 100:5.1f}%  success {res['success']}"
        )
        if best is None or cov > best[0]:
            best = (cov, res["frames"], res["coverage"])

    # Random baseline for context (same seeds, no model).
    base = [
        _peak(pusht.random_baseline(n_steps=cfg.mpc_max_steps, seed=cfg.seed + 100 + i))
        for i in range(min(cfg.num_eval_episodes, 3))
    ]
    pusht.close()

    n_success = sum(int(e["success"]) for e in episodes)
    summary = {
        "num_episodes": len(episodes),
        "success": n_success,
        "success_rate": n_success / max(len(episodes), 1),
        "mean_max_coverage": _mean([e["max_coverage"] for e in episodes]),
        "mean_steps": _mean([e["steps"] for e in episodes]),
        "baseline_mean_max_coverage": _mean(base),
    }
    print(
        f"\nMPC solve: success {n_success}/{len(episodes)}  "
        f"mean max-coverage {summary['mean_max_coverage'] * 100:.1f}%"
    )
    if base:
        print(
            "random baseline: mean max-coverage "
            f"{summary['baseline_mean_max_coverage'] * 100:.1f}%"
        )

    # Best-episode artifacts + machine-readable summary.
    artifacts = []
    if best is not None and cfg.video:
        _, frames, coverage = best
        artifacts = [
            str(vis.frames_to_gif(frames, run_dir / "solve_best.gif")),
            str(
                vis.plot_coverage(
                    coverage,
                    run_dir / "solve_coverage.png",
                    title=f"best episode: max coverage {best[0] * 100:.1f}%",
                )
            ),
        ]

    (run_dir / "solve_results.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "summary": summary,
                "episodes": episodes,
                "artifacts": artifacts,
            },
            indent=2,
            default=str,
        )
    )
    print(f"results: {run_dir / 'solve_results.json'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "mode",
        choices=list(MODE_ALIASES) + ["train", "solve", "train_solve"],
        help=(
            "train: learn the world model. solve/eval: plan to the goal and "
            "report coverage + success. train_solve: both. quick: tiny smoke test."
        ),
    )
    ap.add_argument("--run-dir", default="runs/run0", help="where the run lives")
    ap.add_argument("--no-video", action="store_true", help="skip the solve GIF/plot")
    # Every other flag writes straight onto an ExperimentConfig field.
    for flag, dest, typ, help_ in (
        ("--iters", "num_iters", int, "train steps (default 2000)"),
        ("--batch-size", "batch_size", int, "training batch size"),
        ("--policy", "collect_policy", str, "data collection: random|heuristic|mixed"),
        ("--episodes", "num_eval_episodes", int, "eval episodes for solve"),
        ("--horizon", "mpc_horizon", int, "CEM horizon"),
        ("--samples", "mpc_num_samples", int, "CEM samples per iteration"),
        ("--eval-interval", "eval_interval", int, "validate + save best every N steps"),
        ("--val-batches", "num_val_batches", int, "size of the fixed validation pool"),
        ("--ckpt", "ckpt_tag", str, "which checkpoint `solve` loads"),
        ("--seed", "seed", int, "random seed"),
    ):
        ap.add_argument(flag, dest=dest, type=typ, default=None, help=help_)
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Config for this run: the snapshot in ``--run-dir``, then flags on top.

    Starting from the snapshot is what lets a bare ``solve`` rebuild the exact
    architecture a checkpoint was trained with -- the sizes are not in the
    weights themselves.
    """
    path = Path(args.run_dir) / "run_config.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved:
        print(f"restored config from {path}")

    cfg = ExperimentConfig(**{**saved, "run_dir": args.run_dir})
    if args.mode == "quick":
        for name, value in QUICK.items():
            setattr(cfg, name, value)
        cfg.model = replace(cfg.model, **QUICK_MODEL)

    for name, value in vars(args).items():  # overlay the non-None flags
        if value is not None and hasattr(cfg, name):
            setattr(cfg, name, value)
    if args.no_video:
        cfg.video = False
    if cfg.collect_policy not in POLICIES:
        raise SystemExit(
            f"collect_policy must be one of {POLICIES}, got {cfg.collect_policy!r}"
        )
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    mode = MODE_ALIASES.get(args.mode, args.mode)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if mode in ("train", "train_solve"):
        # Snapshot the config used for training so a later `solve` can rebuild
        # the exact same architecture (sizes are not stored in the weights
        # themselves). Only training writes it, otherwise a `solve --ckpt best`
        # would persist its own flags into the snapshot and quietly change what
        # the *next* bare `solve` loads.
        (run_dir / "run_config.json").write_text(
            json.dumps(asdict(cfg), indent=2, default=str)
        )

    if mode in ("train", "train_solve"):
        run_train(cfg)
    if mode in ("solve", "train_solve"):
        run_solve(cfg)


if __name__ == "__main__":
    main()
