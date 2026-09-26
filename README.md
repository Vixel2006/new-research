# LeWorldModel

Clean, minimal implementation of LeWorldModel (LeWM) in JAX/Flax NNX.

Based on: _LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels_ (arXiv:2603.19312v1)

## Layout

| Path | What it is |
| --- | --- |
| `src/jepa.py` | `ModelConfig` (every model size in one dataclass) + `JEPA` — ViT encoder + action embedder + AR predictor (`encode`, `loss`, `rollout`) |
| `src/modules/` | `ViT`, `Embedder`, `ARPredictor`, `SIGReg` |
| `src/train.py` | `Trainer` — jitted AdamW step, validation + best-checkpoint tracking |
| `src/planner.py` | `CEMPlanner` — cross-entropy optimization over `JEPA.rollout` |
| `src/pusht.py` | `PushT` — the environment: model frames, replay-buffer batches, `run_mpc` / `random_baseline` |
| `src/utils/` | Shared by training and solving: `checkpoint.py` (orbax save/load, `best.json`) and `vis.py` (GIFs, training curves, predictor debug plots, coverage plots) |
| `main.py` | CLI (argparse): train the world model on PushT, then solve it by planning |

## The PushT gym experiment

Goal-conditioned LeWM on `gym_pusht/PushT-v0` (pymunk): the agent must push the
T-shaped block into the fixed green goal zone.

* **Observations** — 64×64×3 uint8 RGB frames (matching the ViT's `img_size`);
  normalized to float32 `[0, 1]` for the model.
* **Actions** — continuous 2-D target of the agent puck, `[0, 512]` (screen coords).
* **Goal** — the fixed goal pose `(256, 256, π/4)` rendered to pixels. Each
  `JEPA.rollout` encodes the goal together with its context frames in one
  BatchNorm batch, then planning pulls the final predicted latent toward the
  resulting goal embedding.
* **Training** — `Trainer` consumes `{pixels: (B, H+1, 64, 64, 3), actions:
  (B, H+1, 2)}` batches streamed online by `PushT.batches()` from env rollouts
  (random / heuristic-push / mixed policies).
* **Solving** — receding-horizon planning with `CEMPlanner` (the model plus the CEM
  search settings in one object; the whole CEM loop is jitted) on the latent
  model's predictions; `PushT.run_mpc` executes the first action, observes,
  replans.

```python
planner = CEMPlanner(
    model,
    horizon=4,
    num_samples=96,
    num_elites=12,
    num_iterations=4,
    action_min=0.0,
    action_max=512.0,
)
action_sequence = planner.plan(
    history, goal_frame, seed=0, history_actions=executed_actions
)
```

## Usage

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .           # deps are declared in pyproject.toml
```

### Typical workflow

```bash
# 0) Sanity check that everything works (tiny model, a few steps)
python main.py quick

# 1) Train a world model on PushT observations + actions.
#    Validates every 100 steps and checkpoints the best model automatically.
python main.py train --run-dir runs/run0

# 2) Solve it: load the model, plan (MPC/CEM) to the goal, report coverage.
#    `eval` is an alias for `solve`.
python main.py solve --run-dir runs/run0 --episodes 5

# 3) Solve from the *best* checkpoint (lowest validation loss), not the final one
python main.py solve --run-dir runs/run0 --ckpt best --episodes 5

# 4) Train and solve in one go (writes runs/run0/solve_best.gif)
python main.py train_solve --run-dir runs/run0

# Full flag list
python main.py --help
```

`solve` knows how to rebuild the exact model architecture used for training: the
full config is snapshotted to `runs/<name>/run_config.json` during training and
restored automatically on load, so a separate `solve` process just needs the
same `--run-dir`.

### Flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--run-dir` | `runs/run0` | output dir: config, checkpoints, curves, GIF, results |
| `--iters` | `2000` | training steps |
| `--batch-size` | `16` | batch size (batches stream from online env rollouts) |
| `--policy` | `mixed` | data-collection policy: `random` \| `heuristic` \| `mixed` |
| `--eval-interval` | `100` | validate + save best checkpoint every N steps |
| `--val-batches` | `8` | size of the fixed validation pool (snapshotted pre-training) |
| `--horizon` | `4` | CEM planning horizon (longer = further-sighted) |
| `--samples` | `96` | CEM samples per iteration (more = better, slower) |
| `--episodes` | `5` | number of solve episodes to plan |
| `--ckpt` | `latest` | which checkpoint `solve` loads: `latest` \| `best` \| `<step>` |
| `--seed` | `0` | random seed (data collection, training, solve all derive from it) |
| `--no-video` | off | skip GIF + coverage-PNG artifacts |

### Example logging/debug runs

```bash
# Quick 100-step model to validate the pipeline end-to-end:
python main.py train --run-dir runs/debug --iters 100 --eval-interval 20

# Longer solve with a deeper CEM search:
python main.py solve --run-dir runs/run0 --horizon 8 --samples 256 --episodes 10
```

### Model size

Sizes are **not** CLI flags — they are fields of `ModelConfig` in `src/jepa.py`
(`embed_dim`, `enc_layers`, `pred_depth`, `pred_mlp_dim`, `sigreg_*`, ...), which
is the single source of truth for the architecture: `JEPA(cfg.model, rngs)`
builds it, and `nnx` keeps the dataclass out of the weights, so it is snapshotted
as JSON in `run_config.json` and a later `solve` rebuilds the exact same shapes.

`main.py` picks a starting point from it: `DEFAULT_MODEL` (the shipped run) and
`QUICK_MODEL` (the `quick` smoke test). Copy either into a named preset, or
`dataclasses.replace(cfg.model, embed_dim=256)`, to scale up. JAX on this box
runs on CPU: the shipped defaults train in a few minutes; scale up `embed_dim` /
`pred_depth` / `num_iters` for real results (ideally on a GPU).

## Checkpoints & results

Every run dir (`runs/<name>/`) accumulates:

| Artifact | What it is |
| --- | --- |
| `run_config.json` | the full `ExperimentConfig` (including the nested `ModelConfig`) used for training — `solve` rebuilds the exact architecture from it |
| `checkpoints/latest` | final-trained model |
| `checkpoints/best_N` + `best.json` | the lowest-validation-loss model (updated whenever it improves) |
| `training_curves.png` | loss / MSE / SIGReg / LR curves with `val_loss` overlaid |
| `predictions.png` | predictor debug: predicted-vs-target latent scatter, latent distribution, per-sample MSE / cosine error |
| `solve_best.gif` | frames of the highest-coverage MPC episode |
| `solve_coverage.png` | per-step coverage of that episode |
| `solve_results.json` | per-episode stats + summary (success rate, mean coverage, random baseline) |
| `metrics.jsonl` | every logged step appended as a JSON line |

Note: JAX on this box runs on CPU. See [Model size](#model-size) for how to
configure bigger models before scaling up training (ideally on a GPU).