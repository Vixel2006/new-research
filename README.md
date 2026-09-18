# LeWorldModel JAX

Clean, minimal implementation of LeWorldModel (LeWM) in JAX/Flax NNX.

Based on: _LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels_ (arXiv:2603.19312v1)

## Features

- **Minimal encoder** - Simple CNN instead of ViT (~5x fewer params)
- **Clean API** - One-liner model creation, training, and planning
- **Gym integration** - Works with any Gymnasium environment
- **SIGReg** - Exact implementation of the paper's Gaussian regularizer

## Installation

```bash
cd lewm_jax
pip install -e .
```

Or with GPU support:

```bash
pip install -e . --extra-index-url https://pypi.nvidia.com
```

## Quick Start

```python
from lewm_jax import LeWorldModel, Trainer, CEMPlanner, Simple2DEnv, collect_trajectories, make_dataset

# 1. Create model (one line!)
model = LeWorldModel.create(embed_dim=192, action_dim=2)

# 2. Collect data from any Gym env
env = Simple2DEnv()  # or GymEnvWrapper("PushT-v1")
trajectories = collect_trajectories(env, num_episodes=100)
train_data = make_dataset(trajectories, batch_size=128)

# 3. Train
trainer = Trainer.create(model, train_data, max_steps=10000)
trainer.train()

# 4. Plan in latent space
planner = CEMPlanner.create(model, horizon=10)
actions = planner.plan(init_obs, goal_obs)
```

## Demo

```bash
# Quick demo (30 seconds)
python -m lewm_jax.demo --mode demo

# Train on real gym env
python -m lewm_jax.demo --mode train --env PushT-v1 --episodes 500 --steps 20000

# Train and test planning
python -m lewm_jax.demo --mode train_plan --env PushT-v1
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│  Pixels     │────▶│  CNN Encoder │────▶│  Projector (BN)  │──▶ z_t
│  (B,T,H,W,C)│     │  (4 layers)  │     │  192→2048→192    │
└─────────────┘     └─────────────┘     └──────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ SIGReg Loss │  (enforces N(0,I) on z)
                    └─────────────┘

┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│  Actions    │────▶│ Action Enc  │────▶│  AdaLN-zero      │
│  (B,T,A)    │     │  (Conv+MLP) │     │  Transformer     │──▶ z_{t+1}
└─────────────┘     └─────────────┘     │  (6 layers)      │
                                        └──────────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ Pred Proj   │
                                        │ (BN)        │
                                        └─────────────┘
```

## Configuration

```python
from lewm_jax import LeWMConfig, TrainConfig, CEMConfig

# Model
model = LeWorldModel(LeWMConfig(
    embed_dim=192,
    img_size=64,
    action_dim=2,
    history_size=3,
    pred_depth=6,
    pred_heads=16,
    sigreg_weight=0.1,
))

# Training
trainer = Trainer(model, TrainConfig(
    lr=5e-5,
    weight_decay=1e-3,
    max_steps=100000,
    batch_size=128,
))

# Planning
planner = CEMPlanner(model, CEMConfig(
    horizon=10,
    num_samples=300,
    num_iterations=10,
))
```

## Key Components

| Component    | File           | Description                                 |
| ------------ | -------------- | ------------------------------------------- |
| SIGReg       | `sigreg.py`    | Epps-Pulley statistic on random projections |
| CNN Encoder  | `encoder.py`   | 4-layer CNN + BatchNorm projector           |
| AR Predictor | `predictor.py` | 6-layer Transformer with AdaLN-zero         |
| LeWorldModel | `model.py`     | Full model with loss & rollout              |
| Trainer      | `train.py`     | Optax + Orbax training loop                 |
| CEM Planner  | `planner.py`   | Cross-Entropy Method in latent space        |
| Env Wrapper  | `env.py`       | Gym integration + data collection           |

## Requirements

- Python 3.10+
- JAX 0.4.25+
- Flax 0.8.0+ (NNX)
- Optax 0.2.0+
- Orbax-checkpoint 0.5.0+
- Gymnasium 0.29.0+

## License

MIT
