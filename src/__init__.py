"""LeWorldModel JAX - Clean, minimal implementation

Usage:
    from lewm import LeWorldModel, Trainer, CEMPlanner, Simple2DEnv
    from lewm import collect_trajectories, make_dataset, SIGReg

    # Create model
    model = LeWorldModel.create(embed_dim=192, action_dim=2)

    # Collect data
    env = Simple2DEnv()
    trajectories = collect_trajectories(env, num_episodes=100)
    train_data = make_dataset(trajectories, batch_size=128)

    # Train
    trainer = Trainer.create(model, train_data, max_steps=10000)
    trainer.train()

    # Plan
    planner = CEMPlanner.create(model, horizon=10)
    actions = planner.plan(init_obs, goal_obs)
"""

# Models
from .models import (
    LeWorldModel,
    WorldModelConfig,
    create_model,
    CNNEncoder,
    ActionEncoder,
    Projector,
    ARPredictor,
    AdaLNBlock,
    PredictorProjector,
    modulate,
)

# Loss
from .loss import SIGReg

# Training
from .train import Trainer, TrainConfig, create_trainer

# Environments
from .env import (
    EnvConfig,
    GymEnvWrapper,
    Simple2DEnv,
    collect_trajectories,
    make_dataset,
)

# Planning
from .planner import CEMConfig, CEMPlanner, create_planner

__all__ = [
    # Models
    "LeWorldModel",
    "WorldModelConfig",
    "create_model",
    "CNNEncoder",
    "ActionEncoder",
    "Projector",
    "ARPredictor",
    "AdaLNBlock",
    "PredictorProjector",
    "modulate",
    # Loss
    "SIGReg",
    # Training
    "Trainer",
    "TrainConfig",
    "create_trainer",
    # Environments
    "EnvConfig",
    "GymEnvWrapper",
    "Simple2DEnv",
    "collect_trajectories",
    "make_dataset",
    # Planning
    "CEMConfig",
    "CEMPlanner",
    "create_planner",
]

__version__ = "0.1.0"