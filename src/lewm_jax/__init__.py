"""LeWorldModel JAX - Clean, minimal implementation

Usage:
    from lewm_jax import LeWorldModel, Trainer, CEMPlanner, Simple2DEnv
    from lewm_jax import collect_trajectories, make_dataset

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

from .model import LeWorldModel, LeWMConfig, create_model
from .train import Trainer, TrainConfig, create_trainer
from .planner import CEMPlanner, CEMConfig, create_planner
from .env import GymEnvWrapper, EnvConfig, collect_trajectories, make_dataset, Simple2DEnv
from .sigreg import SIGReg
from .encoder import CNNEncoder, ActionEncoder, Projector
from .predictor import ARPredictor, AdaLNBlock, PredictorProjector

__all__ = [
    "LeWorldModel",
    "LeWMConfig",
    "create_model",
    "Trainer",
    "TrainConfig",
    "create_trainer",
    "CEMPlanner",
    "CEMConfig",
    "create_planner",
    "GymEnvWrapper",
    "EnvConfig",
    "collect_trajectories",
    "make_dataset",
    "Simple2DEnv",
    "SIGReg",
    "CNNEncoder",
    "ActionEncoder",
    "Projector",
    "ARPredictor",
    "AdaLNBlock",
    "PredictorProjector",
]

__version__ = "0.1.0"