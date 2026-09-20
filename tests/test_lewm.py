"""Tests for LeWorldModel JAX"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from lewm import (
    LeWorldModel,
    SIGReg,
    CNNEncoder,
    ActionEncoder,
    ARPredictor,
    Simple2DEnv,
    collect_trajectories,
    make_dataset,
    create_planner,
)


def test_sigreg():
    """Test SIGReg regularizer"""
    print("Testing SIGReg...")
    sigreg = SIGReg(knots=17, num_proj=128, embed_dim=192)

    # Random embeddings
    embeddings = jax.random.normal(jax.random.key(0), (10, 32, 192))  # (T, B, D)
    loss = sigreg(embeddings)

    assert loss.shape == ()
    assert loss > 0
    print(f"  SIGReg loss: {loss:.4f}")
    print("  ✓ SIGReg works")


def test_encoder():
    """Test CNN encoder"""
    print("\nTesting CNN Encoder...")
    encoder = CNNEncoder(embed_dim=192, in_channels=3, rngs=nnx.Rngs(0))

    # Random images
    pixels = jax.random.uniform(jax.random.key(0), (4, 64, 64, 3))
    emb = encoder(pixels)

    assert emb.shape == (4, 192)
    print(f"  Output shape: {emb.shape}")
    print("  ✓ Encoder works")


def test_action_encoder():
    """Test Action encoder"""
    print("\nTesting Action Encoder...")
    action_enc = ActionEncoder(action_dim=2, embed_dim=192, rngs=nnx.Rngs(0))

    actions = jax.random.normal(jax.random.key(0), (4, 5, 2))  # (B, T, action_dim)
    act_emb = action_enc(actions)

    assert act_emb.shape == (4, 5, 192)
    print(f"  Output shape: {act_emb.shape}")
    print("  ✓ Action Encoder works")


def test_predictor():
    """Test AR Predictor"""
    print("\nTesting AR Predictor...")
    predictor = ARPredictor(
        embed_dim=192,
        num_frames=3,
        depth=2,
        num_heads=4,
        mlp_dim=512,
        rngs=nnx.Rngs(0),
    )

    emb = jax.random.normal(jax.random.key(0), (4, 3, 192))
    act_emb = jax.random.normal(jax.random.key(1), (4, 3, 192))
    pred = predictor(emb, act_emb)

    assert pred.shape == (4, 3, 192)
    print(f"  Output shape: {pred.shape}")
    print("  ✓ Predictor works")


def test_full_model():
    """Test full LeWorldModel"""
    print("\nTesting Full Model...")
    model = LeWorldModel.create(
        embed_dim=128,
        img_size=64,
        action_dim=2,
        history_size=3,
        seed=0,
    )

    # Test compute_loss
    pixels = jax.random.uniform(jax.random.key(0), (4, 4, 64, 64, 3))
    actions = jax.random.normal(jax.random.key(1), (4, 4, 2))

    losses = model.compute_loss(pixels, actions, SIGReg(embed_dim=128))
    print(f"  Losses: {losses}")
    assert "loss" in losses
    assert "pred_loss" in losses
    assert "sigreg_loss" in losses
    print("  ✓ Model loss works")

    # Test rollout
    init_pixels = pixels[:, :3]
    action_seq = actions[:, 3:]
    pred_emb = model.rollout(init_pixels, action_seq)
    print(f"  Rollout shape: {pred_emb.shape}")
    assert pred_emb.shape == (4, 1, 128)
    print("  ✓ Rollout works")


def test_environment():
    """Test Simple2DEnv"""
    print("\nTesting Simple2DEnv...")
    env = Simple2DEnv(img_size=64)

    obs, _ = env.reset(seed=0)
    assert obs.shape == (64, 64, 3)
    print(f"  Obs shape: {obs.shape}")

    action = np.array([0.01, 0.01])
    obs, reward, term, trunc, _ = env.step(action)
    print(f"  Reward: {reward:.4f}")
    env.close()
    print("  ✓ Environment works")


def test_data_pipeline():
    """Test data collection and dataset creation"""
    print("\nTesting Data Pipeline...")
    env = Simple2DEnv(img_size=64)

    trajectories = collect_trajectories(env, num_episodes=5, max_steps=20)
    print(f"  Trajectories: pixels={trajectories['pixels'].shape}, actions={trajectories['actions'].shape}")

    train_iter = make_dataset(trajectories, history_size=3, num_preds=1, batch_size=4)
    batch = next(train_iter)
    print(f"  Batch: pixels={batch['pixels'].shape}, actions={batch['actions'].shape}")

    env.close()
    print("  ✓ Data pipeline works")


def test_planner():
    """Test CEM Planner"""
    print("\nTesting CEM Planner...")
    model = LeWorldModel.create(embed_dim=64, img_size=64, action_dim=2, history_size=3, seed=0)
    planner = create_planner(model, horizon=5, num_samples=20, num_iterations=3)

    init_obs = jax.random.uniform(jax.random.key(0), (1, 64, 64, 3))
    goal_obs = jax.random.uniform(jax.random.key(1), (1, 64, 64, 3))

    actions = planner.plan(init_obs, goal_obs)
    print(f"  Planned actions: {actions.shape}")
    assert actions.shape == (5, 2)
    print("  ✓ Planner works")


def test_training_step():
    """Test a few training steps"""
    print("\nTesting Training Step...")
    from lewm import Trainer, TrainConfig

    model = LeWorldModel.create(embed_dim=64, img_size=64, action_dim=2, history_size=3, seed=0)
    env = Simple2DEnv(img_size=64)
    trajectories = collect_trajectories(env, num_episodes=10, max_steps=20)
    train_iter = make_dataset(trajectories, history_size=3, num_preds=1, batch_size=4)

    config = TrainConfig(max_steps=10, lr=1e-4, batch_size=4)
    trainer = Trainer(model, config, train_iter)
    sigreg_fn = SIGReg(embed_dim=64)

    for i in range(3):
        batch = next(train_iter)
        losses = trainer.train_step(batch, sigreg_fn)
        print(f"  Step {i}: loss={losses['loss']:.4f}, pred={losses['pred_loss']:.4f}, sigreg={losses['sigreg_loss']:.4f}")

    env.close()
    print("  ✓ Training step works")


if __name__ == "__main__":
    print("Running LeWorldModel JAX Tests\n")
    test_sigreg()
    test_encoder()
    test_action_encoder()
    test_predictor()
    test_full_model()
    test_environment()
    test_data_pipeline()
    test_planner()
    test_training_step()
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)