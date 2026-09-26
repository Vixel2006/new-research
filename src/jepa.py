from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

try:
    from .modules import ARPredictor, Embedder, SIGReg, ViT
except ImportError:  # running as a plain script: python src/jepa.py
    from modules import ARPredictor, Embedder, SIGReg, ViT


@dataclass
class ModelConfig:
    """Every size :class:`JEPA` needs: the whole architecture in one object.

    Single source of truth for the model. ``nnx`` keeps a plain dataclass out of
    the checkpoint state, so this rides alongside the weights as JSON in
    ``run_config.json`` and a later ``solve`` rebuilds the exact same shapes
    from it.
    """

    # Pixels -> latents: per-frame ViT.
    img_size: int = 64
    in_channels: int = 3
    patch_size: int = 16
    embed_dim: int = 192  # latent width, shared by every sub-module
    enc_layers: int = 4
    enc_heads: int = 16

    # Actions + trajectory conditioning.
    action_dim: int = 2
    history_size: int = 3  # frames the predictor conditions on
    action_mlp_scale: int = 4

    # Latents -> next latent: causal AR predictor.
    pred_depth: int = 6
    pred_heads: int = 16
    pred_mlp_dim: int = 2048

    # Loss = mse + sigreg_weight * (SIGReg on encoder latents + on predictions).
    sigreg_weight: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 128
    sigreg_t_max: float = 3.0
    sigreg_seed: int = 0  # fixes SIGReg's random projection directions


class JEPA(nnx.Module):
    """End-to-end JEPA from pixels: ViT encoder + action embedder + AR predictor.

    Args:
        config: the architecture, see :class:`ModelConfig`.
        rngs: parameter init, and SIGReg's fixed projection directions.
    """

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.config = config

        # Pixels -> latent: per-frame ViT (includes BN + Linear target projector).
        self.encoder = ViT(
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            img_size=config.img_size,
            embed_dim=config.embed_dim,
            num_layers=config.enc_layers,
            num_heads=config.enc_heads,
            rngs=rngs,
        )

        # Actions -> per-step conditioning embeddings for the predictor.
        self.action_encoder = Embedder(
            input_dim=config.action_dim,
            smoothed_dim=config.embed_dim,
            emb_dim=config.embed_dim,
            mlp_scale=config.action_mlp_scale,
            rngs=rngs,
        )

        # Latent -> latent: causal transformer with AdaLN-zero action conditioning.
        self.predictor = ARPredictor(
            embed_dim=config.embed_dim,
            num_frames=config.history_size,
            depth=config.pred_depth,
            num_heads=config.pred_heads,
            mlp_dim=config.pred_mlp_dim,
            rngs=rngs,
        )

        # The Gaussianity regularizer is part of the model: it owns fixed random
        # projection directions, so `loss` needs no extra argument and the
        # directions ride along in the checkpoint.
        self.sigreg = SIGReg(
            knots=config.sigreg_knots,
            num_proj=config.sigreg_num_proj,
            embed_dim=config.embed_dim,
            t_max=config.sigreg_t_max,
            seed=config.sigreg_seed,
        )

    def encode(self, pixels: jax.Array) -> jax.Array:
        """Encode a batch of frame sequences: (B, T, H, W, C) -> (B, T, embed_dim)."""
        B, T, H, W, C = pixels.shape
        emb = self.encoder(pixels.reshape(B * T, H, W, C))  # (B*T, D)
        return emb.reshape(B, T, -1)  # (B, T, D)

    def loss(self, pixels: jax.Array, actions: jax.Array) -> dict:
        """Training loss: MSE(z_H, z_pred_H) + SIGReg regularizers.

        Args:
            pixels: (B, H + 1, H, W, C) history frames plus the target frame.
            actions: (B, H + 1, action_dim) aligned with pixels in time.
        Returns:
            dict with keys: loss, mse, sigreg, sigreg_pred.
        """
        H = self.config.history_size

        emb = self.encode(pixels)  # (B, H + 1, D)
        act_emb = self.action_encoder(actions)  # (B, H + 1, D)

        # Teacher forcing: the causal predictor reads the H-step trajectory and
        # predicts z_H. Only the last position carries a target.
        pred = self.predictor(emb[:, :H], act_emb[:, :H])  # (B, H, D)
        mse = jnp.mean((pred[:, -1] - emb[:, H]) ** 2)

        # SIGReg over the encoder latents, (T, B, D)...
        sigreg = self.sigreg(jnp.transpose(emb, (1, 0, 2)))
        sigreg_pred = self.sigreg(jnp.transpose(pred, (1, 0, 2)))

        return {
            "loss": mse + self.config.sigreg_weight * (sigreg + sigreg_pred),
            "mse": mse,
            "sigreg": sigreg,
            "sigreg_pred": sigreg_pred,
        }

    def rollout(
        self,
        init_pixels: jax.Array,
        action_sequence: jax.Array,
        goal_pixels: jax.Array,
        history_actions: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Roll the predictor forward over an action sequence.

        Args:
            init_pixels: (B, history_size, H, W, C) trajectory frames.
            action_sequence: (B, T, action_dim) actions to roll out.
            goal_pixels: (H, W, C) frame shared across the batch, or one
                (B, H, W, C) goal per batch element. Encoded alongside the
                trajectory so it lands in the same BatchNorm batch.
            history_actions: (B, history_size - 1, action_dim) actions that
                produced the trajectory frames; zeros when unknown.
        Returns:
            (B, T, embed_dim) predicted latents, and the goal embedding --
            (embed_dim,) for a shared goal, (B, embed_dim) for per-batch goals.
            The planner scores candidates against it as the CEM cost target.
        """
        H = self.config.history_size
        if init_pixels.shape[1] != H:
            raise ValueError(
                f"expected {H} trajectory frames, got {init_pixels.shape[1]}"
            )

        # (H, W, C) -> (1, H, W, C); (G, H, W, C) stays (G, H, W, C), then the
        # goal is broadcast over the batch and appended as one more time step.
        goal = goal_pixels[None] if goal_pixels.ndim == 3 else goal_pixels
        shared_goal = goal.shape[0] == 1
        B, _, height, width, channels = init_pixels.shape
        goal = jnp.broadcast_to(goal[:, None], (B, 1, height, width, channels))
        frames = jnp.concatenate([init_pixels, goal], axis=1)  # (B, H+1, ...)
        encoded = self.encode(frames)  # (B, H + 1, D)
        trajectory = encoded[:, :H]
        goal_embedding = encoded[0, H] if shared_goal else encoded[:, H]

        if history_actions is None:
            history_actions = jnp.zeros(
                (init_pixels.shape[0], H - 1, action_sequence.shape[-1]),
                dtype=action_sequence.dtype,
            )
        act_emb = self.action_encoder(
            jnp.concatenate([history_actions, action_sequence], axis=1)
        )

        predictions = []
        for step in range(action_sequence.shape[1]):
            next_latent = self.predictor(trajectory, act_emb[:, step : step + H])[
                :, -1:
            ]
            predictions.append(next_latent)
            trajectory = jnp.concatenate([trajectory, next_latent], axis=1)[:, -H:]

        return jnp.concatenate(predictions, axis=1), goal_embedding


if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    model = JEPA(ModelConfig(), rngs)

    pixels = jax.random.uniform(rngs.params(), (4, 4, 64, 64, 3))
    actions = jax.random.normal(rngs.params(), (4, 4, 2))

    print(f"encode:        {model.encode(pixels).shape}")
    losses = {k: float(v) for k, v in model.loss(pixels, actions).items()}
    print(f"loss:          {losses}")

    # One goal frame shared by the batch, exactly as the CEM planner passes it.
    preds, goal_embedding = model.rollout(pixels[:, :3], actions[:, 3:], pixels[0, -1])
    print(f"rollout:       {preds.shape}  goal: {goal_embedding.shape}")
