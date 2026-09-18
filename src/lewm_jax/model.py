"""LeWorldModel: Main JEPA Model"""

from flax import nnx
import jax.numpy as jnp
import jax
from dataclasses import dataclass

from .encoder import CNNEncoder, ActionEncoder, Projector
from .predictor import ARPredictor, PredictorProjector
from .sigreg import SIGReg


@dataclass
class LeWMConfig:
    """Configuration for LeWorldModel"""
    embed_dim: int = 192
    img_size: int = 64
    in_channels: int = 3
    action_dim: int = 2
    history_size: int = 3
    num_preds: int = 1

    # Predictor
    pred_depth: int = 6
    pred_heads: int = 16
    pred_mlp_dim: int = 2048
    pred_dropout: float = 0.1

    # Projector
    proj_hidden_dim: int = 2048

    # SIGReg
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    sigreg_weight: float = 0.1


class LeWorldModel(nnx.Module):
    """LeWorldModel: Joint-Embedding Predictive Architecture"""

    def __init__(self, config: LeWMConfig, rngs: nnx.Rngs):
        self.config = config

        # Encoder: pixels -> embeddings
        self.encoder = CNNEncoder(
            embed_dim=config.embed_dim,
            in_channels=config.in_channels,
            rngs=rngs,
        )

        # Projector for encoder outputs (with BatchNorm)
        self.projector = Projector(
            input_dim=config.embed_dim,
            output_dim=config.embed_dim,
            hidden_dim=config.proj_hidden_dim,
            rngs=rngs,
        )

        # Action encoder
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            embed_dim=config.embed_dim,
            rngs=rngs,
        )

        # Predictor: (z_t, a_t) -> z_{t+1}
        self.predictor = ARPredictor(
            embed_dim=config.embed_dim,
            num_frames=config.history_size,
            depth=config.pred_depth,
            num_heads=config.pred_heads,
            mlp_dim=config.pred_mlp_dim,
            dropout=config.pred_dropout,
            rngs=rngs,
        )

        # Projector for predictor outputs
        self.pred_projector = PredictorProjector(
            input_dim=config.embed_dim,
            output_dim=config.embed_dim,
            hidden_dim=config.proj_hidden_dim,
            rngs=rngs,
        )

        # SIGReg regularizer
        self.sigreg = SIGReg(
            knots=config.sigreg_knots,
            num_proj=config.sigreg_num_proj,
            rngs=rngs,
        )

    def encode(self, pixels: jax.Array) -> jax.Array:
        """
        Encode pixels to embeddings.
        Args:
            pixels: (B, T, H, W, C) or (B, H, W, C)
        Returns:
            (B, T, D) or (B, D)
        """
        if pixels.ndim == 4:
            pixels = pixels[:, None]  # (B, 1, H, W, C)

        B, T, H, W, C = pixels.shape
        pixels = pixels.reshape(B * T, H, W, C)

        # Encode each frame
        emb = self.encoder(pixels)  # (B*T, D)
        emb = self.projector(emb)   # (B*T, D)
        emb = emb.reshape(B, T, -1)  # (B, T, D)
        return emb

    def encode_actions(self, actions: jax.Array) -> jax.Array:
        """
        Encode action sequence.
        Args:
            actions: (B, T, action_dim)
        Returns:
            (B, T, D)
        """
        return self.action_encoder(actions)

    def predict(
        self,
        embeddings: jax.Array,
        action_embeddings: jax.Array,
    ) -> jax.Array:
        """
        Predict next embeddings given history.
        Args:
            embeddings: (B, T, D)
            action_embeddings: (B, T, D)
        Returns:
            (B, T, D) - predicted embeddings for each step
        """
        pred = self.predictor(embeddings, action_embeddings)
        pred = self.pred_projector(pred)
        return pred

    def compute_loss(
        self,
        pixels: jax.Array,
        actions: jax.Array,
    ) -> dict:
        """
        Compute LeWM loss (prediction + SIGReg).
        Args:
            pixels: (B, T, H, W, C) where T = history_size + num_preds
            actions: (B, T, action_dim)
        Returns:
            dict with losses
        """
        cfg = self.config
        H = cfg.history_size
        N = cfg.num_preds

        # Encode all frames
        emb = self.encode(pixels)  # (B, T, D)
        act_emb = self.encode_actions(actions)  # (B, T, D)

        # Teacher forcing: predict from history
        ctx_emb = emb[:, :H]           # (B, H, D)
        ctx_act = act_emb[:, :H]       # (B, H, D)
        tgt_emb = emb[:, H:H+N]        # (B, N, D) - targets

        # Predict
        pred_emb = self.predict(ctx_emb, ctx_act)  # (B, H, D)
        pred_emb = pred_emb[:, -N:]    # (B, N, D) - last N predictions

        # Prediction loss (MSE)
        pred_loss = jnp.mean((pred_emb - tgt_emb) ** 2)

        # SIGReg loss on all embeddings
        # Reshape to (T, B, D) for SIGReg
        emb_tbd = jnp.transpose(emb, (1, 0, 2))
        sigreg_loss = self.sigreg(emb_tbd)

        total_loss = pred_loss + cfg.sigreg_weight * sigreg_loss

        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
        }

    def rollout(
        self,
        init_pixels: jax.Array,
        action_sequence: jax.Array,
    ) -> jax.Array:
        """
        Autoregressive rollout in latent space.
        Args:
            init_pixels: (B, H, H, W, C) or (B, H, W, C) - history frames
            action_sequence: (B, T_rollout, action_dim) - future actions
        Returns:
            (B, T_rollout, D) - predicted embeddings
        """
        # Encode initial history
        if init_pixels.ndim == 4:
            init_pixels = init_pixels[:, None]

        emb = self.encode(init_pixels)  # (B, H, D)
        act_emb = self.encode_actions(action_sequence)  # (B, T, D)

        B, H, D = emb.shape
        T = action_sequence.shape[1]

        # Autoregressive rollout
        preds = []
        curr_emb = emb

        for t in range(T):
            # Use last H embeddings
            ctx_emb = curr_emb[:, -H:]
            ctx_act = act_emb[:, t:t+H] if t + H <= T else act_emb[:, t:]

            # Pad if needed
            if ctx_act.shape[1] < H:
                pad = jnp.zeros((B, H - ctx_act.shape[1], D))
                ctx_act = jnp.concatenate([ctx_act, pad], axis=1)

            # Predict next
            pred = self.predict(ctx_emb, ctx_act)[:, -1:]  # (B, 1, D)
            preds.append(pred)

            # Append to history
            curr_emb = jnp.concatenate([curr_emb, pred], axis=1)

        return jnp.concatenate(preds, axis=1)  # (B, T, D)


def create_model(
    embed_dim: int = 192,
    img_size: int = 64,
    action_dim: int = 2,
    history_size: int = 3,
    seed: int = 0,
) -> LeWorldModel:
    """Factory function to create LeWorldModel with default config."""
    config = LeWMConfig(
        embed_dim=embed_dim,
        img_size=img_size,
        action_dim=action_dim,
        history_size=history_size,
    )
    rngs = nnx.Rngs(seed)
    return LeWorldModel(config, rngs)