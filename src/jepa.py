import jax
import jax.numpy as jnp
from flax import nnx

try:
    from .modules import ARPredictor, Embedder, SIGReg, ViT
except ImportError:  # running as a plain script: python src/jepa.py
    from modules import ARPredictor, Embedder, SIGReg, ViT


class JEPA(nnx.Module):
    """End-to-end JEPA from pixels: ViT encoder + action embedder + AR predictor."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        embed_dim: int = 192,
        img_size: int = 64,
        in_channels: int = 3,
        patch_size: int = 16,
        enc_layers: int = 4,
        enc_heads: int = 16,
        action_dim: int = 2,
        history_size: int = 3,
        pred_depth: int = 6,
        pred_heads: int = 16,
        pred_mlp_dim: int = 2048,
        sigreg_weight: float = 0.1,
    ):
        self.embed_dim = embed_dim
        self.history_size = history_size
        self.sigreg_weight = sigreg_weight

        # Pixels -> latent: per-frame ViT (includes BN + Linear target projector).
        self.encoder = ViT(
            in_channels=in_channels,
            patch_size=patch_size,
            img_size=img_size,
            embed_dim=embed_dim,
            num_layers=enc_layers,
            num_heads=enc_heads,
            rngs=rngs,
        )

        # Actions -> per-step conditioning embeddings for the predictor.
        self.action_encoder = Embedder(
            input_dim=action_dim,
            smoothed_dim=embed_dim,
            emb_dim=embed_dim,
            mlp_scale=4,
            rngs=rngs,
        )

        # Latent -> latent: causal transformer with AdaLN-zero action conditioning.
        self.predictor = ARPredictor(
            embed_dim=embed_dim,
            num_frames=history_size,
            depth=pred_depth,
            num_heads=pred_heads,
            mlp_dim=pred_mlp_dim,
            rngs=rngs,
        )

    def encode(self, pixels: jax.Array) -> jax.Array:
        """Encode pixels to latents.

        Args:
            pixels: (B, T, H, W, C) or (B, H, W, C).
        Returns:
            (B, T, D) or (B, 1, D), where D = embed_dim.
        """
        if pixels.ndim == 3:
            pixels = pixels[None, None]  # (1, 1, H, W, C)
        elif pixels.ndim == 4:
            pixels = pixels[:, None]  # (B, 1, H, W, C)

        B, T, H, W, C = pixels.shape
        emb = self.encoder(pixels.reshape(B * T, H, W, C))  # (B*T, D)
        return emb.reshape(B, T, -1)  # (B, T, D)

    def encode_actions(self, actions: jax.Array) -> jax.Array:
        """Encode an action sequence.

        Args:
            actions: (B, T, action_dim).
        Returns:
            (B, T, D) action embeddings.
        """
        return self.action_encoder(actions)

    def predict(
        self,
        embeddings: jax.Array,
        action_embeddings: jax.Array,
        use_running_average: bool | None = None,
    ) -> jax.Array:
        """Predict future latents from context.

        Args:
            embeddings: (B, H, D) context latents.
            action_embeddings: (B, H, D) action embeddings, one per context step.
            use_running_average: pass True for rollout/inference so the head
                BatchNorm uses running stats instead of batch stats.
        Returns:
            (B, H, D) predicted latents; output at step t targets latent t + 1.
        """
        return self.predictor(
            embeddings,
            action_embeddings,
            use_running_average=use_running_average,
        )

    def criterion(self, z_pred: jax.Array, z_target: jax.Array) -> jax.Array:
        """MSE between the predicted next latent and the target next latent."""
        return jnp.mean((z_pred - z_target) ** 2)

    def loss(
        self,
        pixels: jax.Array,
        actions: jax.Array,
        sigreg_fn: SIGReg | None = None,
    ) -> dict:
        """Training loss: MSE(z_{t+1}, z_pred_{t+1}) + SIGReg regularizer.

        Args:
            pixels: (B, H + 1, H, W, C) history frames plus the target frame.
            actions: (B, H + 1, action_dim) aligned with pixels in time.
            sigreg_fn: SIGReg module matching embed_dim; built from defaults
                when omitted.
        Returns:
            dict with keys: loss, mse, sigreg.
        """
        H = self.history_size

        if sigreg_fn is None:
            sigreg_fn = SIGReg(embed_dim=self.embed_dim)

        # Encode all frames: (B, T, D).
        emb = self.encode(pixels)
        act_emb = self.encode_actions(actions)  # (B, T, D)

        # Teacher forcing: predict z_{H} (the next latent) from the H-step context.
        pred = self.predict(emb[:, :H], act_emb[:, :H])  # (B, H, D)
        z_pred = pred[:, -1]  # (B, D) -> z_pred_{t+1}
        z_target = emb[:, H]  # (B, D) -> z_{t+1}

        mse = self.criterion(z_pred, z_target)

        # SIGReg anti-collapse loss over the encoder latents, (T, B, D).
        sigreg = sigreg_fn(jnp.transpose(emb, (1, 0, 2)))

        return {
            "loss": mse + self.sigreg_weight * sigreg,
            "mse": mse,
            "sigreg": sigreg,
        }

    def cost(
        self,
        init_embeddings: jax.Array,
        action: jax.Array,
        goal_embedding: jax.Array,
    ) -> jax.Array:
        """Cost of an action: how far it takes the next latent from a goal.

        Used for planning (e.g. CEM). The candidate action conditions the last
        context step; earlier steps get zero conditioning.

        Args:
            init_embeddings: (B, H, D) encoded context latents.
            action: (B, action_dim) candidate action.
            goal_embedding: (B, D) target latent for z_{t+1}.
        Returns:
            scalar MSE between the predicted next latent and the goal.
        """
        B, H, D = init_embeddings.shape

        act_emb = self.encode_actions(action[:, None, :])  # (B, 1, D)
        act_emb = jnp.zeros((B, H, D)).at[:, -1].set(act_emb[:, 0])

        z_pred = self.predict(init_embeddings, act_emb, use_running_average=True)[
            :, -1
        ]  # (B, D)

        return self.criterion(z_pred, goal_embedding)

    def rollout(
        self,
        init_pixels: jax.Array,
        action_sequence: jax.Array,
    ) -> jax.Array:
        """Autoregressive rollout in latent space.

        Args:
            init_pixels: (B, H, H, W, C) or (B, H, W, C) history frames.
            action_sequence: (B, T_rollout, action_dim) future actions.
        Returns:
            (B, T_rollout, D) predicted latents.
        """
        if init_pixels.ndim == 4:
            init_pixels = init_pixels[:, None]

        ctx = self.encode(init_pixels)  # (B, H, D)
        act_emb = self.encode_actions(action_sequence)  # (B, T, D)

        B, H, D = ctx.shape
        T = action_sequence.shape[1]

        preds = []
        for t in range(T):
            # Keep the last H latents as context, with the corresponding actions.
            ctx_emb = ctx[:, -H:]
            ctx_act = act_emb[:, t : t + H]
            if ctx_act.shape[1] < H:
                # Pad the action window with zeros when it runs past the end.
                pad = jnp.zeros((B, H - ctx_act.shape[1], D))
                ctx_act = jnp.concatenate([ctx_act, pad], axis=1)

            # Predict the next latent; fix BN stats for inference.
            next_emb = self.predict(ctx_emb, ctx_act, use_running_average=True)[
                :, -1:
            ]  # (B, 1, D)
            preds.append(next_emb)
            ctx = jnp.concatenate([ctx, next_emb], axis=1)

        return jnp.concatenate(preds, axis=1)  # (B, T, D)


if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    model = JEPA(
        rngs,
        img_size=64,
        action_dim=2,
        history_size=3,
    )

    pixels = jax.random.uniform(rngs.params(), (4, 4, 64, 64, 3))
    actions = jax.random.normal(rngs.params(), (4, 4, 2))

    emb = model.encode(pixels)
    print(f"encode:        {emb.shape}")

    losses = model.loss(pixels, actions)
    print(f"loss:          { {k: float(v) for k, v in losses.items()} }")

    goal_emb = emb[:, -1]
    sample_action = actions[:, -1:]
    cost = model.cost(emb[:, :3], sample_action[:, 0], goal_emb)
    print(f"action cost:   {float(cost):.4f}")

    preds = model.rollout(pixels[:, :3], actions[:, 3:])
    print(f"rollout:       {preds.shape}")
