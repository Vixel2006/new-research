"""Autoregressive Predictor with AdaLN-zero conditioning"""

from flax import nnx
import jax.numpy as jnp
import jax


def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    """AdaLN-zero modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class AdaLNBlock(nnx.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        rngs: nnx.Rngs = None,
    ):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            dropout_rate=dropout,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(dim, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Dropout(dropout, rngs=rngs),
            nnx.Linear(mlp_dim, dim, rngs=rngs),
            nnx.Dropout(dropout, rngs=rngs),
        )

        # AdaLN modulation - initialized to zero
        self.ada_ln = nnx.Sequential(
            nnx.silu,
            nnx.Linear(dim, 6 * dim, rngs=rngs),
        )
        # Zero init for progressive action conditioning
        self.ada_ln[-1].kernel = nnx.Param(jnp.zeros_like(self.ada_ln[-1].kernel.value))
        self.ada_ln[-1].bias = nnx.Param(jnp.zeros_like(self.ada_ln[-1].bias.value))

    def __call__(
        self,
        x: jax.Array,
        cond: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """
        Args:
            x: (B, T, D)
            cond: (B, T, D) - action conditioning
            mask: (T, T) causal mask
        """
        # AdaLN modulation parameters
        mod = self.ada_ln(cond)  # (B, T, 6*D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(mod, 6, axis=-1)

        # Attention with modulation
        h = self.norm1(x)
        h = modulate(h, shift_msa, scale_msa)
        h = self.attn(h, h, h, mask=mask)
        x = x + gate_msa * h

        # MLP with modulation
        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp * h

        return x


class ARPredictor(nnx.Module):
    """Autoregressive predictor for next-step embedding prediction"""

    def __init__(
        self,
        embed_dim: int = 192,
        num_frames: int = 3,
        depth: int = 6,
        num_heads: int = 16,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Positional embedding
        self.pos_emb = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_frames, embed_dim)) * 0.02
        )
        self.pos_dropout = nnx.Dropout(emb_dropout, rngs=rngs)

        # Transformer blocks with AdaLN
        self.blocks = [
            AdaLNBlock(embed_dim, num_heads, mlp_dim, dropout, rngs=rngs)
            for _ in range(depth)
        ]

        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)

        # Causal mask
        self.causal_mask = nnx.Variable(
            jnp.tril(jnp.ones((num_frames, num_frames), dtype=bool))
        )

    def __call__(
        self,
        embeddings: jax.Array,
        action_embeddings: jax.Array,
    ) -> jax.Array:
        """
        Args:
            embeddings: (B, T, D) - history of state embeddings
            action_embeddings: (B, T, D) - history of action embeddings
        Returns:
            (B, T, D) - predicted next embeddings
        """
        B, T, D = embeddings.shape
        assert T <= self.num_frames, f"T={T} > num_frames={self.num_frames}"

        # Add positional embedding
        x = embeddings + self.pos_emb.value[:, :T]
        x = self.pos_dropout(x)

        # Apply transformer blocks with action conditioning
        mask = self.causal_mask.value[:T, :T]
        for block in self.blocks:
            x = block(x, action_embeddings, mask=mask)

        x = self.norm(x)
        return x


class PredictorProjector(nnx.Module):
    """Projector for predictor outputs (same as encoder projector)"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        rngs: nnx.Rngs = None,
    ):
        self.net = nnx.Sequential(
            nnx.Linear(input_dim, hidden_dim, rngs=rngs),
            nnx.BatchNorm(hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, output_dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B*T, D) or (B, T, D)
        if x.ndim == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            out = self.net(x)
            return out.reshape(B, T, -1)
        return self.net(x)