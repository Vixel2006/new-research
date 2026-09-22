import jax
import jax.numpy as jnp
from flax import nnx


class ARLayer(nnx.Module):
    """Transformer block with temporal causal attention and AdaLN-zero action conditioning."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.mha = nnx.MultiHeadAttention(
            num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs,
        )

        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_dim, embed_dim, rngs=rngs),
        )

        # AdaLN-zero: maps the action embedding to per-token modulation params.
        # The output layer is zero-initialized so conditioning starts as identity.
        self.mod = nnx.Sequential(
            nnx.Linear(embed_dim, embed_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(
                embed_dim,
                6 * embed_dim,
                kernel_init=nnx.initializers.zeros,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            ),
        )

    def __call__(
        self,
        x: jax.Array,
        act_emb: jax.Array,
        mask=None,
    ) -> jax.Array:
        # (B, T, 6*D) -> six (B, T, D) modulation tensors.
        scale1, shift1, gate1, scale2, shift2, gate2 = jnp.split(
            self.mod(act_emb), 6, axis=-1
        )

        h = self.norm1(x)
        h = h * (1.0 + scale1) + shift1
        h = self.mha(h, mask=mask)
        x = x + gate1 * h

        h = self.norm2(x)
        h = h * (1.0 + scale2) + shift2
        h = self.mlp(h)
        x = x + gate2 * h

        return x


class ARPredictor(nnx.Module):
    """Autoregressive latent predictor: (B, T, embed_dim) -> (B, T, embed_dim)."""

    def __init__(
        self,
        embed_dim: int,
        num_frames: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # Learned positional embedding over trajectory/history positions.
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_frames, embed_dim)) * 0.02
        )

        # (1, 1, T, T) broadcastable to (B, heads, T, T): position t attends to
        # frames 0..t (blocks look-ahead to future embeddings).
        self.causal_mask = jnp.tril(jnp.ones((num_frames, num_frames), dtype=bool))[
            None, None
        ]

        self.layers = nnx.List(
            [ARLayer(embed_dim, num_heads, mlp_dim, rngs) for _ in range(depth)]
        )

        # Head projector mirrors the encoder's projection (BN + Linear), per the
        # paper. BN sees the whole (B, T) sequence as its batch; pass
        # use_running_average=True for deployment/rollout (fixed statistics).
        self.head_norm = nnx.BatchNorm(embed_dim, rngs=rngs)
        self.head = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(
        self,
        emb: jax.Array,
        act_emb: jax.Array,
        use_running_average: bool | None = None,
    ) -> jax.Array:
        x = emb + self.pos_embed[...]  # (B, T, embed_dim)

        for layer in self.layers:
            x = layer(x, act_emb, mask=self.causal_mask)

        x = self.head_norm(x, use_running_average=use_running_average)
        x = self.head(x)
        return x


if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    embed_dim = 192
    num_frames = 3

    predictor = ARPredictor(
        embed_dim=embed_dim,
        num_frames=num_frames,
        depth=6,
        num_heads=16,
        mlp_dim=768,
        rngs=rngs,
    )

    emb = jax.random.normal(jax.random.key(0), (4, num_frames, embed_dim))
    act_emb = jax.random.normal(jax.random.key(1), (4, num_frames, embed_dim))

    pred = predictor(emb, act_emb)
    print(pred.shape)  # (4, 3, 192)
