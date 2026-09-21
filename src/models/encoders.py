import jax
import jax.numpy as jnp
from flax import nnx


class PatchEmbeddings(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        trajectory: int,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        rngs: nnx.Rngs,
    ):
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patches per frame.
        self.num_patches = (img_size // patch_size) ** 2
        # Per frame a [CLS] token is prepended to its patches.
        self.tokens_per_frame = self.num_patches + 1
        self.num_tokens = trajectory * self.tokens_per_frame

        self.patch_embed = nnx.Linear(
            patch_size * patch_size * in_channels,
            embed_dim,
            rngs=rngs,
        )

        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))

        self.pos_embed = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (1, self.num_tokens, embed_dim),
            )
            * 0.02
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, H, W, C = x.shape
        P = self.patch_size

        # (B, T, H, W, C) -> (B*T, H, W, C) -> (B*T, H/P, W/P, P, P, C)
        x = x.reshape(B * T, H, W, C)
        x = x.reshape(B * T, H // P, P, W // P, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)

        x = x.reshape(
            B * T,
            self.num_patches,
            P * P * C,
        )

        x = self.patch_embed(x)  # (B*T, N, embed_dim)

        # Prepend one [CLS] token per frame.
        cls = jnp.broadcast_to(
            self.cls_token[...],
            (B * T, 1, self.embed_dim),
        )  # (B*T, 1, embed_dim)

        x = jnp.concatenate([cls, x], axis=1)  # (B*T, N + 1, embed_dim)

        # Interleave frames: (B, T, N + 1, embed_dim) -> (B, T*(N+1), embed_dim).
        # Each frame's tokens stay contiguous as a block.
        x = x.reshape(B, T, self.tokens_per_frame, self.embed_dim)
        x = x.reshape(B, self.num_tokens, self.embed_dim)

        x += self.pos_embed[...]

        return x


class ViTLayer(nnx.Module):
    def __init__(
        self,
        trajectory: int,
        tokens_per_frame: int,
        embed_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        # Frame-level causal mask for autoregressive frame prediction: frame t can
        # only attend to frames 0..t.
        self.mask = jnp.tril(jnp.ones((trajectory, trajectory)))
        self.tokens_per_frame = tokens_per_frame

        # Expand it once to the token sequence: each frame occupies tokens_per_frame
        # consecutive tokens, so the (T, T) tril becomes a block-lower-triangular
        # (L, L) boolean mask with L = T * tokens_per_frame. This is the frame-level
        # boolean applied once per frame: inside a frame block it's fully connected,
        # across frames it's causal.
        tpf = tokens_per_frame
        self.causal_mask = jnp.kron(self.mask, jnp.ones((tpf, tpf), dtype=bool))[
            None, None
        ]  # (1, 1, L, L) broadcastable to (B, heads, L, L)

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
            nnx.Linear(embed_dim, embed_dim * 3, rngs=rngs),
            nnx.gelu,
            nnx.Linear(embed_dim * 3, embed_dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x += self.mha(self.norm1(x), mask=self.causal_mask)
        x += self.mlp(self.norm2(x))
        return x


class ViT(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        trajectory: int,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        self.trajectory = trajectory
        self.embed_dim = embed_dim

        self.embedding = PatchEmbeddings(
            in_channels, trajectory, patch_size, img_size, embed_dim, rngs
        )

        self.tokens_per_frame = self.embedding.tokens_per_frame

        self.layers = nnx.List(
            [
                ViTLayer(trajectory, self.tokens_per_frame, embed_dim, num_heads, rngs)
                for _ in range(num_layers)
            ]
        )

        self.proj = nnx.Sequential(
            nnx.BatchNorm(embed_dim, rngs=rngs),
            nnx.Linear(embed_dim, embed_dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T = x.shape[:2]

        x = self.embedding(x)  # (B, T * (N + 1), embed_dim)
        for layer in self.layers:
            x = layer(x)

        # Collect the [CLS] token of each frame.
        x = x.reshape(B, T, self.tokens_per_frame, self.embed_dim)
        cls = x[:, :, 0]  # (B, T, embed_dim)

        x = self.proj(cls)  # (B, T, embed_dim)
        return x


class Embedder(nnx.Module):
    """Action Encoder and embedder"""

    def __init__(
        self,
        input_dim: int,
        smoothed_dim: int,
        emb_dim: int,
        mlp_scale: int,
        rngs: nnx.Rngs,
    ):
        self.patch_embed = nnx.Conv(
            input_dim, smoothed_dim, kernel_size=1, strides=1, rngs=rngs
        )
        self.embed = nnx.Sequential(
            nnx.Linear(smoothed_dim, mlp_scale * emb_dim, rngs=rngs),
            nnx.selu,
            nnx.Linear(mlp_scale * emb_dim, emb_dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.transpose(0, 2, 1)  # (B, T, D) -> (B, D, T)
        x = self.patch_embed(x)
        x = x.transpose(0, 2, 1)  # (B, D, T) -> (B, T, D)
        x = self.embed(x)
        return x


if __name__ == "__main__":
    rngs = nnx.Rngs(42)
    batch_size = 64
    T = 10
    H = 64
    W = 64
    C = 3
    D = 10
    patch_size = 16
    embed_dim = 192
    num_layers = 4
    num_heads = 16

    o_t = jax.random.normal(rngs.params(), (batch_size, T, H, W, C))
    a_t = jax.random.normal(rngs.params(), (batch_size, T, D))

    vit = ViT(C, T, patch_size, H, embed_dim, num_layers, num_heads, rngs)
    embedder = Embedder(D, D, D, 4, rngs)

    z_t = vit(o_t)
    a_t = embedder(a_t)

    print(z_t.shape)  # (64, 10, 192)
    print(a_t.shape)  # (64, 10, 10)
