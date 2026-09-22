import jax
import jax.numpy as jnp
from flax import nnx


class PatchEmbeddings(nnx.Module):
    """Patchify and embed a single frame: (B, H, W, C) -> (B, N + 1, embed_dim)."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        rngs: nnx.Rngs,
    ):
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patches per frame, plus one [CLS] token prepended to them.
        self.num_patches = (img_size // patch_size) ** 2
        self.tokens_per_frame = self.num_patches + 1

        self.patch_embed = nnx.Linear(
            patch_size * patch_size * in_channels,
            embed_dim,
            rngs=rngs,
        )

        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))

        self.pos_embed = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (1, self.tokens_per_frame, embed_dim),
            )
            * 0.02
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, H, W, C = x.shape
        P = self.patch_size

        # (B, H, W, C) -> (B, H/P, P, W/P, P, C)
        x = x.reshape(B, H // P, P, W // P, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)

        x = x.reshape(
            B,
            self.num_patches,
            P * P * C,
        )

        x = self.patch_embed(x)  # (B, N, embed_dim)

        # Prepend one [CLS] token.
        cls = jnp.broadcast_to(
            self.cls_token[...],
            (B, 1, self.embed_dim),
        )  # (B, 1, embed_dim)

        x = jnp.concatenate([cls, x], axis=1)  # (B, N + 1, embed_dim)

        x += self.pos_embed[...]

        return x


class ViTLayer(nnx.Module):
    """ViT block: spatial self-attention + MLP over a single frame's tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
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
            nnx.Linear(embed_dim, embed_dim * 3, rngs=rngs),
            nnx.gelu,
            nnx.Linear(embed_dim * 3, embed_dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x += self.mha(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x


class ViT(nnx.Module):
    """Per-frame Vision Transformer encoder: (B, H, W, C) -> (B, embed_dim)."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim

        self.embedding = PatchEmbeddings(
            in_channels, patch_size, img_size, embed_dim, rngs
        )

        self.layers = nnx.List(
            [ViTLayer(embed_dim, num_heads, rngs) for _ in range(num_layers)]
        )

        # Project the [CLS] token into the latent space used by SIGReg and the
        # predictor. Needed because the last ViT layer outputs LayerNorm-ed
        # features, which would obstruct the anti-collapse objective.
        self.proj = nnx.Sequential(
            nnx.BatchNorm(embed_dim, rngs=rngs),
            nnx.Linear(embed_dim, embed_dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.embedding(x)  # (B, N + 1, embed_dim)
        for layer in self.layers:
            x = layer(x)

        # Take the [CLS] token of the frame.
        cls = x[:, 0]  # (B, embed_dim)

        x = self.proj(cls)  # (B, embed_dim)
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
        x = self.patch_embed(x)  # (B, T, smoothed_dim)
        x = self.embed(x)  # (B, T, emb_dim)
        return x


if __name__ == "__main__":
    rngs = nnx.Rngs(42)
    batch_size = 64
    H = 64
    W = 64
    C = 3
    D = 10
    T = 10
    patch_size = 16
    embed_dim = 192
    num_layers = 4
    num_heads = 16

    o_t = jax.random.normal(rngs.params(), (batch_size, H, W, C))
    a_t = jax.random.normal(rngs.params(), (batch_size, T, D))

    vit = ViT(C, patch_size, H, embed_dim, num_layers, num_heads, rngs)
    embedder = Embedder(D, D, D, 4, rngs)

    z_t = vit(o_t)
    a_t = embedder(a_t)

    print(z_t.shape)  # (64, 192): one embedding per frame
    print(a_t.shape)  # (64, 10, 10)
