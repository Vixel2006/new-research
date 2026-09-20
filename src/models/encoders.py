from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

# We need to add some good patch embedding image is: (B, H, W , C)
# now when we do the patching we just do like a kind of reshaping as (B, N, P^2 * C)
# as N = HW/P^2. this way we did the reshaping.
# we can do a simple position embedding with it. just maybe the simple sin, cos thing


@dataclass
class ViTConfig:
    rngs: nnx.Rngs
    batch_size: int = 64
    num_patches: int = 16
    H: int = 64
    W: int = 64
    C: int = 3


class PatchEmbeddings(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        rngs: nnx.Rngs,
    ):
        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches = img_size // (patch_size**2)

        self.patch_embed = nnx.Linear(
            patch_size * patch_size * in_channels,
            embed_dim,
            rngs=rngs,
        )

        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))

        self.pos_embed = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (1, self.num_patches + 1, embed_dim),
            )
            * 0.02
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, _, _, C = x.shape
        P = self.patch_size

        x = x.reshape(B, self.num_patches, P * P * C)  # (B, N, P**2 * C = H * W * C)

        x = self.patch_embed(x)  # (B, N, embed_dim)

        cls = jnp.broadcast_to(
            self.cls_token.value,
            (B, 1, self.embed_dim),
        )  # (B, 1, embed_dim)

        x = jnp.concatenate([cls, x], axis=1)  # (B, N + 1, embed_dim)

        x += self.pos_embed.value

        return x


class ViT(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        rngs: nnx.Rngs,
    ):
        self.embedding = PatchEmbeddings(
            in_channels, patch_size, img_size, embed_dim, rngs
        )

        # Now we make the embedding into a layer norm
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)

        # Now it's time for the multihead attention with the mask

    def __call__(self):
        pass


if __name__ == "__main__":
    rngs = nnx.Rngs(42)
    batch_size = 64
    H = 64
    W = 64
    C = 3
    patch_size = 16
    embed_dim = 196

    img = jax.random.normal(rngs.params(), (batch_size, H, W, C))

    embed_model = PatchEmbeddings(C, patch_size, H * W, embed_dim, rngs)

    out = embed_model(img)

    print(out.shape)  # (64, 17, 196)
