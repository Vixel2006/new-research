from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class PatchConfig:
    img_sz: int
    patch_sz: int
    num_chans: int
    hidden_sz: int
    rngs: nnx.Rngs

    @property
    def num_patches(self) -> int:
        return (self.img_sz // self.patch_sz) ** 2


@dataclass
class ViTConfig:
    patch: PatchConfig
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_classes: int = 0

    @property
    def hidden_sz(self) -> int:
        return self.patch.hidden_sz


class PatchEmbeddings(nnx.Module):
    def __init__(self, cfg: PatchConfig):
        super().__init__()
        self.cfg = cfg

        self.projection = nnx.Conv(
            in_features=cfg.num_chans,
            out_features=cfg.hidden_sz,
            kernel_size=(cfg.patch_sz, cfg.patch_sz),
            strides=(cfg.patch_sz, cfg.patch_sz),
            rngs=cfg.rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, H, W, C = x.shape
        chex.assert_shape(x, (B, H, W, C))
        x = self.projection(x)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)
        chex.assert_shape(x, (B, self.cfg.num_patches, D))
        return x


class Attention(nnx.Module):
    def __init__(self, hidden_sz: int, num_heads: int, dropout: float, rngs: nnx.Rngs):
        super().__init__()
        if hidden_sz % num_heads != 0:
            raise ValueError(
                f"hidden_sz={hidden_sz} must be divisible by num_heads={num_heads}"
            )
        self.hidden_sz = hidden_sz
        self.num_heads = num_heads
        self.head_dim = hidden_sz // num_heads

        self.qkv = nnx.Linear(hidden_sz, hidden_sz * 3, rngs=rngs)
        self.proj = nnx.Linear(hidden_sz, hidden_sz, rngs=rngs)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, N, D = x.shape
        chex.assert_shape(x, (B, N, D))
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        chex.assert_equal_shape((q, k, v))
        chex.assert_shape(q, (B, self.num_heads, N, self.head_dim))

        attn = (q @ jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.head_dim)
        attn = nnx.softmax(attn, axis=-1)
        x = attn @ v
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(B, N, D)
        x = self.proj(x)
        chex.assert_shape(x, (B, N, D))
        return x


class MLP(nnx.Module):
    def __init__(self, hidden_sz: int, mlp_ratio: float, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(hidden_sz, int(hidden_sz * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(hidden_sz * mlp_ratio), hidden_sz, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_shape(x, (..., self.fc1.in_features))
        x = nnx.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class ViTBlock(nnx.Module):
    def __init__(
        self,
        hidden_sz: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.norm1 = nnx.LayerNorm(hidden_sz, rngs=rngs)
        self.attn = Attention(hidden_sz, num_heads, dropout, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_sz, rngs=rngs)
        self.mlp = MLP(hidden_sz, mlp_ratio, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, N, D = x.shape
        chex.assert_shape(x, (B, N, D))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        chex.assert_shape(x, (B, N, D))
        return x


class ViT(nnx.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.patch_embed = PatchEmbeddings(cfg.patch)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, cfg.hidden_sz)))
        self.pos_embed = nnx.Param(
            jnp.zeros((1, cfg.patch.num_patches + 1, cfg.hidden_sz))
        )
        self.pos_drop = nnx.Dropout(cfg.dropout, rngs=cfg.patch.rngs)
        self.blocks = nnx.List(
            ViTBlock(cfg.hidden_sz, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, rngs=cfg.patch.rngs)
            for _ in range(cfg.num_layers)
        )
        self.norm = nnx.LayerNorm(cfg.hidden_sz, rngs=cfg.patch.rngs)
        self.head = (
            nnx.Linear(cfg.hidden_sz, cfg.num_classes, rngs=cfg.patch.rngs)
            if cfg.num_classes > 0
            else None
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = jnp.broadcast_to(self.cls_token, (B, 1, x.shape[-1]))
        x = jnp.concatenate([cls, x], axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        chex.assert_shape(x, (B, self.patch_embed.cfg.num_patches + 1, x.shape[-1]))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        chex.assert_shape(x, (B, self.patch_embed.cfg.hidden_sz))

        if self.head is not None:
            x = self.head(x)
        return x