import jax
import jax.numpy as jnp
from flax import nnx


class CNNEncoder(nnx.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        in_channels: int = 3,
        rngs: nnx.Rngs = None,
    ):
        self.conv1 = nnx.Conv(
            in_channels,
            32,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            32, 64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.conv3 = nnx.Conv(
            64, 128, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.conv4 = nnx.Conv(
            128, 256, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs
        )

        self.norm1 = nnx.LayerNorm(32, rngs=rngs)
        self.norm2 = nnx.LayerNorm(64, rngs=rngs)
        self.norm3 = nnx.LayerNorm(128, rngs=rngs)
        self.norm4 = nnx.LayerNorm(256, rngs=rngs)

        # Global average pool + projection
        self.proj = nnx.Linear(256, embed_dim, rngs=rngs)
        self.proj_norm = nnx.BatchNorm(embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (B, H, W, C) in [0, 1] or [0, 255]
        Returns:
            (B, embed_dim)
        """
        # Normalize to [-1, 1] if needed (handle both [0,1] and [0,255])
        x = jnp.where(x.max() > 1.0, x / 127.5 - 1.0, x * 2.0 - 1.0)

        x = nnx.relu(self.norm1(self.conv1(x)))
        x = nnx.relu(self.norm2(self.conv2(x)))
        x = nnx.relu(self.norm3(self.conv3(x)))
        x = nnx.relu(self.norm4(self.conv4(x)))

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 256)

        # Project with BatchNorm (critical for SIGReg!)
        x = self.proj(x)
        x = self.proj_norm(x)
        return x


class ActionEncoder(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        embed_dim: int = 192,
        hidden_mult: int = 4,
        rngs: nnx.Rngs = None,
    ):
        # nnx.Conv expects channels-last: (B, L, C_in) -> (B, L, C_out)
        self.conv = nnx.Conv(
            in_features=action_dim,
            out_features=embed_dim,
            kernel_size=(1,),
            strides=(1,),
            rngs=rngs,
        )
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, hidden_mult * embed_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_mult * embed_dim, embed_dim, rngs=rngs),
        )

    def __call__(self, actions: jax.Array) -> jax.Array:
        """
        Args:
            actions: (B, T, action_dim)
        Returns:
            (B, T, embed_dim)
        """
        # nnx.Conv expects channels-last: (B, T, action_dim) -> (B, T, embed_dim)
        x = self.conv(actions)
        return self.mlp(x)


class Projector(nnx.Module):
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
        return self.net(x)