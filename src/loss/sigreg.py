import jax
import jax.numpy as jnp
from flax import nnx


class SIGReg(nnx.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024, rngs: nnx.Rngs = None, embed_dim: int = None):
        self.num_proj = num_proj
        t = jnp.linspace(0, 3, knots, dtype=jnp.float32)
        dt = 3.0 / (knots - 1)
        weights = jnp.full((knots,), 2 * dt, dtype=jnp.float32)
        weights = weights.at[0].set(dt).at[-1].set(dt)
        window = jnp.exp(-(t**2) / 2.0)
        self.t = t
        self.phi = window
        self.weights = weights * window

        # Fixed random projection matrix - sample at init if embed_dim is known
        if embed_dim is not None:
            key = jax.random.key(42)
            A = jax.random.normal(key, (embed_dim, num_proj))
            A = A / jnp.linalg.norm(A, axis=0, keepdims=True)
            self.A = A
        else:
            self.A = None

    def __call__(self, embeddings: jax.Array) -> jax.Array:
        """
        Args:
            embeddings: (T, B, D) - time, batch, embedding_dim
        Returns:
            scalar loss
        """
        T, _, D = embeddings.shape

        # Use stored projection matrix (must be initialized with correct D)
        if self.A is None:
            raise ValueError("SIGReg projection matrix not initialized. Provide embed_dim at construction.")
        if self.A.shape[0] != D:
            raise ValueError(f"SIGReg projection matrix has wrong shape: {self.A.shape[0]} != {D}")

        # Project: (T, B, D) @ (D, M) -> (T, B, M)
        proj = embeddings @ self.A

        # Epps-Pulley statistic
        # x_t: (T, B, M, K)
        x_t = proj[..., None] * self.t  # broadcast

        # Characteristic function difference
        cos_diff = jnp.mean(jnp.cos(x_t), axis=(0, 1)) - self.phi  # (M, K)
        sin_diff = jnp.mean(jnp.sin(x_t), axis=(0, 1))  # (M, K)
        err = cos_diff**2 + sin_diff**2  # (M, K)

        # Integrate
        stat = jnp.sum(err * self.weights, axis=-1) * T  # (M,)
        return jnp.mean(stat)
