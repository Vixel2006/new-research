import jax
import jax.numpy as jnp
from flax import nnx


class SIGReg(nnx.Module):
    """Epps-Pulley Gaussianity regularizer over random latent projections."""

    def __init__(
        self,
        knots: int = 17,
        num_proj: int = 128,
        embed_dim: int = 192,
        t_max: float = 3.0,
        seed: int = 0,
    ):
        self.knots = knots
        self.num_proj = num_proj
        self.embed_dim = embed_dim

        # Random unit-norm projection directions, sampled once and kept fixed.
        directions = jax.random.normal(jax.random.key(seed), (num_proj, embed_dim))
        self.directions = directions / jnp.linalg.norm(
            directions, axis=1, keepdims=True
        )  # (num_proj, embed_dim)

        # Quadrature nodes + trapezoid weights (with a Gaussian window) for
        # the EP integral, and the N(0, 1) characteristic function target.
        t = jnp.linspace(0.0, t_max, knots)
        dt = t_max / (knots - 1)
        weights = 2.0 * dt * jnp.exp(-(t**2) / 2)
        weights = weights.at[0].mul(0.5).at[-1].mul(0.5)  # trapezoid endpoints
        self.t = t
        self.weights = weights
        self.gaussian_cf = jnp.exp(-(t**2) / 2)

    def __call__(self, embeddings: jax.Array) -> jax.Array:
        assert embeddings.shape[-1] == self.embed_dim, (
            f"SIGReg expects embed_dim {self.embed_dim}, got {embeddings.shape[-1]}"
        )

        # Flatten leading dims (e.g. (T, B, D)) into a single sample axis.
        z = embeddings.reshape(-1, self.embed_dim)

        # Standardize across the sample axis (zero mean, unit variance).
        z = z - z.mean(axis=0, keepdims=True)
        z = z / jnp.maximum(z.std(axis=0, keepdims=True), 1e-6)
        # Clip before the characteristic-function evaluation: on near-degenerate
        # batches the standardization can produce extreme tails whose cos/sin
        # numerics destabilise training. Real training data stays well inside.
        z = jnp.clip(z, -8.0, 8.0)

        # 1D projections: (N, num_proj).
        p = z @ self.directions.T

        # Empirical characteristic function E[e^{i t p}] = E[cos] + i E[sin].
        th = p[:, :, None] * self.t[None, None, :]  # (N, num_proj, knots)
        ecf_real = jnp.cos(th).mean(axis=0)  # (num_proj, knots)
        ecf_imag = jnp.sin(th).mean(axis=0)  # (num_proj, knots)

        # Squared deviation from the N(0, 1) CF (imag part of target is 0).
        err = (ecf_real - self.gaussian_cf[None, :]) ** 2 + ecf_imag**2

        # Weighted quadrature scaled by n, averaged over projections.
        statistics = (err @ self.weights) * p.shape[0]  # (num_proj,)
        return statistics.mean()


if __name__ == "__main__":
    sigreg = SIGReg(knots=17, num_proj=128, embed_dim=192)

    # Random standard-normal embeddings: loss should be small but > 0.
    z = jax.random.normal(jax.random.key(0), (10, 32, 192))  # (T, B, D)
    print(f"Random normal loss: {sigreg(z):.4f}")

    # Collapsed embeddings (all identical): loss should be large.
    z_collapsed = jnp.zeros((10, 32, 192)) + 0.7
    print(f"Collapsed loss:     {sigreg(z_collapsed):.4f}")
