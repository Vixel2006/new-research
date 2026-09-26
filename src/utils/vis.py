from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_uint8(frames) -> list[np.ndarray]:
    """Normalize (T, H, W, 3) float[0,1] or uint8 frames to uint8."""
    out = []
    for f in np.asarray(frames):
        f = np.asarray(f)
        out.append(
            (f * 255.0).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        )
    return out


def frames_to_gif(frames, path, duration_ms: int = 100, scale: int = 1) -> Path:
    """(T, H, W, 3) float[0,1] or uint8 frames -> animated GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = _to_uint8(frames)
    if not imgs:
        print(f"  (no frames to save for {path.name})")
        return path

    from PIL import Image

    if scale > 1:
        imgs = [
            np.asarray(
                Image.fromarray(f).resize(
                    (f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST
                )
            )
            for f in imgs
        ]
    pil = [Image.fromarray(f) for f in imgs]
    pil[0].save(
        path, save_all=True, append_images=pil[1:], duration=duration_ms, loop=0
    )
    print(f"  video: {path} ({len(pil)} frames)")
    return path


def plot_training_curves(history, path) -> Path:
    """Loss / MSE / SIGReg / LR curves from ``Trainer.history`` -> PNG.

    ``history`` is the list of logged metric dicts (``step`` in each).
    Overlays ``val_loss`` (when present) on the loss panel.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = list(history)
    if not history:
        print(f"  (no metrics to plot for {path.name})")
        return path

    steps = np.asarray([m["step"] for m in history])
    ms = history
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key in zip(axes.flat, ("loss", "mse", "sigreg", "lr")):
        good = [
            (s, m[key])
            for s, m in zip(steps, ms)
            if key in m and m[key] is not None and np.isfinite(m[key])
        ]
        if not good:
            ax.set_visible(False)
            continue
        xs, ys = zip(*good)
        ax.plot(xs, ys, label=key)
        if key == "loss":
            vgood = [
                (s, m["val_loss"])
                for s, m in zip(steps, ms)
                if "val_loss" in m and np.isfinite(m["val_loss"])
            ]
            if vgood:
                vxs, vys = zip(*vgood)
                ax.plot(vxs, vys, "o-", label="val_loss")
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  curves: {path}")
    return path


def _mean_dim_r(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over dimensions of the Pearson correlation between two latents.

    One correlation over the flattened arrays would be dominated by whichever
    dimensions happen to have the largest scale.
    """
    scale = a.std(0) * b.std(0)
    per_dim = np.divide(
        (a * b).mean(0), scale, out=np.zeros_like(scale), where=scale > 0
    )
    return float(per_dim.mean())


def plot_predictions(model, batches, path) -> Path:
    """What the predictor actually predicts: final-step latent vs the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = model.config.history_size
    targets, predictions = [], []
    for batch in batches:
        pixels, actions = batch["pixels"], batch["actions"]
        emb = model.encode(pixels)  # (B, H + 1, D)
        act_emb = model.action_encoder(actions)  # (B, H + 1, D)
        predictions.append(model.predictor(emb[:, :H], act_emb[:, :H])[:, -1])
        targets.append(emb[:, H])
    z = np.concatenate([np.asarray(t) for t in targets])  # (N, D)
    y = np.concatenate([np.asarray(p) for p in predictions])

    finite = np.isfinite(z).all(1) & np.isfinite(y).all(1)
    if not finite.any():
        print(f"  (no finite predictions to plot for {path.name})")
        return path
    z, y = z[finite], y[finite]

    zc, yc = z - z.mean(0), y - y.mean(0)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(z))
    r = _mean_dim_r(zc, yc)
    # Shuffle one side only: a common permutation leaves every correlation
    # unchanged, so that would report the true r as its own control.
    r_shuffled = _mean_dim_r(zc, yc[shuffled])

    mse = ((y - z) ** 2).mean(1)
    cosine = (zc * yc).sum(1) / (
        np.linalg.norm(zc, axis=1) * np.linalg.norm(yc, axis=1) + 1e-8
    )
    # The score to beat: MSE of the best constant predictor, i.e. the training
    # loss a model that ignores the input and emits the mean latent would get.
    mean_mse = float(z.var(0).mean())
    r_squared = 1.0 - float(mse.mean()) / mean_mse if mean_mse > 0 else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]  # scatter, subsampled: N*D is far more points than pixels
    n_points = min(2000, z.size)
    pick = rng.choice(z.size, n_points, replace=False)
    ax.scatter(z.ravel()[pick], y.ravel()[pick], s=4, alpha=0.3, edgecolors="none")
    lo = float(min(z.min(), y.min()))
    hi = float(max(z.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], lw=1, ls="--", c="k", label="perfect")
    ax.set_title(f"target vs predicted (r = {r:.3f}, shuffled = {r_shuffled:.3f})")
    ax.set_xlabel("target z_H")
    ax.set_ylabel("predicted z_H")
    ax.legend()

    ax = axes[0, 1]  # marginals: collapse shows up as one narrow blob
    ax.hist(z.ravel(), bins=60, density=True, alpha=0.6, label="target z_H")
    ax.hist(y.ravel(), bins=60, density=True, alpha=0.6, label="predicted z_H")
    ax.set_title(
        f"latent distribution (std {z.std():.3f} vs {y.std():.3f}, "
        f"target norm {np.linalg.norm(z, axis=1).mean():.2f})"
    )
    ax.set_xlabel("latent value")
    ax.legend()

    # Sorted per-sample errors, each against the score of a useless predictor.
    for ax, values, name, baseline, base_name in (
        (axes[1, 0], np.sort(mse), "mse", mean_mse, "predict the mean"),
        (axes[1, 1], np.sort(1.0 - cosine), "cosine error", 1.0, "no information"),
    ):
        ax.plot(np.arange(len(values)), values, lw=1, label=f"mean {values.mean():.4f}")
        ax.axhline(
            baseline, ls="--", lw=1, c="r", label=f"{base_name} ({baseline:.4f})"
        )
        if values.max() > 0:
            ax.set_yscale("log")
        ax.set_title(f"per-sample {name}, sorted (n = {len(values)})")
        ax.set_xlabel("sample rank")
        ax.set_ylabel(name)
        ax.legend()

    fig.suptitle(f"JEPA predictor: {H}-step history, dim {z.shape[1]}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(
        f"  predictions: {path}  (r {r:.3f} vs {r_shuffled:.3f} shuffled, "
        f"mse {mse.mean():.5f} = R^2 {r_squared:+.2f}, "
        f"cosine error {float((1.0 - cosine).mean()):.4f})"
    )
    return path


def plot_coverage(coverage, path, title: str = "coverage over episode") -> Path:
    """Per-step goal coverage of one rollout -> PNG (green line = solved)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cov = np.asarray(coverage)
    if cov.size == 0:
        print(f"  (no coverage to plot for {path.name})")
        return path

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(cov)), cov)
    ax.axhline(0.95, color="g", ls="--", lw=1, label="goal (0.95)")
    ax.set_xlabel("step")
    ax.set_ylabel("coverage")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  coverage: {path}")
    return path
