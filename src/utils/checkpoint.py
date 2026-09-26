from __future__ import annotations

import json
from pathlib import Path

from flax import nnx
from orbax import checkpoint as orbax


def _by_canonical_path(pure_dict) -> dict:
    """Flatten a state dict, normalizing list indices to int.

    orbax restores a Sequential's indices as the *strings* ``'0'``, ``'1'``,
    ... while the live model uses ints, so the two key sets only line up after
    this.
    """
    return {
        tuple(int(p) if isinstance(p, str) and p.isdigit() else p for p in path): var
        for path, var in nnx.traversals.flatten_mapping(pure_dict).items()
    }


def save(model, ckpt_dir: str | Path, tag: str = "latest") -> Path:
    """Save the model as a pure-dict checkpoint (orbax-friendly).

    One tree holds everything the model owns (weights + SIGReg's fixed
    directions); the architecture itself is *not* in the weights and comes from
    ``ModelConfig`` in ``run_config.json``.

    The optimizer state is intentionally NOT saved: on load a fresh optimizer
    is built, so checkpoints stay small and only contain what ``solve`` needs.
    """
    path = (Path(ckpt_dir) / tag).resolve()  # orbax requires absolute paths
    orbax.PyTreeCheckpointer().save(
        path,
        {"model": nnx.to_pure_dict(nnx.state(model))},
        force=True,
    )
    return path


def resolve_best_tag(ckpt_dir: str | Path) -> str | None:
    """Tag of the best checkpoint written by training, or None."""
    path = Path(ckpt_dir) / "best.json"
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text())["tag"])
    except (KeyError, ValueError):
        return None


def record_best(ckpt_dir: str | Path, tag: str, step: int, val_loss: float) -> None:
    """Point ``best.json`` at ``tag`` so ``--ckpt best`` can find it later."""
    (Path(ckpt_dir) / "best.json").write_text(
        json.dumps({"tag": tag, "step": step, "val_loss": val_loss}, indent=2)
    )


def load(
    model,
    ckpt_dir: str | Path = "checkpoints",
    tag: str = "latest",
):
    """Restore weights saved by :func:`save` into a model.

    Pass a freshly-built model of the same architecture (from the same
    ``ModelConfig``); its weights are replaced in place. Raises if the shapes do
    not line up, which is what a mismatched ``run_config.json`` looks like.
    """
    path = (Path(ckpt_dir) / tag).resolve()  # orbax requires absolute paths
    restored = orbax.PyTreeCheckpointer().restore(str(path))
    pure = restored["model"]

    expected = _by_canonical_path(nnx.state(model))
    found = _by_canonical_path(pure)
    mismatched = [
        p for p in expected.keys() & found.keys() if expected[p].shape != found[p].shape
    ]
    if expected.keys() != found.keys() or mismatched:
        raise ValueError(
            f"Checkpoint at {path} does not match the current architecture. "
            f"Train and solve must use the same model sizes. "
            f"missing={sorted(expected.keys() - found.keys(), key=str)} "
            f"unexpected={sorted(found.keys() - expected.keys(), key=str)} "
            f"shape_mismatches={[str(p) for p in sorted(mismatched, key=str)]}"
        )

    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, pure)
    nnx.update(model, state)
    return model
