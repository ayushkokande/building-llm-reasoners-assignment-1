#!/usr/bin/env python3
"""
Evaluate train/val loss at every saved checkpoint and plot the real loss curve.

Produces loss_curve.png in the repo root. All points are measured from the
shipped checkpoints and tokenized data — no synthetic curves.

Measured losses are cached in scripts/loss_curve_data.json; delete that file
to force re-evaluation of the checkpoints.

Usage:
    PYTHONPATH=. uv run --with matplotlib python scripts/plot_loss_curve.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO / "artifacts" / "checkpoints"
CACHE = Path(__file__).resolve().parent / "loss_curve_data.json"
N_BATCHES = 25
BATCH_SIZE = 64
CONTEXT = 256
NUM_HEADS = 8
PARAMS_M = 28.9


def evaluate_checkpoints() -> dict:
    import numpy as np
    import torch

    from student.core import run_cross_entropy, run_get_batch
    from student.decode import _infer_hparams_from_state_dict
    from student.lm import TransformerLM

    def eval_loss(model: TransformerLM, ds) -> float:
        losses = []
        with torch.no_grad():
            for _ in range(N_BATCHES):
                x, y = run_get_batch(ds, BATCH_SIZE, CONTEXT, "cpu")
                logits = model(x)
                losses.append(
                    run_cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
                )
        return sum(losses) / len(losses)

    train_ds = np.load(REPO / "artifacts" / "train.npy", mmap_mode="r").ravel()
    val_ds = np.load(REPO / "artifacts" / "val.npy", mmap_mode="r").ravel()

    ckpts = sorted(
        CKPT_DIR.glob("checkpoint_*.pt"),
        key=lambda p: int(re.search(r"\d+", p.stem).group()),
    )

    torch.manual_seed(0)
    data = {
        "n_batches": N_BATCHES,
        "batch_size": BATCH_SIZE,
        "context_length": CONTEXT,
        "seed": 0,
        "iters": [],
        "train_losses": [],
        "val_losses": [],
    }

    first_state = torch.load(ckpts[0], map_location="cpu")
    first_state = first_state["model"] if "model" in first_state else first_state
    hparams = _infer_hparams_from_state_dict(first_state, context_length=CONTEXT, num_heads=NUM_HEADS)

    # Random-init baseline at iteration 0 (expected ~ln(vocab_size) = 9.21).
    states = [(0, None)] + [(int(re.search(r"\d+", p.stem).group()), p) for p in ckpts]
    for it, ckpt_path in states:
        model = TransformerLM(**hparams)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            state = state["model"] if "model" in state else state
            model.load_state_dict(state)
        model.eval()
        data["iters"].append(it)
        data["train_losses"].append(eval_loss(model, train_ds))
        data["val_losses"].append(eval_loss(model, val_ds))
        print(f"iter {it}: train {data['train_losses'][-1]:.4f} val {data['val_losses'][-1]:.4f}")

    CACHE.write_text(json.dumps(data, indent=2))
    print(f"Cached losses to {CACHE}")
    return data


def main() -> None:
    if CACHE.exists():
        data = json.loads(CACHE.read_text())
        print(f"Using cached losses from {CACHE} (delete to re-evaluate)")
    else:
        data = evaluate_checkpoints()

    iters = data["iters"]
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_full, ax_zoom) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1, 1.3]}
    )

    final_ppl = math.exp(val_losses[-1])
    fig.suptitle(
        f"TinyStories LM ({PARAMS_M}M params) — measured loss at saved checkpoints\n"
        f"final val loss {val_losses[-1]:.3f} (ppl {final_ppl:.2f})"
    )

    ax_full.plot(iters, train_losses, "o-", label="train loss", color="tab:blue")
    ax_full.plot(iters, val_losses, "s-", label="val loss", color="tab:orange")
    ax_full.set_yscale("log")
    ax_full.set_ylabel("Cross-entropy loss (log scale)")
    ax_full.set_title("Full run incl. random-init baseline (loss 9.23 ≈ ln 10000)", fontsize=10)
    ax_full.legend()
    ax_full.grid(alpha=0.3, which="both")

    zi = [i for i, it in enumerate(iters) if it > 0]
    z_iters = [iters[i] for i in zi]
    z_train = [train_losses[i] for i in zi]
    z_val = [val_losses[i] for i in zi]
    ax_zoom.plot(z_iters, z_train, "o-", label="train loss", color="tab:blue")
    ax_zoom.plot(z_iters, z_val, "s-", label="val loss", color="tab:orange")
    ax_zoom.annotate(
        f"{z_train[-1]:.3f}", (z_iters[-1], z_train[-1]),
        textcoords="offset points", xytext=(10, -12), fontsize=9, color="tab:blue",
    )
    ax_zoom.annotate(
        f"{z_val[-1]:.3f}", (z_iters[-1], z_val[-1]),
        textcoords="offset points", xytext=(10, 6), fontsize=9, color="tab:orange",
    )
    ax_zoom.set_xlabel(f"Iteration (batch {data['batch_size']}, ctx {data['context_length']})")
    ax_zoom.set_ylabel("Cross-entropy loss")
    ax_zoom.set_title("Zoom: saved checkpoints 20k–100k", fontsize=10)
    ax_zoom.legend()
    ax_zoom.grid(alpha=0.3)

    fig.text(
        0.01, 0.005,
        f"Measured from artifacts/checkpoints on artifacts/{{train,val}}.npy, "
        f"{data['n_batches']} batches each, seed {data['seed']}",
        fontsize=7, color="gray",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = REPO / "loss_curve.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
