#!/usr/bin/env python3
"""
Training script for the Transformer language model.

Supports:
- Configurable model and optimizer hyperparameters (CLI / config)
- Memory-efficient train/val loading with np.memmap
- Checkpoint serialization to a user-provided path
- Periodic logging to console and optional Weights & Biases (wandb)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from student.assignment1 import (
    get_adamw_cls,
    run_cross_entropy,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_save_checkpoint,
)
from student.lm import TransformerLM


def load_dataset_memmap(
    path: str | Path,
    dtype: npt.DTypeLike = np.uint32,
) -> npt.NDArray:
    """
    Load a 1D array of token IDs using memory-mapping for large datasets.
    - .npy: loaded with np.load(..., mmap_mode="r"), then flattened to 1D.
    - .bin / .dat: raw binary array of dtype, shape (file_size // dtype.itemsize,).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        return arr.ravel()
    if path.suffix in (".bin", ".dat", ""):
        nbytes = path.stat().st_size
        dt = np.dtype(dtype)
        n = nbytes // dt.itemsize
        return np.memmap(path, dtype=dt, mode="r", shape=(n,))
    raise ValueError(f"Unsupported dataset format (use .npy or .bin): {path}")


def prepare_dataset_for_memmap(
    data_path: str | Path,
    output_path: str | Path,
    dtype: npt.DTypeLike = np.uint32,
) -> None:
    """
    If your data is a plain text or token list, convert to a binary file for memmap.
    If data_path is already .npy, copy to output_path as .npy for consistency.
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    if data_path.suffix == ".npy":
        arr = np.load(data_path)
        np.save(output_path, arr, allow_pickle=False)
        return
    # Assume one token ID per line or a single line of space-separated IDs
    with open(data_path) as f:
        content = f.read()
    tokens = [int(x) for x in content.split() if x.strip()]
    arr = np.array(tokens, dtype=dtype)
    np.save(output_path, arr, allow_pickle=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer LM with memmap data and checkpointing.")
    # Data
    p.add_argument("--train_data", type=str, required=True, help="Path to training data (.npy or .bin)")
    p.add_argument("--val_data", type=str, default=None, help="Path to validation data (.npy or .bin)")
    p.add_argument("--data_dtype", type=str, default="uint32", choices=["uint16", "uint32", "int32"],
                   help="Dtype for memmap array (for .bin files)")
    # Model
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=128)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--eps", type=float, default=1e-8)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=10_000)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--cosine_cycle_iters", type=int, default=10_000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--checkpoint_every", type=int, default=1000)    # Logging
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--wandb_project", type=str, default=None, help="If set, enable Weights & Biases logging")
    p.add_argument("--wandb_run_name", type=str, default=None)
    # Device
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    dtype_map = {"uint16": np.uint16, "uint32": np.uint32, "int32": np.int32}
    data_dtype = dtype_map.get(args.data_dtype, np.uint32)

    # Memory-efficient data loading with np.memmap
    train_ds = load_dataset_memmap(args.train_data, dtype=data_dtype)
    val_ds = None
    if args.val_data:
        val_ds = load_dataset_memmap(args.val_data, dtype=data_dtype)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    AdamWCls = get_adamw_cls()
    optimizer = AdamWCls(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        eps=args.eps,
    )

    start_iter = 0

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_project:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as e:
            print(f"wandb init failed: {e}, continuing without wandb")
            args.wandb_project = None

    model.train()
    for it in range(start_iter, args.max_iters):
        lr = run_get_lr_cosine_schedule(
            it,
            max_learning_rate=args.lr,
            min_learning_rate=args.lr * 0.1,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = run_get_batch(
            dataset=train_ds,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )
        optimizer.zero_grad()
        logits = model(x)
        # Flatten batch and sequence for cross_entropy: (B*T, V), (B*T)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = y.reshape(-1)
        loss = run_cross_entropy(logits_flat, targets_flat)
        loss.backward()
        run_gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if it % args.log_every == 0:
            msg = f"iter {it} loss {loss.item():.4f} lr {lr:.2e}"
            print(msg)
            if args.wandb_project:
                try:
                    wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=it)
                except Exception:
                    pass

        if val_ds is not None and it > 0 and it % args.val_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(50):
                    xv, yv = run_get_batch(val_ds, args.batch_size, args.context_length, device)
                    logits_v = model(xv)
                    lv = run_cross_entropy(
                        logits_v.reshape(-1, logits_v.size(-1)),
                        yv.reshape(-1),
                    )
                    val_losses.append(lv.item())
            model.train()
            val_loss = sum(val_losses) / len(val_losses)
            print(f"iter {it} val_loss {val_loss:.4f}")
            if args.wandb_project:
                try:
                    wandb.log({"val/loss": val_loss}, step=it)
                except Exception:
                    pass

        if (it + 1) % args.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_{it + 1}.pt"
            run_save_checkpoint(model, optimizer, it + 1, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
