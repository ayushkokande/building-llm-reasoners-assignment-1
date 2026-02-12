"""
Regenerate transformer LM snapshots for the current platform.

Run from project root: uv run python -m tests.update_transformer_lm_snapshots

Use when tests fail due to small numerical differences across platforms
(CPU arch, PyTorch/NumPy version, Python version).
"""
import json
from pathlib import Path

import numpy as np
import torch

from tests.adapters import run_transformer_lm
from tests.common import FIXTURES_PATH


def main():
    path = FIXTURES_PATH / "ts_tests"
    sd = torch.load(path / "model.pt", map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    cfg = json.load(open(path / "model_config.json"))

    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    d_model = cfg["d_model"]
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]
    d_ff = cfg["d_ff"]
    rope_theta = cfg["rope_theta"]

    torch.manual_seed(6)
    in_indices = torch.randint(0, 10_000, (4, 12))

    out = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        weights=sd,
        in_indices=in_indices,
    )
    snapshot_dir = Path(__file__).parent / "_snapshots"
    np.savez(snapshot_dir / "test_transformer_lm.npz", array=out.detach().cpu().numpy())
    print("Updated test_transformer_lm.npz")

    in_indices_truncated = in_indices[..., :6]
    out_trunc = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        weights=sd,
        in_indices=in_indices_truncated,
    )
    np.savez(
        snapshot_dir / "test_transformer_lm_truncated_input.npz",
        array=out_trunc.detach().cpu().numpy(),
    )
    print("Updated test_transformer_lm_truncated_input.npz")


if __name__ == "__main__":
    main()
