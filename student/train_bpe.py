#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer on a raw text file and serialize it.

Outputs two files in the format expected by `Tokenizer.from_files`:
- vocab.json : {token_id (str) -> token_bytes_as_hex (str)}
- merges.txt : one merge per line, "a_hex b_hex"

Example:
    uv run student/train_bpe.py \
        --input_path data/TinyStoriesV2-GPT4-train.txt \
        --vocab_size 10000 \
        --out_dir artifacts \
        --num_processes 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from student.core import run_train_bpe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    p.add_argument("--input_path", type=str, required=True, help="Path to raw training text (e.g. TinyStories train .txt)")
    p.add_argument("--vocab_size", type=int, default=10000, help="Target vocabulary size (incl. 256 byte tokens + specials)")
    p.add_argument("--out_dir", type=str, default="artifacts", help="Directory to write vocab.json and merges.txt")
    p.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to reserve in the vocab",
    )
    p.add_argument("--num_processes", type=int, default=8, help="Worker processes for pretokenization")
    return p.parse_args()


def save_vocab(vocab: dict[int, bytes], path: Path) -> None:
    serializable = {str(tid): token_bytes.hex() for tid, token_bytes in vocab.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


def save_merges(merges: list[tuple[bytes, bytes]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE: input={args.input_path} vocab_size={args.vocab_size} procs={args.num_processes}")
    t0 = time.time()
    vocab, merges = run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )
    dt = time.time() - t0

    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.txt"
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    print(f"Done in {dt:.1f}s. vocab={len(vocab)} merges={len(merges)}")
    print(f"  -> {vocab_path}")
    print(f"  -> {merges_path}")


if __name__ == "__main__":
    main()
