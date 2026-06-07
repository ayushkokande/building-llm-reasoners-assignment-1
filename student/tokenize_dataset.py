#!/usr/bin/env python3
"""
Encode a raw text file into a flat array of token IDs using a trained BPE
tokenizer, and save it as a .npy file that `student/train.py` can memmap.

Encoding is parallelized: the file is split on the special-token boundary
(`<|endoftext|>`) into N chunks, each chunk is encoded in its own process,
and the per-chunk ID arrays are concatenated in original order.

Example:
    uv run student/tokenize_dataset.py \
        --input_path data/TinyStoriesV2-GPT4-train.txt \
        --vocab_json artifacts/vocab.json \
        --merges_txt artifacts/merges.txt \
        --output_path artifacts/train.npy \
        --num_processes 8
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from student.pretokenization_example import find_chunk_boundaries
from student.tokenizer import Tokenizer

# Globals set per worker via initializer (avoids pickling the tokenizer per task).
_TOKENIZER: Tokenizer | None = None
_INPUT_PATH: str | None = None


def _init_worker(vocab_json: str, merges_txt: str, special_tokens: list[str], input_path: str) -> None:
    global _TOKENIZER, _INPUT_PATH
    _TOKENIZER = Tokenizer.from_files(
        vocab_filepath=vocab_json,
        merges_filepath=merges_txt,
        special_tokens=special_tokens,
    )
    _INPUT_PATH = input_path


def _encode_chunk(span: tuple[int, int, int]) -> tuple[int, np.ndarray]:
    """Encode bytes [start, end) of the input file. Returns (order_index, ids)."""
    order, start, end = span
    assert _TOKENIZER is not None and _INPUT_PATH is not None
    with open(_INPUT_PATH, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    ids = _TOKENIZER.encode(text)
    return order, np.asarray(ids, dtype=np.uint32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenize a text file into a .npy token-id array (parallel).")
    p.add_argument("--input_path", type=str, required=True, help="Raw text file to encode")
    p.add_argument("--vocab_json", type=str, required=True, help="vocab.json from train_bpe.py")
    p.add_argument("--merges_txt", type=str, required=True, help="merges.txt from train_bpe.py")
    p.add_argument("--output_path", type=str, required=True, help="Destination .npy file of token IDs")
    p.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens (must match training); first is also the chunk split token",
    )
    p.add_argument("--num_processes", type=int, default=8, help="Worker processes / chunk count")
    p.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32"],
        help="Output dtype. uint16 ok if vocab_size <= 65535 (smaller files).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_token = args.special_tokens[0].encode("utf-8")
    with open(args.input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_processes, split_token)

    spans = [
        (i, start, end)
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    print(f"Encoding {args.input_path} in {len(spans)} chunk(s) across {args.num_processes} proc(s)...")

    t0 = time.time()
    with Pool(
        processes=args.num_processes,
        initializer=_init_worker,
        initargs=(args.vocab_json, args.merges_txt, args.special_tokens, args.input_path),
    ) as pool:
        results = pool.map(_encode_chunk, spans)

    results.sort(key=lambda r: r[0])
    all_ids = np.concatenate([ids for _, ids in results]) if results else np.zeros(0, dtype=np.uint32)

    out_dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    if args.dtype == "uint16" and all_ids.size and int(all_ids.max()) > 65535:
        raise ValueError("Token IDs exceed uint16 range; rerun with --dtype uint32")
    all_ids = all_ids.astype(out_dtype)

    np.save(out_path, all_ids, allow_pickle=False)
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s. tokens={all_ids.size:,} dtype={args.dtype}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
