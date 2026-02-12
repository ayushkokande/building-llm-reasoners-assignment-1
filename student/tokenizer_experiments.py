from __future__ import annotations

"""
Simple utilities for running tokenizer experiments on TinyStories.
"""

import statistics
import time
from pathlib import Path

import numpy as np

from student.tokenizer import Tokenizer


def _resolve_paths() -> tuple[Path, Path, Path]:
    """Resolve repo root, TinyStories path, and TinyStories BPE artifacts."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Match the logic from tests/train_bpe_tinystories.py
    if (repo_root.parent / "data" / "TinyStoriesV2-GPT4-train.txt").exists():
        tinystories_path = repo_root.parent / "data" / "TinyStoriesV2-GPT4-train.txt"
    else:
        tinystories_path = repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"

    bpe_dir = repo_root / "tests" / "bpe_output"
    return tinystories_path, bpe_dir, repo_root


def load_tokenizer() -> Tokenizer:
    """Load the TinyStories BPE tokenizer trained with vocab_size=10_000."""
    tinystories_path, bpe_dir, _ = _resolve_paths()

    vocab_file = bpe_dir / "tinystories_vocab.json"
    merges_file = bpe_dir / "tinystories_merges.txt"

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_file),
        merges_filepath=str(merges_file),
        special_tokens=["<|endoftext|>"],
    )
    _ = tinystories_path
    return tokenizer


#sample documents from TinyStories
def sample_tinystories_documents(n_docs: int = 10) -> list[str]:
    """Sample `n_docs` documents from TinyStories, split on <|endoftext|>."""
    tinystories_path, _, _ = _resolve_paths()

    with tinystories_path.open("r", encoding="utf-8") as f:
        contents = f.read()

    raw_docs = contents.split("<|endoftext|>")
    docs = [doc.strip() for doc in raw_docs if doc.strip()]

    return docs[:n_docs]


def run_basic_experiment(n_docs: int = 10) -> None:
    """
    Encode `n_docs` sampled TinyStories documents into integer IDs and
    print simple compression statistics (bytes/token).
    """
    tokenizer = load_tokenizer() 
    docs = sample_tinystories_documents(n_docs=n_docs)

    print(f"Encoding {n_docs} TinyStories documents with TinyStories BPE tokenizer...\n")

    ratios: list[float] = []
    for i, doc in enumerate(docs, start=1):
        ids = tokenizer.encode(doc)
        n_tokens = len(ids)
        n_bytes = len(doc.encode("utf-8"))
        ratio = n_bytes / n_tokens if n_tokens > 0 else 0.0
        ratios.append(ratio)

        print(f"Document {i}:")
        print(f"  Characters: {len(doc)}")
        print(f"  Bytes:      {n_bytes}")
        print(f"  Tokens:     {n_tokens}")
        print(f"  Bytes/token: {ratio:.3f}")
        print()

    mean_ratio = statistics.fmean(ratios)
    print(f"Average bytes/token over {n_docs} docs: {mean_ratio:.3f}")


def estimate_throughput_and_pile_tokenization_time(sample_bytes: int = 10_000_000) -> None:
    """
    Estimate tokenizer throughput (bytes/second) on TinyStories and extrapolate
    how long it would take to tokenize The Pile (825 GB).

    We time encoding of roughly `sample_bytes` of TinyStories text, then scale.
    """
    tokenizer = load_tokenizer()
    tinystories_path, _, _ = _resolve_paths()

    with tinystories_path.open("rb") as f:
        data = f.read(sample_bytes)

    text = data.decode("utf-8", errors="ignore")
    n_bytes = len(data)

    t0 = time.time()
    _ = tokenizer.encode(text)
    t1 = time.time()

    elapsed = t1 - t0
    throughput_bps = n_bytes / elapsed  # bytes per second
    throughput_mb_s = throughput_bps / (1024 ** 2)

    # Pile size: 825 GB of text
    pile_bytes = 825 * (1024 ** 3)
    pile_seconds = pile_bytes / throughput_bps
    pile_hours = pile_seconds / 3600
    pile_days = pile_hours / 24

    print("\nThroughput experiment:")
    print(f"  Sample size: {n_bytes} bytes ({n_bytes / (1024 ** 2):.2f} MiB)")
    print(f"  Elapsed:     {elapsed:.3f} s")
    print(f"  Throughput:  {throughput_mb_s:.2f} MiB/s")
    print(f"\nEstimated time to tokenize The Pile (825 GiB):")
    print(f"  ~{pile_hours:.1f} hours (~{pile_days:.2f} days)")

def encode_training_and_development_data(): 
    """
    Encode TinyStories train and dev sets into sequences of token IDs and
    serialize them as NumPy arrays of dtype uint16.

    - Train: TinyStoriesV2-GPT4-train.txt
    - Dev:   TinyStoriesV2-GPT4-valid.txt
    - Output: data/tinystories_train_ids.npy, data/tinystories_valid_ids.npy
    """
    tokenizer = load_tokenizer()

    # Reconstruct repo_root and data paths (mirrors _resolve_paths logic).
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if (repo_root.parent / "data" / "TinyStoriesV2-GPT4-train.txt").exists():
        train_path = repo_root.parent / "data" / "TinyStoriesV2-GPT4-train.txt"
        valid_path = repo_root.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
        out_dir = repo_root.parent / "data"
    else:
        train_path = repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"
        valid_path = repo_root / "data" / "TinyStoriesV2-GPT4-valid.txt"
        out_dir = repo_root / "data"

    out_dir.mkdir(exist_ok=True)

    def _encode_and_save(src: Path, dst_name: str) -> None:
        with src.open("r", encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text)
        ids_array = np.array(ids, dtype=np.uint16)
        np.save(out_dir / dst_name, ids_array)

    _encode_and_save(train_path, "tinystories_train_ids.npy")
    _encode_and_save(valid_path, "tinystories_valid_ids.npy")


if __name__ == "__main__":
    #run_basic_experiment(n_docs=10)
    #estimate_throughput_and_pile_tokenization_time()
    encode_training_and_development_data()

