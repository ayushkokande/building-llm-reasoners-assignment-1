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
    throughput_bps = n_bytes / elapsed  
    throughput_mb_s = throughput_bps / (1024 ** 2)

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
    print("Tokenizer loaded")
    print("Starting to encode training and development data...")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_root = (repo_root.parent / "data") if (repo_root.parent / "data").exists() else (repo_root / "data")

    train_path = data_root / "TinyStoriesV2-GPT4-train.txt"
    valid_path = data_root / "TinyStoriesV2-GPT4-valid.txt"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            "TinyStories text files not found. Expected:\n"
            f"- {train_path}\n"
            f"- {valid_path}\n"
            "Download them as described in README.md."
        )

    out_dir = data_root
    out_dir.mkdir(exist_ok=True)

    end_token = "<|endoftext|>"
    end_id = int(getattr(tokenizer, "_bytes_to_id")[end_token.encode("utf-8")])

    def _iter_docs(fp, delimiter: str, chunk_chars: int):
        buf = ""
        while True:
            chunk = fp.read(chunk_chars)
            if not chunk:
                break
            buf += chunk
            while True:
                idx = buf.find(delimiter)
                if idx < 0:
                    break
                doc = buf[:idx]
                yield doc, True
                buf = buf[idx + len(delimiter) :]
        if buf:
            yield buf, False

    def _count_ids(src: Path, chunk_chars: int) -> int:
        total = 0
        with src.open("r", encoding="utf-8", errors="ignore") as f:
            for doc, had_delim in _iter_docs(f, end_token, chunk_chars=chunk_chars):
                if doc:
                    total += len(tokenizer.encode(doc))
                if had_delim:
                    total += 1
        return total

    def _write_ids(src: Path, dst: Path, n_tokens: int, chunk_chars: int) -> None:
        arr = np.lib.format.open_memmap(dst, mode="w+", dtype=np.uint16, shape=(n_tokens,))
        pos = 0
        with src.open("r", encoding="utf-8", errors="ignore") as f:
            for doc, had_delim in _iter_docs(f, end_token, chunk_chars=chunk_chars):
                if doc:
                    ids = tokenizer.encode(doc)
                    n = len(ids)
                    arr[pos : pos + n] = np.asarray(ids, dtype=np.uint16)
                    pos += n
                if had_delim:
                    arr[pos] = end_id
                    pos += 1
        if pos != n_tokens:
            raise RuntimeError(f"Token count mismatch for {src}: expected {n_tokens}, wrote {pos}")

    chunk_chars = 8_000_000
    train_out = out_dir / "tinystories_train_ids.npy"
    valid_out = out_dir / "tinystories_valid_ids.npy"

    print("Counting train tokens...")
    n_train = _count_ids(train_path, chunk_chars=chunk_chars)
    print(f"train tokens: {n_train:,}")

    print("Counting valid tokens...")
    n_valid = _count_ids(valid_path, chunk_chars=chunk_chars)
    print(f"valid tokens: {n_valid:,}")

    print(f"Writing {train_out} ...")
    _write_ids(train_path, train_out, n_tokens=n_train, chunk_chars=chunk_chars)

    print(f"Writing {valid_out} ...")
    _write_ids(valid_path, valid_out, n_tokens=n_valid, chunk_chars=chunk_chars)

    for name, path in [("train", train_out), ("valid", valid_out)]:
        x = np.load(path, mmap_mode="r").ravel()
        s = np.asarray(x[:1_000_000], dtype=np.int64)
        uniq = len(np.unique(s))
        print(f"{name} sanity: shape={tuple(x.shape)} dtype={x.dtype} min={int(x.min())} max={int(x.max())} sample_unique={uniq}")


if __name__ == "__main__":
    encode_training_and_development_data()

