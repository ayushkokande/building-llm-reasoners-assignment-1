# train_bpe_tinystories.py
import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DATASET_REL = Path("data") / "TinyStoriesV2-GPT4-train.txt"
if (_REPO_ROOT / _DATASET_REL).exists():
    TINYSTORIES_PATH = _REPO_ROOT / _DATASET_REL
else:
    TINYSTORIES_PATH = _REPO_ROOT.parent / _DATASET_REL
OUTPUT_DIR = _SCRIPT_DIR / "bpe_output"

vocab_size = 10_000
special_tokens = ["<|endoftext|>"]


if __name__ == "__main__":
    from tests.adapters import run_train_bpe

    parser = argparse.ArgumentParser(description="Train BPE on TinyStories")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run under cProfile and save/print a time profile",
    )
    parser.add_argument(
        "--profile-sort",
        default="cumulative",
        choices=["cumulative", "tottime", "calls"],
        help="Sort key for profile report (default: cumulative)",
    )
    parser.add_argument(
        "--profile-lines",
        type=int,
        default=40,
        help="Number of lines in the printed profile (default: 40)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Starting BPE training on TinyStories...")
    start_time = time.time()

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()

    vocab, merges = run_train_bpe(
        input_path=TINYSTORIES_PATH,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=8,
    )

    if args.profile:
        prof.disable()
        profile_path = OUTPUT_DIR / "tinystories_train.prof"
        prof.dump_stats(str(profile_path))
        print(f"\nProfile saved to {profile_path}")
        print(f"Top {args.profile_lines} by {args.profile_sort} time:\n")
        ps = pstats.Stats(prof, stream=sys.stdout)
        ps.sort_stats(args.profile_sort)
        ps.print_stats(args.profile_lines)

    end_time = time.time()
    training_time_minutes = (end_time - start_time) / 60
    training_time_hours = training_time_minutes / 60

    print(f"Training completed in {training_time_minutes:.2f} minutes ({training_time_hours:.5f} hours)")

    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 2**20:
            rss_bytes = rss  # assume bytes (e.g. macOS)
        else:
            rss_bytes = rss * 1024  # assume KB (e.g. Linux)
        peak_gb = rss_bytes / (1024**3)
        if peak_gb >= 1.0:
            print(f"Peak memory: {peak_gb:.2f} GB")
        else:
            print(f"Peak memory: {rss_bytes / (1024**2):.2f} MB")
    except Exception:
        print("Peak memory: (use Activity Monitor or top to measure)")

    vocab_output = OUTPUT_DIR / "tinystories_vocab.json"
    merges_output = OUTPUT_DIR / "tinystories_merges.txt"

    vocab_json = {str(k): v.hex() for k, v in vocab.items()}
    with open(vocab_output, "w") as f:
        json.dump(vocab_json, f, indent=2)

    with open(merges_output, "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

    print(f"Saved vocab to {vocab_output}")
    print(f"Saved merges to {merges_output}")

    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_id = next(k for k, v in vocab.items() if v == longest_token_bytes)
    longest_token_length = len(longest_token_bytes)

    print(f"\nLongest token in the vocabulary:")
    print(f"  Length: {longest_token_length} bytes")
    try:
        decoded = longest_token_bytes.decode("utf-8", errors="replace")
        print(f"  Decoded (UTF-8): {repr(decoded)}")
    except Exception:
        decoded = None
        print("  (Not valid UTF-8)")

    if decoded:
        print(f"  Here it decodes to: {decoded!r}")

    print(f"\nSummary:")
    print(f"  Vocab size: {len(vocab)} entries")
    print(f"  Merges count: {len(merges)}")