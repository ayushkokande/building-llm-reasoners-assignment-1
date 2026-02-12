from __future__ import annotations

from typing import Iterable
from pathlib import Path
import json
import regex as re
from student.regexsplitter import RegexSplitter


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    Byte-level BPE tokenizer compatible with the vocab/merges format used
    in this assignment (and GPT-2-style tokenization).

    - `vocab` maps token id -> token bytes
    - `merges` is an ordered list of (left_bytes, right_bytes)
    - `special_tokens` are strings that are never split or merged
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self._bytes_to_id: dict[bytes, int] = {token_bytes: tid for tid, token_bytes in vocab.items()}

        self._token_re = re.compile(PAT)

        self._splitter = RegexSplitter(pat=PAT, special_tokens=self.special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Construct a Tokenizer from serialized vocabulary and merges files,
        in the same format as produced by the BPE training script:
        """
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)

        with vocab_path.open("r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with merges_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                a_hex, b_hex = stripped.split()
                merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _bpe_merge_helper(self, word_bytes: bytes) -> list[int]:
        """
        Apply BPE merges to a single UTF-8 byte string.
        """
        tokens: list[bytes] = [bytes([b]) for b in word_bytes]

        for a, b in self.merges:
            merged = a + b #merging two bytes to form a new byte
            new_tokens: list[bytes] = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self._bytes_to_id[tok] for tok in tokens]

    def encode(self, text: str) -> list[int]:
        """
        Encode input text into token IDs.   
        """
        out: list[int] = []
        parts = self._splitter.split_on_special_tokens(text)
        for part in parts:
            if part in self.special_tokens:
                out.append(self._bytes_to_id[part.encode("utf-8")])
                continue
            for m in self._token_re.finditer(part):
                piece = m.group(0)
                piece_bytes = piece.encode("utf-8")
                out.extend(self._bpe_merge_helper(piece_bytes))

        return out

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into text(UTF-8 string) by
        concatenating their byte values.
        """
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings, i.e. a python file handle, return
        a generator that lazily yields token IDs.
        """
        if hasattr(iterable, "read"):
            content = iterable.read()
        else:
            content = "".join(iterable)
        yield from self.encode(content)

