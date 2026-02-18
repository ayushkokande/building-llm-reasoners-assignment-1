from __future__ import annotations

from typing import Iterable
from pathlib import Path
import json
import regex as re
from student.regexsplitter import RegexSplitter


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    Byte-level BPE tokenizer 
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

        if self.special_tokens:
            next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for st in self.special_tokens:
                b = st.encode("utf-8")
                if b not in self._bytes_to_id:
                    self.vocab[next_id] = b
                    self._bytes_to_id[b] = next_id
                    next_id += 1

        self._token_re = re.compile(PAT)
        self._merge_rank: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}

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
        Uses priority-based merging: repeatedly merge the lowest-rank pair.
        """
        tokens: list[bytes] = [bytes([b]) for b in word_bytes]
        if len(tokens) <= 1:
            return [self._bytes_to_id[tok] for tok in tokens]

        rank = self._merge_rank

        while len(tokens) > 1:
            best_idx = -1
            best_rank = len(self.merges)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                r = rank.get(pair)
                if r is not None and r < best_rank:
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1]
            del tokens[best_idx + 1]

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
                piece_bytes = piece.encode("utf-8") #
                out.extend(self._bpe_merge_helper(piece_bytes))

        return out

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into text(UTF-8 string)
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

