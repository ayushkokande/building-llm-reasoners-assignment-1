from __future__ import annotations

from typing import Dict

import regex as re


class RegexSplitter:
    """
    Helper function to split text into regex tokens, 
    while respecting special tokens that should never be merged across.

    It:
    - reads [start, end) from `filepath` in binary
    - decodes to UTF-8 text
    - splits on special tokens (so merges never cross them)
    - applies a GPT-2-style regex pattern to get pre-tokens
    - returns a dict mapping pre-token strings -> counts
    """

    def __init__(self, pat: str, special_tokens: list[str]) -> None:
        self._special_tokens = set(special_tokens)
        self._token_re = re.compile(pat)

    def split_on_special_tokens(self, text: str) -> list[str]:
        """
        Split `text` on each special token, **keeping** the special tokens
        themselves as separate elements in the result.

        Example:
            text = "[Doc1]<|endoftext|>[Doc2]"
            special_tokens = ["<|endoftext|>"]
            -> ["[Doc1]", "<|endoftext|>", "[Doc2]"]
        """
        if not self._special_tokens:
            return [text]

        pattern = f"({'|'.join(map(re.escape, self._special_tokens))})"
        return [p for p in re.split(pattern, text) if p]

    def seek_and_split(self, filepath: str, start: int, end: int) -> Dict[str, int]:
        """
        Read a byte range [start, end) from `filepath`, split by special tokens
        and the regex pattern, and return pre-token -> count mapping.
        """
        with open(filepath, "rb") as f:
            f.seek(start)
            data = f.read(end - start)

        text = data.decode("utf-8", errors="ignore")

        parts = self.split_on_special_tokens(text)

        counts: Dict[str, int] = {}
        for part in parts:
            if part in self._special_tokens:
                continue

            for m in self._token_re.finditer(part):
                s = m.group(0)
                counts[s] = counts.get(s, 0) + 1

        return counts

