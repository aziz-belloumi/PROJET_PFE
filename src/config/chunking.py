# src/config/chunking.py
"""
Shared token-based chunking utility used by NER, Sentiment, and Topic model classes.
Splits a text into overlapping chunks that respect a tokenizer's max token length.
"""

from __future__ import annotations

from typing import List, Tuple


def token_chunks(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap: int,
) -> List[Tuple[str, int]]:
    """
    Split `text` into overlapping token-based chunks.

    Args:
        text:       Raw input string.
        tokenizer:  A HuggingFace tokenizer with `return_offsets_mapping` support.
        max_tokens: Maximum number of tokens per chunk (excluding special tokens).
        overlap:    Number of tokens to overlap between consecutive chunks.

    Returns:
        List of (chunk_text, start_char_offset) tuples.
        Returns an empty list if the text produces no tokens.
    """
    if not text:
        return []

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )

    input_ids: list = enc.get("input_ids", [])
    offsets: List[Tuple[int, int]] = enc.get("offset_mapping", [])

    if not input_ids or not offsets:
        return []

    chunks: List[Tuple[str, int]] = []
    i = 0
    n = len(input_ids)

    while i < n:
        j = min(i + max_tokens, n)
        start_char = offsets[i][0]
        end_char = offsets[j - 1][1]

        if end_char <= start_char:
            i = j
            continue

        chunks.append((text[start_char:end_char], start_char))

        if j == n:
            break

        i = max(0, j - overlap)

    return chunks
