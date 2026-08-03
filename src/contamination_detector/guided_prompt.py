"""Guided-prompting exact-completion contamination test.

Based on the "guided prompting" technique from Golchin & Surdeanu,
"Time Travel in LLMs: Tracing Data Contamination in Large Language
Models" (2023): split a benchmark example into a prefix and a true
suffix, ask the model to continue the prefix, and measure how closely
the generated continuation matches the true suffix. A model that has
memorized the example from training data tends to reproduce the suffix
almost verbatim; a model seeing it for the first time will not.

This module only needs a `complete_fn(prefix: str) -> str` callable, so
it works with any model provider (local HF model, hosted API, etc.)
without depending on a specific one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .ngram_overlap import tokenize


def split_prefix_suffix(text: str, split_ratio: float = 0.7) -> tuple[str, str]:
    """Split text into a prefix (first `split_ratio` of words) and suffix."""
    words = text.split()
    if len(words) < 2:
        return text, ""
    cut = max(1, round(len(words) * split_ratio))
    cut = min(cut, len(words) - 1)
    return " ".join(words[:cut]), " ".join(words[cut:])


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def lcs_similarity(true_text: str, generated_text: str) -> float:
    """LCS-based similarity in [0, 1], robust to word-order-preserving noise.

    1.0 means the generated text contains the true text as a subsequence
    (or vice versa in equal length); 0.0 means no shared tokens at all.
    """
    true_tokens = tokenize(true_text)
    gen_tokens = tokenize(generated_text)
    if not true_tokens:
        return 0.0
    lcs = _lcs_length(true_tokens, gen_tokens)
    return lcs / len(true_tokens)


@dataclass
class GuidedPromptResult:
    example_id: str
    prefix: str
    true_suffix: str
    generated_suffix: str
    similarity: float


def run_guided_prompt_test(
    example_id: str,
    text: str,
    complete_fn: Callable[[str], str],
    split_ratio: float = 0.7,
) -> GuidedPromptResult:
    prefix, true_suffix = split_prefix_suffix(text, split_ratio=split_ratio)
    generated_suffix = complete_fn(prefix) if true_suffix else ""
    similarity = lcs_similarity(true_suffix, generated_suffix) if true_suffix else 0.0
    return GuidedPromptResult(
        example_id=example_id,
        prefix=prefix,
        true_suffix=true_suffix,
        generated_suffix=generated_suffix,
        similarity=similarity,
    )


def batch_guided_prompt_test(
    examples: dict[str, str],
    complete_fn: Callable[[str], str],
    split_ratio: float = 0.7,
) -> list[GuidedPromptResult]:
    return [
        run_guided_prompt_test(eid, text, complete_fn, split_ratio=split_ratio)
        for eid, text in examples.items()
    ]
