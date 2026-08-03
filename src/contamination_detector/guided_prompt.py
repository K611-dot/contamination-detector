"""Guided-prompting exact-completion contamination test.

Based on the "guided prompting" technique from Golchin & Surdeanu,
"Time Travel in LLMs: Tracing Data Contamination in Large Language
Models" (2023): split a benchmark example into a prefix and a true
suffix, ask the model to continue the prefix, and measure how closely
the generated continuation matches the true suffix. A model that has
memorized the example from training data tends to reproduce the suffix
almost verbatim; a model seeing it for the first time will not.

Similarity is reported as ROUGE-L style precision, recall and F1 over the
longest common subsequence. Recall alone is not safe to threshold on: a
verbose model that emits several hundred generic words will often contain
a short true suffix as a subsequence purely by chance and score a perfect
1.0. Precision penalises that padding, so **F1 is the score to use** and
is what `similarity` returns.

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
    # Iterate with the shorter sequence on the inner axis so the rolling
    # row stays as small as possible.
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        a_i = a[i - 1]
        for j in range(1, len(b) + 1):
            if a_i == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                prev_j, curr_j1 = prev[j], curr[j - 1]
                curr[j] = prev_j if prev_j > curr_j1 else curr_j1
        prev = curr
    return prev[len(b)]


@dataclass
class SimilarityScores:
    """ROUGE-L style scores. Use `f1` for thresholding."""

    precision: float
    recall: float
    f1: float


def lcs_scores(true_text: str, generated_text: str) -> SimilarityScores:
    """LCS precision/recall/F1 between the true suffix and a generated one.

    recall    = LCS / len(true)      — how much of the truth was reproduced
    precision = LCS / len(generated) — how much of the output was on target
    """
    true_tokens = tokenize(true_text)
    gen_tokens = tokenize(generated_text)
    if not true_tokens or not gen_tokens:
        return SimilarityScores(0.0, 0.0, 0.0)

    lcs = _lcs_length(true_tokens, gen_tokens)
    if lcs == 0:
        return SimilarityScores(0.0, 0.0, 0.0)

    precision = lcs / len(gen_tokens)
    recall = lcs / len(true_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return SimilarityScores(precision=precision, recall=recall, f1=f1)


def lcs_similarity(true_text: str, generated_text: str) -> float:
    """LCS F1 in [0, 1]. 1.0 means the continuation matched the suffix exactly."""
    return lcs_scores(true_text, generated_text).f1


@dataclass
class GuidedPromptResult:
    example_id: str
    prefix: str
    true_suffix: str
    generated_suffix: str
    similarity: float  # LCS F1
    precision: float
    recall: float


def run_guided_prompt_test(
    example_id: str,
    text: str,
    complete_fn: Callable[[str], str],
    split_ratio: float = 0.7,
) -> GuidedPromptResult:
    prefix, true_suffix = split_prefix_suffix(text, split_ratio=split_ratio)
    generated_suffix = complete_fn(prefix) if true_suffix else ""
    scores = (
        lcs_scores(true_suffix, generated_suffix)
        if true_suffix
        else SimilarityScores(0.0, 0.0, 0.0)
    )
    return GuidedPromptResult(
        example_id=example_id,
        prefix=prefix,
        true_suffix=true_suffix,
        generated_suffix=generated_suffix,
        similarity=scores.f1,
        precision=scores.precision,
        recall=scores.recall,
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
