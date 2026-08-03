"""N-gram overlap contamination detector.

Checks whether a benchmark example's n-grams appear in a reference corpus.
This mirrors the contamination check GPT-3 and later papers used: an
example is flagged as likely-contaminated if a large fraction of its
n-grams (default n=13, following the GPT-3 paper's convention) already
exist somewhere in the corpus.

No model access is required — this only needs the benchmark text and a
corpus of documents to check against (e.g. a pretraining data sample, a
web crawl, or a suspect dataset).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Deliberately simple and dependency-free."""
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class OverlapResult:
    example_id: str
    overlap_fraction: float
    matched_ngrams: int
    total_ngrams: int


class CorpusIndex:
    """Pre-indexes a reference corpus's n-grams for fast repeated lookups."""

    def __init__(self, documents: list[str], n: int = 13):
        self.n = n
        self._ngram_set: set[tuple[str, ...]] = set()
        for doc in documents:
            self._ngram_set |= ngrams(tokenize(doc), n)

    def __len__(self) -> int:
        return len(self._ngram_set)

    def overlap_score(self, example_text: str, example_id: str = "") -> OverlapResult:
        example_ngrams = ngrams(tokenize(example_text), self.n)
        if not example_ngrams:
            return OverlapResult(example_id, 0.0, 0, 0)
        matched = len(example_ngrams & self._ngram_set)
        return OverlapResult(
            example_id=example_id,
            overlap_fraction=matched / len(example_ngrams),
            matched_ngrams=matched,
            total_ngrams=len(example_ngrams),
        )


def batch_overlap_scores(
    examples: dict[str, str], corpus_documents: list[str], n: int = 13
) -> list[OverlapResult]:
    """Score every example in `examples` (id -> text) against the corpus."""
    index = CorpusIndex(corpus_documents, n=n)
    return [index.overlap_score(text, example_id=eid) for eid, text in examples.items()]
