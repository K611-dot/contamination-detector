"""N-gram overlap contamination detector.

Checks whether a benchmark example's n-grams appear in a reference corpus.
This mirrors the contamination check GPT-3 and later papers used: an
example is flagged as likely-contaminated if a large fraction of its
n-grams (default n=13, following the GPT-3 paper's convention) already
exist somewhere in the corpus.

No model access is required — this only needs the benchmark text and a
corpus of documents to check against (e.g. a pretraining data sample, a
web crawl, or a suspect dataset).

Two signals are reported per example, and they answer different
questions:

- `overlap_fraction` — how much of the example appears in the corpus at
  all. Sensitive, but scattered matches on common phrasing inflate it.
- `longest_match_tokens` — the longest *contiguous* run of example tokens
  found verbatim in the corpus. Far more diagnostic: a 40-word verbatim
  run is close to conclusive, while the same overlap fraction spread over
  disconnected n-grams usually is not.

Prefer the longest run when deciding whether a single example is leaked;
use the fraction when comparing examples against each other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Deliberately simple and dependency-free."""
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """The set of distinct n-grams in `tokens`. Empty if the text is too short."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_hashes(tokens: list[str], n: int) -> list[int]:
    """Hashes of every n-gram, in positional order (so runs stay detectable).

    Hashing rather than keeping the string tuples cuts index memory about
    2.5x (measured: 77.6 MB -> 31.0 MB for 400k 13-grams), which is what
    decides whether a large corpus fits in RAM at all.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return []
    return [hash(tuple(tokens[i : i + n])) for i in range(len(tokens) - n + 1)]


@dataclass
class OverlapResult:
    """Per-example overlap signals.

    overlap_fraction is NaN when the example is shorter than the n-gram
    size — such an example cannot be scored, which is different from
    being clean, and NaN keeps the two apart in downstream statistics.
    """

    example_id: str
    overlap_fraction: float
    matched_ngrams: int
    total_ngrams: int
    longest_match_tokens: int = 0

    @property
    def scorable(self) -> bool:
        return self.total_ngrams > 0


class CorpusIndex:
    """Pre-indexes a reference corpus's n-grams for fast repeated lookups.

    Documents can be supplied as any iterable — including a generator that
    reads files lazily — so a corpus larger than memory can be streamed in
    one document at a time.

    Note: the index holds 64-bit hashes, so a collision could in principle
    register a spurious match. At 10^9 indexed n-grams the expected number
    of collisions is well under one, and an isolated false match cannot
    produce a long contiguous run, which is the signal that actually
    drives a contamination call.
    """

    def __init__(self, documents: Iterable[str], n: int = 13):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self._hashes: set[int] = set()
        for doc in documents:
            # update() from a generator avoids materialising a second full
            # set per document before merging it in.
            self._hashes.update(ngram_hashes(tokenize(doc), n))

    def __len__(self) -> int:
        return len(self._hashes)

    def __contains__(self, ngram: tuple[str, ...]) -> bool:
        return hash(ngram) in self._hashes

    def overlap_score(self, example_text: str, example_id: str = "") -> OverlapResult:
        hashes = ngram_hashes(tokenize(example_text), self.n)
        if not hashes:
            return OverlapResult(
                example_id=example_id,
                overlap_fraction=float("nan"),
                matched_ngrams=0,
                total_ngrams=0,
                longest_match_tokens=0,
            )

        matched_flags = [h in self._hashes for h in hashes]

        # Fraction is computed over *distinct* n-grams so that a repeated
        # boilerplate phrase cannot inflate an example's score.
        distinct = set(hashes)
        distinct_matched = sum(1 for h in distinct if h in self._hashes)

        # Longest run of consecutive matching n-grams. A run of r adjacent
        # matching n-grams covers r + n - 1 contiguous tokens.
        longest_run = 0
        current_run = 0
        for flag in matched_flags:
            current_run = current_run + 1 if flag else 0
            longest_run = max(longest_run, current_run)
        longest_tokens = longest_run + self.n - 1 if longest_run else 0

        return OverlapResult(
            example_id=example_id,
            overlap_fraction=distinct_matched / len(distinct),
            matched_ngrams=distinct_matched,
            total_ngrams=len(distinct),
            longest_match_tokens=longest_tokens,
        )


def batch_overlap_scores(
    examples: dict[str, str], corpus_documents: Iterable[str], n: int = 13
) -> list[OverlapResult]:
    """Score every example in `examples` (id -> text) against the corpus."""
    index = CorpusIndex(corpus_documents, n=n)
    return [index.overlap_score(text, example_id=eid) for eid, text in examples.items()]


@dataclass
class MultiSourceResult:
    """An example scored against each source separately.

    Collapsing every source into one "contaminated" verdict throws away the
    part that determines what to do about it. A match against a training
    split and a match against a held-out test split are different findings
    with different remedies, and a single score cannot distinguish them.
    """

    example_id: str
    per_source: dict[str, OverlapResult]

    @property
    def scorable(self) -> bool:
        return any(r.scorable for r in self.per_source.values())

    @property
    def longest_match_tokens(self) -> int:
        if not self.per_source:
            return 0
        return max(r.longest_match_tokens for r in self.per_source.values())

    @property
    def max_overlap_fraction(self) -> float:
        scorable = [r.overlap_fraction for r in self.per_source.values() if r.scorable]
        return max(scorable) if scorable else float("nan")

    @property
    def best_source(self) -> str | None:
        """Source with the longest verbatim run — where the leak most likely is."""
        matched = {
            name: r for name, r in self.per_source.items() if r.longest_match_tokens > 0
        }
        if not matched:
            return None
        return max(matched, key=lambda name: matched[name].longest_match_tokens)

    def matched_sources(self) -> list[str]:
        """Every source showing a verbatim run, longest first."""
        matched = [
            (name, r.longest_match_tokens)
            for name, r in self.per_source.items()
            if r.longest_match_tokens > 0
        ]
        matched.sort(key=lambda pair: pair[1], reverse=True)
        return [name for name, _ in matched]


class MultiSourceIndex:
    """Indexes several named sources separately and keeps match provenance.

    Use this rather than concatenating a corpus when the sources differ in
    kind — a train split versus a test split, scraped web text versus
    curated prose. Two reasons it matters:

    - **Provenance.** You learn *which* source an example leaked from, which
      is what decides the remedy.
    - **Calibration.** Innocent overlap is not stationary across sources.
      Formulaic or boilerplate-heavy text throws near-duplicate n-grams for
      reasons unrelated to leakage; prose does not. One global threshold
      over-flags the former and under-flags the latter. See
      `contamination_detector.calibration` for per-source thresholds.
    """

    def __init__(self, sources: dict[str, Iterable[str]], n: int = 13):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.indexes: dict[str, CorpusIndex] = {
            name: CorpusIndex(docs, n=n) for name, docs in sources.items()
        }

    @property
    def source_names(self) -> list[str]:
        return list(self.indexes)

    def overlap_score(self, example_text: str, example_id: str = "") -> MultiSourceResult:
        return MultiSourceResult(
            example_id=example_id,
            per_source={
                name: index.overlap_score(example_text, example_id=example_id)
                for name, index in self.indexes.items()
            },
        )

    def batch_overlap_scores(self, examples: dict[str, str]) -> list[MultiSourceResult]:
        return [
            self.overlap_score(text, example_id=eid) for eid, text in examples.items()
        ]
