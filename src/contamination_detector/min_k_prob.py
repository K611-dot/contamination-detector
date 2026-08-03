"""Min-K% Prob / Min-K%++ membership inference scoring.

Implements the two closely related contamination-detection methods:

- Min-K% Prob (Shi et al., "Detecting Pretraining Data from Large
  Language Models", 2023): average the k% lowest per-token log-
  probabilities of a text under the model. Text the model has seen
  during training tends to have fewer very-low-probability tokens than
  unseen text, so a *higher* (less negative) Min-K% score suggests
  training-set membership.

- Min-K%++ (Zhang et al., 2024): normalizes each token's log-probability
  against the mean and standard deviation of the full log-probability
  distribution over the vocabulary at that position, before taking the
  lowest-k% average. This corrects for tokens that are inherently
  low-probability regardless of memorization, and is more robust.

Both are implemented against a small `TokenScore` interface rather than
a specific model, so they can be tested independently of any actual
language model and reused with any provider that can produce per-token
log-probabilities and vocabulary-level statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TokenScore:
    """Per-token statistics needed for Min-K% scoring.

    logprob: log P(token | context) under the model, i.e. the log-
        probability actually assigned to the observed token.
    mean_logprob: mean of the log-softmax distribution over the full
        vocabulary at this position.
    std_logprob: standard deviation of that same distribution.
    """

    logprob: float
    mean_logprob: float = 0.0
    std_logprob: float = 1.0


def _lowest_k_percent(values: list[float], k_percent: float) -> list[float]:
    if not values:
        return []
    k_percent = min(max(k_percent, 0.0), 100.0)
    count = max(1, round(len(values) * k_percent / 100))
    return sorted(values)[:count]


def min_k_percent(scores: list[TokenScore], k_percent: float = 20.0) -> float:
    """Average of the lowest k% token log-probabilities.

    Higher (closer to 0) implies more likely to have been seen in
    training; more negative implies more likely unseen.
    """
    logprobs = [s.logprob for s in scores]
    lowest = _lowest_k_percent(logprobs, k_percent)
    if not lowest:
        return float("nan")
    return sum(lowest) / len(lowest)


def min_k_plus_plus(scores: list[TokenScore], k_percent: float = 20.0) -> float:
    """Min-K%++: same idea, but on standardized (z-scored) log-probabilities."""
    standardized = []
    for s in scores:
        std = s.std_logprob if s.std_logprob > 1e-8 else 1e-8
        standardized.append((s.logprob - s.mean_logprob) / std)
    lowest = _lowest_k_percent(standardized, k_percent)
    if not lowest:
        return float("nan")
    return sum(lowest) / len(lowest)


def perplexity(scores: list[TokenScore]) -> float:
    """Plain perplexity, provided as a naive baseline to compare against."""
    logprobs = [s.logprob for s in scores]
    if not logprobs:
        return float("nan")
    return math.exp(-sum(logprobs) / len(logprobs))
