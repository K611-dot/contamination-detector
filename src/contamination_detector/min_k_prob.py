"""Min-K% Prob / Min-K%++ membership inference scoring.

Implements the two closely related contamination-detection methods:

- Min-K% Prob (Shi et al., "Detecting Pretraining Data from Large
  Language Models", 2023): average the k% lowest per-token log-
  probabilities of a text under the model. Text the model has seen
  during training tends to have fewer very-low-probability tokens than
  unseen text, so a *higher* (less negative) Min-K% score suggests
  training-set membership.

- Min-K%++ (Zhang et al., 2024): calibrates each token's log-probability
  against the model's *own* next-token distribution before taking the
  lowest-k% average:

      score(x_t) = (log p(x_t | x_<t) - mu) / sigma
      mu    = E_{z ~ p(.|x_<t)}[ log p(z | x_<t) ]        (= -entropy)
      sigma = sqrt( E_{z ~ p(.|x_<t)}[ (log p(z|x_<t) - mu)^2 ] )

  Note that mu and sigma are expectations **under the model's own
  probability distribution**, not unweighted averages over the vocabulary
  vector. This distinction matters a great deal: an unweighted mean is
  dominated by the enormous tail of near-zero-probability tokens, which
  pushes mu far below the model's actual expectation and under-disperses
  sigma. The resulting scores lose their sign and most of their
  discriminative power. `TokenScore.from_log_distribution` computes these
  correctly; prefer it over populating the fields by hand.

Both methods are implemented against the small `TokenScore` interface
rather than a specific model, so they can be tested independently of any
language model and reused with any provider that can produce per-token
log-probabilities.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TokenScore:
    """Per-token statistics needed for Min-K% scoring.

    logprob: log p(x_t | x_<t), the log-probability the model assigned to
        the token that was actually observed.
    expected_logprob: mu, the expected log-probability under the model's
        own next-token distribution at this position (equals the negative
        entropy of that distribution).
    logprob_std: sigma, the standard deviation of the log-probability
        under that same distribution.

    The defaults (mu=0, sigma=1) make Min-K%++ degenerate to plain
    Min-K%, which is what you get if a provider cannot supply full
    distribution statistics.
    """

    logprob: float
    expected_logprob: float = 0.0
    logprob_std: float = 1.0

    @classmethod
    def from_log_distribution(
        cls, log_probs: Sequence[float], observed_token_id: int
    ) -> "TokenScore":
        """Build a TokenScore from a full log-softmax vector over the vocabulary.

        Computes mu and sigma as expectations under the distribution
        itself, per the Min-K%++ definition. Use this rather than filling
        the fields manually — it is the single place the calibration
        formula lives.
        """
        mu = 0.0
        for lp in log_probs:
            mu += math.exp(lp) * lp
        variance = 0.0
        for lp in log_probs:
            variance += math.exp(lp) * (lp - mu) ** 2
        return cls(
            logprob=log_probs[observed_token_id],
            expected_logprob=mu,
            logprob_std=math.sqrt(max(variance, 0.0)),
        )


def _mean_of_lowest_k(values: list[float], k_percent: float) -> float:
    """Average the lowest k% of `values`. Returns NaN for an empty input."""
    if not values:
        return float("nan")
    k_percent = min(max(k_percent, 0.0), 100.0)
    count = max(1, round(len(values) * k_percent / 100))
    # nsmallest is O(len * log count), cheaper than a full sort when the
    # tail is small, which is the usual case (k is typically 10-20%).
    lowest = values if count >= len(values) else heapq.nsmallest(count, values)
    return sum(lowest) / len(lowest)


def min_k_percent(scores: Sequence[TokenScore], k_percent: float = 20.0) -> float:
    """Average of the lowest k% token log-probabilities.

    Higher (closer to 0) implies more likely to have been seen in
    training; more negative implies more likely unseen.
    """
    return _mean_of_lowest_k([s.logprob for s in scores], k_percent)


def min_k_plus_plus(scores: Sequence[TokenScore], k_percent: float = 20.0) -> float:
    """Min-K%++: lowest-k% average of distribution-calibrated log-probabilities."""
    standardized = []
    for s in scores:
        # sigma collapses to 0 at a perfectly flat position, where the
        # observed token is exactly as (un)likely as every other and the
        # calibrated score is undefined; 0.0 is the neutral value.
        if s.logprob_std <= 1e-9:
            standardized.append(0.0)
        else:
            standardized.append((s.logprob - s.expected_logprob) / s.logprob_std)
    return _mean_of_lowest_k(standardized, k_percent)


def perplexity(scores: Sequence[TokenScore]) -> float:
    """Plain perplexity, provided as a naive baseline to compare against.

    Unlike the Min-K% scores, *lower* perplexity suggests membership.
    """
    logprobs = [s.logprob for s in scores]
    if not logprobs:
        return float("nan")
    return math.exp(-sum(logprobs) / len(logprobs))
