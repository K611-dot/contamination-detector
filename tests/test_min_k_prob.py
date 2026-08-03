import math

from contamination_detector.min_k_prob import (
    TokenScore,
    min_k_percent,
    min_k_plus_plus,
    perplexity,
)


def _scores(logprobs, mean=0.0, std=1.0):
    return [TokenScore(logprob=lp, mean_logprob=mean, std_logprob=std) for lp in logprobs]


def test_min_k_percent_averages_lowest_tail_only():
    # lowest 20% of 10 values = the 2 smallest: -9 and -8
    scores = _scores([-1, -2, -3, -4, -5, -6, -7, -8, -9, -0.5])
    assert min_k_percent(scores, k_percent=20) == -8.5


def test_memorized_text_scores_higher_than_unseen():
    memorized = _scores([-0.1, -0.2, -0.1, -0.3, -0.2])
    unseen = _scores([-4.0, -0.2, -7.0, -0.3, -5.0])
    assert min_k_percent(memorized) > min_k_percent(unseen)


def test_empty_scores_return_nan():
    assert math.isnan(min_k_percent([]))
    assert math.isnan(min_k_plus_plus([]))
    assert math.isnan(perplexity([]))


def test_min_k_percent_uses_at_least_one_token():
    scores = _scores([-1.0, -2.0])
    assert min_k_percent(scores, k_percent=0.0) == -2.0


def test_min_k_plus_plus_standardizes_against_vocab_distribution():
    # Same raw logprob, but the second token is low-probability everywhere
    # in the vocab, so standardizing should rank it as less surprising.
    raw_only = [TokenScore(logprob=-5.0, mean_logprob=-1.0, std_logprob=1.0)]
    normalized = [TokenScore(logprob=-5.0, mean_logprob=-5.0, std_logprob=1.0)]
    assert min_k_plus_plus(normalized) > min_k_plus_plus(raw_only)


def test_min_k_plus_plus_tolerates_zero_std():
    scores = [TokenScore(logprob=-2.0, mean_logprob=-2.0, std_logprob=0.0)]
    assert not math.isnan(min_k_plus_plus(scores))


def test_perplexity_of_certain_tokens_is_one():
    assert perplexity(_scores([0.0, 0.0, 0.0])) == 1.0
