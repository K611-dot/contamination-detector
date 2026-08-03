import math

from contamination_detector.min_k_prob import (
    TokenScore,
    min_k_percent,
    min_k_plus_plus,
    perplexity,
)


def _scores(logprobs, mu=0.0, sigma=1.0):
    return [
        TokenScore(logprob=lp, expected_logprob=mu, logprob_std=sigma) for lp in logprobs
    ]


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


def test_k_percent_of_one_hundred_uses_every_token():
    scores = _scores([-1.0, -2.0, -3.0])
    assert min_k_percent(scores, k_percent=100.0) == -2.0


def test_min_k_plus_plus_calibrates_against_model_expectation():
    # Same raw logprob, but in the second case the model expected exactly
    # that value, so the calibrated score should be higher (less surprising).
    surprising = [TokenScore(logprob=-5.0, expected_logprob=-1.0, logprob_std=1.0)]
    expected = [TokenScore(logprob=-5.0, expected_logprob=-5.0, logprob_std=1.0)]
    assert min_k_plus_plus(expected) > min_k_plus_plus(surprising)


def test_min_k_plus_plus_is_neutral_when_sigma_collapses():
    # A perfectly flat distribution makes the calibrated score undefined.
    scores = [TokenScore(logprob=-2.0, expected_logprob=-2.0, logprob_std=0.0)]
    assert min_k_plus_plus(scores) == 0.0


def test_min_k_plus_plus_defaults_reduce_to_min_k():
    scores = _scores([-1.0, -2.0, -3.0, -4.0])
    assert min_k_plus_plus(scores, k_percent=50) == min_k_percent(scores, k_percent=50)


def test_perplexity_of_certain_tokens_is_one():
    assert perplexity(_scores([0.0, 0.0, 0.0])) == 1.0


def test_from_log_distribution_uses_probability_weighted_statistics():
    # Peaked distribution: one likely token, many unlikely ones. The
    # probability-weighted mean must sit near the likely token's logprob,
    # NOT near the unweighted average of the vector (which the long tail
    # of near-zero-probability tokens would drag far down).
    p = [0.9] + [0.1 / 999] * 999
    log_probs = [math.log(x) for x in p]

    score = TokenScore.from_log_distribution(log_probs, observed_token_id=0)

    unweighted_mean = sum(log_probs) / len(log_probs)
    assert score.expected_logprob > unweighted_mean + 5  # nowhere near it
    assert score.logprob == log_probs[0]
    # mu is the negative entropy of the distribution
    entropy = -sum(x * math.log(x) for x in p)
    assert math.isclose(score.expected_logprob, -entropy, rel_tol=1e-9)


def test_from_log_distribution_sigma_is_zero_for_uniform():
    log_probs = [math.log(1 / 100)] * 100
    score = TokenScore.from_log_distribution(log_probs, observed_token_id=7)
    assert math.isclose(score.logprob_std, 0.0, abs_tol=1e-9)


def test_from_log_distribution_ranks_confident_token_above_tail_token():
    p = [0.9] + [0.1 / 999] * 999
    log_probs = [math.log(x) for x in p]

    confident = TokenScore.from_log_distribution(log_probs, observed_token_id=0)
    tail = TokenScore.from_log_distribution(log_probs, observed_token_id=500)

    assert min_k_plus_plus([confident]) > min_k_plus_plus([tail])
