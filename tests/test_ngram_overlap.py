import math

import pytest

from contamination_detector.ngram_overlap import (
    CorpusIndex,
    batch_overlap_scores,
    ngram_hashes,
    ngrams,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_ngrams_shorter_than_n_returns_empty():
    assert ngrams(["a", "b"], 3) == set()


def test_ngrams_sliding_window():
    assert ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}


def test_ngram_hashes_are_positional_not_deduplicated():
    # "a b a b a" contains ("a","b") twice; positional output keeps both so
    # contiguous runs remain detectable.
    hashes = ngram_hashes(["a", "b", "a", "b", "a"], 2)
    assert len(hashes) == 4
    assert hashes[0] == hashes[2]


def test_ngram_hashes_agree_with_tuple_hashing():
    tokens = tokenize("the quick brown fox jumps over the lazy dog")
    assert set(ngram_hashes(tokens, 3)) == {hash(g) for g in ngrams(tokens, 3)}


def test_zero_or_negative_n_is_rejected():
    with pytest.raises(ValueError):
        ngrams(["a"], 0)
    with pytest.raises(ValueError):
        CorpusIndex(["some text"], n=-1)


def test_verbatim_example_scores_full_overlap():
    text = "the quick brown fox jumps over the lazy dog"
    index = CorpusIndex([f"prefix text {text} suffix text"], n=5)
    result = index.overlap_score(text)
    assert result.overlap_fraction == 1.0


def test_unrelated_example_scores_zero_overlap():
    index = CorpusIndex(["completely different content about marine biology"], n=5)
    result = index.overlap_score("the quick brown fox jumps over the lazy dog")
    assert result.overlap_fraction == 0.0
    assert result.longest_match_tokens == 0


def test_example_shorter_than_ngram_size_is_unscorable_not_clean():
    index = CorpusIndex(["some corpus text here for testing purposes"], n=13)
    result = index.overlap_score("too short")
    assert math.isnan(result.overlap_fraction)
    assert result.total_ngrams == 0
    assert result.scorable is False


def test_partial_overlap_is_between_zero_and_one():
    example = "alpha beta gamma delta epsilon zeta eta theta"
    index = CorpusIndex(["alpha beta gamma delta unrelated words follow"], n=3)
    result = index.overlap_score(example)
    assert 0.0 < result.overlap_fraction < 1.0


def test_longest_match_counts_contiguous_tokens():
    # A verbatim 6-token run, then unrelated text.
    index = CorpusIndex(["alpha beta gamma delta epsilon zeta nothing else here"], n=3)
    result = index.overlap_score("alpha beta gamma delta epsilon zeta plus other words")
    assert result.longest_match_tokens == 6


def test_longest_match_distinguishes_scattered_from_contiguous():
    corpus = "alpha beta gamma something else entirely delta epsilon zeta"
    index = CorpusIndex([corpus], n=3)

    contiguous = index.overlap_score("alpha beta gamma something else entirely")
    scattered = index.overlap_score("alpha beta gamma zzz delta epsilon zeta")

    # Both find matches, but only one is a single verbatim run.
    assert contiguous.longest_match_tokens > scattered.longest_match_tokens


def test_repeated_phrase_does_not_inflate_fraction():
    # The example repeats one phrase that is NOT in the corpus; deduplicating
    # distinct n-grams keeps the repetition from skewing the score.
    index = CorpusIndex(["entirely unrelated corpus material"], n=3)
    result = index.overlap_score("ping pong ball ping pong ball ping pong ball")
    assert result.overlap_fraction == 0.0


def test_index_supports_membership_check():
    index = CorpusIndex(["alpha beta gamma delta"], n=2)
    assert ("alpha", "beta") in index
    assert ("zeta", "eta") not in index


def test_index_accepts_a_generator_of_documents():
    docs = (d for d in ["alpha beta gamma", "delta epsilon zeta"])
    index = CorpusIndex(docs, n=2)
    assert len(index) == 4


def test_batch_scores_preserve_ids():
    results = batch_overlap_scores(
        {"a": "shared phrase appears here", "b": "nothing alike whatsoever friend"},
        ["shared phrase appears here in the corpus"],
        n=3,
    )
    by_id = {r.example_id: r for r in results}
    assert by_id["a"].overlap_fraction == 1.0
    assert by_id["b"].overlap_fraction == 0.0
