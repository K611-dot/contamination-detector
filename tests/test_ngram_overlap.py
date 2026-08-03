from contamination_detector.ngram_overlap import (
    CorpusIndex,
    batch_overlap_scores,
    ngrams,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_ngrams_shorter_than_n_returns_empty():
    assert ngrams(["a", "b"], 3) == set()


def test_ngrams_sliding_window():
    assert ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}


def test_verbatim_example_scores_full_overlap():
    text = "the quick brown fox jumps over the lazy dog"
    index = CorpusIndex([f"prefix text {text} suffix text"], n=5)
    result = index.overlap_score(text)
    assert result.overlap_fraction == 1.0


def test_unrelated_example_scores_zero_overlap():
    index = CorpusIndex(["completely different content about marine biology"], n=5)
    result = index.overlap_score("the quick brown fox jumps over the lazy dog")
    assert result.overlap_fraction == 0.0


def test_example_shorter_than_ngram_size_is_not_flagged():
    index = CorpusIndex(["some corpus text here for testing purposes"], n=13)
    result = index.overlap_score("too short")
    assert result.overlap_fraction == 0.0
    assert result.total_ngrams == 0


def test_partial_overlap_is_between_zero_and_one():
    example = "alpha beta gamma delta epsilon zeta eta theta"
    index = CorpusIndex(["alpha beta gamma delta unrelated words follow"], n=3)
    result = index.overlap_score(example)
    assert 0.0 < result.overlap_fraction < 1.0


def test_batch_scores_preserve_ids():
    results = batch_overlap_scores(
        {"a": "shared phrase appears here", "b": "nothing alike whatsoever friend"},
        ["shared phrase appears here in the corpus"],
        n=3,
    )
    by_id = {r.example_id: r for r in results}
    assert by_id["a"].overlap_fraction == 1.0
    assert by_id["b"].overlap_fraction == 0.0
