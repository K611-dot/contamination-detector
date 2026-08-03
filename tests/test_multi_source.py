import math

import pytest

from contamination_detector.ngram_overlap import MultiSourceIndex

TRAIN = "the quick brown fox jumps over the lazy dog in the meadow at dawn"
TEST = "a photon travelling through a vacuum moves at a constant speed always"


@pytest.fixture
def index():
    return MultiSourceIndex({"train": [TRAIN], "test": [TEST]}, n=5)


def test_rejects_invalid_ngram_size():
    with pytest.raises(ValueError):
        MultiSourceIndex({"train": ["some text"]}, n=0)


def test_reports_every_source(index):
    result = index.overlap_score(TRAIN, example_id="q1")
    assert set(result.per_source) == {"train", "test"}
    assert index.source_names == ["train", "test"]


def test_identifies_which_source_a_leak_came_from(index):
    """The finding that decides the remedy: train-split or test-split."""
    from_train = index.overlap_score(TRAIN, example_id="q1")
    from_test = index.overlap_score(TEST, example_id="q2")

    assert from_train.best_source == "train"
    assert from_test.best_source == "test"


def test_clean_example_matches_no_source(index):
    result = index.overlap_score(
        "mitochondria generate chemical energy through oxidative phosphorylation daily"
    )
    assert result.best_source is None
    assert result.matched_sources() == []
    assert result.longest_match_tokens == 0


def test_matched_sources_ordered_by_run_length():
    shared = "alpha beta gamma delta epsilon zeta eta theta"
    index = MultiSourceIndex(
        {
            "long_match": [shared + " iota kappa lambda mu nu"],
            "short_match": ["zzz alpha beta gamma delta epsilon yyy"],
        },
        n=3,
    )
    result = index.overlap_score(shared)
    assert result.matched_sources()[0] == "long_match"
    assert len(result.matched_sources()) == 2


def test_aggregate_properties_take_the_worst_source(index):
    result = index.overlap_score(TRAIN)
    per_source_max = max(r.longest_match_tokens for r in result.per_source.values())
    assert result.longest_match_tokens == per_source_max
    assert result.max_overlap_fraction == 1.0


def test_unscorable_example_is_unscorable_across_all_sources(index):
    result = index.overlap_score("too short")
    assert result.scorable is False
    assert math.isnan(result.max_overlap_fraction)


def test_batch_scoring_preserves_ids(index):
    results = index.batch_overlap_scores({"a": TRAIN, "b": TEST})
    assert [r.example_id for r in results] == ["a", "b"]
    assert results[0].best_source == "train"
    assert results[1].best_source == "test"


def test_per_source_scores_are_independent(index):
    """A train-split leak must not inflate the test-split score."""
    result = index.overlap_score(TRAIN)
    assert result.per_source["train"].overlap_fraction == 1.0
    assert result.per_source["test"].overlap_fraction == 0.0
