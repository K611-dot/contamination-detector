import pytest

from contamination_detector.evaluation import make_dataset
from contamination_detector.ngram_overlap import CorpusIndex
from contamination_detector.report import auc_score

# Generating a full-size dataset is the expensive part of these tests, so the
# shared one is built once per session rather than per test.


@pytest.fixture(scope="session")
def dataset():
    return make_dataset(seed=0)


@pytest.fixture(scope="session")
def index_n5(dataset):
    return CorpusIndex(dataset.corpus, n=5)


@pytest.fixture(scope="session")
def index_n13(dataset):
    return CorpusIndex(dataset.corpus, n=13)


def _small(**overrides):
    """A cheap dataset for tests that do not depend on realistic scale."""
    params = dict(n_clean=10, n_per_contamination=4, corpus_docs=6, corpus_doc_length=150)
    params.update(overrides)
    return make_dataset(**params)


def _mean_overlap(index, examples):
    scores = [index.overlap_score(e.text).overlap_fraction for e in examples]
    return sum(scores) / len(scores)


def test_dataset_is_reproducible_for_a_seed():
    a = _small(seed=7)
    b = _small(seed=7)
    assert [e.text for e in a.examples] == [e.text for e in b.examples]
    assert a.corpus == b.corpus


def test_different_seeds_give_different_data():
    assert [e.text for e in _small(seed=1).examples] != [
        e.text for e in _small(seed=2).examples
    ]


def test_labels_align_with_examples():
    data = _small(seed=0)
    assert len(data.labels) == len(data.examples)
    for label, example in zip(data.labels, data.examples):
        assert label == int(example.is_contaminated)


def test_dataset_contains_every_contamination_type():
    types = {e.contamination_type for e in _small(seed=0).examples}
    assert types == {"clean", "verbatim", "partial", "paraphrased"}


def test_subset_keeps_clean_examples_plus_one_type():
    data = _small(seed=0)
    subset = data.subset("verbatim")
    assert {e.contamination_type for e in subset.examples} == {"clean", "verbatim"}
    assert subset.corpus == data.corpus


def test_verbatim_leaks_are_detected_perfectly(dataset, index_n13):
    """The easy case must be solved, or the harness itself is broken."""
    subset = dataset.subset("verbatim")
    pos, neg = [], []
    for e in subset.examples:
        score = index_n13.overlap_score(e.text).overlap_fraction
        (pos if e.is_contaminated else neg).append(score)
    assert auc_score(pos, neg) == 1.0


def test_clean_examples_overlap_the_corpus_enough_to_be_a_real_test(dataset, index_n5):
    """Guards the benchmark's difficulty.

    If clean text shares almost nothing with the corpus, every detector
    scores a perfect AUC and the benchmark measures nothing. Clean
    examples must show real incidental overlap at a small n.
    """
    clean = [e for e in dataset.examples if not e.is_contaminated]
    assert _mean_overlap(index_n5, clean) > 0.15, "clean examples are too easy to separate"


def test_raising_collocation_rate_makes_the_task_harder():
    def mean_clean_overlap(rate):
        data = _small(seed=0, collocation_rate=rate)
        index = CorpusIndex(data.corpus, n=5)
        return _mean_overlap(index, [e for e in data.examples if not e.is_contaminated])

    assert mean_clean_overlap(0.9) > mean_clean_overlap(0.3)


def test_paraphrased_leaks_are_harder_than_verbatim(dataset, index_n13):
    """Documents the known limitation, and fails if it silently changes."""

    def auc_for_type(contamination_type):
        subset = dataset.subset(contamination_type)
        pos, neg = [], []
        for e in subset.examples:
            score = index_n13.overlap_score(e.text).overlap_fraction
            (pos if e.is_contaminated else neg).append(score)
        return auc_score(pos, neg)

    assert auc_for_type("verbatim") > auc_for_type("paraphrased")


@pytest.mark.parametrize("contamination_type", ["verbatim", "partial", "paraphrased"])
def test_contaminated_text_is_actually_present_in_the_corpus(
    dataset, index_n5, contamination_type
):
    """Sanity check that the generator really plants what it claims to."""
    subset = dataset.subset(contamination_type)
    contaminated = [e for e in subset.examples if e.is_contaminated]
    clean = [e for e in subset.examples if not e.is_contaminated]
    assert _mean_overlap(index_n5, contaminated) > _mean_overlap(index_n5, clean)
