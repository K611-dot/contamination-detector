import math

import pytest

from contamination_detector.report import (
    MethodDirection,
    auc_score,
    build_report,
    zscores,
)


def test_zscores_of_constant_values_are_zero():
    assert zscores([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_zscores_center_on_the_mean():
    result = zscores([1.0, 2.0, 3.0])
    assert result[1] == 0.0
    assert result[0] < 0 < result[2]


def test_zscores_preserve_nan_instead_of_treating_it_as_average():
    result = zscores([1.0, 2.0, 3.0, float("nan")])
    assert math.isnan(result[3])
    # The NaN must not shift the statistics of the real values.
    assert result[1] == 0.0


def test_auc_is_one_when_positives_all_score_higher():
    assert auc_score([0.9, 0.8], [0.1, 0.2]) == 1.0


def test_auc_is_half_for_fully_tied_scores():
    assert auc_score([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_auc_is_nan_without_both_classes():
    assert math.isnan(auc_score([0.9], []))


def test_auc_ignores_unscorable_examples():
    assert auc_score([0.9, float("nan")], [0.1]) == 1.0


def test_report_flags_outlier_example():
    report = build_report(
        example_ids=["a", "b", "c", "d", "e", "f"],
        method_scores={"ngram": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
        flag_threshold=2.0,
    )
    flagged = [e.example_id for e in report.examples if e.flagged]
    assert flagged == ["f"]


def test_report_combines_multiple_methods():
    report = build_report(
        example_ids=["a", "b"],
        method_scores={"ngram": [0.0, 1.0], "min_k": [0.0, 1.0]},
        flag_threshold=2.0,
    )
    assert report.most_suspicious(1)[0].example_id == "b"
    assert set(report.examples[0].method_scores) == {"ngram", "min_k"}


def test_lower_is_contaminated_method_is_inverted_before_combining():
    # Perplexity: the contaminated example ("b") has the LOWER score.
    report = build_report(
        example_ids=["a", "b"],
        method_scores={"perplexity": [100.0, 5.0]},
        directions={"perplexity": MethodDirection.LOWER_IS_CONTAMINATED},
    )
    assert report.most_suspicious(1)[0].example_id == "b"


def test_direction_is_applied_to_auc_too():
    report = build_report(
        example_ids=["a", "b", "c", "d"],
        method_scores={"perplexity": [90.0, 80.0, 20.0, 10.0]},
        labels=[0, 0, 1, 1],
        directions={"perplexity": MethodDirection.LOWER_IS_CONTAMINATED},
    )
    assert report.method_auc["perplexity"] == 1.0


def test_report_computes_per_method_auc_when_labels_given():
    report = build_report(
        example_ids=["a", "b", "c", "d"],
        method_scores={"ngram": [0.1, 0.2, 0.8, 0.9]},
        labels=[0, 0, 1, 1],
    )
    assert report.method_auc["ngram"] == 1.0


def test_auc_weighting_suppresses_a_chance_level_method():
    # "good" separates the classes perfectly; "noise" is anti-correlated.
    ids = ["a", "b", "c", "d"]
    good = [0.1, 0.2, 0.8, 0.9]
    noise = [0.9, 0.8, 0.2, 0.1]
    labels = [0, 0, 1, 1]

    unweighted = build_report(ids, {"good": good, "noise": noise}, labels=labels)
    weighted = build_report(
        ids, {"good": good, "noise": noise}, labels=labels, weight_by_auc=True
    )

    # Equal weighting lets the useless method cancel the useful one out.
    assert unweighted.examples[3].combined_zscore == pytest.approx(0.0, abs=1e-9)
    # AUC weighting drops the below-chance method and recovers the signal.
    assert weighted.examples[3].combined_zscore > 1.0
    assert weighted.method_weights["noise"] == 0.0


def test_weight_by_auc_requires_labels():
    with pytest.raises(ValueError):
        build_report(["a"], {"ngram": [0.1]}, weight_by_auc=True)


def test_unscorable_example_is_reported_separately_not_as_clean():
    report = build_report(
        example_ids=["a", "b", "c"],
        method_scores={"ngram": [0.1, 0.9, float("nan")]},
    )
    unscorable = [e.example_id for e in report.unscorable()]
    assert unscorable == ["c"]
    assert "c" not in [e.example_id for e in report.most_suspicious()]


def test_partially_scorable_example_uses_the_methods_that_worked():
    report = build_report(
        example_ids=["a", "b", "c"],
        method_scores={
            "ngram": [0.0, 0.0, float("nan")],
            "min_k": [0.0, 0.0, 5.0],
        },
    )
    c = next(e for e in report.examples if e.example_id == "c")
    assert c.scorable
    assert not math.isnan(c.combined_zscore)


def test_report_rejects_misaligned_score_lists():
    with pytest.raises(ValueError):
        build_report(example_ids=["a", "b"], method_scores={"ngram": [0.1]})


def test_report_rejects_misaligned_labels():
    with pytest.raises(ValueError):
        build_report(
            example_ids=["a", "b"], method_scores={"ngram": [0.1, 0.2]}, labels=[1]
        )
