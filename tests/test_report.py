import math

import pytest

from contamination_detector.report import auc_score, build_report, zscores


def test_zscores_of_constant_values_are_zero():
    assert zscores([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_zscores_center_on_the_mean():
    result = zscores([1.0, 2.0, 3.0])
    assert result[1] == 0.0
    assert result[0] < 0 < result[2]


def test_auc_is_one_when_positives_all_score_higher():
    assert auc_score([0.9, 0.8], [0.1, 0.2]) == 1.0


def test_auc_is_half_for_fully_tied_scores():
    assert auc_score([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_auc_is_nan_without_both_classes():
    assert math.isnan(auc_score([0.9], []))


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


def test_report_computes_per_method_auc_when_labels_given():
    report = build_report(
        example_ids=["a", "b", "c", "d"],
        method_scores={"ngram": [0.1, 0.2, 0.8, 0.9]},
        labels=[0, 0, 1, 1],
    )
    assert report.method_auc["ngram"] == 1.0


def test_report_rejects_misaligned_score_lists():
    with pytest.raises(ValueError):
        build_report(example_ids=["a", "b"], method_scores={"ngram": [0.1]})


def test_report_rejects_misaligned_labels():
    with pytest.raises(ValueError):
        build_report(
            example_ids=["a", "b"], method_scores={"ngram": [0.1, 0.2]}, labels=[1]
        )
