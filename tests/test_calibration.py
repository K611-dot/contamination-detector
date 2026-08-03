import math

import pytest

from contamination_detector.calibration import (
    calibrate_index,
    calibrate_sources,
    percentile,
)
from contamination_detector.ngram_overlap import CorpusIndex, MultiSourceIndex


def test_percentile_endpoints():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 50) == 5.0


def test_percentile_ignores_nan():
    assert percentile([1.0, float("nan"), 3.0], 100) == 3.0


def test_percentile_of_empty_is_nan():
    assert math.isnan(percentile([], 50))


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 150)


def test_calibration_requires_controls():
    index = CorpusIndex(["some corpus text that is reasonably long here"], n=3)
    with pytest.raises(ValueError):
        calibrate_index(index, [])


def test_calibration_rejects_controls_shorter_than_ngram_size():
    index = CorpusIndex(["some corpus text that is reasonably long here"], n=13)
    with pytest.raises(ValueError, match="long enough"):
        calibrate_index(index, ["too short", "also short"])


def test_threshold_sits_above_innocent_overlap():
    """The point of calibration: ordinary text must not clear the bar."""
    corpus = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    index = CorpusIndex([corpus], n=3)
    controls = [
        "alpha beta gamma nu xi omicron pi rho sigma tau",
        "delta epsilon zeta upsilon phi chi psi omega aleph bet",
    ]

    thresholds = calibrate_index(index, controls, percentile_value=100.0)

    for control in controls:
        result = index.overlap_score(control)
        assert result.longest_match_tokens < thresholds.longest_run


def test_formulaic_source_gets_a_higher_threshold_than_prose():
    """The core claim: innocent overlap is not stationary across sources.

    A boilerplate-heavy source produces long innocent runs against ordinary
    control text, so its threshold must land higher than a prose source's.
    A single global threshold cannot serve both.
    """
    boilerplate = " ".join(
        ["please show all working and simplify your answer completely"] * 20
    )
    prose = (
        "the harbour was empty that morning and the gulls had gone inland "
        "before the weather turned, which the fishermen took as a warning"
    )
    controls = [
        "please show all working and simplify your answer completely for question one",
        "please show all working and simplify your answer completely for question two",
    ]

    index = MultiSourceIndex({"formulaic": [boilerplate], "prose": [prose]}, n=5)
    thresholds = calibrate_sources(index, controls, percentile_value=100.0)

    assert thresholds["formulaic"].longest_run > thresholds["prose"].longest_run


def test_min_longest_run_floor_is_respected():
    corpus = "completely unrelated words appearing nowhere near the controls at all"
    index = CorpusIndex([corpus], n=5)
    controls = ["entirely different control text with no shared phrasing whatsoever"]

    thresholds = calibrate_index(index, controls, min_longest_run=25)

    # Controls produce no overlap, so without a floor the threshold would be
    # near zero and flag everything.
    assert thresholds.longest_run >= 25


def test_thresholds_report_whether_they_are_well_estimated():
    corpus = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    index = CorpusIndex([corpus], n=3)

    few = calibrate_index(index, ["some control text here for testing"], percentile_value=99.0)
    assert few.is_well_estimated is False

    many = calibrate_index(
        index,
        [f"control text number {i} with assorted filler words" for i in range(150)],
        percentile_value=99.0,
    )
    assert many.is_well_estimated is True


def test_calibrate_sources_covers_every_source():
    index = MultiSourceIndex(
        {"train": ["alpha beta gamma delta"], "test": ["epsilon zeta eta theta"]}, n=3
    )
    thresholds = calibrate_sources(index, ["some control text for calibration here"])
    assert set(thresholds) == {"train", "test"}
