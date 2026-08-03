"""Per-source threshold calibration from a source's own innocent-overlap null.

A single global threshold assumes innocent overlap is stationary across
sources. It isn't. Formulaic text — templated math problems, boilerplate
scrape, legal or licence text — throws near-duplicate n-grams for reasons
that have nothing to do with leakage, while ordinary prose does not. A
threshold tuned on prose over-flags the formulaic sources into
uselessness; a threshold tuned on formulaic text quietly under-flags the
prose ones.

The fix is to stop guessing a constant and measure each source's own null
distribution instead. Score control text that you know is *not* in the
source — text written after the model's cutoff, held-out data the source
predates, anything provably outside it — and take a high percentile of the
resulting scores as that source's threshold. Whatever overlap innocent
text produces against that particular source, the threshold sits above it
by construction.

    thresholds = calibrate_index(index, control_texts, percentile=99.0)
    result = index.overlap_score(suspect_text)
    if result.longest_match_tokens >= thresholds.longest_run:
        ...

The quality of this depends entirely on the control set being genuinely
clean and genuinely representative of the benchmark's style. Control text
from a different domain calibrates the wrong distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from .ngram_overlap import CorpusIndex, MultiSourceIndex


@dataclass
class SourceThresholds:
    """Flag thresholds derived from one source's innocent-overlap null.

    `n_controls` is carried along because a percentile estimated from a
    handful of controls is not worth much — check it before trusting the
    numbers.
    """

    overlap_fraction: float
    longest_run: int
    percentile: float
    n_controls: int

    @property
    def is_well_estimated(self) -> bool:
        """Whether enough controls back the requested percentile.

        Estimating the 99th percentile needs on the order of 100 samples;
        below that the tail is guesswork.
        """
        tail_fraction = (100.0 - self.percentile) / 100.0
        if tail_fraction <= 0:
            return False
        return self.n_controls >= math.ceil(1.0 / tail_fraction)


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile. NaNs are dropped."""
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return float("nan")
    if not 0.0 <= pct <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if len(clean) == 1:
        return clean[0]
    rank = (pct / 100.0) * (len(clean) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return clean[int(rank)]
    return clean[low] + (clean[high] - clean[low]) * (rank - low)


def calibrate_index(
    index: "CorpusIndex",
    control_texts: Sequence[str],
    percentile_value: float = 99.0,
    min_longest_run: int | None = None,
) -> SourceThresholds:
    """Derive thresholds for one source from control text known to be clean.

    `min_longest_run` sets a floor under the run-length threshold, so a
    source whose controls happen to produce no overlap at all does not end
    up with a threshold of zero that flags everything.
    """
    if not control_texts:
        raise ValueError("need at least one control text to calibrate")

    results = [index.overlap_score(text) for text in control_texts]
    scorable = [r for r in results if r.scorable]
    if not scorable:
        raise ValueError(
            "no control text was long enough to score at this n-gram size; "
            "supply longer controls or lower n"
        )

    overlap_threshold = percentile([r.overlap_fraction for r in scorable], percentile_value)
    run_threshold = percentile(
        [float(r.longest_match_tokens) for r in scorable], percentile_value
    )

    # Land strictly above the observed innocent maximum rather than on it,
    # so text merely as ordinary as the controls does not get flagged.
    run_ceiling = int(math.ceil(run_threshold)) + 1
    if min_longest_run is not None:
        run_ceiling = max(run_ceiling, min_longest_run)

    return SourceThresholds(
        overlap_fraction=min(1.0, overlap_threshold),
        longest_run=run_ceiling,
        percentile=percentile_value,
        n_controls=len(scorable),
    )


def calibrate_sources(
    index: "MultiSourceIndex",
    control_texts: Sequence[str],
    percentile_value: float = 99.0,
    min_longest_run: int | None = None,
) -> dict[str, SourceThresholds]:
    """Calibrate every source in a MultiSourceIndex independently.

    The whole point is that these come out different. If two sources return
    near-identical thresholds they are alike enough that one would have
    done; if they diverge sharply, a global threshold was never going to
    serve both.
    """
    return {
        name: calibrate_index(
            source_index,
            control_texts,
            percentile_value=percentile_value,
            min_longest_run=min_longest_run,
        )
        for name, source_index in index.indexes.items()
    }
