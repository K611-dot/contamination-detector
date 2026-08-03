"""Aggregate per-method contamination scores into a single report.

Each detector (n-gram overlap, Min-K%, guided prompting) produces one
score per example on its own scale. This module standardizes those
scores, combines them, flags outliers, and — when ground-truth
contamination labels are available (e.g. on a held-out calibration set) —
reports AUC per method so a researcher can see which detector is actually
working on their data.

Two things the naive version of this got wrong, both worth knowing about:

- **Direction.** Not every score points the same way. Higher overlap and
  higher Min-K% mean *more* likely contaminated, but higher perplexity
  means *less*. Combining them without accounting for that cancels real
  signal. Declare direction with `MethodDirection`.

- **Weighting.** Averaging methods equally assumes they are equally
  informative. When labels are available, `weight_by_auc=True` weights
  each method by how well it actually separates the classes, so a
  detector that is near-random on your data stops diluting one that works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class MethodDirection(Enum):
    """Which way a method's score points."""

    HIGHER_IS_CONTAMINATED = 1
    LOWER_IS_CONTAMINATED = -1


def zscores(values: list[float]) -> list[float]:
    """Standardize, preserving NaN as NaN rather than coercing it to average."""
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return [float("nan") if math.isnan(v) else 0.0 for v in values]
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / len(clean)
    std = math.sqrt(variance) or 1e-8
    return [float("nan") if math.isnan(v) else (v - mean) / std for v in values]


def auc_score(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Mann-Whitney-U based AUC: P(random positive score > random negative score).

    Implemented via rank-sum so no external stats library is required.
    NaN scores are dropped, since an unscorable example carries no evidence.
    """
    positive_scores = [s for s in positive_scores if not math.isnan(s)]
    negative_scores = [s for s in negative_scores if not math.isnan(s)]
    if not positive_scores or not negative_scores:
        return float("nan")
    labeled = [(s, 1) for s in positive_scores] + [(s, 0) for s in negative_scores]
    labeled.sort(key=lambda x: x[0])

    ranks = [0.0] * len(labeled)
    i = 0
    while i < len(labeled):
        j = i
        while j < len(labeled) and labeled[j][0] == labeled[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-indexed rank, averaged over ties
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    rank_sum_pos = sum(r for r, (_, label) in zip(ranks, labeled) if label == 1)
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


@dataclass
class ExampleReport:
    example_id: str
    method_scores: dict[str, float]
    method_zscores: dict[str, float]
    combined_zscore: float
    flagged: bool

    @property
    def scorable(self) -> bool:
        """False when no method could score this example."""
        return not math.isnan(self.combined_zscore)


@dataclass
class ContaminationReport:
    examples: list[ExampleReport]
    method_auc: dict[str, float] = field(default_factory=dict)
    method_weights: dict[str, float] = field(default_factory=dict)

    def most_suspicious(self, top_n: int = 10) -> list[ExampleReport]:
        scorable = [e for e in self.examples if e.scorable]
        return sorted(scorable, key=lambda e: e.combined_zscore, reverse=True)[:top_n]

    def unscorable(self) -> list[ExampleReport]:
        """Examples no method could score — absence of evidence, not evidence of absence."""
        return [e for e in self.examples if not e.scorable]


def build_report(
    example_ids: list[str],
    method_scores: dict[str, list[float]],
    labels: list[int] | None = None,
    flag_threshold: float = 2.0,
    directions: dict[str, MethodDirection] | None = None,
    weight_by_auc: bool = False,
) -> ContaminationReport:
    """Build a combined report.

    `method_scores` maps a method name (e.g. "ngram_overlap",
    "min_k_plus_plus") to a list of scores aligned with `example_ids`.

    `directions` declares, per method, whether a higher score means more
    contaminated. Anything unlisted defaults to higher-is-contaminated.

    `labels`, if given, is a parallel list of 1 (known contaminated) /
    0 (known clean) used to compute per-method AUC. Set `weight_by_auc`
    to also weight the combination by measured separation quality.
    """
    n = len(example_ids)
    for name, scores in method_scores.items():
        if len(scores) != n:
            raise ValueError(f"method '{name}' has {len(scores)} scores, expected {n}")
    if labels is not None and len(labels) != n:
        raise ValueError(f"labels has {len(labels)} entries, expected {n}")
    if weight_by_auc and labels is None:
        raise ValueError("weight_by_auc requires labels")

    directions = directions or {}
    oriented: dict[str, list[float]] = {}
    for name, scores in method_scores.items():
        sign = directions.get(name, MethodDirection.HIGHER_IS_CONTAMINATED).value
        oriented[name] = [s * sign for s in scores]

    method_z = {name: zscores(scores) for name, scores in oriented.items()}

    method_auc: dict[str, float] = {}
    if labels is not None:
        for name, scores in oriented.items():
            pos = [s for s, label in zip(scores, labels) if label == 1]
            neg = [s for s, label in zip(scores, labels) if label == 0]
            method_auc[name] = auc_score(pos, neg)

    # Weight by how far each method beats chance (AUC 0.5). A method at or
    # below chance contributes nothing rather than adding noise.
    weights: dict[str, float] = {name: 1.0 for name in method_scores}
    if weight_by_auc:
        raw = {
            name: max(0.0, (method_auc.get(name, 0.5) or 0.0) - 0.5)
            for name in method_scores
        }
        total = sum(raw.values())
        if total > 0:
            weights = {name: value / total for name, value in raw.items()}

    examples = []
    for idx, eid in enumerate(example_ids):
        per_method = {name: method_scores[name][idx] for name in method_scores}
        per_method_z = {name: method_z[name][idx] for name in method_z}

        # Average only the methods that could actually score this example,
        # so one unscorable method does not drag the combination toward zero.
        usable = [
            (weights[name], value)
            for name, value in per_method_z.items()
            if not math.isnan(value)
        ]
        weight_sum = sum(w for w, _ in usable)
        if not usable or weight_sum <= 0:
            combined = float("nan")
        else:
            combined = sum(w * v for w, v in usable) / weight_sum

        examples.append(
            ExampleReport(
                example_id=eid,
                method_scores=per_method,
                method_zscores=per_method_z,
                combined_zscore=combined,
                flagged=(not math.isnan(combined)) and combined >= flag_threshold,
            )
        )

    return ContaminationReport(
        examples=examples, method_auc=method_auc, method_weights=weights
    )
