"""Aggregate per-method contamination scores into a single report.

Each detector (n-gram overlap, Min-K%, guided prompting) produces one
score per example on its own scale. This module standardizes those
scores, combines them, flags outliers, and — when ground-truth
contamination labels are available (e.g. on a held-out validation set
built for calibration) — reports AUC per method so a researcher can see
which detector is actually working on their data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def zscores(values: list[float]) -> list[float]:
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return [0.0 for _ in values]
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / len(clean)
    std = math.sqrt(variance) or 1e-8
    return [0.0 if math.isnan(v) else (v - mean) / std for v in values]


def auc_score(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Mann-Whitney-U based AUC: P(random positive score > random negative score).

    Implemented via rank-sum so no external stats library is required.
    """
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


@dataclass
class ContaminationReport:
    examples: list[ExampleReport]
    method_auc: dict[str, float] = field(default_factory=dict)

    def most_suspicious(self, top_n: int = 10) -> list[ExampleReport]:
        return sorted(self.examples, key=lambda e: e.combined_zscore, reverse=True)[:top_n]


def build_report(
    example_ids: list[str],
    method_scores: dict[str, list[float]],
    labels: list[int] | None = None,
    flag_threshold: float = 2.0,
) -> ContaminationReport:
    """Build a combined report.

    `method_scores` maps a method name (e.g. "ngram_overlap", "min_k_plus_plus")
    to a list of scores aligned with `example_ids`. `labels`, if given, is a
    parallel list of 1 (known contaminated) / 0 (known clean) used only to
    compute per-method AUC for calibration — it is not required to run the
    detectors themselves.
    """
    n = len(example_ids)
    for name, scores in method_scores.items():
        if len(scores) != n:
            raise ValueError(f"method '{name}' has {len(scores)} scores, expected {n}")

    method_z = {name: zscores(scores) for name, scores in method_scores.items()}

    examples = []
    for idx, eid in enumerate(example_ids):
        per_method = {name: method_scores[name][idx] for name in method_scores}
        per_method_z = {name: method_z[name][idx] for name in method_z}
        combined = sum(per_method_z.values()) / len(per_method_z) if per_method_z else 0.0
        examples.append(
            ExampleReport(
                example_id=eid,
                method_scores=per_method,
                method_zscores=per_method_z,
                combined_zscore=combined,
                flagged=combined >= flag_threshold,
            )
        )

    method_auc: dict[str, float] = {}
    if labels is not None:
        if len(labels) != n:
            raise ValueError(f"labels has {len(labels)} entries, expected {n}")
        for name, scores in method_scores.items():
            pos = [s for s, label in zip(scores, labels) if label == 1]
            neg = [s for s, label in zip(scores, labels) if label == 0]
            method_auc[name] = auc_score(pos, neg)

    return ContaminationReport(examples=examples, method_auc=method_auc)
