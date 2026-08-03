"""Measure detector accuracy on the synthetic contamination benchmark.

Run: python scripts/benchmark_detectors.py

Reports both ranking quality (AUC) and what happens at a usable operating
threshold. Both are needed: AUC only measures whether contaminated
examples *rank* above clean ones, and a setting can rank well while still
being unusable because the two score distributions overlap so much that no
threshold separates them. False-positive rate at the operating point is
what decides whether you can act on a flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contamination_detector.evaluation import make_dataset  # noqa: E402
from contamination_detector.ngram_overlap import CorpusIndex  # noqa: E402
from contamination_detector.report import auc_score  # noqa: E402

TYPES = ["verbatim", "partial", "paraphrased"]

# Operating thresholds matching the CLI defaults.
THRESHOLDS = {"overlap_fraction": 0.5, "longest_run": 30.0}


def auc_for(dataset, values: dict[str, float]) -> float:
    pos, neg = [], []
    for example in dataset.examples:
        (pos if example.is_contaminated else neg).append(values[example.example_id])
    return auc_score(pos, neg)


def operating_point(dataset, values: dict[str, float], threshold: float):
    """Recall and false-positive rate if we flag everything >= threshold."""
    tp = fn = fp = tn = 0
    for example in dataset.examples:
        flagged = values[example.example_id] >= threshold
        if example.is_contaminated:
            tp, fn = (tp + 1, fn) if flagged else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if flagged else (fp, tn + 1)
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return recall, fpr, precision


def main() -> int:
    dataset = make_dataset(seed=0)
    counts = {t: sum(1 for e in dataset.examples if e.contamination_type == t) for t in TYPES}
    clean = sum(1 for e in dataset.examples if not e.is_contaminated)
    print(
        f"dataset: {clean} clean, "
        + ", ".join(f"{counts[t]} {t}" for t in TYPES)
        + f" | corpus: {len(dataset.corpus)} docs"
    )
    print("AUC 1.0 = perfect ranking, 0.5 = chance. FPR is at the CLI's default threshold.\n")

    header = (
        f"{'n':>3} {'signal':<16} "
        + " ".join(f"{t[:9]:>9}" for t in TYPES)
        + f" {'overall':>8} {'recall':>7} {'FPR':>7} {'prec':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n in [5, 8, 13, 20]:
        index = CorpusIndex(dataset.corpus, n=n)
        scored = {
            e.example_id: index.overlap_score(e.text, e.example_id) for e in dataset.examples
        }

        for signal, extract in [
            ("overlap_fraction", lambda r: 0.0 if r.total_ngrams == 0 else r.overlap_fraction),
            ("longest_run", lambda r: float(r.longest_match_tokens)),
        ]:
            values = {eid: extract(r) for eid, r in scored.items()}
            per_type = [auc_for(dataset.subset(t), values) for t in TYPES]
            overall = auc_for(dataset, values)
            recall, fpr, precision = operating_point(dataset, values, THRESHOLDS[signal])
            rows.append((n, signal, overall, recall, fpr, precision))
            print(
                f"{n:>3} {signal:<16} "
                + " ".join(f"{a:>9.3f}" for a in per_type)
                + f" {overall:>8.3f} {recall:>7.2f} {fpr:>7.2f} {precision:>7.2f}"
            )
        print()

    # A setting is only usable if it almost never flags clean text; among
    # those, prefer the one that catches the most leakage.
    usable = [r for r in rows if r[4] <= 0.05]
    if usable:
        best = max(usable, key=lambda r: r[3])
        print(f"best usable setting (FPR <= 0.05): n={best[0]} {best[1]}")
        print(f"  recall {best[3]:.2f}, FPR {best[4]:.2f}, precision {best[5]:.2f}")

    best_auc = max(rows, key=lambda r: r[2])
    print(f"\nbest raw AUC: n={best_auc[0]} {best_auc[1]} ({best_auc[2]:.3f}) "
          f"- but FPR {best_auc[4]:.2f} at threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
