"""Measure detector accuracy on the synthetic contamination benchmark.

Run: python scripts/benchmark_detectors.py            (10 seeds, the published numbers)
     python scripts/benchmark_detectors.py --seeds 1  (single seed, fast)

Reports both ranking quality (AUC) and what happens at a usable operating
threshold. Both are needed: AUC only measures whether contaminated
examples *rank* above clean ones, and a setting can rank well while still
being unusable because the two score distributions overlap so much that no
threshold separates them. False-positive rate at the operating point is
what decides whether you can act on a flag.

Results are averaged over several seeds by default. A single seed is not
enough to compare settings — the spread between seeds is larger than some
of the differences being compared, so a one-seed run will occasionally
rank the settings differently. The standard deviations printed alongside
each mean tell you which gaps are real.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contamination_detector.evaluation import make_dataset  # noqa: E402
from contamination_detector.ngram_overlap import CorpusIndex  # noqa: E402
from contamination_detector.report import auc_score  # noqa: E402

TYPES = ["verbatim", "partial", "paraphrased"]
NGRAM_SIZES = [5, 8, 13, 20]

# Operating thresholds matching the CLI defaults.
THRESHOLDS = {"overlap_fraction": 0.5, "longest_run": 30.0}

SIGNALS = {
    "overlap_fraction": lambda r: 0.0 if r.total_ngrams == 0 else r.overlap_fraction,
    "longest_run": lambda r: float(r.longest_match_tokens),
}


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


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def run(seeds: int) -> dict[tuple[int, str], dict[str, list[float]]]:
    """Score every (n, signal) pair on each seed."""
    acc: dict[tuple[int, str], dict[str, list[float]]] = {
        (n, signal): {"recall": [], "fpr": [], "precision": [], "overall_auc": [],
                      **{f"auc_{t}": [] for t in TYPES}}
        for n in NGRAM_SIZES
        for signal in SIGNALS
    }

    for seed in range(seeds):
        # One dataset per seed, reused across every n and signal.
        dataset = make_dataset(seed=seed)
        subsets = {t: dataset.subset(t) for t in TYPES}

        for n in NGRAM_SIZES:
            index = CorpusIndex(dataset.corpus, n=n)
            scored = {
                e.example_id: index.overlap_score(e.text, e.example_id)
                for e in dataset.examples
            }
            for signal, extract in SIGNALS.items():
                values = {eid: extract(r) for eid, r in scored.items()}
                bucket = acc[(n, signal)]
                recall, fpr, precision = operating_point(
                    dataset, values, THRESHOLDS[signal]
                )
                bucket["recall"].append(recall)
                bucket["fpr"].append(fpr)
                bucket["precision"].append(precision)
                bucket["overall_auc"].append(auc_for(dataset, values))
                for t in TYPES:
                    bucket[f"auc_{t}"].append(auc_for(subsets[t], values))
    return acc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="number of random seeds to average over (default: 10)",
    )
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    sample = make_dataset(seed=0)
    counts = {t: sum(1 for e in sample.examples if e.contamination_type == t) for t in TYPES}
    clean = sum(1 for e in sample.examples if not e.is_contaminated)
    print(
        f"per seed: {clean} clean, "
        + ", ".join(f"{counts[t]} {t}" for t in TYPES)
        + f" | corpus: {len(sample.corpus)} docs | seeds: {args.seeds}"
    )
    print("AUC 1.0 = perfect ranking, 0.5 = chance. FPR is at the CLI's default threshold.\n")

    acc = run(args.seeds)

    header = (
        f"{'n':>3} {'signal':<16} "
        + " ".join(f"{t[:9]:>9}" for t in TYPES)
        + f" {'overall':>8} {'recall':>15} {'FPR':>15}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n in NGRAM_SIZES:
        for signal in SIGNALS:
            bucket = acc[(n, signal)]
            aucs = [mean_sd(bucket[f"auc_{t}"])[0] for t in TYPES]
            overall = mean_sd(bucket["overall_auc"])[0]
            r_mean, r_sd = mean_sd(bucket["recall"])
            f_mean, f_sd = mean_sd(bucket["fpr"])
            p_mean, _ = mean_sd(bucket["precision"])
            rows.append((n, signal, overall, r_mean, f_mean, p_mean))
            print(
                f"{n:>3} {signal:<16} "
                + " ".join(f"{a:>9.3f}" for a in aucs)
                + f" {overall:>8.3f} "
                + f"{r_mean:.3f} +/- {r_sd:.3f}  "
                + f"{f_mean:.3f} +/- {f_sd:.3f}"
            )
        print()

    # A setting is only usable if it almost never flags clean text; among
    # those, prefer the one that catches the most leakage.
    usable = [r for r in rows if r[4] <= 0.05]
    if usable:
        best = max(usable, key=lambda r: r[3])
        print(f"best usable setting (FPR <= 0.05): n={best[0]} {best[1]}")
        print(f"  recall {best[3]:.3f}, FPR {best[4]:.3f}, precision {best[5]:.3f}")

    best_auc = max(rows, key=lambda r: r[2])
    print(
        f"\nbest raw AUC: n={best_auc[0]} {best_auc[1]} ({best_auc[2]:.3f}) "
        f"- but FPR {best_auc[4]:.3f} at threshold"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
