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
from contamination_detector.report import auc_score, review_queue_size  # noqa: E402

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
    parser.add_argument(
        "--prevalence",
        type=float,
        default=0.01,
        help="assumed real contamination rate for the deployment table (default: 0.01)",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=10_000,
        help="documents in the hypothetical swept corpus (default: 10000)",
    )
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if not 0.0 < args.prevalence < 1.0:
        parser.error("--prevalence must be between 0 and 1")

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

    best_auc = max(rows, key=lambda r: r[2])
    print(
        f"\nbest raw AUC: n={best_auc[0]} {best_auc[1]} ({best_auc[2]:.3f}) "
        f"- but FPR {best_auc[4]:.3f} at threshold"
    )

    # FPR divides by the clean set, which flatters a detector when you sweep
    # a corpus that is almost entirely clean. Precision divides by everything
    # flagged, which is the pile somebody actually has to read.
    print(
        f"\n\nDeployment view: sweeping {args.corpus_size:,} documents at "
        f"{args.prevalence:.0%} real contamination"
    )
    header2 = (
        f"{'n':>3} {'signal':<16} {'flagged':>9} {'real':>7} {'false':>9} "
        f"{'missed':>7} {'precision':>10}"
    )
    print(header2)
    print("-" * len(header2))
    for n, signal, _auc, recall, fpr, _prec in rows:
        q = review_queue_size(args.corpus_size, recall, fpr, args.prevalence)
        print(
            f"{n:>3} {signal:<16} {q['flagged']:>9,.0f} {q['true_positives']:>7,.0f} "
            f"{q['false_positives']:>9,.0f} {q['missed']:>7,.0f} {q['precision']:>10.3f}"
        )
    # Rank on deployment precision, not FPR. Selecting on FPR is the same
    # mistake this table exists to expose: n=5 longest_run looks like the
    # best "usable" setting at FPR 0.018, but it buys a handful of extra
    # true positives with hundreds of false ones.
    deployment = [
        (n, signal, review_queue_size(args.corpus_size, recall, fpr, args.prevalence))
        for n, signal, _auc, recall, fpr, _prec in rows
    ]
    ranked = sorted(
        deployment,
        key=lambda row: (row[2]["precision"], row[2]["true_positives"]),
        reverse=True,
    )
    top_n, top_signal, top_q = ranked[0]
    print(
        f"\nbest at this prevalence: n={top_n} {top_signal} "
        f"- precision {top_q['precision']:.3f}, "
        f"{top_q['true_positives']:,.0f} real leaks found, "
        f"{top_q['false_positives']:,.0f} false flags"
    )
    print(
        "\nA respectable-looking FPR becomes an unreadable review queue once the\n"
        "base rate is realistic. Judge a setting on precision here, not on FPR:\n"
        "selecting on FPR alone would pick a setting that finds a few more real\n"
        "leaks by burying them in hundreds of false ones."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
