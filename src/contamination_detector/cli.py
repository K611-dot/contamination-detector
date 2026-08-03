"""Command-line interface for the n-gram overlap contamination check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .ngram_overlap import batch_overlap_scores
from .report import build_report


def _load_examples(path: Path) -> dict[str, str]:
    """Load benchmark examples from JSONL ({"id": ..., "text": ...} per line)."""
    examples: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            eid = str(record.get("id", line_no))
            examples[eid] = record["text"]
    return examples


def _load_corpus(path: Path) -> list[str]:
    """Load corpus documents: one .txt file, or a directory of .txt files."""
    if path.is_dir():
        return [p.read_text(encoding="utf-8") for p in sorted(path.glob("*.txt"))]
    return [path.read_text(encoding="utf-8")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="contam-detect",
        description="Check whether benchmark examples leaked into a reference corpus.",
    )
    parser.add_argument("--examples", required=True, type=Path, help="JSONL of benchmark examples")
    parser.add_argument("--corpus", required=True, type=Path, help=".txt file or directory of .txt")
    parser.add_argument("-n", "--ngram-size", type=int, default=13, help="n-gram size (default 13)")
    parser.add_argument(
        "--threshold", type=float, default=2.0, help="z-score threshold for flagging"
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.5,
        help="absolute overlap fraction that flags an example regardless of z-score",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    examples = _load_examples(args.examples)
    corpus = _load_corpus(args.corpus)
    if not examples:
        print("No examples found.", file=sys.stderr)
        return 1

    results = batch_overlap_scores(examples, corpus, n=args.ngram_size)
    report = build_report(
        example_ids=[r.example_id for r in results],
        method_scores={"ngram_overlap": [r.overlap_fraction for r in results]},
        flag_threshold=args.threshold,
    )

    # z-scores are relative, so they under-flag when a large share of the
    # benchmark is contaminated (the contaminated examples become the norm).
    # An absolute overlap floor catches that case.
    for entry in report.examples:
        if entry.method_scores["ngram_overlap"] >= args.min_overlap:
            entry.flagged = True

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": e.example_id,
                        "overlap_fraction": e.method_scores["ngram_overlap"],
                        "zscore": e.combined_zscore,
                        "flagged": e.flagged,
                    }
                    for e in report.examples
                ],
                indent=2,
            )
        )
    else:
        print(f"{'example':<20} {'overlap':>9} {'z':>7}  flag")
        print("-" * 45)
        for e in sorted(report.examples, key=lambda x: x.combined_zscore, reverse=True):
            flag = "CONTAMINATED" if e.flagged else ""
            print(
                f"{e.example_id:<20} {e.method_scores['ngram_overlap']:>9.3f} "
                f"{e.combined_zscore:>7.2f}  {flag}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
