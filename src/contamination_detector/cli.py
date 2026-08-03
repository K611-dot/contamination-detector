"""Command-line interface for the n-gram overlap contamination check."""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument(
        "--min-run",
        type=int,
        default=30,
        help="contiguous verbatim token run that flags an example on its own",
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

    by_id = {r.example_id: r for r in results}
    for entry in report.examples:
        result = by_id[entry.example_id]
        # z-scores are relative, so they under-flag when a large share of the
        # benchmark is contaminated (the contaminated examples become the norm).
        # An absolute overlap floor catches that case.
        if result.scorable and result.overlap_fraction >= args.min_overlap:
            entry.flagged = True
        # A long verbatim run is strong evidence on its own, even when the
        # fraction is low because the example is much longer than the leak.
        if result.longest_match_tokens >= args.min_run:
            entry.flagged = True

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": e.example_id,
                        "overlap_fraction": by_id[e.example_id].overlap_fraction,
                        "longest_match_tokens": by_id[e.example_id].longest_match_tokens,
                        "zscore": e.combined_zscore,
                        "scorable": by_id[e.example_id].scorable,
                        "flagged": e.flagged,
                    }
                    for e in report.examples
                ],
                indent=2,
            )
        )
    else:
        print(f"{'example':<20} {'overlap':>9} {'run':>5} {'z':>7}  flag")
        print("-" * 52)
        ordered = sorted(
            report.examples,
            key=lambda x: (
                -1e9 if math.isnan(x.combined_zscore) else x.combined_zscore
            ),
            reverse=True,
        )
        for e in ordered:
            result = by_id[e.example_id]
            if not result.scorable:
                print(
                    f"{e.example_id:<20} {'--':>9} {'--':>5} {'--':>7}  "
                    f"too short for n={args.ngram_size}"
                )
                continue
            flag = "CONTAMINATED" if e.flagged else ""
            print(
                f"{e.example_id:<20} {result.overlap_fraction:>9.3f} "
                f"{result.longest_match_tokens:>5d} {e.combined_zscore:>7.2f}  {flag}"
            )

        unscorable = [r for r in results if not r.scorable]
        if unscorable:
            print(
                f"\nNote: {len(unscorable)} example(s) were shorter than the n-gram "
                f"size and could not be scored. That is not evidence they are clean; "
                f"lower --ngram-size to include them.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
