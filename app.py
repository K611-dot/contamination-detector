"""Streamlit demo for the n-gram overlap contamination detector.

Runs entirely on CPU with no model downloads, so it works on free hosting.
Deploy target: Streamlit Community Cloud.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from contamination_detector.ngram_overlap import CorpusIndex  # noqa: E402
from contamination_detector.report import build_report  # noqa: E402

DEFAULT_BENCHMARK = """The capital of Australia is Canberra, which was purpose built as a compromise between Sydney and Melbourne after federation in nineteen oh one.
A photon travelling through a vacuum moves at a constant speed of roughly three hundred million metres per second regardless of the observer.
In the transformer architecture the attention mechanism computes a weighted sum of value vectors where the weights come from scaled dot products of queries and keys.
Mitochondria generate most of the chemical energy needed to power a cell's biochemical reactions through oxidative phosphorylation of glucose derivatives.
The Great Barrier Reef stretches for over two thousand kilometres along the coast of Queensland and is the largest coral reef system on Earth."""

DEFAULT_CORPUS = """Study notes, mixed sources, uncredited.

The capital of Australia is Canberra, which was purpose built as a compromise between Sydney and Melbourne after federation in nineteen oh one. The city was designed by Walter Burley Griffin following an international competition.

Physics revision: A photon travelling through a vacuum moves at a constant speed of roughly three hundred million metres per second regardless of the observer. This invariance is the founding postulate of special relativity.

Coral reefs are underwater ecosystems built by colonies of tiny animals called polyps. They support enormous biodiversity but are vulnerable to rising sea temperatures.

Neural networks are a family of machine learning models loosely inspired by biological neurons. They are trained by gradient descent on a loss function."""

st.set_page_config(page_title="Benchmark Contamination Detector", page_icon="🔍", layout="wide")

st.title("🔍 Benchmark Contamination Detector")
st.markdown(
    "Check whether your eval examples already appear in a training corpus. "
    "Contaminated benchmarks silently inflate reported model scores — this "
    "flags the leaked examples before you publish."
)

with st.sidebar:
    st.header("Settings")
    ngram_size = st.slider(
        "N-gram size",
        min_value=3,
        max_value=20,
        value=8,
        help="Length of the word sequences matched. Larger = stricter, fewer coincidental matches. "
        "The GPT-3 paper used 13.",
    )
    min_overlap = st.slider(
        "Flag threshold (overlap fraction)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="An example is flagged when at least this fraction of its n-grams appear in the corpus.",
    )
    min_run = st.slider(
        "Flag threshold (contiguous run)",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help="Flag an example that shares this many consecutive words with the corpus, "
        "however small its overall overlap fraction.",
    )
    st.markdown("---")
    st.markdown(
        "**How it works:** every example is split into overlapping n-grams "
        "and checked against the corpus. Two signals come out of that — the "
        "**overlap fraction** (how much of the example appears anywhere in "
        "the corpus) and the **longest contiguous run** of words found "
        "verbatim. The run is the more telling of the two: a long verbatim "
        "stretch is hard to explain innocently, whereas scattered matches "
        "are often just shared phrasing."
    )
    st.markdown("[Source on GitHub](https://github.com/K611-dot/contamination-detector)")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Benchmark examples")
    st.caption("One example per line.")
    benchmark_text = st.text_area(
        "Benchmark examples", value=DEFAULT_BENCHMARK, height=280, label_visibility="collapsed"
    )

with col_right:
    st.subheader("Training corpus")
    st.caption("Paste the text you suspect the examples may have leaked into.")
    corpus_text = st.text_area(
        "Training corpus", value=DEFAULT_CORPUS, height=280, label_visibility="collapsed"
    )

if st.button("Run contamination check", type="primary"):
    examples = [line.strip() for line in benchmark_text.splitlines() if line.strip()]

    if not examples:
        st.warning("Add at least one benchmark example.")
    elif not corpus_text.strip():
        st.warning("Add some corpus text to check against.")
    else:
        index = CorpusIndex([corpus_text], n=ngram_size)
        results = [
            index.overlap_score(text, example_id=f"example_{i + 1}")
            for i, text in enumerate(examples)
        ]
        report = build_report(
            example_ids=[r.example_id for r in results],
            method_scores={"ngram_overlap": [r.overlap_fraction for r in results]},
        )
        for entry, result in zip(report.examples, results):
            entry.flagged = bool(
                (result.scorable and result.overlap_fraction >= min_overlap)
                or result.longest_match_tokens >= min_run
            )

        scorable = [r for r in results if r.scorable]
        unscorable = [r for r in results if not r.scorable]
        flagged_count = sum(1 for e in report.examples if e.flagged)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Examples checked", len(scorable))
        m2.metric("Flagged as contaminated", flagged_count)
        rate = flagged_count / len(scorable) if scorable else 0.0
        m3.metric("Contamination rate", f"{rate:.0%}")

        if flagged_count:
            st.error(
                f"{flagged_count} of {len(scorable)} scorable examples appear in the "
                "corpus. Scores on this benchmark are likely inflated."
            )
        else:
            st.success("No examples exceeded the thresholds.")

        if unscorable:
            st.warning(
                f"{len(unscorable)} example(s) are shorter than the n-gram size "
                f"({ngram_size} words) and could not be scored. That is not evidence "
                "they are clean — lower the n-gram size to include them."
            )

        st.subheader("Per-example results")
        for entry, result, text in zip(report.examples, results, examples):
            if not result.scorable:
                status = "⚪ not scorable"
            elif entry.flagged:
                status = "🚩 CONTAMINATED"
            else:
                status = "✅ clean"
            preview = text if len(text) <= 90 else text[:90] + "…"
            with st.expander(f"{status} — {preview}", expanded=entry.flagged):
                if not result.scorable:
                    st.info(
                        f"Too short to score at n={ngram_size}: an example needs at "
                        f"least {ngram_size} words. Lower the n-gram size to check it."
                    )
                else:
                    st.progress(result.overlap_fraction)
                    c1, c2 = st.columns(2)
                    c1.metric("Overlap fraction", f"{result.overlap_fraction:.1%}")
                    c2.metric("Longest verbatim run", f"{result.longest_match_tokens} words")
                    st.caption(
                        f"{result.matched_ngrams} of {result.total_ngrams} distinct "
                        f"{ngram_size}-grams were found in the corpus."
                    )
                st.text(text)
