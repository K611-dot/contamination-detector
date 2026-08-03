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
    st.markdown("---")
    st.markdown(
        "**How it works:** every example is split into overlapping n-grams "
        "and checked against the corpus's n-gram set. An example whose "
        "n-grams nearly all appear in the corpus is almost certainly leaked."
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
            entry.flagged = result.overlap_fraction >= min_overlap

        flagged_count = sum(1 for e in report.examples if e.flagged)
        contamination_rate = flagged_count / len(examples)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Examples checked", len(examples))
        m2.metric("Flagged as contaminated", flagged_count)
        m3.metric("Contamination rate", f"{contamination_rate:.0%}")

        if flagged_count:
            st.error(
                f"{flagged_count} of {len(examples)} examples appear in the corpus. "
                "Scores on this benchmark are likely inflated."
            )
        else:
            st.success("No examples exceeded the overlap threshold.")

        st.subheader("Per-example results")
        for entry, result, text in zip(report.examples, results, examples):
            status = "🚩 CONTAMINATED" if entry.flagged else "✅ clean"
            preview = text if len(text) <= 90 else text[:90] + "…"
            with st.expander(f"{status} — {preview}", expanded=entry.flagged):
                st.progress(result.overlap_fraction)
                st.write(
                    f"**Overlap:** {result.overlap_fraction:.1%} "
                    f"({result.matched_ngrams} of {result.total_ngrams} n-grams found in corpus)"
                )
                if result.total_ngrams == 0:
                    st.info(
                        f"This example is shorter than the n-gram size ({ngram_size} words), "
                        "so it can't be scored. Lower the n-gram size to check it."
                    )
                st.text(text)
