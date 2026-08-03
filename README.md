# contamination-detector

[![tests](https://github.com/K611-dot/contamination-detector/actions/workflows/tests.yml/badge.svg)](https://github.com/K611-dot/contamination-detector/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Check whether a benchmark leaked into a model's training data — before you publish results on it.**

Benchmark contamination is when eval examples end up inside a model's
pretraining or fine-tuning corpus. The model then partly *recalls*
answers instead of reasoning to them, and the reported score is
inflated. It's a recurring problem: most major benchmarks attract
contamination claims within months of release, usually after the results
have already been cited.

There's no standard tool researchers reach for to check this. This
project puts several published detection methods behind one interface.

---

## Try it without installing anything

The web demo runs the n-gram detector in your browser — paste your eval
examples and a corpus, get a per-example contamination score.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Install

```bash
pip install -e .
```

Core install is dependency-light (numpy only). The model-based detectors
need the optional extra:

```bash
pip install -e ".[hf]"
```

---

## Detection methods

The three methods need different levels of access. Pick based on what
you actually have.

| Method | Needs | Use when |
|---|---|---|
| **N-gram overlap** | Benchmark + a corpus | You can inspect the training data (or a sample of it) |
| **Min-K% / Min-K%++** | Model token log-probs | You have weights or a logprob API, but not the training data |
| **Guided prompting** | Model text generation only | Black-box API access only |

### 1. N-gram overlap

Splits each example into overlapping n-grams and checks how many already
appear in the corpus. An example whose n-grams nearly all appear
verbatim in the corpus is almost certainly leaked. Follows the
contamination check used in the GPT-3 paper (n=13 by default).

```python
from contamination_detector.ngram_overlap import batch_overlap_scores

results = batch_overlap_scores(
    examples={"q1": "the capital of Australia is Canberra ..."},
    corpus_documents=[open("pretrain_sample.txt").read()],
    n=13,
)
for r in results:
    print(r.example_id, r.overlap_fraction)
```

### 2. Min-K% Prob and Min-K%++

Averages the lowest k% of per-token log-probabilities for a text. Text
the model saw in training has fewer very-low-probability tokens, so a
*higher* score suggests membership. **Min-K%++** additionally
standardizes each token against the vocabulary-wide log-prob
distribution at that position, which corrects for tokens that are rare
regardless of memorization — it's the more reliable of the two.

```python
from contamination_detector.min_k_prob import min_k_plus_plus
from contamination_detector.providers.huggingface import HFModelProvider

provider = HFModelProvider("gpt2")
scores = provider.token_scores_for_text("the capital of Australia is Canberra ...")
print(min_k_plus_plus(scores, k_percent=20))
```

These scores are only meaningful **relative to each other**. A single
number tells you nothing; compare a suspect set against a set you know
is clean (e.g. text written after the model's cutoff).

### 3. Guided prompting

Cuts an example into a prefix and a true suffix, asks the model to
continue the prefix, and measures how closely the continuation matches
the real suffix. Near-verbatim reproduction is strong evidence of
memorization. Works through any text-generation endpoint.

```python
from contamination_detector.guided_prompt import run_guided_prompt_test

result = run_guided_prompt_test("q1", example_text, complete_fn=my_api_call)
print(result.similarity)  # 1.0 == model reproduced the suffix exactly
```

---

## CLI

```bash
contam-detect --examples examples/benchmark.jsonl --corpus examples/corpus
```

```
example                overlap       z  flag
---------------------------------------------
q1                       1.000    1.22  CONTAMINATED
q2                       1.000    1.22  CONTAMINATED
q3                       0.000   -0.82
q4                       0.000   -0.82
q5                       0.000   -0.82
```

Examples are JSONL (`{"id": ..., "text": ...}` per line); the corpus is a
`.txt` file or a directory of them. `--json` gives machine-readable
output. Flags on either a z-score outlier or an absolute overlap floor
(`--min-overlap`) — the absolute floor matters because z-scores
under-flag when a large share of the benchmark is contaminated and the
leaked examples become the norm.

## Combining methods

`build_report` standardizes each method's scores, averages them, and
flags outliers. If you have ground-truth labels for a calibration subset,
pass them to get per-method AUC — worth doing, since which detector works
best genuinely varies by model and corpus.

```python
from contamination_detector.report import build_report

report = build_report(
    example_ids=ids,
    method_scores={"ngram_overlap": overlap_scores, "min_k_plus_plus": mink_scores},
    labels=known_labels,   # optional, for AUC calibration
)
print(report.method_auc)
for e in report.most_suspicious(10):
    print(e.example_id, e.combined_zscore, e.flagged)
```

---

## Interpreting results honestly

- **These are signals, not proof.** High overlap on a corpus sample
  means the text is *in that corpus*; it doesn't prove the model trained
  on it. Report it as evidence, not a verdict.
- **Calibrate against a clean control set.** Min-K% and guided-prompting
  scores have no meaningful absolute scale.
- **False positives are real.** Common phrasings, boilerplate, and
  quotations from widely-reproduced sources will overlap innocently.
  Raise `n` if you see this.
- **Absence of a signal isn't absence of contamination.** Paraphrased or
  reformatted leakage defeats n-gram matching entirely.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## References

- Brown et al. (2020), *Language Models are Few-Shot Learners* — n-gram contamination analysis
- Shi et al. (2023), *Detecting Pretraining Data from Large Language Models* — Min-K% Prob
- Zhang et al. (2024), *Min-K%++: Improved Baseline for Detecting Pre-Training Data* — Min-K%++
- Golchin & Surdeanu (2023), *Time Travel in LLMs: Tracing Data Contamination in Large Language Models* — guided prompting

## License

MIT
