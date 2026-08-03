# contamination-detector

[![tests](https://github.com/K611-dot/contamination-detector/actions/workflows/tests.yml/badge.svg)](https://github.com/K611-dot/contamination-detector/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![live demo](https://img.shields.io/badge/live%20demo-streamlit-ff4b4b)](https://contamination-detector.streamlit.app/)

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

### → [contamination-detector.streamlit.app](https://contamination-detector.streamlit.app/)

Paste your eval examples and a corpus, get a per-example contamination
score with the overlap fraction and longest verbatim run. Nothing to
install, no model downloads, and pasted text is processed in memory
only — never stored or transmitted.

The hosted demo caps input size so one visitor can't exhaust the shared
free-tier container. For full-size corpora, run it locally:

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

Splits each example into overlapping n-grams and checks which ones
already appear in the corpus. Follows the contamination check used in the
GPT-3 paper (n=13 by default). Two signals come out of it, and they
answer different questions:

- **`overlap_fraction`** — how much of the example appears in the corpus
  at all. Sensitive, but scattered matches on ordinary phrasing inflate it.
- **`longest_match_tokens`** — the longest *contiguous* run of words found
  verbatim. Much more diagnostic: a 40-word verbatim stretch is hard to
  explain innocently, and a short leak buried in a long example barely
  moves the fraction while showing up clearly here.

Use the run length to judge a single example; use the fraction to compare
examples against each other.

```python
from contamination_detector.ngram_overlap import batch_overlap_scores

results = batch_overlap_scores(
    examples={"q1": "the capital of Australia is Canberra ..."},
    corpus_documents=[open("pretrain_sample.txt").read()],
    n=13,
)
for r in results:
    print(r.example_id, r.overlap_fraction, r.longest_match_tokens)
```

`corpus_documents` accepts any iterable, so a corpus too big for memory
can be streamed a document at a time. The index stores 64-bit hashes
rather than the n-grams themselves, which measured 2.5x smaller
(77.6 MB → 31.0 MB for 400k 13-grams).

Examples shorter than `n` return `overlap_fraction = NaN` and
`scorable = False` — they cannot be checked, which is **not** the same as
being clean, and keeping them distinct stops them quietly counting as
clean results.

### 2. Min-K% Prob and Min-K%++

Averages the lowest k% of per-token log-probabilities for a text. Text
the model saw in training has fewer very-low-probability tokens, so a
*higher* score suggests membership. **Min-K%++** calibrates each token
against the model's own next-token distribution first, which corrects for
tokens that are unlikely regardless of memorization — it's the more
reliable of the two.

```python
from contamination_detector.min_k_prob import min_k_plus_plus
from contamination_detector.providers.huggingface import HFModelProvider

provider = HFModelProvider("gpt2")
scores = provider.token_scores_for_text("the capital of Australia is Canberra ...")
print(min_k_plus_plus(scores, k_percent=20))
```

If you are wiring up your own provider, build token scores with
`TokenScore.from_log_distribution(log_probs, token_id)` rather than
filling the fields by hand. The calibration statistics are expectations
**under the model's own probability distribution** (μ is the negative
entropy), not unweighted averages over the vocabulary vector — an easy
detail to get wrong, and getting it wrong quietly destroys the method. On
a 50k-vocab Zipf distribution the unweighted mean gives μ = −13.40 against
a true −5.40 and under-disperses σ threefold, which pushes every
calibrated score into a narrow positive band and strips out the sign that
separates "likelier than the model expected" from "less likely".

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
print(result.similarity)                    # LCS F1; 1.0 == exact reproduction
print(result.precision, result.recall)      # the two failure modes, separately
```

`similarity` is an LCS **F1**, not plain recall. Normalizing by the true
suffix alone measures only how much of the truth was reproduced, so a
verbose model that emits a few hundred generic words containing the
suffix as a subsequence scores a perfect 1.0 by accident. Precision
penalizes that padding — on the regression case in the test suite, recall
stays at 1.0 while F1 falls below 0.1.

---

## CLI

```bash
contam-detect --examples examples/benchmark.jsonl --corpus examples/corpus
```

```
example                overlap   run       z  flag
----------------------------------------------------
q1                       1.000    23    1.22  CONTAMINATED
q2                       1.000    23    1.22  CONTAMINATED
q3                       0.000     0   -0.82
q4                       0.000     0   -0.82
q5                       0.000     0   -0.82
```

Examples are JSONL (`{"id": ..., "text": ...}` per line); the corpus is a
`.txt` file or a directory of them. `--json` gives machine-readable
output. An example is flagged on any of three grounds:

- a z-score outlier (`--threshold`)
- an absolute overlap floor (`--min-overlap`) — needed because z-scores
  under-flag when much of the benchmark is contaminated and the leaked
  examples *become* the norm
- a contiguous verbatim run (`--min-run`) — catches a short leak inside a
  long example, where the overall fraction stays low

Examples too short to score are listed separately rather than printed as
clean.

## Combining methods

`build_report` standardizes each method's scores, combines them, and
flags outliers. If you have ground-truth labels for a calibration subset,
pass them to get per-method AUC — worth doing, since which detector works
best genuinely varies by model and corpus.

```python
from contamination_detector.report import build_report, MethodDirection

report = build_report(
    example_ids=ids,
    method_scores={
        "ngram_overlap": overlap_scores,
        "min_k_plus_plus": mink_scores,
        "perplexity": ppl_scores,
    },
    directions={"perplexity": MethodDirection.LOWER_IS_CONTAMINATED},
    labels=known_labels,    # optional, enables AUC
    weight_by_auc=True,     # weight methods by measured separation
)
print(report.method_auc, report.method_weights)
for e in report.most_suspicious(10):
    print(e.example_id, e.combined_zscore, e.flagged)
```

Two things worth knowing here. **Declare direction** for any method where
a lower score means more contaminated (perplexity is the common one) —
combining it without that quietly cancels real signal against the methods
that point the other way. And **`weight_by_auc`** stops a method that is
near-random on your data from diluting one that works; a method at or
below chance gets weight zero rather than a vote.

---

## Measured accuracy

Claims that one detector beats another are worth nothing unless they're
measured, so the repo ships a labelled synthetic benchmark and a script
that scores against it:

```bash
python scripts/benchmark_detectors.py
```

It plants three kinds of leakage — `verbatim` (whole example in the
corpus), `partial` (only a span), and `paraphrased` (a quarter of the
words swapped) — and reports AUC per type plus recall and false-positive
rate at the CLI's default thresholds.

Averaged over 10 seeds (60 clean / 60 contaminated examples each):

| Setting | Recall | FPR |
|---|---|---|
| n=5, longest run ≥ 30 | **0.400** ± 0.019 | 0.018 |
| n=8, longest run ≥ 30 | 0.338 ± 0.008 | 0.000 |
| n=13, overlap ≥ 0.5 | 0.333 ± 0.000 | 0.000 |
| n=5, overlap ≥ 0.5 | 0.945 ± 0.045 | **0.457** |

Three things this makes concrete:

- **Verbatim leakage is easy** — AUC 1.000 at every setting. Everything
  interesting happens on partial and paraphrased leaks.
- **The contiguous-run signal is what makes a small `n` usable.** At n=5
  the overlap fraction flags 46% of clean text; the run length at the same
  `n` flags ~2% while catching *more* leakage than the n=13 default.
- **Ranking quality alone will mislead you.** Across all eight settings
  AUC spans just 0.838–0.890, while false-positive rate spans
  0.000–0.457. n=5 on overlap fraction sits mid-pack on AUC (0.869) and
  is by far the least usable of the eight, because its score
  distributions overlap so heavily that no threshold separates them. AUC
  measures ordering and barely distinguishes these settings; FPR at your
  actual threshold is what decides whether you can act on a flag.

The defaults are deliberately **not** tuned to win on this benchmark. It's
synthetic, and fitting the defaults to a generator I wrote myself is
exactly the kind of overfitting this tool exists to catch. Treat the
numbers as relative comparisons between methods, not as expected
performance on real corpora.

Building the benchmark honestly took two attempts, which is itself worth
knowing: the first version sampled words independently, so unrelated texts
shared almost no phrases, every detector scored a perfect AUC, and the
benchmark measured nothing at all. Real prose is built from recurring
collocations and *that* is what creates innocent n-gram matches. Text is
now assembled from a shared phrase pool, clean examples reach 0.82 overlap
at n=5, and a test asserts that difficulty so it can't silently regress.

## Interpreting results honestly

- **These are signals, not proof.** High overlap on a corpus sample
  means the text is *in that corpus*; it doesn't prove the model trained
  on it. Report it as evidence, not a verdict.
- **Calibrate against a clean control set.** Min-K% and guided-prompting
  scores have no meaningful absolute scale.
- **False positives are real.** Common phrasings, boilerplate, and
  quotations from widely-reproduced sources will overlap innocently.
  Raise `n`, or lean on the contiguous-run signal, if you see this.
- **Absence of a signal isn't absence of contamination.** Paraphrased
  leakage defeats n-gram matching: on the benchmark above it drops to
  AUC 0.577 at n=13 and to chance (0.500) at n=20. If you suspect
  reformatted leakage, you need one of the model-based methods.
- **An unscorable example is not a clean one.** Examples shorter than `n`
  are reported separately for exactly this reason.

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
