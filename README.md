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
  `n` flags ~2%.
- **Ranking quality alone will mislead you.** Across all eight settings
  AUC spans just 0.838–0.890, while false-positive rate spans
  0.000–0.457. n=5 on overlap fraction sits mid-pack on AUC (0.869) and
  is by far the least usable of the eight, because its score
  distributions overlap so heavily that no threshold separates them.

### FPR is the wrong denominator

Even false-positive rate flatters a detector, because it divides by the
clean set. When you sweep a corpus that is almost entirely clean, what
matters is precision — the fraction of your *flagged* pile that is real,
because that pile is what somebody has to read. The benchmark reports
this directly:

```bash
python scripts/benchmark_detectors.py --prevalence 0.01
```

Sweeping 10,000 documents at 1% real contamination:

| Setting | Flagged | Real | False | Missed | Precision |
|---|---|---|---|---|---|
| n=5, overlap | 4,616 | 94 | 4,521 | 6 | **0.020** |
| n=5, longest run | 222 | 40 | 182 | 60 | 0.181 |
| n=8, overlap | 70 | 37 | 33 | 63 | 0.527 |
| n=8, longest run | 34 | 34 | 0 | 66 | **1.000** |
| n=13, overlap | 33 | 33 | 0 | 67 | 1.000 |

This reverses a conclusion an earlier version of this README drew. Judged
on FPR, n=5 with the run signal looked like the best usable setting — 0.018
FPR and higher recall than the n=13 default. Judged at realistic
prevalence it finds 7 more real leaks than n=13 and pays 182 false flags
for them. A 46% FPR isn't a noisy signal, it's a review queue nobody
finishes.

`report.precision_at_prevalence()` and `report.review_queue_size()`
compute this for your own numbers. Quote precision at your expected base
rate, not FPR, whenever you report deployment performance.

Credit to [Giulianno Vollmer](https://www.linkedin.com/in/giuliannovollmer/)
for pointing out that FPR understates the deployment cost, and for the two
features in the next section.

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

## Per-source scoring and calibration

Two problems with treating a corpus as one undifferentiated blob.

**Provenance.** A match against a training split and a match against a
held-out test split are different findings with different remedies, and
collapsing both into "contaminated" makes the report unactionable.
`MultiSourceIndex` keeps them apart:

```python
from contamination_detector.ngram_overlap import MultiSourceIndex

index = MultiSourceIndex({"train": train_docs, "test": test_docs}, n=13)
result = index.overlap_score(example_text, example_id="q1")

print(result.best_source)          # "test" — where the longest run sits
print(result.matched_sources())    # every source with a verbatim run
print(result.per_source["train"].longest_match_tokens)
```

**Calibration.** Innocent overlap is not stationary across sources.
Templated math problems, boilerplate scrape, and licence text throw
near-duplicate n-grams for reasons unrelated to leakage; ordinary prose
does not. One global threshold over-flags the formulaic sources into
uselessness and quietly under-flags the prose ones.

So don't guess a constant — measure each source's own null distribution.
Score control text you know is *outside* the source (written after the
cutoff, or held-out data the source predates), and take a high percentile
of those scores as that source's threshold:

```python
from contamination_detector.calibration import calibrate_sources

thresholds = calibrate_sources(index, control_texts, percentile_value=99.0)

for source, t in thresholds.items():
    print(source, t.longest_run, t.is_well_estimated)

result = index.overlap_score(suspect_text)
leaked = [
    name for name, match in result.per_source.items()
    if match.longest_match_tokens >= thresholds[name].longest_run
]
```

Whatever overlap innocent text produces against that particular source,
the threshold sits above it by construction. Check `is_well_estimated` —
a 99th percentile from ten controls is guesswork, and it will tell you so.
This is only as good as your control set: controls from a different domain
calibrate the wrong distribution.

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
