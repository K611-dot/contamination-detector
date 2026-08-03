"""Synthetic benchmark for measuring detector accuracy.

Claims about a detector being "better" are only worth anything if they
are measured, so this module builds a labelled dataset with known
contamination and scores detectors against it by AUC.

The hard part of making this honest is the *clean* examples. If clean
text is drawn from a different distribution than the corpus, every
detector scores a perfect AUC and the benchmark measures nothing. Here
clean examples are generated from the same vocabulary and reuse the same
recurring boilerplate phrases as the corpus, so innocent overlap is
present and detectors have to separate real leakage from ordinary shared
phrasing — which is the actual task.

Three kinds of contamination are simulated, because detectors that look
identical on verbatim copies come apart on the others:

- ``verbatim``   — the whole example appears in the corpus
- ``partial``    — only a span of the example appears (a quoted fragment)
- ``paraphrased``— the example appears with a fraction of words swapped
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

DEFAULT_COLLOCATION_POOL = 250
DEFAULT_COLLOCATION_RATE = 0.75


@dataclass
class LabelledExample:
    example_id: str
    text: str
    is_contaminated: bool
    contamination_type: str  # "clean" | "verbatim" | "partial" | "paraphrased"


@dataclass
class SyntheticDataset:
    examples: list[LabelledExample]
    corpus: list[str] = field(default_factory=list)

    @property
    def labels(self) -> list[int]:
        return [int(e.is_contaminated) for e in self.examples]

    def texts(self) -> dict[str, str]:
        return {e.example_id: e.text for e in self.examples}

    def subset(self, contamination_type: str) -> "SyntheticDataset":
        """Keep the clean examples plus one contamination type, for per-type AUC."""
        kept = [
            e
            for e in self.examples
            if not e.is_contaminated or e.contamination_type == contamination_type
        ]
        return SyntheticDataset(examples=kept, corpus=self.corpus)


class _TextGenerator:
    """Phrase-based text generator producing language-like incidental overlap.

    Sampling words independently is not good enough for this benchmark:
    it makes unrelated texts share almost no multi-word sequences, so
    every detector separates leaked from clean text perfectly and the
    benchmark measures nothing. Real prose is built from recurring
    collocations, and *that* is what produces innocent n-gram matches.

    Here text is assembled mostly from a shared pool of 3-6 word
    collocations (Zipf-weighted, so a few are very common), with rare
    filler words in between. Unrelated documents then share short runs
    the way real documents do, and the detectors face genuine
    false-positive pressure.
    """

    def __init__(
        self,
        rng: random.Random,
        vocab_size: int = 3000,
        collocation_pool: int = DEFAULT_COLLOCATION_POOL,
        collocation_rate: float = DEFAULT_COLLOCATION_RATE,
    ):
        self.rng = rng
        self.vocab = [f"w{i}" for i in range(vocab_size)]
        # Zipf weights: a few very common words, a long tail of rare ones.
        self.weights = [1.0 / (i + 1) ** 1.1 for i in range(vocab_size)]
        self.collocation_rate = collocation_rate

        # Collocations are themselves built from the common end of the
        # vocabulary, as real fixed phrases are.
        common = self.vocab[: max(50, vocab_size // 20)]
        common_weights = self.weights[: len(common)]
        self.collocations = [
            rng.choices(common, weights=common_weights, k=rng.randint(3, 6))
            for _ in range(collocation_pool)
        ]
        self.collocation_weights = [
            1.0 / (i + 1) ** 0.9 for i in range(len(self.collocations))
        ]

    def _word(self) -> str:
        return self.rng.choices(self.vocab, weights=self.weights, k=1)[0]

    def sentence(self, length: int) -> str:
        words: list[str] = []
        while len(words) < length:
            if self.rng.random() < self.collocation_rate:
                words.extend(
                    self.rng.choices(
                        self.collocations, weights=self.collocation_weights, k=1
                    )[0]
                )
            else:
                words.append(self._word())
        return " ".join(words[:length])


def make_dataset(
    n_clean: int = 60,
    n_per_contamination: int = 20,
    example_length: int = 60,
    corpus_docs: int = 40,
    corpus_doc_length: int = 400,
    paraphrase_rate: float = 0.25,
    partial_span: int = 20,
    collocation_rate: float = DEFAULT_COLLOCATION_RATE,
    seed: int = 0,
) -> SyntheticDataset:
    """Build a labelled contamination dataset.

    `paraphrase_rate` is the fraction of words replaced in paraphrased
    leaks; `partial_span` is how many words of a partial leak are copied
    into the corpus. `collocation_rate` controls difficulty — it is the
    share of text drawn from the shared phrase pool, so raising it makes
    clean examples overlap the corpus more and the task harder.
    """
    rng = random.Random(seed)
    gen = _TextGenerator(rng, collocation_rate=collocation_rate)

    corpus = [gen.sentence(corpus_doc_length) for _ in range(corpus_docs)]
    examples: list[LabelledExample] = []
    leaked_fragments: list[str] = []

    for i in range(n_clean):
        examples.append(
            LabelledExample(f"clean_{i}", gen.sentence(example_length), False, "clean")
        )

    for i in range(n_per_contamination):
        text = gen.sentence(example_length)
        leaked_fragments.append(text)
        examples.append(LabelledExample(f"verbatim_{i}", text, True, "verbatim"))

    for i in range(n_per_contamination):
        text = gen.sentence(example_length)
        words = text.split()
        start = rng.randrange(0, max(1, len(words) - partial_span))
        leaked_fragments.append(" ".join(words[start : start + partial_span]))
        examples.append(LabelledExample(f"partial_{i}", text, True, "partial"))

    for i in range(n_per_contamination):
        text = gen.sentence(example_length)
        words = text.split()
        mutated = [
            gen.rng.choices(gen.vocab, weights=gen.weights, k=1)[0]
            if rng.random() < paraphrase_rate
            else w
            for w in words
        ]
        leaked_fragments.append(" ".join(mutated))
        examples.append(LabelledExample(f"paraphrased_{i}", text, True, "paraphrased"))

    # Bury the leaked fragments inside otherwise-normal corpus documents.
    for fragment in leaked_fragments:
        idx = rng.randrange(len(corpus))
        host = corpus[idx].split()
        insert_at = rng.randrange(0, len(host))
        host[insert_at:insert_at] = fragment.split()
        corpus[idx] = " ".join(host)

    rng.shuffle(examples)
    return SyntheticDataset(examples=examples, corpus=corpus)
