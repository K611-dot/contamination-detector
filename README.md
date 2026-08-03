# contamination-detector

A toolkit for researchers to check whether a benchmark or eval set has
leaked into a model's training data. Being built incrementally — see
commit history for progress.

## Why

Benchmark contamination — where eval examples end up in a model's
pretraining or fine-tuning data — quietly inflates reported scores and is
hard to catch after the fact. There's no standard, reusable tool
researchers reach for to check this before publishing results. This
project collects several established detection techniques behind one
interface.

## Status

Early / actively developed.

## License

MIT
