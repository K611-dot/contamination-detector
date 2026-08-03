from contamination_detector.guided_prompt import (
    lcs_scores,
    lcs_similarity,
    run_guided_prompt_test,
    split_prefix_suffix,
)


def test_split_keeps_at_least_one_word_on_each_side():
    prefix, suffix = split_prefix_suffix("one two", split_ratio=0.99)
    assert prefix and suffix


def test_split_single_word_yields_empty_suffix():
    prefix, suffix = split_prefix_suffix("solitary")
    assert prefix == "solitary"
    assert suffix == ""


def test_identical_text_is_fully_similar():
    assert lcs_similarity("alpha beta gamma", "alpha beta gamma") == 1.0


def test_disjoint_text_is_not_similar():
    assert lcs_similarity("alpha beta gamma", "xylophone quokka zeppelin") == 0.0


def test_empty_generation_is_not_similar():
    assert lcs_similarity("alpha beta gamma", "") == 0.0


def test_similarity_ignores_case_and_punctuation():
    assert lcs_similarity("Alpha, beta!", "alpha beta") == 1.0


def test_verbose_padding_is_penalised():
    # The true suffix appears as a subsequence inside a long generic ramble.
    # Recall is perfect, but this is not evidence of memorisation, so the
    # headline score must not be.
    true_suffix = "alpha beta gamma"
    padded = "alpha " + "filler " * 40 + "beta " + "filler " * 40 + "gamma"

    scores = lcs_scores(true_suffix, padded)

    assert scores.recall == 1.0
    assert scores.precision < 0.05
    assert scores.f1 < 0.1
    assert lcs_similarity(true_suffix, padded) == scores.f1


def test_exact_reproduction_beats_padded_reproduction():
    true_suffix = "the mitochondria is the powerhouse of the cell"
    exact = lcs_similarity(true_suffix, true_suffix)
    padded = lcs_similarity(true_suffix, true_suffix + " " + "and so on " * 50)
    assert exact > padded


def test_precision_and_recall_are_reported_separately():
    # Generation covers half the truth, with no padding.
    scores = lcs_scores("alpha beta gamma delta", "alpha beta")
    assert scores.precision == 1.0
    assert scores.recall == 0.5


def test_memorized_completion_scores_higher_than_wrong_completion():
    text = "the mitochondria is the powerhouse of the eukaryotic cell"

    memorized = run_guided_prompt_test("q", text, lambda p: p.split()[-1] + " the eukaryotic cell")
    hallucinated = run_guided_prompt_test("q", text, lambda p: "a fine day for sailing")

    assert memorized.similarity > hallucinated.similarity
    # Echoing the last prefix word costs a little precision, so a near-perfect
    # reproduction lands just under 1.0 rather than at it. What matters is that
    # it still ranks far above an unrelated continuation.
    assert memorized.similarity > 0.8
    assert hallucinated.similarity < 0.2


def test_exact_suffix_reproduction_scores_one():
    text = "the mitochondria is the powerhouse of the eukaryotic cell"
    result = run_guided_prompt_test(
        "q", text, lambda p: " ".join(text.split()[len(p.split()):])
    )
    assert result.similarity == 1.0


def test_result_exposes_precision_and_recall():
    text = "alpha beta gamma delta epsilon zeta"
    result = run_guided_prompt_test("q", text, lambda p: "epsilon zeta", split_ratio=0.66)
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0


def test_completion_fn_receives_only_the_prefix():
    seen = []
    text = "alpha beta gamma delta epsilon"
    result = run_guided_prompt_test("q", text, lambda p: seen.append(p) or "", split_ratio=0.6)
    assert seen == [result.prefix]
    assert result.true_suffix not in result.prefix


def test_lcs_is_symmetric_in_length_handling():
    # Swapping which side is longer must not change the LCS itself.
    a, b = "alpha beta gamma delta", "beta delta"
    assert lcs_scores(a, b).f1 == lcs_scores(b, a).f1
