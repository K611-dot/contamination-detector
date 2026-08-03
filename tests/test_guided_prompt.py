from contamination_detector.guided_prompt import (
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


def test_similarity_ignores_case_and_punctuation():
    assert lcs_similarity("Alpha, beta!", "alpha beta") == 1.0


def test_memorized_completion_scores_higher_than_wrong_completion():
    text = "the mitochondria is the powerhouse of the eukaryotic cell"

    memorized = run_guided_prompt_test("q", text, lambda p: "of the eukaryotic cell")
    hallucinated = run_guided_prompt_test("q", text, lambda p: "a fine day for sailing")

    assert memorized.similarity > hallucinated.similarity
    assert memorized.similarity == 1.0


def test_completion_fn_receives_only_the_prefix():
    seen = []
    text = "alpha beta gamma delta epsilon"
    result = run_guided_prompt_test("q", text, lambda p: seen.append(p) or "", split_ratio=0.6)
    assert seen == [result.prefix]
    assert result.true_suffix not in result.prefix
