import pytest

from llm_eval.metrics.correctness import correctness_scores, normalize_tokens
from llm_eval.scoring import score_run
from llm_eval.types import CaseResult, EvalCase, SystemResponse


def test_normalize_keeps_numbers_intact() -> None:
    assert normalize_tokens("The drone carries 1.8 kg!") == ["drone", "carries", "1.8", "kg"]
    assert normalize_tokens("Budget: 1,000 euros") == ["budget", "1000", "euros"]
    # A comma that is not a thousands separator splits tokens.
    assert normalize_tokens("1,5") == ["1", "5"]
    assert normalize_tokens("An apple, a pear.") == ["apple", "pear"]


def test_exact_match_after_normalisation() -> None:
    assert correctness_scores("The K-200.", "k 200")["exact_match"] == 1.0
    assert correctness_scores("42 minutes", "42 minutes long")["exact_match"] == 0.0


def test_token_f1_and_recall_hand_computed() -> None:
    # pred = [flight, time, is, 42, minutes] (5), gold = [42, minutes] (2), common = 2
    # precision = 2/5, recall = 1, f1 = 2 * 0.4 * 1 / 1.4 = 4/7
    s = correctness_scores("Flight time is 42 minutes", "42 minutes")
    assert s["token_recall"] == 1.0
    assert s["token_f1"] == pytest.approx(4 / 7)


def test_wrong_number_gets_no_credit_for_it() -> None:
    s = correctness_scores("1.8 kg", "18 kg")
    assert s["token_recall"] == 0.5  # "kg" matches, "1.8" != "18"


def test_repeated_tokens_count_once_per_occurrence() -> None:
    # pred = [yes, yes, yes], gold = [yes]; common = 1 -> p = 1/3, r = 1, f1 = 0.5
    s = correctness_scores("yes yes yes", "yes")
    assert s["token_f1"] == pytest.approx(0.5)


def test_empty_answer_or_reference() -> None:
    assert correctness_scores("", "42") == {
        "exact_match": 0.0,
        "token_f1": 0.0,
        "token_recall": 0.0,
    }
    assert correctness_scores("", "") == {"exact_match": 1.0, "token_f1": 1.0, "token_recall": 1.0}


def test_known_limitation_yes_no_references() -> None:
    # Correct in meaning, but shares no tokens with the reference "Yes".
    s = correctness_scores("Premium economy is allowed for flights of 6 hours or longer.", "Yes")
    assert s["token_recall"] == 0.0


def test_scoring_only_labelled_cases_and_errors_zero() -> None:
    cases = [
        EvalCase("a", "q", expected_answer="42 minutes"),
        EvalCase("b", "q", expected_answer="5 kg"),
        EvalCase("c", "q"),
    ]
    results = [
        CaseResult("a", 1.0, SystemResponse("42 minutes")),
        CaseResult("b", 1.0, error="Boom"),
        CaseResult("c", 1.0, SystemResponse("anything")),
    ]
    card = score_run(cases, results)
    by_id = {c.case_id: c for c in card.cases}
    assert by_id["c"].correctness is None
    assert by_id["b"].correctness == {"exact_match": 0.0, "token_f1": 0.0, "token_recall": 0.0}
    em = card.correctness["exact_match"]
    assert (em.mean, em.n, em.n_errored) == (0.5, 2, 1)
