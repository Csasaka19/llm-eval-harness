import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from llm_eval.calibration import (
    CalibrationError,
    LabelledClaim,
    calibrate,
    cohens_kappa,
    load_labels,
    parse_labels,
)
from llm_eval.cli import main
from llm_eval.judges import ClaimVerdict, Judge, load_judge
from llm_eval.judges.overlap import TokenOverlapJudge, content_tokens
from llm_eval.metrics.faithfulness import judge_answer, split_claims
from llm_eval.scoring import score_run
from llm_eval.types import CaseResult, EvalCase, SystemResponse

EXAMPLES = Path(__file__).parent.parent / "examples"
CTX = ["The Eiffel Tower in Paris was completed in 1889 for the World's Fair."]


class KeywordJudge:
    """Test double: supported iff the claim contains 'yes'."""

    name = "keyword"

    def judge_claim(self, claim: str, contexts: Sequence[str]) -> ClaimVerdict:
        return ClaimVerdict(claim, "yes" in claim)


# --- claim splitting -------------------------------------------------------------


def test_split_claims() -> None:
    answer = "It opened in 1889. It is 330 m tall! Is it in Paris?\n- Bullet one\n2) Two"
    assert split_claims(answer) == [
        "It opened in 1889.",
        "It is 330 m tall!",
        "Is it in Paris?",
        "Bullet one",
        "Two",
    ]


def test_split_claims_keeps_decimals() -> None:
    assert split_claims("Growth was 3.5% this year.") == ["Growth was 3.5% this year."]


def test_split_claims_empty() -> None:
    assert split_claims("  \n ") == []


# --- overlap judge ---------------------------------------------------------------


def test_content_tokens_drop_stopwords_keep_numbers() -> None:
    assert content_tokens("The tower is 1,083 ft and 3.5 m.") == [
        "tower",
        "1,083",
        "ft",
        "3.5",
        "m",
    ]


def test_overlap_judge_supported_and_unsupported() -> None:
    judge = TokenOverlapJudge()
    assert judge.judge_claim("The Eiffel Tower was completed in 1889.", CTX).supported
    assert not judge.judge_claim("Bananas are yellow.", CTX).supported


def test_overlap_judge_requires_numbers() -> None:
    verdict = TokenOverlapJudge().judge_claim("The Eiffel Tower was completed in 1887.", CTX)
    assert not verdict.supported
    assert verdict.rationale == "numbers not in context: 1887"


def test_overlap_judge_known_blind_spot_negation() -> None:
    # Documented limitation: negation is invisible to token overlap.
    verdict = TokenOverlapJudge().judge_claim("The Eiffel Tower was not completed in 1889.", CTX)
    assert verdict.supported


def test_overlap_judge_threshold_validation() -> None:
    with pytest.raises(ValueError, match="threshold"):
        TokenOverlapJudge(threshold=0)


def test_load_judge() -> None:
    assert isinstance(load_judge("overlap"), TokenOverlapJudge)
    assert isinstance(load_judge("tests.test_faithfulness:KeywordJudge"), Judge)
    with pytest.raises(TypeError, match="Judge protocol"):
        load_judge("pathlib:PurePath")


# --- faithfulness metric ---------------------------------------------------------


def test_judge_answer_fraction_supported() -> None:
    scores, verdicts = judge_answer("yes one. no two. yes three. no four.", [], KeywordJudge())
    assert scores == {"faithfulness": 0.5, "n_claims": 4.0}
    assert [v.supported for v in verdicts] == [True, False, True, False]


def test_empty_answer_is_vacuously_faithful_but_shows_zero_claims() -> None:
    scores, verdicts = judge_answer("", CTX, KeywordJudge())
    assert scores == {"faithfulness": 1.0, "n_claims": 0.0}
    assert verdicts == ()


def test_scoring_which_cases_get_faithfulness() -> None:
    cases = [
        EvalCase("rag", "q", expected_doc_ids=("d",)),
        EvalCase("rag-crash", "q", expected_doc_ids=("d",)),
        EvalCase("unlabelled-with-ctx", "q"),
        EvalCase("plain", "q"),
        EvalCase("plain-crash", "q"),
    ]
    results = [
        CaseResult("rag", 1.0, SystemResponse("yes. no.", retrieved_contexts=("c",))),
        CaseResult("rag-crash", 1.0, error="Boom"),
        CaseResult("unlabelled-with-ctx", 1.0, SystemResponse("yes.", retrieved_contexts=("c",))),
        CaseResult("plain", 1.0, SystemResponse("no.")),
        CaseResult("plain-crash", 1.0, error="Boom"),
    ]
    card = score_run(cases, results, judge=KeywordJudge())
    by_id = {c.case_id: c for c in card.cases}
    assert by_id["rag"].faithfulness == {"faithfulness": 0.5, "n_claims": 2.0}
    assert by_id["rag"].claims is not None
    assert len(by_id["rag"].claims) == 2
    assert by_id["rag-crash"].faithfulness == {"faithfulness": 0.0, "n_claims": 0.0}
    assert by_id["unlabelled-with-ctx"].faithfulness == {"faithfulness": 1.0, "n_claims": 1.0}
    assert by_id["plain"].faithfulness is None
    assert by_id["plain-crash"].faithfulness is None

    agg = card.faithfulness["faithfulness"]
    assert (agg.n, agg.n_errored) == (3, 1)
    assert agg.mean == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert card.judge == "keyword"


def test_scoring_without_judge_skips_faithfulness() -> None:
    card = score_run(
        [EvalCase("a", "q", expected_doc_ids=("d",))],
        [CaseResult("a", 1.0, SystemResponse("x", retrieved_contexts=("c",)))],
    )
    assert card.faithfulness == {}
    assert card.judge is None


# --- calibration -----------------------------------------------------------------


def test_cohens_kappa_hand_computed() -> None:
    # observed = 3/4; p_h = 0.5, p_j = 0.25; expected = .5*.25 + .5*.75 = 0.5
    # kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5
    assert cohens_kappa([True, True, False, False], [True, False, False, False]) == 0.5


def test_cohens_kappa_extremes() -> None:
    assert cohens_kappa([True, False], [True, False]) == 1.0
    assert cohens_kappa([True, False], [False, True]) == -1.0
    # A constant judge on a balanced set: observed 0.5, expected 0.5 -> 0.
    assert cohens_kappa([True, False, True, False], [True] * 4) == 0.0
    # Both raters constant and identical: chance agreement is 1, kappa undefined.
    assert cohens_kappa([True, True], [True, True]) is None


def test_cohens_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([], [])
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])


def test_accuracy_can_hide_a_useless_judge() -> None:
    # 4 of 5 human labels are "supported"; an always-"supported" judge gets 80% accuracy
    # but kappa 0 and catches none of the unsupported claims.
    class AlwaysSupported:
        name = "always"

        def judge_claim(self, claim: str, contexts: Sequence[str]) -> ClaimVerdict:
            return ClaimVerdict(claim, True)

    labelled = [LabelledClaim(f"c{i}", "x", (), supported=i < 4) for i in range(5)]
    report = calibrate(AlwaysSupported(), labelled)
    assert report.accuracy == 0.8
    assert report.cohens_kappa == 0.0
    assert report.unsupported_recall == 0.0
    assert report.unsupported_precision is None  # judge never says "unsupported"


def test_calibrate_confusion_matrix_and_disagreements() -> None:
    labelled = [
        LabelledClaim("a", "yes", (), supported=True),
        LabelledClaim("b", "no", (), supported=True, note="paraphrase"),
        LabelledClaim("c", "yes", (), supported=False),
        LabelledClaim("d", "no", (), supported=False),
    ]
    r = calibrate(KeywordJudge(), labelled)
    assert (r.both_supported, r.human_supported_judge_unsupported) == (1, 1)
    assert (r.human_unsupported_judge_supported, r.both_unsupported) == (1, 1)
    assert r.accuracy == 0.5
    assert r.cohens_kappa == 0.0
    assert r.unsupported_recall == 0.5
    assert [d.id for d in r.disagreements] == ["b", "c"]
    assert r.disagreements[0].note == "paraphrase"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"claim": "x", "contexts": [], "label": "supported"}', "'id'"),
        ('{"id": "a", "contexts": [], "label": "supported"}', "'claim'"),
        ('{"id": "a", "claim": "x", "contexts": [], "label": "maybe"}', "'label'"),
        ('{"id": "a", "claim": "x", "contexts": "c", "label": "supported"}', "'contexts'"),
        ('{"id": "a", "claim": "x", "contexts": [], "label": "supported", "note": 1}', "'note'"),
    ],
)
def test_parse_labels_validation(line: str, message: str) -> None:
    with pytest.raises(CalibrationError, match=message):
        parse_labels([line])


def test_parse_labels_rejects_duplicates_and_empty() -> None:
    row = '{"id": "a", "claim": "x", "contexts": [], "label": "supported"}'
    with pytest.raises(CalibrationError, match="duplicate"):
        parse_labels([row, row])
    with pytest.raises(CalibrationError, match="no labelled claims"):
        parse_labels(["// only a comment"])


def test_example_calibration_set_loads() -> None:
    labelled = load_labels(EXAMPLES / "faithfulness_calibration.jsonl")
    assert len(labelled) == 24
    assert {item.supported for item in labelled} == {True, False}


def test_calibrate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cal.json"
    labels = str(EXAMPLES / "faithfulness_calibration.jsonl")
    assert main(["calibrate", "--judge", "overlap", "--labels", labels, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "cohen's kappa" in printed
    assert "disagreements" in printed
    report = json.loads(out.read_text())
    assert report["n"] == 24
    assert report["judge"] == "overlap(threshold=0.75)"


def test_score_command_with_judge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset, out = tmp_path / "d.jsonl", tmp_path / "run"
    dataset.write_text('{"id": "a", "input": "q", "expected_doc_ids": ["d1"]}\n')
    assert main(["run", "--dataset", str(dataset), "--system", "echo", "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["score", str(out), "--judge", "overlap"]) == 0
    printed = capsys.readouterr().out
    assert "faithfulness (judge=overlap(threshold=0.75), n=1" in printed
    # Echo answers "q" with no retrieved context, so its one claim is unsupported.
    scores = json.loads((out / "scores.json").read_text())
    assert scores["faithfulness"]["faithfulness"]["mean"] == 0.0
    assert scores["cases"][0]["claims"][0]["supported"] is False
