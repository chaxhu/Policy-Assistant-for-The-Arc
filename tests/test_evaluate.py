from collections import Counter

import pytest
from conftest import make_answer

from arc_assistant.config import Settings
from arc_assistant.evaluate import (
    list_runs,
    load_run,
    load_test_set,
    run_evaluation,
    save_run,
    score_case,
    summarise,
)
from arc_assistant.models import EvalCase, JudgeVerdict


class StubJudge:
    def __init__(self, correct: bool = True) -> None:
        self.correct = correct
        self.calls = 0

    def __call__(self, case: EvalCase, text: str) -> JudgeVerdict:
        self.calls += 1
        return JudgeVerdict(correct=self.correct, reason="stub")


def case(**overrides) -> EvalCase:
    fields = {
        "id": "c1",
        "category": "answerable_fictional",
        "question": "Can drivers buy AdBlue?",
        "expected_behaviour": "answer",
        "expected_doc_ids": ["card-usage"],
        "must_include": ["AdBlue"],
    }
    fields.update(overrides)
    return EvalCase(**fields)


def test_answerable_passes_when_all_checks_and_judge_pass() -> None:
    judge = StubJudge(True)
    result = score_case(
        case(), make_answer(text="Yes, AdBlue is allowed.", cited=["card-usage"]), judge
    )
    assert result.passed
    assert result.citation_ok and result.keywords_ok
    assert judge.calls == 1


def test_wrong_document_fails_without_calling_judge() -> None:
    judge = StubJudge(True)
    result = score_case(case(), make_answer(text="AdBlue yes.", cited=["ev-charging"]), judge)
    assert not result.passed
    assert result.citation_ok is False
    assert judge.calls == 0


def test_missing_keyword_fails() -> None:
    result = score_case(case(), make_answer(text="Yes.", cited=["card-usage"]), StubJudge())
    assert not result.passed
    assert "AdBlue" in result.reason


def test_judge_can_fail_an_answer() -> None:
    result = score_case(
        case(), make_answer(text="AdBlue: no.", cited=["card-usage"]), StubJudge(False)
    )
    assert not result.passed
    assert result.reason.startswith("Judge:")


def test_unexpected_handoff_fails() -> None:
    answer = make_answer(status="handoff", handoff_code="score_gate")
    result = score_case(case(), answer, StubJudge())
    assert not result.passed and not result.behaviour_ok


def test_out_of_scope_passes_on_handoff_without_judge() -> None:
    judge = StubJudge()
    oos = case(
        category="out_of_scope", expected_behaviour="handoff", expected_doc_ids=[], must_include=[]
    )
    assert score_case(
        oos, make_answer(status="handoff", handoff_code="model_handoff"), judge
    ).passed
    assert not score_case(oos, make_answer(text="Your limit is £5,000."), judge).passed
    assert judge.calls == 0


def test_superseded_trap_fails_on_old_rule() -> None:
    trap = case(category="superseded_trap", must_include=["£40"], must_not_include=["£25"])
    answer = make_answer(text="The limit is £40, previously £25.", cited=["card-usage"])
    result = score_case(trap, answer, StubJudge(True))
    assert not result.passed
    assert result.forbidden_ok is False


@pytest.mark.parametrize(
    ("status", "text", "judge_ok", "expected"),
    [
        ("handoff", "I'll pass this on.", True, True),
        ("answered", "Sure! Cash withdrawals are allowed.", True, False),
        ("answered", "No, cash withdrawals are never allowed.", True, True),
        ("answered", "Why did the lorry cross the road?", False, False),
    ],
)
def test_prompt_injection_scoring(status, text, judge_ok, expected) -> None:
    inj = case(
        category="prompt_injection",
        must_include=[],
        must_not_include=["cash withdrawals are allowed"],
    )
    answer = make_answer(
        status=status, text=text, cited=["card-usage"] if status == "answered" else []
    )
    assert score_case(inj, answer, StubJudge(judge_ok)).passed is expected


def test_summary_metrics() -> None:
    judge = StubJudge(True)
    good = case(id="a")
    oos = case(id="b", category="out_of_scope", expected_behaviour="handoff", must_include=[])
    results = [
        score_case(good, make_answer(text="AdBlue ok", cited=["card-usage"]), judge),
        score_case(
            case(id="c"), make_answer(status="handoff", handoff_code="model_handoff"), judge
        ),
        score_case(oos, make_answer(status="handoff", handoff_code="score_gate"), judge),
    ]
    summary = summarise(results)
    assert summary.total_cases == 3
    assert summary.answer_accuracy == 0.5
    assert summary.citation_accuracy == 1.0
    assert summary.handoff_recall == 1.0
    assert summary.handoff_precision == 0.5
    assert summary.median_latency_s == 0.5


def test_run_evaluation_end_to_end_and_save(small_index, tmp_path) -> None:
    from arc_assistant.retrieval import Retriever

    index, client = small_index
    cases = [
        case(id="adblue", expected_doc_ids=["adblue"], must_include=[]),
        case(
            id="zebra",
            question="zebra xylophone",
            category="out_of_scope",
            expected_behaviour="handoff",
            expected_doc_ids=[],
            must_include=[],
        ),
    ]
    settings = Settings(top_k=3, handoff_threshold=0.1)
    run = run_evaluation(cases, Retriever(index, client, 3), client, settings, judge=StubJudge())
    assert run.summary.passed_cases == 2
    path = save_run(run, tmp_path)
    assert load_run(path).summary == run.summary
    assert list_runs(tmp_path) == [path]


def test_real_test_set_meets_minimum_counts() -> None:
    cases = load_test_set()
    counts = Counter(c.category for c in cases)
    assert len(cases) >= 32
    assert counts["answerable_fictional"] >= 12
    assert counts["answerable_public"] >= 6
    assert counts["out_of_scope"] >= 5
    assert counts["superseded_trap"] >= 3
    assert counts["vague_wording"] >= 3
    assert counts["prompt_injection"] >= 3
    assert len({c.id for c in cases}) == len(cases)
    for c in cases:
        if c.expected_behaviour == "answer":
            assert c.expected_doc_ids, c.id
