import pytest

from arc_assistant.config import DOCS_DIR
from arc_assistant.ingest import build_index
from arc_assistant.llm import FakeClient
from arc_assistant.retrieval import IndexMissingError, VectorIndex, asks_about_older_rules


@pytest.mark.parametrize(
    ("question", "expected_doc"),
    [
        ("What is the car fuel benefit multiplier?", "fuel-benefit-company-cars"),
        ("How long do I have to dispute a transaction on my invoice?", "invoicing-billing"),
        ("How fast is a lost card blocked after reporting?", "lost-stolen-cards"),
        ("Which EV charging networks are included?", "ev-charging"),
    ],
)
def test_right_document_ranks_first(real_retriever, question: str, expected_doc: str) -> None:
    retrieved, _ = real_retriever.retrieve(question)
    assert retrieved[0].chunk.doc_id == expected_doc
    scores = [r.score for r in retrieved]
    assert scores == sorted(scores, reverse=True)


def test_superseded_chunks_excluded_by_default(real_retriever) -> None:
    real_retriever.top_k = 100
    retrieved, included = real_retriever.retrieve("Can drivers buy AdBlue or use a car wash?")
    assert not included
    assert retrieved
    assert all(r.chunk.status == "current" for r in retrieved)


def test_superseded_chunks_included_when_asking_about_old_rules(real_retriever) -> None:
    retrieved, included = real_retriever.retrieve("What were the previous rules on AdBlue?")
    assert included
    assert any(r.chunk.status == "superseded" for r in retrieved)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What were the previous rules on AdBlue?", True),
        ("What did the old policy say about car washes?", True),
        ("What changed in version 2.0?", True),
        ("Can my drivers buy AdBlue?", False),
        ("How do I order a replacement card?", False),
    ],
)
def test_detects_questions_about_older_rules(question: str, expected: bool) -> None:
    assert asks_about_older_rules(question) is expected


def test_cache_means_unchanged_docs_are_not_re_embedded(tmp_path) -> None:
    client = FakeClient()
    first = build_index(client, DOCS_DIR, tmp_path)
    assert first.embedded == first.chunks
    second = build_index(client, DOCS_DIR, tmp_path)
    assert second.embedded == 0
    assert second.from_cache == second.chunks


def test_missing_index_raises_friendly_error(tmp_path) -> None:
    with pytest.raises(IndexMissingError, match="not been built"):
        VectorIndex.load(tmp_path)
