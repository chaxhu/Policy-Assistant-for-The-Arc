"""Shared fixtures. Everything runs offline against FakeClient."""

from __future__ import annotations

import numpy as np
import pytest

from arc_assistant.config import DOCS_DIR
from arc_assistant.ingest import build_index
from arc_assistant.llm import FakeClient
from arc_assistant.models import Answer, AnswerMetrics, Chunk, Citation
from arc_assistant.retrieval import Retriever, VectorIndex


def make_chunk(chunk_id: str, text: str, **overrides: object) -> Chunk:
    """A chunk with sensible defaults for tests."""
    doc_id = chunk_id.split("#")[0]
    fields = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "title": doc_id.replace("-", " ").title(),
        "section": "Section",
        "source_type": "fictional",
        "version": "1.0",
        "status": "current",
        "text": text,
    }
    fields.update(overrides)
    return Chunk(**fields)


def make_answer(
    status: str = "answered",
    text: str = "Answer text.",
    cited: list[str] | None = None,
    handoff_code: str | None = None,
) -> Answer:
    """An Answer with citations to the given doc ids."""
    citations = [
        Citation.from_chunk(make_chunk(f"{doc_id}#v1.0#1", "x")) for doc_id in (cited or [])
    ]
    return Answer(
        question="q",
        status=status,
        answer=text,
        citations=citations,
        handoff_code=handoff_code,
        metrics=AnswerMetrics(
            latency_s=0.5,
            prompt_tokens=10,
            completion_tokens=5,
            embedding_tokens=2,
            cost_usd=0.0001,
        ),
    )


@pytest.fixture
def small_index() -> tuple[VectorIndex, FakeClient]:
    """A tiny hand-made index embedded with a FakeClient."""
    client = FakeClient()
    chunks = [
        make_chunk("adblue#v2.0#1", "AdBlue can be bought with the card up to 10 litres."),
        make_chunk("invoices#v1.0#1", "Invoices are issued weekly every Monday."),
        make_chunk(
            "rates#v2026#1",
            "Advisory fuel rates for petrol cars are 17 pence per mile.",
            source_type="public",
            source_url="https://www.gov.uk/guidance/advisory-fuel-rates",
        ),
        make_chunk("adblue#v1.0#1", "AdBlue cannot be bought with the card.", status="superseded"),
    ]
    embeddings = np.array(client.embed([c.embed_text for c in chunks]))
    return VectorIndex(chunks, embeddings, client.embed_model), client


@pytest.fixture(scope="session")
def real_index_dir(tmp_path_factory: pytest.TempPathFactory):
    """The real documents indexed once with fake embeddings."""
    index_dir = tmp_path_factory.mktemp("index")
    build_index(FakeClient(), DOCS_DIR, index_dir)
    return index_dir


@pytest.fixture
def real_retriever(real_index_dir) -> Retriever:
    """Retriever over the real documents, using fake embeddings."""
    client = FakeClient()
    return Retriever(VectorIndex.load(real_index_dir), client, top_k=5)
