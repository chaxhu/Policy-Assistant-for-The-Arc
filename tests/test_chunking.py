from pathlib import Path

from arc_assistant.chunking import (
    chunk_document,
    load_documents,
    parse_document,
    split_paragraphs,
    split_sections,
)
from arc_assistant.config import DOCS_DIR

SAMPLE = """---
doc_id: sample
title: Sample policy
source_type: fictional
version: 2.0
effective_date: 2026-01-01
status: current
---

# Sample policy

> Fictional sample policy for a concept demo. Not the terms of any real company.

## First section

Alpha text.

## Second section

Beta text.

More beta.
"""


def test_front_matter_is_parsed_to_strings() -> None:
    doc = parse_document(SAMPLE, "sample.md")
    assert doc.doc_id == "sample"
    assert doc.version == "2.0"
    assert doc.effective_date == "2026-01-01"
    assert doc.body.startswith("# Sample policy")


def test_sections_split_on_h2_and_skip_title_and_disclaimer() -> None:
    sections = split_sections(parse_document(SAMPLE).body)
    assert sections == [
        ("First section", "Alpha text."),
        ("Second section", "Beta text.\n\nMore beta."),
    ]


def test_chunks_keep_metadata_and_ids() -> None:
    chunks = chunk_document(parse_document(SAMPLE))
    assert [c.chunk_id for c in chunks] == ["sample#v2.0#1", "sample#v2.0#2"]
    first = chunks[0]
    assert (first.doc_id, first.section, first.status, first.version) == (
        "sample",
        "First section",
        "current",
        "2.0",
    )
    assert first.embed_text.startswith("Sample policy\nFirst section")


def test_long_sections_split_with_one_paragraph_overlap() -> None:
    paragraphs = [f"Paragraph {i} " + "word " * 150 for i in range(5)]  # about 190 tokens each
    pieces = split_paragraphs("\n\n".join(paragraphs), max_tokens=500)
    assert len(pieces) > 1
    for previous, current in zip(pieces, pieces[1:], strict=False):
        last_paragraph = previous.split("\n\n")[-1]
        assert current.split("\n\n")[0] == last_paragraph
    assert all(p.strip().startswith("Paragraph") for p in pieces)


def test_short_section_is_one_piece() -> None:
    assert split_paragraphs("one\n\ntwo") == ["one\n\ntwo"]


def test_placeholder_documents_are_excluded(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text(SAMPLE, encoding="utf-8")
    placeholder = SAMPLE.replace("doc_id: sample", "doc_id: todo").split("# Sample")[0]
    (tmp_path / "todo.md").write_text(
        placeholder + "CONTENT NOT YET ADDED: paste a summary here", encoding="utf-8"
    )
    assert [d.doc_id for d in load_documents(tmp_path)] == ["sample"]
    assert len(load_documents(tmp_path, include_placeholders=True)) == 2


def test_real_documents_follow_the_rules() -> None:
    docs = load_documents(DOCS_DIR, include_placeholders=True)
    fictional = [d for d in docs if d.source_type == "fictional"]
    public = [d for d in docs if d.source_type == "public"]
    assert len(fictional) == 7
    assert len(public) == 3
    assert sum(d.status == "superseded" for d in docs) == 1
    for doc in fictional:
        assert "Fictional sample policy for a concept demo" in doc.body
        assert doc.company == "Sample Fuel Card Co. (fictional)"
    for doc in public:
        assert doc.source_url and doc.source_url.startswith("https://www.gov.uk/")
        assert doc.retrieved_date and doc.publisher == "GOV.UK"


def test_deliberate_gaps_are_not_covered() -> None:
    text = " ".join(d.body.lower() for d in load_documents(DOCS_DIR))
    for gap in ["credit limit", "credit check", "notice period", "telematics"]:
        assert gap not in text, f"documents should not cover '{gap}'"
