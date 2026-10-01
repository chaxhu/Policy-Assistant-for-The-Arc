"""Parse markdown policy documents and split them into citeable chunks."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from arc_assistant.llm import estimate_tokens
from arc_assistant.models import Chunk, DocStatus, SourceType

PLACEHOLDER_MARKER = "CONTENT NOT YET ADDED"
MAX_CHUNK_TOKENS = 500
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class Document(BaseModel):
    """A parsed markdown document: front matter fields plus the body text."""

    path: str
    doc_id: str
    title: str
    source_type: SourceType
    version: str
    status: DocStatus
    effective_date: str | None = None
    retrieved_date: str | None = None
    source_url: str | None = None
    publisher: str | None = None
    company: str | None = None
    body: str

    @property
    def is_placeholder(self) -> bool:
        """True if the document is a placeholder waiting for content."""
        return PLACEHOLDER_MARKER in self.body


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def parse_document(text: str, path: str = "") -> Document:
    """Split YAML front matter from the markdown body and validate the fields."""
    match = _FRONT_MATTER.match(text)
    if not match:
        raise ValueError(f"{path or 'document'} has no YAML front matter block")
    meta = yaml.safe_load(match.group(1)) or {}
    fields = {key: _as_text(value) for key, value in meta.items()}
    fields.setdefault("version", fields.get("retrieved_date") or "1.0")
    return Document(path=path, body=text[match.end() :].strip(), **fields)


def load_documents(docs_dir: Path, include_placeholders: bool = False) -> list[Document]:
    """Load every markdown document under docs_dir, sorted by path."""
    documents = []
    for path in sorted(docs_dir.rglob("*.md")):
        doc = parse_document(path.read_text(encoding="utf-8"), str(path.relative_to(docs_dir)))
        if include_placeholders or not doc.is_placeholder:
            documents.append(doc)
    return documents


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body on '## ' headings into (heading, text) pairs.

    The H1 title and the blockquote disclaimer before the first heading are skipped;
    any other text before the first heading becomes an 'Introduction' section.
    """
    sections: list[tuple[str, str]] = []
    heading = "Introduction"
    lines: list[str] = []

    def flush() -> None:
        text = "\n".join(lines).strip()
        if text:
            sections.append((heading, text))

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            heading, lines = line[3:].strip(), []
        elif not sections and heading == "Introduction" and line.startswith(("# ", ">")):
            continue
        else:
            lines.append(line)
    flush()
    return sections


def split_paragraphs(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    """Group paragraphs into pieces of at most max_tokens, repeating one paragraph as overlap.

    A short section comes back as a single piece. A single paragraph longer than
    max_tokens is kept whole rather than cut mid-sentence.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[list[str]] = []
    current: list[str] = []
    for paragraph in paragraphs:
        candidate = current + [paragraph]
        if current and estimate_tokens("\n\n".join(candidate)) > max_tokens:
            pieces.append(current)
            current = [current[-1], paragraph]  # one-paragraph overlap
        else:
            current = candidate
    if current:
        pieces.append(current)
    return ["\n\n".join(piece) for piece in pieces]


def chunk_document(doc: Document, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    """Turn a document into chunks with ids like 'card-usage#v2.0#3'."""
    chunks = []
    for section, text in split_sections(doc.body):
        for piece in split_paragraphs(text, max_tokens):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#v{doc.version}#{len(chunks) + 1}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section,
                    source_type=doc.source_type,
                    version=doc.version,
                    status=doc.status,
                    effective_date=doc.effective_date,
                    retrieved_date=doc.retrieved_date,
                    source_url=doc.source_url,
                    text=piece,
                )
            )
    return chunks
