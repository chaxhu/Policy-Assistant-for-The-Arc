"""Build the vector index: load documents, chunk, embed (with a disk cache) and save.

Run with:  python -m arc_assistant.ingest
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from arc_assistant.chunking import chunk_document, load_documents
from arc_assistant.config import DOCS_DIR, INDEX_DIR, ConfigError, load_settings
from arc_assistant.llm import LLMClient, LLMError, OpenAIClient
from arc_assistant.models import Chunk

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.json"
CACHE_FILE = "embedding_cache.json"


class IngestStats(BaseModel):
    """What happened during one ingestion run."""

    documents: int
    chunks: int
    embedded: int
    from_cache: int
    skipped_placeholders: list[str]


def cache_key(embed_model: str, text: str) -> str:
    """Cache key: hash of the model name and the exact text that gets embedded."""
    return hashlib.sha256(f"{embed_model}\n{text}".encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def index_exists(index_dir: Path = INDEX_DIR) -> bool:
    """True if a built index is present on disk."""
    return (index_dir / EMBEDDINGS_FILE).exists() and (index_dir / CHUNKS_FILE).exists()


def build_index(
    client: LLMClient, docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR
) -> IngestStats:
    """Chunk every document, embed new or changed chunks only, and save the index."""
    all_docs = load_documents(docs_dir, include_placeholders=True)
    placeholders = [doc.path for doc in all_docs if doc.is_placeholder]
    documents = [doc for doc in all_docs if not doc.is_placeholder]
    chunks: list[Chunk] = [chunk for doc in documents for chunk in chunk_document(doc)]

    index_dir.mkdir(parents=True, exist_ok=True)
    cache_path = index_dir / CACHE_FILE
    cache = _load_cache(cache_path)
    keys = [cache_key(client.embed_model, chunk.embed_text) for chunk in chunks]
    missing = [i for i, key in enumerate(keys) if key not in cache]
    if missing:
        vectors = client.embed([chunks[i].embed_text for i in missing])
        for i, vector in zip(missing, vectors, strict=True):
            cache[keys[i]] = vector

    matrix = np.array([cache[key] for key in keys], dtype=np.float32)
    np.save(index_dir / EMBEDDINGS_FILE, matrix)
    payload = {
        "embed_model": client.embed_model,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chunks": [chunk.model_dump() for chunk in chunks],
    }
    (index_dir / CHUNKS_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    used = set(keys)
    cache_path.write_text(json.dumps({k: v for k, v in cache.items() if k in used}))

    return IngestStats(
        documents=len(documents),
        chunks=len(chunks),
        embedded=len(missing),
        from_cache=len(chunks) - len(missing),
        skipped_placeholders=placeholders,
    )


def main() -> int:
    """Command line entry point."""
    try:
        settings = load_settings()
        stats = build_index(OpenAIClient(settings))
    except (ConfigError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Indexed {stats.documents} documents into {stats.chunks} chunks "
        f"({stats.embedded} embedded, {stats.from_cache} from cache)."
    )
    for path in stats.skipped_placeholders:
        print(f"Skipped placeholder document: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
