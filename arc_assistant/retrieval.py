"""Cosine-similarity search over a NumPy matrix of chunk embeddings.

The whole vector store is two files: index/embeddings.npy (one row per chunk) and
index/chunks.json (the chunk text and metadata, in the same order as the rows).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from arc_assistant.config import INDEX_DIR
from arc_assistant.ingest import CHUNKS_FILE, EMBEDDINGS_FILE, build_index, index_exists
from arc_assistant.llm import LLMClient
from arc_assistant.models import Chunk, RetrievedChunk

# Words that show the user is asking about older rules, so superseded chunks are allowed in.
_OLDER_RULES = re.compile(
    r"\b(previous(ly)?|older|old (rules?|polic(y|ies)|version|terms)|superseded|used to|"
    r"version 1|v1(\.0)?|prior|earlier|historic(al)?|originally|what changed|"
    r"changed from|before (january )?2026|in 2024|in 2025|last year)\b",
    re.IGNORECASE,
)


class IndexMissingError(RuntimeError):
    """The index has not been built yet."""


def asks_about_older_rules(question: str) -> bool:
    """True if the question explicitly asks about previous or superseded rules."""
    return bool(_OLDER_RULES.search(question))


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class VectorIndex:
    """In-memory index: a list of chunks and a matching matrix of unit-length vectors."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embed_model: str = "") -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks = chunks
        self.embed_model = embed_model
        self._matrix = _normalise(np.asarray(embeddings, dtype=np.float32))
        self._superseded = np.array([c.status == "superseded" for c in chunks], dtype=bool)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> VectorIndex:
        """Load the index from disk, or raise IndexMissingError."""
        if not index_exists(index_dir):
            raise IndexMissingError("The search index has not been built yet.")
        payload = json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload["chunks"]]
        embeddings = np.load(index_dir / EMBEDDINGS_FILE)
        return cls(chunks, embeddings, payload.get("embed_model", ""))

    def search(
        self, query_vector: list[float], top_k: int, include_superseded: bool = False
    ) -> list[RetrievedChunk]:
        """Return the top_k chunks by cosine similarity, best first."""
        query = _normalise(np.asarray(query_vector, dtype=np.float32))
        scores = self._matrix @ query
        if not include_superseded:
            scores = np.where(self._superseded, -np.inf, scores)
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in order
            if np.isfinite(scores[i])
        ]


class Retriever:
    """Embeds a question and searches the index."""

    def __init__(self, index: VectorIndex, client: LLMClient, top_k: int) -> None:
        self.index = index
        self.client = client
        self.top_k = top_k

    def retrieve(self, question: str) -> tuple[list[RetrievedChunk], bool]:
        """Return (retrieved chunks, whether superseded chunks were allowed)."""
        include_superseded = asks_about_older_rules(question)
        query_vector = self.client.embed([question])[0]
        return self.index.search(query_vector, self.top_k, include_superseded), include_superseded


def load_or_build_retriever(
    client: LLMClient, top_k: int, index_dir: Path = INDEX_DIR
) -> Retriever:
    """Load the index, building it first if it is missing or used a different embedding model."""
    if not index_exists(index_dir):
        build_index(client, index_dir=index_dir)
    index = VectorIndex.load(index_dir)
    if index.embed_model != client.embed_model:
        build_index(client, index_dir=index_dir)
        index = VectorIndex.load(index_dir)
    return Retriever(index, client, top_k)
