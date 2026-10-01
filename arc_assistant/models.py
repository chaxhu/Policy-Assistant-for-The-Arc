"""Pydantic models shared across ingestion, retrieval, answering and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SourceType = Literal["fictional", "public"]
DocStatus = Literal["current", "superseded"]
Behaviour = Literal["answer", "handoff"]
HandoffCode = Literal[
    "score_gate", "model_handoff", "invalid_output", "no_valid_citations", "error"
]


class Chunk(BaseModel):
    """One retrievable piece of a document, with the metadata needed to cite it."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    source_type: SourceType
    version: str
    status: DocStatus
    effective_date: str | None = None
    retrieved_date: str | None = None
    source_url: str | None = None
    text: str

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: title and section give the chunk context."""
        return f"{self.title}\n{self.section}\n\n{self.text}"


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its cosine similarity to the question."""

    chunk: Chunk
    score: float


class Citation(BaseModel):
    """A validated citation shown under an answer."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    source_type: SourceType
    status: DocStatus
    version: str
    source_url: str | None = None

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Citation:
        """Build a citation from the chunk it points to."""
        return cls(**chunk.model_dump(include=set(cls.model_fields)))


class LLMAnswer(BaseModel):
    """The JSON shape the chat model must return. Validated before anything is shown.

    status and answer are required. Confidence defaults to "low" if the model leaves it out,
    because a missing label should not throw away an otherwise valid reply.
    """

    status: Literal["answered", "handoff"]
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    handoff_reason: str | None = None

    @field_validator("status", "confidence", mode="before")
    @classmethod
    def _lowercase(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class TokenUsage(BaseModel):
    """Running token counters for one client."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0

    def minus(self, earlier: TokenUsage) -> TokenUsage:
        """Usage since an earlier snapshot."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens - earlier.prompt_tokens,
            completion_tokens=self.completion_tokens - earlier.completion_tokens,
            embedding_tokens=self.embedding_tokens - earlier.embedding_tokens,
        )


class AnswerMetrics(BaseModel):
    """Latency, tokens and estimated cost for one question."""

    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    embedding_tokens: int
    cost_usd: float | None


class Answer(BaseModel):
    """What the assistant returns for one question, whether answered or handed off."""

    question: str
    status: Literal["answered", "handoff"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    handoff_code: HandoffCode | None = None
    handoff_reason: str | None = None
    handoff_summary: str | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list)
    included_superseded: bool = False
    metrics: AnswerMetrics

    @property
    def cited_doc_ids(self) -> list[str]:
        """Distinct document ids cited by this answer, in order."""
        return list(dict.fromkeys(c.doc_id for c in self.citations))

    @property
    def best_score(self) -> float:
        """Highest retrieval score, or 0 if nothing was retrieved."""
        return max((r.score for r in self.retrieved), default=0.0)


class EvalCase(BaseModel):
    """One question in the evaluation set."""

    id: str
    category: Literal[
        "answerable_fictional",
        "answerable_public",
        "out_of_scope",
        "superseded_trap",
        "vague_wording",
        "prompt_injection",
    ]
    question: str
    expected_behaviour: Behaviour
    expected_doc_ids: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    notes: str = ""


class JudgeVerdict(BaseModel):
    """The LLM judge's decision on one answer."""

    correct: bool
    reason: str


class EvalResult(BaseModel):
    """The outcome of running and scoring one evaluation case."""

    case: EvalCase
    actual_behaviour: Behaviour
    answer_text: str
    cited_doc_ids: list[str]
    handoff_code: HandoffCode | None = None
    best_score: float
    behaviour_ok: bool
    citation_ok: bool | None = None
    keywords_ok: bool | None = None
    forbidden_ok: bool | None = None
    judge: JudgeVerdict | None = None
    passed: bool
    reason: str
    latency_s: float
    cost_usd: float | None = None


class EvalSummary(BaseModel):
    """Headline metrics for one evaluation run. Rates are 0 to 1, or None if no cases apply."""

    total_cases: int
    passed_cases: int
    answer_accuracy: float | None
    citation_accuracy: float | None
    handoff_precision: float | None
    handoff_recall: float | None
    superseded_pass_rate: float | None
    injection_resistance: float | None
    median_latency_s: float | None
    total_cost_usd: float | None
    pass_rate_by_category: dict[str, float]


class EvalRunConfig(BaseModel):
    """Settings that were in force for a run, so runs can be compared fairly."""

    chat_model: str
    embed_model: str
    top_k: int
    handoff_threshold: float
    prompt_version: str


class EvalRun(BaseModel):
    """A full evaluation run, saved as one JSON file."""

    run_id: str
    created_at: str
    label: str = ""
    config: EvalRunConfig
    summary: EvalSummary
    results: list[EvalResult]
