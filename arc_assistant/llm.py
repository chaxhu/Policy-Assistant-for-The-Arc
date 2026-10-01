"""One interface for every model call, with a real OpenAI client and an offline fake for tests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

import openai

from arc_assistant.config import Settings
from arc_assistant.models import TokenUsage


class LLMError(RuntimeError):
    """A model call failed. The message is safe to show to users."""


class LLMClient(Protocol):
    """Everything the app needs from a model provider."""

    chat_model: str
    embed_model: str
    usage: TokenUsage

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    def chat_json(self, system: str, user: str) -> str:
        """Run one chat completion that must return a JSON object, as a raw string."""
        ...


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English). Used for estimates only."""
    return max(1, round(len(text) / 4))


class OpenAIClient:
    """LLMClient backed by the OpenAI API. Temperature 0 for repeatable answers."""

    EMBED_BATCH = 96

    def __init__(self, settings: Settings) -> None:
        self.chat_model = settings.chat_model
        self.embed_model = settings.embed_model
        self.usage = TokenUsage()
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key.get_secret_value(), timeout=60, max_retries=2
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches and add the tokens used to the running total."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.EMBED_BATCH):
            batch = texts[start : start + self.EMBED_BATCH]
            with _friendly_errors():
                response = self._client.embeddings.create(model=self.embed_model, input=batch)
            vectors.extend(item.embedding for item in response.data)
            if response.usage:
                self.usage.embedding_tokens += response.usage.prompt_tokens
        return vectors

    def chat_json(self, system: str, user: str) -> str:
        """Call the chat model in JSON mode and return the raw message content."""
        with _friendly_errors():
            response = self._client.chat.completions.create(
                model=self.chat_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        if response.usage:
            self.usage.prompt_tokens += response.usage.prompt_tokens
            self.usage.completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content or ""


@contextmanager
def _friendly_errors() -> Iterator[None]:
    """Turn OpenAI SDK errors into short, user-safe LLMError messages."""
    try:
        yield
    except openai.OpenAIError as exc:
        if isinstance(exc, openai.AuthenticationError):
            message = "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file."
        elif isinstance(exc, openai.RateLimitError):
            message = "OpenAI rate limit or quota reached. Wait a moment or check your billing."
        elif isinstance(exc, openai.NotFoundError):
            message = (
                "The model name in .env was not found. "
                "Check OPENAI_CHAT_MODEL and OPENAI_EMBED_MODEL."
            )
        elif isinstance(exc, openai.APIConnectionError | openai.APITimeoutError):
            message = "Could not reach OpenAI. Check your internet connection and try again."
        else:
            message = f"OpenAI returned an error ({type(exc).__name__}). Please try again."
        raise LLMError(message) from exc


_STOPWORD_TEXT = (
    "a an and are as at be by can do does for from how i if in is it my of on or our "
    "the their them they this to what when where which who will with you your"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_CHUNK_ID_PATTERN = re.compile(r'chunk_id="([^"]+)"')


def _fake_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOPWORDS]


class FakeClient:
    """Deterministic offline LLMClient for tests.

    Embeddings are hashed bag-of-words vectors, so texts sharing words score higher.
    Chat replies come from a scripted list (used in order), a callable, or a default
    reply that answers and cites the first passage in the prompt.
    """

    DIM = 512

    def __init__(
        self,
        responses: list[str] | Callable[[str, str], str] | None = None,
        chat_model: str = "fake-chat",
        embed_model: str = "fake-embed",
    ) -> None:
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.usage = TokenUsage()
        self.chat_calls: list[tuple[str, str]] = []
        self.embed_calls: list[list[str]] = []
        self._responses = responses

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Hashed bag-of-words vectors, length DIM."""
        self.embed_calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * self.DIM
            for token in _fake_tokens(text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
                vector[bucket] += 1.0
            self.usage.embedding_tokens += estimate_tokens(text)
            vectors.append(vector)
        return vectors

    def chat_json(self, system: str, user: str) -> str:
        """Return the next scripted reply, or the default citing reply."""
        self.chat_calls.append((system, user))
        if callable(self._responses):
            reply = self._responses(system, user)
        elif self._responses:
            reply = self._responses.pop(0)
        else:
            reply = self._default_reply(user)
        self.usage.prompt_tokens += estimate_tokens(system + user)
        self.usage.completion_tokens += estimate_tokens(reply)
        return reply

    @staticmethod
    def _default_reply(user: str) -> str:
        ids = _CHUNK_ID_PATTERN.findall(user)
        if not ids:
            return json.dumps(
                {
                    "status": "handoff",
                    "answer": "",
                    "citations": [],
                    "confidence": "low",
                    "handoff_reason": "No passages provided.",
                }
            )
        return json.dumps(
            {
                "status": "answered",
                "answer": "Fake answer based on the first passage.",
                "citations": [ids[0]],
                "confidence": "high",
                "handoff_reason": None,
            }
        )
