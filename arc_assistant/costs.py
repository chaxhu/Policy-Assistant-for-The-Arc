"""Estimated costs from token counts.

Prices are OpenAI list prices in US dollars per 1 million tokens at the time of writing.
They change, so treat every figure here as an estimate and check the OpenAI pricing page.
"""

from __future__ import annotations

from arc_assistant.models import TokenUsage

# model name: (input price, output price) in USD per 1M tokens
CHAT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}
EMBED_PRICES: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

# Typical tokens per call, used for estimates before a run (measured from this app's prompts).
TYPICAL_ANSWER_PROMPT_TOKENS = 1_900
TYPICAL_ANSWER_COMPLETION_TOKENS = 180
TYPICAL_JUDGE_PROMPT_TOKENS = 1_600
TYPICAL_JUDGE_COMPLETION_TOKENS = 60
TYPICAL_QUESTION_EMBED_TOKENS = 20


def estimate_cost(usage: TokenUsage, chat_model: str, embed_model: str) -> float | None:
    """Estimated USD cost for the given usage, or None if a model's price is unknown."""
    chat_price = CHAT_PRICES.get(chat_model)
    embed_price = EMBED_PRICES.get(embed_model)
    if chat_price is None or embed_price is None:
        return None
    input_price, output_price = chat_price
    return (
        usage.prompt_tokens * input_price
        + usage.completion_tokens * output_price
        + usage.embedding_tokens * embed_price
    ) / 1_000_000


def estimate_eval_run_cost(n_cases: int, chat_model: str, embed_model: str) -> float | None:
    """Upper-end estimate for a full evaluation run: one answer and one judge call per case."""
    usage = TokenUsage(
        prompt_tokens=n_cases * (TYPICAL_ANSWER_PROMPT_TOKENS + TYPICAL_JUDGE_PROMPT_TOKENS),
        completion_tokens=n_cases
        * (TYPICAL_ANSWER_COMPLETION_TOKENS + TYPICAL_JUDGE_COMPLETION_TOKENS),
        embedding_tokens=n_cases * TYPICAL_QUESTION_EMBED_TOKENS,
    )
    return estimate_cost(usage, chat_model, embed_model)


def format_cost(cost_usd: float | None) -> str:
    """Human-friendly cost string."""
    if cost_usd is None:
        return "cost unknown"
    if cost_usd < 0.00001:
        return "under $0.00001"
    if cost_usd < 0.01:
        return f"${cost_usd:.5f}"
    return f"${cost_usd:.3f}"
