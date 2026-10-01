"""Entry point and Ask page. Run with:  streamlit run app.py"""

from __future__ import annotations

import html

import streamlit as st

from arc_assistant import APP_TITLE
from arc_assistant.answer import answer_question
from arc_assistant.config import ConfigError, load_settings
from arc_assistant.costs import format_cost
from arc_assistant.ingest import index_exists
from arc_assistant.llm import LLMError, OpenAIClient
from arc_assistant.models import Answer
from arc_assistant.retrieval import Retriever
from arc_assistant.ui import (
    badge,
    get_client,
    get_retriever,
    header,
    md_safe,
    page_setup,
    source_badge,
)

SUGGESTED_QUESTIONS = [
    "Can my drivers buy AdBlue on their cards?",
    "What is the advisory fuel rate for a 1.6 litre petrol car?",
    "How quickly is a lost card blocked?",
    "Can we use the card for charging at a driver's home?",
    "What is the car fuel benefit multiplier for 2026 to 2027?",
    "How do I cancel my contract?",
]


def render_sources(answer: Answer) -> None:
    """Citation chips with source badges and links."""
    chips = []
    for c in answer.citations:
        title = html.escape(c.title)
        if c.source_url:
            title = f'<a href="{html.escape(c.source_url)}" target="_blank">{title}</a>'
        status = badge("Superseded", "superseded") if c.status == "superseded" else ""
        chips.append(
            f'<span class="pa-chip">{title} · {html.escape(c.section)} '
            f"{source_badge(c.source_type)}{status}</span>"
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


def render_passages(answer: Answer, label: str) -> None:
    """Expander listing every retrieved passage with its score, highlighting cited ones."""
    cited = {c.chunk_id for c in answer.citations}
    with st.expander(label, icon=":material/search:"):
        if answer.included_superseded:
            st.caption(
                "This question asked about older rules, so superseded versions were searched."
            )
        if not answer.retrieved:
            st.caption("No passages were retrieved.")
        for r in answer.retrieved:
            c = r.chunk
            is_cited = c.chunk_id in cited
            text = html.escape(c.text[:420] + ("…" if len(c.text) > 420 else ""))
            st.markdown(
                f'<div class="pa-passage{" cited" if is_cited else ""}">'
                f"<b>{r.score:.3f}</b> · {html.escape(c.title)} › {html.escape(c.section)} "
                f"{source_badge(c.source_type)}"
                f"{badge('Superseded', 'superseded') if c.status == 'superseded' else ''}"
                f"{badge('Cited', 'high') if is_cited else ''}<br>"
                f'<span style="color:#5f6b70">{text}</span></div>',
                unsafe_allow_html=True,
            )


def render_meta(answer: Answer) -> None:
    """Latency, tokens and cost line."""
    m = answer.metrics
    st.markdown(
        f'<div class="pa-meta">{m.latency_s:.2f}s · {m.prompt_tokens:,} prompt + '
        f"{m.completion_tokens:,} completion tokens · {format_cost(m.cost_usd)}</div>",
        unsafe_allow_html=True,
    )


def render_answer(answer: Answer, key: str) -> None:
    """Show an answered card or a distinct hand-off card."""
    if answer.status == "handoff":
        with st.container(key=f"handoff-card-{key}"):
            st.markdown("#### I'll pass this to your account manager")
            if answer.handoff_code == "score_gate":
                st.markdown(
                    badge("No policy closely matched this question", "gate"),
                    unsafe_allow_html=True,
                )
            st.markdown(md_safe(answer.handoff_reason or ""))
            st.caption("Ready-to-send summary for your account manager:")
            st.code(answer.handoff_summary or "", language=None, wrap_lines=True)
            render_passages(answer, "What was searched")
            render_meta(answer)
        return

    with st.container(key=f"answer-card-{key}"):
        pills = badge(f"{answer.confidence.title()} confidence", answer.confidence)
        if any(c.status == "superseded" for c in answer.citations):
            pills += badge("Uses a superseded policy", "superseded")
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown(md_safe(answer.answer))
        render_sources(answer)
        render_passages(answer, "Why this answer?")
        render_meta(answer)


def load_pipeline() -> tuple[OpenAIClient, Retriever, float]:
    """Settings, client and retriever, or a friendly message and stop."""
    try:
        settings = load_settings()
    except ConfigError as exc:
        st.error(str(exc), icon=":material/key:")
        st.info(
            "The Documents and About pages work without a key, and the Evaluation page can "
            "show saved runs.",
            icon=":material/info:",
        )
        st.stop()

    client = get_client(settings)
    try:
        if index_exists():
            retriever = get_retriever(client, settings.top_k)
        else:
            with st.spinner("Building the search index for the first time. One moment…"):
                retriever = get_retriever(client, settings.top_k)
            st.toast("Search index built.", icon=":material/check_circle:")
    except LLMError as exc:
        st.error(f"Could not build the search index. {exc}", icon=":material/error:")
        st.stop()

    with st.sidebar:
        st.caption(
            f"Chat model: {settings.chat_model}  \nEmbeddings: {settings.embed_model}  \n"
            f"Top K: {settings.top_k} · Hand-off threshold: {settings.handoff_threshold:.2f}"
        )
    return client, retriever, settings.handoff_threshold


def ask_page() -> None:
    """The chat page."""
    header(
        APP_TITLE,
        "Ask about Sample Fuel Card Co. policies or UK motoring guidance. "
        "Every answer shows its sources, and anything uncertain goes to a person.",
    )
    client, retriever, threshold = load_pipeline()
    history: list[Answer] = st.session_state.setdefault("history", [])

    with st.sidebar:
        if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
            history.clear()
            st.rerun()

    if not history:
        st.markdown("##### Try a question")
        columns = st.columns(3)
        for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
            if columns[i % 3].button(suggestion, key=f"suggest-{i}", width="stretch"):
                st.session_state["pending"] = suggestion

    for i, answer in enumerate(history):
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(md_safe(answer.question))
        with st.chat_message("assistant", avatar=":material/policy:"):
            render_answer(answer, str(i))

    typed = st.chat_input("Ask about card policies or UK motoring guidance")
    question = (typed or st.session_state.pop("pending", None) or "").strip()
    if not question:
        return
    try:
        with st.spinner("Searching the policies…"):
            history.append(answer_question(question, retriever, client, threshold))
    except LLMError as exc:
        st.error(str(exc), icon=":material/cloud_off:")
        return
    st.rerun()


if __name__ == "__main__":
    page_setup()
    navigation = st.navigation(
        [
            st.Page(ask_page, title="Ask", icon=":material/forum:", default=True),
            st.Page("pages/1_Evaluation.py", title="Evaluation", icon=":material/fact_check:"),
            st.Page("pages/2_Documents.py", title="Documents", icon=":material/description:"),
            st.Page("pages/3_About.py", title="About", icon=":material/info:"),
        ]
    )
    navigation.run()
