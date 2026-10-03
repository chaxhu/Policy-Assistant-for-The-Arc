"""Shared Streamlit helpers: page setup, styling, badges and cached resources."""

from __future__ import annotations

import hashlib
import html

import streamlit as st

from arc_assistant import APP_TITLE, DISCLAIMER
from arc_assistant.config import Settings
from arc_assistant.llm import OpenAIClient
from arc_assistant.retrieval import Retriever, load_or_build_retriever

TEAL = "#0f5257"
CHART_TEAL = "#00968f"
CHART_ORANGE = "#d9731a"
INK = "#1f2a2e"
MUTED = "#5f6b70"

_CSS = """
<style>
h1, h2, h3 { color: #0f5257; letter-spacing: -0.01em; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }
.pa-sub { color: #5f6b70; margin-top: -0.6rem; margin-bottom: 0.4rem; }
.pa-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.74rem;
  font-weight: 600; line-height: 1.6; margin-right: 6px; white-space: nowrap; }
.pa-badge.fictional { background: #fbe6d3; color: #8a4209; }
.pa-badge.public { background: #d5ebe9; color: #0f5257; }
.pa-badge.superseded { background: #f5d6d3; color: #8a1f17; }
.pa-badge.current { background: #e3efe6; color: #24593a; }
.pa-badge.concept { background: #ece8df; color: #4a4f52; border: 1px solid #d8d2c6; }
.pa-badge.high { background: #d5ebe9; color: #0f5257; }
.pa-badge.medium { background: #fbecc8; color: #7a5300; }
.pa-badge.low { background: #eee; color: #555; }
.pa-badge.gate { background: #fbe6d3; color: #8a4209; }
.pa-chip { display: inline-block; border: 1px solid #e3ddd2; background: #fcfbf8;
  border-radius: 10px;
  padding: 6px 10px; margin: 4px 6px 2px 0; font-size: 0.85rem; color: #1f2a2e; }
.pa-chip a { color: #0f5257; text-decoration: none; font-weight: 600; }
.pa-meta { color: #6b7478; font-size: 0.78rem; margin-top: 0.35rem; }
.pa-footer { color: #6b7478; font-size: 0.75rem; line-height: 1.4; }
[class*="st-key-answer-card"] { background: #ffffff; border: 1px solid #e3ddd2;
  border-radius: 14px; padding: 1rem 1.2rem; box-shadow: 0 1px 2px rgba(31, 42, 46, 0.04); }
[class*="st-key-handoff-card"] { background: #fffaf3; border: 1px solid #f0d9bf;
  border-left: 5px solid #d9731a; border-radius: 14px; padding: 1rem 1.2rem; }
[class*="st-key-doc-card"] { background: #ffffff; border: 1px solid #e3ddd2; border-radius: 12px;
  padding: 0.7rem 0.9rem; }
</style>
"""


def page_setup() -> None:
    """Wide layout, styling and the sidebar disclaimer. Called once by the app entry point."""
    st.set_page_config(page_title="Policy Assistant", page_icon=":material/policy:", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f"**{APP_TITLE}**")
        st.markdown(
            f'<div class="pa-footer">{html.escape(DISCLAIMER)}</div>', unsafe_allow_html=True
        )


def header(title: str, subtitle: str) -> None:
    """Page title with a one-line description and the concept badge."""
    st.title(title)
    st.markdown(
        f'<div class="pa-sub">{html.escape(subtitle)} '
        '<span class="pa-badge concept">Unofficial concept demo</span></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str) -> str:
    """HTML for a small coloured badge."""
    return f'<span class="pa-badge {kind}">{html.escape(text)}</span>'


def source_badge(source_type: str) -> str:
    """Badge for the document's source type."""
    if source_type == "public":
        return badge("Public UK guidance", "public")
    return badge("Fictional sample policy", "fictional")


def md_safe(text: str) -> str:
    """Escape dollar signs so Streamlit markdown does not render them as maths."""
    return text.replace("$", r"\$")


def _fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


@st.cache_resource(show_spinner=False)
def _client(_settings: Settings, fingerprint: str, chat: str, embed: str) -> OpenAIClient:
    return OpenAIClient(_settings)


def get_client(settings: Settings) -> OpenAIClient:
    """One shared OpenAI client per key and model pair. The key itself is never a cache key."""
    fingerprint = _fingerprint(settings.openai_api_key.get_secret_value())
    return _client(settings, fingerprint, settings.chat_model, settings.embed_model)


@st.cache_resource(show_spinner=False)
def _retriever(_client: OpenAIClient, embed: str, top_k: int) -> Retriever:
    return load_or_build_retriever(_client, top_k)


def get_retriever(client: OpenAIClient, top_k: int) -> Retriever:
    """Load the index once per process, building it on first start if needed."""
    return _retriever(client, client.embed_model, top_k)
