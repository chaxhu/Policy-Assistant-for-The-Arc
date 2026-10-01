"""Documents page: every source the assistant can use, and the full text of each."""

from __future__ import annotations

import streamlit as st

from arc_assistant.chunking import Document, chunk_document, load_documents
from arc_assistant.config import DOCS_DIR
from arc_assistant.ui import badge, header, md_safe, source_badge


@st.cache_data(show_spinner=False)
def documents_with_chunk_counts() -> list[tuple[Document, int]]:
    """All documents, current first, with the number of chunks each produces."""
    docs = load_documents(DOCS_DIR, include_placeholders=True)
    docs.sort(key=lambda d: (d.source_type, d.status != "current", d.title))
    return [(d, 0 if d.is_placeholder else len(chunk_document(d))) for d in docs]


def date_line(doc: Document) -> str:
    if doc.source_type == "public":
        return f"Retrieved {doc.retrieved_date} from {doc.publisher}"
    return f"Version {doc.version} · effective {doc.effective_date}"


header("Documents", "Everything the assistant is allowed to answer from. Nothing else.")

items = documents_with_chunk_counts()
choice = st.segmented_control(
    "Show", ["All", "Fictional sample policies", "Public UK guidance"], default="All"
)
if choice == "Fictional sample policies":
    items = [i for i in items if i[0].source_type == "fictional"]
elif choice == "Public UK guidance":
    items = [i for i in items if i[0].source_type == "public"]

keys = [f"{d.doc_id}@{d.version}" for d, _ in items]
if st.session_state.get("open_doc") not in keys and keys:
    st.session_state["open_doc"] = keys[0]

list_column, reader_column = st.columns([2, 3], gap="large")
with list_column:
    for (doc, chunks), key in zip(items, keys, strict=True):
        with st.container(key=f"doc-card-{key.replace('@', '-').replace('.', '-')}"):
            status = (
                badge("Superseded", "superseded")
                if doc.status == "superseded"
                else badge("Current", "current")
            )
            st.markdown(f"**{doc.title}**", unsafe_allow_html=True)
            st.markdown(
                f"{source_badge(doc.source_type)}{status}"
                f'<div class="pa-meta">{date_line(doc)} · {chunks} chunks</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "Read",
                key=f"read-{key}",
                icon=":material/menu_book:",
                type="primary" if st.session_state["open_doc"] == key else "secondary",
            ):
                st.session_state["open_doc"] = key
                st.rerun()

with reader_column:
    opened = next(
        (d for (d, _), k in zip(items, keys, strict=True) if k == st.session_state.get("open_doc")),
        None,
    )
    if opened is None:
        st.info("No documents to show.")
    else:
        if opened.status == "superseded":
            st.warning(
                f"Superseded. Version {opened.version} was replaced by a newer version. The "
                "assistant ignores it unless you ask about previous rules.",
                icon=":material/history:",
            )
        if opened.is_placeholder:
            st.error("Content not yet added. This document is excluded from the index.")
        if opened.source_url:
            st.markdown(f"Source: [{opened.source_url}]({opened.source_url})")
        with st.container(border=True, height=720):
            st.markdown(md_safe(opened.body))
