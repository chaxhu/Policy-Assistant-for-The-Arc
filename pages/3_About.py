"""About page: what this is, how it works and what production would need."""

from __future__ import annotations

import streamlit as st

from arc_assistant.evaluate import load_test_set
from arc_assistant.ui import header

ARCHITECTURE = """
digraph G {
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
        color="#d8d2c6", fillcolor="#ffffff", fontcolor="#1f2a2e", margin="0.18,0.08"];
  edge [color="#8a9396", fontname="Helvetica", fontsize=9, fontcolor="#5f6b70"];

  q [label="Question"];
  r [label="Retrieve\\n(cosine similarity,\\ntop K, current only)"];
  g [label="Score gate\\n(best score\\n>= threshold?)", shape=diamond, fillcolor="#f1ede5"];
  l [label="LLM answer\\n(strict JSON,\\none retry)"];
  c [label="Citation check\\n(in code)", shape=diamond, fillcolor="#f1ede5"];
  a [label="Answer\\nwith sources", fillcolor="#d5ebe9", color="#0f5257"];
  h [label="Hand-off\\nto a person", fillcolor="#fbe6d3", color="#d9731a"];
  e [label="Evaluation set\\n(N_CASES cases + LLM judge)", fillcolor="#f1ede5"];
  t [label="Prompt and\\nthreshold changes", fillcolor="#f1ede5"];

  q -> r -> g;
  g -> l [label="yes"];
  g -> h [label="no"];
  l -> c [label="answered"];
  l -> h [label="hand-off or\\ninvalid JSON"];
  c -> a [label="valid citations"];
  c -> h [label="none valid"];
  a -> e [style=dashed];
  h -> e [style=dashed];
  e -> t [style=dashed, label="failures"];
  t -> g [style=dashed];
  t -> l [style=dashed];
}
"""

n_cases = len(load_test_set())
header("About", "What this is, how it works, and what it would take to run it for real.")

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown(
        f"""
This is an **unofficial concept demo** built for a job interview. It is not affiliated with,
endorsed by or connected to Greenarc, and it does not use any Greenarc data, policies, prices or
terms. The company policies it answers from belong to **Sample Fuel Card Co.**, a fictional
company invented for this demo. The public guidance pages are summaries of GOV.UK pages, with a
link and retrieval date on each.

The assistant answers a fleet manager's questions using only its own documents. Every answer
shows which passages it came from, and the citations are checked in code rather than trusted.
When the documents do not clearly cover a question, it hands over to an account manager with a
ready-to-send summary instead of guessing.

The most important part is the **evaluation harness**. A set of {n_cases} test questions covers
normal questions, public guidance, deliberate gaps, a superseded policy trap, vague wording and
prompt injection. Each run is scored with simple checks plus a strict LLM judge, and saved so that
any change can be compared against the last one. The point is not only to build an assistant, but
to measure whether it can be trusted.
"""
    )
with right:
    st.markdown("##### How I'd take this to production")
    st.markdown(
        """
- **Access control per customer account**, so each customer only sees their own contract terms.
- **Documents kept in sync with the source of truth**, with versions and effective dates, and
  automatic re-indexing when a policy changes.
- **Logging and weekly review of hand-offs**, to find gaps in the documents and fix them.
- **Regular re-evaluation** on every prompt, model or document change, with the test set growing
  from real hand-offs.
- **Human approval of policy changes** before they reach the assistant.
- **Cost and latency monitoring** with alerts, plus a cap per account.
"""
    )

st.markdown("##### Architecture")
st.graphviz_chart(ARCHITECTURE.replace("N_CASES", str(n_cases)), width="stretch")
st.caption(
    "Solid arrows are the live question path. Dashed arrows are the evaluation loop: failures "
    "lead to prompt and threshold changes, which are measured again before they ship."
)
