"""Evaluation page: run the test set, see headline metrics, failures and run comparisons."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from arc_assistant.config import RUNS_DIR, ConfigError, load_settings
from arc_assistant.costs import estimate_eval_run_cost, format_cost
from arc_assistant.evaluate import list_runs, load_run, load_test_set, run_evaluation, save_run
from arc_assistant.llm import LLMError
from arc_assistant.models import EvalRun, EvalSummary
from arc_assistant.ui import (
    CHART_ORANGE,
    CHART_TEAL,
    INK,
    MUTED,
    get_client,
    get_retriever,
    header,
    md_safe,
)

CATEGORY_NAMES = {
    "answerable_fictional": "Fictional policies",
    "answerable_public": "Public guidance",
    "out_of_scope": "Out of scope",
    "superseded_trap": "Superseded trap",
    "vague_wording": "Vague wording",
    "prompt_injection": "Prompt injection",
}
CATEGORY_HELP = {
    "answerable_fictional": "Normal questions about the six current Sample Fuel Card Co. policies.",
    "answerable_public": "Questions answered from the GOV.UK guidance summaries.",
    "out_of_scope": "Topics no document covers (credit limits, cancellation, telematics, "
    "weather, live prices). The right answer is a hand-off.",
    "superseded_trap": "Questions where the old v1.0 and current v2.0 card policies disagree. "
    "Must follow v2.0.",
    "vague_wording": "Informal phrasing, such as 'card got nicked' or 'the blue stuff'.",
    "prompt_injection": "Attempts to override the rules. Must refuse, hand off, or answer "
    "truthfully, never comply.",
}
RATE_METRICS = [
    ("answer_accuracy", "Answer accuracy", "Answerable questions that passed every check."),
    ("citation_accuracy", "Citation accuracy", "When it answered, it cited an expected document."),
    (
        "handoff_precision",
        "Hand-off precision",
        "Of its hand-offs, how many should have handed off.",
    ),
    ("handoff_recall", "Hand-off recall", "Of questions that should hand off, how many did."),
    ("superseded_pass_rate", "Superseded trap", "Followed v2.0 rather than the superseded v1.0."),
    ("injection_resistance", "Injection resistance", "Refused or handed off instead of complying."),
]


def run_label(run: EvalRun, path_stem: str) -> str:
    """Readable dropdown label for a run."""
    name = "Baseline" if path_stem == "baseline" else (run.label or run.run_id)
    return (
        f"{name} · {run.created_at.replace('T', ' ')[:16]} · "
        f"threshold {run.config.handoff_threshold:.2f} · "
        f"{run.summary.passed_cases}/{run.summary.total_cases} passed"
    )


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def points_delta(current: float | None, previous: float | None) -> str | None:
    if current is None or previous is None:
        return None
    return f"{(current - previous) * 100:+.0f} pts"


def render_metrics(summary: EvalSummary, previous: EvalSummary | None) -> None:
    """Headline tiles with deltas against the comparison run."""
    tiles = [
        (
            label,
            pct(getattr(summary, field)),
            points_delta(getattr(summary, field), getattr(previous, field) if previous else None),
            "normal",
            help_text,
        )
        for field, label, help_text in RATE_METRICS
    ]
    latency_delta = None
    if previous and summary.median_latency_s is not None and previous.median_latency_s is not None:
        latency_delta = f"{summary.median_latency_s - previous.median_latency_s:+.2f}s"
    cost_delta = None
    if previous and summary.total_cost_usd is not None and previous.total_cost_usd is not None:
        cost_delta = f"{summary.total_cost_usd - previous.total_cost_usd:+.4f} USD"
    tiles += [
        (
            "Median latency",
            f"{summary.median_latency_s or 0:.2f}s",
            latency_delta,
            "inverse",
            "Median time to answer one question, including retrieval.",
        ),
        (
            "Run cost",
            format_cost(summary.total_cost_usd),
            cost_delta,
            "inverse",
            "Estimated cost of answering and judging every case.",
        ),
    ]
    for row_start in range(0, len(tiles), 4):
        columns = st.columns(4)
        for column, (label, value, delta, colour, help_text) in zip(
            columns, tiles[row_start : row_start + 4], strict=False
        ):
            column.metric(label, value, delta, delta_color=colour, help=help_text, border=True)


def _count(results: list, keep) -> tuple[int, int]:
    """(passed, total) for results matching keep."""
    chosen = [r for r in results if keep(r)]
    return sum(r.passed for r in chosen), len(chosen)


def _line(ok: bool, text: str) -> str:
    icon = ":green[:material/check_circle:]" if ok else ":orange[:material/error:]"
    return f"{icon} {text}"


def render_story(run: EvalRun) -> None:
    """The run in plain English, for someone seeing it for the first time."""
    results, summary = run.results, run.summary
    answered, answerable = _count(
        results,
        lambda r: r.case.expected_behaviour == "answer" and r.case.category != "prompt_injection",
    )
    handed, should_hand = _count(results, lambda r: r.case.category == "out_of_scope")
    traps, trap_total = _count(results, lambda r: r.case.category == "superseded_trap")
    resisted, attacks = _count(results, lambda r: r.case.category == "prompt_injection")
    failed = summary.total_cases - summary.passed_cases
    precise = summary.handoff_precision in (None, 1.0)

    with st.container(border=True):
        left, right = st.columns([1, 3], vertical_alignment="center")
        left.metric(
            "Cases passed",
            f"{summary.passed_cases} / {summary.total_cases}",
            help="A case passes only if the code checks and the LLM judge both agree.",
        )
        right.markdown(
            "\n".join(
                f"- {line}"
                for line in [
                    _line(
                        answered == answerable,
                        f"Answered **{answered} of {answerable}** answerable questions correctly.",
                    ),
                    _line(
                        handed == should_hand,
                        f"Handed **{handed} of {should_hand}** out-of-scope questions to a person.",
                    ),
                    _line(
                        precise,
                        "Every hand-off was justified."
                        if precise
                        else "Some questions it could answer were handed off.",
                    ),
                    _line(
                        traps == trap_total,
                        f"Followed the current policy in **{traps} of {trap_total}** "
                        "superseded traps.",
                    ),
                    _line(
                        resisted == attacks,
                        f"Resisted **{resisted} of {attacks}** prompt injection attempts.",
                    ),
                    _line(
                        failed == 0,
                        "No failures." if failed == 0 else f"**{failed}** failed: see below.",
                    ),
                ]
            )
        )


def render_failure(result) -> None:
    """One failing case as a card: question, expected against actual, and why it failed."""
    case = result.case
    with st.container(border=True):
        st.markdown(
            f":orange-badge[{CATEGORY_NAMES[case.category]}] :gray-badge[{case.id}] "
            f"**{md_safe(case.question)}**"
        )
        expected_col, actual_col = st.columns(2)
        with expected_col:
            st.caption("Expected")
            docs = f" from `{', '.join(case.expected_doc_ids)}`" if case.expected_doc_ids else ""
            st.markdown(f"**{case.expected_behaviour.title()}**{docs}")
            if case.notes:
                st.caption(md_safe(case.notes))
        with actual_col:
            st.caption("What happened")
            cited = f", cited `{', '.join(result.cited_doc_ids)}`" if result.cited_doc_ids else ""
            st.markdown(f"**{result.actual_behaviour.title()}**{cited}")
            st.caption(md_safe(result.answer_text[:600]))
        st.markdown(f":orange[:material/gavel:] **Why it failed:** {md_safe(result.reason)}")
        st.caption(f"Best retrieval score: {result.best_score:.3f}")


def category_chart(run: EvalRun, compare: EvalRun | None) -> go.Figure:
    """Pass rate by category, with the comparison run as a second series if chosen."""
    categories = list(CATEGORY_NAMES)
    counts = {c: sum(r.case.category == c for r in run.results) for c in categories}
    names = [f"{CATEGORY_NAMES[c]} ({counts[c]})" for c in categories]
    series = [(run, "Selected run", CHART_TEAL)]
    if compare:
        series.append((compare, "Comparison run", CHART_ORANGE))
    fig = go.Figure()
    for item, name, colour in series:
        values = [item.summary.pass_rate_by_category.get(c, 0) * 100 for c in categories]
        fig.add_bar(
            y=names,
            x=values,
            name=name,
            orientation="h",
            marker={"color": colour, "cornerradius": 4},
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
            textfont={"color": INK, "size": 12},
            hovertemplate="%{y}<br>" + name + ": %{x:.0f}%<extra></extra>",
        )
    fig.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.08,
        height=90 + 26 * len(categories) * len(series),
        margin={"l": 10, "r": 40, "t": 30, "b": 10},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": MUTED},
        showlegend=compare is not None,
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"range": [0, 112], "ticksuffix": "%", "gridcolor": "#ece8df", "zeroline": False},
        yaxis={"tickfont": {"color": INK, "size": 13}, "autorange": "reversed"},
    )
    return fig


def results_frame(run: EvalRun, failures_only: bool) -> pd.DataFrame:
    """Case results as a table."""
    rows = [
        {
            "Case": r.case.id,
            "Category": CATEGORY_NAMES[r.case.category],
            "Question": r.case.question,
            "Expected": r.case.expected_behaviour
            + (f" ({', '.join(r.case.expected_doc_ids)})" if r.case.expected_doc_ids else ""),
            "Actual": r.actual_behaviour
            + (f" ({r.handoff_code})" if r.handoff_code else "")
            + (f" ({', '.join(r.cited_doc_ids)})" if r.cited_doc_ids else ""),
            "Answer": r.answer_text,
            "Why it failed" if failures_only else "Reason": r.reason,
            "Best score": r.best_score,
            "Passed": r.passed,
        }
        for r in run.results
        if not (failures_only and r.passed)
    ]
    frame = pd.DataFrame(rows)
    if failures_only and not frame.empty:
        frame = frame.drop(columns=["Passed"])
    return frame


def changed_cases(run: EvalRun, compare: EvalRun) -> pd.DataFrame:
    """Cases whose pass or fail result differs between two runs."""
    before = {r.case.id: r for r in compare.results}
    rows = []
    for r in run.results:
        old = before.get(r.case.id)
        if old and old.passed != r.passed:
            rows.append(
                {
                    "Case": r.case.id,
                    "Category": CATEGORY_NAMES[r.case.category],
                    "Change": "Now passes" if r.passed else "Now fails",
                    "Question": r.case.question,
                    "Selected run": r.reason,
                    "Comparison run": old.reason,
                }
            )
    return pd.DataFrame(rows)


def config_table(run: EvalRun, compare: EvalRun | None) -> pd.DataFrame:
    """Side-by-side settings and metrics for two runs."""

    def column(item: EvalRun) -> list[str]:
        s, c = item.summary, item.config
        return [
            c.chat_model,
            c.embed_model,
            str(c.top_k),
            f"{c.handoff_threshold:.2f}",
            c.prompt_version,
            f"{s.passed_cases}/{s.total_cases}",
            *[pct(getattr(s, field)) for field, _, _ in RATE_METRICS],
            f"{s.median_latency_s or 0:.2f}s",
            format_cost(s.total_cost_usd),
        ]

    index = [
        "Chat model",
        "Embedding model",
        "Top K",
        "Hand-off threshold",
        "Prompt version",
        "Cases passed",
        *[label for _, label, _ in RATE_METRICS],
        "Median latency",
        "Cost",
    ]
    data = {"Selected run": column(run)}
    if compare:
        data["Comparison run"] = column(compare)
    return pd.DataFrame(data, index=index)


def run_panel(cases_count: int) -> None:
    """Controls to start a new evaluation run."""
    try:
        settings = load_settings()
    except ConfigError as exc:
        st.info(f"To run a new evaluation, add your API key. {exc}", icon=":material/key:")
        return

    estimate = estimate_eval_run_cost(cases_count, settings.chat_model, settings.embed_model)
    st.caption(
        f"{cases_count} cases · {settings.chat_model} · one answer and one judge call per case · "
        f"estimated cost up to {format_cost(estimate)} · takes about 1 to 2 minutes"
    )
    left, middle, right = st.columns([2, 1, 1], vertical_alignment="bottom")
    label = left.text_input("Label for this run", placeholder="for example: threshold 0.40")
    threshold = middle.number_input(
        "Hand-off threshold", 0.0, 1.0, settings.handoff_threshold, 0.01, format="%.2f"
    )
    if not right.button(
        "Run evaluation", type="primary", icon=":material/play_arrow:", width="stretch"
    ):
        return

    settings = settings.model_copy(update={"handoff_threshold": threshold})
    progress = st.progress(0.0, text="Starting…")

    def on_progress(i: int, total: int, case) -> None:
        text = "Finishing…" if i >= total else f"Case {i + 1} of {total}: {case.id}"
        progress.progress(i / total, text=text)

    try:
        client = get_client(settings)
        retriever = get_retriever(client, settings.top_k)
        run = run_evaluation(
            load_test_set(), retriever, client, settings, progress=on_progress, label=label
        )
    except LLMError as exc:
        st.error(str(exc), icon=":material/cloud_off:")
        return
    path = save_run(run)
    st.session_state["selected_run"] = str(path)
    st.rerun()


header("Evaluation", "Where the assistant can be trusted, and where it fails.")

cases = load_test_set()
with st.expander("Run a new evaluation", icon=":material/play_circle:"):
    run_panel(len(cases))

paths = list_runs(RUNS_DIR)
if not paths:
    st.info(
        "No evaluation runs yet. Run one above, or from a terminal with "
        "`python -m arc_assistant.evaluate`.",
        icon=":material/info:",
    )
    st.stop()

runs = {str(p): load_run(p) for p in paths}
labels = {key: run_label(run, p.stem) for (key, run), p in zip(runs.items(), paths, strict=True)}
keys = list(runs)
baseline_key = next((k for k, p in zip(keys, paths, strict=True) if p.stem == "baseline"), None)
selected_default = st.session_state.get("selected_run", baseline_key)
left, right = st.columns(2)
selected = left.selectbox(
    "Run",
    keys,
    index=keys.index(selected_default) if selected_default in keys else 0,
    format_func=labels.get,
)
compare_options = ["none", *[k for k in keys if k != selected]]
first_run = next((k for k, p in zip(keys, paths, strict=True) if p.stem == "first-run"), None)
next_older = next((k for k in keys[keys.index(selected) + 1 :]), "none")
default_compare = first_run if first_run in compare_options else next_older
compare_key = right.selectbox(
    "Compare with",
    compare_options,
    index=compare_options.index(default_compare),
    format_func=lambda k: "No comparison" if k == "none" else labels[k],
)
run = runs[selected]
compare = runs.get(compare_key)

render_story(run)

st.markdown("#### Headline metrics")
if compare:
    st.caption("Changes are shown against the comparison run.")
render_metrics(run.summary, compare.summary if compare else None)

st.markdown("#### Pass rate by category")
st.plotly_chart(category_chart(run, compare), width="stretch", config={"displayModeBar": False})
with st.expander("What each category tests", icon=":material/help:"):
    st.markdown(
        "\n".join(
            f"- **{CATEGORY_NAMES[c]}:** {help_text}" for c, help_text in CATEGORY_HELP.items()
        )
    )

st.markdown("#### Where it fails")
failures = results_frame(run, failures_only=True)
if failures.empty:
    st.success("Every case passed in this run.", icon=":material/check_circle:")
else:
    st.caption(
        f"{len(failures)} of {run.summary.total_cases} cases failed. Each card shows what was "
        "expected, what happened, and why the check or the judge failed it."
    )
    for result in run.results:
        if not result.passed:
            render_failure(result)
    with st.expander("Failures as a table"):
        st.dataframe(
            failures,
            hide_index=True,
            width="stretch",
            column_config={
                "Question": st.column_config.TextColumn(width="medium"),
                "Answer": st.column_config.TextColumn(width="large"),
                "Why it failed": st.column_config.TextColumn(width="large"),
                "Best score": st.column_config.NumberColumn(format="%.3f"),
            },
        )

if compare:
    st.markdown("#### What changed between the two runs")
    changes = changed_cases(run, compare)
    if changes.empty:
        st.caption("No case changed between pass and fail.")
    else:
        st.dataframe(changes, hide_index=True, width="stretch")
    with st.expander("Settings and metrics side by side"):
        st.dataframe(config_table(run, compare), width="stretch")

with st.expander("All case results"):
    st.dataframe(results_frame(run, failures_only=False), hide_index=True, width="stretch")
