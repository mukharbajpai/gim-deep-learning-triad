"""Interactive, auditable explorer for saved grounded QA predictions."""
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import streamlit as st

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from components.charts import (  # noqa: E402
    qa_bottleneck,
    qa_mode_comparison,
    qa_oracle,
    retrieval_metrics,
)
from components.metrics import metric_card, mode_selector, pct  # noqa: E402
from components.styles import apply_styles, hero, sidebar_identity  # noqa: E402
from utils.data_loader import load_metrics, load_predictions  # noqa: E402


def failure_diagnosis(record: dict[str, Any]) -> str:
    """Explain a failure using only booleans and scores in the saved record."""
    if record["exact_match"]:
        return "Exact match: the saved prediction matches at least one normalized reference answer."
    if not record["gold_context_in_top3"]:
        return "Retrieval failure: the gold evidence was absent from the top three TF-IDF contexts."
    if not record["selected_context_correct"]:
        return "Evidence-selection failure: gold evidence was retrieved, but a different context won reranking."
    if record["oracle_context_exact_match"]:
        return "Extraction/reranking failure: the oracle run succeeds on the gold context, but the end-to-end answer does not."
    if float(record["f1"]) > 0:
        return "Partial lexical match: the prediction overlaps the reference but is not an exact normalized match."
    return "Extraction failure: even the saved oracle-context prediction does not exactly match the reference."


def answer_box(label: str, value: str, status: str = "") -> None:
    css = f" {status}" if status else ""
    st.markdown(
        f'<div class="answer-box{css}"><div class="answer-label">{html.escape(label)}</div>'
        f'<div class="answer-text">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Grounded QA · Deep Learning Triad", page_icon="◎", layout="wide")
apply_styles()
sidebar_identity("Grounded QA")
hero(
    "Experiment 03 · Auditable AI workflow",
    'Grounded answers, <span>inspectable evidence</span>',
    "A deterministic tool-routed, retrieval-grounded extractive QA workflow evaluated on SQuAD v1.1.",
)

st.markdown(
    '<div class="callout"><strong>Scientific scope.</strong> This is not a fully autonomous or generative '
    "LLM agent. It is a deterministic router connected to TF-IDF retrieval, extractive DistilBERT QA, "
    "provenance, and an abstention rule.</div>",
    unsafe_allow_html=True,
)

mode = mode_selector("qa_mode")
try:
    metrics = load_metrics("agent", mode)
    predictions = load_predictions(mode)
    fast = load_metrics("agent", "fast")
    full = load_metrics("agent", "full")
except Exception as exc:
    st.error(f"Saved QA results are unavailable: {exc}")
    st.stop()

cols = st.columns(4)
cards = [
    ("End-to-end EM", pct(metrics["exact_match_pct"], already_percent=True), f'{metrics["sample_size"]} questions', "Exact normalized answer match after retrieval and extraction."),
    ("Token F1", pct(metrics["token_f1_pct"], already_percent=True), "End-to-end", "Token overlap F1 against the best reference answer."),
    ("Recall@3", pct(metrics["gold_context_recall_at_3_pct"], already_percent=True), "Gold evidence retrieved", "Fraction with gold context among the top three retrieval hits."),
    ("Oracle EM", pct(metrics["oracle_context_exact_match_pct"], already_percent=True), "Gold context supplied", "Extractor exact match when given gold evidence, never the gold answer."),
]
for column, card in zip(cols, cards):
    with column:
        metric_card(*card)

st.markdown("### Question → Tool Router → TF-IDF Retrieval → Extractive QA → Evidence-backed Answer")
st.caption(f"Closed corpus: {metrics['corpus_contexts']:,} unique SQuAD validation contexts · threshold {metrics['threshold']:.2f}")

tab_explorer, tab_system, tab_scale = st.tabs(["Prediction explorer", "System diagnostics", "Fast ↔ full"])

with tab_explorer:
    filters = st.columns([2, 1])
    with filters[0]:
        query = st.text_input("Search questions", placeholder="Try a name, year, place, or topic…")
    with filters[1]:
        outcome = st.selectbox("Exact-match outcome", ("All", "Correct", "Incorrect"))

    filtered = predictions
    if query.strip():
        needle = query.casefold().strip()
        filtered = [row for row in filtered if needle in row["question"].casefold()]
    if outcome == "Correct":
        filtered = [row for row in filtered if bool(row["exact_match"])]
    elif outcome == "Incorrect":
        filtered = [row for row in filtered if not bool(row["exact_match"])]

    st.caption(f"Showing {len(filtered)} of {len(predictions)} saved predictions")
    if not filtered:
        st.warning("No saved questions match these filters.")
    else:
        selected_id = st.selectbox(
            "Select a recorded question",
            options=[row["id"] for row in filtered],
            format_func=lambda item_id: next(row["question"] for row in filtered if row["id"] == item_id),
        )
        record = next(row for row in filtered if row["id"] == selected_id)
        correct = bool(record["exact_match"])
        answer_box("Question", record["question"])
        left, right = st.columns(2)
        with left:
            answer_box("Predicted answer", record["prediction"], "correct" if correct else "incorrect")
        with right:
            answer_box("Gold answer(s)", " · ".join(dict.fromkeys(record["reference_answers"])), "correct")

        score_cols = st.columns(5)
        score_cols[0].metric("Exact match", "Yes" if correct else "No")
        score_cols[1].metric("Token F1", f'{100 * float(record["f1"]):.2f}%')
        score_cols[2].metric("Retrieval score", f'{float(record["retrieval_score"]):.4f}')
        score_cols[3].metric("QA score", f'{float(record["qa_score"]):.4f}')
        score_cols[4].metric("Gold in top 3", "Yes" if record["gold_context_in_top3"] else "No")

        st.markdown(f"**Diagnostic:** {failure_diagnosis(record)}")
        with st.expander("Inspect selected evidence and oracle result", expanded=True):
            st.markdown("**Saved evidence excerpt**")
            st.write(record["source_excerpt"])
            st.caption(
                "Only this 200-character excerpt was stored; the full selected context was not written "
                "to the prediction artifact."
            )
            st.markdown("**Oracle-context prediction**")
            st.write(record["oracle_context_prediction"])
            st.caption(
                f"Oracle exact match: {'yes' if record['oracle_context_exact_match'] else 'no'} · "
                f"oracle token F1: {100 * float(record['oracle_context_f1']):.2f}% · "
                f"source ID: {record['source_id']} · tool: {record['tool']}"
            )

with tab_system:
    row1, row2 = st.columns(2)
    with row1:
        st.plotly_chart(qa_oracle(metrics), use_container_width=True)
    with row2:
        st.plotly_chart(retrieval_metrics(metrics), use_container_width=True)
    st.plotly_chart(qa_bottleneck(metrics), use_container_width=True)
    gap = metrics["oracle_context_exact_match_pct"] - metrics["exact_match_pct"]
    st.markdown(
        f'<div class="callout"><strong>{gap:.1f}-percentage-point oracle gap.</strong> The same extractor '
        "is substantially stronger when gold evidence is supplied. Combined with retrieval and selected-"
        "context metrics, this identifies evidence retrieval/selection as the dominant bottleneck. "
        f"The saved abstention rate is {metrics['abstention_rate_pct']:.1f}%, so the fixed threshold is not calibrated.</div>",
        unsafe_allow_html=True,
    )

with tab_scale:
    st.plotly_chart(qa_mode_comparison(fast, full), use_container_width=True)
    st.info(
        "Fast mode contains 20 recorded questions; full mode contains 100. The selector above switches "
        "both the aggregate diagnostics and the actual saved prediction set."
    )
