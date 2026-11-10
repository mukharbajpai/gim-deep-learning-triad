"""Metric and result-card primitives."""
from __future__ import annotations

import html

import streamlit as st


def pct(value: float, decimals: int = 2, already_percent: bool = False) -> str:
    number = float(value) if already_percent else 100 * float(value)
    return f"{number:.{decimals}f}%"


def number(value: float, decimals: int = 5) -> str:
    return f"{float(value):,.{decimals}f}"


def metric_card(label: str, value: str, note: str = "", help_text: str = "") -> None:
    tooltip = f' title="{html.escape(help_text)}"' if help_text else ""
    st.markdown(
        f'<div class="metric-card"{tooltip}>'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def experiment_card(
    index: str,
    title: str,
    model: str,
    dataset: str,
    stats: list[tuple[str, str]],
) -> None:
    stat_html = "".join(
        f'<div class="card-stat"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong></div>'
        for label, value in stats
    )
    st.markdown(
        f'<div class="experiment-card"><div class="experiment-index">{html.escape(index)}</div>'
        f'<h3>{html.escape(title)}</h3><div class="experiment-meta">'
        f'{html.escape(model)} · {html.escape(dataset)}</div>{stat_html}</div>',
        unsafe_allow_html=True,
    )


def mode_selector(key: str) -> str:
    label = st.radio(
        "Evaluation scale",
        ("Full run", "Fast smoke run"),
        horizontal=True,
        key=key,
        help="Full runs are final evidence; fast runs are small compatibility checks.",
    )
    return "full" if label == "Full run" else "fast"
