"""Visual system and small HTML helpers for the Streamlit application."""
from __future__ import annotations

import html

import streamlit as st


COLORS = {
    "bg": "#07111f",
    "panel": "#0e1b2c",
    "panel_2": "#12233a",
    "text": "#e8f1ff",
    "muted": "#91a4bf",
    "cyan": "#28d7e5",
    "teal": "#24c7a8",
    "violet": "#9b7cff",
    "amber": "#ffbf69",
    "red": "#ff6b82",
    "grid": "#233852",
}


def apply_styles() -> None:
    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: dark; }}
        .stApp {{
            background:
              radial-gradient(circle at 92% 6%, rgba(155,124,255,.13), transparent 27rem),
              radial-gradient(circle at 8% 25%, rgba(40,215,229,.08), transparent 24rem),
              {COLORS['bg']};
            color: {COLORS['text']};
        }}
        [data-testid="stSidebar"] {{
            background: rgba(8, 20, 35, .97);
            border-right: 1px solid rgba(145,164,191,.14);
        }}
        [data-testid="stHeader"] {{ background: rgba(7,17,31,.78); }}
        .block-container {{ max-width: 1220px; padding-top: 2.2rem; padding-bottom: 4rem; }}
        h1, h2, h3 {{ letter-spacing: -.025em; }}
        h1 {{ font-weight: 760 !important; }}
        p, li {{ color: {COLORS['muted']}; line-height: 1.65; }}
        .eyebrow {{ color: {COLORS['cyan']}; font-size: .75rem; font-weight: 750;
            letter-spacing: .15em; text-transform: uppercase; margin-bottom: .55rem; }}
        .hero-title {{ font-size: clamp(2.25rem, 5vw, 4.35rem); line-height: 1.02;
            letter-spacing: -.055em; font-weight: 800; margin: 0 0 .8rem; color: {COLORS['text']}; }}
        .hero-title span {{ background: linear-gradient(100deg, {COLORS['cyan']}, {COLORS['violet']});
            -webkit-background-clip: text; color: transparent; }}
        .hero-copy {{ max-width: 780px; font-size: 1.08rem; margin-bottom: 1.7rem; }}
        .attribution {{ display: inline-flex; padding: .44rem .72rem; border-radius: 999px;
            border: 1px solid rgba(40,215,229,.26); color: #b9cbe1; font-size: .77rem;
            background: rgba(40,215,229,.06); }}
        .metric-card {{ min-height: 125px; padding: 1.15rem 1.15rem 1rem; border-radius: 16px;
            border: 1px solid rgba(145,164,191,.14); background: linear-gradient(145deg,
            rgba(18,35,58,.94), rgba(11,25,42,.94)); box-shadow: 0 12px 36px rgba(0,0,0,.15);
            transition: transform .18s ease, border-color .18s ease; }}
        .metric-card:hover {{ transform: translateY(-2px); border-color: rgba(40,215,229,.34); }}
        .metric-label {{ color: {COLORS['muted']}; font-size: .76rem; text-transform: uppercase;
            letter-spacing: .075em; font-weight: 720; }}
        .metric-value {{ color: {COLORS['text']}; font-size: 1.72rem; font-weight: 780;
            margin: .35rem 0 .22rem; letter-spacing: -.035em; }}
        .metric-note {{ color: #7f95b2; font-size: .74rem; line-height: 1.35; }}
        .experiment-card {{ padding: 1.35rem; border-radius: 20px; min-height: 330px;
            border: 1px solid rgba(145,164,191,.15); background: linear-gradient(155deg,
            rgba(18,35,58,.92), rgba(9,21,37,.96)); }}
        .experiment-index {{ width: 2rem; height: 2rem; display: grid; place-items: center;
            border-radius: .65rem; color: {COLORS['bg']}; font-weight: 800;
            background: linear-gradient(135deg, {COLORS['cyan']}, {COLORS['violet']}); }}
        .experiment-card h3 {{ margin: .9rem 0 .1rem; }}
        .experiment-meta {{ color: {COLORS['muted']}; font-size: .8rem; margin-bottom: 1rem; }}
        .card-stat {{ display: flex; justify-content: space-between; gap: .75rem; padding: .52rem 0;
            border-top: 1px solid rgba(145,164,191,.1); font-size: .82rem; }}
        .card-stat span {{ color: {COLORS['muted']}; }} .card-stat strong {{ color: {COLORS['text']}; }}
        .callout {{ border-left: 3px solid {COLORS['cyan']}; border-radius: 0 14px 14px 0;
            background: rgba(40,215,229,.055); padding: .95rem 1.05rem; color: #b8c9df;
            margin: .65rem 0 1rem; }}
        .answer-box {{ padding: 1rem 1.1rem; border-radius: 14px; background: rgba(18,35,58,.82);
            border: 1px solid rgba(145,164,191,.15); margin-bottom: .75rem; }}
        .answer-box.correct {{ border-color: rgba(36,199,168,.48); background: rgba(36,199,168,.08); }}
        .answer-box.incorrect {{ border-color: rgba(255,107,130,.46); background: rgba(255,107,130,.07); }}
        .answer-label {{ color: {COLORS['muted']}; font-size: .72rem; letter-spacing: .08em;
            text-transform: uppercase; font-weight: 750; margin-bottom: .4rem; }}
        .answer-text {{ color: {COLORS['text']}; font-size: 1rem; line-height: 1.55; }}
        div[data-testid="stPlotlyChart"] {{ border-radius: 16px; overflow: hidden;
            border: 1px solid rgba(145,164,191,.11); }}
        div[data-testid="stExpander"] {{ border-color: rgba(145,164,191,.15); border-radius: 14px; }}
        .stButton > button, .stLinkButton > a {{ border-radius: 10px; border-color: rgba(40,215,229,.3); }}
        @media (max-width: 700px) {{ .block-container {{ padding: 1.25rem 1rem 3rem; }}
            .hero-title {{ font-size: 2.35rem; }} .metric-card {{ min-height: 108px; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{html.escape(kicker)}</div>'
        f'<div class="hero-title">{title}</div>'
        f'<p class="hero-copy">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )


def attribution() -> None:
    st.markdown(
        '<span class="attribution">Deep Learning &nbsp;|&nbsp; GIM BDA 2025–27 '
        '&nbsp;|&nbsp; Dr. Anup Kumar Maurya</span>',
        unsafe_allow_html=True,
    )


def sidebar_identity(active: str) -> None:
    with st.sidebar:
        st.markdown("### ◈ Deep Learning Triad")
        st.caption(f"Interactive Experiment Lab · {active}")
        st.markdown("---")
        st.page_link("app.py", label="Overview", icon="🏠")
        st.page_link("pages/1_Transformer.py", label="Transformer", icon="🔤")
        st.page_link("pages/2_VAE.py", label="VAE", icon="🧬")
        st.page_link("pages/3_Grounded_QA.py", label="Grounded QA", icon="🔎")
        st.markdown("---")
        st.caption("Saved-result explorer · Seed 42 · No retraining")
