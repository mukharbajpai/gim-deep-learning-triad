"""Landing page for the Deep Learning Triad interactive experiment lab."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

DASHBOARD_ROOT = Path(__file__).resolve().parent
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from components.metrics import experiment_card, number, pct
from components.styles import apply_styles, attribution, hero, sidebar_identity
from utils.data_loader import checkpoint_inventory, load_all_metrics

st.set_page_config(
    page_title="Deep Learning Triad · Experiment Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()
sidebar_identity("Overview")

hero(
    "Interactive experiment lab",
    'Deep Learning Triad <span>— Evidence, explored.</span>',
    "Exploring Attention, Generative Representation Learning, and Grounded AI Workflows.",
)
attribution()
st.write("")

try:
    results = load_all_metrics("full")
except Exception as exc:
    st.error(f"The saved full-run results could not be loaded: {exc}")
    st.stop()

transformer = results["transformer"]
vae = results["vae"]
agent = results["agent"]

st.markdown("## Three experiments. Three evaluation lenses.")
st.caption("All values below are read from `results/full/*.json`; no metrics are embedded in the UI.")

col1, col2, col3 = st.columns(3)
with col1:
    experiment_card(
        "01", "Transformer", "DistilBERT", "IMDb",
        [
            ("Test accuracy", pct(transformer["test_accuracy"])),
            ("Binary F1", pct(transformer["test_f1"])),
            ("Majority baseline", pct(transformer["majority_baseline_test_accuracy"])),
        ],
    )
    st.page_link("pages/1_Transformer.py", label="Explore attention & classification →", use_container_width=True)
with col2:
    experiment_card(
        "02", "Variational Autoencoder", "Convolutional VAE", "Fashion-MNIST",
        [
            ("Reconstruction BCE/image", number(vae["test_reconstruction_bce_per_image"])),
            ("KL/image", number(vae["test_kl_per_image"])),
            ("MSE/pixel", number(vae["test_mse_per_pixel"])),
        ],
    )
    st.page_link("pages/2_VAE.py", label="Explore the latent space →", use_container_width=True)
with col3:
    experiment_card(
        "03", "Grounded QA", "TF-IDF + Extractive DistilBERT", "SQuAD v1.1",
        [
            ("End-to-end exact match", pct(agent["exact_match_pct"], already_percent=True)),
            ("Token F1", pct(agent["token_f1_pct"], already_percent=True)),
            ("Retrieval Recall@3", pct(agent["gold_context_recall_at_3_pct"], already_percent=True)),
        ],
    )
    st.page_link("pages/3_Grounded_QA.py", label="Inspect evidence & predictions →", use_container_width=True)

st.markdown("## A comparison that respects the task")
left, right = st.columns([1.25, 1])
with left:
    st.markdown(
        '<div class="callout"><strong>Do not rank these headline numbers.</strong><br>'
        "Classification accuracy, reconstruction loss, and exact match answer different scientific "
        "questions and use different scales. Each experiment is interpreted only against its own "
        "baseline, oracle, or objective decomposition.</div>",
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        | Workload | Valid comparison |
        |---|---|
        | Sentiment | DistilBERT vs majority class |
        | Reconstruction | VAE vs deterministic AE |
        | Grounded QA | End-to-end vs oracle context |
        """
    )

with st.expander("Saved results and optional checkpoint status"):
    st.write(
        "The complete dashboard is a read-only explorer of committed artifacts and works without "
        "loading a model. Checkpoints are optional, ignored by Git, and never loaded on this page."
    )
    for checkpoint, available in checkpoint_inventory().items():
        st.markdown(f"- {'Available locally' if available else 'Not present'} — **{checkpoint}**")

st.caption("Full evaluation · Seed 42 · Saved artifacts produced 9 November 2026")
