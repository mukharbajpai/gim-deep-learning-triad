"""Interactive convolutional VAE/Fashion-MNIST results page."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from components.charts import (  # noqa: E402
    elbo_decomposition,
    vae_baseline,
    vae_learning_curves,
    vae_mode_comparison,
)
from components.metrics import metric_card, mode_selector, number  # noqa: E402
from components.styles import apply_styles, hero, sidebar_identity  # noqa: E402
from utils.data_loader import artifact_path, load_metrics  # noqa: E402

st.set_page_config(page_title="VAE · Deep Learning Triad", page_icon="◇", layout="wide")
apply_styles()
sidebar_identity("Variational Autoencoder")
hero(
    "Experiment 02 · Generative representation learning",
    'A <span>regularized latent space</span> in motion',
    "Convolutional VAE reconstruction on Fashion-MNIST, evaluated beside a matched deterministic autoencoder.",
)

mode = mode_selector("vae_mode")
try:
    metrics = load_metrics("vae", mode)
    fast = load_metrics("vae", "fast")
    full = load_metrics("vae", "full")
except Exception as exc:
    st.error(f"Saved VAE results are unavailable: {exc}")
    st.stop()

cols = st.columns(6)
cards = [
    ("VAE reconstruction", number(metrics["test_reconstruction_bce_per_image"]), "BCE / image", "Reconstruction term only; lower is better."),
    ("KL divergence", number(metrics["test_kl_per_image"]), "KL / image", "Regularization toward the latent prior."),
    ("VAE ELBO loss", number(metrics["test_elbo_loss_per_image"]), "BCE + KL / image", "The VAE objective combines reconstruction and KL terms."),
    ("VAE pixel error", number(metrics["test_mse_per_pixel"]), "MSE / pixel", "Mean squared reconstruction error per pixel."),
    ("AE reconstruction", number(metrics["deterministic_ae_baseline_test_bce_per_image"]), "BCE / image", "Matched deterministic autoencoder reconstruction."),
    ("AE pixel error", number(metrics["deterministic_ae_baseline_test_mse_per_pixel"]), "MSE / pixel", "Matched deterministic autoencoder pixel error."),
]
for column, card in zip(cols, cards):
    with column:
        metric_card(*card)

tab_learning, tab_recon, tab_compare, tab_arch = st.tabs(
    ["Learning curves", "Reconstruction gallery", "Objective & baseline", "Architecture"]
)

with tab_learning:
    controls = st.columns([1.4, 2])
    with controls[0]:
        selected_metric = st.selectbox(
            "Curve",
            ("ELBO / image", "Reconstruction BCE / image", "MSE / pixel"),
        )
    with controls[1]:
        max_epoch = st.slider("Show through epoch", 1, int(metrics["epochs"]), int(metrics["epochs"]))
    st.plotly_chart(vae_learning_curves(metrics, selected_metric, max_epoch), use_container_width=True)
    st.caption("Interactive curves are reconstructed directly from the saved `history` and `baseline_history` records.")

with tab_recon:
    panel = artifact_path(mode, "vae_reconstructions.png")
    if panel:
        st.image(
            str(panel),
            caption=f"{mode.title()} run · Saved panel: original inputs (top), VAE reconstructions (middle), deterministic AE reconstructions (bottom)",
            use_container_width=True,
        )
    else:
        st.warning("The optional saved reconstruction panel is unavailable; metric exploration remains functional.")
    st.info(
        "Individual image tensors were not saved separately. This page displays the genuine generated "
        "panel rather than splitting or simulating Fashion-MNIST samples."
    )

with tab_compare:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(vae_baseline(metrics), use_container_width=True)
    with right:
        st.plotly_chart(elbo_decomposition(metrics), use_container_width=True)
    st.plotly_chart(vae_mode_comparison(fast, full), use_container_width=True)
    st.markdown(
        '<div class="callout"><strong>Why the AE reconstructs better.</strong> The deterministic AE '
        "optimizes reconstruction alone. The VAE also pays a KL penalty to organize its 16-dimensional "
        "latent distribution so it can be sampled. AE reconstruction BCE and VAE ELBO therefore must "
        "not be treated as equivalent objectives.</div>",
        unsafe_allow_html=True,
    )

with tab_arch:
    st.markdown("### Convolutional encoder → stochastic bottleneck → decoder")
    st.markdown(
        f"""
        1. Two strided convolutions compress each 28×28 grayscale image.
        2. Separate heads predict **μ** and **log σ²** in a {metrics['latent_dim']}-dimensional space.
        3. Reparameterized sampling produces **z = μ + σ ⊙ ε**.
        4. Transpose convolutions reconstruct the image.
        5. The objective is reconstruction BCE + **β × KL**, with β = {metrics['beta']:.1f}.
        """
    )
    st.caption(metrics["split_protocol"])
    st.markdown(
        f"**Run:** {metrics['train_size']:,} train / {metrics['validation_size']:,} validation / "
        f"{metrics['test_size']:,} test · {metrics['epochs']} epochs · seed {metrics['seed']}"
    )
