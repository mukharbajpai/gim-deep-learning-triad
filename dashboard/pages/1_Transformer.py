"""Interactive DistilBERT/IMDb results page."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from components.charts import (  # noqa: E402
    class_metrics,
    confusion_matrix,
    dataset_split,
    transformer_baseline,
    transformer_mode_comparison,
)
from components.metrics import metric_card, mode_selector, pct  # noqa: E402
from components.styles import apply_styles, hero, sidebar_identity  # noqa: E402
from utils.data_loader import load_classification_report, load_metrics  # noqa: E402

st.set_page_config(page_title="Transformer · Deep Learning Triad", page_icon="◫", layout="wide")
apply_styles()
sidebar_identity("Transformer")
hero(
    "Experiment 01 · Discriminative learning",
    'DistilBERT <span>sentiment analytics</span>',
    "Fine-tuned binary sentiment classification on a held-out IMDb test split.",
)

mode = mode_selector("transformer_mode")
try:
    metrics = load_metrics("transformer", mode)
    fast = load_metrics("transformer", "fast")
    full = load_metrics("transformer", "full")
except Exception as exc:
    st.error(f"Saved Transformer results are unavailable: {exc}")
    st.stop()

cols = st.columns(5)
cards = [
    ("Test accuracy", pct(metrics["test_accuracy"]), f'{metrics["test_size"]:,} held-out reviews', "Fraction of held-out reviews assigned the correct sentiment."),
    ("Binary F1", pct(metrics["test_f1"]), "Positive class", "Harmonic mean of positive-class precision and recall."),
    ("Macro F1", pct(metrics["test_macro_f1"]), "Equal class weight", "Unweighted mean of negative- and positive-class F1."),
    ("Validation accuracy", pct(metrics["validation_accuracy"]), f'{metrics["validation_size"]:,} validation reviews', "Accuracy on the non-overlapping validation split."),
    ("Majority baseline", pct(metrics["majority_baseline_test_accuracy"]), "Always predicts class 0", "Accuracy obtained by always choosing the most frequent test label."),
]
for column, card in zip(cols, cards):
    with column:
        metric_card(*card)

tab_eval, tab_scale, tab_protocol = st.tabs(["Evaluation", "Fast ↔ full", "Protocol"])
with tab_eval:
    normalized = st.toggle("Normalize by actual class", value=False, key=f"cm_{mode}")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(confusion_matrix(metrics, normalized), use_container_width=True)
    with right:
        st.plotly_chart(transformer_baseline(metrics), use_container_width=True)
    report = load_classification_report(mode)
    if report:
        st.plotly_chart(class_metrics(report), use_container_width=True)
    else:
        st.info("The optional saved class-level report is not available for this run.")

    improvement = 100 * (metrics["test_accuracy"] - metrics["majority_baseline_test_accuracy"])
    st.markdown(
        f'<div class="callout"><strong>Interpretation.</strong> DistilBERT improves test accuracy by '
        f"<strong>{improvement:.2f} percentage points</strong> over the majority-class baseline. "
        "Validation and test performance are close, while the confusion matrix exposes the remaining "
        "class-specific errors.</div>",
        unsafe_allow_html=True,
    )

with tab_scale:
    st.plotly_chart(transformer_mode_comparison(fast, full), use_container_width=True)
    st.info(
        "Fast mode is a 200-review smoke test after one epoch; full mode is the final 5,000-review "
        "evaluation after two epochs. The two scales are shown for provenance, not as repeated trials."
    )

with tab_protocol:
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(dataset_split(metrics, "IMDb split sizes"), use_container_width=True)
    with right:
        st.markdown("### Run configuration")
        st.markdown(
            f"""
            - **Model:** `{metrics['model']}`
            - **Epochs:** {metrics['epochs']}
            - **Learning rate:** {metrics['learning_rate']:.0e}
            - **Maximum sequence:** 192 tokens
            - **Seed:** {metrics['seed']}
            - **Device recorded:** {metrics['compute_device']}
            """
        )
        st.caption(metrics["split_protocol"])
    st.info(
        "No per-review Transformer predictions were saved in the repository, so this page deliberately "
        "omits a prediction explorer rather than inventing examples."
    )
