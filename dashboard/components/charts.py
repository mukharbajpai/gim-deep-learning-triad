"""Plotly figures built exclusively from saved experiment metrics."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .styles import COLORS


def _layout(fig: go.Figure, title: str, y_title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.03, font=dict(size=18, color=COLORS["text"])),
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["muted"], family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=48, r=25, t=68, b=48),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        hoverlabel=dict(bgcolor=COLORS["panel_2"], font_color=COLORS["text"]),
        yaxis_title=y_title,
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


def confusion_matrix(metrics: dict[str, Any], normalized: bool) -> go.Figure:
    matrix = metrics["test_confusion_matrix"]
    z = [[float(value) for value in row] for row in matrix]
    text = [[str(value) for value in row] for row in matrix]
    if normalized:
        z = [[value / sum(row) for value in row] for row in z]
        text = [[f"{value:.1%}" for value in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted negative", "Predicted positive"],
            y=["Actual negative", "Actual positive"],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, COLORS["panel_2"]], [0.45, "#176f83"], [1, COLORS["cyan"]]],
            hovertemplate="%{y}<br>%{x}<br>Value: %{text}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Test confusion matrix")


def transformer_baseline(metrics: dict[str, Any]) -> go.Figure:
    frame = pd.DataFrame(
        {
            "System": ["DistilBERT", "Majority baseline"],
            "Accuracy": [100 * metrics["test_accuracy"], 100 * metrics["majority_baseline_test_accuracy"]],
        }
    )
    fig = px.bar(frame, x="System", y="Accuracy", color="System", text_auto=".2f",
                 color_discrete_sequence=[COLORS["cyan"], COLORS["violet"]])
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(showlegend=False)
    return _layout(fig, "Model versus trivial baseline", "Accuracy (%)")


def transformer_mode_comparison(fast: dict[str, Any], full: dict[str, Any]) -> go.Figure:
    rows = []
    for label, payload in (("Fast", fast), ("Full", full)):
        for metric, key in (("Accuracy", "test_accuracy"), ("Binary F1", "test_f1"), ("Macro F1", "test_macro_f1")):
            rows.append({"Mode": label, "Metric": metric, "Percent": 100 * payload[key]})
    fig = px.bar(pd.DataFrame(rows), x="Metric", y="Percent", color="Mode", barmode="group",
                 color_discrete_map={"Fast": COLORS["violet"], "Full": COLORS["cyan"]})
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f}%<extra></extra>")
    fig.update_yaxes(range=[0, 100])
    return _layout(fig, "Fast smoke test versus full evaluation", "Score (%)")


def class_metrics(rows: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(rows).melt(id_vars=["class", "support"], value_vars=["precision", "recall", "f1"],
                                    var_name="Metric", value_name="Score")
    fig = px.bar(frame, x="class", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=[COLORS["cyan"], COLORS["violet"], COLORS["teal"]])
    fig.update_traces(hovertemplate="%{x} %{fullData.name}: %{y:.2f}<extra></extra>")
    fig.update_yaxes(range=[0, 1])
    return _layout(fig, "Saved class-level classification report", "Score")


def dataset_split(metrics: dict[str, Any], title: str = "Evaluation protocol") -> go.Figure:
    frame = pd.DataFrame({"Split": ["Train", "Validation", "Test"], "Examples": [
        metrics["train_size"], metrics["validation_size"], metrics["test_size"]]})
    fig = px.bar(frame, x="Split", y="Examples", color="Split", text_auto=",",
                 color_discrete_sequence=[COLORS["cyan"], COLORS["violet"], COLORS["teal"]])
    fig.update_layout(showlegend=False)
    fig.update_traces(hovertemplate="%{x}: %{y:,} examples<extra></extra>")
    return _layout(fig, title, "Examples")


def vae_learning_curves(metrics: dict[str, Any], metric: str, max_epoch: int) -> go.Figure:
    history = pd.DataFrame(metrics["history"])
    baseline = pd.DataFrame(metrics["baseline_history"])
    history = history[history["epoch"] <= max_epoch]
    baseline = baseline[baseline["epoch"] <= max_epoch]
    fig = go.Figure()
    if metric == "ELBO / image":
        fig.add_scatter(x=history.epoch, y=history.train_loss_per_image, name="VAE train ELBO", mode="lines+markers", line=dict(color=COLORS["violet"]))
        fig.add_scatter(x=history.epoch, y=history.val_loss_per_image, name="VAE validation ELBO", mode="lines+markers", line=dict(color=COLORS["cyan"]))
        title, y_title = "VAE ELBO learning curve", "BCE + KL per image"
    elif metric == "Reconstruction BCE / image":
        fig.add_scatter(x=history.epoch, y=history.val_reconstruction_bce_per_image, name="VAE validation reconstruction", mode="lines+markers", line=dict(color=COLORS["cyan"]))
        fig.add_scatter(x=baseline.epoch, y=baseline.val_bce_per_image, name="AE validation reconstruction", mode="lines+markers", line=dict(color=COLORS["teal"]))
        title, y_title = "Validation reconstruction learning curve", "BCE per image"
    else:
        fig.add_scatter(x=history.epoch, y=history.val_mse_per_pixel, name="VAE validation", mode="lines+markers", line=dict(color=COLORS["cyan"]))
        fig.add_scatter(x=baseline.epoch, y=baseline.val_mse_per_pixel, name="AE validation", mode="lines+markers", line=dict(color=COLORS["teal"]))
        title, y_title = "Validation pixel error", "MSE per pixel"
    fig.update_xaxes(dtick=1, title="Epoch")
    return _layout(fig, title, y_title)


def vae_baseline(metrics: dict[str, Any]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["VAE", "Deterministic AE"],
        y=[metrics["test_reconstruction_bce_per_image"], metrics["deterministic_ae_baseline_test_bce_per_image"]],
        marker_color=[COLORS["violet"], COLORS["teal"]],
        text=[f'{metrics["test_reconstruction_bce_per_image"]:.2f}', f'{metrics["deterministic_ae_baseline_test_bce_per_image"]:.2f}'],
        textposition="outside",
        hovertemplate="%{x}: %{y:.5f} BCE/image<extra></extra>",
    ))
    return _layout(fig, "Reconstruction-only comparison", "BCE per image (lower is better)")


def elbo_decomposition(metrics: dict[str, Any]) -> go.Figure:
    rec = metrics["test_reconstruction_bce_per_image"]
    kl = metrics["test_kl_per_image"]
    fig = go.Figure()
    fig.add_bar(x=["VAE test ELBO"], y=[rec], name="Reconstruction BCE", marker_color=COLORS["cyan"])
    fig.add_bar(x=["VAE test ELBO"], y=[kl], name="KL divergence", marker_color=COLORS["violet"])
    fig.update_layout(barmode="stack")
    fig.update_traces(hovertemplate="%{fullData.name}: %{y:.5f}/image<extra></extra>")
    return _layout(fig, "ELBO decomposition: reconstruction + regularization", "Loss per image")


def vae_mode_comparison(fast: dict[str, Any], full: dict[str, Any]) -> go.Figure:
    frame = pd.DataFrame([
        {"Mode": "Fast", "Model": "VAE", "BCE": fast["test_reconstruction_bce_per_image"]},
        {"Mode": "Fast", "Model": "AE", "BCE": fast["deterministic_ae_baseline_test_bce_per_image"]},
        {"Mode": "Full", "Model": "VAE", "BCE": full["test_reconstruction_bce_per_image"]},
        {"Mode": "Full", "Model": "AE", "BCE": full["deterministic_ae_baseline_test_bce_per_image"]},
    ])
    fig = px.bar(frame, x="Mode", y="BCE", color="Model", barmode="group",
                 color_discrete_map={"VAE": COLORS["violet"], "AE": COLORS["teal"]})
    fig.update_traces(hovertemplate="%{x} %{fullData.name}: %{y:.5f}<extra></extra>")
    return _layout(fig, "Fast versus full reconstruction", "Test BCE per image")


def qa_oracle(metrics: dict[str, Any]) -> go.Figure:
    rows = []
    for context, em_key, f1_key in (
        ("End-to-end", "exact_match_pct", "token_f1_pct"),
        ("Oracle context", "oracle_context_exact_match_pct", "oracle_context_token_f1_pct"),
    ):
        rows.extend([{"Context": context, "Metric": "Exact match", "Percent": metrics[em_key]},
                     {"Context": context, "Metric": "Token F1", "Percent": metrics[f1_key]}])
    fig = px.bar(pd.DataFrame(rows), x="Context", y="Percent", color="Metric", barmode="group",
                 color_discrete_sequence=[COLORS["cyan"], COLORS["violet"]])
    fig.update_yaxes(range=[0, 100])
    fig.update_traces(hovertemplate="%{x} %{fullData.name}: %{y:.2f}%<extra></extra>")
    return _layout(fig, "Extraction with retrieved versus gold evidence", "Score (%)")


def retrieval_metrics(metrics: dict[str, Any]) -> go.Figure:
    labels = ["Gold context Recall@1", "Gold context Recall@3", "Selected context correct"]
    values = [metrics["gold_context_recall_at_1_pct"], metrics["gold_context_recall_at_3_pct"], metrics["selected_context_accuracy_pct"]]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=[COLORS["violet"], COLORS["cyan"], COLORS["teal"]], text=[f"{v:.1f}%" for v in values], textposition="outside"))
    fig.update_traces(hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
    fig.update_yaxes(range=[0, 100])
    return _layout(fig, "Retrieval and evidence selection", "Percent")


def qa_bottleneck(metrics: dict[str, Any]) -> go.Figure:
    em = metrics["exact_match_pct"]
    oracle = metrics["oracle_context_exact_match_pct"]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["End-to-end EM", "Gold-context opportunity", "Oracle-context EM"],
        y=[em, oracle - em, oracle],
        text=[f"{em:.1f}%", f"+{oracle-em:.1f} pp", f"{oracle:.1f}%"],
        connector={"line": {"color": COLORS["grid"]}},
        increasing={"marker": {"color": COLORS["violet"]}},
        totals={"marker": {"color": COLORS["cyan"]}},
    ))
    fig.update_yaxes(range=[0, 100])
    return _layout(fig, "Retrieval bottleneck: oracle evidence closes 30 points", "Exact match (%)")


def qa_mode_comparison(fast: dict[str, Any], full: dict[str, Any]) -> go.Figure:
    keys = (("End-to-end EM", "exact_match_pct"), ("Token F1", "token_f1_pct"),
            ("Recall@3", "gold_context_recall_at_3_pct"), ("Oracle EM", "oracle_context_exact_match_pct"))
    rows = [{"Mode": mode, "Metric": label, "Percent": payload[key]}
            for mode, payload in (("Fast", fast), ("Full", full)) for label, key in keys]
    fig = px.bar(pd.DataFrame(rows), x="Metric", y="Percent", color="Mode", barmode="group",
                 color_discrete_map={"Fast": COLORS["violet"], "Full": COLORS["cyan"]})
    fig.update_yaxes(range=[0, 100])
    fig.update_traces(hovertemplate="%{x} %{fullData.name}: %{y:.2f}%<extra></extra>")
    return _layout(fig, "Fast smoke test versus full evaluation", "Percent")
