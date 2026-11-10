import math
from pathlib import Path

import pytest

from dashboard.utils.data_loader import (
    REPO_ROOT,
    artifact_path,
    load_all_metrics,
    load_classification_report,
    load_metrics,
    load_predictions,
)


def test_full_metrics_match_verified_report():
    results = load_all_metrics("full")
    transformer, vae, agent = results["transformer"], results["vae"], results["agent"]

    assert transformer["test_accuracy"] == pytest.approx(0.901)
    assert transformer["test_f1"] == pytest.approx(0.9027695934)
    assert transformer["majority_baseline_test_accuracy"] == pytest.approx(0.4988)
    assert vae["test_reconstruction_bce_per_image"] == pytest.approx(220.05956)
    assert vae["test_kl_per_image"] == pytest.approx(17.46663)
    assert vae["test_mse_per_pixel"] == pytest.approx(0.01261)
    assert agent["exact_match_pct"] == pytest.approx(54.0)
    assert agent["token_f1_pct"] == pytest.approx(58.5396)
    assert agent["gold_context_recall_at_3_pct"] == pytest.approx(74.0)


def test_transformer_confusion_matrix_and_report_are_complete():
    metrics = load_metrics("transformer", "full")
    matrix = metrics["test_confusion_matrix"]
    assert matrix == [[2207, 287], [208, 2298]]
    assert sum(sum(row) for row in matrix) == 5_000

    report = load_classification_report("full")
    assert [row["class"] for row in report] == ["Negative", "Positive"]
    assert sum(int(row["support"]) for row in report) == 5_000


def test_fast_full_switching_returns_distinct_saved_runs():
    for experiment in ("transformer", "vae", "agent"):
        fast = load_metrics(experiment, "fast")
        full = load_metrics(experiment, "full")
        assert fast["mode"] == "fast"
        assert full["mode"] == "full"
        assert fast is not full
    assert load_metrics("transformer", "fast")["test_size"] == 200
    assert load_metrics("transformer", "full")["test_size"] == 5_000


def test_qa_explorer_has_100_auditable_records():
    predictions = load_predictions("full")
    assert len(predictions) == 100
    required = {
        "id", "question", "prediction", "reference_answers", "exact_match", "f1",
        "gold_context_in_top3", "selected_context_correct", "retrieval_score", "qa_score",
        "source_excerpt", "oracle_context_prediction", "oracle_context_exact_match",
    }
    assert all(required <= row.keys() for row in predictions)
    assert sum(int(row["exact_match"]) for row in predictions) == 54
    assert math.isclose(
        100 * sum(float(row["f1"]) for row in predictions) / len(predictions),
        load_metrics("agent", "full")["token_f1_pct"],
        abs_tol=5e-5,
    )


def test_optional_artifact_lookup_is_safe():
    assert artifact_path("full", "vae_reconstructions.png") is not None
    assert artifact_path("full", "not-created.png") is None
    assert artifact_path("full", "../transformer.json") is None


def test_dashboard_sources_compile():
    dashboard = REPO_ROOT / "dashboard"
    sources = list(dashboard.rglob("*.py"))
    assert len(sources) >= 10
    for source in sources:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")


@pytest.mark.parametrize(
    "relative_path",
    ("app.py", "pages/1_Transformer.py", "pages/2_VAE.py", "pages/3_Grounded_QA.py"),
)
def test_streamlit_pages_render_without_exceptions(relative_path):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"))
    if relative_path != "app.py":
        app.switch_page(relative_path)
    app.run(timeout=20)
    assert not app.exception


@pytest.mark.parametrize(
    "relative_path", ("pages/1_Transformer.py", "pages/2_VAE.py", "pages/3_Grounded_QA.py")
)
def test_streamlit_mode_selector_switches_to_fast_run(relative_path):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"))
    app.switch_page(relative_path).run(timeout=20)
    assert not app.exception
    app.radio[0].set_value("Fast smoke run").run(timeout=20)
    assert app.radio[0].value == "Fast smoke run"
    assert not app.exception
