"""Read-only, cached access to committed experiment artifacts.

All paths are anchored to this file so the dashboard works regardless of the
directory from which Streamlit is launched.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
MODES = ("full", "fast")
EXPERIMENTS = ("transformer", "vae", "agent")


class DashboardDataError(RuntimeError):
    """Raised when a required saved artifact is missing or malformed."""


def _mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return normalized


@lru_cache(maxsize=32)
def _read_json(path: str) -> Any:
    file_path = Path(path)
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DashboardDataError(f"Missing saved artifact: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise DashboardDataError(f"Invalid JSON in {file_path}: {exc}") from exc


def load_metrics(experiment: str, mode: str = "full") -> dict[str, Any]:
    """Load one experiment's metrics without mutating the result files."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {experiment!r}")
    selected_mode = _mode(mode)
    payload = _read_json(str(RESULTS_ROOT / selected_mode / f"{experiment}.json"))
    if not isinstance(payload, dict):
        raise DashboardDataError(f"Expected an object for {experiment}/{selected_mode}")
    if payload.get("mode") != selected_mode:
        raise DashboardDataError(
            f"Mode mismatch in {experiment}.json: {payload.get('mode')!r}"
        )
    return payload


def load_all_metrics(mode: str = "full") -> dict[str, dict[str, Any]]:
    return {name: load_metrics(name, mode) for name in EXPERIMENTS}


def load_predictions(mode: str = "full") -> list[dict[str, Any]]:
    selected_mode = _mode(mode)
    payload = _read_json(
        str(RESULTS_ROOT / selected_mode / "agent_predictions.json")
    )
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise DashboardDataError("Agent predictions must be a JSON array of objects")
    return payload


@lru_cache(maxsize=4)
def load_classification_report(mode: str = "full") -> list[dict[str, float | str]]:
    """Parse the saved sklearn text report; never reconstruct class metrics."""
    selected_mode = _mode(mode)
    path = RESULTS_ROOT / selected_mode / "transformer_classification_report.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows: list[dict[str, float | str]] = []
    pattern = re.compile(
        r"^\s*(negative|positive)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$",
        re.MULTILINE,
    )
    for label, precision, recall, f1, support in pattern.findall(text):
        rows.append(
            {
                "class": label.title(),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )
    return rows


def artifact_path(mode: str, filename: str) -> Path | None:
    """Return a safe optional artifact path, or None when it is unavailable."""
    selected_mode = _mode(mode)
    safe_name = Path(filename).name
    if safe_name != filename:
        return None
    path = RESULTS_ROOT / selected_mode / safe_name
    return path if path.is_file() and path.stat().st_size > 0 else None


def checkpoint_inventory() -> dict[str, bool]:
    """Report optional, ignored local checkpoints without loading them."""
    return {
        "Fine-tuned DistilBERT": (OUTPUTS_ROOT / "distilbert" / "final" / "model.safetensors").is_file(),
        "Convolutional VAE": (OUTPUTS_ROOT / "fashion_vae.pt").is_file(),
        "Deterministic AE": (OUTPUTS_ROOT / "fashion_autoencoder_baseline.pt").is_file(),
    }
