# Deep Learning Triad Dashboard

A read-only Streamlit and Plotly application for exploring the repository's
verified fast and full experiment artifacts. It never retrains a model,
downloads a dataset, or writes to `results/`.

## Install and launch on Windows

From the repository root, with Python 3.11:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r dashboard\requirements.txt
python -m streamlit run dashboard\app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
The dashboard works offline once its lightweight packages are installed because
all displayed values and images are local.

To run the dashboard validation suite in the project's fully provisioned
environment:

```powershell
python -m pytest -q tests\test_dashboard.py
```

## Architecture

```text
dashboard/
├── app.py                       # overview and navigation
├── components/
│   ├── charts.py                # Plotly figures from saved values
│   ├── metrics.py               # cards, formatting, mode selector
│   └── styles.py                # visual system and shared navigation
├── pages/
│   ├── 1_Transformer.py         # IMDb classification analysis
│   ├── 2_VAE.py                 # training curves and reconstruction panel
│   └── 3_Grounded_QA.py         # diagnostics and prediction explorer
└── utils/
    └── data_loader.py           # cached, defensive, read-only artifact access
```

All repository paths are derived from `Path(__file__)`; the launch directory
does not determine where data is read from. JSON reads are cached in process,
optional images fail gracefully, and no model is loaded on any results page.

## Data dependencies

| Experience | Required artifacts | Optional artifact |
|---|---|---|
| Transformer | `results/{fast,full}/transformer.json` | `transformer_classification_report.txt` |
| VAE | `results/{fast,full}/vae.json` | `vae_reconstructions.png` |
| Grounded QA | `results/{fast,full}/agent.json`, `agent_predictions.json` | none |

The VAE learning curves use `history` and `baseline_history` from the JSON. The
QA explorer uses only recorded prediction rows. Its context view is correctly
labelled as an excerpt because the artifact stores the selected passage's first
200 characters, not the complete passage.

## Results explorer versus live inference

This release intentionally provides saved-result exploration rather than live
model execution. The local development machine may contain ignored checkpoints
under `outputs/`, but they are not part of a clone and would introduce the full
PyTorch/Transformers stack into a lightweight presentation app. The overview
reports checkpoint presence without loading weights. Nothing shown as a saved
prediction is labelled as live inference.

## Scientific interpretation

- Transformer scores are compared with a majority-class baseline.
- VAE reconstruction is compared with a matched deterministic autoencoder, but
  VAE ELBO is never equated with AE reconstruction BCE.
- Grounded QA is described as a deterministic tool-routed, retrieval-grounded
  extractive workflow—not a generative or autonomous LLM agent.
- Fast runs are compatibility checks; full runs are the final evidence.
