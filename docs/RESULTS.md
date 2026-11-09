# Verified Experimental Results

All numbers in this document were produced by executed code on 9 November 2026.
The root `results/*.json` files contain the full runs; immutable copies of both
scales are under `results/fast/` and `results/full/`.

## Execution evidence

```powershell
.venv\Scripts\python.exe -m compileall -q src scripts
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\run_all.py --fast --offline
.venv\Scripts\python.exe scripts\check_results.py --expect-mode fast
.venv\Scripts\python.exe scripts\run_all.py --offline
.venv\Scripts\python.exe scripts\check_results.py --expect-mode full
```

- Unit tests: **4 passed**.
- Full end-to-end runner time: **1,528.81 seconds**.
- Python: **3.11.15**.
- PyTorch: **2.11.0+cu128**; CUDA build **12.8**.
- GPU: **NVIDIA GeForce GTX 1660 Ti, 6 GiB**.
- OS reported by Python: **Windows 10.0.26200**.
- Seed: **42** for all principal experiments; deterministic AE baseline seed **43**.

## Transformer sentiment classification

Dataset/checkpoint: `stanfordnlp/imdb` and
`distilbert/distilbert-base-uncased`.

| Metric | Fast (800/200/200, 1 epoch) | Full (20k/5k/5k, 2 epochs) |
|---|---:|---:|
| Test accuracy | 69.50% | **90.10%** |
| Test binary F1 | 66.30% | **90.28%** |
| Test macro F1 | 69.22% | **90.10%** |
| Majority accuracy | 52.00% | 49.88% |
| Validation accuracy | 74.50% | 90.30% |
| Confusion matrix | `[[79,25],[36,60]]` | `[[2207,287],[208,2298]]` |
| Train + evaluation time | 18.95 s | 1,064.57 s |

The original IMDb training split supplies non-overlapping train/validation
ranges after a seeded shuffle. The original IMDb test split supplies test data.
The full model improves accuracy by 40.22 percentage points over its majority
baseline. The test and validation scores are close, with no obvious test-set
inflation.

## Fashion-MNIST VAE and autoencoder baseline

Both models use the same seeded train/validation split and untouched original
test set. The deterministic autoencoder uses the same 16-dimensional bottleneck
and convolutional shape, but has no sampling or KL regularization.

| Metric | Fast VAE | Fast AE | Full VAE | Full AE |
|---|---:|---:|---:|---:|
| Test reconstruction BCE/image | 330.15509 | 313.48847 | **220.05956** | **216.22470** |
| Test MSE/pixel | 0.05778 | 0.05073 | **0.01261** | **0.01093** |
| Test KL/image | 17.13585 | n/a | **17.46663** | n/a |
| Test ELBO loss/image | 347.29092 | n/a | **237.52617** | n/a |
| Samples (train/val/test) | 2k/400/500 | same | 50k/10k/10k | same |
| Epochs | 2 | 2 | 12 | 12 |

The full VAE script took **396.27 seconds** (405.31 seconds including process
startup measured by the runner). Validation ELBO fell from 252.02 after epoch 1
to 236.39 after epoch 12. The AE has better pure reconstruction, which is
expected: the VAE pays a KL cost to learn a regularized, sampleable latent
distribution. The reconstruction panel shows the same trade-off qualitatively.

## Grounded SQuAD QA workflow

Dataset/checkpoint: SQuAD v1.1 validation and
`distilbert/distilbert-base-cased-distilled-squad`. The evidence corpus contains
2,067 unique validation contexts. Answer strings are never indexed. The oracle
baseline receives the gold context only during evaluation and never receives the
gold answer.

| Metric | Fast (20 questions) | Full (100 questions) |
|---|---:|---:|
| End-to-end exact match | 55.00% | **54.00%** |
| End-to-end token F1 | 55.83% | **58.54%** |
| Gold-context Recall@1 | 50.00% | **58.00%** |
| Gold-context Recall@3 | 70.00% | **74.00%** |
| Selected-context accuracy | 55.00% | **57.00%** |
| Oracle-context exact match | 90.00% | **84.00%** |
| Oracle-context token F1 | 90.00% | **87.93%** |
| Abstention rate | 0.00% | **0.00%** |
| Full elapsed time | 11.80 s | **18.21 s** |

The 30-point full-run gap between end-to-end and oracle-context exact match
identifies retrieval/evidence selection as the dominant bottleneck. Zero
abstention indicates that the fixed 0.05 threshold is not calibrated for this
in-domain corpus.

## Generated evidence

- `results/full/transformer_diagnostics.png`: confusion matrix and majority comparison.
- `results/full/vae_reconstructions.png`: original, VAE, and AE outputs.
- `results/full/vae_training.png`: full learning curves.
- `results/full/agent_diagnostics.png`: end-to-end/oracle and retrieval results.
- `results/full/agent_predictions.json`: 100 auditable predictions with provenance.
- `results/fast_vs_full_comparison.png`: scale comparison across all three tasks.
- `slides/Deep_Learning_Triad.pptx` and `.pdf`: verified 24-slide deck.

## Limitations

The experiments use one principal seed and one GPU. IMDb inputs are truncated at
192 tokens. Pixel BCE favors smooth reconstructions. The QA corpus is closed and
in-domain; TF-IDF is lexical, and the abstention threshold is not calibrated.
These results demonstrate the coursework concepts but do not establish
production readiness or open-domain generalization.
