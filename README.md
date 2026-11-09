# Deep Learning Triad | GIM BDA 2025–27

**Three distinct coursework demonstrations:**
1. Transformer attention: DistilBERT classification with IMDb movie reviews.
2. Variational autoencoder: ConvVAE reconstruction on Fashion-MNIST.
3. Agentic AI: deterministic tool routing + TF-IDF retrieval + extractive QA + abstention on SQuAD.

> **Status (9 Nov 2026):** all implementations, unit tests, fast runs, and full
> runs have been executed successfully. Verified metrics are in
> `docs/RESULTS.md`; only team identities and the final recording URL remain as
> intentional slide placeholders.

## Setup (verified with Python 3.11.15)

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
# CPU-only (portable):
pip install -r requirements.txt
# NVIDIA alternative used for the verified runs on Windows (run before the
# requirements command so the official CUDA wheels replace CPU-only wheels):
# pip install torch==2.11.0 torchvision==0.26.0 \
#   --index-url https://download.pytorch.org/whl/cu128
pytest -q
python scripts/run_all.py --fast
python scripts/check_results.py
# For final run: python scripts/run_all.py
```

On laptops without dedicated GPUs, the transformer step and full agent QA can be slow. Run each independently and reduce `--fast` settings if necessary. First execution requires internet access to download datasets and model weights; the data and outputs directories are intentionally ignored by Git.

The pinned dependency set was verified as a unit. CUDA 12.8 wheels are optional;
the experiment code automatically falls back to CPU. Every result JSON records
the Python, PyTorch, device, GPU, seed, mode, sample sizes, and elapsed time.

## Individual runs

```bash
python -m src.transformer_sentiment --fast
python -m src.vae_fashion --fast
python -m src.agentic_qa --fast
```

### Data sources
- [IMDb 50k](https://huggingface.co/datasets/stanfordnlp/imdb): use train/validation from original train, held-out original test.
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): torchvision automatically downloads 60k train/10k test images; hold out 10k of train for validation.
- [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad): validation context corpus and evaluation questions; this is a **closed corpus**, not real internet research.

Data submission: `data/` contains Fashion-MNIST after download; Hugging Face datasets are cached under user cache. Since source files may be large, share a viewable dataset directory/link containing official dataset URLs, version info, and cache/export data as required by the professor.

## What is actually novel?
A cross-modal comparison on three different workloads, using controlled splits and evaluation: classification (accuracy/F1), generative reconstruction (ELBO BCE+KL), and question answering (EM/token F1/context retrieval). The 'agent' is a lightweight tool-use **workflow** with deterministic router, not an autonomous LLM planner. Avoid describing it as fully autonomous or presenting a retrieval result as a generative LLM answer.

## Outputs
- `results/transformer.json` + classification report
- `results/vae.json`, reconstructions and loss chart
- `results/agent.json` and per-question evidence/predictions
- weights under `outputs/` (not committed)
- archived fast/full evidence under `results/fast/` and `results/full/`
- factual comparison chart under `results/fast_vs_full_comparison.png`
- final report under `docs/FINAL_REPORT.md`

## Slides and video
- `slides/Deep_Learning_Triad.pptx` and `.pdf`: rendered, verified 24-slide deck with actual results.
- `docs/PRESENTATION_GUIDE.md`: timed member-by-member speaking guide.
- Final slide must include real recording URL, access permission to `bdagim2025@gmail.com` or `anup@gim.ac.in`.
- Submission email: To `bdagim2025@gmail.com`; CC all members; attach code directly and PPT/PDF separately; datasets and video shared with view permission.

## Disclosure
All results should clearly state **fast/full**, sample sizes, seed, and model checkpoint. Do not mix validation with test metrics or claim deployment/production capabilities.
