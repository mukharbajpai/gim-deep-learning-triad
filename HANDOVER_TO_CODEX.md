# CODEX HANDOVER — READ THIS FIRST

You are the implementation, experiment, slides and Git release agent for `deep-learning-triad`. Deliver a defensible GIM BDA Deep Learning group project due 15 Nov 2026, with 3 distinct concepts and 24 slides / 24 minute recording plan.

## Non-negotiable
- **Never invent experimental numbers, successful executions, team identities, Git commit hashes or a video URL.**
- Run, debug, reproduce each experiment, and capture output. If GPU unavailable, transparently report fast-mode metrics; do not claim full-mode metrics.
- Stay in Python 3.11, use a `.venv`; be conservative with model downloads and CPU runtimes.
- Do not commit credential files, Hugging Face cache, datasets, large model weights or video.
- Ask user for repo owner/repo name only if GitHub CLI lacks authenticated profile or if a specific destination is required. Preferred repo name `gim-deep-learning-triad`.
- Before Git push, verify `git remote -v` and get confirmation when publishing to a public remote could expose group identity or coursework. A private repo is preferred.

## Intended project shape
- `src/transformer_sentiment.py` DistilBERT IMDb train/validation/test, accuracy, macro? Code currently binary F1; optionally add macro F1 and a majority baseline.
- `src/vae_fashion.py` ConvVAE Fashion-MNIST, BCE and KL, reconstructed examples, ELBO curves. Improve with a plain deterministic autoencoder baseline if time.
- `src/agentic_qa.py` tool-router TF-IDF retrieval, extractive transformer QA, evidence provenance and deterministic calculator; SQuAD context. Evaluation includes top-3 gold-context recall and selected-context accuracy. Add an oracle-context baseline if time.
- `tests/test_core.py` smoke/unit tests.
- `slides/Deep_Learning_Triad.pptx` existing 24-slide starter.

## Mandatory fix / technical quality gates
1. `python -m compileall src scripts`; `pytest -q`.
2. Check transformers version compatibility: TrainingArguments API/Trainer `processing_class`. Pin a working version in requirements if necessary.
3. Run quick smoke: `python scripts/run_all.py --fast`; fix issues until all 3 return metrics. If too slow, run them independently and record environment/hardware.
4. Run final mode only if compute permits; record speed, size and resources.
5. Validate train/val/test disjointness and no test leakage. SQuAD corpus is intentionally the collection of evaluation contexts and *documents* are visible for retrieval; evaluation *answers* must not enter index.
6. Add factual results to 24 slides; do not replace placeholders with fabricated values. Insert reconstruction and loss curve screenshots; test on 16:9 screens.
7. Run `python scripts/check_results.py` and create `docs/RESULTS.md` with actual metric keys, measured values, hardware, command, date.
8. Verify file submission format and permissions, then package email attachment copies separately from code and dataset share.

## GitHub commits: commit in this order (or adapt to actual changes)

```powershell
# Run inside project folder. Authenticate: gh auth login
# git init only if repo is not already git; protect existing history
# git branch -M main
# gh repo create gim-deep-learning-triad --private --source . --remote origin  # only if no repo exists

git add .gitignore requirements.txt README.md src/common.py src/__init__.py
 git commit -m "chore: initialize reproducible deep learning project"
git add src/transformer_sentiment.py
 git commit -m "feat(transformers): add IMDb DistilBERT sentiment experiment"
git add src/vae_fashion.py
 git commit -m "feat(vae): add Fashion-MNIST convolutional VAE"
git add src/agentic_qa.py
 git commit -m "feat(agent): add tool-routed grounded SQuAD QA workflow"
git add tests scripts
 git commit -m "test: add smoke tests and experiment runner"
git add slides docs HANDOVER_TO_CODEX.md
 git commit -m "docs: add 24-slide deck and team presentation plan"
# After actual experiments: add reports and charts intentionally; current .gitignore
# ignores results/*.json and *.png; explicitly force-add only small verified artifacts.
git add docs/RESULTS.md
 git add -f results/transformer.json results/vae.json results/agent.json
 git commit -m "results: record verified evaluation runs"
git push -u origin main
```

**Important:** Run the commits in sequence on a freshly initialized Git repository. If you receive the generated ZIP (which has no `.git` history), do the steps above after extraction. `git commit` requires user.name and user.email configured. If remote exists, do not recreate it. Avoid any commit or push claiming metrics until verified. Do not blindly run the block if files missing. Do not force push.

## Suggested release timeline
- 9 Nov: setup, code review, API compatibility, GitHub initial commits.
- 10–11 Nov: experiments, QA and baseline comparisons.
- 12 Nov: charts, slides, proofs, references.
- 13 Nov: each of six authors rehearses, recordings and corrections.
- 14 Nov: final recording, ZIP, links, permissions and test email.
- 15 Nov: submission by hard deadline.

## Stop and escalate
If a pretrained model or dataset cannot be downloaded, report the exact error; use documented alternative dataset/model only with explicit disclosure, and keep code/slide descriptions synchronized. If inability to train persists, present the *planned* experiment honestly rather than pretending results exist.
