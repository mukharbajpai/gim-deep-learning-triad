# Deep Learning Triad - Final Technical Report

## Abstract

This project implements and evaluates three distinct deep-learning concepts:
DistilBERT sentiment classification, a convolutional variational autoencoder,
and an auditable tool-routed question-answering workflow. All three were run in
fast and full modes with fixed seeds, disjoint evaluation, task-appropriate
baselines, saved metrics, and generated visual evidence. On full runs,
DistilBERT achieved 90.10% IMDb accuracy; the ConvVAE achieved 220.06 BCE/image
and 0.01261 MSE/pixel on Fashion-MNIST; and the SQuAD workflow achieved 54.0%
exact match and 58.54% token F1. Baseline and oracle comparisons show where the
models add value and where the remaining bottlenecks lie.

## 1. Transformer sentiment classification

The discriminative experiment fine-tunes
`distilbert/distilbert-base-uncased` for binary sentiment. Reviews are truncated
to 192 tokens and dynamically padded. The full protocol uses 20,000 training,
5,000 validation, and 5,000 held-out test reviews for two epochs at learning
rate 2e-5. Training and validation come from non-overlapping regions of the
seeded original training split; test examples come only from the original test
split.

The full model achieved 90.10% accuracy, 90.28% binary F1, and 90.10% macro F1.
The majority baseline achieved 49.88% accuracy and zero positive-class F1. The
confusion matrix `[[2207,287],[208,2298]]` shows slightly fewer false negatives
than false positives. Validation accuracy was 90.30%, close to test accuracy.

## 2. Variational autoencoder

The generative experiment uses two strided convolutions, 16-dimensional mean
and log-variance heads, reparameterized sampling, and a transpose-convolution
decoder. Its objective is reconstruction BCE plus beta-weighted KL divergence,
with beta=1. The full split uses 50,000 training, 10,000 validation, and 10,000
test images for 12 epochs.

A deterministic convolutional autoencoder baseline uses the same data,
bottleneck dimension, and epoch count. The full VAE achieved 220.06 test
reconstruction BCE/image, 17.47 KL/image, 237.53 total ELBO loss/image, and
0.01261 MSE/pixel. The AE achieved 216.22 BCE/image and 0.01093 MSE/pixel. This
does not make the VAE unsuccessful: the AE optimizes only reconstruction, while
the VAE also regularizes a sampleable latent distribution. The images show that
both preserve silhouettes, while fine texture is smoothed.

## 3. Grounded QA workflow

The third experiment is deliberately described as a tool-use workflow, not an
autonomous LLM. A deterministic router handles bounded arithmetic through a
restricted syntax tree. Other questions use bigram TF-IDF to retrieve three of
2,067 unique SQuAD validation contexts. An extractive DistilBERT QA model scores
each passage, and the workflow returns the selected span with source ID,
retrieval score, QA score, and evidence excerpt.

On 100 seeded questions, end-to-end exact match was 54.0% and token F1 was
58.54%. Gold-context Recall@1 was 58.0%, Recall@3 was 74.0%, and selected-context
accuracy was 57.0%. With the gold context supplied (but never the answer), the
same extractor achieved 84.0% exact match and 87.93% token F1. The large oracle
gap identifies retrieval/evidence selection as the main bottleneck. The 0%
abstention rate also shows that threshold calibration is future work.

## 4. Comparative findings

Each experiment needs a different evaluation lens. The transformer is judged by
classification quality against a trivial class baseline. The VAE is judged by
ELBO components, MSE, and reconstruction images against a deterministic AE. The
agent is judged end-to-end and by decomposed retrieval/extraction metrics against
an oracle-context upper bound. Raw values are not directly comparable across
these tasks.

Fast mode proved valuable as a compatibility gate, but its small samples were
not final evidence. Transformer accuracy rose from 69.5% in fast mode to 90.1%
in full mode. VAE reconstruction error fell substantially after using all data
and 12 epochs. QA fast/full variation demonstrates why a 20-question smoke test
should not be presented as a stable estimate.

## 5. Reproducibility, limitations, and conclusion

The repository pins the verified top-level dependencies, records environment
metadata in every metric JSON, seeds Python/NumPy/PyTorch, requests deterministic
PyTorch behavior, and retains both fast and full outputs. Compilation, four unit
tests, schema validation, and PowerPoint rendering checks all passed.

Limitations include a single principal seed, one hardware environment, IMDb
truncation, BCE-induced VAE smoothing, lexical retrieval, a closed QA corpus,
and an uncalibrated abstention threshold. Even with these limits, the experiments
provide defensible evidence for three different concepts and clearly separate
measured results from future work.

Detailed values, commands, artifacts, and hardware provenance are in
`docs/RESULTS.md`.
