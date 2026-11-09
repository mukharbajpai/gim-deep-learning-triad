# Six-Member Presentation Guide (24 slides / about 24 minutes)

Replace Member A-F with real names and rolls before rehearsal. Target 55-60
seconds per slide, a clean handoff every four slides, and 23-24 minutes total.

## Member A - slides 1-4 (0:00-4:00)

- Slide 1: introduce the three concepts and state that all displayed numbers are executed results.
- Slide 2: frame the three research questions and the cross-modal contribution.
- Slide 3: cite Python 3.11.15, PyTorch 2.11 CUDA 12.8, GTX 1660 Ti, seed 42, and four passing tests.
- Slide 4: explain why accuracy/F1, ELBO, and QA metrics cannot be compared as one score.
- Handoff: "Member B will now explain attention and our IMDb experiment."

## Member B - slides 5-8 (4:00-8:00)

- Slide 5: explain Q, K, V and the `sqrt(d_k)` scaling in plain language.
- Slide 6: walk from review through tokenizer, DistilBERT, and classifier.
- Slide 7: emphasize 20k/5k/5k disjoint full split and untouched original test partition.
- Slide 8: point to deterministic Trainer settings and majority/macro-F1 additions.
- Likely question: Why DistilBERT? Answer: compact pretrained encoder with practical fine-tuning cost.
- Handoff: "Member C will interpret the measured transformer result and introduce the VAE."

## Member C - slides 9-12 (8:00-12:00)

- Slide 9: report 90.10% accuracy, 90.28% binary F1, and 49.88% majority accuracy; interpret the confusion matrix.
- Slide 10: distinguish the fast smoke result (69.5%) from the full result (90.1%).
- Slide 11: explain mean, log-variance, reparameterization, reconstruction, and KL.
- Slide 12: follow the convolutional encoder to the 16-D latent and decoder; introduce the deterministic AE baseline.
- Likely question: Why is the VAE stochastic only in training? Answer: evaluation uses the mean for stable reconstruction comparison.
- Handoff: "Member D will cover Fashion-MNIST implementation and evidence."

## Member D - slides 13-16 (12:00-16:00)

- Slide 13: state the 50k/10k/10k full split, 12 epochs, beta=1, and identical baseline data.
- Slide 14: explain `loss = BCE + beta*KL` and why BCE is summed per image.
- Slide 15: report VAE 220.06 BCE and AE 216.22 BCE; explain why the AE wins pure reconstruction.
- Slide 16: compare originals, VAE reconstructions, and AE reconstructions; mention smoothing.
- Likely question: Is lower AE BCE proof it is better overall? Answer: only for reconstruction, not for learning a regularized sampleable distribution.
- Handoff: "Member E will finish the VAE analysis and introduce the QA workflow."

## Member E - slides 17-20 (16:00-20:00)

- Slide 17: show validation ELBO falling from 252.02 to 236.39 and no obvious divergence.
- Slide 18: precisely call the system a transparent tool-use workflow, not an autonomous planner.
- Slide 19: explain route, top-3 retrieval, QA, rerank, answer, and provenance.
- Slide 20: explain why contexts may be indexed while answer strings remain hidden; define oracle context.
- Likely question: Why no web search? Answer: controlled closed-corpus evaluation makes retrieval measurable and reproducible.
- Handoff: "Member F will present agent results, synthesis, and submission status."

## Member F - slides 21-24 (20:00-24:00)

- Slide 21: show that every answer retains source ID, retrieval/QA score, and evidence excerpt.
- Slide 22: report 54.0% EM, 58.54% token F1, 74.0% Recall@3, and 84.0% oracle EM; identify retrieval as the main bottleneck.
- Slide 23: summarize success, limits, and next experiments without claiming production readiness.
- Slide 24: cite the papers, then show the verified video URL after it is inserted.
- Likely question: Why is abstention 0%? Answer: 0.05 is too permissive in-domain and requires calibration on a separate validation protocol.

## Rehearsal checklist

- Every speaker says the split, baseline, metric, and one failure mode for their section.
- Use "fast" and "full" explicitly; never mix their sample sizes.
- Do not call TF-IDF retrieval a generative LLM or the workflow fully autonomous.
- Keep the recording between 20 and 25 minutes and show real CLI/artifact evidence once.
- Verify the final video link and permissions before recording the final take.
