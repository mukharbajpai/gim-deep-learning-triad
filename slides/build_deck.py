"""Build the verified 24-slide 16:9 assignment deck from measured results."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = Path(__file__).with_name('Deep_Learning_Triad.pptx')


def load(mode, name):
    return json.loads((RESULTS / mode / f'{name}.json').read_text(encoding='utf-8'))


fast_t, full_t = load('fast', 'transformer'), load('full', 'transformer')
fast_v, full_v = load('fast', 'vae'), load('full', 'vae')
fast_a, full_a = load('fast', 'agent'), load('full', 'agent')

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

NAVY = RGBColor(20, 35, 56)
TEAL = RGBColor(11, 146, 139)
PALE = RGBColor(241, 245, 249)
INK = RGBColor(38, 54, 74)
MUTED = RGBColor(94, 108, 126)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(55, 130, 90)
ORANGE = RGBColor(221, 126, 55)


def text_box(slide, text, x, y, w, h, size=18, color=INK, bold=False,
             font='Aptos', align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.04)
    tf.margin_top = tf.margin_bottom = Inches(.03)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.name = font
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    return shape


def base(title, kicker='VERIFIED DEEP LEARNING EXPERIMENTS'):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(.14))
    band.fill.solid(); band.fill.fore_color.rgb = TEAL; band.line.fill.background()
    text_box(slide, kicker, .62, .34, 11.8, .27, 9, TEAL, True)
    text_box(slide, title, .6, .78, 12.0, .72, 28, NAVY, True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.62), Inches(6.94),
                                  Inches(12.05), Inches(.012))
    line.fill.solid(); line.fill.fore_color.rgb = RGBColor(215, 224, 232)
    line.line.fill.background()
    text_box(slide, 'Seed 42 | disjoint evaluation | results generated 9 Nov 2026',
             .64, 7.02, 10.8, .2, 8, MUTED)
    text_box(slide, f'{len(prs.slides):02d}', 12.1, 7.01, .55, .2, 9, TEAL, True,
             align=PP_ALIGN.RIGHT)
    return slide


def panel(slide, x, y, w, h, heading=None, fill=PALE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = RGBColor(217, 229, 234)
    if heading:
        text_box(slide, heading, x + .22, y + .18, w - .44, .35, 17, TEAL, True)
    return shape


def bullets(slide, items, x, y, w, h, size=17, color=INK):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(.05); tf.margin_right = Inches(.04)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.font.name = 'Aptos'; p.font.size = Pt(size); p.font.color.rgb = color
        p.space_after = Pt(10); p.level = 0
        p.text = '\u2022 ' + p.text
    return shape


def code_box(slide, code, x, y, w, h):
    shape = panel(slide, x, y, w, h, fill=RGBColor(28, 40, 56))
    tf = shape.text_frame
    tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.18)
    tf.margin_top = Inches(.16)
    p = tf.paragraphs[0]
    p.text = code
    p.font.name = 'Consolas'; p.font.size = Pt(13); p.font.color.rgb = RGBColor(225, 236, 244)
    return shape


def picture(slide, path, x, y, w, h=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w),
                                    height=Inches(h) if h else None)


def table(slide, rows, cols, data, x, y, w, h, col_widths=None):
    shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h))
    tab = shape.table
    if col_widths:
        for i, width in enumerate(col_widths):
            tab.columns[i].width = Inches(width)
    for r in range(rows):
        for c in range(cols):
            cell = tab.cell(r, c)
            cell.text = str(data[r][c])
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else PALE)
            for p in cell.text_frame.paragraphs:
                p.font.name = 'Aptos'; p.font.size = Pt(12 if r else 13)
                p.font.bold = r == 0; p.font.color.rgb = WHITE if r == 0 else INK
                p.alignment = PP_ALIGN.CENTER
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return tab


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# 1
s = base('Deep Learning Triad', 'TRANSFORMERS | VARIATIONAL AUTOENCODERS | AGENTIC AI')
text_box(s, 'Theory, implementation, baselines, and measured evaluation', .78, 2.0, 11.8, .7, 28, NAVY)
text_box(s, 'Deep Learning | Dr. Anup Kumar Maurya | Academic Year 2026-27', .8, 3.05, 11.5, .42, 18, MUTED)
panel(s, .8, 4.05, 11.75, 1.15)
text_box(s, 'TEAM: [NAME + ROLL] x 6 - human input required before submission', 1.08, 4.42, 11.2, .4, 18, TEAL, True, align=PP_ALIGN.CENTER)
notes(s, 'Member A: introduce the three distinct concepts and state that every number shown comes from an executed run.')

# 2
s = base('Assignment objectives and contribution')
panel(s, .68, 1.67, 5.9, 4.85, 'Questions')
bullets(s, ['How does attention support sentiment classification?',
            'How does a variational latent space trade reconstruction for regularization?',
            'How much QA error comes from retrieval versus answer extraction?'], .95, 2.3, 5.35, 3.7)
panel(s, 6.78, 1.67, 5.88, 4.85, 'Contribution')
bullets(s, ['Three modalities: text classification, image reconstruction, grounded QA.',
            'Disjoint splits, fixed seed, baselines, runtime provenance, and saved predictions.',
            'Fast smoke runs plus full experiments on one reproducible GPU environment.'], 7.05, 2.3, 5.32, 3.7)

# 3
s = base('Reproducibility and compute evidence')
data = [['Item', 'Verified configuration'], ['Python', full_t['runtime']['python_version']],
        ['PyTorch / CUDA', f"{full_t['runtime']['torch_version']} / {full_t['runtime']['cuda_build']}"],
        ['GPU', f"{full_t['runtime']['gpu']} ({full_t['runtime']['gpu_memory_gib']:.0f} GiB)"],
        ['Seed', '42 for all three experiments'], ['Validation', '4 unit tests + compileall + result schema checks']]
table(s, len(data), 2, data, .75, 1.72, 7.0, 4.75, [2.4, 4.6])
panel(s, 8.02, 1.72, 4.55, 4.75, 'Integrity gates')
bullets(s, ['No test examples used for fitting or tuning.', 'No answer text enters the QA retrieval index.',
            'Fast and full outputs archived separately.', 'Metrics and plots written only after successful execution.'], 8.3, 2.35, 3.98, 3.5, 15)

# 4
s = base('One workflow, three different evaluation lenses')
for i, (title, body, color) in enumerate([
        ('CLASSIFICATION', 'IMDb text -> DistilBERT -> accuracy / F1', TEAL),
        ('GENERATION', 'Fashion image -> ConvVAE -> BCE / KL / MSE', GREEN),
        ('TOOL WORKFLOW', 'Question -> retrieve -> extract -> EM / token F1', ORANGE)]):
    x = .78 + i * 4.15
    panel(s, x, 2.0, 3.75, 3.65)
    text_box(s, title, x + .2, 2.36, 3.35, .38, 16, color, True, align=PP_ALIGN.CENTER)
    text_box(s, body, x + .28, 3.0, 3.2, 1.65, 19, NAVY, True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
text_box(s, 'Scores are interpreted within each task; raw values are not compared across incompatible metrics.', .9, 6.05, 11.5, .42, 14, MUTED, align=PP_ALIGN.CENTER)

# 5
s = base('Transformer foundation: scaled dot-product attention')
panel(s, .75, 1.75, 7.1, 4.75)
text_box(s, 'Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V', 1.08, 2.18, 6.45, .7, 24, NAVY, True, align=PP_ALIGN.CENTER)
bullets(s, ['Q asks what each token seeks; K identifies what tokens contain; V carries content.',
            'Multi-head attention learns several token relationships in parallel.',
            'DistilBERT is a compact pretrained encoder fine-tuned with a binary head.'], 1.08, 3.15, 6.42, 2.75, 16)
panel(s, 8.12, 1.75, 4.45, 4.75, 'Why pretrained?')
bullets(s, ['IMDb fast mode has only 800 training reviews.', 'Transfer learning supplies broad language representations.',
            'Fine-tuning adapts those representations to positive/negative sentiment.'], 8.4, 2.4, 3.9, 3.35, 15)

# 6
s = base('Transformer experiment architecture')
steps = ['Review', 'Tokenizer', '192 tokens', 'DistilBERT', 'Classifier', 'Sentiment']
for i, step in enumerate(steps):
    x = .65 + i * 2.08
    panel(s, x, 2.42, 1.78, 1.45, fill=RGBColor(232, 245, 243) if i % 2 == 0 else PALE)
    text_box(s, step, x + .1, 2.72, 1.58, .82, 16, NAVY, True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    if i < len(steps) - 1:
        text_box(s, '->', x + 1.78, 2.91, .3, .3, 18, TEAL, True, align=PP_ALIGN.CENTER)
text_box(s, 'AdamW | lr 2e-5 | dynamic padding | 2 full epochs | seed 42', 1.0, 4.55, 11.35, .5, 18, TEAL, True, align=PP_ALIGN.CENTER)

# 7
s = base('IMDb protocol: disjoint train, validation, and test')
data = [['Mode', 'Train', 'Validation', 'Test', 'Epochs'],
        ['Fast', '800', '200', '200', '1'], ['Full', '20,000', '5,000', '5,000', '2']]
table(s, 3, 5, data, .85, 1.8, 7.6, 2.35, [1.5, 1.5, 1.7, 1.5, 1.4])
panel(s, 8.75, 1.8, 3.75, 4.7, 'Leakage control')
bullets(s, ['Train/validation: non-overlapping ranges after seeded shuffle of original train.',
            'Test: sampled only from IMDb original test split.', 'Model and threshold choices never use test labels.'], 9.02, 2.45, 3.22, 3.45, 14)
text_box(s, 'Dataset: stanfordnlp/imdb (25k original train + 25k original test)', .92, 4.68, 7.25, .52, 16, MUTED)

# 8
s = base('Transformer implementation: actual training path')
code_box(s, "args = TrainingArguments(\n    num_train_epochs=2, learning_rate=2e-5,\n    seed=42, data_seed=42, full_determinism=True\n)\ntrainer = Trainer(\n    model=model, processing_class=tokenizer,\n    data_collator=DataCollatorWithPadding(tokenizer),\n    compute_metrics=metric_fn\n)\ntrainer.train()\ntest_prediction = trainer.predict(test_ds)", .7, 1.66, 7.0, 4.95)
panel(s, 8.0, 1.66, 4.58, 4.95, 'Quality additions')
bullets(s, ['Binary F1 plus macro F1.', 'Majority-label baseline on the same test set.',
            'Confusion matrix and classification report.', 'Python, library, GPU, elapsed time stored in JSON.'], 8.28, 2.3, 4.02, 3.7, 15)

# 9
s = base('Transformer full results: 5,000 held-out reviews')
picture(s, RESULTS / 'full' / 'transformer_diagnostics.png', .68, 1.62, 7.45)
panel(s, 8.35, 1.67, 4.22, 4.85, 'Measured results')
bullets(s, [f"Accuracy: {100*full_t['test_accuracy']:.2f}%",
            f"Binary F1: {100*full_t['test_f1']:.2f}%",
            f"Macro F1: {100*full_t['test_macro_f1']:.2f}%",
            f"Majority accuracy: {100*full_t['majority_baseline_test_accuracy']:.2f}%",
            f"Confusion matrix: {full_t['test_confusion_matrix']}",
            f"Train + evaluation: {full_t['training_and_evaluation_seconds']:.1f}s"], 8.65, 2.25, 3.65, 3.85, 15)

# 10
s = base('Fast versus full runs: scale changes the evidence')
picture(s, RESULTS / 'fast_vs_full_comparison.png', .62, 1.58, 8.15)
panel(s, 9.0, 1.72, 3.58, 4.8, 'Interpretation')
bullets(s, [f"IMDb accuracy rose from {100*fast_t['test_accuracy']:.1f}% to {100*full_t['test_accuracy']:.1f}%.",
            'Full VAE training greatly reduced reconstruction error.',
            'QA fast/full differences show why n=20 is only a smoke test.',
            'Baselines contextualize each task; metrics remain task-specific.'], 9.26, 2.32, 3.06, 3.55, 14)

# 11
s = base('VAE foundation: probabilistic latent representation')
panel(s, .72, 1.72, 6.0, 4.82, 'Encoder and reparameterization')
text_box(s, 'q_phi(z|x) -> mu, log(sigma^2)', 1.05, 2.35, 5.35, .5, 22, NAVY, True, align=PP_ALIGN.CENTER)
text_box(s, 'z = mu + sigma * epsilon,   epsilon ~ N(0, I)', 1.05, 3.12, 5.35, .5, 21, TEAL, True, align=PP_ALIGN.CENTER)
bullets(s, ['Sampling remains differentiable through mu and sigma.', 'The decoder reconstructs x from the latent z.'], 1.08, 4.05, 5.2, 1.4, 15)
panel(s, 6.95, 1.72, 5.65, 4.82, 'Objective (negative ELBO)')
text_box(s, 'Loss = BCE(x, x_hat) + beta * KL(q || N(0,I))', 7.23, 2.55, 5.1, .9, 21, NAVY, True, align=PP_ALIGN.CENTER)
bullets(s, ['BCE rewards pixel reconstruction.', 'KL regularizes the latent distribution.', 'beta = 1 in both fast and full runs.'], 7.35, 3.78, 4.9, 2.0, 16)

# 12
s = base('Convolutional VAE architecture')
steps = ['28x28 image', 'Conv 32\n14x14', 'Conv 64\n7x7', 'mu + logvar\n16-D', 'Sample z', 'Transpose conv\n28x28']
for i, step in enumerate(steps):
    x = .6 + i * 2.1
    panel(s, x, 2.35, 1.82, 1.65, fill=RGBColor(232, 245, 243) if i % 2 == 0 else PALE)
    text_box(s, step, x + .1, 2.64, 1.62, .95, 15, NAVY, True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    if i < 5: text_box(s, '->', x + 1.81, 2.95, .3, .3, 17, TEAL, True, align=PP_ALIGN.CENTER)
text_box(s, 'Deterministic baseline: same convolutional encoder/decoder shape, one 16-D bottleneck, no KL term.', .82, 4.75, 11.7, .72, 17, ORANGE, True, align=PP_ALIGN.CENTER)

# 13
s = base('Fashion-MNIST protocol and split integrity')
data = [['Mode', 'Train', 'Validation', 'Test', 'Epochs'],
        ['Fast', '2,000', '400', '500', '2'], ['Full', '50,000', '10,000', '10,000', '12']]
table(s, 3, 5, data, .84, 1.78, 7.65, 2.35, [1.5, 1.5, 1.7, 1.5, 1.45])
panel(s, 8.77, 1.78, 3.8, 4.75, 'Protocol')
bullets(s, ['Seeded permutation of original 60k train.', 'Adjacent non-overlapping train/validation subsets.',
            'Original 10k test remains untouched.', 'Pixels scaled to [0,1]; batch 128; Adam lr 1e-3.'], 9.05, 2.38, 3.25, 3.6, 14)
text_box(s, 'Both VAE and AE baseline see exactly the same examples and epoch count.', .95, 4.75, 7.15, .5, 17, TEAL, True)

# 14
s = base('VAE implementation: actual loss and baseline')
code_box(s, "h = self.enc(x)\nmu, logvar = self.mu(h), self.logvar(h)\nz = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)\nrecon = self.dec(self.fc(z))\n\nrec = BCE(recon, x, reduction='sum') / batch\nkl = -0.5 * sum(1 + logvar - mu**2 - exp(logvar)) / batch\nloss = rec + beta * kl\nloss.backward()", .68, 1.68, 7.05, 4.9)
panel(s, 8.02, 1.68, 4.57, 4.9, 'Fair comparison')
bullets(s, ['VAE evaluation uses latent mean mu for deterministic reconstruction.',
            'AE baseline optimizes reconstruction BCE only.', 'Compare reconstruction BCE/MSE, not AE loss with VAE ELBO.',
            'Full weights saved outside Git under outputs/.'], 8.28, 2.28, 4.02, 3.7, 14)

# 15
s = base('VAE full results and deterministic baseline')
data = [['Metric (test n=10,000)', 'ConvVAE', 'ConvAE baseline'],
        ['Reconstruction BCE / image', f"{full_v['test_reconstruction_bce_per_image']:.2f}",
         f"{full_v['deterministic_ae_baseline_test_bce_per_image']:.2f}"],
        ['MSE / pixel', f"{full_v['test_mse_per_pixel']:.5f}",
         f"{full_v['deterministic_ae_baseline_test_mse_per_pixel']:.5f}"],
        ['KL / image', f"{full_v['test_kl_per_image']:.2f}", 'not applicable'],
        ['ELBO loss / image', f"{full_v['test_elbo_loss_per_image']:.2f}", 'not comparable']]
table(s, len(data), 3, data, .7, 1.72, 8.3, 3.85, [4.0, 2.15, 2.15])
panel(s, 9.25, 1.72, 3.35, 4.85, 'Finding')
bullets(s, ['AE wins pure reconstruction, as expected.', 'VAE pays a KL cost to organize a sampleable latent distribution.',
            f"Both models completed 12 epochs; total script time {full_v['elapsed_seconds']:.1f}s."], 9.5, 2.4, 2.85, 3.15, 14)

# 16
s = base('VAE qualitative evaluation: originals and reconstructions')
picture(s, RESULTS / 'full' / 'vae_reconstructions.png', .78, 1.65, 8.7)
panel(s, 9.72, 1.68, 2.85, 4.85, 'What to inspect')
bullets(s, ['Garment silhouette is retained.', 'Fine texture and sharp edges are smoothed.',
            'AE baseline is slightly sharper, consistent with lower BCE/MSE.', 'Visuals complement, not replace, test metrics.'], 9.95, 2.25, 2.38, 3.65, 13)

# 17
s = base('VAE learning curves: convergence and diminishing returns')
picture(s, RESULTS / 'full' / 'vae_training.png', .7, 1.62, 8.2)
panel(s, 9.15, 1.68, 3.42, 4.86, 'Reading the curves')
bullets(s, ['Largest improvement occurs in early epochs.', 'Validation VAE ELBO falls from 252.02 to 236.39.',
            'Train and validation trends remain close: no strong overfitting signal.', 'AE curve is reconstruction BCE, not ELBO.'], 9.42, 2.3, 2.88, 3.55, 13)

# 18
s = base('Agentic AI: a transparent tool-use workflow')
panel(s, .72, 1.7, 5.9, 4.85, 'Implemented agent')
bullets(s, ['Deterministic router recognizes bounded arithmetic.', 'TF-IDF retrieves top-3 passages from a closed corpus.',
            'Extractive QA proposes spans; score combines relevance and QA confidence.', 'Returns source ID, scores, excerpt, or abstention.'], 1.0, 2.34, 5.32, 3.6, 15)
panel(s, 6.85, 1.7, 5.72, 4.85, 'Claim boundary')
bullets(s, ['Not an autonomous LLM planner.', 'Not web search or open-domain research.',
            'A lightweight, auditable orchestration pipeline.', 'Calculator uses restricted AST operations - never eval().'], 7.15, 2.34, 5.1, 3.6, 15)

# 19
s = base('Agentic QA pipeline and provenance')
steps = ['Question', 'Route', 'TF-IDF\ntop 3', 'QA each\npassage', 'Rerank', 'Answer +\nsource']
for i, step in enumerate(steps):
    x = .62 + i * 2.1
    panel(s, x, 2.22, 1.82, 1.72, fill=RGBColor(232, 245, 243) if i % 2 == 0 else PALE)
    text_box(s, step, x + .12, 2.55, 1.58, .95, 15, NAVY, True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    if i < 5: text_box(s, '->', x + 1.82, 2.92, .28, .3, 17, TEAL, True, align=PP_ALIGN.CENTER)
text_box(s, 'candidate score = QA confidence x max(TF-IDF relevance, epsilon)', 1.0, 4.63, 11.3, .55, 19, TEAL, True, align=PP_ALIGN.CENTER)
text_box(s, 'Threshold 0.05 controls abstention before extraction.', 1.0, 5.42, 11.3, .38, 15, MUTED, align=PP_ALIGN.CENTER)

# 20
s = base('SQuAD closed-corpus evaluation and leakage control')
data = [['Mode', 'Questions', 'Unique contexts', 'Evidence corpus'],
        ['Fast', '20', str(fast_a['corpus_contexts']), 'all validation contexts'],
        ['Full', '100', str(full_a['corpus_contexts']), 'all validation contexts']]
table(s, 3, 4, data, .78, 1.78, 7.3, 2.35, [1.35, 1.5, 1.9, 2.55])
panel(s, 8.35, 1.78, 4.2, 4.75, 'Fairness')
bullets(s, ['Documents are visible because retrieval requires a corpus.', 'Answer strings are never indexed or used for routing.',
            'Gold context is supplied only to the explicitly labelled oracle baseline.', 'EM and normalized token F1 score accepted answers.'], 8.64, 2.38, 3.65, 3.55, 14)
text_box(s, 'Oracle baseline isolates extraction quality from retrieval quality.', .95, 4.78, 6.85, .5, 17, ORANGE, True)

# 21
s = base('Agent implementation: retrieval, extraction, and evidence')
code_box(s, "hits = self.retrieve(question, k=3)\nfor h in hits:\n    out = self.qa(question=question, context=h['context'])\n    score = float(out['score']) * max(h['score'], 1e-9)\n    candidates.append((score, out, h))\n\n_, answer, winner = max(candidates, key=lambda x: x[0])\nreturn {\n    'answer': answer['answer'],\n    'source_id': winner['id'],\n    'retrieval_score': winner['score'],\n    'qa_score': float(answer['score'])\n}", .68, 1.68, 7.25, 4.9)
panel(s, 8.22, 1.68, 4.36, 4.9, 'Saved per question')
bullets(s, ['Prediction and accepted answers.', 'Exact match and token F1.', 'Top-1/top-3 gold-context indicators.',
            'Selected-context correctness.', 'Retrieval score, QA score, source excerpt, oracle prediction.'], 8.48, 2.26, 3.84, 3.85, 14)

# 22
s = base('Agent full results: retrieval is the main bottleneck')
picture(s, RESULTS / 'full' / 'agent_diagnostics.png', .62, 1.58, 7.35)
panel(s, 8.2, 1.68, 4.38, 4.87, '100-question evaluation')
bullets(s, [f"End-to-end EM: {full_a['exact_match_pct']:.1f}%",
            f"End-to-end token F1: {full_a['token_f1_pct']:.2f}%",
            f"Gold-context Recall@3: {full_a['gold_context_recall_at_3_pct']:.1f}%",
            f"Selected-context accuracy: {full_a['selected_context_accuracy_pct']:.1f}%",
            f"Oracle-context EM: {full_a['oracle_context_exact_match_pct']:.1f}%",
            f"Oracle-context token F1: {full_a['oracle_context_token_f1_pct']:.2f}%",
            f"Abstention: {full_a['abstention_rate_pct']:.1f}%"], 8.48, 2.23, 3.8, 3.95, 14)

# 23
s = base('Cross-experiment conclusions, limits, and next steps')
panel(s, .66, 1.68, 4.0, 4.9, 'What worked')
bullets(s, ['Pretrained attention clearly beat the majority baseline.', 'VAE learned useful reconstructions and a regularized latent space.',
            'Explicit evidence tracking exposed retrieval failure separately from QA failure.'], .92, 2.3, 3.5, 3.55, 14)
panel(s, 4.86, 1.68, 3.95, 4.9, 'Limitations')
bullets(s, ['Single seed and one GPU environment.', 'IMDb truncation can miss long-range evidence.', 'Pixel BCE encourages smooth reconstructions.',
            'TF-IDF lexical matching and uncalibrated abstention.'], 5.12, 2.3, 3.42, 3.55, 14)
panel(s, 9.02, 1.68, 3.65, 4.9, 'Next steps')
bullets(s, ['Multi-seed confidence intervals.', 'Transformer length/learning-rate ablations.', 'beta and latent-dimension sweeps.',
            'Dense retrieval and threshold calibration.'], 9.28, 2.3, 3.1, 3.55, 14)

# 24
s = base('References, recording link, and final human inputs')
panel(s, .66, 1.62, 7.7, 4.98, 'Core references')
bullets(s, ['Vaswani et al. (2017), Attention Is All You Need.', 'Sanh et al. (2019), DistilBERT.',
            'Kingma & Welling (2014), Auto-Encoding Variational Bayes.', 'Maas et al. (2011), Learning Word Vectors for Sentiment Analysis.',
            'Xiao et al. (2017), Fashion-MNIST.', 'Rajpurkar et al. (2016), SQuAD.'], .92, 2.2, 7.18, 3.95, 14)
panel(s, 8.62, 1.62, 4.05, 4.98, 'Complete before submission')
bullets(s, ['Replace six team name/roll placeholders.', 'Insert verified video URL.',
            'Grant viewing access to required faculty emails.', 'Attach PPT/PDF and code; CC all members.',
            'Submit by 15 November 2026.'], 8.9, 2.25, 3.48, 3.45, 14)
text_box(s, 'VIDEO LINK: [INSERT VERIFIED GOOGLE DRIVE / ONEDRIVE URL]', 8.9, 5.85, 3.5, .48, 12, TEAL, True, align=PP_ALIGN.CENTER)

assert len(prs.slides) == 24
prs.core_properties.title = 'Deep Learning Triad - Verified Experiments'
prs.core_properties.subject = 'Transformers, variational autoencoders, and agentic QA'
prs.core_properties.author = 'GIM BDA Deep Learning Team'
prs.save(OUT)
print(f'Saved {OUT} with {len(prs.slides)} slides')
