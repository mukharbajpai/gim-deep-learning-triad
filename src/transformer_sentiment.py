"""Fine-tune DistilBERT on disjoint IMDb train/validation/test splits.

Fast mode runs a limited data subset; full mode uses more data.
No metrics are populated before successful execution.
"""
import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import OUTPUTS, RESULTS, device, seed_everything, write_metrics

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MODEL = 'distilbert/distilbert-base-uncased'


def prediction_metrics(y_true, y_pred):
    """Return the same metrics for learned and baseline predictions."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'binary_f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def save_diagnostics(y_true, y_pred, model_scores, baseline_scores):
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    image = axes[0].imshow(cm, cmap='Blues')
    axes[0].set(title='DistilBERT test confusion matrix', xlabel='Predicted', ylabel='Actual',
                xticks=[0, 1], yticks=[0, 1],
                xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
    for row in range(2):
        for col in range(2):
            axes[0].text(col, row, str(cm[row, col]), ha='center', va='center',
                         color='white' if cm[row, col] > cm.max() / 2 else 'black')
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    names = ['accuracy', 'binary_f1', 'macro_f1']
    x = np.arange(len(names))
    axes[1].bar(x - .18, [model_scores[n] for n in names], width=.36, label='DistilBERT')
    axes[1].bar(x + .18, [baseline_scores[n] for n in names], width=.36, label='majority')
    axes[1].set(title='Model versus majority baseline', ylabel='Score', ylim=(0, 1),
                xticks=x, xticklabels=['Accuracy', 'Binary F1', 'Macro F1'])
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / 'transformer_diagnostics.png', dpi=180)
    plt.close(fig)


def run(fast=False, epochs=None, seed=42):
    overall_start = time.time()
    seed_everything(seed)
    raw = load_dataset('stanfordnlp/imdb')
    train_all = raw['train'].shuffle(seed=seed)
    test_all = raw['test'].shuffle(seed=seed)
    # Keep validation separate from the held-out IMDb test split.
    n_train, n_val, n_test = (800, 200, 200) if fast else (20000, 5000, 5000)
    train_ds = train_all.select(range(n_train))
    val_ds = train_all.select(range(n_train, n_train + n_val))
    test_ds = test_all.select(range(n_test))
    # Ranges within the shuffled original training split are disjoint; IMDb's
    # original test split is never used for fitting or model selection.
    assert n_train + n_val <= len(train_all) and n_test <= len(test_all)
    train_labels = np.asarray(train_ds['label'])
    majority_label = int(np.bincount(train_labels).argmax())
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=192)
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    batch = 8 if fast else 16
    args = TrainingArguments(
        output_dir=str(OUTPUTS / 'distilbert'),
        num_train_epochs=epochs if epochs is not None else (1 if fast else 2),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=20,
        save_strategy='no',
        report_to='none',
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        dataloader_num_workers=0,
        fp16=False,
    )
    def metric_fn(pred):
        y_true = pred.label_ids
        y_pred = np.argmax(pred.predictions, axis=-1)
        scores = prediction_metrics(y_true, y_pred)
        return {'accuracy': scores['accuracy'], 'f1': scores['binary_f1'],
                'macro_f1': scores['macro_f1']}
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds,
                      processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=metric_fn)
    train_start = time.time()
    trainer.train()
    validation = trainer.evaluate(eval_dataset=val_ds)
    test_prediction = trainer.predict(test_ds)
    yp = np.argmax(test_prediction.predictions, axis=1)
    yt = np.asarray(test_prediction.label_ids)
    model_scores = prediction_metrics(yt, yp)
    majority_predictions = np.full_like(yt, majority_label)
    baseline_scores = prediction_metrics(yt, majority_predictions)
    save_diagnostics(yt, yp, model_scores, baseline_scores)
    metrics = {
        'experiment': 'distilbert_imdb', 'mode': 'fast' if fast else 'full',
        'seed': seed, 'dataset': 'stanfordnlp/imdb', 'model': MODEL,
        'train_size': n_train, 'validation_size': n_val, 'test_size': n_test,
        'epochs': args.num_train_epochs, 'learning_rate': args.learning_rate,
        'split_protocol': ('train/validation are non-overlapping ranges after seeded shuffle '
                           'of original IMDb train; test is from original IMDb test'),
        'test_accuracy': model_scores['accuracy'],
        'test_f1': model_scores['binary_f1'],
        'test_macro_f1': model_scores['macro_f1'],
        'test_confusion_matrix': confusion_matrix(yt, yp).tolist(),
        'validation_accuracy': float(validation['eval_accuracy']),
        'validation_f1': float(validation['eval_f1']),
        'validation_macro_f1': float(validation['eval_macro_f1']),
        'majority_baseline_label': majority_label,
        'majority_baseline_test_accuracy': baseline_scores['accuracy'],
        'majority_baseline_test_f1': baseline_scores['binary_f1'],
        'majority_baseline_test_macro_f1': baseline_scores['macro_f1'],
        'training_and_evaluation_seconds': round(time.time() - train_start, 2),
        'elapsed_seconds': round(time.time() - overall_start, 2),
        'compute_device': str(device()),
    }
    trainer.save_model(str(OUTPUTS / 'distilbert' / 'final'))
    tokenizer.save_pretrained(str(OUTPUTS / 'distilbert' / 'final'))
    write_metrics('transformer', metrics)
    (RESULTS / 'transformer_classification_report.txt').write_text(
        classification_report(yt, yp, target_names=['negative','positive'], zero_division=0), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fast', action='store_true')
    p.add_argument('--epochs', type=float)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    print(json.dumps(run(a.fast, a.epochs, a.seed), indent=2))
