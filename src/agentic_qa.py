"""Interpretable, tool-using QA agent: route -> retrieve -> extract -> verify -> abstain.

Uses SQuAD validation examples. TF-IDF retrieval operates over unique contexts;
extractive QA model answers from selected passage. Router supports a safe arithmetic
calculator without eval(). The calculator is not used on SQuAD evaluation examples.
"""
import argparse
import ast
import json
import operator
import re
import time
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .common import RESULTS, write_metrics, seed_everything

QA_MODEL = 'distilbert/distilbert-base-cased-distilled-squad'
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv}


def safe_calculate(expression):
    """Only numeric constants and +, -, *, /, parentheses; no Python execution."""
    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = visit(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError('Unsupported arithmetic expression')
    if len(expression) > 80: raise ValueError('Expression too long')
    result = visit(ast.parse(expression, mode='eval'))
    if not np.isfinite(result) or abs(result) > 1e12: raise ValueError('Result out of range')
    return result


def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(w for w in text.split() if w not in ('a', 'an', 'the'))


def exact_match(pred, golds):
    return int(any(normalize(pred) == normalize(g) for g in golds))


def token_f1(pred, golds):
    p = normalize(pred).split()
    scores = []
    for g in golds:
        gt = normalize(g).split()
        shared = sum((Counter(p) & Counter(gt)).values())
        if not p or not gt: scores.append(float(not p and not gt)); continue
        if not shared: scores.append(0.0); continue
        precision, recall = shared / len(p), shared / len(gt)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


class ResearchAgent:
    def __init__(self, contexts, threshold=0.05, qa_model=QA_MODEL):
        self.contexts = list(dict.fromkeys(contexts))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=50000)
        self.matrix = self.vectorizer.fit_transform(self.contexts)
        from transformers import pipeline
        self.qa = pipeline('question-answering', model=qa_model, tokenizer=qa_model,
                           device=0 if __import__('torch').cuda.is_available() else -1)
        self.threshold = threshold

    def retrieve(self, question, k=3):
        scores = linear_kernel(self.vectorizer.transform([question]), self.matrix).ravel()
        indices = np.argsort(scores)[::-1][:k]
        return [{'id': int(i), 'score': float(scores[i]), 'context': self.contexts[i]} for i in indices]

    def answer(self, question, forced_context=None, retrieved_hits=None):
        match = re.fullmatch(r'\s*(?:calculate|what is)\s+([\d.\s+*/()\-]+)\??\s*', question.lower())
        if match:
            try:
                return {'answer': str(safe_calculate(match.group(1))), 'tool': 'calculator', 'source_id': None}
            except (ValueError, SyntaxError, ZeroDivisionError):
                return {'answer': 'Unable to calculate safely.', 'tool': 'calculator', 'source_id': None}
        hits = ([{'id': None, 'score': 1., 'context': forced_context}]
                if forced_context is not None else
                (retrieved_hits if retrieved_hits is not None else self.retrieve(question, k=3)))
        if not hits or hits[0]['score'] < self.threshold:
            return {'answer': 'Insufficient evidence in the dataset.', 'tool': 'abstain', 'source_id': None,
                    'retrieval_score': hits[0]['score'] if hits else 0.}
        # Rerank candidates by QA score * TF-IDF retrieval relevance.
        candidates = []
        for h in hits:
            out = self.qa(question=question, context=h['context'], max_answer_len=40)
            candidates.append((float(out['score']) * max(h['score'], 1e-9), out, h))
        _, answer, winner = max(candidates, key=lambda x: x[0])
        return {'answer': answer['answer'], 'tool': 'retriever+extractive_qa',
                'source_id': winner['id'], 'retrieval_score': winner['score'],
                'qa_score': float(answer['score']), 'source_excerpt': winner['context'][:200]}


def save_diagnostics(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(2)
    axes[0].bar(x - .18,
                [metrics['exact_match_pct'], metrics['oracle_context_exact_match_pct']],
                width=.36, label='Exact match')
    axes[0].bar(x + .18,
                [metrics['token_f1_pct'], metrics['oracle_context_token_f1_pct']],
                width=.36, label='Token F1')
    axes[0].set(title='QA performance and oracle upper bound', ylabel='Percent', ylim=(0, 100))
    axes[0].set_xticks(x, ['End-to-end', 'Oracle context'])
    axes[0].legend()
    retrieval_names = ['Recall@1', 'Recall@3', 'Selected correct']
    retrieval_values = [metrics['gold_context_recall_at_1_pct'],
                        metrics['gold_context_recall_at_3_pct'],
                        metrics['selected_context_accuracy_pct']]
    axes[1].bar(retrieval_names, retrieval_values, color=['#4C78A8', '#59A14F', '#F28E2B'])
    axes[1].set(title='Evidence retrieval and selection', ylabel='Percent', ylim=(0, 100))
    axes[1].tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS / 'agent_diagnostics.png', dpi=180)
    plt.close(fig)


def run(fast=False, seed=42, threshold=0.05):
    overall_start = time.time()
    seed_everything(seed)
    from datasets import load_dataset
    raw = load_dataset('rajpurkar/squad', split='validation')
    # Evaluation examples sampled from validation only; corpus = all validation contexts.
    # This is a bounded evidence retrieval task, NOT out-of-domain/general web QA.
    sample = raw.shuffle(seed=seed).select(range(20 if fast else 100))
    contexts = list(dict.fromkeys(raw['context']))
    model_start = time.time()
    agent = ResearchAgent(contexts, threshold=threshold)
    initialization_seconds = time.time() - model_start
    records = []
    inference_start = time.time()
    for ex in sample:
        hits = agent.retrieve(ex['question'], k=3)
        retrieved_ids = [h['id'] for h in hits]
        pred = agent.answer(ex['question'], retrieved_hits=hits)
        # Oracle context is an evaluation upper bound. The gold answer remains
        # hidden; only the known evidence passage is supplied to the QA model.
        oracle = agent.answer(ex['question'], forced_context=ex['context'])
        golds = ex['answers']['text']
        records.append({'id': ex['id'], 'question': ex['question'],
                        'prediction': pred['answer'], 'reference_answers': golds,
                        'exact_match': exact_match(pred['answer'], golds),
                        'f1': token_f1(pred['answer'], golds),
                        'oracle_context_prediction': oracle['answer'],
                        'oracle_context_exact_match': exact_match(oracle['answer'], golds),
                        'oracle_context_f1': token_f1(oracle['answer'], golds),
                        'gold_context_in_top1': agent.contexts[retrieved_ids[0]] == ex['context'],
                        'gold_context_in_top3': any(agent.contexts[i] == ex['context'] for i in retrieved_ids),
                        'selected_context_correct': pred['source_id'] is not None and
                        agent.contexts[pred['source_id']] == ex['context'],
                        'tool': pred['tool'], 'source_id': pred.get('source_id'),
                        'retrieval_score': pred.get('retrieval_score'),
                        'qa_score': pred.get('qa_score'),
                        'source_excerpt': pred.get('source_excerpt')})
    inference_seconds = time.time() - inference_start
    metrics = {'experiment': 'agentic_squad_qa', 'mode': 'fast' if fast else 'full',
               'seed': seed, 'dataset': 'rajpurkar/squad validation', 'qa_model': QA_MODEL,
               'corpus_contexts': len(contexts), 'sample_size': len(records),
               'threshold': threshold,
               'exact_match_pct': round(100 * float(np.mean(
                   [r['exact_match'] for r in records])), 4),
               'token_f1_pct': round(100 * float(np.mean([r['f1'] for r in records])), 4),
               'oracle_context_exact_match_pct': round(100 * float(np.mean(
                   [r['oracle_context_exact_match'] for r in records])), 4),
               'oracle_context_token_f1_pct': round(100 * float(np.mean(
                   [r['oracle_context_f1'] for r in records])), 4),
               'gold_context_recall_at_1_pct': round(100 * float(np.mean(
                   [r['gold_context_in_top1'] for r in records])), 4),
               'gold_context_recall_at_3_pct': round(100 * float(np.mean(
                   [r['gold_context_in_top3'] for r in records])), 4),
               'selected_context_accuracy_pct': round(100 * float(np.mean(
                   [r['selected_context_correct'] for r in records])), 4),
               'abstention_rate_pct': round(100 * float(np.mean(
                   [r['tool'] == 'abstain' for r in records])), 4),
               'evaluation_protocol': ('all validation contexts are indexed as documents; answer strings are '
                                       'used only after prediction for scoring; oracle baseline receives the '
                                       'gold context but never the answer'),
               'initialization_seconds': round(initialization_seconds, 2),
               'inference_seconds': round(inference_seconds, 2),
               'mean_seconds_per_question_including_oracle': round(
                   inference_seconds / len(records), 4),
               'elapsed_seconds': round(time.time()-overall_start, 2)}
    save_diagnostics(metrics)
    write_metrics('agent', metrics)
    (RESULTS / 'agent_predictions.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fast', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold', type=float, default=0.05)
    a = p.parse_args()
    print(json.dumps(run(a.fast, a.seed, a.threshold), indent=2))
