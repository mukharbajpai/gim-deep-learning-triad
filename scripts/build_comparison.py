"""Build a factual fast-versus-full comparison chart from result JSON files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def load(mode, name):
    path = RESULTS / mode / f'{name}.json'
    return json.loads(path.read_text(encoding='utf-8'))


fast_t, full_t = load('fast', 'transformer'), load('full', 'transformer')
fast_v, full_v = load('fast', 'vae'), load('full', 'vae')
fast_a, full_a = load('fast', 'agent'), load('full', 'agent')

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

x = np.arange(2)
axes[0].bar(x - .18, [100 * fast_t['test_accuracy'], 100 * full_t['test_accuracy']],
            .36, label='DistilBERT')
axes[0].bar(x + .18, [100 * fast_t['majority_baseline_test_accuracy'],
                      100 * full_t['majority_baseline_test_accuracy']], .36,
            label='majority')
axes[0].set(title='IMDb test accuracy', ylabel='Percent', xticks=x,
            xticklabels=['Fast\n(n=200)', 'Full\n(n=5,000)'], ylim=(0, 100))
axes[0].legend(fontsize=8)

axes[1].bar(x - .18, [fast_v['test_reconstruction_bce_per_image'],
                      full_v['test_reconstruction_bce_per_image']], .36, label='VAE')
axes[1].bar(x + .18, [fast_v['deterministic_ae_baseline_test_bce_per_image'],
                      full_v['deterministic_ae_baseline_test_bce_per_image']], .36,
            label='AE baseline')
axes[1].set(title='Fashion-MNIST reconstruction', ylabel='BCE/image (lower is better)',
            xticks=x, xticklabels=['Fast\n(n=500)', 'Full\n(n=10,000)'])
axes[1].legend(fontsize=8)

width = .25
axes[2].bar(x - width, [fast_a['exact_match_pct'], full_a['exact_match_pct']],
            width, label='end-to-end EM')
axes[2].bar(x, [fast_a['oracle_context_exact_match_pct'],
                full_a['oracle_context_exact_match_pct']], width, label='oracle EM')
axes[2].bar(x + width, [fast_a['gold_context_recall_at_3_pct'],
                        full_a['gold_context_recall_at_3_pct']], width, label='Recall@3')
axes[2].set(title='SQuAD closed-corpus QA', ylabel='Percent', xticks=x,
            xticklabels=['Fast\n(n=20)', 'Full\n(n=100)'], ylim=(0, 100))
axes[2].legend(fontsize=8)

fig.suptitle('Fast smoke runs versus full evaluation runs (seed 42)', fontsize=14, weight='bold')
fig.tight_layout()
fig.savefig(RESULTS / 'fast_vs_full_comparison.png', dpi=190, bbox_inches='tight')
print(RESULTS / 'fast_vs_full_comparison.png')
