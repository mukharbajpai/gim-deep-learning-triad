"""Validate presence of genuine experiment metrics before slide finalization."""
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--expect-mode', choices=['fast', 'full'])
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
for name, required in [
    ('transformer', ['test_accuracy', 'test_f1', 'test_macro_f1',
                     'majority_baseline_test_accuracy', 'test_confusion_matrix']),
    ('vae', ['test_reconstruction_bce_per_image', 'test_kl_per_image',
             'deterministic_ae_baseline_test_bce_per_image']),
    ('agent', ['exact_match_pct', 'token_f1_pct', 'gold_context_recall_at_3_pct',
               'oracle_context_exact_match_pct'])]:
    path = root / 'results' / f'{name}.json'
    if not path.exists(): raise SystemExit(f'MISSING: {path}. Run experiments first.')
    obj = json.loads(path.read_text())
    missing = [k for k in required if k not in obj]
    if missing: raise SystemExit(f'{name}: missing keys {missing}')
    invalid = [k for k in required if isinstance(obj[k], float) and not math.isfinite(obj[k])]
    if invalid: raise SystemExit(f'{name}: non-finite values for {invalid}')
    if args.expect_mode and obj.get('mode') != args.expect_mode:
        raise SystemExit(f"{name}: expected {args.expect_mode!r}, found {obj.get('mode')!r}")
    if not obj.get('runtime', {}).get('python_version'):
        raise SystemExit(f'{name}: missing runtime provenance')
    print(name, 'OK', {k: obj[k] for k in required})

for artifact in ('transformer_diagnostics.png', 'vae_reconstructions.png',
                 'vae_training.png', 'agent_diagnostics.png', 'agent_predictions.json'):
    path = root / 'results' / artifact
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f'MISSING OR EMPTY ARTIFACT: {path}')
    print('artifact OK', path.name, path.stat().st_size, 'bytes')
