"""Runs experiments sequentially; requires downloaded pretrained models/datasets."""
import argparse
import os
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--fast', action='store_true')
p.add_argument('--offline', action='store_true',
               help='Use already-cached Hugging Face datasets and models only.')
a = p.parse_args()
env = os.environ.copy()
if a.offline:
    env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
overall_start = time.time()
for module in ('src.transformer_sentiment', 'src.vae_fashion', 'src.agentic_qa'):
    cmd = [sys.executable, '-m', module] + (['--fast'] if a.fast else [])
    print('RUN:', ' '.join(cmd), flush=True)
    started = time.time()
    subprocess.run(cmd, check=True, env=env)
    print(f'COMPLETED {module} in {time.time() - started:.2f}s', flush=True)
print(f'ALL EXPERIMENTS COMPLETED in {time.time() - overall_start:.2f}s', flush=True)
