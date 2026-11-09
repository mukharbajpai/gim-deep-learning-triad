"""Shared reproducibility, result storage, and runtime metadata helpers."""
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUTPUTS = ROOT / 'outputs'
DATA = ROOT / 'data'
CACHE = ROOT / '.cache'
os.environ.setdefault('HF_HOME', str(CACHE / 'huggingface'))
for p in [RESULTS, OUTPUTS, DATA, CACHE]:
    p.mkdir(parents=True, exist_ok=True)


def seed_everything(seed=42):
    """Seed supported RNGs and request deterministic PyTorch behavior."""
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dataloader_generator(seed=42):
    """Return an independently seeded generator for deterministic shuffling."""
    return torch.Generator().manual_seed(seed)


def seed_worker(worker_id):
    """Seed NumPy/Python when DataLoader workers are enabled."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def runtime_metadata():
    """Capture enough environment information to interpret measured runtimes."""
    using_cuda = torch.cuda.is_available()
    return {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'cuda_build': torch.version.cuda,
        'device': 'cuda' if using_cuda else 'cpu',
        'gpu': torch.cuda.get_device_name(0) if using_cuda else None,
        'gpu_memory_gib': (round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                           if using_cuda else None),
    }


def write_metrics(name, metrics):
    metrics = {**metrics, 'runtime': runtime_metadata()}
    path = RESULTS / f'{name}.json'
    with path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
    print(f'Saved {path}')
    return path
