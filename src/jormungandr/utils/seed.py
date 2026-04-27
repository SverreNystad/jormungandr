"""
Reproducibility utilities for seeding Python, NumPy, and PyTorch RNGs.

Functions:
    seed_everything       -- seed all RNGs globally for a fully deterministic run.
    seed_worker           -- DataLoader worker_init_fn for consistent per-worker seeding.
    build_torch_generator -- create a seeded torch.Generator for use with DataLoader.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
