"""Tiện ích chung: I/O, logging, seed helpers."""

import pickle
import random
from pathlib import Path
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def makedirs(*paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_pkl(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def section(title: str) -> None:
    print(f"\n{'─'*60}\n{title}\n{'─'*60}")
