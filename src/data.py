"""Dataset loading utilities."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV dataset. Last column is label."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append([float(v) for v in row])
    data = np.array(rows, dtype=np.float32)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x, y


def generate_synthetic(n_samples: int = 1000, n_features: int = 10, n_classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset."""
    x = np.random.randn(n_samples, n_features).astype(np.float32)
    w = np.random.randn(n_features, n_classes).astype(np.float32)
    y = np.argmax(x @ w, axis=1)
    return x, y
