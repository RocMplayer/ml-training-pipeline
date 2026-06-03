"""Model definitions for training pipeline."""
from __future__ import annotations

import numpy as np


class MLP:
    """Simple multi-layer perceptron (NumPy-based for portability)."""

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, dropout: float = 0.0):
        self.layers = []
        prev = input_size
        for h in hidden_sizes:
            self.layers.append({
                "W": np.random.randn(prev, h).astype(np.float32) * np.sqrt(2.0 / prev),
                "b": np.zeros(h, dtype=np.float32),
            })
            prev = h
        self.layers.append({
            "W": np.random.randn(prev, output_size).astype(np.float32) * np.sqrt(2.0 / prev),
            "b": np.zeros(output_size, dtype=np.float32),
        })
        self.dropout = dropout

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(self.layers):
            x = x @ layer["W"] + layer["b"]
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)  # ReLU
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)
