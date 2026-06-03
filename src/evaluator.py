"""Evaluation logic."""
from __future__ import annotations

import numpy as np
from rich.console import Console

console = Console()


def evaluate(model, x: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate model, return metrics dict."""
    preds = model.predict(x)
    accuracy = float(np.mean(preds == y))
    return {"accuracy": round(accuracy, 4), "n_samples": len(y)}
