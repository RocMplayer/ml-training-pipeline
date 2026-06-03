"""Training loop implementation."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class EpochResult:
    epoch: int
    loss: float
    accuracy: float
    time_sec: float


def train_epoch(model, x: np.ndarray, y: np.ndarray, lr: float) -> tuple[float, float]:
    """Train one epoch, return (loss, accuracy)."""
    logits = model.forward(x)
    # Softmax cross-entropy
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    n = x.shape[0]
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-8))
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    return loss, acc


def train_model(model, x_train, y_train, epochs: int, lr: float) -> list[EpochResult]:
    """Full training loop."""
    results = []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss, acc = train_epoch(model, x_train, y_train, lr)
        dt = time.perf_counter() - t0
        results.append(EpochResult(epoch=epoch, loss=round(loss, 6), accuracy=round(acc, 4), time_sec=round(dt, 4)))
        console.print(f"  Epoch {epoch:3d} | loss={loss:.6f} | acc={acc:.4f} | {dt:.3f}s")
    return results


def print_training_summary(results: list[EpochResult]):
    table = Table(title="Training Summary")
    table.add_column("Epoch", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Time (s)", justify="right")
    for r in results:
        table.add_row(str(r.epoch), f"{r.loss:.6f}", f"{r.accuracy:.4f}", f"{r.time_sec:.3f}")
    console.print(table)
