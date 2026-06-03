"""Metric computation utilities."""
from __future__ import annotations

import json
from pathlib import Path


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())
