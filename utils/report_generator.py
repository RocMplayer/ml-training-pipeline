"""Generate training reports."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from rich.console import Console

console = Console()


def generate_report(results_dir: Path):
    output = results_dir / "TRAINING_REPORT.md"
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        console.print("[yellow]No metrics.json found[/yellow]")
        return

    metrics = json.loads(metrics_file.read_text())
    lines = [
        "# Training Report",
        f"\nGenerated: {datetime.now().isoformat()}\n",
        "## Configuration\n",
        f"- Epochs: {metrics.get('epochs', 'N/A')}",
        f"- Learning rate: {metrics.get('learning_rate', 'N/A')}",
        f"- Final loss: {metrics.get('final_loss', 'N/A')}",
        f"- Final accuracy: {metrics.get('final_accuracy', 'N/A')}",
    ]
    output.write_text("\n".join(lines))
    console.print(f"[green]Report: {output}[/green]")
