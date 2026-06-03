#!/usr/bin/env python3
"""Main training entry point."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    parser = argparse.ArgumentParser(description="ml-training-pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", "-o", default="results")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    lr = args.lr or cfg["training"]["learning_rate"]

    console.print(Panel.fit(
        f"[bold white]ml-training-pipeline[/bold white]\n"
        f"Epochs: {epochs} | LR: {lr}\n"
        f"Mode: {'CPU' if args.cpu else 'GPU (ROCm)'}",
        title="ML Training",
        border_style="blue",
    ))

    from src.data import generate_synthetic
    from src.model import MLP
    from src.trainer import train_model, print_training_summary

    x, y = generate_synthetic(n_samples=1000, n_features=10, n_classes=3)
    model = MLP(10, cfg["model"]["hidden_sizes"], 3, cfg["model"]["dropout"])

    console.print("\n[bold]Training...[/bold]")
    t_start = time.time()
    results = train_model(model, x, y, epochs, lr)
    elapsed = time.time() - t_start

    print_training_summary(results)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "epochs": epochs,
        "learning_rate": lr,
        "final_loss": results[-1].loss,
        "final_accuracy": results[-1].accuracy,
        "total_time": round(elapsed, 2),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    console.print(f"\n[green]Done in {elapsed:.1f}s[/green]")

    from utils.report_generator import generate_report
    generate_report(output)


if __name__ == "__main__":
    main()
