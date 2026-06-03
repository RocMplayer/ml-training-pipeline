#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path


def load_dataset(path: str):
    xs, ys = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return xs, ys


def train(xs, ys, epochs: int, lr: float):
    w = 0.0
    b = 0.0
    n = len(xs)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        dw = 0.0
        db = 0.0
        loss = 0.0

        for x, y in zip(xs, ys):
            pred = w * x + b
            err = pred - y
            loss += err * err
            dw += 2 * err * x
            db += 2 * err

        w -= lr * dw / n
        b -= lr * db / n
        dt = time.perf_counter() - t0

        print(f"[train] epoch={epoch} loss={loss / n:.6f} w={w:.4f} b={b:.4f} time={dt:.6f}s")


def main():
    parser = argparse.ArgumentParser(description="ml-training-pipeline")
    parser.add_argument("--data", default="data/sample.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    xs, ys = load_dataset(args.data)
    train(xs, ys, args.epochs, args.lr)


if __name__ == "__main__":
    main()
