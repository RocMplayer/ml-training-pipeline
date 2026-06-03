# ml-training-pipeline

A minimal training pipeline for reproducible machine learning experiments.

## Overview

This repository is intended for running small training experiments in a structured way.

It includes:
- a configurable training script
- a tiny synthetic dataset
- runtime logging
- a simple benchmark flow

## Goals

- keep experiments reproducible
- separate config from code
- track runtime and training metrics
- create a base pipeline for larger experiments later

## Repository layout

```
ml-training-pipeline/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── sample.csv
├── docs/
│   ├── design.md
│   └── notes.md
├── scripts/
│   └── run_all.sh
└── src/
    └── train.py
```

## Quick start

```bash
python src/train.py --epochs 3 --lr 0.01 --batch-size 32
```

## Notes

This repo is intentionally small so the training flow is easy to follow.
