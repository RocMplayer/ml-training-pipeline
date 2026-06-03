# ml-training-pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ROCm](https://img.shields.io/badge/ROCm-6.x-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/RocMplayer/ml-training-pipeline/actions/workflows/lint.yml/badge.svg)](https://github.com/RocMplayer/ml-training-pipeline/actions)

**Minimal machine learning training pipeline for reproducible experiments on AMD GPUs via ROCm.**

Designed for structured ML experimentation with clean separation of config, training, evaluation, and reporting.

---

## Architecture

```
+--------------------------------------------------------------+
|                        run_train.py                           |
|                   (orchestrator + CLI)                         |
+----------+----------+---------------+-------------------------+
|  Config  | Training | Evaluation    | Reporting               |
|  YAML    | Loop     | Metrics       | JSON + Charts           |
|  Loader  | Epoch    | Loss/Acc      | Markdown Report         |
+----------+----------+---------------+-------------------------+
|                    utils/                                      |
|  metrics.py          |  logger.py                             |
|  loss/accuracy       |  rich console + file logging            |
+--------------------------------------------------------------+
```

## Features

| Component | What it does | Details |
|:---|:---|:---|
| **Config** | YAML-based configuration | Hyperparams, paths, model settings |
| **Training** | Epoch-based training loop | Gradient descent, checkpointing |
| **Evaluation** | Metric computation | Loss, accuracy, per-class metrics |
| **Reporting** | Structured output | JSON results, markdown reports, charts |
| **GPU Support** | ROCm/PyTorch | AMD GPU acceleration via ROCm |

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch with ROCm support (for GPU training)
- pip

### Install

```bash
git clone https://github.com/RocMplayer/ml-training-pipeline.git
cd ml-training-pipeline
pip install -r requirements.txt
```

### Train

```bash
python run_train.py --config configs/default.yaml
python run_train.py --epochs 10 --lr 0.001 --batch-size 64
python run_train.py --cpu  # Force CPU mode
```

### Evaluate

```bash
python run_train.py --eval-only --checkpoint results/checkpoint.pt
```

---

## Project Structure

```
ml-training-pipeline/
+-- src/
|   +-- model.py              # Model architecture
|   +-- trainer.py            # Training loop
|   +-- evaluator.py          # Evaluation logic
|   +-- data.py               # Dataset loading
+-- configs/
|   +-- default.yaml          # Default hyperparameters
+-- utils/
|   +-- metrics.py            # Metric computation
|   +-- logger.py             # Logging utilities
|   +-- report_generator.py   # Report generation
+-- data/                     # Dataset storage
+-- results/                  # Training output
+-- run_train.py              # Main entry point
+-- requirements.txt
+-- pyproject.toml
+-- LICENSE
```

---

## Configuration

```yaml
# configs/default.yaml
model:
  type: mlp
  hidden_sizes: [128, 64]
  dropout: 0.1

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam

data:
  train_path: data/train.csv
  val_path: data/val.csv
  test_path: data/test.csv
```

---

## Output

```
results/
+-- checkpoint.pt
+-- metrics.json
+-- TRAINING_REPORT.md
+-- loss_curve.png
```

---

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest tests/ -v
```

---

## Roadmap

- [ ] Multi-GPU training (DistributedDataParallel)
- [ ] ROCm-specific optimizations
- [ ] Experiment tracking integration (W&B)
- [ ] Hyperparameter sweep support
- [ ] Model export (ONNX, TorchScript)

---

## License

MIT -- see [LICENSE](LICENSE).
