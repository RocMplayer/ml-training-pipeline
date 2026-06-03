#!/usr/bin/env bash
set -euo pipefail

python src/train.py --epochs 3 --lr 0.01 --batch-size 32
