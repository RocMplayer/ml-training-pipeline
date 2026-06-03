# Design

This repository uses a simple training workflow.

## Structure

- config-like arguments are passed through CLI
- training logic stays in `src/train.py`
- small dataset stays under `data/`
- notes are stored under `docs/`

## Principles

- keep dependencies minimal
- keep experiments easy to rerun
- keep results easy to inspect
