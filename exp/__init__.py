"""Experiment modules for adversarial-thought research."""

from pathlib import Path

# Default directories for experiments
DATASET_DIRNAME = str(Path(__file__).parent.parent / "dataset")
OUTPUT_DIRNAME = str(Path(__file__).parent.parent / "out")

__all__ = ["DATASET_DIRNAME", "OUTPUT_DIRNAME"]
