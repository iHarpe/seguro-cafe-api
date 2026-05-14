"""Entry point: train all 3 models, validate artifacts, and report metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train import run_training

if __name__ == "__main__":
    run_training(verbose=True)
