"""Strict, read-only compatibility check; no training or result regeneration."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from checkpoint_io import load_generator_state
from config import OptimizedConfig
from models.generator import AdvancedSeqGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    config = OptimizedConfig()
    config.device = torch.device("cpu")
    model = AdvancedSeqGenerator(config)
    model.load_state_dict(load_generator_state(args.checkpoint), strict=True)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    print(json.dumps({"checkpoint": args.checkpoint.name, "sha256": digest.hexdigest(),
                      "strict_load": True, "parallel_blocks": 5,
                      "input_channels": model.feature_extractor[0].in_channels,
                      "training_loss_verified_by_weights": False}, indent=2))


if __name__ == "__main__":
    main()
